"""Independent local resource-cognition environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.scenario_config import ScenarioConfig
from envs.communication_model import CognitionMessage, NeighborCommunicationModel
from envs.geometry import (
    clip_point_to_ring,
    make_uav_init_positions_center,
    make_uav_init_positions_circle,
    sample_points_in_annulus,
    set_random_seed,
)
from envs.task_model import LocalBeliefBatch, TaskTruthBatch


class ResourceCognitionEnv:
    """Local-observation environment for explicit task sensing."""

    MOVE_ACTIONS = 5

    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.cfg = config or ScenarioConfig(use_resource_cognition=True)
        self.cfg.validate()
        self.rng = set_random_seed(self.cfg.seed)
        self.num_agents = int(self.cfg.max_candidate_uavs)
        self.action_size = self.cfg.get_resource_cognition_action_dim()
        self.local_obs_dim = self._compute_local_obs_dim()
        self.current_step = 0
        self.no_improve_steps = 0
        self.uav_positions = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.active_mask = np.ones((self.num_agents,), dtype=bool)
        self.remaining_time = np.zeros((self.num_agents,), dtype=np.float32)
        self.total_distance_per_uav = np.zeros((self.num_agents,), dtype=np.float32)
        self.task_truth: Optional[TaskTruthBatch] = None
        self.local_beliefs: Optional[LocalBeliefBatch] = None
        self.communication_model: Optional[NeighborCommunicationModel] = None
        self._received_message_cache: List[Dict[int, Tuple[CognitionMessage, bool]]] = []
        self._slot_task_indices: List[np.ndarray] = []
        self._last_info: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = set_random_seed(seed)
        communication_seed = int(self.cfg.seed if seed is None else seed) + 104729
        self.current_step = 0
        self.no_improve_steps = 0
        self._init_uavs()

        positions = sample_points_in_annulus(
            num_points=self.cfg.num_cognition_tasks,
            r_inner=self.cfg.r_safe,
            r_outer=self.cfg.r_disaster,
            rng=self.rng,
        ).astype(np.float32)
        band_ids = self.rng.integers(
            0,
            self.cfg.cognition_num_bands,
            size=self.cfg.num_cognition_tasks,
            dtype=np.int32,
        )
        true_states = self.rng.integers(
            0, 2, size=self.cfg.num_cognition_tasks
        ).astype(np.float32)
        priorities = self.rng.uniform(
            self.cfg.task_priority_min,
            self.cfg.task_priority_max,
            size=self.cfg.num_cognition_tasks,
        ).astype(np.float32)
        self.task_truth = TaskTruthBatch(
            positions_xy=positions,
            band_ids=band_ids,
            true_states=true_states,
            priorities=priorities,
        )
        self.local_beliefs = LocalBeliefBatch(
            num_agents=self.num_agents,
            task_priorities=priorities,
            initial_uncertainty=self.cfg.task_initial_uncertainty,
            initial_aoi=self.cfg.task_initial_aoi,
            max_aoi=self.cfg.task_max_aoi,
        )
        self.communication_model = NeighborCommunicationModel(
            np.random.default_rng(communication_seed),
            delay_steps=self.cfg.cognition_communication_delay_steps,
            packet_loss_rate=self.cfg.cognition_packet_loss_rate,
        )
        self._received_message_cache = [dict() for _ in range(self.num_agents)]
        self.remaining_time[:] = self.cfg.uav_max_time
        self.total_distance_per_uav[:] = 0.0
        self._last_info = self._build_info(
            reward=0.0,
            uncertainty_gain=0.0,
            aoi_gain=0.0,
            repeat_ratio=0.0,
            move_distances=np.zeros((self.num_agents,), dtype=np.float32),
            termination_reason="running",
        )
        return self._build_output()

    def step(
        self,
        actions: List[int] | np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        truth, beliefs = self._require_state()
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.num_agents,):
            raise ValueError(f"actions must have shape ({self.num_agents},).")
        if np.any(action_array < 0) or np.any(action_array >= self.action_size):
            raise ValueError("action out of range.")

        sensing_agents, sensing_tasks = self._decode_sensing_actions(action_array)
        if sensing_tasks.size:
            _, task_counts = np.unique(sensing_tasks, return_counts=True)
            repeat_count = int(np.sum(np.maximum(task_counts - 1, 0)))
        else:
            repeat_count = 0

        self.current_step += 1
        move_distances = self._apply_movement(action_array)
        beliefs.age(self.cfg.cognition_aoi_increment)

        if sensing_tasks.size:
            noise = self.rng.normal(
                0.0,
                self.cfg.cognition_observation_noise_std,
                size=sensing_tasks.shape,
            ).astype(np.float32)
            observations = truth.true_states[sensing_tasks] + noise
        else:
            observations = np.zeros((0,), dtype=np.float32)
        before_sensing_uncertainty = beliefs.uncertainties[
            sensing_agents, sensing_tasks
        ].copy()
        before_sensing_aoi = beliefs.aoi[sensing_agents, sensing_tasks].copy()
        sensing_gain = beliefs.apply_local_sensing(
            sensing_agents,
            sensing_tasks,
            observations,
            uncertainty_reduction=self.cfg.cognition_task_uncertainty_reduction,
            current_step=self.current_step,
        )
        local_information_gains = (
            before_sensing_uncertainty
            - beliefs.uncertainties[sensing_agents, sensing_tasks]
            + (
                before_sensing_aoi - beliefs.aoi[sensing_agents, sensing_tasks]
            )
            / max(self.cfg.task_max_aoi, 1e-6)
        )

        delivered_messages = self._deliver_due_messages()
        fusion_stats = self._fuse_messages(delivered_messages)
        outgoing_messages = self._build_cognition_messages(
            sensing_agents,
            sensing_tasks,
            local_information_gains,
        )
        transmission_stats = self._require_communication().submit(outgoing_messages)
        zero_delay_messages = self._deliver_due_messages()
        if zero_delay_messages:
            zero_delay_fusion = self._fuse_messages(zero_delay_messages)
            delivered_messages.extend(zero_delay_messages)
            fusion_stats["accepted"] += zero_delay_fusion["accepted"]
            fusion_stats["quality_gain"] += zero_delay_fusion["quality_gain"]

        repeat_ratio = float(repeat_count / max(len(truth), 1))
        active_count = max(int(np.sum(self.active_mask)), 1)
        movement_cost = float(
            np.sum(move_distances) / max(active_count * self.cfg.step_size(), 1e-6)
        )
        sensing_cost = self.cfg.cognition_sensing_cost * float(sensing_tasks.size)
        repeat_penalty = self.cfg.cognition_repeat_penalty * repeat_ratio
        communication_penalty = (
            self.cfg.cognition_message_cost * float(transmission_stats.attempted)
        )
        fusion_reward = (
            self.cfg.cognition_fusion_reward_weight * float(fusion_stats["quality_gain"])
        )
        reward = float(
            self.cfg.reward_weight_uncertainty_gain * sensing_gain["uncertainty_gain"]
            + self.cfg.reward_weight_aoi_gain * sensing_gain["aoi_gain"]
            + fusion_reward
            - sensing_cost
            - repeat_penalty
            - communication_penalty
            - self.cfg.reward_weight_movement_cost * movement_cost
        )
        progress = (
            sensing_gain["uncertainty_gain"]
            + sensing_gain["aoi_gain"]
            + float(fusion_stats["quality_gain"])
        )
        self.no_improve_steps = 0 if progress > 1e-6 else self.no_improve_steps + 1
        done, reason = self._check_done()
        info = self._build_info(
            reward=reward,
            uncertainty_gain=sensing_gain["uncertainty_gain"],
            aoi_gain=sensing_gain["aoi_gain"],
            repeat_ratio=repeat_ratio,
            move_distances=move_distances,
            termination_reason=reason,
        )
        info.update(
            {
                "sensing_action_count": int(sensing_tasks.size),
                "selected_task_count": int(np.unique(sensing_tasks).size),
                "repeat_task_count": repeat_count,
                "movement_cost": movement_cost,
                "sensing_cost": float(sensing_cost),
                "repeat_penalty": float(repeat_penalty),
                "messages_attempted": int(transmission_stats.attempted),
                "messages_dropped": int(transmission_stats.dropped),
                "messages_delivered": int(len(delivered_messages)),
                "messages_fused": int(fusion_stats["accepted"]),
                "messages_pending": int(self._require_communication().pending_count),
                "message_acceptance_ratio": float(
                    fusion_stats["accepted"] / max(len(delivered_messages), 1)
                ),
                "communication_cost": float(communication_penalty),
                "communication_penalty": float(communication_penalty),
                "fusion_gain": float(fusion_stats["quality_gain"]),
                "fusion_reward": float(fusion_reward),
            }
        )
        self._last_info = info
        return self._build_output(), reward, done, info

    def get_local_obs(self, agent_id: int) -> np.ndarray:
        truth, beliefs = self._require_state()
        if not 0 <= int(agent_id) < self.num_agents:
            raise ValueError("agent_id out of range.")
        i = int(agent_id)
        position = self.uav_positions[i, :2]
        distances = np.linalg.norm(truth.positions_xy - position[None, :], axis=1)
        visible = np.where(distances <= self.cfg.obs_radius)[0]
        visible = visible[np.argsort(distances[visible])][: self.cfg.cognition_max_task_slots]
        self._slot_task_indices[i] = visible.astype(np.int64)

        self_features = np.array(
            [
                self.uav_positions[i, 0] / self.cfg.r_disaster,
                self.uav_positions[i, 1] / self.cfg.r_disaster,
                self.uav_positions[i, 2] / self.cfg.uav_h_max,
                self.remaining_time[i] / max(self.cfg.uav_max_time, 1e-6),
                self.current_step / max(self.cfg.max_steps, 1),
                beliefs.local_quality(i),
            ],
            dtype=np.float32,
        )
        task_features = np.zeros((self.cfg.cognition_max_task_slots, 8), dtype=np.float32)
        for slot, task_id in enumerate(visible):
            rel = truth.positions_xy[task_id] - position
            task_features[slot] = np.array(
                [
                    rel[0] / self.cfg.obs_radius,
                    rel[1] / self.cfg.obs_radius,
                    truth.band_ids[task_id] / max(self.cfg.cognition_num_bands - 1, 1),
                    beliefs.estimates[i, task_id],
                    beliefs.uncertainties[i, task_id],
                    beliefs.aoi[i, task_id] / max(self.cfg.task_max_aoi, 1e-6),
                    truth.priorities[task_id] / max(self.cfg.task_priority_max, 1e-6),
                    beliefs.confidence[i, task_id],
                ],
                dtype=np.float32,
            )

        neighbor_features = np.zeros((self.cfg.max_obs_uavs, 8), dtype=np.float32)
        received = [
            (sender_id, message, accepted)
            for sender_id, (message, accepted) in self._received_message_cache[i].items()
            if self.current_step - message.created_step <= self.cfg.task_max_aoi
        ]
        received.sort(key=lambda item: (item[1].created_step, item[0]), reverse=True)
        for slot, (sender_id, message, accepted) in enumerate(
            received[: self.cfg.max_obs_uavs]
        ):
            message_age = min(
                max(self.current_step - message.created_step, 0),
                self.cfg.task_max_aoi,
            )
            neighbor_features[slot] = [
                sender_id / max(self.num_agents - 1, 1),
                message.task_id / max(len(truth) - 1, 1),
                message.estimate,
                message.uncertainty,
                message.confidence,
                message_age / max(self.cfg.task_max_aoi, 1e-6),
                1.0 if accepted else 0.0,
                1.0,
            ]
        return np.concatenate(
            [self_features, task_features.flatten(), neighbor_features.flatten()]
        ).astype(np.float32)

    def get_global_state(self) -> np.ndarray:
        """Centralized training/debug state; never used as a local observation."""
        truth, beliefs = self._require_state()
        return np.concatenate(
            [
                truth.positions_xy.flatten(),
                truth.band_ids.astype(np.float32),
                truth.true_states,
                truth.priorities,
                beliefs.estimates.flatten(),
                beliefs.uncertainties.flatten(),
                (beliefs.aoi / max(self.cfg.task_max_aoi, 1e-6)).flatten(),
                beliefs.confidence.flatten(),
                self.uav_positions.flatten(),
                self.remaining_time / max(self.cfg.uav_max_time, 1e-6),
                np.array(
                    [self.current_step / max(self.cfg.max_steps, 1)], dtype=np.float32
                ),
            ]
        ).astype(np.float32)

    def _decode_sensing_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        agent_ids: List[int] = []
        task_ids: List[int] = []
        for agent_id, action in enumerate(actions):
            slot = int(action) - self.MOVE_ACTIONS
            if slot < 0 or slot >= len(self._slot_task_indices[agent_id]):
                continue
            agent_ids.append(agent_id)
            task_ids.append(int(self._slot_task_indices[agent_id][slot]))
        return np.asarray(agent_ids, dtype=np.int64), np.asarray(task_ids, dtype=np.int64)

    def _build_cognition_messages(
        self,
        sensing_agents: np.ndarray,
        sensing_tasks: np.ndarray,
        local_information_gains: np.ndarray,
    ) -> List[CognitionMessage]:
        if not self.cfg.cognition_enable_communication or sensing_tasks.size == 0:
            return []
        truth, beliefs = self._require_state()
        messages: List[CognitionMessage] = []
        for sender_id, task_id, information_gain in zip(
            sensing_agents, sensing_tasks, local_information_gains
        ):
            sender_id = int(sender_id)
            task_id = int(task_id)
            confidence = float(beliefs.confidence[sender_id, task_id])
            aoi = float(beliefs.aoi[sender_id, task_id])
            freshness = float(np.exp(-self.cfg.cognition_freshness_decay * aoi))
            message_value = float(
                truth.priorities[task_id]
                * max(float(information_gain), 0.0)
                * confidence
                * freshness
            )
            if message_value < self.cfg.cognition_message_value_threshold:
                continue

            sender_xy = self.uav_positions[sender_id, :2]
            receiver_distances = {
                receiver_id: float(
                    np.linalg.norm(self.uav_positions[receiver_id, :2] - sender_xy)
                )
                for receiver_id in range(self.num_agents)
                if receiver_id != sender_id and self.active_mask[receiver_id]
            }
            receivers = sorted(
                (
                    receiver_id
                    for receiver_id, distance in receiver_distances.items()
                    if distance <= self.cfg.cognition_communication_radius
                ),
                key=receiver_distances.get,
            )[: self.cfg.cognition_max_messages_per_agent]
            for receiver_id in receivers:
                messages.append(
                    CognitionMessage(
                        sender_id=sender_id,
                        receiver_id=int(receiver_id),
                        task_id=task_id,
                        estimate=float(beliefs.estimates[sender_id, task_id]),
                        uncertainty=float(beliefs.uncertainties[sender_id, task_id]),
                        confidence=confidence,
                        aoi=aoi,
                        created_step=self.current_step,
                        arrival_step=(
                            self.current_step
                            + self._require_communication().delay_steps
                        ),
                    )
                )
        return messages

    def _deliver_due_messages(self) -> List[CognitionMessage]:
        if not self.cfg.cognition_enable_communication:
            return []
        return self._require_communication().deliver(self.current_step)

    def _fuse_messages(self, messages: List[CognitionMessage]) -> Dict[str, float]:
        _, beliefs = self._require_state()
        accepted = 0
        quality_gain = 0.0
        for message in messages:
            result = beliefs.fuse_neighbor_message(
                receiver_id=message.receiver_id,
                task_id=message.task_id,
                estimate=message.estimate,
                uncertainty=message.uncertainty,
                confidence=message.confidence,
                message_aoi=message.aoi,
                source_update_step=message.created_step,
                current_step=self.current_step,
                confidence_threshold=self.cfg.cognition_fusion_confidence_threshold,
                freshness_decay=self.cfg.cognition_freshness_decay,
            )
            accepted += int(result["accepted"])
            quality_gain += float(result["quality_gain"])
            self._received_message_cache[message.receiver_id][message.sender_id] = (
                message,
                bool(result["accepted"]),
            )
        return {"accepted": float(accepted), "quality_gain": float(quality_gain)}

    def _init_uavs(self) -> None:
        if self.cfg.uav_init_mode == "circle":
            xy = make_uav_init_positions_circle(self.num_agents, self.cfg.r_safe)
        elif self.cfg.uav_init_mode == "center":
            xy = make_uav_init_positions_center(self.num_agents, radius_spread=30.0)
        elif self.cfg.uav_init_mode == "custom":
            xy = np.asarray(self.cfg.custom_uav_init_xy, dtype=np.float32)
            if xy.shape != (self.num_agents, 2):
                raise ValueError("custom_uav_init_xy must have shape [num_agents, 2].")
        else:
            raise ValueError(f"Unsupported uav_init_mode: {self.cfg.uav_init_mode}")
        self.uav_positions = np.concatenate(
            [
                xy.astype(np.float32),
                np.full(
                    (self.num_agents, 1),
                    self.cfg.uav_init_height,
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )
        self.active_mask[:] = True
        self.remaining_time[:] = self.cfg.uav_max_time
        self._slot_task_indices = [
            np.zeros((0,), dtype=np.int64) for _ in range(self.num_agents)
        ]

    def _apply_movement(self, actions: np.ndarray) -> np.ndarray:
        distances = np.zeros((self.num_agents,), dtype=np.float32)
        deltas = self.cfg.action_to_delta_xy()
        for i, action in enumerate(actions):
            if (
                int(action) >= self.MOVE_ACTIONS
                or not self.active_mask[i]
                or self.remaining_time[i] <= 0.0
            ):
                continue
            dx, dy = deltas[int(action)]
            old = self.uav_positions[i, :2].copy()
            x, y = clip_point_to_ring(
                float(old[0] + dx),
                float(old[1] + dy),
                self.cfg.r_safe,
                self.cfg.r_disaster,
            )
            distance = float(
                np.linalg.norm(np.array([x, y], dtype=np.float32) - old)
            )
            self.uav_positions[i, :2] = [x, y]
            self.total_distance_per_uav[i] += distance
            self.remaining_time[i] = max(self.remaining_time[i] - self.cfg.dt, 0.0)
            distances[i] = distance
        return distances

    def _check_done(self) -> Tuple[bool, str]:
        _, beliefs = self._require_state()
        if self.current_step >= self.cfg.max_steps:
            return True, "max_steps"
        if beliefs.mean_uncertainty() <= self.cfg.trusted_sensing_uncertainty_target:
            return True, "uncertainty_target"
        if not np.any(self.remaining_time > 0.0):
            return True, "energy_timeout"
        if self.no_improve_steps >= self.cfg.stagnation_patience:
            return True, "stagnation"
        return False, "running"

    def _build_output(self) -> Dict[str, Any]:
        local_obs = np.stack(
            [self.get_local_obs(i) for i in range(self.num_agents)], axis=0
        )
        return {
            "global_state": self.get_global_state(),
            "local_obs": local_obs.astype(np.float32),
            "action_mask": self._get_action_mask(),
        }

    def _get_action_mask(self) -> np.ndarray:
        mask = np.ones((self.num_agents, self.action_size), dtype=np.float32)
        for i, visible in enumerate(self._slot_task_indices):
            mask[i, self.MOVE_ACTIONS + len(visible):] = 0.0
        return mask

    def _build_info(
        self,
        *,
        reward: float,
        uncertainty_gain: float,
        aoi_gain: float,
        repeat_ratio: float,
        move_distances: np.ndarray,
        termination_reason: str,
    ) -> Dict[str, Any]:
        truth, beliefs = self._require_state()
        per_agent_quality = np.array(
            [beliefs.local_quality(i) for i in range(self.num_agents)],
            dtype=np.float32,
        )
        return {
            "reward_total": float(reward),
            "uncertainty_gain": float(uncertainty_gain),
            "aoi_gain": float(aoi_gain),
            "reward_uncertainty_gain": float(uncertainty_gain),
            "reward_aoi_gain": float(aoi_gain),
            "repeat_sensing_ratio": float(repeat_ratio),
            "mean_task_uncertainty": beliefs.mean_uncertainty(),
            "mean_task_aoi": beliefs.mean_aoi(),
            "cognitive_quality": beliefs.mean_quality(),
            "per_agent_cognitive_quality": per_agent_quality,
            "mean_estimation_error": beliefs.mean_estimation_error(truth.true_states),
            "move_distance_total_step": float(np.sum(move_distances)),
            "total_distance_per_uav": self.total_distance_per_uav.copy(),
            "remaining_time": self.remaining_time.copy(),
            "active_uav_count": int(np.sum(self.active_mask)),
            "step": int(self.current_step),
            "termination_reason": termination_reason,
        }

    def _compute_local_obs_dim(self) -> int:
        return int(6 + self.cfg.cognition_max_task_slots * 8 + self.cfg.max_obs_uavs * 8)

    def _require_state(self) -> Tuple[TaskTruthBatch, LocalBeliefBatch]:
        if self.task_truth is None or self.local_beliefs is None:
            raise RuntimeError("Call reset before accessing task state.")
        return self.task_truth, self.local_beliefs

    def _require_communication(self) -> NeighborCommunicationModel:
        if self.communication_model is None:
            raise RuntimeError("Call reset before accessing communication state.")
        return self.communication_model
