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
    SENSE_ACTION_START = MOVE_ACTIONS

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
        demand_levels = self.rng.beta(
            2.0, 2.0, size=self.cfg.num_cognition_tasks
        ).astype(np.float32)
        arrival_rates = self.rng.uniform(
            self.cfg.cognition_arrival_rate_min,
            self.cfg.cognition_arrival_rate_max,
            size=self.cfg.num_cognition_tasks,
        ).astype(np.float32)
        initial_queues = self.rng.uniform(
            self.cfg.cognition_initial_queue_min,
            self.cfg.cognition_initial_queue_max,
            size=self.cfg.num_cognition_tasks,
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
            demand_levels=demand_levels,
            priorities=priorities,
            arrival_rates=arrival_rates,
            queue_lengths=initial_queues,
            queue_capacity=self.cfg.cognition_queue_capacity,
        )
        self.local_beliefs = LocalBeliefBatch(
            num_agents=self.num_agents,
            task_priorities=priorities,
            initial_uncertainty=self.cfg.task_initial_uncertainty,
            initial_aoi=self.cfg.task_initial_aoi,
            max_aoi=self.cfg.task_max_aoi,
            spectrum_quality_weight=self.cfg.cognition_spectrum_quality_weight,
            demand_quality_weight=self.cfg.cognition_demand_quality_weight,
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
        schedule_agents, schedule_tasks = self._decode_schedule_actions(action_array)
        repeat_cost_units = np.zeros((self.num_agents,), dtype=np.float32)
        if sensing_tasks.size:
            unique_tasks, task_counts = np.unique(sensing_tasks, return_counts=True)
            repeat_count = int(np.sum(np.maximum(task_counts - 1, 0)))
            count_by_task = dict(zip(unique_tasks.tolist(), task_counts.tolist()))
            for agent_id, task_id in zip(sensing_agents, sensing_tasks):
                count = int(count_by_task[int(task_id)])
                if count > 1:
                    repeat_cost_units[int(agent_id)] = (count - 1) / count / len(truth)
        else:
            repeat_count = 0

        self.current_step += 1
        move_distances = self._apply_movement(action_array)
        beliefs.age(self.cfg.cognition_aoi_increment)
        business_arrivals = self._sample_business_arrivals()
        arrival_stats = truth.advance_business(business_arrivals)
        quality_before_sensing = self._per_agent_quality()

        if sensing_tasks.size:
            noise = self.rng.normal(
                0.0,
                self.cfg.cognition_observation_noise_std,
                size=sensing_tasks.shape,
            ).astype(np.float32)
            observations = truth.true_states[sensing_tasks] + noise
            demand_noise = self.rng.normal(
                0.0,
                self.cfg.cognition_observation_noise_std,
                size=sensing_tasks.shape,
            ).astype(np.float32)
            demand_observations = truth.demand_levels[sensing_tasks] + demand_noise
            queue_noise = self.rng.normal(
                0.0,
                self.cfg.cognition_arrival_noise_std,
                size=sensing_tasks.shape,
            ).astype(np.float32)
            queue_observations = truth.queue_lengths[sensing_tasks] + queue_noise
            arrival_observations = truth.arrival_rates[sensing_tasks] + queue_noise
        else:
            observations = np.zeros((0,), dtype=np.float32)
            demand_observations = np.zeros((0,), dtype=np.float32)
            queue_observations = np.zeros((0,), dtype=np.float32)
            arrival_observations = np.zeros((0,), dtype=np.float32)
        sensing_gain = beliefs.apply_local_sensing(
            sensing_agents,
            sensing_tasks,
            observations,
            demand_observations=demand_observations,
            queue_observations=queue_observations,
            arrival_observations=arrival_observations,
            uncertainty_reduction=self.cfg.cognition_task_uncertainty_reduction,
            demand_uncertainty_reduction=self.cfg.cognition_demand_uncertainty_reduction,
            current_step=self.current_step,
        )
        quality_after_sensing = self._per_agent_quality()
        team_quality_after_sensing = float(np.mean(quality_after_sensing))
        counterfactual_quality_without_agent = (
            team_quality_after_sensing
            - (quality_after_sensing - quality_before_sensing) / self.num_agents
        )
        sensing_difference = (
            team_quality_after_sensing - counterfactual_quality_without_agent
        ).astype(np.float32)
        local_information_gains = sensing_gain["information_gain"]

        delivered_messages = self._deliver_due_messages()
        fusion_stats = self._fuse_messages(delivered_messages)
        outgoing_messages = self._build_cognition_messages(
            sensing_agents,
            sensing_tasks,
            local_information_gains,
        )
        messages_attempted_by_sender = np.bincount(
            [message.sender_id for message in outgoing_messages],
            minlength=self.num_agents,
        ).astype(np.float32)
        transmission_stats = self._require_communication().submit(outgoing_messages)
        zero_delay_messages = self._deliver_due_messages()
        if zero_delay_messages:
            zero_delay_fusion = self._fuse_messages(zero_delay_messages)
            delivered_messages.extend(zero_delay_messages)
            fusion_stats["accepted"] += zero_delay_fusion["accepted"]
            fusion_stats["quality_gain"] += zero_delay_fusion["quality_gain"]
            fusion_stats["quality_gain_by_sender"] += zero_delay_fusion[
                "quality_gain_by_sender"
            ]
            fusion_stats["accepted_by_sender"] += zero_delay_fusion[
                "accepted_by_sender"
            ]

        scheduling_stats = self._execute_scheduling(schedule_agents, schedule_tasks)

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
        service_reward = (
            self.cfg.cognition_service_reward_weight
            * float(scheduling_stats["served_data"])
            / max(float(len(truth) * self.cfg.cognition_max_service_per_step), 1e-6)
        )
        queue_backlog_penalty = (
            self.cfg.cognition_queue_reward_weight
            * float(np.mean(truth.queue_lengths / self.cfg.cognition_queue_capacity))
        )
        shared_reward = float(
            self.cfg.reward_weight_uncertainty_gain * sensing_gain["uncertainty_gain"]
            + self.cfg.reward_weight_aoi_gain * sensing_gain["aoi_gain"]
            + fusion_reward
            + self.cfg.cognition_scheduling_reward_weight
            * len(truth)
            * scheduling_stats["team_utility"]
            + service_reward
            + self.cfg.cognition_priority_service_weight
            * scheduling_stats["high_priority_service_rate"]
            - scheduling_stats["energy_penalty"]
            - scheduling_stats["conflict_penalty"]
            - queue_backlog_penalty
            - sensing_cost
            - repeat_penalty
            - communication_penalty
            - self.cfg.reward_weight_movement_cost * movement_cost
        )
        per_agent_rewards = None
        if self.cfg.cognition_use_per_agent_rewards:
            task_scale = float(len(truth))
            sensing_rewards = (
                self.cfg.cognition_difference_reward_weight
                * task_scale
                * sensing_difference
            )
            fusion_rewards = (
                self.cfg.cognition_fusion_reward_weight
                * task_scale
                * fusion_stats["quality_gain_by_sender"]
                / self.num_agents
            )
            movement_costs = (
                self.cfg.reward_weight_movement_cost
                * move_distances
                / max(self.cfg.step_size(), 1e-6)
            )
            sensing_costs = np.zeros((self.num_agents,), dtype=np.float32)
            sensing_costs[sensing_agents] = self.cfg.cognition_sensing_cost
            repeat_costs = self.cfg.cognition_repeat_penalty * repeat_cost_units
            communication_costs = (
                self.cfg.cognition_message_cost * messages_attempted_by_sender
            )
            scheduling_rewards = (
                self.cfg.cognition_scheduling_reward_weight
                * task_scale
                * scheduling_stats["difference_by_agent"]
            )
            per_agent_rewards = (
                sensing_rewards
                + fusion_rewards
                + scheduling_rewards
                - movement_costs
                - sensing_costs
                - repeat_costs
                - communication_costs
                - scheduling_stats["energy_penalty_by_agent"]
                - scheduling_stats["conflict_penalty_by_agent"]
            ).astype(np.float32)
            reward = float(np.mean(per_agent_rewards))
        else:
            reward = shared_reward
        progress = (
            sensing_gain["uncertainty_gain"]
            + sensing_gain["aoi_gain"]
            + float(fusion_stats["quality_gain"])
            + float(scheduling_stats["team_utility"])
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
            "scheduled_task_count": int(scheduling_stats["scheduled_count"]),
                "scheduling_team_utility": float(scheduling_stats["team_utility"]),
                "scheduling_estimated_utility": float(
                    scheduling_stats["estimated_team_utility"]
                ),
                "scheduling_reward": float(
                    self.cfg.cognition_scheduling_reward_weight
                    * len(truth)
                    * scheduling_stats["team_utility"]
                ),
                "scheduling_conflict_count": int(scheduling_stats["conflict_count"]),
                "scheduling_conflict_penalty": float(
                    scheduling_stats["conflict_penalty"]
                ),
                "scheduling_energy_consumption": float(
                    scheduling_stats["energy_consumption"]
                ),
                "scheduling_energy_penalty": float(
                    scheduling_stats["energy_penalty"]
                ),
                "total_arrivals": float(arrival_stats["total_arrivals"]),
                "queue_overflow": float(arrival_stats["queue_overflow"]),
                "total_queue_length": float(np.sum(truth.queue_lengths)),
                "scheduling_served_data": float(scheduling_stats["served_data"]),
                "service_rate": float(scheduling_stats["service_rate"]),
                "weighted_demand_satisfaction": float(
                    scheduling_stats["weighted_demand_satisfaction"]
                ),
                "high_priority_service_rate": float(
                    scheduling_stats["high_priority_service_rate"]
                ),
                "service_energy_consumption": float(
                    scheduling_stats["energy_consumption"]
                ),
                "per_agent_served_data": scheduling_stats["service_by_agent"].copy(),
                "service_reward": float(service_reward),
                "queue_backlog_penalty": float(queue_backlog_penalty),
                "per_agent_scheduling_difference": scheduling_stats[
                    "difference_by_agent"
                ].copy(),
                "per_agent_scheduled_task": scheduling_stats[
                    "scheduled_task_by_agent"
                ].copy(),
                "shared_resource_reward": float(shared_reward),
                "sensing_difference_mean": float(np.mean(sensing_difference)),
                "counterfactual_team_quality_without_agent": (
                    counterfactual_quality_without_agent.copy()
                ),
            }
        )
        if per_agent_rewards is not None:
            info.update(
                {
                    "per_agent_rewards": per_agent_rewards.copy(),
                    "per_agent_sensing_difference": sensing_difference.copy(),
                    "per_agent_fusion_gain_by_sender": fusion_stats[
                        "quality_gain_by_sender"
                    ].copy(),
                    "per_agent_messages_attempted": messages_attempted_by_sender.copy(),
                    "reward_scheduling": float(np.mean(scheduling_rewards)),
                    "reward_counterfactual_sensing": float(np.mean(sensing_rewards)),
                    "reward_sender_fusion": float(np.mean(fusion_rewards)),
                    "counterfactual_movement_penalty": float(
                        np.mean(movement_costs)
                    ),
                    "counterfactual_sensing_penalty": float(np.mean(sensing_costs)),
                    "counterfactual_repeat_penalty": float(np.mean(repeat_costs)),
                    "counterfactual_communication_penalty": float(
                        np.mean(communication_costs)
                    ),
                    "counterfactual_scheduling_energy_penalty": float(
                        np.mean(scheduling_stats["energy_penalty_by_agent"])
                    ),
                    "counterfactual_scheduling_conflict_penalty": float(
                        np.mean(scheduling_stats["conflict_penalty_by_agent"])
                    ),
                    "per_agent_reward_std": float(np.std(per_agent_rewards)),
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
                beliefs.local_quality(i, use_task_priorities=False),
            ],
            dtype=np.float32,
        )
        task_features = np.zeros((self.cfg.cognition_max_task_slots, 12), dtype=np.float32)
        for slot, task_id in enumerate(visible):
            rel = truth.positions_xy[task_id] - position
            link_quality = float(np.exp(-distances[task_id] / max(self.cfg.sensing_radius, 1e-6)))
            task_features[slot] = np.array(
                [
                    rel[0] / self.cfg.obs_radius,
                    rel[1] / self.cfg.obs_radius,
                    truth.band_ids[task_id] / max(self.cfg.cognition_num_bands - 1, 1),
                    beliefs.estimates[i, task_id],
                    beliefs.uncertainties[i, task_id],
                    beliefs.aoi[i, task_id] / max(self.cfg.task_max_aoi, 1e-6),
                    beliefs.confidence[i, task_id],
                    beliefs.demand_estimates[i, task_id],
                    beliefs.demand_uncertainties[i, task_id],
                    beliefs.demand_aoi[i, task_id] / max(self.cfg.task_max_aoi, 1e-6),
                    beliefs.demand_confidence[i, task_id],
                    link_quality,
                ],
                dtype=np.float32,
            )

        neighbor_features = np.zeros((self.cfg.max_obs_uavs, 12), dtype=np.float32)
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
                message.demand_estimate,
                message.demand_uncertainty,
                message.demand_confidence,
                min(max(message.demand_aoi, 0.0), self.cfg.task_max_aoi)
                / max(self.cfg.task_max_aoi, 1e-6),
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
                truth.demand_levels,
                truth.priorities,
                beliefs.estimates.flatten(),
                beliefs.uncertainties.flatten(),
                (beliefs.aoi / max(self.cfg.task_max_aoi, 1e-6)).flatten(),
                beliefs.confidence.flatten(),
                beliefs.demand_estimates.flatten(),
                beliefs.demand_uncertainties.flatten(),
                (beliefs.demand_aoi / max(self.cfg.task_max_aoi, 1e-6)).flatten(),
                beliefs.demand_confidence.flatten(),
                self.uav_positions.flatten(),
                self.remaining_time / max(self.cfg.uav_max_time, 1e-6),
                np.array(
                    [self.current_step / max(self.cfg.max_steps, 1)], dtype=np.float32
                ),
            ]
        ).astype(np.float32)

    def _decode_sensing_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._decode_task_actions(actions, self.SENSE_ACTION_START)

    def _decode_schedule_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cfg.cognition_enable_scheduling:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        return self._decode_task_actions(actions, self._schedule_action_start())

    def _decode_task_actions(
        self,
        actions: np.ndarray,
        action_start: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        agent_ids: List[int] = []
        task_ids: List[int] = []
        for agent_id, action in enumerate(actions):
            slot = int(action) - int(action_start)
            if slot < 0 or slot >= len(self._slot_task_indices[agent_id]):
                continue
            agent_ids.append(agent_id)
            task_ids.append(int(self._slot_task_indices[agent_id][slot]))
        return np.asarray(agent_ids, dtype=np.int64), np.asarray(task_ids, dtype=np.int64)

    def _schedule_action_start(self) -> int:
        return int(self.SENSE_ACTION_START + self.cfg.cognition_max_task_slots)

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
                        demand_estimate=float(beliefs.demand_estimates[sender_id, task_id]),
                        demand_uncertainty=float(
                            beliefs.demand_uncertainties[sender_id, task_id]
                        ),
                        demand_confidence=float(
                            beliefs.demand_confidence[sender_id, task_id]
                        ),
                        demand_aoi=float(beliefs.demand_aoi[sender_id, task_id]),
                        queue_estimate=float(beliefs.queue_estimates[sender_id, task_id]),
                        queue_uncertainty=float(
                            beliefs.queue_uncertainties[sender_id, task_id]
                        ),
                        queue_confidence=float(
                            beliefs.queue_confidence[sender_id, task_id]
                        ),
                        queue_aoi=float(beliefs.queue_aoi[sender_id, task_id]),
                        arrival_estimate=float(
                            beliefs.arrival_estimates[sender_id, task_id]
                        ),
                    )
                )
        return messages

    def _deliver_due_messages(self) -> List[CognitionMessage]:
        if not self.cfg.cognition_enable_communication:
            return []
        return self._require_communication().deliver(self.current_step)

    def _fuse_messages(self, messages: List[CognitionMessage]) -> Dict[str, Any]:
        _, beliefs = self._require_state()
        accepted = 0
        quality_gain = 0.0
        quality_gain_by_sender = np.zeros((self.num_agents,), dtype=np.float32)
        accepted_by_sender = np.zeros((self.num_agents,), dtype=np.float32)
        for message in messages:
            result = beliefs.fuse_neighbor_message(
                receiver_id=message.receiver_id,
                task_id=message.task_id,
                estimate=message.estimate,
                uncertainty=message.uncertainty,
                confidence=message.confidence,
                message_aoi=message.aoi,
                demand_estimate=message.demand_estimate,
                demand_uncertainty=message.demand_uncertainty,
                demand_confidence=message.demand_confidence,
                demand_aoi=message.demand_aoi,
                queue_estimate=message.queue_estimate,
                queue_uncertainty=message.queue_uncertainty,
                queue_confidence=message.queue_confidence,
                queue_aoi=message.queue_aoi,
                arrival_estimate=message.arrival_estimate,
                source_update_step=message.created_step,
                current_step=self.current_step,
                confidence_threshold=self.cfg.cognition_fusion_confidence_threshold,
                freshness_decay=self.cfg.cognition_freshness_decay,
            )
            accepted += int(result["accepted"])
            quality_gain += float(result["quality_gain"])
            quality_gain_by_sender[message.sender_id] += float(result["quality_gain"])
            accepted_by_sender[message.sender_id] += float(result["accepted"])
            self._received_message_cache[message.receiver_id][message.sender_id] = (
                message,
                bool(result["accepted"]),
            )
        return {
            "accepted": float(accepted),
            "quality_gain": float(quality_gain),
            "quality_gain_by_sender": quality_gain_by_sender,
            "accepted_by_sender": accepted_by_sender,
        }

    def _per_agent_quality(self) -> np.ndarray:
        _, beliefs = self._require_state()
        return np.fromiter(
            (beliefs.local_quality(i) for i in range(self.num_agents)),
            dtype=np.float32,
            count=self.num_agents,
        )

    def _execute_scheduling(
        self,
        schedule_agents: np.ndarray,
        schedule_tasks: np.ndarray,
    ) -> Dict[str, Any]:
        truth, _ = self._require_state()
        assignments = np.full((self.num_agents,), -1, dtype=np.int64)
        if self.cfg.cognition_enable_scheduling and schedule_tasks.size:
            assignments[schedule_agents] = schedule_tasks

        team_utility, service_by_agent, conflict_counts, capacity_by_agent = self._evaluate_schedule(
            assignments, use_truth=True
        )
        estimated_utility, _, _, _ = self._evaluate_schedule(
            assignments, use_truth=False
        )
        difference_by_agent = np.zeros((self.num_agents,), dtype=np.float32)
        for agent_id in np.flatnonzero(assignments >= 0):
            counterfactual = assignments.copy()
            counterfactual[agent_id] = -1
            counterfactual_utility, _, _, _ = self._evaluate_schedule(
                counterfactual, use_truth=True
            )
            difference_by_agent[agent_id] = max(
                float(team_utility - counterfactual_utility), 0.0
            )

        queue_before = truth.queue_lengths.copy()
        served_by_task = np.zeros((len(truth),), dtype=np.float32)
        for agent_id in np.flatnonzero(assignments >= 0):
            served_by_task[int(assignments[agent_id])] += service_by_agent[agent_id]
        served_data = truth.apply_service(served_by_task)
        satisfaction_by_task = np.divide(
            served_by_task,
            np.maximum(queue_before, 1e-6),
            out=np.zeros_like(served_by_task),
            where=queue_before > 1e-6,
        )
        priority_weights = truth.priorities
        weighted_demand_satisfaction = float(
            np.sum(priority_weights * satisfaction_by_task)
            / max(float(np.sum(priority_weights)), 1e-6)
        )
        high_priority = priority_weights >= np.percentile(priority_weights, 75.0)
        high_priority_service_rate = float(
            np.mean(satisfaction_by_task[high_priority])
            if np.any(high_priority) else 0.0
        )
        service_rate = float(
            np.clip(served_data / max(float(np.sum(queue_before)), 1e-6), 0.0, 1.0)
        )

        energy_consumption_by_agent = np.zeros((self.num_agents,), dtype=np.float32)
        for agent_id in np.flatnonzero(assignments >= 0):
            if service_by_agent[agent_id] <= 1e-6:
                continue
            consumed = min(
                float(self.cfg.cognition_service_energy_cost),
                float(self.remaining_time[agent_id]),
            )
            self.remaining_time[agent_id] = max(
                self.remaining_time[agent_id] - consumed, 0.0
            )
            energy_consumption_by_agent[agent_id] = consumed

        energy_penalty_by_agent = (
            self.cfg.reward_weight_movement_cost
            * energy_consumption_by_agent
            / max(self.cfg.step_size(), 1e-6)
        ).astype(np.float32)
        conflict_penalty_by_agent = (
            self.cfg.cognition_scheduling_conflict_penalty
            * conflict_counts
        ).astype(np.float32)
        return {
            "team_utility": float(team_utility),
            "estimated_team_utility": float(estimated_utility),
            "service_by_agent": service_by_agent,
            "capacity_by_agent": capacity_by_agent,
            "served_by_task": served_by_task,
            "served_data": float(served_data),
            "service_rate": service_rate,
            "weighted_demand_satisfaction": weighted_demand_satisfaction,
            "high_priority_service_rate": high_priority_service_rate,
            "difference_by_agent": difference_by_agent,
            "scheduled_task_by_agent": assignments,
            "scheduled_count": int(np.count_nonzero(assignments >= 0)),
            "conflict_counts": conflict_counts,
            "conflict_count": int(np.sum(conflict_counts > 0.0)),
            "conflict_penalty_by_agent": conflict_penalty_by_agent,
            "conflict_penalty": float(np.mean(conflict_penalty_by_agent)),
            "energy_consumption_by_agent": energy_consumption_by_agent,
            "energy_consumption": float(np.sum(energy_consumption_by_agent)),
            "energy_penalty_by_agent": energy_penalty_by_agent,
            "energy_penalty": float(np.mean(energy_penalty_by_agent)),
        }

    def _sample_business_arrivals(self) -> np.ndarray:
        truth, _ = self._require_state()
        demand_noise = self.rng.normal(
            0.0,
            self.cfg.cognition_arrival_noise_std,
            size=truth.demand_levels.shape,
        ).astype(np.float32)
        truth.demand_levels[:] = np.clip(
            truth.demand_levels + demand_noise,
            0.0,
            1.0,
        )
        expected = truth.arrival_rates * (0.5 + truth.demand_levels)
        arrival_noise = self.rng.normal(
            0.0,
            self.cfg.cognition_arrival_noise_std,
            size=expected.shape,
        ).astype(np.float32)
        return np.maximum(expected + arrival_noise, 0.0).astype(np.float32)

    def _evaluate_schedule(
        self,
        assignments: np.ndarray,
        *,
        use_truth: bool,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        truth, beliefs = self._require_state()
        assignments = np.asarray(assignments, dtype=np.int64)
        conflict_counts = np.zeros((self.num_agents,), dtype=np.float32)
        assigned_agents = np.flatnonzero(assignments >= 0)
        for left, agent_id in enumerate(assigned_agents):
            task_id = int(assignments[agent_id])
            for other_id in assigned_agents[left + 1:]:
                other_task_id = int(assignments[other_id])
                same_band = truth.band_ids[task_id] == truth.band_ids[other_task_id]
                task_distance = float(
                    np.linalg.norm(
                        truth.positions_xy[task_id] - truth.positions_xy[other_task_id]
                    )
                )
                if same_band and task_distance <= self.cfg.cognition_interference_radius:
                    conflict_counts[agent_id] += 1.0
                    conflict_counts[other_id] += 1.0

        capacity_by_agent = np.zeros((self.num_agents,), dtype=np.float32)
        service_by_agent = np.zeros((self.num_agents,), dtype=np.float32)
        for agent_id in assigned_agents:
            task_id = int(assignments[agent_id])
            distance = float(
                np.linalg.norm(
                    self.uav_positions[agent_id, :2] - truth.positions_xy[task_id]
                )
            )
            link_quality = float(
                np.exp(-distance / max(self.cfg.sensing_radius, 1e-6))
            )
            energy_factor = float(
                np.clip(
                    self.remaining_time[agent_id]
                    / max(self.cfg.uav_max_time, 1e-6),
                    0.0,
                    1.0,
                )
            )
            availability_value = (
                1.0 - float(truth.true_states[task_id])
                if use_truth
                else 1.0 - float(beliefs.estimates[agent_id, task_id])
            )
            raw_capacity = (
                self.cfg.cognition_base_service_rate
                * link_quality
                * np.clip(availability_value, 0.0, 1.0)
                * energy_factor
            )
            capacity_by_agent[agent_id] = float(
                min(raw_capacity, self.cfg.cognition_max_service_per_step)
            )

        effective_capacity = capacity_by_agent / (1.0 + conflict_counts)
        for task_id in np.unique(assignments[assigned_agents]):
            task_agents = assigned_agents[assignments[assigned_agents] == task_id]
            total_capacity = float(np.sum(effective_capacity[task_agents]))
            queue_value = (
                float(truth.queue_lengths[task_id])
                if use_truth
                else float(np.max(beliefs.queue_estimates[:, task_id]))
            )
            actual_service = min(max(queue_value, 0.0), total_capacity)
            if total_capacity > 1e-6:
                service_by_agent[task_agents] = (
                    effective_capacity[task_agents] / total_capacity * actual_service
                )

        if use_truth:
            queues = truth.queue_lengths
            weights = truth.priorities
        else:
            queues = np.max(beliefs.queue_estimates, axis=0)
            weights = np.ones((len(truth),), dtype=np.float32)
        served_by_task = np.zeros((len(truth),), dtype=np.float32)
        for agent_id in assigned_agents:
            served_by_task[int(assignments[agent_id])] += service_by_agent[agent_id]
        utility = float(
            np.sum(weights * np.divide(
                served_by_task,
                np.maximum(queues, 1e-6),
                out=np.zeros_like(served_by_task),
                where=queues > 1e-6,
            ))
            / max(float(np.sum(weights[queues > 1e-6])), 1e-6)
        )
        return utility, service_by_agent, conflict_counts, capacity_by_agent

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
        sense_end = self.SENSE_ACTION_START + self.cfg.cognition_max_task_slots
        for i, visible in enumerate(self._slot_task_indices):
            mask[i, self.SENSE_ACTION_START + len(visible):sense_end] = 0.0
            if self.cfg.cognition_enable_scheduling:
                mask[i, self._schedule_action_start() + len(visible):] = 0.0
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
            "mean_estimation_error": beliefs.mean_estimation_error(
                truth.true_states, truth.demand_levels
            ),
            "mean_spectrum_estimation_error": beliefs.mean_spectrum_estimation_error(
                truth.true_states
            ),
            "mean_demand_estimation_error": beliefs.mean_demand_estimation_error(
                truth.demand_levels
            ),
            "move_distance_total_step": float(np.sum(move_distances)),
            "total_distance_per_uav": self.total_distance_per_uav.copy(),
            "remaining_time": self.remaining_time.copy(),
            "active_uav_count": int(np.sum(self.active_mask)),
            "step": int(self.current_step),
            "termination_reason": termination_reason,
        }

    def _compute_local_obs_dim(self) -> int:
        return int(6 + self.cfg.cognition_max_task_slots * 12 + self.cfg.max_obs_uavs * 12)

    def _require_state(self) -> Tuple[TaskTruthBatch, LocalBeliefBatch]:
        if self.task_truth is None or self.local_beliefs is None:
            raise RuntimeError("Call reset before accessing task state.")
        return self.task_truth, self.local_beliefs

    def _require_communication(self) -> NeighborCommunicationModel:
        if self.communication_model is None:
            raise RuntimeError("Call reset before accessing communication state.")
        return self.communication_model
