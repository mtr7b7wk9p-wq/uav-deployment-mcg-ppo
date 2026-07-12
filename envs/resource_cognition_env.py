"""Independent local resource-cognition environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.scenario_config import ScenarioConfig
from envs.geometry import (
    clip_point_to_ring,
    make_uav_init_positions_center,
    make_uav_init_positions_circle,
    sample_points_in_annulus,
    set_random_seed,
)
from envs.task_model import TaskStateBatch


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
        self.tasks: Optional[TaskStateBatch] = None
        self._slot_task_indices: List[np.ndarray] = []
        self._last_info: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = set_random_seed(seed)
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
        self.tasks = TaskStateBatch(
            positions,
            band_ids,
            true_states,
            priorities,
            initial_uncertainty=self.cfg.task_initial_uncertainty,
            initial_aoi=self.cfg.task_initial_aoi,
            max_aoi=self.cfg.task_max_aoi,
        )
        self.remaining_time[:] = self.cfg.uav_max_time
        self.total_distance_per_uav[:] = 0.0
        self._last_info = self._build_info(
            reward=0.0,
            uncertainty_gain=0.0,
            aoi_gain=0.0,
            repeat_ratio=0.0,
            termination_reason="running",
        )
        return self._build_output()

    def step(self, actions: List[int] | np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.tasks is None:
            raise RuntimeError("Call reset before step.")
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.num_agents,):
            raise ValueError(f"actions must have shape ({self.num_agents},).")
        if np.any(action_array < 0) or np.any(action_array >= self.action_size):
            raise ValueError("action out of range.")

        selected_tasks = [
            int(self._slot_task_indices[i][int(action) - self.MOVE_ACTIONS])
            for i, action in enumerate(action_array)
            if int(action) >= self.MOVE_ACTIONS
            and int(action) - self.MOVE_ACTIONS < len(self._slot_task_indices[i])
        ]
        unique_tasks, counts = np.unique(selected_tasks, return_counts=True) if selected_tasks else (
            np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        )
        repeat_count = int(np.sum(np.maximum(counts - 1, 0)))

        self.current_step += 1
        self.tasks.age(self.cfg.cognition_aoi_increment)
        move_distances = self._apply_movement(action_array)
        if unique_tasks.size:
            noise = self.rng.normal(
                0.0,
                self.cfg.cognition_observation_noise_std,
                size=unique_tasks.shape,
            ).astype(np.float32)
            observations = self.tasks.true_states[unique_tasks] + noise
        else:
            observations = None
        sensing_gain = self.tasks.sense(
            unique_tasks,
            uncertainty_reduction=self.cfg.cognition_task_uncertainty_reduction,
            noisy_observations=observations,
        )

        repeat_ratio = float(repeat_count / max(len(self.tasks), 1))
        active_count = max(int(np.sum(self.active_mask)), 1)
        movement_cost = float(
            np.sum(move_distances) / max(active_count * self.cfg.step_size(), 1e-6)
        )
        sensing_cost = self.cfg.cognition_sensing_cost * float(unique_tasks.size)
        repeat_penalty = self.cfg.cognition_repeat_penalty * repeat_ratio
        reward = float(
            self.cfg.reward_weight_uncertainty_gain * sensing_gain["uncertainty_gain"]
            + self.cfg.reward_weight_aoi_gain * sensing_gain["aoi_gain"]
            - sensing_cost
            - repeat_penalty
            - self.cfg.reward_weight_movement_cost * movement_cost
        )
        progress = sensing_gain["uncertainty_gain"] + sensing_gain["aoi_gain"]
        self.no_improve_steps = 0 if progress > 1e-6 else self.no_improve_steps + 1
        done, reason = self._check_done()
        info = self._build_info(
            reward=reward,
            uncertainty_gain=sensing_gain["uncertainty_gain"],
            aoi_gain=sensing_gain["aoi_gain"],
            repeat_ratio=repeat_ratio,
            termination_reason=reason,
        )
        info.update(
            {
                "selected_task_count": int(unique_tasks.size),
                "repeat_task_count": repeat_count,
                "movement_cost": movement_cost,
                "sensing_cost": float(sensing_cost),
                "repeat_penalty": float(repeat_penalty),
            }
        )
        self._last_info = info
        return self._build_output(), reward, done, info

    def get_local_obs(self, agent_id: int) -> np.ndarray:
        if self.tasks is None:
            raise RuntimeError("Call reset before get_local_obs.")
        if not 0 <= int(agent_id) < self.num_agents:
            raise ValueError("agent_id out of range.")
        i = int(agent_id)
        position = self.uav_positions[i, :2]
        distances = np.linalg.norm(self.tasks.positions_xy - position[None, :], axis=1)
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
                self.tasks.cognitive_quality(),
            ],
            dtype=np.float32,
        )
        task_features = np.zeros((self.cfg.cognition_max_task_slots, 7), dtype=np.float32)
        for slot, task_idx in enumerate(visible):
            rel = self.tasks.positions_xy[task_idx] - position
            task_features[slot] = np.array(
                [
                    rel[0] / self.cfg.obs_radius,
                    rel[1] / self.cfg.obs_radius,
                    self.tasks.band_ids[task_idx] / max(self.cfg.cognition_num_bands - 1, 1),
                    self.tasks.uncertainty[task_idx],
                    self.tasks.aoi[task_idx] / max(self.cfg.task_max_aoi, 1e-6),
                    self.tasks.priorities[task_idx] / max(self.cfg.task_priority_max, 1e-6),
                    self.tasks.confidence[task_idx],
                ],
                dtype=np.float32,
            )
        neighbor_features = np.zeros((self.cfg.max_obs_uavs, 4), dtype=np.float32)
        neighbor_ids = [j for j in range(self.num_agents) if j != i]
        neighbor_ids.sort(key=lambda j: float(np.linalg.norm(self.uav_positions[j, :2] - position)))
        for slot, j in enumerate(neighbor_ids[: self.cfg.max_obs_uavs]):
            rel = self.uav_positions[j, :2] - position
            neighbor_features[slot] = [
                rel[0] / self.cfg.obs_radius,
                rel[1] / self.cfg.obs_radius,
                self.remaining_time[j] / max(self.cfg.uav_max_time, 1e-6),
                1.0 if self.active_mask[j] else 0.0,
            ]
        return np.concatenate([self_features, task_features.flatten(), neighbor_features.flatten()])

    def get_global_state(self) -> np.ndarray:
        if self.tasks is None:
            raise RuntimeError("Call reset before get_global_state.")
        return np.concatenate(
            [
                self.tasks.positions_xy.flatten(),
                self.tasks.band_ids.astype(np.float32),
                self.tasks.true_states,
                self.tasks.estimate,
                self.tasks.uncertainty,
                self.tasks.aoi / max(self.cfg.task_max_aoi, 1e-6),
                self.tasks.priorities,
                self.uav_positions.flatten(),
                self.remaining_time / max(self.cfg.uav_max_time, 1e-6),
                np.array([self.current_step / max(self.cfg.max_steps, 1)], dtype=np.float32),
            ]
        ).astype(np.float32)

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
            [xy.astype(np.float32), np.full((self.num_agents, 1), self.cfg.uav_init_height, dtype=np.float32)],
            axis=1,
        )
        self.active_mask[:] = True
        self.remaining_time[:] = self.cfg.uav_max_time
        self._slot_task_indices = [np.zeros((0,), dtype=np.int64) for _ in range(self.num_agents)]

    def _apply_movement(self, actions: np.ndarray) -> np.ndarray:
        distances = np.zeros((self.num_agents,), dtype=np.float32)
        deltas = self.cfg.action_to_delta_xy()
        for i, action in enumerate(actions):
            if int(action) >= self.MOVE_ACTIONS or not self.active_mask[i] or self.remaining_time[i] <= 0.0:
                continue
            dx, dy = deltas[int(action)]
            old = self.uav_positions[i, :2].copy()
            x, y = clip_point_to_ring(float(old[0] + dx), float(old[1] + dy), self.cfg.r_safe, self.cfg.r_disaster)
            distance = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - old))
            self.uav_positions[i, :2] = [x, y]
            self.total_distance_per_uav[i] += distance
            self.remaining_time[i] = max(self.remaining_time[i] - self.cfg.dt, 0.0)
            distances[i] = distance
        return distances

    def _check_done(self) -> Tuple[bool, str]:
        if self.current_step >= self.cfg.max_steps:
            return True, "max_steps"
        if self.tasks is not None and float(np.average(self.tasks.uncertainty, weights=self.tasks.priorities)) <= self.cfg.trusted_sensing_uncertainty_target:
            return True, "uncertainty_target"
        if not np.any(self.remaining_time > 0.0):
            return True, "energy_timeout"
        if self.no_improve_steps >= self.cfg.stagnation_patience:
            return True, "stagnation"
        return False, "running"

    def _build_output(self) -> Dict[str, Any]:
        local_obs = np.stack([self.get_local_obs(i) for i in range(self.num_agents)], axis=0)
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
        termination_reason: str,
    ) -> Dict[str, Any]:
        if self.tasks is None:
            raise RuntimeError("Task state is not initialized.")
        weighted_uncertainty = float(np.average(self.tasks.uncertainty, weights=self.tasks.priorities))
        weighted_aoi = float(np.average(self.tasks.aoi, weights=self.tasks.priorities))
        return {
            "reward_total": float(reward),
            "uncertainty_gain": float(uncertainty_gain),
            "aoi_gain": float(aoi_gain),
            "reward_uncertainty_gain": float(uncertainty_gain),
            "reward_aoi_gain": float(aoi_gain),
            "repeat_sensing_ratio": float(repeat_ratio),
            "mean_task_uncertainty": weighted_uncertainty,
            "mean_task_aoi": weighted_aoi,
            "cognitive_quality": self.tasks.cognitive_quality(),
            "active_uav_count": int(np.sum(self.active_mask)),
            "step": int(self.current_step),
            "termination_reason": termination_reason,
        }

    def _compute_local_obs_dim(self) -> int:
        return int(6 + self.cfg.cognition_max_task_slots * 7 + self.cfg.max_obs_uavs * 4)
