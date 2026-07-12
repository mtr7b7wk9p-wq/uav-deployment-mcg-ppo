from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.scenario_config import ScenarioConfig
from envs.channel import AirToGroundChannelConfig, build_cover_matrix, nearest_feasible_uav
from envs.geometry import (
    clip_point_to_ring,
    make_uav_init_positions_center,
    make_uav_init_positions_circle,
    pad_or_trim_rows,
    sample_points_in_annulus,
    sample_points_in_annulus_weighted,
    sample_user_points_clustered,
    sample_user_points_mixed,
    set_random_seed,
    validate_user_distribution,
)


class DisasterDeploymentEnv:
    """
    Activation-aware deployment environment.

    New objective:
    Given maximum candidate UAVs M_max, the policy should activate as few UAVs as
    possible while achieving high QoS coverage.
    """

    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.cfg = config or ScenarioConfig()
        self.cfg.validate()
        self.rng = set_random_seed(self.cfg.seed)

        self.bs_pos = np.array(self.cfg.bs_pos, dtype=np.float32)
        self.num_agents = self.cfg.max_candidate_uavs

        self.channel_cfg = AirToGroundChannelConfig(
            mode="simplified" if self.cfg.use_simplified_qos else "paper_atg",
            simplified_coverage_radius=self.cfg.simplified_coverage_radius,
            carrier_freq_ghz=2.0,
            qos_threshold_db=self.cfg.qos_threshold_db,
            los_a=9.61,
            los_b=0.16,
            eta_los_db=1.0,
            eta_nlos_db=20.0,
        )

        self.current_step = 0
        self.elapsed_time = 0.0
        self.no_improve_steps = 0

        self.ue_positions = np.zeros((self.cfg.num_users, 2), dtype=np.float32)
        self.uav_positions = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.active_mask = np.zeros((self.num_agents,), dtype=bool)
        self.just_activated_mask = np.zeros((self.num_agents,), dtype=bool)

        self.remaining_time = np.full((self.num_agents,), self.cfg.uav_max_time, dtype=np.float32)
        self.total_distance_per_uav = np.zeros((self.num_agents,), dtype=np.float32)

        self.covered_mask = np.zeros((self.cfg.num_users,), dtype=bool)
        self.cover_count_per_user = np.zeros((self.cfg.num_users,), dtype=np.int32)
        self.assigned_uav_idx = np.full((self.cfg.num_users,), -1, dtype=np.int32)
        self.cover_mat = np.zeros((self.cfg.num_users, self.num_agents), dtype=bool)
        self.cover_aux_metric = np.zeros((self.cfg.num_users, self.num_agents), dtype=np.float32)

        self.coverage_ratio = 0.0
        self.last_coverage_count = 0
        self.prev_mean_target_distance = 0.0
        self.prev_dispersion_penalty = 0.0

        self.user_cluster_centers = np.zeros((0, 2), dtype=np.float32)
        self.user_cluster_ids = np.full((self.cfg.num_users,), -1, dtype=np.int32)
        self.user_is_clustered = np.zeros((self.cfg.num_users,), dtype=bool)
        self.user_distribution_stats: Dict[str, Any] = {}

        self.task_uncertainty = np.zeros((self.cfg.num_users,), dtype=np.float32)
        self.task_aoi = np.zeros((self.cfg.num_users,), dtype=np.float32)
        self.task_priority = np.ones((self.cfg.num_users,), dtype=np.float32)
        self.task_sensing_count = np.zeros((self.cfg.num_users,), dtype=np.int32)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = set_random_seed(seed)

        self.current_step = 0
        self.elapsed_time = 0.0
        self.no_improve_steps = 0

        self._sample_user_positions()
        self._reset_sensing_state()

        self._init_uavs()
        self.remaining_time[:] = self.cfg.uav_max_time
        self.total_distance_per_uav[:] = 0.0

        self.active_mask[:] = False
        self.just_activated_mask[:] = False
        if self.cfg.initially_active_uavs > 0:
            self.active_mask[: self.cfg.initially_active_uavs] = True

        self._update_coverage()
        self.last_coverage_count = int(self.covered_mask.sum())
        self.prev_mean_target_distance = float(self._compute_mean_target_distance())
        self.prev_dispersion_penalty = float(self._compute_dispersion_penalty())

        return self._build_reset_output()

    def step(self, actions: List[int] | np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._validate_actions(actions)
        self.current_step += 1
        self.elapsed_time += self.cfg.dt
        self.just_activated_mask[:] = False

        prev_coverage_count = int(self.covered_mask.sum())
        prev_covered_mask = self.covered_mask.copy()
        prev_cover_count_per_user = self.cover_count_per_user.copy()
        prev_cover_mat = self.cover_mat.copy()
        prev_uav_positions = self.uav_positions.copy()

        move_distances, out_of_bound_count, new_activation_count = self._apply_actions(actions)
        self._update_coverage()
        sensing_stats = self._update_sensing_state()

        new_coverage_count = int(self.covered_mask.sum())
        delta_new_cover = new_coverage_count - prev_coverage_count
        overlap_count = int(np.sum(self.cover_count_per_user > 1))
        timeout_flag = bool(np.any(self.remaining_time[self.active_mask] <= 0.0)) if np.any(self.active_mask) else False

        reward, reward_breakdown = self._compute_reward(
            delta_new_cover=delta_new_cover,
            move_distances=move_distances,
            overlap_count=overlap_count,
            timeout_flag=timeout_flag,
            out_of_bound_count=out_of_bound_count,
            prev_covered_mask=prev_covered_mask,
            prev_cover_count_per_user=prev_cover_count_per_user,
            prev_cover_mat=prev_cover_mat,
            prev_uav_positions=prev_uav_positions,
            sensing_stats=sensing_stats,
        )

        if self._should_use_trusted_sensing():
            cognitive_progress = (
                float(sensing_stats["uncertainty_reduction"])
                + float(sensing_stats["aoi_reduction"])
            )
            self.no_improve_steps = 0 if cognitive_progress > 1e-6 else self.no_improve_steps + 1
        elif new_coverage_count > self.last_coverage_count:
            self.no_improve_steps = 0
            self.last_coverage_count = new_coverage_count
        else:
            self.no_improve_steps += 1

        done, termination_reason = self._check_done()

        obs = self._build_step_output()
        info = self._build_info(
            delta_new_cover=delta_new_cover,
            overlap_count=overlap_count,
            move_distances=move_distances,
            out_of_bound_count=out_of_bound_count,
            timeout_flag=timeout_flag,
            new_activation_count=new_activation_count,
            reward_breakdown=reward_breakdown,
            sensing_stats=sensing_stats,
            termination_reason=termination_reason,
        )
        return obs, float(reward), done, info

    def get_global_state(self) -> np.ndarray:
        extra = np.array([
            float(self.current_step),
            float(self.elapsed_time),
            float(self.coverage_ratio),
            float(np.mean(self.active_mask.astype(np.float32))),
            float(np.mean((~self.covered_mask).astype(np.float32))),
        ], dtype=np.float32)
        return np.concatenate([
            self.ue_positions.flatten(),
            self.uav_positions.flatten(),
            self.active_mask.astype(np.float32),
            self.covered_mask.astype(np.float32),
            self.remaining_time.astype(np.float32),
            self.total_distance_per_uav.astype(np.float32),
            extra,
        ]).astype(np.float32)

    def get_local_obs(self, agent_id: int) -> np.ndarray:
        """
        统一 observation 入口：
        - ppo_main 走原始局部观测
        - mcg_ppo 走增强局部观测
        """
        if self._should_use_enhanced_obs():
            return self.get_local_obs_mcg(agent_id)
        return self.get_local_obs_ppo_main(agent_id)

    def get_local_obs_ppo_main(self, agent_id: int) -> np.ndarray:
        """
        保留 ppo_main 的原始 observation 逻辑，不做行为修改。
        """
        self._check_agent_id(agent_id)

        self_pos = self.uav_positions[agent_id]
        self_xy = self_pos[:2]

        ue_rel = self.ue_positions - self_xy[None, :]
        ue_dist = np.linalg.norm(ue_rel, axis=1)

        visible_user_idx = np.where(ue_dist <= self.cfg.obs_radius)[0]
        near_user_idx = visible_user_idx[np.argsort(ue_dist[visible_user_idx])][: self.cfg.max_obs_users]

        user_feats = []
        for idx in near_user_idx:
            if self._should_use_trusted_sensing():
                user_feats.append([
                    float(ue_rel[idx, 0] / max(self.cfg.obs_radius, 1e-6)),
                    float(ue_rel[idx, 1] / max(self.cfg.obs_radius, 1e-6)),
                    float(self.task_uncertainty[idx]),
                    float(self.task_aoi[idx] / max(self.cfg.task_max_aoi, 1e-6)),
                    float(self.task_priority[idx] / max(self.cfg.task_priority_max, 1e-6)),
                ])
            else:
                user_feats.append([
                    float(ue_rel[idx, 0] / max(self.cfg.obs_radius, 1e-6)),
                    float(ue_rel[idx, 1] / max(self.cfg.obs_radius, 1e-6)),
                    1.0 if self.covered_mask[idx] else 0.0,
                    1.0 if self.assigned_uav_idx[idx] == agent_id else 0.0,
                    float(self.cover_count_per_user[idx]) / max(float(self.cfg.max_obs_uavs), 1.0),
                ])

        user_arr = (
            np.array(user_feats, dtype=np.float32)
            if user_feats else np.zeros((0, 5), dtype=np.float32)
        )
        user_arr = pad_or_trim_rows(user_arr, self.cfg.max_obs_users, fill_value=0.0)

        other_feats = []
        nearby_uav_count = 0

        if self.cfg.use_neighbor_uav_obs:
            for j in range(self.num_agents):
                if j == agent_id:
                    continue

                rel_xy = self.uav_positions[j, :2] - self_xy
                d = float(np.linalg.norm(rel_xy))

                if d <= self.cfg.obs_radius:
                    nearby_uav_count += 1
                    neighbor_load = float(np.sum(self.assigned_uav_idx == j)) / max(float(self.cfg.num_users), 1.0)

                    other_feats.append([
                        float(rel_xy[0] / max(self.cfg.obs_radius, 1e-6)),
                        float(rel_xy[1] / max(self.cfg.obs_radius, 1e-6)),
                        float(d / max(self.cfg.obs_radius, 1e-6)),
                        float(self.remaining_time[j] / max(self.cfg.uav_max_time, 1e-6)),
                        neighbor_load,
                    ])

        other_arr = (
            np.array(other_feats, dtype=np.float32)
            if other_feats else np.zeros((0, 5), dtype=np.float32)
        )
        other_arr = pad_or_trim_rows(other_arr, self.cfg.max_obs_uavs, fill_value=0.0)

        local_visible_count = int(visible_user_idx.size)
        max_total_travel = max(float(self.cfg.step_size() * self.cfg.max_steps), 1e-6)

        if self._should_use_trusted_sensing() and local_visible_count > 0:
            self_context = [
                float(np.mean(self.task_uncertainty[visible_user_idx])),
                float(np.mean(self.task_aoi[visible_user_idx]) / max(self.cfg.task_max_aoi, 1e-6)),
                float(np.mean(self.task_priority[visible_user_idx]) / max(self.cfg.task_priority_max, 1e-6)),
                float(np.mean((self.task_sensing_count[visible_user_idx] > 0).astype(np.float32))),
            ]
        elif self._should_use_trusted_sensing():
            self_context = [0.0, 0.0, 0.0, 0.0]
        else:
            self_assigned_users = float(np.sum(self.assigned_uav_idx == agent_id))
            self_overlap_users = float(
                np.sum((self.assigned_uav_idx == agent_id) & (self.cover_count_per_user > 1))
            )
            local_uncovered_count = (
                int(np.sum(~self.covered_mask[visible_user_idx])) if local_visible_count > 0 else 0
            )
            self_context = [
                float(self_assigned_users / max(float(self.cfg.num_users), 1.0)),
                float(self_overlap_users / max(float(self.cfg.num_users), 1.0)),
                float(min(local_visible_count / max(float(self.cfg.max_obs_users), 1.0), 1.0)),
                float(local_uncovered_count / max(float(local_visible_count), 1.0)) if local_visible_count > 0 else 0.0,
            ]

        self_feat = np.array([
            float(self_pos[0] / max(self.cfg.r_disaster, 1e-6)),
            float(self_pos[1] / max(self.cfg.r_disaster, 1e-6)),
            float(self_pos[2] / max(self.cfg.uav_h_max, 1e-6)),
            float(self.remaining_time[agent_id] / max(self.cfg.uav_max_time, 1e-6)),
            float(self.total_distance_per_uav[agent_id] / max_total_travel),
            float(self.current_step / max(float(self.cfg.max_steps), 1.0)),
            *self_context,
            float(min(nearby_uav_count / max(float(self.cfg.max_obs_uavs), 1.0), 1.0)),
        ], dtype=np.float32)

        local_neighborhood_summary = self._compute_local_neighborhood_summary(agent_id)
        uncovered_guidance = self._compute_uncovered_guidance(agent_id)

        obs = np.concatenate(
            [
                self_feat,
                user_arr.flatten(),
                other_arr.flatten(),
                local_neighborhood_summary,
                uncovered_guidance,
            ],
            axis=0,
        ).astype(np.float32)

        return obs

    def get_local_obs_mcg(self, agent_id: int) -> np.ndarray:
        """
        MCG-PPO 的增强 observation：
        在不改 PPO 主体的前提下，在原 observation 后拼接固定长度统计摘要，
        保持完全分布式，只使用当前位置、局部可见用户、邻近 UAV 与局部重叠风险信息。
        """
        base_obs = self.get_local_obs_ppo_main(agent_id)
        enhanced_blocks = [base_obs]

        enhanced_blocks.append(self._compute_mcg_self_state_features(agent_id))
        enhanced_blocks.append(self._compute_mcg_user_summary_features(agent_id))
        enhanced_blocks.append(self._compute_mcg_neighbor_summary_features(agent_id))
        enhanced_blocks.append(self._compute_mcg_overlap_risk_features(agent_id))

        return np.concatenate(enhanced_blocks, axis=0).astype(np.float32)

    def _should_use_enhanced_obs(self) -> bool:
        return bool(getattr(self.cfg, "use_enhanced_obs", False))

    def _should_use_mcg_reward(self) -> bool:
        return bool(getattr(self.cfg, "use_mcg_reward", False))

    def _should_use_trusted_sensing(self) -> bool:
        return bool(getattr(self.cfg, "use_trusted_sensing", False))

    def _reset_sensing_state(self) -> None:
        self.task_sensing_count[:] = 0
        if not self._should_use_trusted_sensing():
            self.task_uncertainty[:] = 0.0
            self.task_aoi[:] = 0.0
            self.task_priority[:] = 1.0
            return

        self.task_uncertainty[:] = self.cfg.task_initial_uncertainty
        self.task_aoi[:] = self.cfg.task_initial_aoi
        self.task_priority[:] = self.rng.uniform(
            self.cfg.task_priority_min,
            self.cfg.task_priority_max,
            size=self.cfg.num_users,
        ).astype(np.float32)

    def _empty_sensing_stats(self) -> Dict[str, float]:
        return {
            "uncertainty_reduction": 0.0,
            "aoi_reduction": 0.0,
            "repeat_sensing_ratio": 0.0,
            "sensed_task_ratio": 0.0,
            "active_sensing_ratio": 0.0,
            "mean_task_uncertainty": 0.0,
            "mean_task_aoi": 0.0,
            "cognitive_quality": 0.0,
        }

    def _update_sensing_state(self) -> Dict[str, float]:
        if not self._should_use_trusted_sensing():
            return self._empty_sensing_stats()

        priority_sum = max(float(np.sum(self.task_priority)), 1e-6)
        previous_uncertainty = self.task_uncertainty.copy()
        aoi_before_sensing = np.minimum(self.task_aoi + 1.0, self.cfg.task_max_aoi)
        self.task_aoi[:] = aoi_before_sensing
        self.task_sensing_count[:] = 0

        active_idx = np.where(self.active_mask)[0]
        for agent_id in active_idx:
            distances = np.linalg.norm(self.ue_positions - self.uav_positions[agent_id, :2], axis=1)
            sensed_idx = np.where(distances <= self.cfg.sensing_radius)[0]
            self.task_sensing_count[sensed_idx] += 1

        sensed_mask = self.task_sensing_count > 0
        if np.any(sensed_mask):
            reduced_uncertainty = previous_uncertainty[sensed_mask] * (
                1.0 - self.cfg.task_uncertainty_reduction
            )
            self.task_uncertainty[sensed_mask] = np.maximum(
                reduced_uncertainty,
                self.cfg.task_min_uncertainty,
            )
            self.task_aoi[sensed_mask] = 0.0

        uncertainty_reduction = float(
            np.sum(self.task_priority * (previous_uncertainty - self.task_uncertainty)) / priority_sum
        )
        aoi_reduction = float(
            np.sum(self.task_priority * (aoi_before_sensing - self.task_aoi))
            / max(priority_sum * self.cfg.task_max_aoi, 1e-6)
        )
        repeat_sensing_count = int(np.sum(np.maximum(self.task_sensing_count - 1, 0)))
        repeat_sensing_ratio = float(repeat_sensing_count / max(self.cfg.num_users, 1))
        mean_task_uncertainty = float(np.sum(self.task_priority * self.task_uncertainty) / priority_sum)
        mean_task_aoi = float(np.sum(self.task_priority * self.task_aoi) / priority_sum)
        normalized_aoi = mean_task_aoi / max(self.cfg.task_max_aoi, 1e-6)
        cognitive_quality = float(np.clip(1.0 - 0.5 * (mean_task_uncertainty + normalized_aoi), 0.0, 1.0))

        return {
            "uncertainty_reduction": uncertainty_reduction,
            "aoi_reduction": aoi_reduction,
            "repeat_sensing_ratio": repeat_sensing_ratio,
            "sensed_task_ratio": float(np.mean(sensed_mask.astype(np.float32))),
            "active_sensing_ratio": float(active_idx.size / max(self.num_agents, 1)),
            "mean_task_uncertainty": mean_task_uncertainty,
            "mean_task_aoi": mean_task_aoi,
            "cognitive_quality": cognitive_quality,
        }

    def _get_visible_user_context(self, agent_id: int) -> Dict[str, Any]:
        self_xy = self.uav_positions[agent_id, :2]
        ue_rel = self.ue_positions - self_xy[None, :]
        ue_dist = np.linalg.norm(ue_rel, axis=1)

        visible_idx = np.where(ue_dist <= self.cfg.obs_radius)[0]
        visible_rel = ue_rel[visible_idx] if visible_idx.size > 0 else np.zeros((0, 2), dtype=np.float32)
        visible_dist = ue_dist[visible_idx] if visible_idx.size > 0 else np.zeros((0,), dtype=np.float32)
        visible_covered = self.covered_mask[visible_idx] if visible_idx.size > 0 else np.zeros((0,), dtype=bool)

        uncovered_mask = ~visible_covered if visible_idx.size > 0 else np.zeros((0,), dtype=bool)
        uncovered_rel = visible_rel[uncovered_mask] if visible_idx.size > 0 else np.zeros((0, 2), dtype=np.float32)
        uncovered_dist = visible_dist[uncovered_mask] if visible_idx.size > 0 else np.zeros((0,), dtype=np.float32)

        return {
            "self_xy": self_xy,
            "visible_idx": visible_idx,
            "visible_rel": visible_rel,
            "visible_dist": visible_dist,
            "visible_covered": visible_covered,
            "uncovered_mask": uncovered_mask,
            "uncovered_rel": uncovered_rel,
            "uncovered_dist": uncovered_dist,
        }

    def _get_neighbor_context(self, agent_id: int) -> Dict[str, Any]:
        self_xy = self.uav_positions[agent_id, :2]
        rel_list = []
        dist_list = []
        neighbor_ids = []

        if self.cfg.use_neighbor_uav_obs:
            for j in range(self.num_agents):
                if j == agent_id:
                    continue
                rel_xy = self.uav_positions[j, :2] - self_xy
                d = float(np.linalg.norm(rel_xy))
                if d <= self.cfg.obs_radius:
                    rel_list.append(rel_xy.astype(np.float32))
                    dist_list.append(d)
                    neighbor_ids.append(j)

        rel_arr = np.array(rel_list, dtype=np.float32) if rel_list else np.zeros((0, 2), dtype=np.float32)
        dist_arr = np.array(dist_list, dtype=np.float32) if dist_list else np.zeros((0,), dtype=np.float32)
        return {
            "neighbor_ids": neighbor_ids,
            "rel": rel_arr,
            "dist": dist_arr,
        }

    def _compute_mcg_self_state_features(self, agent_id: int) -> np.ndarray:
        if not getattr(self.cfg, "use_enhanced_obs", False):
            return np.zeros((7,), dtype=np.float32)

        ctx = self._get_visible_user_context(agent_id)
        self_xy = ctx["self_xy"]
        visible_idx = ctx["visible_idx"]

        if self._should_use_trusted_sensing():
            if visible_idx.size > 0:
                mean_uncertainty = float(np.mean(self.task_uncertainty[visible_idx]))
                mean_aoi = float(np.mean(self.task_aoi[visible_idx]) / max(self.cfg.task_max_aoi, 1e-6))
            else:
                mean_uncertainty = 0.0
                mean_aoi = 0.0

            center_rel = -self_xy
            radial_distance = float(np.linalg.norm(self_xy))
            return np.array([
                float(center_rel[0] / max(self.cfg.r_disaster, 1e-6)),
                float(center_rel[1] / max(self.cfg.r_disaster, 1e-6)),
                float(radial_distance / max(self.cfg.r_disaster, 1e-6)),
                float(self.current_step / max(float(self.cfg.max_steps), 1.0)),
                float(visible_idx.size / max(float(self.cfg.max_obs_users), 1.0)),
                mean_uncertainty,
                mean_aoi,
            ], dtype=np.float32)

        uncovered_mask = ctx["uncovered_mask"]

        local_visible_count = int(visible_idx.size)
        local_uncovered_count = int(np.sum(uncovered_mask)) if local_visible_count > 0 else 0
        local_covered_count = local_visible_count - local_uncovered_count

        center_rel = -self_xy
        radial_distance = float(np.linalg.norm(self_xy))

        return np.array([
            float(center_rel[0] / max(self.cfg.r_disaster, 1e-6)),
            float(center_rel[1] / max(self.cfg.r_disaster, 1e-6)),
            float(radial_distance / max(self.cfg.r_disaster, 1e-6)),
            float(self.current_step / max(float(self.cfg.max_steps), 1.0)),
            float(local_visible_count / max(float(self.cfg.max_obs_users), 1.0)),
            float(local_covered_count / max(float(self.cfg.max_obs_users), 1.0)),
            float(local_uncovered_count / max(float(self.cfg.max_obs_users), 1.0)),
        ], dtype=np.float32)
    def _compute_mcg_user_summary_features(self, agent_id: int) -> np.ndarray:
        if not getattr(self.cfg, "use_user_summary_features", False):
            return np.zeros((8,), dtype=np.float32)

        ctx = self._get_visible_user_context(agent_id)
        visible_idx = ctx["visible_idx"]
        visible_rel = ctx["visible_rel"]
        visible_dist = ctx["visible_dist"]

        if self._should_use_trusted_sensing():
            if visible_idx.size == 0:
                return np.zeros((8,), dtype=np.float32)

            normalized_aoi = self.task_aoi[visible_idx] / max(self.cfg.task_max_aoi, 1e-6)
            normalized_priority = self.task_priority[visible_idx] / max(self.cfg.task_priority_max, 1e-6)
            task_scores = normalized_priority * (self.task_uncertainty[visible_idx] + normalized_aoi)
            target_idx = int(np.argmax(task_scores))
            target_rel = visible_rel[target_idx]
            target_dist = float(visible_dist[target_idx])
            centroid_rel = np.average(visible_rel, axis=0, weights=np.maximum(task_scores, 1e-6))

            return np.array([
                float(visible_idx.size / max(float(self.cfg.max_obs_users), 1.0)),
                float(np.mean(self.task_uncertainty[visible_idx])),
                float(np.mean(normalized_aoi)),
                float(target_rel[0] / max(self.cfg.obs_radius, 1e-6)),
                float(target_rel[1] / max(self.cfg.obs_radius, 1e-6)),
                float(target_dist / max(self.cfg.obs_radius, 1e-6)),
                float(centroid_rel[0] / max(self.cfg.obs_radius, 1e-6)),
                float(centroid_rel[1] / max(self.cfg.obs_radius, 1e-6)),
            ], dtype=np.float32)

        uncovered_rel = ctx["uncovered_rel"]
        uncovered_dist = ctx["uncovered_dist"]

        local_visible_count = int(visible_idx.size)
        local_uncovered_count = int(uncovered_rel.shape[0])
        local_uncovered_ratio = float(local_uncovered_count / max(float(local_visible_count), 1.0)) if local_visible_count > 0 else 0.0

        if local_uncovered_count > 0:
            nearest_idx = int(np.argmin(uncovered_dist))
            nearest_rel = uncovered_rel[nearest_idx]
            nearest_dist = float(uncovered_dist[nearest_idx])

            weights = 1.0 / np.maximum(uncovered_dist, 1.0)
            centroid_rel = np.average(uncovered_rel, axis=0, weights=weights)
        else:
            nearest_rel = np.zeros((2,), dtype=np.float32)
            nearest_dist = 0.0
            centroid_rel = np.zeros((2,), dtype=np.float32)

        return np.array([
            float(local_visible_count / max(float(self.cfg.max_obs_users), 1.0)),
            float(local_uncovered_count / max(float(self.cfg.max_obs_users), 1.0)),
            float(local_uncovered_ratio),
            float(nearest_rel[0] / max(self.cfg.obs_radius, 1e-6)),
            float(nearest_rel[1] / max(self.cfg.obs_radius, 1e-6)),
            float(nearest_dist / max(self.cfg.obs_radius, 1e-6)),
            float(centroid_rel[0] / max(self.cfg.obs_radius, 1e-6)),
            float(centroid_rel[1] / max(self.cfg.obs_radius, 1e-6)),
        ], dtype=np.float32)

    def _compute_mcg_neighbor_summary_features(self, agent_id: int) -> np.ndarray:
        if not getattr(self.cfg, "use_neighbor_summary_features", False):
            return np.zeros((6,), dtype=np.float32)

        ctx = self._get_neighbor_context(agent_id)
        rel = ctx["rel"]
        dist = ctx["dist"]
        count = int(rel.shape[0])

        if count > 0:
            nearest_idx = int(np.argmin(dist))
            nearest_rel = rel[nearest_idx]
            nearest_dist = float(dist[nearest_idx])
            centroid_rel = np.mean(rel, axis=0)
            mean_dist = float(np.mean(dist))
        else:
            nearest_rel = np.zeros((2,), dtype=np.float32)
            nearest_dist = 0.0
            centroid_rel = np.zeros((2,), dtype=np.float32)
            mean_dist = 0.0

        return np.array([
            float(count / max(float(max(self.num_agents - 1, 1)), 1.0)),
            float(nearest_rel[0] / max(self.cfg.obs_radius, 1e-6)),
            float(nearest_rel[1] / max(self.cfg.obs_radius, 1e-6)),
            float(nearest_dist / max(self.cfg.obs_radius, 1e-6)),
            float(centroid_rel[0] / max(self.cfg.obs_radius, 1e-6)),
            float(centroid_rel[1] / max(self.cfg.obs_radius, 1e-6)),
        ], dtype=np.float32)

    def _compute_mcg_overlap_risk_features(self, agent_id: int) -> np.ndarray:
        if not getattr(self.cfg, "use_overlap_risk_features", False):
            return np.zeros((4,), dtype=np.float32)

        ctx = self._get_neighbor_context(agent_id)
        dist = ctx["dist"]
        count = int(dist.shape[0])

        if self.cfg.use_simplified_qos:
            risk_distance_scale = float(max(self.cfg.simplified_coverage_radius, 1e-6))
        else:
            risk_distance_scale = float(max(self.cfg.obs_radius * 0.5, 1e-6))

        close_threshold = max(2.0 * risk_distance_scale, 1e-6)

        if count > 0:
            close_mask = dist <= close_threshold
            close_count = int(np.sum(close_mask))
            close_exist = 1.0 if close_count > 0 else 0.0
            mean_dist = float(np.mean(dist))
            risk_score = float(np.mean(np.clip((close_threshold - dist) / close_threshold, a_min=0.0, a_max=1.0)))
        else:
            close_count = 0
            close_exist = 0.0
            mean_dist = 0.0
            risk_score = 0.0

        return np.array([
            float(close_exist),
            float(close_count / max(float(max(self.num_agents - 1, 1)), 1.0)),
            float(risk_score),
            float(mean_dist / max(self.cfg.obs_radius, 1e-6)),
        ], dtype=np.float32)

    def get_all_local_obs(self) -> np.ndarray:
        return np.stack([self.get_local_obs(i) for i in range(self.num_agents)], axis=0).astype(np.float32)

    def render(self) -> Dict[str, Any]:
        return {
            "bs_pos": self.bs_pos.copy(),
            "ue_positions": self.ue_positions.copy(),
            "uav_positions": self.uav_positions.copy(),
            "active_mask": self.active_mask.copy(),
            "covered_mask": self.covered_mask.copy(),
            "cover_count_per_user": self.cover_count_per_user.copy(),
            "assigned_uav_idx": self.assigned_uav_idx.copy(),
            "cover_mat": self.cover_mat.copy(),
            "user_cluster_centers": self.user_cluster_centers.copy(),
            "user_cluster_ids": self.user_cluster_ids.copy(),
            "user_is_clustered": self.user_is_clustered.copy(),
            "user_distribution_stats": dict(self.user_distribution_stats),
            "task_uncertainty": self.task_uncertainty.copy(),
            "task_aoi": self.task_aoi.copy(),
            "task_priority": self.task_priority.copy(),
            "task_sensing_count": self.task_sensing_count.copy(),
        }

    def get_env_metadata(self) -> Dict[str, Any]:
        meta = asdict(self.cfg)
        meta["channel_mode"] = self.channel_cfg.mode
        meta["state_dim"] = int(self.get_global_state().shape[0])
        meta["local_obs_dim"] = int(self.get_local_obs(0).shape[0])
        meta["ppo_main_local_obs_dim"] = int(self.get_local_obs_ppo_main(0).shape[0])
        meta["mcg_local_obs_dim"] = int(self.get_local_obs_mcg(0).shape[0])
        meta["num_agents"] = self.num_agents
        return meta

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _sample_user_positions(self) -> None:
        mode = self.cfg.user_distribution_mode

        if mode == "uniform":
            self.ue_positions = sample_points_in_annulus(
                num_points=self.cfg.num_users,
                r_inner=self.cfg.r_safe,
                r_outer=self.cfg.r_disaster,
                rng=self.rng,
            ).astype(np.float32)
            self.user_cluster_centers = np.zeros((0, 2), dtype=np.float32)
            self.user_cluster_ids = np.full((self.cfg.num_users,), -1, dtype=np.int32)
            self.user_is_clustered = np.zeros((self.cfg.num_users,), dtype=bool)
        elif mode == "clustered":
            pts, meta = sample_user_points_clustered(
                num_points=self.cfg.num_users,
                r_inner=self.cfg.r_safe,
                r_outer=self.cfg.r_disaster,
                rng=self.rng,
                num_clusters=self.cfg.num_user_clusters,
                cluster_radius=self.cfg.cluster_radius,
                edge_avoidance_ratio=self.cfg.edge_avoidance_ratio,
                edge_soft_limit_ratio=self.cfg.edge_soft_limit_ratio,
                cluster_center_min_radius_ratio=self.cfg.cluster_center_min_radius_ratio,
                cluster_center_max_radius_ratio=self.cfg.cluster_center_max_radius_ratio,
            )
            self.ue_positions = pts.astype(np.float32)
            self.user_cluster_centers = meta["user_cluster_centers"].astype(np.float32)
            self.user_cluster_ids = meta["user_cluster_ids"].astype(np.int32)
            self.user_is_clustered = meta["user_is_clustered"].astype(bool)
        elif mode == "mixed":
            pts, meta = sample_user_points_mixed(
                num_points=self.cfg.num_users,
                r_inner=self.cfg.r_safe,
                r_outer=self.cfg.r_disaster,
                rng=self.rng,
                num_clusters=self.cfg.num_user_clusters,
                clustered_ratio=self.cfg.clustered_user_ratio,
                cluster_radius=self.cfg.cluster_radius,
                edge_avoidance_ratio=self.cfg.edge_avoidance_ratio,
                edge_soft_limit_ratio=self.cfg.edge_soft_limit_ratio,
                cluster_center_min_radius_ratio=self.cfg.cluster_center_min_radius_ratio,
                cluster_center_max_radius_ratio=self.cfg.cluster_center_max_radius_ratio,
                radial_beta_a=self.cfg.user_radial_beta_a,
                radial_beta_b=self.cfg.user_radial_beta_b,
            )
            self.ue_positions = pts.astype(np.float32)
            self.user_cluster_centers = meta["user_cluster_centers"].astype(np.float32)
            self.user_cluster_ids = meta["user_cluster_ids"].astype(np.int32)
            self.user_is_clustered = meta["user_is_clustered"].astype(bool)
        else:
            raise ValueError(f"Unsupported user_distribution_mode: {mode}")

        self.user_distribution_stats = validate_user_distribution(
            points_xy=self.ue_positions,
            r_inner=self.cfg.r_safe,
            r_outer=self.cfg.r_disaster,
            edge_soft_limit_ratio=self.cfg.edge_soft_limit_ratio,
            edge_avoidance_ratio=self.cfg.edge_avoidance_ratio,
            generation_mode=mode,
            user_is_clustered=self.user_is_clustered,
            user_cluster_ids=self.user_cluster_ids,
        )

        if not self.user_distribution_stats.get("is_valid", True):
            raise RuntimeError(
                f"User generation validation failed: {self.user_distribution_stats}"
            )

    def _init_uavs(self) -> None:
        if self.cfg.uav_init_mode == "circle":
            init_xy = make_uav_init_positions_circle(self.num_agents, self.cfg.r_safe)
        elif self.cfg.uav_init_mode == "center":
            init_xy = make_uav_init_positions_center(self.num_agents, radius_spread=30.0)
        elif self.cfg.uav_init_mode == "custom":
            if len(self.cfg.custom_uav_init_xy) != self.num_agents:
                raise ValueError("custom_uav_init_xy length must equal max_candidate_uavs.")
            init_xy = np.array(self.cfg.custom_uav_init_xy, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported uav_init_mode: {self.cfg.uav_init_mode}")
        heights = np.full((self.num_agents, 1), self.cfg.uav_init_height, dtype=np.float32)
        self.uav_positions = np.concatenate([init_xy, heights], axis=1).astype(np.float32)

    def _validate_actions(self, actions: List[int] | np.ndarray) -> None:
        actions = actions.tolist() if isinstance(actions, np.ndarray) else actions
        if len(actions) != self.num_agents:
            raise ValueError(f"actions length mismatch, expect {self.num_agents}, got {len(actions)}")
        for a in actions:
            if not (0 <= int(a) < self.cfg.action_size):
                raise ValueError(f"invalid action {a}")

    def _apply_actions(self, actions: List[int] | np.ndarray) -> Tuple[np.ndarray, int, int]:
        actions = actions.tolist() if isinstance(actions, np.ndarray) else actions
        move_distances = np.zeros((self.num_agents,), dtype=np.float32)
        out_of_bound_count = 0
        new_activation_count = 0
        action_map = self.cfg.action_to_delta_xy()
        activation_action = 5

        for i, a in enumerate(actions):
            if int(a) == activation_action and self.cfg.allow_activation_action:
                if not self.active_mask[i]:
                    self.active_mask[i] = True
                    self.just_activated_mask[i] = True
                    new_activation_count += 1
                continue

            if not self.active_mask[i]:
                continue
            if self.remaining_time[i] <= 0.0:
                continue

            dx, dy = action_map[int(a)]
            old_xy = self.uav_positions[i, :2].copy()
            new_x = float(old_xy[0] + dx)
            new_y = float(old_xy[1] + dy)
            clipped_x, clipped_y = clip_point_to_ring(new_x, new_y, self.cfg.r_safe, self.cfg.r_disaster)
            if abs(clipped_x - new_x) > 1e-6 or abs(clipped_y - new_y) > 1e-6:
                out_of_bound_count += 1
            actual_move = np.linalg.norm(np.array([clipped_x, clipped_y], dtype=np.float32) - old_xy)
            self.uav_positions[i, 0] = clipped_x
            self.uav_positions[i, 1] = clipped_y
            move_distances[i] = float(actual_move)
            self.total_distance_per_uav[i] += float(actual_move)
            self.remaining_time[i] -= self.cfg.dt

        return move_distances, out_of_bound_count, new_activation_count

    def _update_coverage(self) -> None:
        active_idx = np.where(self.active_mask)[0]
        self.cover_mat[:] = False
        self.cover_aux_metric[:] = 0.0
        self.cover_count_per_user[:] = 0
        self.covered_mask[:] = False
        self.assigned_uav_idx[:] = -1
        self.coverage_ratio = 0.0

        if active_idx.size == 0:
            return

        cover_mat_active, aux_metric_active = build_cover_matrix(
            ue_xy=self.ue_positions,
            uav_xyz=self.uav_positions[active_idx],
            cfg=self.channel_cfg,
        )
        self.cover_mat[:, active_idx] = cover_mat_active
        self.cover_aux_metric[:, active_idx] = aux_metric_active

        self.cover_count_per_user = np.sum(self.cover_mat, axis=1).astype(np.int32)
        self.covered_mask = self.cover_count_per_user > 0
        prefer = "nearest" if self.channel_cfg.mode == "simplified" else "lowest_pathloss"
        assigned_local = nearest_feasible_uav(cover_mat_active, aux_metric_active, prefer=prefer)
        for k in range(self.cfg.num_users):
            if assigned_local[k] >= 0:
                self.assigned_uav_idx[k] = int(active_idx[assigned_local[k]])
        self.coverage_ratio = float(np.mean(self.covered_mask.astype(np.float32)))

    def _compute_mean_target_distance(self) -> float:
        active_idx = np.where(self.active_mask)[0]
        if active_idx.size == 0:
            return float(self.cfg.r_disaster)
        uav_xy = self.uav_positions[active_idx, :2]
        ue_xy = self.ue_positions

        if self.cfg.shaping_target_mode == "assigned_users":
            vals = []
            for local_i, global_i in enumerate(active_idx):
                assigned_idx = np.where(self.assigned_uav_idx == global_i)[0]
                if assigned_idx.size == 0:
                    continue
                d = np.linalg.norm(ue_xy[assigned_idx] - uav_xy[local_i][None, :], axis=1)
                vals.append(float(np.mean(d)))
            return float(np.mean(vals)) if vals else float(self.cfg.r_disaster)

        uncovered_idx = np.where(~self.covered_mask)[0]
        if uncovered_idx.size == 0:
            return 0.0
        uncovered_xy = ue_xy[uncovered_idx]
        vals = []
        for i in range(uav_xy.shape[0]):
            d = np.linalg.norm(uncovered_xy - uav_xy[i][None, :], axis=1)
            vals.append(float(np.min(d)))
        return float(np.mean(vals)) if vals else 0.0

    def _compute_dispersion_penalty(self) -> float:
        active_idx = np.where(self.active_mask)[0]
        if active_idx.size <= 1:
            return 0.0
        uav_xy = self.uav_positions[active_idx, :2]
        min_sep = float(self.cfg.min_uav_separation)
        penalty = 0.0
        for i in range(uav_xy.shape[0]):
            for j in range(i + 1, uav_xy.shape[0]):
                d = float(np.linalg.norm(uav_xy[i] - uav_xy[j]))
                if d < min_sep:
                    penalty += (min_sep - d) / max(min_sep, 1e-6)
        return float(penalty)

    def _compute_local_neighborhood_summary(self, agent_id: int) -> np.ndarray:
        """
        基于局部感知构造局部邻域摘要。
        若 use_local_neighborhood_summary=False，则返回同维度零向量，
        保持主方法与消融版本接口兼容。
        """
        num_dir = int(self.cfg.num_direction_sectors)
        num_rad = int(self.cfg.num_radial_bins)
        summary_dim = num_dir + num_rad + 4

        if not self.cfg.use_local_neighborhood_summary:
            return np.zeros((summary_dim,), dtype=np.float32)

        self_xy = self.uav_positions[agent_id, :2]
        ue_rel = self.ue_positions - self_xy[None, :]
        ue_dist = np.linalg.norm(ue_rel, axis=1)

        visible_idx = np.where(ue_dist <= self.cfg.obs_radius)[0]
        if visible_idx.size == 0:
            return np.zeros((summary_dim,), dtype=np.float32)

        visible_rel = ue_rel[visible_idx]
        visible_dist = ue_dist[visible_idx]

        if self._should_use_trusted_sensing():
            task_scores = self.task_priority[visible_idx] * (
                self.task_uncertainty[visible_idx]
                + self.task_aoi[visible_idx] / max(self.cfg.task_max_aoi, 1e-6)
            )
            direction_counts = np.zeros((num_dir,), dtype=np.float32)
            radial_counts = np.zeros((num_rad,), dtype=np.float32)
            angles = (np.arctan2(visible_rel[:, 1], visible_rel[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
            sector_ids = np.clip(
                np.floor(angles / (2.0 * np.pi / max(num_dir, 1))).astype(np.int32),
                0,
                max(num_dir - 1, 0),
            )
            radial_edges = np.linspace(0.0, self.cfg.obs_radius, num_rad + 1)
            radial_ids = np.clip(
                np.digitize(visible_dist, radial_edges[1:-1], right=False),
                0,
                max(num_rad - 1, 0),
            )
            for score, sector_id, radial_id in zip(task_scores, sector_ids, radial_ids):
                direction_counts[sector_id] += float(score)
                radial_counts[radial_id] += float(score)

            score_total = max(float(np.sum(task_scores)), 1e-6)
            nearby_uav_count = sum(
                1
                for j in range(self.num_agents)
                if j != agent_id
                and np.linalg.norm(self.uav_positions[j, :2] - self_xy) <= self.cfg.obs_radius
            )
            local_stats = np.array([
                float(np.mean(self.task_uncertainty[visible_idx])),
                float(np.mean(self.task_aoi[visible_idx]) / max(self.cfg.task_max_aoi, 1e-6)),
                float(np.mean(self.task_priority[visible_idx]) / max(self.cfg.task_priority_max, 1e-6)),
                float(min(nearby_uav_count / max(float(self.cfg.max_obs_uavs), 1.0), 1.0)),
            ], dtype=np.float32)
            return np.concatenate(
                [direction_counts / score_total, radial_counts / score_total, local_stats],
                axis=0,
            ).astype(np.float32)

        visible_covered = self.covered_mask[visible_idx]
        visible_assigned_to_me = (self.assigned_uav_idx[visible_idx] == agent_id)
        visible_overlap = (self.cover_count_per_user[visible_idx] > 1)

        uncovered_mask_local = ~visible_covered
        uncovered_rel = visible_rel[uncovered_mask_local]
        uncovered_dist = visible_dist[uncovered_mask_local]

        direction_counts = np.zeros((num_dir,), dtype=np.float32)
        radial_counts = np.zeros((num_rad,), dtype=np.float32)

        if uncovered_rel.shape[0] > 0:
            angles = np.arctan2(uncovered_rel[:, 1], uncovered_rel[:, 0])
            angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)

            sector_width = 2.0 * np.pi / max(num_dir, 1)
            sector_ids = np.floor(angles / sector_width).astype(np.int32)
            sector_ids = np.clip(sector_ids, 0, max(num_dir - 1, 0))

            for s in sector_ids:
                direction_counts[s] += 1.0

            radial_edges = np.linspace(0.0, self.cfg.obs_radius, num_rad + 1)
            radial_ids = np.digitize(uncovered_dist, radial_edges[1:-1], right=False)
            radial_ids = np.clip(radial_ids, 0, max(num_rad - 1, 0))

            for b in radial_ids:
                radial_counts[b] += 1.0

        denom_users = max(float(visible_idx.size), 1.0)
        direction_ratios = direction_counts / denom_users
        radial_ratios = radial_counts / denom_users

        nearby_uav_count = 0
        if self.cfg.use_neighbor_uav_obs:
            for j in range(self.num_agents):
                if j == agent_id:
                    continue
                d = float(np.linalg.norm(self.uav_positions[j, :2] - self_xy))
                if d <= self.cfg.obs_radius:
                    nearby_uav_count += 1

        local_stats = np.array([
            float(np.sum(uncovered_mask_local)) / denom_users,
            float(np.sum(visible_overlap)) / denom_users,
            float(np.sum(visible_assigned_to_me)) / denom_users,
            float(min(nearby_uav_count / max(float(self.cfg.max_obs_uavs), 1.0), 1.0)),
        ], dtype=np.float32)

        return np.concatenate(
            [direction_ratios.astype(np.float32), radial_ratios.astype(np.float32), local_stats],
            axis=0,
        ).astype(np.float32)

    def _compute_global_uncovered_summary(self) -> np.ndarray:
        """
        Build a compact global summary of uncovered users.

        Summary layout:
        [direction sector ratios..., radial bin ratios..., uncovered_ratio, active_uav_ratio]

        - direction sector ratios: length = cfg.num_direction_sectors
        - radial bin ratios: length = cfg.num_radial_bins
        - uncovered_ratio: scalar
        - active_uav_ratio: scalar
        """
        if not self.cfg.use_global_uncovered_summary:
            return np.zeros((0,), dtype=np.float32)

        num_dir = int(self.cfg.num_direction_sectors)
        num_rad = int(self.cfg.num_radial_bins)

        total_users = self.cfg.num_users
        if total_users <= 0:
            return np.zeros((num_dir + num_rad + 2,), dtype=np.float32)

        uncovered_idx = np.where(~self.covered_mask)[0]
        uncovered_xy = self.ue_positions[uncovered_idx] if uncovered_idx.size > 0 else np.zeros((0, 2),
                                                                                                dtype=np.float32)

        direction_counts = np.zeros((num_dir,), dtype=np.float32)
        radial_counts = np.zeros((num_rad,), dtype=np.float32)

        if uncovered_xy.shape[0] > 0:
            # -------------------------
            # 1) direction sector stats
            # -------------------------
            angles = np.arctan2(uncovered_xy[:, 1], uncovered_xy[:, 0])  # [-pi, pi]
            angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)  # [0, 2pi)

            sector_width = 2.0 * np.pi / num_dir
            sector_ids = np.floor(angles / sector_width).astype(np.int32)
            sector_ids = np.clip(sector_ids, 0, num_dir - 1)

            for s in sector_ids:
                direction_counts[s] += 1.0

            # -------------------------
            # 2) radial bin stats
            # -------------------------
            radii = np.linalg.norm(uncovered_xy, axis=1)  # [Nu]
            radial_edges = np.linspace(self.cfg.r_safe, self.cfg.r_disaster, num_rad + 1)

            # bin index in [0, num_rad-1]
            radial_ids = np.digitize(radii, radial_edges[1:-1], right=False)
            radial_ids = np.clip(radial_ids, 0, num_rad - 1)

            for b in radial_ids:
                radial_counts[b] += 1.0

        # normalize by total users so ratios are stable across episodes
        direction_ratios = direction_counts / max(float(total_users), 1.0)
        radial_ratios = radial_counts / max(float(total_users), 1.0)

        uncovered_ratio = float(uncovered_idx.size / max(total_users, 1))
        active_uav_ratio = float(np.sum(self.active_mask) / max(self.cfg.max_candidate_uavs, 1))

        summary = np.concatenate(
            [
                direction_ratios.astype(np.float32),
                radial_ratios.astype(np.float32),
                np.array([uncovered_ratio, active_uav_ratio], dtype=np.float32),
            ],
            axis=0,
        )
        return summary.astype(np.float32)

    def _compute_uncovered_guidance(self, agent_id: int) -> np.ndarray:
        """
        仅基于局部可见用户构造引导特征。
        若 use_uncovered_guidance=False，则返回固定零向量，保持维度兼容。
        """
        if not self.cfg.use_uncovered_guidance:
            return np.zeros((8,), dtype=np.float32)

        self_xy = self.uav_positions[agent_id, :2]

        ue_rel = self.ue_positions - self_xy[None, :]
        ue_dists = np.linalg.norm(ue_rel, axis=1)

        visible_idx = np.where(ue_dists <= self.cfg.obs_radius)[0]
        if visible_idx.size == 0:
            return np.zeros((8,), dtype=np.float32)

        visible_rel = ue_rel[visible_idx]
        visible_dists = ue_dists[visible_idx]

        if self._should_use_trusted_sensing():
            nearest_local_idx = int(np.argmin(visible_dists))
            nearest_rel = visible_rel[nearest_local_idx]
            task_scores = self.task_priority[visible_idx] * (
                self.task_uncertainty[visible_idx]
                + self.task_aoi[visible_idx] / max(self.cfg.task_max_aoi, 1e-6)
            )
            target_local_idx = int(np.argmax(task_scores))
            target_rel = visible_rel[target_local_idx]
            target_dist = float(visible_dists[target_local_idx])
            density_center = np.average(visible_rel, axis=0, weights=np.maximum(task_scores, 1e-6))

            return np.array([
                float(nearest_rel[0] / max(self.cfg.obs_radius, 1e-6)),
                float(nearest_rel[1] / max(self.cfg.obs_radius, 1e-6)),
                float(visible_dists[nearest_local_idx] / max(self.cfg.obs_radius, 1e-6)),
                float(target_rel[0] / max(self.cfg.obs_radius, 1e-6)),
                float(target_rel[1] / max(self.cfg.obs_radius, 1e-6)),
                float(target_dist / max(self.cfg.obs_radius, 1e-6)),
                float(density_center[0] / max(self.cfg.obs_radius, 1e-6)),
                float(density_center[1] / max(self.cfg.obs_radius, 1e-6)),
            ], dtype=np.float32)

        visible_covered = self.covered_mask[visible_idx]

        nearest_local_idx = int(np.argmin(visible_dists))
        nearest_dx = float(visible_rel[nearest_local_idx, 0] / max(self.cfg.obs_radius, 1e-6))
        nearest_dy = float(visible_rel[nearest_local_idx, 1] / max(self.cfg.obs_radius, 1e-6))
        nearest_dist = float(visible_dists[nearest_local_idx] / max(self.cfg.obs_radius, 1e-6))

        local_uncovered_mask = ~visible_covered
        if np.any(local_uncovered_mask):
            uncovered_rel = visible_rel[local_uncovered_mask]
            uncovered_dists = visible_dists[local_uncovered_mask]

            nearest_uncovered_local_idx = int(np.argmin(uncovered_dists))
            nearest_uncovered_dx = float(
                uncovered_rel[nearest_uncovered_local_idx, 0] / max(self.cfg.obs_radius, 1e-6)
            )
            nearest_uncovered_dy = float(
                uncovered_rel[nearest_uncovered_local_idx, 1] / max(self.cfg.obs_radius, 1e-6)
            )
            nearest_uncovered_dist = float(
                uncovered_dists[nearest_uncovered_local_idx] / max(self.cfg.obs_radius, 1e-6)
            )

            weights = 1.0 / (uncovered_dists + 1.0)
            density_center = np.average(uncovered_rel, axis=0, weights=weights)
            density_dx = float(density_center[0] / max(self.cfg.obs_radius, 1e-6))
            density_dy = float(density_center[1] / max(self.cfg.obs_radius, 1e-6))
        else:
            nearest_uncovered_dx, nearest_uncovered_dy, nearest_uncovered_dist = 0.0, 0.0, 0.0
            density_dx, density_dy = 0.0, 0.0

        return np.array([
            nearest_dx, nearest_dy, nearest_dist,
            nearest_uncovered_dx, nearest_uncovered_dy, nearest_uncovered_dist,
            density_dx, density_dy,
        ], dtype=np.float32)

    def _compute_reward(
            self,
            delta_new_cover: int,
            move_distances: np.ndarray,
            overlap_count: int,
            timeout_flag: bool,
            out_of_bound_count: int,
            prev_covered_mask: Optional[np.ndarray] = None,
            prev_cover_count_per_user: Optional[np.ndarray] = None,
            prev_cover_mat: Optional[np.ndarray] = None,
            prev_uav_positions: Optional[np.ndarray] = None,
            sensing_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        if self._should_use_trusted_sensing():
            return self._compute_reward_trusted_sensing(
                move_distances=move_distances,
                timeout_flag=timeout_flag,
                out_of_bound_count=out_of_bound_count,
                sensing_stats=sensing_stats or self._empty_sensing_stats(),
            )
        if self._should_use_mcg_reward():
            return self._compute_reward_mcg(
                delta_new_cover=delta_new_cover,
                move_distances=move_distances,
                overlap_count=overlap_count,
                timeout_flag=timeout_flag,
                out_of_bound_count=out_of_bound_count,
                prev_covered_mask=prev_covered_mask,
                prev_cover_count_per_user=prev_cover_count_per_user,
                prev_cover_mat=prev_cover_mat,
                prev_uav_positions=prev_uav_positions,
            )
        return self._compute_reward_base(
            delta_new_cover=delta_new_cover,
            move_distances=move_distances,
            overlap_count=overlap_count,
            timeout_flag=timeout_flag,
            out_of_bound_count=out_of_bound_count,
        )

    def _compute_reward_trusted_sensing(
            self,
            move_distances: np.ndarray,
            timeout_flag: bool,
            out_of_bound_count: int,
            sensing_stats: Dict[str, float],
    ) -> Tuple[float, Dict[str, Any]]:
        active_idx = np.where(self.active_mask)[0]
        active_count = int(active_idx.size)
        if active_count > 0 and self.cfg.step_size() > 0:
            mean_move_ratio = float(np.sum(move_distances[active_idx])) / (
                float(active_count) * float(self.cfg.step_size())
            )
        else:
            mean_move_ratio = 0.0

        uncertainty_gain_reward = (
            self.cfg.reward_weight_uncertainty_gain * float(sensing_stats["uncertainty_reduction"])
        )
        aoi_gain_reward = self.cfg.reward_weight_aoi_gain * float(sensing_stats["aoi_reduction"])
        repeat_sensing_penalty = (
            self.cfg.reward_weight_repeat_sensing_penalty * float(sensing_stats["repeat_sensing_ratio"])
        )
        sensing_cost_penalty = (
            self.cfg.reward_weight_sensing_cost * float(sensing_stats["active_sensing_ratio"])
        )
        movement_penalty = self.cfg.reward_weight_movement_cost * mean_move_ratio
        out_of_bound_penalty = self.cfg.w_out_of_bound_penalty * float(out_of_bound_count)
        timeout_penalty = self.cfg.w_timeout_penalty if timeout_flag else 0.0
        terminal_success_bonus = (
            self.cfg.w_terminal_success
            if float(sensing_stats["mean_task_uncertainty"]) <= self.cfg.trusted_sensing_uncertainty_target
            else 0.0
        )

        core_reward_pre_clip = (
            uncertainty_gain_reward
            + aoi_gain_reward
            - repeat_sensing_penalty
            - sensing_cost_penalty
            - movement_penalty
        )
        core_reward = core_reward_pre_clip
        if self.cfg.reward_normalize_for_mcg and self.cfg.mc_reward_clip > 0.0:
            core_reward = float(np.clip(core_reward, -self.cfg.mc_reward_clip, self.cfg.mc_reward_clip))

        total_reward = core_reward - out_of_bound_penalty - timeout_penalty + terminal_success_bonus
        zero_per_uav = np.zeros((self.num_agents,), dtype=np.float32)
        return float(total_reward), {
            "reward_mode": "trusted_sensing",
            "coverage_reward": 0.0,
            "marginal_contribution_reward": 0.0,
            "movement_penalty": float(movement_penalty),
            "overlap_penalty": 0.0,
            "uncovered_guidance_reward": 0.0,
            "uncertainty_gain_reward": float(uncertainty_gain_reward),
            "aoi_gain_reward": float(aoi_gain_reward),
            "repeat_sensing_penalty": float(repeat_sensing_penalty),
            "sensing_cost_penalty": float(sensing_cost_penalty),
            "out_of_bound_penalty": float(out_of_bound_penalty),
            "timeout_penalty": float(timeout_penalty),
            "terminal_success_bonus": float(terminal_success_bonus),
            "total_reward": float(total_reward),
            "reward_coverage": 0.0,
            "reward_marginal_contribution": 0.0,
            "reward_move_penalty": float(movement_penalty),
            "reward_overlap_penalty": 0.0,
            "reward_uncovered_guidance": 0.0,
            "reward_uncertainty_gain": float(uncertainty_gain_reward),
            "reward_aoi_gain": float(aoi_gain_reward),
            "reward_repeat_sensing_penalty": float(repeat_sensing_penalty),
            "reward_sensing_cost_penalty": float(sensing_cost_penalty),
            "reward_out_of_bound_penalty": float(out_of_bound_penalty),
            "reward_timeout_penalty": float(timeout_penalty),
            "reward_terminal_success": float(terminal_success_bonus),
            "reward_total": float(total_reward),
            "unique_new_covered_users": 0,
            "per_uav_unique_new_cover": zero_per_uav,
            "per_uav_guidance_progress": zero_per_uav,
            "overlap_user_ratio": 0.0,
            "near_neighbor_overlap_risk": 0.0,
            "core_reward_pre_clip": float(core_reward_pre_clip),
            "core_reward_post_clip": float(core_reward),
        }

    def _compute_reward_base(
            self,
            delta_new_cover: int,
            move_distances: np.ndarray,
            overlap_count: int,
            timeout_flag: bool,
            out_of_bound_count: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        ppo_main 原始团队共享奖励：覆盖收益 - 移动代价 - 冗余重叠 - 约束违规。
        """
        active_count = int(np.sum(self.active_mask))

        reward_coverage = self.cfg.w_newly_served_users * float(delta_new_cover)

        if active_count > 0 and self.cfg.step_size() > 0:
            mean_move_ratio = float(np.sum(move_distances)) / (float(active_count) * float(self.cfg.step_size()))
        else:
            mean_move_ratio = 0.0
        reward_move_penalty = self.cfg.w_step_move_cost * mean_move_ratio

        overlap_ratio = float(overlap_count) / max(float(self.cfg.num_users), 1.0)
        reward_overlap_penalty = self.cfg.w_overlap_penalty * overlap_ratio

        reward_out_of_bound_penalty = self.cfg.w_out_of_bound_penalty * float(out_of_bound_count)
        reward_timeout_penalty = self.cfg.w_timeout_penalty if timeout_flag else 0.0
        reward_terminal_success = self.cfg.w_terminal_success if np.all(self.covered_mask) else 0.0

        total_reward = (
                reward_coverage
                - reward_move_penalty
                - reward_overlap_penalty
                - reward_out_of_bound_penalty
                - reward_timeout_penalty
                + reward_terminal_success
        )

        breakdown: Dict[str, Any] = {
            "reward_mode": "base",
            "coverage_reward": float(reward_coverage),
            "marginal_contribution_reward": 0.0,
            "movement_penalty": float(reward_move_penalty),
            "overlap_penalty": float(reward_overlap_penalty),
            "uncovered_guidance_reward": 0.0,
            "out_of_bound_penalty": float(reward_out_of_bound_penalty),
            "timeout_penalty": float(reward_timeout_penalty),
            "terminal_success_bonus": float(reward_terminal_success),
            "total_reward": float(total_reward),
            "reward_coverage": float(reward_coverage),
            "reward_marginal_contribution": 0.0,
            "reward_move_penalty": float(reward_move_penalty),
            "reward_overlap_penalty": float(reward_overlap_penalty),
            "reward_uncovered_guidance": 0.0,
            "reward_out_of_bound_penalty": float(reward_out_of_bound_penalty),
            "reward_timeout_penalty": float(reward_timeout_penalty),
            "reward_terminal_success": float(reward_terminal_success),
            "reward_total": float(total_reward),
            "unique_new_covered_users": 0,
            "per_uav_unique_new_cover": np.zeros((self.num_agents,), dtype=np.float32),
            "per_uav_guidance_progress": np.zeros((self.num_agents,), dtype=np.float32),
            "overlap_user_ratio": float(overlap_ratio),
            "near_neighbor_overlap_risk": 0.0,
            "core_reward_pre_clip": float(total_reward),
            "core_reward_post_clip": float(total_reward),
        }
        return float(total_reward), breakdown

    def _compute_reward_mcg(
            self,
            delta_new_cover: int,
            move_distances: np.ndarray,
            overlap_count: int,
            timeout_flag: bool,
            out_of_bound_count: int,
            prev_covered_mask: Optional[np.ndarray],
            prev_cover_count_per_user: Optional[np.ndarray],
            prev_cover_mat: Optional[np.ndarray],
            prev_uav_positions: Optional[np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        mcg_ppo 增强 reward：
        1) 新增覆盖作为主收益；
        2) 独特新增覆盖作为边际贡献增强；
        3) 以移动代价、重叠抑制、未覆盖区域引导作为辅助塑形；
        4) 不改变训练主体，仍返回团队共享 reward，但 reward 内部可解释、可分项记录。
        """
        if prev_covered_mask is None:
            prev_covered_mask = np.zeros_like(self.covered_mask, dtype=bool)
        if prev_uav_positions is None:
            prev_uav_positions = self.uav_positions.copy()

        active_idx = np.where(self.active_mask)[0]
        active_count = int(active_idx.size)

        coverage_reward = self.cfg.reward_weight_coverage * float(delta_new_cover)

        unique_new_count, per_uav_unique_new_cover = self._compute_unique_new_cover_counts(prev_covered_mask)
        marginal_contribution_reward = 0.0
        if bool(getattr(self.cfg, "enable_marginal_contribution_reward", True)):
            marginal_contribution_reward = (
                self.cfg.reward_weight_marginal_contribution * float(unique_new_count)
            )

        if active_count > 0 and self.cfg.step_size() > 0:
            mean_move_ratio = float(np.sum(move_distances[active_idx])) / (float(active_count) * float(self.cfg.step_size()))
        else:
            mean_move_ratio = 0.0
        movement_penalty = self.cfg.reward_weight_movement_cost * float(mean_move_ratio)

        overlap_user_ratio = float(overlap_count) / max(float(self.cfg.num_users), 1.0)
        near_neighbor_overlap_risk = self._compute_near_neighbor_overlap_risk(active_idx)
        overlap_risk = 0.5 * overlap_user_ratio + 0.5 * near_neighbor_overlap_risk
        overlap_penalty = 0.0
        if bool(getattr(self.cfg, "enable_overlap_penalty", True)):
            overlap_penalty = self.cfg.reward_weight_overlap_penalty * float(overlap_risk)

        uncovered_guidance_raw, per_uav_guidance_progress = self._compute_uncovered_guidance_progress(
            prev_covered_mask=prev_covered_mask,
            prev_uav_positions=prev_uav_positions,
            active_idx=active_idx,
        )
        uncovered_guidance_reward = self.cfg.reward_weight_uncovered_guidance * float(uncovered_guidance_raw)

        out_of_bound_penalty = self.cfg.w_out_of_bound_penalty * float(out_of_bound_count)
        timeout_penalty = self.cfg.w_timeout_penalty if timeout_flag else 0.0
        terminal_success_bonus = self.cfg.w_terminal_success if np.all(self.covered_mask) else 0.0

        core_reward_pre_clip = (
            coverage_reward
            + marginal_contribution_reward
            + uncovered_guidance_reward
            - movement_penalty
            - overlap_penalty
        )

        core_reward_post_clip = float(core_reward_pre_clip)
        if bool(getattr(self.cfg, "reward_normalize_for_mcg", True)) and float(getattr(self.cfg, "mc_reward_clip", 0.0)) > 0.0:
            clip_value = float(self.cfg.mc_reward_clip)
            core_reward_post_clip = float(np.clip(core_reward_pre_clip, -clip_value, clip_value))

        total_reward = (
            core_reward_post_clip
            - out_of_bound_penalty
            - timeout_penalty
            + terminal_success_bonus
        )

        breakdown: Dict[str, Any] = {
            "reward_mode": "mcg",
            "coverage_reward": float(coverage_reward),
            "marginal_contribution_reward": float(marginal_contribution_reward),
            "movement_penalty": float(movement_penalty),
            "overlap_penalty": float(overlap_penalty),
            "uncovered_guidance_reward": float(uncovered_guidance_reward),
            "out_of_bound_penalty": float(out_of_bound_penalty),
            "timeout_penalty": float(timeout_penalty),
            "terminal_success_bonus": float(terminal_success_bonus),
            "total_reward": float(total_reward),
            "reward_coverage": float(coverage_reward),
            "reward_marginal_contribution": float(marginal_contribution_reward),
            "reward_move_penalty": float(movement_penalty),
            "reward_overlap_penalty": float(overlap_penalty),
            "reward_uncovered_guidance": float(uncovered_guidance_reward),
            "reward_out_of_bound_penalty": float(out_of_bound_penalty),
            "reward_timeout_penalty": float(timeout_penalty),
            "reward_terminal_success": float(terminal_success_bonus),
            "reward_total": float(total_reward),
            "unique_new_covered_users": int(unique_new_count),
            "per_uav_unique_new_cover": per_uav_unique_new_cover.astype(np.float32).copy(),
            "per_uav_guidance_progress": per_uav_guidance_progress.astype(np.float32).copy(),
            "overlap_user_ratio": float(overlap_user_ratio),
            "near_neighbor_overlap_risk": float(near_neighbor_overlap_risk),
            "core_reward_pre_clip": float(core_reward_pre_clip),
            "core_reward_post_clip": float(core_reward_post_clip),
        }
        return float(total_reward), breakdown

    def _compute_unique_new_cover_counts(self, prev_covered_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        per_uav_unique_new_cover = np.zeros((self.num_agents,), dtype=np.float32)
        newly_covered_idx = np.where((~prev_covered_mask) & self.covered_mask)[0]
        if newly_covered_idx.size == 0:
            return 0, per_uav_unique_new_cover

        for user_idx in newly_covered_idx:
            coverers = np.where(self.cover_mat[user_idx])[0]
            if coverers.size == 1:
                per_uav_unique_new_cover[int(coverers[0])] += 1.0

        unique_new_count = int(np.sum(per_uav_unique_new_cover))
        return unique_new_count, per_uav_unique_new_cover

    def _compute_near_neighbor_overlap_risk(self, active_idx: np.ndarray) -> float:
        if active_idx.size <= 1:
            return 0.0

        threshold = float(max(getattr(self.cfg, "overlap_distance_threshold", 1.0), 1.0))
        uav_xy = self.uav_positions[active_idx, :2]
        risks: List[float] = []
        for i in range(uav_xy.shape[0]):
            for j in range(i + 1, uav_xy.shape[0]):
                d = float(np.linalg.norm(uav_xy[i] - uav_xy[j]))
                risks.append(max(0.0, (threshold - d) / threshold))
        return float(np.mean(risks)) if risks else 0.0

    def _compute_uncovered_guidance_progress(
            self,
            prev_covered_mask: np.ndarray,
            prev_uav_positions: np.ndarray,
            active_idx: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        per_uav_guidance_progress = np.zeros((self.num_agents,), dtype=np.float32)
        target_user_idx = np.where(~prev_covered_mask)[0]
        if active_idx.size == 0 or target_user_idx.size == 0:
            return 0.0, per_uav_guidance_progress

        target_xy = self.ue_positions[target_user_idx]
        distance_scale = float(max(getattr(self.cfg, "guidance_distance_scale", self.cfg.obs_radius), 1.0))

        progress_values: List[float] = []
        for agent_id in active_idx:
            prev_xy = prev_uav_positions[agent_id, :2]
            curr_xy = self.uav_positions[agent_id, :2]
            prev_dist = float(np.min(np.linalg.norm(target_xy - prev_xy[None, :], axis=1)))
            curr_dist = float(np.min(np.linalg.norm(target_xy - curr_xy[None, :], axis=1)))
            progress = max(0.0, prev_dist - curr_dist) / distance_scale
            progress = float(np.clip(progress, 0.0, 1.0))
            per_uav_guidance_progress[agent_id] = progress
            progress_values.append(progress)

        guidance_raw = float(np.mean(progress_values)) if progress_values else 0.0
        return guidance_raw, per_uav_guidance_progress

    def _check_done(self) -> Tuple[bool, str]:
        if self.current_step >= self.cfg.max_steps:
            return True, "max_steps"

        if self._should_use_trusted_sensing():
            priority_sum = max(float(np.sum(self.task_priority)), 1e-6)
            weighted_uncertainty = float(np.sum(self.task_priority * self.task_uncertainty) / priority_sum)
            if weighted_uncertainty <= self.cfg.trusted_sensing_uncertainty_target:
                return True, "cognitive_target"
        elif np.all(self.covered_mask):
            return True, "all_covered"

        if np.any(self.active_mask) and np.all(self.remaining_time[self.active_mask] <= 0.0):
            return True, "all_timeout"

        # 更克制的早停：至少给轨迹展开留出前半段窗口
        stagnation_ready = self.current_step >= max(int(self.cfg.stagnation_patience), 15)
        if stagnation_ready and self.no_improve_steps >= self.cfg.stagnation_patience and np.any(self.active_mask):
            return True, "stagnation"

        return False, "running"

    def _build_info(
            self,
            delta_new_cover: int,
            overlap_count: int,
            move_distances: np.ndarray,
            out_of_bound_count: int,
            timeout_flag: bool,
            new_activation_count: int,
            reward_breakdown: Dict[str, float],
            sensing_stats: Dict[str, float],
            termination_reason: str,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step": self.current_step,
            "elapsed_time": float(self.elapsed_time),
            "coverage_ratio": float(self.coverage_ratio),
            "covered_users": int(self.covered_mask.sum()),
            "delta_new_cover": int(delta_new_cover),
            "overlap_users": int(overlap_count),
            "move_distance_total_step": float(np.sum(move_distances)),
            "move_distance_per_uav": move_distances.astype(np.float32).copy(),
            "total_distance_per_uav": self.total_distance_per_uav.astype(np.float32).copy(),
            "remaining_time": self.remaining_time.astype(np.float32).copy(),
            "out_of_bound_count": int(out_of_bound_count),
            "timeout_flag": bool(timeout_flag),
            "channel_mode": self.channel_cfg.mode,
            "active_uav_count": int(np.sum(self.active_mask)),
            "new_activation_count": int(new_activation_count),
            "active_ratio": float(np.mean(self.active_mask.astype(np.float32))),
            "termination_reason": termination_reason,
            "mean_task_uncertainty": float(sensing_stats["mean_task_uncertainty"]),
            "mean_task_aoi": float(sensing_stats["mean_task_aoi"]),
            "cognitive_quality": float(sensing_stats["cognitive_quality"]),
            "sensed_task_ratio": float(sensing_stats["sensed_task_ratio"]),
            "repeat_sensing_ratio": float(sensing_stats["repeat_sensing_ratio"]),
            "user_distribution_mode": self.cfg.user_distribution_mode,
            "num_edge_users": int(self.user_distribution_stats.get("num_edge_users", 0)),
            "num_clustered_users": int(self.user_distribution_stats.get("num_clustered_users", 0)),
            "num_independent_users": int(self.user_distribution_stats.get("num_independent_users", 0)),
            "user_radius_mean": float(self.user_distribution_stats.get("user_radius_mean", 0.0)),
            "user_radius_std": float(self.user_distribution_stats.get("user_radius_std", 0.0)),
            "user_distribution_valid": bool(self.user_distribution_stats.get("is_valid", True)),

            # reward breakdown
            "reward_mode": str(reward_breakdown.get("reward_mode", "base")),
            "coverage_reward": float(reward_breakdown.get("coverage_reward", reward_breakdown.get("reward_coverage", 0.0))),
            "marginal_contribution_reward": float(reward_breakdown.get("marginal_contribution_reward", reward_breakdown.get("reward_marginal_contribution", 0.0))),
            "movement_penalty": float(reward_breakdown.get("movement_penalty", reward_breakdown.get("reward_move_penalty", 0.0))),
            "overlap_penalty": float(reward_breakdown.get("overlap_penalty", reward_breakdown.get("reward_overlap_penalty", 0.0))),
            "uncovered_guidance_reward": float(reward_breakdown.get("uncovered_guidance_reward", reward_breakdown.get("reward_uncovered_guidance", 0.0))),
            "uncertainty_gain_reward": float(reward_breakdown.get("uncertainty_gain_reward", 0.0)),
            "aoi_gain_reward": float(reward_breakdown.get("aoi_gain_reward", 0.0)),
            "repeat_sensing_penalty": float(reward_breakdown.get("repeat_sensing_penalty", 0.0)),
            "sensing_cost_penalty": float(reward_breakdown.get("sensing_cost_penalty", 0.0)),
            "out_of_bound_penalty": float(reward_breakdown.get("out_of_bound_penalty", reward_breakdown.get("reward_out_of_bound_penalty", 0.0))),
            "timeout_penalty": float(reward_breakdown.get("timeout_penalty", reward_breakdown.get("reward_timeout_penalty", 0.0))),
            "terminal_success_bonus": float(reward_breakdown.get("terminal_success_bonus", reward_breakdown.get("reward_terminal_success", 0.0))),
            "total_reward": float(reward_breakdown.get("total_reward", reward_breakdown.get("reward_total", 0.0))),
            "reward_coverage": float(reward_breakdown.get("reward_coverage", reward_breakdown.get("coverage_reward", 0.0))),
            "reward_marginal_contribution": float(reward_breakdown.get("reward_marginal_contribution", reward_breakdown.get("marginal_contribution_reward", 0.0))),
            "reward_move_penalty": float(reward_breakdown.get("reward_move_penalty", reward_breakdown.get("movement_penalty", 0.0))),
            "reward_overlap_penalty": float(reward_breakdown.get("reward_overlap_penalty", reward_breakdown.get("overlap_penalty", 0.0))),
            "reward_uncovered_guidance": float(reward_breakdown.get("reward_uncovered_guidance", reward_breakdown.get("uncovered_guidance_reward", 0.0))),
            "reward_out_of_bound_penalty": float(reward_breakdown.get("reward_out_of_bound_penalty", reward_breakdown.get("out_of_bound_penalty", 0.0))),
            "reward_timeout_penalty": float(reward_breakdown.get("reward_timeout_penalty", reward_breakdown.get("timeout_penalty", 0.0))),
            "reward_terminal_success": float(reward_breakdown.get("reward_terminal_success", reward_breakdown.get("terminal_success_bonus", 0.0))),
            "reward_total": float(reward_breakdown.get("reward_total", reward_breakdown.get("total_reward", 0.0))),
            "unique_new_covered_users": int(reward_breakdown.get("unique_new_covered_users", 0)),
            "per_uav_unique_new_cover": np.array(reward_breakdown.get("per_uav_unique_new_cover", np.zeros((self.num_agents,), dtype=np.float32)), dtype=np.float32),
            "per_uav_guidance_progress": np.array(reward_breakdown.get("per_uav_guidance_progress", np.zeros((self.num_agents,), dtype=np.float32)), dtype=np.float32),
            "overlap_user_ratio": float(reward_breakdown.get("overlap_user_ratio", 0.0)),
            "near_neighbor_overlap_risk": float(reward_breakdown.get("near_neighbor_overlap_risk", 0.0)),
            "core_reward_pre_clip": float(reward_breakdown.get("core_reward_pre_clip", reward_breakdown.get("total_reward", reward_breakdown.get("reward_total", 0.0)))),
            "core_reward_post_clip": float(reward_breakdown.get("core_reward_post_clip", reward_breakdown.get("total_reward", reward_breakdown.get("reward_total", 0.0)))),
        }
        return info

    def _build_reset_output(self) -> Dict[str, Any]:
        return {
            "global_state": self.get_global_state(),
            "local_obs": self.get_all_local_obs(),
            "action_mask": self._get_action_mask(),
            "info": {
                "coverage_ratio": float(self.coverage_ratio),
                "covered_users": int(self.covered_mask.sum()),
                "channel_mode": self.channel_cfg.mode,
                "active_uav_count": int(np.sum(self.active_mask)),
                "mean_task_uncertainty": float(np.mean(self.task_uncertainty)) if self._should_use_trusted_sensing() else 0.0,
                "mean_task_aoi": float(np.mean(self.task_aoi)) if self._should_use_trusted_sensing() else 0.0,
                "user_distribution_mode": self.cfg.user_distribution_mode,
                "num_edge_users": int(self.user_distribution_stats.get("num_edge_users", 0)),
                "num_clustered_users": int(self.user_distribution_stats.get("num_clustered_users", 0)),
                "num_independent_users": int(self.user_distribution_stats.get("num_independent_users", 0)),
                "user_radius_mean": float(self.user_distribution_stats.get("user_radius_mean", 0.0)),
                "user_radius_std": float(self.user_distribution_stats.get("user_radius_std", 0.0)),
                "user_distribution_valid": bool(self.user_distribution_stats.get("is_valid", True)),
            },
        }

    def _build_step_output(self) -> Dict[str, Any]:
        return {
            "global_state": self.get_global_state(),
            "local_obs": self.get_all_local_obs(),
            "action_mask": self._get_action_mask(),
        }

    def _get_action_mask(self) -> np.ndarray:
        mask = np.ones((self.num_agents, self.cfg.action_size), dtype=np.float32)
        if self.cfg.allow_activation_action:
            activation_action = 5
            for i in range(self.num_agents):
                if self.active_mask[i]:
                    mask[i, activation_action] = 0.0
                else:
                    # inactive agents only stay or activate
                    mask[i, 1:5] = 0.0
        return mask

    def _check_agent_id(self, agent_id: int) -> None:
        if not (0 <= agent_id < self.num_agents):
            raise ValueError(f"Invalid agent_id={agent_id}")

    def debug_print_scene(self) -> None:
        print("===== DisasterDeploymentEnv Scene =====")
        print(f"Channel mode: {self.channel_cfg.mode}")
        print(f"Active UAVs: {int(np.sum(self.active_mask))}/{self.num_agents}")
        print(f"Covered users: {int(self.covered_mask.sum())}/{self.cfg.num_users}")
        print(f"Coverage ratio: {self.coverage_ratio:.4f}")
        print(f"User mode: {self.cfg.user_distribution_mode}")
        print(f"Edge users: {int(self.user_distribution_stats.get('num_edge_users', 0))}")
        print(f"Clustered users: {int(self.user_distribution_stats.get('num_clustered_users', 0))}")
        print(f"Independent users: {int(self.user_distribution_stats.get('num_independent_users', 0))}")

    def set_channel_mode(self, mode: str) -> None:
        if mode not in ("simplified", "paper_atg"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.channel_cfg.mode = mode
        self._update_coverage()

def _debug_main() -> None:
    cfg = ScenarioConfig(
        max_candidate_uavs=6,
        initially_active_uavs=0,
        allow_activation_action=True,
        use_simplified_qos=True,
    )
    env = DisasterDeploymentEnv(cfg)
    obs = env.reset(seed=123)
    print("local_obs shape:", obs["local_obs"].shape)

    activation_action = 5
    for _ in range(5):
        acts = np.zeros((cfg.max_candidate_uavs,), dtype=np.int64)
        acts[0] = activation_action
        obs, rew, done, info = env.step(acts)
        print(rew, info["active_uav_count"], info["coverage_ratio"])


if __name__ == "__main__":
    _debug_main()
