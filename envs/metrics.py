from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _should_collect_reward_key(key: str, value: Any) -> bool:
    if not _is_scalar_number(value):
        return False
    return key.startswith("reward_") or key.endswith("_reward") or key.endswith("_penalty")


def _mean_named_dict(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    all_keys = sorted({k for item in dict_list for k in item.keys()})
    out: Dict[str, float] = {}
    for key in all_keys:
        values = [float(item[key]) for item in dict_list if key in item]
        if values:
            out[key] = float(np.mean(np.array(values, dtype=np.float32)))
    return out


@dataclass
class EpisodeMetrics:
    """
    存储单个 episode 的过程指标，并统一输出标准 summary 字段。
    同时兼容 reward 分项的动态收集，便于后续 mcg_ppo 的 reward breakdown
    进入 train / eval / compare / summary 全链路。
    """
    rewards: List[float] = field(default_factory=list)
    coverage_ratios: List[float] = field(default_factory=list)
    covered_users: List[int] = field(default_factory=list)
    overlap_users: List[int] = field(default_factory=list)
    move_distance_total_step: List[float] = field(default_factory=list)
    task_uncertainty: List[float] = field(default_factory=list)
    task_aoi: List[float] = field(default_factory=list)
    cognitive_quality: List[float] = field(default_factory=list)
    estimation_error: List[float] = field(default_factory=list)
    spectrum_estimation_error: List[float] = field(default_factory=list)
    demand_estimation_error: List[float] = field(default_factory=list)
    repeat_sensing_ratio: List[float] = field(default_factory=list)
    messages_attempted: List[int] = field(default_factory=list)
    messages_dropped: List[int] = field(default_factory=list)
    messages_delivered: List[int] = field(default_factory=list)
    messages_fused: List[int] = field(default_factory=list)
    communication_cost: List[float] = field(default_factory=list)
    fusion_gain: List[float] = field(default_factory=list)
    scheduling_team_utility: List[float] = field(default_factory=list)
    scheduling_estimated_utility: List[float] = field(default_factory=list)
    scheduling_conflict_count: List[int] = field(default_factory=list)
    scheduling_energy_consumption: List[float] = field(default_factory=list)
    arrivals: List[float] = field(default_factory=list)
    served_data: List[float] = field(default_factory=list)
    queue_length: List[float] = field(default_factory=list)
    service_rate: List[float] = field(default_factory=list)
    weighted_demand_satisfaction: List[float] = field(default_factory=list)
    high_priority_service_rate: List[float] = field(default_factory=list)
    queue_overflow: List[float] = field(default_factory=list)
    service_energy_consumption: List[float] = field(default_factory=list)
    reward_component_history: Dict[str, List[float]] = field(default_factory=dict)

    final_total_distance_per_uav: Optional[np.ndarray] = None
    final_remaining_time: Optional[np.ndarray] = None
    final_coverage_ratio: float = 0.0
    final_covered_users: int = 0
    final_active_uav_count: int = 0
    final_task_uncertainty: float = 0.0
    final_task_aoi: float = 0.0
    final_cognitive_quality: float = 0.0
    final_estimation_error: float = 0.0
    final_spectrum_estimation_error: float = 0.0
    final_demand_estimation_error: float = 0.0
    final_scheduling_team_utility: float = 0.0
    final_total_queue: float = 0.0
    final_service_rate: float = 0.0
    final_weighted_demand_satisfaction: float = 0.0
    final_high_priority_service_rate: float = 0.0
    episode_length: int = 0
    done_reason: str = "unknown"

    def update(self, reward: float, info: Dict[str, Any]) -> None:
        self.rewards.append(float(info.get("reward_total", reward)))
        self.coverage_ratios.append(float(info.get("coverage_ratio", 0.0)))
        self.covered_users.append(int(info.get("covered_users", 0)))
        self.overlap_users.append(int(info.get("overlap_users", 0)))
        self.move_distance_total_step.append(float(info.get("move_distance_total_step", 0.0)))
        self.task_uncertainty.append(float(info.get("mean_task_uncertainty", 0.0)))
        self.task_aoi.append(float(info.get("mean_task_aoi", 0.0)))
        self.cognitive_quality.append(float(info.get("cognitive_quality", 0.0)))
        self.estimation_error.append(float(info.get("mean_estimation_error", 0.0)))
        self.spectrum_estimation_error.append(
            float(info.get("mean_spectrum_estimation_error", 0.0))
        )
        self.demand_estimation_error.append(
            float(info.get("mean_demand_estimation_error", 0.0))
        )
        self.repeat_sensing_ratio.append(float(info.get("repeat_sensing_ratio", 0.0)))
        self.messages_attempted.append(int(info.get("messages_attempted", 0)))
        self.messages_dropped.append(int(info.get("messages_dropped", 0)))
        self.messages_delivered.append(int(info.get("messages_delivered", 0)))
        self.messages_fused.append(int(info.get("messages_fused", 0)))
        self.communication_cost.append(float(info.get("communication_cost", 0.0)))
        self.fusion_gain.append(float(info.get("fusion_gain", 0.0)))
        self.scheduling_team_utility.append(
            float(info.get("scheduling_team_utility", 0.0))
        )
        self.scheduling_estimated_utility.append(
            float(info.get("scheduling_estimated_utility", 0.0))
        )
        self.scheduling_conflict_count.append(
            int(info.get("scheduling_conflict_count", 0))
        )
        self.scheduling_energy_consumption.append(
            float(info.get("scheduling_energy_consumption", 0.0))
        )
        self.arrivals.append(float(info.get("total_arrivals", 0.0)))
        self.served_data.append(float(info.get("scheduling_served_data", 0.0)))
        self.queue_length.append(float(info.get("total_queue_length", 0.0)))
        self.service_rate.append(float(info.get("service_rate", 0.0)))
        self.weighted_demand_satisfaction.append(
            float(info.get("weighted_demand_satisfaction", 0.0))
        )
        self.high_priority_service_rate.append(
            float(info.get("high_priority_service_rate", 0.0))
        )
        self.queue_overflow.append(float(info.get("queue_overflow", 0.0)))
        self.service_energy_consumption.append(
            float(info.get("service_energy_consumption", 0.0))
        )

        for key, value in info.items():
            if _should_collect_reward_key(key, value):
                self.reward_component_history.setdefault(key, []).append(float(value))

        if "total_distance_per_uav" in info:
            self.final_total_distance_per_uav = np.array(info["total_distance_per_uav"], dtype=np.float32)

        if "remaining_time" in info:
            self.final_remaining_time = np.array(info["remaining_time"], dtype=np.float32)

        self.final_coverage_ratio = float(info.get("coverage_ratio", self.final_coverage_ratio))
        self.final_covered_users = int(info.get("covered_users", self.final_covered_users))
        self.final_active_uav_count = int(info.get("active_uav_count", self.final_active_uav_count))
        self.final_task_uncertainty = float(info.get("mean_task_uncertainty", self.final_task_uncertainty))
        self.final_task_aoi = float(info.get("mean_task_aoi", self.final_task_aoi))
        self.final_cognitive_quality = float(info.get("cognitive_quality", self.final_cognitive_quality))
        self.final_estimation_error = float(info.get("mean_estimation_error", self.final_estimation_error))
        self.final_spectrum_estimation_error = float(
            info.get("mean_spectrum_estimation_error", self.final_spectrum_estimation_error)
        )
        self.final_demand_estimation_error = float(
            info.get("mean_demand_estimation_error", self.final_demand_estimation_error)
        )
        self.final_scheduling_team_utility = float(
            info.get("scheduling_team_utility", self.final_scheduling_team_utility)
        )
        self.final_total_queue = float(
            info.get("total_queue_length", self.final_total_queue)
        )
        self.final_service_rate = float(info.get("service_rate", self.final_service_rate))
        self.final_weighted_demand_satisfaction = float(
            info.get(
                "weighted_demand_satisfaction",
                self.final_weighted_demand_satisfaction,
            )
        )
        self.final_high_priority_service_rate = float(
            info.get("high_priority_service_rate", self.final_high_priority_service_rate)
        )
        self.episode_length = int(info.get("step", len(self.rewards)))
        self.done_reason = str(info.get("termination_reason", self.done_reason))

    def summary(self) -> Dict[str, Any]:
        episode_return = float(np.sum(self.rewards)) if self.rewards else 0.0
        mean_step_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        mean_overlap_users_step = float(np.mean(self.overlap_users)) if self.overlap_users else 0.0
        mean_step_move_distance = float(np.mean(self.move_distance_total_step)) if self.move_distance_total_step else 0.0

        total_move_distance = (
            float(np.sum(self.final_total_distance_per_uav))
            if self.final_total_distance_per_uav is not None else 0.0
        )

        active_uav_count = max(int(self.final_active_uav_count), 1)
        avg_move_distance_per_uav = total_move_distance / float(active_uav_count)

        full_coverage_success = 1 if self.final_coverage_ratio >= 0.999999 else 0

        reward_component_episode_totals = {
            key: float(np.sum(np.array(values, dtype=np.float32)))
            for key, values in sorted(self.reward_component_history.items())
        }
        reward_component_step_means = {
            key: float(np.mean(np.array(values, dtype=np.float32)))
            for key, values in sorted(self.reward_component_history.items())
        }
        total_messages_attempted = int(np.sum(self.messages_attempted))
        total_messages_dropped = int(np.sum(self.messages_dropped))
        total_messages_delivered = int(np.sum(self.messages_delivered))
        total_messages_fused = int(np.sum(self.messages_fused))

        summary = {
            # 统一后的主字段
            "episode_length": int(self.episode_length),
            "episode_return": episode_return,
            "mean_step_reward": mean_step_reward,
            "final_coverage_ratio": float(self.final_coverage_ratio),
            "final_covered_users": int(self.final_covered_users),
            "full_coverage_success": int(full_coverage_success),
            "total_move_distance": total_move_distance,
            "avg_move_distance_per_uav": float(avg_move_distance_per_uav),
            "mean_step_move_distance": mean_step_move_distance,
            "mean_overlap_users_step": mean_overlap_users_step,
            "mean_repeat_sensing_ratio": float(np.mean(self.repeat_sensing_ratio)) if self.repeat_sensing_ratio else 0.0,
            "final_task_uncertainty": self.final_task_uncertainty,
            "final_task_aoi": self.final_task_aoi,
            "final_cognitive_quality": self.final_cognitive_quality,
            "final_estimation_error": self.final_estimation_error,
            "final_spectrum_estimation_error": self.final_spectrum_estimation_error,
            "final_demand_estimation_error": self.final_demand_estimation_error,
            "final_scheduling_team_utility": self.final_scheduling_team_utility,
            "total_arrivals": float(np.sum(self.arrivals)),
            "total_served_data": float(np.sum(self.served_data)),
            "final_total_queue": self.final_total_queue,
            "service_rate": float(np.mean(self.service_rate)) if self.service_rate else 0.0,
            "weighted_demand_satisfaction": (
                float(np.mean(self.weighted_demand_satisfaction))
                if self.weighted_demand_satisfaction else 0.0
            ),
            "high_priority_service_rate": (
                float(np.mean(self.high_priority_service_rate))
                if self.high_priority_service_rate else 0.0
            ),
            "total_queue_overflow": float(np.sum(self.queue_overflow)),
            "total_service_energy_consumption": float(
                np.sum(self.service_energy_consumption)
            ),
            "mean_scheduling_team_utility": float(
                np.mean(self.scheduling_team_utility)
            ) if self.scheduling_team_utility else 0.0,
            "total_scheduling_conflicts": int(np.sum(self.scheduling_conflict_count)),
            "total_scheduling_energy_consumption": float(
                np.sum(self.scheduling_energy_consumption)
            ),
            "total_messages_attempted": total_messages_attempted,
            "total_messages_dropped": total_messages_dropped,
            "total_messages_delivered": total_messages_delivered,
            "total_messages_fused": total_messages_fused,
            "message_acceptance_ratio": float(
                total_messages_fused / max(total_messages_delivered, 1)
            ),
            "total_communication_cost": float(np.sum(self.communication_cost)),
            "total_fusion_gain": float(np.sum(self.fusion_gain)),
            "done_reason": self.done_reason,
            "final_active_uav_count": int(self.final_active_uav_count),
            "final_total_distance_per_uav": (
                self.final_total_distance_per_uav.copy()
                if self.final_total_distance_per_uav is not None else None
            ),
            "final_remaining_time": (
                self.final_remaining_time.copy()
                if self.final_remaining_time is not None else None
            ),
            "reward_component_episode_totals": reward_component_episode_totals,
            "reward_component_step_means": reward_component_step_means,

            # 兼容旧字段
            "avg_step_reward": mean_step_reward,
            "avg_overlap_users": mean_overlap_users_step,
            "avg_step_move_distance": mean_step_move_distance,
            "final_total_distance": total_move_distance,
        }
        return summary


class MetricTracker:
    """
    跨 episode 汇总指标，训练 / 评估 / compare 共用。
    """

    def __init__(self):
        self.episode_summaries: List[Dict[str, Any]] = []

    def add_episode(self, episode_metrics: EpisodeMetrics) -> Dict[str, Any]:
        summary = episode_metrics.summary()
        self.episode_summaries.append(summary)
        return summary

    def reset(self) -> None:
        self.episode_summaries.clear()

    def num_episodes(self) -> int:
        return len(self.episode_summaries)

    def aggregate(self) -> Dict[str, Any]:
        if not self.episode_summaries:
            return {
                "num_episodes": 0,
                "mean_episode_return": 0.0,
                "std_episode_return": 0.0,
                "mean_final_coverage_ratio": 0.0,
                "std_final_coverage_ratio": 0.0,
                "mean_final_covered_users": 0.0,
                "std_final_covered_users": 0.0,
                "full_coverage_success_rate": 0.0,
                "mean_total_move_distance": 0.0,
                "std_total_move_distance": 0.0,
                "mean_avg_move_distance_per_uav": 0.0,
                "mean_mean_overlap_users_step": 0.0,
                "mean_episode_length": 0.0,
                "std_episode_length": 0.0,
                "mean_final_active_uav_count": 0.0,
                "mean_final_task_uncertainty": 0.0,
                "mean_final_task_aoi": 0.0,
                "mean_final_cognitive_quality": 0.0,
                "mean_final_estimation_error": 0.0,
                "mean_final_spectrum_estimation_error": 0.0,
                "mean_final_demand_estimation_error": 0.0,
                "mean_final_scheduling_team_utility": 0.0,
                "mean_total_scheduling_conflicts": 0.0,
                "mean_total_scheduling_energy_consumption": 0.0,
                "mean_total_arrivals": 0.0,
                "mean_total_served_data": 0.0,
                "mean_final_total_queue": 0.0,
                "mean_service_rate": 0.0,
                "mean_weighted_demand_satisfaction": 0.0,
                "mean_high_priority_service_rate": 0.0,
                "mean_total_queue_overflow": 0.0,
                "mean_total_service_energy_consumption": 0.0,
                "mean_total_messages_attempted": 0.0,
                "mean_total_messages_dropped": 0.0,
                "mean_total_messages_delivered": 0.0,
                "mean_total_messages_fused": 0.0,
                "mean_message_acceptance_ratio": 0.0,
                "mean_total_communication_cost": 0.0,
                "mean_total_fusion_gain": 0.0,
                "mean_mean_repeat_sensing_ratio": 0.0,
                "done_reason_histogram": {},
                "reward_component_episode_total_means": {},
                "reward_component_step_mean_across_episodes": {},
                # 兼容旧字段
                "success_rate": 0.0,
                "mean_total_distance": 0.0,
                "std_total_distance": 0.0,
            }

        episode_returns = np.array([x["episode_return"] for x in self.episode_summaries], dtype=np.float32)
        final_coverage_ratios = np.array([x["final_coverage_ratio"] for x in self.episode_summaries], dtype=np.float32)
        final_covered_users = np.array([x["final_covered_users"] for x in self.episode_summaries], dtype=np.float32)
        full_coverage_success = np.array([x["full_coverage_success"] for x in self.episode_summaries], dtype=np.float32)
        total_move_distances = np.array([x["total_move_distance"] for x in self.episode_summaries], dtype=np.float32)
        avg_move_distance_per_uav = np.array([x["avg_move_distance_per_uav"] for x in self.episode_summaries], dtype=np.float32)
        mean_overlap_users_step = np.array([x["mean_overlap_users_step"] for x in self.episode_summaries], dtype=np.float32)
        episode_lengths = np.array([x["episode_length"] for x in self.episode_summaries], dtype=np.float32)
        final_active_uav_count = np.array([x["final_active_uav_count"] for x in self.episode_summaries], dtype=np.float32)
        final_task_uncertainty = np.array([x.get("final_task_uncertainty", 0.0) for x in self.episode_summaries], dtype=np.float32)
        final_task_aoi = np.array([x.get("final_task_aoi", 0.0) for x in self.episode_summaries], dtype=np.float32)
        final_cognitive_quality = np.array([x.get("final_cognitive_quality", 0.0) for x in self.episode_summaries], dtype=np.float32)
        final_estimation_error = np.array([x.get("final_estimation_error", 0.0) for x in self.episode_summaries], dtype=np.float32)
        final_spectrum_estimation_error = np.array(
            [x.get("final_spectrum_estimation_error", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        final_demand_estimation_error = np.array(
            [x.get("final_demand_estimation_error", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        mean_scheduling_team_utility = np.array(
            [x.get("mean_scheduling_team_utility", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        total_scheduling_conflicts = np.array(
            [x.get("total_scheduling_conflicts", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        total_scheduling_energy_consumption = np.array(
            [x.get("total_scheduling_energy_consumption", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        total_arrivals = np.array(
            [x.get("total_arrivals", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        total_served_data = np.array(
            [x.get("total_served_data", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        final_total_queue = np.array(
            [x.get("final_total_queue", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        service_rates = np.array(
            [x.get("service_rate", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        weighted_demand_satisfaction = np.array(
            [x.get("weighted_demand_satisfaction", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        high_priority_service_rate = np.array(
            [x.get("high_priority_service_rate", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        total_queue_overflow = np.array(
            [x.get("total_queue_overflow", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        total_service_energy_consumption = np.array(
            [x.get("total_service_energy_consumption", 0.0) for x in self.episode_summaries],
            dtype=np.float32,
        )
        total_messages_attempted = np.array([x.get("total_messages_attempted", 0.0) for x in self.episode_summaries], dtype=np.float32)
        total_messages_dropped = np.array([x.get("total_messages_dropped", 0.0) for x in self.episode_summaries], dtype=np.float32)
        total_messages_delivered = np.array([x.get("total_messages_delivered", 0.0) for x in self.episode_summaries], dtype=np.float32)
        total_messages_fused = np.array([x.get("total_messages_fused", 0.0) for x in self.episode_summaries], dtype=np.float32)
        message_acceptance_ratio = np.array([x.get("message_acceptance_ratio", 0.0) for x in self.episode_summaries], dtype=np.float32)
        total_communication_cost = np.array([x.get("total_communication_cost", 0.0) for x in self.episode_summaries], dtype=np.float32)
        total_fusion_gain = np.array([x.get("total_fusion_gain", 0.0) for x in self.episode_summaries], dtype=np.float32)
        mean_repeat_sensing_ratio = np.array([x.get("mean_repeat_sensing_ratio", 0.0) for x in self.episode_summaries], dtype=np.float32)

        done_reason_histogram: Dict[str, int] = {}
        for x in self.episode_summaries:
            reason = str(x.get("done_reason", "unknown"))
            done_reason_histogram[reason] = done_reason_histogram.get(reason, 0) + 1

        reward_component_episode_total_means = _mean_named_dict([
            x.get("reward_component_episode_totals", {}) for x in self.episode_summaries
        ])
        reward_component_step_mean_across_episodes = _mean_named_dict([
            x.get("reward_component_step_means", {}) for x in self.episode_summaries
        ])

        agg = {
            "num_episodes": int(len(self.episode_summaries)),
            "mean_episode_return": float(np.mean(episode_returns)),
            "std_episode_return": float(np.std(episode_returns)),
            "mean_final_coverage_ratio": float(np.mean(final_coverage_ratios)),
            "std_final_coverage_ratio": float(np.std(final_coverage_ratios)),
            "mean_final_covered_users": float(np.mean(final_covered_users)),
            "std_final_covered_users": float(np.std(final_covered_users)),
            "full_coverage_success_rate": float(np.mean(full_coverage_success)),
            "mean_total_move_distance": float(np.mean(total_move_distances)),
            "std_total_move_distance": float(np.std(total_move_distances)),
            "mean_avg_move_distance_per_uav": float(np.mean(avg_move_distance_per_uav)),
            "std_avg_move_distance_per_uav": float(np.std(avg_move_distance_per_uav)),
            "mean_mean_overlap_users_step": float(np.mean(mean_overlap_users_step)),
            "std_mean_overlap_users_step": float(np.std(mean_overlap_users_step)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "std_episode_length": float(np.std(episode_lengths)),
            "mean_final_active_uav_count": float(np.mean(final_active_uav_count)),
            "std_final_active_uav_count": float(np.std(final_active_uav_count)),
            "mean_final_task_uncertainty": float(np.mean(final_task_uncertainty)),
            "mean_final_task_aoi": float(np.mean(final_task_aoi)),
            "mean_final_cognitive_quality": float(np.mean(final_cognitive_quality)),
            "mean_final_estimation_error": float(np.mean(final_estimation_error)),
            "mean_final_spectrum_estimation_error": float(
                np.mean(final_spectrum_estimation_error)
            ),
            "mean_final_demand_estimation_error": float(
                np.mean(final_demand_estimation_error)
            ),
            "mean_final_scheduling_team_utility": float(
                np.mean(mean_scheduling_team_utility)
            ),
            "mean_total_scheduling_conflicts": float(
                np.mean(total_scheduling_conflicts)
            ),
            "mean_total_scheduling_energy_consumption": float(
                np.mean(total_scheduling_energy_consumption)
            ),
            "mean_total_arrivals": float(np.mean(total_arrivals)),
            "mean_total_served_data": float(np.mean(total_served_data)),
            "mean_final_total_queue": float(np.mean(final_total_queue)),
            "mean_service_rate": float(np.mean(service_rates)),
            "mean_weighted_demand_satisfaction": float(
                np.mean(weighted_demand_satisfaction)
            ),
            "mean_high_priority_service_rate": float(
                np.mean(high_priority_service_rate)
            ),
            "mean_total_queue_overflow": float(np.mean(total_queue_overflow)),
            "mean_total_service_energy_consumption": float(
                np.mean(total_service_energy_consumption)
            ),
            "mean_total_messages_attempted": float(np.mean(total_messages_attempted)),
            "mean_total_messages_dropped": float(np.mean(total_messages_dropped)),
            "mean_total_messages_delivered": float(np.mean(total_messages_delivered)),
            "mean_total_messages_fused": float(np.mean(total_messages_fused)),
            "mean_message_acceptance_ratio": float(np.mean(message_acceptance_ratio)),
            "mean_total_communication_cost": float(np.mean(total_communication_cost)),
            "mean_total_fusion_gain": float(np.mean(total_fusion_gain)),
            "mean_mean_repeat_sensing_ratio": float(np.mean(mean_repeat_sensing_ratio)),
            "done_reason_histogram": done_reason_histogram,
            "reward_component_episode_total_means": reward_component_episode_total_means,
            "reward_component_step_mean_across_episodes": reward_component_step_mean_across_episodes,

            # 兼容旧字段
            "success_rate": float(np.mean(full_coverage_success)),
            "mean_total_distance": float(np.mean(total_move_distances)),
            "std_total_distance": float(np.std(total_move_distances)),
        }
        return agg

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.episode_summaries:
            return None
        return self.episode_summaries[-1]


def extract_episode_metrics_from_infos(rewards: List[float], infos: List[Dict[str, Any]]) -> EpisodeMetrics:
    ep = EpisodeMetrics()
    for r, info in zip(rewards, infos):
        ep.update(r, info)
    return ep


def format_metric_summary(summary: Dict[str, Any]) -> str:
    parts = []

    if "num_episodes" in summary:
        parts.append(f"episodes={summary['num_episodes']}")
    if "episode_return" in summary:
        parts.append(f"return={summary['episode_return']:.3f}")
    if "mean_episode_return" in summary:
        parts.append(f"mean_return={summary['mean_episode_return']:.3f}")
    if "final_coverage_ratio" in summary:
        parts.append(f"final_coverage={summary['final_coverage_ratio']:.3f}")
    if "mean_final_coverage_ratio" in summary:
        parts.append(f"mean_final_coverage={summary['mean_final_coverage_ratio']:.3f}")
    if "final_covered_users" in summary:
        parts.append(f"final_covered_users={summary['final_covered_users']}")
    if "mean_final_covered_users" in summary:
        parts.append(f"mean_final_covered_users={summary['mean_final_covered_users']:.3f}")
    if "full_coverage_success" in summary:
        parts.append(f"full_success={summary['full_coverage_success']}")
    if "full_coverage_success_rate" in summary:
        parts.append(f"full_success_rate={summary['full_coverage_success_rate']:.3f}")
    if "episode_length" in summary:
        parts.append(f"ep_len={summary['episode_length']}")
    if "mean_episode_length" in summary:
        parts.append(f"mean_ep_len={summary['mean_episode_length']:.3f}")
    if "total_move_distance" in summary:
        parts.append(f"move_total={summary['total_move_distance']:.3f}")
    if "mean_total_move_distance" in summary:
        parts.append(f"mean_move_total={summary['mean_total_move_distance']:.3f}")
    if "mean_overlap_users_step" in summary:
        parts.append(f"overlap_mean={summary['mean_overlap_users_step']:.3f}")
    if "mean_mean_overlap_users_step" in summary:
        parts.append(f"mean_overlap={summary['mean_mean_overlap_users_step']:.3f}")
    if "done_reason" in summary:
        parts.append(f"done={summary['done_reason']}")

    return "  ".join(parts)
