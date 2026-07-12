from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from baselines.method_registry import get_method_meta
from utils.experiment_schema import (
    normalize_episode_record,
    normalize_method_aggregate,
    safe_bool,
    safe_float,
    safe_int,
)


@dataclass
class CompareMethodEntry:
    method_name: str
    display_name: str
    method_type: str
    category: str
    checkpoint_name: Optional[str] = None
    checkpoint_dir_name: Optional[str] = None
    output_dir_name: Optional[str] = None
    trainer_family: Optional[str] = None
    policy_family: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "display_name": self.display_name,
            "method_type": self.method_type,
            "category": self.category,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_dir_name": self.checkpoint_dir_name,
            "output_dir_name": self.output_dir_name,
            "trainer_family": self.trainer_family,
            "policy_family": self.policy_family,
        }


def build_compare_method_entries(method_names: List[str]) -> List[CompareMethodEntry]:
    entries: List[CompareMethodEntry] = []
    for method_name in method_names:
        meta = get_method_meta(method_name)
        entries.append(
            CompareMethodEntry(
                method_name=meta.method_name,
                display_name=meta.display_name,
                method_type=meta.method_type,
                category=meta.category,
                checkpoint_name=meta.checkpoint_name,
                checkpoint_dir_name=meta.checkpoint_dir_name,
                output_dir_name=meta.output_dir_name,
                trainer_family=meta.trainer_family,
                policy_family=meta.policy_family,
            )
        )
    return entries


def standardize_run_result(
    result: Any,
    scenario_cfg: Any,
    scene_seed: int,
    episode_index: int,
) -> Dict[str, Any]:
    """
    把 baseline / PPO / IPPO / MADDPG 的单次运行结果统一成 compare 记录。
    """
    raw = getattr(result, "raw", {}) or {}
    episode_summary = raw.get("episode_summary", {}) or {}
    env_info = raw.get("env_info", {}) or {}

    method_name = str(getattr(result, "method_name", raw.get("method_name", "")))
    display_name = str(raw.get("display_name", method_name))

    record = {
        "scene_seed": int(scene_seed),
        "episode_index": int(episode_index),
        "num_users": safe_int(getattr(scenario_cfg, "num_users", 0)),
        "num_uavs": safe_int(getattr(scenario_cfg, "max_candidate_uavs", 0)),
        "method_name": method_name,
        "display_name": display_name,

        "final_coverage_ratio": safe_float(
            getattr(result, "final_coverage_ratio", episode_summary.get("final_coverage_ratio", 0.0))
        ),
        "final_covered_users": safe_int(
            getattr(result, "final_covered_users", episode_summary.get("final_covered_users", 0))
        ),
        "total_move_distance": safe_float(
            getattr(result, "total_distance", episode_summary.get("total_move_distance", 0.0))
        ),
        "mean_overlap_users_step": safe_float(
            episode_summary.get("mean_overlap_users_step", env_info.get("overlap_users", 0.0))
        ),
        "episode_length": safe_int(
            getattr(result, "episode_length", episode_summary.get("episode_length", env_info.get("step", 0)))
        ),
        "full_coverage_success": safe_bool(
            getattr(result, "success", episode_summary.get("full_coverage_success", False))
        ),

        "active_uav_count": safe_int(getattr(result, "active_uav_count", 0)),
        "trainer_family": raw.get("trainer_family"),
        "policy_family": raw.get("policy_family"),
        "method_type": raw.get("method_type"),
        "method_category": raw.get("method_category"),
        "control_mode": raw.get("control_mode"),
    }

    return normalize_episode_record(
        record,
        fallback_method_name=method_name,
        fallback_display_name=display_name,
    )


def aggregate_method_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return normalize_method_aggregate({})

    normalized = [
        normalize_episode_record(
            x,
            fallback_method_name=str(x.get("method_name", "")),
            fallback_display_name=str(x.get("display_name", "")),
        )
        for x in records
    ]

    method_name = str(normalized[0].get("method_name", ""))
    display_name = str(normalized[0].get("display_name", method_name))

    n = len(normalized)

    def _mean(key: str) -> float:
        if n == 0:
            return 0.0
        return sum(safe_float(x.get(key, 0.0)) for x in normalized) / float(n)

    aggregate = {
        "method_name": method_name,
        "display_name": display_name,
        "num_episodes": int(n),

        "mean_final_coverage_ratio": _mean("final_coverage_ratio"),
        "mean_final_covered_users": _mean("final_covered_users"),
        "mean_total_move_distance": _mean("total_move_distance"),
        "mean_mean_overlap_users_step": _mean("mean_overlap_users_step"),
        "mean_episode_length": _mean("episode_length"),
        "full_coverage_success_rate": _mean("full_coverage_success"),

        "std_final_coverage_ratio": (
            sum((safe_float(x.get("final_coverage_ratio", 0.0)) - _mean("final_coverage_ratio")) ** 2 for x in normalized) / float(n)
        ) ** 0.5,
        "std_total_move_distance": (
            sum((safe_float(x.get("total_move_distance", 0.0)) - _mean("total_move_distance")) ** 2 for x in normalized) / float(n)
        ) ** 0.5,
    }

    return normalize_method_aggregate(
        aggregate,
        fallback_method_name=method_name,
        fallback_display_name=display_name,
    )


def build_compare_items(
    method_aggregates: Dict[str, Dict[str, Any]],
    method_order: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if method_order is None:
        method_order = list(method_aggregates.keys())

    items: List[Dict[str, Any]] = []
    for method_name in method_order:
        agg = method_aggregates.get(method_name, {})
        meta = get_method_meta(method_name)
        normalized = normalize_method_aggregate(
            aggregate=agg,
            fallback_method_name=meta.method_name,
            fallback_display_name=meta.display_name,
        )
        items.append(normalized)

    return items


def build_paper_table_rows(compare_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in compare_items:
        normalized = normalize_method_aggregate(
            item,
            fallback_method_name=str(item.get("method_name", "")),
            fallback_display_name=str(item.get("display_name", "")),
        )
        rows.append(
            {
                "method_name": normalized["method_name"],
                "display_name": normalized["display_name"],
                "final_coverage_ratio": safe_float(normalized["mean_final_coverage_ratio"]),
                "final_covered_users": safe_float(normalized["mean_final_covered_users"]),
                "total_move_distance": safe_float(normalized["mean_total_move_distance"]),
                "mean_overlap_users_step": safe_float(normalized["mean_mean_overlap_users_step"]),
                "episode_length": safe_float(normalized["mean_episode_length"]),
                "full_coverage_success": safe_float(normalized["full_coverage_success_rate"]),
            }
        )
    return rows