from __future__ import annotations

import os
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "paper_exp_v1"

TRAIN_LOG_FILENAME = "training_log.json"
EVAL_LOG_FILENAME = "eval_results.json"
COMPARE_PROTOCOL_FILENAME = "compare_protocol.json"
COMPARE_EPISODES_FILENAME = "episode_records.json"
COMPARE_AGGREGATES_FILENAME = "method_aggregates.json"
COMPARE_ALL_FILENAME = "deployment_compare_all.json"
SUMMARY_FILENAME = "summary.json"

TRAIN_PLOT_FILENAME = "training_history.png"
EVAL_SCENE_PLOT_FILENAME = "scene_example.png"
COMPARE_PLOT_FILENAME = "deployment_compare.png"


TRAINING_CURVE_SPECS: Dict[str, Dict[str, Any]] = {
    "final_coverage_ratio": {
        "title": "Training Final Coverage Ratio",
        "ylabel": "Coverage Ratio",
        "aliases": [
            "final_coverage_ratio",
            "mean_final_coverage_ratio",
        ],
        "preferred_source": "episode",
        "default": 0.0,
    },
    "episode_return": {
        "title": "Training Episode Return",
        "ylabel": "Episode Return",
        "aliases": [
            "episode_return",
            "mean_episode_return",
            "return",
        ],
        "preferred_source": "episode",
        "default": 0.0,
    },
    "total_move_distance": {
        "title": "Training Total Move Distance",
        "ylabel": "Distance (m)",
        "aliases": [
            "total_move_distance",
            "mean_total_move_distance",
            "mean_total_distance",
        ],
        "preferred_source": "episode",
        "default": 0.0,
    },
    "mean_overlap_users_step": {
        "title": "Training Mean Overlap Users / Step",
        "ylabel": "Users",
        "aliases": [
            "mean_overlap_users_step",
            "mean_mean_overlap_users_step",
            "overlap_users_per_step",
        ],
        "preferred_source": "episode",
        "default": 0.0,
    },
}

DEFAULT_TRAINING_METRICS: List[str] = list(TRAINING_CURVE_SPECS.keys())


def _sanitize_numeric_sequence(values: Sequence[Any]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            cleaned.append(numeric)
    return cleaned


def _drop_anomalous_tail_point(values: Sequence[Any]) -> List[float]:
    cleaned = _sanitize_numeric_sequence(values)
    if len(cleaned) < 8:
        return cleaned

    head = cleaned[:-1]
    tail_value = cleaned[-1]
    recent = head[-min(10, len(head)) :]
    if len(recent) < 5:
        return cleaned

    recent_median = float(median(recent))
    if abs(recent_median) < 1e-8:
        return cleaned

    abs_devs = [abs(x - recent_median) for x in recent]
    mad = float(median(abs_devs)) if abs_devs else 0.0
    recent_std = float(np.std(np.asarray(recent, dtype=np.float32)))
    robust_scale = max(mad * 1.4826, recent_std, 1e-6)

    drop_ratio = tail_value / recent_median
    is_large_downward_jump = drop_ratio < 0.60
    is_statistical_outlier = (recent_median - tail_value) > (3.0 * robust_scale)

    if is_large_downward_jump and is_statistical_outlier:
        return head
    return cleaned


def clean_training_curve_values(values: Sequence[Any]) -> List[float]:
    return _drop_anomalous_tail_point(values)


def make_method_identity(
    method_name: str,
    display_name: str,
    config_name: str,
    checkpoint_dir_name: Optional[str] = None,
    output_dir_name: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    summary_name: Optional[str] = None,
    plot_name: Optional[str] = None,
    method_label: Optional[str] = None,
    trainer_family: Optional[str] = None,
    policy_family: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "method_name": method_name,
        "display_name": display_name,
        "method_label": method_label or display_name,
        "config_name": config_name,
        "checkpoint_dir_name": checkpoint_dir_name or method_name,
        "output_dir_name": output_dir_name or method_name,
        "checkpoint_name": checkpoint_name,
        "summary_name": summary_name or method_name,
        "plot_name": plot_name or display_name,
        "trainer_family": trainer_family,
        "policy_family": policy_family,
        "agent_type": agent_type,
    }


def make_paths_block(
    run_dir: str,
    ckpt_dir: Optional[str] = None,
    plot_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    stable_output_dir: Optional[str] = None,
    training_log_filename: str = TRAIN_LOG_FILENAME,
    eval_log_filename: str = EVAL_LOG_FILENAME,
    summary_filename: str = SUMMARY_FILENAME,
    compare_protocol_filename: str = COMPARE_PROTOCOL_FILENAME,
    compare_aggregates_filename: str = COMPARE_AGGREGATES_FILENAME,
    compare_all_filename: str = COMPARE_ALL_FILENAME,
    training_plot_filename: str = TRAIN_PLOT_FILENAME,
    eval_scene_plot_filename: str = EVAL_SCENE_PLOT_FILENAME,
    compare_plot_filename: str = COMPARE_PLOT_FILENAME,
) -> Dict[str, Any]:
    paths: Dict[str, Any] = {
        "run_dir": run_dir,
        "stable_output_dir": stable_output_dir,
    }

    if ckpt_dir is not None:
        paths["checkpoint_dir"] = ckpt_dir
    if plot_dir is not None:
        paths["plot_dir"] = plot_dir
        paths["training_plot_path"] = os.path.join(plot_dir, training_plot_filename)
        paths["eval_scene_plot_path"] = os.path.join(plot_dir, eval_scene_plot_filename)
        paths["compare_plot_path"] = os.path.join(plot_dir, compare_plot_filename)
    if log_dir is not None:
        paths["log_dir"] = log_dir
        paths["training_log_path"] = os.path.join(log_dir, training_log_filename)
        paths["eval_log_path"] = os.path.join(log_dir, eval_log_filename)
        paths["compare_protocol_path"] = os.path.join(log_dir, compare_protocol_filename)
        paths["compare_aggregates_path"] = os.path.join(log_dir, compare_aggregates_filename)
        paths["compare_all_path"] = os.path.join(log_dir, compare_all_filename)

    paths["summary_path"] = os.path.join(run_dir, summary_filename)
    return paths


def make_train_stats_block(train_stats: Dict[str, float]) -> Dict[str, float]:
    return {
        "train_policy_loss": float(train_stats.get("policy_loss", 0.0)),
        "train_value_loss": float(train_stats.get("value_loss", 0.0)),
        "train_entropy": float(train_stats.get("entropy", 0.0)),
        "train_total_loss": float(train_stats.get("total_loss", 0.0)),
        "train_approx_kl": float(train_stats.get("approx_kl", 0.0)),
        "train_clip_frac": float(train_stats.get("clip_frac", 0.0)),
        "train_buffer_size": float(train_stats.get("buffer_size", 0.0)),
    }


def make_ippo_train_stats_block(train_stats: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "train_policy_loss": float(train_stats.get("policy_loss", 0.0)),
        "train_value_loss": float(train_stats.get("value_loss", 0.0)),
        "train_entropy": float(train_stats.get("entropy", 0.0)),
        "train_total_loss": float(train_stats.get("total_loss", 0.0)),
        "train_approx_kl": float(train_stats.get("approx_kl", 0.0)),
        "train_clip_frac": float(train_stats.get("clip_frac", 0.0)),
        "train_buffer_size": float(train_stats.get("buffer_size", 0.0)),
        "num_independent_agents": int(train_stats.get("num_independent_agents", 0)),
        "independent_agent_stats": list(train_stats.get("independent_agent_stats", [])),
    }
    return out


def make_maddpg_train_stats_block(train_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    MADDPG 与 PPO 的训练统计字段不同，所以单独保留一套统一导出字段。
    这样 compare / plot 不会依赖这些字段，但日志分析会更清楚。
    """
    return {
        "train_actor_loss": float(train_stats.get("actor_loss", 0.0)),
        "train_critic_loss": float(train_stats.get("critic_loss", 0.0)),
        "train_q_mean": float(train_stats.get("q_mean", 0.0)),
        "train_target_q_mean": float(train_stats.get("target_q_mean", 0.0)),
        "train_actor_q_mean": float(train_stats.get("actor_q_mean", 0.0)),
        "train_buffer_size": float(train_stats.get("buffer_size", 0.0)),
        "num_maddpg_agents": int(train_stats.get("num_agents", 0)),
        "per_agent_stats": list(train_stats.get("per_agent_stats", [])),
        "skipped_update": bool(train_stats.get("skipped_update", False)),
    }

def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return float(default)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        if isinstance(value, bool):
            return int(value)
        value = int(float(value))
        return value
    except (TypeError, ValueError):
        return int(default)


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    return bool(default)


def normalize_episode_record(
    record: Dict[str, Any],
    fallback_method_name: str = "",
    fallback_display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将单次 episode 记录统一成 compare/eval/export 可复用的标准格式。
    """
    record = dict(record or {})

    method_name = str(record.get("method_name") or fallback_method_name or "")
    display_name = str(
        record.get("display_name")
        or fallback_display_name
        or method_name
        or "Unknown"
    )

    normalized = {
        "method_name": method_name,
        "display_name": display_name,
        "scene_seed": safe_int(record.get("scene_seed", 0)),
        "episode_index": safe_int(record.get("episode_index", 0)),
        "num_users": safe_int(record.get("num_users", 0)),
        "num_uavs": safe_int(record.get("num_uavs", 0)),

        "final_coverage_ratio": safe_float(
            record.get("final_coverage_ratio", record.get("coverage_ratio", 0.0))
        ),
        "final_covered_users": safe_int(
            record.get("final_covered_users", record.get("covered_users", 0))
        ),
        "total_move_distance": safe_float(
            record.get("total_move_distance", record.get("total_distance", 0.0))
        ),
        "mean_overlap_users_step": safe_float(
            record.get("mean_overlap_users_step", record.get("overlap_users_per_step", 0.0))
        ),
        "episode_length": safe_int(record.get("episode_length", 0)),
        "full_coverage_success": safe_bool(
            record.get("full_coverage_success", record.get("success", False))
        ),
        "final_task_uncertainty": safe_float(record.get("final_task_uncertainty", 0.0)),
        "final_task_aoi": safe_float(record.get("final_task_aoi", 0.0)),
        "final_cognitive_quality": safe_float(record.get("final_cognitive_quality", 0.0)),
        "mean_repeat_sensing_ratio": safe_float(record.get("mean_repeat_sensing_ratio", 0.0)),
        "evaluation_domain": str(record.get("evaluation_domain", "coverage")),

        "active_uav_count": safe_int(record.get("active_uav_count", 0)),
        "trainer_family": record.get("trainer_family"),
        "policy_family": record.get("policy_family"),
        "method_type": record.get("method_type"),
        "method_category": record.get("method_category"),
        "control_mode": record.get("control_mode"),
    }

    return normalized


def normalize_method_aggregate(
    aggregate: Dict[str, Any],
    fallback_method_name: str = "",
    fallback_display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将方法级聚合结果统一成 compare/export/table 可用格式。
    """
    aggregate = dict(aggregate or {})

    method_name = str(aggregate.get("method_name") or fallback_method_name or "")
    display_name = str(
        aggregate.get("display_name")
        or fallback_display_name
        or method_name
        or "Unknown"
    )

    normalized = {
        "method_name": method_name,
        "display_name": display_name,
        "num_episodes": safe_int(aggregate.get("num_episodes", 0)),

        "mean_final_coverage_ratio": safe_float(
            aggregate.get("mean_final_coverage_ratio", aggregate.get("final_coverage_ratio", 0.0))
        ),
        "mean_final_covered_users": safe_float(
            aggregate.get("mean_final_covered_users", aggregate.get("final_covered_users", 0.0))
        ),
        "mean_total_move_distance": safe_float(
            aggregate.get("mean_total_move_distance", aggregate.get("mean_total_distance", 0.0))
        ),
        "mean_mean_overlap_users_step": safe_float(
            aggregate.get("mean_mean_overlap_users_step", aggregate.get("mean_overlap_users_step", 0.0))
        ),
        "mean_episode_length": safe_float(
            aggregate.get("mean_episode_length", aggregate.get("episode_length", 0.0))
        ),
        "full_coverage_success_rate": safe_float(
            aggregate.get("full_coverage_success_rate", aggregate.get("success_rate", 0.0))
        ),
        "mean_final_task_uncertainty": safe_float(aggregate.get("mean_final_task_uncertainty", 0.0)),
        "mean_final_task_aoi": safe_float(aggregate.get("mean_final_task_aoi", 0.0)),
        "mean_final_cognitive_quality": safe_float(aggregate.get("mean_final_cognitive_quality", 0.0)),
        "mean_mean_repeat_sensing_ratio": safe_float(aggregate.get("mean_mean_repeat_sensing_ratio", 0.0)),
        "evaluation_domain": str(aggregate.get("evaluation_domain", "coverage")),

        "std_final_coverage_ratio": safe_float(
            aggregate.get("std_final_coverage_ratio", 0.0)
        ),
        "std_total_move_distance": safe_float(
            aggregate.get("std_total_move_distance", 0.0)
        ),

        "trainer_family": aggregate.get("trainer_family"),
        "policy_family": aggregate.get("policy_family"),
        "method_type": aggregate.get("method_type"),
        "method_category": aggregate.get("method_category"),
        "control_mode": aggregate.get("control_mode"),
    }

    return normalized

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_number(mapping: Dict[str, Any], aliases: Sequence[str], default: float = 0.0) -> float:
    for key in aliases:
        if key in mapping and mapping.get(key) is not None:
            return _safe_float(mapping.get(key), default=default)
    return float(default)


def get_training_curve_specs(metric_keys: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    if metric_keys is None:
        keys = list(DEFAULT_TRAINING_METRICS)
    else:
        keys = [str(x) for x in metric_keys]
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        if key in TRAINING_CURVE_SPECS:
            out[key] = dict(TRAINING_CURVE_SPECS[key])
    return out


def get_training_curve_metric_keys(metric_keys: Optional[Sequence[str]] = None) -> List[str]:
    return list(get_training_curve_specs(metric_keys).keys())


def normalize_training_metric_series(payload: Dict[str, Any], metric_keys: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    specs = get_training_curve_specs(metric_keys)
    update_logs = list(payload.get("update_logs", []) or [])
    train_episode_history = list(payload.get("train_episode_history", []) or [])

    series: Dict[str, Dict[str, Any]] = {}
    for metric_key, spec in specs.items():
        episode_values: List[float] = []
        update_values: List[float] = []

        aliases = list(spec.get("aliases", []))
        default = _safe_float(spec.get("default", 0.0))

        for item in train_episode_history:
            episode_values.append(_first_number(item, aliases, default=default))

        for item in update_logs:
            rollout_summary = item.get("rollout_summary") or {}
            if isinstance(rollout_summary, dict) and rollout_summary:
                update_values.append(_first_number(rollout_summary, aliases, default=default))
            else:
                update_values.append(_first_number(item, aliases, default=default))

        preferred_source = str(spec.get("preferred_source", "episode"))
        if preferred_source == "update" and update_values:
            chosen_values = update_values
            source = "update_logs"
        elif preferred_source == "episode" and episode_values:
            chosen_values = episode_values
            source = "train_episode_history"
        elif episode_values:
            chosen_values = episode_values
            source = "train_episode_history"
        else:
            chosen_values = update_values
            source = "update_logs"

        cleaned_values = clean_training_curve_values(chosen_values)

        series[metric_key] = {
            "metric_key": metric_key,
            "title": str(spec.get("title", metric_key)),
            "ylabel": str(spec.get("ylabel", metric_key)),
            "values": list(cleaned_values),
            "raw_values": list(chosen_values),
            "source": source,
            "available_sources": {
                "train_episode_history": episode_values,
                "update_logs": update_values,
            },
        }

    return series


def build_training_plot_series(payload: Dict[str, Any]) -> Dict[str, List[float]]:
    normalized = normalize_training_metric_series(payload)
    return {
        "episode_returns": list(normalized.get("episode_return", {}).get("values", [])),
        "final_coverages": list(normalized.get("final_coverage_ratio", {}).get("values", [])),
        "total_move_distances": list(normalized.get("total_move_distance", {}).get("values", [])),
        "mean_overlap_users_step": list(normalized.get("mean_overlap_users_step", {}).get("values", [])),
    }


def resolve_training_log_path(exp_path: str) -> str:
    exp_path = os.path.abspath(exp_path)
    if os.path.isfile(exp_path):
        return exp_path

    candidate_paths = [
        os.path.join(exp_path, "logs", TRAIN_LOG_FILENAME),
        os.path.join(exp_path, TRAIN_LOG_FILENAME),
        os.path.join(exp_path, "results", TRAIN_LOG_FILENAME),
        os.path.join(exp_path, "output", TRAIN_LOG_FILENAME),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    for root, _, files in os.walk(exp_path):
        if TRAIN_LOG_FILENAME in files:
            return os.path.join(root, TRAIN_LOG_FILENAME)

    raise FileNotFoundError(f"未找到训练日志: {exp_path}")


def infer_run_dir_from_training_log_path(log_path: str, payload: Optional[Dict[str, Any]] = None) -> str:
    payload = payload or {}
    run_dir = (
        (payload.get("paths") or {}).get("run_dir")
        or payload.get("run_dir")
        or payload.get("output_dir")
    )
    if run_dir:
        return os.path.abspath(str(run_dir))

    log_dir = os.path.abspath(os.path.dirname(log_path))
    if os.path.basename(log_dir).lower() == "logs":
        return os.path.dirname(log_dir)
    return log_dir


def extract_method_display_name(payload: Dict[str, Any], fallback: str = "Training") -> str:
    method = payload.get("method") or {}
    return str(
        method.get("display_name")
        or payload.get("display_name")
        or payload.get("method_name")
        or fallback
    )


def build_paper_metric_row(method_identity: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method_name": method_identity["method_name"],
        "display_name": method_identity["display_name"],
        "final_coverage_ratio": float(aggregate.get("mean_final_coverage_ratio", 0.0)),
        "final_covered_users": float(aggregate.get("mean_final_covered_users", 0.0)),
        "full_coverage_success_rate": float(
            aggregate.get("full_coverage_success_rate", aggregate.get("success_rate", 0.0))
        ),
        "total_move_distance": float(
            aggregate.get("mean_total_move_distance", aggregate.get("mean_total_distance", 0.0))
        ),
        "mean_overlap_users_step": float(aggregate.get("mean_mean_overlap_users_step", 0.0)),
        "episode_length": float(aggregate.get("mean_episode_length", 0.0)),
        "final_task_uncertainty": float(aggregate.get("mean_final_task_uncertainty", 0.0)),
        "final_task_aoi": float(aggregate.get("mean_final_task_aoi", 0.0)),
        "final_cognitive_quality": float(aggregate.get("mean_final_cognitive_quality", 0.0)),
        "mean_repeat_sensing_ratio": float(aggregate.get("mean_mean_repeat_sensing_ratio", 0.0)),
        "trainer_family": method_identity.get("trainer_family"),
        "policy_family": method_identity.get("policy_family"),
        "agent_type": method_identity.get("agent_type"),
    }


def build_reward_tail_mean(train_episode_history: List[Dict[str, Any]], tail_size: int = 10) -> Dict[str, float]:
    tail = train_episode_history[-tail_size:] if len(train_episode_history) >= tail_size else train_episode_history
    if not tail:
        return {}

    keys = sorted({k for x in tail for k in x.get("reward_component_episode_totals", {}).keys()})
    out: Dict[str, float] = {}
    for key in keys:
        values = [
            float(x["reward_component_episode_totals"][key])
            for x in tail
            if key in x.get("reward_component_episode_totals", {})
        ]
        if values:
            out[key] = sum(values) / float(len(values))
    return out
