from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.plot_scene import plot_training_history
from utils.experiment_schema import (
    EVAL_LOG_FILENAME,
    SUMMARY_FILENAME,
    TRAIN_LOG_FILENAME,
    build_reward_tail_mean,
    build_training_plot_series,
)
from utils.io import load_json, save_json


# =========================================================
# 基础工具
# =========================================================
def _configure_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _smooth_series(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    window = int(max(1, window))
    window = min(window, arr.size)
    if window <= 1:
        return arr.copy()

    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def _plot_raw_and_smooth(
    ax: plt.Axes,
    y: Sequence[float],
    title: str,
    ylabel: str,
    smooth_window: int = 30,
    color: str = "#1f77b4",
    show_raw: bool = True,
) -> None:
    values = np.asarray(y, dtype=np.float32)
    if values.size == 0:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Index")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    x = np.arange(1, len(values) + 1)

    if show_raw:
        ax.plot(x, values, alpha=0.25, linewidth=0.9, color=color)

    smooth = _smooth_series(values, smooth_window)
    ax.plot(x, smooth, linewidth=2.0, color=color)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _maybe_load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        return load_json(path)
    except Exception as e:
        print(f"[Warn] JSON 读取失败: {path} | {e}")
        return None


def _first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# =========================================================
# 文件定位
# =========================================================
def _resolve_exp_dir(exp_dir: str) -> str:
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"实验目录不存在: {exp_dir}")
    if not os.path.isdir(exp_dir):
        raise NotADirectoryError(f"--exp_dir 不是目录: {exp_dir}")
    return exp_dir


def _find_run_files(exp_dir: str) -> Dict[str, Optional[str]]:
    logs_dir = os.path.join(exp_dir, "logs")

    training_log_path = _first_existing([
        os.path.join(logs_dir, TRAIN_LOG_FILENAME),
        os.path.join(exp_dir, TRAIN_LOG_FILENAME),
    ])

    eval_log_path = _first_existing([
        os.path.join(logs_dir, EVAL_LOG_FILENAME),
        os.path.join(exp_dir, EVAL_LOG_FILENAME),
    ])

    summary_path = _first_existing([
        os.path.join(exp_dir, SUMMARY_FILENAME),
        os.path.join(logs_dir, SUMMARY_FILENAME),
    ])

    manifest_path = _first_existing([
        os.path.join(exp_dir, "manifest.json"),
        os.path.join(logs_dir, "manifest.json"),
    ])

    return {
        "exp_dir": exp_dir,
        "logs_dir": logs_dir if os.path.isdir(logs_dir) else None,
        "training_log_path": training_log_path,
        "eval_log_path": eval_log_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "plots_dir": os.path.join(exp_dir, "plots"),
    }


# =========================================================
# 元信息提取
# =========================================================
def _infer_run_meta(
    exp_dir: str,
    training_payload: Optional[Dict[str, Any]],
    eval_payload: Optional[Dict[str, Any]],
    summary_payload: Optional[Dict[str, Any]],
    manifest_payload: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    payloads = [training_payload, eval_payload, summary_payload, manifest_payload]

    def pick(*keys: str) -> Optional[str]:
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            for key in keys:
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            method = payload.get("method", {})
            if isinstance(method, dict):
                for key in keys:
                    val = method.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        return None

    run_name = pick("run_name") or os.path.basename(exp_dir)
    method_name = pick("method_name") or "unknown_method"
    display_name = pick("display_name") or pick("method_label") or method_name

    return {
        "run_name": run_name,
        "method_name": method_name,
        "display_name": display_name,
    }


# =========================================================
# 数据抽取
# =========================================================
def _extract_update_logs(training_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(training_payload, dict):
        return []
    update_logs = training_payload.get("update_logs", [])
    return update_logs if isinstance(update_logs, list) else []


def _extract_train_episode_history(training_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(training_payload, dict):
        return []
    history = training_payload.get("train_episode_history", [])
    return history if isinstance(history, list) else []


def _extract_eval_episode_records(eval_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(eval_payload, dict):
        return []

    candidate_keys = [
        "episode_records",
        "eval_episode_records",
        "eval_episode_history",
        "episodes",
        "records",
        "rollout_episodes",
    ]
    for key in candidate_keys:
        value = eval_payload.get(key)
        if isinstance(value, list):
            return value

    aggregates = eval_payload.get("aggregates", {})
    if isinstance(aggregates, dict):
        maybe_records = aggregates.get("episode_records")
        if isinstance(maybe_records, list):
            return maybe_records

    return []


def _extract_scalar_metrics(
    training_payload: Optional[Dict[str, Any]],
    eval_payload: Optional[Dict[str, Any]],
    summary_payload: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def update_from_dict(src: Dict[str, Any]) -> None:
        aliases = {
            "final_coverage_ratio": [
                "final_coverage_ratio",
                "mean_final_coverage_ratio",
                "coverage_ratio",
            ],
            "final_covered_users": [
                "final_covered_users",
                "mean_final_covered_users",
                "covered_users",
            ],
            "total_move_distance": [
                "total_move_distance",
                "mean_total_move_distance",
                "mean_total_distance",
            ],
            "mean_overlap_users_step": [
                "mean_overlap_users_step",
                "mean_mean_overlap_users_step",
            ],
            "episode_length": [
                "episode_length",
                "mean_episode_length",
            ],
            "success_rate": [
                "full_coverage_success_rate",
                "success_rate",
            ],
            "episode_return": [
                "mean_episode_return",
                "episode_return",
                "return",
            ],
        }

        for dst_key, src_keys in aliases.items():
            if dst_key in metrics:
                continue
            for key in src_keys:
                if key in src:
                    metrics[dst_key] = _safe_float(src[key])
                    break

    for payload in [summary_payload, eval_payload, training_payload]:
        if not isinstance(payload, dict):
            continue

        update_from_dict(payload)

        method_agg = payload.get("aggregate", {})
        if isinstance(method_agg, dict):
            update_from_dict(method_agg)

        rollout_summary = payload.get("rollout_summary", {})
        if isinstance(rollout_summary, dict):
            update_from_dict(rollout_summary)

        final_eval = payload.get("final_eval", {})
        if isinstance(final_eval, dict):
            update_from_dict(final_eval)

    update_logs = _extract_update_logs(training_payload)
    if update_logs:
        last_summary = (update_logs[-1].get("rollout_summary") or {})
        if isinstance(last_summary, dict):
            update_from_dict(last_summary)

    return metrics


def _extract_train_stats_curves(training_payload: Optional[Dict[str, Any]]) -> Dict[str, List[float]]:
    update_logs = _extract_update_logs(training_payload)

    curves: Dict[str, List[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": [],
        "approx_kl": [],
        "clip_frac": [],
    }

    for item in update_logs:
        train_stats = item.get("train_stats", {})
        if not isinstance(train_stats, dict):
            train_stats = {}

        curves["policy_loss"].append(_safe_float(
            train_stats.get("train_policy_loss", item.get("train_policy_loss", 0.0))
        ))
        curves["value_loss"].append(_safe_float(
            train_stats.get("train_value_loss", item.get("train_value_loss", 0.0))
        ))
        curves["entropy"].append(_safe_float(
            train_stats.get("train_entropy", item.get("train_entropy", 0.0))
        ))
        curves["total_loss"].append(_safe_float(
            train_stats.get("train_total_loss", item.get("train_total_loss", 0.0))
        ))
        curves["approx_kl"].append(_safe_float(
            train_stats.get("train_approx_kl", item.get("train_approx_kl", 0.0))
        ))
        curves["clip_frac"].append(_safe_float(
            train_stats.get("train_clip_frac", item.get("train_clip_frac", 0.0))
        ))

    return curves


def _extract_eval_metric_curves(eval_records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    metrics = {
        "final_coverage_ratio": [],
        "final_covered_users": [],
        "total_move_distance": [],
        "mean_overlap_users_step": [],
        "episode_return": [],
        "episode_length": [],
    }

    for rec in eval_records:
        metrics["final_coverage_ratio"].append(_safe_float(rec.get("final_coverage_ratio", rec.get("coverage_ratio", 0.0))))
        metrics["final_covered_users"].append(_safe_float(rec.get("final_covered_users", rec.get("covered_users", 0.0))))
        metrics["total_move_distance"].append(_safe_float(rec.get("total_move_distance", rec.get("total_distance", 0.0))))
        metrics["mean_overlap_users_step"].append(_safe_float(rec.get("mean_overlap_users_step", 0.0)))
        metrics["episode_return"].append(_safe_float(rec.get("episode_return", rec.get("return", 0.0))))
        metrics["episode_length"].append(_safe_float(rec.get("episode_length", 0.0)))

    return metrics


def _extract_tail_reward_components(training_payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    history = _extract_train_episode_history(training_payload)
    if history:
        return build_reward_tail_mean(history, tail_size=10)

    update_logs = _extract_update_logs(training_payload)
    if not update_logs:
        return {}

    tail = update_logs[-10:] if len(update_logs) >= 10 else update_logs
    collector: Dict[str, List[float]] = {}

    for item in tail:
        rollout_summary = item.get("rollout_summary", {})
        if not isinstance(rollout_summary, dict):
            continue
        comp = rollout_summary.get("reward_component_episode_total_means", {})
        if not isinstance(comp, dict):
            continue
        for k, v in comp.items():
            collector.setdefault(k, []).append(_safe_float(v))

    out: Dict[str, float] = {}
    for k, values in collector.items():
        if values:
            out[k] = float(sum(values) / len(values))
    return out


# =========================================================
# 绘图函数
# =========================================================
def _plot_training_history_figure(
    training_payload: Dict[str, Any],
    save_path: str,
    title: str,
    smooth_window: int,
    show: bool,
) -> bool:
    plot_series = build_training_plot_series(training_payload)
    if not plot_series["episode_returns"]:
        return False

    plot_training_history(
        episode_returns=plot_series["episode_returns"],
        final_coverages=plot_series["final_coverages"],
        total_move_distances=plot_series["total_move_distances"],
        mean_overlap_users_step=plot_series["mean_overlap_users_step"],
        title=title,
        save_path=save_path,
        show=show,
        smooth_window=smooth_window,
        show_raw=True,
    )
    plt.close("all")
    return True


def _plot_train_stats_figure(
    train_stat_curves: Dict[str, List[float]],
    save_path: str,
    title: str,
    smooth_window: int,
    show: bool,
) -> bool:
    has_any = any(len(v) > 0 for v in train_stat_curves.values())
    if not has_any:
        return False

    _configure_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    items = [
        ("policy_loss", "Policy Loss", "Loss", "#1f77b4"),
        ("value_loss", "Value Loss", "Loss", "#ff7f0e"),
        ("entropy", "Entropy", "Value", "#2ca02c"),
        ("total_loss", "Total Loss", "Loss", "#d62728"),
        ("approx_kl", "Approx KL", "Value", "#9467bd"),
        ("clip_frac", "Clip Fraction", "Value", "#8c564b"),
    ]

    for ax, (key, ttl, ylabel, color) in zip(axes, items):
        _plot_raw_and_smooth(
            ax=ax,
            y=train_stat_curves.get(key, []),
            title=ttl,
            ylabel=ylabel,
            smooth_window=smooth_window,
            color=color,
            show_raw=True,
        )
        ax.set_xlabel("Update")

    fig.suptitle(title)
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return True


def _plot_eval_episode_figure(
    eval_metric_curves: Dict[str, List[float]],
    save_path: str,
    title: str,
    smooth_window: int,
    show: bool,
) -> bool:
    has_any = any(len(v) > 0 for v in eval_metric_curves.values())
    if not has_any:
        return False

    _configure_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    items = [
        ("final_coverage_ratio", "Eval Final Coverage Ratio", "Coverage Ratio", "#1f77b4"),
        ("final_covered_users", "Eval Final Covered Users", "Users", "#ff7f0e"),
        ("total_move_distance", "Eval Total Move Distance", "Distance (m)", "#2ca02c"),
        ("mean_overlap_users_step", "Eval Mean Overlap Users / Step", "Users", "#d62728"),
        ("episode_return", "Eval Episode Return", "Return", "#9467bd"),
        ("episode_length", "Eval Episode Length", "Steps", "#8c564b"),
    ]

    for ax, (key, ttl, ylabel, color) in zip(axes, items):
        _plot_raw_and_smooth(
            ax=ax,
            y=eval_metric_curves.get(key, []),
            title=ttl,
            ylabel=ylabel,
            smooth_window=smooth_window,
            color=color,
            show_raw=True,
        )
        ax.set_xlabel("Eval Episode")

    fig.suptitle(title)
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return True


def _plot_scalar_summary_figure(
    scalar_metrics: Dict[str, float],
    save_path: str,
    title: str,
    show: bool,
) -> bool:
    wanted_order = [
        "final_coverage_ratio",
        "final_covered_users",
        "success_rate",
        "total_move_distance",
        "mean_overlap_users_step",
        "episode_length",
        "episode_return",
    ]
    labels_map = {
        "final_coverage_ratio": "Coverage Ratio",
        "final_covered_users": "Covered Users",
        "success_rate": "Success Rate",
        "total_move_distance": "Move Distance",
        "mean_overlap_users_step": "Overlap/Step",
        "episode_length": "Episode Length",
        "episode_return": "Episode Return",
    }

    keys = [k for k in wanted_order if k in scalar_metrics]
    if not keys:
        return False

    values = [scalar_metrics[k] for k in keys]
    labels = [labels_map[k] for k in keys]

    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(keys))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Value")

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return True


def _plot_reward_components_figure(
    reward_components: Dict[str, float],
    save_path: str,
    title: str,
    show: bool,
) -> bool:
    if not reward_components:
        return False

    items = sorted(reward_components.items(), key=lambda kv: abs(kv[1]), reverse=True)
    keys = [k for k, _ in items]
    values = [v for _, v in items]

    _configure_plot_style()
    fig_h = max(5.0, 0.35 * len(keys) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y = np.arange(len(keys))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Mean Reward Component Value")

    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", ha="left", fontsize=9)

    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return True


# =========================================================
# 主流程
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot figures for a single experiment directory."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="实验目录，例如 outputs/xxx_run 或 results/train/xxx_run",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否弹窗显示图像",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=30,
        help="曲线平滑窗口大小",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exp_dir = _resolve_exp_dir(args.exp_dir)
    files = _find_run_files(exp_dir)
    plots_dir = files["plots_dir"]
    _ensure_dir(plots_dir)

    training_payload = _maybe_load_json(files["training_log_path"])
    eval_payload = _maybe_load_json(files["eval_log_path"])
    summary_payload = _maybe_load_json(files["summary_path"])
    manifest_payload = _maybe_load_json(files["manifest_path"])

    meta = _infer_run_meta(
        exp_dir=exp_dir,
        training_payload=training_payload,
        eval_payload=eval_payload,
        summary_payload=summary_payload,
        manifest_payload=manifest_payload,
    )

    print("========== 单实验绘图 ==========")
    print("exp_dir           :", exp_dir)
    print("training_log_path :", files["training_log_path"])
    print("eval_log_path     :", files["eval_log_path"])
    print("summary_path      :", files["summary_path"])
    print("manifest_path     :", files["manifest_path"])
    print("plots_dir         :", plots_dir)
    print("run_name          :", meta["run_name"])
    print("method_name       :", meta["method_name"])
    print("display_name      :", meta["display_name"])

    generated_files: List[str] = []

    # 1) 训练历史图
    if training_payload is not None:
        save_path = os.path.join(plots_dir, "single_experiment_training_history.png")
        ok = _plot_training_history_figure(
            training_payload=training_payload,
            save_path=save_path,
            title=f"{meta['display_name']} - Training History",
            smooth_window=args.smooth_window,
            show=args.show,
        )
        if ok:
            generated_files.append(save_path)

    # 2) 训练统计图
    train_stat_curves = _extract_train_stats_curves(training_payload)
    save_path = os.path.join(plots_dir, "single_experiment_train_stats.png")
    ok = _plot_train_stats_figure(
        train_stat_curves=train_stat_curves,
        save_path=save_path,
        title=f"{meta['display_name']} - Training Stats",
        smooth_window=args.smooth_window,
        show=args.show,
    )
    if ok:
        generated_files.append(save_path)

    # 3) eval 各 episode 指标图
    eval_records = _extract_eval_episode_records(eval_payload)
    eval_metric_curves = _extract_eval_metric_curves(eval_records)
    save_path = os.path.join(plots_dir, "single_experiment_eval_episode_metrics.png")
    ok = _plot_eval_episode_figure(
        eval_metric_curves=eval_metric_curves,
        save_path=save_path,
        title=f"{meta['display_name']} - Eval Episode Metrics",
        smooth_window=max(3, min(args.smooth_window, 10)),
        show=args.show,
    )
    if ok:
        generated_files.append(save_path)

    # 4) 汇总指标柱状图
    scalar_metrics = _extract_scalar_metrics(
        training_payload=training_payload,
        eval_payload=eval_payload,
        summary_payload=summary_payload,
    )
    save_path = os.path.join(plots_dir, "single_experiment_summary_metrics.png")
    ok = _plot_scalar_summary_figure(
        scalar_metrics=scalar_metrics,
        save_path=save_path,
        title=f"{meta['display_name']} - Summary Metrics",
        show=args.show,
    )
    if ok:
        generated_files.append(save_path)

    # 5) reward 组成图
    reward_components = _extract_tail_reward_components(training_payload)
    save_path = os.path.join(plots_dir, "single_experiment_reward_components.png")
    ok = _plot_reward_components_figure(
        reward_components=reward_components,
        save_path=save_path,
        title=f"{meta['display_name']} - Tail Reward Components",
        show=args.show,
    )
    if ok:
        generated_files.append(save_path)

    summary_out = {
        "exp_dir": exp_dir,
        "run_name": meta["run_name"],
        "method_name": meta["method_name"],
        "display_name": meta["display_name"],
        "source_files": {
            "training_log_path": files["training_log_path"],
            "eval_log_path": files["eval_log_path"],
            "summary_path": files["summary_path"],
            "manifest_path": files["manifest_path"],
        },
        "generated_plot_files": generated_files,
        "num_generated_plots": len(generated_files),
        "available_data": {
            "has_training_payload": training_payload is not None,
            "has_eval_payload": eval_payload is not None,
            "has_summary_payload": summary_payload is not None,
            "has_manifest_payload": manifest_payload is not None,
            "num_update_logs": len(_extract_update_logs(training_payload)),
            "num_train_episode_history": len(_extract_train_episode_history(training_payload)),
            "num_eval_episode_records": len(eval_records),
        },
        "scalar_metrics_used": scalar_metrics,
    }

    save_json(summary_out, os.path.join(plots_dir, "single_experiment_plot_summary.json"))

    print("\n===== 绘图完成 =====")
    if generated_files:
        for p in generated_files:
            print("生成:", p)
    else:
        print("未生成图片：没有找到足够的训练/评估数据。")
    print("summary :", os.path.join(plots_dir, "single_experiment_plot_summary.json"))


if __name__ == "__main__":
    main()