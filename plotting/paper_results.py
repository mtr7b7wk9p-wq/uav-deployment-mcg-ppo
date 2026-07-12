from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_scene import _configure_plot_style, _plot_raw_and_smooth
from utils.experiment_schema import DEFAULT_TRAINING_METRICS, get_training_curve_specs


METHOD_LABELS = {
    "ppo_main": "PPO",
    "ppo_localcritic": "PPO",
    "mcg_ppo": "MCG-PPO",
    "mcg_ppo_no_graph": "MCG-PPO w/o Graph",
    "mcg_ppo_no_mc_reward": "MCG-PPO w/o MC Reward",
    "mcg_ppo_no_overlap_guidance": "MCG-PPO w/o Guidance",
    "mcg_ppo_no_overlap_penalty": "MCG-PPO w/o Overlap",
    "ippo": "IPPO",
    "maddpg": "MADDPG",
    "ppo_wo_local_summary": "w/o Local Summary",
    "ppo_wo_guidance": "w/o Guidance",
    "ppo_wo_neighbor_uav": "w/o Neighbor UAV",
    "ppo_reward_coverage_only": "Coverage-Only Reward",
    "random": "Random",
    "random_masked": "Random",
    "greedy_local": "Greedy Local",
    "constrained_kmeans": "Constrained KMeans",
}

METHOD_COLORS = {
    "ppo_main": "#1f77b4",
    "mcg_ppo": "#d62728",
    "mcg_ppo_no_graph": "#17becf",
    "mcg_ppo_no_mc_reward": "#2ca02c",
    "mcg_ppo_no_overlap_guidance": "#bcbd22",
    "mcg_ppo_no_overlap_penalty": "#9467bd",
    "ippo": "#ff7f0e",
    "maddpg": "#8c564b",
    "random_masked": "#7f7f7f",
    "greedy_local": "#e377c2",
    "constrained_kmeans": "#8c564b",
}

EXPERIMENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
]

METRIC_PLOT_SPECS = {
    "final_coverage_ratio": {
        "title": "Final Coverage Ratio",
        "ylabel": "Coverage Ratio",
    },
    "final_covered_users": {
        "title": "Final Covered Users",
        "ylabel": "Users",
    },
    "total_move_distance": {
        "title": "Mean Total Move Distance",
        "ylabel": "Distance (m)",
    },
    "mean_overlap_users_step": {
        "title": "Mean Overlap Users / Step",
        "ylabel": "Users",
    },
    "episode_length": {
        "title": "Mean Episode Length",
        "ylabel": "Steps",
    },
    "full_coverage_success": {
        "title": "Full Coverage Success Rate",
        "ylabel": "Success Rate",
    },
}

DEFAULT_SEPARATE_METRICS = [
    "final_coverage_ratio",
    "final_covered_users",
    "total_move_distance",
    "mean_overlap_users_step",
]


def configure_paper_style() -> None:
    """
    这里直接复用 train_ppo_deployment 最终调用的 plot_scene 风格。
    这样 replot 出来的图和训练完成时自动生成的图保持一致。
    """
    _configure_plot_style()


def _ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _method_label(method_name: str) -> str:
    return METHOD_LABELS.get(method_name, method_name)


def _method_color(method_name: str) -> str:
    return METHOD_COLORS.get(method_name, "#1f77b4")


def _select_methods(
    compare_items: List[Dict[str, Any]],
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    items = list(compare_items)
    if include_methods is not None:
        include_set = set(include_methods)
        items = [x for x in items if x.get("method_name") in include_set]
    if exclude_methods is not None:
        exclude_set = set(exclude_methods)
        items = [x for x in items if x.get("method_name") not in exclude_set]
    return items


def _normalize_experiment_payloads(
    experiment_payloads: Sequence[Dict[str, Any]],
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, payload in enumerate(experiment_payloads):
        compare_items = list(payload.get("compare_items", []))
        compare_items = _select_methods(compare_items, include_methods, exclude_methods)
        normalized.append({
            "experiment_label": str(payload.get("experiment_label", f"Exp {idx + 1}")),
            "compare_items": compare_items,
        })
    return normalized


def _collect_method_order(
    experiment_payloads: Sequence[Dict[str, Any]],
) -> List[str]:
    method_order: List[str] = []
    seen = set()
    for payload in experiment_payloads:
        for item in payload.get("compare_items", []):
            name = item.get("method_name")
            if name and name not in seen:
                seen.add(name)
                method_order.append(name)
    return method_order


def _metric_value(item: Dict[str, Any], metric_key: str) -> float:
    value = item.get(metric_key, np.nan)
    if value is None:
        return float("nan")
    return float(value)


def _sanitize_series(values: Sequence[Any]) -> np.ndarray:
    clean: List[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            clean.append(f)
    return np.asarray(clean, dtype=np.float32)


def _choose_curve_color(item: Dict[str, Any], idx: int = 0) -> str:
    method_name = str(item.get("method_name") or "")
    color = item.get("color")
    if isinstance(color, str) and color:
        return color
    if method_name:
        return _method_color(method_name)
    return EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]


def _choose_curve_label(item: Dict[str, Any], idx: int = 0) -> str:
    label = item.get("label") or item.get("display_name") or item.get("experiment_label")
    if label:
        return str(label)
    method_name = item.get("method_name")
    if method_name:
        return _method_label(str(method_name))
    return f"Curve {idx + 1}"


def _make_safe_filename(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "figure"


# =========================
# 柱状图部分（保持原有功能）
# =========================
def plot_main_results_bars(
    compare_items: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Main results across different deployment methods",
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
    show: bool = False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    configure_paper_style()
    items = _select_methods(compare_items, include_methods, exclude_methods)

    methods = [x["method_name"] for x in items]
    labels = [_method_label(m) for m in methods]
    colors = [_method_color(m) for m in methods]

    final_coverage = [float(x.get("final_coverage_ratio", 0.0)) for x in items]
    total_move = [float(x.get("total_move_distance", 0.0)) for x in items]
    overlap = [float(x.get("mean_overlap_users_step", 0.0)) for x in items]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes = list(axes)

    axes[0].bar(x, final_coverage, color=colors)
    axes[0].set_title("Final Coverage Ratio")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Coverage Ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")

    axes[1].bar(x, total_move, color=colors)
    axes[1].set_title("Mean Total Move Distance")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Distance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")

    axes[2].bar(x, overlap, color=colors)
    axes[2].set_title("Mean Overlap Users / Step")
    axes[2].set_xlabel("Method")
    axes[2].set_ylabel("Users")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")

    fig.suptitle(title)

    if save_path is not None:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig, axes


def plot_ablation_bars(
    compare_items: List[Dict[str, Any]],
    main_method: str = "ppo_main",
    save_path: Optional[str] = None,
    title: str = "Ablation results of the proposed distributed PPO framework",
    show: bool = False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    ablation_names = [
        x["method_name"] for x in compare_items
        if x["method_name"].startswith("ppo_wo_") or x["method_name"].startswith("ppo_reward_")
    ]
    include_methods = [main_method] + ablation_names
    return plot_main_results_bars(
        compare_items=compare_items,
        save_path=save_path,
        title=title,
        include_methods=include_methods,
        show=show,
    )


def plot_single_metric_multi_experiment_bars(
    experiment_payloads: Sequence[Dict[str, Any]],
    metric_key: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
    max_experiments: int = 4,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    if metric_key not in METRIC_PLOT_SPECS:
        raise ValueError(f"Unsupported metric_key: {metric_key}")

    normalized = _normalize_experiment_payloads(
        experiment_payloads=experiment_payloads,
        include_methods=include_methods,
        exclude_methods=exclude_methods,
    )
    if len(normalized) == 0:
        raise ValueError("No experiment payloads were provided.")
    if len(normalized) > max_experiments:
        raise ValueError(f"At most {max_experiments} experiment payloads are supported, got {len(normalized)}.")

    configure_paper_style()
    method_order = _collect_method_order(normalized)
    if len(method_order) == 0:
        raise ValueError("No methods were found in experiment payloads.")

    labels = [_method_label(x) for x in method_order]
    x = np.arange(len(method_order))
    width = 0.76 / max(len(normalized), 1)

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8))
    for exp_idx, payload in enumerate(normalized):
        label = payload["experiment_label"]
        item_map = {item["method_name"]: item for item in payload.get("compare_items", [])}
        values = [_metric_value(item_map.get(method_name, {}), metric_key) for method_name in method_order]
        offsets = x + (exp_idx - (len(normalized) - 1) / 2.0) * width
        ax.bar(
            offsets,
            values,
            width=width * 0.92,
            label=label,
            color=EXPERIMENT_COLORS[exp_idx % len(EXPERIMENT_COLORS)],
            alpha=0.92,
        )

    spec = METRIC_PLOT_SPECS[metric_key]
    ax.set_title(spec["title"] if title is None else title)
    ax.set_xlabel("Method")
    ax.set_ylabel(spec["ylabel"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(frameon=False, ncol=min(len(normalized), 4))

    if save_path is not None:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig, ax


def plot_separate_metric_figures(
    experiment_payloads: Sequence[Dict[str, Any]],
    save_dir: str,
    file_prefix: str = "paper_compare",
    metric_keys: Optional[Sequence[str]] = None,
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
    max_experiments: int = 4,
    show: bool = False,
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    metric_list = list(metric_keys) if metric_keys is not None else list(DEFAULT_SEPARATE_METRICS)

    output_paths: Dict[str, str] = {}
    for metric_key in metric_list:
        save_path = os.path.join(save_dir, f"{file_prefix}_{metric_key}.png")
        plot_single_metric_multi_experiment_bars(
            experiment_payloads=experiment_payloads,
            metric_key=metric_key,
            save_path=save_path,
            include_methods=include_methods,
            exclude_methods=exclude_methods,
            max_experiments=max_experiments,
            show=show,
        )
        output_paths[metric_key] = save_path
    return output_paths


# =========================
# 训练曲线部分：按 train_ppo_deployment 同风格重绘
# =========================
def plot_training_curves(
    training_payloads: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Training convergence of the proposed method",
    show: bool = False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    configure_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes = list(axes)

    for method_name, payload in training_payloads.items():
        history = payload.get("train_episode_history", [])
        if len(history) == 0:
            continue

        eps = np.arange(1, len(history) + 1)
        coverage = [float(x.get("final_coverage_ratio", 0.0)) for x in history]
        returns = [float(x.get("episode_return", 0.0)) for x in history]

        label = _method_label(method_name)
        color = _method_color(method_name)

        axes[0].plot(eps, coverage, label=label, color=color)
        axes[1].plot(eps, returns, label=label, color=color)

    axes[0].set_title("Final Coverage Ratio vs Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Coverage Ratio")

    axes[1].set_title("Episode Return vs Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Return")

    for ax in axes:
        ax.legend(frameon=False)

    fig.suptitle(title)

    if save_path is not None:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig, axes


def plot_single_metric_training_curves(
    curve_items,
    metric_key,
    save_path=None,
    title=None,
    xlabel="Episode",
    smooth_window=200,
    show_raw=True,
    figure_size=(12.8, 7.2),
    show=False,
):
    """
    这里完全复用 train_ppo_deployment 的绘图细节：
    - 同样的 _configure_plot_style
    - 同样的 _plot_raw_and_smooth
    - 同样的 smooth_window 默认值风格
    区别只有一个：原来是一条方法曲线，现在是一张图里画多条方法曲线。
    """
    configure_paper_style()

    specs = get_training_curve_specs([metric_key])
    if metric_key not in specs:
        raise ValueError(f"Unsupported training metric: {metric_key}")
    spec = specs[metric_key]

    fig, ax = plt.subplots(figsize=figure_size)

    plotted_count = 0
    for idx, item in enumerate(curve_items):
        values = _sanitize_series(item.get("values", []))
        if len(values) == 0:
            continue

        x = np.arange(1, len(values) + 1)
        color = _choose_curve_color(item, idx)
        label = _choose_curve_label(item, idx)
        method_name = str(item.get("method_name", "")).strip()

        _plot_raw_and_smooth(
            ax=ax,
            x=x,
            y=values,
            color=color,
            label=label,
            smooth_window=smooth_window,
            show_raw=show_raw,
            metric_name=metric_key,
            method_name=method_name,
        )
        plotted_count += 1

    ax.set_title(title or spec.get("title", metric_key))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(spec.get("ylabel", metric_key))

    if plotted_count > 0:
        ax.legend(frameon=False)

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig, ax


def plot_training_curve_groups(
    compare_curve_items: Sequence[Dict[str, Any]],
    ablation_curve_items: Sequence[Dict[str, Any]],
    save_dir: str,
    metric_keys: Optional[Sequence[str]] = None,
    smooth_window: int = 200,
    show_raw: bool = True,
    show: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    现在不再输出“大图里四个子图”。
    改成：
    - compare 的每个指标单独一张图
    - ablation 的每个指标单独一张图
    """
    os.makedirs(save_dir, exist_ok=True)
    metric_list = list(metric_keys) if metric_keys is not None else list(DEFAULT_TRAINING_METRICS)

    outputs: Dict[str, Dict[str, str]] = {
        "compare_group": {},
        "ablation_group": {},
    }

    for metric_key in metric_list:
        metric_title = get_training_curve_specs([metric_key])[metric_key]["title"]

        if compare_curve_items:
            compare_path = os.path.join(save_dir, f"train_compare_{_make_safe_filename(metric_key)}.png")
            plot_single_metric_training_curves(
                curve_items=compare_curve_items,
                metric_key=metric_key,
                save_path=compare_path,
                title=metric_title,
                smooth_window=smooth_window,
                show_raw=show_raw,
                show=show,
            )
            outputs["compare_group"][metric_key] = compare_path

        if ablation_curve_items:
            ablation_path = os.path.join(save_dir, f"train_ablation_{_make_safe_filename(metric_key)}.png")
            plot_single_metric_training_curves(
                curve_items=ablation_curve_items,
                metric_key=metric_key,
                save_path=ablation_path,
                title=metric_title,
                smooth_window=smooth_window,
                show_raw=show_raw,
                show=show,
            )
            outputs["ablation_group"][metric_key] = ablation_path

    return outputs


def plot_step_level_coverage_growth(
    step_traces: Dict[str, List[Dict[str, Any]]],
    save_path: Optional[str] = None,
    title: str = "Coverage growth over deployment steps",
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    configure_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))

    for method_name, trace in step_traces.items():
        if len(trace) == 0:
            continue
        xs = [int(x.get("step", 0)) for x in trace]
        ys = [float(x.get("coverage_ratio", 0.0)) for x in trace]
        ax.plot(xs, ys, label=_method_label(method_name), color=_method_color(method_name))

    ax.set_title("Coverage Growth over Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage Ratio")
    ax.legend(frameon=False)

    fig.suptitle(title)

    if save_path is not None:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig, ax


def build_caption_templates() -> Dict[str, str]:
    return {
        "figure_main_results": "Comparison of final deployment performance across different methods under identical disaster response settings.",
        "figure_training_curve": "Training convergence of the proposed distributed PPO method in terms of final coverage ratio and episode return.",
        "figure_behavior": "Representative deployment trajectories and final coverage states produced by different methods in the same disaster scenario.",
        "figure_step_coverage": "Step-level coverage growth curves of different deployment methods under the same environment initialization.",
        "figure_ablation": "Impact of removing key components from the proposed distributed PPO framework.",
        "figure_separate_metrics": "Separate single-metric figures for paper-ready comparison, where each figure can merge up to four experimental result groups.",
        "figure_train_compare_group": "Training-stage comparison curves under the unified experimental protocol.",
        "figure_train_ablation_group": "Training-stage ablation curves for the proposed method and its reduced variants.",
        "table_main_results": "Quantitative comparison of coverage performance, movement cost, and overlap control across different deployment methods.",
        "table_ablation": "Ablation study of the proposed distributed PPO framework under the unified evaluation protocol.",
    }


def build_metric_definitions() -> Dict[str, Dict[str, str]]:
    return {
        "final_coverage_ratio": {
            "field_name": "final_coverage_ratio",
            "display_name": "Coverage Ratio",
            "formula": "Coverage Ratio = N_covered / N_users",
            "definition": "episode 结束时，被成功覆盖的用户数占总用户数的比例。",
        },
        "final_covered_users": {
            "field_name": "final_covered_users",
            "display_name": "Final Covered Users",
            "formula": "N_covered",
            "definition": "episode 结束时被覆盖的用户总数。",
        },
        "full_coverage_success_rate": {
            "field_name": "full_coverage_success_rate",
            "display_name": "Success Rate",
            "formula": "Success Rate = (1/M) * sum I(Coverage Ratio = 1)",
            "definition": "在多个 episode 中，实现全覆盖的比例。",
        },
        "mean_total_move_distance": {
            "field_name": "mean_total_move_distance",
            "display_name": "Total Movement Distance",
            "formula": "D_move = sum_i sum_t d_i^(t)",
            "definition": "一个 episode 内所有 UAV 的累计移动距离总和，再对多个 episode 取平均。",
        },
        "mean_mean_overlap_users_step": {
            "field_name": "mean_mean_overlap_users_step",
            "display_name": "Overlap Users per Step",
            "formula": "Overlap = (1/T) * sum_t N_overlap^(t)",
            "definition": "一个 episode 内每一步被多个 UAV 同时覆盖的用户数量的平均值，再对多个 episode 取平均。",
        },
        "mean_episode_length": {
            "field_name": "mean_episode_length",
            "display_name": "Episode Length",
            "formula": "T",
            "definition": "episode 的实际执行步数，再对多个 episode 取平均。",
        },
        "mean_episode_return": {
            "field_name": "mean_episode_return",
            "display_name": "Episode Return",
            "formula": "R = sum_t r_t",
            "definition": "一个 episode 内逐步奖励的累计回报。",
        },
    }