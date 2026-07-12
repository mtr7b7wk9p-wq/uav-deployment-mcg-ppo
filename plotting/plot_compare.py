from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "ppo_main": "#1f77b4",
    "ippo": "#ff7f0e",
    "maddpg": "#2ca02c",
    "mcg_ppo": "#d62728",

    "mcg_ppo_no_graph": "#9467bd",
    "mcg_ppo_no_mc_reward": "#17becf",
    "mcg_ppo_no_overlap_penalty": "#e377c2",
    "mcg_ppo_no_guidance": "#7f7f7f",

    "ppo_wo_local_summary": "#9467bd",
    "ppo_wo_guidance": "#17becf",
    "ppo_wo_neighbor_uav": "#e377c2",
    "ppo_reward_coverage_only": "#7f7f7f",

    "random_masked": "#7f7f7f",
    "random": "#7f7f7f",
    "greedy_local": "#bcbd22",
    "constrained_kmeans": "#8c564b",
}

METHOD_ORDER_PRIORITY = [
    "random_masked",
    "greedy_local",
    "constrained_kmeans",
    "ppo_main",
    "ippo",
    "maddpg",
    "mcg_ppo",
    "mcg_ppo_no_graph",
    "mcg_ppo_no_mc_reward",
    "mcg_ppo_no_overlap_penalty",
    "mcg_ppo_no_guidance",
    "ppo_wo_local_summary",
    "ppo_wo_guidance",
    "ppo_wo_neighbor_uav",
    "ppo_reward_coverage_only",
]


def _configure_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.alpha": 0.25,
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


def _pick(d: Dict, *keys, default=0.0):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _color_for_method(method_name: str) -> str:
    return METHOD_COLORS.get(method_name, "#4c4c4c")


def _label_for_item(item: Dict) -> str:
    if item.get("display_name"):
        return str(item["display_name"])
    if item.get("method_label"):
        return str(item["method_label"])
    return str(item.get("method_name", ""))


def _sort_items(items: List[Dict]) -> List[Dict]:
    def _key(x: Dict) -> Tuple[int, str]:
        name = str(x.get("method_name", ""))
        if name in METHOD_ORDER_PRIORITY:
            return (METHOD_ORDER_PRIORITY.index(name), name)
        return (10_000, name)

    return sorted(items, key=_key)


def plot_deployment_comparison(
    result_dicts: List[Dict],
    title: str = "Deployment Method Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    _configure_plot_style()

    result_dicts = _sort_items(result_dicts)

    method_names = [x["method_name"] for x in result_dicts]
    method_labels = [_label_for_item(x) for x in result_dicts]
    colors = [_color_for_method(m) for m in method_names]

    coverages = [_pick(x, "final_coverage_ratio", "mean_final_coverage_ratio") for x in result_dicts]
    covered_users = [_pick(x, "final_covered_users", "mean_final_covered_users") for x in result_dicts]
    total_distance = [_pick(x, "total_move_distance", "mean_total_move_distance", "mean_total_distance") for x in result_dicts]
    overlap_users = [_pick(x, "mean_overlap_users_step", "mean_mean_overlap_users_step") for x in result_dicts]

    x = np.arange(len(result_dicts))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = list(axes.flatten())

    axes[0].bar(x, coverages, color=colors)
    axes[0].set_title("Final Coverage Ratio")
    axes[0].set_ylabel("Coverage Ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels, rotation=15, ha="right")

    axes[1].bar(x, covered_users, color=colors)
    axes[1].set_title("Final Covered Users")
    axes[1].set_ylabel("Users")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_labels, rotation=15, ha="right")

    axes[2].bar(x, total_distance, color=colors)
    axes[2].set_title("Mean Total Move Distance")
    axes[2].set_ylabel("Distance (m)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(method_labels, rotation=15, ha="right")

    axes[3].bar(x, overlap_users, color=colors)
    axes[3].set_title("Mean Overlap Users / Step")
    axes[3].set_ylabel("Users")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(method_labels, rotation=15, ha="right")

    fig.suptitle(title)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    return fig, axes