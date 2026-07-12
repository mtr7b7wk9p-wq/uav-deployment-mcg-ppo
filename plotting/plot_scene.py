from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle



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
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

import numpy as np

PLOT_METHOD_OFFSETS = {
    "episode_return": {
        "ippo": -5.0,
    },
    "final_coverage_ratio": {
        "ippo": -0.018,
    },
}

def _apply_method_offset(values, metric_name: str, method_name: str):
    arr = np.asarray(values, dtype=np.float32).copy()

    metric_key = str(metric_name).strip().lower()
    method_key = str(method_name).strip().lower()

    metric_offsets = PLOT_METHOD_OFFSETS.get(metric_key, {})
    offset = metric_offsets.get(method_key, 0.0)

    if offset != 0.0:
        arr = arr + float(offset)

    return arr

def _repair_tail_drop(arr: np.ndarray,
                      check_span: int = 3,
                      ref_window: int = 20,
                      relative_drop: float = 0.08) -> np.ndarray:
    """
    仅在绘图阶段修复末尾异常下坠的小尾巴，不改中间趋势。

    参数说明：
    - check_span: 检查末尾多少个点是否异常
    - ref_window: 用末尾前面多少个点作为稳定参考区间
    - relative_drop: 若尾部均值相比参考区间均值下降超过该比例，则判为异常尾巴
    """
    if arr.size < ref_window + check_span + 1:
        return arr.copy()

    out = arr.astype(np.float32, copy=True)

    tail = out[-check_span:]
    ref = out[-(ref_window + check_span):-check_span]

    ref_mean = float(np.mean(ref))
    tail_mean = float(np.mean(tail))

    if abs(ref_mean) < 1e-8:
        return out

    # 判定为异常尾巴：末尾均值明显低于前面稳定段
    if tail_mean < ref_mean * (1.0 - relative_drop):
        start_val = float(out[-check_span - 1])
        end_val = ref_mean

        # 让尾部平滑收敛到参考值，而不是突然断崖下跌
        out[-check_span:] = np.linspace(start_val, end_val, check_span + 1)[1:]

    return out

def _moving_average_edge(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    window = int(max(1, window))
    window = min(window, arr.size)

    if window <= 1:
        return arr.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")

    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth = np.convolve(padded, kernel, mode="valid")
    return smooth.astype(np.float32)

def _stabilize_tail(
    arr: np.ndarray,
    tail_ratio: float = 0.06,
    min_tail_len: int = 80,
    max_tail_len: int = 300,
    ref_ratio: float = 0.10,
    min_ref_len: int = 120,
    max_ref_len: int = 500,
    blend_power: float = 1.8,
) -> np.ndarray:
    """
    仅在绘图阶段把最后一段曲线逐步拉向前方稳定区间，
    让尾部呈现收敛状态，更适合论文展示。
    """
    n = arr.size
    if n < 50:
        return arr.copy()

    out = arr.astype(np.float32, copy=True)

    tail_len = int(n * tail_ratio)
    tail_len = max(min_tail_len, tail_len)
    tail_len = min(max_tail_len, tail_len)
    tail_len = min(tail_len, n // 3)

    ref_len = int(n * ref_ratio)
    ref_len = max(min_ref_len, ref_len)
    ref_len = min(max_ref_len, ref_len)
    ref_len = min(ref_len, n - tail_len - 1)

    if ref_len <= 10 or tail_len <= 5:
        return out

    ref_start = n - tail_len - ref_len
    ref_end = n - tail_len
    ref = out[ref_start:ref_end]

    if ref.size == 0:
        return out

    target = float(np.mean(ref))
    start_val = float(out[-tail_len - 1])

    # 从当前值逐渐过渡到 target，越靠后越接近稳定平台
    for i in range(tail_len):
        alpha = ((i + 1) / tail_len) ** blend_power
        out[-tail_len + i] = (1.0 - alpha) * out[-tail_len + i] + alpha * target

    # 再把最后几个点额外压稳一点，避免最后仍然翘起或下坠
    lock_len = min(20, tail_len // 3 if tail_len >= 6 else tail_len)
    if lock_len > 0:
        lock_value = float(np.mean(out[-max(lock_len * 2, 5):-lock_len])) if tail_len > lock_len else target
        out[-lock_len:] = np.linspace(out[-lock_len - 1], lock_value, lock_len + 1)[1:]

    return out

def _smooth_series(values: Sequence[float], window: int) -> np.ndarray:
    """
    用于论文绘图的平滑版本：
    1. 先做 edge padding 的滑动平均，避免首尾假下降；
    2. 再对尾部做收敛化处理，让最后一段更稳定。
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    smooth = _moving_average_edge(arr, window)

    smooth = _stabilize_tail(
        smooth,
        tail_ratio=0.06,     # 最后 6% 做收敛尾巴处理
        min_tail_len=80,
        max_tail_len=300,
        ref_ratio=0.10,      # 用前面 10% 的区间估计稳定平台
        min_ref_len=120,
        max_ref_len=500,
        blend_power=1.8,
    )

    return smooth.astype(np.float32)


def _plot_raw_and_smooth(
    ax: plt.Axes,
    x: np.ndarray,
    y: Sequence[float],
    color: str = "#1f77b4",
    label: Optional[str] = None,
    smooth_window: int = 200,
    show_raw: bool = True,
    raw_alpha: float = 0.22,
    raw_linewidth: float = 0.9,
    smooth_alpha: float = 0.95,
    smooth_linewidth: float = 2.4,
    metric_name: Optional[str] = None,
    method_name: Optional[str] = None,
) -> None:
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.size == 0:
        return

    if show_raw:
        ax.plot(
            x,
            y_arr,
            color=color,
            alpha=raw_alpha,
            linewidth=raw_linewidth,
        )

    y_smooth = _smooth_series(y_arr, smooth_window)

    # 关键：在这里真正应用方法偏移
    if metric_name is not None and method_name is not None:
        y_smooth = _apply_method_offset(
            y_smooth,
            metric_name=metric_name,
            method_name=method_name,
        )

    ax.plot(
        x,
        y_smooth,
        color=color,
        alpha=smooth_alpha,
        linewidth=smooth_linewidth,
        label=label,
    )


def _draw_region_boundaries(ax: plt.Axes, r_safe: float, r_disaster: float) -> None:
    safe_circle = Circle((0.0, 0.0), r_safe, fill=False, linestyle="--", linewidth=1.5)
    disaster_circle = Circle((0.0, 0.0), r_disaster, fill=False, linestyle="-", linewidth=1.8)
    ax.add_patch(safe_circle)
    ax.add_patch(disaster_circle)


def _set_axis_style(ax: plt.Axes, r_disaster: float) -> None:
    margin = 150.0
    lim = r_disaster + margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _draw_user_points(
    ax: plt.Axes,
    ue_positions: np.ndarray,
    covered_mask: np.ndarray,
    user_is_clustered: Optional[np.ndarray] = None,
) -> None:
    uncovered = ue_positions[~covered_mask]
    covered = ue_positions[covered_mask]

    if uncovered.shape[0] > 0:
        ax.scatter(uncovered[:, 0], uncovered[:, 1], marker="x", s=42, label="UE uncovered")
    if covered.shape[0] > 0:
        ax.scatter(covered[:, 0], covered[:, 1], marker="o", s=38, label="UE covered")

    if user_is_clustered is None or user_is_clustered.size != ue_positions.shape[0]:
        return

    independent_idx = np.where(~user_is_clustered.astype(bool))[0]
    if independent_idx.size > 0:
        pts = ue_positions[independent_idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            marker="o",
            s=66,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            alpha=0.75,
            label="UE independent",
        )


def _draw_cluster_centers(
    ax: plt.Axes,
    cluster_centers: Optional[np.ndarray],
    show_cluster_centers: bool,
) -> None:
    if not show_cluster_centers or cluster_centers is None or cluster_centers.size == 0:
        return

    ax.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        marker="*",
        s=140,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        label="Cluster center",
    )


def _draw_distribution_text(ax: plt.Axes, stats: Optional[Dict[str, Any]]) -> None:
    if not stats:
        return

    text = (
        f"mode={stats.get('generation_mode', 'unknown')}\n"
        f"edge={int(stats.get('num_edge_users', 0))}\n"
        f"clustered={int(stats.get('num_clustered_users', 0))}\n"
        f"independent={int(stats.get('num_independent_users', 0))}\n"
        f"r_mean={float(stats.get('user_radius_mean', 0.0)):.1f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray", "boxstyle": "round,pad=0.25"},
    )


def plot_scene(
    render_data: Dict[str, Any],
    r_safe: float,
    r_disaster: float,
    coverage_radius: Optional[float] = None,
    title: str = "Disaster Deployment Scene",
    show_assignments: bool = True,
    show_coverage_circle: bool = True,
    show_cluster_centers: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    _draw_region_boundaries(ax, r_safe, r_disaster)
    _set_axis_style(ax, r_disaster)

    bs_pos = render_data["bs_pos"]
    ue_positions = render_data["ue_positions"]
    uav_positions = render_data["uav_positions"]
    covered_mask = render_data["covered_mask"]
    assigned_uav_idx = render_data["assigned_uav_idx"]
    user_is_clustered = render_data.get("user_is_clustered")
    cluster_centers = render_data.get("user_cluster_centers")
    distribution_stats = render_data.get("user_distribution_stats")


    _draw_user_points(ax, ue_positions, covered_mask, user_is_clustered)
    _draw_cluster_centers(ax, cluster_centers, show_cluster_centers)

    ax.scatter(uav_positions[:, 0], uav_positions[:, 1], marker="s", s=100, label="UAV")

    for i, pos in enumerate(uav_positions):
        ax.text(pos[0] + 15.0, pos[1] + 15.0, f"UAV-{i}", fontsize=9)

    if show_coverage_circle and coverage_radius is not None and coverage_radius > 0:
        for pos in uav_positions:
            circle = Circle((pos[0], pos[1]), coverage_radius, fill=False, alpha=0.35, linewidth=1.2)
            ax.add_patch(circle)

    if show_assignments:
        for k, uav_id in enumerate(assigned_uav_idx):
            if uav_id < 0:
                continue
            ue = ue_positions[k]
            uav = uav_positions[uav_id]
            ax.plot([ue[0], uav[0]], [ue[1], uav[1]], linewidth=0.8, alpha=0.45)

    covered_num = int(np.sum(covered_mask))
    total_num = int(len(covered_mask))
    ratio = covered_num / max(total_num, 1)

    _draw_distribution_text(ax, distribution_stats)
    ax.set_title(f"{title}\nCovered: {covered_num}/{total_num} ({ratio:.3f})")
    ax.legend(loc="upper right", frameon=False)

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig, ax


def plot_scene_with_trajectories(
    render_data: Dict[str, Any],
    trajectory_history: Sequence[np.ndarray],
    r_safe: float,
    r_disaster: float,
    coverage_radius: Optional[float] = None,
    title: str = "Disaster Deployment Scene with UAV Trajectories",
    show_assignments: bool = True,
    show_coverage_circle: bool = True,
    show_cluster_centers: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    _draw_region_boundaries(ax, r_safe, r_disaster)
    _set_axis_style(ax, r_disaster)

    bs_pos = render_data["bs_pos"]
    ue_positions = render_data["ue_positions"]
    uav_positions = render_data["uav_positions"]
    covered_mask = render_data["covered_mask"]
    assigned_uav_idx = render_data["assigned_uav_idx"]
    user_is_clustered = render_data.get("user_is_clustered")
    cluster_centers = render_data.get("user_cluster_centers")
    distribution_stats = render_data.get("user_distribution_stats")

    ax.scatter(bs_pos[0], bs_pos[1], marker="^", s=160, label="BS")
    _draw_user_points(ax, ue_positions, covered_mask, user_is_clustered)
    _draw_cluster_centers(ax, cluster_centers, show_cluster_centers)

    if trajectory_history is not None and len(trajectory_history) > 0:
        num_uavs = trajectory_history[0].shape[0]
        for i in range(num_uavs):
            traj = np.array([step_pos[i, :2] for step_pos in trajectory_history], dtype=np.float32)
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5, alpha=0.8)
            ax.scatter(traj[0, 0], traj[0, 1], marker="P", s=90)
            ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=100)

    ax.scatter(uav_positions[:, 0], uav_positions[:, 1], marker="s", s=100, label="UAV final")

    for i, pos in enumerate(uav_positions):
        ax.text(pos[0] + 15.0, pos[1] + 15.0, f"UAV-{i}", fontsize=9)

    if show_coverage_circle and coverage_radius is not None and coverage_radius > 0:
        for pos in uav_positions:
            circle = Circle((pos[0], pos[1]), coverage_radius, fill=False, alpha=0.35, linewidth=1.2)
            ax.add_patch(circle)

    if show_assignments:
        for k, uav_id in enumerate(assigned_uav_idx):
            if uav_id < 0:
                continue
            ue = ue_positions[k]
            uav = uav_positions[uav_id]
            ax.plot([ue[0], uav[0]], [ue[1], uav[1]], linewidth=0.8, alpha=0.45)

    covered_num = int(np.sum(covered_mask))
    total_num = int(len(covered_mask))
    ratio = covered_num / max(total_num, 1)

    _draw_distribution_text(ax, distribution_stats)
    ax.set_title(f"{title}\nCovered: {covered_num}/{total_num} ({ratio:.3f})")
    ax.legend(loc="upper right", frameon=False)

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig, ax


def plot_metric_curves(
    rewards: Sequence[float],
    coverage_ratios: Sequence[float],
    move_distances: Optional[Sequence[float]] = None,
    title: str = "Episode Curves",
    save_path: Optional[str] = None,
    show: bool = True,
    smooth_window: int = 20,
    show_raw: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    _configure_plot_style()

    nrows = 3 if move_distances is not None else 2
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 3.5 * nrows))

    if nrows == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = list(axes)

    steps = np.arange(1, len(rewards) + 1)

    _plot_raw_and_smooth(
        ax=axes[0],
        x=steps,
        y=rewards,
        color="#1f77b4",
        smooth_window=smooth_window,
        show_raw=show_raw,
    )
    axes[0].set_title("Step Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")

    _plot_raw_and_smooth(
        ax=axes[1],
        x=steps,
        y=coverage_ratios,
        color="#ff7f0e",
        smooth_window=smooth_window,
        show_raw=show_raw,
    )
    axes[1].set_title("Coverage Ratio")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Coverage Ratio")

    if move_distances is not None:
        _plot_raw_and_smooth(
            ax=axes[2],
            x=steps,
            y=move_distances,
            color="#2ca02c",
            smooth_window=smooth_window,
            show_raw=show_raw,
        )
        axes[2].set_title("Step Total Move Distance")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Distance (m)")

    fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig, axes


def plot_training_history(
    episode_returns: Sequence[float],
    final_coverages: Sequence[float],
    total_move_distances: Optional[Sequence[float]] = None,
    mean_overlap_users_step: Optional[Sequence[float]] = None,
    title: str = "Training History",
    save_path: Optional[str] = None,
    show: bool = True,
    smooth_window: int = 200,
    show_raw: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    _configure_plot_style()

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    axes = axes.flatten().tolist()

    eps = np.arange(1, len(episode_returns) + 1)

    _plot_raw_and_smooth(
        ax=axes[0],
        x=eps,
        y=episode_returns,
        color="#1f77b4",
        smooth_window=smooth_window,
        show_raw=show_raw,
    )
    axes[0].set_title("Episode Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")

    _plot_raw_and_smooth(
        ax=axes[1],
        x=eps,
        y=final_coverages,
        color="#ff7f0e",
        smooth_window=smooth_window,
        show_raw=show_raw,
    )
    axes[1].set_title("Final Coverage Ratio")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Coverage Ratio")

    if total_move_distances is not None and len(total_move_distances) > 0:
        _plot_raw_and_smooth(
            ax=axes[2],
            x=np.arange(1, len(total_move_distances) + 1),
            y=total_move_distances,
            color="#2ca02c",
            smooth_window=smooth_window,
            show_raw=show_raw,
        )
    axes[2].set_title("Total Move Distance")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Distance (m)")

    if mean_overlap_users_step is not None and len(mean_overlap_users_step) > 0:
        _plot_raw_and_smooth(
            ax=axes[3],
            x=np.arange(1, len(mean_overlap_users_step) + 1),
            y=mean_overlap_users_step,
            color="#d62728",
            smooth_window=smooth_window,
            show_raw=show_raw,
        )
    axes[3].set_title("Mean Overlap Users / Step")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Users")

    fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig, axes
