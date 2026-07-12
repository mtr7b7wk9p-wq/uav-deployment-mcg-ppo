import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def set_random_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def euclidean_distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def euclidean_distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1[:3] - p2[:3]))


def norm_2d(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def is_in_ring(x: float, y: float, r_inner: float, r_outer: float) -> bool:
    r = norm_2d(x, y)
    return r_inner <= r <= r_outer


def clip_point_to_ring(
    x: float,
    y: float,
    r_inner: float,
    r_outer: float,
) -> Tuple[float, float]:
    """
    Project a point onto the valid annulus [r_inner, r_outer].
    """
    r = norm_2d(x, y)

    if r == 0.0:
        return r_inner, 0.0

    if r < r_inner:
        scale = r_inner / r
        return x * scale, y * scale

    if r > r_outer:
        scale = r_outer / r
        return x * scale, y * scale

    return x, y


def sample_points_in_annulus(
    num_points: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Uniformly sample points in an annulus by area.
    Return shape: [num_points, 2]
    """
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    theta = rng.uniform(0.0, 2.0 * math.pi, size=num_points)
    r2 = rng.uniform(r_inner * r_inner, r_outer * r_outer, size=num_points)
    r = np.sqrt(r2)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y], axis=1).astype(np.float32)
    return pts


def _sample_radius_beta(
    num_points: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
    beta_a: float = 2.2,
    beta_b: float = 3.0,
) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0,), dtype=np.float32)

    t = rng.beta(beta_a, beta_b, size=num_points)
    r = r_inner + (r_outer - r_inner) * t
    return r.astype(np.float32)


def _sample_points_from_radius(
    radii: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if radii.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    theta = rng.uniform(0.0, 2.0 * math.pi, size=radii.shape[0])
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_points_in_annulus_weighted(
    num_points: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
    edge_avoidance_ratio: float = 0.15,
    edge_soft_limit_ratio: float = 0.20,
    radial_beta_a: float = 2.2,
    radial_beta_b: float = 3.0,
    max_resample_rounds: int = 8,
) -> np.ndarray:
    """
    Bias point sampling toward the middle annulus region.

    Compared with uniform-by-area sampling, this function reduces the probability
    that too many points fall into the outermost edge band.
    """
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    radii = _sample_radius_beta(
        num_points=num_points,
        r_inner=r_inner,
        r_outer=r_outer,
        rng=rng,
        beta_a=radial_beta_a,
        beta_b=radial_beta_b,
    )
    pts = _sample_points_from_radius(radii, rng)

    span = max(r_outer - r_inner, 1e-6)
    edge_threshold = r_outer - edge_soft_limit_ratio * span
    max_edge_points = int(math.floor(num_points * max(edge_avoidance_ratio, 0.0)))

    for _ in range(max_resample_rounds):
        radii_now = np.linalg.norm(pts, axis=1)
        edge_idx = np.where(radii_now >= edge_threshold)[0]
        if edge_idx.size <= max_edge_points:
            break

        need_resample = edge_idx[max_edge_points:]
        repl_r = _sample_radius_beta(
            num_points=need_resample.size,
            r_inner=r_inner,
            r_outer=edge_threshold,
            rng=rng,
            beta_a=max(radial_beta_a, 1.2),
            beta_b=max(radial_beta_b, 1.8),
        )
        pts[need_resample] = _sample_points_from_radius(repl_r, rng)

    return pts.astype(np.float32)


def sample_cluster_centers_in_annulus(
    num_clusters: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
    min_radius_ratio: float = 0.18,
    max_radius_ratio: float = 0.72,
    min_center_separation: Optional[float] = None,
    radial_beta_a: float = 2.0,
    radial_beta_b: float = 2.4,
    max_trials_per_center: int = 60,
) -> np.ndarray:
    """
    Sample cluster centers in the middle annulus band.
    """
    if num_clusters <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    span = max(r_outer - r_inner, 1e-6)
    center_r_min = r_inner + min_radius_ratio * span
    center_r_max = r_inner + max_radius_ratio * span
    center_r_min = min(max(center_r_min, r_inner), r_outer)
    center_r_max = min(max(center_r_max, center_r_min + 1e-6), r_outer)

    if min_center_separation is None:
        min_center_separation = max(120.0, 0.8 * span / max(num_clusters, 1))

    centers: List[np.ndarray] = []
    for _ in range(num_clusters):
        picked = None
        for _ in range(max_trials_per_center):
            radius = _sample_radius_beta(
                1,
                center_r_min,
                center_r_max,
                rng,
                beta_a=radial_beta_a,
                beta_b=radial_beta_b,
            )
            cand = _sample_points_from_radius(radius, rng)[0]
            if len(centers) == 0:
                picked = cand
                break

            d = np.linalg.norm(np.stack(centers, axis=0) - cand[None, :], axis=1)
            if np.all(d >= float(min_center_separation)):
                picked = cand
                break

        if picked is None:
            radius = _sample_radius_beta(1, center_r_min, center_r_max, rng)
            picked = _sample_points_from_radius(radius, rng)[0]
        centers.append(np.asarray(picked, dtype=np.float32))

    return np.stack(centers, axis=0).astype(np.float32)


def _allocate_cluster_sizes(num_points: int, num_clusters: int) -> np.ndarray:
    if num_points <= 0 or num_clusters <= 0:
        return np.zeros((0,), dtype=np.int32)

    sizes = np.full((num_clusters,), num_points // num_clusters, dtype=np.int32)
    sizes[: num_points % num_clusters] += 1
    return sizes


def sample_user_points_clustered(
    num_points: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
    num_clusters: int = 3,
    cluster_radius: float = 220.0,
    edge_avoidance_ratio: float = 0.15,
    edge_soft_limit_ratio: float = 0.20,
    cluster_center_min_radius_ratio: float = 0.18,
    cluster_center_max_radius_ratio: float = 0.72,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if num_points <= 0:
        meta = {
            "user_is_clustered": np.zeros((0,), dtype=bool),
            "user_cluster_ids": np.zeros((0,), dtype=np.int32),
            "user_cluster_centers": np.zeros((0, 2), dtype=np.float32),
            "generation_mode": "clustered",
        }
        return np.zeros((0, 2), dtype=np.float32), meta

    num_clusters = max(1, min(int(num_clusters), num_points))
    centers = sample_cluster_centers_in_annulus(
        num_clusters=num_clusters,
        r_inner=r_inner,
        r_outer=r_outer,
        rng=rng,
        min_radius_ratio=cluster_center_min_radius_ratio,
        max_radius_ratio=cluster_center_max_radius_ratio,
        min_center_separation=max(140.0, cluster_radius * 1.35),
    )

    span = max(r_outer - r_inner, 1e-6)
    edge_threshold = r_outer - edge_soft_limit_ratio * span
    max_edge_points = int(math.floor(num_points * max(edge_avoidance_ratio, 0.0)))
    sigma = max(cluster_radius * 0.42, 1.0)

    pts = np.zeros((num_points, 2), dtype=np.float32)
    cluster_ids = np.full((num_points,), -1, dtype=np.int32)
    cursor = 0
    edge_count = 0

    cluster_sizes = _allocate_cluster_sizes(num_points, num_clusters)
    for cluster_id, size in enumerate(cluster_sizes.tolist()):
        center = centers[cluster_id]
        for _ in range(size):
            picked = None
            for _ in range(120):
                offset = rng.normal(loc=0.0, scale=sigma, size=2).astype(np.float32)
                if float(np.linalg.norm(offset)) > cluster_radius:
                    continue
                cand = center + offset
                radius = float(np.linalg.norm(cand))
                if radius < r_inner or radius > r_outer:
                    continue
                if radius >= edge_threshold and edge_count >= max_edge_points:
                    continue
                picked = cand
                break

            if picked is None:
                direction = center / max(float(np.linalg.norm(center)), 1e-6)
                fallback_radius = min(max(float(np.linalg.norm(center)), r_inner + 5.0), edge_threshold - 5.0)
                picked = direction * fallback_radius

            if float(np.linalg.norm(picked)) >= edge_threshold:
                edge_count += 1

            pts[cursor] = picked.astype(np.float32)
            cluster_ids[cursor] = cluster_id
            cursor += 1

    user_is_clustered = np.ones((num_points,), dtype=bool)
    meta = {
        "user_is_clustered": user_is_clustered,
        "user_cluster_ids": cluster_ids,
        "user_cluster_centers": centers.astype(np.float32),
        "generation_mode": "clustered",
    }
    return pts.astype(np.float32), meta


def sample_user_points_mixed(
    num_points: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
    num_clusters: int = 3,
    clustered_ratio: float = 0.80,
    cluster_radius: float = 220.0,
    edge_avoidance_ratio: float = 0.15,
    edge_soft_limit_ratio: float = 0.20,
    cluster_center_min_radius_ratio: float = 0.18,
    cluster_center_max_radius_ratio: float = 0.72,
    radial_beta_a: float = 2.2,
    radial_beta_b: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if num_points <= 0:
        meta = {
            "user_is_clustered": np.zeros((0,), dtype=bool),
            "user_cluster_ids": np.zeros((0,), dtype=np.int32),
            "user_cluster_centers": np.zeros((0, 2), dtype=np.float32),
            "generation_mode": "mixed",
        }
        return np.zeros((0, 2), dtype=np.float32), meta

    clustered_ratio = float(np.clip(clustered_ratio, 0.0, 1.0))
    num_clustered = int(round(num_points * clustered_ratio))
    if num_points >= 2:
        num_clustered = min(max(num_clustered, 1), num_points - 1)
    else:
        num_clustered = num_points
    num_independent = num_points - num_clustered

    clustered_pts, clustered_meta = sample_user_points_clustered(
        num_points=num_clustered,
        r_inner=r_inner,
        r_outer=r_outer,
        rng=rng,
        num_clusters=num_clusters,
        cluster_radius=cluster_radius,
        edge_avoidance_ratio=edge_avoidance_ratio,
        edge_soft_limit_ratio=edge_soft_limit_ratio,
        cluster_center_min_radius_ratio=cluster_center_min_radius_ratio,
        cluster_center_max_radius_ratio=cluster_center_max_radius_ratio,
    )

    independent_pts = sample_points_in_annulus_weighted(
        num_points=num_independent,
        r_inner=r_inner,
        r_outer=r_outer,
        rng=rng,
        edge_avoidance_ratio=edge_avoidance_ratio,
        edge_soft_limit_ratio=edge_soft_limit_ratio,
        radial_beta_a=radial_beta_a,
        radial_beta_b=radial_beta_b,
    )

    pts = np.concatenate([clustered_pts, independent_pts], axis=0).astype(np.float32)
    user_is_clustered = np.concatenate(
        [
            np.ones((num_clustered,), dtype=bool),
            np.zeros((num_independent,), dtype=bool),
        ],
        axis=0,
    )
    user_cluster_ids = np.concatenate(
        [
            clustered_meta["user_cluster_ids"].astype(np.int32),
            np.full((num_independent,), -1, dtype=np.int32),
        ],
        axis=0,
    )

    perm = rng.permutation(num_points)
    pts = pts[perm]
    user_is_clustered = user_is_clustered[perm]
    user_cluster_ids = user_cluster_ids[perm]

    meta = {
        "user_is_clustered": user_is_clustered,
        "user_cluster_ids": user_cluster_ids,
        "user_cluster_centers": clustered_meta["user_cluster_centers"].astype(np.float32),
        "generation_mode": "mixed",
    }
    return pts.astype(np.float32), meta


def compute_user_distribution_stats(
    points_xy: np.ndarray,
    r_inner: float,
    r_outer: float,
    edge_soft_limit_ratio: float,
    user_is_clustered: Optional[np.ndarray] = None,
    user_cluster_ids: Optional[np.ndarray] = None,
    generation_mode: str = "unknown",
) -> Dict[str, Any]:
    num_users = int(points_xy.shape[0])
    radii = np.linalg.norm(points_xy, axis=1) if num_users > 0 else np.zeros((0,), dtype=np.float32)
    span = max(r_outer - r_inner, 1e-6)
    edge_threshold = float(r_outer - edge_soft_limit_ratio * span)
    edge_mask = radii >= edge_threshold if num_users > 0 else np.zeros((0,), dtype=bool)

    stats: Dict[str, Any] = {
        "generation_mode": generation_mode,
        "num_users": num_users,
        "edge_threshold_radius": edge_threshold,
        "num_edge_users": int(np.sum(edge_mask)),
        "edge_user_ratio": float(np.mean(edge_mask.astype(np.float32))) if num_users > 0 else 0.0,
        "user_radius_mean": float(np.mean(radii)) if num_users > 0 else 0.0,
        "user_radius_std": float(np.std(radii)) if num_users > 0 else 0.0,
        "num_clustered_users": 0,
        "num_independent_users": num_users,
        "num_clusters_present": 0,
    }

    if user_is_clustered is not None and user_is_clustered.size == num_users:
        stats["num_clustered_users"] = int(np.sum(user_is_clustered.astype(np.int32)))
        stats["num_independent_users"] = int(num_users - stats["num_clustered_users"])

    if user_cluster_ids is not None and user_cluster_ids.size == num_users:
        valid_ids = user_cluster_ids[user_cluster_ids >= 0]
        stats["num_clusters_present"] = int(np.unique(valid_ids).size) if valid_ids.size > 0 else 0

    return stats


def validate_user_distribution(
    points_xy: np.ndarray,
    r_inner: float,
    r_outer: float,
    edge_soft_limit_ratio: float,
    edge_avoidance_ratio: float,
    generation_mode: str,
    user_is_clustered: Optional[np.ndarray] = None,
    user_cluster_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    stats = compute_user_distribution_stats(
        points_xy=points_xy,
        r_inner=r_inner,
        r_outer=r_outer,
        edge_soft_limit_ratio=edge_soft_limit_ratio,
        user_is_clustered=user_is_clustered,
        user_cluster_ids=user_cluster_ids,
        generation_mode=generation_mode,
    )

    num_users = max(int(stats["num_users"]), 1)
    radii = np.linalg.norm(points_xy, axis=1) if points_xy.size > 0 else np.zeros((0,), dtype=np.float32)
    inside_ring = bool(np.all((radii >= (r_inner - 1e-6)) & (radii <= (r_outer + 1e-6))))
    max_edge_users = int(math.ceil(num_users * max(edge_avoidance_ratio, 0.0)))
    edge_ok = int(stats["num_edge_users"]) <= max_edge_users

    mixed_ok = True
    if generation_mode == "mixed":
        mixed_ok = (
            int(stats["num_clustered_users"]) > 0
            and int(stats["num_independent_users"]) > 0
            and int(stats["num_clusters_present"]) > 0
        )

    result: Dict[str, Any] = dict(stats)
    result.update({
        "inside_ring_ok": inside_ring,
        "edge_user_limit": max_edge_users,
        "edge_user_ok": edge_ok if generation_mode != "uniform" else True,
        "mixed_mode_ok": mixed_ok,
        "is_valid": bool(inside_ring and (edge_ok if generation_mode != "uniform" else True) and mixed_ok),
    })
    return result


def make_uav_init_positions_circle(
    num_uavs: int,
    radius: float,
) -> np.ndarray:
    """
    Equally place UAVs on a circle boundary.
    Return shape: [num_uavs, 2]
    """
    if num_uavs <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for i in range(num_uavs):
        theta = 2.0 * math.pi * i / num_uavs
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        pts.append([x, y])

    return np.array(pts, dtype=np.float32)


def make_uav_init_positions_center(
    num_uavs: int,
    radius_spread: float = 30.0,
) -> np.ndarray:
    """
    Place UAVs around center with small spread.
    """
    if num_uavs <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for i in range(num_uavs):
        theta = 2.0 * math.pi * i / max(1, num_uavs)
        x = radius_spread * math.cos(theta)
        y = radius_spread * math.sin(theta)
        pts.append([x, y])

    return np.array(pts, dtype=np.float32)


def pairwise_distances_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N, 2]
    b: [M, 2]
    return: [N, M]
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    diff = a[:, None, :] - b[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.astype(np.float32)


def pad_or_trim_rows(arr: np.ndarray, target_rows: int, fill_value: float = 0.0) -> np.ndarray:
    """
    Pad or trim a 2D array on row dimension to fixed size.
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D.")

    rows, cols = arr.shape
    if rows == target_rows:
        return arr

    if rows > target_rows:
        return arr[:target_rows]

    pad_rows = target_rows - rows
    pad = np.full((pad_rows, cols), fill_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def angles_from_center(points_xy: np.ndarray) -> np.ndarray:
    """
    Return polar angles in [-pi, pi].
    """
    return np.arctan2(points_xy[:, 1], points_xy[:, 0]).astype(np.float32)


def radii_from_center(points_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points_xy, axis=1).astype(np.float32)
