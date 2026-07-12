from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from configs.scenario_config import ScenarioConfig
from envs.channel import AirToGroundChannelConfig, build_cover_matrix, nearest_feasible_uav


@dataclass
class KMeansDeployResult:
    success: bool
    num_uavs_used: int
    uav_positions: np.ndarray
    cluster_labels: np.ndarray
    covered_mask: np.ndarray
    assigned_uav_idx: np.ndarray
    coverage_ratio: float
    iterations: int
    mean_intra_cluster_distance: float
    info: Dict[str, Any]


class ConstrainedKMeansBaseline:
    def __init__(self, scenario_cfg: ScenarioConfig):
        self.cfg = scenario_cfg
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

    def solve(self, ue_positions: np.ndarray, m_min: int = 1, m_max: Optional[int] = None,
              n_init: int = 5, max_iter: int = 50, tol: float = 1e-3,
              random_seed: Optional[int] = None) -> KMeansDeployResult:
        rng = np.random.default_rng(self.cfg.seed if random_seed is None else random_seed)
        num_users = ue_positions.shape[0]
        m_max = min(self.cfg.max_candidate_uavs if m_max is None else m_max, num_users)

        best_partial = None
        best_feasible = None
        for m in range(m_max, m_min - 1, -1):
            res = self._solve_fixed_m(ue_positions, m, n_init, max_iter, tol, rng)
            if best_partial is None or res.coverage_ratio > best_partial.coverage_ratio:
                best_partial = res
            if res.success:
                best_feasible = res
        return best_feasible if best_feasible is not None else best_partial

    def _solve_fixed_m(self, ue_positions: np.ndarray, m: int, n_init: int, max_iter: int,
                       tol: float, rng: np.random.Generator) -> KMeansDeployResult:
        best_result = None
        best_obj = float("inf")
        for _ in range(n_init):
            centers = self._init_centers(ue_positions, m, rng)
            centers, labels, iters = self._run_kmeans(ue_positions, centers, max_iter, tol)
            uav_pos = np.concatenate([centers, np.full((m, 1), self.cfg.uav_init_height, dtype=np.float32)], axis=1)
            cover_mat, aux = build_cover_matrix(ue_positions, uav_pos, self.channel_cfg)
            cover_count = np.sum(cover_mat, axis=1).astype(np.int32)
            covered_mask = cover_count > 0
            coverage_ratio = float(np.mean(covered_mask.astype(np.float32)))
            prefer = "nearest" if self.channel_cfg.mode == "simplified" else "lowest_pathloss"
            assigned = nearest_feasible_uav(cover_mat, aux, prefer=prefer)
            mean_intra = self._mean_intra_cluster_distance(ue_positions, centers, labels)
            obj = mean_intra - 10000.0 * coverage_ratio + 10.0 * m
            success = bool(np.all(covered_mask))
            result = KMeansDeployResult(
                success=success,
                num_uavs_used=m,
                uav_positions=uav_pos.astype(np.float32),
                cluster_labels=labels.astype(np.int32),
                covered_mask=covered_mask.astype(bool),
                assigned_uav_idx=assigned.astype(np.int32),
                coverage_ratio=coverage_ratio,
                iterations=iters,
                mean_intra_cluster_distance=float(mean_intra),
                info={"cover_mat": cover_mat, "aux_metric": aux, "channel_mode": self.channel_cfg.mode},
            )
            if best_result is None or obj < best_obj:
                best_result, best_obj = result, obj
        return best_result

    def _init_centers(self, points: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        first_idx = int(rng.integers(0, points.shape[0]))
        centers = [points[first_idx]]
        for _ in range(1, m):
            d2 = self._min_sq_dist_to_centers(points, np.array(centers, dtype=np.float32))
            probs = d2 / np.maximum(np.sum(d2), 1e-8)
            idx = int(rng.choice(points.shape[0], p=probs))
            centers.append(points[idx])
        return np.array(centers, dtype=np.float32)

    def _run_kmeans(self, points: np.ndarray, centers: np.ndarray, max_iter: int, tol: float):
        labels = np.zeros((points.shape[0],), dtype=np.int32)
        for it in range(1, max_iter + 1):
            d = self._pairwise_dist(points, centers)
            labels = np.argmin(d, axis=1).astype(np.int32)
            new_centers = centers.copy()
            for j in range(centers.shape[0]):
                members = points[labels == j]
                if members.shape[0] > 0:
                    new_centers[j] = np.mean(members, axis=0)
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= tol:
                return centers.astype(np.float32), labels.astype(np.int32), it
        return centers.astype(np.float32), labels.astype(np.int32), max_iter

    def _pairwise_dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).astype(np.float32)

    def _min_sq_dist_to_centers(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        d = self._pairwise_dist(points, centers)
        return np.min(d, axis=1).astype(np.float32) ** 2

    def _mean_intra_cluster_distance(self, points: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean([np.linalg.norm(points[i] - centers[labels[i]]) for i in range(points.shape[0])]))
