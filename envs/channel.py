from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


ChannelMode = Literal["simplified", "paper_atg"]


@dataclass
class AirToGroundChannelConfig:
    """
    Air-to-ground channel parameters.

    This version focuses on deployment-stage QoS feasibility:
    - simplified mode: horizontal-distance threshold
    - paper_atg mode: probabilistic LoS/NLoS average path loss threshold

    Units:
    - distance: meter
    - frequency: GHz
    - path loss: dB
    """

    mode: ChannelMode = "simplified"

    # -------------------------
    # Simplified mode
    # -------------------------
    simplified_coverage_radius: float = 350.0

    # -------------------------
    # Paper-like ATG mode
    # -------------------------
    # Carrier frequency in GHz
    carrier_freq_ghz: float = 2.0

    # QoS threshold in dB
    qos_threshold_db: float = 110.0

    # Environment parameters commonly used in UAV ATG models
    # Urban-like default values
    los_a: float = 9.61
    los_b: float = 0.16

    # Additional excessive losses (dB)
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0

    # Numerical safety
    eps: float = 1e-8


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
def horizontal_distance(uav_xy: np.ndarray, ue_xy: np.ndarray) -> np.ndarray:
    """
    Compute pairwise horizontal distances.

    Args:
        uav_xy: [M, 2]
        ue_xy:  [K, 2]

    Returns:
        dist_xy: [K, M]
    """
    diff = ue_xy[:, None, :] - uav_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.astype(np.float32)


def slant_distance(dist_xy: np.ndarray, uav_h: np.ndarray) -> np.ndarray:
    """
    Compute slant distance sqrt(d_xy^2 + h^2).

    Args:
        dist_xy: [K, M]
        uav_h:   [M]

    Returns:
        dist_3d: [K, M]
    """
    h = uav_h.reshape(1, -1)
    d3 = np.sqrt(np.maximum(dist_xy ** 2 + h ** 2, 1e-8))
    return d3.astype(np.float32)


def elevation_angle_deg(dist_xy: np.ndarray, uav_h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Elevation angle in degrees:
        theta = arctan(h / d_xy)

    Special case:
        if d_xy == 0, angle -> 90 deg
    """
    h = uav_h.reshape(1, -1)
    theta_rad = np.arctan(h / np.maximum(dist_xy, eps))
    theta_deg = np.degrees(theta_rad)
    return theta_deg.astype(np.float32)


# ----------------------------------------------------------------------
# Simplified coverage model
# ----------------------------------------------------------------------
def simplified_cover_matrix(
    ue_xy: np.ndarray,
    uav_xy: np.ndarray,
    coverage_radius: float,
) -> np.ndarray:
    """
    Simplified QoS feasibility:
        feasible iff horizontal distance <= coverage_radius

    Returns:
        cover_mat: [K, M], bool
    """
    dist_xy = horizontal_distance(uav_xy, ue_xy)
    return (dist_xy <= coverage_radius)


# ----------------------------------------------------------------------
# Paper-like ATG model
# ----------------------------------------------------------------------
def los_probability(
    elevation_deg: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    r"""
    Probabilistic LoS model widely used in UAV ATG links:

        P_LoS = 1 / (1 + a * exp(-b * (theta - a)))

    Args:
        elevation_deg: [K, M]
        a, b: environment constants

    Returns:
        p_los: [K, M]
    """
    p = 1.0 / (1.0 + a * np.exp(-b * (elevation_deg - a)))
    return p.astype(np.float32)


def free_space_path_loss_db(
    dist_3d_m: np.ndarray,
    carrier_freq_ghz: float,
) -> np.ndarray:
    r"""
    Free-space path loss in dB with:
        d in km, f in GHz

        FSPL = 32.44 + 20log10(d_km) + 20log10(f_MHz)
    Since f is in GHz, f_MHz = 1000 * f_GHz:
        FSPL = 32.44 + 20log10(d_km) + 20log10(1000 f_GHz)

    Equivalent numeric implementation below.

    Args:
        dist_3d_m: [K, M]
        carrier_freq_ghz: scalar

    Returns:
        fspl_db: [K, M]
    """
    d_km = np.maximum(dist_3d_m / 1000.0, 1e-12)
    f_mhz = carrier_freq_ghz * 1000.0
    fspl = 32.44 + 20.0 * np.log10(d_km) + 20.0 * np.log10(f_mhz)
    return fspl.astype(np.float32)


def channel_gain_from_path_loss_db(path_loss_db: np.ndarray) -> np.ndarray:
    """Convert path loss in dB into a linear power gain."""
    path_loss = np.asarray(path_loss_db, dtype=np.float32)
    return np.power(10.0, -path_loss / 10.0).astype(np.float32)


def shannon_rate_mbps(bandwidth_mhz: float, sinr: np.ndarray) -> np.ndarray:
    """Return Shannon rate in Mbit/s for a bandwidth expressed in MHz."""
    if bandwidth_mhz <= 0.0:
        raise ValueError("bandwidth_mhz must be positive.")
    values = np.maximum(np.asarray(sinr, dtype=np.float32), 0.0)
    return (float(bandwidth_mhz) * np.log2(1.0 + values)).astype(np.float32)


def average_atg_path_loss_db(
    ue_xy: np.ndarray,
    uav_xyz: np.ndarray,
    cfg: AirToGroundChannelConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Average ATG path loss based on probabilistic LoS/NLoS model.

    Model:
        PL_avg = FSPL(d_3d, f) + P_LoS * eta_LoS + (1 - P_LoS) * eta_NLoS

    Args:
        ue_xy:   [K, 2]
        uav_xyz: [M, 3]
        cfg: channel config

    Returns:
        pl_avg_db: [K, M]
        p_los:     [K, M]
        dist_xy:   [K, M]
        dist_3d:   [K, M]
    """
    uav_xy = uav_xyz[:, :2]
    uav_h = uav_xyz[:, 2]

    dist_xy = horizontal_distance(uav_xy, ue_xy)                  # [K, M]
    dist_3d = slant_distance(dist_xy, uav_h)                      # [K, M]
    elev_deg = elevation_angle_deg(dist_xy, uav_h, eps=cfg.eps)   # [K, M]

    p_los = los_probability(
        elevation_deg=elev_deg,
        a=cfg.los_a,
        b=cfg.los_b,
    )

    fspl_db = free_space_path_loss_db(
        dist_3d_m=dist_3d,
        carrier_freq_ghz=cfg.carrier_freq_ghz,
    )

    pl_avg_db = fspl_db + p_los * cfg.eta_los_db + (1.0 - p_los) * cfg.eta_nlos_db
    return pl_avg_db.astype(np.float32), p_los.astype(np.float32), dist_xy.astype(np.float32), dist_3d.astype(np.float32)


def paper_atg_cover_matrix(
    ue_xy: np.ndarray,
    uav_xyz: np.ndarray,
    cfg: AirToGroundChannelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper-like QoS feasibility:
        feasible iff average ATG path loss <= qos_threshold_db

    Returns:
        cover_mat: [K, M], bool
        pl_avg_db: [K, M]
    """
    pl_avg_db, _, _, _ = average_atg_path_loss_db(ue_xy, uav_xyz, cfg)
    cover_mat = (pl_avg_db <= cfg.qos_threshold_db)
    return cover_mat, pl_avg_db


# ----------------------------------------------------------------------
# Unified interface
# ----------------------------------------------------------------------
def build_cover_matrix(
    ue_xy: np.ndarray,
    uav_xyz: np.ndarray,
    cfg: AirToGroundChannelConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Unified QoS feasibility interface.

    Args:
        ue_xy:   [K, 2]
        uav_xyz: [M, 3]
        cfg: channel config

    Returns:
        cover_mat: [K, M], bool
        aux_metric:
            - simplified mode: horizontal distance matrix [K, M]
            - paper_atg mode: average path loss matrix [K, M]
    """
    if cfg.mode == "simplified":
        uav_xy = uav_xyz[:, :2]
        dist_xy = horizontal_distance(uav_xy, ue_xy)
        cover_mat = (dist_xy <= cfg.simplified_coverage_radius)
        return cover_mat, dist_xy.astype(np.float32)

    if cfg.mode == "paper_atg":
        cover_mat, pl_avg_db = paper_atg_cover_matrix(
            ue_xy=ue_xy,
            uav_xyz=uav_xyz,
            cfg=cfg,
        )
        return cover_mat, pl_avg_db.astype(np.float32)

    raise ValueError(f"Unsupported channel mode: {cfg.mode}")


# ----------------------------------------------------------------------
# Optional helpers for later analysis / debugging
# ----------------------------------------------------------------------
def nearest_feasible_uav(
    cover_mat: np.ndarray,
    aux_metric: np.ndarray,
    prefer: Literal["nearest", "lowest_pathloss"] = "nearest",
) -> np.ndarray:
    """
    Assign each UE to one feasible UAV.

    Args:
        cover_mat:  [K, M], bool
        aux_metric: [K, M]
            - if prefer == "nearest": smaller is better (distance)
            - if prefer == "lowest_pathloss": smaller is better (path loss)

    Returns:
        assigned_uav_idx: [K], -1 if uncovered
    """
    k_num, m_num = cover_mat.shape
    assigned = np.full((k_num,), -1, dtype=np.int32)

    for k in range(k_num):
        feasible = np.where(cover_mat[k])[0]
        if feasible.size == 0:
            continue

        best_local = np.argmin(aux_metric[k, feasible])
        assigned[k] = int(feasible[best_local])

    return assigned


def count_overlap_users(cover_mat: np.ndarray) -> int:
    """
    Number of users covered by more than one UAV.
    """
    cover_count = np.sum(cover_mat, axis=1)
    return int(np.sum(cover_count > 1))


def compute_coverage_ratio(cover_mat: np.ndarray) -> float:
    """
    Ratio of users covered by at least one UAV.
    """
    covered = np.sum(np.sum(cover_mat, axis=1) > 0)
    total = cover_mat.shape[0]
    return float(covered / max(total, 1))


# ----------------------------------------------------------------------
# Minimal self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ue_xy = np.array(
        [
            [700.0, 200.0],
            [900.0, -100.0],
            [1300.0, 500.0],
        ],
        dtype=np.float32,
    )

    uav_xyz = np.array(
        [
            [500.0, 0.0, 100.0],
            [1000.0, 300.0, 120.0],
        ],
        dtype=np.float32,
    )

    # Simplified
    cfg1 = AirToGroundChannelConfig(
        mode="simplified",
        simplified_coverage_radius=350.0,
    )
    cover1, metric1 = build_cover_matrix(ue_xy, uav_xyz, cfg1)
    print("=== simplified ===")
    print("cover:\n", cover1.astype(np.int32))
    print("dist_xy:\n", metric1)

    # Paper-like ATG
    cfg2 = AirToGroundChannelConfig(
        mode="paper_atg",
        carrier_freq_ghz=2.0,
        qos_threshold_db=110.0,
        los_a=9.61,
        los_b=0.16,
        eta_los_db=1.0,
        eta_nlos_db=20.0,
    )
    cover2, metric2 = build_cover_matrix(ue_xy, uav_xyz, cfg2)
    print("=== paper_atg ===")
    print("cover:\n", cover2.astype(np.int32))
    print("avg_path_loss_db:\n", metric2)
