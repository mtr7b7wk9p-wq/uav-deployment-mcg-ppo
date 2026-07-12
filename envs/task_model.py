"""State model for local, time-sensitive resource cognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class TaskState:
    """Static task metadata and the current cognitive state."""

    position_xy: tuple[float, float]
    band_id: int
    true_state: float
    estimate: float
    uncertainty: float
    aoi: float
    priority: float
    confidence: float


class TaskStateBatch:
    """Vectorized task state with bounded aging and sensing updates."""

    def __init__(
        self,
        positions_xy: np.ndarray,
        band_ids: np.ndarray,
        true_states: np.ndarray,
        priorities: np.ndarray,
        *,
        initial_uncertainty: float = 1.0,
        initial_aoi: float = 0.0,
        max_aoi: float = 40.0,
    ) -> None:
        positions = np.asarray(positions_xy, dtype=np.float32)
        bands = np.asarray(band_ids, dtype=np.int32)
        truth = np.asarray(true_states, dtype=np.float32)
        priority = np.asarray(priorities, dtype=np.float32)
        n = positions.shape[0]
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions_xy must have shape [N, 2].")
        if any(array.shape != (n,) for array in (bands, truth, priority)):
            raise ValueError("Task arrays must have the same length.")
        if max_aoi <= 0.0:
            raise ValueError("max_aoi must be positive.")

        self.positions_xy = positions.copy()
        self.band_ids = bands.copy()
        self.true_states = np.clip(truth, 0.0, 1.0)
        self.priorities = np.maximum(priority, 0.0)
        self.max_aoi = float(max_aoi)
        self.initial_uncertainty = float(np.clip(initial_uncertainty, 0.0, 1.0))
        self.initial_aoi = float(np.clip(initial_aoi, 0.0, self.max_aoi))
        self.estimate = np.zeros((n,), dtype=np.float32)
        self.uncertainty = np.zeros((n,), dtype=np.float32)
        self.aoi = np.zeros((n,), dtype=np.float32)
        self.confidence = np.zeros((n,), dtype=np.float32)
        self.reset()

    def __len__(self) -> int:
        return int(self.positions_xy.shape[0])

    def reset(self) -> None:
        self.estimate[:] = 0.5
        self.uncertainty[:] = self.initial_uncertainty
        self.aoi[:] = self.initial_aoi
        self.confidence[:] = 1.0 - self.uncertainty

    def age(self, steps: float = 1.0) -> None:
        """Increase information age without changing the hidden truth."""
        if steps < 0.0:
            raise ValueError("steps must be non-negative.")
        self.aoi[:] = np.minimum(self.aoi + float(steps), self.max_aoi)

    def sense(
        self,
        task_indices: Iterable[int],
        *,
        uncertainty_reduction: float,
        noisy_observations: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Update selected tasks and return weighted cognitive gains."""
        indices = np.asarray(list(task_indices), dtype=np.int64)
        if indices.size == 0:
            return {"uncertainty_gain": 0.0, "aoi_gain": 0.0}
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("task index out of range.")
        reduction = float(np.clip(uncertainty_reduction, 0.0, 1.0))
        unique_indices = np.unique(indices)
        before_uncertainty = self.uncertainty[unique_indices].copy()
        before_aoi = self.aoi[unique_indices].copy()

        if noisy_observations is None:
            observations = self.true_states[unique_indices]
        else:
            observations = np.asarray(noisy_observations, dtype=np.float32)
            if observations.shape != unique_indices.shape:
                raise ValueError("noisy_observations must match unique task indices.")
        self.estimate[unique_indices] = np.clip(observations, 0.0, 1.0)
        self.uncertainty[unique_indices] = np.maximum(
            before_uncertainty * (1.0 - reduction), 0.0
        )
        self.aoi[unique_indices] = 0.0
        self.confidence[unique_indices] = 1.0 - self.uncertainty[unique_indices]

        weight_sum = max(float(np.sum(self.priorities[unique_indices])), 1e-6)
        uncertainty_gain = float(
            np.sum(self.priorities[unique_indices] * (before_uncertainty - self.uncertainty[unique_indices]))
            / weight_sum
        )
        aoi_gain = float(
            np.sum(self.priorities[unique_indices] * before_aoi)
            / (weight_sum * self.max_aoi)
        )
        return {"uncertainty_gain": uncertainty_gain, "aoi_gain": aoi_gain}

    def cognitive_quality(self) -> float:
        """Return a priority-weighted quality score in [0, 1]."""
        weight_sum = max(float(np.sum(self.priorities)), 1e-6)
        normalized_aoi = self.aoi / self.max_aoi
        quality = 1.0 - 0.5 * (self.uncertainty + normalized_aoi)
        return float(np.clip(np.sum(self.priorities * quality) / weight_sum, 0.0, 1.0))

    def snapshot(self, index: int) -> TaskState:
        if not 0 <= int(index) < len(self):
            raise IndexError("task index out of range.")
        i = int(index)
        return TaskState(
            position_xy=(float(self.positions_xy[i, 0]), float(self.positions_xy[i, 1])),
            band_id=int(self.band_ids[i]),
            true_state=float(self.true_states[i]),
            estimate=float(self.estimate[i]),
            uncertainty=float(self.uncertainty[i]),
            aoi=float(self.aoi[i]),
            priority=float(self.priorities[i]),
            confidence=float(self.confidence[i]),
        )
