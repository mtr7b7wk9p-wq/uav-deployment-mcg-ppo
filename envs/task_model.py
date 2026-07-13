"""Truth and per-UAV belief models for resource cognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TaskTruth:
    """Environment-only task state that must not enter local observations."""

    position_xy: tuple[float, float]
    band_id: int
    true_state: float
    priority: float


@dataclass(frozen=True)
class LocalTaskBelief:
    """One UAV's current belief about one task."""

    estimate: float
    uncertainty: float
    aoi: float
    confidence: float
    last_update_step: int


class TaskTruthBatch:
    """Vectorized hidden task truth owned by the environment."""

    def __init__(
        self,
        positions_xy: np.ndarray,
        band_ids: np.ndarray,
        true_states: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        positions = np.asarray(positions_xy, dtype=np.float32)
        bands = np.asarray(band_ids, dtype=np.int32)
        truth = np.asarray(true_states, dtype=np.float32)
        priority = np.asarray(priorities, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions_xy must have shape [N, 2].")
        num_tasks = positions.shape[0]
        if any(array.shape != (num_tasks,) for array in (bands, truth, priority)):
            raise ValueError("Task truth arrays must have the same length.")
        if num_tasks <= 0:
            raise ValueError("At least one task is required.")

        self.positions_xy = positions.copy()
        self.band_ids = bands.copy()
        self.true_states = np.clip(truth, 0.0, 1.0)
        self.priorities = np.maximum(priority, 0.0)

    def __len__(self) -> int:
        return int(self.positions_xy.shape[0])

    def snapshot(self, task_id: int) -> TaskTruth:
        task_id = self._validate_task_id(task_id)
        return TaskTruth(
            position_xy=(
                float(self.positions_xy[task_id, 0]),
                float(self.positions_xy[task_id, 1]),
            ),
            band_id=int(self.band_ids[task_id]),
            true_state=float(self.true_states[task_id]),
            priority=float(self.priorities[task_id]),
        )

    def _validate_task_id(self, task_id: int) -> int:
        task_id = int(task_id)
        if not 0 <= task_id < len(self):
            raise IndexError("task_id out of range.")
        return task_id


class LocalBeliefBatch:
    """Independent cognitive state for every UAV-task pair."""

    def __init__(
        self,
        num_agents: int,
        task_priorities: np.ndarray,
        *,
        initial_uncertainty: float = 1.0,
        initial_aoi: float = 0.0,
        max_aoi: float = 40.0,
    ) -> None:
        priorities = np.asarray(task_priorities, dtype=np.float32)
        if num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        if priorities.ndim != 1 or priorities.size == 0:
            raise ValueError("task_priorities must be a non-empty vector.")
        if max_aoi <= 0.0:
            raise ValueError("max_aoi must be positive.")

        self.num_agents = int(num_agents)
        self.num_tasks = int(priorities.size)
        self.task_priorities = np.maximum(priorities, 0.0)
        self.max_aoi = float(max_aoi)
        self.initial_uncertainty = float(np.clip(initial_uncertainty, 0.0, 1.0))
        self.initial_aoi = float(np.clip(initial_aoi, 0.0, self.max_aoi))

        shape = (self.num_agents, self.num_tasks)
        self.estimates = np.full(shape, 0.5, dtype=np.float32)
        self.uncertainties = np.full(shape, self.initial_uncertainty, dtype=np.float32)
        self.aoi = np.full(shape, self.initial_aoi, dtype=np.float32)
        self.confidence = np.full(shape, 1.0 - self.initial_uncertainty, dtype=np.float32)
        self.last_update_step = np.full(shape, -1, dtype=np.int32)

    def reset(self) -> None:
        self.estimates[:] = 0.5
        self.uncertainties[:] = self.initial_uncertainty
        self.aoi[:] = self.initial_aoi
        self.confidence[:] = 1.0 - self.initial_uncertainty
        self.last_update_step[:] = -1

    def age(self, steps: float = 1.0) -> None:
        """Age every UAV's local information independently."""
        if steps < 0.0:
            raise ValueError("steps must be non-negative.")
        self.aoi[:] = np.minimum(self.aoi + float(steps), self.max_aoi)

    def apply_local_sensing(
        self,
        agent_ids: Iterable[int],
        task_ids: Iterable[int],
        observations: np.ndarray,
        *,
        uncertainty_reduction: float,
        current_step: int,
    ) -> dict[str, float]:
        """Update only the selected UAV-task belief entries."""
        agents = np.asarray(list(agent_ids), dtype=np.int64)
        tasks = np.asarray(list(task_ids), dtype=np.int64)
        values = np.asarray(observations, dtype=np.float32)
        if agents.size == 0:
            return {"uncertainty_gain": 0.0, "aoi_gain": 0.0}
        if agents.shape != tasks.shape or agents.shape != values.shape:
            raise ValueError("agent_ids, task_ids, and observations must have matching shapes.")
        if np.any(agents < 0) or np.any(agents >= self.num_agents):
            raise IndexError("agent_id out of range.")
        if np.any(tasks < 0) or np.any(tasks >= self.num_tasks):
            raise IndexError("task_id out of range.")

        before_uncertainty = self.uncertainties[agents, tasks].copy()
        before_aoi = self.aoi[agents, tasks].copy()
        reduction = float(np.clip(uncertainty_reduction, 0.0, 1.0))

        self.estimates[agents, tasks] = np.clip(values, 0.0, 1.0)
        self.uncertainties[agents, tasks] = before_uncertainty * (1.0 - reduction)
        self.aoi[agents, tasks] = 0.0
        self.confidence[agents, tasks] = 1.0 - self.uncertainties[agents, tasks]
        self.last_update_step[agents, tasks] = int(current_step)

        weights = self.task_priorities[tasks]
        weight_sum = max(float(np.sum(weights)), 1e-6)
        uncertainty_gain = float(
            np.sum(weights * (before_uncertainty - self.uncertainties[agents, tasks]))
            / weight_sum
        )
        aoi_gain = float(
            np.sum(weights * before_aoi) / (weight_sum * self.max_aoi)
        )
        return {"uncertainty_gain": uncertainty_gain, "aoi_gain": aoi_gain}

    def local_quality(self, agent_id: int) -> float:
        agent_id = self._validate_agent_id(agent_id)
        weights_sum = max(float(np.sum(self.task_priorities)), 1e-6)
        normalized_aoi = self.aoi[agent_id] / self.max_aoi
        quality = 1.0 - 0.5 * (self.uncertainties[agent_id] + normalized_aoi)
        return float(
            np.clip(np.sum(self.task_priorities * quality) / weights_sum, 0.0, 1.0)
        )

    def fuse_neighbor_message(
        self,
        receiver_id: int,
        task_id: int,
        *,
        estimate: float,
        uncertainty: float,
        confidence: float,
        message_aoi: float,
        source_update_step: int,
        current_step: int,
        confidence_threshold: float,
        freshness_decay: float,
    ) -> dict[str, float]:
        """Fuse one delivered message into only the addressed local belief."""
        receiver_id = self._validate_agent_id(receiver_id)
        task_id = self._validate_task_id(task_id)
        transit_age = max(int(current_step) - int(source_update_step), 0)
        effective_aoi = min(float(message_aoi) + float(transit_age), self.max_aoi)
        effective_confidence = float(
            np.clip(
                float(confidence) * np.exp(-float(freshness_decay) * effective_aoi),
                0.0,
                1.0,
            )
        )
        effective_uncertainty = max(
            float(np.clip(uncertainty, 0.0, 1.0)),
            1.0 - effective_confidence,
        )
        if effective_confidence < float(confidence_threshold):
            return {"accepted": 0.0, "quality_gain": 0.0}

        local_uncertainty = float(self.uncertainties[receiver_id, task_id])
        local_aoi = float(self.aoi[receiver_id, task_id])
        improves_uncertainty = effective_uncertainty < local_uncertainty - 1e-6
        improves_freshness = effective_aoi < local_aoi - 1e-6
        if not improves_uncertainty and not improves_freshness:
            return {"accepted": 0.0, "quality_gain": 0.0}

        quality_before = self.local_quality(receiver_id)
        local_confidence = float(self.confidence[receiver_id, task_id])
        weight_sum = max(local_confidence + effective_confidence, 1e-6)
        fused_estimate = (
            local_confidence * float(self.estimates[receiver_id, task_id])
            + effective_confidence * float(np.clip(estimate, 0.0, 1.0))
        ) / weight_sum

        self.estimates[receiver_id, task_id] = float(np.clip(fused_estimate, 0.0, 1.0))
        self.uncertainties[receiver_id, task_id] = min(
            local_uncertainty, effective_uncertainty
        )
        self.aoi[receiver_id, task_id] = min(local_aoi, effective_aoi)
        self.confidence[receiver_id, task_id] = max(
            local_confidence, effective_confidence
        )
        self.last_update_step[receiver_id, task_id] = max(
            int(self.last_update_step[receiver_id, task_id]), int(source_update_step)
        )
        quality_gain = max(self.local_quality(receiver_id) - quality_before, 0.0)
        return {"accepted": 1.0, "quality_gain": float(quality_gain)}

    def mean_quality(self) -> float:
        return float(np.mean([self.local_quality(i) for i in range(self.num_agents)]))

    def mean_uncertainty(self) -> float:
        weights = np.broadcast_to(self.task_priorities, self.uncertainties.shape)
        return float(np.sum(weights * self.uncertainties) / max(float(np.sum(weights)), 1e-6))

    def mean_aoi(self) -> float:
        weights = np.broadcast_to(self.task_priorities, self.aoi.shape)
        return float(np.sum(weights * self.aoi) / max(float(np.sum(weights)), 1e-6))

    def mean_estimation_error(self, true_states: np.ndarray) -> float:
        truth = np.asarray(true_states, dtype=np.float32)
        if truth.shape != (self.num_tasks,):
            raise ValueError("true_states must have shape [num_tasks].")
        errors = np.abs(self.estimates - truth[None, :])
        weights = np.broadcast_to(self.task_priorities, errors.shape)
        return float(np.sum(weights * errors) / max(float(np.sum(weights)), 1e-6))

    def snapshot(self, agent_id: int, task_id: int) -> LocalTaskBelief:
        agent_id = self._validate_agent_id(agent_id)
        task_id = self._validate_task_id(task_id)
        return LocalTaskBelief(
            estimate=float(self.estimates[agent_id, task_id]),
            uncertainty=float(self.uncertainties[agent_id, task_id]),
            aoi=float(self.aoi[agent_id, task_id]),
            confidence=float(self.confidence[agent_id, task_id]),
            last_update_step=int(self.last_update_step[agent_id, task_id]),
        )

    def _validate_agent_id(self, agent_id: int) -> int:
        agent_id = int(agent_id)
        if not 0 <= agent_id < self.num_agents:
            raise IndexError("agent_id out of range.")
        return agent_id

    def _validate_task_id(self, task_id: int) -> int:
        task_id = int(task_id)
        if not 0 <= task_id < self.num_tasks:
            raise IndexError("task_id out of range.")
        return task_id
