"""Hidden resource truth and independent per-UAV cognition beliefs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TaskTruth:
    """Environment-only task state that must not enter local observations."""

    position_xy: tuple[float, float]
    band_id: int
    true_state: float
    demand_level: float
    priority: float
    arrival_rate: float
    queue_length: float


@dataclass(frozen=True)
class LocalTaskBelief:
    """One UAV's spectrum and demand belief about one task."""

    estimate: float
    uncertainty: float
    aoi: float
    confidence: float
    demand_estimate: float
    demand_uncertainty: float
    demand_aoi: float
    demand_confidence: float
    queue_estimate: float
    queue_uncertainty: float
    queue_aoi: float
    queue_confidence: float
    arrival_estimate: float
    last_update_step: int


class TaskTruthBatch:
    """Vectorized hidden task truth owned by the environment."""

    def __init__(
        self,
        positions_xy: np.ndarray,
        band_ids: np.ndarray,
        true_states: np.ndarray,
        priorities: np.ndarray,
        demand_levels: np.ndarray | None = None,
        arrival_rates: np.ndarray | None = None,
        queue_lengths: np.ndarray | None = None,
        queue_capacity: float = 20.0,
    ) -> None:
        positions = np.asarray(positions_xy, dtype=np.float32)
        bands = np.asarray(band_ids, dtype=np.int32)
        truth = np.asarray(true_states, dtype=np.float32)
        priority = np.asarray(priorities, dtype=np.float32)
        demand = truth.copy() if demand_levels is None else np.asarray(demand_levels, dtype=np.float32)
        arrivals = (
            np.zeros_like(truth, dtype=np.float32)
            if arrival_rates is None
            else np.asarray(arrival_rates, dtype=np.float32)
        )
        queues = (
            np.zeros_like(truth, dtype=np.float32)
            if queue_lengths is None
            else np.asarray(queue_lengths, dtype=np.float32)
        )
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions_xy must have shape [N, 2].")
        num_tasks = positions.shape[0]
        if any(
            array.shape != (num_tasks,)
            for array in (bands, truth, demand, priority, arrivals, queues)
        ):
            raise ValueError("Task truth arrays must have the same length.")
        if num_tasks <= 0:
            raise ValueError("At least one task is required.")
        if queue_capacity <= 0.0:
            raise ValueError("queue_capacity must be positive.")

        self.positions_xy = positions.copy()
        self.band_ids = bands.copy()
        self.true_states = np.clip(truth, 0.0, 1.0)
        self.demand_levels = np.clip(demand, 0.0, 1.0)
        self.priorities = np.maximum(priority, 0.0)
        self.arrival_rates = np.maximum(arrivals, 0.0)
        self.queue_capacity = float(queue_capacity)
        self.queue_lengths = np.clip(queues, 0.0, self.queue_capacity)

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
            demand_level=float(self.demand_levels[task_id]),
            priority=float(self.priorities[task_id]),
            arrival_rate=float(self.arrival_rates[task_id]),
            queue_length=float(self.queue_lengths[task_id]),
        )

    def advance_business(self, arrivals: np.ndarray) -> dict[str, float]:
        values = np.asarray(arrivals, dtype=np.float32)
        if values.shape != self.queue_lengths.shape:
            raise ValueError("arrivals must match queue_lengths shape.")
        values = np.maximum(values, 0.0)
        before = self.queue_lengths.copy()
        raw = before + values
        self.queue_lengths[:] = np.minimum(raw, self.queue_capacity)
        accepted = self.queue_lengths - before
        return {
            "total_arrivals": float(np.sum(accepted)),
            "queue_overflow": float(np.sum(np.maximum(raw - self.queue_capacity, 0.0))),
        }

    def apply_service(self, served: np.ndarray) -> float:
        values = np.asarray(served, dtype=np.float32)
        if values.shape != self.queue_lengths.shape:
            raise ValueError("served must match queue_lengths shape.")
        actual = np.minimum(np.maximum(values, 0.0), self.queue_lengths)
        self.queue_lengths[:] -= actual
        return float(np.sum(actual))

    def _validate_task_id(self, task_id: int) -> int:
        task_id = int(task_id)
        if not 0 <= task_id < len(self):
            raise IndexError("task_id out of range.")
        return task_id


class LocalBeliefBatch:
    """Independent spectrum and demand cognition for every UAV-task pair."""

    def __init__(
        self,
        num_agents: int,
        task_priorities: np.ndarray,
        *,
        initial_uncertainty: float = 1.0,
        initial_aoi: float = 0.0,
        max_aoi: float = 40.0,
        spectrum_quality_weight: float = 0.6,
        demand_quality_weight: float = 0.4,
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
        weight_sum = max(float(spectrum_quality_weight + demand_quality_weight), 1e-6)
        self.spectrum_quality_weight = float(max(spectrum_quality_weight, 0.0) / weight_sum)
        self.demand_quality_weight = float(max(demand_quality_weight, 0.0) / weight_sum)

        shape = (self.num_agents, self.num_tasks)
        self.estimates = np.full(shape, 0.5, dtype=np.float32)
        self.uncertainties = np.full(shape, self.initial_uncertainty, dtype=np.float32)
        self.aoi = np.full(shape, self.initial_aoi, dtype=np.float32)
        self.confidence = np.full(shape, 1.0 - self.initial_uncertainty, dtype=np.float32)
        self.demand_estimates = np.full(shape, 0.5, dtype=np.float32)
        self.demand_uncertainties = np.full(shape, self.initial_uncertainty, dtype=np.float32)
        self.demand_aoi = np.full(shape, self.initial_aoi, dtype=np.float32)
        self.demand_confidence = np.full(shape, 1.0 - self.initial_uncertainty, dtype=np.float32)
        self.queue_estimates = np.zeros(shape, dtype=np.float32)
        self.queue_uncertainties = np.full(shape, 1.0, dtype=np.float32)
        self.queue_aoi = np.full(shape, self.initial_aoi, dtype=np.float32)
        self.queue_confidence = np.zeros(shape, dtype=np.float32)
        self.arrival_estimates = np.zeros(shape, dtype=np.float32)
        self.last_update_step = np.full(shape, -1, dtype=np.int32)

    def reset(self) -> None:
        self.estimates[:] = 0.5
        self.uncertainties[:] = self.initial_uncertainty
        self.aoi[:] = self.initial_aoi
        self.confidence[:] = 1.0 - self.initial_uncertainty
        self.demand_estimates[:] = 0.5
        self.demand_uncertainties[:] = self.initial_uncertainty
        self.demand_aoi[:] = self.initial_aoi
        self.demand_confidence[:] = 1.0 - self.initial_uncertainty
        self.queue_estimates[:] = 0.0
        self.queue_uncertainties[:] = 1.0
        self.queue_aoi[:] = self.initial_aoi
        self.queue_confidence[:] = 0.0
        self.arrival_estimates[:] = 0.0
        self.last_update_step[:] = -1

    def age(self, steps: float = 1.0) -> None:
        if steps < 0.0:
            raise ValueError("steps must be non-negative.")
        increment = float(steps)
        self.aoi[:] = np.minimum(self.aoi + increment, self.max_aoi)
        self.demand_aoi[:] = np.minimum(self.demand_aoi + increment, self.max_aoi)
        self.queue_aoi[:] = np.minimum(self.queue_aoi + increment, self.max_aoi)

    def apply_local_sensing(
        self,
        agent_ids: Iterable[int],
        task_ids: Iterable[int],
        observations: np.ndarray,
        *,
        demand_observations: np.ndarray | None = None,
        queue_observations: np.ndarray | None = None,
        arrival_observations: np.ndarray | None = None,
        uncertainty_reduction: float,
        demand_uncertainty_reduction: float | None = None,
        current_step: int,
    ) -> dict[str, Any]:
        """Update only selected UAV-task spectrum and demand beliefs."""
        agents = np.asarray(list(agent_ids), dtype=np.int64)
        tasks = np.asarray(list(task_ids), dtype=np.int64)
        spectrum_values = np.asarray(observations, dtype=np.float32)
        if agents.size == 0:
            return {
                "uncertainty_gain": 0.0,
                "aoi_gain": 0.0,
                "demand_uncertainty_gain": 0.0,
                "demand_aoi_gain": 0.0,
                "queue_uncertainty_gain": 0.0,
                "queue_aoi_gain": 0.0,
                "information_gain": np.zeros((0,), dtype=np.float32),
            }
        if agents.shape != tasks.shape or agents.shape != spectrum_values.shape:
            raise ValueError("agent_ids, task_ids, and observations must have matching shapes.")
        if demand_observations is None:
            demand_values = self.demand_estimates[agents, tasks].copy()
        else:
            demand_values = np.asarray(demand_observations, dtype=np.float32)
            if demand_values.shape != agents.shape:
                raise ValueError("demand_observations must match agent_ids shape.")
        if queue_observations is None:
            queue_values = self.queue_estimates[agents, tasks].copy()
        else:
            queue_values = np.asarray(queue_observations, dtype=np.float32)
            if queue_values.shape != agents.shape:
                raise ValueError("queue_observations must match agent_ids shape.")
        if arrival_observations is None:
            arrival_values = self.arrival_estimates[agents, tasks].copy()
        else:
            arrival_values = np.asarray(arrival_observations, dtype=np.float32)
            if arrival_values.shape != agents.shape:
                raise ValueError("arrival_observations must match agent_ids shape.")
        if np.any(agents < 0) or np.any(agents >= self.num_agents):
            raise IndexError("agent_id out of range.")
        if np.any(tasks < 0) or np.any(tasks >= self.num_tasks):
            raise IndexError("task_id out of range.")

        before_uncertainty = self.uncertainties[agents, tasks].copy()
        before_aoi = self.aoi[agents, tasks].copy()
        before_demand_uncertainty = self.demand_uncertainties[agents, tasks].copy()
        before_demand_aoi = self.demand_aoi[agents, tasks].copy()
        before_queue_uncertainty = self.queue_uncertainties[agents, tasks].copy()
        before_queue_aoi = self.queue_aoi[agents, tasks].copy()
        before_quality = self._task_quality(agents, tasks)
        reduction = float(np.clip(uncertainty_reduction, 0.0, 1.0))
        demand_reduction = float(
            np.clip(
                uncertainty_reduction if demand_uncertainty_reduction is None else demand_uncertainty_reduction,
                0.0,
                1.0,
            )
        )

        self.estimates[agents, tasks] = np.clip(spectrum_values, 0.0, 1.0)
        self.uncertainties[agents, tasks] = before_uncertainty * (1.0 - reduction)
        self.aoi[agents, tasks] = 0.0
        self.confidence[agents, tasks] = 1.0 - self.uncertainties[agents, tasks]
        self.demand_estimates[agents, tasks] = np.clip(demand_values, 0.0, 1.0)
        self.demand_uncertainties[agents, tasks] = before_demand_uncertainty * (1.0 - demand_reduction)
        self.demand_aoi[agents, tasks] = 0.0
        self.demand_confidence[agents, tasks] = 1.0 - self.demand_uncertainties[agents, tasks]
        self.queue_estimates[agents, tasks] = np.maximum(queue_values, 0.0)
        self.queue_uncertainties[agents, tasks] = before_queue_uncertainty * (1.0 - demand_reduction)
        self.queue_aoi[agents, tasks] = 0.0
        self.queue_confidence[agents, tasks] = 1.0 - self.queue_uncertainties[agents, tasks]
        self.arrival_estimates[agents, tasks] = np.maximum(arrival_values, 0.0)
        self.last_update_step[agents, tasks] = int(current_step)

        after_quality = self._task_quality(agents, tasks)
        weights = self.task_priorities[tasks]
        weight_sum = max(float(np.sum(weights)), 1e-6)
        spectrum_uncertainty_gain = float(
            np.sum(weights * (before_uncertainty - self.uncertainties[agents, tasks])) / weight_sum
        )
        spectrum_aoi_gain = float(np.sum(weights * before_aoi) / (weight_sum * self.max_aoi))
        demand_uncertainty_gain = float(
            np.sum(weights * (before_demand_uncertainty - self.demand_uncertainties[agents, tasks])) / weight_sum
        )
        demand_aoi_gain = float(np.sum(weights * before_demand_aoi) / (weight_sum * self.max_aoi))
        queue_uncertainty_gain = float(
            np.sum(
                weights
                * (before_queue_uncertainty - self.queue_uncertainties[agents, tasks])
            )
            / weight_sum
        )
        queue_aoi_gain = float(np.sum(weights * before_queue_aoi) / (weight_sum * self.max_aoi))
        return {
            "uncertainty_gain": 0.5 * (spectrum_uncertainty_gain + demand_uncertainty_gain),
            "aoi_gain": 0.5 * (spectrum_aoi_gain + demand_aoi_gain),
            "spectrum_uncertainty_gain": spectrum_uncertainty_gain,
            "spectrum_aoi_gain": spectrum_aoi_gain,
            "demand_uncertainty_gain": demand_uncertainty_gain,
            "demand_aoi_gain": demand_aoi_gain,
            "queue_uncertainty_gain": queue_uncertainty_gain,
            "queue_aoi_gain": queue_aoi_gain,
            "information_gain": np.maximum(after_quality - before_quality, 0.0).astype(np.float32),
        }

    def local_quality(self, agent_id: int, *, use_task_priorities: bool = True) -> float:
        agent_id = self._validate_agent_id(agent_id)
        task_weights = self.task_priorities if use_task_priorities else np.ones(self.num_tasks, dtype=np.float32)
        weights_sum = max(float(np.sum(task_weights)), 1e-6)
        spectrum_quality = 1.0 - 0.5 * (self.uncertainties[agent_id] + self.aoi[agent_id] / self.max_aoi)
        demand_quality = 1.0 - 0.5 * (
            self.demand_uncertainties[agent_id] + self.demand_aoi[agent_id] / self.max_aoi
        )
        quality = (
            self.spectrum_quality_weight * spectrum_quality
            + self.demand_quality_weight * demand_quality
        )
        return float(np.clip(np.sum(task_weights * quality) / weights_sum, 0.0, 1.0))

    def fuse_neighbor_message(
        self,
        receiver_id: int,
        task_id: int,
        *,
        estimate: float,
        uncertainty: float,
        confidence: float,
        message_aoi: float,
        demand_estimate: float = 0.5,
        demand_uncertainty: float = 1.0,
        demand_confidence: float = 0.0,
        demand_aoi: float = 0.0,
        queue_estimate: float = 0.0,
        queue_uncertainty: float = 1.0,
        queue_confidence: float = 0.0,
        queue_aoi: float = 0.0,
        arrival_estimate: float = 0.0,
        source_update_step: int,
        current_step: int,
        confidence_threshold: float,
        freshness_decay: float,
    ) -> dict[str, float]:
        receiver_id = self._validate_agent_id(receiver_id)
        task_id = self._validate_task_id(task_id)
        quality_before = self.local_quality(receiver_id)
        spectrum_accepted = self._fuse_dimension(
            self.estimates, self.uncertainties, self.aoi, self.confidence,
            receiver_id, task_id, estimate, uncertainty, confidence, message_aoi,
            source_update_step, current_step, confidence_threshold, freshness_decay,
        )
        demand_accepted = self._fuse_dimension(
            self.demand_estimates, self.demand_uncertainties, self.demand_aoi, self.demand_confidence,
            receiver_id, task_id, demand_estimate, demand_uncertainty, demand_confidence, demand_aoi,
            source_update_step, current_step, confidence_threshold, freshness_decay,
        )
        queue_accepted = self._fuse_dimension(
            self.queue_estimates, self.queue_uncertainties, self.queue_aoi, self.queue_confidence,
            receiver_id, task_id, queue_estimate, queue_uncertainty, queue_confidence, queue_aoi,
            source_update_step, current_step, confidence_threshold, freshness_decay,
            estimate_max=None,
        )
        if queue_accepted:
            self.arrival_estimates[receiver_id, task_id] = max(float(arrival_estimate), 0.0)
        accepted = spectrum_accepted or demand_accepted or queue_accepted
        if accepted:
            self.last_update_step[receiver_id, task_id] = max(
                int(self.last_update_step[receiver_id, task_id]), int(source_update_step)
            )
        quality_gain = max(self.local_quality(receiver_id) - quality_before, 0.0) if accepted else 0.0
        return {
            "accepted": float(accepted),
            "quality_gain": float(quality_gain),
            "spectrum_accepted": float(spectrum_accepted),
            "demand_accepted": float(demand_accepted),
            "queue_accepted": float(queue_accepted),
        }

    def _fuse_dimension(
        self,
        estimates: np.ndarray,
        uncertainties: np.ndarray,
        aois: np.ndarray,
        confidences: np.ndarray,
        receiver_id: int,
        task_id: int,
        estimate: float,
        uncertainty: float,
        confidence: float,
        message_aoi: float,
        source_update_step: int,
        current_step: int,
        confidence_threshold: float,
        freshness_decay: float,
        estimate_max: float | None = 1.0,
    ) -> bool:
        transit_age = max(int(current_step) - int(source_update_step), 0)
        effective_aoi = min(float(message_aoi) + transit_age, self.max_aoi)
        effective_confidence = float(
            np.clip(float(confidence) * np.exp(-float(freshness_decay) * effective_aoi), 0.0, 1.0)
        )
        effective_uncertainty = max(float(np.clip(uncertainty, 0.0, 1.0)), 1.0 - effective_confidence)
        if effective_confidence < float(confidence_threshold):
            return False

        local_uncertainty = float(uncertainties[receiver_id, task_id])
        local_aoi = float(aois[receiver_id, task_id])
        if effective_uncertainty >= local_uncertainty - 1e-6 and effective_aoi >= local_aoi - 1e-6:
            return False

        local_confidence = float(confidences[receiver_id, task_id])
        weight_sum = max(local_confidence + effective_confidence, 1e-6)
        value = float(max(float(estimate), 0.0))
        if estimate_max is not None:
            value = min(value, float(estimate_max))
        estimates[receiver_id, task_id] = (
            local_confidence * float(estimates[receiver_id, task_id])
            + effective_confidence * value
        ) / weight_sum
        uncertainties[receiver_id, task_id] = min(local_uncertainty, effective_uncertainty)
        aois[receiver_id, task_id] = min(local_aoi, effective_aoi)
        confidences[receiver_id, task_id] = max(local_confidence, effective_confidence)
        return True

    def mean_quality(self) -> float:
        return float(np.mean([self.local_quality(i) for i in range(self.num_agents)]))

    def mean_uncertainty(self) -> float:
        weights = np.broadcast_to(self.task_priorities, self.uncertainties.shape)
        spectrum = np.sum(weights * self.uncertainties) / max(float(np.sum(weights)), 1e-6)
        demand = np.sum(weights * self.demand_uncertainties) / max(float(np.sum(weights)), 1e-6)
        return float(self.spectrum_quality_weight * spectrum + self.demand_quality_weight * demand)

    def mean_aoi(self) -> float:
        weights = np.broadcast_to(self.task_priorities, self.aoi.shape)
        spectrum = np.sum(weights * self.aoi) / max(float(np.sum(weights)), 1e-6)
        demand = np.sum(weights * self.demand_aoi) / max(float(np.sum(weights)), 1e-6)
        return float(self.spectrum_quality_weight * spectrum + self.demand_quality_weight * demand)

    def mean_estimation_error(
        self,
        true_states: np.ndarray,
        demand_levels: np.ndarray | None = None,
    ) -> float:
        truth = np.asarray(true_states, dtype=np.float32)
        if truth.shape != (self.num_tasks,):
            raise ValueError("true_states must have shape [num_tasks].")
        weights = np.broadcast_to(self.task_priorities, self.estimates.shape)
        spectrum_error = np.abs(self.estimates - truth[None, :])
        spectrum = np.sum(weights * spectrum_error) / max(float(np.sum(weights)), 1e-6)
        if demand_levels is None:
            return float(spectrum)
        demand_truth = np.asarray(demand_levels, dtype=np.float32)
        if demand_truth.shape != (self.num_tasks,):
            raise ValueError("demand_levels must have shape [num_tasks].")
        demand_error = np.abs(self.demand_estimates - demand_truth[None, :])
        demand = np.sum(weights * demand_error) / max(float(np.sum(weights)), 1e-6)
        return float(self.spectrum_quality_weight * spectrum + self.demand_quality_weight * demand)

    def mean_spectrum_estimation_error(self, true_states: np.ndarray) -> float:
        truth = np.asarray(true_states, dtype=np.float32)
        if truth.shape != (self.num_tasks,):
            raise ValueError("true_states must have shape [num_tasks].")
        weights = np.broadcast_to(self.task_priorities, self.estimates.shape)
        return float(
            np.sum(weights * np.abs(self.estimates - truth[None, :]))
            / max(float(np.sum(weights)), 1e-6)
        )

    def mean_demand_estimation_error(self, demand_levels: np.ndarray) -> float:
        truth = np.asarray(demand_levels, dtype=np.float32)
        if truth.shape != (self.num_tasks,):
            raise ValueError("demand_levels must have shape [num_tasks].")
        weights = np.broadcast_to(self.task_priorities, self.demand_estimates.shape)
        return float(
            np.sum(weights * np.abs(self.demand_estimates - truth[None, :]))
            / max(float(np.sum(weights)), 1e-6)
        )

    def snapshot(self, agent_id: int, task_id: int) -> LocalTaskBelief:
        agent_id = self._validate_agent_id(agent_id)
        task_id = self._validate_task_id(task_id)
        return LocalTaskBelief(
            estimate=float(self.estimates[agent_id, task_id]),
            uncertainty=float(self.uncertainties[agent_id, task_id]),
            aoi=float(self.aoi[agent_id, task_id]),
            confidence=float(self.confidence[agent_id, task_id]),
            demand_estimate=float(self.demand_estimates[agent_id, task_id]),
            demand_uncertainty=float(self.demand_uncertainties[agent_id, task_id]),
            demand_aoi=float(self.demand_aoi[agent_id, task_id]),
            demand_confidence=float(self.demand_confidence[agent_id, task_id]),
            queue_estimate=float(self.queue_estimates[agent_id, task_id]),
            queue_uncertainty=float(self.queue_uncertainties[agent_id, task_id]),
            queue_aoi=float(self.queue_aoi[agent_id, task_id]),
            queue_confidence=float(self.queue_confidence[agent_id, task_id]),
            arrival_estimate=float(self.arrival_estimates[agent_id, task_id]),
            last_update_step=int(self.last_update_step[agent_id, task_id]),
        )

    def _task_quality(self, agent_ids: np.ndarray, task_ids: np.ndarray) -> np.ndarray:
        spectrum = 1.0 - 0.5 * (
            self.uncertainties[agent_ids, task_ids] + self.aoi[agent_ids, task_ids] / self.max_aoi
        )
        demand = 1.0 - 0.5 * (
            self.demand_uncertainties[agent_ids, task_ids]
            + self.demand_aoi[agent_ids, task_ids] / self.max_aoi
        )
        return (
            self.spectrum_quality_weight * spectrum + self.demand_quality_weight * demand
        ).astype(np.float32)

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
