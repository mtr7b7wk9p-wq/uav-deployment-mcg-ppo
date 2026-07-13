"""Delayed and lossy transport for local cognition messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class CognitionMessage:
    sender_id: int
    receiver_id: int
    task_id: int
    estimate: float
    uncertainty: float
    confidence: float
    aoi: float
    created_step: int
    arrival_step: int
    demand_estimate: float = 0.5
    demand_uncertainty: float = 1.0
    demand_confidence: float = 0.0
    demand_aoi: float = 0.0
    queue_estimate: float = 0.0
    queue_uncertainty: float = 1.0
    queue_confidence: float = 0.0
    queue_aoi: float = 0.0
    arrival_estimate: float = 0.0


@dataclass(frozen=True)
class TransmissionStats:
    attempted: int
    dropped: int
    queued: int


class NeighborCommunicationModel:
    """Own message delivery timing and loss without reading agent beliefs."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        delay_steps: int,
        packet_loss_rate: float,
    ) -> None:
        self.rng = rng
        self.delay_steps = int(max(delay_steps, 0))
        self.packet_loss_rate = float(np.clip(packet_loss_rate, 0.0, 1.0))
        self._pending: List[CognitionMessage] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()

    def submit(self, messages: Iterable[CognitionMessage]) -> TransmissionStats:
        attempted = 0
        dropped = 0
        queued = 0
        for message in messages:
            attempted += 1
            if float(self.rng.random()) < self.packet_loss_rate:
                dropped += 1
                continue
            self._pending.append(message)
            queued += 1
        return TransmissionStats(attempted=attempted, dropped=dropped, queued=queued)

    def deliver(self, current_step: int) -> List[CognitionMessage]:
        due: List[CognitionMessage] = []
        waiting: List[CognitionMessage] = []
        for message in self._pending:
            if message.arrival_step <= int(current_step):
                due.append(message)
            else:
                waiting.append(message)
        self._pending = waiting
        return due
