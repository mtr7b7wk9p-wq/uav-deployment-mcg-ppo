from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RandomPolicy:
    action_size: int
    num_agents: int
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, action_mask: Optional[np.ndarray] = None) -> List[int]:
        """
        If action_mask is provided, choose uniformly among valid actions.
        action_mask shape: [num_agents, action_size]
        """
        actions = []

        if action_mask is None:
            for _ in range(self.num_agents):
                a = int(self.rng.integers(0, self.action_size))
                actions.append(a)
            return actions

        for i in range(self.num_agents):
            valid = np.where(action_mask[i] > 0.5)[0]
            if valid.size == 0:
                actions.append(0)
            else:
                idx = int(self.rng.integers(0, valid.size))
                actions.append(int(valid[idx]))

        return actions