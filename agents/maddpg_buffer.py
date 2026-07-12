from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class MADDPGBatch:
    obs: torch.Tensor                  # [B, N, obs_dim]
    action_mask: torch.Tensor          # [B, N, action_dim]
    actions: torch.Tensor              # [B, N]
    rewards: torch.Tensor              # [B, 1]
    next_obs: torch.Tensor             # [B, N, obs_dim]
    next_action_mask: torch.Tensor     # [B, N, action_dim]
    dones: torch.Tensor                # [B, 1]


class MADDPGReplayBuffer:
    """
    Multi-agent replay buffer for centralized-critic off-policy training.

    存储的是联合 transition：
    - obs: [num_agents, obs_dim]
    - action_mask: [num_agents, action_dim]
    - actions: [num_agents]
    - reward: scalar team reward
    - next_obs: [num_agents, obs_dim]
    - next_action_mask: [num_agents, action_dim]
    - done: scalar
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
        self.action_mask = np.zeros((self.capacity, self.num_agents, self.action_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.num_agents), dtype=np.int64)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
        self.next_action_mask = np.zeros((self.capacity, self.num_agents, self.action_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs.astype(np.float32)
        self.action_mask[self.ptr] = action_mask.astype(np.float32)
        self.actions[self.ptr] = actions.astype(np.int64)
        self.rewards[self.ptr, 0] = float(reward)
        self.next_obs[self.ptr] = next_obs.astype(np.float32)
        self.next_action_mask[self.ptr] = next_action_mask.astype(np.float32)
        self.dones[self.ptr, 0] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: str,
    ) -> MADDPGBatch:
        if self.size < batch_size:
            raise ValueError(
                f"Replay buffer size {self.size} is smaller than batch_size {batch_size}."
            )

        indices = np.random.randint(0, self.size, size=batch_size)

        return MADDPGBatch(
            obs=torch.tensor(self.obs[indices], dtype=torch.float32, device=device),
            action_mask=torch.tensor(self.action_mask[indices], dtype=torch.float32, device=device),
            actions=torch.tensor(self.actions[indices], dtype=torch.long, device=device),
            rewards=torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            next_obs=torch.tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            next_action_mask=torch.tensor(self.next_action_mask[indices], dtype=torch.float32, device=device),
            dones=torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
        )

    def summary(self) -> Dict[str, int]:
        return {
            "capacity": int(self.capacity),
            "size": int(self.size),
            "ptr": int(self.ptr),
            "num_agents": int(self.num_agents),
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
        }