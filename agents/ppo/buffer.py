from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class PPOMiniBatch:
    local_obs: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class PPOBuffer:
    """
    Local-critic PPO buffer.

    每个 agent 在每一步贡献一条样本：
    - local_obs_i
    - action_mask_i
    - action_i
    - scalar shared reward or one reward per agent
    - done
    - value_i = critic(local_obs_i)
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.local_obs: List[np.ndarray] = []
        self.action_masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []

        self.returns: np.ndarray | None = None
        self.advantages: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        local_obs: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: float,
        value: float,
    ) -> None:
        self.local_obs.append(local_obs.astype(np.float32))
        self.action_masks.append(action_mask.astype(np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def add_multi_agent_step(
        self,
        local_obs_batch: np.ndarray,     # [N, local_obs_dim]
        action_mask_batch: np.ndarray,   # [N, action_dim]
        action_batch: np.ndarray,        # [N]
        log_prob_batch: np.ndarray,      # [N]
        reward: float | np.ndarray,
        done: bool,
        value_batch: np.ndarray,         # [N]
    ) -> None:
        num_agents = local_obs_batch.shape[0]
        done_f = 1.0 if done else 0.0
        reward_array = np.asarray(reward, dtype=np.float32)
        if reward_array.ndim == 0:
            reward_array = np.full((num_agents,), float(reward_array), dtype=np.float32)
        elif reward_array.shape != (num_agents,):
            raise ValueError(f"reward must be scalar or have shape ({num_agents},).")

        for i in range(num_agents):
            self.add(
                local_obs=local_obs_batch[i],
                action_mask=action_mask_batch[i],
                action=int(action_batch[i]),
                log_prob=float(log_prob_batch[i]),
                reward=float(reward_array[i]),
                done=done_f,
                value=float(value_batch[i]),
            )

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
        num_agents: int,
    ) -> None:
        n = len(self.rewards)
        if n == 0:
            raise ValueError("Cannot compute returns/advantages on empty buffer.")
        if n % num_agents != 0:
            raise ValueError("Buffer size must be divisible by num_agents.")

        horizon = n // num_agents

        rewards = np.array(self.rewards, dtype=np.float32).reshape(horizon, num_agents)
        dones = np.array(self.dones, dtype=np.float32).reshape(horizon, num_agents)
        values = np.array(self.values, dtype=np.float32).reshape(horizon, num_agents)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        next_adv = np.zeros((num_agents,), dtype=np.float32)
        next_value = last_values.astype(np.float32).copy()

        for t in reversed(range(horizon)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            next_adv = delta + gamma * gae_lambda * mask * next_adv
            advantages[t] = next_adv
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        self.advantages = advantages.astype(np.float32)
        self.returns = returns.astype(np.float32)

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.advantages is None or self.returns is None:
            raise ValueError("Call compute_returns_and_advantages() before as_tensors().")

        data = {
            "local_obs": torch.tensor(np.array(self.local_obs, dtype=np.float32), device=device),
            "action_mask": torch.tensor(np.array(self.action_masks, dtype=np.float32), device=device),
            "actions": torch.tensor(np.array(self.actions, dtype=np.int64), device=device),
            "old_log_probs": torch.tensor(np.array(self.log_probs, dtype=np.float32), device=device),
            "old_values": torch.tensor(np.array(self.values, dtype=np.float32), device=device),
            "returns": torch.tensor(self.returns, dtype=torch.float32, device=device),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32, device=device),
        }
        return data

    def iterate_minibatches(
        self,
        device: torch.device,
        mini_batch_size: int,
        shuffle: bool = True,
    ):
        data = self.as_tensors(device=device)
        total_size = data["actions"].shape[0]
        indices = np.arange(total_size)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, total_size, mini_batch_size):
            end = min(start + mini_batch_size, total_size)
            batch_idx = indices[start:end]

            yield PPOMiniBatch(
                local_obs=data["local_obs"][batch_idx],
                action_mask=data["action_mask"][batch_idx],
                actions=data["actions"][batch_idx],
                old_log_probs=data["old_log_probs"][batch_idx],
                old_values=data["old_values"][batch_idx],
                returns=data["returns"][batch_idx],
                advantages=data["advantages"][batch_idx],
            )
