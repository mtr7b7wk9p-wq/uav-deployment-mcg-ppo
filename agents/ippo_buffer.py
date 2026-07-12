from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from agents.ppo.buffer import PPOBuffer


@dataclass
class AgentBufferSummary:
    agent_idx: int
    buffer_size: int


class IPPOBuffer:
    """
    Independent PPO buffer container.

    每个 UAV / agent 持有一个独立 PPOBuffer：
    - 各自存 local_obs
    - 各自存 action_mask
    - 各自存 action / log_prob / value
    - reward 目前仍使用团队共享 reward（与当前环境兼容）
    - done 独立记录，但来源于同一个 episode 终止

    这样做的关键区别在于：
    Shared PPO:
        所有 agent 的样本最终混在一个共享参数 buffer 中更新
    IPPO:
        每个 agent 的样本只用于该 agent 自己的 PPO 更新
    """

    def __init__(self, num_agents: int):
        if num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        self.num_agents = int(num_agents)
        self.buffers: List[PPOBuffer] = [PPOBuffer() for _ in range(self.num_agents)]

    def reset(self) -> None:
        for buf in self.buffers:
            buf.reset()

    def __len__(self) -> int:
        return int(sum(len(buf) for buf in self.buffers))

    def add_step(
        self,
        local_obs_batch: np.ndarray,     # [N, obs_dim]
        action_mask_batch: np.ndarray,   # [N, action_dim]
        action_batch: np.ndarray,        # [N]
        log_prob_batch: np.ndarray,      # [N]
        reward: float,
        done: bool,
        value_batch: np.ndarray,         # [N]
    ) -> None:
        if local_obs_batch.shape[0] != self.num_agents:
            raise ValueError(
                f"local_obs_batch first dim {local_obs_batch.shape[0]} != num_agents {self.num_agents}"
            )

        done_f = 1.0 if done else 0.0
        for agent_idx in range(self.num_agents):
            self.buffers[agent_idx].add(
                local_obs=local_obs_batch[agent_idx],
                action_mask=action_mask_batch[agent_idx],
                action=int(action_batch[agent_idx]),
                log_prob=float(log_prob_batch[agent_idx]),
                reward=float(reward),
                done=done_f,
                value=float(value_batch[agent_idx]),
            )

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,   # [N]
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if last_values.shape[0] != self.num_agents:
            raise ValueError(
                f"last_values length {last_values.shape[0]} != num_agents {self.num_agents}"
            )

        for agent_idx, buf in enumerate(self.buffers):
            buf.compute_returns_and_advantages(
                last_values=np.array([float(last_values[agent_idx])], dtype=np.float32),
                gamma=gamma,
                gae_lambda=gae_lambda,
                num_agents=1,
            )

    def get_agent_buffer(self, agent_idx: int) -> PPOBuffer:
        return self.buffers[agent_idx]

    def summaries(self) -> List[Dict[str, int]]:
        return [
            {
                "agent_idx": agent_idx,
                "buffer_size": len(buf),
            }
            for agent_idx, buf in enumerate(self.buffers)
        ]