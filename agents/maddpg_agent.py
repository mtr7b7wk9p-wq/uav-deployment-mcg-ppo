from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from agents.maddpg_buffer import MADDPGBatch, MADDPGReplayBuffer


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_hidden_layers: int = 2,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = in_dim
    for _ in range(max(1, num_hidden_layers)):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class DiscreteActor(nn.Module):
    """
    离散动作 actor：
    输入局部观测，输出每个离散动作的 logits。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            in_dim=obs_dim,
            hidden_dim=hidden_dim,
            out_dim=action_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralizedCritic(nn.Module):
    """
    Centralized critic:
    输入联合局部观测 + 联合动作 one-hot
    输出当前 agent 对应的 Q 值
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        input_dim = num_agents * obs_dim + num_agents * action_dim
        self.net = _build_mlp(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(
        self,
        joint_obs: torch.Tensor,          # [B, N, obs_dim]
        joint_action_onehot: torch.Tensor # [B, N, action_dim]
    ) -> torch.Tensor:
        obs_flat = joint_obs.reshape(joint_obs.shape[0], -1)
        act_flat = joint_action_onehot.reshape(joint_action_onehot.shape[0], -1)
        x = torch.cat([obs_flat, act_flat], dim=-1)
        return self.net(x)


@dataclass
class MADDPGConfig:
    num_agents: int
    local_obs_dim: int
    action_dim: int

    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    num_hidden_layers: int = 2

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    gamma: float = 0.99
    tau: float = 0.01

    replay_size: int = 200000
    batch_size: int = 256
    update_after: int = 1000
    update_every: int = 50
    gradient_steps: int = 50
    policy_update_freq: int = 2

    gumbel_tau: float = 1.0
    explore_epsilon: float = 0.10

    max_grad_norm: float = 10.0
    device: str = "cuda"


class MADDPGAgent:
    """
    离散化 MADDPG 风格最小实现：

    为什么不是标准连续 MADDPG？
    - 当前环境动作空间是离散动作；
    - 若强行输出连续动作再硬映射，会让工程结构不自然且训练不稳定；
    - 因此这里采用“离散 actor logits + centralized critic + Gumbel-Softmax actor update”的最小可训练方案。

    核心结构仍然保留：
    - 每个 UAV 独立 actor
    - 每个 UAV 一个 centralized critic
    - target actor / target critic
    - replay buffer
    - soft update
    """

    def __init__(self, config: MADDPGConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.num_agents = int(config.num_agents)
        self.obs_dim = int(config.local_obs_dim)
        self.action_dim = int(config.action_dim)

        self.actors: List[DiscreteActor] = []
        self.target_actors: List[DiscreteActor] = []
        self.actor_optimizers: List[Adam] = []

        self.critics: List[CentralizedCritic] = []
        self.target_critics: List[CentralizedCritic] = []
        self.critic_optimizers: List[Adam] = []

        for _ in range(self.num_agents):
            actor = DiscreteActor(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=config.actor_hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
            ).to(self.device)
            target_actor = copy.deepcopy(actor).to(self.device)
            actor_opt = Adam(actor.parameters(), lr=config.actor_lr)

            critic = CentralizedCritic(
                num_agents=self.num_agents,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=config.critic_hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
            ).to(self.device)
            target_critic = copy.deepcopy(critic).to(self.device)
            critic_opt = Adam(critic.parameters(), lr=config.critic_lr)

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.actor_optimizers.append(actor_opt)

            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.critic_optimizers.append(critic_opt)

        self.total_train_steps = 0

    @staticmethod
    def _apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        # action_mask=1 表示可选；=0 表示禁用
        invalid = (action_mask <= 0.0)
        masked_logits = logits.masked_fill(invalid, -1e9)
        return masked_logits

    @staticmethod
    def _actions_to_onehot(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
        # actions: [B, N]
        onehot = F.one_hot(actions.long(), num_classes=action_dim).float()
        return onehot

    def _joint_onehot_from_target_actors(
        self,
        next_obs: torch.Tensor,           # [B, N, obs_dim]
        next_action_mask: torch.Tensor,   # [B, N, action_dim]
    ) -> torch.Tensor:
        next_onehots = []
        for agent_idx in range(self.num_agents):
            logits = self.target_actors[agent_idx](next_obs[:, agent_idx, :])
            logits = self._apply_action_mask(logits, next_action_mask[:, agent_idx, :])
            target_actions = torch.argmax(logits, dim=-1)
            onehot = F.one_hot(target_actions, num_classes=self.action_dim).float()
            next_onehots.append(onehot.unsqueeze(1))
        return torch.cat(next_onehots, dim=1)  # [B, N, action_dim]

    def _joint_onehot_from_current_actors_detached(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        onehots = []
        for agent_idx in range(self.num_agents):
            logits = self.actors[agent_idx](obs[:, agent_idx, :])
            logits = self._apply_action_mask(logits, action_mask[:, agent_idx, :])
            probs = F.softmax(logits, dim=-1)
            actions = torch.argmax(probs, dim=-1)
            onehot = F.one_hot(actions, num_classes=self.action_dim).float()
            if detach:
                onehot = onehot.detach()
            onehots.append(onehot.unsqueeze(1))
        return torch.cat(onehots, dim=1)

    def _agent_actor_action_onehot_st(
        self,
        agent_idx: int,
        obs_i: torch.Tensor,              # [B, obs_dim]
        action_mask_i: torch.Tensor,      # [B, action_dim]
    ) -> torch.Tensor:
        logits = self.actors[agent_idx](obs_i)
        logits = self._apply_action_mask(logits, action_mask_i)
        # Straight-through Gumbel-Softmax for discrete action relaxation
        onehot = F.gumbel_softmax(
            logits,
            tau=self.cfg.gumbel_tau,
            hard=True,
            dim=-1,
        )
        return onehot

    @torch.no_grad()
    def act(
        self,
        local_obs_batch: np.ndarray,       # [N, obs_dim]
        action_mask_batch: np.ndarray,     # [N, action_dim]
        explore: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if local_obs_batch.shape[0] != self.num_agents:
            raise ValueError(
                f"MADDPG act expects num_agents={self.num_agents}, "
                f"but got first dim={local_obs_batch.shape[0]}"
            )

        actions: List[int] = []
        for agent_idx in range(self.num_agents):
            obs_t = torch.tensor(
                local_obs_batch[agent_idx:agent_idx + 1],
                dtype=torch.float32,
                device=self.device,
            )
            mask_t = torch.tensor(
                action_mask_batch[agent_idx:agent_idx + 1],
                dtype=torch.float32,
                device=self.device,
            )

            logits = self.actors[agent_idx](obs_t)
            logits = self._apply_action_mask(logits, mask_t)

            valid_actions = np.where(action_mask_batch[agent_idx] > 0.0)[0]
            if valid_actions.size == 0:
                actions.append(0)
                continue

            if explore and np.random.rand() < self.cfg.explore_epsilon:
                action = int(np.random.choice(valid_actions))
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = int(dist.sample().item()) if explore else int(torch.argmax(probs, dim=-1).item())

            actions.append(action)

        dummy_log_probs = np.zeros((self.num_agents,), dtype=np.float32)
        dummy_values = np.zeros((self.num_agents,), dtype=np.float32)
        return (
            np.array(actions, dtype=np.int64),
            dummy_log_probs,
            dummy_values,
        )

    @torch.no_grad()
    def greedy_act(
        self,
        local_obs_batch: np.ndarray,
        action_mask_batch: np.ndarray,
    ) -> np.ndarray:
        actions, _, _ = self.act(
            local_obs_batch=local_obs_batch,
            action_mask_batch=action_mask_batch,
            explore=False,
        )
        return actions

    def can_update(self, replay_buffer: MADDPGReplayBuffer) -> bool:
        return len(replay_buffer) >= int(self.cfg.update_after)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = float(self.cfg.tau)
        with torch.no_grad():
            for src_p, tgt_p in zip(source.parameters(), target.parameters()):
                tgt_p.data.mul_(1.0 - tau).add_(tau * src_p.data)

    def _update_critic(
        self,
        agent_idx: int,
        batch: MADDPGBatch,
    ) -> Dict[str, float]:
        critic = self.critics[agent_idx]
        target_critic = self.target_critics[agent_idx]
        critic_opt = self.critic_optimizers[agent_idx]

        with torch.no_grad():
            next_joint_onehot = self._joint_onehot_from_target_actors(
                next_obs=batch.next_obs,
                next_action_mask=batch.next_action_mask,
            )
            target_q = target_critic(batch.next_obs, next_joint_onehot)
            y = batch.rewards + (1.0 - batch.dones) * float(self.cfg.gamma) * target_q

        current_joint_onehot = self._actions_to_onehot(batch.actions, self.action_dim)
        current_q = critic(batch.obs, current_joint_onehot)

        critic_loss = F.mse_loss(current_q, y)

        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.max_grad_norm)
        critic_opt.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(current_q.mean().item()),
            "target_q_mean": float(y.mean().item()),
        }

    def _update_actor(
        self,
        agent_idx: int,
        batch: MADDPGBatch,
    ) -> Dict[str, float]:
        actor = self.actors[agent_idx]
        actor_opt = self.actor_optimizers[agent_idx]
        critic = self.critics[agent_idx]

        joint_actions = []
        for j in range(self.num_agents):
            if j == agent_idx:
                onehot = self._agent_actor_action_onehot_st(
                    agent_idx=j,
                    obs_i=batch.obs[:, j, :],
                    action_mask_i=batch.action_mask[:, j, :],
                )
            else:
                with torch.no_grad():
                    logits = self.actors[j](batch.obs[:, j, :])
                    logits = self._apply_action_mask(logits, batch.action_mask[:, j, :])
                    probs = F.softmax(logits, dim=-1)
                    act = torch.argmax(probs, dim=-1)
                    onehot = F.one_hot(act, num_classes=self.action_dim).float()
            joint_actions.append(onehot.unsqueeze(1))

        joint_action_onehot = torch.cat(joint_actions, dim=1)
        actor_q = critic(batch.obs, joint_action_onehot)
        actor_loss = -actor_q.mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
        actor_opt.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "actor_q_mean": float(actor_q.mean().item()),
        }

    def update(
        self,
        replay_buffer: MADDPGReplayBuffer,
    ) -> Dict[str, Any]:
        if not self.can_update(replay_buffer):
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "q_mean": 0.0,
                "target_q_mean": 0.0,
                "actor_q_mean": 0.0,
                "buffer_size": float(len(replay_buffer)),
                "num_agents": int(self.num_agents),
                "per_agent_stats": [],
                "skipped_update": True,
            }

        actor_loss_meter: List[float] = []
        critic_loss_meter: List[float] = []
        q_mean_meter: List[float] = []
        target_q_mean_meter: List[float] = []
        actor_q_mean_meter: List[float] = []
        per_agent_stats: List[Dict[str, float]] = []

        for grad_step in range(int(self.cfg.gradient_steps)):
            batch = replay_buffer.sample(
                batch_size=int(self.cfg.batch_size),
                device=str(self.device),
            )

            critic_stats_all = []
            for agent_idx in range(self.num_agents):
                critic_stats = self._update_critic(agent_idx=agent_idx, batch=batch)
                critic_stats_all.append(critic_stats)

            actor_stats_all = []
            if (self.total_train_steps + grad_step) % int(self.cfg.policy_update_freq) == 0:
                for agent_idx in range(self.num_agents):
                    actor_stats = self._update_actor(agent_idx=agent_idx, batch=batch)
                    actor_stats_all.append(actor_stats)

                    self._soft_update(self.actors[agent_idx], self.target_actors[agent_idx])
                    self._soft_update(self.critics[agent_idx], self.target_critics[agent_idx])

            for agent_idx in range(self.num_agents):
                merged = {
                    "agent_idx": int(agent_idx),
                    **critic_stats_all[agent_idx],
                }
                if agent_idx < len(actor_stats_all):
                    merged.update(actor_stats_all[agent_idx])
                else:
                    merged.update({
                        "actor_loss": 0.0,
                        "actor_q_mean": 0.0,
                    })
                per_agent_stats.append(merged)

                actor_loss_meter.append(float(merged["actor_loss"]))
                critic_loss_meter.append(float(merged["critic_loss"]))
                q_mean_meter.append(float(merged["q_mean"]))
                target_q_mean_meter.append(float(merged["target_q_mean"]))
                actor_q_mean_meter.append(float(merged["actor_q_mean"]))

        self.total_train_steps += int(self.cfg.gradient_steps)

        return {
            "actor_loss": float(np.mean(actor_loss_meter)) if actor_loss_meter else 0.0,
            "critic_loss": float(np.mean(critic_loss_meter)) if critic_loss_meter else 0.0,
            "q_mean": float(np.mean(q_mean_meter)) if q_mean_meter else 0.0,
            "target_q_mean": float(np.mean(target_q_mean_meter)) if target_q_mean_meter else 0.0,
            "actor_q_mean": float(np.mean(actor_q_mean_meter)) if actor_q_mean_meter else 0.0,
            "buffer_size": float(len(replay_buffer)),
            "num_agents": int(self.num_agents),
            "per_agent_stats": per_agent_stats,
            "skipped_update": False,
        }

    def save(self, path: str) -> None:
        payload = {
            "agent_type": "maddpg",
            "config": self.cfg.__dict__,
            "actors": [actor.state_dict() for actor in self.actors],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "total_train_steps": int(self.total_train_steps),
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)

        def _load_module_list(modules: List[nn.Module], states: List[Dict[str, Any]], name: str) -> None:
            if len(modules) != len(states):
                raise ValueError(
                    f"{name} checkpoint length {len(states)} != current num_agents {len(modules)}"
                )
            for module, state in zip(modules, states):
                module.load_state_dict(state, strict=strict)

        _load_module_list(self.actors, payload["actors"], "actors")
        _load_module_list(self.target_actors, payload["target_actors"], "target_actors")
        _load_module_list(self.critics, payload["critics"], "critics")
        _load_module_list(self.target_critics, payload["target_critics"], "target_critics")

        actor_opt_states = payload.get("actor_optimizers", [])
        critic_opt_states = payload.get("critic_optimizers", [])

        if len(actor_opt_states) == len(self.actor_optimizers):
            for opt, state in zip(self.actor_optimizers, actor_opt_states):
                opt.load_state_dict(state)

        if len(critic_opt_states) == len(self.critic_optimizers):
            for opt, state in zip(self.critic_optimizers, critic_opt_states):
                opt.load_state_dict(state)

        self.total_train_steps = int(payload.get("total_train_steps", 0))
        return payload