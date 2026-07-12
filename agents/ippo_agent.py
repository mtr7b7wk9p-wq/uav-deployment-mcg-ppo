from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from agents.ippo_buffer import IPPOBuffer
from agents.ppo.models import LocalActorCritic


@dataclass
class IPPOConfig:
    num_agents: int
    local_obs_dim: int
    action_dim: int

    # 与现有 PPOConfig 对齐，方便工程复用
    global_state_dim: int = 0

    hidden_dim: int = 256
    num_hidden_layers: int = 2

    lr: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_eps: float = 0.2
    ppo_epochs: int = 10
    mini_batch_size: int = 256

    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    device: str = "cuda"

    # structured obs reservation
    use_structured_obs_encoder: bool = False
    use_neighbor_encoder: bool = False

    neighbor_encoder_hidden_dim: int = 64
    neighbor_context_dim: int = 64
    neighbor_pooling_type: str = "mean_max"

    max_obs_users: int = 20
    max_obs_uavs: int = 2
    num_direction_sectors: int = 4
    num_radial_bins: int = 3


class IndependentPPOAgent:
    """
    Independent PPO:
    - 每个 agent 一套独立 actor-critic 参数
    - 每个 agent 独立采样动作
    - 每个 agent 独立 PPO update
    - 团队级指标由训练脚本汇总

    这样它与参数共享 PPO 的核心区别是：
    1. 参数不共享
    2. buffer 不共享
    3. update 不共享
    """

    def __init__(self, config: IPPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.num_agents = int(config.num_agents)

        if self.num_agents <= 0:
            raise ValueError("IPPO num_agents must be positive.")

        self.models: List[LocalActorCritic] = []
        self.optimizers: List[Adam] = []

        for _ in range(self.num_agents):
            model = LocalActorCritic(
                local_obs_dim=config.local_obs_dim,
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                use_structured_obs_encoder=config.use_structured_obs_encoder,
                use_neighbor_encoder=config.use_neighbor_encoder,
                max_obs_users=config.max_obs_users,
                max_obs_uavs=config.max_obs_uavs,
                num_direction_sectors=config.num_direction_sectors,
                num_radial_bins=config.num_radial_bins,
                neighbor_encoder_hidden_dim=config.neighbor_encoder_hidden_dim,
                neighbor_context_dim=config.neighbor_context_dim,
                neighbor_pooling_type=config.neighbor_pooling_type,
            ).to(self.device)

            optimizer = Adam(model.parameters(), lr=config.lr)

            self.models.append(model)
            self.optimizers.append(optimizer)

    @torch.no_grad()
    def act(
        self,
        local_obs_batch: np.ndarray,      # [N, obs_dim]
        action_mask_batch: np.ndarray,    # [N, action_dim]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if local_obs_batch.shape[0] != self.num_agents:
            raise ValueError(
                f"IPPO act expects batch first dim == num_agents ({self.num_agents}), "
                f"but got {local_obs_batch.shape[0]}"
            )

        actions: List[int] = []
        log_probs: List[float] = []
        values: List[float] = []

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

            out = self.models[agent_idx].act(
                local_obs=obs_t,
                action_mask=mask_t,
            )

            actions.append(int(out.actions.item()))
            log_probs.append(float(out.log_probs.item()))
            values.append(float(out.values.item()))

        return (
            np.array(actions, dtype=np.int64),
            np.array(log_probs, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )

    @torch.no_grad()
    def greedy_act(
        self,
        local_obs_batch: np.ndarray,
        action_mask_batch: np.ndarray,
    ) -> np.ndarray:
        if local_obs_batch.shape[0] != self.num_agents:
            raise ValueError(
                f"IPPO greedy_act expects batch first dim == num_agents ({self.num_agents}), "
                f"but got {local_obs_batch.shape[0]}"
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
            logits = self.models[agent_idx].actor(obs_t, mask_t)
            action = int(torch.argmax(logits, dim=-1).item())
            actions.append(action)

        return np.array(actions, dtype=np.int64)

    @torch.no_grad()
    def get_values(self, local_obs_batch: np.ndarray) -> np.ndarray:
        if local_obs_batch.shape[0] != self.num_agents:
            raise ValueError(
                f"IPPO get_values expects batch first dim == num_agents ({self.num_agents}), "
                f"but got {local_obs_batch.shape[0]}"
            )

        values: List[float] = []
        for agent_idx in range(self.num_agents):
            obs_t = torch.tensor(
                local_obs_batch[agent_idx:agent_idx + 1],
                dtype=torch.float32,
                device=self.device,
            )
            value = self.models[agent_idx].get_values(obs_t)
            values.append(float(value.item()))
        return np.array(values, dtype=np.float32)

    def _update_single_agent(
        self,
        agent_idx: int,
        local_buffer,
    ) -> Dict[str, float]:
        model = self.models[agent_idx]
        optimizer = self.optimizers[agent_idx]

        policy_loss_meter: List[float] = []
        value_loss_meter: List[float] = []
        entropy_meter: List[float] = []
        approx_kl_meter: List[float] = []
        clip_frac_meter: List[float] = []
        total_loss_meter: List[float] = []

        for _ in range(self.cfg.ppo_epochs):
            for batch in local_buffer.iterate_minibatches(
                device=self.device,
                mini_batch_size=self.cfg.mini_batch_size,
                shuffle=True,
            ):
                log_probs, entropy, values = model.evaluate_actions(
                    local_obs=batch.local_obs,
                    actions=batch.actions,
                    action_mask=batch.action_mask,
                )

                new_log_probs = log_probs
                entropy_mean = entropy.mean()
                new_values = values

                log_ratio = new_log_probs - batch.old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_eps,
                    1.0 + self.cfg.clip_eps,
                ) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, batch.returns)

                total_loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_mean
                )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )

                policy_loss_meter.append(float(policy_loss.item()))
                value_loss_meter.append(float(value_loss.item()))
                entropy_meter.append(float(entropy_mean.item()))
                approx_kl_meter.append(float(approx_kl))
                clip_frac_meter.append(float(clip_frac))
                total_loss_meter.append(float(total_loss.item()))

        return {
            "agent_idx": int(agent_idx),
            "policy_loss": float(np.mean(policy_loss_meter)) if policy_loss_meter else 0.0,
            "value_loss": float(np.mean(value_loss_meter)) if value_loss_meter else 0.0,
            "entropy": float(np.mean(entropy_meter)) if entropy_meter else 0.0,
            "approx_kl": float(np.mean(approx_kl_meter)) if approx_kl_meter else 0.0,
            "clip_frac": float(np.mean(clip_frac_meter)) if clip_frac_meter else 0.0,
            "total_loss": float(np.mean(total_loss_meter)) if total_loss_meter else 0.0,
            "buffer_size": float(len(local_buffer)),
        }

    def update(
        self,
        buffer: IPPOBuffer,
        last_values: np.ndarray,
    ) -> Dict[str, Any]:
        buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        agent_stats: List[Dict[str, float]] = []
        for agent_idx in range(self.num_agents):
            local_stats = self._update_single_agent(
                agent_idx=agent_idx,
                local_buffer=buffer.get_agent_buffer(agent_idx),
            )
            agent_stats.append(local_stats)

        def _mean(key: str) -> float:
            if not agent_stats:
                return 0.0
            return float(np.mean([float(x.get(key, 0.0)) for x in agent_stats]))

        return {
            "policy_loss": _mean("policy_loss"),
            "value_loss": _mean("value_loss"),
            "entropy": _mean("entropy"),
            "approx_kl": _mean("approx_kl"),
            "clip_frac": _mean("clip_frac"),
            "total_loss": _mean("total_loss"),
            "buffer_size": float(sum(float(x.get("buffer_size", 0.0)) for x in agent_stats)),
            "num_independent_agents": int(self.num_agents),
            "independent_agent_stats": agent_stats,
        }

    def save(self, path: str) -> None:
        payload = {
            "agent_type": "ippo",
            "config": self.cfg.__dict__,
            "models": [model.state_dict() for model in self.models],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        model_states = payload["models"]

        if len(model_states) != self.num_agents:
            raise ValueError(
                f"Checkpoint num_agents={len(model_states)} does not match current IPPO num_agents={self.num_agents}"
            )

        ignored_keys: List[Dict[str, Any]] = []

        for agent_idx in range(self.num_agents):
            ckpt_state = model_states[agent_idx]
            model = self.models[agent_idx]

            if strict:
                model.load_state_dict(ckpt_state, strict=True)
                continue

            current_state = model.state_dict()
            filtered_state = {}
            ignored = []

            for k, v in ckpt_state.items():
                if k in current_state and current_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    ignored.append(k)

            model.load_state_dict(filtered_state, strict=False)
            ignored_keys.append(
                {
                    "agent_idx": agent_idx,
                    "ignored_keys": ignored,
                }
            )

        payload["ignored_keys"] = ignored_keys
        return payload