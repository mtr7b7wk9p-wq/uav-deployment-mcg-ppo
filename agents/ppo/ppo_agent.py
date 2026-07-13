from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from agents.ppo.buffer import PPOBuffer
from agents.ppo.models import LocalActorCritic


@dataclass
class PPOConfig:
    local_obs_dim: int
    action_dim: int

    # 为兼容旧 checkpoint 配置字典，保留但不再使用
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

    # -------------------------
    # Step-3: structured obs / neighbor aggregation
    # ppo_main 默认关闭
    # mcg_ppo 通过 ablation_config 打开
    # -------------------------
    use_structured_obs_encoder: bool = False
    use_resource_cognition_encoder: bool = False
    use_neighbor_encoder: bool = False

    neighbor_encoder_hidden_dim: int = 64
    neighbor_context_dim: int = 64
    neighbor_pooling_type: str = "mean_max"

    # -------------------------
    # local_obs 结构配置
    # 必须与 env 中的 local_obs 组织方式保持一致
    # -------------------------
    max_obs_users: int = 20
    max_obs_uavs: int = 2
    num_direction_sectors: int = 4
    num_radial_bins: int = 3
    resource_num_task_slots: int = 8
    resource_num_message_slots: int = 4
    resource_context_dim: int = 64


class SharedPPOAgent:
    """
    参数共享的局部 actor-critic PPO。
    actor 和 critic 都只依赖 local_obs，不再依赖 global_state。

    - ppo_main: 使用原始 MLP actor/critic
    - mcg_ppo: 使用结构化观测编码 + 邻域协同聚合
    """

    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.model = LocalActorCritic(
            local_obs_dim=config.local_obs_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            use_structured_obs_encoder=config.use_structured_obs_encoder,
            use_resource_cognition_encoder=config.use_resource_cognition_encoder,
            use_neighbor_encoder=config.use_neighbor_encoder,
            max_obs_users=config.max_obs_users,
            max_obs_uavs=config.max_obs_uavs,
            num_direction_sectors=config.num_direction_sectors,
            num_radial_bins=config.num_radial_bins,
            neighbor_encoder_hidden_dim=config.neighbor_encoder_hidden_dim,
            neighbor_context_dim=config.neighbor_context_dim,
            neighbor_pooling_type=config.neighbor_pooling_type,
            resource_num_task_slots=config.resource_num_task_slots,
            resource_num_message_slots=config.resource_num_message_slots,
            resource_context_dim=config.resource_context_dim,
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=config.lr)

    @torch.no_grad()
    def act(
        self,
        local_obs_batch: np.ndarray,      # [N, local_obs_dim]
        action_mask_batch: np.ndarray,    # [N, action_dim]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_obs_t = torch.tensor(local_obs_batch, dtype=torch.float32, device=self.device)
        action_mask_t = torch.tensor(action_mask_batch, dtype=torch.float32, device=self.device)

        out = self.model.act(
            local_obs=local_obs_t,
            action_mask=action_mask_t,
        )

        return (
            out.actions.cpu().numpy(),
            out.log_probs.cpu().numpy(),
            out.values.cpu().numpy(),
        )

    @torch.no_grad()
    def greedy_act(
        self,
        local_obs_batch: np.ndarray,      # [N, local_obs_dim]
        action_mask_batch: np.ndarray,    # [N, action_dim]
    ) -> np.ndarray:
        local_obs_t = torch.tensor(local_obs_batch, dtype=torch.float32, device=self.device)
        action_mask_t = torch.tensor(action_mask_batch, dtype=torch.float32, device=self.device)

        logits = self.model.actor(local_obs_t, action_mask_t)
        actions = torch.argmax(logits, dim=-1)
        return actions.cpu().numpy()

    @torch.no_grad()
    def get_values(self, local_obs_batch: np.ndarray) -> np.ndarray:
        local_obs_t = torch.tensor(local_obs_batch, dtype=torch.float32, device=self.device)
        values_t = self.model.get_values(local_obs_t)
        return values_t.cpu().numpy()

    def update(
        self,
        buffer: PPOBuffer,
        last_values: np.ndarray,
        num_agents: int,
    ) -> Dict[str, float]:
        buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            num_agents=num_agents,
        )

        policy_loss_meter = []
        value_loss_meter = []
        entropy_meter = []
        approx_kl_meter = []
        clip_frac_meter = []
        total_loss_meter = []

        for _ in range(self.cfg.ppo_epochs):
            for batch in buffer.iterate_minibatches(
                device=self.device,
                mini_batch_size=self.cfg.mini_batch_size,
                shuffle=True,
            ):
                log_probs, entropy, values = self.model.evaluate_actions(
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

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )

                policy_loss_meter.append(policy_loss.item())
                value_loss_meter.append(value_loss.item())
                entropy_meter.append(entropy_mean.item())
                approx_kl_meter.append(approx_kl)
                clip_frac_meter.append(clip_frac)
                total_loss_meter.append(total_loss.item())

        stats = {
            "policy_loss": float(np.mean(policy_loss_meter)) if policy_loss_meter else 0.0,
            "value_loss": float(np.mean(value_loss_meter)) if value_loss_meter else 0.0,
            "entropy": float(np.mean(entropy_meter)) if entropy_meter else 0.0,
            "approx_kl": float(np.mean(approx_kl_meter)) if approx_kl_meter else 0.0,
            "clip_frac": float(np.mean(clip_frac_meter)) if clip_frac_meter else 0.0,
            "total_loss": float(np.mean(total_loss_meter)) if total_loss_meter else 0.0,
            "buffer_size": float(len(buffer)),
        }
        return stats

    def save(self, path: str) -> None:
        payload = {
            "model": self.model.state_dict(),
            "config": self.cfg.__dict__,
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """
        strict=True:
            严格加载，要求 checkpoint 与当前模型结构完全一致

        strict=False:
            允许部分加载。对旧 checkpoint 或结构变化后的 checkpoint，
            自动忽略 shape 不匹配的参数。
        """
        payload = torch.load(path, map_location=self.device)
        ckpt_state = payload["model"]

        if strict:
            self.model.load_state_dict(ckpt_state, strict=True)
            return payload

        current_state = self.model.state_dict()
        filtered_state = {}
        ignored_keys = []

        for k, v in ckpt_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                ignored_keys.append(k)

        self.model.load_state_dict(filtered_state, strict=False)
        payload["ignored_keys"] = ignored_keys
        return payload
