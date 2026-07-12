from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int = 2,
    activation: nn.Module = nn.Tanh,
) -> nn.Sequential:
    layers = []
    last_dim = input_dim

    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class StructuredObsSliceSpec:
    """
    按当前环境中 local_obs 的固定布局做切片。

    base obs:
        self_feat                                  = 11
        user_slots                                = max_obs_users * 5
        neighbor_slots                            = max_obs_uavs * 5
        local_neighborhood_summary                = num_direction_sectors + num_radial_bins + 4
        uncovered_guidance                        = 8

    mcg extra obs:
        mcg_self_state_features                   = 7
        mcg_user_summary_features                 = 8
        mcg_neighbor_summary_features             = 6
        mcg_overlap_risk_features                 = 4
    """

    local_obs_dim: int
    max_obs_users: int = 20
    max_obs_uavs: int = 2
    num_direction_sectors: int = 4
    num_radial_bins: int = 3

    self_feat_dim: int = 11
    user_slot_dim: int = 5
    neighbor_slot_dim: int = 5
    guidance_dim: int = 8

    mcg_self_feat_dim: int = 7
    mcg_user_summary_dim: int = 8
    mcg_neighbor_summary_dim: int = 6
    mcg_overlap_risk_dim: int = 4

    @property
    def local_summary_dim(self) -> int:
        return int(self.num_direction_sectors + self.num_radial_bins + 4)

    @property
    def base_obs_dim(self) -> int:
        return int(
            self.self_feat_dim
            + self.max_obs_users * self.user_slot_dim
            + self.max_obs_uavs * self.neighbor_slot_dim
            + self.local_summary_dim
            + self.guidance_dim
        )

    @property
    def mcg_extra_obs_dim(self) -> int:
        return int(
            self.mcg_self_feat_dim
            + self.mcg_user_summary_dim
            + self.mcg_neighbor_summary_dim
            + self.mcg_overlap_risk_dim
        )

    @property
    def expected_mcg_obs_dim(self) -> int:
        return int(self.base_obs_dim + self.mcg_extra_obs_dim)

    def _slice_or_zero(self, local_obs: torch.Tensor, start: int, length: int) -> torch.Tensor:
        end = start + length
        if start >= local_obs.shape[-1]:
            return local_obs.new_zeros((local_obs.shape[0], length))

        if end <= local_obs.shape[-1]:
            return local_obs[:, start:end]

        available = local_obs[:, start: local_obs.shape[-1]]
        missing = length - available.shape[-1]
        pad = local_obs.new_zeros((local_obs.shape[0], missing))
        return torch.cat([available, pad], dim=-1)

    def split(self, local_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if local_obs.dim() != 2:
            raise ValueError(f"Expected local_obs with shape [B, D], got {tuple(local_obs.shape)}")

        if local_obs.shape[-1] < self.base_obs_dim:
            raise ValueError(
                f"local_obs_dim={local_obs.shape[-1]} is smaller than required base_obs_dim={self.base_obs_dim}."
            )

        offset = 0

        self_feat = self._slice_or_zero(local_obs, offset, self.self_feat_dim)
        offset += self.self_feat_dim

        user_flat = self._slice_or_zero(local_obs, offset, self.max_obs_users * self.user_slot_dim)
        user_slots = user_flat.view(local_obs.shape[0], self.max_obs_users, self.user_slot_dim)
        offset += self.max_obs_users * self.user_slot_dim

        neighbor_flat = self._slice_or_zero(local_obs, offset, self.max_obs_uavs * self.neighbor_slot_dim)
        neighbor_slots = neighbor_flat.view(local_obs.shape[0], self.max_obs_uavs, self.neighbor_slot_dim)
        offset += self.max_obs_uavs * self.neighbor_slot_dim

        local_summary = self._slice_or_zero(local_obs, offset, self.local_summary_dim)
        offset += self.local_summary_dim

        guidance = self._slice_or_zero(local_obs, offset, self.guidance_dim)
        offset += self.guidance_dim

        mcg_self = self._slice_or_zero(local_obs, offset, self.mcg_self_feat_dim)
        offset += self.mcg_self_feat_dim

        mcg_user_summary = self._slice_or_zero(local_obs, offset, self.mcg_user_summary_dim)
        offset += self.mcg_user_summary_dim

        mcg_neighbor_summary = self._slice_or_zero(local_obs, offset, self.mcg_neighbor_summary_dim)
        offset += self.mcg_neighbor_summary_dim

        mcg_overlap_risk = self._slice_or_zero(local_obs, offset, self.mcg_overlap_risk_dim)

        user_mask = (user_slots.abs().sum(dim=-1) > 1e-8).float()
        neighbor_mask = (neighbor_slots.abs().sum(dim=-1) > 1e-8).float()

        return {
            "self_feat": self_feat,
            "user_slots": user_slots,
            "user_mask": user_mask,
            "neighbor_slots": neighbor_slots,
            "neighbor_mask": neighbor_mask,
            "local_summary": local_summary,
            "guidance": guidance,
            "mcg_self": mcg_self,
            "mcg_user_summary": mcg_user_summary,
            "mcg_neighbor_summary": mcg_neighbor_summary,
            "mcg_overlap_risk": mcg_overlap_risk,
        }


class RowSetEncoder(nn.Module):
    """
    对一组定长 row features 先逐行共享编码，再做 masked mean pooling。
    用于局部用户摘要编码。
    """

    def __init__(self, row_dim: int, row_hidden_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.row_encoder = build_mlp(
            input_dim=row_dim,
            hidden_dim=row_hidden_dim,
            output_dim=out_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )

    def masked_mean(self, row_embed: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        weights = row_mask.unsqueeze(-1)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        pooled = (row_embed * weights).sum(dim=1) / denom
        no_valid = (row_mask.sum(dim=1, keepdim=True) <= 0.0).float()
        return pooled * (1.0 - no_valid)

    def forward(self, rows: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        if rows.shape[1] == 0:
            return rows.new_zeros((rows.shape[0], self.out_dim))
        row_embed = self.row_encoder(rows)
        return self.masked_mean(row_embed, row_mask)


class SelfFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserSummaryEncoder(nn.Module):
    """
    局部用户态势编码：
    - 用户 row set pooling
    - mcg user summary features
    - 最后融合
    """

    def __init__(self, user_slot_dim: int, user_summary_dim: int, hidden_dim: int):
        super().__init__()
        self.user_set_encoder = RowSetEncoder(
            row_dim=user_slot_dim,
            row_hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.summary_encoder = build_mlp(
            input_dim=user_summary_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )
        self.fuse = build_mlp(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )

    def forward(
        self,
        user_slots: torch.Tensor,
        user_mask: torch.Tensor,
        user_summary: torch.Tensor,
    ) -> torch.Tensor:
        slot_context = self.user_set_encoder(user_slots, user_mask)
        summary_context = self.summary_encoder(user_summary)
        return self.fuse(torch.cat([slot_context, summary_context], dim=-1))


class NeighborFeatureAggregator(nn.Module):
    """
    轻量邻域协同聚合器：
    1) 每个邻居 row feature 先经过共享小 MLP 编码
    2) 再做 masked pooling（mean / mean_max / attention）
    3) 最后与 neighbor summary features 融合，得到固定维度 neighbor context
    """

    def __init__(
        self,
        neighbor_slot_dim: int,
        neighbor_summary_dim: int,
        self_context_dim: int,
        neighbor_encoder_hidden_dim: int,
        neighbor_context_dim: int,
        pooling_type: str = "mean_max",
    ):
        super().__init__()
        self.pooling_type = str(pooling_type).lower()
        self.neighbor_encoder_hidden_dim = int(neighbor_encoder_hidden_dim)
        self.neighbor_context_dim = int(neighbor_context_dim)

        self.neighbor_encoder = build_mlp(
            input_dim=neighbor_slot_dim,
            hidden_dim=self.neighbor_encoder_hidden_dim,
            output_dim=self.neighbor_encoder_hidden_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )
        self.summary_encoder = build_mlp(
            input_dim=neighbor_summary_dim,
            hidden_dim=self.neighbor_context_dim,
            output_dim=self.neighbor_context_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )

        if self.pooling_type == "attention":
            self.query_proj = nn.Linear(self_context_dim, self.neighbor_encoder_hidden_dim)
            self.key_proj = nn.Linear(self.neighbor_encoder_hidden_dim, self.neighbor_encoder_hidden_dim)
            self.value_proj = nn.Linear(self.neighbor_encoder_hidden_dim, self.neighbor_encoder_hidden_dim)
            self.pooled_dim = self.neighbor_encoder_hidden_dim
        elif self.pooling_type == "mean":
            self.pooled_dim = self.neighbor_encoder_hidden_dim
        elif self.pooling_type == "mean_max":
            self.pooled_dim = 2 * self.neighbor_encoder_hidden_dim
        else:
            raise ValueError(f"Unsupported neighbor_pooling_type: {pooling_type}")

        self.out_proj = build_mlp(
            input_dim=self.pooled_dim + self.neighbor_context_dim,
            hidden_dim=max(self.neighbor_context_dim, self.neighbor_encoder_hidden_dim),
            output_dim=self.neighbor_context_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return x.new_zeros((x.shape[0], x.shape[-1]))
        weights = mask.unsqueeze(-1)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        pooled = (x * weights).sum(dim=1) / denom
        no_valid = (mask.sum(dim=1, keepdim=True) <= 0.0).float()
        return pooled * (1.0 - no_valid)

    def _masked_max(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return x.new_zeros((x.shape[0], x.shape[-1]))
        neg_inf = torch.full_like(x, -1e9)
        masked_x = torch.where(mask.unsqueeze(-1) > 0.5, x, neg_inf)
        pooled = masked_x.max(dim=1).values
        no_valid = (mask.sum(dim=1, keepdim=True) <= 0.0).float()
        return pooled * (1.0 - no_valid)

    def _attention_pool(self, self_context: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return x.new_zeros((x.shape[0], x.shape[-1]))

        q = self.query_proj(self_context).unsqueeze(1)  # [B,1,H]
        k = self.key_proj(x)                            # [B,N,H]
        v = self.value_proj(x)                          # [B,N,H]

        scores = (q * k).sum(dim=-1) / (float(self.neighbor_encoder_hidden_dim) ** 0.5)
        scores = scores.masked_fill(mask <= 0.5, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = attn * mask

        denom = torch.clamp(attn.sum(dim=-1, keepdim=True), min=1.0)
        attn = attn / denom

        pooled = torch.bmm(attn.unsqueeze(1), v).squeeze(1)
        no_valid = (mask.sum(dim=1, keepdim=True) <= 0.0).float()
        return pooled * (1.0 - no_valid)

    def forward(
        self,
        self_context: torch.Tensor,
        neighbor_slots: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_summary: torch.Tensor,
    ) -> torch.Tensor:
        if neighbor_slots.shape[1] == 0:
            if self.pooling_type == "mean_max":
                pooled = neighbor_slots.new_zeros((neighbor_slots.shape[0], self.pooled_dim))
            else:
                pooled = neighbor_slots.new_zeros((neighbor_slots.shape[0], self.neighbor_encoder_hidden_dim))
        else:
            row_embed = self.neighbor_encoder(neighbor_slots)

            if self.pooling_type == "mean":
                pooled = self._masked_mean(row_embed, neighbor_mask)
            elif self.pooling_type == "mean_max":
                pooled_mean = self._masked_mean(row_embed, neighbor_mask)
                pooled_max = self._masked_max(row_embed, neighbor_mask)
                pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
            else:
                pooled = self._attention_pool(self_context, row_embed, neighbor_mask)

        summary_context = self.summary_encoder(neighbor_summary)
        return self.out_proj(torch.cat([pooled, summary_context], dim=-1))


class GuidanceEncoder(nn.Module):
    """
    局部引导 / 风险摘要编码：
    - local_neighborhood_summary
    - uncovered_guidance
    - mcg_overlap_risk
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointPolicyBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_hidden_layers=1,
            activation=nn.Tanh,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuredObsEncoder(nn.Module):
    """
    结构化观测编码器：
    - SelfFeatureEncoder
    - UserSummaryEncoder
    - NeighborFeatureAggregator / fallback neighbor encoder
    - GuidanceEncoder
    - JointPolicyBackbone
    """

    def __init__(
        self,
        slice_spec: StructuredObsSliceSpec,
        hidden_dim: int,
        neighbor_encoder_hidden_dim: int,
        neighbor_context_dim: int,
        neighbor_pooling_type: str,
        use_neighbor_encoder: bool,
    ):
        super().__init__()
        self.slice_spec = slice_spec
        self.hidden_dim = int(hidden_dim)
        self.use_neighbor_encoder = bool(use_neighbor_encoder)
        self.branch_dim = int(hidden_dim)

        self.self_encoder = SelfFeatureEncoder(
            input_dim=self.slice_spec.self_feat_dim + self.slice_spec.mcg_self_feat_dim,
            hidden_dim=self.branch_dim,
        )
        self.user_encoder = UserSummaryEncoder(
            user_slot_dim=self.slice_spec.user_slot_dim,
            user_summary_dim=self.slice_spec.mcg_user_summary_dim,
            hidden_dim=self.branch_dim,
        )
        self.guidance_encoder = GuidanceEncoder(
            input_dim=self.slice_spec.local_summary_dim + self.slice_spec.guidance_dim + self.slice_spec.mcg_overlap_risk_dim,
            hidden_dim=self.branch_dim,
        )

        if self.use_neighbor_encoder:
            self.neighbor_encoder = NeighborFeatureAggregator(
                neighbor_slot_dim=self.slice_spec.neighbor_slot_dim,
                neighbor_summary_dim=self.slice_spec.mcg_neighbor_summary_dim,
                self_context_dim=self.branch_dim,
                neighbor_encoder_hidden_dim=neighbor_encoder_hidden_dim,
                neighbor_context_dim=neighbor_context_dim,
                pooling_type=neighbor_pooling_type,
            )
        else:
            fallback_input_dim = (
                self.slice_spec.max_obs_uavs * self.slice_spec.neighbor_slot_dim
                + self.slice_spec.mcg_neighbor_summary_dim
            )
            self.neighbor_encoder = build_mlp(
                input_dim=fallback_input_dim,
                hidden_dim=max(self.branch_dim, neighbor_context_dim),
                output_dim=neighbor_context_dim,
                num_hidden_layers=1,
                activation=nn.Tanh,
            )

        self.joint_backbone = JointPolicyBackbone(
            input_dim=self.branch_dim + self.branch_dim + neighbor_context_dim + self.branch_dim,
            hidden_dim=self.hidden_dim,
        )

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        parts = self.slice_spec.split(local_obs)

        self_input = torch.cat([parts["self_feat"], parts["mcg_self"]], dim=-1)
        self_context = self.self_encoder(self_input)

        user_context = self.user_encoder(
            user_slots=parts["user_slots"],
            user_mask=parts["user_mask"],
            user_summary=parts["mcg_user_summary"],
        )

        if self.use_neighbor_encoder:
            neighbor_context = self.neighbor_encoder(
                self_context=self_context,
                neighbor_slots=parts["neighbor_slots"],
                neighbor_mask=parts["neighbor_mask"],
                neighbor_summary=parts["mcg_neighbor_summary"],
            )
        else:
            neighbor_fallback = torch.cat(
                [
                    parts["neighbor_slots"].reshape(local_obs.shape[0], -1),
                    parts["mcg_neighbor_summary"],
                ],
                dim=-1,
            )
            neighbor_context = self.neighbor_encoder(neighbor_fallback)

        guidance_input = torch.cat(
            [
                parts["local_summary"],
                parts["guidance"],
                parts["mcg_overlap_risk"],
            ],
            dim=-1,
        )
        guidance_context = self.guidance_encoder(guidance_input)

        joint = torch.cat(
            [
                self_context,
                user_context,
                neighbor_context,
                guidance_context,
            ],
            dim=-1,
        )
        return self.joint_backbone(joint)


class MaskedActorNet(nn.Module):
    """
    Actor:
    input = local observation
    output = masked action logits
    """

    def __init__(
        self,
        local_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=local_obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_hidden_layers=num_hidden_layers,
            activation=nn.Tanh,
        )

    def forward(self, local_obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(local_obs)

        if action_mask is not None:
            huge_neg = torch.full_like(logits, -1e9)
            logits = torch.where(action_mask > 0.5, logits, huge_neg)

        return logits

    def get_dist(self, local_obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Categorical:
        logits = self.forward(local_obs, action_mask)
        return Categorical(logits=logits)


class StructuredMaskedActorNet(nn.Module):
    """
    MCG-PPO actor:
    先做结构化观测编码，再输出动作 logits。
    """

    def __init__(
        self,
        local_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_obs_users: int = 20,
        max_obs_uavs: int = 2,
        num_direction_sectors: int = 4,
        num_radial_bins: int = 3,
        neighbor_encoder_hidden_dim: int = 64,
        neighbor_context_dim: int = 64,
        neighbor_pooling_type: str = "mean_max",
        use_neighbor_encoder: bool = True,
    ):
        super().__init__()
        self.slice_spec = StructuredObsSliceSpec(
            local_obs_dim=local_obs_dim,
            max_obs_users=max_obs_users,
            max_obs_uavs=max_obs_uavs,
            num_direction_sectors=num_direction_sectors,
            num_radial_bins=num_radial_bins,
        )
        self.structured_encoder = StructuredObsEncoder(
            slice_spec=self.slice_spec,
            hidden_dim=hidden_dim,
            neighbor_encoder_hidden_dim=neighbor_encoder_hidden_dim,
            neighbor_context_dim=neighbor_context_dim,
            neighbor_pooling_type=neighbor_pooling_type,
            use_neighbor_encoder=use_neighbor_encoder,
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, local_obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        joint_feat = self.structured_encoder(local_obs)
        logits = self.policy_head(joint_feat)

        if action_mask is not None:
            huge_neg = torch.full_like(logits, -1e9)
            logits = torch.where(action_mask > 0.5, logits, huge_neg)

        return logits

    def get_dist(self, local_obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Categorical:
        logits = self.forward(local_obs, action_mask)
        return Categorical(logits=logits)


class LocalCriticNet(nn.Module):
    """
    Local critic:
    input = local observation
    output = scalar value
    """

    def __init__(
        self,
        local_obs_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=local_obs_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            activation=nn.Tanh,
        )

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        value = self.net(local_obs)
        return value.squeeze(-1)


class StructuredLocalCriticNet(nn.Module):
    """
    MCG-PPO critic:
    与 actor 一样先做结构化观测编码，但仍然只依赖 local obs，
    不引入 centralized critic。
    """

    def __init__(
        self,
        local_obs_dim: int,
        hidden_dim: int = 256,
        max_obs_users: int = 20,
        max_obs_uavs: int = 2,
        num_direction_sectors: int = 4,
        num_radial_bins: int = 3,
        neighbor_encoder_hidden_dim: int = 64,
        neighbor_context_dim: int = 64,
        neighbor_pooling_type: str = "mean_max",
        use_neighbor_encoder: bool = True,
    ):
        super().__init__()
        self.slice_spec = StructuredObsSliceSpec(
            local_obs_dim=local_obs_dim,
            max_obs_users=max_obs_users,
            max_obs_uavs=max_obs_uavs,
            num_direction_sectors=num_direction_sectors,
            num_radial_bins=num_radial_bins,
        )
        self.structured_encoder = StructuredObsEncoder(
            slice_spec=self.slice_spec,
            hidden_dim=hidden_dim,
            neighbor_encoder_hidden_dim=neighbor_encoder_hidden_dim,
            neighbor_context_dim=neighbor_context_dim,
            neighbor_pooling_type=neighbor_pooling_type,
            use_neighbor_encoder=use_neighbor_encoder,
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        joint_feat = self.structured_encoder(local_obs)
        return self.value_head(joint_feat).squeeze(-1)


@dataclass
class ActorCriticOutput:
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    entropy: torch.Tensor


class LocalActorCritic(nn.Module):
    """
    参数共享的局部 actor-critic：
    - ppo_main: 保持原始 MLP actor/critic
    - mcg_ppo: 使用结构化观测编码 + 邻域协同聚合
    """

    def __init__(
        self,
        local_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        use_structured_obs_encoder: bool = False,
        use_neighbor_encoder: bool = False,
        max_obs_users: int = 20,
        max_obs_uavs: int = 2,
        num_direction_sectors: int = 4,
        num_radial_bins: int = 3,
        neighbor_encoder_hidden_dim: int = 64,
        neighbor_context_dim: int = 64,
        neighbor_pooling_type: str = "mean_max",
    ):
        super().__init__()
        self.local_obs_dim = int(local_obs_dim)
        self.action_dim = int(action_dim)
        self.use_structured_obs_encoder = bool(use_structured_obs_encoder)

        if self.use_structured_obs_encoder:
            self.actor = StructuredMaskedActorNet(
                local_obs_dim=local_obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                max_obs_users=max_obs_users,
                max_obs_uavs=max_obs_uavs,
                num_direction_sectors=num_direction_sectors,
                num_radial_bins=num_radial_bins,
                neighbor_encoder_hidden_dim=neighbor_encoder_hidden_dim,
                neighbor_context_dim=neighbor_context_dim,
                neighbor_pooling_type=neighbor_pooling_type,
                use_neighbor_encoder=use_neighbor_encoder,
            )
            self.critic = StructuredLocalCriticNet(
                local_obs_dim=local_obs_dim,
                hidden_dim=hidden_dim,
                max_obs_users=max_obs_users,
                max_obs_uavs=max_obs_uavs,
                num_direction_sectors=num_direction_sectors,
                num_radial_bins=num_radial_bins,
                neighbor_encoder_hidden_dim=neighbor_encoder_hidden_dim,
                neighbor_context_dim=neighbor_context_dim,
                neighbor_pooling_type=neighbor_pooling_type,
                use_neighbor_encoder=use_neighbor_encoder,
            )
        else:
            self.actor = MaskedActorNet(
                local_obs_dim=local_obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            self.critic = LocalCriticNet(
                local_obs_dim=local_obs_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )

    def act(
        self,
        local_obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> ActorCriticOutput:
        dist = self.actor.get_dist(local_obs, action_mask)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(local_obs)

        return ActorCriticOutput(
            actions=actions,
            log_probs=log_probs,
            values=values,
            entropy=entropy,
        )

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ):
        dist = self.actor.get_dist(local_obs, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(local_obs)
        return log_probs, entropy, values

    def get_values(self, local_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(local_obs)