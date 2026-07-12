from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from configs.scenario_config import ScenarioConfig


@dataclass
class AblationSpec:
    method_name: str
    config_name: str
    method_label: str
    paper_name: str
    description: str
    category: str
    scenario_overrides: Dict[str, Any] = field(default_factory=dict)
    ppo_overrides: Dict[str, Any] = field(default_factory=dict)

    # 统一命名 / 输出协议
    display_name: Optional[str] = None
    checkpoint_dir_name: Optional[str] = None
    output_dir_name: Optional[str] = None
    compare_name: Optional[str] = None
    summary_name: Optional[str] = None
    plot_name: Optional[str] = None
    default_checkpoint_name: str = "best_model.pt"
    is_reserved_ablation: bool = False

    # 新增：训练骨架保留字段
    trainer_family: str = "ppo_shared"
    policy_family: str = "shared_actor_critic"

    @property
    def effective_display_name(self) -> str:
        return self.display_name or self.method_label

    @property
    def effective_checkpoint_dir_name(self) -> str:
        return self.checkpoint_dir_name or self.method_name

    @property
    def effective_output_dir_name(self) -> str:
        return self.output_dir_name or self.method_name

    @property
    def effective_compare_name(self) -> str:
        return self.compare_name or self.method_name

    @property
    def effective_summary_name(self) -> str:
        return self.summary_name or self.method_name

    @property
    def effective_plot_name(self) -> str:
        return self.plot_name or self.method_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "config_name": self.config_name,
            "method_label": self.method_label,
            "paper_name": self.paper_name,
            "description": self.description,
            "category": self.category,
            "scenario_overrides": dict(self.scenario_overrides),
            "ppo_overrides": dict(self.ppo_overrides),
            "display_name": self.effective_display_name,
            "checkpoint_dir_name": self.effective_checkpoint_dir_name,
            "output_dir_name": self.effective_output_dir_name,
            "compare_name": self.effective_compare_name,
            "summary_name": self.effective_summary_name,
            "plot_name": self.effective_plot_name,
            "default_checkpoint_name": self.default_checkpoint_name,
            "is_reserved_ablation": bool(self.is_reserved_ablation),
            "trainer_family": self.trainer_family,
            "policy_family": self.policy_family,
        }


DEFAULT_PPO_CONFIG: Dict[str, Any] = {
    "hidden_dim": 256,
    "num_hidden_layers": 2,
    "lr": 1e-4,
    "gamma": 0.995,
    "gae_lambda": 0.97,
    "clip_eps": 0.1,
    "ppo_epochs": 15,
    "mini_batch_size": 256,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,

    # Step-3: structured obs / neighbor aggregation
    "use_structured_obs_encoder": False,
    "use_neighbor_encoder": False,
    "neighbor_encoder_hidden_dim": 64,
    "neighbor_context_dim": 64,
    "neighbor_pooling_type": "mean_max",
}


ABLATION_REGISTRY: Dict[str, AblationSpec] = {
    "ppo_main": AblationSpec(
        method_name="ppo_main",
        config_name="ppo_main",
        method_label="PPO-Main",
        paper_name="PPO-Main",
        display_name="PPO-Main",
        description="基础 PPO 主方法入口，使用当前默认局部观测、奖励与 critic 结构。",
        category="main",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "ppo_main",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
        },
    ),
    "mcg_ppo": AblationSpec(
        method_name="mcg_ppo",
        config_name="mcg_ppo",
        method_label="MCG-PPO",
        paper_name="MCG-PPO",
        display_name="MCG-PPO",
        description="论文主方法入口；在策略网络与价值网络前端增加结构化观测编码与轻量邻域协同特征聚合模块。",
        category="main",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "mcg_ppo",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_mcg_reward": True,
            "use_enhanced_obs": True,
            "enhanced_obs_for_mcg_only": True,
            "use_user_summary_features": True,
            "use_neighbor_summary_features": True,
            "use_overlap_risk_features": True,
            "use_uncovered_guidance": True,
        },
        ppo_overrides={
            "use_structured_obs_encoder": True,
            "use_neighbor_encoder": True,
            "neighbor_encoder_hidden_dim": 64,
            "neighbor_context_dim": 64,
            "neighbor_pooling_type": "mean_max",
        },
    ),
    "mcg_ppo_sensing": AblationSpec(
        method_name="mcg_ppo_sensing",
        config_name="mcg_ppo_sensing",
        method_label="MCG-PPO-Sensing",
        paper_name="MCG-PPO-Sensing",
        display_name="MCG-PPO-Sensing",
        description="Distributed MCG-PPO with uncertainty, information-age, and repeated-sensing objectives.",
        category="main",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "mcg_ppo_sensing",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_mcg_reward": True,
            "use_trusted_sensing": True,
            "use_enhanced_obs": True,
            "enhanced_obs_for_mcg_only": True,
            "use_user_summary_features": True,
            "use_neighbor_summary_features": True,
            "use_overlap_risk_features": True,
            "use_uncovered_guidance": True,
        },
        ppo_overrides={
            "use_structured_obs_encoder": True,
            "use_neighbor_encoder": True,
            "neighbor_encoder_hidden_dim": 64,
            "neighbor_context_dim": 64,
            "neighbor_pooling_type": "mean_max",
        },
    ),
    "mcg_ppo_resource_cognition": AblationSpec(
        method_name="mcg_ppo_resource_cognition",
        config_name="mcg_ppo_resource_cognition",
        method_label="MCG-PPO-Resource-Cognition",
        paper_name="MCG-PPO-Resource-Cognition",
        display_name="MCG-PPO-Resource-Cognition",
        description="Independent local trusted-state cognition environment with explicit sensing actions.",
        category="main",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "mcg_ppo_resource_cognition",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_resource_cognition": True,
        },
        ppo_overrides={},
    ),
    "ippo": AblationSpec(
        method_name="ippo",
        config_name="ippo",
        method_label="IPPO",
        paper_name="IPPO",
        display_name="IPPO",
        description="IPPO 工程骨架入口；本轮先复用共享 PPO 训练壳子，后续可替换为真正的独立 PPO 多智能体实现。",
        category="main",
        trainer_family="ippo",
        policy_family="independent_ppo_placeholder",
        scenario_overrides={
            "method_name": "ippo",
            "rl_method_family": "ippo",
            "policy_architecture": "independent_ppo_placeholder",
        },
    ),
    "maddpg": AblationSpec(
        method_name="maddpg",
        config_name="maddpg",
        method_label="MADDPG",
        paper_name="MADDPG",
        display_name="MADDPG",
        description="MADDPG 工程骨架入口；本轮先保留独立方法身份、目录与训练路由，内部采用最小可运行占位实现。",
        category="main",
        trainer_family="maddpg",
        policy_family="maddpg_placeholder",
        scenario_overrides={
            "method_name": "maddpg",
            "rl_method_family": "maddpg",
            "policy_architecture": "maddpg_placeholder",
        },
    ),
    "mcg_ppo_no_graph": AblationSpec(
        method_name="mcg_ppo_no_graph",
        config_name="mcg_ppo_no_graph",
        method_label="MCG-PPO w/o Graph",
        paper_name="MCG-PPO w/o Graph",
        display_name="MCG-PPO w/o Graph",
        description="预留给后续消融：关闭结构化邻域协同相关分支，仅保留基础局部特征。当前已具备正式工程入口。",
        category="ablation",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "mcg_ppo_no_graph",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_mcg_reward": True,
            "use_enhanced_obs": False,
            "use_user_summary_features": False,
            "use_neighbor_summary_features": False,
            "use_overlap_risk_features": False,
            "use_local_neighborhood_summary": False,
            "use_neighbor_uav_obs": False,
        },
        ppo_overrides={
            "use_structured_obs_encoder": False,
            "use_neighbor_encoder": False,
        },
    ),
    "mcg_ppo_no_mc_reward": AblationSpec(
        method_name="mcg_ppo_no_mc_reward",
        config_name="mcg_ppo_no_mc_reward",
        method_label="MCG-PPO w/o MC Reward",
        paper_name="MCG-PPO w/o MC Reward",
        display_name="MCG-PPO w/o MC Reward",
        description="预留给后续消融：边际贡献奖励分支的正式方法入口。本轮先保留完整壳子与统一命名，便于后续最小代价接入。",
        category="ablation",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "mcg_ppo_no_mc_reward",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_mcg_reward": True,
            "use_enhanced_obs": True,
            "enhanced_obs_for_mcg_only": True,
            "use_user_summary_features": True,
            "use_neighbor_summary_features": True,
            "use_overlap_risk_features": True,
            "use_uncovered_guidance": True,
            "enable_marginal_contribution_reward": False,
            "reward_weight_marginal_contribution": 0.0,
        },
        ppo_overrides={
            "use_structured_obs_encoder": True,
            "use_neighbor_encoder": True,
            "neighbor_encoder_hidden_dim": 64,
            "neighbor_context_dim": 64,
            "neighbor_pooling_type": "mean_max",
        },
        is_reserved_ablation=True,
    ),
    "mcg_ppo_no_overlap_penalty": AblationSpec(
        method_name="mcg_ppo_no_overlap_penalty",
        config_name="mcg_ppo_no_overlap_penalty",
        method_label="MCG-PPO w/o Overlap Penalty",
        paper_name="MCG-PPO w/o Overlap Penalty",
        display_name="MCG-PPO w/o Overlap Penalty",
        description="预留给后续消融：重叠惩罚项的正式方法入口。本轮先保留工程位置与统一命名。",
        category="ablation",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "mcg_ppo_no_overlap_penalty",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_mcg_reward": True,
            "use_enhanced_obs": True,
            "enhanced_obs_for_mcg_only": True,
            "use_user_summary_features": True,
            "use_neighbor_summary_features": True,
            "use_overlap_risk_features": True,
            "use_uncovered_guidance": True,
            "enable_overlap_penalty": False,
            "reward_weight_overlap_penalty": 0.0,
        },
        ppo_overrides={
            "use_structured_obs_encoder": True,
            "use_neighbor_encoder": True,
            "neighbor_encoder_hidden_dim": 64,
            "neighbor_context_dim": 64,
            "neighbor_pooling_type": "mean_max",
        },
        is_reserved_ablation=True,
    ),
    "mcg_ppo_no_guidance": AblationSpec(
        method_name="mcg_ppo_no_guidance",
        config_name="mcg_ppo_no_guidance",
        method_label="MCG-PPO w/o Guidance",
        paper_name="MCG-PPO w/o Guidance",
        display_name="MCG-PPO w/o Guidance",
        description="关闭未覆盖用户引导特征，为后续论文消融做正式入口预留。",
        category="ablation",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "mcg_ppo_no_guidance",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_mcg_reward": True,
            "use_enhanced_obs": True,
            "enhanced_obs_for_mcg_only": True,
            "use_user_summary_features": True,
            "use_neighbor_summary_features": True,
            "use_overlap_risk_features": True,
            "use_uncovered_guidance": False,
            "reward_weight_uncovered_guidance": 0.0,
        },
        ppo_overrides={
            "use_structured_obs_encoder": True,
            "use_neighbor_encoder": True,
            "neighbor_encoder_hidden_dim": 64,
            "neighbor_context_dim": 64,
            "neighbor_pooling_type": "mean_max",
        },
    ),

    # legacy aliases
    "ppo_wo_local_summary": AblationSpec(
        method_name="ppo_wo_local_summary",
        config_name="ppo_wo_local_summary",
        method_label="PPO w/o Local Summary",
        paper_name="w/o Local Summary",
        display_name="PPO w/o Local Summary",
        description="兼容旧命名：移除局部邻域统计摘要，仅保留 self/users/neighbor_uav/guidance。",
        category="legacy_alias",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "ppo_wo_local_summary",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_local_neighborhood_summary": False,
        },
    ),
    "ppo_wo_guidance": AblationSpec(
        method_name="ppo_wo_guidance",
        config_name="ppo_wo_guidance",
        method_label="PPO w/o Guidance",
        paper_name="w/o Guidance",
        display_name="PPO w/o Guidance",
        description="兼容旧命名：移除局部未覆盖引导向量，仅保留其他观测块。",
        category="legacy_alias",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "ppo_wo_guidance",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_uncovered_guidance": False,
        },
    ),
    "ppo_wo_neighbor_uav": AblationSpec(
        method_name="ppo_wo_neighbor_uav",
        config_name="ppo_wo_neighbor_uav",
        method_label="PPO w/o Neighbor UAV",
        paper_name="w/o Neighbor UAV",
        display_name="PPO w/o Neighbor UAV",
        description="兼容旧命名：移除邻近 UAV 特征块，测试协同邻域信息的作用。",
        category="legacy_alias",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "ppo_wo_neighbor_uav",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "use_neighbor_uav_obs": False,
        },
    ),
    "ppo_reward_coverage_only": AblationSpec(
        method_name="ppo_reward_coverage_only",
        config_name="ppo_reward_coverage_only",
        method_label="PPO Coverage-Only Reward",
        paper_name="Coverage-Only Reward",
        display_name="PPO Coverage-Only Reward",
        description="兼容旧命名：仅保留覆盖收益，去掉移动代价与重叠惩罚。",
        category="legacy_alias",
        trainer_family="ppo_shared",
        policy_family="shared_actor_critic",
        scenario_overrides={
            "method_name": "ppo_reward_coverage_only",
            "rl_method_family": "ppo_shared",
            "policy_architecture": "shared_actor_critic",
            "w_step_move_cost": 0.0,
            "w_overlap_penalty": 0.0,
        },
    ),
}


_CONFIG_NAME_TO_METHOD: Dict[str, str] = {
    spec.config_name: spec.method_name for spec in ABLATION_REGISTRY.values()
}


def list_ablation_specs() -> List[AblationSpec]:
    return [ABLATION_REGISTRY[k] for k in ABLATION_REGISTRY.keys()]


def list_registered_method_names(include_legacy: bool = True) -> List[str]:
    names: List[str] = []
    for spec in list_ablation_specs():
        if not include_legacy and spec.category == "legacy_alias":
            continue
        names.append(spec.method_name)
    return names


def get_ablation_spec(method_name: str) -> AblationSpec:
    if method_name not in ABLATION_REGISTRY:
        raise KeyError(f"Unknown ablation method_name: {method_name}")
    return ABLATION_REGISTRY[method_name]


def get_method_display_name(method_name: str) -> str:
    return get_ablation_spec(method_name).effective_display_name


def get_compare_label_map(method_names: Optional[List[str]] = None) -> Dict[str, str]:
    if method_names is None:
        method_names = list_registered_method_names(include_legacy=True)
    out: Dict[str, str] = {}
    for method_name in method_names:
        if method_name in ABLATION_REGISTRY:
            out[method_name] = get_ablation_spec(method_name).effective_display_name
    return out


def resolve_method_name(
    method_name: Optional[str] = None,
    config_name: Optional[str] = None,
    default_method_name: str = "ppo_main",
) -> str:
    if method_name is not None and str(method_name).strip() != "":
        candidate = str(method_name).strip()
        if candidate not in ABLATION_REGISTRY:
            raise KeyError(f"Unknown method_name: {candidate}")
        return candidate

    if config_name is not None and str(config_name).strip() != "":
        candidate = str(config_name).strip()
        if candidate not in _CONFIG_NAME_TO_METHOD:
            raise KeyError(f"Unknown config_name: {candidate}")
        return _CONFIG_NAME_TO_METHOD[candidate]

    if default_method_name not in ABLATION_REGISTRY:
        raise KeyError(f"Unknown default_method_name: {default_method_name}")
    return default_method_name


def build_base_scenario_config(**overrides: Any) -> ScenarioConfig:
    cfg = ScenarioConfig(
        num_users=20,
        user_distribution_mode="mixed",
        num_user_clusters=3,
        clustered_user_ratio=0.80,
        cluster_radius=220.0,
        edge_avoidance_ratio=0.15,
        edge_soft_limit_ratio=0.20,
        cluster_center_min_radius_ratio=0.18,
        cluster_center_max_radius_ratio=0.72,
        max_candidate_uavs=5,
        initially_active_uavs=5,
        uav_init_height=100.0,
        use_simplified_qos=True,
        max_obs_users=20,
        max_obs_uavs=4,
        seed=42,
        allow_activation_action=False,
        activation_cost_once=0.0,
        w_new_activation_penalty=0.0,
        w_active_count_penalty=0.0,
        active_alive_cost=0.0,
        inactive_idle_cost=0.0,

        use_enhanced_obs=False,
        enhanced_obs_for_mcg_only=True,
        use_user_summary_features=False,
        use_neighbor_summary_features=False,
        use_overlap_risk_features=False,

        method_name="ppo_main",
        rl_method_family="ppo_shared",
        policy_architecture="shared_actor_critic",
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"ScenarioConfig has no field '{key}'")
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


def apply_ablation_to_scenario(base_cfg: ScenarioConfig, method_name: str) -> ScenarioConfig:
    spec = get_ablation_spec(method_name)
    cfg = replace(base_cfg)
    for key, value in spec.scenario_overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"ScenarioConfig has no field '{key}' for method {method_name}")
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


def build_base_ppo_config_kwargs(**overrides: Any) -> Dict[str, Any]:
    cfg = dict(DEFAULT_PPO_CONFIG)
    cfg.update(overrides)
    return cfg


def build_method_ppo_config_kwargs(method_name: str, **extra_overrides: Any) -> Dict[str, Any]:
    spec = get_ablation_spec(method_name)
    cfg = build_base_ppo_config_kwargs(**spec.ppo_overrides)
    cfg.update(extra_overrides)
    return cfg


def resolve_method_checkpoint_path(
    method_name: str,
    checkpoint_name: Optional[str] = None,
    train_root: str = os.path.join("results", "train"),
) -> str:
    spec = get_ablation_spec(method_name)
    filename = checkpoint_name or spec.default_checkpoint_name
    return os.path.join(train_root, spec.effective_checkpoint_dir_name, "checkpoints", filename)
