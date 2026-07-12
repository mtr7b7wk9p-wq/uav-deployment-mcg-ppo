from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from configs.ablation_config import get_ablation_spec


@dataclass(frozen=True)
class MethodMeta:
    method_name: str
    display_name: str
    category: str
    method_type: str
    config_name: str
    default_compare_enabled: bool = True
    description: str = ""
    legacy_target_method: Optional[str] = None
    checkpoint_name: Optional[str] = None
    checkpoint_dir_name: Optional[str] = None
    output_dir_name: Optional[str] = None
    trainer_family: Optional[str] = None
    policy_family: Optional[str] = None

    @property
    def is_baseline(self) -> bool:
        return self.method_type == "baseline"

    @property
    def is_rl_family(self) -> bool:
        return self.method_type in {"rl", "ablation", "legacy"}

    @property
    def is_main_rl(self) -> bool:
        return self.category == "main_rl"

    @property
    def is_ablation(self) -> bool:
        return self.category == "ablation"

    @property
    def is_legacy_alias(self) -> bool:
        return self.category == "legacy_alias"

    def to_dict(self) -> Dict[str, object]:
        return {
            "method_name": self.method_name,
            "display_name": self.display_name,
            "category": self.category,
            "method_type": self.method_type,
            "config_name": self.config_name,
            "default_compare_enabled": bool(self.default_compare_enabled),
            "description": self.description,
            "legacy_target_method": self.legacy_target_method,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_dir_name": self.checkpoint_dir_name,
            "output_dir_name": self.output_dir_name,
            "trainer_family": self.trainer_family,
            "policy_family": self.policy_family,
        }


BASELINE_METHODS: List[str] = [
    "random_masked",
    "greedy_local",
    "constrained_kmeans",
]

MAIN_RL_METHODS: List[str] = [
    "ppo_main",
    "mcg_ppo",
    "mcg_ppo_sensing",
    "ippo",
    "maddpg",
]

ABLATION_METHODS: List[str] = [
    "mcg_ppo_no_graph",
    "mcg_ppo_no_mc_reward",
    "mcg_ppo_no_overlap_penalty",
    "mcg_ppo_no_guidance",
]

LEGACY_ALIAS_METHODS: List[str] = [
    "ppo_wo_local_summary",
    "ppo_wo_guidance",
    "ppo_wo_neighbor_uav",
    "ppo_reward_coverage_only",
]

DEFAULT_COMPARE_METHODS: List[str] = [
    "random_masked",
    "greedy_local",
    "constrained_kmeans",
    "ppo_main",
    "mcg_ppo",
    "ippo",
    "maddpg",
]


def _build_baseline_meta() -> Dict[str, MethodMeta]:
    return {
        "random_masked": MethodMeta(
            method_name="random_masked",
            display_name="Random-Masked",
            category="baseline",
            method_type="baseline",
            config_name="random_masked",
            description="随机 masked baseline。",
        ),
        "greedy_local": MethodMeta(
            method_name="greedy_local",
            display_name="Greedy-Local",
            category="baseline",
            method_type="baseline",
            config_name="greedy_local",
            description="局部贪心启发式 baseline。",
        ),
        "constrained_kmeans": MethodMeta(
            method_name="constrained_kmeans",
            display_name="Constrained-KMeans",
            category="baseline",
            method_type="baseline",
            config_name="constrained_kmeans",
            description="集中式一次性部署 baseline。",
        ),
    }


def _build_rl_meta() -> Dict[str, MethodMeta]:
    out: Dict[str, MethodMeta] = {}

    for method_name in MAIN_RL_METHODS:
        spec = get_ablation_spec(method_name)
        out[method_name] = MethodMeta(
            method_name=method_name,
            display_name=spec.effective_display_name,
            category="main_rl",
            method_type="rl",
            config_name=spec.config_name,
            description=spec.description,
            checkpoint_name=spec.default_checkpoint_name,
            checkpoint_dir_name=spec.effective_checkpoint_dir_name,
            output_dir_name=spec.effective_output_dir_name,
            trainer_family=spec.trainer_family,
            policy_family=spec.policy_family,
        )

    for method_name in ABLATION_METHODS:
        spec = get_ablation_spec(method_name)
        out[method_name] = MethodMeta(
            method_name=method_name,
            display_name=spec.effective_display_name,
            category="ablation",
            method_type="ablation",
            config_name=spec.config_name,
            description=spec.description,
            checkpoint_name=spec.default_checkpoint_name,
            checkpoint_dir_name=spec.effective_checkpoint_dir_name,
            output_dir_name=spec.effective_output_dir_name,
            trainer_family=spec.trainer_family,
            policy_family=spec.policy_family,
        )

    legacy_target_map = {
        "ppo_wo_local_summary": "ppo_main",
        "ppo_wo_guidance": "ppo_main",
        "ppo_wo_neighbor_uav": "ppo_main",
        "ppo_reward_coverage_only": "ppo_main",
    }

    for method_name in LEGACY_ALIAS_METHODS:
        spec = get_ablation_spec(method_name)
        out[method_name] = MethodMeta(
            method_name=method_name,
            display_name=spec.effective_display_name,
            category="legacy_alias",
            method_type="legacy",
            config_name=spec.config_name,
            default_compare_enabled=False,
            description=spec.description,
            legacy_target_method=legacy_target_map.get(method_name),
            checkpoint_name=spec.default_checkpoint_name,
            checkpoint_dir_name=spec.effective_checkpoint_dir_name,
            output_dir_name=spec.effective_output_dir_name,
            trainer_family=spec.trainer_family,
            policy_family=spec.policy_family,
        )

    return out


_METHOD_REGISTRY: Dict[str, MethodMeta] = {}
_METHOD_REGISTRY.update(_build_baseline_meta())
_METHOD_REGISTRY.update(_build_rl_meta())


def get_method_meta(method_name: str) -> MethodMeta:
    if method_name not in _METHOD_REGISTRY:
        raise KeyError(f"Unknown method_name: {method_name}")
    return _METHOD_REGISTRY[method_name]


def list_method_metas(
    category: Optional[str] = None,
    method_type: Optional[str] = None,
    include_legacy: bool = True,
    only_default_compare_enabled: bool = False,
) -> List[MethodMeta]:
    metas: List[MethodMeta] = []
    for _, meta in _METHOD_REGISTRY.items():
        if category is not None and meta.category != category:
            continue
        if method_type is not None and meta.method_type != method_type:
            continue
        if not include_legacy and meta.is_legacy_alias:
            continue
        if only_default_compare_enabled and not meta.default_compare_enabled:
            continue
        metas.append(meta)
    return metas


def list_method_names(
    category: Optional[str] = None,
    method_type: Optional[str] = None,
    include_legacy: bool = True,
    only_default_compare_enabled: bool = False,
) -> List[str]:
    return [
        meta.method_name
        for meta in list_method_metas(
            category=category,
            method_type=method_type,
            include_legacy=include_legacy,
            only_default_compare_enabled=only_default_compare_enabled,
        )
    ]


def list_baseline_method_names() -> List[str]:
    return list(BASELINE_METHODS)


def list_main_rl_method_names() -> List[str]:
    return list(MAIN_RL_METHODS)


def list_ablation_method_names() -> List[str]:
    return list(ABLATION_METHODS)


def list_legacy_alias_method_names() -> List[str]:
    return list(LEGACY_ALIAS_METHODS)


def list_default_compare_method_names() -> List[str]:
    return list(DEFAULT_COMPARE_METHODS)


def get_display_name(method_name: str) -> str:
    return get_method_meta(method_name).display_name


def get_compare_label_map(method_names: Optional[List[str]] = None) -> Dict[str, str]:
    if method_names is None:
        method_names = list_default_compare_method_names()
    return {name: get_method_meta(name).display_name for name in method_names}


def is_baseline_method(method_name: str) -> bool:
    return get_method_meta(method_name).is_baseline


def is_main_rl_method(method_name: str) -> bool:
    return get_method_meta(method_name).is_main_rl


def is_ablation_method(method_name: str) -> bool:
    return get_method_meta(method_name).is_ablation


def is_legacy_alias_method(method_name: str) -> bool:
    return get_method_meta(method_name).is_legacy_alias
