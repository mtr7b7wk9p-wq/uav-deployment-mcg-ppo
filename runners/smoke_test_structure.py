from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from agents.ppo.ppo_agent import PPOConfig, SharedPPOAgent
from baselines.compare_adapter import build_compare_method_entries
from baselines.method_registry import (
    list_baseline_method_names,
    list_default_compare_method_names,
)
from configs.ablation_config import build_base_scenario_config
from envs.disaster_deployment_env import DisasterDeploymentEnv


def main() -> None:
    scenario_cfg = build_base_scenario_config()
    env = DisasterDeploymentEnv(scenario_cfg)
    obs = env.reset(seed=scenario_cfg.seed)
    local_obs_dim = int(obs["local_obs"].shape[-1])

    ppo_cfg = PPOConfig(
        local_obs_dim=local_obs_dim,
        action_dim=int(scenario_cfg.action_size),
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_obs_users=int(scenario_cfg.max_obs_users),
        max_obs_uavs=int(scenario_cfg.max_obs_uavs),
        num_direction_sectors=int(scenario_cfg.num_direction_sectors),
        num_radial_bins=int(scenario_cfg.num_radial_bins),
    )
    agent = SharedPPOAgent(ppo_cfg)

    default_methods = list_default_compare_method_names()
    compare_entries = build_compare_method_entries(default_methods)

    baseline_name_set = set(list_baseline_method_names())
    baseline_entries = [entry for entry in compare_entries if entry.method_name in baseline_name_set]
    _ = [entry.to_dict() for entry in baseline_entries]

    print("[OK] train_init")
    print(f"local_obs_dim={ppo_cfg.local_obs_dim}")
    print(f"agent_device={agent.device}")
    print("[OK] eval_init")
    print("[OK] compare_init")
    print(f"baseline_methods={[entry.method_name for entry in baseline_entries]}")


if __name__ == "__main__":
    main()
