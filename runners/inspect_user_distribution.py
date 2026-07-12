from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.scenario_config import ScenarioConfig
from envs.disaster_deployment_env import DisasterDeploymentEnv
from plotting.plot_scene import plot_scene


def main() -> None:
    scenario_cfg = ScenarioConfig(
        num_users=20,
        user_distribution_mode="mixed",
        num_user_clusters=3,
        clustered_user_ratio=0.80,
        cluster_radius=220.0,
        max_candidate_uavs=3,
        initially_active_uavs=3,
        max_obs_users=20,
        max_obs_uavs=2,
        seed=123,
    )

    env = DisasterDeploymentEnv(scenario_cfg)
    obs = env.reset(seed=scenario_cfg.seed)
    render_data = env.render()

    out_dir = os.path.join("results", "debug_user_distribution")
    os.makedirs(out_dir, exist_ok=True)

    plot_scene(
        render_data=render_data,
        r_safe=scenario_cfg.r_safe,
        r_disaster=scenario_cfg.r_disaster,
        coverage_radius=scenario_cfg.simplified_coverage_radius if scenario_cfg.use_simplified_qos else None,
        title="User Distribution Debug",
        show_assignments=False,
        show_coverage_circle=False,
        show_cluster_centers=True,
        save_path=os.path.join(out_dir, "user_distribution_debug.png"),
        show=False,
    )

    print("===== User Distribution Debug =====")
    print("local_obs shape:", obs["local_obs"].shape)
    for key, value in render_data["user_distribution_stats"].items():
        print(f"{key}: {value}")
    print("saved figure:", os.path.join(out_dir, "user_distribution_debug.png"))


if __name__ == "__main__":
    main()
