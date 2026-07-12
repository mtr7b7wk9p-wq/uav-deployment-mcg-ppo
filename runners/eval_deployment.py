from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from baselines.compare_adapter import aggregate_method_records, standardize_run_result
from baselines.deployment_executor import execute_method_once
from baselines.method_registry import get_method_meta
from configs.ablation_config import apply_ablation_to_scenario, build_base_scenario_config
from envs.disaster_deployment_env import DisasterDeploymentEnv
from plotting.plot_scene import plot_scene_with_trajectories
from utils.experiment_schema import (
    EVAL_LOG_FILENAME,
    EVAL_SCENE_PLOT_FILENAME,
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    build_paper_metric_row,
    make_method_identity,
    make_paths_block,
    normalize_method_aggregate,
)
from utils.io import save_json
from utils.run_manager import build_run_dirs, build_run_name, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation entry for baseline and RL methods.")
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="results/eval")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--greedy", dest="greedy", action="store_true")
    parser.add_argument("--sample", dest="greedy", action="store_false")
    parser.set_defaults(greedy=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = get_method_meta(args.method_name)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_scenario_cfg = build_base_scenario_config()
    if meta.method_type == "baseline":
        scenario_cfg = base_scenario_cfg
    else:
        scenario_cfg = apply_ablation_to_scenario(base_scenario_cfg, args.method_name)

    run_tag = (
        f"{args.method_name}"
        f"_m{scenario_cfg.max_candidate_uavs}"
        f"_u{scenario_cfg.num_users}"
        f"_eps{args.eval_episodes}"
    )
    run_name = build_run_name(prefix="eval", method=args.method_name, tag=run_tag)
    dirs = build_run_dirs(args.output_root, run_name)

    method_identity = make_method_identity(
        method_name=meta.method_name,
        display_name=meta.display_name,
        config_name=meta.config_name,
        checkpoint_dir_name=meta.checkpoint_dir_name,
        output_dir_name=meta.output_dir_name,
        checkpoint_name=meta.checkpoint_name,
        trainer_family=meta.trainer_family,
        policy_family=meta.policy_family,
        agent_type=(
            "ippo" if args.method_name == "ippo"
            else "maddpg" if args.method_name == "maddpg"
            else "shared_ppo" if meta.method_type != "baseline"
            else "baseline"
        ),
    )
    paths_block = make_paths_block(
        run_dir=dirs["run_dir"],
        ckpt_dir=dirs["ckpt_dir"],
        plot_dir=dirs["plot_dir"],
        log_dir=dirs["log_dir"],
        stable_output_dir=None,
    )

    save_manifest(
        run_dir=dirs["run_dir"],
        run_type="eval",
        run_name=run_name,
        note="Unified baseline / RL evaluation entry with common compare fields.",
        schema_version=SCHEMA_VERSION,
        method=method_identity,
        paths=paths_block,
        scenario_cfg=scenario_cfg,
        eval_episodes=args.eval_episodes,
        device=device,
    )

    episode_results: List[Dict[str, Any]] = []
    scene_plot_done = False
    agent_cache: Dict[str, Any] = {}

    for ep in range(args.eval_episodes):
        env = DisasterDeploymentEnv(scenario_cfg)
        result = execute_method_once(
            method_name=args.method_name,
            env=env,
            max_steps=scenario_cfg.max_steps,
            seed=args.seed_base + ep,
            device=device,
            greedy=args.greedy,
            checkpoint_name=args.checkpoint_name,
            agent_cache=agent_cache,
        )

        record = standardize_run_result(
            result=result,
            scenario_cfg=scenario_cfg,
            scene_seed=args.seed_base + ep,
            episode_index=ep,
        )
        episode_results.append(record)

        if not scene_plot_done:
            render_data = env.render()
            plot_scene_with_trajectories(
                render_data=render_data,
                trajectory_history=result.trajectory_history,
                r_safe=scenario_cfg.r_safe,
                r_disaster=scenario_cfg.r_disaster,
                coverage_radius=scenario_cfg.simplified_coverage_radius if scenario_cfg.use_simplified_qos else None,
                title=f"{method_identity['display_name']}\n{run_name}",
                save_path=os.path.join(dirs["plot_dir"], EVAL_SCENE_PLOT_FILENAME),
                show=False,
            )
            scene_plot_done = True

    aggregate = normalize_method_aggregate(
        aggregate_method_records(episode_results),
        fallback_method_name=method_identity["method_name"],
        fallback_display_name=method_identity["display_name"],
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "eval",
        "run_name": run_name,
        "method": method_identity,
        "paths": paths_block,
        "scenario_cfg": scenario_cfg.__dict__,
        "episode_results": episode_results,
        "aggregate": aggregate,
        "compare_fields": {
            "final_coverage_ratio": aggregate["mean_final_coverage_ratio"],
            "final_covered_users": aggregate["mean_final_covered_users"],
            "total_move_distance": aggregate["mean_total_move_distance"],
            "mean_overlap_users_step": aggregate["mean_mean_overlap_users_step"],
            "episode_length": aggregate["mean_episode_length"],
            "full_coverage_success": aggregate["full_coverage_success_rate"],
            "method_name": aggregate["method_name"],
            "display_name": aggregate["display_name"],
        },
    }
    save_json(payload, os.path.join(dirs["log_dir"], EVAL_LOG_FILENAME))

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "summary_type": "eval_run_summary",
        "run_name": run_name,
        "method": method_identity,
        "paths": paths_block,
        "aggregate": aggregate,
        "compare_fields": payload["compare_fields"],
        "paper_metric_row": build_paper_metric_row(method_identity, aggregate),
    }
    save_json(summary_payload, os.path.join(dirs["run_dir"], SUMMARY_FILENAME))

    print("===== Unified Eval Done =====")
    print("run_name:", run_name)
    print("run_dir:", dirs["run_dir"])
    print("aggregate:", aggregate)


if __name__ == "__main__":
    main()