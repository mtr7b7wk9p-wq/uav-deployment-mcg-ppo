from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from baselines.compare_adapter import aggregate_method_records, standardize_run_result
from baselines.deployment_executor import build_agent_from_checkpoint, run_ppo_deployment
from configs.ablation_config import (
    apply_ablation_to_scenario,
    build_base_scenario_config,
    get_ablation_spec,
    resolve_method_checkpoint_path,
    resolve_method_name,
)
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
    parser = argparse.ArgumentParser(description="Evaluate registered RL deployment methods.")
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--output-root", type=str, default="results/eval")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--greedy", dest="greedy", action="store_true")
    parser.add_argument("--sample", dest="greedy", action="store_false")
    parser.set_defaults(greedy=True)
    return parser.parse_args()


def _copy_if_exists(src_path: str, dst_path: str) -> None:
    if not os.path.exists(src_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)


def sync_eval_alias_artifacts(output_root: str, method_dir_name: str, checkpoint_stem: str, dirs: Dict[str, str]) -> None:
    alias_root = os.path.join(output_root, method_dir_name, checkpoint_stem)
    _copy_if_exists(os.path.join(dirs["run_dir"], "manifest.json"), os.path.join(alias_root, "manifest.json"))
    _copy_if_exists(os.path.join(dirs["run_dir"], SUMMARY_FILENAME), os.path.join(alias_root, SUMMARY_FILENAME))
    _copy_if_exists(os.path.join(dirs["log_dir"], EVAL_LOG_FILENAME), os.path.join(alias_root, "logs", EVAL_LOG_FILENAME))
    _copy_if_exists(os.path.join(dirs["plot_dir"], EVAL_SCENE_PLOT_FILENAME), os.path.join(alias_root, "plots", EVAL_SCENE_PLOT_FILENAME))


def main() -> None:
    args = parse_args()
    method_name = resolve_method_name(
        method_name=args.method_name,
        config_name=args.config_name,
        default_method_name="ppo_main",
    )
    ablation_spec = get_ablation_spec(method_name)

    ckpt_path = args.ckpt_path or resolve_method_checkpoint_path(
        method_name=method_name,
        checkpoint_name=args.checkpoint_name,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    greedy = args.greedy

    base_scenario_cfg = build_base_scenario_config()
    scenario_cfg = apply_ablation_to_scenario(base_scenario_cfg, method_name)

    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_tag = (
        f"{method_name}"
        f"_cfg{ablation_spec.config_name}"
        f"_{ckpt_name}"
        f"_m{scenario_cfg.max_candidate_uavs}"
        f"_u{scenario_cfg.num_users}"
        f"_{'greedy' if greedy else 'sample'}"
    )
    run_name = build_run_name(prefix="eval", method=method_name, tag=run_tag)
    dirs = build_run_dirs(args.output_root, run_name)

    if method_name == "ippo":
        agent_type = "ippo"
    elif method_name == "maddpg":
        agent_type = "maddpg"
    else:
        agent_type = "shared_ppo"

    method_identity = make_method_identity(
        method_name=method_name,
        display_name=ablation_spec.effective_display_name,
        config_name=ablation_spec.config_name,
        checkpoint_dir_name=ablation_spec.effective_checkpoint_dir_name,
        output_dir_name=ablation_spec.effective_output_dir_name,
        checkpoint_name=os.path.basename(ckpt_path),
        method_label=ablation_spec.method_label,
        trainer_family=ablation_spec.trainer_family,
        policy_family=ablation_spec.policy_family,
        agent_type=agent_type,
    )
    stable_output_dir = os.path.join(args.output_root, method_identity["output_dir_name"], ckpt_name)
    paths_block = make_paths_block(
        run_dir=dirs["run_dir"],
        ckpt_dir=dirs["ckpt_dir"],
        plot_dir=dirs["plot_dir"],
        log_dir=dirs["log_dir"],
        stable_output_dir=stable_output_dir,
    )

    save_manifest(
        run_dir=dirs["run_dir"],
        run_type="eval",
        run_name=run_name,
        note="Single-method evaluation with unified compare schema",
        schema_version=SCHEMA_VERSION,
        method=method_identity,
        checkpoint=ckpt_path,
        paths=paths_block,
        ablation_spec=ablation_spec.to_dict(),
        scenario_cfg=scenario_cfg,
        greedy=greedy,
        eval_episodes=args.eval_episodes,
        device=device,
    )

    agent = build_agent_from_checkpoint(ckpt_path=ckpt_path, device=device)
    episode_results: List[Dict[str, Any]] = []

    for ep in range(args.eval_episodes):
        env = DisasterDeploymentEnv(scenario_cfg)
        result = run_ppo_deployment(
            env=env,
            agent=agent,
            max_steps=scenario_cfg.max_steps,
            seed=args.seed_base + ep,
            greedy=greedy,
            method_name=method_name,
        )

        episode_results.append(
            standardize_run_result(
                result=result,
                scenario_cfg=scenario_cfg,
                scene_seed=args.seed_base + ep,
                episode_index=ep,
            )
        )

        if ep == 0:
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

    aggregate = normalize_method_aggregate(
        aggregate_method_records(episode_results),
        fallback_method_name=method_identity["method_name"],
        fallback_display_name=method_identity["display_name"],
    )

    eval_log_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "eval",
        "run_name": run_name,
        "method": method_identity,
        "paths": paths_block,
        "checkpoint_path": ckpt_path,
        "checkpoint_name": os.path.basename(ckpt_path),
        "greedy": greedy,
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
    save_json(eval_log_payload, os.path.join(dirs["log_dir"], EVAL_LOG_FILENAME))

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "summary_type": "eval_run_summary",
        "run_name": run_name,
        "method": method_identity,
        "paths": paths_block,
        "checkpoint_path": ckpt_path,
        "checkpoint_name": os.path.basename(ckpt_path),
        "greedy": greedy,
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
        "paper_metric_row": build_paper_metric_row(method_identity, aggregate),
    }
    save_json(summary_payload, os.path.join(dirs["run_dir"], SUMMARY_FILENAME))

    sync_eval_alias_artifacts(
        output_root=args.output_root,
        method_dir_name=method_identity["output_dir_name"],
        checkpoint_stem=ckpt_name,
        dirs=dirs,
    )

    print("===== Eval Done =====")
    print("run_name:", run_name)
    print("run_dir:", dirs["run_dir"])
    print("method_name:", method_identity["method_name"])
    print("display_name:", method_identity["display_name"])
    print("checkpoint:", ckpt_path)
    print("aggregate:", aggregate)


if __name__ == "__main__":
    main()