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

from baselines.compare_adapter import (
    aggregate_method_records,
    build_compare_items,
    build_compare_method_entries,
    build_paper_table_rows,
    standardize_run_result,
)
from baselines.deployment_executor import execute_method_once
from baselines.method_registry import (
    get_method_meta,
    is_baseline_method,
    list_default_compare_method_names,
    list_method_names,
)
from configs.ablation_config import (
    apply_ablation_to_scenario,
    build_base_scenario_config,
    get_ablation_spec,
    resolve_method_checkpoint_path,
)
from envs.disaster_deployment_env import DisasterDeploymentEnv
from plotting.plot_compare import plot_deployment_comparison
from plotting.plot_scene import plot_scene_with_trajectories
from utils.experiment_schema import (
    COMPARE_AGGREGATES_FILENAME,
    COMPARE_ALL_FILENAME,
    COMPARE_EPISODES_FILENAME,
    COMPARE_PLOT_FILENAME,
    COMPARE_PROTOCOL_FILENAME,
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    make_paths_block,
    normalize_method_aggregate,
)
from utils.io import save_json
from utils.run_manager import build_run_dirs, build_run_name, save_manifest


DEFAULT_METHODS = list_default_compare_method_names()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified compare pipeline for baselines and registered RL methods.")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--output-root", type=str, default="results/compare")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--include-reserved-ablations", action="store_true")
    return parser.parse_args()


def _validate_requested_methods(
    method_names: List[str],
    include_legacy: bool,
    include_reserved_ablations: bool,
    checkpoint_name: str,
) -> List[Dict[str, str]]:
    skipped: List[Dict[str, str]] = []
    allowed_names = set(list_method_names(include_legacy=include_legacy))

    for method_name in method_names:
        if method_name not in allowed_names:
            skipped.append({
                "method_name": method_name,
                "reason": "unknown_or_legacy_not_enabled",
            })
            continue

        meta = get_method_meta(method_name)
        if meta.method_type == "ablation":
            abspec = get_ablation_spec(method_name)
            if abspec.is_reserved_ablation and not include_reserved_ablations:
                skipped.append({
                    "method_name": method_name,
                    "reason": "reserved_ablation_shell_not_enabled",
                })
                continue

        if not is_baseline_method(method_name):
            ckpt_path = resolve_method_checkpoint_path(
                method_name=method_name,
                checkpoint_name=checkpoint_name,
            )
            if not os.path.exists(ckpt_path):
                skipped.append({
                    "method_name": method_name,
                    "reason": f"checkpoint_not_found: {ckpt_path}",
                })

    return skipped


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_scenario_cfg = build_base_scenario_config()
    num_compare_episodes = int(args.episodes)
    scene_seed_list = [int(args.seed_base) + i for i in range(num_compare_episodes)]

    requested_methods = list(args.methods or DEFAULT_METHODS)
    skipped_methods = _validate_requested_methods(
        method_names=requested_methods,
        include_legacy=args.include_legacy,
        include_reserved_ablations=args.include_reserved_ablations,
        checkpoint_name=args.checkpoint_name,
    )
    skipped_names = {item["method_name"] for item in skipped_methods}
    active_methods = [name for name in requested_methods if name not in skipped_names]
    compare_method_entries = build_compare_method_entries(active_methods)

    run_tag = (
        f"formal_methods"
        f"_m{base_scenario_cfg.max_candidate_uavs}"
        f"_u{base_scenario_cfg.num_users}"
        f"_eps{num_compare_episodes}"
    )
    run_name = build_run_name(prefix="compare", method="formal_methods", tag=run_tag)
    dirs = build_run_dirs(output_root, run_name)
    paths_block = make_paths_block(
        run_dir=dirs["run_dir"],
        ckpt_dir=dirs["ckpt_dir"],
        plot_dir=dirs["plot_dir"],
        log_dir=dirs["log_dir"],
        stable_output_dir=None,
    )

    save_manifest(
        run_dir=dirs["run_dir"],
        run_type="compare",
        run_name=run_name,
        note="Unified compare pipeline with normalized result schema across baseline / PPO / IPPO / MADDPG.",
        schema_version=SCHEMA_VERSION,
        base_scenario_cfg=base_scenario_cfg,
        scene_seed_list=scene_seed_list,
        requested_methods=requested_methods,
        active_methods=active_methods,
        skipped_methods=skipped_methods,
        compare_method_entries=[entry.to_dict() for entry in compare_method_entries],
        method_meta_snapshot=[get_method_meta(name).to_dict() for name in active_methods],
        paths=paths_block,
    )

    episode_records: List[Dict[str, Any]] = []
    method_records: Dict[str, List[Dict[str, Any]]] = {entry.method_name: [] for entry in compare_method_entries}
    method_examples: Dict[str, Any] = {}
    agent_cache: Dict[str, Any] = {}

    for episode_index, scene_seed in enumerate(scene_seed_list):
        for entry in compare_method_entries:
            if entry.method_type == "baseline":
                scenario_cfg = base_scenario_cfg
            else:
                scenario_cfg = apply_ablation_to_scenario(base_scenario_cfg, entry.method_name)

            env = DisasterDeploymentEnv(scenario_cfg)
            result = execute_method_once(
                method_name=entry.method_name,
                env=env,
                max_steps=scenario_cfg.max_steps,
                seed=scene_seed,
                device=device,
                greedy=True,
                checkpoint_name=args.checkpoint_name,
                agent_cache=agent_cache,
            )

            record = standardize_run_result(
                result=result,
                scenario_cfg=scenario_cfg,
                scene_seed=scene_seed,
                episode_index=episode_index,
            )
            episode_records.append(record)
            method_records[entry.method_name].append(record)

            if entry.method_name not in method_examples:
                method_examples[entry.method_name] = {
                    "env": env,
                    "result": result,
                }

    method_aggregates = {
        method_name: normalize_method_aggregate(
            aggregate_method_records(records),
            fallback_method_name=method_name,
            fallback_display_name=get_method_meta(method_name).display_name,
        )
        for method_name, records in method_records.items()
    }

    method_order = [entry.method_name for entry in compare_method_entries]
    compare_items = build_compare_items(method_aggregates, method_order)
    paper_table_rows = build_paper_table_rows(compare_items)

    for entry in compare_method_entries:
        example = method_examples.get(entry.method_name)
        if example is None:
            continue

        env_ref = example["env"]
        result_ref = example["result"]
        render_data = env_ref.render()

        plot_scene_with_trajectories(
            render_data=render_data,
            trajectory_history=result_ref.trajectory_history,
            r_safe=env_ref.cfg.r_safe,
            r_disaster=env_ref.cfg.r_disaster,
            coverage_radius=env_ref.cfg.simplified_coverage_radius if env_ref.cfg.use_simplified_qos else None,
            title=f"{entry.display_name}\n{run_name}",
            save_path=os.path.join(dirs["plot_dir"], f"{entry.method_name}_scene_example.png"),
            show=False,
        )

    plot_deployment_comparison(
        result_dicts=compare_items,
        title=f"Formal Method Comparison\n{run_name}",
        save_path=os.path.join(dirs["plot_dir"], COMPARE_PLOT_FILENAME),
        show=False,
    )

    save_json(
        {
            "schema_version": SCHEMA_VERSION,
            "run_type": "compare",
            "run_name": run_name,
            "requested_methods": requested_methods,
            "active_methods": active_methods,
            "skipped_methods": skipped_methods,
            "compare_method_entries": [entry.to_dict() for entry in compare_method_entries],
            "method_meta_snapshot": [get_method_meta(name).to_dict() for name in active_methods],
            "protocol": {
                "num_compare_episodes": num_compare_episodes,
                "scene_seed_list": scene_seed_list,
                "base_scenario_cfg": base_scenario_cfg.__dict__,
                "shared_max_steps": base_scenario_cfg.max_steps,
                "shared_initialization_rule": "same_scene_seed_per_method",
                "unified_compare_fields": [
                    "final_coverage_ratio",
                    "final_covered_users",
                    "total_move_distance",
                    "mean_overlap_users_step",
                    "episode_length",
                    "full_coverage_success",
                    "method_name",
                    "display_name",
                ],
            },
        },
        os.path.join(dirs["log_dir"], COMPARE_PROTOCOL_FILENAME),
    )

    save_json(
        {
            "schema_version": SCHEMA_VERSION,
            "run_type": "compare",
            "run_name": run_name,
            "episode_records": episode_records,
        },
        os.path.join(dirs["log_dir"], COMPARE_EPISODES_FILENAME),
    )

    save_json(
        {
            "schema_version": SCHEMA_VERSION,
            "run_type": "compare",
            "run_name": run_name,
            "method_aggregates": method_aggregates,
            "compare_items": compare_items,
            "paper_table_rows": paper_table_rows,
        },
        os.path.join(dirs["log_dir"], COMPARE_AGGREGATES_FILENAME),
    )

    save_json(
        {
            "schema_version": SCHEMA_VERSION,
            "run_type": "compare",
            "run_name": run_name,
            "requested_methods": requested_methods,
            "active_methods": active_methods,
            "skipped_methods": skipped_methods,
            "compare_method_entries": [entry.to_dict() for entry in compare_method_entries],
            "episode_records": episode_records,
            "method_aggregates": method_aggregates,
            "compare_items": compare_items,
            "paper_table_rows": paper_table_rows,
        },
        os.path.join(dirs["log_dir"], COMPARE_ALL_FILENAME),
    )

    save_json(
        {
            "schema_version": SCHEMA_VERSION,
            "summary_type": "compare_run_summary",
            "run_name": run_name,
            "requested_methods": requested_methods,
            "active_methods": active_methods,
            "skipped_methods": skipped_methods,
            "compare_method_entries": [entry.to_dict() for entry in compare_method_entries],
            "method_aggregates": method_aggregates,
            "compare_items": compare_items,
            "paper_table_rows": paper_table_rows,
        },
        os.path.join(dirs["run_dir"], SUMMARY_FILENAME),
    )

    print("===== Formal Compare Done =====")
    print("run_name:", run_name)
    print("run_dir:", dirs["run_dir"])
    if skipped_methods:
        print("skipped_methods:", skipped_methods)
    for method_name in method_order:
        print(f"{method_name}: {method_aggregates[method_name]}")


if __name__ == "__main__":
    main()