from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from plotting.paper_results import plot_separate_metric_figures
from utils.experiment_schema import normalize_method_aggregate
from utils.io import load_json, save_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将最多 4 组 compare 结果合并为论文可用的单指标独立图。",
    )
    parser.add_argument(
        "--compare-jsons",
        nargs="+",
        required=True,
        help="一个或多个 deployment_compare_all.json 路径，最多 4 个。",
    )
    parser.add_argument(
        "--experiment-labels",
        nargs="*",
        default=None,
        help="与 compare-jsons 对应的实验标签。",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出目录。",
    )
    parser.add_argument(
        "--file-prefix",
        default="paper_compare",
        help="输出图片前缀。",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "final_coverage_ratio",
            "final_covered_users",
            "total_move_distance",
            "mean_overlap_users_step",
        ],
        help="要导出的指标列表。",
    )
    parser.add_argument(
        "--include-methods",
        nargs="*",
        default=["ppo_main", "ippo", "maddpg", "mcg_ppo"],
        help="只绘制这些 method_name。默认直接用于 RL 主方法对比。",
    )
    parser.add_argument(
        "--exclude-methods",
        nargs="*",
        default=None,
        help="排除这些 method_name。",
    )
    return parser.parse_args()


def _build_experiment_payloads(
    compare_json_paths: Sequence[str],
    experiment_labels: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if len(compare_json_paths) == 0:
        raise ValueError("compare_json_paths is empty")
    if len(compare_json_paths) > 4:
        raise ValueError("At most 4 compare json files are supported.")

    payloads: List[Dict[str, Any]] = []
    for idx, path in enumerate(compare_json_paths):
        compare_payload = load_json(path)
        label = None
        if experiment_labels is not None and idx < len(experiment_labels):
            label = experiment_labels[idx]
        if label is None or str(label).strip() == "":
            label = f"Exp{idx + 1}"

        compare_items = [
            normalize_method_aggregate(x)
            for x in compare_payload.get("compare_items", [])
        ]

        payloads.append({
            "experiment_label": str(label),
            "compare_items": compare_items,
            "source_json": path,
            "run_dir": compare_payload.get("run_dir", ""),
        })
    return payloads


def main() -> None:
    args = _parse_args()
    experiment_payloads = _build_experiment_payloads(
        compare_json_paths=args.compare_jsons,
        experiment_labels=args.experiment_labels,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    output_paths = plot_separate_metric_figures(
        experiment_payloads=experiment_payloads,
        save_dir=args.out_dir,
        file_prefix=args.file_prefix,
        metric_keys=args.metrics,
        include_methods=args.include_methods,
        exclude_methods=args.exclude_methods,
        max_experiments=4,
        show=False,
    )

    summary = {
        "out_dir": args.out_dir,
        "file_prefix": args.file_prefix,
        "metrics": list(args.metrics),
        "include_methods": list(args.include_methods) if args.include_methods is not None else None,
        "exclude_methods": list(args.exclude_methods) if args.exclude_methods is not None else None,
        "experiment_payloads": experiment_payloads,
        "output_paths": output_paths,
    }
    save_json(summary, os.path.join(args.out_dir, f"{args.file_prefix}_summary.json"))

    print("===== Separate Paper Figures Export Done =====")
    print("out_dir:", args.out_dir)
    for metric_key, path in output_paths.items():
        print(metric_key, "->", path)


if __name__ == "__main__":
    main()