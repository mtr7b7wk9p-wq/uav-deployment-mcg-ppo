from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from plotting.paper_results import (
    build_analysis_templates,
    build_caption_templates,
    build_metric_definitions,
    plot_ablation_bars,
    plot_main_results_bars,
    plot_separate_metric_figures,
    plot_step_level_coverage_growth,
    plot_training_curves,
)
from utils.experiment_schema import normalize_method_aggregate
from utils.io import load_json, save_json
from utils.table_export import (
    build_table_rows,
    export_table_csv,
    export_table_latex,
    export_table_markdown,
)


DEFAULT_MAIN_METHODS = [
    "random_masked",
    "greedy_local",
    "constrained_kmeans",
    "ppo_main",
    "ippo",
    "maddpg",
    "mcg_ppo",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出论文可用的结果图、表格与文本模板。")
    parser.add_argument("--compare-json", type=str, required=True, help="deployment_compare_all.json 路径。")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录。默认使用 compare run_dir 下的 paper_exports。")
    parser.add_argument("--training-log-paths", nargs="*", default=None, help="可选：多个 training_log.json 路径。")
    parser.add_argument("--training-method-names", nargs="*", default=None, help="可选：与 training-log-paths 对应的方法名。")
    parser.add_argument("--main-methods", nargs="*", default=None, help="主结果图方法列表。默认包含 ppo_main/ippo/maddpg/mcg_ppo 与 baseline。")
    parser.add_argument("--ablation-main-method", type=str, default="mcg_ppo", help="消融图主方法名。")
    parser.add_argument("--file-prefix", type=str, default="paper", help="论文图输出前缀。")
    return parser.parse_args()


def _write_text(path: str, text: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _build_step_trace_payload(example_rollouts: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    traces: Dict[str, List[Dict[str, Any]]] = {}
    for method_name, payload in example_rollouts.items():
        trace = payload.get("step_info_trace", [])
        if isinstance(trace, list) and len(trace) > 0:
            traces[method_name] = trace
    return traces


def _collect_training_payloads(
    training_log_paths: Optional[Sequence[str]],
    training_method_names: Optional[Sequence[str]],
) -> Dict[str, Dict[str, Any]]:
    if not training_log_paths:
        return {}

    payloads: Dict[str, Dict[str, Any]] = {}
    for idx, path in enumerate(training_log_paths):
        if not os.path.exists(path):
            continue

        payload = load_json(path)
        if training_method_names is not None and idx < len(training_method_names):
            method_name = str(training_method_names[idx])
        else:
            method_name = (
                payload.get("method", {}).get("method_name")
                or payload.get("method_name")
                or f"method_{idx + 1}"
            )
        payloads[method_name] = payload
    return payloads


def main() -> None:
    args = _parse_args()

    compare_payload = load_json(args.compare_json)
    compare_items = [normalize_method_aggregate(x) for x in compare_payload.get("compare_items", [])]
    method_aggregates = {
        k: normalize_method_aggregate(v, fallback_method_name=k, fallback_display_name=k)
        for k, v in compare_payload.get("method_aggregates", {}).items()
    }
    example_rollouts = compare_payload.get("example_rollouts", {})

    run_dir = compare_payload.get("run_dir") or os.path.dirname(os.path.dirname(args.compare_json))
    paper_root = args.out_dir or os.path.join(run_dir, "paper_exports")

    paper_fig_dir = os.path.join(paper_root, "paper_figures")
    paper_table_dir = os.path.join(paper_root, "paper_tables")
    paper_text_dir = os.path.join(paper_root, "paper_text")

    os.makedirs(paper_fig_dir, exist_ok=True)
    os.makedirs(paper_table_dir, exist_ok=True)
    os.makedirs(paper_text_dir, exist_ok=True)

    captions = build_caption_templates()
    metric_defs = build_metric_definitions()
    analysis_templates = build_analysis_templates()

    main_methods = list(args.main_methods) if args.main_methods else list(DEFAULT_MAIN_METHODS)

    # -----------------------------
    # Figure 1: 主结果图
    # -----------------------------
    plot_main_results_bars(
        compare_items=compare_items,
        include_methods=main_methods,
        title="Main results across different deployment methods",
        save_path=os.path.join(paper_fig_dir, f"{args.file_prefix}_main_results.png"),
        show=False,
    )

    # -----------------------------
    # Figure 1b: 四张独立指标图
    # -----------------------------
    plot_separate_metric_figures(
        experiment_payloads=[{
            "experiment_label": "Current Run",
            "compare_items": compare_items,
        }],
        save_dir=paper_fig_dir,
        file_prefix=f"{args.file_prefix}_single_metric",
        metric_keys=[
            "final_coverage_ratio",
            "final_covered_users",
            "total_move_distance",
            "mean_overlap_users_step",
        ],
        include_methods=main_methods,
        max_experiments=4,
        show=False,
    )

    # -----------------------------
    # Figure 2: RL 主方法对比（论文里最常用）
    # -----------------------------
    rl_core_methods = [m for m in ["ppo_main", "ippo", "maddpg", "mcg_ppo"] if any(x["method_name"] == m for x in compare_items)]
    if len(rl_core_methods) > 0:
        plot_separate_metric_figures(
            experiment_payloads=[{
                "experiment_label": "RL Compare",
                "compare_items": compare_items,
            }],
            save_dir=paper_fig_dir,
            file_prefix=f"{args.file_prefix}_rl_compare",
            metric_keys=[
                "final_coverage_ratio",
                "final_covered_users",
                "total_move_distance",
                "mean_overlap_users_step",
            ],
            include_methods=rl_core_methods,
            max_experiments=4,
            show=False,
        )

    # -----------------------------
    # Figure 3: 消融图（若存在）
    # -----------------------------
    candidate_names = [x["method_name"] for x in compare_items]
    ablation_names = [
        name for name in candidate_names
        if (
            name.startswith("mcg_ppo_no_")
            or name.startswith("ppo_wo_")
            or name.startswith("ppo_reward_")
        )
    ]
    if len(ablation_names) > 0 and args.ablation_main_method in candidate_names:
        plot_ablation_bars(
            compare_items=compare_items,
            main_method=args.ablation_main_method,
            title="Ablation results of the proposed framework",
            save_path=os.path.join(paper_fig_dir, f"{args.file_prefix}_ablation.png"),
            show=False,
        )

    # -----------------------------
    # Figure 4: 训练曲线（可选）
    # -----------------------------
    training_payloads = _collect_training_payloads(
        training_log_paths=args.training_log_paths,
        training_method_names=args.training_method_names,
    )
    if len(training_payloads) > 0:
        plot_training_curves(
            training_payloads=training_payloads,
            title="Training convergence of compared reinforcement learning methods",
            save_path=os.path.join(paper_fig_dir, f"{args.file_prefix}_training_curves.png"),
            show=False,
        )

    # -----------------------------
    # Figure 5: step-level 曲线（可选）
    # -----------------------------
    step_traces = _build_step_trace_payload(example_rollouts)
    if len(step_traces) > 0:
        plot_step_level_coverage_growth(
            step_traces=step_traces,
            title="Coverage growth over deployment steps",
            save_path=os.path.join(paper_fig_dir, f"{args.file_prefix}_step_coverage.png"),
            show=False,
        )

    # -----------------------------
    # Table 1: 主结果表
    # -----------------------------
    main_rows = build_table_rows(method_aggregates=method_aggregates, include_methods=main_methods)
    export_table_csv(main_rows, os.path.join(paper_table_dir, "results_table_main.csv"))
    export_table_markdown(main_rows, os.path.join(paper_table_dir, "results_table_main.md"))
    export_table_latex(
        main_rows,
        os.path.join(paper_table_dir, "results_table_main.tex"),
        caption=captions["table_main_results"],
        label="tab:main_results",
    )

    # -----------------------------
    # Table 2: RL 主方法对比表
    # -----------------------------
    if len(rl_core_methods) > 0:
        rl_rows = build_table_rows(method_aggregates=method_aggregates, include_methods=rl_core_methods)
        export_table_csv(rl_rows, os.path.join(paper_table_dir, "results_table_rl_compare.csv"))
        export_table_markdown(rl_rows, os.path.join(paper_table_dir, "results_table_rl_compare.md"))
        export_table_latex(
            rl_rows,
            os.path.join(paper_table_dir, "results_table_rl_compare.tex"),
            caption="Quantitative comparison among PPO Main, IPPO, MADDPG and MCG-PPO.",
            label="tab:rl_compare",
        )

    # -----------------------------
    # Table 3: 消融表
    # -----------------------------
    if len(ablation_names) > 0 and args.ablation_main_method in candidate_names:
        ablation_include = [args.ablation_main_method] + ablation_names
        ablation_rows = build_table_rows(method_aggregates=method_aggregates, include_methods=ablation_include)
        export_table_csv(ablation_rows, os.path.join(paper_table_dir, "results_table_ablation.csv"))
        export_table_markdown(ablation_rows, os.path.join(paper_table_dir, "results_table_ablation.md"))
        export_table_latex(
            ablation_rows,
            os.path.join(paper_table_dir, "results_table_ablation.tex"),
            caption=captions["table_ablation"],
            label="tab:ablation_results",
        )

    # -----------------------------
    # Caption / 指标定义 / 分析模板
    # -----------------------------
    save_json(captions, os.path.join(paper_text_dir, "captions.json"))
    save_json(metric_defs, os.path.join(paper_text_dir, "metric_definitions.json"))
    save_json(analysis_templates, os.path.join(paper_text_dir, "analysis_templates.json"))

    _write_text(
        os.path.join(paper_text_dir, "captions.md"),
        "\n\n".join([f"### {k}\n{v}" for k, v in captions.items()]),
    )
    _write_text(
        os.path.join(paper_text_dir, "metric_definitions.md"),
        "\n\n".join([
            f"### {k}\n"
            f"- 字段名：{v['field_name']}\n"
            f"- 指标名：{v['display_name']}\n"
            f"- 数学表达：{v['formula']}\n"
            f"- 含义：{v['definition']}"
            for k, v in metric_defs.items()
        ]),
    )
    _write_text(
        os.path.join(paper_text_dir, "analysis_templates.md"),
        "\n\n".join([f"### {k}\n{v}" for k, v in analysis_templates.items()]),
    )

    save_json(
        {
            "paper_root": paper_root,
            "paper_figures_dir": paper_fig_dir,
            "paper_tables_dir": paper_table_dir,
            "paper_text_dir": paper_text_dir,
            "generated_main_methods": main_methods,
            "generated_rl_core_methods": rl_core_methods,
            "generated_ablation_methods": ablation_names,
            "compare_json": args.compare_json,
        },
        os.path.join(paper_root, "paper_results_summary.json"),
    )

    print("===== Paper Results Export Done =====")
    print("paper_root:", paper_root)
    print("paper_figures_dir:", paper_fig_dir)
    print("paper_tables_dir:", paper_table_dir)
    print("paper_text_dir:", paper_text_dir)


if __name__ == "__main__":
    main()