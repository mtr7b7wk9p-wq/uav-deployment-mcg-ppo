from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.paper_results import plot_training_curve_groups
from utils.experiment_schema import (
    DEFAULT_TRAINING_METRICS,
    extract_method_display_name,
    get_training_curve_metric_keys,
    normalize_training_metric_series,
    resolve_training_log_path,
)
from utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 train_ppo_deployment 相同绘图风格，重绘多方法训练曲线，并将四个指标分别输出。"
    )
    parser.add_argument(
        "--compare-dirs",
        nargs="*",
        default=[],
        help="对比实验目录或 training_log.json 路径。",
    )
    parser.add_argument(
        "--ablation-dirs",
        nargs="*",
        default=[],
        help="消融实验目录或 training_log.json 路径。",
    )
    parser.add_argument(
        "--compare-labels",
        nargs="*",
        default=[],
        help="与 --compare-dirs 对齐的显示名称。",
    )
    parser.add_argument(
        "--ablation-labels",
        nargs="*",
        default=[],
        help="与 --ablation-dirs 对齐的显示名称。",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_TRAINING_METRICS),
        help="默认四个指标：final_coverage_ratio episode_return total_move_distance mean_overlap_users_step",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认 PROJECT_ROOT/runners/results/training_curves",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=200,
        help="和 train_ppo_deployment 保持一致，默认 200。",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="显示浅色 raw 曲线。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否弹窗显示。",
    )
    return parser.parse_args()


def _normalize_metric_args(metric_args: Sequence[str]) -> List[str]:
    metric_keys = get_training_curve_metric_keys(metric_args)
    if not metric_keys:
        raise ValueError("No valid metrics were provided.")
    return metric_keys


def _build_curve_item(exp_path: str, custom_label: str = "") -> Dict[str, Any]:
    log_path = resolve_training_log_path(exp_path)
    payload = load_json(log_path)

    method_name = str(
        payload.get("method_name")
        or (payload.get("method") or {}).get("method_name")
        or "unknown"
    )
    display_name = custom_label or extract_method_display_name(payload, fallback=method_name)

    normalized = normalize_training_metric_series(payload)

    return {
        "exp_path": os.path.abspath(exp_path),
        "log_path": os.path.abspath(log_path),
        "method_name": method_name,
        "display_name": display_name,
        "metrics": normalized,
    }


def _build_curve_items(paths: Sequence[str], labels: Sequence[str]) -> List[Dict[str, Any]]:
    curve_items: List[Dict[str, Any]] = []
    for idx, exp_path in enumerate(paths):
        custom_label = labels[idx] if idx < len(labels) else ""
        try:
            curve_items.append(_build_curve_item(exp_path, custom_label=custom_label))
        except FileNotFoundError as exc:
            print(f"[Skip] {exc}")
        except Exception as exc:
            print(f"[Skip] 读取训练曲线失败: {exp_path} -> {exc}")
    return curve_items


def _flatten_group_for_metric(group_items: Sequence[Dict[str, Any]], metric_key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(group_items):
        metric_payload = item.get("metrics", {}).get(metric_key, {})
        values = list(metric_payload.get("values", []) or [])
        out.append({
            "method_name": item.get("method_name"),
            "display_name": item.get("display_name"),
            "label": item.get("display_name"),
            "values": values,
            "source": metric_payload.get("source"),
            "curve_index": idx,
        })
    return out


def main() -> None:
    args = parse_args()
    metric_keys = _normalize_metric_args(args.metrics)

    compare_items = _build_curve_items(args.compare_dirs, args.compare_labels)
    ablation_items = _build_curve_items(args.ablation_dirs, args.ablation_labels)

    if not compare_items and not ablation_items:
        raise ValueError("未找到任何可用实验。请至少提供 --compare-dirs 或 --ablation-dirs。")

    output_dir = args.output_dir or str(PROJECT_ROOT / "runners" / "results" / "training_curves")
    os.makedirs(output_dir, exist_ok=True)

    for metric_key in metric_keys:
        compare_group = _flatten_group_for_metric(compare_items, metric_key)
        ablation_group = _flatten_group_for_metric(ablation_items, metric_key)

        plot_training_curve_groups(
            compare_curve_items=compare_group,
            ablation_curve_items=ablation_group,
            save_dir=output_dir,
            metric_keys=[metric_key],
            smooth_window=args.smooth_window,
            show_raw=args.show_raw,
            show=args.show,
        )

    print("训练曲线重绘完成")
    print("输出目录：", output_dir)
    print("Compare 实验数：", len(compare_items))
    print("Ablation 实验数：", len(ablation_items))
    print("指标：", ", ".join(metric_keys))


if __name__ == "__main__":
    main()