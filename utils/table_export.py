from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Sequence, Tuple


TABLE_COLUMNS = [
    "method_name",
    "final_coverage_ratio",
    "final_covered_users",
    "full_coverage_success_rate",
    "mean_total_move_distance",
    "mean_mean_overlap_users_step",
    "mean_episode_length",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda x: (
            -float(x.get("final_coverage_ratio", 0.0)),
            -float(x.get("full_coverage_success_rate", 0.0)),
            float(x.get("mean_total_move_distance", 1e18)),
        ),
    )


def _best_flags(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, str], bool]:
    if len(rows) == 0:
        return {}

    best_map: Dict[Tuple[int, str], bool] = {}

    max_coverage = max(float(r["final_coverage_ratio"]) for r in rows)
    max_covered = max(float(r["final_covered_users"]) for r in rows)
    max_success = max(float(r["full_coverage_success_rate"]) for r in rows)
    min_move = min(float(r["mean_total_move_distance"]) for r in rows)
    min_overlap = min(float(r["mean_mean_overlap_users_step"]) for r in rows)
    min_len = min(float(r["mean_episode_length"]) for r in rows)

    for i, r in enumerate(rows):
        best_map[(i, "final_coverage_ratio")] = float(r["final_coverage_ratio"]) == max_coverage
        best_map[(i, "final_covered_users")] = float(r["final_covered_users"]) == max_covered
        best_map[(i, "full_coverage_success_rate")] = float(r["full_coverage_success_rate"]) == max_success
        best_map[(i, "mean_total_move_distance")] = float(r["mean_total_move_distance"]) == min_move
        best_map[(i, "mean_mean_overlap_users_step")] = float(r["mean_mean_overlap_users_step"]) == min_overlap
        best_map[(i, "mean_episode_length")] = float(r["mean_episode_length"]) == min_len

    return best_map


def build_table_rows(
    method_aggregates: Dict[str, Dict[str, Any]],
    include_methods: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method_name, agg in method_aggregates.items():
        if include_methods is not None and method_name not in include_methods:
            continue
        rows.append({
            "method_name": method_name,
            "final_coverage_ratio": float(agg.get("mean_final_coverage_ratio", 0.0)),
            "final_covered_users": float(agg.get("mean_final_covered_users", 0.0)),
            "full_coverage_success_rate": float(agg.get("full_coverage_success_rate", 0.0)),
            "mean_total_move_distance": float(agg.get("mean_total_move_distance", 0.0)),
            "mean_mean_overlap_users_step": float(agg.get("mean_mean_overlap_users_step", 0.0)),
            "mean_episode_length": float(agg.get("mean_episode_length", 0.0)),
        })
    return _sort_rows(rows)


def export_table_csv(rows: List[Dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_table_markdown(rows: List[Dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    best_map = _best_flags(rows)

    header = "| " + " | ".join(TABLE_COLUMNS) + " |\n"
    sep = "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |\n"

    lines = [header, sep]
    for i, row in enumerate(rows):
        vals = []
        for col in TABLE_COLUMNS:
            val = row[col]
            if col == "method_name":
                cell = str(val)
            else:
                cell = f"{float(val):.4f}"
            if (i, col) in best_map and best_map[(i, col)] and col != "method_name":
                cell = f"**{cell}**"
            vals.append(cell)
        lines.append("| " + " | ".join(vals) + " |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def export_table_latex(rows: List[Dict[str, Any]], path: str, caption: str = "", label: str = "tab:results") -> None:
    _ensure_dir(os.path.dirname(path))
    best_map = _best_flags(rows)

    col_spec = "l" + "c" * (len(TABLE_COLUMNS) - 1)
    lines = []
    lines.append("\\begin{table}[t]\n")
    lines.append("\\centering\n")
    lines.append(f"\\caption{{{caption}}}\n")
    lines.append(f"\\label{{{label}}}\n")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}\n")
    lines.append("\\hline\n")
    lines.append(" & ".join(TABLE_COLUMNS) + " \\\\\n")
    lines.append("\\hline\n")

    for i, row in enumerate(rows):
        cells = []
        for col in TABLE_COLUMNS:
            val = row[col]
            if col == "method_name":
                cell = str(val).replace("_", "\\_")
            else:
                cell = f"{float(val):.4f}"
                if best_map.get((i, col), False):
                    cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\\n")

    lines.append("\\hline\n")
    lines.append("\\end{tabular}\n")
    lines.append("\\end{table}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)