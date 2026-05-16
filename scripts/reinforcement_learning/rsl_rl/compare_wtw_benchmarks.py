# SPDX-License-Identifier: Apache-2.0
"""Compare and visualize Go2 Walk-These-Ways benchmark summaries.

This script is intentionally offline: it only reads the summary.json files
produced by play_wtw_benchmark.py and does not launch IsaacLab.

Examples:
    python scripts/reinforcement_learning/rsl_rl/compare_wtw_benchmarks.py \
        --inputs logs/rsl_rl/go2_walk_these_ways/*/benchmark_play/*/summary.json \
        --output_dir logs/rsl_rl/go2_walk_these_ways/benchmark_compare/latest

    python scripts/reinforcement_learning/rsl_rl/compare_wtw_benchmarks.py \
        --inputs path/to/a1/summary.json path/to/a2/summary.json \
        --labels A1 A2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


AGG_METRICS = [
    "overall_score",
    "tracking_score",
    "gait_score_v2",
    "height_score",
    "done_rate",
    "vx_rmse",
    "height_rmse",
    "contact_duty_error_mean",
    "contact_freq_ratio",
    "contact_phase_r_mean",
]

CASE_HEATMAP_METRICS = [
    "overall_score",
    "tracking_score",
    "gait_score_v2",
    "height_score",
    "contact_freq_ratio",
    "contact_duty_error_mean",
]

GAIT_NAMES = ("trot", "pace", "bound", "pronk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare WTW benchmark summary.json files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="summary.json files, benchmark output directories, run directories, or glob patterns.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching the resolved summary files. Defaults to run/timestamp-derived labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for CSV, Markdown, and PNG outputs. Defaults to benchmark_compare/<timestamp>.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="overall_score_mean",
        help="Aggregate metric used for ranking in the report.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return parser.parse_args()


def resolve_summary_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        matches = glob.glob(item)
        if not matches:
            matches = [item]
        for match in matches:
            path = Path(match).expanduser().resolve()
            candidates: list[Path]
            if path.is_file():
                candidates = [path]
            elif path.is_dir():
                if (path / "summary.json").is_file():
                    candidates = [path / "summary.json"]
                else:
                    candidates = sorted(path.glob("benchmark_play/*/summary.json"))
            else:
                continue
            for candidate in candidates:
                candidate = candidate.resolve()
                if candidate.name == "summary.json" and candidate not in seen:
                    seen.add(candidate)
                    paths.append(candidate)
    return paths


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _min(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _max(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "benchmark"


def infer_label(path: Path, metadata: dict[str, Any]) -> str:
    checkpoint = str(metadata.get("checkpoint", ""))
    run_name = ""
    if checkpoint:
        parts = Path(checkpoint).parts
        if "go2_walk_these_ways" in parts:
            idx = parts.index("go2_walk_these_ways")
            if idx + 1 < len(parts):
                run_name = parts[idx + 1]
        elif len(parts) >= 2:
            run_name = Path(checkpoint).parent.name
    if not run_name:
        parts = path.parts
        if "go2_walk_these_ways" in parts:
            idx = parts.index("go2_walk_these_ways")
            if idx + 1 < len(parts):
                run_name = parts[idx + 1]
    stamp = path.parent.name
    if run_name:
        return f"{run_name}/{stamp}"
    return path.parent.name


def load_benchmark(path: Path, label: str | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{path} does not contain a list in key 'cases'")
    return {
        "label": label or infer_label(path, metadata),
        "path": str(path),
        "metadata": metadata,
        "cases": cases,
    }


def flatten_cases(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        metadata = benchmark["metadata"]
        for case in benchmark["cases"]:
            row = dict(case)
            row["experiment"] = benchmark["label"]
            row["summary_path"] = benchmark["path"]
            row["checkpoint"] = metadata.get("checkpoint", "")
            row["suite"] = metadata.get("suite", "")
            row["seed"] = metadata.get("seed", "")
            rows.append(row)
    return rows


def aggregate_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    cases = benchmark["cases"]
    row: dict[str, Any] = {
        "experiment": benchmark["label"],
        "summary_path": benchmark["path"],
        "checkpoint": benchmark["metadata"].get("checkpoint", ""),
        "suite": benchmark["metadata"].get("suite", ""),
        "seed": benchmark["metadata"].get("seed", ""),
        "num_cases": len(cases),
    }
    for metric in AGG_METRICS:
        values = [_safe_float(case.get(metric)) for case in cases]
        row[f"{metric}_mean"] = _mean(values)
        row[f"{metric}_min"] = _min(values)
        row[f"{metric}_max"] = _max(values)
    for gait in GAIT_NAMES:
        gait_cases = [case for case in cases if case.get("gait") == gait]
        for metric in ("gait_score_v2", "contact_freq_ratio", "contact_duty_error_mean"):
            row[f"{gait}_{metric}_mean"] = _mean([_safe_float(case.get(metric)) for case in gait_cases])
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "experiment",
        "case",
        "gait",
        "overall_score_mean",
        "tracking_score_mean",
        "gait_score_v2_mean",
        "height_score_mean",
        "done_rate_mean",
        "overall_score",
        "tracking_score",
        "gait_score_v2",
        "height_score",
        "done_rate",
        "contact_freq_ratio",
        "contact_duty_error_mean",
        "summary_path",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PNG plots. Install it or use the CSV/Markdown outputs.") from exc
    return plt


def _fmt(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def plot_aggregate_bars(aggregate_rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    plt = require_matplotlib()
    labels = [row["experiment"] for row in aggregate_rows]
    x = list(range(len(labels)))
    metrics = [
        ("overall_score_mean", "Overall"),
        ("tracking_score_mean", "Velocity"),
        ("gait_score_v2_mean", "Gait"),
        ("height_score_mean", "Height"),
    ]
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.4), 5.2))
    for idx, (metric, name) in enumerate(metrics):
        offsets = [pos + (idx - 1.5) * width for pos in x]
        ax.bar(offsets, [_safe_float(row.get(metric), 0.0) for row in aggregate_rows], width=width, label=name)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Benchmark score comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_scores.png", dpi=dpi)
    plt.close(fig)


def plot_metric_heatmap(case_rows: list[dict[str, Any]], metric: str, output_dir: Path, dpi: int) -> None:
    plt = require_matplotlib()
    experiments = list(dict.fromkeys(str(row["experiment"]) for row in case_rows))
    cases = list(dict.fromkeys(str(row["case"]) for row in case_rows))
    values_by_key = {(str(row["experiment"]), str(row["case"])): _safe_float(row.get(metric)) for row in case_rows}
    matrix = [[values_by_key.get((experiment, case), math.nan) for case in cases] for experiment in experiments]

    fig, ax = plt.subplots(figsize=(max(10, len(cases) * 0.65), max(3.5, len(experiments) * 0.55)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(metric)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=45, ha="right")
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments)
    fig.colorbar(image, ax=ax, shrink=0.85)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if math.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(output_dir / f"case_heatmap_{_slug(metric)}.png", dpi=dpi)
    plt.close(fig)


def plot_gait_breakdown(aggregate_rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    plt = require_matplotlib()
    labels = [row["experiment"] for row in aggregate_rows]
    x = list(range(len(labels)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.4), 5.0))
    for idx, gait in enumerate(GAIT_NAMES):
        metric = f"{gait}_gait_score_v2_mean"
        offsets = [pos + (idx - 1.5) * width for pos in x]
        ax.bar(offsets, [_safe_float(row.get(metric), 0.0) for row in aggregate_rows], width=width, label=gait)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("gait_score_v2")
    ax.set_title("Gait-specific score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "gait_score_breakdown.png", dpi=dpi)
    plt.close(fig)


def plot_height_response(case_rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    plt = require_matplotlib()
    height_rows = [
        row
        for row in case_rows
        if math.isfinite(_safe_float(row.get("target_height"))) and math.isfinite(_safe_float(row.get("height_mean")))
    ]
    if not height_rows:
        return
    experiments = list(dict.fromkeys(str(row["experiment"]) for row in height_rows))
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for experiment in experiments:
        rows = [row for row in height_rows if row["experiment"] == experiment]
        ax.scatter(
            [_safe_float(row.get("target_height")) for row in rows],
            [_safe_float(row.get("height_mean")) for row in rows],
            label=experiment,
            alpha=0.85,
        )
    all_values = [
        value
        for row in height_rows
        for value in (_safe_float(row.get("target_height")), _safe_float(row.get("height_mean")))
        if math.isfinite(value)
    ]
    lo = min(all_values)
    hi = max(all_values)
    pad = max(0.02, (hi - lo) * 0.08)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1.0, alpha=0.55)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("target_height")
    ax.set_ylabel("height_mean")
    ax.set_title("Height command response")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "height_response.png", dpi=dpi)
    plt.close(fig)


def write_report(path: Path, aggregate_rows: list[dict[str, Any]], sort_by: str) -> None:
    ranked = sorted(aggregate_rows, key=lambda row: _safe_float(row.get(sort_by), -math.inf), reverse=True)
    lines = [
        "# WTW Benchmark Compare",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- ranked_by: `{sort_by}`",
        f"- experiments: `{len(aggregate_rows)}`",
        "",
        "## Ranking",
        "",
        "| rank | experiment | overall | track | gait_v2 | height | done | freq_ratio | duty_err |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{rank} | {row['experiment']} | {_fmt(row.get('overall_score_mean'))} | "
            f"{_fmt(row.get('tracking_score_mean'))} | {_fmt(row.get('gait_score_v2_mean'))} | "
            f"{_fmt(row.get('height_score_mean'))} | {_fmt(row.get('done_rate_mean'))} | "
            f"{_fmt(row.get('contact_freq_ratio_mean'))} | {_fmt(row.get('contact_duty_error_mean_mean'))} |"
        )
    lines += [
        "",
        "## Gait Breakdown",
        "",
        "| experiment | trot | pace | bound | pronk |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            "| "
            f"{row['experiment']} | {_fmt(row.get('trot_gait_score_v2_mean'))} | "
            f"{_fmt(row.get('pace_gait_score_v2_mean'))} | {_fmt(row.get('bound_gait_score_v2_mean'))} | "
            f"{_fmt(row.get('pronk_gait_score_v2_mean'))} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `aggregate.csv`: 每个实验一行的均值、最小值、最大值。",
        "- `cases_long.csv`: 每个实验每个 case 一行，用于进一步筛选。",
        "- `aggregate_scores.png`: 总分、速度、高度、步态分项对比。",
        "- `gait_score_breakdown.png`: 不同步态的 `gait_score_v2` 对比。",
        "- `height_response.png`: 高度命令目标值和实际均值的散点图。",
        "- `case_heatmap_*.png`: 每个 case 的关键指标热力图。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_paths = resolve_summary_paths(args.inputs)
    if not summary_paths:
        raise SystemExit("No summary.json files were found from --inputs.")
    if args.labels and len(args.labels) != len(summary_paths):
        raise SystemExit(f"--labels count ({len(args.labels)}) must match resolved summaries ({len(summary_paths)}).")

    labels = args.labels or [None] * len(summary_paths)
    benchmarks = [load_benchmark(path, label) for path, label in zip(summary_paths, labels)]
    duplicate_labels = {label for label in [b["label"] for b in benchmarks] if [b["label"] for b in benchmarks].count(label) > 1}
    if duplicate_labels:
        for idx, benchmark in enumerate(benchmarks):
            if benchmark["label"] in duplicate_labels:
                benchmark["label"] = f"{benchmark['label']}#{idx + 1}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"benchmark_compare/{timestamp}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = [aggregate_benchmark(benchmark) for benchmark in benchmarks]
    case_rows = flatten_cases(benchmarks)
    sort_by = args.sort_by
    if sort_by not in aggregate_rows[0]:
        sort_by = "overall_score_mean"

    aggregate_rows = sorted(aggregate_rows, key=lambda row: _safe_float(row.get(sort_by), -math.inf), reverse=True)
    write_csv(output_dir / "aggregate.csv", aggregate_rows)
    write_csv(output_dir / "cases_long.csv", case_rows)
    write_report(output_dir / "report.md", aggregate_rows, sort_by)

    plot_aggregate_bars(aggregate_rows, output_dir, args.dpi)
    plot_gait_breakdown(aggregate_rows, output_dir, args.dpi)
    plot_height_response(case_rows, output_dir, args.dpi)
    for metric in CASE_HEATMAP_METRICS:
        plot_metric_heatmap(case_rows, metric, output_dir, args.dpi)

    print(f"[INFO] Loaded {len(benchmarks)} benchmark summaries.")
    for row in aggregate_rows:
        print(
            "[RESULT] "
            f"{row['experiment']}: overall={_fmt(row.get('overall_score_mean'))} "
            f"track={_fmt(row.get('tracking_score_mean'))} gait={_fmt(row.get('gait_score_v2_mean'))} "
            f"height={_fmt(row.get('height_score_mean'))} done={_fmt(row.get('done_rate_mean'))}"
        )
    print(f"[INFO] Wrote comparison outputs to: {output_dir}")


if __name__ == "__main__":
    main()
