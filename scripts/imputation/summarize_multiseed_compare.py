from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


METRICS = ["loss", "mae", "rmse", "pearson", "r2"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize two multiseed imputation experiment directories.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/imputation_multiseed"))
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _seed_from_name(path: Path) -> int:
    return int(path.name.split("_")[-1])


def _load_split_metrics(exp_dir: Path, split: str) -> Dict[int, Dict[str, float]]:
    by_seed: Dict[int, Dict[str, float]] = {}
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
    for seed_dir in sorted(exp_dir.glob("seed_*"), key=_seed_from_name):
        summary_path = seed_dir / "metrics_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if split not in payload:
            continue
        by_seed[_seed_from_name(seed_dir)] = {metric: float(payload[split][metric]) for metric in METRICS}
    return by_seed


def _mean_var(rows: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    if not rows:
        return {metric: {"mean": float("nan"), "var": float("nan")} for metric in METRICS}
    summary: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "var": float(values.var()),
        }
    return summary


def _render_table(title: str, baseline_name: str, candidate_name: str, baseline_rows: Dict[int, Dict[str, float]], candidate_rows: Dict[int, Dict[str, float]]) -> List[str]:
    baseline_stats = _mean_var(baseline_rows.values())
    candidate_stats = _mean_var(candidate_rows.values())
    lines = [f"## {title}", "", f"| metric | {baseline_name} mean | {baseline_name} var | {candidate_name} mean | {candidate_name} var | delta mean ({candidate_name} - {baseline_name}) | delta var |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for metric in METRICS:
        base_mean = baseline_stats[metric]["mean"]
        cand_mean = candidate_stats[metric]["mean"]
        base_var = baseline_stats[metric]["var"]
        cand_var = candidate_stats[metric]["var"]
        lines.append(
            f"| {metric} | {base_mean:.6f} | {base_var:.6f} | {cand_mean:.6f} | {cand_var:.6f} | {cand_mean - base_mean:.6f} | {cand_var - base_var:.6f} |"
        )
    lines.append("")
    return lines


def _render_seed_table(baseline_name: str, candidate_name: str, baseline_rows: Dict[int, Dict[str, float]], candidate_rows: Dict[int, Dict[str, float]]) -> List[str]:
    seeds = sorted(set(baseline_rows) & set(candidate_rows))
    lines = [
        "## 逐 seed Test 对照",
        "",
        f"| seed | {baseline_name} mae | {candidate_name} mae | delta mae | {baseline_name} rmse | {candidate_name} rmse | delta rmse | {baseline_name} r2 | {candidate_name} r2 | delta r2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in seeds:
        base = baseline_rows[seed]
        cand = candidate_rows[seed]
        lines.append(
            f"| {seed} | {base['mae']:.6f} | {cand['mae']:.6f} | {cand['mae'] - base['mae']:.6f} | {base['rmse']:.6f} | {cand['rmse']:.6f} | {cand['rmse'] - base['rmse']:.6f} | {base['r2']:.6f} | {cand['r2']:.6f} | {cand['r2'] - base['r2']:.6f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    args = _parse_args()
    baseline_dir = args.runs_root / args.baseline
    candidate_dir = args.runs_root / args.candidate

    baseline_test = _load_split_metrics(baseline_dir, "test")
    candidate_test = _load_split_metrics(candidate_dir, "test")
    baseline_best_val = _load_split_metrics(baseline_dir, "best_val")
    candidate_best_val = _load_split_metrics(candidate_dir, "best_val")

    shared_test_seeds = sorted(set(baseline_test) & set(candidate_test))
    shared_best_val_seeds = sorted(set(baseline_best_val) & set(candidate_best_val))

    lines: List[str] = ["# Multiseed Imputation Compare", ""]
    lines.append(f"- baseline: `{args.baseline}`")
    lines.append(f"- candidate: `{args.candidate}`")
    lines.append(f"- shared test seeds: `{shared_test_seeds}`")
    lines.append(f"- shared best-val seeds: `{shared_best_val_seeds}`")
    lines.append("")

    lines.extend(
        _render_table(
            title="Test 指标：均值与方差",
            baseline_name=args.baseline,
            candidate_name=args.candidate,
            baseline_rows={seed: baseline_test[seed] for seed in shared_test_seeds},
            candidate_rows={seed: candidate_test[seed] for seed in shared_test_seeds},
        )
    )
    lines.extend(
        _render_table(
            title="Best validation 指标：均值与方差",
            baseline_name=args.baseline,
            candidate_name=args.candidate,
            baseline_rows={seed: baseline_best_val[seed] for seed in shared_best_val_seeds},
            candidate_rows={seed: candidate_best_val[seed] for seed in shared_best_val_seeds},
        )
    )
    lines.extend(
        _render_seed_table(
            baseline_name=args.baseline,
            candidate_name=args.candidate,
            baseline_rows={seed: baseline_test[seed] for seed in shared_test_seeds},
            candidate_rows={seed: candidate_test[seed] for seed in shared_test_seeds},
        )
    )

    output_text = "\n".join(lines)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
