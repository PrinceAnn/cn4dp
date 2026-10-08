#!/usr/bin/env python
"""Evaluate standard tabular imputation baselines for IDP reconstruction.

The script fits sklearn imputers on the part2 training split only, applies the
same synthetic masks used by the neural diagnostics, and evaluates only the
artificially masked observed cells on the held-out split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenotype_encoder.data import PreprocessArtifacts, build_preprocess_pipeline, group_feature_columns_by_organ
from scripts.downstream.utils import split_indices
from scripts.imputation.evaluate_imputation_diagnostics import (
    _collect_metrics,
    _eval_mask,
    _load_table,
    _load_yaml,
    _scenario_to_masking,
)


DEFAULT_SEEDS = [7, 17, 27]
DEFAULT_SCENARIOS = ["mixedmask", "random70"]
DEFAULT_METHODS = ["knn", "iterative_bayes", "iterative_rf"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sklearn tabular imputation baselines")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo/imputation.yaml"),
        help="Part2 imputation config whose data/split settings are reused.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/imputation_tabular_baselines_pilot"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenarios.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods: knn,iterative_bayes,iterative_rf.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--n-nearest-features", type=int, default=80)
    parser.add_argument("--rf-n-estimators", type=int, default=30)
    parser.add_argument("--rf-max-depth", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="If positive, fit sklearn imputers on a deterministic subsample of train rows.",
    )
    parser.add_argument(
        "--clip-to-train-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip imputed values to per-feature min/max observed in the imputer fitting rows.",
    )
    return parser.parse_args()


def _build_imputer(method: str, args: argparse.Namespace, seed: int):
    method = str(method).lower()
    if method == "knn":
        from sklearn.impute import KNNImputer

        return KNNImputer(n_neighbors=int(args.knn_neighbors), weights="distance")

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    if method == "iterative_bayes":
        from sklearn.linear_model import BayesianRidge

        return IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=int(args.iterative_max_iter),
            n_nearest_features=int(args.n_nearest_features),
            initial_strategy="median",
            random_state=int(seed),
            skip_complete=True,
        )

    if method == "iterative_rf":
        from sklearn.ensemble import RandomForestRegressor

        estimator = RandomForestRegressor(
            n_estimators=int(args.rf_n_estimators),
            max_depth=int(args.rf_max_depth),
            min_samples_leaf=5,
            n_jobs=int(args.n_jobs),
            random_state=int(seed),
        )
        return IterativeImputer(
            estimator=estimator,
            max_iter=int(args.iterative_max_iter),
            n_nearest_features=int(args.n_nearest_features),
            initial_strategy="median",
            random_state=int(seed),
            skip_complete=True,
        )

    raise ValueError(f"Unsupported method: {method}")


def _standardize(raw: np.ndarray, preprocess: PreprocessArtifacts) -> np.ndarray:
    mean = np.asarray(preprocess.mean_, dtype=np.float32)
    std = np.asarray(preprocess.std_, dtype=np.float32)
    values = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=mean, posinf=mean, neginf=mean)
    return (values - mean) / std


def _tabular_predictions(
    *,
    imputer,
    raw_features: np.ndarray,
    eval_indices: np.ndarray,
    eval_mask: np.ndarray,
    preprocess: PreprocessArtifacts,
    clip_min: np.ndarray | None,
    clip_max: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_raw = raw_features[eval_indices].copy()
    mask_matrix = eval_mask[eval_indices].astype(bool)
    eval_raw[mask_matrix] = np.nan
    pred_raw = imputer.transform(eval_raw).astype(np.float32)
    if clip_min is not None and clip_max is not None:
        pred_raw = np.clip(pred_raw, clip_min[None, :], clip_max[None, :]).astype(np.float32)
    target_raw = raw_features[eval_indices].astype(np.float32)
    pred_std = _standardize(pred_raw, preprocess)
    target_std = _standardize(target_raw, preprocess)
    return pred_std, target_std, mask_matrix


def _write_summary(
    *,
    out_dir: Path,
    seed_df: pd.DataFrame,
    organ_df: pd.DataFrame,
    win_rows: List[dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Tabular Imputation Baselines",
        "",
        f"Config: `{args.config}`",
        f"Split: `{args.split}`",
        f"Seeds: `{args.seeds}`",
        f"Scenarios: `{args.scenarios}`",
        f"Methods: `{args.methods}`",
        "",
        "## Overall Metrics",
        "",
        "| scenario | method | seeds | fit rows | MAE | std MAE | RMSE | NRMSE | R2 | fit seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (scenario, method), group in seed_df.groupby(["scenario", "method"], sort=True):
        lines.append(
            "| {scenario} | {method} | {seeds} | {fit_rows:.0f} | {mae:.6f} | {std_mae:.6f} | {rmse:.6f} | {nrmse:.6f} | {r2:.6f} | {fit_seconds:.2f} |".format(
                scenario=scenario,
                method=method,
                seeds=int(group["seed"].nunique()),
                fit_rows=float(group["fit_rows"].mean()) if "fit_rows" in group.columns else float("nan"),
                mae=float(group["mae"].mean()),
                std_mae=float(group["std_mae"].mean()),
                rmse=float(group["rmse"].mean()),
                nrmse=float(group["nrmse"].mean()),
                r2=float(group["r2"].mean()),
                fit_seconds=float(group["fit_seconds"].mean()),
            )
        )

    lines.extend(["", "## Per-Feature Win Rates", ""])
    lines.append("| scenario | comparison | seeds | win rate | mean delta MAE |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in win_rows:
        lines.append(
            "| {scenario} | {comparison} | {seeds} | {win_rate:.4f} | {mean_delta_mae:.6f} |".format(**row)
        )

    lines.extend(["", "## Organ-Wise Mean NRMSE", ""])
    lines.append("| scenario | method | organ | NRMSE | std MAE | raw MAE |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    organ_summary = organ_df.groupby(["scenario", "method", "organ"], as_index=False)[["nrmse", "std_mae", "mae"]].mean()
    for _, row in organ_summary.sort_values(["scenario", "method", "organ"]).iterrows():
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['organ']} | {row['nrmse']:.6f} | {row['std_mae']:.6f} | {row['mae']:.6f} |"
        )

    (out_dir / "summary_tables.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "config": str(args.config),
        "split": args.split,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "methods": args.methods,
        "outputs": {
            "seed_metrics": "seed_metrics.csv",
            "organ_metrics": "organ_metrics.csv",
            "feature_metrics": "feature_metrics.csv",
            "feature_win_rates": "feature_win_rates.csv",
            "summary": "summary_tables.md",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _pairwise_feature_win_rates(feature_df: pd.DataFrame, methods: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    method_list = [str(method) for method in methods]
    comparisons = [(left, right) for left in method_list for right in method_list if left < right]
    for scenario in sorted(feature_df["scenario"].unique()):
        scenario_df = feature_df[feature_df["scenario"] == scenario]
        for left, right in comparisons:
            left_df = scenario_df[scenario_df["method"] == left]
            right_df = scenario_df[scenario_df["method"] == right]
            merged = left_df.merge(
                right_df,
                on=["seed", "scenario", "feature"],
                suffixes=("_left", "_right"),
            )
            if merged.empty:
                continue
            delta = merged["mae_left"].to_numpy(dtype=np.float64) - merged["mae_right"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "scenario": scenario,
                    "comparison": f"{left} - {right}",
                    "seeds": int(merged["seed"].nunique()),
                    "n_feature_seed_pairs": int(len(merged)),
                    "win_rate": float(np.mean(delta < 0.0)),
                    "mean_delta_mae": float(np.mean(delta)),
                    "median_delta_mae": float(np.median(delta)),
                }
            )
    return rows


def main() -> None:
    args = _parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    scenarios = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    methods = [value.strip().lower() for value in args.methods.split(",") if value.strip()]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(args.config)
    merged, feature_cols, raw_features, observed_mask = _load_table(cfg)
    organ_groups = group_feature_columns_by_organ(feature_cols)
    organ_to_indices = {
        organ: np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
        for organ, columns in organ_groups.items()
    }

    seed_rows: List[dict] = []
    organ_rows: List[dict] = []
    feature_rows: List[dict] = []

    for seed in seeds:
        train_idx, val_idx, test_idx = split_indices(
            len(merged),
            seed=seed,
            train_ratio=float(cfg.get("train_ratio", 0.8)),
            val_ratio=float(cfg.get("val_ratio", 0.1)),
        )
        eval_idx = val_idx if args.split == "val" else test_idx
        preprocess = build_preprocess_pipeline()
        preprocess.fit(raw_features[train_idx])
        preprocess_artifacts = preprocess.to_artifacts()

        scenario_masks: Dict[str, np.ndarray] = {}
        for scenario in scenarios:
            scenario_masks[scenario] = _eval_mask(
                observed_mask=observed_mask,
                indices=eval_idx,
                masking_cfg=_scenario_to_masking(scenario),
                organ_to_indices=organ_to_indices,
                seed=seed + (23 if args.split == "val" else 37),
            )

        fit_idx = train_idx
        if int(args.max_train_rows) > 0 and len(train_idx) > int(args.max_train_rows):
            rng = np.random.default_rng(int(seed) + 101)
            fit_idx = np.sort(rng.choice(train_idx, size=int(args.max_train_rows), replace=False))
        train_raw = raw_features[fit_idx].astype(np.float32, copy=True)
        clip_min = None
        clip_max = None
        if bool(args.clip_to_train_range):
            clip_min = np.nanmin(train_raw, axis=0).astype(np.float32)
            clip_max = np.nanmax(train_raw, axis=0).astype(np.float32)
            global_min = float(np.nanmin(train_raw)) if np.isfinite(np.nanmin(train_raw)) else 0.0
            global_max = float(np.nanmax(train_raw)) if np.isfinite(np.nanmax(train_raw)) else 0.0
            clip_min = np.nan_to_num(clip_min, nan=global_min, posinf=global_min, neginf=global_min)
            clip_max = np.nan_to_num(clip_max, nan=global_max, posinf=global_max, neginf=global_max)
        for method in methods:
            imputer = _build_imputer(method, args, seed)
            start = time.perf_counter()
            imputer.fit(train_raw)
            fit_seconds = time.perf_counter() - start

            for scenario in scenarios:
                pred_std, target_std, mask_matrix = _tabular_predictions(
                    imputer=imputer,
                    raw_features=raw_features,
                    eval_indices=eval_idx,
                    eval_mask=scenario_masks[scenario],
                    preprocess=preprocess_artifacts,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                overall, organs, features, _ = _collect_metrics(
                    pred_std_matrix=pred_std,
                    target_std_matrix=target_std,
                    mask_matrix=mask_matrix,
                    preprocess=preprocess_artifacts,
                    feature_cols=feature_cols,
                    seed=seed,
                    scenario=scenario,
                    method=method,
                    split=args.split,
                )
                overall["fit_seconds"] = float(fit_seconds)
                overall["fit_rows"] = int(len(fit_idx))
                seed_rows.append(overall)
                organ_rows.extend(organs)
                feature_rows.extend(features)
                print(
                    f"done seed={seed} scenario={scenario} method={method} "
                    f"mae={overall['mae']:.4f} nrmse={overall['nrmse']:.4f} "
                    f"fit_rows={len(fit_idx)} fit_s={fit_seconds:.1f}",
                    flush=True,
                )

    seed_df = pd.DataFrame(seed_rows)
    organ_df = pd.DataFrame(organ_rows)
    feature_df = pd.DataFrame(feature_rows)
    win_rows = _pairwise_feature_win_rates(feature_df, methods)

    seed_df.to_csv(out_dir / "seed_metrics.csv", index=False)
    organ_df.to_csv(out_dir / "organ_metrics.csv", index=False)
    feature_df.to_csv(out_dir / "feature_metrics.csv", index=False)
    pd.DataFrame(win_rows).to_csv(out_dir / "feature_win_rates.csv", index=False)
    _write_summary(out_dir=out_dir, seed_df=seed_df, organ_df=organ_df, win_rows=win_rows, args=args)


if __name__ == "__main__":
    main()
