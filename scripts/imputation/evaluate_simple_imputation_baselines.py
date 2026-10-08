#!/usr/bin/env python
"""Evaluate mean/median imputation baselines under the imputation masking setup.

This is an evaluation-only helper. It reuses the same cohort, feature table,
train/val/test split, and masking policy as `train_idp_imputation.py`, then
computes simple per-feature statistics from the train split and evaluates them
on masked val/test cells.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenotype_encoder.data import group_feature_columns_by_organ
from scripts.downstream.utils import split_indices
from scripts.imputation.train_idp_imputation import (
    _align_feature_columns_for_pretrained,
    _load_allowed_eids,
    _load_pretrained_distill_ckpt,
    _regression_metrics,
    _sample_loss_mask,
)


DEFAULT_SEEDS = [7, 17, 27, 37, 42, 47, 57, 67, 77, 87]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mean/median imputation baselines")
    parser.add_argument("--config", type=Path, required=True, help="Imputation YAML whose data/masking setup to reuse")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for JSON/Markdown summary")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated split/mask seeds",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="mean,median",
        help="Comma-separated methods: mean,median",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_table_from_config(cfg: dict) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    eid_col = str(cfg.get("eid_col", "eid"))
    imaging_date_col = str(cfg.get("imaging_date_col", "imaging_date"))
    phenotype_csv = Path(cfg["phenotype_csv"])
    cohort_csv = cfg.get("cohort_csv", None)
    allowed_eids_csv = cfg.get("allowed_eids_csv", None)

    phenotype_df = pd.read_csv(phenotype_csv)
    if eid_col not in phenotype_df.columns:
        raise ValueError(f"phenotype_csv must contain column {eid_col}")
    phenotype_df[eid_col] = phenotype_df[eid_col].astype(int)

    if cohort_csv is not None:
        cohort_df = pd.read_csv(Path(cohort_csv))
        if eid_col not in cohort_df.columns or imaging_date_col not in cohort_df.columns:
            raise ValueError(f"cohort_csv must contain columns {eid_col} and {imaging_date_col}")
        cohort_df[eid_col] = cohort_df[eid_col].astype(int)
        cohort_df[imaging_date_col] = pd.to_datetime(cohort_df[imaging_date_col], errors="coerce")
        merged = cohort_df[[eid_col, imaging_date_col]].merge(phenotype_df, on=eid_col, how="inner")
    else:
        if imaging_date_col not in phenotype_df.columns:
            raise ValueError("Either cohort_csv is required or phenotype_csv must include imaging_date")
        phenotype_df[imaging_date_col] = pd.to_datetime(phenotype_df[imaging_date_col], errors="coerce")
        merged = phenotype_df.copy()

    if allowed_eids_csv is not None:
        allowed_eids = set(_load_allowed_eids(Path(allowed_eids_csv), eid_col=eid_col).tolist())
        merged = merged[merged[eid_col].isin(allowed_eids)].reset_index(drop=True)

    exclude = {eid_col, imaging_date_col}
    feature_cols = [column for column in merged.columns if column not in exclude]

    pretrained_cfg = cfg.get("pretrained", {}) or {}
    pretrained_ckpt_path = pretrained_cfg.get("ckpt_path", None)
    if pretrained_ckpt_path is not None:
        _, pretrained_feature_names, _ = _load_pretrained_distill_ckpt(Path(pretrained_ckpt_path))
        feature_cols, _ = _align_feature_columns_for_pretrained(
            current_feature_cols=feature_cols,
            pretrained_feature_names=pretrained_feature_names,
        )

    raw_features = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)
    observed_mask = np.isfinite(raw_features)
    min_observed_features = int(cfg.get("min_observed_features", 2))
    eligible = observed_mask.sum(axis=1) >= min_observed_features
    merged = merged.loc[eligible].reset_index(drop=True)
    raw_features = raw_features[eligible]
    observed_mask = observed_mask[eligible]
    eids = merged[eid_col].to_numpy(dtype=np.int64, copy=True)
    return merged, feature_cols, raw_features, observed_mask


def _feature_stats(train_raw: np.ndarray, method: str) -> np.ndarray:
    method = str(method).lower()
    if method == "mean":
        values = np.nanmean(train_raw, axis=0)
    elif method == "median":
        values = np.nanmedian(train_raw, axis=0)
    else:
        raise ValueError(f"Unsupported method={method!r}")

    # Extremely defensive fallback for all-missing columns.
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        fallback = np.nanmean(train_raw)
        if not np.isfinite(fallback):
            fallback = 0.0
        values = np.nan_to_num(values, nan=float(fallback), posinf=float(fallback), neginf=float(fallback))
    return values


def _standardization(train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train_raw, axis=0).astype(np.float32)
    std = np.nanstd(train_raw, axis=0).astype(np.float32)
    fallback_mean = float(np.nanmean(train_raw)) if np.isfinite(np.nanmean(train_raw)) else 0.0
    mean = np.nan_to_num(mean, nan=fallback_mean, posinf=fallback_mean, neginf=fallback_mean)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std <= 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _build_eval_mask(
    *,
    observed_mask: np.ndarray,
    indices: np.ndarray,
    masking_cfg: dict,
    organ_to_indices: Dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    eval_mask = np.zeros_like(observed_mask, dtype=bool)

    mixture_cfg = masking_cfg.get("mixture", None)
    if mixture_cfg is not None:
        if not isinstance(mixture_cfg, list) or not mixture_cfg:
            raise ValueError("masking.mixture must be a non-empty list")
        masking_specs = [dict(item) for item in mixture_cfg]
        weights = np.asarray([float(item.get("weight", 1.0)) for item in masking_specs], dtype=np.float64)
        weights = weights / float(weights.sum())
    else:
        masking_specs = [dict(masking_cfg)]
        weights = np.asarray([1.0], dtype=np.float64)

    for row_index in indices.tolist():
        spec_index = int(rng.choice(np.arange(len(masking_specs)), p=weights))
        spec = masking_specs[spec_index]
        fixed_organ = spec.get("organ", None)
        mask = _sample_loss_mask(
            observed_mask=observed_mask[row_index].astype(bool, copy=False),
            mode=str(spec.get("mode", "random")).lower(),
            ratio=float(spec.get("ratio", 0.3)),
            organ_to_indices=organ_to_indices,
            rng=rng,
            fixed_organ=str(fixed_organ) if fixed_organ is not None else None,
        )
        eval_mask[row_index] = mask
    return eval_mask


def _evaluate_method(
    *,
    raw_features: np.ndarray,
    train_indices: np.ndarray,
    eval_mask: np.ndarray,
    method: str,
) -> Dict[str, float]:
    stats = _feature_stats(raw_features[train_indices], method)
    mean, std = _standardization(raw_features[train_indices])

    rows, cols = np.where(eval_mask)
    target_raw = raw_features[rows, cols].astype(np.float32)
    pred_raw = stats[cols].astype(np.float32)

    target_std = (target_raw - mean[cols]) / std[cols]
    pred_std = (pred_raw - mean[cols]) / std[cols]
    loss = float(np.mean((pred_std - target_std) ** 2)) if target_std.size else float("nan")

    metrics = {"loss": loss, "n_masked_values": float(target_raw.size)}
    metrics.update(_regression_metrics(pred_raw, target_raw))
    return metrics


def _aggregate(items: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({key for item in items for key in item.keys()})
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(item[key]) for item in items if key in item], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = {
            "mean": float(values.mean()) if values.size else float("nan"),
            "var": float(values.var(ddof=0)) if values.size else float("nan"),
        }
    return out


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Simple Imputation Baselines",
        "",
        f"Config: `{summary['config']}`",
        f"Seeds: `{', '.join(str(seed) for seed in summary['seeds'])}`",
        "",
        "## Test Mean/Variance",
        "",
        "| method | loss mean | loss var | mae mean | mae var | rmse mean | rmse var | pearson mean | r2 mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, data in summary["methods"].items():
        agg = data["test_aggregate"]
        lines.append(
            "| {method} | {loss_mean:.6f} | {loss_var:.6f} | {mae_mean:.6f} | {mae_var:.6f} | "
            "{rmse_mean:.6f} | {rmse_var:.6f} | {pearson_mean:.6f} | {r2_mean:.6f} |".format(
                method=method,
                loss_mean=agg["loss"]["mean"],
                loss_var=agg["loss"]["var"],
                mae_mean=agg["mae"]["mean"],
                mae_var=agg["mae"]["var"],
                rmse_mean=agg["rmse"]["mean"],
                rmse_var=agg["rmse"]["var"],
                pearson_mean=agg["pearson"]["mean"],
                r2_mean=agg["r2"]["mean"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    cfg = _load_yaml(args.config)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    methods = [value.strip().lower() for value in args.methods.split(",") if value.strip()]

    _, feature_cols, raw_features, observed_mask = _load_table_from_config(cfg)
    organ_groups = group_feature_columns_by_organ(feature_cols)
    organ_to_indices = {
        organ_name: np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
        for organ_name, columns in organ_groups.items()
    }
    masking_cfg = cfg.get("masking", {}) or {}

    summary: Dict[str, Any] = {
        "config": str(args.config),
        "seeds": seeds,
        "methods": {},
        "n_rows": int(raw_features.shape[0]),
        "n_features": int(raw_features.shape[1]),
    }

    for method in methods:
        val_metrics = []
        test_metrics = []
        per_seed = []
        for seed in seeds:
            train_idx, val_idx, test_idx = split_indices(
                raw_features.shape[0],
                seed=int(seed),
                train_ratio=float(cfg.get("train_ratio", 0.8)),
                val_ratio=float(cfg.get("val_ratio", 0.1)),
            )
            val_mask = _build_eval_mask(
                observed_mask=observed_mask,
                indices=val_idx,
                masking_cfg=masking_cfg,
                organ_to_indices=organ_to_indices,
                seed=int(seed) + 23,
            )
            test_mask = _build_eval_mask(
                observed_mask=observed_mask,
                indices=test_idx,
                masking_cfg=masking_cfg,
                organ_to_indices=organ_to_indices,
                seed=int(seed) + 37,
            )
            val = _evaluate_method(raw_features=raw_features, train_indices=train_idx, eval_mask=val_mask, method=method)
            test = _evaluate_method(raw_features=raw_features, train_indices=train_idx, eval_mask=test_mask, method=method)
            val_metrics.append(val)
            test_metrics.append(test)
            per_seed.append({"seed": int(seed), "val": val, "test": test})

        summary["methods"][method] = {
            "per_seed": per_seed,
            "val_aggregate": _aggregate(val_metrics),
            "test_aggregate": _aggregate(test_metrics),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "simple_imputation_baselines_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_markdown(args.out_dir / "simple_imputation_baselines_summary.md", summary)
    print(f"Wrote summary to {args.out_dir}")


if __name__ == "__main__":
    main()
