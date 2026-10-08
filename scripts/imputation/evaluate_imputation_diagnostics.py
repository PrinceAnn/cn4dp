#!/usr/bin/env python
"""Detailed reconstruction diagnostics for trained IDP imputers.

This script evaluates frozen part2 imputation checkpoints under multiple
synthetic missingness scenarios and writes overall, organ-wise, and feature-wise
metrics. It is intended for paper diagnostics, not for model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenotype_encoder.data import PreprocessArtifacts, group_feature_columns_by_organ
from phenotype_encoder.model import build_encoder_from_config
from scripts.downstream.utils import load_delphi_checkpoint, split_indices
from scripts.imputation.train_idp_imputation import (
    FrozenTrajectoryEncoder,
    ImputationDataset,
    TrajectoryConditionedImputer,
    _build_pre_imaging_sequences,
    _collate_left_pad,
    _load_allowed_eids,
    _load_pretrained_distill_ckpt,
    _regression_metrics,
    _sample_loss_mask,
)


DEFAULT_SEEDS = [7, 17, 27, 37, 42, 47, 57, 67, 77, 87]
DEFAULT_SCENARIOS = ["random30", "random50", "random70", "organ_block", "mixedmask"]
DEFAULT_METHODS = ["median", "idp_imputer", "traj_organ_imputer"]
DEFAULT_CKPT_TEMPLATE = "runs/imputation_multiseed_part2_mixedmask/{model}/seed_{seed}/best.pt"
METHOD_MODEL_DIRS = {
    "idp_imputer": "part2_mixedmask_distill_init",
    "traj_organ_imputer": "part2_mixedmask_traj_organ_residual",
    "traj_organ_shuffled": "part2_mixedmask_traj_organ_residual",
    "traj_organ_disable_full_mask": "part2_mixedmask_traj_organ_residual",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detailed imputation diagnostics")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo/imputation.yaml"),
        help="Part2 imputation config whose cohort/split settings are reused.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/imputation_diagnostics"),
        help="Output directory for CSV/Markdown diagnostics.",
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
        help="Comma-separated scenarios: random30,random50,random70,organ_block,mixedmask.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods: median,idp_imputer,traj_organ_imputer,traj_organ_shuffled.",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default=DEFAULT_CKPT_TEMPLATE,
        help="Template for neural checkpoints. Fields: {model}, {seed}.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Evaluation split.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _scenario_to_masking(scenario: str) -> dict:
    scenario = str(scenario).lower()
    if scenario.startswith("random"):
        ratio = float(scenario.replace("random", "")) / 100.0
        return {"mode": "random", "ratio": ratio}
    if scenario == "organ_block":
        return {"mode": "organ_block"}
    if scenario == "mixedmask":
        return {
            "mixture": [
                {"mode": "random", "ratio": 0.3, "weight": 1.0},
                {"mode": "random", "ratio": 0.5, "weight": 1.0},
                {"mode": "random", "ratio": 0.7, "weight": 1.0},
                {"mode": "organ_block", "weight": 1.0},
            ]
        }
    raise ValueError(f"Unsupported scenario: {scenario}")


def _load_table(cfg: dict, feature_cols: Sequence[str] | None = None) -> tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray]:
    eid_col = str(cfg.get("eid_col", "eid"))
    imaging_date_col = str(cfg.get("imaging_date_col", "imaging_date"))
    phenotype_df = pd.read_csv(Path(cfg["phenotype_csv"]))
    if eid_col not in phenotype_df.columns:
        raise ValueError(f"phenotype_csv must contain {eid_col!r}")
    phenotype_df[eid_col] = phenotype_df[eid_col].astype(int)

    cohort_csv = cfg.get("cohort_csv", None)
    if cohort_csv is not None:
        cohort_df = pd.read_csv(Path(cohort_csv))
        cohort_df[eid_col] = cohort_df[eid_col].astype(int)
        cohort_df[imaging_date_col] = pd.to_datetime(cohort_df[imaging_date_col], errors="coerce")
        merged = cohort_df[[eid_col, imaging_date_col]].merge(phenotype_df, on=eid_col, how="inner")
    else:
        if imaging_date_col not in phenotype_df.columns:
            raise ValueError("Either cohort_csv is required or phenotype_csv must include imaging_date")
        phenotype_df[imaging_date_col] = pd.to_datetime(phenotype_df[imaging_date_col], errors="coerce")
        merged = phenotype_df.copy()

    allowed_eids_csv = cfg.get("allowed_eids_csv", None)
    if allowed_eids_csv is not None:
        allowed = set(_load_allowed_eids(Path(allowed_eids_csv), eid_col=eid_col).tolist())
        merged = merged[merged[eid_col].isin(allowed)].reset_index(drop=True)

    if feature_cols is None:
        exclude = {eid_col, imaging_date_col}
        feature_cols = [column for column in merged.columns if column not in exclude]
    else:
        missing = [column for column in feature_cols if column not in merged.columns]
        if missing:
            raise ValueError(f"Missing feature columns in phenotype table: {missing[:5]}")
        feature_cols = list(feature_cols)

    raw_features = merged[list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    observed_mask = np.isfinite(raw_features)
    eligible = observed_mask.sum(axis=1) >= int(cfg.get("min_observed_features", 2))
    merged = merged.loc[eligible].reset_index(drop=True)
    raw_features = raw_features[eligible]
    observed_mask = observed_mask[eligible]
    return merged, list(feature_cols), raw_features, observed_mask


def _preprocess_from_checkpoint(ckpt: dict) -> PreprocessArtifacts:
    preprocess = ckpt["preprocess"]
    return PreprocessArtifacts(
        mean_=np.asarray(preprocess["mean_"], dtype=np.float32),
        std_=np.asarray(preprocess["std_"], dtype=np.float32),
    )


def _transform_with_artifacts(raw: np.ndarray, preprocess: PreprocessArtifacts) -> np.ndarray:
    mean = np.asarray(preprocess.mean_, dtype=np.float32)
    std = np.asarray(preprocess.std_, dtype=np.float32)
    x = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=mean, posinf=mean, neginf=mean)
    return (x - mean) / std


def _feature_stats(train_raw: np.ndarray, method: str) -> np.ndarray:
    if method != "median":
        raise ValueError(f"Only median is supported here, got {method}")
    values = np.nanmedian(train_raw, axis=0).astype(np.float32)
    fallback = float(np.nanmedian(train_raw))
    if not np.isfinite(fallback):
        fallback = 0.0
    return np.nan_to_num(values, nan=fallback, posinf=fallback, neginf=fallback)


def _eval_mask(
    *,
    observed_mask: np.ndarray,
    indices: np.ndarray,
    masking_cfg: dict,
    organ_to_indices: Dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros_like(observed_mask, dtype=bool)
    mixture = masking_cfg.get("mixture", None)
    if mixture is None:
        specs = [dict(masking_cfg)]
        weights = np.asarray([1.0], dtype=np.float64)
    else:
        specs = [dict(item) for item in mixture]
        weights = np.asarray([float(item.get("weight", 1.0)) for item in specs], dtype=np.float64)
        weights = weights / float(weights.sum())

    for row_index in indices.tolist():
        spec = specs[int(rng.choice(np.arange(len(specs)), p=weights))]
        fixed_organ = spec.get("organ", None)
        out[row_index] = _sample_loss_mask(
            observed_mask=observed_mask[row_index].astype(bool, copy=False),
            mode=str(spec.get("mode", "random")).lower(),
            ratio=float(spec.get("ratio", 0.3)),
            organ_to_indices=organ_to_indices,
            rng=rng,
            fixed_organ=str(fixed_organ) if fixed_organ is not None else None,
        )
    return out


def _build_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[TrajectoryConditionedImputer, FrozenTrajectoryEncoder | None, dict, List[str], PreprocessArtifacts]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    feature_cols = list(ckpt["feature_cols"])
    preprocess = _preprocess_from_checkpoint(ckpt)

    pretrained_cfg = cfg.get("pretrained", {}) or {}
    pretrained_ckpt_path = pretrained_cfg.get("ckpt_path", None)
    if pretrained_ckpt_path is not None:
        distill_ckpt, _, _ = _load_pretrained_distill_ckpt(Path(pretrained_ckpt_path))
        distill_cfg = distill_ckpt.get("config", {}) or {}
        encoder_cfg = distill_cfg.get("encoder", {}) or {}
    else:
        encoder_cfg = cfg.get("encoder", {}) or {}

    encoder, encoder_dim = build_encoder_from_config(encoder_cfg=encoder_cfg, feature_cols=feature_cols)
    use_trajectory = bool(cfg.get("use_trajectory", True))
    trajectory_fusion = str(cfg.get("trajectory_fusion", "concat" if use_trajectory else "none")).lower()
    trajectory_encoder: FrozenTrajectoryEncoder | None = None
    trajectory_dim = 0
    if use_trajectory:
        delphi_model, _ = load_delphi_checkpoint(Path(cfg.get("delphi_ckpt_dir", "runs/demo/teacher")), device=torch.device("cpu"))
        trajectory_encoder = FrozenTrajectoryEncoder(
            delphi_model,
            pooling=str(cfg.get("teacher_pooling", "sum")),
            freeze_delphi=bool(cfg.get("freeze_delphi", True)),
            se_reduction=int(cfg.get("teacher_senet_reduction", 16)),
            se_dropout=float(cfg.get("teacher_senet_dropout", 0.0)),
            temperature=float(cfg.get("teacher_senet_temperature", 1.0)),
        )
        if ckpt.get("trajectory_pooler_state", None) is not None:
            trajectory_encoder.pooler.load_state_dict(ckpt["trajectory_pooler_state"], strict=True)
        trajectory_dim = int(trajectory_encoder.output_dim)

    organ_groups = group_feature_columns_by_organ(feature_cols)
    organ_to_indices = {
        organ: np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
        for organ, columns in organ_groups.items()
    }
    model = TrajectoryConditionedImputer(
        encoder=encoder,
        encoder_dim=int(encoder_dim),
        feature_dim=len(feature_cols),
        use_trajectory=use_trajectory,
        trajectory_dim=trajectory_dim,
        trajectory_fusion=trajectory_fusion,
        organ_to_indices=organ_to_indices,
        decoder_hidden_dims=[int(value) for value in cfg.get("decoder_hidden_dims", [512, 256])],
        dropout=float(cfg.get("decoder_dropout", 0.1)),
        act=str(cfg.get("decoder_act", "relu")),
    )
    model.load_state_dict(ckpt["imputer_state"], strict=True)
    model.to(device).eval()
    if trajectory_encoder is not None:
        trajectory_encoder.to(device).eval()
    return model, trajectory_encoder, cfg, feature_cols, preprocess


def _neural_predictions(
    *,
    model: TrajectoryConditionedImputer,
    trajectory_encoder: FrozenTrajectoryEncoder | None,
    cfg: dict,
    standardized: np.ndarray,
    observed_mask: np.ndarray,
    eids: np.ndarray,
    token_list: Sequence[torch.Tensor],
    age_list: Sequence[torch.Tensor],
    indices: np.ndarray,
    masking_cfg: dict,
    organ_to_indices: Dict[str, np.ndarray],
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    shuffle_trajectory: bool = False,
    residual_policy: str = "normal",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scripts.imputation.train_idp_imputation import _build_samples

    method_token_list = token_list
    method_age_list = age_list
    if shuffle_trajectory:
        rng = np.random.default_rng(int(seed) + 104729)
        source = rng.permutation(indices)
        for _ in range(100):
            if not np.any(source == indices):
                break
            source = rng.permutation(indices)
        if np.any(source == indices):
            source = np.roll(indices, 1)
        shuffled_tokens = list(token_list)
        shuffled_ages = list(age_list)
        for target_idx, source_idx in zip(indices.tolist(), source.tolist()):
            shuffled_tokens[target_idx] = token_list[source_idx]
            shuffled_ages[target_idx] = age_list[source_idx]
        method_token_list = shuffled_tokens
        method_age_list = shuffled_ages

    samples = _build_samples(
        standardized_features=standardized,
        observed_mask=observed_mask,
        eids=eids,
        token_list=method_token_list,
        age_list=method_age_list,
        indices=indices,
        masking_cfg=masking_cfg,
        organ_to_indices=organ_to_indices,
        seed=seed,
    )
    loader = DataLoader(
        ImputationDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_left_pad,
    )

    pred_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    use_trajectory = bool(cfg.get("use_trajectory", True))
    model.residual_policy = str(residual_policy)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            observed = batch["observed_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            input_missing = (~observed) | loss_mask
            trajectory = None
            if use_trajectory and trajectory_encoder is not None:
                trajectory = trajectory_encoder(batch["tokens"].to(device), batch["ages_days"].to(device))
            pred = model(x, input_missing, trajectory)
            pred_chunks.append(pred.detach().cpu().numpy().astype(np.float32))
            target_chunks.append(x.detach().cpu().numpy().astype(np.float32))
            mask_chunks.append(loss_mask.detach().cpu().numpy().astype(bool))

    if not pred_chunks:
        feature_dim = standardized.shape[1]
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, feature_dim), dtype=bool),
        )
    return np.concatenate(pred_chunks), np.concatenate(target_chunks), np.concatenate(mask_chunks)


def _median_predictions(
    *,
    raw_features: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    eval_mask: np.ndarray,
    preprocess: PreprocessArtifacts,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fill = _feature_stats(raw_features[train_indices], "median")
    target_raw = np.nan_to_num(raw_features[eval_indices], nan=preprocess.mean_, posinf=preprocess.mean_, neginf=preprocess.mean_)
    pred_raw = np.broadcast_to(fill[None, :], target_raw.shape).astype(np.float32)
    pred_std = (pred_raw - preprocess.mean_[None, :]) / preprocess.std_[None, :]
    target_std = (target_raw - preprocess.mean_[None, :]) / preprocess.std_[None, :]
    return pred_std.astype(np.float32), target_std.astype(np.float32), eval_mask[eval_indices].astype(bool)


def _safe_r2(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    denom = float(np.sum((target - float(np.mean(target))) ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.sum((pred - target) ** 2)) / denom


def _collect_metrics(
    *,
    pred_std_matrix: np.ndarray,
    target_std_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    preprocess: PreprocessArtifacts,
    feature_cols: Sequence[str],
    seed: int,
    scenario: str,
    method: str,
    split: str,
) -> tuple[dict, List[dict], List[dict], Dict[str, float]]:
    pred_raw = pred_std_matrix * preprocess.std_[None, :] + preprocess.mean_[None, :]
    target_raw = target_std_matrix * preprocess.std_[None, :] + preprocess.mean_[None, :]
    mask = mask_matrix.astype(bool)
    pred_values = pred_raw[mask]
    target_values = target_raw[mask]
    pred_std_values = pred_std_matrix[mask]
    target_std_values = target_std_matrix[mask]
    error_std = pred_std_values - target_std_values
    raw_metrics = _regression_metrics(pred_values, target_values)
    std_mae = float(np.mean(np.abs(error_std))) if error_std.size else float("nan")
    std_rmse = float(np.sqrt(np.mean(error_std**2))) if error_std.size else float("nan")
    overall = {
        "seed": int(seed),
        "scenario": scenario,
        "method": method,
        "split": split,
        "n_masked_values": int(mask.sum()),
        "mae": raw_metrics["mae"],
        "rmse": raw_metrics["rmse"],
        "pearson": raw_metrics["pearson"],
        "r2": raw_metrics["r2"],
        "std_mae": std_mae,
        "std_rmse": std_rmse,
        "nrmse": std_rmse,
    }

    feature_rows: List[dict] = []
    feature_mae: Dict[str, float] = {}
    for feature_idx, feature in enumerate(feature_cols):
        fm = mask[:, feature_idx]
        if not np.any(fm):
            continue
        p_raw = pred_raw[fm, feature_idx]
        t_raw = target_raw[fm, feature_idx]
        p_std = pred_std_matrix[fm, feature_idx]
        t_std = target_std_matrix[fm, feature_idx]
        mae = float(np.mean(np.abs(p_raw - t_raw)))
        rmse = float(np.sqrt(np.mean((p_raw - t_raw) ** 2)))
        std_mae_f = float(np.mean(np.abs(p_std - t_std)))
        std_rmse_f = float(np.sqrt(np.mean((p_std - t_std) ** 2)))
        feature_mae[str(feature)] = mae
        feature_rows.append(
            {
                "seed": int(seed),
                "scenario": scenario,
                "method": method,
                "split": split,
                "feature": str(feature),
                "organ": str(feature).split("__", 1)[0] if "__" in str(feature) else "global",
                "n_masked_values": int(fm.sum()),
                "mae": mae,
                "rmse": rmse,
                "r2": _safe_r2(p_raw, t_raw),
                "std_mae": std_mae_f,
                "std_rmse": std_rmse_f,
                "nrmse": std_rmse_f,
            }
        )

    organ_groups = group_feature_columns_by_organ(list(feature_cols))
    organ_rows: List[dict] = []
    for organ, columns in organ_groups.items():
        indices = np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
        om = mask[:, indices]
        if not np.any(om):
            continue
        p_raw = pred_raw[:, indices][om]
        t_raw = target_raw[:, indices][om]
        p_std = pred_std_matrix[:, indices][om]
        t_std = target_std_matrix[:, indices][om]
        raw = _regression_metrics(p_raw, t_raw)
        err_std = p_std - t_std
        organ_rows.append(
            {
                "seed": int(seed),
                "scenario": scenario,
                "method": method,
                "split": split,
                "organ": organ,
                "n_features": int(len(indices)),
                "n_masked_values": int(om.sum()),
                "mae": raw["mae"],
                "rmse": raw["rmse"],
                "r2": raw["r2"],
                "std_mae": float(np.mean(np.abs(err_std))),
                "std_rmse": float(np.sqrt(np.mean(err_std**2))),
                "nrmse": float(np.sqrt(np.mean(err_std**2))),
            }
        )

    return overall, organ_rows, feature_rows, feature_mae


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size <= 1:
        return mean, float("nan"), float("nan")
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _write_summary(
    *,
    out_dir: Path,
    seed_df: pd.DataFrame,
    organ_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    win_rows: List[dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Imputation Reconstruction Diagnostics",
        "",
        f"Config: `{args.config}`",
        f"Split: `{args.split}`",
        f"Seeds: `{args.seeds}`",
        f"Scenarios: `{args.scenarios}`",
        f"Methods: `{args.methods}`",
        "",
        "## Overall Metrics",
        "",
        "| scenario | method | seeds | MAE | std MAE | RMSE | NRMSE | R2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (scenario, method), group in seed_df.groupby(["scenario", "method"], sort=True):
        row = {"scenario": scenario, "method": method, "seeds": int(group["seed"].nunique())}
        for metric in ["mae", "std_mae", "rmse", "nrmse", "r2"]:
            row[metric] = float(group[metric].mean())
        lines.append(
            "| {scenario} | {method} | {seeds} | {mae:.6f} | {std_mae:.6f} | {rmse:.6f} | {nrmse:.6f} | {r2:.6f} |".format(
                **row
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
    meta = {
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
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _feature_win_rates(feature_df: pd.DataFrame) -> List[dict]:
    rows: List[dict] = []
    comparisons = [
        ("traj_organ_imputer", "idp_imputer"),
        ("traj_organ_imputer", "median"),
        ("idp_imputer", "median"),
    ]
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
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    cfg_for_table = _load_yaml(args.config)
    neural_ckpt_cache: Dict[Tuple[str, int], tuple[TrajectoryConditionedImputer, FrozenTrajectoryEncoder | None, dict, List[str], PreprocessArtifacts]] = {}
    sequence_cache: Dict[int, tuple[np.ndarray, Sequence[torch.Tensor], Sequence[torch.Tensor]]] = {}
    seed_rows: List[dict] = []
    organ_rows: List[dict] = []
    feature_rows: List[dict] = []

    for seed in seeds:
        reference_ckpt_path = Path(
            args.checkpoint_template.format(model=METHOD_MODEL_DIRS["idp_imputer"], seed=seed)
        )
        reference_ckpt = torch.load(reference_ckpt_path, map_location="cpu", weights_only=False)
        feature_cols = list(reference_ckpt["feature_cols"])
        preprocess = _preprocess_from_checkpoint(reference_ckpt)
        merged, feature_cols, raw_features, observed_mask = _load_table(cfg_for_table, feature_cols=feature_cols)
        train_idx, val_idx, test_idx = split_indices(
            len(merged),
            seed=seed,
            train_ratio=float(cfg_for_table.get("train_ratio", 0.8)),
            val_ratio=float(cfg_for_table.get("val_ratio", 0.1)),
        )
        eval_idx = val_idx if args.split == "val" else test_idx
        standardized = _transform_with_artifacts(raw_features, preprocess)
        if seed not in sequence_cache:
            eid_col = str(cfg_for_table.get("eid_col", "eid"))
            imaging_date_col = str(cfg_for_table.get("imaging_date_col", "imaging_date"))
            eids = merged[eid_col].to_numpy(dtype=np.int64, copy=True)
            token_list, age_list = _build_pre_imaging_sequences(
                eids=eids,
                imaging_date=merged[imaging_date_col].to_numpy(copy=True),
                disease_onset_csv=Path(cfg_for_table.get("disease_onset_csv", "data/demo/disease_onsets.csv")),
                basic_csv=Path(cfg_for_table.get("basic_csv", "data/demo/basic_info.csv")),
                labels_csv=Path(cfg_for_table.get("delphi_labels_csv", "Delphi/data/demo/labels.csv")),
                eid_col=eid_col,
                block_size=int(cfg_for_table.get("block_size", 256)),
                append_query_token=bool(cfg_for_table.get("append_query_token", True)),
            )
            sequence_cache[seed] = (eids, token_list, age_list)
        eids, token_list, age_list = sequence_cache[seed]
        organ_groups = group_feature_columns_by_organ(feature_cols)
        organ_to_indices = {
            organ: np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
            for organ, columns in organ_groups.items()
        }

        for scenario in scenarios:
            masking_cfg = _scenario_to_masking(scenario)
            mask_seed = seed + (23 if args.split == "val" else 37)
            eval_mask = _eval_mask(
                observed_mask=observed_mask,
                indices=eval_idx,
                masking_cfg=masking_cfg,
                organ_to_indices=organ_to_indices,
                seed=mask_seed,
            )
            for method in methods:
                method_preprocess = preprocess
                if method == "median":
                    pred_std, target_std, mask_matrix = _median_predictions(
                        raw_features=raw_features,
                        train_indices=train_idx,
                        eval_indices=eval_idx,
                        eval_mask=eval_mask,
                        preprocess=preprocess,
                    )
                else:
                    if method not in METHOD_MODEL_DIRS:
                        raise ValueError(f"Unknown neural method: {method}")
                    cache_key = (method, seed)
                    if cache_key not in neural_ckpt_cache:
                        ckpt_path = Path(args.checkpoint_template.format(model=METHOD_MODEL_DIRS[method], seed=seed))
                        neural_ckpt_cache[cache_key] = _build_model_from_checkpoint(ckpt_path, device)
                    model, trajectory_encoder, ckpt_cfg, ckpt_feature_cols, ckpt_preprocess = neural_ckpt_cache[cache_key]
                    if ckpt_feature_cols != feature_cols:
                        raise ValueError(f"Feature mismatch for {method} seed={seed}")
                    method_preprocess = ckpt_preprocess
                    pred_std, target_std, mask_matrix = _neural_predictions(
                        model=model,
                        trajectory_encoder=trajectory_encoder,
                        cfg=ckpt_cfg,
                        standardized=standardized,
                        observed_mask=observed_mask,
                        eids=eids,
                        token_list=token_list,
                        age_list=age_list,
                        indices=eval_idx,
                        masking_cfg=masking_cfg,
                        organ_to_indices=organ_to_indices,
                        seed=mask_seed,
                        device=device,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        shuffle_trajectory=(method == "traj_organ_shuffled"),
                        residual_policy=(
                            "disable_full_organ_mask"
                            if method == "traj_organ_disable_full_mask"
                            else "normal"
                        ),
                    )

                overall, organs, features, _ = _collect_metrics(
                    pred_std_matrix=pred_std,
                    target_std_matrix=target_std,
                    mask_matrix=mask_matrix,
                    preprocess=method_preprocess,
                    feature_cols=feature_cols,
                    seed=seed,
                    scenario=scenario,
                    method=method,
                    split=args.split,
                )
                seed_rows.append(overall)
                organ_rows.extend(organs)
                feature_rows.extend(features)
                print(
                    f"done seed={seed} scenario={scenario} method={method} "
                    f"mae={overall['mae']:.4f} nrmse={overall['nrmse']:.4f}",
                    flush=True,
                )

    seed_df = pd.DataFrame(seed_rows)
    organ_df = pd.DataFrame(organ_rows)
    feature_df = pd.DataFrame(feature_rows)
    win_rows = _feature_win_rates(feature_df)

    seed_df.to_csv(out_dir / "seed_metrics.csv", index=False)
    organ_df.to_csv(out_dir / "organ_metrics.csv", index=False)
    feature_df.to_csv(out_dir / "feature_metrics.csv", index=False)
    pd.DataFrame(win_rows).to_csv(out_dir / "feature_win_rates.csv", index=False)
    _write_summary(out_dir=out_dir, seed_df=seed_df, organ_df=organ_df, feature_df=feature_df, win_rows=win_rows, args=args)


if __name__ == "__main__":
    main()
