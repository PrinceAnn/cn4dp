#!/usr/bin/env python
"""Export downstream-ready imputed phenotype features from an imputation checkpoint.

This script loads an existing IDP imputation checkpoint, rebuilds the same input
cohort, runs inference, and writes a phenotype CSV with `eid` plus imputed
feature columns. It is designed to bridge the current imputation runs to the
existing downstream scripts without retraining the imputer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenotype_encoder.data import PreprocessArtifacts, group_feature_columns_by_organ
from phenotype_encoder.model import build_encoder_from_config
from scripts.downstream.utils import load_delphi_checkpoint, seed_everything, split_indices
from scripts.imputation.train_idp_imputation import (
    FrozenTrajectoryEncoder,
    ImputationDataset,
    ImputationSample,
    TrajectoryConditionedImputer,
    _build_pre_imaging_sequences,
    _collate_left_pad,
    _configure_logging,
    _device_from_cfg,
    _load_allowed_eids,
    _load_pretrained_distill_ckpt,
    logger,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export imputed phenotype CSV from a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt from imputation training")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to write the exported phenotype CSV")
    parser.add_argument(
        "--phenotype-csv",
        type=Path,
        default=None,
        help="Optional phenotype CSV override for applying a checkpoint to an external cohort",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=None,
        help="Optional cohort CSV override with eid and imaging_date for the external phenotype CSV",
    )
    parser.add_argument(
        "--allowed-eids-csv",
        type=Path,
        default=None,
        help="Optional allowed eid CSV override. Use 'none' behavior by omitting this argument.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which imputation split to export (default: all)",
    )
    parser.add_argument(
        "--fill-mode",
        type=str,
        default="missing_only",
        choices=["missing_only", "full_reconstruction"],
        help="Whether to fill only originally-missing cells or replace every feature with the model reconstruction",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size (default: 256)")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke tests / debugging")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cpu or cuda:0 (default: use checkpoint config)",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint type at {path}: {type(checkpoint)}")
    return checkpoint


def _load_preprocess_from_checkpoint(checkpoint: Dict[str, Any]) -> PreprocessArtifacts:
    preprocess = checkpoint.get("preprocess", None)
    if not isinstance(preprocess, dict) or "mean_" not in preprocess or "std_" not in preprocess:
        raise ValueError("Checkpoint is missing preprocess statistics")
    return PreprocessArtifacts(
        mean_=np.asarray(preprocess["mean_"], dtype=np.float32),
        std_=np.asarray(preprocess["std_"], dtype=np.float32),
    )


def _load_merged_table(
    *,
    cfg: Dict[str, Any],
    feature_cols: Sequence[str],
    max_rows: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
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
        if cohort_df[imaging_date_col].isna().any():
            raise ValueError(f"Found invalid {imaging_date_col} values in cohort_csv")
        merged = cohort_df[[eid_col, imaging_date_col]].merge(phenotype_df, on=eid_col, how="inner")
    else:
        if imaging_date_col not in phenotype_df.columns:
            raise ValueError(
                "Either provide cohort_csv with imaging dates or include imaging_date_col in phenotype_csv"
            )
        phenotype_df[imaging_date_col] = pd.to_datetime(phenotype_df[imaging_date_col], errors="coerce")
        if phenotype_df[imaging_date_col].isna().any():
            raise ValueError(f"Found invalid {imaging_date_col} values in phenotype_csv")
        merged = phenotype_df.copy()

    if allowed_eids_csv is not None:
        allowed_eids = set(_load_allowed_eids(Path(allowed_eids_csv), eid_col=eid_col).tolist())
        merged = merged[merged[eid_col].isin(allowed_eids)].reset_index(drop=True)

    if max_rows is not None:
        merged = merged.iloc[: int(max_rows)].copy()
    elif cfg.get("max_samples", None) is not None:
        merged = merged.iloc[: int(cfg["max_samples"])].copy()

    missing_features = [column for column in feature_cols if column not in merged.columns]
    if missing_features:
        raise ValueError(
            "Merged cohort is missing features required by the checkpoint. "
            f"Missing {len(missing_features)} columns, e.g. {missing_features[:10]}"
        )

    raw_features = merged[list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    observed_mask = np.isfinite(raw_features)

    min_observed_features = int(cfg.get("min_observed_features", 2))
    eligible = observed_mask.sum(axis=1) >= min_observed_features
    if not np.any(eligible):
        raise ValueError("No rows have enough observed features for export")

    merged = merged.loc[eligible].reset_index(drop=True)
    raw_features = raw_features[eligible]
    observed_mask = observed_mask[eligible]

    eids = merged[eid_col].to_numpy(dtype=np.int64, copy=True)
    imaging_dates = merged[imaging_date_col].to_numpy(copy=True)
    return merged, eids, imaging_dates, raw_features, observed_mask, eid_col, imaging_date_col


def _select_indices(n_rows: int, cfg: Dict[str, Any], split_name: str) -> np.ndarray:
    if split_name == "all":
        return np.arange(n_rows, dtype=np.int64)

    train_idx, val_idx, test_idx = split_indices(
        n_rows,
        seed=int(cfg.get("seed", 42)),
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
    )
    by_split = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    return np.asarray(by_split[split_name], dtype=np.int64)


def _build_inference_samples(
    *,
    standardized_features: np.ndarray,
    observed_mask: np.ndarray,
    eids: np.ndarray,
    token_list: Sequence[torch.Tensor],
    age_list: Sequence[torch.Tensor],
    indices: np.ndarray,
) -> list[ImputationSample]:
    samples: list[ImputationSample] = []
    for row_index in indices.tolist():
        samples.append(
            ImputationSample(
                eid=torch.tensor(int(eids[row_index]), dtype=torch.long),
                x=torch.from_numpy(standardized_features[row_index].astype(np.float32)),
                observed_mask=torch.from_numpy(observed_mask[row_index].astype(np.bool_)),
                loss_mask=torch.zeros_like(torch.from_numpy(observed_mask[row_index].astype(np.bool_))),
                tokens=token_list[row_index],
                ages_days=age_list[row_index],
            )
        )
    return samples


def _build_model(
    *,
    checkpoint: Dict[str, Any],
    cfg: Dict[str, Any],
    feature_cols: Sequence[str],
    device: torch.device,
) -> tuple[TrajectoryConditionedImputer, FrozenTrajectoryEncoder | None, bool]:
    pretrained_cfg = cfg.get("pretrained", {}) or {}
    pretrained_ckpt_path = pretrained_cfg.get("ckpt_path", None)

    if pretrained_ckpt_path is not None:
        distill_ckpt, _, _ = _load_pretrained_distill_ckpt(Path(pretrained_ckpt_path))
        distill_cfg = (distill_ckpt.get("config", {}) or {})
        encoder_cfg = (distill_cfg.get("encoder", {}) or {})
    else:
        encoder_cfg = cfg.get("encoder", None)
        if encoder_cfg is None:
            raise ValueError("Missing required encoder config")

    encoder, encoder_dim = build_encoder_from_config(encoder_cfg=encoder_cfg, feature_cols=list(feature_cols))

    organ_groups = group_feature_columns_by_organ(list(feature_cols))
    organ_to_indices = {
        organ_name: np.asarray([list(feature_cols).index(column) for column in columns], dtype=np.int64)
        for organ_name, columns in organ_groups.items()
    }

    use_trajectory = bool(cfg.get("use_trajectory", True))
    trajectory_encoder: FrozenTrajectoryEncoder | None = None
    trajectory_dim = 0
    if use_trajectory:
        delphi_ckpt_dir = Path(cfg.get("delphi_ckpt_dir", "runs/demo/teacher"))
        delphi_model, _ = load_delphi_checkpoint(delphi_ckpt_dir, device=torch.device("cpu"))
        trajectory_encoder = FrozenTrajectoryEncoder(
            delphi_model,
            pooling=str(cfg.get("teacher_pooling", "sum")),
            freeze_delphi=bool(cfg.get("freeze_delphi", True)),
            se_reduction=int(cfg.get("teacher_senet_reduction", 16)),
            se_dropout=float(cfg.get("teacher_senet_dropout", 0.0)),
            temperature=float(cfg.get("teacher_senet_temperature", 1.0)),
        )
        trajectory_dim = int(trajectory_encoder.output_dim)

    model = TrajectoryConditionedImputer(
        encoder=encoder,
        encoder_dim=int(encoder_dim),
        feature_dim=len(feature_cols),
        use_trajectory=use_trajectory,
        trajectory_dim=trajectory_dim,
        trajectory_fusion=str(cfg.get("trajectory_fusion", "concat" if use_trajectory else "none")).lower(),
        organ_to_indices=organ_to_indices,
        decoder_hidden_dims=[int(value) for value in cfg.get("decoder_hidden_dims", [512, 256])],
        dropout=float(cfg.get("decoder_dropout", 0.1)),
        act=str(cfg.get("decoder_act", "relu")),
    )
    model.load_state_dict(checkpoint["imputer_state"], strict=True)
    model.to(device)
    model.eval()

    if trajectory_encoder is not None:
        pooler_state = checkpoint.get("trajectory_pooler_state", None)
        if pooler_state is not None:
            trajectory_encoder.pooler.load_state_dict(pooler_state, strict=True)
        trajectory_encoder.to(device)
        trajectory_encoder.eval()

    return model, trajectory_encoder, use_trajectory


def _inverse_standardize(x: np.ndarray, preprocess: PreprocessArtifacts) -> np.ndarray:
    return x * preprocess.std_[None, :] + preprocess.mean_[None, :]


def _standardize_with_checkpoint_stats(x: np.ndarray, preprocess: PreprocessArtifacts) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=preprocess.mean_, posinf=preprocess.mean_, neginf=preprocess.mean_)
    return (x - preprocess.mean_) / preprocess.std_


def _run_export(
    *,
    model: TrajectoryConditionedImputer,
    trajectory_encoder: FrozenTrajectoryEncoder | None,
    use_trajectory: bool,
    samples: Sequence[ImputationSample],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataloader = DataLoader(
        ImputationDataset(samples),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
        collate_fn=_collate_left_pad,
    )

    pred_batches: list[np.ndarray] = []
    eid_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            observed_mask = batch["observed_mask"].to(device)
            tokens = batch["tokens"].to(device)
            ages_days = batch["ages_days"].to(device)
            input_missing_mask = ~observed_mask

            trajectory = trajectory_encoder(tokens, ages_days) if (use_trajectory and trajectory_encoder is not None) else None
            pred = model(x, input_missing_mask, trajectory)

            pred_batches.append(pred.detach().cpu().numpy())
            eid_batches.append(batch["eid"].detach().cpu().numpy())

    return np.concatenate(pred_batches, axis=0), np.concatenate(eid_batches, axis=0)


def main() -> None:
    args = _parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    cfg = dict(checkpoint.get("config", {}) or {})
    if not cfg:
        raise ValueError("Checkpoint is missing the original config")

    if args.phenotype_csv is not None:
        cfg["phenotype_csv"] = str(args.phenotype_csv)
    if args.cohort_csv is not None:
        cfg["cohort_csv"] = str(args.cohort_csv)
    if args.allowed_eids_csv is not None:
        cfg["allowed_eids_csv"] = str(args.allowed_eids_csv)

    _configure_logging(str(cfg.get("log_level", "INFO")))
    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))

    device = torch.device(args.device) if args.device else _device_from_cfg(cfg)
    feature_cols = list(checkpoint.get("feature_cols", []) or [])
    if not feature_cols:
        raise ValueError("Checkpoint is missing feature_cols")
    preprocess = _load_preprocess_from_checkpoint(checkpoint)

    merged, eids, imaging_dates, raw_features, observed_mask, eid_col, _ = _load_merged_table(
        cfg=cfg,
        feature_cols=feature_cols,
        max_rows=args.max_rows,
    )
    export_indices = _select_indices(len(merged), cfg, args.split)
    if export_indices.size == 0:
        raise ValueError(f"Split {args.split!r} contains no rows")

    standardized_features = _standardize_with_checkpoint_stats(raw_features, preprocess)
    if not np.isfinite(standardized_features).all():
        raise ValueError("Found NaN/Inf after applying checkpoint preprocessing")

    token_list, age_list = _build_pre_imaging_sequences(
        eids=eids,
        imaging_date=imaging_dates,
        disease_onset_csv=Path(cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv")),
        basic_csv=Path(cfg.get("basic_csv", "data/demo/basic_info.csv")),
        labels_csv=Path(cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv")),
        eid_col=eid_col,
        block_size=int(cfg.get("block_size", 256)),
        append_query_token=bool(cfg.get("append_query_token", True)),
    )

    model, trajectory_encoder, use_trajectory = _build_model(
        checkpoint=checkpoint,
        cfg=cfg,
        feature_cols=feature_cols,
        device=device,
    )

    samples = _build_inference_samples(
        standardized_features=standardized_features,
        observed_mask=observed_mask,
        eids=eids,
        token_list=token_list,
        age_list=age_list,
        indices=export_indices,
    )
    pred_std, pred_eids = _run_export(
        model=model,
        trajectory_encoder=trajectory_encoder,
        use_trajectory=use_trajectory,
        samples=samples,
        batch_size=int(args.batch_size),
        device=device,
    )

    selected_eids = eids[export_indices]
    if pred_std.shape[0] != selected_eids.shape[0] or not np.array_equal(pred_eids.astype(np.int64), selected_eids.astype(np.int64)):
        raise RuntimeError("EID order mismatch during export")

    pred_raw = _inverse_standardize(pred_std.astype(np.float32, copy=False), preprocess)
    selected_raw = raw_features[export_indices].copy()
    selected_observed = observed_mask[export_indices]

    if args.fill_mode == "missing_only":
        output_raw = selected_raw
        output_raw[~selected_observed] = pred_raw[~selected_observed]
    else:
        output_raw = pred_raw

    output_df = pd.DataFrame(output_raw, columns=feature_cols)
    output_df.insert(0, eid_col, selected_eids.astype(np.int64))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "source_phenotype_csv": str(cfg.get("phenotype_csv")),
        "split": args.split,
        "fill_mode": args.fill_mode,
        "n_rows": int(output_df.shape[0]),
        "n_features": int(len(feature_cols)),
        "n_filled_cells": int((~selected_observed).sum()) if args.fill_mode == "missing_only" else int(output_df.shape[0] * len(feature_cols)),
        "device": str(device),
    }
    with open(args.output_csv.with_suffix(args.output_csv.suffix + ".meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    logger.info(
        "Exported %d rows x %d features to %s using split=%s fill_mode=%s",
        output_df.shape[0],
        len(feature_cols),
        str(args.output_csv),
        args.split,
        args.fill_mode,
    )


if __name__ == "__main__":
    main()
