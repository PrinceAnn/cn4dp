#!/usr/bin/env python
"""Downstream disease prediction on a matched riskset (case/control) CSV.

This script verifies the quality of a (pretrained) phenotype encoder by training
a simple multi-task head on riskset labels:

1) Predict whether the disease occurs (binary classification).
2) Predict time-to-onset (regression) for incident cases, using `delta_time`.

Inputs
------
- riskset CSV (labels + sample selection): contains `eid` and `label` at least.
- phenotype feature CSV: contains `eid` and many numeric IDP/phenotype columns.

The script joins on `eid`, standardizes features (mean-impute then z-score, as in
`phenotype_encoder`), trains an encoder+classifier, and reports metrics.

Run
---
python scripts/downstream/train_phenotype.py --config configs/demo/idp_scratch.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader



import pytorch_lightning as pl  # type: ignore
from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore
from pytorch_lightning.loggers import CSVLogger  # type: ignore


# Allow running directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenotype_encoder.data import PreprocessArtifacts, build_preprocess_pipeline
from phenotype_encoder.model import build_encoder_from_config

from scripts.downstream.utils import (  # type: ignore
    configure_logger,
    datetime_to_year_fraction,
    load_delphi_checkpoint,
    load_labels,
    roc_auc_score,
    seed_everything,
    split_indices,
    split_subject_indices,
)


logger = logging.getLogger("downstream.riskset")


def _configure_logging(level: str = "INFO") -> None:
    configure_logger(logger, level=level)


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed RNGs for reproducible runs.

    Note: full determinism can still be affected by GPU ops / driver versions.
    """
    seed_everything(int(seed), deterministic=deterministic, lightning_seed_fn=getattr(pl, "seed_everything", None))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream riskset classifier")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)



def _split_indices(n: int, seed: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split_indices(n, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn.

    Uses the rank-based Mann–Whitney U statistic.
    Returns NaN if only one class is present.
    """
    return float(roc_auc_score(y_true, y_score))


def _apply_synthetic_missingness(
    X_raw: np.ndarray,
    *,
    feature_cols: list[str],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask additional observed IDP cells for downstream imputation utility tests."""
    missing_cfg = cfg.get("synthetic_missingness", {}) or {}
    mixture_cfg = missing_cfg.get("mixture", None)
    mode = str(missing_cfg.get("mode", "none")).lower()
    if mixture_cfg is None and mode in {"", "none", "off", "false"}:
        return X_raw, np.zeros_like(np.isfinite(X_raw), dtype=bool)

    seed = int(missing_cfg.get("seed", cfg.get("seed", 42)))
    rng = np.random.default_rng(seed)

    observed = np.isfinite(X_raw)
    synthetic_mask = np.zeros_like(observed, dtype=bool)

    from phenotype_encoder.data import group_feature_columns_by_organ

    organ_groups = group_feature_columns_by_organ(feature_cols)
    organ_to_indices = {
        organ: np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
        for organ, columns in organ_groups.items()
    }
    organ_names = list(organ_to_indices.keys())

    if mixture_cfg is not None:
        if not isinstance(mixture_cfg, list) or not mixture_cfg:
            raise ValueError("synthetic_missingness.mixture must be a non-empty list")
        specs = [dict(item) for item in mixture_cfg]
        weights = np.asarray([float(item.get("weight", 1.0)) for item in specs], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            raise ValueError("synthetic_missingness.mixture weights must be finite and positive")
        weights = weights / float(weights.sum())
    else:
        specs = [dict(missing_cfg)]
        weights = np.asarray([1.0], dtype=np.float64)

    for row in range(X_raw.shape[0]):
        spec = specs[int(rng.choice(np.arange(len(specs)), p=weights))]
        row_mode = str(spec.get("mode", "random")).lower()
        ratio = min(max(float(spec.get("ratio", 0.0)), 0.0), 1.0)

        if row_mode == "random":
            candidates = np.flatnonzero(observed[row])
            n_mask = int(round(float(candidates.size) * ratio))
            if n_mask > 0:
                chosen = rng.choice(candidates, size=min(n_mask, candidates.size), replace=False)
                synthetic_mask[row, chosen] = True

        elif row_mode in {"organ", "organ_block", "block"}:
            fixed_organs = spec.get("organs", None)
            fixed_organs = [str(value) for value in fixed_organs] if fixed_organs else None
            candidates = np.flatnonzero(observed[row])

            if fixed_organs:
                selected = [organ for organ in fixed_organs if organ in organ_to_indices]
            elif ratio <= 0:
                selected = [str(rng.choice(np.asarray(organ_names, dtype=object)))]
            else:
                target = int(round(float(candidates.size) * ratio))
                if target <= 0:
                    continue
                shuffled = list(rng.permutation(organ_names))
                selected = []
                covered = 0
                for organ in shuffled:
                    selected.append(organ)
                    idx = organ_to_indices[organ]
                    covered += int(observed[row, idx].sum())
                    if covered >= target:
                        break

            for organ in selected:
                idx = organ_to_indices[organ]
                synthetic_mask[row, idx] |= observed[row, idx]

        else:
            raise ValueError(f"Unsupported synthetic_missingness.mode={row_mode!r}")

    X_masked = X_raw.copy()
    X_masked[synthetic_mask] = np.nan
    logger.info(
        "Applied synthetic missingness: mode=%s masked_cells=%d observed_cells=%d",
        "mixture" if mixture_cfg is not None else mode,
        int(synthetic_mask.sum()),
        int(observed.sum()),
    )
    return X_masked, synthetic_mask


def _impute_features_with_checkpoint(
    X_raw: np.ndarray,
    *,
    eids: np.ndarray,
    feature_cols: list[str],
    cfg: dict,
    shuffle_groups: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    """Apply a frozen imputation checkpoint in memory, without exporting a phenotype table."""
    impute_cfg = cfg.get("imputation", {}) or {}
    if not bool(impute_cfg.get("enabled", False)):
        return X_raw

    ckpt_path = impute_cfg.get("checkpoint", None)
    if not ckpt_path:
        raise ValueError("imputation.enabled=true requires imputation.checkpoint")

    from scripts.imputation.export_imputed_features import (
        _build_inference_samples,
        _build_model,
        _inverse_standardize,
        _load_checkpoint,
        _load_preprocess_from_checkpoint,
        _run_export,
        _standardize_with_checkpoint_stats,
    )
    from scripts.imputation.train_idp_imputation import _build_pre_imaging_sequences

    import pandas as pd

    checkpoint = _load_checkpoint(Path(ckpt_path))
    impute_model_cfg = dict(checkpoint.get("config", {}) or {})
    checkpoint_feature_cols = list(checkpoint.get("feature_cols", []) or [])
    if checkpoint_feature_cols != list(feature_cols):
        missing = [column for column in checkpoint_feature_cols if column not in feature_cols]
        extra = [column for column in feature_cols if column not in checkpoint_feature_cols]
        if missing or extra:
            raise ValueError(
                "Downstream feature columns do not match imputation checkpoint. "
                f"missing_from_downstream={missing[:5]} extra_in_downstream={extra[:5]}"
            )
        reorder = [feature_cols.index(column) for column in checkpoint_feature_cols]
        X_for_imputer = X_raw[:, reorder]
        inverse_reorder = np.argsort(np.asarray(reorder))
    else:
        X_for_imputer = X_raw
        inverse_reorder = None

    preprocess = _load_preprocess_from_checkpoint(checkpoint)
    observed_mask = np.isfinite(X_for_imputer)
    standardized_features = _standardize_with_checkpoint_stats(X_for_imputer, preprocess)

    cohort_csv = impute_cfg.get("cohort_csv", impute_model_cfg.get("cohort_csv", None))
    if cohort_csv is None:
        raise ValueError("Dynamic imputation requires imputation.cohort_csv or checkpoint config cohort_csv")
    eid_col = str(cfg.get("eid_col", "eid"))
    imaging_date_col = str(impute_model_cfg.get("imaging_date_col", "imaging_date"))
    cohort = pd.read_csv(Path(cohort_csv), usecols=[eid_col, imaging_date_col])
    cohort[eid_col] = cohort[eid_col].astype(int)
    date_by_eid = cohort.set_index(eid_col)[imaging_date_col]
    missing_dates = [int(eid) for eid in eids.tolist() if int(eid) not in date_by_eid.index]
    if missing_dates:
        raise ValueError(f"Missing imaging dates for {len(missing_dates)} eids, e.g. {missing_dates[:5]}")
    imaging_dates = date_by_eid.reindex(eids.astype(int)).to_numpy(copy=True)

    impute_exclude_codes: set[str] = set()
    if bool(impute_cfg.get("exclude_target_codes", False)):
        explicit_codes = impute_cfg.get("target_codes", None)
        if explicit_codes:
            impute_exclude_codes.update(str(code).upper() for code in explicit_codes)
        else:
            riskset_path = str(cfg.get("riskset_csv", ""))
            stem = Path(riskset_path).stem
            if stem.startswith("idp_riskset_"):
                impute_exclude_codes.add(stem.removeprefix("idp_riskset_").split("_")[0].upper())

    token_list, age_list = _build_pre_imaging_sequences(
        eids=eids.astype(np.int64, copy=False),
        imaging_date=imaging_dates,
        disease_onset_csv=Path(impute_model_cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv")),
        basic_csv=Path(impute_model_cfg.get("basic_csv", "data/demo/basic_info.csv")),
        labels_csv=Path(impute_model_cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv")),
        eid_col=eid_col,
        block_size=int(impute_model_cfg.get("block_size", 256)),
        append_query_token=bool(impute_model_cfg.get("append_query_token", True)),
        exclude_codes=impute_exclude_codes,
    )
    if impute_exclude_codes:
        logger.info("Dynamic imputation trajectory target-code exclusion: %s", ",".join(sorted(impute_exclude_codes)))

    if bool(impute_cfg.get("shuffle_trajectory", False)):
        shuffle_seed = int(
            impute_cfg.get(
                "shuffle_seed",
                (cfg.get("synthetic_missingness", {}) or {}).get("seed", cfg.get("seed", 42)),
            )
        )
        rng = np.random.default_rng(shuffle_seed + 104729)
        groups = shuffle_groups or (np.arange(len(token_list), dtype=np.int64),)
        shuffled_tokens = list(token_list)
        shuffled_ages = list(age_list)
        fixed_points = 0
        shuffled_subjects = 0
        for group in groups:
            group = np.asarray(group, dtype=np.int64)
            if group.size <= 1:
                fixed_points += int(group.size)
                continue
            source = rng.permutation(group)
            for _ in range(100):
                if not np.any(source == group):
                    break
                source = rng.permutation(group)
            if np.any(source == group):
                source = np.roll(group, 1)
            fixed_points += int(np.sum(source == group))
            shuffled_subjects += int(group.size)
            for target_idx, source_idx in zip(group.tolist(), source.tolist()):
                shuffled_tokens[target_idx] = token_list[source_idx]
                shuffled_ages[target_idx] = age_list[source_idx]
        token_list = shuffled_tokens
        age_list = shuffled_ages
        logger.info(
            "Applied split-local trajectory shuffle: seed=%d subjects=%d fixed_points=%d",
            shuffle_seed,
            shuffled_subjects,
            fixed_points,
        )

    device_name = str(impute_cfg.get("device", cfg.get("device", impute_model_cfg.get("device", "cpu"))))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model, trajectory_encoder, use_trajectory = _build_model(
        checkpoint=checkpoint,
        cfg=impute_model_cfg,
        feature_cols=checkpoint_feature_cols,
        device=device,
    )

    indices = np.arange(X_for_imputer.shape[0], dtype=np.int64)
    samples = _build_inference_samples(
        standardized_features=standardized_features,
        observed_mask=observed_mask,
        eids=eids.astype(np.int64, copy=False),
        token_list=token_list,
        age_list=age_list,
        indices=indices,
    )
    pred_std, pred_eids = _run_export(
        model=model,
        trajectory_encoder=trajectory_encoder,
        use_trajectory=use_trajectory,
        samples=samples,
        batch_size=int(impute_cfg.get("batch_size", 256)),
        device=device,
    )
    if not np.array_equal(pred_eids.astype(np.int64), eids.astype(np.int64)):
        raise RuntimeError("EID order mismatch during dynamic imputation")

    pred_raw = _inverse_standardize(pred_std.astype(np.float32, copy=False), preprocess)
    output = X_for_imputer.copy()
    fill_mask = ~observed_mask
    output[fill_mask] = pred_raw[fill_mask]
    if inverse_reorder is not None:
        output = output[:, inverse_reorder]

    logger.info(
        "Applied dynamic imputation: checkpoint=%s rows=%d filled_cells=%d",
        str(ckpt_path),
        int(output.shape[0]),
        int(fill_mask.sum()),
    )
    return output


def _apply_simple_feature_imputation(
    X_raw: np.ndarray,
    *,
    train_indices: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    simple_cfg = cfg.get("simple_imputation", {}) or {}
    method = str(simple_cfg.get("method", "none")).lower()
    if method in {"", "none", "off", "false"}:
        return X_raw

    train_raw = X_raw[train_indices]
    if method == "mean":
        fill_values = np.nanmean(train_raw, axis=0)
    elif method == "median":
        fill_values = np.nanmedian(train_raw, axis=0)
    else:
        raise ValueError(f"Unsupported simple_imputation.method={method!r}")

    fill_values = np.asarray(fill_values, dtype=np.float32)
    if not np.isfinite(fill_values).all():
        fallback = np.nanmean(train_raw)
        if not np.isfinite(fallback):
            fallback = 0.0
        fill_values = np.nan_to_num(fill_values, nan=float(fallback), posinf=float(fallback), neginf=float(fallback))

    missing = ~np.isfinite(X_raw)
    output = X_raw.copy()
    if missing.any():
        _, col_idx = np.where(missing)
        output[missing] = fill_values[col_idx]
    logger.info(
        "Applied simple feature imputation: method=%s filled_cells=%d",
        method,
        int(missing.sum()),
    )
    return output


def _collate_trajectory_batch(
    token_list: list[torch.Tensor],
    age_list: list[torch.Tensor],
    *,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(token_list[int(i)].numel()) for i in indices.tolist())
    tokens = torch.zeros((len(indices), max_len), dtype=torch.long)
    ages = torch.full((len(indices), max_len), -10000.0, dtype=torch.float32)
    for row, item_idx in enumerate(indices.tolist()):
        tok = token_list[int(item_idx)]
        age = age_list[int(item_idx)]
        length = int(tok.numel())
        tokens[row, -length:] = tok
        ages[row, -length:] = age
    return tokens, ages


def _build_trajectory_sequences(
    *,
    eids: np.ndarray,
    imaging_dates: np.ndarray,
    disease_onset_csv: Path,
    basic_csv: Path,
    labels_csv: Path,
    eid_col: str,
    block_size: int,
    append_query_token: bool,
    exclude_codes: set[str],
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, int]]:
    """Build pre-imaging Delphi token sequences in the same order as eids."""
    import pandas as pd

    basic = pd.read_csv(basic_csv)
    if eid_col not in basic.columns:
        raise ValueError(f"basic_csv must contain column {eid_col}")

    birth_year_col = None
    for candidate in ["birth_year"]:
        if candidate in basic.columns:
            birth_year_col = candidate
            break
    if birth_year_col is None:
        raise ValueError("basic_csv must contain birth year column (expected 'birth_year')")

    basic[eid_col] = basic[eid_col].astype(int)
    basic["birth_year"] = pd.to_numeric(basic[birth_year_col], errors="coerce")
    basic = basic.set_index(eid_col)[["birth_year"]]
    birth_year = basic.reindex(eids.astype(int))["birth_year"].to_numpy(dtype=np.float64)
    if np.isnan(birth_year).any():
        raise ValueError(f"Missing birth_year for {int(np.isnan(birth_year).sum())} eids")

    labels_map = load_labels(labels_csv)
    disease = pd.read_csv(disease_onset_csv)
    if eid_col not in disease.columns:
        raise ValueError(f"disease_onset_csv must contain column {eid_col}")
    disease[eid_col] = disease[eid_col].astype(int)
    disease = disease.set_index(eid_col)

    exclude_upper = {str(code).upper() for code in exclude_codes}
    candidate_cols = [column for column in disease.columns if column in labels_map and str(column).upper() not in exclude_upper]
    if not candidate_cols:
        raise ValueError("No disease onset columns overlap with Delphi labels after exclusions")

    disease_sub = disease.reindex(eids.astype(int))
    imaging_year = np.array([datetime_to_year_fraction(value) for value in imaging_dates], dtype=np.float64)
    imaging_age_years = imaging_year - birth_year

    token_list: list[torch.Tensor] = []
    age_list: list[torch.Tensor] = []
    stats = {"target_tokens_excluded": 0, "pre_imaging_events": 0}
    query_token_shifted = 1

    for row_index, eid in enumerate(eids.tolist()):
        imaging_age = float(imaging_age_years[row_index])
        if not math.isfinite(imaging_age) or imaging_age <= 0:
            raise ValueError(f"Invalid imaging age for eid={int(eid)}: {imaging_age}")
        imaging_age_days = imaging_age * 365.25

        disease_row = disease_sub.iloc[row_index]
        ages: list[float] = []
        tokens: list[int] = []

        for disease_code in disease.columns:
            if disease_code not in labels_map:
                continue
            onset_value = disease_row[disease_code]
            if onset_value is None or (isinstance(onset_value, float) and not math.isfinite(onset_value)):
                continue
            try:
                onset_age = float(onset_value)
            except Exception:
                continue
            if not math.isfinite(onset_age) or onset_age > imaging_age:
                continue

            if str(disease_code).upper() in exclude_upper:
                stats["target_tokens_excluded"] += 1
                continue

            token_index = int(labels_map[disease_code]) + 1
            if token_index <= 0:
                continue
            ages.append(onset_age * 365.25)
            tokens.append(token_index)

        stats["pre_imaging_events"] += len(tokens)
        if ages:
            order = sorted(range(len(ages)), key=lambda idx: (ages[idx], tokens[idx]))
            ages = [ages[idx] for idx in order]
            tokens = [tokens[idx] for idx in order]

        if append_query_token:
            ages.append(imaging_age_days)
            tokens.append(query_token_shifted)

        if not tokens:
            ages = [imaging_age_days]
            tokens = [query_token_shifted]

        if len(tokens) > int(block_size):
            ages = ages[-int(block_size) :]
            tokens = tokens[-int(block_size) :]

        token_list.append(torch.tensor(tokens, dtype=torch.long))
        age_list.append(torch.tensor(ages, dtype=torch.float32))

    return token_list, age_list, stats


def _build_trajectory_features(
    *,
    eids: np.ndarray,
    riskset_df,
    train_indices: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray | None, list[str], dict[str, Any]]:
    traj_cfg = cfg.get("trajectory_features", {}) or {}
    if not bool(traj_cfg.get("enabled", False)):
        return None, [], {}

    eid_col = str(cfg.get("eid_col", "eid"))
    imaging_date_col = str(traj_cfg.get("imaging_date_col", cfg.get("imaging_date_col", "imaging_date")))
    if imaging_date_col not in riskset_df.columns:
        raise ValueError(f"trajectory_features requires riskset column {imaging_date_col!r}")

    exclude_codes: set[str] = set()
    if bool(traj_cfg.get("exclude_target_codes", False)):
        explicit = traj_cfg.get("target_codes", None)
        if explicit:
            exclude_codes.update(str(code).upper() for code in explicit)
        elif "disease" in riskset_df.columns:
            exclude_codes.update(str(code).upper() for code in riskset_df["disease"].dropna().unique().tolist())
        else:
            riskset_path = str(cfg.get("riskset_csv", ""))
            stem = Path(riskset_path).stem
            if stem.startswith("idp_riskset_"):
                exclude_codes.add(stem.removeprefix("idp_riskset_").split("_")[0].upper())

    token_list, age_list, seq_stats = _build_trajectory_sequences(
        eids=eids.astype(np.int64, copy=False),
        imaging_dates=riskset_df[imaging_date_col].to_numpy(copy=True),
        disease_onset_csv=Path(traj_cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv")),
        basic_csv=Path(traj_cfg.get("basic_csv", "data/demo/basic_info.csv")),
        labels_csv=Path(traj_cfg.get("labels_csv", traj_cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv"))),
        eid_col=eid_col,
        block_size=int(traj_cfg.get("block_size", 256)),
        append_query_token=bool(traj_cfg.get("append_query_token", True)),
        exclude_codes=exclude_codes,
    )

    from scripts.imputation.train_idp_imputation import FrozenTrajectoryEncoder

    device_name = str(traj_cfg.get("device", cfg.get("device", "cpu")))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    delphi, delphi_conf = load_delphi_checkpoint(Path(traj_cfg.get("delphi_ckpt_dir", "runs/demo/teacher")), device=torch.device("cpu"))
    encoder = FrozenTrajectoryEncoder(
        delphi,
        pooling=str(traj_cfg.get("pooling", "sum")),
        freeze_delphi=True,
        se_reduction=int(traj_cfg.get("se_reduction", traj_cfg.get("teacher_senet_reduction", 16))),
        se_dropout=float(traj_cfg.get("se_dropout", traj_cfg.get("teacher_senet_dropout", 0.0))),
        temperature=float(traj_cfg.get("temperature", traj_cfg.get("teacher_senet_temperature", 1.0))),
    ).to(device)
    encoder.eval()

    embeddings = np.zeros((len(eids), int(delphi_conf.n_embd)), dtype=np.float32)
    batch_size = int(traj_cfg.get("batch_size", 256))
    with torch.no_grad():
        for start in range(0, len(eids), batch_size):
            batch_idx = np.arange(start, min(start + batch_size, len(eids)), dtype=np.int64)
            tokens, ages = _collate_trajectory_batch(token_list, age_list, indices=batch_idx)
            z = encoder(tokens.to(device), ages.to(device)).detach().cpu().numpy().astype(np.float32)
            embeddings[batch_idx] = z

    # Train-only standardization for trajectory features before concatenation.
    mean = embeddings[train_indices].mean(axis=0).astype(np.float32)
    std = embeddings[train_indices].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    embeddings = (embeddings - mean) / std

    names = [f"trajectory__delphi_{i:03d}" for i in range(embeddings.shape[1])]
    meta = {
        "trajectory_dim": int(embeddings.shape[1]),
        "pooling": str(traj_cfg.get("pooling", "sum")),
        "exclude_target_codes": bool(traj_cfg.get("exclude_target_codes", False)),
        "excluded_codes": sorted(exclude_codes),
        **seq_stats,
    }
    logger.info(
        "Built trajectory features: dim=%d pooling=%s exclude_target_codes=%s excluded=%s",
        int(embeddings.shape[1]),
        meta["pooling"],
        meta["exclude_target_codes"],
        ",".join(meta["excluded_codes"]) if meta["excluded_codes"] else "none",
    )
    return embeddings, names, meta


class RisksetDataset(torch.utils.data.Dataset):
    """Dataset for multi-task learning.

    - Classification uses all samples.
    - Time regression uses only samples with finite delta_time and y==1.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, eid: np.ndarray, delta_time: np.ndarray | None = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.eid = torch.from_numpy(np.asarray(eid, dtype=np.int64))

        if delta_time is None:
            delta_time = np.full((len(X),), np.nan, dtype=np.float32)
        delta_time = np.asarray(delta_time, dtype=np.float32)
        dt_mask = np.isfinite(delta_time) & (y.astype(np.int64) == 1)

        # Store dt with NaNs replaced for stable tensor creation.
        dt_filled = np.nan_to_num(delta_time, nan=0.0, posinf=0.0, neginf=0.0)
        self.delta_time = torch.from_numpy(dt_filled.astype(np.float32))
        self.dt_mask = torch.from_numpy(dt_mask.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "eid": self.eid[idx],
            "delta_time": self.delta_time[idx],
            "dt_mask": self.dt_mask[idx],
        }


class EncoderMultiTask(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        embed_dim: int,
        classifier_dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.classifier_dropout = nn.Dropout(classifier_dropout) if classifier_dropout and classifier_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, 1)
        self.time_head = nn.Linear(embed_dim, 1)
        self.time_act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z = self.classifier_dropout(z)
        logit = self.classifier(z).squeeze(-1)
        dt = self.time_act(self.time_head(z).squeeze(-1))
        return logit, dt


class RisksetDataModule(pl.LightningDataModule):  # type: ignore[misc]
    def __init__(
        self,
        *,
        ds_train: torch.utils.data.Dataset,
        ds_val: torch.utils.data.Dataset | None,
        ds_test: torch.utils.data.Dataset | None,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
    ):
        super().__init__()
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.seed = int(seed)

    def _worker_init_fn(self, worker_id: int) -> None:
        # Ensure deterministic behavior when num_workers > 0.
        s = self.seed + int(worker_id)
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            generator=g,
        )

    def val_dataloader(self):
        if self.ds_val is None:
            return None
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        if self.ds_test is None:
            return None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
        )


class MultiTaskRisksetModule(pl.LightningModule):  # type: ignore[misc]
    def __init__(
        self,
        *,
        net: EncoderMultiTask,
        cfg: dict,
        feature_cols: list[str],
        preprocess_artifacts: PreprocessArtifacts,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.preprocess_artifacts = preprocess_artifacts

        # Loss weights
        self.w_cls = float(cfg.get("cls_loss_weight", 1.0))
        self.w_time = float(cfg.get("time_loss_weight", 1.0))

        if pos_weight is not None:
            self.criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.criterion_cls = nn.BCEWithLogitsLoss()

        time_loss = str(cfg.get("time_loss", "smoothl1")).lower()
        if time_loss in {"smoothl1", "huber"}:
            self.criterion_time = nn.SmoothL1Loss(reduction="none")
        elif time_loss in {"mse", "l2"}:
            self.criterion_time = nn.MSELoss(reduction="none")
        elif time_loss in {"mae", "l1"}:
            self.criterion_time = nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Unsupported time_loss={time_loss}")

        # Buffers for epoch-end metrics (no torchmetrics dependency)
        self._val_y: list[np.ndarray] = []
        self._val_p: list[np.ndarray] = []
        self._val_dt_y: list[np.ndarray] = []
        self._val_dt_p: list[np.ndarray] = []
        self._test_y: list[np.ndarray] = []
        self._test_p: list[np.ndarray] = []
        self._test_eid: list[np.ndarray] = []
        self._test_logit: list[np.ndarray] = []
        self._test_dt: list[np.ndarray] = []
        self._test_dt_pred: list[np.ndarray] = []
        self._test_dt_mask: list[np.ndarray] = []
        self._test_dt_y: list[np.ndarray] = []
        self._test_dt_p: list[np.ndarray] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]
        eid = batch.get("eid", None)
        dt = batch["delta_time"]
        dt_mask = batch["dt_mask"]

        logit, dt_pred = self(x)
        loss_cls = self.criterion_cls(logit, y)

        # Regression loss only for valid incident cases
        per_ex_loss_time = self.criterion_time(dt_pred, dt)
        denom = torch.clamp(dt_mask.sum(), min=1.0)
        loss_time = (per_ex_loss_time * dt_mask).sum() / denom

        loss = self.w_cls * loss_cls + self.w_time * loss_time

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_cls", loss_cls, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_time", loss_time, prog_bar=False, on_step=False, on_epoch=True)

        # Collect metrics on CPU
        logit_np = logit.detach().cpu().numpy()
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        dt_pred_np = dt_pred.detach().cpu().numpy()
        dt_np = dt.detach().cpu().numpy()
        dt_mask_np = dt_mask.detach().cpu().numpy().astype(bool)

        if stage == "val":
            self._val_y.append(y_np)
            self._val_p.append(prob)
            if dt_mask_np.any():
                self._val_dt_y.append(dt_np[dt_mask_np])
                self._val_dt_p.append(dt_pred_np[dt_mask_np])
        elif stage == "test":
            self._test_y.append(y_np)
            self._test_p.append(prob)
            if eid is not None:
                self._test_eid.append(eid.detach().cpu().numpy().astype(np.int64))
            self._test_logit.append(logit_np)
            self._test_dt.append(dt_np)
            self._test_dt_pred.append(dt_pred_np)
            self._test_dt_mask.append(dt_mask_np.astype(np.int64))
            if dt_mask_np.any():
                self._test_dt_y.append(dt_np[dt_mask_np])
                self._test_dt_p.append(dt_pred_np[dt_mask_np])

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        _ = self._shared_step(batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        _ = self._shared_step(batch, stage="test")

    def on_validation_epoch_end(self) -> None:
        if not self._val_y:
            return
        y = np.concatenate(self._val_y)
        p = np.concatenate(self._val_p)
        auc = _roc_auc_score(y, p)
        acc = float(((p >= 0.5).astype(np.int64) == y.astype(np.int64)).mean())
        self.log("val/auc", auc, prog_bar=True)
        self.log("val/acc", acc, prog_bar=False)

        if self._val_dt_y:
            dt_y = np.concatenate(self._val_dt_y)
            dt_p = np.concatenate(self._val_dt_p)
            mae = float(np.mean(np.abs(dt_p - dt_y)))
            self.log("val/dt_mae", mae, prog_bar=True)
        else:
            self.log("val/dt_mae", float("nan"), prog_bar=False)

        self._val_y.clear()
        self._val_p.clear()
        self._val_dt_y.clear()
        self._val_dt_p.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_y:
            return
        y = np.concatenate(self._test_y)
        p = np.concatenate(self._test_p)
        auc = _roc_auc_score(y, p)
        acc = float(((p >= 0.5).astype(np.int64) == y.astype(np.int64)).mean())
        self.log("test/auc", auc)
        self.log("test/acc", acc)

        # Save per-sample predictions for downstream ROC/AUC plotting.
        try:
            out_dir = Path(self.cfg["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.csv"

            eid = np.concatenate(self._test_eid) if self._test_eid else np.full_like(y, fill_value=-1, dtype=np.int64)
            logit = np.concatenate(self._test_logit) if self._test_logit else np.full_like(p, fill_value=np.nan, dtype=np.float32)
            dt = np.concatenate(self._test_dt) if self._test_dt else np.full_like(p, fill_value=np.nan, dtype=np.float32)
            dt_pred = (
                np.concatenate(self._test_dt_pred) if self._test_dt_pred else np.full_like(p, fill_value=np.nan, dtype=np.float32)
            )
            dt_mask = (
                np.concatenate(self._test_dt_mask) if self._test_dt_mask else np.zeros_like(y, dtype=np.int64)
            )

            with open(pred_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["eid", "y", "prob", "logit", "delta_time", "dt_pred", "dt_mask"])
                for i in range(len(y)):
                    w.writerow(
                        [
                            int(eid[i]),
                            int(y[i]),
                            float(p[i]),
                            float(logit[i]),
                            float(dt[i]),
                            float(dt_pred[i]),
                            int(dt_mask[i]),
                        ]
                    )
        except Exception as e:
            logger.warning("Failed to write test_predictions.csv: %s", str(e))

        if self._test_dt_y:
            dt_y = np.concatenate(self._test_dt_y)
            dt_p = np.concatenate(self._test_dt_p)
            mae = float(np.mean(np.abs(dt_p - dt_y)))
            self.log("test/dt_mae", mae)
        else:
            self.log("test/dt_mae", float("nan"))

        self._test_y.clear()
        self._test_p.clear()
        self._test_eid.clear()
        self._test_logit.clear()
        self._test_dt.clear()
        self._test_dt_pred.clear()
        self._test_dt_mask.clear()
        self._test_dt_y.clear()
        self._test_dt_p.clear()

    def configure_optimizers(self):
        opt_name = str(self.cfg.get("optimizer", "adamw")).lower()
        lr = float(self.cfg.get("lr", 1e-3))
        weight_decay = float(self.cfg.get("weight_decay", 0.0))

        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters")

        if opt_name == "adamw":
            betas = self.cfg.get("betas", None)
            if betas is None:
                betas = (0.9, 0.999)
            else:
                betas = (float(betas[0]), float(betas[1]))
            eps = float(self.cfg.get("eps", 1e-8))
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        elif opt_name == "sgd":
            momentum = float(self.cfg.get("momentum", 0.9))
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer={opt_name}")

        # Warmup + cosine decay (per-step)
        warmup_steps = self.cfg.get("warmup_steps", None)
        warmup_ratio = float(self.cfg.get("warmup_ratio", 0.0))
        min_lr_ratio = float(self.cfg.get("min_lr_ratio", 0.1))

        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps <= 0:
            # Fallback (rare, but keep safe)
            total_steps = int(self.cfg.get("total_steps", 0) or 0)

        if warmup_steps is None:
            warmup_steps_i = int(total_steps * warmup_ratio)
        else:
            warmup_steps_i = int(warmup_steps)
        warmup_steps_i = max(0, min(warmup_steps_i, total_steps))

        def lr_lambda(step: int) -> float:
            if total_steps <= 0:
                return 1.0
            if warmup_steps_i > 0 and step < warmup_steps_i:
                return float(step) / float(max(1, warmup_steps_i))
            # cosine from 1 -> min_lr_ratio
            progress = float(step - warmup_steps_i) / float(max(1, total_steps - warmup_steps_i))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Persist data artifacts so the best checkpoint is self-contained.
        checkpoint["config"] = self.cfg
        checkpoint["feature_cols"] = self.feature_cols
        checkpoint["preprocess"] = {
            # NOTE: Keep this weights_only-safe (no numpy objects), otherwise
            # PyTorch 2.6+ safe unpickling may fail when Lightning loads ckpts.
            "mean_": np.asarray(self.preprocess_artifacts.mean_, dtype=np.float32).tolist(),
            "std_": np.asarray(self.preprocess_artifacts.std_, dtype=np.float32).tolist(),
        }


def _load_pretrained_distill_ckpt(ckpt_path: Path) -> Tuple[dict, list[str], PreprocessArtifacts]:
    """Load distillation checkpoint from scripts/distill/train_align.py.

    Expected keys:
    - model_state: phenotype encoder state_dict
    - feature_names: list of phenotype feature columns used during distillation
    - preprocess: artifacts for phenotype preprocessing
    """

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type at {ckpt_path}: {type(ckpt)}")

    for k in ["model_state", "feature_names", "preprocess"]:
        if k not in ckpt:
            raise ValueError(
                f"Checkpoint {ckpt_path} missing key '{k}'. "
                "Make sure you pass a distill alignment checkpoint produced by scripts/distill/train_align.py"
            )

    feature_names = list(ckpt["feature_names"])
    preprocess_artifacts = ckpt["preprocess"]
    if isinstance(preprocess_artifacts, dict):
        # Rebuild as dataclass.
        preprocess_artifacts = PreprocessArtifacts(
            mean_=np.asarray(preprocess_artifacts["mean_"], dtype=np.float32),
            std_=np.asarray(preprocess_artifacts["std_"], dtype=np.float32),
        )
    if not isinstance(preprocess_artifacts, PreprocessArtifacts):
        raise ValueError(f"Unexpected preprocess artifacts type in {ckpt_path}: {type(preprocess_artifacts)}")
    return ckpt, feature_names, preprocess_artifacts


def train_from_config(config_path: Path) -> None:
    cfg = _load_yaml(config_path)
    _configure_logging(str(cfg.get("log_level", "INFO")))

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    _seed_everything(seed, deterministic=deterministic)

    if pl is None:  # pragma: no cover
        raise RuntimeError(
            "Lightning is required for this script. Install with: pip install lightning (or pytorch-lightning)."
        )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    riskset_csv = Path(cfg["riskset_csv"])
    phenotype_csv = Path(cfg["phenotype_csv"])
    eid_col = str(cfg.get("eid_col", "eid"))
    label_col = str(cfg.get("label_col", "label"))
    delta_time_col = str(cfg.get("delta_time_col", "delta_time"))

    import pandas as pd

    rs = pd.read_csv(riskset_csv)
    ph = pd.read_csv(phenotype_csv)

    if eid_col not in rs.columns or label_col not in rs.columns:
        raise ValueError(f"riskset_csv must contain columns {eid_col} and {label_col}")
    if eid_col not in ph.columns:
        raise ValueError(f"phenotype_csv must contain column {eid_col}")

    keep_cols = [eid_col, label_col]
    if delta_time_col in rs.columns:
        keep_cols.append(delta_time_col)
    for optional_col in ["disease", str(cfg.get("imaging_date_col", "imaging_date"))]:
        if optional_col in rs.columns and optional_col not in keep_cols:
            keep_cols.append(optional_col)

    rs2 = rs[keep_cols].copy()
    rs2[eid_col] = rs2[eid_col].astype(int)
    rs2[label_col] = rs2[label_col].astype(int)
    if delta_time_col in rs2.columns:
        rs2[delta_time_col] = pd.to_numeric(rs2[delta_time_col], errors="coerce")

    ph[eid_col] = ph[eid_col].astype(int)
    idp_feature_cfg = cfg.get("idp_features", {}) or {}
    use_idp_features = bool(idp_feature_cfg.get("enabled", True))
    if use_idp_features:
        merged = rs2.merge(ph, on=eid_col, how="inner")
    else:
        merged = rs2.merge(ph[[eid_col]], on=eid_col, how="inner")
    if len(merged) == 0:
        raise ValueError("No samples after joining riskset and phenotype CSV on eid")

    eids = merged[eid_col].to_numpy(dtype=np.int64, copy=True)

    # Decide IDP feature set + preprocessing.
    pretrained_cfg = cfg.get("pretrained", {}) or {}
    ckpt_path = pretrained_cfg.get("ckpt_path")
    use_pretrained = bool(ckpt_path)

    if use_pretrained and not use_idp_features:
        raise ValueError("pretrained.ckpt_path cannot be used when idp_features.enabled=false")

    if not use_idp_features:
        feature_cols = []
        X_raw = np.zeros((len(merged), 0), dtype=np.float32)
        preprocess = build_preprocess_pipeline()
        logger.info("IDP features disabled; downstream input will use non-IDP feature blocks only")
    elif use_pretrained:
        ckpt_path = Path(ckpt_path)
        ckpt, feature_cols, preprocess_artifacts = _load_pretrained_distill_ckpt(ckpt_path)
        missing = [c for c in feature_cols if c not in merged.columns]
        if missing:
            raise ValueError(
                "phenotype_csv is missing features required by pretrained checkpoint. "
                f"Missing {len(missing)} columns, e.g. {missing[:10]}"
            )
        # We intentionally ignore extra phenotype columns not in checkpoint.
        X_raw = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)
        preprocess = build_preprocess_pipeline()
        preprocess.mean_ = preprocess_artifacts.mean_.astype(np.float32)
        preprocess.std_ = preprocess_artifacts.std_.astype(np.float32)
        logger.info("Using pretrained encoder from %s", str(ckpt_path))
    else:
        # IMPORTANT: exclude outcome/target-derived columns from features to avoid leakage.
        # - label_col is the classification target
        # - delta_time_col is only used as the regression target (and often directly encodes case status)
        exclude = {eid_col, label_col, str(cfg.get("imaging_date_col", "imaging_date")), "disease", "matched_to", "case_time"}
        if delta_time_col in merged.columns:
            exclude.add(delta_time_col)
        feature_cols = [c for c in merged.columns if c not in exclude]
        if not feature_cols:
            raise ValueError("No feature columns found after merge")
        X_raw = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)
        preprocess = build_preprocess_pipeline()

    # Extra safety: never allow targets in feature set.
    if label_col in feature_cols:
        raise RuntimeError(f"Leakage detected: label_col={label_col!r} is in feature_cols")
    if delta_time_col in feature_cols:
        raise RuntimeError(f"Leakage detected: delta_time_col={delta_time_col!r} is in feature_cols")

    tr, va, te = split_subject_indices(
        eids, seed=seed,
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        split_csv=cfg.get("split_csv"), eid_col=eid_col,
    )

    if use_idp_features:
        X_raw, synthetic_mask = _apply_synthetic_missingness(X_raw, feature_cols=feature_cols, cfg=cfg)
        X_raw = _impute_features_with_checkpoint(
            X_raw,
            eids=eids,
            feature_cols=feature_cols,
            cfg=cfg,
            shuffle_groups=(tr, va, te),
        )
        X_raw = _apply_simple_feature_imputation(X_raw, train_indices=tr, cfg=cfg)
        if not use_pretrained:
            preprocess.fit(X_raw[tr])
        X_idp = preprocess.transform(X_raw)
    else:
        synthetic_mask = np.zeros_like(X_raw, dtype=bool)
        X_idp = X_raw

    y = merged[label_col].to_numpy(dtype=np.int64, copy=True)
    delta_time = merged[delta_time_col].to_numpy(dtype=np.float32, copy=True) if delta_time_col in merged.columns else None
    trajectory_features, trajectory_feature_cols, trajectory_meta = _build_trajectory_features(
        eids=eids,
        riskset_df=merged,
        train_indices=tr,
        cfg=cfg,
    )

    final_feature_cols = list(feature_cols)
    if trajectory_features is not None:
        X = np.concatenate([X_idp, trajectory_features], axis=1).astype(np.float32, copy=False)
        final_feature_cols.extend(trajectory_feature_cols)
        preprocess_artifacts_for_checkpoint = PreprocessArtifacts(
            mean_=np.zeros((X.shape[1],), dtype=np.float32),
            std_=np.ones((X.shape[1],), dtype=np.float32),
        )
    else:
        X = X_idp
        preprocess_artifacts_for_checkpoint = preprocess.to_artifacts()

    if X.shape[1] == 0:
        raise ValueError("No downstream input features are enabled")

    if bool(cfg.get("check_finite", True)) and (not np.isfinite(X).all()):
        raise ValueError("Found NaN/Inf in standardized phenotype features")

    ds_train = RisksetDataset(X[tr], y[tr], eids[tr], None if delta_time is None else delta_time[tr])
    ds_val = RisksetDataset(X[va], y[va], eids[va], None if delta_time is None else delta_time[va]) if len(va) else None
    ds_test = RisksetDataset(X[te], y[te], eids[te], None if delta_time is None else delta_time[te]) if len(te) else None

    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))

    dm = RisksetDataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    enc_cfg = cfg.get("encoder", {}) or {}
    embed_dim = int(enc_cfg.get("embed_dim", enc_cfg.get("out_dim", 128)))

    if use_pretrained:
        distill_cfg = (ckpt.get("config", {}) or {})
        distill_encoder_cfg = (distill_cfg.get("encoder", {}) or {})

        # Build encoder exactly as distill used, based on feature_cols order.
        if trajectory_features is not None:
            raise ValueError("pretrained phenotype encoders are not supported with appended trajectory_features")
        encoder, out_dim = build_encoder_from_config(encoder_cfg=distill_encoder_cfg, feature_cols=final_feature_cols)
        encoder.load_state_dict(ckpt["model_state"], strict=True)

        freeze = bool(pretrained_cfg.get("freeze_encoder", True))
        if freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)

        embed_dim = int(out_dim)
        logger.info("Loaded pretrained encoder weights. freeze_encoder=%s out_dim=%d", freeze, embed_dim)
    else:
        # Train encoder from scratch.
        encoder, out_dim = build_encoder_from_config(encoder_cfg=enc_cfg, feature_cols=final_feature_cols)
        embed_dim = int(out_dim)

    # Optional pos_weight for BCE when classes are imbalanced.
    pos_weight = cfg.get("pos_weight", None)
    if pos_weight is None:
        n_pos = float((y[tr] == 1).sum())
        n_neg = float((y[tr] == 0).sum())
        if n_pos > 0:
            pos_weight = n_neg / max(1.0, n_pos)
        else:
            pos_weight = None
    else:
        pos_weight = float(pos_weight)

    classifier_dropout = float(cfg.get("classifier_dropout", 0.0))
    net = EncoderMultiTask(encoder=encoder, embed_dim=embed_dim, classifier_dropout=classifier_dropout)

    logger.info("Samples after join: n=%d (pos=%d neg=%d)", len(X), int((y == 1).sum()), int((y == 0).sum()))
    logger.info("Features=%d", X.shape[1])
    if trajectory_meta:
        logger.info("Trajectory feature meta: %s", json.dumps(trajectory_meta, ensure_ascii=False))

    module = MultiTaskRisksetModule(
        net=net,
        cfg=cfg,
        feature_cols=final_feature_cols,
        preprocess_artifacts=preprocess_artifacts_for_checkpoint,
        pos_weight=pos_weight,
    )

    # Lightning trainer
    max_epochs = int(cfg.get("epochs", cfg.get("max_epochs", 20)))
    grad_clip = cfg.get("gradient_clip_val", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    logger_csv = CSVLogger(save_dir=str(out_dir), name="logs")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="best",
        monitor=str(cfg.get("monitor", "val/loss")),
        mode=str(cfg.get("monitor_mode", "min")),
        save_top_k=1,
        save_last=True,
    )

    accelerator = "gpu" if (str(cfg.get("device", "cpu")).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    devices = int(cfg.get("devices", 1))

    trainer = pl.Trainer(
        default_root_dir=str(out_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger_csv,
        callbacks=[ckpt_cb],
        log_every_n_steps=int(cfg.get("log_every", 50)),
        gradient_clip_val=grad_clip,
        enable_progress_bar=bool(cfg.get("progress_bar", True)),
        deterministic=deterministic,
    )

    trainer.fit(module, datamodule=dm)

    # Evaluate test set using the best checkpoint.
    if ds_test is not None:
        best_path = getattr(ckpt_cb, "best_model_path", None) or "best"

        # PyTorch 2.6 changed torch.load default to weights_only=True.
        # Some Lightning versions pass weights_only=True when loading ckpts,
        # which can fail for older checkpoints or checkpoints containing
        # non-tensor objects. Prefer a safe fallback that works for local ckpts.
        try:
            trainer.test(datamodule=dm, ckpt_path=best_path, weights_only=False)
        except TypeError:
            # Older PL versions don't expose weights_only kwarg.
            if isinstance(best_path, str) and best_path == "best":
                ckpt_path = getattr(ckpt_cb, "best_model_path", "")
            else:
                ckpt_path = str(best_path)
            if not ckpt_path:
                raise
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            module.load_state_dict(state_dict, strict=False)
            trainer.test(model=module, datamodule=dm, ckpt_path=None)

    with open(out_dir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(final_feature_cols, f, ensure_ascii=False, indent=2)

    if trajectory_meta:
        with open(out_dir / "trajectory_feature_meta.json", "w", encoding="utf-8") as f:
            json.dump(trajectory_meta, f, ensure_ascii=False, indent=2)

    # Persist config for reproducibility
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    logger.info("Done. Saved to %s", str(out_dir))


def main() -> None:
    args = _parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
