#!/usr/bin/env python
"""Generative baseline evaluation for a specific disease (Delphi).

This baseline aligns with `scripts/downstream/train_delphi_trajectory_head.py`:

- Uses the SAME riskset CSV for labels and imaging date.
- Constructs a masked trajectory per person: only disease events with
  onset_age <= imaging_age are kept.
- Appends a query token at imaging time (same as trajectory-head script).

Scoring (baseline)
------------------
We use Delphi as a generative hazard model. For a target disease token k, the
model produces a log-rate (log hazard per day) at the last position.
We convert it to a probability within a chosen horizon (in years):

    hazard_per_year = exp(logit_k) * 365.25
    p_horizon = 1 - exp(-hazard_per_year * horizon_years)

That gives a per-sample probability score, directly comparable across samples
for ROC-AUC.

Notes
-----
- This is a *scoring* baseline (uses the generator's parametric hazard) rather
  than Monte-Carlo sampling. It's much faster and deterministic.
- Token indexing:
  - In Delphi, token 0 is padding, token 1 is "No event".
  - `Delphi/data/demo/labels.csv` is pre-shift (row index is token id in the
    original vocab). We shift by +1 when feeding the model.

Run
---
python scripts/downstream/eval_delphi_generative_baseline.py --config configs/demo/generative.yaml --target_disease_code SYN_TARGET

Config
------
This script reuses most fields from the trajectory-head config and adds:
- horizon_years: float (default 5.0)
- target_disease_code: str (can be passed via CLI)

Outputs
-------
Writes `baseline_metrics.json` and `baseline_predictions.csv` in out_dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Delphi.model import Delphi, DelphiConfig  # type: ignore

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


logger = logging.getLogger("downstream.delphi_gen_baseline")

MASK_TIME = -10000.0


def _configure_logging(level: str = "INFO") -> None:
    configure_logger(logger, level=level)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_everything(seed: int) -> None:
    seed_everything(int(seed), deterministic=True)


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def _datetime_to_year_fraction(dt) -> float:
    return float(datetime_to_year_fraction(dt))


def _load_delphi_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[Delphi, DelphiConfig]:
    return load_delphi_checkpoint(ckpt_dir, device)


def _load_labels(labels_csv: Path) -> Dict[str, int]:
    return load_labels(labels_csv)


@dataclass
class Sample:
    eid: int
    tokens: torch.Tensor  # (T,) int64, shifted (+1)
    ages_days: torch.Tensor  # (T,) float32
    y: int


def _build_samples(
    *,
    riskset_csv: Path,
    disease_onset_csv: Path,
    basic_csv: Path,
    labels_csv: Path,
    eid_col: str,
    label_col: str,
    imaging_date_col: str,
    block_size: int,
    append_query_token: bool,
) -> Tuple[List[Sample], Dict[str, int]]:
    import pandas as pd

    rs = pd.read_csv(riskset_csv)
    if eid_col not in rs.columns or label_col not in rs.columns:
        raise ValueError(f"riskset_csv must contain columns {eid_col} and {label_col}")
    if imaging_date_col not in rs.columns:
        raise ValueError(f"riskset_csv must contain column {imaging_date_col}")

    rs[eid_col] = rs[eid_col].astype(int)
    rs[label_col] = rs[label_col].astype(int)

    rs[imaging_date_col] = pd.to_datetime(rs[imaging_date_col], errors="coerce")
    if rs[imaging_date_col].isna().any():
        raise ValueError(f"Found invalid {imaging_date_col} in riskset_csv")

    basic = pd.read_csv(basic_csv)
    if eid_col not in basic.columns:
        raise ValueError(f"basic_csv must contain column {eid_col}")

    birth_year_col = "birth_year"
    basic[eid_col] = basic[eid_col].astype(int)
    basic["birth_year"] = pd.to_numeric(basic[birth_year_col], errors="coerce")
    basic = basic[[eid_col, "birth_year"]]

    rs = rs.merge(basic, on=eid_col, how="left")
    if rs["birth_year"].isna().any():
        missing = int(rs["birth_year"].isna().sum())
        raise ValueError(f"Missing birth_year for {missing} eids after merge with basic_csv")

    labels_map = _load_labels(labels_csv)

    disease = pd.read_csv(disease_onset_csv)
    if eid_col not in disease.columns:
        raise ValueError(f"disease_onset_csv must contain column {eid_col}")
    disease[eid_col] = disease[eid_col].astype(int)
    disease = disease.set_index(eid_col)

    candidate_cols = [c for c in disease.columns if c in labels_map]
    if not candidate_cols:
        raise ValueError("No disease columns in disease_onset_csv match Delphi labels vocab")

    eids = rs[eid_col].to_numpy(dtype=np.int64)
    disease_sub = disease.reindex(eids)

    imaging_year = rs[imaging_date_col].apply(_datetime_to_year_fraction).to_numpy(dtype=np.float64)
    birth_year = rs["birth_year"].to_numpy(dtype=np.float64)
    imaging_age_years = imaging_year - birth_year

    samples: List[Sample] = []

    query_token_shifted = 1

    for i in range(len(rs)):
        eid = int(eids[i])
        y_i = int(rs.loc[i, label_col])

        img_age_y = float(imaging_age_years[i])
        if not math.isfinite(img_age_y) or img_age_y <= 0:
            raise ValueError(f"Invalid imaging_age_years for eid={eid}: {img_age_y}")
        img_age_days = img_age_y * 365.25

        row = disease_sub.iloc[i]
        ages: List[float] = []
        toks: List[int] = []

        for dis in candidate_cols:
            v = row[dis]
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            try:
                onset_age = float(v)
            except Exception:
                continue
            if not math.isfinite(onset_age) or onset_age >= img_age_y:
                continue

            tok_pre_shift = int(labels_map[dis])
            tok = tok_pre_shift + 1
            if tok <= 0:
                continue
            ages.append(onset_age * 365.25)
            toks.append(tok)

        if ages:
            order = sorted(range(len(ages)), key=lambda j: (ages[j], toks[j]))
            ages = [ages[j] for j in order]
            toks = [toks[j] for j in order]

        if append_query_token:
            ages.append(img_age_days)
            toks.append(query_token_shifted)

        if not toks:
            ages = [img_age_days]
            toks = [query_token_shifted]

        if len(toks) > int(block_size):
            ages = ages[-int(block_size) :]
            toks = toks[-int(block_size) :]

        samples.append(
            Sample(
                eid=eid,
                tokens=torch.tensor(toks, dtype=torch.long),
                ages_days=torch.tensor(ages, dtype=torch.float32),
                y=y_i,
            )
        )

    return samples, labels_map


def _split_indices(n: int, seed: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split_indices(n, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)


def _left_pad(tokens: torch.Tensor, ages: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Defensive: some configs may build trajectories longer than the checkpoint
    # block_size. Clip from the left (keep most recent events, incl. query token).
    if int(tokens.numel()) > int(max_len):
        tokens = tokens[-int(max_len) :]
        ages = ages[-int(max_len) :]

    t = int(tokens.numel())
    x = torch.zeros((max_len,), dtype=torch.long)
    a = torch.full((max_len,), float(MASK_TIME), dtype=torch.float32)
    x[-t:] = tokens
    a[-t:] = ages
    return x, a


def _compute_scores(
    *,
    model: Delphi,
    conf: DelphiConfig,
    samples: List[Sample],
    idx_eval: np.ndarray,
    target_token_shifted: int,
    horizon_years: float,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = int(conf.block_size)

    eids_out: List[int] = []
    y_out: List[int] = []
    p_out: List[float] = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(idx_eval), batch_size):
            batch_idx = idx_eval[start : start + batch_size]

            x_list = []
            a_list = []
            e_list = []
            y_list = []

            for j in batch_idx:
                s = samples[int(j)]
                x_j, a_j = _left_pad(s.tokens, s.ages_days, max_len=max_len)
                x_list.append(x_j)
                a_list.append(a_j)
                e_list.append(s.eid)
                y_list.append(int(s.y))

            x = torch.stack(x_list, dim=0).to(device)
            a = torch.stack(a_list, dim=0).to(device)

            if x.is_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _, _ = model(x, a)
            else:
                logits, _, _ = model(x, a)

            last_logits = logits[:, -1, :]
            logit_k = last_logits[:, int(target_token_shifted)]

            hazard_per_year = torch.exp(logit_k).clamp(min=0) * 365.25
            p_h = 1.0 - torch.exp(-hazard_per_year * float(horizon_years))

            eids_out.extend(e_list)
            y_out.extend(y_list)
            p_out.extend(p_h.detach().float().cpu().numpy().tolist())

    return np.asarray(eids_out, dtype=np.int64), np.asarray(y_out, dtype=np.int64), np.asarray(p_out, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Delphi generative baseline for a specific disease")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target_disease_code", type=str, default=None, help="Disease code string as in labels.csv (pre-shift)")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    _configure_logging(str(cfg.get("log_level", "INFO")))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    dtype_str = str(cfg.get("dtype", "float32")).lower()
    amp_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

    horizon_years = float(cfg.get("horizon_years", 5.0))

    riskset_csv = Path(cfg["riskset_csv"])
    disease_onset_csv = Path(cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv"))
    basic_csv = Path(cfg.get("basic_csv", "data/demo/basic_info.csv"))
    labels_csv = Path(cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv"))

    eid_col = str(cfg.get("eid_col", "eid"))
    label_col = str(cfg.get("label_col", "label"))
    imaging_date_col = str(cfg.get("imaging_date_col", "imaging_date"))

    block_size = int(cfg.get("block_size", 256))
    append_query_token = bool(cfg.get("append_query_token", True))

    target_code = args.target_disease_code or cfg.get("target_disease_code", None)
    if not target_code:
        raise SystemExit("Need --target_disease_code or config target_disease_code")

    logger.info("Building masked trajectories (post-imaging events dropped)...")
    samples, labels_map = _build_samples(
        riskset_csv=riskset_csv,
        disease_onset_csv=disease_onset_csv,
        basic_csv=basic_csv,
        labels_csv=labels_csv,
        eid_col=eid_col,
        label_col=label_col,
        imaging_date_col=imaging_date_col,
        block_size=block_size,
        append_query_token=append_query_token,
    )

    if target_code not in labels_map:
        raise ValueError(f"target_disease_code={target_code!r} not found in labels vocab")

    target_token_shifted = int(labels_map[target_code]) + 1

    tr, va, te = split_subject_indices(
        np.array([int(s.eid) for s in samples]), seed=seed,
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        split_csv=cfg.get("split_csv"), eid_col=eid_col,
    )

    idx_eval = {"train": tr, "val": va, "test": te}[args.split]

    device_str = str(cfg.get("device", "cpu"))
    device = torch.device("cuda" if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    delphi_ckpt_dir = Path(cfg.get("delphi_ckpt_dir", "runs/demo/teacher"))
    logger.info("Loading Delphi checkpoint from %s", str(delphi_ckpt_dir))
    model, conf = _load_delphi_checkpoint(delphi_ckpt_dir, device=device)

    batch_size = int(cfg.get("eval_batch_size", cfg.get("batch_size", 64)))

    logger.info(
        "Scoring split=%s target=%s (token_shifted=%d) horizon=%.2f years ...",
        args.split,
        target_code,
        target_token_shifted,
        horizon_years,
    )

    eids, y_true, y_score = _compute_scores(
        model=model,
        conf=conf,
        samples=samples,
        idx_eval=idx_eval,
        target_token_shifted=target_token_shifted,
        horizon_years=horizon_years,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
    )

    auc = _roc_auc_score(y_true, y_score)
    acc = float(((y_score >= 0.5).astype(np.int64) == y_true.astype(np.int64)).mean())

    metrics = {
        "split": args.split,
        "target_disease_code": target_code,
        "target_token_shifted": int(target_token_shifted),
        "horizon_years": float(horizon_years),
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auc": float(auc) if auc == auc else None,
        "acc@0.5": float(acc),
    }

    (out_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import pandas as pd

        df = pd.DataFrame({"eid": eids, "y": y_true, "score": y_score})
        df.to_csv(out_dir / "baseline_predictions.csv", index=False)
    except Exception as e:
        logger.warning("Failed to write baseline_predictions.csv (pandas missing?): %s", e)

    logger.info("Done. AUC=%s  n_pos=%d n_neg=%d", metrics["auc"], metrics["n_pos"], metrics["n_neg"])


if __name__ == "__main__":
    main()
