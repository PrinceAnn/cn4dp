#!/usr/bin/env python
"""Distillation alignment training entrypoint.

This script intentionally lives under `scripts/distill/`.

It trains a phenotype encoder network to regress to teacher embeddings computed
on-the-fly from Delphi disease trajectories.

Run:

    python scripts/distill/train_align.py --config configs/demo/distill.yaml

Config notes
------------
This trainer builds per-patient Delphi trajectories directly from onset-age tables
and runs the pretrained Delphi transformer during training/validation.

Pooling options (applied inside this script on Delphi hidden states):

- `teacher_pooling`: `mean` (default), `last`, or `senet`.
    - `mean`: average over valid timesteps.
    - `last`: last valid timestep.
    - `senet`: **learnable** token-wise gated pooling: an MLP produces a scalar score per timestep, softmax gives weights,
        then we compute a weighted sum of the original vectors (i.e., scalar * vector, summed over time).
- `teacher_senet_reduction` (default: 16)
- `teacher_senet_dropout` (default: 0.0)
- `teacher_senet_temperature` (default: 1.0)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

try:
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tqdm is required: pip install tqdm") from exc

from torch.utils.data import DataLoader

# Allow running this file directly (python scripts/distill/train_align.py ...)
# by adding the repository root to sys.path so local packages resolve.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenotype_encoder.data import build_preprocess_pipeline, load_phenotype_table
from phenotype_encoder.model import build_encoder_from_config

from scripts.downstream.utils import load_delphi_checkpoint, load_labels  # type: ignore


MASK_TIME = -10000.0


class DelphiHiddenExtractor(nn.Module):
    """Wrap Delphi and expose final hidden states (after ln_f) via a permanent hook."""

    def __init__(
        self,
        delphi: nn.Module,
        *,
        pooling: str = "mean",
        se_reduction: int = 16,
        se_dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.delphi = delphi
        self._last_hidden: torch.Tensor | None = None
        self.pooling = str(pooling).lower()
        self.temperature = float(temperature)

        d = int(getattr(getattr(delphi, "config", None), "n_embd"))
        r = max(1, int(se_reduction))
        hidden = max(1, d // r)

        if self.pooling in {"senet", "se"}:
            drop = nn.Dropout(float(se_dropout)) if se_dropout and se_dropout > 0 else nn.Identity()
            self.token_gate: nn.Module | None = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                drop,
                nn.Linear(hidden, 1),
            )
        elif self.pooling in {"mean", "avg", "average"}:
            self.token_gate = None
        else:
            raise ValueError(f"Unsupported pooling={pooling!r}. Use one of: mean | senet.")

        def _hook(_module, _inputs, output):
            self._last_hidden = output

        # Delphi keeps ln_f under transformer.ln_f
        getattr(self.delphi, "transformer").ln_f.register_forward_hook(_hook)

    def encode(self, x: torch.Tensor, a: torch.Tensor, *, dtype: torch.dtype, delphi_no_grad: bool) -> torch.Tensor:
        """Return pooled embedding over non-pad tokens: (B, D)."""
        self._last_hidden = None

        if x.is_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                if delphi_no_grad:
                    with torch.no_grad():
                        _ = self.delphi(x, a)
                else:
                    _ = self.delphi(x, a)
        else:
            if delphi_no_grad:
                with torch.no_grad():
                    _ = self.delphi(x, a)
            else:
                _ = self.delphi(x, a)

        if self._last_hidden is None:
            raise RuntimeError("Failed to capture Delphi hidden state from ln_f")

        hid = self._last_hidden  # (B,T,D)
        hid_fp32 = hid.to(dtype=torch.float32)
        mask_bool = x > 0

        if self.pooling in {"mean", "avg", "average"}:
            m = mask_bool.to(dtype=torch.float32)
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (hid_fp32 * m.unsqueeze(-1)).sum(dim=1) / denom

        if self.token_gate is None:
            raise RuntimeError("token_gate is not initialized for weighted pooling")

        scores = self.token_gate(hid_fp32).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(~mask_bool, -1e9)
        temp = self.temperature if self.temperature else 1.0
        scores = scores / max(1e-6, float(temp))
        w = torch.softmax(scores, dim=1)
        return (hid_fp32 * w.unsqueeze(-1)).sum(dim=1)


def _collate_left_pad_tokens(batch: list[dict[str, Any]], *, mask_time: float = MASK_TIME) -> dict[str, torch.Tensor]:
    """Collate variable-length token/age sequences with left padding.

    Returns:
      x_tokens: (B, T)
      x_ages:   (B, T)
      x:        (B, F) phenotype features
    """
    max_len = max(int(b["tokens"].numel()) for b in batch)
    B = len(batch)

    x_tokens = torch.zeros((B, max_len), dtype=torch.long)
    x_ages = torch.full((B, max_len), float(mask_time), dtype=torch.float32)
    x = torch.stack([b["x"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        t = int(b["tokens"].numel())
        x_tokens[i, -t:] = b["tokens"]
        x_ages[i, -t:] = b["ages_days"]

    return {"x": x, "x_tokens": x_tokens, "x_ages": x_ages}


class TrajPool(nn.Module):
    """Pool a (B,T,D) teacher trajectory to (B,D)."""

    def __init__(self, *, mode: str, dim: int, se_reduction: int = 16, se_dropout: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.mode = str(mode).lower()
        self.temperature = float(temperature)

        self.token_gate: nn.Module | None = None
        if self.mode in {"senet", "se"}:
            r = max(1, int(se_reduction))
            hidden = max(1, int(dim // r))
            self.token_gate = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(float(se_dropout)) if se_dropout and se_dropout > 0 else nn.Identity(),
                nn.Linear(hidden, 1),
            )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """seq: (B,T,D), mask: (B,T) bool where True means valid token."""
        if self.mode in {"mean", "avg", "average"}:
            denom = torch.clamp(mask.sum(dim=1, keepdim=True).to(seq.dtype), min=1.0)
            return (seq * mask.unsqueeze(-1).to(seq.dtype)).sum(dim=1) / denom
        if self.mode in {"last"}:
            # last valid token per row
            idx = mask.long().sum(dim=1) - 1
            idx = torch.clamp(idx, min=0)
            return seq[torch.arange(seq.shape[0], device=seq.device), idx]
        if self.mode in {"senet", "se"}:
            if self.token_gate is None:
                raise RuntimeError("token_gate not initialized")
            scores = self.token_gate(seq).squeeze(-1)  # (B,T)
            scores = scores.masked_fill(~mask, -1e9)
            temp = self.temperature if self.temperature else 1.0
            scores = scores / max(1e-6, float(temp))
            w = torch.softmax(scores, dim=1)  # (B,T)
            return (seq * w.unsqueeze(-1)).sum(dim=1)
        raise ValueError(f"Unsupported pooling mode: {self.mode}")


class ProjectionHead(nn.Module):
    """Small MLP to map between embedding spaces.

    Used when `teacher_dim != encoder.out_dim`.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.0, act: str = "relu"):
        super().__init__()
        hidden_dims = hidden_dims or []

        if act == "relu":
            act_fn = nn.ReLU
        elif act == "gelu":
            act_fn = nn.GELU
        elif act in {"silu", "swish"}:
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {act}")

        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(act_fn())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = int(h)
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveDistillLoss(nn.Module):
    """In-batch contrastive distillation loss (NT-Xent / InfoNCE).

    We treat each (student_i, teacher_i) pair in the batch as a positive match.
    All other teachers in the same batch act as negatives for student_i.

    This is a simple and effective way to do distillation alignment without
    requiring labels.

    Inputs:
      student: (B, D)
      teacher: (B, D)

    Returns:
      scalar loss
    """

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        normalize: bool = True,
        symmetric: bool = True,
        gather_with_grad: bool = False,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.symmetric = bool(symmetric)
        self.gather_with_grad = bool(gather_with_grad)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.dim() != 2 or teacher.dim() != 2:
            raise ValueError(f"Expected 2D (B,D) tensors, got {tuple(student.shape)} and {tuple(teacher.shape)}")
        if student.shape[0] != teacher.shape[0]:
            raise ValueError("student and teacher must have same batch size")
        if student.shape[1] != teacher.shape[1]:
            raise ValueError("student and teacher must have same embedding dim")

        z_s = student
        z_t = teacher
        if self.normalize:
            z_s = F.normalize(z_s, dim=-1)
            z_t = F.normalize(z_t, dim=-1)

        # Teacher acts as the "class prototypes" for each student.
        # logits[i, j] = sim(student_i, teacher_j)
        temp = self.temperature if self.temperature else 0.07
        logits = (z_s @ z_t.t()) / max(1e-6, float(temp))
        targets = torch.arange(logits.shape[0], device=logits.device)

        loss_s2t = F.cross_entropy(logits, targets)
        if not self.symmetric:
            return loss_s2t

        # Symmetric direction: teacher matches student (helps stabilize).
        loss_t2s = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_s2t + loss_t2s)


logger = logging.getLogger("distill.align")


def _configure_logging(level: str = "INFO") -> None:
    if logger.handlers:
        return
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phenotype encoder alignment to Delphi teacher embeddings")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _device_from_cfg(cfg: dict) -> torch.device:
    dev = str(cfg.get("device", "cpu"))
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    return torch.device(dev)


def _split_indices(n: int, seed: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


class PhenotypeTeacherOnlineDataset(torch.utils.data.Dataset):
    """Phenotype rows filtered/enriched with an online teacher trajectory spec.

    Each item returns:
      - x: phenotype vector (F,)
      - tokens: disease token ids (T,) (shifted +1 and includes query token)
      - ages_days: ages aligned to tokens (T,)

    Trajectories are constructed by masking out post-imaging disease events,
    following `scripts/downstream/train_delphi_trajectory_head.py`.
    """

    def __init__(
        self,
        *,
        phenotype_df,
        feature_names: list[str],
        preprocess,
        disease_onset_csv: Path,
        basic_csv: Path | None,
        labels_csv: Path,
        eid_col: str,
        block_size: int,
        append_query_token: bool,
        traj_mode: str = "pre_imaging",
        max_age_years: float | None = None,
    ):
        import math
        import pandas as pd

        self.feature_names = list(feature_names)
        self.preprocess = preprocess
        self.eid_col = str(eid_col)
        self.traj_mode = str(traj_mode).lower()
        self.max_age_years = float(max_age_years) if max_age_years is not None else None

        if self.traj_mode not in {"full_history", "pre_imaging"}:
            raise ValueError(f"Unsupported traj_mode={traj_mode!r}. Use full_history | pre_imaging")

        # Phenotype table indexed by eid for fast lookup.
        ph = phenotype_df.copy()
        ph[self.eid_col] = ph[self.eid_col].astype(int)
        self._pheno_by_eid = ph.set_index(self.eid_col)

        labels_map = load_labels(labels_csv)

        disease = pd.read_csv(disease_onset_csv)
        if self.eid_col not in disease.columns:
            raise ValueError(f"disease_onset_csv must contain column {self.eid_col}")
        disease[self.eid_col] = disease[self.eid_col].astype(int)
        disease = disease.set_index(self.eid_col)

        candidate_cols = [c for c in disease.columns if c in labels_map]
        if not candidate_cols:
            raise ValueError("No disease columns in disease_onset_csv match Delphi labels vocab")

        # Optional per-eid cutoff age (years). Only needed for pre_imaging mode.
        imaging_age_by_eid: dict[int, float] | None = None
        if self.traj_mode == "pre_imaging":
            if basic_csv is None:
                raise ValueError("basic_csv is required when traj_mode=pre_imaging")
            basic = pd.read_csv(basic_csv)
            required = {self.eid_col, "birth_year", "imaging_date"}
            if not required.issubset(basic.columns):
                raise ValueError(f"basic_csv must contain {sorted(required)}")
            if basic[self.eid_col].duplicated().any():
                raise ValueError("basic_csv must contain one row per subject")
            cutoff = basic["imaging_date"].apply(_datetime_to_year_fraction) - pd.to_numeric(basic["birth_year"], errors="raise")
            if not np.isfinite(cutoff).all() or (cutoff <= 0).any():
                raise ValueError("Invalid imaging age in basic_csv")
            imaging_age_by_eid = dict(zip(basic[self.eid_col].astype(int), cutoff))

        self.samples: list[dict[str, Any]] = []
        query_token_shifted = 1  # "No event" 0 -> +1 shift

        # Iterate over phenotype eids (ensures we only build samples we can train on).
        for eid in self._pheno_by_eid.index.astype(int).tolist():
            if int(eid) not in disease.index:
                continue
            row = disease.loc[int(eid)]

            # Determine cutoff/query age.
            cutoff_y: float | None = None
            if imaging_age_by_eid is not None:
                cutoff_y = imaging_age_by_eid.get(int(eid))
            if imaging_age_by_eid is not None and cutoff_y is None:
                raise ValueError(f"Missing imaging date/birth year for subject {eid}")
            if cutoff_y is None:
                cutoff_y = self.max_age_years
            # If still None, we don't apply a cutoff; query age becomes last observed event age.
            ages: list[float] = []
            toks: list[int] = []

            for dis in candidate_cols:
                v = row[dis]
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    continue
                try:
                    onset_age = float(v)
                except Exception:
                    continue
                if not math.isfinite(onset_age):
                    continue
                if cutoff_y is not None and onset_age >= float(cutoff_y):
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

            # Default query age: last event age; if none -> 0.
            query_age_days = float(ages[-1]) if ages else 0.0
            if cutoff_y is not None:
                query_age_days = float(cutoff_y) * 365.25

            if append_query_token:
                ages.append(query_age_days)
                toks.append(query_token_shifted)

            if not toks:
                ages = [query_age_days]
                toks = [query_token_shifted]

            if len(toks) > int(block_size):
                ages = ages[-int(block_size) :]
                toks = toks[-int(block_size) :]

            self.samples.append(
                {
                    "eid": eid,
                    "tokens": torch.tensor(toks, dtype=torch.long),
                    "ages_days": torch.tensor(ages, dtype=torch.float32),
                }
            )

        if not self.samples:
            raise ValueError("No samples built for online teacher alignment; check CSV paths and eid overlap")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[int(idx)]
        eid = int(s["eid"])
        row = self._pheno_by_eid.loc[eid]
        x = row[self.feature_names].to_numpy(dtype=np.float32, copy=True)
        x = self.preprocess.transform(x[None, :])[0]
        return {"pid": eid, "x": torch.from_numpy(x), "tokens": s["tokens"], "ages_days": s["ages_days"]}


def _datetime_to_year_fraction(dt) -> float:
    # Keep dependency-light: use the already-tested helper.
    from scripts.downstream.utils import datetime_to_year_fraction  # type: ignore

    return float(datetime_to_year_fraction(dt))


def train_from_config(config_path: Path) -> None:
    cfg = _load_yaml(config_path)

    _configure_logging(str(cfg.get("log_level", "INFO")))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Data --------
    phenotype_csv = Path(cfg["phenotype_csv"])
    eid_col = str(cfg.get("eid_col", "eid"))
    teacher_mode = str(cfg.get("teacher_mode", "online_delphi")).lower()

    # Pooling config (used by online teacher encoder, and for offline seq->vec pooling).
    teacher_pooling = str(cfg.get("teacher_pooling", "mean"))
    teacher_senet_reduction = int(cfg.get("teacher_senet_reduction", 16))
    teacher_senet_dropout = float(cfg.get("teacher_senet_dropout", 0.0))
    teacher_senet_temperature = float(cfg.get("teacher_senet_temperature", 1.0))

    pheno_df, feature_names = load_phenotype_table(
        phenotype_csv,
        eid_col=eid_col,
    )

    preprocess = build_preprocess_pipeline()
    X_raw = pheno_df[feature_names].to_numpy(dtype=np.float32, copy=True)

    if teacher_mode not in {"online", "online_delphi", "delphi"}:
        raise ValueError(
            f"Unsupported teacher_mode={teacher_mode!r}. This script only supports online Delphi teacher now. "
            "Use teacher_mode=online_delphi."
        )

    if teacher_mode in {"online", "online_delphi", "delphi"}:
        disease_onset_csv = Path(cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv"))
        # basic_csv is only needed for pre_imaging cutoff; full_history does not require it.
        basic_csv_raw = cfg.get("basic_csv", None)
        basic_csv = Path(basic_csv_raw) if basic_csv_raw else None
        labels_csv = Path(cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv"))
        block_size = int(cfg.get("block_size", 256))
        append_query_token = bool(cfg.get("append_query_token", True))
        traj_mode = str(cfg.get("teacher_traj_mode", "pre_imaging"))
        max_age_years = cfg.get("max_age_years", None)

        ds_full: torch.utils.data.Dataset = PhenotypeTeacherOnlineDataset(
            phenotype_df=pheno_df,
            feature_names=feature_names,
            preprocess=preprocess,
            disease_onset_csv=disease_onset_csv,
            basic_csv=basic_csv,
            labels_csv=labels_csv,
            eid_col=eid_col,
            block_size=block_size,
            append_query_token=append_query_token,
            traj_mode=traj_mode,
            max_age_years=max_age_years,
        )

    seed = int(cfg.get("seed", 42))
    train_ratio = float(cfg.get("train_ratio", 0.8))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    tr_idx, va_idx, _ = _split_indices(len(ds_full), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)

    training_ids = {int(ds_full.samples[int(i)]["eid"]) for i in tr_idx}
    preprocess.fit(pheno_df.loc[pheno_df[eid_col].isin(training_ids), feature_names].to_numpy(dtype=np.float32))

    class _Subset(torch.utils.data.Dataset):
        def __init__(self, base, idxs):
            self.base = base
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            return self.base[int(self.idxs[int(i)])]

    ds_train = _Subset(ds_full, tr_idx)
    ds_val = _Subset(ds_full, va_idx) if len(va_idx) else None

    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return _collate_left_pad_tokens(batch)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
    val_loader = (
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=_collate)
        if ds_val is not None
        else None
    )

    # -------- Model --------
    device = _device_from_cfg(cfg)

    dtype_str = str(cfg.get("dtype", "float32")).lower()
    amp_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

    # Student phenotype encoder
    enc_cfg = cfg.get("encoder", {}) or {}
    model, student_dim = build_encoder_from_config(encoder_cfg=enc_cfg, feature_cols=feature_names)
    model = model.to(device)

    teacher_encoder: DelphiHiddenExtractor | None = None
    freeze_delphi = bool(cfg.get("freeze_delphi", True))

    delphi_ckpt_dir = Path(cfg["delphi_ckpt_dir"])
    delphi_model, delphi_conf = load_delphi_checkpoint(delphi_ckpt_dir, device)
    delphi_model.eval()
    if freeze_delphi:
        for p in delphi_model.parameters():
            p.requires_grad_(False)

    teacher_encoder = DelphiHiddenExtractor(
        delphi_model,
        pooling=teacher_pooling,
        se_reduction=teacher_senet_reduction,
        se_dropout=teacher_senet_dropout,
        temperature=teacher_senet_temperature,
    ).to(device)
    teacher_dim = int(getattr(delphi_conf, "n_embd"))

    # Optional projection head if dimensions don't match
    projector_cfg = cfg.get("projector", {}) or {}
    projector_enabled = bool(projector_cfg.get("enabled", True))
    # We project the phenotype embedding (student) into teacher space.
    # This matches the usual distillation alignment setup.
    projector_hidden = projector_cfg.get("hidden_dims", [])
    projector_dropout = float(projector_cfg.get("dropout", 0.0))
    projector_act = str(projector_cfg.get("act", "relu"))

    projector: nn.Module | None = None

    if teacher_dim != int(student_dim):
        if not projector_enabled:
            raise ValueError(
                f"encoder_out_dim={int(student_dim)} but teacher_dim={teacher_dim}. "
                "Enable projector.enabled to map encoder embedding to teacher dim, or set encoder.out_dim to match."
            )

        # map phenotype embedding -> teacher space
        projector = ProjectionHead(
            in_dim=int(student_dim),
            out_dim=teacher_dim,
            hidden_dims=[int(x) for x in projector_hidden],
            dropout=projector_dropout,
            act=projector_act,
        ).to(device)
        logger.info("Using projector: encoder_out %d -> teacher %d", int(student_dim), teacher_dim)

    # NOTE: `teacher_pooling` is handled inside `teacher_encoder` for online Delphi.

    loss_name = str(cfg.get("loss", "mse")).lower()
    if loss_name == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif loss_name == "smooth_l1":
        criterion = nn.SmoothL1Loss(beta=float(cfg.get("smooth_l1_beta", 1.0)))
    elif loss_name == "cosine":
        criterion = lambda a, b: (1.0 - nn.functional.cosine_similarity(a, b, dim=-1)).mean()  # type: ignore
    elif loss_name in {"contrastive", "infonce", "ntxent"}:
        c_cfg = cfg.get("contrastive", {}) or {}
        criterion = ContrastiveDistillLoss(
            temperature=float(c_cfg.get("temperature", 0.07)),
            normalize=bool(c_cfg.get("normalize", True)),
            symmetric=bool(c_cfg.get("symmetric", True)),
        )
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    params = list(model.parameters())
    if projector is not None:
        params += list(projector.parameters())
    if teacher_encoder is not None:
        params += [p for p in teacher_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    epochs = int(cfg.get("epochs", 50))
    log_every = int(cfg.get("log_every", 50))
    save_every = int(cfg.get("save_every", 1))

    best_val = float("inf")
    best_state: Dict[str, Any] | None = None

    logger.info("Train samples=%d, Val samples=%s", len(ds_train), (len(ds_val) if ds_val is not None else "0"))
    logger.info("Features=%d, Student dim=%d", len(feature_names), int(student_dim))

    # Optional early NaN/Inf check to avoid silent NaN losses.
    check_finite = bool(cfg.get("check_finite", True))

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            x = batch["x"].to(device)
            x_tokens = batch["x_tokens"].to(device)
            x_ages = batch["x_ages"].to(device)
            teacher = teacher_encoder.encode(x_tokens, x_ages, dtype=amp_dtype, delphi_no_grad=freeze_delphi)

            if check_finite:
                if not torch.isfinite(x).all():
                    raise ValueError("Found NaN/Inf in phenotype input batch")
                if not torch.isfinite(teacher).all():
                    raise ValueError("Found NaN/Inf in teacher feature batch")

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            if projector is not None:
                pred = projector(pred)

            loss = criterion(pred, teacher)
            loss.backward()
            optimizer.step()

            global_step += 1
            l = float(loss.detach().cpu().item())
            epoch_losses.append(l)
            pbar.set_postfix(loss=f"{l:.4f}")
            if log_every and global_step % log_every == 0:
                logger.info("epoch=%d step=%d loss=%.6f", epoch, global_step, l)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    x_tokens = batch["x_tokens"].to(device)
                    x_ages = batch["x_ages"].to(device)
                    teacher = teacher_encoder.encode(x_tokens, x_ages, dtype=amp_dtype, delphi_no_grad=True)
                    if check_finite:
                        if not torch.isfinite(x).all():
                            raise ValueError("Found NaN/Inf in phenotype input batch (val)")
                        if not torch.isfinite(teacher).all():
                            raise ValueError("Found NaN/Inf in teacher feature batch (val)")
                    pred = model(x)
                    if projector is not None:
                        pred = projector(pred)
                    loss = criterion(pred, teacher)
                    vals.append(float(loss.detach().cpu().item()))
            val_loss = float(np.mean(vals)) if vals else float("nan")

        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        if val_loader is not None and np.isfinite(val_loss) and val_loss < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if save_every and (epoch % save_every == 0):
            ckpt_path = out_dir / f"ckpt_epoch_{epoch}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "projector_state": (projector.state_dict() if projector is not None else None),
                    "config": cfg,
                    "student_dim": int(student_dim),
                    "feature_names": list(feature_names),
                    "preprocess": preprocess.to_artifacts(),
                },
                ckpt_path,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(
            {
                "model_state": model.state_dict(),
                "projector_state": (projector.state_dict() if projector is not None else None),
                "config": cfg,
                "student_dim": int(student_dim),
                "feature_names": list(feature_names),
                "preprocess": preprocess.to_artifacts(),
            },
            out_dir / "best.pt",
        )

    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(list(feature_names), f, ensure_ascii=False, indent=2)

    logger.info("Done. Saved to %s", str(out_dir))


def main() -> None:
    args = _parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
