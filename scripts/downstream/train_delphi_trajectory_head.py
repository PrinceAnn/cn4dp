#!/usr/bin/env python
"""Train a multi-task head on Delphi disease-trajectory embeddings (Lightning).

Goal
----
Use a pretrained Delphi transformer as an encoder for disease trajectories, then
train a small multi-task head for:

1) Binary classification: whether the target disease occurs (label).
2) Regression: time-to-onset (delta_time) for incident cases.

Key requirement
---------------
Disease events after the imaging time must be masked out (not provided to the
encoder). This script constructs, per sample, a sequence of disease tokens
sorted by onset time and keeps only events with onset_age <= imaging_age.

Implementation notes
--------------------
- Uses Lightning Trainer + checkpointing.
- Does NOT save intermediate embeddings; embeddings are computed on-the-fly.
- Default is to freeze Delphi and train only the head.

Run
---
python scripts/downstream/train_delphi_trajectory_head.py --config configs/demo/trajectory.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader


# Allow running directly.
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


def _import_lightning():
    """Import Lightning with compatibility across package variants."""
    try:
        import lightning.pytorch as pl  # type: ignore
        from lightning.pytorch.callbacks import ModelCheckpoint  # type: ignore
        from lightning.pytorch.loggers import CSVLogger  # type: ignore
        return pl, ModelCheckpoint, CSVLogger
    except Exception:
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore
        from pytorch_lightning.loggers import CSVLogger  # type: ignore
        return pl, ModelCheckpoint, CSVLogger


pl, ModelCheckpoint, CSVLogger = _import_lightning()


logger = logging.getLogger("downstream.delphi_traj")


MASK_TIME = -10000.0


def _configure_logging(level: str = "INFO") -> None:
    configure_logger(logger, level=level)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train head on Delphi trajectory embeddings")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    seed_everything(int(seed), deterministic=deterministic, lightning_seed_fn=getattr(pl, "seed_everything", None))


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def _datetime_to_year_fraction(dt) -> float:
    return float(datetime_to_year_fraction(dt))


def _load_delphi_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[Delphi, DelphiConfig]:
    return load_delphi_checkpoint(ckpt_dir, device)


def _load_labels(labels_csv: Path) -> Dict[str, int]:
    mapping = load_labels(labels_csv)
    if mapping.get("No event", None) != 0:
        logger.warning("labels[0] is not 'No event' (got %r)", next(iter(mapping.keys()), None))
    return mapping


@dataclass
class Sample:
    eid: torch.Tensor  # () int64
    tokens: torch.Tensor  # (T,) int64, already shifted (+1) and includes query token
    ages_days: torch.Tensor  # (T,) float32
    y: torch.Tensor  # () float32
    delta_time: torch.Tensor  # () float32 (filled)
    dt_mask: torch.Tensor  # () float32


class DelphiTrajDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def _collate_left_pad(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    max_len = max(int(s.tokens.numel()) for s in batch)
    B = len(batch)

    eid = torch.zeros((B,), dtype=torch.long)
    x = torch.zeros((B, max_len), dtype=torch.long)
    a = torch.full((B, max_len), float(MASK_TIME), dtype=torch.float32)

    y = torch.zeros((B,), dtype=torch.float32)
    dt = torch.zeros((B,), dtype=torch.float32)
    dt_mask = torch.zeros((B,), dtype=torch.float32)

    for i, s in enumerate(batch):
        eid[i] = s.eid
        t = int(s.tokens.numel())
        x[i, -t:] = s.tokens
        a[i, -t:] = s.ages_days
        y[i] = s.y
        dt[i] = s.delta_time
        dt_mask[i] = s.dt_mask

    return {"eid": eid, "x": x, "a": a, "y": y, "delta_time": dt, "dt_mask": dt_mask}


class DelphiHiddenExtractor(nn.Module):
    """Wrap Delphi and expose final hidden states (after ln_f) via a permanent hook."""

    def __init__(
        self,
        delphi: Delphi,
        *,
        pooling: str = "mean",
        se_reduction: int = 16,
        se_dropout: float = 0.0,
    ):
        super().__init__()
        self.delphi = delphi
        self._last_hidden: Optional[torch.Tensor] = None
        self.pooling = str(pooling).lower()

        # Token-wise gating network ("SENet-like"), producing a scalar weight per token.
        # This is closer to attention pooling than channel-SE, but matches the requested behavior.
        d = int(getattr(delphi.config, "n_embd"))
        r = max(1, int(se_reduction))
        hidden = max(1, d // r)

        if self.pooling == "senet":
            drop = nn.Dropout(float(se_dropout)) if se_dropout and se_dropout > 0 else nn.Identity()
            self.token_gate = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                drop,
                nn.Linear(hidden, 1),
            )
        elif self.pooling in {"mean"}:
            self.token_gate = None
        else:
            raise ValueError(
                f"Unsupported pooling={pooling!r}. Use one of: mean | senet."
            )

        def _hook(_module, _inputs, output):
            self._last_hidden = output

        self.delphi.transformer.ln_f.register_forward_hook(_hook)

    @torch.no_grad()
    def encode_last(self, x: torch.Tensor, a: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """Backward-compatible alias; uses the configured pooling."""
        return self.encode(x, a, dtype=dtype, delphi_no_grad=True)

    def encode(self, x: torch.Tensor, a: torch.Tensor, *, dtype: torch.dtype, delphi_no_grad: bool) -> torch.Tensor:
        """Return pooled embedding over non-pad tokens: (B, D).

        pooling:
        - mean: average over non-pad tokens
        - se/weighted/attn: token-wise gated weighted sum (softmax weights)
        """
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

        hid = self._last_hidden  # (B, T, D)
        hid_fp32 = hid.to(dtype=torch.float32)

        mask_bool = x > 0  # (B, T)
        if self.pooling in {"mean", "avg", "average"}:
            mask = mask_bool.to(dtype=torch.float32)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (hid_fp32 * mask.unsqueeze(-1)).sum(dim=1) / denom
            return pooled

        # token-wise weights
        if self.token_gate is None:
            raise RuntimeError("token_gate is not initialized for weighted pooling")

        scores = self.token_gate(hid_fp32).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask_bool, -1e9)
        w = torch.softmax(scores, dim=1)  # (B, T)
        pooled = (hid_fp32 * w.unsqueeze(-1)).sum(dim=1)
        return pooled


class HeadMultiTask(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.cls = nn.Linear(in_dim, 1)
        self.time = nn.Linear(in_dim, 1)
        self.time_act = nn.Softplus()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.drop(z)
        logit = self.cls(z).squeeze(-1)
        dt = self.time_act(self.time(z).squeeze(-1))
        return logit, dt


class DelphiTrajDataModule(pl.LightningDataModule):  # type: ignore[misc]
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
            collate_fn=_collate_left_pad,
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
            collate_fn=_collate_left_pad,
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
            collate_fn=_collate_left_pad,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
        )


class MultiTaskDelphiTrajModule(pl.LightningModule):  # type: ignore[misc]
    def __init__(
        self,
        *,
        encoder: DelphiHiddenExtractor,
        head: HeadMultiTask,
        cfg: dict,
        amp_dtype: torch.dtype,
        pos_weight: float | None,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.cfg = cfg
        self.amp_dtype = amp_dtype

        self.freeze_delphi = bool(cfg.get("freeze_delphi", True))
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

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep Delphi optionally frozen, but allow gradients for pooling/head params.
        z = self.encoder.encode(x, a, dtype=self.amp_dtype, delphi_no_grad=self.freeze_delphi)
        return self.head(z)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        eid = batch.get("eid", None)
        x = batch["x"]
        a = batch["a"]
        y = batch["y"]
        dt = batch["delta_time"]
        dt_mask = batch["dt_mask"]

        logit, dt_pred = self(x, a)

        loss_cls = self.criterion_cls(logit, y)
        per_ex = self.criterion_time(dt_pred, dt)
        denom = torch.clamp(dt_mask.sum(), min=1.0)
        loss_time = (per_ex * dt_mask).sum() / denom

        loss = self.w_cls * loss_cls + self.w_time * loss_time

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_time", loss_time, on_step=False, on_epoch=True)

        # Collect epoch metrics on CPU.
        logit_np = logit.detach().cpu().numpy()
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        dt_pred_np = dt_pred.detach().cpu().numpy()
        dt_np = dt.detach().cpu().numpy()
        m_np = dt_mask.detach().cpu().numpy().astype(bool)

        if stage == "val":
            self._val_y.append(y_np)
            self._val_p.append(prob)
            if m_np.any():
                self._val_dt_y.append(dt_np[m_np])
                self._val_dt_p.append(dt_pred_np[m_np])
        elif stage == "test":
            self._test_y.append(y_np)
            self._test_p.append(prob)
            if eid is not None:
                self._test_eid.append(eid.detach().cpu().numpy().astype(np.int64))
            self._test_logit.append(logit_np)
            self._test_dt.append(dt_np)
            self._test_dt_pred.append(dt_pred_np)
            self._test_dt_mask.append(m_np.astype(np.int64))
            if m_np.any():
                self._test_dt_y.append(dt_np[m_np])
                self._test_dt_p.append(dt_pred_np[m_np])

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
        self.log("val/auc", _roc_auc_score(y, p), prog_bar=True)
        self.log("val/acc", float(((p >= 0.5).astype(np.int64) == y.astype(np.int64)).mean()))

        if self._val_dt_y:
            dt_y = np.concatenate(self._val_dt_y)
            dt_p = np.concatenate(self._val_dt_p)
            self.log("val/dt_mae", float(np.mean(np.abs(dt_p - dt_y))), prog_bar=True)
        else:
            self.log("val/dt_mae", float("nan"))

        self._val_y.clear()
        self._val_p.clear()
        self._val_dt_y.clear()
        self._val_dt_p.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_y:
            return
        y = np.concatenate(self._test_y)
        p = np.concatenate(self._test_p)
        self.log("test/auc", _roc_auc_score(y, p))
        self.log("test/acc", float(((p >= 0.5).astype(np.int64) == y.astype(np.int64)).mean()))

        # Save per-sample predictions for downstream analysis.
        try:
            out_dir = Path(self.cfg["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.csv"

            eid = np.concatenate(self._test_eid) if self._test_eid else np.full_like(y, fill_value=-1, dtype=np.int64)
            logit = (
                np.concatenate(self._test_logit)
                if self._test_logit
                else np.full_like(p, fill_value=np.nan, dtype=np.float32)
            )
            dt = np.concatenate(self._test_dt) if self._test_dt else np.full_like(p, fill_value=np.nan, dtype=np.float32)
            dt_pred = (
                np.concatenate(self._test_dt_pred)
                if self._test_dt_pred
                else np.full_like(p, fill_value=np.nan, dtype=np.float32)
            )
            dt_mask = (
                np.concatenate(self._test_dt_mask)
                if self._test_dt_mask
                else np.zeros_like(y, dtype=np.int64)
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
            self.log("test/dt_mae", float(np.mean(np.abs(dt_p - dt_y))))
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
        lr = float(self.cfg.get("lr", 1e-3))
        weight_decay = float(self.cfg.get("weight_decay", 0.0))
        trainable = [p for p in list(self.head.parameters()) + list(self.encoder.parameters()) if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        warmup_steps = self.cfg.get("warmup_steps", None)
        warmup_ratio = float(self.cfg.get("warmup_ratio", 0.1))
        min_lr_ratio = float(self.cfg.get("min_lr_ratio", 0.1))

        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps <= 0:
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
            progress = float(step - warmup_steps_i) / float(max(1, total_steps - warmup_steps_i))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Keep checkpoint weights_only-safe (no numpy objects).
        checkpoint["config"] = self.cfg


def _split_indices(n: int, seed: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split_indices(n, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)


def _build_samples(
    *,
    riskset_csv: Path,
    disease_onset_csv: Path,
    basic_csv: Path,
    labels_csv: Path,
    eid_col: str,
    label_col: str,
    delta_time_col: str,
    imaging_date_col: str,
    block_size: int,
    append_query_token: bool,
) -> List[Sample]:
    import pandas as pd

    rs = pd.read_csv(riskset_csv)
    if eid_col not in rs.columns or label_col not in rs.columns:
        raise ValueError(f"riskset_csv must contain columns {eid_col} and {label_col}")
    if imaging_date_col not in rs.columns:
        raise ValueError(f"riskset_csv must contain column {imaging_date_col}")

    rs[eid_col] = rs[eid_col].astype(int)
    rs[label_col] = rs[label_col].astype(int)
    if delta_time_col in rs.columns:
        rs[delta_time_col] = pd.to_numeric(rs[delta_time_col], errors="coerce")
    else:
        rs[delta_time_col] = np.nan

    rs[imaging_date_col] = pd.to_datetime(rs[imaging_date_col], errors="coerce")
    if rs[imaging_date_col].isna().any():
        raise ValueError(f"Found invalid {imaging_date_col} in riskset_csv")

    basic = pd.read_csv(basic_csv)
    if eid_col not in basic.columns:
        raise ValueError(f"basic_csv must contain column {eid_col}")

    birth_year_col = 'birth_year'

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

    query_token_shifted = 1  # "No event" row 0 -> +1 shift

    for i in range(len(rs)):
        eid = int(eids[i])
        y_i = float(rs.loc[i, label_col])
        dt_raw = rs.loc[i, delta_time_col]
        dt_val = float(dt_raw) if dt_raw is not None and math.isfinite(float(dt_raw)) else float("nan")
        dt_mask = 1.0 if (y_i == 1.0 and math.isfinite(dt_val)) else 0.0
        dt_filled = dt_val if math.isfinite(dt_val) else 0.0

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
                eid=torch.tensor(eid, dtype=torch.long),
                tokens=torch.tensor(toks, dtype=torch.long),
                ages_days=torch.tensor(ages, dtype=torch.float32),
                y=torch.tensor(y_i, dtype=torch.float32),
                delta_time=torch.tensor(dt_filled, dtype=torch.float32),
                dt_mask=torch.tensor(dt_mask, dtype=torch.float32),
            )
        )

    return samples


def train_from_config(config_path: Path) -> None:
    cfg = _load_yaml(config_path)
    _configure_logging(str(cfg.get("log_level", "INFO")))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    _seed_everything(seed, deterministic=deterministic)

    dtype_str = str(cfg.get("dtype", "float32")).lower()
    amp_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

    riskset_csv = Path(cfg["riskset_csv"])
    disease_onset_csv = Path(cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv"))
    basic_csv = Path(cfg.get("basic_csv", "data/demo/basic_info.csv"))
    labels_csv = Path(cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv"))

    eid_col = str(cfg.get("eid_col", "eid"))
    label_col = str(cfg.get("label_col", "label"))
    delta_time_col = str(cfg.get("delta_time_col", "delta_time"))
    imaging_date_col = str(cfg.get("imaging_date_col", "imaging_date"))

    block_size = int(cfg.get("block_size", 256))
    append_query_token = bool(cfg.get("append_query_token", True))

    logger.info("Building masked trajectories (post-imaging events dropped)...")
    samples = _build_samples(
        riskset_csv=riskset_csv,
        disease_onset_csv=disease_onset_csv,
        basic_csv=basic_csv,
        labels_csv=labels_csv,
        eid_col=eid_col,
        label_col=label_col,
        delta_time_col=delta_time_col,
        imaging_date_col=imaging_date_col,
        block_size=block_size,
        append_query_token=append_query_token,
    )

    tr, va, te = split_subject_indices(
        np.array([int(s.eid) for s in samples]), seed=seed,
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        split_csv=cfg.get("split_csv"), eid_col=eid_col,
    )

    ds_train = DelphiTrajDataset([samples[i] for i in tr])
    ds_val = DelphiTrajDataset([samples[i] for i in va]) if len(va) else None
    ds_test = DelphiTrajDataset([samples[i] for i in te]) if len(te) else None

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    device_str = str(cfg.get("device", "cpu"))
    default_pin = bool(device_str.startswith("cuda") and torch.cuda.is_available())
    pin_memory = bool(cfg.get("pin_memory", default_pin))

    dm = DelphiTrajDataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    delphi_ckpt_dir = Path(cfg.get("delphi_ckpt_dir", "runs/demo/teacher"))
    logger.info("Loading Delphi checkpoint from %s", str(delphi_ckpt_dir))
    # Load on CPU first; Lightning will move modules to target device.
    delphi, delphi_conf = _load_delphi_checkpoint(delphi_ckpt_dir, device=torch.device("cpu"))

    freeze_delphi = bool(cfg.get("freeze_delphi", True))
    if freeze_delphi:
        for p in delphi.parameters():
            p.requires_grad_(False)
        delphi.eval()

    encoder = DelphiHiddenExtractor(delphi,pooling=str(cfg.get("pooling", "mean")),se_reduction=int(cfg.get("se_reduction", 16)),se_dropout=float(cfg.get("se_dropout", 0.0)),)
    head = HeadMultiTask(in_dim=int(delphi_conf.n_embd), dropout=float(cfg.get("head_dropout", 0.0)))

    pos_weight = cfg.get("pos_weight", None)
    if pos_weight is None:
        y_train = np.array([float(samples[i].y.item()) for i in tr], dtype=np.float32)
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = n_neg / max(1.0, n_pos) if n_pos > 0 else None
    else:
        pos_weight = float(pos_weight)

    module = MultiTaskDelphiTrajModule(
        encoder=encoder,
        head=head,
        cfg=cfg,
        amp_dtype=amp_dtype,
        pos_weight=pos_weight,
    )

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
        try:
            trainer.test(datamodule=dm, ckpt_path=best_path, weights_only=False)
        except TypeError:
            # Older PL versions don't expose weights_only kwarg.
            ckpt_path = str(best_path) if best_path != "best" else str(getattr(ckpt_cb, "best_model_path", ""))
            if not ckpt_path:
                raise
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            module.load_state_dict(state_dict, strict=False)
            trainer.test(model=module, datamodule=dm, ckpt_path=None)

        # Persist metrics summary if available.
        try:
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if isinstance(v, (int, float, torch.Tensor))}
        except Exception:
            metrics = {}
        if metrics:
            with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = _parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
