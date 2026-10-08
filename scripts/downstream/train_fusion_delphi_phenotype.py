#!/usr/bin/env python
"""Fusion downstream model: IDP (organ_transformer) + Delphi trajectory (senet pooling).

This script trains a multi-task predictor on a matched riskset (case/control) by
*joining two modalities* per eid and concatenating their embeddings:

1) Delphi trajectory embedding: built from pre-imaging disease events and encoded
   by a pretrained Delphi transformer (frozen by default).
2) Phenotype/IDP features: standardized numeric features from a phenotype CSV.
   Optionally passed through a pretrained distillation encoder.

Outputs
-------
- Binary classification: whether the target disease occurs (label)
- Regression: time-to-onset (delta_time) for incident cases (masked loss)

Fusion types
------------
1) concat: concatenate pooled Delphi embedding + IDP embedding.
2) cross_attn: use IDP embedding as query and attend over Delphi token latents.

Run
---
python scripts/downstream/train_fusion_delphi_phenotype.py --config configs/demo/fusion_xattn.yaml
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

logger = logging.getLogger("downstream.fusion")

MASK_TIME = -10000.0


def _configure_logging(level: str = "INFO") -> None:
    configure_logger(logger, level=level)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion model: Delphi trajectory + phenotype")
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
    """Convert a datetime-like scalar to a fractional year.

    Accepts pandas Timestamp, python datetime/date, or numpy.datetime64.
    """
    return float(datetime_to_year_fraction(dt))


def _load_delphi_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[Delphi, DelphiConfig]:
    return load_delphi_checkpoint(ckpt_dir, device)


def _load_labels(labels_csv: Path) -> Dict[str, int]:
    return load_labels(labels_csv)


def _split_indices(n: int, seed: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split_indices(n, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)


def _load_pretrained_distill_ckpt(ckpt_path: Path) -> Tuple[dict, list[str], PreprocessArtifacts]:
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
        preprocess_artifacts = PreprocessArtifacts(
            mean_=np.asarray(preprocess_artifacts["mean_"], dtype=np.float32),
            std_=np.asarray(preprocess_artifacts["std_"], dtype=np.float32),
        )
    if not isinstance(preprocess_artifacts, PreprocessArtifacts):
        raise ValueError(f"Unexpected preprocess artifacts type in {ckpt_path}: {type(preprocess_artifacts)}")

    return ckpt, feature_names, preprocess_artifacts


class DelphiHiddenExtractor(nn.Module):
    """Wrap Delphi and expose final hidden states (after ln_f) via a permanent hook.

    Also supports pooling over token latents.
    """

    def __init__(
        self,
        delphi: Delphi,
        *,
        pooling: str = "senet",
        se_reduction: int = 16,
        se_dropout: float = 0.0,
    ):
        super().__init__()
        self.delphi = delphi
        self._last_hidden: Optional[torch.Tensor] = None

        self.pooling = str(pooling).lower()

        d = int(getattr(delphi.config, "n_embd"))
        r = max(1, int(se_reduction))
        hidden = max(1, d // r)

        self.token_gate: nn.Module | None
        if self.pooling == "senet":
            self.token_gate = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Dropout(float(se_dropout)) if se_dropout and float(se_dropout) > 0 else nn.Identity(),
                nn.Linear(hidden, 1),
            )
        elif self.pooling in {"mean", "avg", "average"}:
            self.token_gate = None
        else:
            raise ValueError(f"Unsupported pooling={pooling!r} (expected mean|senet)")

        def _hook(_module, _inputs, output):
            self._last_hidden = output

        self.delphi.transformer.ln_f.register_forward_hook(_hook)

    def _forward_delphi(self, x: torch.Tensor, a: torch.Tensor, *, dtype: torch.dtype, delphi_no_grad: bool) -> None:
        if delphi_no_grad:
            ctx = torch.no_grad()
        else:
            # Should not happen in this script; Delphi is expected frozen.
            ctx = torch.enable_grad()

        with ctx:
            if x.is_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    _ = self.delphi(x, a)
            else:
                _ = self.delphi(x, a)

    def encode_tokens(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        *,
        dtype: torch.dtype,
        delphi_no_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token latents and pad mask.

        Returns:
          hid_fp32: (B,T,D) float32
          mask_bool: (B,T) True for real tokens (x>0)
        """
        self._last_hidden = None
        self._forward_delphi(x, a, dtype=dtype, delphi_no_grad=delphi_no_grad)
        if self._last_hidden is None:
            raise RuntimeError("Failed to capture Delphi hidden state from ln_f")
        hid = self._last_hidden  # (B,T,D)
        hid_fp32 = hid.to(dtype=torch.float32)
        mask_bool = x > 0
        return hid_fp32, mask_bool

    def pool(self, hid_fp32: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        """Pool token latents to (B,D)."""
        if self.pooling in {"mean", "avg", "average"}:
            denom = mask_bool.sum(dim=1, keepdim=True).clamp(min=1)
            return (hid_fp32 * mask_bool.unsqueeze(-1)).sum(dim=1) / denom

        if self.token_gate is None:
            raise RuntimeError("token_gate is not initialized")
        scores = self.token_gate(hid_fp32).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(~mask_bool, -1e9)
        w = torch.softmax(scores, dim=1)  # (B,T)
        return (hid_fp32 * w.unsqueeze(-1)).sum(dim=1)

    def token_weights(self, hid_fp32: torch.Tensor, mask_bool: torch.Tensor) -> Optional[torch.Tensor]:
        """Return senet softmax weights (B,T) if pooling=senet, else None."""
        if self.pooling != "senet":
            return None
        if self.token_gate is None:
            raise RuntimeError("token_gate is not initialized")
        scores = self.token_gate(hid_fp32).squeeze(-1)
        scores = scores.masked_fill(~mask_bool, -1e9)
        return torch.softmax(scores, dim=1)


class CrossAttentionFusion(nn.Module):
    """Cross-attention: query=IDP embedding, key/value=Delphi token latents."""

    def __init__(
        self,
        *,
        q_dim: int,
        kv_dim: int,
        attn_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.q_proj = nn.Linear(int(q_dim), int(attn_dim))
        self.k_proj = nn.Linear(int(kv_dim), int(attn_dim))
        self.v_proj = nn.Linear(int(kv_dim), int(attn_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(attn_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        """Args:
        q: (B,Q)
        kv: (B,T,K)
        kv_mask: (B,T) True for real tokens
        """
        q_e = self.q_proj(q).unsqueeze(1)  # (B,1,A)
        k_e = self.k_proj(kv)
        v_e = self.v_proj(kv)
        key_padding_mask = ~kv_mask  # True for pad
        out, _ = self.attn(q_e, k_e, v_e, key_padding_mask=key_padding_mask, need_weights=False)
        return out.squeeze(1)  # (B,A)


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


@dataclass
class FusionSample:
    eid: torch.Tensor  # () int64
    x_feat: torch.Tensor  # (F,) float32
    x_tok: torch.Tensor  # (T,) int64 (shifted +1)
    x_age: torch.Tensor  # (T,) float32 ages in days
    y: torch.Tensor  # () float32
    delta_time: torch.Tensor  # () float32
    dt_mask: torch.Tensor  # () float32


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[FusionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FusionSample:
        return self.samples[idx]


def _collate_fusion_left_pad(batch: List[FusionSample]) -> Dict[str, torch.Tensor]:
    max_len = max(int(s.x_tok.numel()) for s in batch)
    B = len(batch)

    eid = torch.zeros((B,), dtype=torch.long)
    feat = torch.stack([s.x_feat for s in batch], dim=0)
    tok = torch.zeros((B, max_len), dtype=torch.long)
    age = torch.full((B, max_len), float(MASK_TIME), dtype=torch.float32)

    y = torch.zeros((B,), dtype=torch.float32)
    dt = torch.zeros((B,), dtype=torch.float32)
    dt_mask = torch.zeros((B,), dtype=torch.float32)

    for i, s in enumerate(batch):
        eid[i] = s.eid
        t = int(s.x_tok.numel())
        tok[i, -t:] = s.x_tok
        age[i, -t:] = s.x_age
        y[i] = s.y
        dt[i] = s.delta_time
        dt_mask[i] = s.dt_mask

    return {"eid": eid, "feat": feat, "tok": tok, "age": age, "y": y, "delta_time": dt, "dt_mask": dt_mask}


class FusionDataModule(pl.LightningDataModule):  # type: ignore[misc]
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
            collate_fn=_collate_fusion_left_pad,
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
            collate_fn=_collate_fusion_left_pad,
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
            collate_fn=_collate_fusion_left_pad,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
        )


class FusionModule(pl.LightningModule):  # type: ignore[misc]
    def __init__(
        self,
        *,
        delphi_encoder: DelphiHiddenExtractor,
        delphi_dim: int,
        pheno_encoder: nn.Module,
        pheno_dim: int,
        fusion: nn.Module,
        head: HeadMultiTask,
        cfg: dict,
        amp_dtype: torch.dtype,
        pos_weight: float | None,
        feature_cols: list[str],
        preprocess_artifacts: PreprocessArtifacts,
    ):
        super().__init__()
        self.delphi_encoder = delphi_encoder
        self.pheno_encoder = pheno_encoder
        self.fusion = fusion
        self.head = head
        self.cfg = cfg
        self.amp_dtype = amp_dtype

        self.feature_cols = feature_cols
        self.preprocess_artifacts = preprocess_artifacts

        self.freeze_delphi = bool(cfg.get("freeze_delphi", True))
        if not self.freeze_delphi:
            raise NotImplementedError("Finetuning Delphi is not supported in this fusion script (keep freeze_delphi=true).")

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

        self._delphi_dim = int(delphi_dim)
        self._pheno_dim = int(pheno_dim)

    def forward(self, feat: torch.Tensor, tok: torch.Tensor, age: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Delphi is expected frozen.
        hid_fp32, mask_bool = self.delphi_encoder.encode_tokens(tok, age, dtype=self.amp_dtype, delphi_no_grad=True)

        # Pooling always uses senet (or mean), as configured.
        z_traj = self.delphi_encoder.pool(hid_fp32, mask_bool)

        z_pheno = self.pheno_encoder(feat)

        fusion_type = str(self.cfg.get("fusion_type", "concat")).lower()
        if fusion_type == "concat":
            z = torch.cat([z_traj, z_pheno], dim=1)
        elif fusion_type in {"cross_attn", "xattn", "cross-attn"}:
            # Optionally weight Delphi tokens using senet gate before attention.
            w = self.delphi_encoder.token_weights(hid_fp32, mask_bool)
            kv = hid_fp32 if w is None else (hid_fp32 * w.unsqueeze(-1))
            ctx = self.fusion(z_pheno, kv, mask_bool)
            z = torch.cat([z_pheno, ctx], dim=1)
        else:
            raise ValueError(f"Unsupported fusion_type={fusion_type!r} (expected concat|cross_attn)")

        return self.head(z)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        eid = batch.get("eid", None)
        feat = batch["feat"]
        tok = batch["tok"]
        age = batch["age"]
        y = batch["y"]
        dt = batch["delta_time"]
        dt_mask = batch["dt_mask"]

        logit, dt_pred = self(feat, tok, age)

        loss_cls = self.criterion_cls(logit, y)
        per_ex = self.criterion_time(dt_pred, dt)
        denom = torch.clamp(dt_mask.sum(), min=1.0)
        loss_time = (per_ex * dt_mask).sum() / denom
        loss = self.w_cls * loss_cls + self.w_time * loss_time

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_time", loss_time, on_step=False, on_epoch=True)

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
        opt_name = str(self.cfg.get("optimizer", "adamw")).lower()
        lr = float(self.cfg.get("lr", 1e-3))
        weight_decay = float(self.cfg.get("weight_decay", 0.0))

        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters")

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = float(self.cfg.get("momentum", 0.9))
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer={opt_name}")

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
        checkpoint["config"] = self.cfg
        checkpoint["feature_cols"] = self.feature_cols
        checkpoint["preprocess"] = {
            "mean_": np.asarray(self.preprocess_artifacts.mean_, dtype=np.float32).tolist(),
            "std_": np.asarray(self.preprocess_artifacts.std_, dtype=np.float32).tolist(),
        }


def _build_delphi_sequences_for_eids(
    *,
    eids: np.ndarray,
    imaging_date: np.ndarray,
    label: np.ndarray,
    delta_time: np.ndarray,
    disease_onset_csv: Path,
    basic_csv: Path,
    labels_csv: Path,
    eid_col: str,
    imaging_date_col: str,
    block_size: int,
    append_query_token: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Build (tokens, ages_days) per sample in the same order as eids."""
    import pandas as pd

    basic = pd.read_csv(basic_csv)
    if eid_col not in basic.columns:
        raise ValueError(f"basic_csv must contain column {eid_col}")

    birth_year_col = None
    for cand in ["birth_year"]:
        if cand in basic.columns:
            birth_year_col = cand
            break
    if birth_year_col is None:
        raise ValueError("basic_csv must contain birth year column (expected 'birth_year')")

    basic[eid_col] = basic[eid_col].astype(int)
    basic["birth_year"] = pd.to_numeric(basic[birth_year_col], errors="coerce")
    basic = basic.set_index(eid_col)[["birth_year"]]

    birth_year = basic.reindex(eids)["birth_year"].to_numpy(dtype=np.float64)
    if np.isnan(birth_year).any():
        missing = int(np.isnan(birth_year).sum())
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

    disease_sub = disease.reindex(eids)

    img_year = np.array([_datetime_to_year_fraction(d) for d in imaging_date], dtype=np.float64)
    img_age_years = img_year - birth_year

    query_token_shifted = 1

    tok_list: List[torch.Tensor] = []
    age_list: List[torch.Tensor] = []

    dt_mask = np.isfinite(delta_time.astype(np.float64)) & (label.astype(np.int64) == 1)

    for i in range(len(eids)):
        img_age_y = float(img_age_years[i])
        if not math.isfinite(img_age_y) or img_age_y <= 0:
            raise ValueError(f"Invalid imaging_age_years for eid={int(eids[i])}: {img_age_y}")
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

        tok_list.append(torch.tensor(toks, dtype=torch.long))
        age_list.append(torch.tensor(ages, dtype=torch.float32))

    return tok_list, age_list, torch.tensor(dt_mask.astype(np.float32))


def train_from_config(config_path: Path) -> None:
    import pandas as pd

    cfg = _load_yaml(config_path)
    _configure_logging(str(cfg.get("log_level", "INFO")))

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    _seed_everything(seed, deterministic=deterministic)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    eid_col = str(cfg.get("eid_col", "eid"))
    label_col = str(cfg.get("label_col", "label"))
    delta_time_col = str(cfg.get("delta_time_col", "delta_time"))
    imaging_date_col = str(cfg.get("imaging_date_col", "imaging_date"))

    riskset_csv = Path(cfg["riskset_csv"])
    phenotype_csv = Path(cfg["phenotype_csv"])

    disease_onset_csv = Path(cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv"))
    basic_csv = Path(cfg.get("basic_csv", "data/demo/basic_info.csv"))
    labels_csv = Path(cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv"))

    delphi_ckpt_dir = Path(cfg.get("delphi_ckpt_dir", "runs/demo/teacher"))

    # Load riskset + phenotype and join.
    rs = pd.read_csv(riskset_csv)
    ph = pd.read_csv(phenotype_csv)

    need_cols = [eid_col, label_col, imaging_date_col]
    for c in need_cols:
        if c not in rs.columns:
            raise ValueError(f"riskset_csv must contain column {c}")
    if eid_col not in ph.columns:
        raise ValueError(f"phenotype_csv must contain column {eid_col}")

    rs2 = rs[[c for c in [eid_col, label_col, delta_time_col, imaging_date_col] if c in rs.columns]].copy()
    rs2[eid_col] = rs2[eid_col].astype(int)
    rs2[label_col] = rs2[label_col].astype(int)
    if delta_time_col in rs2.columns:
        rs2[delta_time_col] = pd.to_numeric(rs2[delta_time_col], errors="coerce")
    else:
        rs2[delta_time_col] = np.nan

    rs2[imaging_date_col] = pd.to_datetime(rs2[imaging_date_col], errors="coerce")
    if rs2[imaging_date_col].isna().any():
        raise ValueError(f"Found invalid {imaging_date_col} in riskset_csv")

    ph[eid_col] = ph[eid_col].astype(int)
    merged = rs2.merge(ph, on=eid_col, how="inner")
    if len(merged) == 0:
        raise ValueError("No samples after joining riskset and phenotype CSV on eid")

    eids = merged[eid_col].to_numpy(dtype=np.int64, copy=True)
    tr, va, te = split_subject_indices(
        eids, seed=seed,
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        split_csv=cfg.get("split_csv"), eid_col=eid_col,
    )

    # Phenotype preprocessing / encoder.
    pretrained_cfg = cfg.get("pretrained", {}) or {}
    ckpt_path = pretrained_cfg.get("ckpt_path")
    use_pretrained = bool(ckpt_path)

    if use_pretrained:
        ckpt_path = Path(ckpt_path)
        distill_ckpt, feature_cols, preprocess_artifacts = _load_pretrained_distill_ckpt(ckpt_path)
        missing = [c for c in feature_cols if c not in merged.columns]
        if missing:
            raise ValueError(
                "phenotype_csv is missing features required by pretrained checkpoint. "
                f"Missing {len(missing)} columns, e.g. {missing[:10]}"
            )
        X_raw = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)
        preprocess = build_preprocess_pipeline()
        preprocess.mean_ = preprocess_artifacts.mean_.astype(np.float32)
        preprocess.std_ = preprocess_artifacts.std_.astype(np.float32)

        distill_cfg = (distill_ckpt.get("config", {}) or {})
        distill_encoder_cfg = (distill_cfg.get("encoder", {}) or {})

        # Build encoder exactly as distill used.
        pheno_encoder, out_dim = build_encoder_from_config(encoder_cfg=distill_encoder_cfg, feature_cols=feature_cols)
        pheno_encoder.load_state_dict(distill_ckpt["model_state"], strict=True)

        freeze = bool(pretrained_cfg.get("freeze_encoder", True))
        if freeze:
            for p in pheno_encoder.parameters():
                p.requires_grad_(False)
        pheno_dim = int(out_dim)
        logger.info("Using pretrained phenotype encoder from %s (freeze=%s)", str(ckpt_path), str(freeze))
    else:
        # IMPORTANT: exclude target-derived columns from IDP features.
        exclude = {eid_col, label_col, imaging_date_col}
        if delta_time_col in merged.columns:
            exclude.add(delta_time_col)
        feature_cols = [c for c in merged.columns if c not in exclude]
        if not feature_cols:
            raise ValueError("No phenotype feature columns found after merge")
        X_raw = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)
        preprocess = build_preprocess_pipeline()
        preprocess.fit(X_raw[tr])

        enc_cfg = cfg.get("encoder", None)
        if enc_cfg is None:
            raise ValueError("Missing required 'encoder' config for IDP model (expected organ_transformer).")
        pheno_encoder, out_dim = build_encoder_from_config(encoder_cfg=enc_cfg, feature_cols=feature_cols)
        pheno_dim = int(out_dim)

    # Extra safety: never allow targets in feature set.
    if label_col in feature_cols:
        raise RuntimeError(f"Leakage detected: label_col={label_col!r} is in feature_cols")
    if delta_time_col in feature_cols:
        raise RuntimeError(f"Leakage detected: delta_time_col={delta_time_col!r} is in feature_cols")

    X = preprocess.transform(X_raw)
    if bool(cfg.get("check_finite", True)) and (not np.isfinite(X).all()):
        raise ValueError("Found NaN/Inf in standardized phenotype features")

    # Build Delphi trajectories for same sample order.
    eids = merged[eid_col].to_numpy(dtype=np.int64, copy=True)
    y = merged[label_col].to_numpy(dtype=np.int64, copy=True)
    dt = merged[delta_time_col].to_numpy(dtype=np.float32, copy=True)
    imaging_date = merged[imaging_date_col].to_numpy(copy=True)

    block_size = int(cfg.get("block_size", 256))
    append_query_token = bool(cfg.get("append_query_token", True))

    logger.info("Building masked Delphi trajectories for fused samples...")
    tok_list, age_list, dt_mask = _build_delphi_sequences_for_eids(
        eids=eids,
        imaging_date=imaging_date,
        label=y,
        delta_time=dt,
        disease_onset_csv=disease_onset_csv,
        basic_csv=basic_csv,
        labels_csv=labels_csv,
        eid_col=eid_col,
        imaging_date_col=imaging_date_col,
        block_size=block_size,
        append_query_token=append_query_token,
    )

    # Split.


    # Package samples.
    dt_filled = np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def make_samples(indices: np.ndarray) -> List[FusionSample]:
        out: List[FusionSample] = []
        for i in indices.tolist():
            out.append(
                FusionSample(
                    eid=torch.tensor(int(eids[i]), dtype=torch.long),
                    x_feat=torch.from_numpy(X[i].astype(np.float32)),
                    x_tok=tok_list[i],
                    x_age=age_list[i],
                    y=torch.tensor(float(y[i]), dtype=torch.float32),
                    delta_time=torch.tensor(float(dt_filled[i]), dtype=torch.float32),
                    dt_mask=torch.tensor(float(dt_mask[i].item()), dtype=torch.float32),
                )
            )
        return out

    ds_train = FusionDataset(make_samples(tr))
    ds_val = FusionDataset(make_samples(va)) if len(va) else None
    ds_test = FusionDataset(make_samples(te)) if len(te) else None

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    device_str = str(cfg.get("device", "cpu"))
    default_pin = bool(device_str.startswith("cuda") and torch.cuda.is_available())
    pin_memory = bool(cfg.get("pin_memory", default_pin))

    dm = FusionDataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    # Delphi encoder.
    dtype_str = str(cfg.get("dtype", "float32")).lower()
    amp_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

    logger.info("Loading Delphi checkpoint from %s", str(delphi_ckpt_dir))
    delphi, delphi_conf = _load_delphi_checkpoint(delphi_ckpt_dir, device=torch.device("cpu"))

    freeze_delphi = bool(cfg.get("freeze_delphi", True))
    if freeze_delphi:
        for p in delphi.parameters():
            p.requires_grad_(False)
        delphi.eval()

    delphi_pooling = str(cfg.get("delphi_pooling", cfg.get("traj_pooling", "senet"))).lower()
    delphi_encoder = DelphiHiddenExtractor(
        delphi,
        pooling=delphi_pooling,
        se_reduction=int(cfg.get("se_reduction", 16)),
        se_dropout=float(cfg.get("se_dropout", 0.0)),
    )
    delphi_dim = int(delphi_conf.n_embd)

    # Compute pos_weight on training split.
    pos_weight = cfg.get("pos_weight", None)
    if pos_weight is None:
        n_pos = float((y[tr] == 1).sum())
        n_neg = float((y[tr] == 0).sum())
        pos_weight = n_neg / max(1.0, n_pos) if n_pos > 0 else None
    else:
        pos_weight = float(pos_weight)

    # Fusion modules + head.
    fusion_type = str(cfg.get("fusion_type", "concat")).lower()
    head_dropout = float(cfg.get("head_dropout", 0.0))

    if fusion_type == "concat":
        fusion = nn.Identity()
        fusion_dim = delphi_dim + int(pheno_dim)
    elif fusion_type in {"cross_attn", "xattn", "cross-attn"}:
        attn_dim = int(cfg.get("xattn_dim", pheno_dim))
        num_heads = int(cfg.get("xattn_heads", 4))
        attn_drop = float(cfg.get("xattn_dropout", 0.0))
        fusion = CrossAttentionFusion(q_dim=int(pheno_dim), kv_dim=int(delphi_dim), attn_dim=attn_dim, num_heads=num_heads, dropout=attn_drop)
        fusion_dim = int(pheno_dim) + int(attn_dim)
    else:
        raise ValueError(f"Unsupported fusion_type={fusion_type!r} (expected concat|cross_attn)")

    head = HeadMultiTask(in_dim=int(fusion_dim), dropout=head_dropout)

    preprocess_artifacts = preprocess.to_artifacts()

    module = FusionModule(
        delphi_encoder=delphi_encoder,
        delphi_dim=delphi_dim,
        pheno_encoder=pheno_encoder,
        pheno_dim=int(pheno_dim),
        fusion=fusion,
        head=head,
        cfg=cfg,
        amp_dtype=amp_dtype,
        pos_weight=pos_weight,
        feature_cols=feature_cols,
        preprocess_artifacts=preprocess_artifacts,
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

    accelerator = "gpu" if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu"
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

    if ds_test is not None:
        best_path = getattr(ckpt_cb, "best_model_path", None) or "best"
        try:
            trainer.test(datamodule=dm, ckpt_path=best_path, weights_only=False)
        except TypeError:
            ckpt_path = str(best_path) if best_path != "best" else str(getattr(ckpt_cb, "best_model_path", ""))
            if not ckpt_path:
                raise
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            module.load_state_dict(state_dict, strict=False)
            trainer.test(model=module, datamodule=dm, ckpt_path=None)

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = _parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
