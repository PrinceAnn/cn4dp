"""Small shared utilities for downstream scripts.

These helpers exist because several scripts under `scripts/downstream/` were
written as standalone entrypoints and accumulated duplicated boilerplate.

Keep this module dependency-light and stable: it should be safe to import from
any downstream script.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from Delphi.model import Delphi, DelphiConfig  # type: ignore


def configure_logger(logger: logging.Logger, level: str = "INFO") -> None:
    """Configure a simple stream logger once (idempotent)."""
    if logger.handlers:
        return
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)


def seed_everything(seed: int, deterministic: bool = True, *, lightning_seed_fn=None) -> None:
    """Seed python/numpy/torch (optionally also Lightning if seed fn passed)."""
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if lightning_seed_fn is not None:
        try:
            lightning_seed_fn(seed_i, workers=True)
        except Exception:
            pass


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn (rank-based Mann–Whitney U)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="stable")
    sorted_scores = y_score[order]
    ranks = np.empty_like(order, dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(order)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    rank_sum_pos = float(ranks[pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def datetime_to_year_fraction(dt) -> float:
    """Convert a datetime-like scalar to a fractional year."""
    import pandas as pd

    ts = pd.to_datetime(dt, errors="coerce")
    if ts is pd.NaT or (hasattr(pd, "isna") and pd.isna(ts)):
        raise ValueError(f"Invalid datetime value: {dt!r}")

    doy = float(ts.dayofyear)
    return float(ts.year) + (doy - 1.0) / 365.25


def split_indices(n: int, seed: int, train_ratio: float, val_ratio: float):
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must leave nonempty training and test fractions")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def split_subject_indices(eids, *, seed, train_ratio, val_ratio, split_csv=None, eid_col="eid"):
    """Keep every occurrence of a subject in the same partition.

    For nested case-control data, assign subjects before sampling controls and
    pass the same assignment CSV (columns: eid, split) to every trainer.
    """
    eids = np.asarray(eids, dtype=np.int64)
    if split_csv:
        import pandas as pd

        table = pd.read_csv(split_csv)
        if not {eid_col, "split"}.issubset(table.columns) or table[eid_col].duplicated().any():
            raise ValueError("Split CSV must contain unique subject IDs and a split column")
        table[eid_col] = table[eid_col].astype(np.int64)
        assignment = table.set_index(eid_col)["split"].reindex(eids)
        if not assignment.isin(["train", "val", "test"]).all():
            raise ValueError("Every subject needs a train/val/test assignment")
        values = assignment.to_numpy()
        parts = tuple(np.flatnonzero(values == name) for name in ["train", "val", "test"])
    else:
        unique = np.unique(eids)
        groups = split_indices(len(unique), seed, train_ratio, val_ratio)
        parts = tuple(np.flatnonzero(np.isin(eids, unique[group])) for group in groups)
    if any(len(part) == 0 for part in parts):
        raise ValueError("All three subject partitions must be nonempty")
    return parts


def load_labels(labels_csv: Path) -> Dict[str, int]:
    """Return mapping: disease_code(str) -> token_index_in_labels(int)."""
    import pandas as pd

    df = pd.read_csv(labels_csv)
    if df.shape[1] != 1:
        raise ValueError(f"Expected 1 column in {labels_csv}, got {df.columns.tolist()}")
    col = df.columns[0]
    labels = df[col].astype(str).tolist()
    return {name: int(i) for i, name in enumerate(labels)}


def load_delphi_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[Delphi, DelphiConfig]:
    """Load `ckpt.pt` and return (model, config)."""
    ckpt_path = ckpt_dir / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    conf = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(conf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."  # torch.compile prefix
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, conf
