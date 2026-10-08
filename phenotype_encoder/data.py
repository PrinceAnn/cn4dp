from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class PreprocessArtifacts:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


def build_preprocess_pipeline(eps: float = 1e-6) -> "_Standardize":
    return _Standardize(eps=eps)


class _Standardize:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < self.eps, 1.0, std)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)

    def to_artifacts(self) -> PreprocessArtifacts:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Preprocess pipeline not fitted")
        return PreprocessArtifacts(mean_=self.mean_, std_=self.std_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Preprocess pipeline not fitted")
        # Replace NaN/Inf with the training-set mean so missing values become 0
        # after standardization.
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=self.mean_, posinf=self.mean_, neginf=self.mean_)
        return (x - self.mean_) / self.std_
def load_phenotype_table(
    csv_path: Path,
    *,
    eid_col: str = "eid",
) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    if eid_col not in df.columns:
        raise ValueError(f"eid_col={eid_col} not found in {csv_path}")

    feature_cols = [c for c in df.columns if c != eid_col]

    if not feature_cols:
        raise ValueError("No phenotype feature columns selected")

    return df, feature_cols


def feature_to_organ(feature_name: str, *, sep: str = "__", default: str = "global") -> str:
    """Infer organ name from a phenotype feature column.

    Expected input format: "{organ}{sep}{rest}" (e.g., "lung__feature_001").
    If `sep` is not present, returns `default`.
    """

    s = str(feature_name)
    if sep in s:
        organ = s.split(sep, 1)[0].strip()
        return organ if organ else default
    return default


def group_feature_columns_by_organ(
    feature_cols: List[str], *, sep: str = "__", default_organ: str = "global", sort_organs: bool = True
) -> Dict[str, List[str]]:
    """Group phenotype feature columns by organ prefix.

    Returns a dict mapping organ -> list of column names, preserving the original
    order within each organ.
    """

    groups: Dict[str, List[str]] = {}
    for c in feature_cols:
        organ = feature_to_organ(c, sep=sep, default=default_organ)
        groups.setdefault(organ, []).append(c)

    if sort_organs:
        # Stable order across runs/configs.
        return {k: groups[k] for k in sorted(groups.keys())}
    return groups
