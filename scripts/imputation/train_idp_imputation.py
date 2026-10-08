#!/usr/bin/env python
"""训练 IDP 缺失值补全模型，并可选地引入冻结的 Delphi 轨迹先验。

当前实现对应这个新任务的第一版可运行基线：

1) 从 cohort 表和 phenotype 表中构建有影像时间的受试者集合。
2) 截断影像时间之后的疾病事件，并用冻结的 Delphi 编码影像前轨迹。
3) 对已观测到的 IDP 特征做随机遮挡，或整器官 block 遮挡。
4) 用剩余可见 IDP 和可选的轨迹表征去重建被遮挡的目标值。

默认的条件融合方式是简单 concat；
如果想更贴近“轨迹提供缺失先验”的设定，也支持把轨迹分支当成一个 gated residual prior，
只在 decoder 端以残差方式补充疾病轨迹带来的预测偏置。

进一步地，也支持 organ-aware residual：
轨迹先验不再同时作用于全部 IDP，而是按器官分头地产生 residual 和 gate，
让疾病轨迹只去修正更相关的器官块。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Delphi.model import Delphi  # type: ignore
from phenotype_encoder.data import PreprocessArtifacts, build_preprocess_pipeline, group_feature_columns_by_organ
from phenotype_encoder.model import build_encoder_from_config
from scripts.downstream.utils import configure_logger, datetime_to_year_fraction, load_delphi_checkpoint, load_labels, seed_everything, split_indices


logger = logging.getLogger("imputation.train")
MASK_TIME = -10000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IDP imputation with optional Delphi trajectory priors")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_allowed_eids(path: Path, eid_col: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    series = df[eid_col] if eid_col in df.columns else df.iloc[:, 0] if df.shape[1] == 1 else None
    if series is None:
        raise ValueError(f"allowed_eids_csv must contain column {eid_col!r} or a single eid column")
    return pd.to_numeric(series, errors="raise").astype(np.int64).to_numpy(copy=True)


def _load_pretrained_distill_ckpt(ckpt_path: Path) -> Tuple[dict, list[str], PreprocessArtifacts]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type at {ckpt_path}: {type(ckpt)}")

    for key in ["model_state", "feature_names", "preprocess"]:
        if key not in ckpt:
            raise ValueError(
                f"Checkpoint {ckpt_path} missing key {key!r}. "
                "Make sure it is produced by scripts/distill/train_align.py"
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


def _configure_logging(level: str = "INFO") -> None:
    configure_logger(logger, level=level)


def _attach_file_logger(log_path: Path, level: str = "INFO") -> None:
    # 额外把训练日志写入文件，便于回看每次实验输出。
    resolved = log_path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved:
            return

    file_handler = logging.FileHandler(resolved, encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)


def _device_from_cfg(cfg: dict) -> torch.device:
    device_name = str(cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    return torch.device(device_name)


def _feature_identity(column: str) -> Tuple[str, str]:
    column_str = str(column)
    if "__" in column_str:
        organ_name, raw_name = column_str.split("__", 1)
    else:
        organ_name, raw_name = "global", column_str
    field_id = raw_name.split("-", 1)[0]
    return organ_name, field_id


def _align_feature_columns_for_pretrained(
    *,
    current_feature_cols: Sequence[str],
    pretrained_feature_names: Sequence[str],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    if len(current_feature_cols) != len(pretrained_feature_names):
        raise ValueError(
            "Current phenotype columns do not match pretrained encoder width: "
            f"got {len(current_feature_cols)} vs pretrained {len(pretrained_feature_names)}"
        )

    current_cols = [str(column) for column in current_feature_cols]
    current_set = set(current_cols)
    current_by_identity: Dict[Tuple[str, str], List[str]] = {}
    for column in current_cols:
        current_by_identity.setdefault(_feature_identity(column), []).append(column)

    aligned_columns: List[str] = []
    fallback_pairs: List[Tuple[str, str]] = []
    used_current: set[str] = set()

    for pretrained_name in pretrained_feature_names:
        pretrained_name = str(pretrained_name)
        if pretrained_name in current_set and pretrained_name not in used_current:
            chosen = pretrained_name
        else:
            identity = _feature_identity(pretrained_name)
            candidates = [column for column in current_by_identity.get(identity, []) if column not in used_current]
            if len(candidates) != 1:
                raise ValueError(
                    "Failed to align pretrained feature to current phenotype table: "
                    f"pretrained='{pretrained_name}', candidates={candidates}"
                )
            chosen = candidates[0]
            fallback_pairs.append((pretrained_name, chosen))

        aligned_columns.append(chosen)
        used_current.add(chosen)

    if len(used_current) != len(current_cols):
        unused = [column for column in current_cols if column not in used_current]
        raise ValueError(
            "Current phenotype table contains columns that could not be aligned to the pretrained encoder: "
            f"{unused[:10]}"
        )

    return aligned_columns, fallback_pairs


def _build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, *, dropout: float, act: str) -> nn.Sequential:
    act_name = str(act).lower()
    if act_name == "relu":
        act_fn = nn.ReLU
    elif act_name == "gelu":
        act_fn = nn.GELU
    elif act_name in {"silu", "swish"}:
        act_fn = nn.SiLU
    else:
        raise ValueError(f"Unsupported activation: {act}")

    layers: List[nn.Module] = []
    prev_dim = int(in_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(act_fn())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(out_dim)))
    return nn.Sequential(*layers)


class SequencePool(nn.Module):
    """把 Delphi 的 token 序列表征池化成单个受试者级向量。"""

    def __init__(self, *, mode: str, dim: int, se_reduction: int = 16, se_dropout: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.mode = str(mode).lower()
        self.temperature = float(temperature)
        self.token_gate: nn.Module | None = None

        if self.mode in {"senet", "se"}:
            reduction = max(1, int(se_reduction))
            hidden_dim = max(1, int(dim) // reduction)
            self.token_gate = nn.Sequential(
                nn.Linear(int(dim), hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(se_dropout)) if se_dropout and se_dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, sequence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.mode in {"sum", "add"}:
            return (sequence * valid_mask.unsqueeze(-1).to(sequence.dtype)).sum(dim=1)

        if self.mode in {"mean", "avg", "average"}:
            denom = torch.clamp(valid_mask.sum(dim=1, keepdim=True).to(sequence.dtype), min=1.0)
            return (sequence * valid_mask.unsqueeze(-1).to(sequence.dtype)).sum(dim=1) / denom

        if self.mode == "last":
            lengths = valid_mask.long().sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            return sequence[torch.arange(sequence.shape[0], device=sequence.device), lengths]

        if self.mode in {"senet", "se"}:
            if self.token_gate is None:
                raise RuntimeError("token_gate is not initialized")
            scores = self.token_gate(sequence).squeeze(-1)
            scores = scores.masked_fill(~valid_mask, -1e9)
            temp = self.temperature if self.temperature else 1.0
            weights = torch.softmax(scores / max(1e-6, float(temp)), dim=1)
            return (sequence * weights.unsqueeze(-1)).sum(dim=1)

        raise ValueError(f"Unsupported teacher_pooling mode: {self.mode}")


class FrozenTrajectoryEncoder(nn.Module):
    """取出 Delphi 最后一层隐藏状态，并通过轻量池化头得到轨迹向量。"""

    def __init__(
        self,
        delphi: Delphi,
        *,
        pooling: str,
        freeze_delphi: bool,
        se_reduction: int,
        se_dropout: float,
        temperature: float,
    ):
        super().__init__()
        self.delphi = delphi
        self.freeze_delphi = bool(freeze_delphi)
        self.pooler = SequencePool(
            mode=pooling,
            dim=int(delphi.config.n_embd),
            se_reduction=se_reduction,
            se_dropout=se_dropout,
            temperature=temperature,
        )
        self._last_hidden: Optional[torch.Tensor] = None

        if self.freeze_delphi:
            for parameter in self.delphi.parameters():
                parameter.requires_grad_(False)
            self.delphi.eval()

        def _hook(_module, _inputs, output):
            # 从 Delphi 最后一层归一化之后截取 token 表征，作为后续池化输入。
            self._last_hidden = output

        self.delphi.transformer.ln_f.register_forward_hook(_hook)

    @property
    def output_dim(self) -> int:
        return int(self.delphi.config.n_embd)

    def encode_tokens(self, tokens: torch.Tensor, ages_days: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._last_hidden = None
        if self.freeze_delphi:
            with torch.no_grad():
                _ = self.delphi(tokens, ages_days)
        else:
            _ = self.delphi(tokens, ages_days)

        if self._last_hidden is None:
            raise RuntimeError("Failed to capture Delphi hidden states from ln_f")

        hidden = self._last_hidden.to(dtype=torch.float32)
        valid_mask = tokens > 0
        return hidden, valid_mask

    def forward(self, tokens: torch.Tensor, ages_days: torch.Tensor) -> torch.Tensor:
        hidden, valid_mask = self.encode_tokens(tokens, ages_days)
        return self.pooler(hidden, valid_mask)


class TrajectoryConditionedImputer(nn.Module):
    """根据可见 IDP 和可选轨迹先验，重建被遮挡的 IDP。"""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_dim: int,
        feature_dim: int,
        use_trajectory: bool,
        trajectory_dim: int,
        trajectory_fusion: str,
        organ_to_indices: Optional[Dict[str, np.ndarray]],
        decoder_hidden_dims: Sequence[int],
        dropout: float,
        act: str,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_trajectory = bool(use_trajectory)
        self.encoder_dim = int(encoder_dim)
        self.feature_dim = int(feature_dim)
        self.trajectory_dim = int(trajectory_dim)
        self.trajectory_fusion = str(trajectory_fusion).lower()
        self.mask_token = nn.Parameter(torch.zeros(int(feature_dim)))
        self.organ_slices: List[Tuple[str, List[int]]] = []
        self.organ_context_dim = 32
        self.residual_policy = "normal"

        if not self.use_trajectory:
            self.trajectory_fusion = "none"

        if self.trajectory_fusion in {"none", "concat"}:
            decoder_input_dim = self.encoder_dim + (self.trajectory_dim if self.use_trajectory else 0)
            self.decoder = _build_mlp(decoder_input_dim, decoder_hidden_dims, self.feature_dim, dropout=dropout, act=act)
            self.idp_decoder = None
            self.trajectory_decoder = None
            self.trajectory_gate = None
            self.organ_context_encoders = None
            self.organ_trajectory_decoders = None
            self.organ_trajectory_gates = None
        elif self.trajectory_fusion in {"residual", "gated_residual", "residual_prior"}:
            if not self.use_trajectory:
                raise ValueError("trajectory_fusion=residual requires use_trajectory=true")
            self.decoder = None
            self.idp_decoder = _build_mlp(self.encoder_dim, decoder_hidden_dims, self.feature_dim, dropout=dropout, act=act)
            self.trajectory_decoder = _build_mlp(self.trajectory_dim, decoder_hidden_dims, self.feature_dim, dropout=dropout, act=act)
            self.trajectory_gate = nn.Linear(self.encoder_dim + self.trajectory_dim, self.feature_dim)
            self.organ_context_encoders = None
            self.organ_trajectory_decoders = None
            self.organ_trajectory_gates = None
        elif self.trajectory_fusion in {"organ_residual", "organ_gated_residual", "organ_residual_prior"}:
            if not self.use_trajectory:
                raise ValueError("trajectory_fusion=organ_residual requires use_trajectory=true")
            if organ_to_indices is None or not organ_to_indices:
                raise ValueError("trajectory_fusion=organ_residual requires non-empty organ_to_indices")
            self.decoder = None
            self.idp_decoder = _build_mlp(self.encoder_dim, decoder_hidden_dims, self.feature_dim, dropout=dropout, act=act)
            self.trajectory_decoder = None
            self.trajectory_gate = None
            self.organ_context_encoders = nn.ModuleDict()
            self.organ_trajectory_decoders = nn.ModuleDict()
            self.organ_trajectory_gates = nn.ModuleDict()

            ordered_organs = sorted(
                ((organ_name, indices) for organ_name, indices in organ_to_indices.items() if len(indices) > 0),
                key=lambda item: int(np.min(item[1])),
            )
            for organ_idx, (organ_name, indices) in enumerate(ordered_organs):
                organ_key = f"organ_{organ_idx}"
                organ_indices = [int(index) for index in np.asarray(indices, dtype=np.int64).tolist()]
                self.organ_slices.append((organ_key, organ_indices))
                organ_dim = len(organ_indices)
                organ_context_dim = min(64, max(8, organ_dim))
                self.organ_context_encoders[organ_key] = _build_mlp(
                    organ_dim,
                    [],
                    organ_context_dim,
                    dropout=0.0,
                    act=act,
                )
                self.organ_trajectory_decoders[organ_key] = _build_mlp(
                    self.trajectory_dim,
                    decoder_hidden_dims,
                    organ_dim,
                    dropout=dropout,
                    act=act,
                )
                self.organ_trajectory_gates[organ_key] = nn.Linear(organ_context_dim + self.trajectory_dim, organ_dim)
        else:
            raise ValueError(f"Unsupported trajectory_fusion={trajectory_fusion}")

    def build_encoder_input(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        # 被遮挡/原本缺失的位置不用真实值，而是换成可学习的 mask token，
        # 这样编码器能区分“数值本来就是 0”和“这个位置当前不可见”。
        x_input = x.masked_fill(missing_mask, 0.0)
        x_input = x_input + missing_mask.to(x.dtype) * self.mask_token.unsqueeze(0)
        return x_input

    def encode_inputs(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.build_encoder_input(x, missing_mask))

    def decode(
        self,
        z_idp: torch.Tensor,
        trajectory: torch.Tensor | None = None,
        *,
        encoder_input: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.trajectory_fusion == "none":
            if self.decoder is None:
                raise RuntimeError("decoder is not initialized")
            return self.decoder(z_idp)

        if self.trajectory_fusion == "concat":
            if trajectory is None:
                raise ValueError("trajectory features are required when use_trajectory=true")
            fused = torch.cat([z_idp, trajectory], dim=1)
            if self.decoder is None:
                raise RuntimeError("decoder is not initialized")
            return self.decoder(fused)

        if self.trajectory_fusion in {"residual", "gated_residual", "residual_prior"}:
            if trajectory is None:
                raise ValueError("trajectory features are required when trajectory_fusion uses residual priors")
            if self.idp_decoder is None or self.trajectory_decoder is None or self.trajectory_gate is None:
                raise RuntimeError("Residual trajectory fusion modules are not initialized")
            base_pred = self.idp_decoder(z_idp)
            traj_pred = self.trajectory_decoder(trajectory)
            gate = torch.sigmoid(self.trajectory_gate(torch.cat([z_idp, trajectory], dim=1)))
            return base_pred + gate * traj_pred

        if self.trajectory_fusion in {"organ_residual", "organ_gated_residual", "organ_residual_prior"}:
            if trajectory is None:
                raise ValueError("trajectory features are required when trajectory_fusion uses organ residual priors")
            if encoder_input is None:
                raise ValueError("encoder_input is required when trajectory_fusion uses organ residual priors")
            if (
                self.idp_decoder is None
                or self.organ_context_encoders is None
                or self.organ_trajectory_decoders is None
                or self.organ_trajectory_gates is None
            ):
                raise RuntimeError("Organ-aware residual trajectory fusion modules are not initialized")

            base_pred = self.idp_decoder(z_idp)
            residual = torch.zeros_like(base_pred)
            # 每个 organ 都有各自的轨迹残差头和 gate，
            # gate 只看该 organ 的可见 IDP 子表示，而不是全局 z_idp，
            # 避免局部器官先验被全局表征稀释。
            for organ_key, organ_indices in self.organ_slices:
                organ_input = encoder_input[:, organ_indices]
                organ_context = self.organ_context_encoders[organ_key](organ_input)
                organ_pred = self.organ_trajectory_decoders[organ_key](trajectory)
                organ_gate = torch.sigmoid(
                    self.organ_trajectory_gates[organ_key](torch.cat([organ_context, trajectory], dim=1))
                )
                if self.residual_policy == "disable_full_organ_mask":
                    if missing_mask is None:
                        raise ValueError("missing_mask is required by disable_full_organ_mask")
                    fully_missing = missing_mask[:, organ_indices].all(dim=1, keepdim=True)
                    organ_gate = organ_gate.masked_fill(fully_missing, 0.0)
                residual[:, organ_indices] = organ_gate * organ_pred
            return base_pred + residual

        raise ValueError(f"Unsupported trajectory_fusion={self.trajectory_fusion}")

    def forward(
        self,
        x: torch.Tensor,
        missing_mask: torch.Tensor,
        trajectory: torch.Tensor | None = None,
        *,
        return_latent: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        encoder_input = self.build_encoder_input(x, missing_mask)
        z_idp = self.encoder(encoder_input)
        pred = self.decode(z_idp, trajectory, encoder_input=encoder_input, missing_mask=missing_mask)
        if return_latent:
            return pred, z_idp
        return pred


@dataclass
class ImputationSample:
    eid: torch.Tensor
    x: torch.Tensor
    observed_mask: torch.Tensor
    loss_mask: torch.Tensor
    tokens: torch.Tensor
    ages_days: torch.Tensor


class ImputationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[ImputationSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ImputationSample:
        return self.samples[index]


def _collate_left_pad(batch: Sequence[ImputationSample]) -> Dict[str, torch.Tensor]:
    max_len = max(int(item.tokens.numel()) for item in batch)
    batch_size = len(batch)

    eid = torch.zeros((batch_size,), dtype=torch.long)
    x = torch.stack([item.x for item in batch], dim=0)
    observed_mask = torch.stack([item.observed_mask for item in batch], dim=0)
    loss_mask = torch.stack([item.loss_mask for item in batch], dim=0)
    tokens = torch.zeros((batch_size, max_len), dtype=torch.long)
    ages_days = torch.full((batch_size, max_len), float(MASK_TIME), dtype=torch.float32)

    for row_index, item in enumerate(batch):
        eid[row_index] = item.eid
        seq_len = int(item.tokens.numel())
        # 左侧 padding，让每个人最近的事件都右对齐，便于统一送入 Delphi。
        tokens[row_index, -seq_len:] = item.tokens
        ages_days[row_index, -seq_len:] = item.ages_days

    return {
        "eid": eid,
        "x": x,
        "observed_mask": observed_mask,
        "loss_mask": loss_mask,
        "tokens": tokens,
        "ages_days": ages_days,
    }


def _build_pre_imaging_sequences(
    *,
    eids: np.ndarray,
    imaging_date: np.ndarray,
    disease_onset_csv: Path,
    basic_csv: Path,
    labels_csv: Path,
    eid_col: str,
    block_size: int,
    append_query_token: bool,
    exclude_codes: Optional[set[str]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # 把每个受试者在影像时间之前的疾病轨迹整理成 Delphi 可用的 token 序列。
    import pandas as pd

    basic_df = pd.read_csv(basic_csv)
    if eid_col not in basic_df.columns:
        raise ValueError(f"basic_csv must contain column {eid_col}")

    birth_year_col = None
    for candidate in ["birth_year"]:
        if candidate in basic_df.columns:
            birth_year_col = candidate
            break
    if birth_year_col is None:
        raise ValueError("basic_csv must contain birth year column (expected 'birth_year')")

    basic_df[eid_col] = basic_df[eid_col].astype(int)
    basic_df["birth_year"] = pd.to_numeric(basic_df[birth_year_col], errors="coerce")
    basic_df = basic_df.set_index(eid_col)[["birth_year"]]

    birth_year = basic_df.reindex(eids)["birth_year"].to_numpy(dtype=np.float64)
    if np.isnan(birth_year).any():
        missing_count = int(np.isnan(birth_year).sum())
        raise ValueError(f"Missing birth_year for {missing_count} eids after merge with basic_csv")

    labels_map = load_labels(labels_csv)

    disease_df = pd.read_csv(disease_onset_csv)
    if eid_col not in disease_df.columns:
        raise ValueError(f"disease_onset_csv must contain column {eid_col}")
    disease_df[eid_col] = disease_df[eid_col].astype(int)
    disease_df = disease_df.set_index(eid_col)

    excluded = {str(code).upper() for code in (exclude_codes or set())}
    candidate_cols = [column for column in disease_df.columns if column in labels_map and str(column).upper() not in excluded]
    if not candidate_cols:
        raise ValueError("No disease onset columns overlap with Delphi labels vocabulary")

    disease_sub = disease_df.reindex(eids)
    imaging_year = np.array([datetime_to_year_fraction(value) for value in imaging_date], dtype=np.float64)
    imaging_age_years = imaging_year - birth_year

    query_token_shifted = 1
    token_list: List[torch.Tensor] = []
    age_list: List[torch.Tensor] = []

    for row_index, eid in enumerate(eids.tolist()):
        imaging_age = float(imaging_age_years[row_index])
        if not math.isfinite(imaging_age) or imaging_age <= 0:
            raise ValueError(f"Invalid imaging age for eid={eid}: {imaging_age}")
        imaging_age_days = imaging_age * 365.25

        disease_row = disease_sub.iloc[row_index]
        ages: List[float] = []
        tokens: List[int] = []

        for disease_code in candidate_cols:
            onset_value = disease_row[disease_code]
            if onset_value is None or (isinstance(onset_value, float) and not math.isfinite(onset_value)):
                continue
            try:
                onset_age = float(onset_value)
            except Exception:
                continue
            if not math.isfinite(onset_age) or onset_age > imaging_age:
                continue

            # Delphi 中 0 留给 padding，所以真实疾病 token 要整体加 1。
            token_index = int(labels_map[disease_code]) + 1
            if token_index <= 0:
                continue
            ages.append(onset_age * 365.25)
            tokens.append(token_index)

        if ages:
            order = sorted(range(len(ages)), key=lambda idx: (ages[idx], tokens[idx]))
            ages = [ages[idx] for idx in order]
            tokens = [tokens[idx] for idx in order]

        if append_query_token:
            # 在序列末尾追加“影像时点”这个查询位置，
            # 让轨迹表示显式对应扫描前可见的病史。
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

    return token_list, age_list


def _sample_loss_mask(
    *,
    observed_mask: np.ndarray,
    mode: str,
    ratio: float,
    organ_to_indices: Dict[str, np.ndarray],
    rng: np.random.Generator,
    fixed_organ: str | None,
) -> np.ndarray:
    mask = np.zeros_like(observed_mask, dtype=bool)
    observed_indices = np.where(observed_mask)[0]
    if observed_indices.size == 0:
        return mask

    if mode == "random":
        if observed_indices.size <= 1:
            return mask
        # 至少保留一个可见特征，避免模型在该样本上完全失去重建上下文。
        n_mask = max(1, int(round(float(ratio) * float(observed_indices.size))))
        n_mask = min(n_mask, observed_indices.size - 1)
        chosen = rng.choice(observed_indices, size=n_mask, replace=False)
        mask[chosen] = True
        return mask

    if mode == "organ_block":
        if fixed_organ is not None:
            candidate_organs: Iterable[str] = [fixed_organ]
        else:
            candidate_organs = organ_to_indices.keys()

        valid_organs: List[str] = []
        for organ_name in candidate_organs:
            organ_indices = organ_to_indices[organ_name]
            organ_observed = organ_indices[observed_mask[organ_indices]]
            outside_observed = observed_indices[~np.isin(observed_indices, organ_indices)]
            # 整器官遮挡时，也必须保证器官外仍有可见特征，
            # 否则这个样本没有足够条件信息可用来重建。
            if organ_observed.size > 0 and outside_observed.size > 0:
                valid_organs.append(organ_name)

        if not valid_organs:
            return mask

        chosen_organ = str(rng.choice(np.asarray(valid_organs, dtype=object)))
        organ_indices = organ_to_indices[chosen_organ]
        mask[organ_indices[observed_mask[organ_indices]]] = True
        return mask

    raise ValueError(f"Unsupported masking mode: {mode}")


def _masked_loss(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor, *, loss_name: str) -> torch.Tensor:
    if loss_mask.sum() <= 0:
        return pred.new_tensor(0.0)

    if loss_name == "mae":
        per_value = torch.abs(pred - target)
    elif loss_name == "mse":
        per_value = (pred - target) ** 2
    elif loss_name in {"smoothl1", "huber"}:
        per_value = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_name={loss_name}")

    return per_value[loss_mask].mean()


def _prior_alignment_loss(current_repr: torch.Tensor, prior_repr: torch.Tensor, *, loss_name: str) -> torch.Tensor:
    loss_name = str(loss_name).lower()
    if current_repr.shape != prior_repr.shape:
        raise ValueError(
            "Current encoder representation and prior representation must share the same shape: "
            f"got {tuple(current_repr.shape)} vs {tuple(prior_repr.shape)}"
        )

    if loss_name in {"cosine", "cos"}:
        return 1.0 - nn.functional.cosine_similarity(current_repr, prior_repr.detach(), dim=-1).mean()
    if loss_name == "mse":
        return nn.functional.mse_loss(current_repr, prior_repr.detach())
    if loss_name in {"smoothl1", "huber"}:
        return nn.functional.smooth_l1_loss(current_repr, prior_repr.detach())

    raise ValueError(f"Unsupported prior alignment loss: {loss_name}")


def _flatten_masked_values(pred_std: np.ndarray, target_std: np.ndarray, loss_mask: np.ndarray, preprocess: PreprocessArtifacts) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(preprocess.mean_, dtype=np.float32)
    std = np.asarray(preprocess.std_, dtype=np.float32)
    # 指标回到原始 IDP 尺度上更直观，避免只看标准化后的误差。
    pred_raw = pred_std * std[None, :] + mean[None, :]
    target_raw = target_std * std[None, :] + mean[None, :]
    mask_bool = loss_mask.astype(bool)
    return pred_raw[mask_bool], target_raw[mask_bool]


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if pred.size == 0 or target.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "r2": float("nan")}

    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    if pred.size > 1 and pred_std > 0.0 and target_std > 0.0:
        pearson = float(np.corrcoef(pred, target)[0, 1])
    else:
        pearson = float("nan")

    target_mean = float(np.mean(target))
    total_sum = float(np.sum((target - target_mean) ** 2))
    if total_sum > 0.0:
        r2 = 1.0 - float(np.sum((pred - target) ** 2)) / total_sum
    else:
        r2 = float("nan")

    return {"mae": mae, "rmse": rmse, "pearson": pearson, "r2": r2}


def _evaluate(
    *,
    model: TrajectoryConditionedImputer,
    trajectory_encoder: FrozenTrajectoryEncoder | None,
    prior_encoder: nn.Module | None,
    dataloader: DataLoader,
    device: torch.device,
    preprocess: PreprocessArtifacts,
    use_trajectory: bool,
    loss_name: str,
    prior_loss_name: str | None,
    prior_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    if trajectory_encoder is not None:
        trajectory_encoder.eval()
    if prior_encoder is not None:
        prior_encoder.eval()

    losses: List[float] = []
    objective_losses: List[float] = []
    prior_losses: List[float] = []
    pred_values: List[np.ndarray] = []
    target_values: List[np.ndarray] = []
    masked_value_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            observed_mask = batch["observed_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            tokens = batch["tokens"].to(device)
            ages_days = batch["ages_days"].to(device)
            input_missing_mask = (~observed_mask) | loss_mask

            trajectory = trajectory_encoder(tokens, ages_days) if (use_trajectory and trajectory_encoder is not None) else None
            if prior_encoder is not None:
                pred, z_idp = model(x, input_missing_mask, trajectory, return_latent=True)
            else:
                pred = model(x, input_missing_mask, trajectory)
                z_idp = None

            recon_loss = _masked_loss(pred, x, loss_mask, loss_name=loss_name)
            losses.append(float(recon_loss.detach().cpu().item()))

            if prior_encoder is not None:
                prior_input = model.build_encoder_input(x, input_missing_mask)
                z_prior = prior_encoder(prior_input)
                prior_loss = _prior_alignment_loss(
                    z_idp,
                    z_prior,
                    loss_name=str(prior_loss_name or "cosine"),
                )
                prior_losses.append(float(prior_loss.detach().cpu().item()))
                objective_losses.append(float((recon_loss + float(prior_loss_weight) * prior_loss).detach().cpu().item()))
            else:
                objective_losses.append(float(recon_loss.detach().cpu().item()))

            pred_np, target_np = _flatten_masked_values(
                pred.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                loss_mask.detach().cpu().numpy(),
                preprocess,
            )
            masked_value_count += int(pred_np.size)
            if pred_np.size:
                pred_values.append(pred_np)
                target_values.append(target_np)

    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "objective_loss": float(np.mean(objective_losses)) if objective_losses else float("nan"),
        "prior_loss": float(np.mean(prior_losses)) if prior_losses else 0.0,
        "n_masked_values": float(masked_value_count),
    }
    if pred_values:
        pred_all = np.concatenate(pred_values)
        target_all = np.concatenate(target_values)
        metrics.update(_regression_metrics(pred_all, target_all))
    else:
        metrics.update(_regression_metrics(np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
    return metrics


def _build_samples(
    *,
    standardized_features: np.ndarray,
    observed_mask: np.ndarray,
    eids: np.ndarray,
    token_list: Sequence[torch.Tensor],
    age_list: Sequence[torch.Tensor],
    indices: np.ndarray,
    masking_cfg: dict,
    organ_to_indices: Dict[str, np.ndarray],
    seed: int,
) -> List[ImputationSample]:
    rng = np.random.default_rng(int(seed))
    samples: List[ImputationSample] = []

    mixture_cfg = masking_cfg.get("mixture", None)
    if mixture_cfg is not None:
        if not isinstance(mixture_cfg, list) or not mixture_cfg:
            raise ValueError("masking.mixture must be a non-empty list")
        masking_specs = [dict(item) for item in mixture_cfg]
        weights = np.asarray([float(item.get("weight", 1.0)) for item in masking_specs], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            raise ValueError("masking.mixture weights must be finite and sum to a positive value")
        weights = weights / float(weights.sum())
    else:
        masking_specs = [dict(masking_cfg)]
        weights = np.asarray([1.0], dtype=np.float64)

    for row_index in indices.tolist():
        row_observed = observed_mask[row_index].astype(bool, copy=False)
        spec_index = int(rng.choice(np.arange(len(masking_specs)), p=weights))
        spec = masking_specs[spec_index]
        mode = str(spec.get("mode", "random")).lower()
        ratio = float(spec.get("ratio", 0.3))
        fixed_organ = spec.get("organ", None)
        fixed_organ = str(fixed_organ) if fixed_organ is not None else None
        loss_mask = _sample_loss_mask(
            observed_mask=row_observed,
            mode=mode,
            ratio=ratio,
            organ_to_indices=organ_to_indices,
            rng=rng,
            fixed_organ=fixed_organ,
        )
        if not np.any(loss_mask):
            continue

        samples.append(
            ImputationSample(
                eid=torch.tensor(int(eids[row_index]), dtype=torch.long),
                x=torch.from_numpy(standardized_features[row_index].astype(np.float32)),
                observed_mask=torch.from_numpy(row_observed.astype(np.bool_)),
                loss_mask=torch.from_numpy(loss_mask.astype(np.bool_)),
                tokens=token_list[row_index],
                ages_days=age_list[row_index],
            )
        )

    return samples


def _save_checkpoint(
    *,
    path: Path,
    model: TrajectoryConditionedImputer,
    trajectory_encoder: FrozenTrajectoryEncoder | None,
    config: dict,
    feature_cols: Sequence[str],
    preprocess: PreprocessArtifacts,
    best_metrics: Dict[str, float],
    epoch: int,
) -> None:
    checkpoint = {
        "imputer_state": model.state_dict(),
        "trajectory_pooler_state": (None if trajectory_encoder is None else trajectory_encoder.pooler.state_dict()),
        "config": config,
        "feature_cols": list(feature_cols),
        "preprocess": {
            "mean_": np.asarray(preprocess.mean_, dtype=np.float32).tolist(),
            "std_": np.asarray(preprocess.std_, dtype=np.float32).tolist(),
        },
        "best_metrics": dict(best_metrics),
        "epoch": int(epoch),
    }
    torch.save(checkpoint, path)


def train_from_config(config_path: Path) -> None:
    import pandas as pd

    # 1. 读取配置并初始化实验环境。
    cfg = _load_yaml(config_path)
    _configure_logging(str(cfg.get("log_level", "INFO")))

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{config_path.stem}.log"
    _attach_file_logger(log_path, level=str(cfg.get("log_level", "INFO")))
    logger.info("日志将写入 %s", str(log_path))

    device = _device_from_cfg(cfg)
    eid_col = str(cfg.get("eid_col", "eid"))
    imaging_date_col = str(cfg.get("imaging_date_col", "imaging_date"))

    cohort_csv = cfg.get("cohort_csv", None)
    phenotype_csv = Path(cfg["phenotype_csv"])
    disease_onset_csv = Path(cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv"))
    basic_csv = Path(cfg.get("basic_csv", "data/demo/basic_info.csv"))
    labels_csv = Path(cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv"))
    allowed_eids_csv = cfg.get("allowed_eids_csv", None)
    pretrained_cfg = cfg.get("pretrained", {}) or {}
    pretrained_ckpt_path = pretrained_cfg.get("ckpt_path", None)
    distill_ckpt: dict | None = None
    pretrained_feature_names: List[str] | None = None
    if pretrained_ckpt_path is not None:
        distill_ckpt, pretrained_feature_names, _ = _load_pretrained_distill_ckpt(Path(pretrained_ckpt_path))

    prior_cfg = cfg.get("prior_preserving", {}) or {}
    use_prior_preserving = bool(prior_cfg.get("enabled", False))
    prior_loss_name = str(prior_cfg.get("loss", "cosine")).lower()
    prior_loss_weight = float(prior_cfg.get("weight", 0.0))

    # 2. 读取 phenotype/cohort，并整理出带影像日期的建模样本表。
    phenotype_df = pd.read_csv(phenotype_csv)
    if eid_col not in phenotype_df.columns:
        raise ValueError(f"phenotype_csv must contain column {eid_col}")
    phenotype_df[eid_col] = phenotype_df[eid_col].astype(int)

    if cohort_csv is not None:
        # 常规路径：cohort 提供 imaging_date，phenotype 提供 IDP 特征。
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

    if len(merged) == 0:
        raise ValueError("No samples remain after joining the cohort and phenotype tables")

    if allowed_eids_csv is not None:
        allowed_eids = set(_load_allowed_eids(Path(allowed_eids_csv), eid_col=eid_col).tolist())
        merged = merged[merged[eid_col].isin(allowed_eids)].reset_index(drop=True)
        logger.info("Applied allowed_eids filter from %s: %d rows remain", str(allowed_eids_csv), len(merged))
        if len(merged) == 0:
            raise ValueError("No samples remain after applying allowed_eids_csv filter")

    max_samples = cfg.get("max_samples", None)
    if max_samples is not None:
        merged = merged.iloc[: int(max_samples)].copy()

    exclude = {eid_col, imaging_date_col}
    feature_cols = [column for column in merged.columns if column not in exclude]
    if pretrained_feature_names is not None:
        feature_cols, fallback_pairs = _align_feature_columns_for_pretrained(
            current_feature_cols=feature_cols,
            pretrained_feature_names=pretrained_feature_names,
        )
        if fallback_pairs:
            logger.info(
                "Aligned %d pretrained feature names via organ+field fallback; examples=%s",
                len(fallback_pairs),
                fallback_pairs[:5],
            )
    if not feature_cols:
        raise ValueError("No phenotype feature columns were found for imputation")

    raw_features = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)
    observed_mask = np.isfinite(raw_features)

    min_observed_features = int(cfg.get("min_observed_features", 2))
    eligible = observed_mask.sum(axis=1) >= min_observed_features
    if not np.any(eligible):
        raise ValueError("No rows have enough observed IDP features for masking/imputation")

    merged = merged.loc[eligible].reset_index(drop=True)
    raw_features = raw_features[eligible]
    observed_mask = observed_mask[eligible]

    eids = merged[eid_col].to_numpy(dtype=np.int64, copy=True)
    imaging_dates = merged[imaging_date_col].to_numpy(copy=True)

    train_indices, val_indices, test_indices = split_indices(
        len(merged),
        seed=seed,
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
    )

    # 3. 仅用训练集拟合预处理，再把全体样本映射到统一标准化空间。
    preprocess = build_preprocess_pipeline()
    # 只在训练集上拟合均值和方差，避免把验证/测试信息泄漏进来。
    preprocess.fit(raw_features[train_indices])
    preprocess_artifacts = preprocess.to_artifacts()
    standardized_features = preprocess.transform(raw_features)

    if bool(cfg.get("check_finite", True)) and not np.isfinite(standardized_features).all():
        raise ValueError("Found NaN/Inf after phenotype standardization")

    organ_groups = group_feature_columns_by_organ(feature_cols)
    organ_to_indices = {
        organ_name: np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
        for organ_name, columns in organ_groups.items()
    }

    logger.info("Building pre-imaging Delphi trajectories for %d samples...", len(merged))
    # 4. 把每个人影像之前的病史转换成 Delphi 轨迹输入。
    token_list, age_list = _build_pre_imaging_sequences(
        eids=eids,
        imaging_date=imaging_dates,
        disease_onset_csv=disease_onset_csv,
        basic_csv=basic_csv,
        labels_csv=labels_csv,
        eid_col=eid_col,
        block_size=int(cfg.get("block_size", 256)),
        append_query_token=bool(cfg.get("append_query_token", True)),
    )

    masking_cfg = cfg.get("masking", {}) or {}
    # 5. 构造自监督样本：先遮掉一部分已观测 IDP，再要求模型只预测这些被遮挡位置。
    train_samples = _build_samples(
        standardized_features=standardized_features,
        observed_mask=observed_mask,
        eids=eids,
        token_list=token_list,
        age_list=age_list,
        indices=train_indices,
        masking_cfg=masking_cfg,
        organ_to_indices=organ_to_indices,
        seed=seed + 11,
    )
    val_samples = _build_samples(
        standardized_features=standardized_features,
        observed_mask=observed_mask,
        eids=eids,
        token_list=token_list,
        age_list=age_list,
        indices=val_indices,
        masking_cfg=masking_cfg,
        organ_to_indices=organ_to_indices,
        seed=seed + 23,
    )
    test_samples = _build_samples(
        standardized_features=standardized_features,
        observed_mask=observed_mask,
        eids=eids,
        token_list=token_list,
        age_list=age_list,
        indices=test_indices,
        masking_cfg=masking_cfg,
        organ_to_indices=organ_to_indices,
        seed=seed + 37,
    )

    if not train_samples:
        raise ValueError("No valid train samples remain after applying the masking strategy")

    logger.info(
        "Prepared imputation splits: train=%d val=%d test=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", device.type == "cuda"))

    # 6. 组装 dataloader。轨迹序列在 collate 阶段做左侧 padding。
    train_loader = DataLoader(
        ImputationDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_left_pad,
    )
    val_loader = DataLoader(
        ImputationDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_left_pad,
    ) if val_samples else None
    test_loader = DataLoader(
        ImputationDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_left_pad,
    ) if test_samples else None

    encoder_cfg = cfg.get("encoder", None)
    if encoder_cfg is None:
        raise ValueError("Missing required 'encoder' section in the config")

    # 7. 构建 phenotype encoder；如果配置要求，再额外加载冻结的 Delphi 轨迹编码器。
    if distill_ckpt is not None:
        distill_cfg = (distill_ckpt.get("config", {}) or {})
        distill_encoder_cfg = (distill_cfg.get("encoder", {}) or {})
        encoder, encoder_dim = build_encoder_from_config(encoder_cfg=distill_encoder_cfg, feature_cols=feature_cols)
        encoder.load_state_dict(distill_ckpt["model_state"], strict=True)
        freeze_pretrained = bool(pretrained_cfg.get("freeze_encoder", False))
        if freeze_pretrained:
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
        logger.info(
            "Loaded pretrained phenotype encoder from %s (freeze=%s)",
            str(pretrained_ckpt_path),
            str(freeze_pretrained),
        )
    else:
        encoder, encoder_dim = build_encoder_from_config(encoder_cfg=encoder_cfg, feature_cols=feature_cols)

    use_trajectory = bool(cfg.get("use_trajectory", True))
    trajectory_fusion = str(cfg.get("trajectory_fusion", "concat" if use_trajectory else "none")).lower()
    trajectory_encoder: FrozenTrajectoryEncoder | None = None
    trajectory_dim = 0
    if use_trajectory:
        delphi_ckpt_dir = Path(cfg.get("delphi_ckpt_dir", "runs/demo/teacher"))
        logger.info("Loading frozen Delphi checkpoint from %s", str(delphi_ckpt_dir))
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

    prior_encoder: nn.Module | None = None
    if use_prior_preserving:
        if distill_ckpt is None:
            raise ValueError(
                "prior_preserving.enabled=true currently requires a distill checkpoint loaded via pretrained.ckpt_path"
            )
        distill_cfg = (distill_ckpt.get("config", {}) or {})
        distill_encoder_cfg = (distill_cfg.get("encoder", {}) or {})
        prior_encoder, prior_encoder_dim = build_encoder_from_config(
            encoder_cfg=distill_encoder_cfg,
            feature_cols=feature_cols,
        )
        prior_encoder.load_state_dict(distill_ckpt["model_state"], strict=True)
        for parameter in prior_encoder.parameters():
            parameter.requires_grad_(False)
        if int(prior_encoder_dim) != int(encoder_dim):
            raise ValueError(
                f"Prior encoder dim mismatch: trainable encoder={encoder_dim}, prior encoder={prior_encoder_dim}"
            )
        logger.info(
            "Enabled prior-preserving training with loss=%s weight=%.6f using %s",
            prior_loss_name,
            prior_loss_weight,
            str(pretrained_ckpt_path),
        )

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

    # 8. 优化器同时更新补全模型参数，以及未冻结的轨迹池化头参数。
    model.to(device)
    if trajectory_encoder is not None:
        trajectory_encoder.to(device)
    if prior_encoder is not None:
        prior_encoder.to(device)
        prior_encoder.eval()

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if trajectory_encoder is not None:
        params.extend(parameter for parameter in trajectory_encoder.parameters() if parameter.requires_grad)

    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    grad_clip = cfg.get("gradient_clip_val", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None
    loss_name = str(cfg.get("loss", "mse")).lower()
    epochs = int(cfg.get("epochs", 20))
    monitor = str(cfg.get("monitor", "mae")).lower()
    monitor_mode = str(cfg.get("monitor_mode", "min")).lower()

    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    best_val_metrics: Dict[str, float] = {}

    # 9. 标准训练循环：训练一个 epoch，再在验证集上挑选最优 checkpoint。
    for epoch in range(1, epochs + 1):
        model.train()
        if trajectory_encoder is not None:
            trajectory_encoder.train()
        if prior_encoder is not None:
            prior_encoder.eval()

        batch_recon_losses: List[float] = []
        batch_prior_losses: List[float] = []
        batch_total_losses: List[float] = []
        for batch in train_loader:
            x = batch["x"].to(device)
            observed_mask = batch["observed_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            tokens = batch["tokens"].to(device)
            ages_days = batch["ages_days"].to(device)
            # 输入里既要屏蔽“原本就缺失”的位置，也要屏蔽“这次人为遮挡”的位置。
            # 但 loss 只会算在人为遮挡、且原本真实存在的那些值上。
            input_missing_mask = (~observed_mask) | loss_mask

            optimizer.zero_grad(set_to_none=True)
            trajectory = trajectory_encoder(tokens, ages_days) if (use_trajectory and trajectory_encoder is not None) else None
            if prior_encoder is not None:
                pred, z_idp = model(x, input_missing_mask, trajectory, return_latent=True)
            else:
                pred = model(x, input_missing_mask, trajectory)
                z_idp = None

            recon_loss = _masked_loss(pred, x, loss_mask, loss_name=loss_name)
            if prior_encoder is not None:
                prior_input = model.build_encoder_input(x, input_missing_mask)
                with torch.no_grad():
                    z_prior = prior_encoder(prior_input)
                prior_loss = _prior_alignment_loss(z_idp, z_prior, loss_name=prior_loss_name)
            else:
                prior_loss = recon_loss.new_tensor(0.0)

            loss = recon_loss + prior_loss_weight * prior_loss
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            batch_recon_losses.append(float(recon_loss.detach().cpu().item()))
            batch_prior_losses.append(float(prior_loss.detach().cpu().item()))
            batch_total_losses.append(float(loss.detach().cpu().item()))

        train_recon_loss = float(np.mean(batch_recon_losses)) if batch_recon_losses else float("nan")
        train_prior_loss = float(np.mean(batch_prior_losses)) if batch_prior_losses else float("nan")
        train_total_loss = float(np.mean(batch_total_losses)) if batch_total_losses else float("nan")
        val_metrics = _evaluate(
            model=model,
            trajectory_encoder=trajectory_encoder,
            prior_encoder=prior_encoder,
            dataloader=val_loader if val_loader is not None else train_loader,
            device=device,
            preprocess=preprocess_artifacts,
            use_trajectory=use_trajectory,
            loss_name=loss_name,
            prior_loss_name=prior_loss_name if prior_encoder is not None else None,
            prior_loss_weight=prior_loss_weight,
        )

        current_metric = float(val_metrics.get(monitor, float("nan")))
        improved = False
        if math.isfinite(current_metric):
            if monitor_mode == "min":
                improved = current_metric < best_metric
            else:
                improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_val_metrics = dict(val_metrics)
            _save_checkpoint(
                path=out_dir / "best.pt",
                model=model,
                trajectory_encoder=trajectory_encoder,
                config=cfg,
                feature_cols=feature_cols,
                preprocess=preprocess_artifacts,
                best_metrics=best_val_metrics,
                epoch=epoch,
            )

        logger.info(
            "epoch=%d train_recon=%.6f train_prior=%.6f train_total=%.6f val_loss=%.6f val_prior=%.6f val_obj=%.6f val_mae=%.6f val_rmse=%.6f val_pearson=%s val_r2=%s",
            epoch,
            train_recon_loss,
            train_prior_loss,
            train_total_loss,
            float(val_metrics.get("loss", float("nan"))),
            float(val_metrics.get("prior_loss", float("nan"))),
            float(val_metrics.get("objective_loss", float("nan"))),
            float(val_metrics.get("mae", float("nan"))),
            float(val_metrics.get("rmse", float("nan"))),
            f"{float(val_metrics.get('pearson', float('nan'))):.6f}" if math.isfinite(float(val_metrics.get("pearson", float("nan")))) else "nan",
            f"{float(val_metrics.get('r2', float('nan'))):.6f}" if math.isfinite(float(val_metrics.get("r2", float("nan")))) else "nan",
        )

    if not (out_dir / "best.pt").exists():
        _save_checkpoint(
            path=out_dir / "best.pt",
            model=model,
            trajectory_encoder=trajectory_encoder,
            config=cfg,
            feature_cols=feature_cols,
            preprocess=preprocess_artifacts,
            best_metrics=best_val_metrics,
            epoch=epochs,
        )

    checkpoint = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    # 10. 用验证集选出的最佳 checkpoint 做最终测试，避免直接拿最后一个 epoch 的结果。
    model.load_state_dict(checkpoint["imputer_state"], strict=True)
    model.to(device)
    if trajectory_encoder is not None and checkpoint.get("trajectory_pooler_state") is not None:
        trajectory_encoder.pooler.load_state_dict(checkpoint["trajectory_pooler_state"], strict=True)
        trajectory_encoder.to(device)

    summary: Dict[str, Any] = {
        "config": cfg,
        "best_val": dict(checkpoint.get("best_metrics", {})),
    }

    if test_loader is not None:
        test_metrics = _evaluate(
            model=model,
            trajectory_encoder=trajectory_encoder,
            prior_encoder=prior_encoder,
            dataloader=test_loader,
            device=device,
            preprocess=preprocess_artifacts,
            use_trajectory=use_trajectory,
            loss_name=loss_name,
            prior_loss_name=prior_loss_name if prior_encoder is not None else None,
            prior_loss_weight=prior_loss_weight,
        )
        summary["test"] = test_metrics
        logger.info(
            "test_loss=%.6f test_prior=%.6f test_obj=%.6f test_mae=%.6f test_rmse=%.6f test_pearson=%s test_r2=%s",
            float(test_metrics.get("loss", float("nan"))),
            float(test_metrics.get("prior_loss", float("nan"))),
            float(test_metrics.get("objective_loss", float("nan"))),
            float(test_metrics.get("mae", float("nan"))),
            float(test_metrics.get("rmse", float("nan"))),
            f"{float(test_metrics.get('pearson', float('nan'))):.6f}" if math.isfinite(float(test_metrics.get("pearson", float("nan")))) else "nan",
            f"{float(test_metrics.get('r2', float('nan'))):.6f}" if math.isfinite(float(test_metrics.get("r2", float("nan")))) else "nan",
        )

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = _parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
