from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    in_dim: int
    hidden_dims: List[int]
    out_dim: int
    dropout: float = 0.0
    act: str = "relu"  # relu | gelu | silu


class PhenotypeMLP(nn.Module):
    """Simple configurable MLP encoder for tabular phenotype features."""

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.act == "relu":
            act_fn = nn.ReLU
        elif cfg.act == "gelu":
            act_fn = nn.GELU
        elif cfg.act in {"silu", "swish"}:
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {cfg.act}")

        layers: List[nn.Module] = []
        d_in = cfg.in_dim
        for d_h in cfg.hidden_dims:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(act_fn())
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            d_in = d_h

        layers.append(nn.Linear(d_in, cfg.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class OrganEncoderConfig:
    """Configuration for organ-aware phenotype encoder.

    The encoder splits the input feature vector into organ-specific subvectors
    based on feature name prefixes (e.g. "lung__...") and processes each organ
    with a per-organ linear alignment layer plus a shared MLP.
    """

    # Dimensions
    align_dim: int = 128  # h_o dim
    organ_embed_dim: int = 32  # e_o dim
    token_dim: int = 128  # z_o dim (before aggregation)
    shared_hidden_dims: List[int] = None  # MLP hidden dims for SharedMLP
    dropout: float = 0.0
    act: str = "relu"  # relu | gelu | silu

    # Aggregation
    aggregator: Literal["concat", "weighted_pool", "transformer"] = "concat"

    # Weighted pooling
    pool_temperature: float = 1.0
    pool_hidden_dim: int = 64

    # Transformer aggregation
    nhead: int = 4
    num_layers: int = 2
    ff_mult: int = 4
    transformer_dropout: float = 0.0


def _act_from_name(name: str):
    n = str(name).lower()
    if n == "relu":
        return nn.ReLU
    if n == "gelu":
        return nn.GELU
    if n in {"silu", "swish"}:
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


class _SharedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, *, dropout: float, act: str):
        super().__init__()
        act_fn = _act_from_name(act)
        layers: List[nn.Module] = []
        d = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(act_fn())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrganPhenotypeEncoder(nn.Module):
    """Organ-aware phenotype encoder.

    Pipeline:
      - Split input x into per-organ subvectors x_o.
      - Per-organ alignment: h_o = Linear_o(x_o)  (R^{align_dim})
      - Organ embedding: e_o = Embedding[o_id]    (R^{organ_embed_dim})
      - Shared MLP: z_o = SharedMLP(concat(h_o, e_o)) (R^{token_dim})

    Aggregation options:
      1) concat (baseline): z_all = concat(z_1..z_O)  -> output_dim = O * token_dim
      2) weighted_pool: softmax weights over organs, pooled sum -> output_dim = token_dim
      3) transformer: Transformer over tokens with CLS -> output_dim = token_dim
    """

    def __init__(
        self,
        *,
        feature_cols: Sequence[str],
        organ_to_cols: Dict[str, Sequence[str]],
        cfg: OrganEncoderConfig,
        feature_sep: str = "__",
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.organ_names = list(organ_to_cols.keys())
        self.cfg = cfg
        self.feature_sep = str(feature_sep)

        if cfg.shared_hidden_dims is None:
            shared_hidden = [256]
        else:
            shared_hidden = [int(x) for x in cfg.shared_hidden_dims]

        # Map feature name -> position in input x.
        feat_pos = {c: i for i, c in enumerate(self.feature_cols)}

        # Per-organ indices and per-organ Linear align.
        self.organ_linears = nn.ModuleDict()
        self._organ_index_keys: List[str] = []
        for organ in self.organ_names:
            cols = list(organ_to_cols[organ])
            idx = [feat_pos[c] for c in cols if c in feat_pos]
            if not idx:
                raise ValueError(f"Organ {organ!r} has no columns after alignment with feature_cols")

            safe = str(organ).replace(" ", "_").replace("/", "_").replace("-", "_")
            key = f"idx__{safe}"
            # Keep indices in checkpoint for reproducibility/debug.
            self.register_buffer(key, torch.tensor(idx, dtype=torch.long), persistent=True)
            self._organ_index_keys.append(key)

            self.organ_linears[organ] = nn.Linear(len(idx), int(cfg.align_dim))

        self.organ_embedding = nn.Embedding(len(self.organ_names), int(cfg.organ_embed_dim))

        shared_in = int(cfg.align_dim) + int(cfg.organ_embed_dim)
        self.shared_mlp = _SharedMLP(
            shared_in,
            hidden_dims=shared_hidden,
            out_dim=int(cfg.token_dim),
            dropout=float(cfg.dropout),
            act=str(cfg.act),
        )

        agg = str(cfg.aggregator).lower()
        if agg not in {"concat", "weighted_pool", "transformer"}:
            raise ValueError(f"Unsupported aggregator={cfg.aggregator!r}")
        self.aggregator = agg

        # Weighted pooling head over tokens.
        self.pool_gate: nn.Module | None = None
        if self.aggregator == "weighted_pool":
            gate_hidden = int(cfg.pool_hidden_dim)
            self.pool_gate = nn.Sequential(
                nn.Linear(int(cfg.token_dim), gate_hidden),
                _act_from_name(str(cfg.act))(),
                nn.Linear(gate_hidden, 1),
            )

        # Transformer aggregation.
        self.cls_token: nn.Parameter | None = None
        self.transformer: nn.Module | None = None
        if self.aggregator == "transformer":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(cfg.token_dim)))
            layer = nn.TransformerEncoderLayer(
                d_model=int(cfg.token_dim),
                nhead=int(cfg.nhead),
                dim_feedforward=int(cfg.token_dim) * int(cfg.ff_mult),
                dropout=float(cfg.transformer_dropout),
                batch_first=True,
                activation=str(cfg.act),
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=int(cfg.num_layers))

    @property
    def output_dim(self) -> int:
        if self.aggregator == "concat":
            return int(len(self.organ_names) * int(self.cfg.token_dim))
        return int(self.cfg.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,F)
        if x.dim() != 2:
            raise ValueError(f"Expected x with shape (B,F), got {tuple(x.shape)}")

        B = x.shape[0]
        tokens: List[torch.Tensor] = []
        for organ_id, organ in enumerate(self.organ_names):
            idx_key = self._organ_index_keys[organ_id]
            idx = getattr(self, idx_key)
            x_o = x.index_select(dim=1, index=idx)
            h_o = self.organ_linears[organ](x_o)
            e_o = self.organ_embedding(torch.full((B,), organ_id, device=x.device, dtype=torch.long))
            z_o = self.shared_mlp(torch.cat([h_o, e_o], dim=-1))
            tokens.append(z_o)

        if self.aggregator == "concat":
            return torch.cat(tokens, dim=-1)

        z = torch.stack(tokens, dim=1)  # (B,O,D)

        if self.aggregator == "weighted_pool":
            if self.pool_gate is None:
                raise RuntimeError("pool_gate is not initialized")
            scores = self.pool_gate(z).squeeze(-1)  # (B,O)
            temp = float(self.cfg.pool_temperature) if self.cfg.pool_temperature else 1.0
            scores = scores / max(1e-6, temp)
            w = torch.softmax(scores, dim=1)
            return (z * w.unsqueeze(-1)).sum(dim=1)

        if self.aggregator == "transformer":
            if self.transformer is None or self.cls_token is None:
                raise RuntimeError("transformer/cls_token not initialized")
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, z], dim=1)  # (B,1+O,D)
            out = self.transformer(seq)
            return out[:, 0, :]

        raise RuntimeError(f"Unhandled aggregator: {self.aggregator}")


def build_encoder_from_config(
    *,
    encoder_cfg: Dict,
    feature_cols: Sequence[str],
    feature_sep: str = "__",
) -> Tuple[nn.Module, int]:
    """Factory to build phenotype encoders from config dicts.

    Supports:
      - type: "mlp" (default)
      - type: "organ" (organ-aware)

    Returns (encoder_module, output_dim).
    """

    cfg = dict(encoder_cfg or {})
    enc_type = str(cfg.get("type", "mlp")).lower()

    if enc_type in {"mlp", "baseline"}:
        out_dim = int(cfg.get("out_dim", cfg.get("embed_dim", 128)))
        hidden_dims = [int(x) for x in cfg.get("hidden_dims", [512, 256])]
        dropout = float(cfg.get("dropout", 0.1))
        act = str(cfg.get("act", "relu"))
        enc = PhenotypeMLP(MLPConfig(in_dim=len(feature_cols), hidden_dims=hidden_dims, out_dim=out_dim, dropout=dropout, act=act))
        return enc, out_dim

    if enc_type in {"organ", "organ_encoder", "organ-encoder"}:
        # Lazy import to avoid tight coupling.
        from phenotype_encoder.data import group_feature_columns_by_organ

        organ_groups = group_feature_columns_by_organ(list(feature_cols), sep=str(feature_sep), default_organ=str(cfg.get("default_organ", "global")))

        organ_cfg = OrganEncoderConfig(
            align_dim=int(cfg.get("align_dim", cfg.get("h_dim", 128))),
            organ_embed_dim=int(cfg.get("organ_embed_dim", cfg.get("e_dim", 32))),
            token_dim=int(cfg.get("token_dim", cfg.get("out_dim", cfg.get("embed_dim", 128)))),
            shared_hidden_dims=[int(x) for x in cfg.get("shared_hidden_dims", cfg.get("hidden_dims", [256]))],
            dropout=float(cfg.get("dropout", 0.0)),
            act=str(cfg.get("act", "relu")),
            aggregator=str(cfg.get("aggregator", "concat")).lower(),
            pool_temperature=float(cfg.get("pool_temperature", 1.0)),
            pool_hidden_dim=int(cfg.get("pool_hidden_dim", 64)),
            nhead=int(cfg.get("nhead", 4)),
            num_layers=int(cfg.get("num_layers", 2)),
            ff_mult=int(cfg.get("ff_mult", 4)),
            transformer_dropout=float(cfg.get("transformer_dropout", cfg.get("dropout", 0.0))),
        )

        enc = OrganPhenotypeEncoder(feature_cols=feature_cols, organ_to_cols=organ_groups, cfg=organ_cfg, feature_sep=str(feature_sep))
        return enc, int(enc.output_dim)

    raise ValueError(f"Unsupported encoder type: {enc_type!r}")
