from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenotype_encoder.data import build_preprocess_pipeline, group_feature_columns_by_organ
from phenotype_encoder.model import build_encoder_from_config
from scripts.downstream.utils import load_delphi_checkpoint, split_indices
from scripts.imputation.train_idp_imputation import (
    FrozenTrajectoryEncoder,
    ImputationDataset,
    TrajectoryConditionedImputer,
    _align_feature_columns_for_pretrained,
    _build_pre_imaging_sequences,
    _build_samples,
    _collate_left_pad,
    _device_from_cfg,
    _load_allowed_eids,
    _load_pretrained_distill_ckpt,
    _load_yaml,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two imputation models by organ on the same multiseed test splits.")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/demo/imputation.yaml"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/demo/imputation.yaml"),
    )
    parser.add_argument(
        "--baseline-runs-root",
        type=Path,
        default=Path("runs/imputation_multiseed/idp_only_random_distill_init_traj_residual_noleak"),
    )
    parser.add_argument(
        "--candidate-runs-root",
        type=Path,
        default=Path("runs/imputation_multiseed/idp_only_random_distill_init_traj_organ_residual_noleak"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _seed_from_dir(path: Path) -> int:
    return int(path.name.split("_")[-1])


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - target)))


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(array.mean()), "std": float(array.std())}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(float(denominator)) < 1e-12:
        return float("nan")
    return float(numerator) / float(denominator)


def _prepare_shared_inputs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cohort_csv = cfg.get("cohort_csv", None)
    phenotype_csv = Path(cfg["phenotype_csv"])
    disease_onset_csv = Path(cfg.get("disease_onset_csv", "data/demo/disease_onsets.csv"))
    basic_csv = Path(cfg.get("basic_csv", "data/demo/basic_info.csv"))
    labels_csv = Path(cfg.get("delphi_labels_csv", "Delphi/data/demo/labels.csv"))
    allowed_eids_csv = cfg.get("allowed_eids_csv", None)
    eid_col = str(cfg.get("eid_col", "eid"))
    imaging_date_col = str(cfg.get("imaging_date_col", "imaging_date"))

    pretrained_cfg = cfg.get("pretrained", {}) or {}
    pretrained_ckpt_path = pretrained_cfg.get("ckpt_path", None)
    distill_ckpt: Dict[str, Any] | None = None
    pretrained_feature_names: List[str] | None = None
    if pretrained_ckpt_path is not None:
        distill_ckpt, pretrained_feature_names, _ = _load_pretrained_distill_ckpt(Path(pretrained_ckpt_path))

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

    max_samples = cfg.get("max_samples", None)
    if max_samples is not None:
        merged = merged.iloc[: int(max_samples)].copy()

    exclude = {eid_col, imaging_date_col}
    feature_cols = [column for column in merged.columns if column not in exclude]
    if pretrained_feature_names is not None:
        feature_cols, _ = _align_feature_columns_for_pretrained(
            current_feature_cols=feature_cols,
            pretrained_feature_names=pretrained_feature_names,
        )

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

    organ_groups = group_feature_columns_by_organ(feature_cols)
    organ_to_indices = {
        organ_name: np.asarray([feature_cols.index(column) for column in columns], dtype=np.int64)
        for organ_name, columns in organ_groups.items()
    }

    return {
        "cfg": cfg,
        "feature_cols": feature_cols,
        "raw_features": raw_features,
        "observed_mask": observed_mask,
        "eids": eids,
        "token_list": token_list,
        "age_list": age_list,
        "organ_to_indices": organ_to_indices,
        "distill_ckpt": distill_ckpt,
        "pretrained_cfg": pretrained_cfg,
    }


def _build_seed_test_loader(shared: Dict[str, Any], seed: int, batch_size: int, device: torch.device) -> Dict[str, Any]:
    cfg = shared["cfg"]
    raw_features = shared["raw_features"]
    observed_mask = shared["observed_mask"]
    eids = shared["eids"]
    token_list = shared["token_list"]
    age_list = shared["age_list"]
    organ_to_indices = shared["organ_to_indices"]

    train_indices, _, test_indices = split_indices(
        len(raw_features),
        seed=seed,
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
    )

    preprocess = build_preprocess_pipeline()
    preprocess.fit(raw_features[train_indices])
    preprocess_artifacts = preprocess.to_artifacts()
    standardized_features = preprocess.transform(raw_features)

    test_samples = _build_samples(
        standardized_features=standardized_features,
        observed_mask=observed_mask,
        eids=eids,
        token_list=token_list,
        age_list=age_list,
        indices=test_indices,
        masking_cfg=cfg.get("masking", {}) or {},
        organ_to_indices=organ_to_indices,
        seed=seed + 37,
    )
    if not test_samples:
        raise ValueError(f"No valid test samples remain for seed={seed}")

    dataloader = DataLoader(
        ImputationDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
        collate_fn=_collate_left_pad,
    )
    return {
        "preprocess": preprocess_artifacts,
        "dataloader": dataloader,
    }


def _build_seed_model(shared: Dict[str, Any], seed_dir: Path, device: torch.device) -> Dict[str, Any]:
    cfg = shared["cfg"]
    feature_cols = shared["feature_cols"]
    organ_to_indices = shared["organ_to_indices"]
    distill_ckpt = shared["distill_ckpt"]
    pretrained_cfg = shared["pretrained_cfg"]

    encoder_cfg = cfg.get("encoder", None)
    if encoder_cfg is None:
        raise ValueError("Missing required 'encoder' section in the config")

    if distill_ckpt is not None:
        distill_cfg = (distill_ckpt.get("config", {}) or {})
        distill_encoder_cfg = (distill_cfg.get("encoder", {}) or {})
        encoder, encoder_dim = build_encoder_from_config(encoder_cfg=distill_encoder_cfg, feature_cols=feature_cols)
        encoder.load_state_dict(distill_ckpt["model_state"], strict=True)
        if bool(pretrained_cfg.get("freeze_encoder", False)):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
    else:
        encoder, encoder_dim = build_encoder_from_config(encoder_cfg=encoder_cfg, feature_cols=feature_cols)

    use_trajectory = bool(cfg.get("use_trajectory", True))
    trajectory_fusion = str(cfg.get("trajectory_fusion", "concat" if use_trajectory else "none")).lower()

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
        trajectory_fusion=trajectory_fusion,
        organ_to_indices=organ_to_indices,
        decoder_hidden_dims=[int(value) for value in cfg.get("decoder_hidden_dims", [512, 256])],
        dropout=float(cfg.get("decoder_dropout", 0.1)),
        act=str(cfg.get("decoder_act", "relu")),
    )

    checkpoint = torch.load(seed_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["imputer_state"], strict=True)
    model.to(device)
    model.eval()

    if trajectory_encoder is not None and checkpoint.get("trajectory_pooler_state") is not None:
        trajectory_encoder.pooler.load_state_dict(checkpoint["trajectory_pooler_state"], strict=True)
        trajectory_encoder.to(device)
        trajectory_encoder.eval()

    return {
        "model": model,
        "trajectory_encoder": trajectory_encoder,
    }


def _collect_predictions_by_organ(
    *,
    shared: Dict[str, Any],
    seed: int,
    seed_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    run_data = _build_seed_test_loader(shared, seed, batch_size=batch_size, device=device)
    model_data = _build_seed_model(shared, seed_dir=seed_dir, device=device)

    preprocess = run_data["preprocess"]
    dataloader = run_data["dataloader"]
    model: TrajectoryConditionedImputer = model_data["model"]
    trajectory_encoder: FrozenTrajectoryEncoder | None = model_data["trajectory_encoder"]
    organ_to_indices = shared["organ_to_indices"]

    outputs: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            observed_mask = batch["observed_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            tokens = batch["tokens"].to(device)
            ages_days = batch["ages_days"].to(device)
            input_missing_mask = (~observed_mask) | loss_mask

            trajectory = trajectory_encoder(tokens, ages_days) if trajectory_encoder is not None else None
            pred = model(x, input_missing_mask, trajectory)

            for organ_name, organ_indices in organ_to_indices.items():
                organ_loss_mask = loss_mask[:, organ_indices]
                if organ_loss_mask.sum().item() == 0:
                    continue
                mean = torch.as_tensor(preprocess.mean_[organ_indices], dtype=x.dtype, device=device).unsqueeze(0)
                std = torch.as_tensor(preprocess.std_[organ_indices], dtype=x.dtype, device=device).unsqueeze(0)
                pred_raw = pred[:, organ_indices] * std + mean
                target_raw = x[:, organ_indices] * std + mean
                mask_np = organ_loss_mask.detach().cpu().numpy().astype(bool)
                outputs[organ_name]["pred"].append(pred_raw.detach().cpu().numpy()[mask_np])
                outputs[organ_name]["target"].append(target_raw.detach().cpu().numpy()[mask_np])

    final_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for organ_name, arrays in outputs.items():
        final_outputs[organ_name] = {
            "pred": np.concatenate(arrays["pred"]) if arrays["pred"] else np.array([], dtype=np.float32),
            "target": np.concatenate(arrays["target"]) if arrays["target"] else np.array([], dtype=np.float32),
        }

    del model
    if trajectory_encoder is not None:
        del trajectory_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_outputs


def _collect_seed_comparison(
    *,
    baseline_shared: Dict[str, Any],
    candidate_shared: Dict[str, Any],
    seed: int,
    baseline_seed_dir: Path,
    candidate_seed_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    baseline_outputs = _collect_predictions_by_organ(
        shared=baseline_shared,
        seed=seed,
        seed_dir=baseline_seed_dir,
        device=device,
        batch_size=batch_size,
    )
    candidate_outputs = _collect_predictions_by_organ(
        shared=candidate_shared,
        seed=seed,
        seed_dir=candidate_seed_dir,
        device=device,
        batch_size=batch_size,
    )

    organs = sorted(set(baseline_outputs.keys()) & set(candidate_outputs.keys()))
    result: Dict[str, Dict[str, float]] = {}
    for organ_name in organs:
        baseline_pred = baseline_outputs[organ_name]["pred"]
        baseline_target = baseline_outputs[organ_name]["target"]
        candidate_pred = candidate_outputs[organ_name]["pred"]
        candidate_target = candidate_outputs[organ_name]["target"]
        if baseline_target.size != candidate_target.size or not np.allclose(baseline_target, candidate_target):
            raise ValueError(f"Target mismatch for organ={organ_name}, seed={seed}")
        target_std = float(candidate_target.std()) if candidate_target.size else float("nan")
        baseline_mae = _mae(baseline_pred, baseline_target)
        candidate_mae = _mae(candidate_pred, candidate_target)
        baseline_rmse = _rmse(baseline_pred, baseline_target)
        candidate_rmse = _rmse(candidate_pred, candidate_target)
        mae_gain = baseline_mae - candidate_mae
        rmse_gain = baseline_rmse - candidate_rmse
        result[organ_name] = {
            "masked_values": float(candidate_target.size),
            "target_std": target_std,
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
            "mae_gain": mae_gain,
            "mae_gain_pct": 100.0 * _safe_ratio(mae_gain, baseline_mae),
            "mae_gain_over_target_std": _safe_ratio(mae_gain, target_std),
            "baseline_rmse": baseline_rmse,
            "candidate_rmse": candidate_rmse,
            "rmse_gain": rmse_gain,
            "rmse_gain_pct": 100.0 * _safe_ratio(rmse_gain, baseline_rmse),
            "rmse_gain_over_target_std": _safe_ratio(rmse_gain, target_std),
        }
    return result


def _summarize(per_seed: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    by_organ: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for seed, organ_stats in per_seed.items():
        for organ_name, metrics in organ_stats.items():
            row = dict(metrics)
            row["seed"] = float(seed)
            by_organ[organ_name].append(row)

    organ_summary: List[Dict[str, Any]] = []
    for organ_name, rows in by_organ.items():
        item: Dict[str, Any] = {"organ": organ_name, "seeds": len(rows)}
        for metric_name in [
            "masked_values",
            "target_std",
            "baseline_mae",
            "candidate_mae",
            "mae_gain",
            "mae_gain_pct",
            "mae_gain_over_target_std",
            "baseline_rmse",
            "candidate_rmse",
            "rmse_gain",
            "rmse_gain_pct",
            "rmse_gain_over_target_std",
        ]:
            stats = _mean_std([float(row[metric_name]) for row in rows])
            item[f"{metric_name}_mean"] = stats["mean"]
            item[f"{metric_name}_std"] = stats["std"]
        organ_summary.append(item)

    organ_summary.sort(key=lambda item: item["mae_gain_mean"], reverse=True)
    return {
        "organ_summary": organ_summary,
        "seeds": sorted(per_seed.keys()),
        "per_seed": per_seed,
    }


def _top_lines(rows: Sequence[Dict[str, Any]], metric_key: str, n: int, reverse: bool) -> List[str]:
    ordered = sorted(rows, key=lambda item: item[metric_key], reverse=reverse)[:n]
    return [f"- {item['organ']}: {metric_key}={item[metric_key]:.4f}" for item in ordered]


def _render_markdown(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    rows = summary["organ_summary"]
    lines: List[str] = ["# Organ-Level Model Compare", ""]
    lines.append(f"- baseline_config: `{args.baseline_config}`")
    lines.append(f"- candidate_config: `{args.candidate_config}`")
    lines.append(f"- baseline_runs_root: `{args.baseline_runs_root}`")
    lines.append(f"- candidate_runs_root: `{args.candidate_runs_root}`")
    lines.append(f"- seeds: `{summary['seeds']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("Top organs by candidate MAE gain:")
    lines.extend(_top_lines(rows, "mae_gain_mean", n=min(5, len(rows)), reverse=True))
    lines.append("")
    lines.append("Top organs by candidate relative MAE gain (%):")
    lines.extend(_top_lines(rows, "mae_gain_pct_mean", n=min(5, len(rows)), reverse=True))
    lines.append("")
    lines.append("Note: positive gain means the candidate beats the baseline.")
    lines.append("Note: `*_over_target_std` divides by the masked target standard deviation within that organ for scale-aware comparison.")
    lines.append("")
    lines.append("## Organ Table")
    lines.append("")
    lines.append("| organ | masked values/seed | target std | baseline mae | candidate mae | mae gain | mae gain % | mae gain / target std | baseline rmse | candidate rmse | rmse gain | rmse gain % | rmse gain / target std |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in rows:
        lines.append(
            "| {organ} | {masked_values_mean:.1f} | {target_std_mean:.4f} | {baseline_mae_mean:.4f} | {candidate_mae_mean:.4f} | {mae_gain_mean:.4f} | {mae_gain_pct_mean:.2f} | {mae_gain_over_target_std_mean:.4f} | {baseline_rmse_mean:.4f} | {candidate_rmse_mean:.4f} | {rmse_gain_mean:.4f} | {rmse_gain_pct_mean:.2f} | {rmse_gain_over_target_std_mean:.4f} |".format(
                **item
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    baseline_cfg = _load_yaml(args.baseline_config)
    candidate_cfg = _load_yaml(args.candidate_config)
    if args.device is not None:
        baseline_cfg = dict(baseline_cfg)
        candidate_cfg = dict(candidate_cfg)
        baseline_cfg["device"] = args.device
        candidate_cfg["device"] = args.device
    device = _device_from_cfg(candidate_cfg)

    baseline_shared = _prepare_shared_inputs(baseline_cfg)
    candidate_shared = _prepare_shared_inputs(candidate_cfg)

    baseline_seed_dirs = { _seed_from_dir(path): path for path in args.baseline_runs_root.glob("seed_*") if (path / "best.pt").exists() }
    candidate_seed_dirs = { _seed_from_dir(path): path for path in args.candidate_runs_root.glob("seed_*") if (path / "best.pt").exists() }
    shared_seeds = sorted(set(baseline_seed_dirs) & set(candidate_seed_dirs))
    if not shared_seeds:
        raise FileNotFoundError("No shared completed seed directories found between baseline and candidate runs")

    per_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed in shared_seeds:
        per_seed[seed] = _collect_seed_comparison(
            baseline_shared=baseline_shared,
            candidate_shared=candidate_shared,
            seed=seed,
            baseline_seed_dir=baseline_seed_dirs[seed],
            candidate_seed_dir=candidate_seed_dirs[seed],
            device=device,
            batch_size=int(args.batch_size),
        )

    summary = _summarize(per_seed)
    markdown = _render_markdown(summary, args)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
