#!/usr/bin/env python3
"""Audit incident/prevalent leakage properties of downstream IDP risksets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit downstream riskset leakage controls")
    parser.add_argument("--riskset-dir", type=Path, default=Path("data/demo/risksets"))
    parser.add_argument("--pattern", type=str, default="idp_riskset_*_K4_samples.csv")
    parser.add_argument("--disease-onset-csv", type=Path, default=Path("data/demo/disease_onsets.csv"))
    parser.add_argument("--basic-csv", type=Path, default=Path("data/demo/basic_info.csv"))
    parser.add_argument("--labels-csv", type=Path, default=Path("Delphi/data/demo/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo/audit"))
    parser.add_argument("--diseases", type=str, default="", help="Optional comma-separated ICD codes, e.g. N18,E66")
    return parser.parse_args()


def _year_fraction(values: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(values, errors="coerce")
    return dt.dt.year.to_numpy(dtype=np.float64) + (dt.dt.dayofyear.to_numpy(dtype=np.float64) - 1.0) / 365.25


def _load_birth_years(path: Path) -> pd.DataFrame:
    basic = pd.read_csv(path)
    birth_col = None
    for candidate in ["birth_year"]:
        if candidate in basic.columns:
            birth_col = candidate
            break
    if birth_col is None:
        raise ValueError("basic_csv must contain birth year column (expected 'birth_year')")
    return basic[["eid", birth_col]].rename(columns={birth_col: "birth_year"})


def _disease_from_path(path: Path) -> str:
    name = path.name
    prefix = "idp_riskset_"
    suffix = "_K4_samples.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)].upper()
    return path.stem.upper()


def _summarize_one(
    *,
    path: Path,
    disease: str,
    onset: pd.DataFrame,
    birth: pd.DataFrame,
    labels: set[str],
) -> Dict[str, Any]:
    rs = pd.read_csv(path)
    if "disease" in rs.columns:
        disease_values = sorted(str(value).upper() for value in rs["disease"].dropna().unique().tolist())
        if len(disease_values) == 1:
            disease = disease_values[0]

    if disease not in onset.columns:
        raise ValueError(f"{disease} is not present in disease_onset_csv")

    data = rs.merge(birth, on="eid", how="left")
    disease_onset = onset[["eid", disease]].rename(columns={disease: "onset_age"})
    data = data.merge(disease_onset, on="eid", how="left")
    data["onset_age"] = pd.to_numeric(data["onset_age"], errors="coerce")
    data["birth_year"] = pd.to_numeric(data["birth_year"], errors="coerce")
    data["onset_year"] = data["onset_age"] + data["birth_year"]
    data["imaging_year"] = _year_fraction(data["imaging_date"])
    data["case_time_num"] = pd.to_numeric(data.get("case_time", np.nan), errors="coerce")
    data["delta_time_num"] = pd.to_numeric(data.get("delta_time", np.nan), errors="coerce")

    label = data["label"].astype(int)
    pos = label == 1
    neg = label == 0
    finite_onset = np.isfinite(data["onset_year"].to_numpy(dtype=np.float64))

    onset_year = data["onset_year"].to_numpy(dtype=np.float64)
    imaging_year = data["imaging_year"].to_numpy(dtype=np.float64)
    case_time = data["case_time_num"].to_numpy(dtype=np.float64)

    pos_prevalent = pos.to_numpy() & finite_onset & (onset_year <= imaging_year)
    pos_incident = pos.to_numpy() & finite_onset & (onset_year > imaging_year)
    pos_missing_onset = pos.to_numpy() & ~finite_onset

    neg_pre_imaging = neg.to_numpy() & finite_onset & (onset_year <= imaging_year)
    neg_before_case_time = neg.to_numpy() & finite_onset & np.isfinite(case_time) & (onset_year <= case_time)

    target_in_vocab = disease in labels
    target_pre_imaging_token = finite_onset & (onset_year <= imaging_year) & target_in_vocab

    dt = data.loc[pos, "delta_time_num"].dropna()
    return {
        "disease": disease,
        "riskset_csv": str(path),
        "n_rows": int(len(data)),
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
        "target_in_delphi_vocab": bool(target_in_vocab),
        "pos_incident": int(pos_incident.sum()),
        "pos_prevalent_or_at_scan": int(pos_prevalent.sum()),
        "pos_missing_onset": int(pos_missing_onset.sum()),
        "neg_target_before_imaging": int(neg_pre_imaging.sum()),
        "neg_target_before_or_at_case_time": int(neg_before_case_time.sum()),
        "pre_imaging_target_token_rows": int(target_pre_imaging_token.sum()),
        "pre_imaging_target_token_pos": int((target_pre_imaging_token & pos.to_numpy()).sum()),
        "pre_imaging_target_token_neg": int((target_pre_imaging_token & neg.to_numpy()).sum()),
        "delta_time_min": float(dt.min()) if len(dt) else float("nan"),
        "delta_time_median": float(dt.median()) if len(dt) else float("nan"),
        "delta_time_max": float(dt.max()) if len(dt) else float("nan"),
        "passes_incident_positive_check": bool(pos_prevalent.sum() == 0 and pos_missing_onset.sum() == 0),
        "passes_control_riskset_check": bool(neg_before_case_time.sum() == 0),
        "passes_pre_imaging_target_token_check": bool(target_pre_imaging_token.sum() == 0),
    }


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    n = len(df)
    lines: List[str] = [
        "# Downstream Riskset Leakage Audit",
        "",
        f"Audited diseases: `{n}`",
        "",
        "## Summary",
        "",
        f"- Incident positive check failures: `{int((~df['passes_incident_positive_check']).sum())}`",
        f"- Control riskset check failures: `{int((~df['passes_control_riskset_check']).sum())}`",
        f"- Pre-imaging target-token check failures: `{int((~df['passes_pre_imaging_target_token_check']).sum())}`",
        f"- Total pre-imaging target-token rows: `{int(df['pre_imaging_target_token_rows'].sum())}`",
        "",
        "## Per-Disease Audit",
        "",
        "| disease | rows | pos | pos prevalent | neg before case time | target token rows | pass incident | pass controls | pass target-token |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for _, row in df.sort_values("disease").iterrows():
        lines.append(
            "| {disease} | {n_rows} | {n_pos} | {pos_prevalent_or_at_scan} | "
            "{neg_target_before_or_at_case_time} | {pre_imaging_target_token_rows} | "
            "{passes_incident_positive_check} | {passes_control_riskset_check} | "
            "{passes_pre_imaging_target_token_check} |".format(**row.to_dict())
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    selected = {item.strip().upper() for item in args.diseases.split(",") if item.strip()}

    labels_df = pd.read_csv(args.labels_csv)
    label_col = labels_df.columns[0]
    labels = set(labels_df[label_col].astype(str).tolist())
    onset = pd.read_csv(args.disease_onset_csv)
    birth = _load_birth_years(args.basic_csv)

    rows: List[Dict[str, Any]] = []
    for path in sorted(args.riskset_dir.glob(args.pattern)):
        disease = _disease_from_path(path)
        if selected and disease not in selected:
            continue
        rows.append(_summarize_one(path=path, disease=disease, onset=onset, birth=birth, labels=labels))

    if not rows:
        raise SystemExit("No risksets matched the requested inputs")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("disease")
    csv_path = out_dir / "riskset_leakage_audit.csv"
    md_path = out_dir / "riskset_leakage_audit.md"
    df.to_csv(csv_path, index=False)
    _write_markdown(df, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
