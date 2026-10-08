#!/usr/bin/env python3
"""Sample incident cases and event-free controls using a generic cohort schema."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_risksets(*, idp, basic, disease, disease_cols, k=4, seed=42, min_pos=1, split_csv=None):
    if k < 1:
        raise ValueError("k must be positive")
    ids = pd.read_csv(idp, usecols=["eid"])
    info = pd.read_csv(basic, usecols=["eid", "birth_year", "imaging_date"])
    onsets = pd.read_csv(disease, usecols=["eid", *disease_cols])
    for table in [ids, info, onsets]:
        if table.eid.duplicated().any():
            raise ValueError("Input tables must have one row per subject")
    base = ids.merge(info, on="eid", validate="one_to_one").merge(onsets, on="eid", validate="one_to_one")
    dates = pd.to_datetime(base.imaging_date, errors="raise")
    base["imaging_year"] = dates.dt.year + (dates.dt.dayofyear - 1) / 365.25
    base["birth_year"] = pd.to_numeric(base.birth_year, errors="raise")
    if not np.isfinite(base.birth_year).all():
        raise ValueError("Every subject needs a finite birth year")
    if split_csv:
        splits = pd.read_csv(split_csv, usecols=["eid", "split"])
        if splits.eid.duplicated().any():
            raise ValueError("Split assignments must be unique")
        base = base.merge(splits, on="eid", how="left", validate="one_to_one")
        if not base.split.isin(["train", "val", "test"]).all():
            raise ValueError("Every subject needs a train/val/test assignment")
    else:
        base["split"] = "unspecified"
    rng = np.random.default_rng(seed)
    rows = []
    for split, subset in base.groupby("split", sort=True):
        for code in disease_cols:
            frame = subset.copy()
            frame["onset_year"] = pd.to_numeric(frame[code], errors="coerce") + frame.birth_year
            frame = frame[frame.onset_year.isna() | (frame.onset_year > frame.imaging_year)].reset_index(drop=True)
            cases = frame[frame.onset_year.notna()]
            if len(cases) < min_pos:
                continue
            for idx, case in cases.iterrows():
                eligible = frame[(frame.imaging_year < case.onset_year) &
                                 (frame.onset_year.isna() | (frame.onset_year > case.onset_year)) &
                                 (frame.eid != case.eid)]
                if eligible.empty:
                    continue
                # Fewer than k available subjects yield fewer controls, never duplicate draws.
                chosen = rng.choice(eligible.index.to_numpy(), size=min(k, len(eligible)), replace=False)
                common = {"disease": code, "case_time": float(case.onset_year), "split": split}
                rows.append({**common, "eid": int(case.eid), "label": 1, "matched_to": int(case.eid),
                             "imaging_date": case.imaging_date,
                             "delta_time": float(case.onset_year - case.imaging_year)})
                for control_idx in chosen:
                    control = frame.loc[control_idx]
                    rows.append({**common, "eid": int(control.eid), "label": 0, "matched_to": int(case.eid),
                                 "imaging_date": control.imaging_date, "delta_time": np.nan})
    if not rows:
        raise ValueError("No incident risk sets could be built")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idp", default="data/demo/phenotypes_downstream.csv")
    parser.add_argument("--basic", default="data/demo/basic_info.csv")
    parser.add_argument("--disease", default="data/demo/disease_onsets.csv")
    parser.add_argument("--disease-cols", required=True, help="Comma-separated onset column names")
    parser.add_argument("--split-csv", help="Subject assignments made before risk-set sampling")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pos", type=int, default=1)
    parser.add_argument("--out", default="data/demo/riskset.csv")
    args = parser.parse_args()
    table = build_risksets(idp=args.idp, basic=args.basic, disease=args.disease,
                          disease_cols=args.disease_cols.split(","), k=args.K, seed=args.seed,
                          min_pos=args.min_pos, split_csv=args.split_csv)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Wrote {len(table)} synthetic or locally supplied rows to {path}")


if __name__ == "__main__":
    main()
