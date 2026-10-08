#!/usr/bin/env python
"""Convert the CN4DP wide table to Delphi train/val .bin files.

Expected input format
---------------------
CSV with columns:
- ``eid`` (or another patient id column via ``--patient-id``)
- one column per ICD code containing first-onset age in **years** (NaN if never occurred)
- optional ``death`` column containing age at death in **years``.

Output
------
Creates ``train.bin``/``val.bin`` in Delphi format (np.uint32, columns: patient_id, age_days, token_id)
plus a ``labels.csv`` that aligns token ids with readable labels (Padding/No event are inserted first).

Usage
-----
python Delphi/data/prepare_cn4dp.py \
  --input-csv data/demo/disease_onsets.csv \
  --exclude-csv data/demo/phenotypes_all.csv \
  --basic-info-csv data/demo/basic_info.csv \
  --output-dir Delphi/data/demo \
  --val-ratio 0.1 --include-death
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CN4DP data for Delphi")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to the wide CSV file")
    parser.add_argument("--basic-info-csv", type=Path, default=None, help="Optional CSV with eid and sex (0=female, 1=male)")
    parser.add_argument("--exclude-csv", type=Path, default=None, help="Path to CSV with patient IDs to exclude")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Delphi/data/demo"), help="Where to write .bin and labels.csv"
    )
    parser.add_argument("--patient-id", type=str, default="eid", help="Patient identifier column name")
    parser.add_argument(
        "--death-column",
        type=str,
        default="death",
        help="Column with age at death (years). Ignored if missing in CSV or --include-death is False",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of patients for validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient split")
    parser.add_argument(
        "--include-death",
        action="store_true",
        help="If set, create a death token and include death age as an event when available",
    )
    return parser.parse_args()


def years_to_days(age_years: float) -> int:
    """Convert age in years to rounded integer days."""
    return int(round(float(age_years) * 365.25))


def build_token_map(disease_cols: List[str], include_death: bool, death_label: str) -> Tuple[Dict[str, int], List[str]]:
    tokens: Dict[str, int] = {}
    labels: List[str] = ["Padding", "No event"]
    for idx, col in enumerate(disease_cols, start=1):
        tokens[col] = idx
        labels.append(col)
    if include_death:
        death_token = len(tokens) + 1
        tokens[death_label] = death_token
        labels.append(death_label)
    return tokens, labels


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.patient_id not in df.columns:
        raise ValueError(f"Patient id column '{args.patient_id}' not found in {args.input_csv}")

    if args.exclude_csv is not None:
        exclude_df = pd.read_csv(args.exclude_csv)
        exclude_ids = set(exclude_df[args.patient_id])
        df = df[~df[args.patient_id].isin(exclude_ids)]

    if args.basic_info_csv is not None:
        basic_info_df = pd.read_csv(args.basic_info_csv)
        # now only use sex: sex
        basic_info_df = basic_info_df[[args.patient_id, 'sex']]
        basic_info_df['female'] = basic_info_df['sex'].apply(lambda x: 0.0 if x == 0 else pd.NA)
        basic_info_df['male'] = basic_info_df['sex'].apply(lambda x: 0.0 if x == 1 else pd.NA)
        df = df.merge(basic_info_df[[args.patient_id,'female', 'male']], on=args.patient_id, how='left')


    has_death_column = args.death_column in df.columns
    death_col = args.death_column if has_death_column and args.include_death else None
    disease_cols = [c for c in df.columns if c not in {args.patient_id, args.death_column}]
    if len(disease_cols) == 0:
        raise ValueError("No disease columns found after removing patient/death columns")

    tokens, labels = build_token_map(disease_cols, include_death=bool(death_col), death_label=args.death_column)

    patient_events: Dict[int, List[Tuple[int, int]]] = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing patients"):
        pid = int(row[args.patient_id])
        events: List[Tuple[int, int]] = []
        for col in disease_cols:
            age_years = row[col]
            if pd.notna(age_years) and float(age_years) >= 0:
                events.append((years_to_days(age_years), tokens[col]))
        if death_col:
            death_age = row[death_col]
            if pd.notna(death_age) and float(death_age) >= 0:
                events.append((years_to_days(death_age), tokens[death_col]))
        if not events:
            continue  # skip patients with no events at all
        events.sort(key=lambda x: (x[0], x[1]))
        patient_events[pid] = events

    patient_ids = list(patient_events.keys())
    if len(patient_ids) == 0:
        raise ValueError("No patients with events found after filtering")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(patient_ids)
    split = int(len(patient_ids) * (1 - args.val_ratio))
    train_ids = set(patient_ids[:split])
    val_ids = set(patient_ids[split:])
    if len(val_ids) == 0:
        raise ValueError("Validation set is empty; try increasing --val-ratio")

    def flatten(pid_set: set) -> np.ndarray:
        rows: List[Tuple[int, int, int]] = []
        for pid in sorted(pid_set):
            for age_days, token in patient_events[pid]:
                rows.append((pid, age_days, token))
        return np.array(rows, dtype=np.uint32)

    train_arr = flatten(train_ids)
    val_arr = flatten(val_ids)
    all_arr = np.vstack([train_arr, val_arr])

    (out_dir / "train.bin").write_bytes(train_arr.tobytes())
    (out_dir / "val.bin").write_bytes(val_arr.tobytes())
    (out_dir / "all.bin").write_bytes(all_arr.tobytes())

    pd.Series(labels).to_csv(out_dir / "labels.csv", index=False, header=False)

    meta = {
        "num_patients": len(patient_ids),
        "num_train_patients": len(train_ids),
        "num_val_patients": len(val_ids),
        "num_events_train": int(train_arr.shape[0]),
        "num_events_val": int(val_arr.shape[0]),
        "num_diseases": len(disease_cols),
        "include_death": bool(death_col),
        "vocab_size": max(tokens.values()) + 1,  # +1 for padding after shift in get_batch
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote train.bin ({train_arr.shape[0]} rows), val.bin ({val_arr.shape[0]} rows), and all.bin ({all_arr.shape[0]} rows) to {out_dir}")
    print(f"Vocab size (before +1 shift in training) = {max(tokens.values()) + 1}")


if __name__ == "__main__":
    main()
