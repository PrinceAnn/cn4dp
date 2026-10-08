#!/usr/bin/env python3
"""Generate toy data from random draws, without reading any external dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.build_idp_riskset import build_risksets

ORGANS = ["brain", "skeleton", "heart", "abdomen", "lung"]
DISEASES = ["SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_E", "SYN_TARGET"]


def generate(output: Path, seed=42, teacher_subjects=128, distill_subjects=96, downstream_subjects=240):
    if min(teacher_subjects, distill_subjects, downstream_subjects) < 32:
        raise ValueError("Each cohort needs at least 32 synthetic subjects")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count = teacher_subjects + distill_subjects + downstream_subjects
    eids = np.arange(1, count + 1)
    birth_years = rng.integers(1950, 1986, count)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 365, count), unit="D")
    imaging_years = dates.year.to_numpy() + (dates.dayofyear.to_numpy() - 1) / 365.25
    ages = imaging_years - birth_years
    latent = rng.normal(size=(count, len(ORGANS)))
    info = pd.DataFrame({"eid": eids, "birth_year": birth_years, "imaging_date": dates.strftime("%Y-%m-%d")})
    onset = pd.DataFrame({"eid": eids})
    for i, code in enumerate(DISEASES[:-1]):
        probability = 1 / (1 + np.exp(-latent[:, i]))
        present = rng.random(count) < probability
        if i == 0:
            present[:] = True  # Every teacher subject has at least one historical event.
        onset[code] = np.where(present, ages - rng.uniform(2 + i, 12 + i, count), np.nan)
    risk_score = latent[:, 0] + 0.6 * latent[:, 2] + rng.normal(scale=0.7, size=count)
    has_target = risk_score > 0
    onset["SYN_TARGET"] = np.where(has_target, ages + rng.uniform(0.5, 5, count), np.nan)
    pheno = pd.DataFrame({"eid": eids})
    for i, organ in enumerate(ORGANS):
        for j in range(4):
            pheno[f"{organ}__feature_{j + 1:03d}"] = latent[:, i] + rng.normal(scale=0.3 + 0.1 * j, size=count)
    distill_start, downstream_start = teacher_subjects, teacher_subjects + distill_subjects
    imaging = pheno.iloc[distill_start:].copy()
    imaging.to_csv(output / "phenotypes_all.csv", index=False)
    pheno.iloc[distill_start:downstream_start].to_csv(output / "phenotypes_distill.csv", index=False)
    pheno.iloc[downstream_start:].to_csv(output / "phenotypes_downstream.csv", index=False)
    info.to_csv(output / "basic_info.csv", index=False)
    onset.to_csv(output / "disease_onsets.csv", index=False)
    downstream_ids = eids[downstream_start:]
    shuffled = rng.permutation(downstream_ids)
    n_train, n_val = int(0.6 * len(shuffled)), int(0.2 * len(shuffled))
    assignment = pd.DataFrame({"eid": shuffled, "split": ["train"] * n_train + ["val"] * n_val +
                               ["test"] * (len(shuffled) - n_train - n_val)})
    assignment.to_csv(output / "downstream_splits.csv", index=False)
    risksets = build_risksets(idp=output / "phenotypes_downstream.csv", basic=output / "basic_info.csv",
                             disease=output / "disease_onsets.csv", disease_cols=["SYN_TARGET"],
                             split_csv=output / "downstream_splits.csv", k=2, seed=seed)
    risksets.to_csv(output / "riskset.csv", index=False)
    info.loc[info.eid.isin(eids[distill_start:downstream_start]), ["eid", "imaging_date"]].to_csv(
        output / "imaging_cohort.csv", index=False)
    manifest = {"synthetic": True, "seed": seed, "source": "Independent random draws; no external data inputs",
                "cohorts": {"teacher": teacher_subjects, "distill": distill_subjects, "downstream": downstream_subjects},
                "organ_features": {organ: 4 for organ in ORGANS}, "disease_codes": DISEASES,
                "riskset_rows": len(risksets)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {count} artificial subjects and {len(risksets)} risk-set rows in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-subjects", type=int, default=128)
    parser.add_argument("--distill-subjects", type=int, default=96)
    parser.add_argument("--downstream-subjects", type=int, default=240)
    args = parser.parse_args()
    generate(args.output_dir, args.seed, args.teacher_subjects, args.distill_subjects, args.downstream_subjects)


if __name__ == "__main__":
    main()
