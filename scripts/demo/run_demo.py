#!/usr/bin/env python3
"""Run the existing research entrypoints on newly generated synthetic data."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(args, cwd=ROOT):
    print("\n> " + " ".join(map(str, args)), flush=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-imputation", action="store_true", help="Also run the optional imputation extension")
    args = parser.parse_args()
    python = sys.executable
    run([python, "scripts/demo/generate_synthetic.py"])
    run([python, "Delphi/data/prepare_cn4dp.py", "--input-csv", "data/demo/disease_onsets.csv",
         "--exclude-csv", "data/demo/phenotypes_all.csv", "--output-dir", "Delphi/data/demo", "--val-ratio", "0.2"])
    run([python, "train.py", "config/train_demo.py"], cwd=ROOT / "Delphi")
    run([python, "scripts/distill/train_align.py", "--config", "configs/demo/distill.yaml"])
    for script, config in [
        ("train_phenotype.py", "idp_scratch"), ("train_phenotype.py", "idp_pretrained"),
        ("train_delphi_trajectory_head.py", "trajectory"),
        ("train_fusion_delphi_phenotype.py", "fusion_concat"),
        ("train_fusion_delphi_phenotype.py", "fusion_xattn"),
    ]:
        run([python, f"scripts/downstream/{script}", "--config", f"configs/demo/{config}.yaml"])
    run([python, "scripts/downstream/eval_delphi_generative_baseline.py", "--config", "configs/demo/generative.yaml"])
    if args.with_imputation:
        run([python, "scripts/imputation/train_idp_imputation.py", "--config", "configs/demo/imputation.yaml"])
    print("\nSynthetic demo completed. Local outputs: runs/demo/")


if __name__ == "__main__":
    main()
