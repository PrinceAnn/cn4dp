#!/usr/bin/env python3
"""为 IDP 补全任务构建通用的影像 cohort 文件。

这个脚本会从 basic_info 中为每个有 IDP 的受试者挑出首个可用影像日期，
不依赖特定疾病或 downstream risk-set 的定义。
输出结果就是第一阶段补全任务所需的 cohort 表。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a generic imaging cohort CSV for IDP imputation")
    parser.add_argument("--idp", default="data/demo/phenotypes_distill.csv")
    parser.add_argument("--basic", default="data/demo/basic_info.csv")
    parser.add_argument("--out", default="data/demo/imaging_cohort.csv")
    return parser.parse_args()


def _pick_first_nonnull_across_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for column in cols:
        if column not in df.columns:
            continue
        values = df[column]
        # 对每一行，依次选取候选列里第一个非空的影像日期。
        mask = out.isna() & values.notna() & (values.astype(str).str.len() > 0)
        out.loc[mask] = values.loc[mask]
    return out


def main() -> None:
    args = _parse_args()

    # 1. 读取 IDP 样本集合和 basic_info。
    idp = pd.read_csv(args.idp, usecols=["eid"])
    basic = pd.read_csv(args.basic)
    if "eid" not in basic.columns:
        raise SystemExit("basic file must contain 'eid'")

    if "imaging_date" not in basic.columns:
        raise SystemExit("basic file must contain 'imaging_date'")
    basic = basic.copy()
    basic["imaging_date"] = pd.to_datetime(basic["imaging_date"], errors="coerce")

    # 3. 只保留同时出现在 IDP 表里、且至少有一个合法影像日期的受试者。
    cohort = idp.merge(basic[["eid", "imaging_date"]], on="eid", how="inner")
    cohort = cohort[cohort["imaging_date"].notna()].drop_duplicates(subset=["eid"]).reset_index(drop=True)
    cohort["imaging_date"] = cohort["imaging_date"].dt.strftime("%Y-%m-%d")

    # 4. 输出给后续补全训练直接使用的 cohort 文件。
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out_path, index=False)
    print(f"Saved {len(cohort)} rows to {out_path}")


if __name__ == "__main__":
    main()
