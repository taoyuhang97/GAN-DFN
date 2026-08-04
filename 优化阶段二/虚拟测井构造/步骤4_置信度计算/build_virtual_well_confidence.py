from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.append(str(WORKFLOW_ROOT))

from common.config import STEP2_DIR, STEP3_DIR, STEP4_DIR
from common.io_utils import read_csv_flexible, write_csv_utf8


SIMILARITY_FEATURES = [
    "COHERENCE",
    "ANT_TRACK",
    "CURVATURE_MAX",
    "CURVATURE_POS",
    "SEIS_TRUE",
    "COHERENCE_WIN_MEAN",
    "ANT_TRACK_WIN_MEAN",
    "CURVATURE_MAX_WIN_MEAN",
    "CURVATURE_POS_WIN_MEAN",
]


def build_similarity_weight(attr_df: pd.DataFrame) -> pd.DataFrame:
    work = attr_df.copy()
    for col in SIMILARITY_FEATURES:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    source_ref = (
        work.groupby("VirtualWellName", sort=False)[SIMILARITY_FEATURES]
        .first()
        .rename(columns=lambda c: f"{c}_SRC")
        .reset_index()
    )
    work = work.merge(source_ref, on="VirtualWellName", how="left")

    score_cols: list[str] = []
    for col in SIMILARITY_FEATURES:
        if col not in work.columns or f"{col}_SRC" not in work.columns:
            continue
        src_col = f"{col}_SRC"
        denom = pd.concat(
            [
                work[col].abs(),
                work[src_col].abs(),
                pd.Series(1.0, index=work.index),
            ],
            axis=1,
        ).max(axis=1)
        diff = (work[col] - work[src_col]).abs() / denom
        score_col = f"{col}_SIM"
        work[score_col] = (1.0 - diff).clip(lower=0.0, upper=1.0)
        work.loc[work[col].isna() | work[src_col].isna(), score_col] = np.nan
        score_cols.append(score_col)

    if score_cols:
        work["SimilarityWeight"] = work[score_cols].mean(axis=1, skipna=True).fillna(1.0)
    else:
        work["SimilarityWeight"] = 1.0
    work["SimilarityWeight"] = work["SimilarityWeight"].clip(lower=0.05, upper=1.0)

    keep_cols = [
        col
        for col in ["VirtualWellName", "TIME", "SimilarityWeight"]
        if col in work.columns
    ]
    return work[keep_cols].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weaklabel-csv", type=Path, default=STEP3_DIR / "virtual_well_density_weaklabel.csv")
    parser.add_argument("--attribute-csv", type=Path, default=STEP2_DIR / "virtual_well_attributes.csv")
    parser.add_argument("--output-csv", type=Path, default=STEP4_DIR / "virtual_well_confidence.csv")
    args = parser.parse_args()

    weak_df = read_csv_flexible(args.weaklabel_csv)
    attr_df = read_csv_flexible(args.attribute_csv)
    if weak_df.empty:
        write_csv_utf8(pd.DataFrame(), args.output_csv)
        print(args.output_csv)
        print("rows=0")
        return 0

    weak_df = weak_df.copy()
    weak_df["TIME"] = pd.to_numeric(weak_df["TIME"], errors="coerce")
    weak_df["DistanceDecayWeight"] = pd.to_numeric(
        weak_df["DistanceDecayWeight"], errors="coerce"
    ).fillna(0.0)

    attr_df = attr_df.copy()
    attr_df["TIME"] = pd.to_numeric(attr_df["TIME"], errors="coerce")
    similarity_df = build_similarity_weight(attr_df)

    out = weak_df.merge(similarity_df, on=["VirtualWellName", "TIME"], how="left")
    out["SimilarityWeight"] = pd.to_numeric(out["SimilarityWeight"], errors="coerce").fillna(1.0)
    out["PointConfidence"] = (
        out["DistanceDecayWeight"] * out["SimilarityWeight"]
    ).clip(lower=0.0, upper=1.0)

    well_conf = (
        out.groupby("VirtualWellName", as_index=False)["PointConfidence"]
        .mean()
        .rename(columns={"PointConfidence": "WellConfidence"})
    )
    out = out.merge(well_conf, on="VirtualWellName", how="left")
    out = out[
        [
            "SourceWellName",
            "VirtualWellName",
            "VirtualX",
            "VirtualY",
            "TIME",
            "FractureDensityWeak",
            "HasFractureWeak",
            "DistanceToSource",
            "DistanceDecayWeight",
            "SimilarityWeight",
            "PointConfidence",
            "WellConfidence",
        ]
    ]
    write_csv_utf8(out, args.output_csv)
    print(args.output_csv)
    print(f"rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
