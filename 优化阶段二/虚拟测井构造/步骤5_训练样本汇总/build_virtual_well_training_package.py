from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.append(str(WORKFLOW_ROOT))

from common.config import OUTPUT_ROOT, STEP2_DIR, STEP3_DIR, STEP4_DIR
from common.io_utils import read_csv_flexible, write_csv_utf8


STEP5_DIR = OUTPUT_ROOT / "步骤5_训练样本汇总"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute-csv", type=Path, default=STEP2_DIR / "virtual_well_attributes.csv")
    parser.add_argument("--weaklabel-csv", type=Path, default=STEP3_DIR / "virtual_well_density_weaklabel.csv")
    parser.add_argument("--confidence-csv", type=Path, default=STEP4_DIR / "virtual_well_confidence.csv")
    parser.add_argument("--output-csv", type=Path, default=STEP5_DIR / "virtual_well_training_samples.csv")
    parser.add_argument("--summary-csv", type=Path, default=STEP5_DIR / "virtual_well_training_summary.csv")
    args = parser.parse_args()

    attr_df = read_csv_flexible(args.attribute_csv)
    weak_df = read_csv_flexible(args.weaklabel_csv)
    conf_df = read_csv_flexible(args.confidence_csv)

    key_cols = ["SourceWellName", "VirtualWellName", "VirtualX", "VirtualY", "TIME"]
    weak_keep = [
        "SourceWellName",
        "VirtualWellName",
        "VirtualX",
        "VirtualY",
        "TIME",
        "DistanceToSource",
        "DistanceDecayWeight",
        "FractureDensityWeak",
        "HasFractureWeak",
        "DensitySourceType",
    ]
    conf_keep = [
        "SourceWellName",
        "VirtualWellName",
        "VirtualX",
        "VirtualY",
        "TIME",
        "PointConfidence",
        "WellConfidence",
    ]

    merged = attr_df.merge(
        weak_df[[col for col in weak_keep if col in weak_df.columns]],
        on=[col for col in key_cols if col in attr_df.columns and col in weak_df.columns],
        how="outer",
    )
    merged = merged.merge(
        conf_df[[col for col in conf_keep if col in conf_df.columns]],
        on=[col for col in key_cols if col in merged.columns and col in conf_df.columns],
        how="left",
    )

    out = pd.DataFrame()
    out["SourceWellName"] = merged.get("SourceWellName")
    out["VirtualWellName"] = merged.get("VirtualWellName")
    out["X"] = pd.to_numeric(merged.get("VirtualX"), errors="coerce")
    out["Y"] = pd.to_numeric(merged.get("VirtualY"), errors="coerce")
    out["TIME"] = pd.to_numeric(merged.get("TIME"), errors="coerce")
    out["Density"] = pd.to_numeric(merged.get("FractureDensityWeak"), errors="coerce")
    out["HasFracture"] = pd.to_numeric(merged.get("HasFractureWeak"), errors="coerce").fillna(0).astype("Int64")
    out["PointConfidence"] = pd.to_numeric(merged.get("PointConfidence"), errors="coerce")
    out["WellConfidence"] = pd.to_numeric(merged.get("WellConfidence"), errors="coerce")
    out["DistanceToSource"] = pd.to_numeric(merged.get("DistanceToSource"), errors="coerce")
    out["DistanceDecayWeight"] = pd.to_numeric(merged.get("DistanceDecayWeight"), errors="coerce")
    out["DensitySourceType"] = merged.get("DensitySourceType")

    for optional_col in [
        "TVD",
        "SEIS_TRUE",
        "COHERENCE",
        "ANT_TRACK",
        "CURVATURE_MAX",
        "CURVATURE_POS",
        "FRACTURE_INV",
        "SEIS_TRUE_WIN_MEAN",
        "SEIS_TRUE_WIN_STD",
        "SEIS_TRUE_WIN_MIN",
        "SEIS_TRUE_WIN_MAX",
        "SEIS_TRUE_WIN_VALID_COUNT",
        "COHERENCE_WIN_MEAN",
        "COHERENCE_WIN_STD",
        "COHERENCE_WIN_MIN",
        "COHERENCE_WIN_MAX",
        "COHERENCE_WIN_VALID_COUNT",
        "ANT_TRACK_WIN_MEAN",
        "ANT_TRACK_WIN_STD",
        "ANT_TRACK_WIN_MIN",
        "ANT_TRACK_WIN_MAX",
        "ANT_TRACK_WIN_VALID_COUNT",
        "CURVATURE_MAX_WIN_MEAN",
        "CURVATURE_MAX_WIN_STD",
        "CURVATURE_MAX_WIN_MIN",
        "CURVATURE_MAX_WIN_MAX",
        "CURVATURE_MAX_WIN_VALID_COUNT",
        "CURVATURE_POS_WIN_MEAN",
        "CURVATURE_POS_WIN_STD",
        "CURVATURE_POS_WIN_MIN",
        "CURVATURE_POS_WIN_MAX",
        "CURVATURE_POS_WIN_VALID_COUNT",
    ]:
        if optional_col in merged.columns:
            out[optional_col] = merged[optional_col]

    out = out.dropna(subset=["X", "Y", "TIME"]).sort_values(
        ["SourceWellName", "VirtualWellName", "TIME"]
    ).reset_index(drop=True)

    write_csv_utf8(out, args.output_csv)

    summary = (
        out.groupby(["SourceWellName", "VirtualWellName"], as_index=False)
        .agg(
            NumSamples=("TIME", "size"),
            NumFractureSamples=("HasFracture", "sum"),
            MeanDensity=("Density", "mean"),
            MaxDensity=("Density", "max"),
            MeanPointConfidence=("PointConfidence", "mean"),
            MeanWellConfidence=("WellConfidence", "mean"),
        )
    )
    write_csv_utf8(summary, args.summary_csv)

    print(args.output_csv)
    print(args.summary_csv)
    print(f"rows={len(out)}")
    print(f"virtual_wells={summary['VirtualWellName'].nunique() if not summary.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
