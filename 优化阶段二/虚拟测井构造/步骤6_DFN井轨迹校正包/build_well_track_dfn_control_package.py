from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.append(str(WORKFLOW_ROOT))

from common.config import OUTPUT_ROOT, PREDICTION_ROOT
from common.io_utils import write_csv_utf8


STEP6_DIR = OUTPUT_ROOT / "步骤6_DFN井轨迹校正包"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--output-csv", type=Path, default=STEP6_DIR / "well_track_dfn_control_package.csv")
    parser.add_argument("--summary-csv", type=Path, default=STEP6_DIR / "well_track_dfn_control_summary.csv")
    args = parser.parse_args()

    rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for final_csv in sorted(args.prediction_root.rglob("final_log_with_fractures.csv")):
        run_dir = final_csv.parent
        run_name = run_dir.name
        well_name = run_name.replace("stage2_t6_boundary_", "") if run_name.startswith("stage2_t6_boundary_") else run_name
        try:
            df = pd.read_csv(final_csv, low_memory=False, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(final_csv, low_memory=False)

        out = pd.DataFrame()
        out["WellName"] = df["WellName"] if "WellName" in df.columns else well_name
        out["X"] = pd.to_numeric(df.get("X"), errors="coerce")
        out["Y"] = pd.to_numeric(df.get("Y"), errors="coerce")
        out["TIME"] = pd.to_numeric(df.get("TIME"), errors="coerce")
        out["Density"] = pd.to_numeric(df.get("PredDensityMassPerLength"), errors="coerce")
        out["HasFracture"] = pd.to_numeric(df.get("PredFractureFlag"), errors="coerce").fillna(0).astype("Int64")
        out["PredAzimuth"] = pd.to_numeric(df.get("PredAzimuth"), errors="coerce")
        out["PredDip"] = pd.to_numeric(df.get("PredDip"), errors="coerce")
        out["PredSegmentID"] = df.get("PredSegmentID")
        out["PredStrataName"] = df.get("PredStrataName")
        out["ControlPriority"] = out["HasFracture"].fillna(0).astype(float).map(lambda v: 2 if v > 0 else 1)
        out["ControlSource"] = "well_track_prediction"
        out["SourceResultDir"] = str(run_dir)
        for optional_col in [
            "SEIS_TRUE",
            "车西_相干体T4_T7_CENTER",
            "车西_蚂蚁体T4_T7_CENTER",
            "车西_最大曲率T4_T7_CENTER",
            "车西_最大正曲率T4_T7_CENTER",
        ]:
            if optional_col in df.columns:
                out[optional_col] = df[optional_col]

        out = out.dropna(subset=["X", "Y", "TIME"]).reset_index(drop=True)
        rows.append(out)
        summary_rows.append(
            {
                "WellName": well_name,
                "NumSamples": int(len(out)),
                "NumFractureSamples": int((out["HasFracture"].fillna(0).astype(float) > 0).sum()),
                "MinTIME": float(out["TIME"].min()) if len(out) else None,
                "MaxTIME": float(out["TIME"].max()) if len(out) else None,
            }
        )

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_csv_utf8(result, args.output-csv if False else args.output_csv)
    write_csv_utf8(pd.DataFrame(summary_rows), args.summary_csv)

    print(args.output_csv)
    print(args.summary_csv)
    print(f"rows={len(result)}")
    print(f"wells={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
