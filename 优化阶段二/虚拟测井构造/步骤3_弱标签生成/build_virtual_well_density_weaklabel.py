from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.append(str(WORKFLOW_ROOT))

from common.config import STEP1_DIR, STEP3_DIR, WEAK_LABEL_CONFIG
from common.io_utils import read_csv_flexible, write_csv_utf8


def build_density_proxy(log_df: pd.DataFrame) -> pd.Series:
    if "PredDensityMassPerLength" in log_df.columns:
        return pd.to_numeric(log_df["PredDensityMassPerLength"], errors="coerce").fillna(0.0)
    if "FractureFlag" in log_df.columns:
        return pd.to_numeric(log_df["FractureFlag"], errors="coerce").fillna(0.0)
    if "PredFractureFlag" in log_df.columns:
        return pd.to_numeric(log_df["PredFractureFlag"], errors="coerce").fillna(0.0)
    if "PredProb" in log_df.columns:
        return pd.to_numeric(log_df["PredProb"], errors="coerce").fillna(0.0)
    point_cols = [col for col in log_df.columns if "fracture" in col.lower() and "point" in col.lower()]
    if point_cols:
        return pd.to_numeric(log_df[point_cols[0]], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=log_df.index)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", type=Path, default=STEP1_DIR / "virtual_well_index.csv")
    parser.add_argument("--output-csv", type=Path, default=STEP3_DIR / "virtual_well_density_weaklabel.csv")
    args = parser.parse_args()

    index_df = read_csv_flexible(args.index_csv)
    rows: list[pd.DataFrame] = []
    for item in index_df.to_dict(orient="records"):
        source_log_csv = Path(str(item["SourceFinalLogCsv"]))
        if not source_log_csv.exists():
            continue
        log_df = read_csv_flexible(source_log_csv)
        density = build_density_proxy(log_df)
        dist = float(item["DistanceToSource"])
        distance_weight = max(
            WEAK_LABEL_CONFIG.min_confidence,
            math.exp(-dist / WEAK_LABEL_CONFIG.distance_decay_length),
        )
        weak_density = density * distance_weight
        out = pd.DataFrame(
            {
                "SourceWellName": item["SourceWellName"],
                "VirtualWellName": item["VirtualWellName"],
                "VirtualX": item.get("VirtualX"),
                "VirtualY": item.get("VirtualY"),
                "TIME": log_df["TIME"] if "TIME" in log_df.columns else None,
                "DistanceToSource": dist,
                "DistanceDecayWeight": distance_weight,
                "FractureDensityWeak": weak_density,
                "HasFractureWeak": (weak_density > 0).astype(int),
                "DensitySourceType": "copy_and_distance_decay",
            }
        )
        rows.append(out)

    result_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_csv_utf8(result_df, args.output_csv)
    print(args.output_csv)
    print(f"rows={len(result_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
