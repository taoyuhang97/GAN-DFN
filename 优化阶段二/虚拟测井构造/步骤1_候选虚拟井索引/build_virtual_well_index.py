from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.append(str(WORKFLOW_ROOT))

from common.config import NEIGHBOR_CONFIG, PREDICTION_ROOT, STEP1_DIR, TRACE_HEADER_CSV
from common.io_utils import read_csv_flexible, write_csv_utf8
from common.well_utils import discover_available_prediction_wells


def build_grid(trace_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_coords = np.sort(pd.to_numeric(trace_df["X"], errors="coerce").dropna().unique())
    y_coords = np.sort(pd.to_numeric(trace_df["Y"], errors="coerce").dropna().unique())
    return x_coords, y_coords


def nearest_idx(coords: np.ndarray, value: float) -> int:
    idx = int(np.searchsorted(coords, value))
    if idx <= 0:
        return 0
    if idx >= len(coords):
        return len(coords) - 1
    return idx - 1 if abs(value - coords[idx - 1]) <= abs(value - coords[idx]) else idx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=STEP1_DIR / "virtual_well_index.csv")
    args = parser.parse_args()

    trace_df = read_csv_flexible(TRACE_HEADER_CSV)
    trace_df["X"] = pd.to_numeric(trace_df["X"], errors="coerce")
    trace_df["Y"] = pd.to_numeric(trace_df["Y"], errors="coerce")
    trace_df = trace_df.dropna(subset=["X", "Y"]).copy()

    wells_df = discover_available_prediction_wells(PREDICTION_ROOT)
    x_coords, y_coords = build_grid(trace_df)

    rows: list[dict[str, object]] = []
    half = NEIGHBOR_CONFIG.xy_window_size // 2
    for well in wells_df.to_dict(orient="records"):
        if pd.isna(well["X"]) or pd.isna(well["Y"]):
            continue
        x0 = float(well["X"])
        y0 = float(well["Y"])
        xi = nearest_idx(x_coords, x0)
        yi = nearest_idx(y_coords, y0)
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                xx = min(max(xi + dx, 0), len(x_coords) - 1)
                yy = min(max(yi + dy, 0), len(y_coords) - 1)
                virtual_x = float(x_coords[xx])
                virtual_y = float(y_coords[yy])
                is_source = int(dx == 0 and dy == 0)
                distance = float(((virtual_x - x0) ** 2 + (virtual_y - y0) ** 2) ** 0.5)
                rows.append(
                    {
                        "SourceWellName": well["SourceWellName"],
                        "VirtualWellName": f'{well["SourceWellName"]}_VW_{dx + half}_{dy + half}',
                        "SourceX": x0,
                        "SourceY": y0,
                        "VirtualX": virtual_x,
                        "VirtualY": virtual_y,
                        "GridOffsetX": dx,
                        "GridOffsetY": dy,
                        "DistanceToSource": distance,
                        "IsSourceTrace": is_source,
                        "SourceResultDir": well["ResultDir"],
                        "SourceFinalLogCsv": well["FinalLogCsv"],
                        "SourceFinalPointsCsv": well["FinalPointsCsv"],
                        "SourceFinalSegmentsCsv": well["FinalSegmentsCsv"],
                    }
                )

    out_df = pd.DataFrame(rows)
    write_csv_utf8(out_df, args.output_csv)
    print(args.output_csv)
    print(f"rows={len(out_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
