from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import read_csv_flexible


def discover_available_prediction_wells(prediction_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for well_dir in sorted(prediction_root.glob("stage2_*")):
        if not well_dir.is_dir():
            continue
        log_csv = well_dir / "final_log_with_fractures.csv"
        points_csv = well_dir / "final_fracture_points.csv"
        segments_csv = well_dir / "final_fracture_segments.csv"
        if not log_csv.exists():
            continue
        well_name = well_dir.name
        well_name = well_name.replace("stage2_t6_boundary_", "").replace("stage2_new_t4t7_", "")
        try:
            log_df = read_csv_flexible(log_csv, nrows=5)
        except Exception:
            continue
        x_val = pd.to_numeric(log_df.get("X"), errors="coerce").dropna()
        y_val = pd.to_numeric(log_df.get("Y"), errors="coerce").dropna()
        rows.append(
            {
                "SourceWellName": well_name,
                "ResultDir": str(well_dir),
                "FinalLogCsv": str(log_csv),
                "FinalPointsCsv": str(points_csv) if points_csv.exists() else "",
                "FinalSegmentsCsv": str(segments_csv) if segments_csv.exists() else "",
                "X": float(x_val.iloc[0]) if not x_val.empty else None,
                "Y": float(y_val.iloc[0]) if not y_val.empty else None,
            }
        )
    return pd.DataFrame(rows)
