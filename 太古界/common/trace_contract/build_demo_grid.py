#!/usr/bin/env python3
"""Build the 5 km x 5 km demo-area trace grid aligned to OBN trace headers.

Demo area (confirmed 2026-08-22): center (663000, 4240900)
  X in [660500, 665500], Y in [4238400, 4243400]

The grid is the rectangular set of OBN header traces whose X/Y fall inside the
box. Outputs:
  demo_grid.csv            TraceIdx,X,Y,IX,IY (IX = rank of X, IY = rank of Y)
  trace_mapping.npz        output_trace_index/source_trace_idx/x/y/ix/iy
  demo_grid_summary.json   status=pass when the box is a complete rectangle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build demo-area trace grid from OBN header map.")
    parser.add_argument("--trace-header-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-x", type=float, default=663000.0)
    parser.add_argument("--center-y", type=float, default=4240900.0)
    parser.add_argument("--half-size-m", type=float, default=2500.0)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_csv = output_dir / "demo_grid.csv"
    mapping_npz = output_dir / "trace_mapping.npz"
    summary_path = output_dir / "demo_grid_summary.json"
    for path in (grid_csv, mapping_npz, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing demo grid output: {path}")

    trace_df = pd.read_csv(args.trace_header_csv, encoding="utf-8-sig")
    for column in ("TraceIdx", "X", "Y"):
        trace_df[column] = pd.to_numeric(trace_df[column], errors="coerce")
    trace_df = trace_df.dropna(subset=["TraceIdx", "X", "Y"]).drop_duplicates("TraceIdx").copy()
    trace_df["TraceIdx"] = trace_df["TraceIdx"].astype(np.int64)

    x_min = args.center_x - args.half_size_m
    x_max = args.center_x + args.half_size_m
    y_min = args.center_y - args.half_size_m
    y_max = args.center_y + args.half_size_m
    mask = (
        trace_df["X"].between(x_min, x_max)
        & trace_df["Y"].between(y_min, y_max)
    )
    work = trace_df.loc[mask, ["TraceIdx", "X", "Y"]].copy()
    if work.empty:
        raise RuntimeError("demo box contains no trace-header rows")

    x_values = np.sort(work["X"].unique())
    y_values = np.sort(work["Y"].unique())
    x_rank = {float(value): idx for idx, value in enumerate(x_values)}
    y_rank = {float(value): idx for idx, value in enumerate(y_values)}
    work["IX"] = work["X"].map(x_rank).astype(np.int32)
    work["IY"] = work["Y"].map(y_rank).astype(np.int32)
    work = work.sort_values("TraceIdx").reset_index(drop=True)

    expected_count = int(len(x_values) * len(y_values))
    checks = {
        "box_is_full_rectangle": int(len(work)) == expected_count,
        "no_duplicate_grid_cells": not work[["IX", "IY"]].duplicated().any(),
        "x_line_count_is_401": int(len(x_values)) == 401,
        "y_line_count_is_401": int(len(y_values)) == 401,
        "half_size_is_5000m": abs(2 * args.half_size_m - 5000.0) < 1.0e-6,
    }
    status = "pass" if all(checks.values()) else "fail"
    work.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        mapping_npz,
        output_trace_index=np.arange(len(work), dtype=np.int32),
        source_trace_idx=work["TraceIdx"].to_numpy(dtype=np.int64),
        x=work["X"].to_numpy(dtype=np.float64),
        y=work["Y"].to_numpy(dtype=np.float64),
        ix=work["IX"].to_numpy(dtype=np.int32),
        iy=work["IY"].to_numpy(dtype=np.int32),
    )
    summary = {
        "status": status,
        "trace_header_csv": str(args.trace_header_csv.resolve()),
        "target_box": {
            "center_x": float(args.center_x),
            "center_y": float(args.center_y),
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
            "half_size_m": float(args.half_size_m),
        },
        "grid": {
            "trace_count": int(len(work)),
            "x_line_count": int(len(x_values)),
            "y_line_count": int(len(y_values)),
            "expected_full_rectangle_trace_count": expected_count,
            "missing_trace_count": int(expected_count - len(work)),
            "x_min": float(work["X"].min()),
            "x_max": float(work["X"].max()),
            "y_min": float(work["Y"].min()),
            "y_max": float(work["Y"].max()),
        },
        "output_paths": {
            "demo_grid_csv": str(grid_csv),
            "trace_mapping_npz": str(mapping_npz),
            "summary_json": str(summary_path),
        },
        "checks": checks,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
