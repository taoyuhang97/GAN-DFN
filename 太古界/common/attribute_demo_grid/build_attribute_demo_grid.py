#!/usr/bin/env python3
"""Build a Cartesian demo grid whose TraceIdx belongs to the attribute volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Taigu attribute-volume demo-grid contract.")
    parser.add_argument("--attribute-trace-header-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-x", type=float, default=663000.0)
    parser.add_argument("--center-y", type=float, default=4240900.0)
    parser.add_argument("--half-size-m", type=float, default=2500.0)
    parser.add_argument("--grid-spacing-m", type=float, default=12.5)
    parser.add_argument("--max-match-distance-m", type=float, default=2.0)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rounded_axis(start: float, stop: float, spacing: float) -> np.ndarray:
    count_float = (stop - start) / spacing
    count = int(round(count_float))
    if not np.isclose(count_float, count):
        raise ValueError("demo bounds must be evenly divisible by grid spacing")
    # Source headers store integer metres; use half-up rounding to reproduce the
    # confirmed 12.5 m Cartesian demo lattice without depending on OBN headers.
    return np.floor(start + np.arange(count + 1, dtype=np.float64) * spacing + 0.5)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "attribute_demo_grid.csv"
    mapping_path = output_dir / "attribute_demo_grid_mapping.npz"
    summary_path = output_dir / "attribute_demo_grid_summary.json"
    for path in (grid_path, mapping_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite attribute demo-grid output: {path}")

    header_path = args.attribute_trace_header_csv.resolve()
    header = pd.read_csv(header_path, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    for column in ("TraceIdx", "X", "Y"):
        header[column] = pd.to_numeric(header[column], errors="coerce")
    header = header.dropna().copy()
    header["TraceIdx"] = header["TraceIdx"].astype(np.int64)
    if header["TraceIdx"].duplicated().any():
        raise RuntimeError("attribute trace header contains duplicate TraceIdx")

    x_axis = rounded_axis(
        args.center_x - args.half_size_m,
        args.center_x + args.half_size_m,
        args.grid_spacing_m,
    )
    y_axis = rounded_axis(
        args.center_y - args.half_size_m,
        args.center_y + args.half_size_m,
        args.grid_spacing_m,
    )
    ix = np.repeat(np.arange(len(x_axis), dtype=np.int32), len(y_axis))
    iy = np.tile(np.arange(len(y_axis), dtype=np.int32), len(x_axis))
    target_x = x_axis[ix]
    target_y = y_axis[iy]

    tree = cKDTree(header[["X", "Y"]].to_numpy(dtype=np.float64))
    distance, position = tree.query(np.column_stack([target_x, target_y]), k=1)
    trace_idx = header.iloc[position]["TraceIdx"].to_numpy(dtype=np.int64)
    unique_matches = int(np.unique(trace_idx).size)
    checks = {
        "all_targets_matched_within_tolerance": bool(np.all(distance <= args.max_match_distance_m)),
        "attribute_trace_match_is_one_to_one": unique_matches == len(trace_idx),
        "grid_is_complete_rectangle": len(trace_idx) == len(x_axis) * len(y_axis),
        "grid_cells_are_unique": len(np.unique(np.column_stack([ix, iy]), axis=0)) == len(trace_idx),
    }
    status = "pass" if all(checks.values()) else "fail"
    if status != "pass":
        raise RuntimeError(
            "attribute demo-grid mapping failed: "
            f"max_distance={float(distance.max()):.3f}m, unique={unique_matches}/{len(trace_idx)}"
        )

    grid = pd.DataFrame({"TraceIdx": trace_idx, "X": target_x, "Y": target_y, "IX": ix, "IY": iy})
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        mapping_path,
        output_trace_index=np.arange(len(grid), dtype=np.int64),
        source_trace_idx=trace_idx,
        x=target_x,
        y=target_y,
        ix=ix,
        iy=iy,
    )
    summary = {
        "status": status,
        "contract": "attribute_trace_idx_only_no_obn_trace_idx",
        "attribute_trace_header_csv": str(header_path),
        "target_box": {
            "center_x": float(args.center_x),
            "center_y": float(args.center_y),
            "half_size_m": float(args.half_size_m),
            "grid_spacing_m": float(args.grid_spacing_m),
        },
        "grid": {
            "trace_count": int(len(grid)),
            "x_line_count": int(len(x_axis)),
            "y_line_count": int(len(y_axis)),
            "x_min": float(target_x.min()),
            "x_max": float(target_x.max()),
            "y_min": float(target_y.min()),
            "y_max": float(target_y.max()),
        },
        "attribute_match": {
            "max_allowed_distance_m": float(args.max_match_distance_m),
            "maximum_distance_m": float(distance.max()),
            "median_distance_m": float(np.median(distance)),
            "unique_attribute_trace_count": unique_matches,
        },
        "checks": checks,
        "outputs": {
            "grid_csv": str(grid_path),
            "mapping_npz": str(mapping_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
