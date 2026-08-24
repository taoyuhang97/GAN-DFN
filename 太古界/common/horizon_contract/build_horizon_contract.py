#!/usr/bin/env python3
"""Build the per-trace three-surface horizon contract for the 太古界 demo area.

Surfaces (five-column files: inline, xline, X, Y, TIME[ms]):
  Top   = Grid_上部复合层T-a-1.dat   (top of 上部复合层)
  Mid   = Grid_太古界顶Art_1.dat     (太古界顶, boundary to 太古界风化壳)
  Base  = Grid_风化壳底Art_d1-1.dat  (base of 太古界风化壳)

Windows per trace:
  main window : Top -> Base
    upper layer: Top -> Mid     (上部复合层)
    crust layer: Mid -> Base    (太古界风化壳)
  ext window  : Base -> Base + 50 ms   (下伏 50 ms 延伸窗, WindowCode=ext50)

Outputs (output_dir):
  demo_horizon_contract.csv     per-trace Top/Mid/Base + validity
  horizon_windows_2ms.npz       compact 2 ms window start/stop indices
  surface_lookup_cache.npz      full-mine surface X/Y/T for arbitrary queries
  horizon_contract_summary.json status=pass
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SURFACE_FILES = {
    "Top": "Grid_上部复合层T-a-1.dat",
    "Mid": "Grid_太古界顶Art_1.dat",
    "Base": "Grid_风化壳底Art_d1-1.dat",
}
EXTENSION_WINDOW_MS = 50.0
MAX_MATCH_DISTANCE_M = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-trace three-surface horizon contract for demo area.")
    parser.add_argument("--horizon-dir", type=Path, required=True)
    parser.add_argument("--demo-grid-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ext-window-ms", type=float, default=EXTENSION_WINDOW_MS)
    parser.add_argument("--replace-output", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = np.loadtxt(path, dtype=np.float64)
    return table[:, 0], table[:, 1], table[:, 2], table[:, 3], table[:, 4]


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_csv = output_dir / "demo_horizon_contract.csv"
    windows_npz = output_dir / "horizon_windows_2ms.npz"
    cache_npz = output_dir / "surface_lookup_cache.npz"
    summary_path = output_dir / "horizon_contract_summary.json"
    for path in (contract_csv, windows_npz, cache_npz, summary_path):
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing horizon contract output: {path}")

    grid = pd.read_csv(args.demo_grid_csv, encoding="utf-8-sig")
    for column in ("TraceIdx", "X", "Y", "IX", "IY"):
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    grid = grid.dropna(subset=["TraceIdx", "X", "Y"]).reset_index(drop=True)
    grid_xy = grid[["X", "Y"]].to_numpy(dtype=np.float64)

    surface_tables: dict[str, dict[str, np.ndarray]] = {}
    surface_meta: dict[str, dict[str, Any]] = {}
    for code, filename in SURFACE_FILES.items():
        inline, xline, xs, ys, times = load_surface(args.horizon_dir / filename)
        surface_tables[code] = {"X": xs, "Y": ys, "T": times}
        surface_meta[code] = {
            "rows": int(len(xs)),
            "inline_range": [float(inline.min()), float(inline.max())],
            "xline_range": [float(xline.min()), float(xline.max())],
            "x_range": [float(xs.min()), float(xs.max())],
            "y_range": [float(ys.min()), float(ys.max())],
            "time_range_ms": [float(times.min()), float(times.max())],
        }

    matched: dict[str, np.ndarray] = {}
    match_dist: dict[str, np.ndarray] = {}
    for code, table in surface_tables.items():
        tree = cKDTree(np.column_stack([table["X"], table["Y"]]))
        distance, index = tree.query(grid_xy, k=1, p=1)
        matched[code] = table["T"][np.asarray(index, dtype=np.int64)]
        match_dist[code] = np.asarray(distance, dtype=np.float64)

    top = matched["Top"].copy()
    mid = matched["Mid"].copy()
    base = matched["Base"].copy()
    surface_valid = (
        (match_dist["Top"] <= MAX_MATCH_DISTANCE_M)
        & (match_dist["Mid"] <= MAX_MATCH_DISTANCE_M)
        & (match_dist["Base"] <= MAX_MATCH_DISTANCE_M)
        & (top < mid)
        & (mid < base)
    )
    order_reversal_mask = ~(
        (match_dist["Top"] <= MAX_MATCH_DISTANCE_M)
        & (match_dist["Mid"] <= MAX_MATCH_DISTANCE_M)
        & (match_dist["Base"] <= MAX_MATCH_DISTANCE_M)
        & (top < mid)
        & (mid < base)
    ) & (
        (match_dist["Top"] <= MAX_MATCH_DISTANCE_M)
        & (match_dist["Mid"] <= MAX_MATCH_DISTANCE_M)
        & (match_dist["Base"] <= MAX_MATCH_DISTANCE_M)
    )
    top = np.where(surface_valid, top, np.nan)
    mid = np.where(surface_valid, mid, np.nan)
    base = np.where(surface_valid, base, np.nan)

    valid_top = top[surface_valid]
    valid_base_plus = (base + args.ext_window_ms)[surface_valid]
    sample_interval_ms = 2.0
    axis_start = float(np.floor(valid_top.min() / sample_interval_ms) * sample_interval_ms)
    axis_stop = float(np.ceil(valid_base_plus.max() / sample_interval_ms) * sample_interval_ms)
    sample_axis = np.arange(axis_start, axis_stop + sample_interval_ms * 0.5, sample_interval_ms, dtype=np.float64)
    times_2d = sample_axis[None, :]

    def window_indices(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        start = np.searchsorted(sample_axis, lo, side="left").astype(np.int32)
        stop = np.searchsorted(sample_axis, hi, side="right").astype(np.int32)
        start = np.clip(start, 0, len(sample_axis))
        stop = np.clip(stop, 0, len(sample_axis))
        return start, stop

    upper_start, upper_stop = window_indices(
        np.where(surface_valid, top, -np.inf), np.where(surface_valid, mid, -np.inf)
    )
    crust_start, crust_stop = window_indices(
        np.where(surface_valid, mid, -np.inf), np.where(surface_valid, base, -np.inf)
    )
    ext_start, ext_stop = window_indices(
        np.where(surface_valid, base, -np.inf), np.where(surface_valid, base + args.ext_window_ms, -np.inf)
    )
    upper_start = np.where(surface_valid, upper_start, 0).astype(np.int32)
    upper_stop = np.where(surface_valid, upper_stop, 0).astype(np.int32)
    crust_start = np.where(surface_valid, crust_start, 0).astype(np.int32)
    crust_stop = np.where(surface_valid, crust_stop, 0).astype(np.int32)
    ext_start = np.where(surface_valid, ext_start, 0).astype(np.int32)
    ext_stop = np.where(surface_valid, ext_stop, 0).astype(np.int32)
    present = {
        "upper": (surface_valid & (upper_stop > upper_start)).astype(np.uint8),
        "crust": (surface_valid & (crust_stop > crust_start)).astype(np.uint8),
        "ext": (surface_valid & (ext_stop > ext_start)).astype(np.uint8),
        "main": (surface_valid & ((upper_stop > upper_start) | (crust_stop > crust_start))).astype(np.uint8),
    }

    contract = pd.DataFrame(
        {
            "TraceIdx": grid["TraceIdx"].astype(np.int64),
            "X": grid["X"],
            "Y": grid["Y"],
            "IX": grid["IX"].astype(np.int32),
            "IY": grid["IY"].astype(np.int32),
            "TopTimeMs": top,
            "MidTimeMs": mid,
            "BaseTimeMs": base,
            "TopDistM": np.where(surface_valid, match_dist["Top"], np.nan),
            "MidDistM": np.where(surface_valid, match_dist["Mid"], np.nan),
            "BaseDistM": np.where(surface_valid, match_dist["Base"], np.nan),
            "SurfaceValid": surface_valid.astype(np.uint8),
            "UpperThicknessMs": np.where(surface_valid, mid - top, np.nan),
            "CrustThicknessMs": np.where(surface_valid, base - mid, np.nan),
        }
    )
    contract.to_csv(contract_csv, index=False, encoding="utf-8-sig")

    np.savez_compressed(
        windows_npz,
        sample_axis=sample_axis.astype(np.float32),
        sample_interval_ms=np.float32(sample_interval_ms),
        ext_window_ms=np.float32(args.ext_window_ms),
        trace_idx=grid["TraceIdx"].to_numpy(dtype=np.int64),
        surface_valid=surface_valid.astype(np.uint8),
        upper_start=upper_start,
        upper_stop=upper_stop,
        crust_start=crust_start,
        crust_stop=crust_stop,
        ext_start=ext_start,
        ext_stop=ext_stop,
        upper_present=present["upper"],
        crust_present=present["crust"],
        ext_present=present["ext"],
        main_present=present["main"],
    )
    np.savez_compressed(
        cache_npz,
        top_x=surface_tables["Top"]["X"],
        top_y=surface_tables["Top"]["Y"],
        top_t=surface_tables["Top"]["T"],
        mid_x=surface_tables["Mid"]["X"],
        mid_y=surface_tables["Mid"]["Y"],
        mid_t=surface_tables["Mid"]["T"],
        base_x=surface_tables["Base"]["X"],
        base_y=surface_tables["Base"]["Y"],
        base_t=surface_tables["Base"]["T"],
    )

    valid_count = int(surface_valid.sum())
    order_ok_valid = bool(
        (np.nanmedian(mid[surface_valid] - top[surface_valid]) > 0)
        and (np.nanmedian(base[surface_valid] - mid[surface_valid]) > 0)
    )
    checks = {
        "sample_interval_is_2ms": abs(sample_interval_ms - 2.0) < 1.0e-9,
        "ext_window_is_50ms": abs(args.ext_window_ms - 50.0) < 1.0e-9,
        "surface_coverage_ge_0p99": valid_count / len(grid) >= 0.99,
        "all_valid_traces_order_top_lt_mid_lt_base": order_ok_valid,
        "windows_nonempty_for_valid": bool(
            (present["upper"][surface_valid].mean() >= 0.999)
            and (present["crust"][surface_valid].mean() >= 0.999)
            and (present["ext"][surface_valid].mean() >= 0.999)
        ),
        "grid_rows_match": len(contract) == len(grid),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "horizon_dir": str(args.horizon_dir.resolve()),
        "demo_grid_csv": str(args.demo_grid_csv.resolve()),
        "surface_files": {code: str(args.horizon_dir / name) for code, name in SURFACE_FILES.items()},
        "surface_meta": surface_meta,
        "sample_axis": {
            "time_min_ms": float(sample_axis[0]),
            "time_max_ms": float(sample_axis[-1]),
            "sample_interval_ms": sample_interval_ms,
            "sample_count": int(len(sample_axis)),
        },
        "extension_window_ms": float(args.ext_window_ms),
        "coverage": {
            "trace_count": int(len(grid)),
            "surface_valid_trace_count": valid_count,
            "surface_valid_fraction": float(valid_count / len(grid)),
            "main_window_trace_count": int(present["main"].sum()),
            "upper_window_trace_count": int(present["upper"].sum()),
            "crust_window_trace_count": int(present["crust"].sum()),
            "ext_window_trace_count": int(present["ext"].sum()),
        },
        "known_issues": {
            "surface_invalid_trace_count": int((~surface_valid).sum()),
            "order_reversal_trace_count": int(order_reversal_mask.sum()),
            "note": "局部层序倒转（风化壳底/太古界顶/上部复合层顶交错）道标记 SurfaceValid=0，下游预测置 NaN，不参与训练",
        },
        "window_stats": {
            "upper_thickness_ms": {"min": float(np.nanmin(contract["UpperThicknessMs"])), "max": float(np.nanmax(contract["UpperThicknessMs"])), "mean": float(np.nanmean(contract["UpperThicknessMs"]))},
            "crust_thickness_ms": {"min": float(np.nanmin(contract["CrustThicknessMs"])), "max": float(np.nanmax(contract["CrustThicknessMs"])), "mean": float(np.nanmean(contract["CrustThicknessMs"]))},
        },
        "output_paths": {
            "demo_horizon_contract_csv": str(contract_csv),
            "horizon_windows_2ms_npz": str(windows_npz),
            "surface_lookup_cache_npz": str(cache_npz),
            "summary_json": str(summary_path),
        },
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
