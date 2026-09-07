#!/usr/bin/env python3
"""Build a compact per-trace horizon contract for the Taigu area.

The contract stores only TraceIdx and horizon/window quality fields.  Spatial
coordinates remain in the independent OBN trace-header table.
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
from tqdm import tqdm

SURFACES = {
    "Top": "Grid_上部复合层T-a-1.dat",
    "Mid": "Grid_太古界顶Art_1.dat",
    "Base": "Grid_风化壳底Art_d1-1.dat",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build compact Taigu per-trace horizon contract")
    p.add_argument("--trace-header-csv", type=Path, required=True)
    p.add_argument("--horizon-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-match-distance-m", type=float, default=10.0)
    p.add_argument("--min-upper-thickness-ms", type=float, default=1.0)
    p.add_argument("--min-crust-thickness-ms", type=float, default=1.0)
    p.add_argument("--ext-window-ms", type=float, default=50.0)
    p.add_argument("--replace-output", action="store_true")
    return p.parse_args()


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.loadtxt(path, dtype=np.float64)
    return table[:, 2], table[:, 3], table[:, 4]


def main() -> int:
    a = args(); started = time.time(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    npy = out / "horizon_trace_table.npy"; csv = out / "horizon_trace_table.csv"; summary = out / "horizon_contract_summary.json"
    for p in (npy, csv, summary):
        if p.exists() and not a.replace_output:
            raise FileExistsError(f"refusing to overwrite {p}")

    headers = pd.read_csv(a.trace_header_csv, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    headers["TraceIdx"] = pd.to_numeric(headers["TraceIdx"], errors="raise").astype(np.int64)
    headers["X"] = pd.to_numeric(headers["X"], errors="raise"); headers["Y"] = pd.to_numeric(headers["Y"], errors="raise")
    headers = headers.sort_values("TraceIdx").drop_duplicates("TraceIdx").reset_index(drop=True)
    trace_idx = headers.TraceIdx.to_numpy(np.int64); xy = headers[["X", "Y"]].to_numpy(np.float64)
    if not np.array_equal(trace_idx, np.arange(len(trace_idx), dtype=np.int64)):
        raise ValueError("TraceIdx must be continuous from zero for the compact contract")

    matched: dict[str, np.ndarray] = {}; distances: dict[str, np.ndarray] = {}
    for code, filename in tqdm(SURFACES.items(), desc="匹配三层层位", unit="surface"):
        xs, ys, ts = load_surface(a.horizon_dir / filename)
        tree = cKDTree(np.column_stack([xs, ys])); dist, pos = tree.query(xy, k=1, p=1)
        matched[code] = ts[np.asarray(pos, dtype=np.int64)]; distances[code] = np.asarray(dist, np.float64)

    top, mid, base = matched["Top"], matched["Mid"], matched["Base"]
    finite = np.isfinite(top) & np.isfinite(mid) & np.isfinite(base)
    near = (distances["Top"] <= a.max_match_distance_m) & (distances["Mid"] <= a.max_match_distance_m) & (distances["Base"] <= a.max_match_distance_m)
    order = finite & (top < mid) & (mid < base)
    upper = order & ((mid - top) >= a.min_upper_thickness_ms)
    crust = order & ((base - mid) >= a.min_crust_thickness_ms)
    surface_valid = near & order
    # Keep times for valid rows only; invalid rows are explicitly NaN.
    top = np.where(surface_valid, top, np.nan); mid = np.where(surface_valid, mid, np.nan); base = np.where(surface_valid, base, np.nan)
    upper_present = surface_valid & ((mid - top) >= a.min_upper_thickness_ms)
    crust_present = surface_valid & ((base - mid) >= a.min_crust_thickness_ms)
    ext_present = surface_valid & np.isfinite(base)
    correction = np.zeros(len(trace_idx), dtype=np.uint8)
    correction[~near & finite] = 4       # source horizon is too far from the trace
    correction[near & finite & ~order] = 5  # local order reversal
    correction[near & ~finite] = 3        # missing source horizon
    confidence = np.where(surface_valid, 1.0, 0.0).astype(np.float32)

    dtype = np.dtype([("TraceIdx", "<i8"), ("TopTimeMs", "<f4"), ("MidTimeMs", "<f4"), ("BaseTimeMs", "<f4"),
                      ("SurfaceOrderValid", "u1"), ("UpperPresent", "u1"), ("CrustPresent", "u1"), ("ExtPresent", "u1"),
                      ("HorizonCorrectionCode", "u1"), ("HorizonConfidence", "<f4")])
    table = np.empty(len(trace_idx), dtype=dtype); table["TraceIdx"] = trace_idx; table["TopTimeMs"] = top.astype(np.float32); table["MidTimeMs"] = mid.astype(np.float32); table["BaseTimeMs"] = base.astype(np.float32)
    table["SurfaceOrderValid"] = surface_valid.astype(np.uint8); table["UpperPresent"] = upper_present.astype(np.uint8); table["CrustPresent"] = crust_present.astype(np.uint8); table["ExtPresent"] = ext_present.astype(np.uint8); table["HorizonCorrectionCode"] = correction; table["HorizonConfidence"] = confidence
    np.save(npy, table)
    pd.DataFrame.from_records(table).to_csv(csv, index=False, encoding="utf-8-sig")
    counts = {str(int(k)): int(v) for k, v in zip(*np.unique(correction, return_counts=True))}
    checks = {"trace_count_matches_header": len(table) == len(headers), "trace_idx_contiguous": True, "valid_order_ok": bool(np.all((top[surface_valid] < mid[surface_valid]) & (mid[surface_valid] < base[surface_valid]))), "compact_fields_only": set(table.dtype.names) == set(dtype.names)}
    payload: dict[str, Any] = {"status": "pass" if all(checks.values()) else "fail", "trace_header_csv": str(a.trace_header_csv.resolve()), "horizon_dir": str(a.horizon_dir.resolve()), "output_paths": {"npy": str(npy), "csv": str(csv), "summary": str(summary)}, "trace_count": int(len(table)), "surface_valid_trace_count": int(surface_valid.sum()), "surface_valid_fraction": float(surface_valid.mean()), "upper_present_trace_count": int(upper_present.sum()), "crust_present_trace_count": int(crust_present.sum()), "ext_present_trace_count": int(ext_present.sum()), "correction_code_counts": counts, "checks": checks, "parameters": {"max_match_distance_m": a.max_match_distance_m, "min_upper_thickness_ms": a.min_upper_thickness_ms, "min_crust_thickness_ms": a.min_crust_thickness_ms, "ext_window_ms": a.ext_window_ms}, "elapsed_seconds": time.time() - started}
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
