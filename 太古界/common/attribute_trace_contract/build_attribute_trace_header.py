#!/usr/bin/env python3
"""Build and validate the attribute-cube trace contract.

CurvatureMax is the reference ordering.  Coherence and AntTrack are checked
against it by trace count and by the sampled (and, by default, full) XY order.
The OBN header is intentionally not an input: it is an external coverage QC
dataset and no longer defines the Step5-9 spatial grid.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build attribute trace header contract")
    p.add_argument("--curvature", type=Path, required=True)
    p.add_argument("--coherence", type=Path, required=True)
    p.add_argument("--anttrack", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-check", type=int, default=0, help="0=full XY order check")
    p.add_argument("--xy-tolerance-m", type=float, default=2.0)
    p.add_argument("--replace-output", action="store_true")
    return p.parse_args()


def read_headers(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, object]]:
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        f.mmap()
        n = int(f.tracecount)
        rows = np.empty((n, 5), dtype=np.float64)
        for i in tqdm(range(n), desc=f"读取{label}道头", unit="trace"):
            h = f.header[i]
            rows[i] = (i, h[segyio.TraceField.INLINE_3D], h[segyio.TraceField.CROSSLINE_3D],
                       h[segyio.TraceField.SourceX], h[segyio.TraceField.SourceY])
        meta = {"trace_count": n, "sample_count": int(f.bin[segyio.BinField.Samples]),
                "sample_interval_us": int(f.bin[segyio.BinField.Interval])}
    return pd.DataFrame(rows, columns=["TraceIdx", "Inline3D", "Crossline3D", "X", "Y"]), meta


def compare(reference: pd.DataFrame, candidate: pd.DataFrame, sample_check: int, xy_tolerance_m: float) -> dict[str, object]:
    n = min(len(reference), len(candidate))
    if sample_check > 0 and n > sample_check:
        idx = np.unique(np.linspace(0, n - 1, sample_check, dtype=np.int64))
    else:
        idx = np.arange(n, dtype=np.int64)
    ref = reference[["X", "Y"]].to_numpy(float)
    cur = candidate[["X", "Y"]].to_numpy(float)
    delta = np.linalg.norm(ref[idx] - cur[idx], axis=1) if len(idx) else np.empty(0, float)
    mismatch = delta > xy_tolerance_m
    return {"trace_count": int(len(candidate)), "same_trace_count": bool(len(candidate) == len(reference)),
            "checked_trace_count": int(len(idx)), "same_header_order_checked": bool(not mismatch.any()),
            "header_order_mismatch_count": int(mismatch.sum()),
            "xy_tolerance_m": float(xy_tolerance_m),
            "xy_offset_max_m": float(delta.max()) if len(delta) else 0.0,
            "xy_offset_median_m": float(np.median(delta)) if len(delta) else 0.0}


def main() -> int:
    a = parse_args(); started = time.time(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    outputs = [out / "attribute_trace_header.csv", out / "attribute_trace_header.npy", out / "attribute_trace_summary.json"]
    if any(p.exists() for p in outputs) and not a.replace_output:
        raise FileExistsError("attribute trace contract exists; use a new version or --replace-output")
    curvature, cur_meta = read_headers(a.curvature, "曲率体")
    coherence, coh_meta = read_headers(a.coherence, "相干体")
    anttrack, ant_meta = read_headers(a.anttrack, "蚂蚁体")
    checks = {"Coherence": compare(curvature, coherence, a.sample_check, a.xy_tolerance_m),
              "AntTrack": compare(curvature, anttrack, a.sample_check, a.xy_tolerance_m)}
    # The contract is deliberately the compact spatial identity used downstream.
    contract = curvature.copy()
    contract["TraceIdx"] = contract["TraceIdx"].astype(np.int64)
    contract["Inline3D"] = contract["Inline3D"].astype(np.int32)
    contract["Crossline3D"] = contract["Crossline3D"].astype(np.int32)
    contract.to_csv(outputs[0], index=False, encoding="utf-8-sig")
    np.save(outputs[1], contract.to_records(index=False))
    payload = {"status": "pass" if all(v["same_trace_count"] and v["same_header_order_checked"] for v in checks.values()) else "fail",
               "reference": "CurvatureMax", "contract_fields": list(contract.columns),
               "inputs": {"CurvatureMax": str(a.curvature.resolve()), "Coherence": str(a.coherence.resolve()), "AntTrack": str(a.anttrack.resolve())},
               "volumes": {"CurvatureMax": cur_meta, "Coherence": coh_meta, "AntTrack": ant_meta},
               "checks_against_curvature": checks, "trace_count": int(len(contract)),
               "bounds": {k: [float(contract[k].min()), float(contract[k].max())] for k in ("X", "Y")},
               "outputs": {"csv": str(outputs[0]), "npy": str(outputs[1]), "summary": str(outputs[2])},
               "parameters": {"sample_check": int(a.sample_check), "xy_tolerance_m": float(a.xy_tolerance_m)}, "elapsed_seconds": time.time() - started}
    outputs[2].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
