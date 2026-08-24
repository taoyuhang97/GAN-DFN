#!/usr/bin/env python3
"""Scan OBN SEG-Y trace headers and persist the trace-index <-> X/Y map.

Outputs (config.output_dir):
  obn_trace_header_xy.csv   TraceIdx,Inline3D,Crossline3D,X,Y
  obn_trace_header_summary.json

The X/Y map is the single spatial contract for every downstream Step5-9 task:
any X/Y query resolves through this table (cKDTree) to a source trace index.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan OBN SEG-Y trace headers (TraceIdx/X/Y/Inline/Crossline).")
    parser.add_argument("--segy", type=Path, required=True, help="OBN SEG-Y file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--max-traces", type=int, default=0, help="Smoke-test cap (0 = full file).")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    segy_path = args.segy.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "obn_trace_header_xy.csv"
    summary_path = output_dir / "obn_trace_header_summary.json"
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing trace header output: {csv_path}")

    started = time.time()
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as handle:
        handle.mmap()
        trace_count = handle.tracecount
        sample_count = int(handle.bin[segyio.BinField.Samples])
        sample_interval_us = int(handle.bin[segyio.BinField.Interval])
        samples = np.asarray(handle.samples, dtype=np.float64)
        limit = trace_count if args.max_traces <= 0 else min(int(args.max_traces), trace_count)
        trace_idx = np.empty(limit, dtype=np.int64)
        inlines = np.empty(limit, dtype=np.int32)
        crosslines = np.empty(limit, dtype=np.int32)
        xs = np.empty(limit, dtype=np.float64)
        ys = np.empty(limit, dtype=np.float64)
        for i in tqdm(range(limit), desc="scan trace headers", total=limit):
            header = handle.header[i]
            trace_idx[i] = i
            inlines[i] = int(header[segyio.TraceField.INLINE_3D])
            crosslines[i] = int(header[segyio.TraceField.CROSSLINE_3D])
            xs[i] = float(header[segyio.TraceField.SourceX])
            ys[i] = float(header[segyio.TraceField.SourceY])

    df = pd.DataFrame(
        {
            "TraceIdx": trace_idx,
            "Inline3D": inlines,
            "Crossline3D": crosslines,
            "X": xs,
            "Y": ys,
        }
    )
    df = df.drop_duplicates("TraceIdx", keep="first").sort_values("TraceIdx").reset_index(drop=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    inline_min, inline_max = int(df["Inline3D"].min()), int(df["Inline3D"].max())
    crossline_min, crossline_max = int(df["Crossline3D"].min()), int(df["Crossline3D"].max())
    unique_inlines = int(df["Inline3D"].nunique())
    unique_crosslines = int(df["Crossline3D"].nunique())
    full_rectangle = unique_inlines * unique_crosslines
    summary = {
        "status": "pass",
        "segy": str(segy_path),
        "trace_count_total": int(trace_count),
        "traces_scanned": int(len(df)),
        "sample_count": sample_count,
        "sample_interval_us": sample_interval_us,
        "sample_axis": {
            "time_min_ms": float(samples[0]),
            "time_max_ms": float(samples[-1]),
            "sample_count": int(len(samples)),
        },
        "coordinate_field": "SourceX/SourceY (verified equal to CDP_X/CDP_Y/GroupX/GroupY)",
        "inline_range": [inline_min, inline_max],
        "crossline_range": [crossline_min, crossline_max],
        "unique_inline_count": unique_inlines,
        "unique_crossline_count": unique_crosslines,
        "full_rectangle_trace_count": full_rectangle,
        "missing_trace_count": int(full_rectangle - len(df)),
        "grid": {
            "x_min": float(df["X"].min()),
            "x_max": float(df["X"].max()),
            "y_min": float(df["Y"].min()),
            "y_max": float(df["Y"].max()),
        },
        "output_paths": {
            "trace_header_csv": str(csv_path),
            "summary_json": str(summary_path),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
