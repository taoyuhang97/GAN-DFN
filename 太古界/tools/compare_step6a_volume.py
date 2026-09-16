#!/usr/bin/env python3
"""Step6A 密度体逐样点等价性判定（P0-4 验收用）。

用于决定 Step6B/6C/7B/7C 是否可以跳过：只要新旧体的有效样点最大绝对差为 0，
就说明上游改动（ID 列/回接逻辑）没有影响三维预测结果。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import segyio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Step6A density volumes sample by sample.")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--chunk-traces", type=int, default=4000)
    return parser.parse_args()


def compare(old_path: Path, new_path: Path, chunk: int) -> dict:
    with segyio.open(str(old_path), "r", ignore_geometry=True) as old, segyio.open(
        str(new_path), "r", ignore_geometry=True
    ) as new:
        if old.tracecount != new.tracecount:
            return {
                "identical": False,
                "reason": "trace_count_mismatch",
                "old_trace_count": int(old.tracecount),
                "new_trace_count": int(new.tracecount),
            }
        if old.samples.size != new.samples.size:
            return {
                "identical": False,
                "reason": "sample_count_mismatch",
                "old_sample_count": int(old.samples.size),
                "new_sample_count": int(new.samples.size),
            }
        max_diff = 0.0
        differing_traces = 0
        differing_samples = 0
        both_finite = 0
        nan_mismatch = 0
        for start in range(0, old.tracecount, chunk):
            stop = min(start + chunk, old.tracecount)
            a = old.trace.raw[start:stop].astype(np.float64)
            b = new.trace.raw[start:stop].astype(np.float64)
            finite_a = np.isfinite(a)
            finite_b = np.isfinite(b)
            nan_mismatch += int(np.count_nonzero(finite_a != finite_b))
            both = finite_a & finite_b
            both_finite += int(np.count_nonzero(both))
            diff = np.zeros_like(a)
            diff[both] = np.abs(a[both] - b[both])
            trace_max = diff.max(axis=1) if diff.size else np.zeros(0)
            differing_traces += int(np.count_nonzero(trace_max > 1.0e-9))
            differing_samples += int(np.count_nonzero(diff > 1.0e-9))
            max_diff = max(max_diff, float(diff.max()) if diff.size else 0.0)
    return {
        "identical": bool(
            max_diff <= 1.0e-9 and nan_mismatch == 0 and differing_traces == 0
        ),
        "max_abs_diff": max_diff,
        "differing_trace_count": differing_traces,
        "differing_sample_count": differing_samples,
        "nan_pattern_mismatch_count": nan_mismatch,
        "finite_sample_compared": both_finite,
        "old_path": str(old_path),
        "new_path": str(new_path),
    }


def main() -> int:
    args = parse_args()
    report = compare(args.old, args.new, int(args.chunk_traces))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    # 退出码恒为 0：本工具只做判定，是否重跑下游由 runner 依据 JSON 决定。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
