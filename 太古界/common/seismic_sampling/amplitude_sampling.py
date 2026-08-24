#!/usr/bin/env python3
"""Streaming OBN amplitude sampling utilities for the 太古界 Step5-9 chain.

Design:
  * Trace headers (TraceIdx <-> X/Y) come from the common trace contract.
  * Amplitude is read from the OBN SEG-Y in contiguous blocks (mmap), never
    loading the full volume into memory.
  * Time resampling: 1 ms source -> arbitrary target times (linear).

Usage:
  from amplitude_sampling import ObnAmplitudeSampler
  sampler = ObnAmplitudeSampler(segy, trace_header_csv)
  amp = sampler.sample_at_xy(x, y, times_ms)        # per-row sampling
  block = sampler.read_2ms_block(trace_indices, samples_2ms)

CLI smoke test:
  python amplitude_sampling.py --segy ... --trace-header-csv ... --output-dir ...
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
from scipy.spatial import cKDTree


NULL_THRESHOLD = -1.0e6


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return []
    boundaries = np.where(np.diff(values) != 1)[0] + 1
    chunks = np.split(values, boundaries)
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks]


class ObnAmplitudeSampler:
    """Streaming amplitude sampler over the OBN SEG-Y."""

    def __init__(self, segy_path: Path, trace_header_csv: Path):
        self.segy_path = str(segy_path.resolve())
        self.handle = segyio.open(self.segy_path, "r", ignore_geometry=True)
        self.handle.mmap()
        self.source_samples = np.asarray(self.handle.samples, dtype=np.float64)
        if len(self.source_samples) < 2 or not np.allclose(
            np.diff(self.source_samples), np.diff(self.source_samples)[0], atol=1.0e-6
        ):
            raise ValueError("source sample axis is not regular")
        self.source_dt_ms = float(np.median(np.diff(self.source_samples)))
        self.trace_df = pd.read_csv(trace_header_csv, encoding="utf-8-sig")
        for column in ("TraceIdx", "X", "Y"):
            self.trace_df[column] = pd.to_numeric(self.trace_df[column], errors="coerce")
        self.trace_df = self.trace_df.dropna(subset=["TraceIdx", "X", "Y"]).drop_duplicates("TraceIdx").copy()
        self.trace_df["TraceIdx"] = self.trace_df["TraceIdx"].astype(np.int64)
        self.xy = self.trace_df[["X", "Y"]].to_numpy(dtype=np.float64)
        self.tree = cKDTree(self.xy)
        self.trace_index_lookup = self.trace_df.set_index("TraceIdx")["X"].to_dict()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "ObnAmplitudeSampler":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def nearest_trace(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
        distance, index = self.tree.query(points, k=1, p=1)
        return (
            self.trace_df["TraceIdx"].to_numpy(dtype=np.int64)[np.asarray(index, dtype=np.int64)],
            np.asarray(distance, dtype=np.float64),
        )

    def read_trace_matrix(self, trace_indices: np.ndarray) -> np.ndarray:
        """Read raw 1 ms amplitude matrix for (possibly non-contiguous) trace indices."""
        indices = np.asarray(trace_indices, dtype=np.int64)
        parts = [
            np.asarray(self.handle.trace.raw[start:stop], dtype=np.float32)
            for start, stop in contiguous_runs(indices)
        ]
        if not parts:
            return np.empty((0, len(self.source_samples)), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    @staticmethod
    def resample_1ms(matrix: np.ndarray, source_samples: np.ndarray, target_samples: np.ndarray) -> np.ndarray:
        """Linear resample each trace (axis 1) from source 1 ms axis to target times."""
        source = np.asarray(source_samples, dtype=np.float64)
        dt = float(np.median(np.diff(source)))
        positions = (np.asarray(target_samples, dtype=np.float64) - source[0]) / dt
        left = np.floor(positions).astype(np.int64)
        alpha = (positions - left).astype(np.float32)
        valid = (left >= 0) & (left < len(source) - 1)
        output = np.full((matrix.shape[0], len(target_samples)), np.nan, dtype=np.float32)
        if valid.any():
            left_valid = left[valid]
            a = alpha[valid][None, :]
            v0 = matrix[:, left_valid]
            v1 = matrix[:, left_valid + 1]
            output[:, valid] = v0 * (1.0 - a) + v1 * a
        return output

    def sample_at_trace(self, trace_idx: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
        """Amplitude sampled at arbitrary times (ms) for given source trace indices."""
        unique_idx, inverse = np.unique(np.asarray(trace_idx, dtype=np.int64), return_inverse=True)
        matrix = self.read_trace_matrix(unique_idx)
        matrix[(~np.isfinite(matrix)) | (matrix <= NULL_THRESHOLD)] = np.nan
        resampled = self.resample_1ms(matrix, self.source_samples, np.asarray(times_ms, dtype=np.float64))
        return resampled[inverse, np.arange(len(inverse))] if len(inverse) else np.empty(0, dtype=np.float32)

    def sample_at_xy(self, x: np.ndarray, y: np.ndarray, times_ms: np.ndarray) -> dict[str, np.ndarray]:
        """Per-row nearest-trace amplitude sampling at given X/Y/TIME."""
        trace_idx, distance = self.nearest_trace(np.asarray(x), np.asarray(y))
        amplitude = self.sample_at_trace(trace_idx, np.asarray(times_ms))
        return {"TraceIdx": trace_idx, "NearestTraceDistM": distance, "SeisAmp": amplitude}

    def read_2ms_block(self, trace_indices: np.ndarray, samples_2ms: np.ndarray) -> np.ndarray:
        """Raw 1 ms block resampled to a 2 ms axis (used by Step6A prediction)."""
        matrix = self.read_trace_matrix(np.asarray(trace_indices, dtype=np.int64))
        matrix[(~np.isfinite(matrix)) | (matrix <= NULL_THRESHOLD)] = np.nan
        return self.resample_1ms(matrix, self.source_samples, np.asarray(samples_2ms, dtype=np.float64))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the OBN amplitude sampler.")
    parser.add_argument("--segy", type=Path, required=True)
    parser.add_argument("--trace-header-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "amplitude_sampling_smoke_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke summary: {summary_path}")
    started = time.time()
    with ObnAmplitudeSampler(args.segy, args.trace_header_csv) as sampler:
        rng = np.random.default_rng(42)
        probe = np.sort(rng.choice(np.arange(10_000, 200_000, dtype=np.int64), size=64, replace=False))
        block = sampler.read_trace_matrix(probe)
        direct = np.stack([np.asarray(sampler.handle.trace[int(i)], dtype=np.float32) for i in probe])
        block_equals_direct = bool(np.allclose(block, direct, equal_nan=True))

        target = np.arange(2250.0, 3450.0 + 1.0e-6, 2.0)
        resampled = sampler.resample_1ms(block, sampler.source_samples, target)
        ref = np.empty((len(probe), len(target)), dtype=np.float32)
        for row in range(len(probe)):
            ref[row] = np.interp(target, sampler.source_samples, block[row])
        resample_ok = bool(np.allclose(resampled, ref, atol=1.0e-5, equal_nan=True))

        sample_x = sampler.trace_df["X"].to_numpy()[:4]
        sample_y = sampler.trace_df["Y"].to_numpy()[:4]
        sample_t = np.array([2500.0, 2564.0, 2632.0, 2720.0])
        out = sampler.sample_at_xy(sample_x, sample_y, sample_t)
        nearest_ok = bool(np.all(out["NearestTraceDistM"] < 1.0e-6))
        amplitude_finite_count = int(np.isfinite(out["SeisAmp"]).sum())

    checks = {
        "block_read_matches_direct": block_equals_direct,
        "resample_matches_np_interp": resample_ok,
        "nearest_trace_exact_for_header_xy": nearest_ok,
        "sampled_amplitudes_mostly_finite": amplitude_finite_count >= 3,
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "segy": str(args.segy.resolve()),
        "trace_header_csv": str(args.trace_header_csv.resolve()),
        "source_axis": {
            "time_min_ms": float(sampler.source_samples[0]),
            "time_max_ms": float(sampler.source_samples[-1]),
            "sample_interval_ms": sampler.source_dt_ms,
        },
        "smoke": {
            "probe_trace_count": int(len(probe)),
            "probe_trace_index_range": [int(probe.min()), int(probe.max())],
            "block_equals_direct": block_equals_direct,
            "resample_ok": resample_ok,
            "nearest_trace_ok": nearest_ok,
            "sample_finite_count": amplitude_finite_count,
        },
        "output_paths": {"smoke_summary_json": str(summary_path)},
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
