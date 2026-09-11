#!/usr/bin/env python3
"""Streaming unified-TraceIdx sampler for 太古界 seismic attributes.

The three attribute SEG-Y files share the project attribute-grid TraceIdx;
this is deliberately independent of the OBN trace order.  X/Y is used once
to locate that reference TraceIdx, then all three attributes are read using
the same index and interpolated in absolute TWT milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from scipy.spatial import cKDTree


ATTRIBUTE_NAMES = ("Coherence", "AntTrack", "CurvatureMax")
REFERENCE_ATTRIBUTE = "CurvatureMax"


def _header_xy(handle: Any) -> np.ndarray:
    """Read source X/Y with SEG-Y scalar handling."""
    count = len(handle.trace)
    xy = np.empty((count, 2), dtype=np.float64)
    scalar_field = segyio.TraceField.SourceGroupScalar
    for index in range(count):
        header = handle.header[index]
        scalar = float(header[scalar_field])
        factor = 1.0 if scalar == 0 else (scalar if scalar > 0 else 1.0 / abs(scalar))
        xy[index, 0] = float(header[segyio.TraceField.SourceX]) * factor
        xy[index, 1] = float(header[segyio.TraceField.SourceY]) * factor
    return xy


class SegyAttributeSampler:
    """Nearest-trace, linear-time sampler for one attribute SEG-Y."""

    def __init__(
        self,
        name: str,
        segy_path: Path,
        invalid_le: float | None = None,
        xy: np.ndarray | None = None,
        time_origin_ms: float | None = None,
    ):
        self.name = str(name)
        self.path = Path(segy_path).resolve()
        self.invalid_le = invalid_le
        self.handle = segyio.open(str(self.path), "r", ignore_geometry=True)
        self.handle.mmap()
        self.raw_samples = np.asarray(self.handle.samples, dtype=np.float64)
        # Attribute files use different stored axes.  The project contract is
        # absolute TWT, so expose every volume on the configured TWT origin.
        origin = float(self.raw_samples[0]) if time_origin_ms is None else float(time_origin_ms)
        self.samples = self.raw_samples - self.raw_samples[0] + origin
        if len(self.samples) < 2 or not np.allclose(np.diff(self.samples), np.diff(self.samples)[0], atol=1.0e-6):
            raise ValueError(f"{self.name}: source time axis is not regular")
        self.xy = _header_xy(self.handle) if xy is None else np.asarray(xy, dtype=np.float64)
        if len(self.xy) != len(self.handle.trace):
            raise ValueError(f"{self.name}: shared XY count does not match trace count")
        self.tree = cKDTree(self.xy)
        self.time_interval_ms = float(np.median(np.diff(self.samples)))

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "SegyAttributeSampler":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def nearest_trace(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.tree.query(np.column_stack([x, y]), k=1, p=1)
        return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)

    def _clean(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).copy()
        result[~np.isfinite(result)] = np.nan
        if self.invalid_le is not None:
            result[result <= float(self.invalid_le)] = np.nan
        return result

    def sample_at_trace(self, trace_indices: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
        traces = np.asarray(trace_indices, dtype=np.int64)
        times = np.asarray(times_ms, dtype=np.float64)
        out = np.full(len(traces), np.nan, dtype=np.float32)
        if len(traces) != len(times):
            raise ValueError("trace_indices and times_ms must have equal length")
        for trace_index in np.unique(traces):
            mask = traces == trace_index
            values = self._clean(np.asarray(self.handle.trace[int(trace_index)], dtype=np.float32))
            local_times = times[mask]
            valid = np.isfinite(local_times) & (local_times >= self.samples[0]) & (local_times <= self.samples[-1])
            if valid.any():
                out_indices = np.flatnonzero(mask)[valid]
                out[out_indices] = np.interp(local_times[valid], self.samples, values, left=np.nan, right=np.nan)
        return out

    def sample_at_xy(self, x: np.ndarray, y: np.ndarray, times_ms: np.ndarray) -> dict[str, np.ndarray]:
        trace_idx, distance = self.nearest_trace(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return {
            "TraceIdx": trace_idx,
            "NearestTraceDistM": distance,
            self.name: self.sample_at_trace(trace_idx, np.asarray(times_ms, dtype=np.float64)),
        }

    def read_block_at_xy(self, x: np.ndarray, y: np.ndarray, target_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trace_idx, distance = self.nearest_trace(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        unique, inverse = np.unique(trace_idx, return_inverse=True)
        matrix = np.stack([self._clean(np.asarray(self.handle.trace[int(index)], dtype=np.float32)) for index in unique])
        target = np.asarray(target_samples, dtype=np.float64)
        output = np.full((len(unique), len(target)), np.nan, dtype=np.float32)
        for row in range(len(unique)):
            output[row] = np.interp(target, self.samples, matrix[row], left=np.nan, right=np.nan)
        return output[inverse], trace_idx, distance


class MultiAttributeSampler:
    """Coordinated sampler for the three post-stack attributes."""

    def __init__(
        self,
        volume_paths: dict[str, str],
        invalid_rules: dict[str, dict[str, float]] | None = None,
        time_origins_ms: dict[str, float] | None = None,
    ):
        invalid_rules = invalid_rules or {}
        time_origins_ms = time_origins_ms or {}
        first_name = REFERENCE_ATTRIBUTE
        # AntTrack=-1 is a valid low-fracture response, not a SEG-Y null.
        # Reject stale caller configurations explicitly instead of silently
        # converting physical low values into missing data.
        ant_rule = invalid_rules.get("AntTrack", {})
        if ant_rule.get("invalid_le") is not None:
            raise ValueError(
                "AntTrack invalid_le is forbidden: -1 is valid low fracture evidence; "
                "remove this stale invalid-value rule."
            )
        first = SegyAttributeSampler(
            first_name,
            Path(volume_paths[first_name]),
            invalid_le=invalid_rules.get(first_name, {}).get("invalid_le"),
            time_origin_ms=time_origins_ms.get(first_name),
        )
        self.samplers = {first_name: first}
        for name in ATTRIBUTE_NAMES:
            if name == first_name:
                continue
            self.samplers[name] = SegyAttributeSampler(
                name,
                Path(volume_paths[name]),
                invalid_le=invalid_rules.get(name, {}).get("invalid_le"),
                xy=first.xy,
                time_origin_ms=time_origins_ms.get(name),
            )

    def close(self) -> None:
        for sampler in self.samplers.values():
            sampler.close()

    def __enter__(self) -> "MultiAttributeSampler":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def sample_at_xy(self, x: np.ndarray, y: np.ndarray, times_ms: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        times = np.asarray(times_ms, dtype=np.float64)
        trace_idx, distance = self.samplers[REFERENCE_ATTRIBUTE].nearest_trace(x, y)
        output: dict[str, np.ndarray] = {
            "TraceIdx": trace_idx,
            "NearestTraceDistM": distance,
        }
        for name, sampler in self.samplers.items():
            output[name] = sampler.sample_at_trace(trace_idx, times)
            # Kept temporarily for downstream compatibility; all are equal
            # by construction under the unified TraceIdx contract.
            output[f"{name}TraceIdx"] = trace_idx
            output[f"{name}NearestTraceDistM"] = distance
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            name: {
                "path": str(sampler.path),
                "trace_count": int(len(sampler.xy)),
                "time_min_ms": float(sampler.samples[0]),
                "time_max_ms": float(sampler.samples[-1]),
                "raw_time_min_ms": float(sampler.raw_samples[0]),
                "sample_interval_ms": sampler.time_interval_ms,
                "invalid_le": sampler.invalid_le,
            }
            for name, sampler in self.samplers.items()
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
