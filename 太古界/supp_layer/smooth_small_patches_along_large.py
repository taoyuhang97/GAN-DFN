#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smooth small-patch density along one large-fracture XZ trace.

The transformation preserves patch count, dip, dip azimuth, and lateral
offset from the selected large fracture.  It only redistributes patch centers
along the large-fracture tangent using a partial uniform-rank transform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from 太古界.common.orientation_frame import convention as orientation  # noqa: E402


VERSION = "large_wake_small_density_smoothing_v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Redistribute small DFN patches smoothly along a large fracture"
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--anchor-source-index", type=int, required=True)
    p.add_argument("--time-min-ms", type=float, required=True)
    p.add_argument("--time-max-ms", type=float, required=True)
    p.add_argument("--smoothing-strength", type=float, default=0.85)
    p.add_argument("--along-jitter-sigma-m", type=float, default=0.8)
    p.add_argument("--bin-width-ms", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20261122)
    return p


def protected_mask(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(False, index=frame.index)
    for column in (
        "IsImagingMatchPatch",
        "IsWellControlPatch",
        "IsImagingGroundTruth",
        "ImagingMatchEventLocked",
    ):
        if column in frame:
            result |= (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0)
                .ne(0)
            )
    return result


def trace_geometry(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    trace = orientation.trace_on_section(
        float(row["DipAzimuthDeg"]),
        float(row["DipDeg"]),
        "XZ",
    )
    if trace is None:
        raise RuntimeError("target large fracture has no XZ trace")
    length = float(np.clip(float(row["LengthM"]), 10.0, 500.0))
    tangent = np.array([float(trace[0]), 2.0 * float(trace[1])])
    tangent /= np.linalg.norm(tangent)
    center = np.array([float(row["CenterX"]), 2.0 * float(row["CenterTime"])])
    half = 0.5 * length
    endpoints = np.vstack([center - half * tangent, center + half * tangent])
    return endpoints, tangent


def bin_counts(
    times: np.ndarray,
    time_min_ms: float,
    time_max_ms: float,
    bin_width_ms: float,
) -> tuple[list[float], list[int]]:
    edges = np.arange(
        time_min_ms,
        time_max_ms + 0.5 * bin_width_ms,
        bin_width_ms,
    )
    counts = np.histogram(times, bins=edges)[0].astype(float)
    return edges.tolist(), counts.astype(int).tolist()


def main() -> int:
    args = parser().parse_args()
    if not (0.0 <= args.smoothing_strength <= 1.0):
        raise ValueError("--smoothing-strength must be in [0, 1]")
    if args.time_min_ms >= args.time_max_ms:
        raise ValueError("--time-min-ms must be smaller than --time-max-ms")
    if args.bin_width_ms <= 0.0:
        raise ValueError("--bin-width-ms must be positive")

    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    frame = pd.read_csv(input_csv, low_memory=False)
    required = {
        "SourceIndex",
        "PatchID",
        "FractureScale",
        "LargeAnchorID",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "DipAzimuthDeg",
        "DipDeg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"input CSV missing required columns: {missing}")
    for column in (
        "SourceIndex",
        "LargeAnchorID",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "DipAzimuthDeg",
        "DipDeg",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    anchors = frame[
        frame["SourceIndex"].eq(int(args.anchor_source_index))
        & frame["FractureScale"].astype(str).str.strip().str.lower().eq("large")
    ]
    if len(anchors) != 1:
        raise RuntimeError(
            "expected exactly one large-fracture anchor row, "
            f"found {len(anchors)} for SourceIndex={args.anchor_source_index}"
        )
    endpoints, tangent = trace_geometry(anchors.iloc[0])

    small = frame["FractureScale"].astype(str).str.strip().str.lower().eq("small")
    assigned = frame["LargeAnchorID"].eq(int(args.anchor_source_index))
    active_time = frame["CenterTime"].between(
        float(args.time_min_ms),
        float(args.time_max_ms),
        inclusive="both",
    )
    protected = protected_mask(frame)
    active_mask = small & assigned & active_time & ~protected
    active_positions = np.flatnonzero(active_mask.to_numpy())
    if len(active_positions) < 10:
        raise RuntimeError("not enough active small patches to smooth")

    points = np.column_stack(
        [
            frame.loc[active_positions, "CenterX"].to_numpy(dtype=float),
            2.0 * frame.loc[active_positions, "CenterTime"].to_numpy(dtype=float),
        ]
    )
    order = np.lexsort(
        (
            frame.loc[active_positions, "PatchID"].astype(str).to_numpy(),
            points[:, 1],
        )
    )
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    quantiles = (ranks + 0.5) / float(len(order))
    time_uniform = (
        float(args.time_min_ms)
        + quantiles
        * (float(args.time_max_ms) - float(args.time_min_ms))
    )
    time_before = points[:, 1] / 2.0
    rng = np.random.default_rng(int(args.seed))
    time_target = (
        (1.0 - float(args.smoothing_strength)) * time_before
        + float(args.smoothing_strength) * time_uniform
        + rng.normal(
            0.0,
            float(args.along_jitter_sigma_m) / 2.0,
            len(time_before),
        )
    ).clip(float(args.time_min_ms), float(args.time_max_ms))
    trace_times = endpoints[:, 1] / 2.0
    trace_x = endpoints[:, 0]
    horizontal_offset = (
        points[:, 0] - np.interp(time_before, trace_times, trace_x)
    )
    new_x = np.interp(time_target, trace_times, trace_x) + horizontal_offset
    before_times = time_before
    after_times = time_target
    shifts = (after_times - before_times) * 2.0 / max(
        float(abs(tangent[1])),
        1.0e-9,
    )

    output = frame.copy()
    output["LargeWakeSmoothingVersion"] = VERSION
    output["LargeWakeSmoothingBeforeTimeMs"] = pd.NA
    output["LargeWakeSmoothingAfterTimeMs"] = pd.NA
    output["LargeWakeSmoothingAlongShiftM"] = pd.NA
    output["LargeWakeSmoothingHorizontalOffsetM"] = pd.NA
    output.loc[
        active_positions,
        "LargeWakeSmoothingBeforeTimeMs",
    ] = before_times
    output.loc[
        active_positions,
        "LargeWakeSmoothingAfterTimeMs",
    ] = after_times
    output.loc[
        active_positions,
        "LargeWakeSmoothingAlongShiftM",
    ] = shifts
    output.loc[
        active_positions,
        "LargeWakeSmoothingHorizontalOffsetM",
    ] = horizontal_offset
    output.loc[active_positions, "CenterX"] = new_x
    output.loc[active_positions, "CenterTime"] = after_times

    edges, before_counts = bin_counts(
        before_times,
        float(args.time_min_ms),
        float(args.time_max_ms),
        float(args.bin_width_ms),
    )
    _, after_counts = bin_counts(
        after_times,
        float(args.time_min_ms),
        float(args.time_max_ms),
        float(args.bin_width_ms),
    )
    before_array = np.asarray(before_counts, dtype=float)
    after_array = np.asarray(after_counts, dtype=float)
    before_cv = float(before_array.std() / max(before_array.mean(), 1.0e-12))
    after_cv = float(after_array.std() / max(after_array.mean(), 1.0e-12))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary: dict[str, Any] = {
        "status": "pass",
        "version": VERSION,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "anchor_source_index": int(args.anchor_source_index),
        "time_window_ms": [float(args.time_min_ms), float(args.time_max_ms)],
        "smoothing_strength": float(args.smoothing_strength),
        "bin_width_ms": float(args.bin_width_ms),
        "active_small_patch_count": int(len(active_positions)),
        "unchanged_patch_count": int(len(frame) - len(active_positions)),
        "protected_assigned_small_count": int(
            (small & assigned & active_time & protected).sum()
        ),
        "bin_edges_ms": edges,
        "counts_before": before_counts,
        "counts_after": after_counts,
        "coefficient_of_variation_before": before_cv,
        "coefficient_of_variation_after": after_cv,
        "max_to_min_before": float(
            before_array.max() / max(before_array.min(), 1.0)
        ),
        "max_to_min_after": float(
            after_array.max() / max(after_array.min(), 1.0)
        ),
        "along_shift_m": {
            "min": float(shifts.min()),
            "median": float(np.median(shifts)),
            "max": float(shifts.max()),
            "mean_abs": float(np.mean(np.abs(shifts))),
        },
        "preserved": {
            "total_row_count": True,
            "dip_and_dip_azimuth": True,
            "horizontal_offset_from_large_fracture": True,
            "imaging_and_well_control_patches": True,
        },
    }
    if after_cv >= before_cv:
        summary["status"] = "fail"
        summary["reason"] = "smoothing did not reduce coefficient of variation"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[wake-smoothing] status={summary['status']} "
        f"active={len(active_positions)} cv={before_cv:.3f}->{after_cv:.3f} "
        f"output={output_csv}",
        flush=True,
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
