#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concentrate small patches laterally toward a large-fracture XZ trace.

TIME coordinates, patch count, fracture length, dip, and dip azimuth are kept
unchanged.  Only the horizontal offset to the large-fracture trace is reduced.
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


VERSION = "large_wake_small_lateral_concentration_v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Concentrate small patches laterally around a large fracture"
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--anchor-source-index", type=int, required=True)
    p.add_argument("--time-min-ms", type=float, required=True)
    p.add_argument("--time-max-ms", type=float, required=True)
    p.add_argument("--lateral-scale-factor", type=float, default=0.55)
    p.add_argument("--well-track-dir", type=Path, required=True)
    p.add_argument("--well-name", required=True)
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


def endpoint_geometry(row: pd.Series) -> np.ndarray:
    trace = orientation.trace_on_section(
        float(row["DipAzimuthDeg"]),
        float(row["DipDeg"]),
        "XZ",
    )
    if trace is None:
        raise RuntimeError("target large fracture has no XZ trace")
    tangent = np.array([float(trace[0]), 2.0 * float(trace[1])])
    tangent /= np.linalg.norm(tangent)
    center = np.array([float(row["CenterX"]), 2.0 * float(row["CenterTime"])])
    half = 0.5 * float(np.clip(float(row["LengthM"]), 10.0, 500.0))
    return np.vstack([center - half * tangent, center + half * tangent])


def load_track(track_dir: Path, well_name: str) -> pd.DataFrame:
    frames = []
    for path in sorted((track_dir / well_name).glob("*.csv")):
        frame = pd.read_csv(
            path,
            usecols=lambda value: value in {"X", "Y", "TIME"},
            low_memory=False,
        )
        if {"X", "Y", "TIME"}.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"no well-track CSV files found under {track_dir / well_name}")
    track = pd.concat(frames, ignore_index=True)
    for column in ("X", "Y", "TIME"):
        track[column] = pd.to_numeric(track[column], errors="coerce")
    return (
        track.dropna(subset=["X", "Y", "TIME"])
        .sort_values("TIME")
        .drop_duplicates("TIME")
        .reset_index(drop=True)
    )


def patch_line(
    x: float,
    time_ms: float,
    length_m: float,
    dip_azimuth_deg: float,
    dip_deg: float,
) -> np.ndarray:
    trace = orientation.trace_on_section(
        float(dip_azimuth_deg),
        float(dip_deg),
        "XZ",
    )
    if trace is None:
        raise RuntimeError("small patch has no XZ trace")
    half = 0.5 * float(np.clip(float(length_m), 10.0, 500.0))
    return np.asarray(
        [
            [float(x) - half * trace[0], float(time_ms) - half * trace[1]],
            [float(x) + half * trace[0], float(time_ms) + half * trace[1]],
        ],
        dtype=float,
    )


def line_crosses_well(
    line: np.ndarray,
    track: pd.DataFrame,
    time_range: tuple[float, float],
) -> bool:
    times = track["TIME"].to_numpy(dtype=float)
    x_values = track["X"].to_numpy(dtype=float)
    low = max(float(line[:, 1].min()), float(time_range[0]))
    high = min(float(line[:, 1].max()), float(time_range[1]))
    if high <= low:
        return False
    sampled_t = np.linspace(low, high, 2001)
    sampled_h = np.interp(sampled_t, line[:, 1], line[:, 0])
    sampled_well = np.interp(sampled_t, times, x_values)
    delta = sampled_h - sampled_well
    return bool(np.any(np.signbit(delta[:-1]) != np.signbit(delta[1:])))


def lateral_distances(
    x: np.ndarray,
    time_ms: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    points = np.column_stack([x, 2.0 * time_ms])
    start = endpoints[0]
    delta = endpoints[1] - start
    fraction = np.clip(
        ((points - start) @ delta) / float(delta @ delta),
        0.0,
        1.0,
    )
    projected = start + fraction[:, None] * delta
    tangent = delta / np.linalg.norm(delta)
    normal = np.array([-tangent[1], tangent[0]])
    return np.abs((points - projected) @ normal)


def main() -> int:
    args = parser().parse_args()
    if not (0.0 < args.lateral_scale_factor <= 1.0):
        raise ValueError("--lateral-scale-factor must be in (0, 1]")
    if args.time_min_ms >= args.time_max_ms:
        raise ValueError("--time-min-ms must be smaller than --time-max-ms")

    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    frame = pd.read_csv(input_csv, low_memory=False)
    required = {
        "SourceIndex",
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
    endpoints = endpoint_geometry(anchors.iloc[0])
    trace_times = endpoints[:, 1] / 2.0
    trace_x = endpoints[:, 0]

    small = frame["FractureScale"].astype(str).str.strip().str.lower().eq("small")
    assigned = frame["LargeAnchorID"].eq(int(args.anchor_source_index))
    active_time = frame["CenterTime"].between(
        float(args.time_min_ms),
        float(args.time_max_ms),
        inclusive="both",
    )
    protected = protected_mask(frame)
    active_mask = small & assigned & active_time & ~protected
    positions = np.flatnonzero(active_mask.to_numpy())
    if len(positions) < 10:
        raise RuntimeError("not enough small patches to concentrate")

    old_x = frame.loc[positions, "CenterX"].to_numpy(dtype=float)
    times = frame.loc[positions, "CenterTime"].to_numpy(dtype=float)
    line_x = np.interp(times, trace_times, trace_x)
    horizontal_offset = old_x - line_x
    new_x = line_x + float(args.lateral_scale_factor) * horizontal_offset
    track = load_track(args.well_track_dir.resolve(), str(args.well_name))
    crossing_before = np.zeros(len(positions), dtype=bool)
    crossing_after = np.zeros(len(positions), dtype=bool)
    reverted = np.zeros(len(positions), dtype=bool)
    for index, frame_index in enumerate(positions):
        row = frame.loc[frame_index]
        before_line = patch_line(
            float(row["CenterX"]),
            float(row["CenterTime"]),
            float(row["LengthM"]),
            float(row["DipAzimuthDeg"]),
            float(row["DipDeg"]),
        )
        after_line = patch_line(
            float(new_x[index]),
            float(row["CenterTime"]),
            float(row["LengthM"]),
            float(row["DipAzimuthDeg"]),
            float(row["DipDeg"]),
        )
        crossing_before[index] = line_crosses_well(
            before_line,
            track,
            (float(args.time_min_ms), float(args.time_max_ms)),
        )
        crossing_after[index] = line_crosses_well(
            after_line,
            track,
            (float(args.time_min_ms), float(args.time_max_ms)),
        )
        if crossing_after[index] and not crossing_before[index]:
            reverted[index] = True
            new_x[index] = float(row["CenterX"])
            crossing_after[index] = False
    before_distance = lateral_distances(old_x, times, endpoints)
    after_distance = lateral_distances(new_x, times, endpoints)

    output = frame.copy()
    output["LargeWakeConcentrationVersion"] = VERSION
    output["LargeWakeConcentrationTimeWindowMs"] = pd.NA
    output["LargeWakeConcentrationBeforeX"] = pd.NA
    output["LargeWakeConcentrationAfterX"] = pd.NA
    output["LargeWakeConcentrationXShiftM"] = pd.NA
    output["LargeWakeConcentrationDistanceBeforeM"] = pd.NA
    output["LargeWakeConcentrationDistanceAfterM"] = pd.NA
    output.loc[positions, "LargeWakeConcentrationTimeWindowMs"] = (
        f"{float(args.time_min_ms):g}-{float(args.time_max_ms):g}"
    )
    output.loc[positions, "LargeWakeConcentrationBeforeX"] = old_x
    output.loc[positions, "LargeWakeConcentrationAfterX"] = new_x
    output.loc[positions, "LargeWakeConcentrationXShiftM"] = new_x - old_x
    output.loc[positions, "LargeWakeConcentrationDistanceBeforeM"] = before_distance
    output.loc[positions, "LargeWakeConcentrationDistanceAfterM"] = after_distance
    output.loc[positions, "CenterX"] = new_x

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary: dict[str, Any] = {
        "status": "pass",
        "version": VERSION,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "anchor_source_index": int(args.anchor_source_index),
        "time_window_ms": [float(args.time_min_ms), float(args.time_max_ms)],
        "lateral_scale_factor": float(args.lateral_scale_factor),
        "active_small_patch_count": int(len(positions)),
        "unchanged_patch_count": int(len(frame) - len(positions)),
        "well_crossing_before": int(crossing_before.sum()),
        "well_crossing_after": int(crossing_after.sum()),
        "reverted_to_avoid_new_well_crossing": int(reverted.sum()),
        "protected_assigned_small_count": int(
            (small & assigned & active_time & protected).sum()
        ),
        "counts_within_lateral_band": {
            str(radius): {
                "before": int((before_distance <= radius).sum()),
                "after": int((after_distance <= radius).sum()),
            }
            for radius in (15, 30, 45, 60, 80, 100)
        },
        "mean_lateral_distance_m": {
            "before": float(before_distance.mean()),
            "after": float(after_distance.mean()),
        },
        "x_shift_m": {
            "min": float((new_x - old_x).min()),
            "median": float(np.median(new_x - old_x)),
            "max": float((new_x - old_x).max()),
            "mean_abs": float(np.mean(np.abs(new_x - old_x))),
        },
        "preserved": {
            "total_row_count": True,
            "center_time": True,
            "dip_and_dip_azimuth": True,
            "center_y": True,
            "imaging_and_well_control_patches": True,
            "no_new_well_crossings": bool(
                not np.any(crossing_after & ~crossing_before)
            ),
        },
    }
    if after_distance.mean() >= before_distance.mean():
        summary["status"] = "fail"
        summary["reason"] = "mean distance did not decrease"
    if np.any(crossing_after & ~crossing_before):
        summary["status"] = "fail"
        summary["reason"] = "new well crossings remain after correction"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[wake-concentration] status={summary['status']} "
        f"active={len(positions)} "
        f"mean_distance={before_distance.mean():.1f}->{after_distance.mean():.1f}m "
        f"within30={summary['counts_within_lateral_band']['30']['before']}"
        f"->{summary['counts_within_lateral_band']['30']['after']} "
        f"output={output_csv}",
        flush=True,
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
