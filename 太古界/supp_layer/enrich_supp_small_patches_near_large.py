#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enrich small patches around one conditioned large-fracture anchor.

The input is already conditioned against the imaging interval.  This utility
only adds jittered small-patch copies around a selected XZ large-fracture
trace; existing rows and imaging-locked patches are preserved.
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add jittered small patches around a conditioned large fracture"
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--anchor-source-index", type=int, required=True)
    p.add_argument("--added-count", type=int, required=True)
    p.add_argument(
        "--enrichment-stage",
        default="upper_right_large_fracture_enrichment_v1",
    )
    p.add_argument("--time-min-ms", type=float)
    p.add_argument("--time-max-ms", type=float)
    p.add_argument("--parent-radius-m", type=float, default=150.0)
    p.add_argument("--lateral-sigma-m", type=float, default=32.0)
    p.add_argument("--max-lateral-offset-m", type=float, default=95.0)
    p.add_argument("--longitudinal-sigma-m", type=float, default=16.0)
    p.add_argument("--max-longitudinal-offset-m", type=float, default=60.0)
    p.add_argument("--azimuth-sigma-deg", type=float, default=3.5)
    p.add_argument("--dip-sigma-deg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=405)
    return p


def trace_geometry(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    trace = orientation.trace_on_section(
        float(row["DipAzimuthDeg"]),
        float(row["DipDeg"]),
        "XZ",
    )
    if trace is None:
        raise RuntimeError("target large fracture has no XZ trace")
    length = float(np.clip(float(row["LengthM"]), 10.0, 500.0))
    half = 0.5 * length
    center = np.array([float(row["CenterX"]), 2.0 * float(row["CenterTime"])])
    tangent_m = np.array([float(trace[0]), 2.0 * float(trace[1])])
    tangent_m /= np.linalg.norm(tangent_m)
    endpoints = np.vstack([center - half * tangent_m, center + half * tangent_m])
    return endpoints, tangent_m


def trace_offsets(
    frame: pd.DataFrame,
    endpoints: np.ndarray,
    tangent_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start = endpoints[0]
    delta = endpoints[1] - start
    denominator = float(np.dot(delta, delta))
    points = np.column_stack(
        [
            frame["CenterX"].to_numpy(dtype=float),
            2.0 * frame["CenterTime"].to_numpy(dtype=float),
        ]
    )
    fraction = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
    projected = start + fraction[:, None] * delta
    lateral = ((points - projected) @ np.array([-tangent_m[1], tangent_m[0]]))
    longitudinal = ((points - start) @ tangent_m)
    return lateral, longitudinal


def main() -> int:
    args = parser().parse_args()
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    if args.added_count <= 0:
        raise ValueError("--added-count must be positive")
    if (args.time_min_ms is None) != (args.time_max_ms is None):
        raise ValueError("--time-min-ms and --time-max-ms must be used together")
    if (
        args.time_min_ms is not None
        and float(args.time_min_ms) >= float(args.time_max_ms)
    ):
        raise ValueError("--time-min-ms must be smaller than --time-max-ms")
    enrichment_stage = str(args.enrichment_stage).strip()
    if not enrichment_stage:
        raise ValueError("--enrichment-stage must not be empty")

    frame = pd.read_csv(input_csv, low_memory=False)
    required = {
        "SourceIndex",
        "PatchID",
        "FractureScale",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "DipAzimuthDeg",
        "DipDeg",
        "LargeAnchorID",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"input CSV missing required columns: {missing}")
    numeric_columns = [
        "SourceIndex",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "DipAzimuthDeg",
        "DipDeg",
        "LargeAnchorID",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    anchor_matches = frame[
        frame["SourceIndex"].eq(int(args.anchor_source_index))
        & frame["FractureScale"].astype(str).str.strip().str.lower().eq("large")
    ]
    if len(anchor_matches) != 1:
        raise RuntimeError(
            "expected exactly one large-fracture anchor row for "
            f"SourceIndex={args.anchor_source_index}, found {len(anchor_matches)}"
        )
    anchor = anchor_matches.iloc[0]
    endpoints, tangent_m = trace_geometry(anchor)

    all_lateral, _ = trace_offsets(frame, endpoints, tangent_m)
    frame["_TargetLargeLateralDistanceM"] = np.abs(all_lateral)
    small_mask = frame["FractureScale"].astype(str).str.strip().str.lower().eq("small")
    immutable_mask = pd.Series(False, index=frame.index)
    for column in (
        "ImagingMatchEventLocked",
        "IsImagingMatchPatch",
        "IsWellControlPatch",
    ):
        if column in frame:
            immutable_mask |= pd.to_numeric(
                frame[column],
                errors="coerce",
            ).fillna(0).ne(0)
    previous_enrichment_mask = (
        frame["TargetedLargeEnrichmentParentPatchID"].notna()
        if "TargetedLargeEnrichmentParentPatchID" in frame
        else pd.Series(False, index=frame.index)
    )
    time_mask = pd.Series(True, index=frame.index)
    if args.time_min_ms is not None:
        time_mask = frame["CenterTime"].between(
            float(args.time_min_ms),
            float(args.time_max_ms),
            inclusive="both",
        )
    parents = frame[
        small_mask
        & frame["LargeAnchorID"].eq(int(args.anchor_source_index))
        & frame["_TargetLargeLateralDistanceM"].le(float(args.parent_radius_m))
        & ~immutable_mask
        & ~previous_enrichment_mask
        & time_mask
    ].copy()
    if parents.empty:
        raise RuntimeError("no movable small-patch parents found around anchor")

    rng = np.random.default_rng(int(args.seed))
    replace = len(parents) < int(args.added_count)
    parent_positions = rng.choice(
        np.arange(len(parents)),
        size=int(args.added_count),
        replace=replace,
    )
    added = parents.iloc[parent_positions].copy().reset_index(drop=True)

    lateral_offset = np.clip(
        rng.normal(0.0, float(args.lateral_sigma_m), len(added)),
        -float(args.max_lateral_offset_m),
        float(args.max_lateral_offset_m),
    )
    longitudinal_offset = np.clip(
        rng.normal(0.0, float(args.longitudinal_sigma_m), len(added)),
        -float(args.max_longitudinal_offset_m),
        float(args.max_longitudinal_offset_m),
    )
    normal_m = np.array([-tangent_m[1], tangent_m[0]])
    position_offset_m = (
        normal_m[None, :] * lateral_offset[:, None]
        + tangent_m[None, :] * longitudinal_offset[:, None]
    )
    added["CenterX"] = added["CenterX"].to_numpy(dtype=float) + position_offset_m[:, 0]
    added["CenterTime"] = (
        added["CenterTime"].to_numpy(dtype=float) + position_offset_m[:, 1] / 2.0
    )
    if args.time_min_ms is not None:
        added["CenterTime"] = added["CenterTime"].clip(
            float(args.time_min_ms),
            float(args.time_max_ms),
        )
    added["DipAzimuthDeg"] = (
        added["DipAzimuthDeg"].to_numpy(dtype=float)
        + rng.normal(0.0, float(args.azimuth_sigma_deg), len(added))
    ) % 360.0
    added["DipDeg"] = np.clip(
        added["DipDeg"].to_numpy(dtype=float)
        + rng.normal(0.0, float(args.dip_sigma_deg), len(added)),
        1.0,
        89.0,
    )

    parent_patch_ids = added["PatchID"].astype(str).to_numpy()
    added["PatchID"] = [
        f"{parent}__{enrichment_stage}_{index:05d}"
        for index, parent in enumerate(parent_patch_ids)
    ]
    added["SourceIndex"] = -900000 - np.arange(len(added), dtype=int)
    if "GenerationStage" in added:
        added["GenerationStage"] = (
            added["GenerationStage"].astype(str)
            + f"|{enrichment_stage}"
        )
    added["TargetedLargeEnrichmentAnchorSourceIndex"] = int(
        args.anchor_source_index
    )
    added["TargetedLargeEnrichmentParentPatchID"] = parent_patch_ids
    added["TargetedLargeEnrichmentLateralOffsetM"] = lateral_offset
    added["TargetedLargeEnrichmentLongitudinalOffsetM"] = longitudinal_offset
    added["LargeAnchorConditioned"] = 3
    added["LargeAnchorID"] = int(args.anchor_source_index)
    added["LargeAnchorDistanceM"] = np.abs(lateral_offset)
    added["LargeAnchorOrientationWeight"] = 0.0
    for column in (
        "ImagingMatchEventLocked",
        "IsImagingMatchPatch",
        "IsWellControlPatch",
        "IsDensityCompensationPatch",
    ):
        if column in added:
            added[column] = 0
    if "WellControlSampleID" in added:
        added["WellControlSampleID"] = ""
    added = added.drop(columns=["_TargetLargeLateralDistanceM"])

    output = pd.concat([frame.drop(columns=["_TargetLargeLateralDistanceM"]), added])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")

    small = output[
        output["FractureScale"].astype(str).str.strip().str.lower().eq("small")
    ]
    target_after = int(small["LargeAnchorID"].eq(int(args.anchor_source_index)).sum())
    scale_counts = {
        str(key): int(value)
        for key, value in (
            output["FractureScale"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .items()
        )
    }
    summary: dict[str, Any] = {
        "status": "pass",
        "method": enrichment_stage,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "anchor_source_index": int(args.anchor_source_index),
        "anchor_center_x_m": float(anchor["CenterX"]),
        "anchor_center_time_ms": float(anchor["CenterTime"]),
        "anchor_trace_xz_endpoints": endpoints.tolist(),
        "input_total_rows": int(len(frame)),
        "output_total_rows": int(len(output)),
        "input_small_scale_counts": {
            str(key): int(value)
            for key, value in (
                frame["FractureScale"]
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
                .items()
            )
        },
        "output_scale_counts": scale_counts,
        "eligible_parent_count": int(len(parents)),
        "added_small_patch_count": int(len(added)),
        "time_filter_ms": (
            [float(args.time_min_ms), float(args.time_max_ms)]
            if args.time_min_ms is not None
            else None
        ),
        "previous_enrichment_rows_excluded_from_parents": int(
            previous_enrichment_mask.sum()
        ),
        "target_small_count_before": int(
            frame[
                small_mask
                & frame["LargeAnchorID"].eq(int(args.anchor_source_index))
            ].shape[0]
        ),
        "target_small_count_after": target_after,
        "parent_radius_m": float(args.parent_radius_m),
        "jitter": {
            "lateral_sigma_m": float(args.lateral_sigma_m),
            "max_lateral_offset_m": float(args.max_lateral_offset_m),
            "longitudinal_sigma_m": float(args.longitudinal_sigma_m),
            "max_longitudinal_offset_m": float(args.max_longitudinal_offset_m),
            "azimuth_sigma_deg": float(args.azimuth_sigma_deg),
            "dip_sigma_deg": float(args.dip_sigma_deg),
            "seed": int(args.seed),
        },
        "preserved": {
            "imaging_locked_patches": True,
            "well_control_patches": True,
            "medium_and_large_patch_counts": True,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[large-enrichment] "
        f"anchor={args.anchor_source_index} added={len(added)} "
        f"target_before={summary['target_small_count_before']} "
        f"target_after={target_after} output={output_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
