#!/usr/bin/env python3
"""Thin one local coordinate-time box and densify small patches outside it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rebalance local small-patch density")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--coordinate-column", choices=("CenterX", "CenterY"), required=True)
    p.add_argument("--coordinate-min", type=float, required=True)
    p.add_argument("--coordinate-max", type=float, required=True)
    p.add_argument("--time-min-ms", type=float, required=True)
    p.add_argument("--time-max-ms", type=float, required=True)
    p.add_argument("--sparse-keep-fraction", type=float, required=True)
    p.add_argument("--outside-multiplier", type=float, required=True)
    p.add_argument("--position-sigma-m", type=float, default=10.0)
    p.add_argument("--min-position-offset-m", type=float, default=3.0)
    p.add_argument("--time-sigma-ms", type=float, default=1.0)
    p.add_argument("--azimuth-sigma-deg", type=float, default=8.0)
    p.add_argument("--dip-sigma-deg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20261121)
    return p


def replicate(
    frame: pd.DataFrame,
    factor: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    frame = frame.reset_index(drop=True)
    target = int(round(len(frame) * float(factor)))
    copies_total = max(target - len(frame), 0)
    base_copies = copies_total // len(frame)
    remainder = copies_total % len(frame)
    ordinals = np.arange(len(frame), dtype=int)
    extra_flags = (
        ((ordinals + 1) * remainder) // len(frame)
        > (ordinals * remainder) // len(frame)
    ).astype(int)
    copy_counts = base_copies + extra_flags
    repeated = np.repeat(np.arange(len(frame), dtype=int), copy_counts)
    copies = frame.iloc[repeated].copy().reset_index(drop=True)
    if copies.empty:
        return frame.copy()
    radius = np.clip(
        np.abs(rng.normal(0.0, float(args.position_sigma_m), len(copies))),
        float(args.min_position_offset_m),
        50.0,
    )
    angle = rng.uniform(0.0, 2.0 * np.pi, len(copies))
    copies["CenterX"] += radius * np.cos(angle)
    copies["CenterY"] += radius * np.sin(angle)
    copies["CenterTime"] += float(args.time_sigma_ms) * rng.normal(
        0.0, 1.0, len(copies)
    )
    copies["DipAzimuthDeg"] += float(args.azimuth_sigma_deg) * rng.normal(
        0.0, 1.0, len(copies)
    )
    copies["DipAzimuthDeg"] %= 360.0
    copies["DipDeg"] = np.clip(
        copies["DipDeg"]
        + float(args.dip_sigma_deg) * rng.normal(0.0, 1.0, len(copies)),
        1.0,
        89.0,
    )
    if "PatchID" in copies.columns:
        parent = frame.loc[repeated, "PatchID"].astype(str).to_numpy()
        copies["PatchID"] = [
            f"{value}__density_rebalance_{index}"
            for index, value in enumerate(parent)
        ]
    return pd.concat([frame, copies], ignore_index=True)


def main() -> int:
    args = parser().parse_args()
    frame = pd.read_csv(args.input_csv.resolve(), low_memory=False)
    required = {
        "FractureScale",
        "CenterX",
        "CenterY",
        "CenterTime",
        "DipAzimuthDeg",
        "DipDeg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"local patch CSV missing columns: {missing}")
    for column in (
        "CenterX",
        "CenterY",
        "CenterTime",
        "DipAzimuthDeg",
        "DipDeg",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    scale = frame["FractureScale"].astype(str).str.strip().str.lower()
    small = frame[scale.eq("small")].copy()
    high = small[
        small[args.coordinate_column].between(
            float(args.coordinate_min),
            float(args.coordinate_max),
        )
        & small["CenterTime"].between(
            float(args.time_min_ms),
            float(args.time_max_ms),
        )
    ]
    outside = small.drop(index=high.index)

    rng_state = np.random.default_rng(int(args.seed))
    keep_count = int(round(len(high) * float(args.sparse_keep_fraction)))
    if keep_count < len(high):
        selected = rng_state.choice(np.arange(len(high)), keep_count, replace=False)
        high_kept = high.iloc[np.sort(selected)].copy()
    else:
        high_kept = high.copy()
    outside_boosted = replicate(
        outside,
        float(args.outside_multiplier),
        args,
        rng_state,
    )
    non_small = frame[~scale.eq("small")].copy()
    output = pd.concat([non_small, high_kept, outside_boosted], ignore_index=True)

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "status": "pass",
        "input_csv": str(args.input_csv.resolve()),
        "output_csv": str(output_csv),
        "mask": {
            "coordinate_column": args.coordinate_column,
            "coordinate_range": [float(args.coordinate_min), float(args.coordinate_max)],
            "time_ms": [float(args.time_min_ms), float(args.time_max_ms)],
        },
        "small_input_count": int(len(small)),
        "sparse_region_input_count": int(len(high)),
        "sparse_region_output_count": int(len(high_kept)),
        "outside_input_count": int(len(outside)),
        "outside_output_count": int(len(outside_boosted)),
        "non_small_count": int(len(non_small)),
        "output_total_count": int(len(output)),
        "sparse_keep_fraction": float(args.sparse_keep_fraction),
        "outside_multiplier": float(args.outside_multiplier),
    }
    summary_path = args.summary_json.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[density-rebalance] status=pass output={output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
