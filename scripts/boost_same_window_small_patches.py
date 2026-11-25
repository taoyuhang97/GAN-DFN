#!/usr/bin/env python3
"""Densify small DFN patches in an imaging-time window with controlled jitter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Boost small DFN patch count with position/orientation jitter"
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--factor", type=float, required=True)
    p.add_argument("--time-min-ms", type=float, required=True)
    p.add_argument("--time-max-ms", type=float, required=True)
    p.add_argument("--x-min", type=float, required=True)
    p.add_argument("--x-max", type=float, required=True)
    p.add_argument("--y-min", type=float, required=True)
    p.add_argument("--y-max", type=float, required=True)
    p.add_argument("--position-sigma-m", type=float, default=12.0)
    p.add_argument("--min-position-offset-m", type=float, default=3.0)
    p.add_argument("--max-position-offset-m", type=float, default=50.0)
    p.add_argument("--time-sigma-ms", type=float, default=1.2)
    p.add_argument("--azimuth-sigma-deg", type=float, default=8.0)
    p.add_argument("--dip-sigma-deg", type=float, default=5.0)
    p.add_argument("--chunksize", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20261121)
    return p


def target_mask(frame: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    return (
        frame["FractureScale"].astype(str).str.strip().str.lower().eq("small")
        & frame["CenterTime"].between(
            float(args.time_min_ms),
            float(args.time_max_ms),
        )
        & frame["CenterX"].between(float(args.x_min), float(args.x_max))
        & frame["CenterY"].between(float(args.y_min), float(args.y_max))
    )


def count_rows(path: Path, args: argparse.Namespace) -> tuple[int, int]:
    total = 0
    selected = 0
    for chunk in pd.read_csv(
        path,
        usecols=["FractureScale", "CenterX", "CenterY", "CenterTime"],
        chunksize=int(args.chunksize),
        low_memory=False,
    ):
        total += int(len(chunk))
        selected += int(target_mask(chunk, args).sum())
    if selected == 0:
        raise RuntimeError("no small DFN patches found inside the requested window")
    return total, selected


def main() -> int:
    args = parser().parse_args()
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    input_header = pd.read_csv(input_csv, nrows=0)
    boost_markers = {
        "SameWindowBoostFactor",
        "SameWindowBoostParentPatchID",
        "SameWindowBoostCopyIndex",
    }
    if boost_markers.intersection(input_header.columns) or "same_window_boost" in input_csv.name:
        raise ValueError(
            "input CSV is already same-window boosted; refusing double densification"
        )

    started = time.time()
    input_total, target_original = count_rows(input_csv, args)
    target_total = int(round(float(target_original) * float(args.factor)))
    duplicate_total = target_total - target_original
    base_copies = duplicate_total // target_original
    remainder = duplicate_total % target_original
    if duplicate_total < 0:
        raise ValueError("factor must preserve or increase the target row count")

    rng = np.random.default_rng(int(args.seed))
    first_write = True
    target_cursor = 0
    output_rows = 0
    added_rows = 0
    chunk_index = 0
    for chunk in pd.read_csv(
        input_csv,
        chunksize=int(args.chunksize),
        low_memory=False,
    ):
        chunk_index += 1
        mask = target_mask(chunk, args).to_numpy()
        selected_positions = np.flatnonzero(mask)
        extras: list[pd.DataFrame] = []
        if len(selected_positions):
            ordinals = np.arange(
                target_cursor,
                target_cursor + len(selected_positions),
                dtype=np.int64,
            )
            numerator_end = (ordinals + 1) * remainder
            numerator_start = ordinals * remainder
            copy_counts = base_copies + (
                numerator_end // target_original
                > numerator_start // target_original
            ).astype(int)

            repeated_positions = np.repeat(selected_positions, copy_counts)
            copy_indices = np.concatenate(
                [np.arange(int(count), dtype=int) for count in copy_counts]
            )
            parent_ordinals = np.repeat(ordinals, copy_counts)
            added = chunk.iloc[repeated_positions].copy()

            radius = np.abs(
                rng.normal(0.0, float(args.position_sigma_m), size=len(added))
            )
            radius = np.clip(
                radius,
                float(args.min_position_offset_m),
                float(args.max_position_offset_m),
            )
            angle = rng.uniform(0.0, 2.0 * np.pi, size=len(added))
            added["CenterX"] = np.clip(
                added["CenterX"].to_numpy(dtype=float) + radius * np.cos(angle),
                float(args.x_min),
                float(args.x_max),
            )
            added["CenterY"] = np.clip(
                added["CenterY"].to_numpy(dtype=float) + radius * np.sin(angle),
                float(args.y_min),
                float(args.y_max),
            )
            added["CenterTime"] = np.clip(
                added["CenterTime"].to_numpy(dtype=float)
                + rng.normal(0.0, float(args.time_sigma_ms), size=len(added)),
                float(args.time_min_ms),
                float(args.time_max_ms),
            )
            added["DipAzimuthDeg"] = (
                added["DipAzimuthDeg"].to_numpy(dtype=float)
                + rng.normal(0.0, float(args.azimuth_sigma_deg), size=len(added))
            ) % 360.0
            added["DipDeg"] = np.clip(
                added["DipDeg"].to_numpy(dtype=float)
                + rng.normal(0.0, float(args.dip_sigma_deg), size=len(added)),
                1.0,
                89.0,
            )
            if "PatchID" in added.columns:
                added["PatchID"] = [
                    f"{parent}__same_window_boost_{ordinal}_{copy}"
                    for parent, ordinal, copy in zip(
                        added["PatchID"].astype(str),
                        parent_ordinals,
                        copy_indices,
                    )
                ]
            added["SameWindowBoostParentPatchID"] = chunk.iloc[
                repeated_positions
            ]["PatchID"].astype(str).to_numpy()
            added["SameWindowBoostCopyIndex"] = copy_indices
            added["SameWindowBoostFactor"] = float(args.factor)
            if "GenerationStage" in added.columns:
                added["GenerationStage"] = (
                    added["GenerationStage"].astype(str)
                    + "|same_window_density_boost_v1"
                )
            extras.append(added)
            target_cursor += len(selected_positions)
            added_rows += len(added)

        combined = (
            pd.concat([chunk, *extras], ignore_index=True)
            if extras
            else chunk
        )
        combined.to_csv(
            temporary_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig" if first_write else "utf-8",
            lineterminator="\n",
        )
        first_write = False
        output_rows += len(combined)
        if chunk_index % 4 == 0:
            print(
                f"[same-window-boost] chunks={chunk_index} "
                f"target_seen={target_cursor}/{target_original} "
                f"rows={output_rows}",
                flush=True,
            )

    if target_cursor != target_original:
        raise RuntimeError(
            f"target row mismatch: {target_cursor} != {target_original}"
        )
    expected_output = input_total - target_original + target_total
    if output_rows != expected_output:
        raise RuntimeError(
            f"output row mismatch: {output_rows} != {expected_output}"
        )
    temporary_csv.replace(output_csv)
    summary = {
        "status": "pass",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "factor": float(args.factor),
        "target_region": {
            "time_ms": [float(args.time_min_ms), float(args.time_max_ms)],
            "x_m": [float(args.x_min), float(args.x_max)],
            "y_m": [float(args.y_min), float(args.y_max)],
            "fracture_scale": "small",
        },
        "input_total_rows": int(input_total),
        "target_original_small_rows": int(target_original),
        "target_boosted_small_rows": int(target_total),
        "added_small_rows": int(added_rows),
        "output_total_rows": int(output_rows),
        "jitter": {
            "position_sigma_m": float(args.position_sigma_m),
            "min_position_offset_m": float(args.min_position_offset_m),
            "max_position_offset_m": float(args.max_position_offset_m),
            "time_sigma_ms": float(args.time_sigma_ms),
            "azimuth_sigma_deg": float(args.azimuth_sigma_deg),
            "dip_sigma_deg": float(args.dip_sigma_deg),
            "seed": int(args.seed),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[same-window-boost] status=pass output={output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
