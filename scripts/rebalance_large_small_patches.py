#!/usr/bin/env python3
"""Rebalance large-profile small patches by coordinate-time mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MASK64 = (1 << 64) - 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rebalance large-profile small patches")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--coordinate-column", choices=("CenterX", "CenterY"), required=True)
    p.add_argument("--coordinate-min", type=float, required=True)
    p.add_argument("--coordinate-max", type=float, required=True)
    p.add_argument("--time-min-ms", type=float, required=True)
    p.add_argument("--time-max-ms", type=float, required=True)
    p.add_argument("--block-x-min", type=float, required=True)
    p.add_argument("--block-x-max", type=float, required=True)
    p.add_argument("--block-y-min", type=float, required=True)
    p.add_argument("--block-y-max", type=float, required=True)
    p.add_argument("--sparse-keep-fraction", type=float, required=True)
    p.add_argument("--outside-multiplier", type=int, required=True)
    p.add_argument("--position-sigma-m", type=float, default=10.0)
    p.add_argument("--min-position-offset-m", type=float, default=3.0)
    p.add_argument("--time-sigma-ms", type=float, default=1.0)
    p.add_argument("--azimuth-sigma-deg", type=float, default=8.0)
    p.add_argument("--dip-sigma-deg", type=float, default=5.0)
    p.add_argument("--chunksize", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20261121)
    return p


def splitmix64(value: int, seed: int) -> int:
    x = (int(value) + int(seed) + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def masks(frame: pd.DataFrame, args: argparse.Namespace):
    small = frame["FractureScale"].astype(str).str.strip().str.lower().eq("small")
    block = (
        frame["CenterX"].between(float(args.block_x_min), float(args.block_x_max))
        & frame["CenterY"].between(float(args.block_y_min), float(args.block_y_max))
    )
    sparse = (
        small
        & block
        & frame[args.coordinate_column].between(
            float(args.coordinate_min),
            float(args.coordinate_max),
        )
        & frame["CenterTime"].between(
            float(args.time_min_ms),
            float(args.time_max_ms),
        )
    )
    dense = small & block & ~sparse
    return sparse.to_numpy(), dense.to_numpy()


def first_pass(args: argparse.Namespace):
    row_cursor = 0
    sparse_priorities: list[tuple[int, int]] = []
    dense_count = 0
    input_total = 0
    for frame in pd.read_csv(
        args.input_csv.resolve(),
        usecols=[
            "FractureScale",
            "CenterX",
            "CenterY",
            "CenterTime",
        ],
        chunksize=int(args.chunksize),
        low_memory=False,
    ):
        for column in ("CenterX", "CenterY", "CenterTime"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        sparse, dense = masks(frame, args)
        positions = np.arange(row_cursor, row_cursor + len(frame), dtype=np.int64)
        for position in positions[sparse]:
            sparse_priorities.append(
                (splitmix64(int(position), int(args.seed)), int(position))
            )
        dense_count += int(dense.sum())
        input_total += len(frame)
        row_cursor += len(frame)
    target_sparse = int(round(len(sparse_priorities) * args.sparse_keep_fraction))
    sparse_priorities.sort()
    selected_sparse = {
        position for _, position in sparse_priorities[:target_sparse]
    }
    return {
        "input_total": input_total,
        "sparse_count": len(sparse_priorities),
        "dense_count": dense_count,
        "target_sparse": target_sparse,
        "selected_sparse_positions": selected_sparse,
    }


def copy_rows(
    source: pd.DataFrame,
    copies: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    ordinal_start: int,
) -> pd.DataFrame:
    if copies <= 0 or source.empty:
        return source.iloc[0:0].copy()
    positions = np.repeat(np.arange(len(source), dtype=int), copies)
    result = source.iloc[positions].copy().reset_index(drop=True)
    radius = np.clip(
        np.abs(rng.normal(0.0, float(args.position_sigma_m), len(result))),
        float(args.min_position_offset_m),
        50.0,
    )
    angle = rng.uniform(0.0, 2.0 * np.pi, len(result))
    result["CenterX"] = np.clip(
        result["CenterX"].to_numpy(dtype=float) + radius * np.cos(angle),
        float(args.block_x_min),
        float(args.block_x_max),
    )
    result["CenterY"] = np.clip(
        result["CenterY"].to_numpy(dtype=float) + radius * np.sin(angle),
        float(args.block_y_min),
        float(args.block_y_max),
    )
    result["CenterTime"] = (
        result["CenterTime"].to_numpy(dtype=float)
        + rng.normal(0.0, float(args.time_sigma_ms), len(result))
    )
    result["DipAzimuthDeg"] = (
        result["DipAzimuthDeg"].to_numpy(dtype=float)
        + rng.normal(0.0, float(args.azimuth_sigma_deg), len(result))
    ) % 360.0
    result["DipDeg"] = np.clip(
        result["DipDeg"].to_numpy(dtype=float)
        + rng.normal(0.0, float(args.dip_sigma_deg), len(result)),
        1.0,
        89.0,
    )
    if "PatchID" in result.columns:
        parent = source.iloc[positions]["PatchID"].astype(str).to_numpy()
        result["PatchID"] = [
            f"{value}__large_density_copy_{ordinal_start + index}"
            for index, value in enumerate(parent)
        ]
    result["DensityRebalanceRegion"] = (
        f"outside_sparse_x{int(args.outside_multiplier)}"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    stats = first_pass(args)
    header = pd.read_csv(args.input_csv.resolve(), nrows=0)
    output_columns = [*header.columns, "DensityRebalanceRegion"]
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    rng = np.random.default_rng(int(args.seed))
    row_cursor = 0
    added_dense = 0
    kept_sparse = 0
    output_count = 0
    first_write = True
    for chunk_index, frame in enumerate(
        pd.read_csv(
            args.input_csv.resolve(),
            chunksize=int(args.chunksize),
            low_memory=False,
        ),
        start=1,
    ):
        sparse, dense = masks(frame, args)
        positions = np.arange(row_cursor, row_cursor + len(frame), dtype=np.int64)
        keep_sparse = np.asarray(
            [
                int(position) in stats["selected_sparse_positions"]
                for position in positions
            ],
            dtype=bool,
        ) & sparse
        regular = frame.loc[~(sparse | dense)].copy()
        regular["DensityRebalanceRegion"] = "unchanged"
        sparse_rows = frame.loc[keep_sparse].copy()
        sparse_rows["DensityRebalanceRegion"] = "sparse_kept"
        dense_rows = frame.loc[dense]
        dense_copies = copy_rows(
            dense_rows,
            int(args.outside_multiplier) - 1,
            args,
            rng,
            row_cursor * 10,
        )
        dense_original = dense_rows.copy()
        dense_original["DensityRebalanceRegion"] = "dense_original"
        combined = pd.concat(
            [regular, sparse_rows, dense_original, dense_copies],
            ignore_index=True,
        ).reindex(columns=output_columns)
        combined.to_csv(
            temporary,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig" if first_write else "utf-8",
            lineterminator="\n",
        )
        first_write = False
        added_dense += len(dense_copies)
        kept_sparse += int(keep_sparse.sum())
        output_count += len(combined)
        row_cursor += len(frame)
        if chunk_index % 4 == 0:
            print(
                f"[large-density-rebalance] chunks={chunk_index} "
                f"rows={output_count} added={added_dense}",
                flush=True,
            )
    expected = (
        stats["input_total"]
        - stats["sparse_count"]
        + stats["target_sparse"]
        + stats["dense_count"] * (int(args.outside_multiplier) - 1)
    )
    if output_count != expected:
        raise RuntimeError(f"output count mismatch: {output_count} != {expected}")
    temporary.replace(output_csv)
    summary = {
        "status": "pass",
        "input_csv": str(args.input_csv.resolve()),
        "output_csv": str(output_csv),
        "sparse_mask": {
            "coordinate_column": args.coordinate_column,
            "coordinate_range": [float(args.coordinate_min), float(args.coordinate_max)],
            "time_ms": [float(args.time_min_ms), float(args.time_max_ms)],
        },
        "target_block": {
            "x_m": [float(args.block_x_min), float(args.block_x_max)],
            "y_m": [float(args.block_y_min), float(args.block_y_max)],
        },
        "input_small_sparse_count": int(stats["sparse_count"]),
        "output_small_sparse_count": int(stats["target_sparse"]),
        "input_small_dense_count": int(stats["dense_count"]),
        "output_small_dense_count": int(stats["dense_count"] * args.outside_multiplier),
        "outside_multiplier": int(args.outside_multiplier),
        "sparse_keep_fraction": float(args.sparse_keep_fraction),
        "input_total_rows": int(stats["input_total"]),
        "output_total_rows": int(output_count),
    }
    args.summary_json.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[large-density-rebalance] status=pass output={output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
