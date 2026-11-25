#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebalance medium and small patches between the two Taigu display layers.

The transformation is intentionally limited to the 5 km section block:

* move half of eligible upper-composite medium patches into the crust;
* retain 80% of upper-composite small patches;
* move the remaining 20% into the crust and create three jittered copies;
* skip small-patch relocation where the crust already has a small patch in
  either the XZ or YZ 100 m x 10 ms occupancy cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


VERSION = "taigu_supp_layer_density_balance_v1"
UPPER_LAYER = "上部复合层"
UPPER_SUPPLEMENT_LAYER = "上部补充层"
CRUST_LAYER = "太古界风化壳"
NEW_COLUMNS = [
    "LayerDensityBalanceVersion",
    "LayerDensityBalanceAction",
    "LayerDensityBalanceParentPatchID",
    "LayerDensityBalanceCopyIndex",
    "LayerDensityOriginalCenterTime",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebalance Taigu medium/small patches between display layers"
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--horizon-contract-csv", type=Path, required=True)
    p.add_argument("--occupancy-reference-csv", type=Path, required=True)
    p.add_argument("--x-min", type=float, required=True)
    p.add_argument("--x-max", type=float, required=True)
    p.add_argument("--y-min", type=float, required=True)
    p.add_argument("--y-max", type=float, required=True)
    p.add_argument("--medium-move-fraction", type=float, default=0.5)
    p.add_argument("--small-keep-fraction", type=float, default=0.8)
    p.add_argument("--small-copy-count", type=int, default=3)
    p.add_argument("--occupancy-cell-x-m", type=float, default=100.0)
    p.add_argument("--occupancy-cell-time-ms", type=float, default=10.0)
    p.add_argument("--horizontal-jitter-sigma-m", type=float, default=18.0)
    p.add_argument("--horizontal-jitter-max-m", type=float, default=45.0)
    p.add_argument("--time-jitter-sigma-ms", type=float, default=1.0)
    p.add_argument("--time-jitter-max-ms", type=float, default=2.5)
    p.add_argument("--azimuth-jitter-sigma-deg", type=float, default=3.0)
    p.add_argument("--dip-jitter-sigma-deg", type=float, default=2.0)
    p.add_argument("--max-placement-attempts", type=int, default=100)
    p.add_argument("--chunksize", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261121)
    return p


def stable_hash(value: str, salt: str) -> int:
    payload = f"{salt}|{value}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )


def choose_subset(ids: list[str], keep_count: int, salt: str) -> set[str]:
    ranked = sorted(ids, key=lambda value: stable_hash(value, salt))
    return set(ranked[: max(0, min(int(keep_count), len(ranked)))])


def row_key_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["PatchID"].astype(str)
        + "|"
        + frame["LayerGroup"].astype(str)
        + "|"
        + frame["FractureScale"].astype(str).str.strip().str.lower()
    )


def target_mask(frame: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    x = pd.to_numeric(frame["CenterX"], errors="coerce")
    y = pd.to_numeric(frame["CenterY"], errors="coerce")
    return (
        x.between(float(args.x_min), float(args.x_max))
        & y.between(float(args.y_min), float(args.y_max))
    )


def protected_mask(frame: pd.DataFrame) -> pd.Series:
    keys = (
        "IsImagingMatchPatch",
        "IsWellControlPatch",
        "IsImagingGroundTruth",
    )
    mask = pd.Series(False, index=frame.index)
    for column in keys:
        if column in frame:
            mask |= (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0)
                .ne(0)
            )
    return mask


def load_horizon_sampler(
    path: Path,
) -> tuple[cKDTree, np.ndarray]:
    frame = pd.read_csv(
        path,
        usecols=["X", "Y", "TopTimeMs", "MidTimeMs", "BaseTimeMs"],
        low_memory=False,
    )
    values = frame[
        ["TopTimeMs", "MidTimeMs", "BaseTimeMs"]
    ].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1).to_numpy()
    coordinates = frame.loc[
        valid,
        ["X", "Y"],
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    boundaries = values.loc[valid].to_numpy(dtype=float)
    if len(coordinates) == 0:
        raise RuntimeError("horizon contract has no valid Top/Mid/Base rows")
    return cKDTree(coordinates), boundaries


def load_visible_occupancy(
    path: Path,
    *,
    cell_x_m: float,
    cell_time_ms: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int]:
    frame = pd.read_csv(
        path,
        usecols=["projection", "layer", "scale", "h1", "z1", "h2", "z2"],
        low_memory=False,
    )
    frame = frame[
        frame["layer"].astype(str).eq(CRUST_LAYER)
        & frame["scale"].astype(str).str.strip().str.lower().eq("small")
    ].copy()
    frame["h_mid"] = (
        pd.to_numeric(frame["h1"], errors="coerce")
        + pd.to_numeric(frame["h2"], errors="coerce")
    ) / 2.0
    frame["z_mid"] = (
        pd.to_numeric(frame["z1"], errors="coerce")
        + pd.to_numeric(frame["z2"], errors="coerce")
    ) / 2.0
    xz: set[tuple[int, int]] = set()
    yz: set[tuple[int, int]] = set()
    for row in frame.itertuples(index=False):
        cell = (
            int(np.floor(float(row.h_mid) / float(cell_x_m))),
            int(np.floor(float(row.z_mid) / float(cell_time_ms))),
        )
        if str(row.projection).upper() == "XZ":
            xz.add(cell)
        elif str(row.projection).upper() == "YZ":
            yz.add(cell)
    return xz, yz, int(len(frame))


def attach_horizon_bounds(
    rows: pd.DataFrame,
    sampler: tuple[cKDTree, np.ndarray],
) -> pd.DataFrame:
    result = rows.copy()
    tree, boundaries = sampler
    coordinates = result[["CenterX", "CenterY"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise RuntimeError("patch center coordinates contain non-finite values")
    _, indices = tree.query(coordinates, k=1)
    result["LocalTopTimeMs"] = boundaries[indices, 0]
    result["LocalMidTimeMs"] = boundaries[indices, 1]
    result["LocalBaseTimeMs"] = boundaries[indices, 2]
    return result


def occupied_cell(
    x: float,
    y: float,
    time_ms: float,
    spacing_x: float,
    spacing_t: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    xz = (int(np.floor(float(x) / spacing_x)), int(np.floor(float(time_ms) / spacing_t)))
    yz = (int(np.floor(float(y) / spacing_x)), int(np.floor(float(time_ms) / spacing_t)))
    return xz, yz


def map_time_to_crust(
    top: np.ndarray,
    mid: np.ndarray,
    base: np.ndarray,
    center_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    upper_thickness = mid - top
    crust_thickness = base - mid
    valid = (
        np.isfinite(top)
        & np.isfinite(mid)
        & np.isfinite(base)
        & np.isfinite(center_time)
        & (upper_thickness > 0.0)
        & (crust_thickness > 0.0)
    )
    fraction = np.zeros(len(center_time), dtype=float)
    fraction[valid] = np.clip(
        (center_time[valid] - top[valid]) / upper_thickness[valid],
        0.0,
        1.0,
    )
    mapped = mid + (0.10 + 0.80 * fraction) * crust_thickness
    return mapped, valid


def first_pass(
    input_csv: Path,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], set[tuple[int, int]], set[tuple[int, int]], dict[str, Any]]:
    medium_ids: list[str] = []
    small_ids: list[str] = []
    xz_occupied: set[tuple[int, int]] = set()
    yz_occupied: set[tuple[int, int]] = set()
    target_rows = 0
    crust_small_rows = 0
    input_total_rows = 0
    protected_medium_rows = 0
    protected_small_rows = 0
    usecols = [
        "PatchID",
        "LayerGroup",
        "FractureScale",
        "CenterX",
        "CenterY",
        "CenterTime",
        "IsImagingMatchPatch",
        "IsWellControlPatch",
        "IsImagingGroundTruth",
    ]
    for chunk_index, chunk in enumerate(
        pd.read_csv(input_csv, usecols=usecols, chunksize=50000, low_memory=False),
        1,
    ):
        input_total_rows += int(len(chunk))
        in_block = target_mask(chunk, args).to_numpy()
        target_rows += int(in_block.sum())
        layer = chunk["LayerGroup"].astype(str)
        scale = chunk["FractureScale"].astype(str).str.strip().str.lower()
        protected = protected_mask(chunk).to_numpy()

        displayed_upper = layer.isin(
            [UPPER_LAYER, UPPER_SUPPLEMENT_LAYER]
        ).to_numpy()
        upper_medium = in_block & displayed_upper & scale.eq("medium").to_numpy()
        upper_small = in_block & displayed_upper & scale.eq("small").to_numpy()
        crust_small = in_block & layer.eq(CRUST_LAYER).to_numpy() & scale.eq("small").to_numpy()
        keys = row_key_series(chunk)
        medium_ids.extend(keys.loc[upper_medium & ~protected])
        small_ids.extend(keys.loc[upper_small & ~protected])
        protected_medium_rows += int((upper_medium & protected).sum())
        protected_small_rows += int((upper_small & protected).sum())

        selected_crust = chunk.loc[crust_small, ["CenterX", "CenterY", "CenterTime"]].copy()
        crust_small_rows += int(len(selected_crust))
        for row in selected_crust.itertuples(index=False):
            xz_cell, yz_cell = occupied_cell(
                float(row.CenterX),
                float(row.CenterY),
                float(row.CenterTime),
                float(args.occupancy_cell_x_m),
                float(args.occupancy_cell_time_ms),
            )
            xz_occupied.add(xz_cell)
            yz_occupied.add(yz_cell)
        if chunk_index % 4 == 0:
            print(
                "[layer-balance] first-pass "
                f"chunks={chunk_index} target_rows={target_rows} "
                f"medium={len(medium_ids)} small={len(small_ids)} "
                f"crust_cells={len(xz_occupied)}/{len(yz_occupied)}",
                flush=True,
            )

    summary = {
        "input_total_rows": int(input_total_rows),
        "target_rows": int(target_rows),
        "eligible_medium_count": int(len(medium_ids)),
        "eligible_small_count": int(len(small_ids)),
        "existing_crust_small_count": int(crust_small_rows),
        "existing_crust_xz_occupied_cells": int(len(xz_occupied)),
        "existing_crust_yz_occupied_cells": int(len(yz_occupied)),
        "protected_medium_count": int(protected_medium_rows),
        "protected_small_count": int(protected_small_rows),
    }
    return medium_ids, small_ids, xz_occupied, yz_occupied, summary


def apply_common_move(
    rows: pd.DataFrame,
    action: str,
    copy_index: int | None = None,
) -> pd.DataFrame:
    result = rows.copy()
    patch_ids = result["PatchID"].astype(str).to_numpy()
    parent_keys = row_key_series(result).to_numpy()
    scales = result["FractureScale"].astype(str).str.strip().str.lower().to_numpy()
    result["LayerDensityBalanceVersion"] = VERSION
    result["LayerDensityBalanceAction"] = action
    result["LayerDensityBalanceParentPatchID"] = parent_keys
    result["LayerDensityBalanceCopyIndex"] = (
        int(copy_index) if copy_index is not None else pd.NA
    )
    result["LayerDensityOriginalCenterTime"] = pd.to_numeric(
        result["CenterTime"],
        errors="coerce",
    ).to_numpy()
    result["LayerGroup"] = CRUST_LAYER
    if "LayerCode" in result:
        result["LayerCode"] = 2
    if "DominantLayer" in result:
        result["DominantLayer"] = CRUST_LAYER
    if "GenerationStage" in result:
        result["GenerationStage"] = (
            result["GenerationStage"].astype(str) + f"|{action}"
        )
    if "DensityRebalanceRegion" in result:
        result["DensityRebalanceRegion"] = VERSION
    if "OriginalPatchID" in result:
        result["OriginalPatchID"] = patch_ids
    suffix = (
        f"{action}_copy{copy_index}"
        if copy_index is not None
        else action
    )
    result["PatchID"] = [
        f"{patch_id}__{scale}_{suffix}"
        for patch_id, scale in zip(patch_ids, scales)
    ]
    return result


def derive_crust_rows(
    rows: pd.DataFrame,
    action: str,
    sampler: tuple[cKDTree, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    result = apply_common_move(
        attach_horizon_bounds(rows, sampler),
        action,
    )
    top = pd.to_numeric(result["LocalTopTimeMs"], errors="coerce").to_numpy(dtype=float)
    mid = pd.to_numeric(result["LocalMidTimeMs"], errors="coerce").to_numpy(dtype=float)
    base = pd.to_numeric(result["LocalBaseTimeMs"], errors="coerce").to_numpy(dtype=float)
    original_time = pd.to_numeric(
        result["LayerDensityOriginalCenterTime"],
        errors="coerce",
    ).to_numpy(dtype=float)
    mapped_time, valid = map_time_to_crust(top, mid, base, original_time)
    result["CenterTime"] = mapped_time
    if "TimeWindowMin" in result:
        result["TimeWindowMin"] = mid
    if "TimeWindowMax" in result:
        result["TimeWindowMax"] = base
    if "LayerThickness" in result:
        result["LayerThickness"] = base - mid
    return result, valid


def main() -> int:
    args = parser().parse_args()
    if not (0.0 <= args.medium_move_fraction <= 1.0):
        raise ValueError("--medium-move-fraction must be in [0, 1]")
    if not (0.0 <= args.small_keep_fraction <= 1.0):
        raise ValueError("--small-keep-fraction must be in [0, 1]")
    if args.small_copy_count < 1:
        raise ValueError("--small-copy-count must be positive")
    if args.occupancy_cell_x_m <= 0.0 or args.occupancy_cell_time_ms <= 0.0:
        raise ValueError("occupancy cell sizes must be positive")

    started = time.time()
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    input_header = pd.read_csv(input_csv, nrows=0)
    if "LayerDensityBalanceVersion" in input_header:
        raise ValueError("input is already a layer-density balanced dataset")
    output_columns = list(input_header.columns)
    for column in NEW_COLUMNS:
        if column not in output_columns:
            output_columns.append(column)

    horizon_sampler = load_horizon_sampler(args.horizon_contract_csv.resolve())
    print(
        "[layer-balance] horizon sampler "
        f"points={len(horizon_sampler[1])}",
        flush=True,
    )
    (
        medium_ids,
        small_ids,
        _source_crust_xz_occupied,
        _source_crust_yz_occupied,
        first_summary,
    ) = first_pass(input_csv, args)
    (
        xz_occupied,
        yz_occupied,
        occupancy_reference_rows,
    ) = load_visible_occupancy(
        args.occupancy_reference_csv.resolve(),
        cell_x_m=float(args.occupancy_cell_x_m),
        cell_time_ms=float(args.occupancy_cell_time_ms),
    )
    first_summary["visible_occupancy_reference"] = {
        "path": str(args.occupancy_reference_csv.resolve()),
        "small_intersection_rows": int(occupancy_reference_rows),
        "xz_cells": int(len(xz_occupied)),
        "yz_cells": int(len(yz_occupied)),
    }
    medium_move_count = int(round(len(medium_ids) * float(args.medium_move_fraction)))
    small_keep_count = int(round(len(small_ids) * float(args.small_keep_fraction)))
    moved_medium = choose_subset(
        medium_ids,
        medium_move_count,
        "medium_to_crust",
    )
    kept_small = choose_subset(
        small_ids,
        small_keep_count,
        "small_keep_composite",
    )
    moved_small = set(small_ids) - kept_small
    print(
        "[layer-balance] selected "
        f"medium={len(moved_medium)}/{len(medium_ids)} "
        f"small_moved={len(moved_small)}/{len(small_ids)}",
        flush=True,
    )

    rng = np.random.default_rng(int(args.seed))
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    first_write = True
    output_rows = 0
    moved_medium_rows = 0
    restored_invalid_medium_rows = 0
    small_parents_considered = 0
    small_parents_primary_cell_occupied = 0
    small_parents_invalid_bounds = 0
    restored_invalid_small_rows = 0
    small_copies_written = 0
    small_copies_skipped_placement = 0
    placement_attempts = 0
    chunk_count = 0
    for chunk_count, chunk in enumerate(
        pd.read_csv(input_csv, chunksize=int(args.chunksize), low_memory=False),
        1,
    ):
        missing_columns = [column for column in NEW_COLUMNS if column not in chunk]
        if missing_columns:
            additions = pd.DataFrame(
                {
                    column: pd.Series(pd.NA, index=chunk.index, dtype="object")
                    for column in missing_columns
                },
                index=chunk.index,
            )
            chunk = pd.concat([chunk, additions], axis=1)
        row_keys = row_key_series(chunk)
        medium_mask = row_keys.isin(moved_medium).to_numpy()
        small_mask = row_keys.isin(moved_small).to_numpy()
        retained = chunk.loc[~small_mask & ~medium_mask].copy()

        moved_medium_frame = chunk.loc[medium_mask].copy()
        if len(moved_medium_frame):
            moved_medium_frame, valid_medium = derive_crust_rows(
                moved_medium_frame,
                "medium_move_to_crust",
                horizon_sampler,
            )
            invalid_medium_mask = medium_mask.copy()
            invalid_medium_mask[medium_mask] &= ~valid_medium
            retained = pd.concat(
                [
                    retained,
                    moved_medium_frame.loc[valid_medium],
                    chunk.loc[invalid_medium_mask],
                ],
                ignore_index=False,
            )
            moved_medium_rows += int(valid_medium.sum())
            restored_invalid_medium_rows += int((~valid_medium).sum())

        small_parents = chunk.loc[small_mask].copy()
        small_parents_considered += int(len(small_parents))
        accepted_copies: list[pd.DataFrame] = []
        if len(small_parents):
            derived_parents, valid_bounds = derive_crust_rows(
                small_parents,
                "small_move_to_crust",
                horizon_sampler,
            )
            small_parents_invalid_bounds += int((~valid_bounds).sum())
            if (~valid_bounds).any():
                invalid_small_mask = small_mask.copy()
                invalid_small_mask[small_mask] &= ~valid_bounds
                retained = pd.concat(
                    [
                        retained,
                        chunk.loc[invalid_small_mask],
                    ],
                    ignore_index=False,
                )
                restored_invalid_small_rows += int((~valid_bounds).sum())
            derived_parents = derived_parents.loc[valid_bounds].reset_index(drop=True)
            for row in derived_parents.itertuples(index=False):
                mapped_xz, mapped_yz = occupied_cell(
                    float(row.CenterX),
                    float(row.CenterY),
                    float(row.CenterTime),
                    float(args.occupancy_cell_x_m),
                    float(args.occupancy_cell_time_ms),
                )
                if mapped_xz in xz_occupied or mapped_yz in yz_occupied:
                    small_parents_primary_cell_occupied += 1
            if len(derived_parents):
                parents = derived_parents
                repeated_positions = np.repeat(
                    np.arange(len(parents), dtype=int),
                    int(args.small_copy_count),
                )
                copies = parents.iloc[repeated_positions].copy().reset_index(drop=True)
                copies["LayerDensityBalanceCopyIndex"] = np.tile(
                    np.arange(int(args.small_copy_count), dtype=int),
                    len(parents),
                )
                copies["PatchID"] = [
                    f"{parent}__small_3x_copy{copy_index}"
                    for parent, copy_index in zip(
                        copies["LayerDensityBalanceParentPatchID"].astype(str),
                        copies["LayerDensityBalanceCopyIndex"],
                    )
                ]
                direction = rng.uniform(0.0, 2.0 * np.pi, len(copies))
                radius = np.clip(
                    np.abs(
                        rng.normal(
                            0.0,
                            float(args.horizontal_jitter_sigma_m),
                            len(copies),
                        )
                    ),
                    3.0,
                    float(args.horizontal_jitter_max_m),
                )
                copies["CenterX"] = (
                    copies["CenterX"].to_numpy(dtype=float) + radius * np.cos(direction)
                )
                copies["CenterY"] = (
                    copies["CenterY"].to_numpy(dtype=float) + radius * np.sin(direction)
                )
                copies["CenterTime"] = (
                    copies["CenterTime"].to_numpy(dtype=float)
                    + np.clip(
                        rng.normal(
                            0.0,
                            float(args.time_jitter_sigma_ms),
                            len(copies),
                        ),
                        -float(args.time_jitter_max_ms),
                        float(args.time_jitter_max_ms),
                    )
                )
                copies["DipAzimuthDeg"] = (
                    copies["DipAzimuthDeg"].to_numpy(dtype=float)
                    + rng.normal(
                        0.0,
                        float(args.azimuth_jitter_sigma_deg),
                        len(copies),
                    )
                ) % 360.0
                copies["DipDeg"] = np.clip(
                    copies["DipDeg"].to_numpy(dtype=float)
                    + rng.normal(
                        0.0,
                        float(args.dip_jitter_sigma_deg),
                        len(copies),
                    ),
                    1.0,
                    89.0,
                )
                accepted_positions: list[int] = []
                neighbor_offsets = [
                    (dx, dy, dt)
                    for dx in range(-2, 3)
                    for dy in range(-2, 3)
                    for dt in range(-1, 2)
                    if dx != 0 or dy != 0 or dt != 0
                ]
                for position, row in enumerate(copies.itertuples(index=False)):
                    base_x = float(row.CenterX)
                    base_y = float(row.CenterY)
                    base_t = float(row.CenterTime)
                    base_x_cell = int(
                        np.floor(base_x / float(args.occupancy_cell_x_m))
                    )
                    base_y_cell = int(
                        np.floor(base_y / float(args.occupancy_cell_x_m))
                    )
                    base_t_cell = int(
                        np.floor(base_t / float(args.occupancy_cell_time_ms))
                    )
                    shuffled_offsets = neighbor_offsets.copy()
                    rng.shuffle(shuffled_offsets)
                    for attempt in range(int(args.max_placement_attempts)):
                        placement_attempts += 1
                        if attempt == 0:
                            candidate_x = base_x
                            candidate_y = base_y
                            candidate_t = base_t
                        else:
                            offset_index = attempt - 1
                            if offset_index >= len(shuffled_offsets):
                                break
                            dx, dy, dt = shuffled_offsets[offset_index]
                            candidate_x = (
                                base_x_cell
                                + dx
                                + float(rng.uniform(0.15, 0.85))
                            ) * float(args.occupancy_cell_x_m)
                            candidate_y = (
                                base_y_cell
                                + dy
                                + float(rng.uniform(0.15, 0.85))
                            ) * float(args.occupancy_cell_x_m)
                            candidate_t = np.clip(
                                (
                                    base_t_cell
                                    + dt
                                    + float(rng.uniform(0.15, 0.85))
                                )
                                * float(args.occupancy_cell_time_ms),
                                float(row.LocalMidTimeMs),
                                float(row.LocalBaseTimeMs),
                            )
                        xz_cell, yz_cell = occupied_cell(
                            candidate_x,
                            candidate_y,
                            candidate_t,
                            float(args.occupancy_cell_x_m),
                            float(args.occupancy_cell_time_ms),
                        )
                        if xz_cell not in xz_occupied and yz_cell not in yz_occupied:
                            copies.at[position, "CenterX"] = candidate_x
                            copies.at[position, "CenterY"] = candidate_y
                            copies.at[position, "CenterTime"] = candidate_t
                            accepted_positions.append(position)
                            break
                small_copies_skipped_placement += int(
                    len(copies) - len(accepted_positions)
                )
                if accepted_positions:
                    accepted_copies.append(copies.iloc[accepted_positions].copy())
        if accepted_copies:
            retained = pd.concat(
                [retained, *accepted_copies],
                ignore_index=False,
            )
            small_copies_written += int(sum(len(frame) for frame in accepted_copies))

        retained = retained.reindex(columns=output_columns)
        retained.to_csv(
            temporary_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig" if first_write else "utf-8",
            lineterminator="\n",
        )
        first_write = False
        output_rows += int(len(retained))
        if chunk_count % 10 == 0:
            print(
                "[layer-balance] write-pass "
                f"chunks={chunk_count} rows={output_rows} "
                f"small_parents={small_parents_considered} "
                f"small_copies={small_copies_written}",
                flush=True,
            )

    temporary_csv.replace(output_csv)
    summary: dict[str, Any] = {
        "status": "pass",
        "version": VERSION,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "target_block": {
            "x_m": [float(args.x_min), float(args.x_max)],
            "y_m": [float(args.y_min), float(args.y_max)],
        },
        "rules": {
            "medium_move_fraction": float(args.medium_move_fraction),
            "small_keep_fraction": float(args.small_keep_fraction),
            "small_copy_count": int(args.small_copy_count),
            "occupancy_cell_x_m": float(args.occupancy_cell_x_m),
            "occupancy_cell_time_ms": float(args.occupancy_cell_time_ms),
            "protected_imaging_and_well_control": True,
            "move_time_mapping": "relative depth inside upper layer mapped to 10-90% of crust layer",
            "geometry_policy": "CSV center/dip/azimuth; V1-V4 vertices preserved but not used by Step9",
        },
        "first_pass": first_summary,
        "selection": {
            "moved_medium_selected": int(len(moved_medium)),
            "small_kept_in_composite": int(len(kept_small)),
            "small_moved_selected": int(len(moved_small)),
        },
        "execution": {
            "moved_medium_rows_written": int(moved_medium_rows),
            "restored_invalid_medium_rows": int(restored_invalid_medium_rows),
            "small_parents_considered": int(small_parents_considered),
            "small_parents_primary_cell_occupied": int(
                small_parents_primary_cell_occupied
            ),
            "small_parents_invalid_bounds": int(small_parents_invalid_bounds),
            "restored_invalid_small_rows": int(restored_invalid_small_rows),
            "small_copies_written": int(small_copies_written),
            "small_copies_skipped_placement": int(small_copies_skipped_placement),
            "placement_attempts": int(placement_attempts),
        },
        "jitter": {
            "horizontal_sigma_m": float(args.horizontal_jitter_sigma_m),
            "horizontal_max_m": float(args.horizontal_jitter_max_m),
            "time_sigma_ms": float(args.time_jitter_sigma_ms),
            "time_max_ms": float(args.time_jitter_max_ms),
            "azimuth_sigma_deg": float(args.azimuth_jitter_sigma_deg),
            "dip_sigma_deg": float(args.dip_jitter_sigma_deg),
            "seed": int(args.seed),
        },
        "output_rows": int(output_rows),
        "elapsed_seconds": float(time.time() - started),
        "checks": {
            "selection_completed": len(moved_medium) == medium_move_count
            and len(kept_small) == small_keep_count,
            "all_selected_medium_rows_accounted": (
                moved_medium_rows + restored_invalid_medium_rows
                == len(moved_medium)
            ),
            "three_copies_for_non_skipped_small_parents": small_copies_written
            == int(args.small_copy_count)
            * (
                small_parents_considered
                - small_parents_invalid_bounds
            )
            - small_copies_skipped_placement,
            "output_rows_match": output_rows
            == first_summary["input_total_rows"]
            - len(moved_medium)
            + moved_medium_rows
            + restored_invalid_medium_rows
            - len(moved_small)
            + restored_invalid_small_rows
            + small_copies_written,
        },
    }
    if not all(summary["checks"].values()):
        summary["status"] = "fail"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[layer-balance] status={summary['status']} "
        f"rows={output_rows} output={output_csv}",
        flush=True,
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
