"""Shared 5 km section-window and DFN time-extent helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERTEX_Z_COLUMNS = [f"V{i}Z" for i in range(1, 5)]


def centered_window(
    well_df: pd.DataFrame,
    size_m: float,
    *,
    outer_bounds: dict[str, Any] | None = None,
    fit_to_outer_bounds: bool = False,
    name: str = "well_centered_5km",
) -> dict[str, float | str]:
    if well_df.empty:
        raise ValueError("cannot build section window from an empty well trajectory")
    size_m = float(size_m)
    if size_m <= 0.0:
        raise ValueError("section window size must be positive")
    x_min = float(well_df["X"].min())
    x_max = float(well_df["X"].max())
    y_min = float(well_df["Y"].min())
    y_max = float(well_df["Y"].max())
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    block = {
        "name": name,
        "requested_size_m": size_m,
        "x_min": center_x - 0.5 * size_m,
        "x_max": center_x + 0.5 * size_m,
        "y_min": center_y - 0.5 * size_m,
        "y_max": center_y + 0.5 * size_m,
    }
    if outer_bounds:
        for axis in ("x", "y"):
            block_min_key = f"{axis}_min"
            block_max_key = f"{axis}_max"
            outer_min = float(outer_bounds[block_min_key])
            outer_max = float(outer_bounds[block_max_key])
            if fit_to_outer_bounds and outer_max - outer_min >= size_m:
                if float(block[block_min_key]) < outer_min:
                    shift = outer_min - float(block[block_min_key])
                    block[block_min_key] = outer_min
                    block[block_max_key] = float(block[block_max_key]) + shift
                elif float(block[block_max_key]) > outer_max:
                    shift = outer_max - float(block[block_max_key])
                    block[block_min_key] = float(block[block_min_key]) + shift
                    block[block_max_key] = outer_max
            else:
                block[block_min_key] = max(float(block[block_min_key]), outer_min)
                block[block_max_key] = min(float(block[block_max_key]), outer_max)
        block["center_shift_x_m"] = 0.5 * (
            float(block["x_min"]) + float(block["x_max"])
        ) - center_x
        block["center_shift_y_m"] = 0.5 * (
            float(block["y_min"]) + float(block["y_max"])
        ) - center_y
    if float(block["x_max"]) <= float(block["x_min"]) or float(block["y_max"]) <= float(block["y_min"]):
        raise ValueError(f"invalid centered section window: {block}")
    return block


def dfn_time_extent_in_block(
    csv_path: Path,
    block: dict[str, Any],
    *,
    padding_ms: float = 20.0,
    sample_interval_ms: float = 2.0,
) -> tuple[float, float, dict[str, Any]]:
    """Return aligned top/bottom time bounds for predicted DFN in a map block."""

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = {"CenterX", "CenterY", "CenterTime"}
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"DFN CSV missing required columns: {missing}")
    usecols = [column for column in [
        "CenterX", "CenterY", "CenterTime", "HeightTimeMs", "LayerGroup", "FractureScale",
        *VERTEX_Z_COLUMNS,
    ] if column in header]
    frame = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    for column in usecols:
        if column not in {"LayerGroup", "FractureScale"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["CenterX"].between(float(block["x_min"]), float(block["x_max"]))
        & frame["CenterY"].between(float(block["y_min"]), float(block["y_max"]))
        & frame["CenterTime"].notna()
    ].copy()
    if frame.empty:
        raise RuntimeError("no DFN patches fall inside the requested section window")

    available_z = [column for column in VERTEX_Z_COLUMNS if column in frame.columns]
    if available_z:
        z_values = frame[available_z].to_numpy(dtype=float).reshape(-1)
        z_values = z_values[np.isfinite(z_values)]
    else:
        if "HeightTimeMs" not in frame.columns:
            z_values = frame["CenterTime"].to_numpy(dtype=float)
        else:
            half_height = 0.5 * pd.to_numeric(frame["HeightTimeMs"], errors="coerce").fillna(0.0)
            z_values = np.concatenate(
                [
                    frame["CenterTime"].to_numpy(dtype=float) - half_height.to_numpy(dtype=float),
                    frame["CenterTime"].to_numpy(dtype=float) + half_height.to_numpy(dtype=float),
                ]
            )
    if not len(z_values):
        raise RuntimeError("DFN patches in section window have no finite Z extents")
    raw_top = float(np.min(z_values))
    raw_bottom = float(np.max(z_values))
    interval = max(float(sample_interval_ms), 1.0e-9)
    padded_top = raw_top - float(padding_ms)
    padded_bottom = raw_bottom + float(padding_ms)
    aligned_top = math.floor(padded_top / interval) * interval
    aligned_bottom = math.ceil(padded_bottom / interval) * interval
    layers = {
        str(key): int(value)
        for key, value in frame.get("LayerGroup", pd.Series(dtype=str)).astype(str).value_counts().items()
    }
    scales = {
        str(key): int(value)
        for key, value in frame.get("FractureScale", pd.Series(dtype=str)).astype(str).value_counts().items()
    }
    summary = {
        "csv": str(csv_path),
        "map_block": block,
        "patch_count_in_block": int(len(frame)),
        "layer_counts": layers,
        "scale_counts": scales,
        "dfn_raw_top_ms": raw_top,
        "dfn_raw_bottom_ms": raw_bottom,
        "padding_ms": float(padding_ms),
        "sample_interval_ms": interval,
        "aligned_top_ms": aligned_top,
        "aligned_bottom_ms": aligned_bottom,
        "extent_source": "DFN vertex Z" if available_z else "CenterTime +/- HeightTimeMs/2",
        "original_faults_included": False,
    }
    return aligned_top, aligned_bottom, summary
