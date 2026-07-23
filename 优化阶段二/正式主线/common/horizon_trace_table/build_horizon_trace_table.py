# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


TABLE_DTYPE = np.dtype(
    [
        ("TraceIdx", "<i4"),
        ("T4", "<f4"),
        ("T5", "<f4"),
        ("T6", "<f4"),
        ("T7", "<f4"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable per-seismic-trace T4-T7 horizon table.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def required_path(config: dict[str, Any], key: str) -> Path:
    path = Path(str(config[key])).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def load_trace_grid(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trace_df = pd.read_csv(path, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    for column in ["TraceIdx", "X", "Y"]:
        trace_df[column] = pd.to_numeric(trace_df[column], errors="raise")
    trace_df = trace_df.sort_values("TraceIdx").reset_index(drop=True)
    trace_idx = trace_df["TraceIdx"].to_numpy(dtype=np.int64)
    expected = np.arange(len(trace_df), dtype=np.int64)
    if not np.array_equal(trace_idx, expected):
        raise ValueError("TraceIdx must be contiguous, zero-based, and sorted")
    x_values = np.sort(trace_df["X"].unique()).astype(np.float64)
    y_values = np.sort(trace_df["Y"].unique()).astype(np.float64)
    if len(x_values) * len(y_values) != len(trace_df):
        raise ValueError("trace headers do not form a complete Cartesian XY grid")
    trace_x = trace_df["X"].to_numpy(dtype=np.float64)
    trace_y = trace_df["Y"].to_numpy(dtype=np.float64)
    ix = np.searchsorted(x_values, trace_x)
    iy = np.searchsorted(y_values, trace_y)
    if not np.allclose(x_values[ix], trace_x) or not np.allclose(y_values[iy], trace_y):
        raise ValueError("trace XY values do not map exactly to the derived axes")
    grid_trace_idx = np.full((len(x_values), len(y_values)), -1, dtype=np.int32)
    grid_trace_idx[ix, iy] = trace_idx.astype(np.int32)
    if np.any(grid_trace_idx < 0):
        raise ValueError("trace XY grid contains missing cells")
    return trace_df, x_values, y_values, ix, iy


def nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.searchsorted(axis, values, side="left")
    right = np.clip(right, 0, len(axis) - 1)
    left = np.clip(right - 1, 0, len(axis) - 1)
    choose_left = np.abs(values - axis[left]) <= np.abs(values - axis[right])
    return np.where(choose_left, left, right).astype(np.int64)


def map_full_grid_horizon(
    path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    trace_ix: np.ndarray,
    trace_iy: np.ndarray,
    chunksize: int,
    coordinate_tolerance_m: float,
    code: str,
    expected_source_rows: int,
    show_progress: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    grid = np.full((len(x_values), len(y_values)), np.nan, dtype=np.float32)
    parsed_count = 0
    duplicate_count = 0
    max_coordinate_error = 0.0
    reader = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["X", "Y", "TIME"],
        dtype=np.float64,
        chunksize=chunksize,
    )
    total_chunks = math.ceil(expected_source_rows / chunksize) if expected_source_rows > 0 else None
    chunk_iterator = reader
    if tqdm is not None and show_progress:
        chunk_iterator = tqdm(reader, total=total_chunks, desc=f"{code} source mapping", unit="chunk")
    for chunk_index, chunk in enumerate(chunk_iterator, start=1):
        xyz = chunk[["X", "Y", "TIME"]].to_numpy(dtype=np.float64)
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        ix = nearest_axis_indices(x_values, xyz[:, 0])
        iy = nearest_axis_indices(y_values, xyz[:, 1])
        errors = np.hypot(xyz[:, 0] - x_values[ix], xyz[:, 1] - y_values[iy])
        if errors.size:
            max_coordinate_error = max(max_coordinate_error, float(np.max(errors)))
        if np.any(errors > coordinate_tolerance_m):
            raise ValueError(
                f"{code} contains horizon points farther than {coordinate_tolerance_m:g} m from the trace grid"
            )
        duplicate_count += int(np.isfinite(grid[ix, iy]).sum())
        grid[ix, iy] = xyz[:, 2].astype(np.float32)
        parsed_count += int(len(xyz))
        if tqdm is None or not show_progress:
            print(f"[horizon-table] {code} parsed chunks={chunk_index} rows={parsed_count}", flush=True)

    finite_before_fill = int(np.isfinite(grid).sum())
    if finite_before_fill == 0:
        raise ValueError(f"{code} contains no valid horizon values")
    missing = ~np.isfinite(grid)
    if np.any(missing):
        nearest_indices = distance_transform_edt(missing, return_distances=False, return_indices=True)
        grid[missing] = grid[tuple(nearest_indices[:, missing])]
    values = grid[trace_ix, trace_iy].astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{code} nearest-grid fill left non-finite values")
    return values, {
        "source_point_count": parsed_count,
        "mapped_grid_count": finite_before_fill,
        "nearest_filled_trace_count": int(len(values) - finite_before_fill),
        "duplicate_grid_assignment_count": duplicate_count,
        "max_coordinate_error_m": max_coordinate_error,
    }


def load_t4_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["X", "Y", "TIME"],
        dtype=np.float64,
    )
    xyz = frame[["X", "Y", "TIME"]].to_numpy(dtype=np.float64)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) == 0:
        raise ValueError(f"T4 contains no valid XYZ rows: {path}")
    source = pd.DataFrame(xyz, columns=["X", "Y", "TIME"]).groupby(["X", "Y"], as_index=False)["TIME"].mean()
    return source[["X", "Y"]].to_numpy(dtype=np.float64), source["TIME"].to_numpy(dtype=np.float64)


def interpolate_t4_idw(
    source_xy: np.ndarray,
    source_time: np.ndarray,
    target_xy: np.ndarray,
    neighbors: int,
    power: float,
    chunksize: int,
    workers: int,
    show_progress: bool,
) -> np.ndarray:
    tree = cKDTree(source_xy)
    k = min(max(1, neighbors), len(source_xy))
    output = np.empty(len(target_xy), dtype=np.float32)
    starts = range(0, len(target_xy), chunksize)
    start_iterator = starts
    if tqdm is not None and show_progress:
        start_iterator = tqdm(starts, total=math.ceil(len(target_xy) / chunksize), desc="T4 IDW interpolation", unit="chunk")
    for start in start_iterator:
        stop = min(start + chunksize, len(target_xy))
        distances, indices = tree.query(target_xy[start:stop], k=k, workers=workers)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        values = source_time[np.asarray(indices, dtype=np.int64)]
        exact = distances[:, 0] <= 1.0e-8
        weights = 1.0 / np.maximum(distances, 1.0e-6) ** power
        interpolated = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
        interpolated[exact] = values[exact, 0]
        output[start:stop] = interpolated.astype(np.float32)
        if tqdm is None or not show_progress:
            print(f"[horizon-table] T4 interpolated traces={stop}/{len(target_xy)}", flush=True)
    if not np.isfinite(output).all():
        raise RuntimeError("T4 interpolation produced non-finite values")
    return output


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def main() -> int:
    cli = parse_args()
    config_path = cli.config.resolve()
    config = read_json(config_path)
    trace_header_path = required_path(config, "trace_header_csv")
    horizon_paths = {code: required_path(config["horizon_paths"], code) for code in ["T4", "T5", "T6", "T7"]}
    output_dir = Path(str(config["output_dir"])).resolve()
    output_path = output_dir / str(config.get("output_filename", "horizon_trace_table.npy"))
    metadata_path = output_dir / "horizon_trace_table_metadata.json"

    trace_df, x_values, y_values, trace_ix, trace_iy = load_trace_grid(trace_header_path)
    print(
        f"[horizon-table] traces={len(trace_df)} grid={len(x_values)}x{len(y_values)} "
        f"X=({x_values[0]}, {x_values[-1]}) Y=({y_values[0]}, {y_values[-1]})",
        flush=True,
    )
    if cli.check_only:
        print("[horizon-table] check-only=pass", flush=True)
        return 0

    chunksize = int(config.get("read_chunksize", 250000))
    coordinate_tolerance_m = float(config.get("coordinate_tolerance_m", 2.0))
    show_progress = bool(config.get("show_progress", True))
    expected_source_rows = {str(key): int(value) for key, value in dict(config.get("expected_source_rows", {})).items()}
    layers: dict[str, np.ndarray] = {}
    mapping_summary: dict[str, Any] = {}
    target_xy = trace_df[["X", "Y"]].to_numpy(dtype=np.float64)

    t4_xy, t4_time = load_t4_points(horizon_paths["T4"])
    layers["T4"] = interpolate_t4_idw(
        t4_xy,
        t4_time,
        target_xy,
        neighbors=int(config.get("t4_idw_neighbors", 8)),
        power=float(config.get("t4_idw_power", 2.0)),
        chunksize=int(config.get("t4_query_chunksize", 100000)),
        workers=int(config.get("t4_query_workers", -1)),
        show_progress=show_progress,
    )
    mapping_summary["T4"] = {"source_point_count": int(len(t4_xy)), "method": "IDW"}

    for code in ["T5", "T6", "T7"]:
        layers[code], mapping_summary[code] = map_full_grid_horizon(
            horizon_paths[code],
            x_values,
            y_values,
            trace_ix,
            trace_iy,
            chunksize,
            coordinate_tolerance_m,
            code,
            expected_source_rows.get(code, 0),
            show_progress,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    table = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=TABLE_DTYPE, shape=(len(trace_df),))
    table["TraceIdx"] = trace_df["TraceIdx"].to_numpy(dtype=np.int32)
    for code in ["T4", "T5", "T6", "T7"]:
        table[code] = layers[code]
    table.flush()
    del table
    os.replace(temporary_path, output_path)

    metadata = {
        "version": "horizon_trace_table_v1",
        "config_path": str(config_path),
        "table_path": str(output_path),
        "table_format": "NumPy structured NPY; memory-map compatible",
        "row_count": int(len(trace_df)),
        "fields": {name: str(TABLE_DTYPE.fields[name][0]) for name in TABLE_DTYPE.names or []},
        "trace_index_contract": "rows are sorted by TraceIdx and row index equals TraceIdx",
        "time_unit": "TWT_ms",
        "trace_header": file_metadata(trace_header_path),
        "horizon_sources": {code: file_metadata(path) for code, path in horizon_paths.items()},
        "methods": {
            "T4": {
                "method": "inverse_distance_weighted_interpolation",
                "neighbors": int(config.get("t4_idw_neighbors", 8)),
                "power": float(config.get("t4_idw_power", 2.0)),
            },
            "T5_T6_T7": "map to nearest trace-grid coordinate within tolerance, then fill missing cells from nearest available horizon grid cell",
        },
        "mapping_counts": mapping_summary,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_order = (layers["T4"] < layers["T5"]) & (layers["T5"] < layers["T6"]) & (layers["T6"] < layers["T7"])
    print(f"[horizon-table] output={output_path}", flush=True)
    print(f"[horizon-table] metadata={metadata_path}", flush=True)
    print(f"[horizon-table] ordered_trace_count={int(valid_order.sum())}/{len(valid_order)}", flush=True)
    print("[horizon-table] status=pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
