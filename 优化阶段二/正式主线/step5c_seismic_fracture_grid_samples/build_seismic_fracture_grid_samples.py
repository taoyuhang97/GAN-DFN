#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import segyio


CORNER_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)
CORNER_LABELS = tuple(f"{dx}{dy}{dt}" for dx, dy, dt in CORNER_OFFSETS)
ATTRIBUTE_NAMES = ("AntTrack", "Coherence", "CurvatureMax")
POINT_COLUMNS = (
    "X",
    "Y",
    "TIME",
    "PresenceLabel",
    "DensityLabel",
    "PointConfidence",
)
OUTPUT_COLUMNS = (
    "FractureDensity",
    *(f"{name}_{label}" for name in ATTRIBUTE_NAMES for label in CORNER_LABELS),
)


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


class SegyVolume:
    def __init__(self, name: str, path: Path, time_origin_ms: float | None, rule: dict[str, float] | None):
        self.name = str(name)
        self.path = Path(path).resolve()
        self.handle = segyio.open(str(self.path), "r", ignore_geometry=True)
        self.handle.mmap()
        raw_samples = np.asarray(self.handle.samples, dtype=np.float64)
        self.samples = raw_samples if time_origin_ms is None else raw_samples - raw_samples[0] + float(time_origin_ms)
        self.rule = rule or {}

    def close(self) -> None:
        self.handle.close()

    def trace_at(self, trace_index: int) -> np.ndarray:
        return np.asarray(self.handle.trace[int(trace_index)], dtype=np.float64)

    def clean(self, values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float64).copy()
        out[~np.isfinite(out)] = np.nan
        if self.rule:
            out[(out < float(self.rule["min"])) | (out > float(self.rule["max"]))] = np.nan
        return out


class SandGrid:
    def __init__(self, config: dict[str, Any], volumes: dict[str, SegyVolume]):
        mapping_path = Path(config["trace_mapping_npz"]).resolve()
        with np.load(mapping_path, allow_pickle=False) as packed:
            ix = packed["ix"].astype(np.int64)
            iy = packed["iy"].astype(np.int64)
            x = packed["x"].astype(float)
            y = packed["y"].astype(float)
            source_trace_idx = packed["source_trace_idx"].astype(np.int64)
        self.nx = int(ix.max()) + 1
        self.ny = int(iy.max()) + 1
        self.x_axis = np.full(self.nx, np.nan, dtype=float)
        self.y_axis = np.full(self.ny, np.nan, dtype=float)
        self.x_axis[ix] = x
        self.y_axis[iy] = y
        if not np.isfinite(self.x_axis).all() or not np.isfinite(self.y_axis).all():
            raise ValueError("sand trace mapping does not cover a complete X/Y grid")
        if not np.all(np.diff(self.x_axis) > 0) or not np.all(np.diff(self.y_axis) > 0):
            raise ValueError("sand trace mapping axes must be strictly increasing")
        self.trace_by_ij = np.full((self.ny, self.nx), -1, dtype=np.int64)
        self.trace_by_ij[iy, ix] = source_trace_idx
        if np.any(self.trace_by_ij < 0):
            raise ValueError("sand trace mapping contains missing grid nodes")
        self.t_axis = volumes["CurvatureMax"].samples.copy()
        if len(self.t_axis) < 2 or not np.allclose(np.diff(self.t_axis), 2.0):
            raise ValueError("sand corner grid requires the native 2 ms curvature axis")
        self.nt = len(self.t_axis)

    def map_points(self, x: np.ndarray, y: np.ndarray, time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ix0 = np.searchsorted(self.x_axis, x, side="right") - 1
        iy0 = np.searchsorted(self.y_axis, y, side="right") - 1
        it0 = np.searchsorted(self.t_axis, time_ms, side="right") - 1
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(time_ms)
            & (ix0 >= 0)
            & (ix0 < self.nx - 1)
            & (iy0 >= 0)
            & (iy0 < self.ny - 1)
            & (it0 >= 0)
            & (it0 < self.nt - 1)
        )
        return ix0.astype(np.int64), iy0.astype(np.int64), it0.astype(np.int64), valid, np.full(len(x), np.nan)

    def corner_trace_indices(self, ix0: np.ndarray, iy0: np.ndarray, it0: np.ndarray) -> np.ndarray:
        result = np.empty((len(ix0), 8), dtype=np.int64)
        for index, (dx, dy, _) in enumerate(CORNER_OFFSETS):
            result[:, index] = self.trace_by_ij[iy0 + dy, ix0 + dx]
        return result

    def corner_times(self, it0: np.ndarray) -> np.ndarray:
        result = np.empty((len(it0), 8), dtype=np.float64)
        for index, (_, _, dt) in enumerate(CORNER_OFFSETS):
            result[:, index] = self.t_axis[it0 + dt]
        return result


class TaiguGrid:
    def __init__(self, config: dict[str, Any], volumes: dict[str, SegyVolume]):
        header = pd.read_csv(
            Path(config["trace_header_csv"]).resolve(),
            encoding="utf-8-sig",
            usecols=["TraceIdx", "X", "Y"],
        ).sort_values("TraceIdx")
        self.trace_indices = pd.to_numeric(header["TraceIdx"], errors="coerce").to_numpy(dtype=np.int64)
        self.trace_xy = header[["X", "Y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if len(self.trace_indices) != int(config["trace_count"]) or not np.array_equal(self.trace_indices, np.arange(len(self.trace_indices))):
            raise ValueError("Taigu attribute trace indexes are not contiguous")
        if not np.isfinite(self.trace_xy).all():
            raise ValueError("Taigu attribute trace coordinates contain non-finite values")
        self.origin_x = float(config["grid_origin_x_m"])
        self.origin_y = float(config["grid_origin_y_m"])
        self.spacing = float(config["grid_spacing_m"])
        self.nx = int(config["expected_grid_nx"])
        self.ny = int(config["expected_grid_ny"])
        self.coordinate_tolerance = float(config.get("coordinate_tolerance_m", 2.5))
        if self.nx * self.ny != int(config["trace_count"]):
            raise ValueError("Taigu expected grid dimensions do not match trace count")

        # The external trace header stores a nominal 12.5 m Cartesian lattice with
        # metre-level coordinate rounding. Recover each trace's lattice node from
        # its external X/Y instead of treating the nearest trace as a cell corner.
        ix = np.rint((self.trace_xy[:, 0] - self.origin_x) / self.spacing).astype(np.int64)
        iy = np.rint((self.trace_xy[:, 1] - self.origin_y) / self.spacing).astype(np.int64)
        residual_x = self.trace_xy[:, 0] - (self.origin_x + ix * self.spacing)
        residual_y = self.trace_xy[:, 1] - (self.origin_y + iy * self.spacing)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        residual_within_tolerance = (np.abs(residual_x) <= self.coordinate_tolerance) & (np.abs(residual_y) <= self.coordinate_tolerance)
        self.trace_by_ij = np.full((self.ny, self.nx), -1, dtype=np.int64)
        self.trace_by_ij[iy[inside], ix[inside]] = self.trace_indices[inside]
        node_count = int(np.count_nonzero(self.trace_by_ij >= 0))
        if not np.all(inside & residual_within_tolerance):
            raise ValueError(
                "Taigu trace coordinates cannot be uniquely assigned to the configured lattice: "
                f"inside={int(inside.sum())}/{len(inside)}, residual_max=({np.abs(residual_x).max():.3f}, {np.abs(residual_y).max():.3f}) m"
            )
        if node_count != len(self.trace_indices) or node_count != self.nx * self.ny:
            raise ValueError(
                "Taigu trace coordinates do not form a one-to-one complete lattice: "
                f"unique_nodes={node_count}, traces={len(self.trace_indices)}, expected={self.nx * self.ny}"
            )
        self.t_axis = volumes["AntTrack"].samples.copy()
        if len(self.t_axis) < 2 or not np.allclose(np.diff(self.t_axis), 1.0):
            raise ValueError("Taigu corner grid requires a native 1 ms attribute axis")
        self.nt = len(self.t_axis)

    def map_points(self, x: np.ndarray, y: np.ndarray, time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ix0 = np.floor((x - self.origin_x) / self.spacing).astype(np.int64, copy=False)
        iy0 = np.floor((y - self.origin_y) / self.spacing).astype(np.int64, copy=False)
        it0 = np.searchsorted(self.t_axis, time_ms, side="right") - 1
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(time_ms)
            & (ix0 >= 0)
            & (ix0 < self.nx - 1)
            & (iy0 >= 0)
            & (iy0 < self.ny - 1)
            & (it0 >= 0)
            & (it0 < self.nt - 1)
        )
        return ix0, iy0, it0.astype(np.int64), valid, np.full(len(x), np.nan)

    def corner_trace_indices(self, ix0: np.ndarray, iy0: np.ndarray, it0: np.ndarray) -> np.ndarray:
        result = np.empty((len(ix0), 8), dtype=np.int64)
        for index, (dx, dy, _) in enumerate(CORNER_OFFSETS):
            result[:, index] = self.trace_by_ij[iy0 + dy, ix0 + dx]
        return result

    def corner_times(self, it0: np.ndarray) -> np.ndarray:
        result = np.empty((len(it0), 8), dtype=np.float64)
        for index, (_, _, dt) in enumerate(CORNER_OFFSETS):
            result[:, index] = self.t_axis[it0 + dt]
        return result


def sample_volume_corners(volume: SegyVolume, trace_ids: np.ndarray, corner_times: np.ndarray) -> np.ndarray:
    flat_traces = trace_ids.reshape(-1)
    flat_times = corner_times.reshape(-1)
    positions = np.interp(
        flat_times,
        volume.samples,
        np.arange(len(volume.samples), dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    nearest = np.rint(positions).astype(np.int64, copy=False)
    valid = np.isfinite(positions) & (nearest >= 0) & (nearest < len(volume.samples))
    valid &= np.isclose(positions, nearest, atol=1.0e-6, equal_nan=False)
    values = np.full(len(flat_traces), np.nan, dtype=np.float64)
    for trace_index in np.unique(flat_traces[valid]):
        mask = valid & (flat_traces == trace_index)
        trace = volume.trace_at(int(trace_index))
        values[mask] = trace[nearest[mask]]
    return volume.clean(values.reshape(trace_ids.shape))


def aggregate_density(points: pd.DataFrame, ii: np.ndarray, jj: np.ndarray, it0: np.ndarray, valid: np.ndarray, nx: int, ny: int) -> pd.DataFrame:
    work = points.loc[valid].copy()
    ii = ii[valid]
    jj = jj[valid]
    it0 = it0[valid]
    if work.empty:
        return pd.DataFrame(columns=["IX", "IY", "IT", "FractureDensity", "PointCount"])
    presence = numeric(work, "PresenceLabel", 0.0)
    density_label = numeric(work, "DensityLabel", 0.0)
    confidence = numeric(work, "PointConfidence", 1.0)
    confidence = np.where(np.isfinite(confidence) & (confidence > 0), confidence, 1.0)
    positive_density = np.where((presence == 1.0) & np.isfinite(density_label) & (density_label >= 0), density_label, 0.0)
    keys = (it0.astype(np.int64) * int(ny) + jj.astype(np.int64)) * int(nx) + ii.astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    weight_sum = np.bincount(inverse, weights=confidence, minlength=len(unique_keys))
    density_sum = np.bincount(inverse, weights=confidence * positive_density, minlength=len(unique_keys))
    point_count = np.bincount(inverse, minlength=len(unique_keys))
    density = np.divide(density_sum, weight_sum, out=np.zeros_like(density_sum), where=weight_sum > 0)
    it0_out = unique_keys // (int(ny) * int(nx))
    remainder = unique_keys % (int(ny) * int(nx))
    jj_out = remainder // int(nx)
    ii_out = remainder % int(nx)
    return pd.DataFrame(
        {
            "IX": ii_out.astype(np.int64),
            "IY": jj_out.astype(np.int64),
            "IT": it0_out.astype(np.int64),
            "FractureDensity": np.maximum(density, 0.0),
            "PointCount": point_count.astype(np.int64),
        }
    ).sort_values(["IT", "IY", "IX"], kind="stable").reset_index(drop=True)


def build_sample_frame(points: pd.DataFrame, grid: SandGrid | TaiguGrid, volumes: dict[str, SegyVolume]) -> pd.DataFrame:
    x = numeric(points, "X")
    y = numeric(points, "Y")
    time_ms = numeric(points, "TIME")
    ii, jj, it0, valid, _ = grid.map_points(x, y, time_ms)
    nx = grid.nx
    ny = grid.ny
    density = aggregate_density(points, ii, jj, it0, valid, nx=nx, ny=ny)
    if density.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    cell_i = density["IX"].to_numpy(dtype=np.int64)
    cell_j = density["IY"].to_numpy(dtype=np.int64)
    cell_k = density["IT"].to_numpy(dtype=np.int64)
    trace_ids = grid.corner_trace_indices(cell_i, cell_j, cell_k)
    times = grid.corner_times(cell_k)
    output = pd.DataFrame({"FractureDensity": density["FractureDensity"].to_numpy(dtype=float)})
    for name in ATTRIBUTE_NAMES:
        values = sample_volume_corners(volumes[name], trace_ids, times)
        for index, label in enumerate(CORNER_LABELS):
            output[f"{name}_{label}"] = values[:, index]
    return output.loc[:, OUTPUT_COLUMNS]


def validate_output_frame(frame: pd.DataFrame, path: Path) -> None:
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{path}: output columns do not match the fixed 25-column contract")
    if frame.shape[1] != 25:
        raise ValueError(f"{path}: expected 25 columns, found {frame.shape[1]}")


def write_sample_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    validate_output_frame(frame, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"sample file already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.10g")
    temporary.replace(path)


def discover_virtual_files(well_root: Path, config: dict[str, Any]) -> list[tuple[str, Path]]:
    virtual_root = well_root / "virtual_wells"
    virtual_dirs = sorted((path for path in virtual_root.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name))
    expected = int(config.get("virtual_per_source", 25))
    if len(virtual_dirs) != expected:
        raise RuntimeError(f"{well_root.name}: expected {expected} virtual wells, found {len(virtual_dirs)}")
    result: list[tuple[str, Path]] = []
    for virtual_dir in virtual_dirs:
        path = well_root / str(config["virtual_relative_template"]).format(virtual_well_name=virtual_dir.name)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing virtual sample file: {path}")
        result.append((virtual_dir.name, path))
    return result


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 25-column seismic-grid fracture samples for real and virtual wells.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-well", action="append", default=[], help="Optional source-well filter for smoke tests.")
    parser.add_argument("--max-source-wells", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.monotonic()
    config_path = args.config.resolve()
    config = read_json(config_path)
    source_wells = [str(value) for value in config["source_wells"]]
    if args.source_well:
        requested = set(str(value) for value in args.source_well)
        missing = requested.difference(source_wells)
        if missing:
            raise ValueError(f"source wells not present in config: {sorted(missing)}")
        source_wells = [well for well in source_wells if well in requested]
    if args.max_source_wells > 0:
        source_wells = source_wells[: int(args.max_source_wells)]
    expected_source_count = int(config["expected_source_wells"])
    if not args.source_well and args.max_source_wells == 0 and len(source_wells) != expected_source_count:
        raise ValueError(f"config must contain {expected_source_count} source wells, found {len(source_wells)}")

    output_root = (args.output_root or Path(config["output_root"])).resolve()
    wells_root = Path(config["wells_root"]).resolve()
    volumes = {
        name: SegyVolume(
            name=name,
            path=Path(config["volume_paths"][name]),
            time_origin_ms=(config.get("time_origins_ms") or {}).get(name),
            rule=(config.get("attribute_rules") or {}).get(name),
        )
        for name in ATTRIBUTE_NAMES
    }
    grid: SandGrid | TaiguGrid
    if config["grid_type"] == "sand_trace_mapping":
        grid = SandGrid(config, volumes)
    elif config["grid_type"] == "taigu_inline_xline":
        grid = TaiguGrid(config, volumes)
    else:
        raise ValueError(f"unsupported grid_type: {config['grid_type']}")

    print(f"[fracture-grid] region={config['region']} source_wells={len(source_wells)} output_root={output_root}", flush=True)
    print(f"[fracture-grid] grid={config['grid_type']} columns={len(OUTPUT_COLUMNS)}", flush=True)
    manifest_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    try:
        for source_position, source_well in enumerate(source_wells, start=1):
            well_root = wells_root / source_well
            if not well_root.is_dir():
                raise FileNotFoundError(f"source well directory not found: {well_root}")
            real_path = well_root / str(config["real_relative_path"])
            if not real_path.is_file() or real_path.stat().st_size == 0:
                raise FileNotFoundError(f"missing real-well sample file: {real_path}")
            virtual_files = discover_virtual_files(well_root, config)
            source_rows.append(
                {
                    "Region": config["region"],
                    "SourceWellName": source_well,
                    "RealSampleFile": str(real_path),
                    "VirtualWellCount": len(virtual_files),
                    "ExpectedSampleFileCount": 1 + len(virtual_files),
                }
            )
            sample_inputs: list[tuple[str, str, Path]] = [("RealWell", source_well, real_path)]
            sample_inputs.extend(("VirtualWell", virtual_name, path) for virtual_name, path in virtual_files)
            for sample_position, (well_type, well_name, input_path) in enumerate(sample_inputs, start=1):
                points = pd.read_csv(input_path, encoding="utf-8-sig", usecols=list(POINT_COLUMNS))
                frame = build_sample_frame(points, grid, volumes)
                if frame.empty:
                    raise RuntimeError(f"{well_name}: no valid seismic-grid cells after alignment")
                if well_type == "RealWell":
                    relative_output = Path("real_wells") / f"{source_well}.csv"
                else:
                    relative_output = Path("virtual_wells") / source_well / f"{well_name}.csv"
                output_path = output_root / relative_output
                write_sample_csv(frame, output_path, overwrite=args.overwrite)
                numeric_values = frame.to_numpy(dtype=float)
                manifest_rows.append(
                    {
                        "Region": config["region"],
                        "SampleID": f"{config['region_code']}__{well_name}",
                        "SourceWellName": source_well,
                        "WellType": well_type,
                        "WellName": well_name,
                        "GridCellRows": int(len(frame)),
                        "NonzeroDensityRows": int((frame["FractureDensity"] > 0).sum()),
                        "DensityMean": float(frame["FractureDensity"].mean()),
                        "DensityMax": float(frame["FractureDensity"].max()),
                        "CornerCompleteRows": int(np.isfinite(numeric_values).all(axis=1).sum()),
                        "InputPointRows": int(len(points)),
                        "FilePath": str(output_path),
                    }
                )
                print(
                    f"[fracture-grid] {source_position}/{len(source_wells)} {source_well} "
                    f"{sample_position}/26 {well_type} {well_name} rows={len(frame)}",
                    flush=True,
                )
    finally:
        for volume in volumes.values():
            volume.close()

    manifest_fields = [
        "Region",
        "SampleID",
        "SourceWellName",
        "WellType",
        "WellName",
        "GridCellRows",
        "NonzeroDensityRows",
        "DensityMean",
        "DensityMax",
        "CornerCompleteRows",
        "InputPointRows",
        "FilePath",
    ]
    source_fields = ["Region", "SourceWellName", "RealSampleFile", "VirtualWellCount", "ExpectedSampleFileCount"]
    write_csv(output_root / "sample_file_manifest.csv", manifest_rows, manifest_fields)
    write_csv(output_root / "source_well_selection.csv", source_rows, source_fields)
    expected_files = len(source_wells) * (1 + int(config["virtual_per_source"]))
    summary_rows = [
        {"Metric": "status", "Value": "pass"},
        {"Metric": "region", "Value": config["region"]},
        {"Metric": "source_well_count", "Value": len(source_wells)},
        {"Metric": "sample_csv_count", "Value": len(manifest_rows)},
        {"Metric": "expected_sample_csv_count", "Value": expected_files},
        {"Metric": "columns_per_sample_csv", "Value": len(OUTPUT_COLUMNS)},
        {"Metric": "total_grid_cell_rows", "Value": int(sum(row["GridCellRows"] for row in manifest_rows))},
        {"Metric": "elapsed_seconds", "Value": round(time.monotonic() - started, 3)},
        {"Metric": "output_root", "Value": str(output_root)},
        {"Metric": "config_path", "Value": str(config_path)},
    ]
    write_csv(output_root / "build_summary.csv", summary_rows, ["Metric", "Value"])
    if len(manifest_rows) != expected_files:
        raise RuntimeError(f"sample file count mismatch: expected {expected_files}, found {len(manifest_rows)}")
    print(f"[fracture-grid] finished sample_csv={len(manifest_rows)} rows={sum(row['GridCellRows'] for row in manifest_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
