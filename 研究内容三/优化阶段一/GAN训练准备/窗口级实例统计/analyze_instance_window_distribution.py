# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OPT_STAGE_DIR = THIS_DIR.parent.parent
INSTANCE_DIR = OPT_STAGE_DIR / "DFN实例表达互转实验"
if str(INSTANCE_DIR) not in sys.path:
    sys.path.append(str(INSTANCE_DIR))

from instance_roundtrip_common import (  # type: ignore
    DEFAULT_DOCX_PATH,
    DEFAULT_TRACE_HEADER_CSV,
    DEFAULT_UNIT_DFN_ROOT,
    build_grid_spec,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    load_layer_table,
    load_patch_table,
    load_unit_summary,
    open_or_create_doc,
    patch_row_to_instance_geometry,
    physical_to_index,
    read_csv_utf8,
    write_csv_utf8,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/GAN训练准备/窗口级实例统计"
)
DEFAULT_BLOCK_SIZE = 25
DEFAULT_WINDOW_SIZES = [128, 160]
DEFAULT_CANDIDATE_K = [4, 8, 12, 16, 24, 32, 40]
PHASE1_RELIABILITY = {"real_controlled"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计单元 DFN 实例表达在窗口级的冲突分布，为 GAN 训练确定 K/T 参数。")
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=f"full_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=DEFAULT_WINDOW_SIZES)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--limit-units", type=int)
    return parser


def compute_unit_bounds_from_trace_arrays(
    unique_x: np.ndarray,
    unique_y: np.ndarray,
    block_x: int,
    block_y: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[float, float, float, float]:
    start_x = int(block_x) * (int(block_size) - 1)
    start_y = int(block_y) * (int(block_size) - 1)
    end_x = start_x + int(block_size)
    end_y = start_y + int(block_size)
    if end_x > len(unique_x) or end_y > len(unique_y):
        raise ValueError(f"BlockX={block_x}, BlockY={block_y} exceeds trace header grid extent")
    target_x = unique_x[start_x:end_x]
    target_y = unique_y[start_y:end_y]
    return float(target_x.min()), float(target_x.max()), float(target_y.min()), float(target_y.max())


def load_trace_header_axes(trace_header_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    trace_df = read_csv_utf8(trace_header_csv)
    return np.sort(trace_df["X"].dropna().unique()), np.sort(trace_df["Y"].dropna().unique())


def quantile_or_none(values: list[float] | np.ndarray, q: float) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def build_unit_instance_table(patch_df: pd.DataFrame, grid) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_idx, row in patch_df.iterrows():
        geom = patch_row_to_instance_geometry(row)
        center_phys = np.asarray(geom["center_phys"], dtype=float)
        center_idx = physical_to_index(center_phys.reshape(1, 3), grid)[0]
        i = int(np.clip(math.floor(center_idx[0]), 0, grid.nx - 1))
        j = int(np.clip(math.floor(center_idx[1]), 0, grid.ny - 1))
        rows.append(
            {
                "PatchIndex": int(pd.to_numeric(row.get("PatchIndex"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("PatchIndex"), errors="coerce")) else int(row_idx + 1),
                "PatchID": str(row.get("PatchID", "")),
                "GeoIntervalKey": str(row.get("GeoIntervalKey", "")),
                "StrataName": str(row.get("StrataName", "")),
                "TopSurfaceCode": str(row.get("TopSurfaceCode", "")),
                "BaseSurfaceCode": str(row.get("BaseSurfaceCode", "")),
                "SourceKind": str(row.get("SourceKind", "")),
                "SeedType": str(row.get("SeedType", "")),
                "CenterX": float(center_phys[0]),
                "CenterY": float(center_phys[1]),
                "CenterTIME": float(center_phys[2]),
                "VoxelI": int(i),
                "VoxelJ": int(j),
                "PatchLength": float(pd.to_numeric(row.get("PatchLength"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("PatchLength"), errors="coerce")) else float(geom["length_phys"]),
                "PatchHeight": float(pd.to_numeric(row.get("PatchHeight"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("PatchHeight"), errors="coerce")) else float(geom["height_phys"]),
                "Azimuth": float(pd.to_numeric(row.get("Azimuth"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("Azimuth"), errors="coerce")) else np.nan,
                "Dip": float(pd.to_numeric(row.get("Dip"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("Dip"), errors="coerce")) else np.nan,
                "Confidence": float(pd.to_numeric(row.get("Confidence"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("Confidence"), errors="coerce")) else float(geom["confidence"]),
            }
        )
    return pd.DataFrame(rows)


def generate_layer_windows(
    layer_top: float,
    layer_base: float,
    unit_z_min: float,
    unit_z_max: float,
    window_size: int,
    z_step_ms: float,
    overlap_ratio: float,
) -> list[tuple[float, float]]:
    lower = float(min(layer_top, layer_base))
    upper = float(max(layer_top, layer_base))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return []

    window_length = float(window_size) * float(z_step_ms)
    stride = max(float(window_length * (1.0 - float(overlap_ratio))), float(z_step_ms))
    windows: list[tuple[float, float]] = []

    def clamp_window(top: float) -> tuple[float, float]:
        base = top + window_length
        if top < unit_z_min:
            top = float(unit_z_min)
            base = top + window_length
        if base > unit_z_max:
            base = float(unit_z_max)
            top = base - window_length
        if top < unit_z_min:
            top = float(unit_z_min)
        return float(top), float(base)

    if (upper - lower) <= window_length:
        centered_top = ((lower + upper) / 2.0) - window_length / 2.0
        windows.append(clamp_window(centered_top))
        return windows

    start = lower
    while (start + window_length) < (upper - 1e-9):
        windows.append(clamp_window(start))
        start += stride

    last_window = clamp_window(upper - window_length)
    if not windows or abs(windows[-1][0] - last_window[0]) > max(1e-6, z_step_ms / 2.0):
        windows.append(last_window)
    return windows


def analyze_single_window(
    subset_df: pd.DataFrame,
    window_top: float,
    window_base: float,
    window_size: int,
    z_step_ms: float,
    candidate_k: list[int],
) -> tuple[dict[str, Any], list[int]]:
    row: dict[str, Any] = {
        "WindowTopTime": float(window_top),
        "WindowBaseTime": float(window_base),
        "WindowCenterTime": float((window_top + window_base) / 2.0),
        "WindowLengthMs": float(window_size * z_step_ms),
        "PatchCount": int(len(subset_df)),
        "RealPatchCount": int((subset_df["SourceKind"] == "real").sum()) if not subset_df.empty else 0,
        "VirtualPatchCount": int((subset_df["SourceKind"] == "virtual").sum()) if not subset_df.empty else 0,
        "GradientFillPatchCount": int((subset_df["SourceKind"] == "seismic_gradient_fill").sum()) if not subset_df.empty else 0,
        "IsPositiveWindow": int(len(subset_df) > 0),
    }

    if subset_df.empty:
        row.update(
            {
                "UniqueOccupiedVoxelCount": 0,
                "MaxInstancesInSingleVoxel": 0,
                "MeanInstancesPerOccupiedVoxel": 0.0,
                "PatchLengthMean": None,
                "PatchHeightMean": None,
                "AzimuthMean": None,
                "DipMean": None,
            }
        )
        for k in candidate_k:
            row[f"OverflowInstances_K{k}"] = 0
            row[f"OverflowVoxelCount_K{k}"] = 0
        return row, []

    local_k = np.floor((subset_df["CenterTIME"].to_numpy(dtype=float) - float(window_top)) / float(z_step_ms)).astype(int)
    local_k = np.clip(local_k, 0, int(window_size) - 1)
    voxel_keys = pd.DataFrame(
        {
            "VoxelI": subset_df["VoxelI"].to_numpy(dtype=int),
            "VoxelJ": subset_df["VoxelJ"].to_numpy(dtype=int),
            "VoxelK": local_k.astype(int),
        }
    )
    voxel_counts = voxel_keys.value_counts(sort=False).to_numpy(dtype=int)

    row.update(
        {
            "UniqueOccupiedVoxelCount": int(len(voxel_counts)),
            "MaxInstancesInSingleVoxel": int(voxel_counts.max()) if len(voxel_counts) else 0,
            "MeanInstancesPerOccupiedVoxel": float(voxel_counts.mean()) if len(voxel_counts) else 0.0,
            "PatchLengthMean": float(pd.to_numeric(subset_df["PatchLength"], errors="coerce").dropna().mean()),
            "PatchHeightMean": float(pd.to_numeric(subset_df["PatchHeight"], errors="coerce").dropna().mean()),
            "AzimuthMean": float(pd.to_numeric(subset_df["Azimuth"], errors="coerce").dropna().mean()),
            "DipMean": float(pd.to_numeric(subset_df["Dip"], errors="coerce").dropna().mean()),
        }
    )
    for k in candidate_k:
        overflow_instances = int(np.maximum(voxel_counts - int(k), 0).sum())
        overflow_voxels = int(np.count_nonzero(voxel_counts > int(k)))
        row[f"OverflowInstances_K{k}"] = overflow_instances
        row[f"OverflowVoxelCount_K{k}"] = overflow_voxels
    return row, voxel_counts.tolist()


def build_empty_window_df(candidate_k: list[int]) -> pd.DataFrame:
    columns = [
        "WindowTopTime",
        "WindowBaseTime",
        "WindowCenterTime",
        "WindowLengthMs",
        "PatchCount",
        "RealPatchCount",
        "VirtualPatchCount",
        "GradientFillPatchCount",
        "IsPositiveWindow",
        "UniqueOccupiedVoxelCount",
        "MaxInstancesInSingleVoxel",
        "MeanInstancesPerOccupiedVoxel",
        "PatchLengthMean",
        "PatchHeightMean",
        "AzimuthMean",
        "DipMean",
        "UnitID",
        "BlockX",
        "BlockY",
        "ReliabilityClass",
        "DataMode",
        "HasRealData",
        "HasVirtualData",
        "GeoIntervalKey",
        "StrataName",
        "TopSurfaceCode",
        "BaseSurfaceCode",
        "LayerTopTime",
        "LayerBaseTime",
        "LayerThicknessMs",
        "WindowIndex",
        "WindowTag",
    ]
    for k in candidate_k:
        columns.append(f"OverflowInstances_K{int(k)}")
        columns.append(f"OverflowVoxelCount_K{int(k)}")
    return pd.DataFrame(columns=columns)


def build_unit_inventory_row(unit_dir: Path, summary: dict[str, Any], patch_count: int, layer_count: int) -> dict[str, Any]:
    reliability = str(summary.get("ReliabilityClass", ""))
    has_real = bool(summary.get("HasRealData", False))
    phase1 = has_real and reliability in PHASE1_RELIABILITY and int(summary.get("RealSeedCount", 0)) > 0
    phase2 = has_real and patch_count > 0
    return {
        "UnitID": str(summary.get("UnitID", unit_dir.name)),
        "BlockX": int(summary.get("BlockX", 0)),
        "BlockY": int(summary.get("BlockY", 0)),
        "ReliabilityClass": reliability,
        "DataMode": str(summary.get("DataMode", "")),
        "HasRealData": has_real,
        "HasVirtualData": bool(summary.get("HasVirtualData", False)),
        "PatchCount": int(patch_count),
        "LayerCount": int(layer_count),
        "RealSeedCount": int(summary.get("RealSeedCount", 0)),
        "VirtualSeedCount": int(summary.get("VirtualSeedCount", 0)),
        "GradientFillSeedCount": int(summary.get("GradientFillSeedCount", 0)),
        "RecommendPhase1Train": int(phase1),
        "RecommendPhase2Train": int(phase2),
        "RecommendExclude": int(not phase2),
        "ExcludeReason": "" if phase2 else "no_real_seed_or_empty_patch",
        "UnitDir": str(unit_dir),
    }


def summarize_candidate_k(window_df: pd.DataFrame, voxel_count_values: list[int], candidate_k: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    total_windows = int(len(window_df))
    positive_df = window_df[window_df["IsPositiveWindow"] == 1].copy()
    positive_windows = int(len(positive_df))
    total_instances = int(window_df["PatchCount"].sum())
    total_occupied_voxels = int(window_df["UniqueOccupiedVoxelCount"].sum())

    rows: list[dict[str, Any]] = []
    for k in candidate_k:
        overflow_instances = int(window_df[f"OverflowInstances_K{k}"].sum())
        overflow_voxels = int(window_df[f"OverflowVoxelCount_K{k}"].sum())
        rows.append(
            {
                "CandidateK": int(k),
                "OverflowInstanceCount": overflow_instances,
                "OverflowInstanceRatio": float(overflow_instances / total_instances) if total_instances else 0.0,
                "OverflowVoxelCount": overflow_voxels,
                "OverflowVoxelRatio": float(overflow_voxels / total_occupied_voxels) if total_occupied_voxels else 0.0,
                "PositiveWindowCoverage": float((positive_df["MaxInstancesInSingleVoxel"] <= int(k)).mean()) if positive_windows else 1.0,
                "AllWindowCoverage": float((window_df["MaxInstancesInSingleVoxel"] <= int(k)).mean()) if total_windows else 1.0,
            }
        )
    candidate_df = pd.DataFrame(rows)

    def pick_recommendation(column: str, threshold: float) -> int | None:
        if candidate_df.empty:
            return None
        if column == "OverflowInstanceRatio":
            selected = candidate_df[candidate_df[column] <= threshold]
        else:
            selected = candidate_df[candidate_df[column] >= threshold]
        if selected.empty:
            return int(candidate_df.iloc[-1]["CandidateK"]) if not candidate_df.empty else None
        return int(selected.iloc[0]["CandidateK"])

    summary = {
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "positive_window_ratio": float(positive_windows / total_windows) if total_windows else 0.0,
        "total_window_instances": total_instances,
        "total_occupied_voxels": total_occupied_voxels,
        "occupied_voxel_instance_p50": quantile_or_none(voxel_count_values, 0.50),
        "occupied_voxel_instance_p95": quantile_or_none(voxel_count_values, 0.95),
        "occupied_voxel_instance_p99": quantile_or_none(voxel_count_values, 0.99),
        "occupied_voxel_instance_max": int(max(voxel_count_values)) if voxel_count_values else 0,
        "positive_window_max_p95": quantile_or_none(positive_df["MaxInstancesInSingleVoxel"].tolist(), 0.95),
        "positive_window_max_p99": quantile_or_none(positive_df["MaxInstancesInSingleVoxel"].tolist(), 0.99),
        "positive_window_max_max": int(positive_df["MaxInstancesInSingleVoxel"].max()) if positive_windows else 0,
        "recommended_k_by_overflow_le_1pct": pick_recommendation("OverflowInstanceRatio", 0.01),
        "recommended_k_by_overflow_le_0p5pct": pick_recommendation("OverflowInstanceRatio", 0.005),
        "recommended_k_by_positive_window_ge_95pct": pick_recommendation("PositiveWindowCoverage", 0.95),
        "recommended_k_by_positive_window_ge_99pct": pick_recommendation("PositiveWindowCoverage", 0.99),
    }
    return candidate_df, summary


def append_summary_to_docx(
    docx_path: Path,
    title: str,
    config: dict[str, Any],
    inventory_df: pd.DataFrame,
    window_summaries: dict[str, dict[str, Any]],
) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    lines = [
        "task: GAN 训练准备 - 窗口级实例分布统计",
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"output_dir: {config.get('output_dir', '')}",
        f"unit_dfn_root: {config.get('unit_dfn_root', '')}",
        f"trace_header_csv: {config.get('trace_header_csv', '')}",
        f"xy_resolution: {config.get('xy_resolution', '')}",
        f"z_step_ms: {config.get('z_step_ms', '')}",
        f"window_sizes: {config.get('window_sizes', '')}",
        f"overlap_ratio: {config.get('overlap_ratio', '')}",
        f"candidate_k: {config.get('candidate_k', '')}",
        f"processed_unit_count: {int(len(inventory_df))}",
        f"phase1_train_unit_count: {int(inventory_df['RecommendPhase1Train'].sum()) if not inventory_df.empty else 0}",
        f"phase2_train_unit_count: {int(inventory_df['RecommendPhase2Train'].sum()) if not inventory_df.empty else 0}",
    ]
    for key, summary in window_summaries.items():
        lines.extend(
            [
                f"[{key}] total_windows: {summary.get('total_windows', '')}",
                f"[{key}] positive_windows: {summary.get('positive_windows', '')}",
                f"[{key}] total_window_instances: {summary.get('total_window_instances', '')}",
                f"[{key}] occupied_voxel_instance_p95: {summary.get('occupied_voxel_instance_p95', '')}",
                f"[{key}] occupied_voxel_instance_p99: {summary.get('occupied_voxel_instance_p99', '')}",
                f"[{key}] occupied_voxel_instance_max: {summary.get('occupied_voxel_instance_max', '')}",
                f"[{key}] recommended_k_by_overflow_le_1pct: {summary.get('recommended_k_by_overflow_le_1pct', '')}",
                f"[{key}] recommended_k_by_positive_window_ge_95pct: {summary.get('recommended_k_by_positive_window_ge_95pct', '')}",
            ]
        )
    doc.add_paragraph("\n".join(lines))
    doc.add_paragraph("")
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.output_root / args.run_name
    aggregate_dir = run_dir / "aggregated"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    unique_x, unique_y = load_trace_header_axes(args.trace_header_csv)
    unit_dirs = sorted([path for path in Path(args.unit_dfn_root).iterdir() if path.is_dir()])
    if args.limit_units:
        unit_dirs = unit_dirs[: int(args.limit_units)]

    inventory_rows: list[dict[str, Any]] = []
    processed_units: list[dict[str, Any]] = []
    skipped_units: list[dict[str, Any]] = []

    for unit_dir in unit_dirs:
        patch_csv = unit_dir / "unit_dfn_patches.csv"
        if not patch_csv.exists():
            skipped_units.append({"UnitDir": str(unit_dir), "Reason": "missing_unit_dfn_patches_csv"})
            continue

        patch_df = load_patch_table(patch_csv)
        if patch_df.empty:
            skipped_units.append({"UnitDir": str(unit_dir), "Reason": "empty_patch_table"})
            continue

        layers_df = load_layer_table(unit_dir)
        if layers_df.empty:
            skipped_units.append({"UnitDir": str(unit_dir), "Reason": "missing_layer_table"})
            continue

        summary = load_unit_summary(unit_dir)
        unit_id = str(summary.get("UnitID", patch_df["UnitID"].iloc[0] or unit_dir.name))
        block_x = int(summary.get("BlockX", patch_df["BlockX"].iloc[0]))
        block_y = int(summary.get("BlockY", patch_df["BlockY"].iloc[0]))

        try:
            x_min, x_max, y_min, y_max = compute_unit_bounds_from_trace_arrays(unique_x, unique_y, block_x, block_y)
        except Exception:
            x_min, x_max, y_min, y_max = compute_unit_bounds_from_patch_table(patch_df)

        z_min, z_max = compute_time_bounds(patch_df, layers_df, z_padding_ms=0.0)
        grid = build_grid_spec(
            unit_id=unit_id,
            block_x=block_x,
            block_y=block_y,
            x_bounds=(x_min, x_max),
            y_bounds=(y_min, y_max),
            z_bounds=(z_min, z_max),
            xy_resolution=args.xy_resolution,
            z_step_ms=args.z_step_ms,
        )
        instance_df = build_unit_instance_table(patch_df, grid)
        if instance_df.empty:
            skipped_units.append({"UnitDir": str(unit_dir), "Reason": "empty_instance_table"})
            continue

        inventory_rows.append(build_unit_inventory_row(unit_dir, summary, len(patch_df), len(layers_df)))
        processed_units.append(
            {
                "UnitDir": str(unit_dir),
                "UnitID": unit_id,
                "BlockX": block_x,
                "BlockY": block_y,
                "ReliabilityClass": str(summary.get("ReliabilityClass", "")),
                "DataMode": str(summary.get("DataMode", "")),
                "HasRealData": bool(summary.get("HasRealData", False)),
                "HasVirtualData": bool(summary.get("HasVirtualData", False)),
                "PatchCount": int(len(patch_df)),
                "LayerCount": int(len(layers_df)),
                "ZMin": float(z_min),
                "ZMax": float(z_max),
                "Instances": instance_df,
                "Layers": layers_df.copy(),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)
    write_csv_utf8(inventory_df, aggregate_dir / "unit_inventory.csv")
    write_csv_utf8(pd.DataFrame(skipped_units), aggregate_dir / "skipped_units.csv")
    if not inventory_df.empty:
        write_csv_utf8(inventory_df[inventory_df["RecommendPhase1Train"] == 1].copy(), aggregate_dir / "phase1_train_units.csv")
        write_csv_utf8(inventory_df[inventory_df["RecommendPhase2Train"] == 1].copy(), aggregate_dir / "phase2_train_units.csv")

    window_summaries: dict[str, dict[str, Any]] = {}
    unit_summary_rows: list[dict[str, Any]] = []
    window_size_comparison_rows: list[dict[str, Any]] = []

    for window_size in sorted({int(v) for v in args.window_sizes}):
        tag = f"T{int(window_size)}_OV{int(round(float(args.overlap_ratio) * 100.0))}"
        size_dir = run_dir / tag
        size_dir.mkdir(parents=True, exist_ok=True)

        window_rows: list[dict[str, Any]] = []
        voxel_count_values: list[int] = []

        for unit in processed_units:
            unit_instance_df = unit["Instances"]
            layers_df = unit["Layers"]
            unit_window_rows: list[dict[str, Any]] = []

            for _, layer_row in layers_df.iterrows():
                layer_key = str(layer_row.get("GeoIntervalKey", ""))
                top_time = pd.to_numeric(layer_row.get("TopTime"), errors="coerce")
                base_time = pd.to_numeric(layer_row.get("BaseTime"), errors="coerce")
                if pd.isna(top_time) or pd.isna(base_time):
                    continue
                windows = generate_layer_windows(
                    layer_top=float(top_time),
                    layer_base=float(base_time),
                    unit_z_min=float(unit["ZMin"]),
                    unit_z_max=float(unit["ZMax"]),
                    window_size=window_size,
                    z_step_ms=args.z_step_ms,
                    overlap_ratio=args.overlap_ratio,
                )
                layer_subset = unit_instance_df[unit_instance_df["GeoIntervalKey"] == layer_key].copy()
                for window_idx, (window_top, window_base) in enumerate(windows, start=1):
                    mask = (layer_subset["CenterTIME"] >= float(window_top)) & (layer_subset["CenterTIME"] < float(window_base) + 1e-9)
                    subset_df = layer_subset.loc[mask].copy()
                    row, voxel_counts = analyze_single_window(
                        subset_df=subset_df,
                        window_top=window_top,
                        window_base=window_base,
                        window_size=window_size,
                        z_step_ms=args.z_step_ms,
                        candidate_k=[int(v) for v in args.candidate_k],
                    )
                    row.update(
                        {
                            "UnitID": unit["UnitID"],
                            "BlockX": int(unit["BlockX"]),
                            "BlockY": int(unit["BlockY"]),
                            "ReliabilityClass": unit["ReliabilityClass"],
                            "DataMode": unit["DataMode"],
                            "HasRealData": int(unit["HasRealData"]),
                            "HasVirtualData": int(unit["HasVirtualData"]),
                            "GeoIntervalKey": layer_key,
                            "StrataName": str(layer_row.get("StrataName", "")),
                            "TopSurfaceCode": str(layer_row.get("TopSurfaceCode", "")),
                            "BaseSurfaceCode": str(layer_row.get("BaseSurfaceCode", "")),
                            "LayerTopTime": float(min(top_time, base_time)),
                            "LayerBaseTime": float(max(top_time, base_time)),
                            "LayerThicknessMs": float(abs(float(base_time) - float(top_time))),
                            "WindowIndex": int(window_idx),
                            "WindowTag": f"{unit['UnitID']}::{layer_key}::W{window_idx:03d}",
                        }
                    )
                    window_rows.append(row)
                    unit_window_rows.append(row)
                    voxel_count_values.extend(voxel_counts)

            unit_window_df = pd.DataFrame(unit_window_rows)
            if unit_window_df.empty:
                continue
            unit_summary_row = {
                "UnitID": unit["UnitID"],
                "BlockX": int(unit["BlockX"]),
                "BlockY": int(unit["BlockY"]),
                "ReliabilityClass": unit["ReliabilityClass"],
                "DataMode": unit["DataMode"],
                "WindowSize": int(window_size),
                "WindowCount": int(len(unit_window_df)),
                "PositiveWindowCount": int(unit_window_df["IsPositiveWindow"].sum()),
                "TotalWindowPatchCount": int(unit_window_df["PatchCount"].sum()),
                "MaxWindowPatchCount": int(unit_window_df["PatchCount"].max()),
                "MaxInstancesInSingleVoxel": int(unit_window_df["MaxInstancesInSingleVoxel"].max()),
            }
            for k in args.candidate_k:
                unit_summary_row[f"OverflowInstances_K{int(k)}"] = int(unit_window_df[f"OverflowInstances_K{int(k)}"].sum())
            unit_summary_rows.append(unit_summary_row)

        window_df = pd.DataFrame(window_rows) if window_rows else build_empty_window_df([int(v) for v in args.candidate_k])
        write_csv_utf8(window_df, size_dir / "window_stats.csv")

        unit_window_summary_df = pd.DataFrame([row for row in unit_summary_rows if int(row["WindowSize"]) == int(window_size)])
        write_csv_utf8(unit_window_summary_df, size_dir / "unit_window_summary.csv")

        if not window_df.empty:
            interval_stats_df = (
                window_df.groupby("GeoIntervalKey", dropna=False)
                .agg(
                    UnitCount=("UnitID", "nunique"),
                    WindowCount=("WindowTag", "count"),
                    PositiveWindowCount=("IsPositiveWindow", "sum"),
                    TotalWindowPatchCount=("PatchCount", "sum"),
                    MaxInstancesInSingleVoxel=("MaxInstancesInSingleVoxel", "max"),
                    MeanPatchCount=("PatchCount", "mean"),
                )
                .reset_index()
            )
        else:
            interval_stats_df = pd.DataFrame(columns=["GeoIntervalKey", "UnitCount", "WindowCount", "PositiveWindowCount", "TotalWindowPatchCount", "MaxInstancesInSingleVoxel", "MeanPatchCount"])
        write_csv_utf8(interval_stats_df, size_dir / "interval_stats.csv")

        candidate_df, summary = summarize_candidate_k(window_df, voxel_count_values, [int(v) for v in args.candidate_k])
        write_csv_utf8(candidate_df, size_dir / "candidate_k_coverage.csv")
        if not window_df.empty:
            top_conflict_df = window_df.sort_values(["MaxInstancesInSingleVoxel", "PatchCount"], ascending=[False, False]).head(200).copy()
        else:
            top_conflict_df = pd.DataFrame()
        write_csv_utf8(top_conflict_df, size_dir / "top_conflict_windows.csv")
        write_json(size_dir / "window_stats_summary.json", summary)
        window_summaries[tag] = summary
        window_size_comparison_rows.append({"WindowSize": int(window_size), **summary})

    comparison_df = pd.DataFrame(window_size_comparison_rows)
    write_csv_utf8(comparison_df, aggregate_dir / "window_size_comparison.csv")

    reliability_counts = (
        inventory_df["ReliabilityClass"].value_counts(dropna=False).to_dict() if not inventory_df.empty else {}
    )
    data_mode_counts = inventory_df["DataMode"].value_counts(dropna=False).to_dict() if not inventory_df.empty else {}

    global_summary = {
        "run_name": args.run_name,
        "output_dir": str(run_dir),
        "unit_dfn_root": str(args.unit_dfn_root),
        "trace_header_csv": str(args.trace_header_csv),
        "xy_resolution": int(args.xy_resolution),
        "z_step_ms": float(args.z_step_ms),
        "window_sizes": [int(v) for v in args.window_sizes],
        "overlap_ratio": float(args.overlap_ratio),
        "candidate_k": [int(v) for v in args.candidate_k],
        "processed_unit_count": int(len(inventory_df)),
        "skipped_unit_count": int(len(skipped_units)),
        "phase1_train_unit_count": int(inventory_df["RecommendPhase1Train"].sum()) if not inventory_df.empty else 0,
        "phase2_train_unit_count": int(inventory_df["RecommendPhase2Train"].sum()) if not inventory_df.empty else 0,
        "reliability_counts": reliability_counts,
        "data_mode_counts": data_mode_counts,
        "window_summaries": window_summaries,
        "unit_inventory_csv": str(aggregate_dir / "unit_inventory.csv"),
        "phase1_train_units_csv": str(aggregate_dir / "phase1_train_units.csv"),
        "phase2_train_units_csv": str(aggregate_dir / "phase2_train_units.csv"),
        "window_size_comparison_csv": str(aggregate_dir / "window_size_comparison.csv"),
    }
    write_json(aggregate_dir / "global_summary.json", global_summary)

    append_summary_to_docx(
        docx_path=args.docx_path,
        title="GAN训练准备 - 窗口级实例分布统计",
        config={
            "output_dir": str(run_dir),
            "unit_dfn_root": str(args.unit_dfn_root),
            "trace_header_csv": str(args.trace_header_csv),
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
            "window_sizes": [int(v) for v in args.window_sizes],
            "overlap_ratio": float(args.overlap_ratio),
            "candidate_k": [int(v) for v in args.candidate_k],
        },
        inventory_df=inventory_df,
        window_summaries=window_summaries,
    )

    print(f"run_dir: {run_dir}")
    print(f"processed_unit_count: {len(inventory_df)}")
    print(f"global_summary: {aggregate_dir / 'global_summary.json'}")


if __name__ == "__main__":
    main()
