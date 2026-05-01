# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio

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
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/GAN训练准备/训练样本打包"
)
DEFAULT_SGY_FILE = Path(r"/data/shared/project-oil/wx数据/砂砾岩/psdm_final_time.sgy")
DEFAULT_BLOCK_SIZE = 25
DEFAULT_SOURCE_WEIGHTS = {
    "real": 1.0,
    "virtual": 0.8,
    "seismic_gradient_fill": 0.4,
}
TARGET_GEOM_CHANNELS = [
    "offset_x",
    "offset_y",
    "offset_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "u_dir_x",
    "u_dir_y",
    "u_dir_z",
    "length",
    "height",
    "confidence",
]
INPUT_CHANNELS = [
    "seismic_amp",
    "grad_x",
    "grad_y",
    "grad_z",
    "grad_mag",
    "dist_to_top",
    "dist_to_base",
    "rel_depth_in_interval",
    "block_x_norm",
    "block_y_norm",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于单元 DFN 实例表达与地震条件打包 GAN 训练样本。")
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--sgy-file", type=Path, default=DEFAULT_SGY_FILE)
    parser.add_argument("--stats-run-dir", type=Path, required=True)
    parser.add_argument("--train-unit-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=f"phase1_t128_k16_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--slots-per-voxel", type=int, default=16)
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
        raise ValueError(f"BlockX={block_x}, BlockY={block_y} exceeds trace header extent")
    target_x = unique_x[start_x:end_x]
    target_y = unique_y[start_y:end_y]
    return float(target_x.min()), float(target_x.max()), float(target_y.min()), float(target_y.max())


def load_trace_header(trace_header_csv: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    trace_df = read_csv_utf8(trace_header_csv)
    unique_x = np.sort(trace_df["X"].dropna().unique())
    unique_y = np.sort(trace_df["Y"].dropna().unique())
    return trace_df, unique_x, unique_y


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
        return [clamp_window(centered_top)]

    start = lower
    while (start + window_length) < (upper - 1e-9):
        windows.append(clamp_window(start))
        start += stride
    last_window = clamp_window(upper - window_length)
    if not windows or abs(windows[-1][0] - last_window[0]) > max(1e-6, z_step_ms / 2.0):
        windows.append(last_window)
    return windows


def build_unit_trace_block(
    trace_df: pd.DataFrame,
    unique_x: np.ndarray,
    unique_y: np.ndarray,
    block_x: int,
    block_y: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> pd.DataFrame:
    start_x = int(block_x) * (int(block_size) - 1)
    start_y = int(block_y) * (int(block_size) - 1)
    end_x = start_x + int(block_size)
    end_y = start_y + int(block_size)
    target_x = unique_x[start_x:end_x]
    target_y = unique_y[start_y:end_y]
    block_df = trace_df[trace_df["X"].isin(target_x) & trace_df["Y"].isin(target_y)].copy()
    if block_df.empty:
        raise ValueError(f"empty trace block for BlockX={block_x}, BlockY={block_y}")
    x_order = {float(x): idx for idx, x in enumerate(target_x.tolist())}
    y_order = {float(y): idx for idx, y in enumerate(target_y.tolist())}
    block_df["XOrder"] = block_df["X"].map(x_order)
    block_df["YOrder"] = block_df["Y"].map(y_order)
    block_df = block_df.sort_values(["XOrder", "YOrder"]).reset_index(drop=True)
    if len(block_df) != int(block_size * block_size):
        raise ValueError(f"trace block size mismatch for BlockX={block_x}, BlockY={block_y}: {len(block_df)}")
    return block_df


def extract_unit_trace_cube(
    sgy_handle,
    block_df: pd.DataFrame,
    target_z_centers: np.ndarray,
) -> np.ndarray:
    trace_indices = block_df["TraceIdx"].astype(int).to_numpy()
    sample_times = np.asarray(sgy_handle.samples, dtype=float)
    raw = np.stack([np.asarray(sgy_handle.trace[idx], dtype=np.float32) for idx in trace_indices], axis=0)
    interp = np.empty((raw.shape[0], len(target_z_centers)), dtype=np.float32)
    for i in range(raw.shape[0]):
        interp[i] = np.interp(
            x=target_z_centers.astype(float),
            xp=sample_times,
            fp=raw[i].astype(float),
            left=float(raw[i, 0]),
            right=float(raw[i, -1]),
        ).astype(np.float32)
    return interp.reshape(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE, len(target_z_centers))


def traces_to_center_volume(trace_cube: np.ndarray) -> np.ndarray:
    return (
        trace_cube[:-1, :-1, :]
        + trace_cube[1:, :-1, :]
        + trace_cube[:-1, 1:, :]
        + trace_cube[1:, 1:, :]
    ) / 4.0


def robust_scale_signed(volume: np.ndarray, q: float = 0.995) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    scale = float(np.quantile(np.abs(finite), q))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    if scale <= 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip(volume / scale, -1.0, 1.0).astype(np.float32)


def build_input_features(
    unit_center_volume: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    block_x: int,
    block_y: int,
    max_block_x: int,
    max_block_y: int,
    window_top: float,
    window_base: float,
    z_centers: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    gx, gy, gz = np.gradient(unit_center_volume, dx, dy, dz, axis=(0, 1, 2))
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    dist_to_top = (z_centers.astype(np.float32) - float(window_top)).astype(np.float32)
    dist_to_base = (float(window_base) - z_centers.astype(np.float32)).astype(np.float32)
    layer_thickness = max(float(window_base - window_top), 1e-6)
    rel_depth = ((z_centers.astype(np.float32) - float(window_top)) / layer_thickness).astype(np.float32)

    amp_scaled = robust_scale_signed(unit_center_volume.astype(np.float32))
    gx_scaled = robust_scale_signed(gx.astype(np.float32))
    gy_scaled = robust_scale_signed(gy.astype(np.float32))
    gz_scaled = robust_scale_signed(gz.astype(np.float32))
    grad_mag_scaled = robust_scale_signed(grad_mag.astype(np.float32))

    dist_to_top_scaled = np.clip(dist_to_top / layer_thickness, -1.0, 2.0)
    dist_to_base_scaled = np.clip(dist_to_base / layer_thickness, -1.0, 2.0)
    rel_depth_scaled = np.clip(rel_depth, 0.0, 1.0)

    block_x_norm = np.full_like(unit_center_volume, float(block_x / max(max_block_x, 1)), dtype=np.float32)
    block_y_norm = np.full_like(unit_center_volume, float(block_y / max(max_block_y, 1)), dtype=np.float32)

    channels = np.stack(
        [
            amp_scaled,
            gx_scaled,
            gy_scaled,
            gz_scaled,
            grad_mag_scaled,
            np.broadcast_to(dist_to_top_scaled.reshape(1, 1, -1), unit_center_volume.shape).astype(np.float32),
            np.broadcast_to(dist_to_base_scaled.reshape(1, 1, -1), unit_center_volume.shape).astype(np.float32),
            np.broadcast_to(rel_depth_scaled.reshape(1, 1, -1), unit_center_volume.shape).astype(np.float32),
            block_x_norm.astype(np.float32),
            block_y_norm.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    stats = {
        "amp_abs_q995": float(np.quantile(np.abs(unit_center_volume[np.isfinite(unit_center_volume)]), 0.995)),
        "grad_mag_abs_q995": float(np.quantile(np.abs(grad_mag[np.isfinite(grad_mag)]), 0.995)),
        "layer_thickness_ms": float(layer_thickness),
    }
    return channels, stats


def sort_window_instances(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return window_df.copy()
    work = window_df.copy()
    work["SortOffsetZ"] = pd.to_numeric(work["OffsetZ"], errors="coerce").fillna(0.0)
    work["SortAzimuth"] = pd.to_numeric(work["Azimuth"], errors="coerce").fillna(0.0) % 360.0
    work["SortLength"] = pd.to_numeric(work["PatchLength"], errors="coerce").fillna(0.0)
    work["SortHeight"] = pd.to_numeric(work["PatchHeight"], errors="coerce").fillna(0.0)
    work["SortPatchIndex"] = pd.to_numeric(work["PatchIndex"], errors="coerce").fillna(0).astype(int)
    return work.sort_values(
        ["SortOffsetZ", "SortAzimuth", "SortLength", "SortHeight", "SortPatchIndex"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)


def encode_window_targets(
    patch_df: pd.DataFrame,
    grid,
    window_top: float,
    window_size: int,
    z_step_ms: float,
    slots_per_voxel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    center = np.zeros((int(slots_per_voxel), grid.nx, grid.ny, int(window_size)), dtype=np.float32)
    count = np.zeros((1, grid.nx, grid.ny, int(window_size)), dtype=np.float32)
    geom = np.zeros((int(slots_per_voxel), len(TARGET_GEOM_CHANNELS), grid.nx, grid.ny, int(window_size)), dtype=np.float32)
    weight = np.zeros((int(slots_per_voxel), grid.nx, grid.ny, int(window_size)), dtype=np.float32)

    if patch_df.empty:
        return center, count, geom, weight, {
            "PatchCount": 0,
            "OverflowInstanceCount": 0,
            "OverflowVoxelCount": 0,
            "MaxInstancesInSingleVoxel": 0,
            "ActiveVoxelCount": 0,
            "RealPatchCount": 0,
            "VirtualPatchCount": 0,
            "GradientFillPatchCount": 0,
        }

    rows: list[dict[str, Any]] = []
    for row_idx, row in patch_df.iterrows():
        geom_info = patch_row_to_instance_geometry(row)
        center_phys = np.asarray(geom_info["center_phys"], dtype=float)
        center_idx = physical_to_index(center_phys.reshape(1, 3), grid)[0]
        i = int(np.clip(math.floor(center_idx[0]), 0, grid.nx - 1))
        j = int(np.clip(math.floor(center_idx[1]), 0, grid.ny - 1))
        k_local = int(np.clip(math.floor((float(center_phys[2]) - float(window_top)) / float(z_step_ms)), 0, int(window_size) - 1))
        voxel_center = np.array(
            [grid.x_centers[i], grid.y_centers[j], float(window_top) + (k_local + 0.5) * float(z_step_ms)],
            dtype=float,
        )
        patch_index = pd.to_numeric(row.get("PatchIndex"), errors="coerce")
        azimuth = pd.to_numeric(row.get("Azimuth"), errors="coerce")
        rows.append(
            {
                "PatchIndex": int(patch_index) if pd.notna(patch_index) else int(row_idx + 1),
                "PatchID": str(row.get("PatchID", "")),
                "SourceKind": str(row.get("SourceKind", "")),
                "CenterTIME": float(center_phys[2]),
                "VoxelI": int(i),
                "VoxelJ": int(j),
                "VoxelK": int(k_local),
                "OffsetX": float(center_phys[0] - voxel_center[0]),
                "OffsetY": float(center_phys[1] - voxel_center[1]),
                "OffsetZ": float(center_phys[2] - voxel_center[2]),
                "NormalX": float(np.asarray(geom_info["normal_phys"], dtype=float)[0]),
                "NormalY": float(np.asarray(geom_info["normal_phys"], dtype=float)[1]),
                "NormalZ": float(np.asarray(geom_info["normal_phys"], dtype=float)[2]),
                "UDirX": float(np.asarray(geom_info["u_dir_phys"], dtype=float)[0]),
                "UDirY": float(np.asarray(geom_info["u_dir_phys"], dtype=float)[1]),
                "UDirZ": float(np.asarray(geom_info["u_dir_phys"], dtype=float)[2]),
                "PatchLength": float(geom_info["length_phys"]),
                "PatchHeight": float(geom_info["height_phys"]),
                "Confidence": float(geom_info["confidence"]),
                "Azimuth": float(azimuth) if pd.notna(azimuth) else np.nan,
            }
        )

    inst_df = pd.DataFrame(rows)
    voxel_group = inst_df.groupby(["VoxelI", "VoxelJ", "VoxelK"], dropna=False)
    overflow_instances = 0
    overflow_voxels = 0
    max_instances = 0
    for (i, j, k_local), group in voxel_group:
        group_sorted = sort_window_instances(group)
        raw_count = int(len(group_sorted))
        count[0, int(i), int(j), int(k_local)] = float(min(raw_count, int(slots_per_voxel)))
        max_instances = max(max_instances, raw_count)
        if raw_count > int(slots_per_voxel):
            overflow_instances += raw_count - int(slots_per_voxel)
            overflow_voxels += 1
        for slot, (_, item) in enumerate(group_sorted.iterrows()):
            if slot >= int(slots_per_voxel):
                continue
            center[slot, int(i), int(j), int(k_local)] = 1.0
            geom[slot, :, int(i), int(j), int(k_local)] = np.array(
                [
                    item["OffsetX"],
                    item["OffsetY"],
                    item["OffsetZ"],
                    item["NormalX"],
                    item["NormalY"],
                    item["NormalZ"],
                    item["UDirX"],
                    item["UDirY"],
                    item["UDirZ"],
                    item["PatchLength"],
                    item["PatchHeight"],
                    item["Confidence"],
                ],
                dtype=np.float32,
            )
            weight[slot, int(i), int(j), int(k_local)] = float(DEFAULT_SOURCE_WEIGHTS.get(str(item["SourceKind"]), 0.6))

    summary = {
        "PatchCount": int(len(inst_df)),
        "OverflowInstanceCount": int(overflow_instances),
        "OverflowVoxelCount": int(overflow_voxels),
        "MaxInstancesInSingleVoxel": int(max_instances),
        "ActiveVoxelCount": int(len(voxel_group)),
        "RealPatchCount": int((inst_df["SourceKind"] == "real").sum()),
        "VirtualPatchCount": int((inst_df["SourceKind"] == "virtual").sum()),
        "GradientFillPatchCount": int((inst_df["SourceKind"] == "seismic_gradient_fill").sum()),
    }
    return center, count, geom, weight, summary


def append_summary_to_docx(
    docx_path: Path,
    title: str,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    lines = [
        "task: GAN 训练准备 - 实例表达训练样本打包",
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"output_dir: {config.get('output_dir', '')}",
        f"stats_run_dir: {config.get('stats_run_dir', '')}",
        f"train_unit_csv: {config.get('train_unit_csv', '')}",
        f"sgy_file: {config.get('sgy_file', '')}",
        f"xy_resolution: {config.get('xy_resolution', '')}",
        f"z_step_ms: {config.get('z_step_ms', '')}",
        f"window_size: {config.get('window_size', '')}",
        f"overlap_ratio: {config.get('overlap_ratio', '')}",
        f"slots_per_voxel: {config.get('slots_per_voxel', '')}",
        f"processed_unit_count: {summary.get('processed_unit_count', '')}",
        f"total_sample_count: {summary.get('total_sample_count', '')}",
        f"positive_sample_count: {summary.get('positive_sample_count', '')}",
        f"negative_sample_count: {summary.get('negative_sample_count', '')}",
        f"total_patch_count: {summary.get('total_patch_count', '')}",
        f"total_overflow_instance_count: {summary.get('total_overflow_instance_count', '')}",
        f"samples_with_overflow: {summary.get('samples_with_overflow', '')}",
        f"recommended_training_subset: {summary.get('recommended_training_subset', '')}",
        f"manifest_csv: {summary.get('manifest_csv', '')}",
        f"summary_json: {summary.get('summary_json', '')}",
    ]
    doc.add_paragraph("\n".join(lines))
    doc.add_paragraph("")
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))


def main() -> None:
    args = build_parser().parse_args()
    stats_run_dir = Path(args.stats_run_dir)
    train_unit_csv = Path(args.train_unit_csv) if args.train_unit_csv else (stats_run_dir / "aggregated" / "phase1_train_units.csv")
    if not train_unit_csv.exists():
        raise FileNotFoundError(f"train_unit_csv not found: {train_unit_csv}")

    run_dir = Path(args.output_root) / args.run_name
    samples_dir = run_dir / "samples"
    aggregated_dir = run_dir / "aggregated"
    samples_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    train_units_df = read_csv_utf8(train_unit_csv)
    if args.limit_units:
        train_units_df = train_units_df.head(int(args.limit_units)).copy()
    selected_unit_ids = set(train_units_df["UnitID"].astype(str).tolist())

    trace_df, unique_x, unique_y = load_trace_header(args.trace_header_csv)
    max_block_x = int(max(train_units_df["BlockX"].max() if not train_units_df.empty else 1, 1))
    max_block_y = int(max(train_units_df["BlockY"].max() if not train_units_df.empty else 1, 1))

    sample_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    processed_units = 0
    total_patch_count = 0
    total_overflow = 0
    samples_with_overflow = 0

    with segyio.open(str(args.sgy_file), "r", ignore_geometry=True) as sgy:
        for unit_id in train_units_df["UnitID"].astype(str).tolist():
            unit_dir = Path(args.unit_dfn_root) / unit_id
            patch_csv = unit_dir / "unit_dfn_patches.csv"
            if not patch_csv.exists():
                skipped_rows.append({"UnitID": unit_id, "Reason": "missing_unit_dfn_patches_csv"})
                continue

            patch_df = load_patch_table(patch_csv)
            if patch_df.empty:
                skipped_rows.append({"UnitID": unit_id, "Reason": "empty_patch_table"})
                continue

            layers_df = load_layer_table(unit_dir)
            if layers_df.empty:
                skipped_rows.append({"UnitID": unit_id, "Reason": "missing_layer_table"})
                continue

            unit_summary = load_unit_summary(unit_dir)
            block_x = int(unit_summary.get("BlockX", patch_df["BlockX"].iloc[0]))
            block_y = int(unit_summary.get("BlockY", patch_df["BlockY"].iloc[0]))

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

            block_trace_df = build_unit_trace_block(trace_df, unique_x, unique_y, block_x, block_y)
            trace_cube = extract_unit_trace_cube(sgy, block_trace_df, grid.z_centers.astype(float))
            unit_center_volume = traces_to_center_volume(trace_cube).astype(np.float32)
            processed_units += 1

            for _, layer_row in layers_df.iterrows():
                layer_key = str(layer_row.get("GeoIntervalKey", ""))
                layer_top = pd.to_numeric(layer_row.get("TopTime"), errors="coerce")
                layer_base = pd.to_numeric(layer_row.get("BaseTime"), errors="coerce")
                if pd.isna(layer_top) or pd.isna(layer_base):
                    continue

                layer_windows = generate_layer_windows(
                    layer_top=float(layer_top),
                    layer_base=float(layer_base),
                    unit_z_min=float(z_min),
                    unit_z_max=float(z_max),
                    window_size=args.window_size,
                    z_step_ms=args.z_step_ms,
                    overlap_ratio=args.overlap_ratio,
                )
                layer_patch_df = patch_df[patch_df["GeoIntervalKey"].astype(str) == layer_key].copy()
                for window_idx, (window_top, window_base) in enumerate(layer_windows, start=1):
                    z_mask = (grid.z_centers >= float(window_top)) & (grid.z_centers < float(window_base) + 1e-9)
                    if not np.any(z_mask):
                        continue
                    z_indices = np.flatnonzero(z_mask)
                    window_volume = unit_center_volume[:, :, z_indices].astype(np.float32)
                    input_channels, input_stats = build_input_features(
                        unit_center_volume=window_volume,
                        dx=grid.dx,
                        dy=grid.dy,
                        dz=grid.dz,
                        block_x=block_x,
                        block_y=block_y,
                        max_block_x=max_block_x,
                        max_block_y=max_block_y,
                        window_top=float(window_top),
                        window_base=float(window_base),
                        z_centers=grid.z_centers[z_indices],
                    )

                    subset_mask = (layer_patch_df["CenterTIME"] >= float(window_top)) & (layer_patch_df["CenterTIME"] < float(window_base) + 1e-9)
                    window_patch_df = layer_patch_df.loc[subset_mask].copy()
                    target_center, target_count, target_geom, target_weight, target_summary = encode_window_targets(
                        patch_df=window_patch_df,
                        grid=grid,
                        window_top=float(window_top),
                        window_size=int(len(z_indices)),
                        z_step_ms=float(args.z_step_ms),
                        slots_per_voxel=int(args.slots_per_voxel),
                    )

                    sample_id = f"{unit_id}__{layer_key}__W{window_idx:03d}"
                    sample_dir = samples_dir / unit_id / layer_key / f"W{window_idx:03d}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    input_path = sample_dir / "input.npy"
                    center_path = sample_dir / "target_center.npy"
                    count_path = sample_dir / "target_count.npy"
                    geom_path = sample_dir / "target_geom.npy"
                    weight_path = sample_dir / "target_weight.npy"
                    meta_path = sample_dir / "meta.json"

                    np.save(input_path, input_channels.astype(np.float32))
                    np.save(center_path, target_center.astype(np.float32))
                    np.save(count_path, target_count.astype(np.float32))
                    np.save(geom_path, target_geom.astype(np.float32))
                    np.save(weight_path, target_weight.astype(np.float32))

                    sample_meta = {
                        "SampleID": sample_id,
                        "UnitID": unit_id,
                        "BlockX": block_x,
                        "BlockY": block_y,
                        "GeoIntervalKey": layer_key,
                        "StrataName": str(layer_row.get("StrataName", "")),
                        "TopSurfaceCode": str(layer_row.get("TopSurfaceCode", "")),
                        "BaseSurfaceCode": str(layer_row.get("BaseSurfaceCode", "")),
                        "WindowIndex": int(window_idx),
                        "WindowTopTime": float(window_top),
                        "WindowBaseTime": float(window_base),
                        "WindowLengthMs": float(window_base - window_top),
                        "WindowSize": int(len(z_indices)),
                        "XYResolution": int(args.xy_resolution),
                        "ZStepMs": float(args.z_step_ms),
                        "SlotsPerVoxel": int(args.slots_per_voxel),
                        "InputChannels": list(INPUT_CHANNELS),
                        "TargetGeomChannels": list(TARGET_GEOM_CHANNELS),
                        "InputStats": input_stats,
                        "TargetSummary": target_summary,
                        "PatchCSV": str(patch_csv),
                        "SamplePaths": {
                            "input": str(input_path),
                            "target_center": str(center_path),
                            "target_count": str(count_path),
                            "target_geom": str(geom_path),
                            "target_weight": str(weight_path),
                        },
                    }
                    write_json(meta_path, sample_meta)

                    total_patch_count += int(target_summary["PatchCount"])
                    total_overflow += int(target_summary["OverflowInstanceCount"])
                    if int(target_summary["OverflowInstanceCount"]) > 0:
                        samples_with_overflow += 1

                    sample_rows.append(
                        {
                            "SampleID": sample_id,
                            "UnitID": unit_id,
                            "BlockX": block_x,
                            "BlockY": block_y,
                            "GeoIntervalKey": layer_key,
                            "WindowIndex": int(window_idx),
                            "WindowTopTime": float(window_top),
                            "WindowBaseTime": float(window_base),
                            "WindowLengthMs": float(window_base - window_top),
                            "WindowSize": int(len(z_indices)),
                            "IsPositiveWindow": int(target_summary["PatchCount"] > 0),
                            "PatchCount": int(target_summary["PatchCount"]),
                            "RealPatchCount": int(target_summary["RealPatchCount"]),
                            "VirtualPatchCount": int(target_summary["VirtualPatchCount"]),
                            "GradientFillPatchCount": int(target_summary["GradientFillPatchCount"]),
                            "OverflowInstanceCount": int(target_summary["OverflowInstanceCount"]),
                            "OverflowVoxelCount": int(target_summary["OverflowVoxelCount"]),
                            "MaxInstancesInSingleVoxel": int(target_summary["MaxInstancesInSingleVoxel"]),
                            "ActiveVoxelCount": int(target_summary["ActiveVoxelCount"]),
                            "InputPath": str(input_path),
                            "TargetCenterPath": str(center_path),
                            "TargetCountPath": str(count_path),
                            "TargetGeomPath": str(geom_path),
                            "TargetWeightPath": str(weight_path),
                            "MetaPath": str(meta_path),
                        }
                    )

    manifest_df = pd.DataFrame(sample_rows)
    write_csv_utf8(manifest_df, aggregated_dir / "sample_manifest.csv")
    write_csv_utf8(pd.DataFrame(skipped_rows), aggregated_dir / "skipped_units.csv")
    if not manifest_df.empty:
        unit_manifest_df = (
            manifest_df.groupby(["UnitID", "BlockX", "BlockY"], dropna=False)
            .agg(
                SampleCount=("SampleID", "count"),
                PositiveSampleCount=("IsPositiveWindow", "sum"),
                TotalPatchCount=("PatchCount", "sum"),
                OverflowInstanceCount=("OverflowInstanceCount", "sum"),
                MaxInstancesInSingleVoxel=("MaxInstancesInSingleVoxel", "max"),
            )
            .reset_index()
        )
    else:
        unit_manifest_df = pd.DataFrame(columns=["UnitID", "BlockX", "BlockY", "SampleCount", "PositiveSampleCount", "TotalPatchCount", "OverflowInstanceCount", "MaxInstancesInSingleVoxel"])
    write_csv_utf8(unit_manifest_df, aggregated_dir / "unit_manifest.csv")

    summary = {
        "run_name": args.run_name,
        "output_dir": str(run_dir),
        "stats_run_dir": str(stats_run_dir),
        "train_unit_csv": str(train_unit_csv),
        "sgy_file": str(args.sgy_file),
        "processed_unit_count": int(processed_units),
        "selected_unit_count": int(len(selected_unit_ids)),
        "total_sample_count": int(len(manifest_df)),
        "positive_sample_count": int(manifest_df["IsPositiveWindow"].sum()) if not manifest_df.empty else 0,
        "negative_sample_count": int((manifest_df["IsPositiveWindow"] == 0).sum()) if not manifest_df.empty else 0,
        "total_patch_count": int(total_patch_count),
        "total_overflow_instance_count": int(total_overflow),
        "samples_with_overflow": int(samples_with_overflow),
        "recommended_training_subset": "phase1_train_units + T128 + K16",
        "input_channels": list(INPUT_CHANNELS),
        "target_geom_channels": list(TARGET_GEOM_CHANNELS),
        "source_weights": dict(DEFAULT_SOURCE_WEIGHTS),
        "manifest_csv": str(aggregated_dir / "sample_manifest.csv"),
        "unit_manifest_csv": str(aggregated_dir / "unit_manifest.csv"),
    }
    summary_path = aggregated_dir / "dataset_summary.json"
    summary["summary_json"] = str(summary_path)
    write_json(summary_path, summary)

    append_summary_to_docx(
        docx_path=args.docx_path,
        title="GAN训练准备 - 实例表达训练样本打包",
        config={
            "output_dir": str(run_dir),
            "stats_run_dir": str(stats_run_dir),
            "train_unit_csv": str(train_unit_csv),
            "sgy_file": str(args.sgy_file),
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
            "window_size": int(args.window_size),
            "overlap_ratio": float(args.overlap_ratio),
            "slots_per_voxel": int(args.slots_per_voxel),
        },
        summary=summary,
    )

    print(f"run_dir: {run_dir}")
    print(f"processed_unit_count: {processed_units}")
    print(f"total_sample_count: {len(manifest_df)}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
