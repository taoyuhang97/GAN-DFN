# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
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
DEFAULT_INPUT_CHANNELS = [
    "seismic_amp",
    "grad_x",
    "grad_y",
    "grad_z",
    "rel_depth_in_interval",
]
OPTIONAL_INPUT_CHANNELS = {
    "seismic_amp",
    "grad_x",
    "grad_y",
    "grad_z",
    "grad_mag",
    "rel_depth_in_interval",
    "dist_to_top",
    "dist_to_base",
}
SPARSE_GEOM_CHANNELS = [
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
SOURCE_CODE_MAP = {
    "real": 1,
    "virtual": 2,
    "seismic_gradient_fill": 3,
}
SOURCE_WEIGHT_MAP = {
    "real": 1.0,
    "virtual": 0.8,
    "seismic_gradient_fill": 0.4,
}
DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将单元 DFN 样本打包为轻量版地震体 + 稀疏裂缝实例标签。")
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--sgy-file", type=Path, default=DEFAULT_SGY_FILE)
    parser.add_argument("--stats-run-dir", type=Path, required=True)
    parser.add_argument("--train-unit-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-name",
        type=str,
        default=f"phase1_t128_sparse_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--limit-units", type=int)
    parser.add_argument(
        "--input-channels",
        nargs="+",
        default=list(DEFAULT_INPUT_CHANNELS),
        choices=sorted(OPTIONAL_INPUT_CHANNELS),
    )
    parser.add_argument("--input-dtype", choices=sorted(DTYPE_MAP), default="float16")
    parser.add_argument("--geom-dtype", choices=sorted(DTYPE_MAP), default="float16")
    parser.add_argument("--omit-count-volume", action="store_true")
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
    expected_count = int(block_size * block_size)
    if len(block_df) != expected_count:
        raise ValueError(
            f"trace block size mismatch for BlockX={block_x}, BlockY={block_y}: {len(block_df)} != {expected_count}"
        )
    return block_df


def extract_unit_trace_cube(
    sgy_handle: Any,
    block_df: pd.DataFrame,
    target_z_centers: np.ndarray,
) -> np.ndarray:
    trace_indices = block_df["TraceIdx"].astype(int).to_numpy()
    sample_times = np.asarray(sgy_handle.samples, dtype=float)
    raw = np.stack([np.asarray(sgy_handle.trace[idx], dtype=np.float32) for idx in trace_indices], axis=0)
    interp = np.empty((raw.shape[0], len(target_z_centers)), dtype=np.float32)
    for idx in range(raw.shape[0]):
        interp[idx] = np.interp(
            x=target_z_centers.astype(float),
            xp=sample_times,
            fp=raw[idx].astype(float),
            left=float(raw[idx, 0]),
            right=float(raw[idx, -1]),
        ).astype(np.float32)
    return interp.reshape(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE, len(target_z_centers))


def traces_to_center_volume(trace_cube: np.ndarray) -> np.ndarray:
    return (
        trace_cube[:-1, :-1, :]
        + trace_cube[1:, :-1, :]
        + trace_cube[:-1, 1:, :]
        + trace_cube[1:, 1:, :]
    ) / 4.0


def robust_scale_signed(
    volume: np.ndarray,
    valid_mask: np.ndarray | None = None,
    q: float = 0.995,
) -> np.ndarray:
    data = np.asarray(volume, dtype=np.float32)
    finite_mask = np.isfinite(data)
    if valid_mask is not None:
        finite_mask &= np.asarray(valid_mask, dtype=bool)
    finite = data[finite_mask]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    scale = float(np.quantile(np.abs(finite), q))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip(data / scale, -1.0, 1.0).astype(np.float32)


def extract_fixed_window_volume(
    unit_center_volume: np.ndarray,
    unit_z_centers: np.ndarray,
    window_top: float,
    window_size: int,
    z_step_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, _ = unit_center_volume.shape
    target_z_centers = float(window_top) + (np.arange(int(window_size), dtype=np.float32) + 0.5) * float(z_step_ms)
    lower_bound = float(unit_z_centers[0] - z_step_ms / 2.0)
    upper_bound = float(unit_z_centers[-1] + z_step_ms / 2.0)
    valid_z_mask = (target_z_centers >= lower_bound) & (target_z_centers <= upper_bound)

    window_volume = np.zeros((nx, ny, int(window_size)), dtype=np.float32)
    if np.any(valid_z_mask):
        flat = unit_center_volume.reshape(-1, unit_center_volume.shape[-1])
        interp_flat = window_volume.reshape(-1, int(window_size))
        target_valid = target_z_centers[valid_z_mask].astype(float)
        for idx in range(flat.shape[0]):
            interp_flat[idx, valid_z_mask] = np.interp(
                x=target_valid,
                xp=unit_z_centers.astype(float),
                fp=flat[idx].astype(float),
                left=float(flat[idx, 0]),
                right=float(flat[idx, -1]),
            ).astype(np.float32)
    return window_volume, target_z_centers.astype(np.float32), valid_z_mask.astype(np.uint8)


def build_input_features(
    unit_center_volume: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    target_z_centers: np.ndarray,
    valid_z_mask: np.ndarray,
    layer_top: float,
    layer_base: float,
    input_channels: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    gx, gy, gz = np.gradient(unit_center_volume, dx, dy, dz, axis=(0, 1, 2))
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    valid_mask_3d = np.broadcast_to(valid_z_mask.reshape(1, 1, -1).astype(bool), unit_center_volume.shape)

    layer_thickness = max(abs(float(layer_base) - float(layer_top)), float(dz))
    dist_to_top = ((target_z_centers.astype(np.float32) - float(layer_top)) / layer_thickness).astype(np.float32)
    dist_to_base = ((float(layer_base) - target_z_centers.astype(np.float32)) / layer_thickness).astype(np.float32)
    rel_depth = ((target_z_centers.astype(np.float32) - float(layer_top)) / layer_thickness).astype(np.float32)

    feature_map: dict[str, np.ndarray] = {
        "seismic_amp": robust_scale_signed(unit_center_volume.astype(np.float32), valid_mask_3d),
        "grad_x": robust_scale_signed(gx.astype(np.float32), valid_mask_3d),
        "grad_y": robust_scale_signed(gy.astype(np.float32), valid_mask_3d),
        "grad_z": robust_scale_signed(gz.astype(np.float32), valid_mask_3d),
        "grad_mag": robust_scale_signed(grad_mag.astype(np.float32), valid_mask_3d),
        "rel_depth_in_interval": np.broadcast_to(np.clip(rel_depth, 0.0, 1.0).reshape(1, 1, -1), unit_center_volume.shape)
        .astype(np.float32),
        "dist_to_top": np.broadcast_to(np.clip(dist_to_top, -1.0, 2.0).reshape(1, 1, -1), unit_center_volume.shape)
        .astype(np.float32),
        "dist_to_base": np.broadcast_to(np.clip(dist_to_base, -1.0, 2.0).reshape(1, 1, -1), unit_center_volume.shape)
        .astype(np.float32),
    }
    for name in feature_map:
        feature_map[name] = (feature_map[name] * valid_mask_3d.astype(np.float32)).astype(np.float32)

    channels = np.stack([feature_map[name] for name in input_channels], axis=0).astype(np.float32)
    amp_valid = unit_center_volume[valid_mask_3d]
    grad_valid = grad_mag[valid_mask_3d]
    stats = {
        "valid_z_count": int(valid_z_mask.sum()),
        "window_depth_ms": float(len(valid_z_mask) * dz),
        "layer_thickness_ms": float(layer_thickness),
        "amp_abs_q995": float(np.quantile(np.abs(amp_valid), 0.995)) if amp_valid.size else 0.0,
        "grad_mag_abs_q995": float(np.quantile(np.abs(grad_valid), 0.995)) if grad_valid.size else 0.0,
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
        ["VoxelI", "VoxelJ", "VoxelK", "SortOffsetZ", "SortAzimuth", "SortLength", "SortHeight", "SortPatchIndex"],
        ascending=[True, True, True, True, True, False, False, True],
    ).reset_index(drop=True)


def encode_window_sparse_targets(
    patch_df: pd.DataFrame,
    grid: Any,
    window_top: float,
    window_size: int,
    z_step_ms: float,
    include_count_volume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if patch_df.empty:
        payload: dict[str, np.ndarray] = {
            "instance_ijk": np.zeros((0, 3), dtype=np.uint8),
            "instance_geom": np.zeros((0, len(SPARSE_GEOM_CHANNELS)), dtype=np.float32),
            "instance_source": np.zeros((0,), dtype=np.uint8),
            "instance_weight": np.zeros((0,), dtype=np.float32),
            "instance_patch_index": np.zeros((0,), dtype=np.int32),
        }
        if include_count_volume:
            payload["count_volume"] = np.zeros((grid.nx, grid.ny, int(window_size)), dtype=np.uint8)
        summary = {
            "PatchCount": 0,
            "InstanceCount": 0,
            "ActiveVoxelCount": 0,
            "MaxInstancesInSingleVoxel": 0,
            "RealPatchCount": 0,
            "VirtualPatchCount": 0,
            "GradientFillPatchCount": 0,
        }
        return payload, summary

    rows: list[dict[str, Any]] = []
    for row_idx, row in patch_df.iterrows():
        geom_info = patch_row_to_instance_geometry(row)
        center_phys = np.asarray(geom_info["center_phys"], dtype=float)
        center_idx = physical_to_index(center_phys.reshape(1, 3), grid)[0]
        i = int(np.clip(math.floor(center_idx[0]), 0, grid.nx - 1))
        j = int(np.clip(math.floor(center_idx[1]), 0, grid.ny - 1))
        k_local = int(
            np.clip(
                np.floor((float(center_phys[2]) - float(window_top)) / float(z_step_ms)),
                0,
                int(window_size) - 1,
            )
        )
        voxel_center = np.array(
            [grid.x_centers[i], grid.y_centers[j], float(window_top) + (k_local + 0.5) * float(z_step_ms)],
            dtype=float,
        )
        patch_index = pd.to_numeric(row.get("PatchIndex"), errors="coerce")
        azimuth = pd.to_numeric(row.get("Azimuth"), errors="coerce")
        source_kind = str(row.get("SourceKind", "")).strip()
        source_code = int(SOURCE_CODE_MAP.get(source_kind, 0))
        source_weight = float(SOURCE_WEIGHT_MAP.get(source_kind, 0.6))
        rows.append(
            {
                "PatchIndex": int(patch_index) if pd.notna(patch_index) else int(row_idx + 1),
                "PatchID": str(row.get("PatchID", "")),
                "SourceKind": source_kind,
                "SourceCode": int(source_code),
                "SourceWeight": float(source_weight),
                "Azimuth": float(azimuth) if pd.notna(azimuth) else np.nan,
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
            }
        )

    inst_df = sort_window_instances(pd.DataFrame(rows))
    if inst_df.empty:
        payload = {
            "instance_ijk": np.zeros((0, 3), dtype=np.uint8),
            "instance_geom": np.zeros((0, len(SPARSE_GEOM_CHANNELS)), dtype=np.float32),
            "instance_source": np.zeros((0,), dtype=np.uint8),
            "instance_weight": np.zeros((0,), dtype=np.float32),
            "instance_patch_index": np.zeros((0,), dtype=np.int32),
        }
        if include_count_volume:
            payload["count_volume"] = np.zeros((grid.nx, grid.ny, int(window_size)), dtype=np.uint8)
        summary = {
            "PatchCount": 0,
            "InstanceCount": 0,
            "ActiveVoxelCount": 0,
            "MaxInstancesInSingleVoxel": 0,
            "RealPatchCount": 0,
            "VirtualPatchCount": 0,
            "GradientFillPatchCount": 0,
        }
        return payload, summary

    payload = {
        "instance_ijk": inst_df[["VoxelI", "VoxelJ", "VoxelK"]].to_numpy(dtype=np.uint8),
        "instance_geom": inst_df[
            [
                "OffsetX",
                "OffsetY",
                "OffsetZ",
                "NormalX",
                "NormalY",
                "NormalZ",
                "UDirX",
                "UDirY",
                "UDirZ",
                "PatchLength",
                "PatchHeight",
                "Confidence",
            ]
        ].to_numpy(dtype=np.float32),
        "instance_source": inst_df["SourceCode"].to_numpy(dtype=np.uint8),
        "instance_weight": inst_df["SourceWeight"].to_numpy(dtype=np.float32),
        "instance_patch_index": inst_df["PatchIndex"].to_numpy(dtype=np.int32),
    }
    voxel_counts = inst_df.groupby(["VoxelI", "VoxelJ", "VoxelK"], dropna=False).size()
    if include_count_volume:
        count_volume = np.zeros((grid.nx, grid.ny, int(window_size)), dtype=np.uint8)
        for (i, j, k_local), count_value in voxel_counts.items():
            count_volume[int(i), int(j), int(k_local)] = np.uint8(min(int(count_value), 255))
        payload["count_volume"] = count_volume

    summary = {
        "PatchCount": int(len(inst_df)),
        "InstanceCount": int(len(inst_df)),
        "ActiveVoxelCount": int(len(voxel_counts)),
        "MaxInstancesInSingleVoxel": int(voxel_counts.max()) if len(voxel_counts) else 0,
        "RealPatchCount": int((inst_df["SourceKind"] == "real").sum()),
        "VirtualPatchCount": int((inst_df["SourceKind"] == "virtual").sum()),
        "GradientFillPatchCount": int((inst_df["SourceKind"] == "seismic_gradient_fill").sum()),
    }
    return payload, summary


def save_sparse_sample_package(
    output_path: Path,
    input_features: np.ndarray,
    valid_z_mask: np.ndarray,
    sparse_payload: dict[str, np.ndarray],
    input_dtype: str,
    geom_dtype: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload: dict[str, np.ndarray] = {
        "input_features": np.asarray(input_features, dtype=DTYPE_MAP[input_dtype]),
        "valid_z_mask": np.asarray(valid_z_mask, dtype=np.uint8),
        "instance_ijk": np.asarray(sparse_payload["instance_ijk"], dtype=np.uint8),
        "instance_geom": np.asarray(sparse_payload["instance_geom"], dtype=DTYPE_MAP[geom_dtype]),
        "instance_source": np.asarray(sparse_payload["instance_source"], dtype=np.uint8),
        "instance_weight": np.asarray(sparse_payload["instance_weight"], dtype=np.float16),
        "instance_patch_index": np.asarray(sparse_payload["instance_patch_index"], dtype=np.int32),
    }
    if "count_volume" in sparse_payload:
        save_payload["count_volume"] = np.asarray(sparse_payload["count_volume"], dtype=np.uint8)
    np.savez_compressed(output_path, **save_payload)
    return output_path


def append_summary_to_docx(
    docx_path: Path,
    title: str,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    lines = [
        "task: GAN 训练准备 - 轻量版稀疏实例训练样本打包",
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"output_dir: {config.get('output_dir', '')}",
        f"stats_run_dir: {config.get('stats_run_dir', '')}",
        f"train_unit_csv: {config.get('train_unit_csv', '')}",
        f"sgy_file: {config.get('sgy_file', '')}",
        f"xy_resolution: {config.get('xy_resolution', '')}",
        f"z_step_ms: {config.get('z_step_ms', '')}",
        f"window_size: {config.get('window_size', '')}",
        f"overlap_ratio: {config.get('overlap_ratio', '')}",
        f"input_channels: {', '.join(config.get('input_channels', []))}",
        f"input_dtype: {config.get('input_dtype', '')}",
        f"geom_dtype: {config.get('geom_dtype', '')}",
        f"include_count_volume: {config.get('include_count_volume', '')}",
        f"processed_unit_count: {summary.get('processed_unit_count', '')}",
        f"total_sample_count: {summary.get('total_sample_count', '')}",
        f"positive_sample_count: {summary.get('positive_sample_count', '')}",
        f"negative_sample_count: {summary.get('negative_sample_count', '')}",
        f"total_patch_count: {summary.get('total_patch_count', '')}",
        f"total_size_gb: {summary.get('total_size_gb', '')}",
        f"avg_sample_size_mb: {summary.get('avg_sample_size_mb', '')}",
        f"mean_instance_count_per_positive_window: {summary.get('mean_instance_count_per_positive_window', '')}",
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
    train_unit_csv = (
        Path(args.train_unit_csv)
        if args.train_unit_csv
        else (stats_run_dir / "aggregated" / "phase1_train_units.csv")
    )
    if not train_unit_csv.exists():
        raise FileNotFoundError(f"train_unit_csv not found: {train_unit_csv}")

    input_channels = list(dict.fromkeys(args.input_channels))
    if not input_channels:
        raise ValueError("input_channels must not be empty")

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
                    window_volume, target_z_centers, valid_z_mask = extract_fixed_window_volume(
                        unit_center_volume=unit_center_volume,
                        unit_z_centers=grid.z_centers.astype(np.float32),
                        window_top=float(window_top),
                        window_size=int(args.window_size),
                        z_step_ms=float(args.z_step_ms),
                    )
                    if int(valid_z_mask.sum()) <= 0:
                        continue

                    input_features, input_stats = build_input_features(
                        unit_center_volume=window_volume,
                        dx=grid.dx,
                        dy=grid.dy,
                        dz=grid.dz,
                        target_z_centers=target_z_centers,
                        valid_z_mask=valid_z_mask,
                        layer_top=float(layer_top),
                        layer_base=float(layer_base),
                        input_channels=input_channels,
                    )

                    subset_mask = (layer_patch_df["CenterTIME"] >= float(window_top)) & (
                        layer_patch_df["CenterTIME"] < float(window_base) + 1e-9
                    )
                    window_patch_df = layer_patch_df.loc[subset_mask].copy()
                    sparse_payload, target_summary = encode_window_sparse_targets(
                        patch_df=window_patch_df,
                        grid=grid,
                        window_top=float(window_top),
                        window_size=int(args.window_size),
                        z_step_ms=float(args.z_step_ms),
                        include_count_volume=not args.omit_count_volume,
                    )

                    sample_id = f"{unit_id}__{layer_key}__W{window_idx:03d}"
                    sample_path = samples_dir / unit_id / layer_key / f"W{window_idx:03d}.npz"
                    save_sparse_sample_package(
                        output_path=sample_path,
                        input_features=input_features,
                        valid_z_mask=valid_z_mask,
                        sparse_payload=sparse_payload,
                        input_dtype=args.input_dtype,
                        geom_dtype=args.geom_dtype,
                    )
                    package_bytes = int(sample_path.stat().st_size)

                    total_patch_count += int(target_summary["PatchCount"])
                    sample_rows.append(
                        {
                            "SampleID": sample_id,
                            "UnitID": unit_id,
                            "BlockX": block_x,
                            "BlockY": block_y,
                            "BlockXNorm": float(block_x / max(max_block_x, 1)),
                            "BlockYNorm": float(block_y / max(max_block_y, 1)),
                            "GeoIntervalKey": layer_key,
                            "StrataName": str(layer_row.get("StrataName", "")),
                            "TopSurfaceCode": str(layer_row.get("TopSurfaceCode", "")),
                            "BaseSurfaceCode": str(layer_row.get("BaseSurfaceCode", "")),
                            "WindowIndex": int(window_idx),
                            "WindowTopTime": float(window_top),
                            "WindowBaseTime": float(window_base),
                            "WindowNominalLengthMs": float(args.window_size * args.z_step_ms),
                            "WindowValidZCount": int(valid_z_mask.sum()),
                            "WindowValidLengthMs": float(valid_z_mask.sum() * args.z_step_ms),
                            "IsPositiveWindow": int(target_summary["PatchCount"] > 0),
                            "PatchCount": int(target_summary["PatchCount"]),
                            "InstanceCount": int(target_summary["InstanceCount"]),
                            "RealPatchCount": int(target_summary["RealPatchCount"]),
                            "VirtualPatchCount": int(target_summary["VirtualPatchCount"]),
                            "GradientFillPatchCount": int(target_summary["GradientFillPatchCount"]),
                            "ActiveVoxelCount": int(target_summary["ActiveVoxelCount"]),
                            "MaxInstancesInSingleVoxel": int(target_summary["MaxInstancesInSingleVoxel"]),
                            "InputChannelCount": int(len(input_channels)),
                            "InputChannels": ",".join(input_channels),
                            "InputDType": str(args.input_dtype),
                            "GeomDType": str(args.geom_dtype),
                            "CountVolumeIncluded": int(not args.omit_count_volume),
                            "AmpAbsQ995": float(input_stats["amp_abs_q995"]),
                            "GradMagAbsQ995": float(input_stats["grad_mag_abs_q995"]),
                            "LayerThicknessMs": float(input_stats["layer_thickness_ms"]),
                            "PackagePath": str(sample_path),
                            "PackageBytes": int(package_bytes),
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
                TotalInstanceCount=("InstanceCount", "sum"),
                MaxInstancesInSingleVoxel=("MaxInstancesInSingleVoxel", "max"),
                TotalPackageBytes=("PackageBytes", "sum"),
            )
            .reset_index()
        )
    else:
        unit_manifest_df = pd.DataFrame(
            columns=[
                "UnitID",
                "BlockX",
                "BlockY",
                "SampleCount",
                "PositiveSampleCount",
                "TotalPatchCount",
                "TotalInstanceCount",
                "MaxInstancesInSingleVoxel",
                "TotalPackageBytes",
            ]
        )
    write_csv_utf8(unit_manifest_df, aggregated_dir / "unit_manifest.csv")

    total_package_bytes = int(manifest_df["PackageBytes"].sum()) if not manifest_df.empty else 0
    positive_count = int(manifest_df["IsPositiveWindow"].sum()) if not manifest_df.empty else 0
    mean_instance_count = (
        float(manifest_df.loc[manifest_df["IsPositiveWindow"] == 1, "InstanceCount"].mean())
        if positive_count > 0
        else 0.0
    )
    summary = {
        "run_name": args.run_name,
        "output_dir": str(run_dir),
        "stats_run_dir": str(stats_run_dir),
        "train_unit_csv": str(train_unit_csv),
        "sgy_file": str(args.sgy_file),
        "processed_unit_count": int(processed_units),
        "selected_unit_count": int(len(selected_unit_ids)),
        "total_sample_count": int(len(manifest_df)),
        "positive_sample_count": positive_count,
        "negative_sample_count": int((manifest_df["IsPositiveWindow"] == 0).sum()) if not manifest_df.empty else 0,
        "total_patch_count": int(total_patch_count),
        "total_size_gb": round(total_package_bytes / (1024.0 ** 3), 4),
        "avg_sample_size_mb": round(total_package_bytes / max(len(manifest_df), 1) / (1024.0 ** 2), 4),
        "mean_instance_count_per_positive_window": round(mean_instance_count, 4),
        "input_channels": list(input_channels),
        "input_dtype": str(args.input_dtype),
        "geom_dtype": str(args.geom_dtype),
        "sparse_geom_channels": list(SPARSE_GEOM_CHANNELS),
        "source_code_map": dict(SOURCE_CODE_MAP),
        "source_weight_map": dict(SOURCE_WEIGHT_MAP),
        "count_volume_included": bool(not args.omit_count_volume),
        "manifest_csv": str(aggregated_dir / "sample_manifest.csv"),
        "unit_manifest_csv": str(aggregated_dir / "unit_manifest.csv"),
    }
    summary_path = aggregated_dir / "dataset_summary.json"
    summary["summary_json"] = str(summary_path)
    write_json(summary_path, summary)

    append_summary_to_docx(
        docx_path=args.docx_path,
        title="GAN训练准备 - 轻量版稀疏实例训练样本打包",
        config={
            "output_dir": str(run_dir),
            "stats_run_dir": str(stats_run_dir),
            "train_unit_csv": str(train_unit_csv),
            "sgy_file": str(args.sgy_file),
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
            "window_size": int(args.window_size),
            "overlap_ratio": float(args.overlap_ratio),
            "input_channels": list(input_channels),
            "input_dtype": str(args.input_dtype),
            "geom_dtype": str(args.geom_dtype),
            "include_count_volume": bool(not args.omit_count_volume),
        },
        summary=summary,
    )

    print(f"run_dir: {run_dir}")
    print(f"processed_unit_count: {processed_units}")
    print(f"total_sample_count: {len(manifest_df)}")
    print(f"total_size_gb: {summary['total_size_gb']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
