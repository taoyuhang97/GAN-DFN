# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
LEGACY_DIR = THIS_DIR.parent / "DFN体素互转实验"
if str(LEGACY_DIR) not in sys.path:
    sys.path.append(str(LEGACY_DIR))

from roundtrip_common import (  # type: ignore
    DEFAULT_DOCX_PATH as LEGACY_DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT as LEGACY_DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRACE_HEADER_CSV,
    DEFAULT_UNIT_DFN_ROOT,
    GridSpec,
    LAYER_SURFACE_PAIR_KEY_COL,
    PATCH_OUTPUT_COLUMNS,
    UNIT_LAYER_SEGMENT_KEY_COL,
    VtkPatchExportConfig,
    assign_layer_by_time,
    build_grid_spec,
    build_patch_match_metrics,
    build_voxel_overlap_metrics,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    compute_unit_bounds_from_trace_header,
    export_patch_comparison_vtk,
    export_patch_vtk_files,
    load_layer_table,
    load_patch_table,
    load_unit_summary,
    normal_to_azimuth_dip,
    normalize,
    open_or_create_doc,
    patch_row_to_vertices,
    physical_to_index,
    rasterize_patches_to_voxel,
    read_csv_utf8,
    resolve_layer_surface_pair_key_from_row,
    resolve_patch_csv,
    resolve_unit_layer_segment_key_from_row,
    summarize_patch_statistics,
    write_csv_utf8,
    write_json,
)

DEFAULT_OUTPUT_ROOT = LEGACY_DEFAULT_OUTPUT_ROOT.parent / "DFN实例表达互转实验"
DEFAULT_DOCX_PATH = LEGACY_DEFAULT_DOCX_PATH
DEFAULT_SLOTS_PER_VOXEL = 16

INSTANCE_LABEL_CHANNELS = [
    "center_heatmap",
    "center_count",
    "offsets",
    "normals",
    "u_dirs",
    "lengths",
    "heights",
    "confidence",
    "source_patch_index",
    "source_kind_code",
    "layer_surface_pair_code",
    "unit_layer_segment_code",
    "patch_area",
]

INSTANCE_PATCH_EXTRA_COLUMNS = [
    "CenterVoxelI",
    "CenterVoxelJ",
    "CenterVoxelK",
    "CenterSlot",
    "CenterHeatmap",
    "OriginalSourceKind",
    "OriginalLayerSurfacePairKey",
    "OriginalUnitLayerSegmentKey",
    "PatchArea",
]
INSTANCE_PATCH_OUTPUT_COLUMNS = PATCH_OUTPUT_COLUMNS + INSTANCE_PATCH_EXTRA_COLUMNS

SOURCE_KIND_PRIORITY = {
    "real": 0,
    "virtual": 1,
    "seismic_gradient_fill": 2,
}


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def build_value_code_maps(values: pd.Series | list[Any]) -> tuple[dict[str, int], dict[int, str]]:
    series = pd.Series(values, dtype="object")
    normalized_values = [normalize_text(value) for value in series.tolist()]
    unique_values = sorted({value for value in normalized_values if value})
    value_to_code = {value: idx + 1 for idx, value in enumerate(unique_values)}
    code_to_value = {code: value for value, code in value_to_code.items()}
    return value_to_code, code_to_value


def canonicalize_vector_sign(vec: np.ndarray, axis_priority: tuple[int, ...]) -> np.ndarray:
    work = normalize(np.asarray(vec, dtype=float))
    if np.linalg.norm(work) <= 1e-8:
        return work
    for axis in axis_priority:
        component = float(work[axis])
        if abs(component) > 1e-8:
            return work if component > 0.0 else -work
    return work


def canonicalize_frame(normal_vec: np.ndarray, u_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_vec = normalize(np.asarray(normal_vec, dtype=float))
    if np.linalg.norm(normal_vec) <= 1e-8:
        normal_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    normal_vec = canonicalize_vector_sign(normal_vec, axis_priority=(2, 1, 0))

    u_vec = np.asarray(u_vec, dtype=float)
    u_vec = u_vec - float(np.dot(u_vec, normal_vec)) * normal_vec
    if np.linalg.norm(u_vec) <= 1e-8:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, normal_vec))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u_vec = ref - float(np.dot(ref, normal_vec)) * normal_vec
    u_vec = normalize(u_vec)
    if np.linalg.norm(u_vec) <= 1e-8:
        u_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    u_vec = canonicalize_vector_sign(u_vec, axis_priority=(0, 1, 2))

    v_vec = normalize(np.cross(normal_vec, u_vec))
    if np.linalg.norm(v_vec) <= 1e-8:
        v_vec = np.array([0.0, 1.0, 0.0], dtype=float)
    return normal_vec, u_vec, v_vec


def build_slot_selection_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
    source_kind = normalize_text(item.get("OriginalSourceKind", ""))
    return (
        float(SOURCE_KIND_PRIORITY.get(source_kind, 9)),
        -float(item.get("Confidence", 0.0)),
        -float(item.get("PatchArea", 0.0)),
        int(item.get("PatchIndex", 0)),
    )


def build_slot_order_key(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(item.get("OffsetZ", 0.0)),
        float(item.get("OffsetX", 0.0)),
        float(item.get("OffsetY", 0.0)),
        float(item.get("NormalZ", 0.0)),
        float(item.get("NormalY", 0.0)),
        float(item.get("NormalX", 0.0)),
        float(item.get("UDirX", 0.0)),
        float(item.get("UDirY", 0.0)),
        float(item.get("UDirZ", 0.0)),
        -float(item.get("PatchLength", 0.0)),
        -float(item.get("PatchHeight", 0.0)),
        int(item.get("PatchIndex", 0)),
    )


def resolve_layer_info_from_code(
    layer_surface_pair_code: int,
    unit_layer_segment_code: int,
    center_time: float,
    layers_df: pd.DataFrame,
    label_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    code_to_pair = {
        int(key): str(value)
        for key, value in dict((label_metadata or {}).get("layer_surface_pair_mapping", {})).items()
        if str(key).strip()
    }
    code_to_segment = {
        int(key): str(value)
        for key, value in dict((label_metadata or {}).get("unit_layer_segment_mapping", {})).items()
        if str(key).strip()
    }
    target_pair = code_to_pair.get(int(layer_surface_pair_code), "")
    target_segment = code_to_segment.get(int(unit_layer_segment_code), "")
    if not layers_df.empty:
        if target_segment and UNIT_LAYER_SEGMENT_KEY_COL in layers_df.columns:
            mask = layers_df[UNIT_LAYER_SEGMENT_KEY_COL].fillna("").astype(str).eq(target_segment)
            if mask.any():
                return dict(layers_df.loc[mask].iloc[0])
        if target_pair and LAYER_SURFACE_PAIR_KEY_COL in layers_df.columns:
            mask = layers_df[LAYER_SURFACE_PAIR_KEY_COL].fillna("").astype(str).eq(target_pair)
            if mask.any():
                return dict(layers_df.loc[mask].iloc[0])
    return assign_layer_by_time(float(center_time), layers_df)


def resolve_source_kind_from_code(source_kind_code: int, label_metadata: dict[str, Any] | None = None) -> str:
    code_to_value = {
        int(key): str(value)
        for key, value in dict((label_metadata or {}).get("source_kind_mapping", {})).items()
        if str(key).strip()
    }
    return code_to_value.get(int(source_kind_code), "")


def patch_row_to_instance_geometry(row: pd.Series) -> dict[str, Any]:
    vertices = patch_row_to_vertices(row)
    center_phys = np.mean(vertices, axis=0)
    edge_a = vertices[1] - vertices[0]
    edge_b = vertices[3] - vertices[0]
    len_a = float(np.linalg.norm(edge_a))
    len_b = float(np.linalg.norm(edge_b))
    if len_a >= len_b:
        u_raw, v_raw = edge_a, edge_b
        length_phys, height_phys = len_a, len_b
    else:
        u_raw, v_raw = edge_b, edge_a
        length_phys, height_phys = len_b, len_a
    normal_phys = normalize(np.cross(u_raw, v_raw))
    if np.linalg.norm(normal_phys) <= 1e-8:
        row_normal = np.array(
            [
                float(pd.to_numeric(row.get("NormalX"), errors="coerce")),
                float(pd.to_numeric(row.get("NormalY"), errors="coerce")),
                float(pd.to_numeric(row.get("NormalZ"), errors="coerce")),
            ],
            dtype=float,
        )
        if np.all(np.isfinite(row_normal)):
            normal_phys = normalize(row_normal)
    if np.linalg.norm(normal_phys) <= 1e-8:
        normal_phys = np.array([0.0, 0.0, 1.0], dtype=float)
    u_dir = u_raw - float(np.dot(u_raw, normal_phys)) * normal_phys
    if np.linalg.norm(u_dir) <= 1e-8:
        u_dir = v_raw - float(np.dot(v_raw, normal_phys)) * normal_phys
    if np.linalg.norm(u_dir) <= 1e-8:
        u_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    normal_phys, u_dir, _ = canonicalize_frame(normal_phys, u_dir)
    row_length = pd.to_numeric(row.get("PatchLength"), errors="coerce")
    row_height = pd.to_numeric(row.get("PatchHeight"), errors="coerce")
    if pd.notna(row_length) and float(row_length) > 0:
        length_phys = float(row_length)
    if pd.notna(row_height) and float(row_height) > 0:
        height_phys = float(row_height)
    patch_area = float(max(length_phys, 1e-6) * max(height_phys, 1e-6))
    confidence = pd.to_numeric(row.get("Confidence"), errors="coerce")
    patch_index = pd.to_numeric(row.get("PatchIndex"), errors="coerce")
    return {
        "center_phys": center_phys.astype(float),
        "normal_phys": normalize(normal_phys).astype(float),
        "u_dir_phys": normalize(u_dir).astype(float),
        "length_phys": float(max(length_phys, 1e-6)),
        "height_phys": float(max(height_phys, 1e-6)),
        "patch_area": patch_area,
        "confidence": float(confidence) if pd.notna(confidence) and np.isfinite(confidence) else 1.0,
        "patch_index": int(patch_index) if pd.notna(patch_index) else -1,
    }


def build_empty_instance_label(grid: GridSpec, slots_per_voxel: int) -> dict[str, np.ndarray]:
    shape = (grid.nx, grid.ny, grid.nz, int(slots_per_voxel))
    return {
        "center_heatmap": np.zeros(shape, dtype=np.float32),
        "center_count": np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.int16),
        "offsets": np.zeros(shape + (3,), dtype=np.float32),
        "normals": np.zeros(shape + (3,), dtype=np.float32),
        "u_dirs": np.zeros(shape + (3,), dtype=np.float32),
        "lengths": np.zeros(shape, dtype=np.float32),
        "heights": np.zeros(shape, dtype=np.float32),
        "confidence": np.zeros(shape, dtype=np.float32),
        "source_patch_index": np.full(shape, -1, dtype=np.int32),
        "source_kind_code": np.zeros(shape, dtype=np.int16),
        "layer_surface_pair_code": np.zeros(shape, dtype=np.int16),
        "unit_layer_segment_code": np.zeros(shape, dtype=np.int16),
        "patch_area": np.zeros(shape, dtype=np.float32),
    }


def encode_patches_to_instance_label(
    patch_df: pd.DataFrame,
    grid: GridSpec,
    slots_per_voxel: int = DEFAULT_SLOTS_PER_VOXEL,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    slots_per_voxel = int(slots_per_voxel)
    if slots_per_voxel <= 0:
        raise ValueError("slots_per_voxel must be > 0")
    label_payload = build_empty_instance_label(grid, slots_per_voxel)
    source_kind_to_code, source_kind_mapping = build_value_code_maps(
        patch_df.get("SourceKind", pd.Series(dtype="object"))
    )
    layer_pair_to_code, layer_pair_mapping = build_value_code_maps(
        patch_df.get(LAYER_SURFACE_PAIR_KEY_COL, pd.Series(dtype="object"))
    )
    unit_segment_to_code, unit_segment_mapping = build_value_code_maps(
        patch_df.get(UNIT_LAYER_SEGMENT_KEY_COL, pd.Series(dtype="object"))
    )
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row_idx, row in patch_df.iterrows():
        geom = patch_row_to_instance_geometry(row)
        center_phys = np.asarray(geom["center_phys"], dtype=float)
        center_idx = physical_to_index(center_phys.reshape(1, 3), grid)[0]
        i = int(np.clip(math.floor(center_idx[0]), 0, grid.nx - 1))
        j = int(np.clip(math.floor(center_idx[1]), 0, grid.ny - 1))
        k = int(np.clip(math.floor(center_idx[2]), 0, grid.nz - 1))
        voxel_center = np.array([grid.x_centers[i], grid.y_centers[j], grid.z_centers[k]], dtype=float)
        patch_index = int(geom["patch_index"]) if int(geom["patch_index"]) > 0 else int(row_idx + 1)
        original_source_kind = normalize_text(row.get("SourceKind", ""))
        original_layer_pair_key = normalize_text(row.get(LAYER_SURFACE_PAIR_KEY_COL, ""))
        original_unit_segment_key = normalize_text(row.get(UNIT_LAYER_SEGMENT_KEY_COL, ""))
        groups.setdefault((i, j, k), []).append(
            {
                "PatchIndex": patch_index,
                "PatchID": str(row.get("PatchID", "")),
                "CenterVoxelI": i,
                "CenterVoxelJ": j,
                "CenterVoxelK": k,
                "CenterX": float(center_phys[0]),
                "CenterY": float(center_phys[1]),
                "CenterTIME": float(center_phys[2]),
                "OffsetX": float(center_phys[0] - voxel_center[0]),
                "OffsetY": float(center_phys[1] - voxel_center[1]),
                "OffsetZ": float(center_phys[2] - voxel_center[2]),
                "NormalX": float(np.asarray(geom["normal_phys"], dtype=float)[0]),
                "NormalY": float(np.asarray(geom["normal_phys"], dtype=float)[1]),
                "NormalZ": float(np.asarray(geom["normal_phys"], dtype=float)[2]),
                "UDirX": float(np.asarray(geom["u_dir_phys"], dtype=float)[0]),
                "UDirY": float(np.asarray(geom["u_dir_phys"], dtype=float)[1]),
                "UDirZ": float(np.asarray(geom["u_dir_phys"], dtype=float)[2]),
                "PatchLength": float(geom["length_phys"]),
                "PatchHeight": float(geom["height_phys"]),
                "PatchArea": float(geom["patch_area"]),
                "Confidence": float(geom["confidence"]),
                "OriginalSourceKind": original_source_kind,
                "OriginalLayerSurfacePairKey": original_layer_pair_key,
                "OriginalUnitLayerSegmentKey": original_unit_segment_key,
                "SourceKindCode": int(source_kind_to_code.get(original_source_kind, 0)),
                "LayerSurfacePairCode": int(layer_pair_to_code.get(original_layer_pair_key, 0)),
                "UnitLayerSegmentCode": int(unit_segment_to_code.get(original_unit_segment_key, 0)),
            }
        )

    assign_rows: list[dict[str, Any]] = []
    overflow_rows: list[dict[str, Any]] = []
    multi_voxel_count = 0
    max_voxel_count = 0
    encoded_count = 0
    for (i, j, k), items in groups.items():
        selected_items = sorted(items, key=build_slot_selection_key)
        kept_items = sorted(selected_items[:slots_per_voxel], key=build_slot_order_key)
        overflow_items = selected_items[slots_per_voxel:]
        item_count = len(items)
        label_payload["center_count"][i, j, k] = np.int16(item_count)
        max_voxel_count = max(max_voxel_count, item_count)
        if item_count > 1:
            multi_voxel_count += 1
        for item in overflow_items:
            item["CenterSlot"] = int(slots_per_voxel)
            item["CenterHeatmap"] = 1.0
            overflow_rows.append(item)
        for slot, item in enumerate(kept_items):
            item["CenterSlot"] = int(slot)
            item["CenterHeatmap"] = 1.0
            label_payload["center_heatmap"][i, j, k, slot] = 1.0
            label_payload["offsets"][i, j, k, slot, :] = np.array(
                [item["OffsetX"], item["OffsetY"], item["OffsetZ"]], dtype=np.float32
            )
            label_payload["normals"][i, j, k, slot, :] = np.array(
                [item["NormalX"], item["NormalY"], item["NormalZ"]], dtype=np.float32
            )
            label_payload["u_dirs"][i, j, k, slot, :] = np.array(
                [item["UDirX"], item["UDirY"], item["UDirZ"]], dtype=np.float32
            )
            label_payload["lengths"][i, j, k, slot] = float(item["PatchLength"])
            label_payload["heights"][i, j, k, slot] = float(item["PatchHeight"])
            label_payload["confidence"][i, j, k, slot] = float(item["Confidence"])
            label_payload["source_patch_index"][i, j, k, slot] = int(item["PatchIndex"])
            label_payload["source_kind_code"][i, j, k, slot] = np.int16(item["SourceKindCode"])
            label_payload["layer_surface_pair_code"][i, j, k, slot] = np.int16(item["LayerSurfacePairCode"])
            label_payload["unit_layer_segment_code"][i, j, k, slot] = np.int16(item["UnitLayerSegmentCode"])
            label_payload["patch_area"][i, j, k, slot] = float(item["PatchArea"])
            assign_rows.append(item)
            encoded_count += 1
    assign_df = pd.DataFrame(assign_rows)
    overflow_df = pd.DataFrame(overflow_rows)
    summary = {
        "input_patch_count": int(len(patch_df)),
        "encoded_instance_count": int(encoded_count),
        "active_center_voxel_count": int(len(groups)),
        "multi_instance_voxel_count": int(multi_voxel_count),
        "max_instances_in_single_voxel": int(max_voxel_count),
        "overflow_patch_count": int(len(overflow_rows)),
        "slots_per_voxel": int(slots_per_voxel),
        "label_density": float(encoded_count / max(grid.nx * grid.ny * grid.nz * slots_per_voxel, 1)),
        "source_kind_mapping": {int(key): value for key, value in source_kind_mapping.items()},
        "layer_surface_pair_mapping": {int(key): value for key, value in layer_pair_mapping.items()},
        "unit_layer_segment_mapping": {int(key): value for key, value in unit_segment_mapping.items()},
    }
    return label_payload, assign_df, overflow_df, summary


def ensure_valid_frame(normal_vec: np.ndarray, u_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return canonicalize_frame(normal_vec, u_vec)


def decode_instance_label_to_patches(
    label_payload: dict[str, np.ndarray],
    grid: GridSpec,
    layers_df: pd.DataFrame,
    threshold: float = 0.5,
    label_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    heatmap = np.asarray(label_payload["center_heatmap"], dtype=np.float32)
    active_idx = np.argwhere(heatmap >= float(threshold))
    offsets = np.asarray(label_payload["offsets"], dtype=np.float32)
    normals = np.asarray(label_payload["normals"], dtype=np.float32)
    u_dirs = np.asarray(label_payload["u_dirs"], dtype=np.float32)
    lengths = np.asarray(label_payload["lengths"], dtype=np.float32)
    heights = np.asarray(label_payload["heights"], dtype=np.float32)
    confidence = np.asarray(label_payload["confidence"], dtype=np.float32)
    source_patch_index = np.asarray(label_payload["source_patch_index"], dtype=np.int32)
    source_kind_code = np.asarray(
        label_payload.get("source_kind_code", np.zeros_like(heatmap, dtype=np.int16)),
        dtype=np.int16,
    )
    layer_surface_pair_code = np.asarray(
        label_payload.get("layer_surface_pair_code", np.zeros_like(heatmap, dtype=np.int16)),
        dtype=np.int16,
    )
    unit_layer_segment_code = np.asarray(
        label_payload.get("unit_layer_segment_code", np.zeros_like(heatmap, dtype=np.int16)),
        dtype=np.int16,
    )
    patch_area = np.asarray(
        label_payload.get("patch_area", np.maximum(lengths * heights, 0.0)),
        dtype=np.float32,
    )
    rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []
    for i, j, k, slot in active_idx.tolist():
        voxel_center = np.array([grid.x_centers[i], grid.y_centers[j], grid.z_centers[k]], dtype=float)
        center_phys = voxel_center + offsets[i, j, k, slot, :].astype(float)
        normal_vec, u_vec, v_vec = ensure_valid_frame(
            normals[i, j, k, slot, :].astype(float),
            u_dirs[i, j, k, slot, :].astype(float),
        )
        length_phys = float(max(lengths[i, j, k, slot], 1e-6))
        height_phys = float(max(heights[i, j, k, slot], 1e-6))
        vertices = np.array(
            [
                center_phys - (length_phys / 2.0) * u_vec - (height_phys / 2.0) * v_vec,
                center_phys + (length_phys / 2.0) * u_vec - (height_phys / 2.0) * v_vec,
                center_phys + (length_phys / 2.0) * u_vec + (height_phys / 2.0) * v_vec,
                center_phys - (length_phys / 2.0) * u_vec + (height_phys / 2.0) * v_vec,
            ],
            dtype=float,
        )
        azimuth, dip = normal_to_azimuth_dip(normal_vec)
        layer_info = resolve_layer_info_from_code(
            layer_surface_pair_code=int(layer_surface_pair_code[i, j, k, slot]),
            unit_layer_segment_code=int(unit_layer_segment_code[i, j, k, slot]),
            center_time=float(center_phys[2]),
            layers_df=layers_df,
            label_metadata=label_metadata,
        )
        original_source_kind = resolve_source_kind_from_code(
            source_kind_code=int(source_kind_code[i, j, k, slot]),
            label_metadata=label_metadata,
        )
        original_layer_pair_key = normalize_text(layer_info.get(LAYER_SURFACE_PAIR_KEY_COL, ""))
        original_unit_segment_key = normalize_text(layer_info.get(UNIT_LAYER_SEGMENT_KEY_COL, ""))
        if label_metadata is not None:
            code_to_pair = {
                int(key): str(value)
                for key, value in dict(label_metadata.get("layer_surface_pair_mapping", {})).items()
                if str(key).strip()
            }
            code_to_segment = {
                int(key): str(value)
                for key, value in dict(label_metadata.get("unit_layer_segment_mapping", {})).items()
                if str(key).strip()
            }
            original_layer_pair_key = code_to_pair.get(
                int(layer_surface_pair_code[i, j, k, slot]),
                original_layer_pair_key,
            )
            original_unit_segment_key = code_to_segment.get(
                int(unit_layer_segment_code[i, j, k, slot]),
                original_unit_segment_key,
            )
        row: dict[str, Any] = {
            "PatchIndex": int(len(rows) + 1),
            "PatchID": f"{grid.unit_id}_INSTANCE_{len(rows) + 1:04d}",
            "UnitID": str(grid.unit_id),
            "BlockX": int(grid.block_x),
            "BlockY": int(grid.block_y),
            "GeoIntervalKey": str(layer_info.get("GeoIntervalKey", "")),
            LAYER_SURFACE_PAIR_KEY_COL: resolve_layer_surface_pair_key_from_row(layer_info),
            UNIT_LAYER_SEGMENT_KEY_COL: resolve_unit_layer_segment_key_from_row(layer_info),
            "StrataName": str(layer_info.get("StrataName", "")),
            "TopSurfaceCode": str(layer_info.get("TopSurfaceCode", "")),
            "BaseSurfaceCode": str(layer_info.get("BaseSurfaceCode", "")),
            "SeedID": f"instance_center_{i:03d}_{j:03d}_{k:03d}_slot_{slot:02d}",
            "SourceKind": "instance_roundtrip",
            "SourceName": "instance_label",
            "SeedType": "center_slot",
            "ParentSeedID": str(int(source_patch_index[i, j, k, slot])),
            "ParentSourceKind": "instance_center",
            "CenterX": float(center_phys[0]),
            "CenterY": float(center_phys[1]),
            "CenterTIME": float(center_phys[2]),
            "CenterDepth": float(center_phys[2]),
            "Azimuth": float(azimuth),
            "Dip": float(dip),
            "DensityWeight": 1.0,
            "LengthWeight": 1.0,
            "Confidence": float(confidence[i, j, k, slot]),
            "PatchLength": float(length_phys),
            "PatchHeight": float(height_phys),
            "NormalX": float(normal_vec[0]),
            "NormalY": float(normal_vec[1]),
            "NormalZ": float(normal_vec[2]),
            "CenterVoxelI": int(i),
            "CenterVoxelJ": int(j),
            "CenterVoxelK": int(k),
            "CenterSlot": int(slot),
            "CenterHeatmap": float(heatmap[i, j, k, slot]),
            "OriginalSourceKind": original_source_kind,
            "OriginalLayerSurfacePairKey": original_layer_pair_key,
            "OriginalUnitLayerSegmentKey": original_unit_segment_key,
            "PatchArea": float(max(patch_area[i, j, k, slot], 0.0)),
        }
        for vertex_idx, vertex in enumerate(vertices, start=1):
            row[f"V{vertex_idx}X"] = float(vertex[0])
            row[f"V{vertex_idx}Y"] = float(vertex[1])
            row[f"V{vertex_idx}Z"] = float(vertex[2])
        rows.append(row)
        decoded_rows.append(
            {
                "PatchID": row["PatchID"],
                "ParentPatchIndex": int(source_patch_index[i, j, k, slot]),
                "CenterVoxelI": int(i),
                "CenterVoxelJ": int(j),
                "CenterVoxelK": int(k),
                "CenterSlot": int(slot),
                "CenterX": float(center_phys[0]),
                "CenterY": float(center_phys[1]),
                "CenterTIME": float(center_phys[2]),
                "PatchLength": float(length_phys),
                "PatchHeight": float(height_phys),
                "Azimuth": float(azimuth),
                "Dip": float(dip),
                "Confidence": float(confidence[i, j, k, slot]),
                "OriginalSourceKind": original_source_kind,
                "OriginalLayerSurfacePairKey": original_layer_pair_key,
                "OriginalUnitLayerSegmentKey": original_unit_segment_key,
                "PatchArea": float(max(patch_area[i, j, k, slot], 0.0)),
            }
        )
    patch_df = pd.DataFrame(rows)
    if patch_df.empty:
        patch_df = pd.DataFrame(columns=INSTANCE_PATCH_OUTPUT_COLUMNS)
    else:
        for col in INSTANCE_PATCH_OUTPUT_COLUMNS:
            if col not in patch_df.columns:
                patch_df[col] = np.nan if col not in {
                    "PatchID",
                    "UnitID",
                    "GeoIntervalKey",
                    "StrataName",
                    "TopSurfaceCode",
                    "BaseSurfaceCode",
                    "SeedID",
                    "SourceKind",
                    "SourceName",
                    "SeedType",
                    "ParentSeedID",
                    "ParentSourceKind",
                    "OriginalSourceKind",
                    "OriginalLayerSurfacePairKey",
                    "OriginalUnitLayerSegmentKey",
                } else ""
        patch_df = patch_df[INSTANCE_PATCH_OUTPUT_COLUMNS].copy()
    decoded_df = pd.DataFrame(decoded_rows)
    summary = {
        "decoded_patch_count": int(len(patch_df)),
        "center_threshold": float(threshold),
        "active_slot_count": int(len(active_idx)),
    }
    return patch_df, decoded_df, summary


def save_instance_label_package(
    output_path: Path,
    label_payload: dict[str, np.ndarray],
    grid: GridSpec,
    metadata: dict[str, Any],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload: dict[str, Any] = {
        "x_centers": grid.x_centers.astype(np.float32),
        "y_centers": grid.y_centers.astype(np.float32),
        "z_centers": grid.z_centers.astype(np.float32),
    }
    for name in INSTANCE_LABEL_CHANNELS:
        save_payload[name] = np.asarray(label_payload[name])
    np.savez_compressed(output_path, **save_payload)
    meta_payload = dict(metadata)
    meta_payload["grid"] = grid.to_json_dict()
    meta_payload["label_channels"] = list(INSTANCE_LABEL_CHANNELS)
    write_json(output_path.with_suffix(".json"), meta_payload)
    return output_path


def load_instance_label_package(npz_path: Path) -> tuple[dict[str, np.ndarray], GridSpec, dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    label_payload = {name: data[name] for name in INSTANCE_LABEL_CHANNELS if name in data.files}
    meta_path = npz_path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    grid = GridSpec(**metadata["grid"])
    return label_payload, grid, metadata


def append_instance_roundtrip_summary_to_docx(
    docx_path: Path,
    title: str,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    lines = [
        "task: 单元 DFN 实例表达互转验证",
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"unit_id: {config.get('unit_id', '')}",
        f"patch_csv: {config.get('patch_csv', '')}",
        f"output_dir: {config.get('output_dir', '')}",
        f"xy_resolution: {config.get('xy_resolution', '')}",
        f"z_step_ms: {config.get('z_step_ms', '')}",
        f"slots_per_voxel: {config.get('slots_per_voxel', '')}",
        f"center_threshold: {config.get('center_threshold', '')}",
        f"thickness_vox_for_compare: {config.get('thickness_vox_for_compare', '')}",
        f"input_patch_count: {summary.get('input_patch_count', '')}",
        f"encoded_instance_count: {summary.get('encoded_instance_count', '')}",
        f"decoded_patch_count: {summary.get('decoded_patch_count', '')}",
        f"active_center_voxel_count: {summary.get('active_center_voxel_count', '')}",
        f"multi_instance_voxel_count: {summary.get('multi_instance_voxel_count', '')}",
        f"max_instances_in_single_voxel: {summary.get('max_instances_in_single_voxel', '')}",
        f"overflow_patch_count: {summary.get('overflow_patch_count', '')}",
        f"voxel_iou: {summary.get('voxel_iou', '')}",
        f"matched_patch_count: {summary.get('matched_patch_count', '')}",
        f"center_offset_median: {summary.get('center_offset_median', '')}",
        f"azimuth_diff_median_deg: {summary.get('azimuth_diff_median_deg', '')}",
        f"dip_diff_median_deg: {summary.get('dip_diff_median_deg', '')}",
        f"length_rel_error_median: {summary.get('length_rel_error_median', '')}",
        f"height_rel_error_median: {summary.get('height_rel_error_median', '')}",
        f"input_patches_display_vtk: {summary.get('input_patches_display_vtk', '')}",
        f"roundtrip_patches_display_vtk: {summary.get('roundtrip_patches_display_vtk', '')}",
        f"patch_compare_display_vtk: {summary.get('patch_compare_display_vtk', '')}",
        f"summary_json: {summary.get('summary_json', '')}",
    ]
    doc.add_paragraph("\n".join(lines))
    doc.add_paragraph("")
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))
