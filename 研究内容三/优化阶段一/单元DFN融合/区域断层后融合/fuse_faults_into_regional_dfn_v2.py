# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merge_unit_dfn_vtks import merge_vtk_payloads, read_legacy_vtk_polygons
from fault_postfusion_common import (
    DEFAULT_DOCX_PATH,
    FAULT_ACTION_TEXT_TO_CODE,
    PATCH_ORIGIN_TEXT_TO_CODE,
    append_lines_to_docx,
    make_patch_row_from_axes,
    normalize_vector,
    predict_fault_time_at_xy,
    read_regional_vtk_to_df,
    scale_patch_row_geometry,
    write_legacy_vtk_polygons_preserve_patch_area,
    write_csv_utf8,
    write_df_to_regional_vtk,
    write_json,
)


SIZE_LABELS = ["small", "medium", "large"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse regional fault panels into a postprocessed regional DFN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--fault-panel-csv", type=Path, required=True)
    parser.add_argument("--fault-surface-vtk", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=f"regional_fault_postfusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--fault-half-band-ms", type=float, default=100.0)
    parser.add_argument("--fault-remove-ms", type=float, default=50.0)
    parser.add_argument("--fault-transition-ms", type=float, default=100.0)
    parser.add_argument("--panel-xy-buffer", type=float, default=180.0)
    parser.add_argument("--parallel-ratio", type=float, default=0.65)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def load_fault_panels(panel_csv: Path) -> pd.DataFrame:
    panel_df = pd.read_csv(panel_csv, encoding="utf-8-sig")
    if panel_df.empty:
        raise ValueError(f"fault panel csv is empty: {panel_csv}")
    numeric_cols = [
        "FaultPanelID",
        "SourcePatchCount",
        "SourceUnitCount",
        "CenterX",
        "CenterY",
        "CenterTIME",
        "StrikeDeg",
        "DipDeg",
        "PanelLength",
        "PanelHeight",
        "PanelArea",
        "NormalX",
        "NormalY",
        "NormalZ",
        "StrikeVecX",
        "StrikeVecY",
        "StrikeVecZ",
        "DipVecX",
        "DipVecY",
        "DipVecZ",
        "BBoxXMin",
        "BBoxXMax",
        "BBoxYMin",
        "BBoxYMax",
        "BBoxZMin",
        "BBoxZMax",
    ] + [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
    for col in numeric_cols:
        if col in panel_df.columns:
            panel_df[col] = pd.to_numeric(panel_df[col], errors="coerce")
    return panel_df


def build_zone_specs(fault_half_band_ms: float) -> list[dict[str, Any]]:
    half_band = max(float(fault_half_band_ms), 12.0)
    edges = np.linspace(0.0, half_band, 4)
    return [
        {
            "name": "near",
            "d_min": float(edges[0]),
            "d_max": float(edges[1]),
            "count_weight": 1.50,
            "size_probs": [0.64, 0.27, 0.09],
            "confidence": 0.92,
            "size_scale": 1.55,
        },
        {
            "name": "mid",
            "d_min": float(edges[1]),
            "d_max": float(edges[2]),
            "count_weight": 0.95,
            "size_probs": [0.78, 0.18, 0.04],
            "confidence": 0.82,
            "size_scale": 1.12,
        },
        {
            "name": "far",
            "d_min": float(edges[2]),
            "d_max": float(edges[3]),
            "count_weight": 0.60,
            "size_probs": [0.88, 0.10, 0.02],
            "confidence": 0.72,
            "size_scale": 0.86,
        },
    ]


def assign_fault_influence(
    df: pd.DataFrame,
    panel_df: pd.DataFrame,
    fault_half_band_ms: float,
    fault_remove_ms: float,
    fault_transition_ms: float,
    panel_xy_buffer: float,
) -> pd.DataFrame:
    result = df.copy()
    result["PatchOriginCode"] = pd.to_numeric(result.get("PatchOriginCode", 0), errors="coerce").fillna(0).astype(int)
    result["PatchOriginText"] = result.get("PatchOriginText", "original")
    result["FaultActionCode"] = 0
    result["FaultActionText"] = "keep"
    result["FaultDistanceMs"] = np.nan
    result["FaultInfluenceWeight"] = 0.0
    result["NearestFaultPanelID"] = -1
    result["NearestFaultName"] = ""

    if panel_df.empty or result.empty:
        return result

    panel_arrays = {
        "id": panel_df["FaultPanelID"].to_numpy(dtype=int),
        "name": panel_df["FaultName"].astype(str).to_numpy(),
        "xmin": panel_df["BBoxXMin"].to_numpy(dtype=float) - float(panel_xy_buffer),
        "xmax": panel_df["BBoxXMax"].to_numpy(dtype=float) + float(panel_xy_buffer),
        "ymin": panel_df["BBoxYMin"].to_numpy(dtype=float) - float(panel_xy_buffer),
        "ymax": panel_df["BBoxYMax"].to_numpy(dtype=float) + float(panel_xy_buffer),
        "cx": panel_df["CenterX"].to_numpy(dtype=float),
        "cy": panel_df["CenterY"].to_numpy(dtype=float),
        "cz": panel_df["CenterTIME"].to_numpy(dtype=float),
        "nx": panel_df["NormalX"].to_numpy(dtype=float),
        "ny": panel_df["NormalY"].to_numpy(dtype=float),
        "nz": panel_df["NormalZ"].to_numpy(dtype=float),
    }
    valid_nz = np.abs(panel_arrays["nz"]) > 1e-8

    for idx, row in result.iterrows():
        x = float(row["CenterX"])
        y = float(row["CenterY"])
        z = float(row["CenterTIME"])
        candidate_mask = (
            (panel_arrays["xmin"] <= x)
            & (x <= panel_arrays["xmax"])
            & (panel_arrays["ymin"] <= y)
            & (y <= panel_arrays["ymax"])
        )
        if not np.any(candidate_mask):
            continue
        fault_z = panel_arrays["cz"][candidate_mask].copy()
        cand_valid_nz = valid_nz[candidate_mask]
        if np.any(cand_valid_nz):
            fault_z[cand_valid_nz] = (
                panel_arrays["cz"][candidate_mask][cand_valid_nz]
                - (
                    panel_arrays["nx"][candidate_mask][cand_valid_nz] * (x - panel_arrays["cx"][candidate_mask][cand_valid_nz])
                    + panel_arrays["ny"][candidate_mask][cand_valid_nz] * (y - panel_arrays["cy"][candidate_mask][cand_valid_nz])
                )
                / panel_arrays["nz"][candidate_mask][cand_valid_nz]
            )
        distances = np.abs(z - fault_z)
        min_pos = int(np.argmin(distances))
        min_dist = float(distances[min_pos])
        chosen_panel_id = int(panel_arrays["id"][candidate_mask][min_pos])
        chosen_fault_name = str(panel_arrays["name"][candidate_mask][min_pos])
        result.at[idx, "NearestFaultPanelID"] = chosen_panel_id
        result.at[idx, "NearestFaultName"] = chosen_fault_name
        result.at[idx, "FaultDistanceMs"] = min_dist
        influence_weight = max(0.0, 1.0 - min_dist / max(float(fault_half_band_ms), 1e-6))
        result.at[idx, "FaultInfluenceWeight"] = influence_weight
        if min_dist <= float(fault_remove_ms):
            result.at[idx, "FaultActionText"] = "remove"
            result.at[idx, "FaultActionCode"] = FAULT_ACTION_TEXT_TO_CODE["remove"]
        elif min_dist <= float(fault_transition_ms):
            result.at[idx, "FaultActionText"] = "shrink"
            result.at[idx, "FaultActionCode"] = FAULT_ACTION_TEXT_TO_CODE["shrink"]
    return result


def apply_fault_filter_and_shrink(
    influenced_df: pd.DataFrame,
    fault_remove_ms: float,
    fault_transition_ms: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    kept_rows: list[pd.Series] = []
    removed_count = 0
    shrunk_count = 0
    for _, row in influenced_df.iterrows():
        action = str(row.get("FaultActionText", "keep"))
        if action == "remove":
            removed_count += 1
            continue
        updated = row.copy()
        if action == "shrink":
            distance_ms = float(row.get("FaultDistanceMs", np.nan))
            denom = max(float(fault_transition_ms) - float(fault_remove_ms), 1e-6)
            area_ratio = np.clip((distance_ms - float(fault_remove_ms)) / denom, 0.0, 1.0)
            updated = scale_patch_row_geometry(updated, float(area_ratio))
            updated["FaultShrinkAreaRatio"] = float(area_ratio)
            shrunk_count += 1
        else:
            updated["FaultShrinkAreaRatio"] = 1.0
        kept_rows.append(updated)
    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else influenced_df.iloc[0:0].copy()
    stats = {
        "input_patch_count": int(len(influenced_df)),
        "kept_patch_count": int(len(kept_df)),
        "removed_patch_count": int(removed_count),
        "shrunk_patch_count": int(shrunk_count),
    }
    return kept_df, stats


def sample_patch_dimensions(panel_row: pd.Series, zone_spec: dict[str, Any], rng: np.random.Generator) -> tuple[float, float, str]:
    panel_length = max(float(panel_row.get("PanelLength", 20.0)), 20.0)
    panel_height = max(float(panel_row.get("PanelHeight", 8.0)), 8.0)
    size_label = str(rng.choice(SIZE_LABELS, p=np.asarray(zone_spec["size_probs"], dtype=float)))
    size_scale = float(zone_spec["size_scale"])
    if size_label == "small":
        max_length = min(max(panel_length * 0.16 * size_scale, 18.0), 42.0 * size_scale)
        max_height = min(max(panel_height * 0.18 * size_scale, 5.0), 14.0 * size_scale)
        min_length = max(6.0, min(12.0, max_length * 0.55))
        min_height = max(2.0, min(4.0, max_height * 0.45))
    elif size_label == "medium":
        max_length = min(max(panel_length * 0.34 * size_scale, 35.0), 95.0 * size_scale)
        max_height = min(max(panel_height * 0.32 * size_scale, 10.0), 28.0 * size_scale)
        min_length = max(18.0, min(36.0, max_length * 0.55))
        min_height = max(5.0, min(10.0, max_height * 0.45))
    else:
        max_length = min(max(panel_length * 0.62 * size_scale, 80.0), 220.0 * size_scale)
        max_height = min(max(panel_height * 0.58 * size_scale, 18.0), 70.0 * size_scale)
        min_length = max(35.0, min(80.0, max_length * 0.50))
        min_height = max(8.0, min(18.0, max_height * 0.45))
    if max_length <= min_length:
        max_length = min_length
    if max_height <= min_height:
        max_height = min_height
    patch_length = float(rng.uniform(min_length, max_length))
    patch_height = float(rng.uniform(min_height, max_height))
    return patch_length, patch_height, size_label


def build_fault_horizontal_normal(panel_row: pd.Series, strike_vec: np.ndarray) -> np.ndarray:
    horizontal = np.array(
        [
            float(panel_row.get("NormalX", 0.0)),
            float(panel_row.get("NormalY", 0.0)),
            0.0,
        ],
        dtype=float,
    )
    if float(np.linalg.norm(horizontal)) > 1e-8:
        return normalize_vector(horizontal, fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    fallback = np.array([-float(strike_vec[1]), float(strike_vec[0]), 0.0], dtype=float)
    return normalize_vector(fallback, fallback=np.array([1.0, 0.0, 0.0], dtype=float))


def build_vertical_perpendicular_axes(panel_row: pd.Series, strike_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal_normal = build_fault_horizontal_normal(panel_row, strike_vec)
    vertical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    return horizontal_normal, vertical_axis


def sample_panel_offsets(
    panel_length: float,
    panel_height: float,
    patch_length: float,
    patch_height: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    max_strike_offset = max(0.0, 0.5 * float(panel_length) - 0.55 * float(patch_length))
    max_dip_offset = max(0.0, 0.5 * float(panel_height) - 0.55 * float(patch_height))
    strike_offset = float(rng.uniform(-max_strike_offset, max_strike_offset)) if max_strike_offset > 0.0 else 0.0
    dip_offset = float(rng.uniform(-max_dip_offset, max_dip_offset)) if max_dip_offset > 0.0 else 0.0
    return strike_offset, dip_offset


def build_fault_offset_center(
    panel_row: pd.Series,
    base_point: np.ndarray,
    fault_time: float,
    distance_ms: float,
    fault_half_band_ms: float,
    side_sign: float,
    horizontal_normal: np.ndarray,
    panel_length: float,
) -> np.ndarray:
    base_xy_offset = float(np.clip(0.18 * float(panel_length), 20.0, 65.0))
    zone_ratio = float(np.clip(distance_ms / max(float(fault_half_band_ms), 1e-6), 0.0, 1.0))
    xy_offset = (0.35 + 0.65 * zone_ratio) * base_xy_offset
    return np.array(
        [
            float(base_point[0] + side_sign * xy_offset * horizontal_normal[0]),
            float(base_point[1] + side_sign * xy_offset * horizontal_normal[1]),
            float(fault_time + side_sign * distance_ms),
        ],
        dtype=float,
    )


def center_is_far_enough(
    candidate_center: np.ndarray,
    existing_centers: list[np.ndarray],
    min_xy_distance: float,
    min_time_distance: float,
) -> bool:
    if not existing_centers:
        return True
    for existing in existing_centers:
        xy_distance = float(np.linalg.norm(candidate_center[:2] - existing[:2]))
        time_distance = abs(float(candidate_center[2] - existing[2]))
        if xy_distance < float(min_xy_distance) and time_distance < float(min_time_distance):
            return False
    return True


def build_generated_patch_rows(
    panel_df: pd.DataFrame,
    parallel_ratio: float,
    random_seed: int,
    fault_half_band_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(random_seed))
    parallel_rows: list[dict[str, Any]] = []
    perpendicular_rows: list[dict[str, Any]] = []
    parallel_ratio = float(np.clip(parallel_ratio, 0.0, 1.0))
    zone_specs = build_zone_specs(float(fault_half_band_ms))

    for _, panel_row in panel_df.iterrows():
        panel_id = int(panel_row["FaultPanelID"])
        fault_name = str(panel_row["FaultName"])
        center = np.array([float(panel_row["CenterX"]), float(panel_row["CenterY"]), float(panel_row["CenterTIME"])], dtype=float)
        strike_vec = np.array([float(panel_row["StrikeVecX"]), float(panel_row["StrikeVecY"]), float(panel_row["StrikeVecZ"])], dtype=float)
        dip_vec = np.array([float(panel_row["DipVecX"]), float(panel_row["DipVecY"]), float(panel_row["DipVecZ"])], dtype=float)
        normal = np.array([float(panel_row["NormalX"]), float(panel_row["NormalY"]), float(panel_row["NormalZ"])], dtype=float)
        panel_length = max(float(panel_row["PanelLength"]), 10.0)
        panel_height = max(float(panel_row["PanelHeight"]), 6.0)
        source_patch_count = max(int(panel_row.get("SourcePatchCount", 1)), 1)
        source_unit_count = max(int(panel_row.get("SourceUnitCount", 1)), 1)
        horizontal_normal = build_fault_horizontal_normal(panel_row, strike_vec)
        perpendicular_u_vec, perpendicular_v_vec = build_vertical_perpendicular_axes(panel_row, strike_vec)
        accepted_parallel_centers: list[np.ndarray] = []
        accepted_perpendicular_centers: list[np.ndarray] = []

        for zone_spec in zone_specs:
            total_count = max(1, int(round(source_patch_count * float(zone_spec["count_weight"]))))
            parallel_count = max(1, int(round(total_count * parallel_ratio)))
            perpendicular_count = max(0, total_count - parallel_count)
            if perpendicular_count == 0:
                perpendicular_count = 1

            for _ in range(parallel_count):
                patch_length, patch_height, size_label = sample_patch_dimensions(panel_row, zone_spec, rng)
                min_xy_distance = max(12.0, 0.45 * patch_length)
                min_time_distance = max(1.0, 0.35 * patch_height)
                patch_center = None
                zone_center_distance = 0.0
                zone_weight = 0.0
                for _attempt in range(16):
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, patch_length, patch_height, rng)
                    base_point = center + strike_offset * strike_vec + dip_offset * dip_vec
                    fault_time = predict_fault_time_at_xy(panel_row, float(base_point[0]), float(base_point[1]))
                    zone_center_distance = max(float(rng.uniform(zone_spec["d_min"], zone_spec["d_max"])), 1.0)
                    zone_weight = max(0.0, 1.0 - zone_center_distance / max(float(fault_half_band_ms), 1e-6))
                    time_sign = -1.0 if rng.random() < 0.5 else 1.0
                    candidate_center = build_fault_offset_center(
                        panel_row=panel_row,
                        base_point=base_point,
                        fault_time=fault_time,
                        distance_ms=zone_center_distance,
                        fault_half_band_ms=float(fault_half_band_ms),
                        side_sign=time_sign,
                        horizontal_normal=horizontal_normal,
                        panel_length=panel_length,
                    )
                    if center_is_far_enough(candidate_center, accepted_parallel_centers, min_xy_distance, min_time_distance):
                        patch_center = candidate_center
                        accepted_parallel_centers.append(candidate_center)
                        break
                if patch_center is None:
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, patch_length, patch_height, rng)
                    base_point = center + strike_offset * strike_vec + dip_offset * dip_vec
                    fault_time = predict_fault_time_at_xy(panel_row, float(base_point[0]), float(base_point[1]))
                    zone_center_distance = max(float(rng.uniform(zone_spec["d_min"], zone_spec["d_max"])), 1.0)
                    zone_weight = max(0.0, 1.0 - zone_center_distance / max(float(fault_half_band_ms), 1e-6))
                    time_sign = -1.0 if rng.random() < 0.5 else 1.0
                    patch_center = build_fault_offset_center(
                        panel_row=panel_row,
                        base_point=base_point,
                        fault_time=fault_time,
                        distance_ms=zone_center_distance,
                        fault_half_band_ms=float(fault_half_band_ms),
                        side_sign=time_sign,
                        horizontal_normal=horizontal_normal,
                        panel_length=panel_length,
                    )
                parallel_rows.append(
                    make_patch_row_from_axes(
                        center=patch_center,
                        u_vec=strike_vec,
                        v_vec=dip_vec,
                        length=patch_length,
                        height=patch_height,
                        extra={
                            "Confidence": float(zone_spec["confidence"]),
                            "PatchOriginCode": PATCH_ORIGIN_TEXT_TO_CODE["fault_parallel"],
                            "PatchOriginText": "fault_parallel",
                            "FaultActionCode": FAULT_ACTION_TEXT_TO_CODE["induced"],
                            "FaultActionText": "induced",
                            "FaultDistanceMs": float(zone_center_distance),
                            "FaultInfluenceWeight": float(zone_weight),
                            "FaultPanelID": panel_id,
                            "NearestFaultPanelID": panel_id,
                            "NearestFaultName": fault_name,
                            "FaultName": fault_name,
                            "SourcePatchCount": source_patch_count,
                            "SourceUnitCount": source_unit_count,
                            "ParentPatchCount": 1,
                            "ReliabilityLevel": "high" if size_label == "large" else "medium",
                            "ConnectionType": "original",
                            "AggregationMode": "original",
                            "ScaleClass": "macro_core" if size_label == "large" else ("meso_link" if size_label == "medium" else "micro_bg"),
                            "CorridorSupport": 1,
                            "IsSupplemented": 0,
                            "FractureSet": -1,
                        },
                    )
                )
            for _ in range(perpendicular_count):
                patch_length, patch_height, size_label = sample_patch_dimensions(panel_row, zone_spec, rng)
                patch_height = max(patch_height, zone_spec["size_scale"] * 10.0)
                min_xy_distance = max(12.0, 0.40 * patch_length)
                min_time_distance = max(1.0, 0.30 * patch_height)
                patch_center = None
                zone_center_distance = 0.0
                zone_weight = 0.0
                for _attempt in range(16):
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, patch_length, patch_height, rng)
                    base_point = center + strike_offset * strike_vec + dip_offset * dip_vec
                    fault_time = predict_fault_time_at_xy(panel_row, float(base_point[0]), float(base_point[1]))
                    zone_center_distance = max(float(rng.uniform(zone_spec["d_min"], zone_spec["d_max"])), 1.0)
                    zone_weight = max(0.0, 1.0 - zone_center_distance / max(float(fault_half_band_ms), 1e-6))
                    time_sign = -1.0 if rng.random() < 0.5 else 1.0
                    candidate_center = build_fault_offset_center(
                        panel_row=panel_row,
                        base_point=base_point,
                        fault_time=fault_time,
                        distance_ms=zone_center_distance,
                        fault_half_band_ms=float(fault_half_band_ms),
                        side_sign=time_sign,
                        horizontal_normal=horizontal_normal,
                        panel_length=panel_length,
                    )
                    if center_is_far_enough(candidate_center, accepted_perpendicular_centers, min_xy_distance, min_time_distance):
                        patch_center = candidate_center
                        accepted_perpendicular_centers.append(candidate_center)
                        break
                if patch_center is None:
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, patch_length, patch_height, rng)
                    base_point = center + strike_offset * strike_vec + dip_offset * dip_vec
                    fault_time = predict_fault_time_at_xy(panel_row, float(base_point[0]), float(base_point[1]))
                    zone_center_distance = max(float(rng.uniform(zone_spec["d_min"], zone_spec["d_max"])), 1.0)
                    zone_weight = max(0.0, 1.0 - zone_center_distance / max(float(fault_half_band_ms), 1e-6))
                    time_sign = -1.0 if rng.random() < 0.5 else 1.0
                    patch_center = build_fault_offset_center(
                        panel_row=panel_row,
                        base_point=base_point,
                        fault_time=fault_time,
                        distance_ms=zone_center_distance,
                        fault_half_band_ms=float(fault_half_band_ms),
                        side_sign=time_sign,
                        horizontal_normal=horizontal_normal,
                        panel_length=panel_length,
                    )
                perpendicular_rows.append(
                    make_patch_row_from_axes(
                        center=patch_center,
                        u_vec=perpendicular_u_vec,
                        v_vec=perpendicular_v_vec,
                        length=patch_length,
                        height=patch_height,
                        extra={
                            "Confidence": float(zone_spec["confidence"]) * 0.95,
                            "PatchOriginCode": PATCH_ORIGIN_TEXT_TO_CODE["fault_perpendicular"],
                            "PatchOriginText": "fault_perpendicular",
                            "FaultActionCode": FAULT_ACTION_TEXT_TO_CODE["induced"],
                            "FaultActionText": "induced",
                            "FaultDistanceMs": float(zone_center_distance),
                            "FaultInfluenceWeight": float(zone_weight),
                            "FaultPanelID": panel_id,
                            "NearestFaultPanelID": panel_id,
                            "NearestFaultName": fault_name,
                            "FaultName": fault_name,
                            "SourcePatchCount": source_patch_count,
                            "SourceUnitCount": source_unit_count,
                            "ParentPatchCount": 1,
                            "ReliabilityLevel": "high" if size_label == "large" else "medium",
                            "ConnectionType": "original",
                            "AggregationMode": "original",
                            "ScaleClass": "macro_core" if size_label == "large" else ("meso_link" if size_label == "medium" else "micro_bg"),
                            "CorridorSupport": 1,
                            "IsSupplemented": 0,
                            "FractureSet": -1,
                        },
                    )
                )
    return pd.DataFrame(parallel_rows), pd.DataFrame(perpendicular_rows)


def finalize_output_dataframe(
    original_df: pd.DataFrame,
    parallel_df: pd.DataFrame,
    perpendicular_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = [frame for frame in [original_df, parallel_df, perpendicular_df] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged_df = pd.concat(frames, ignore_index=True, sort=False)
    defaults: dict[str, Any] = {
        "Confidence": 0.5,
        "PatchOriginCode": 0,
        "PatchOriginText": "original",
        "FaultActionCode": 0,
        "FaultActionText": "keep",
        "FaultInfluenceWeight": 0.0,
        "FaultPanelID": -1,
        "NearestFaultPanelID": -1,
        "NearestFaultName": "",
        "FaultName": "",
        "SourcePatchCount": 1,
        "SourceUnitCount": 1,
        "ParentPatchCount": 1,
        "ReliabilityLevel": "medium",
        "ConnectionType": "original",
        "AggregationMode": "original",
        "ScaleClass": "micro_bg",
        "CorridorSupport": 0,
        "IsSupplemented": 0,
        "FractureSet": -1,
    }
    for col, default in defaults.items():
        if col not in merged_df.columns:
            merged_df[col] = default
        else:
            merged_df[col] = merged_df[col].fillna(default)
    if "FaultDistanceMs" not in merged_df.columns:
        merged_df["FaultDistanceMs"] = np.nan
    if "FaultShrinkAreaRatio" not in merged_df.columns:
        merged_df["FaultShrinkAreaRatio"] = 1.0
    preferred_cols = list(original_df.columns)
    for col in merged_df.columns:
        if col not in preferred_cols:
            preferred_cols.append(col)
    return merged_df.loc[:, preferred_cols]


def combine_with_fault_surface_vtk(
    fracture_vtk: Path,
    fault_surface_vtk: Path | None,
    output_vtk: Path,
    title: str,
) -> dict[str, Any]:
    fracture_payload = read_legacy_vtk_polygons(Path(fracture_vtk))
    surface_polygon_count = 0
    if fault_surface_vtk is None or not Path(fault_surface_vtk).exists():
        write_legacy_vtk_polygons_preserve_patch_area(
            path=Path(output_vtk),
            title=str(title),
            points=np.asarray(fracture_payload["points"], dtype=float),
            polygons=fracture_payload["polygons"],
            cell_data=fracture_payload["cell_data"],
            scalar_types=fracture_payload["scalar_types"],
            recompute_patch_area=True,
        )
        return {
            "surface_polygon_count": 0,
            "final_polygon_count": int(len(fracture_payload["polygons"])),
        }

    surface_payload = read_legacy_vtk_polygons(Path(fault_surface_vtk))
    surface_polygon_count = int(len(surface_payload["polygons"]))
    merged_payload = merge_vtk_payloads([fracture_payload, surface_payload])
    write_legacy_vtk_polygons_preserve_patch_area(
        path=Path(output_vtk),
        title=str(title),
        points=np.asarray(merged_payload["points"], dtype=float),
        polygons=merged_payload["polygons"],
        cell_data=merged_payload["cell_data"],
        scalar_types=merged_payload["scalar_types"],
        recompute_patch_area=True,
    )
    return {
        "surface_polygon_count": surface_polygon_count,
        "final_polygon_count": int(len(merged_payload["polygons"])),
    }


def run_fault_postfusion(
    input_vtk: Path,
    fault_panel_csv: Path,
    fault_surface_vtk: Path | None,
    output_root: Path,
    run_name: str,
    fault_half_band_ms: float,
    fault_remove_ms: float,
    fault_transition_ms: float,
    panel_xy_buffer: float,
    parallel_ratio: float,
    random_seed: int,
) -> dict[str, Any]:
    run_dir = Path(output_root) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    regional_df, scalar_types, title = read_regional_vtk_to_df(Path(input_vtk))
    panel_df = load_fault_panels(Path(fault_panel_csv))
    influenced_df = assign_fault_influence(
        df=regional_df,
        panel_df=panel_df,
        fault_half_band_ms=float(fault_half_band_ms),
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
        panel_xy_buffer=float(panel_xy_buffer),
    )
    kept_df, filter_stats = apply_fault_filter_and_shrink(
        influenced_df=influenced_df,
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
    )
    parallel_df, perpendicular_df = build_generated_patch_rows(
        panel_df=panel_df,
        parallel_ratio=float(parallel_ratio),
        random_seed=int(random_seed),
        fault_half_band_ms=float(fault_half_band_ms),
    )
    final_df = finalize_output_dataframe(kept_df, parallel_df, perpendicular_df)

    output_csv = run_dir / "regional_dfn_fault_embedded_fractures.csv"
    fractures_vtk = run_dir / "regional_dfn_fault_embedded_fractures_raw.vtk"
    output_vtk = run_dir / "regional_dfn_fault_embedded_raw.vtk"
    write_csv_utf8(final_df, output_csv)
    write_df_to_regional_vtk(final_df, f"{title}_fault_embedded_fractures", fractures_vtk, scalar_types)
    vtk_merge_stats = combine_with_fault_surface_vtk(
        fracture_vtk=fractures_vtk,
        fault_surface_vtk=Path(fault_surface_vtk) if fault_surface_vtk else None,
        output_vtk=output_vtk,
        title=f"{title}_fault_embedded",
    )

    parallel_csv = run_dir / "fault_parallel_patches.csv"
    perpendicular_csv = run_dir / "fault_perpendicular_patches.csv"
    parallel_vtk = run_dir / "fault_parallel_patches_raw.vtk"
    perpendicular_vtk = run_dir / "fault_perpendicular_patches_raw.vtk"
    if not parallel_df.empty:
        write_csv_utf8(parallel_df, parallel_csv)
        write_df_to_regional_vtk(parallel_df, f"{title}_fault_parallel", parallel_vtk, scalar_types)
    if not perpendicular_df.empty:
        write_csv_utf8(perpendicular_df, perpendicular_csv)
        write_df_to_regional_vtk(perpendicular_df, f"{title}_fault_perpendicular", perpendicular_vtk, scalar_types)

    origin_counts = final_df["PatchOriginText"].value_counts(dropna=False).to_dict() if "PatchOriginText" in final_df.columns else {}
    action_counts = final_df["FaultActionText"].value_counts(dropna=False).to_dict() if "FaultActionText" in final_df.columns else {}
    if int(vtk_merge_stats["surface_polygon_count"]) > 0:
        origin_counts["fault_surface"] = int(vtk_merge_stats["surface_polygon_count"])
        action_counts["surface"] = int(vtk_merge_stats["surface_polygon_count"])
    summary = {
        "input_vtk": str(input_vtk),
        "fault_panel_csv": str(fault_panel_csv),
        "fault_surface_vtk": str(fault_surface_vtk) if fault_surface_vtk else "",
        "fault_half_band_ms": float(fault_half_band_ms),
        "fault_remove_ms": float(fault_remove_ms),
        "fault_transition_ms": float(fault_transition_ms),
        "panel_xy_buffer": float(panel_xy_buffer),
        "parallel_ratio": float(parallel_ratio),
        "input_patch_count": int(len(regional_df)),
        "fault_panel_count": int(len(panel_df)),
        "fault_parallel_patch_count": int(len(parallel_df)),
        "fault_perpendicular_patch_count": int(len(perpendicular_df)),
        "fracture_patch_count": int(len(final_df)),
        "fault_surface_polygon_count": int(vtk_merge_stats["surface_polygon_count"]),
        "final_polygon_count": int(vtk_merge_stats["final_polygon_count"]),
        "output_csv": str(output_csv),
        "fractures_vtk": str(fractures_vtk),
        "output_vtk": str(output_vtk),
        "fault_parallel_csv": str(parallel_csv) if not parallel_df.empty else "",
        "fault_parallel_vtk": str(parallel_vtk) if not parallel_df.empty else "",
        "fault_perpendicular_csv": str(perpendicular_csv) if not perpendicular_df.empty else "",
        "fault_perpendicular_vtk": str(perpendicular_vtk) if not perpendicular_df.empty else "",
        "filter_stats": filter_stats,
        "origin_counts": origin_counts,
        "action_counts": action_counts,
    }
    summary_path = run_dir / "regional_dfn_fault_embedded_summary.json"
    write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / str(args.run_name)
    if run_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output run dir already exists: {run_dir}")
    summary = run_fault_postfusion(
        input_vtk=Path(args.input_vtk),
        fault_panel_csv=Path(args.fault_panel_csv),
        fault_surface_vtk=Path(args.fault_surface_vtk) if args.fault_surface_vtk else None,
        output_root=Path(args.output_root),
        run_name=str(args.run_name),
        fault_half_band_ms=float(args.fault_half_band_ms),
        fault_remove_ms=float(args.fault_remove_ms),
        fault_transition_ms=float(args.fault_transition_ms),
        panel_xy_buffer=float(args.panel_xy_buffer),
        parallel_ratio=float(args.parallel_ratio),
        random_seed=int(args.random_seed),
    )
    append_lines_to_docx(
        docx_path=args.docx_path,
        title=f"regional fault postfusion {args.run_name}",
        lines=[
            f"input_vtk: {args.input_vtk}",
            f"fault_panel_csv: {args.fault_panel_csv}",
            f"fault_surface_vtk: {args.fault_surface_vtk}" if args.fault_surface_vtk else "fault_surface_vtk: ",
            f"input_patch_count: {summary['input_patch_count']}",
            f"fault_panel_count: {summary['fault_panel_count']}",
            f"fracture_patch_count: {summary['fracture_patch_count']}",
            f"fault_surface_polygon_count: {summary['fault_surface_polygon_count']}",
            f"final_polygon_count: {summary['final_polygon_count']}",
            f"origin_counts: {summary['origin_counts']}",
            f"action_counts: {summary['action_counts']}",
            f"output_vtk: {summary['output_vtk']}",
        ],
    )
    print(f"output_vtk: {summary['output_vtk']}")
    print(f"final_polygon_count: {summary['final_polygon_count']}")
    print(f"origin_counts: {summary['origin_counts']}")


if __name__ == "__main__":
    main()
