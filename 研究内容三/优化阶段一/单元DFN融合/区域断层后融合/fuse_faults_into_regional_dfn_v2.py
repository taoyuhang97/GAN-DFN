# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from merge_unit_dfn_vtks import merge_vtk_payloads, read_legacy_vtk_polygons
from fault_postfusion_common import (
    DEFAULT_DOCX_PATH,
    FAULT_ACTION_TEXT_TO_CODE,
    PATCH_ORIGIN_TEXT_TO_CODE,
    append_lines_to_docx,
    azimuth_diff_deg,
    build_plane_axes_from_normal,
    make_patch_row_from_axes,
    normalize_vector,
    predict_fault_time_at_xy,
    read_regional_vtk_to_df,
    strike_dip_from_normal,
    write_legacy_vtk_polygons_preserve_patch_area,
    write_csv_utf8,
    write_df_to_regional_vtk,
    write_json,
)


SIZE_LABELS = ["small", "medium", "large"]
PATCH_GEOMETRY_COLUMNS = [
    "CenterX",
    "CenterY",
    "CenterTIME",
    "PatchLength",
    "PatchHeight",
    "PatchArea",
    "Azimuth",
    "Dip",
    "NormalX",
    "NormalY",
    "NormalZ",
    "BBoxXMin",
    "BBoxXMax",
    "BBoxYMin",
    "BBoxYMax",
    "BBoxZMin",
    "BBoxZMax",
] + [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]


def emit_fault_postfusion_progress(stage: str, detail: str | None = None) -> None:
    if detail:
        print(f"[fault-fuse] {stage} | {detail}", flush=True)
    else:
        print(f"[fault-fuse] {stage}", flush=True)


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
    parser.add_argument("--fault-induced-count-scale", type=float, default=5.5)
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


def build_zone_specs(
    fault_remove_ms: float,
    fault_transition_ms: float,
) -> list[dict[str, Any]]:
    remove_ms = max(float(fault_remove_ms), 1.0)
    transition_ms = max(float(fault_transition_ms), remove_ms + 1.0)
    transition_mid = remove_ms + 0.5 * (transition_ms - remove_ms)
    return [
        {
            "name": "core",
            "d_min": 0.0,
            "d_max": float(remove_ms),
            "count_weight": 1.50,
            "size_probs": [0.64, 0.27, 0.09],
            "confidence": 0.92,
            "size_scale": 1.55,
            "transition_blend": 0,
        },
        {
            "name": "transition_inner",
            "d_min": float(remove_ms),
            "d_max": float(transition_mid),
            "count_weight": 0.95,
            "size_probs": [0.78, 0.18, 0.04],
            "confidence": 0.82,
            "size_scale": 1.12,
            "transition_blend": 1,
        },
        {
            "name": "transition_outer",
            "d_min": float(transition_mid),
            "d_max": float(transition_ms),
            "count_weight": 0.60,
            "size_probs": [0.88, 0.10, 0.02],
            "confidence": 0.72,
            "size_scale": 0.86,
            "transition_blend": 1,
        },
    ]


def compute_transition_fault_weight(
    distance_ms: float,
    fault_remove_ms: float,
    fault_transition_ms: float,
) -> float:
    span = max(float(fault_transition_ms) - float(fault_remove_ms), 1e-6)
    ratio = (float(distance_ms) - float(fault_remove_ms)) / span
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def blend_axial_angles_deg(fault_angle_deg: float, ref_angle_deg: float, fault_weight: float) -> float:
    blend_weight = float(np.clip(fault_weight, 0.0, 1.0))
    ref_weight = 1.0 - blend_weight
    fault_rad = np.deg2rad(2.0 * (float(fault_angle_deg) % 180.0))
    ref_rad = np.deg2rad(2.0 * (float(ref_angle_deg) % 180.0))
    vec = (
        blend_weight * np.array([np.cos(fault_rad), np.sin(fault_rad)], dtype=float)
        + ref_weight * np.array([np.cos(ref_rad), np.sin(ref_rad)], dtype=float)
    )
    if float(np.linalg.norm(vec)) <= 1e-8:
        return float(fault_angle_deg if blend_weight >= ref_weight else ref_angle_deg) % 180.0
    return float(0.5 * np.degrees(np.arctan2(vec[1], vec[0]))) % 180.0


def normal_from_azimuth_dip_deg(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    azimuth_rad = np.deg2rad(float(azimuth_deg) % 180.0)
    dip_rad = np.deg2rad(float(np.clip(dip_deg, 0.0, 89.999)))
    horizontal_norm = float(np.cos(dip_rad))
    normal = np.array(
        [
            np.sin(azimuth_rad) * horizontal_norm,
            np.cos(azimuth_rad) * horizontal_norm,
            np.sin(dip_rad),
        ],
        dtype=float,
    )
    return normalize_vector(normal, fallback=np.array([0.0, 0.0, 1.0], dtype=float))


def build_axes_from_azimuth_dip_deg(azimuth_deg: float, dip_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plane_normal = normal_from_azimuth_dip_deg(azimuth_deg, dip_deg)
    return build_plane_axes_from_normal(plane_normal)


def orientation_from_axes(u_vec: np.ndarray, v_vec: np.ndarray) -> tuple[float, float]:
    plane_normal = normalize_vector(np.cross(np.asarray(u_vec, dtype=float), np.asarray(v_vec, dtype=float)))
    return strike_dip_from_normal(plane_normal)


def build_patch_geometry_from_orientation(
    center: np.ndarray,
    azimuth_deg: float,
    dip_deg: float,
    patch_length: float,
    patch_height: float,
) -> dict[str, Any]:
    _, strike_vec, dip_vec = build_axes_from_azimuth_dip_deg(azimuth_deg, dip_deg)
    return make_patch_row_from_axes(
        center=np.asarray(center, dtype=float),
        u_vec=strike_vec,
        v_vec=dip_vec,
        length=float(patch_length),
        height=float(patch_height),
    )


def build_transition_reference_lookup(
    influenced_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    fault_remove_ms: float,
    fault_transition_ms: float,
) -> dict[int, dict[str, np.ndarray]]:
    if influenced_df.empty or panel_df.empty:
        return {}
    if "NearestFaultPanelID" not in influenced_df.columns or "FaultDistanceMs" not in influenced_df.columns:
        return {}
    distance = pd.to_numeric(influenced_df["FaultDistanceMs"], errors="coerce")
    panel_id = pd.to_numeric(influenced_df["NearestFaultPanelID"], errors="coerce")
    valid_mask = (
        panel_id.notna()
        & (panel_id.astype(int) >= 0)
        & distance.notna()
        & (distance > float(fault_remove_ms))
        & (distance <= float(fault_transition_ms))
    )
    if not bool(valid_mask.any()):
        return {}
    cols = [
        "NearestFaultPanelID",
        "CenterX",
        "CenterY",
        "CenterTIME",
        "Azimuth",
        "Dip",
        "PatchLength",
        "PatchHeight",
        "PatchArea",
        "FaultDistanceMs",
    ]
    work = influenced_df.loc[valid_mask, cols].copy()
    if "PatchArea" not in work.columns:
        work["PatchArea"] = (
            pd.to_numeric(work.get("PatchLength"), errors="coerce").fillna(0.0)
            * pd.to_numeric(work.get("PatchHeight"), errors="coerce").fillna(0.0)
        )
    panel_lookup = {
        int(row["FaultPanelID"]): row
        for _, row in panel_df.iterrows()
    }
    ref_lookup: dict[int, dict[str, np.ndarray]] = {}
    for panel_key, group in work.groupby("NearestFaultPanelID", sort=False):
        panel_id_int = int(panel_key)
        panel_row = panel_lookup.get(panel_id_int)
        if panel_row is None:
            continue
        center_x = pd.to_numeric(group["CenterX"], errors="coerce").to_numpy(dtype=float)
        center_y = pd.to_numeric(group["CenterY"], errors="coerce").to_numpy(dtype=float)
        center_z = pd.to_numeric(group["CenterTIME"], errors="coerce").to_numpy(dtype=float)
        azimuth = pd.to_numeric(group["Azimuth"], errors="coerce").to_numpy(dtype=float)
        dip = pd.to_numeric(group["Dip"], errors="coerce").to_numpy(dtype=float)
        patch_length = pd.to_numeric(group["PatchLength"], errors="coerce").to_numpy(dtype=float)
        patch_height = pd.to_numeric(group["PatchHeight"], errors="coerce").to_numpy(dtype=float)
        patch_area = pd.to_numeric(group["PatchArea"], errors="coerce").to_numpy(dtype=float)
        distance_ms = pd.to_numeric(group["FaultDistanceMs"], errors="coerce").to_numpy(dtype=float)
        nx = float(panel_row.get("NormalX", 0.0))
        ny = float(panel_row.get("NormalY", 0.0))
        nz = float(panel_row.get("NormalZ", 1.0))
        cx = float(panel_row.get("CenterX", 0.0))
        cy = float(panel_row.get("CenterY", 0.0))
        cz = float(panel_row.get("CenterTIME", 0.0))
        if abs(nz) > 1e-8:
            fault_time = cz - (nx * (center_x - cx) + ny * (center_y - cy)) / nz
        else:
            fault_time = np.full(len(group), cz, dtype=float)
        side_sign = np.where(center_z >= fault_time, 1.0, -1.0)
        valid = (
            np.isfinite(center_x)
            & np.isfinite(center_y)
            & np.isfinite(center_z)
            & np.isfinite(azimuth)
            & np.isfinite(dip)
            & np.isfinite(patch_length)
            & np.isfinite(patch_height)
            & np.isfinite(patch_area)
            & np.isfinite(distance_ms)
        )
        if not np.any(valid):
            continue
        ref_lookup[panel_id_int] = {
            "CenterX": center_x[valid],
            "CenterY": center_y[valid],
            "CenterTIME": center_z[valid],
            "Azimuth": azimuth[valid],
            "Dip": dip[valid],
            "PatchLength": np.clip(patch_length[valid], 1e-6, None),
            "PatchHeight": np.clip(patch_height[valid], 1e-6, None),
            "PatchArea": np.clip(patch_area[valid], 1e-6, None),
            "FaultDistanceMs": distance_ms[valid],
            "SideSign": side_sign[valid],
        }
    return ref_lookup


def sample_transition_reference(
    ref_lookup: dict[int, dict[str, np.ndarray]],
    panel_id: int,
    side_sign: float,
    target_distance_ms: float,
    rng: np.random.Generator,
) -> dict[str, float] | None:
    refs = ref_lookup.get(int(panel_id))
    if not refs:
        return None
    base_mask = np.isfinite(refs["FaultDistanceMs"])
    if not np.any(base_mask):
        return None
    same_side_mask = base_mask & (refs["SideSign"] == (1.0 if float(side_sign) >= 0.0 else -1.0))
    use_mask = same_side_mask if np.any(same_side_mask) else base_mask
    candidate_positions = np.flatnonzero(use_mask)
    if len(candidate_positions) <= 0:
        return None
    distance_delta = np.abs(refs["FaultDistanceMs"][candidate_positions] - float(target_distance_ms))
    area_weights = np.sqrt(np.clip(refs["PatchArea"][candidate_positions], 1e-6, None))
    weights = area_weights / np.clip(distance_delta + 5.0, 1.0, None)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        chosen_pos = int(rng.choice(candidate_positions))
    else:
        probs = weights / weights.sum()
        chosen_pos = int(rng.choice(candidate_positions, p=probs))
    return {
        "CenterX": float(refs["CenterX"][chosen_pos]),
        "CenterY": float(refs["CenterY"][chosen_pos]),
        "CenterTIME": float(refs["CenterTIME"][chosen_pos]),
        "Azimuth": float(refs["Azimuth"][chosen_pos]),
        "Dip": float(refs["Dip"][chosen_pos]),
        "PatchLength": float(refs["PatchLength"][chosen_pos]),
        "PatchHeight": float(refs["PatchHeight"][chosen_pos]),
        "PatchArea": float(refs["PatchArea"][chosen_pos]),
        "FaultDistanceMs": float(refs["FaultDistanceMs"][chosen_pos]),
        "SourceCount": int(len(candidate_positions)),
    }


def build_panel_spatial_index(
    panel_arrays: dict[str, np.ndarray],
    grid_size_xy: float,
) -> dict[tuple[int, int], np.ndarray]:
    safe_grid_size = max(float(grid_size_xy), 1.0)
    grid_lookup: dict[tuple[int, int], list[int]] = {}
    grid_x_min = np.floor(panel_arrays["xmin"] / safe_grid_size).astype(int)
    grid_x_max = np.floor(panel_arrays["xmax"] / safe_grid_size).astype(int)
    grid_y_min = np.floor(panel_arrays["ymin"] / safe_grid_size).astype(int)
    grid_y_max = np.floor(panel_arrays["ymax"] / safe_grid_size).astype(int)
    for panel_pos in range(len(panel_arrays["id"])):
        for grid_x in range(int(grid_x_min[panel_pos]), int(grid_x_max[panel_pos]) + 1):
            for grid_y in range(int(grid_y_min[panel_pos]), int(grid_y_max[panel_pos]) + 1):
                grid_lookup.setdefault((grid_x, grid_y), []).append(int(panel_pos))
    return {
        key: np.asarray(value, dtype=np.int32)
        for key, value in grid_lookup.items()
    }


def iter_grouped_xy_row_positions(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid_size_xy: float,
) -> list[tuple[tuple[int, int], np.ndarray]]:
    if len(x_values) <= 0:
        return []
    safe_grid_size = max(float(grid_size_xy), 1.0)
    grid_x = np.floor(np.asarray(x_values, dtype=float) / safe_grid_size).astype(np.int32)
    grid_y = np.floor(np.asarray(y_values, dtype=float) / safe_grid_size).astype(np.int32)
    order = np.lexsort((grid_y, grid_x))
    sorted_grid_x = grid_x[order]
    sorted_grid_y = grid_y[order]
    change_positions = np.flatnonzero(
        (sorted_grid_x[1:] != sorted_grid_x[:-1]) | (sorted_grid_y[1:] != sorted_grid_y[:-1])
    ) + 1
    starts = np.concatenate([[0], change_positions])
    ends = np.concatenate([change_positions, [len(order)]])
    return [
        ((int(sorted_grid_x[start]), int(sorted_grid_y[start])), order[start:end])
        for start, end in zip(starts, ends)
    ]


def assign_fault_influence(
    df: pd.DataFrame,
    panel_df: pd.DataFrame,
    fault_half_band_ms: float,
    fault_remove_ms: float,
    fault_transition_ms: float,
    panel_xy_buffer: float,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    if "PatchOriginCode" in result.columns:
        result["PatchOriginCode"] = pd.to_numeric(result["PatchOriginCode"], errors="coerce").fillna(0).astype(int)
    else:
        result["PatchOriginCode"] = 0
    if "PatchOriginText" not in result.columns:
        result["PatchOriginText"] = "original"
    total_rows = int(len(result))

    fault_action_codes = np.zeros(total_rows, dtype=int)
    fault_action_texts = np.full(total_rows, "keep", dtype=object)
    fault_distance_ms = np.full(total_rows, np.nan, dtype=float)
    fault_influence_weight = np.zeros(total_rows, dtype=float)
    nearest_fault_panel_id = np.full(total_rows, -1, dtype=int)
    nearest_fault_name = np.full(total_rows, "", dtype=object)

    if panel_df.empty or result.empty:
        result["FaultActionCode"] = fault_action_codes
        result["FaultActionText"] = fault_action_texts
        result["FaultDistanceMs"] = fault_distance_ms
        result["FaultInfluenceWeight"] = fault_influence_weight
        result["NearestFaultPanelID"] = nearest_fault_panel_id
        result["NearestFaultName"] = nearest_fault_name
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
    grid_size_xy = max(float(panel_xy_buffer), 1.0)
    panel_spatial_index = build_panel_spatial_index(panel_arrays, grid_size_xy)

    center_x = result["CenterX"].to_numpy(dtype=float)
    center_y = result["CenterY"].to_numpy(dtype=float)
    center_z = result["CenterTIME"].to_numpy(dtype=float)
    grouped_positions = iter_grouped_xy_row_positions(center_x, center_y, grid_size_xy)
    affected_count = 0
    processed_rows = 0
    emit_step = max(1, total_rows // 20) if total_rows > 0 else 1
    next_emit_row = 1

    def maybe_emit_progress() -> None:
        nonlocal next_emit_row
        if progress_hook is None:
            return
        if processed_rows < next_emit_row and processed_rows != total_rows:
            return
        progress_hook("断层影响赋值进度", f"{processed_rows}/{total_rows}, affected={affected_count}")
        while next_emit_row <= processed_rows:
            next_emit_row += emit_step

    for grid_key, row_positions in grouped_positions:
        processed_rows += int(len(row_positions))
        candidate_positions = panel_spatial_index.get(grid_key)
        if candidate_positions is None or len(candidate_positions) <= 0:
            maybe_emit_progress()
            continue

        group_x = center_x[row_positions][:, None]
        group_y = center_y[row_positions][:, None]
        group_z = center_z[row_positions][:, None]
        candidate_xmin = panel_arrays["xmin"][candidate_positions][None, :]
        candidate_xmax = panel_arrays["xmax"][candidate_positions][None, :]
        candidate_ymin = panel_arrays["ymin"][candidate_positions][None, :]
        candidate_ymax = panel_arrays["ymax"][candidate_positions][None, :]

        inside_mask = (
            (candidate_xmin <= group_x)
            & (group_x <= candidate_xmax)
            & (candidate_ymin <= group_y)
            & (group_y <= candidate_ymax)
        )
        if not np.any(inside_mask):
            maybe_emit_progress()
            continue

        candidate_cz = panel_arrays["cz"][candidate_positions]
        fault_z = np.broadcast_to(candidate_cz[None, :], inside_mask.shape).astype(float, copy=True)
        candidate_valid_nz = valid_nz[candidate_positions]
        if np.any(candidate_valid_nz):
            candidate_cx = panel_arrays["cx"][candidate_positions][candidate_valid_nz][None, :]
            candidate_cy = panel_arrays["cy"][candidate_positions][candidate_valid_nz][None, :]
            candidate_nx = panel_arrays["nx"][candidate_positions][candidate_valid_nz][None, :]
            candidate_ny = panel_arrays["ny"][candidate_positions][candidate_valid_nz][None, :]
            candidate_nz = panel_arrays["nz"][candidate_positions][candidate_valid_nz][None, :]
            fault_z[:, candidate_valid_nz] = candidate_cz[candidate_valid_nz][None, :] - (
                candidate_nx * (group_x - candidate_cx) + candidate_ny * (group_y - candidate_cy)
            ) / candidate_nz

        distances = np.abs(group_z - fault_z)
        distances[~inside_mask] = np.inf
        min_pos = np.argmin(distances, axis=1)
        row_pos = np.arange(len(row_positions))
        min_dist = distances[row_pos, min_pos]
        valid_rows = np.isfinite(min_dist)
        if np.any(valid_rows):
            valid_row_positions = row_positions[valid_rows]
            chosen_candidate_positions = candidate_positions[min_pos[valid_rows]]
            valid_dist = min_dist[valid_rows]
            affected_count += int(valid_rows.sum())
            nearest_fault_panel_id[valid_row_positions] = panel_arrays["id"][chosen_candidate_positions]
            nearest_fault_name[valid_row_positions] = panel_arrays["name"][chosen_candidate_positions]
            fault_distance_ms[valid_row_positions] = valid_dist
            fault_influence_weight[valid_row_positions] = np.maximum(
                0.0,
                1.0 - valid_dist / max(float(fault_half_band_ms), 1e-6),
            )
            remove_mask = valid_dist <= float(fault_remove_ms)
            shrink_mask = (valid_dist > float(fault_remove_ms)) & (valid_dist <= float(fault_transition_ms))
            if np.any(remove_mask):
                remove_positions = valid_row_positions[remove_mask]
                fault_action_texts[remove_positions] = "remove"
                fault_action_codes[remove_positions] = FAULT_ACTION_TEXT_TO_CODE["remove"]
            if np.any(shrink_mask):
                shrink_positions = valid_row_positions[shrink_mask]
                fault_action_texts[shrink_positions] = "shrink"
                fault_action_codes[shrink_positions] = FAULT_ACTION_TEXT_TO_CODE["shrink"]
        maybe_emit_progress()

    result["FaultActionCode"] = fault_action_codes
    result["FaultActionText"] = fault_action_texts
    result["FaultDistanceMs"] = fault_distance_ms
    result["FaultInfluenceWeight"] = fault_influence_weight
    result["NearestFaultPanelID"] = nearest_fault_panel_id
    result["NearestFaultName"] = nearest_fault_name
    return result


def apply_fault_filter_and_shrink(
    influenced_df: pd.DataFrame,
    fault_remove_ms: float,
    fault_transition_ms: float,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    total_rows = int(len(influenced_df))
    if "FaultActionText" in influenced_df.columns:
        action_text = influenced_df["FaultActionText"].astype(str).to_numpy()
    else:
        action_text = np.full(total_rows, "keep", dtype=object)
    remove_mask = action_text == "remove"
    shrink_mask = action_text == "shrink"
    removed_count = int(remove_mask.sum())
    shrunk_count = int(shrink_mask.sum())

    if progress_hook is not None:
        progress_hook(
            "断层控制区过滤进度",
            f"0/{total_rows}, kept=0, removed=0, shrunk=0",
        )

    kept_df = influenced_df.loc[~remove_mask].copy().reset_index(drop=True)
    kept_df["FaultShrinkAreaRatio"] = 1.0
    if not kept_df.empty and "FaultActionText" in kept_df.columns and np.any(kept_df["FaultActionText"].astype(str).to_numpy() == "shrink"):
        kept_shrink_mask = kept_df["FaultActionText"].astype(str).to_numpy() == "shrink"
        shrink_idx = np.flatnonzero(kept_shrink_mask)
        denom = max(float(fault_transition_ms) - float(fault_remove_ms), 1e-6)
        distance_ms = kept_df.loc[shrink_idx, "FaultDistanceMs"].to_numpy(dtype=float)
        area_ratio = np.clip((distance_ms - float(fault_remove_ms)) / denom, 0.0, 1.0)
        scale = np.sqrt(area_ratio)
        kept_df.loc[shrink_idx, "FaultShrinkAreaRatio"] = area_ratio
        kept_df.loc[shrink_idx, "PatchLength"] = kept_df.loc[shrink_idx, "PatchLength"].to_numpy(dtype=float) * scale
        kept_df.loc[shrink_idx, "PatchHeight"] = kept_df.loc[shrink_idx, "PatchHeight"].to_numpy(dtype=float) * scale
        if "PatchArea" in kept_df.columns:
            kept_df.loc[shrink_idx, "PatchArea"] = kept_df.loc[shrink_idx, "PatchArea"].to_numpy(dtype=float) * area_ratio
        if "PatchArea3D" in kept_df.columns:
            kept_df.loc[shrink_idx, "PatchArea3D"] = kept_df.loc[shrink_idx, "PatchArea3D"].to_numpy(dtype=float) * area_ratio

        for vertex_idx in range(1, 5):
            center_x = kept_df.loc[shrink_idx, "CenterX"].to_numpy(dtype=float)
            center_y = kept_df.loc[shrink_idx, "CenterY"].to_numpy(dtype=float)
            center_z = kept_df.loc[shrink_idx, "CenterTIME"].to_numpy(dtype=float)
            for axis, center_values in zip(
                ("X", "Y", "Z"),
                (center_x, center_y, center_z),
            ):
                col = f"V{vertex_idx}{axis}"
                vertex_values = kept_df.loc[shrink_idx, col].to_numpy(dtype=float)
                kept_df.loc[shrink_idx, col] = center_values + scale * (vertex_values - center_values)

        vertex_x = np.column_stack([
            kept_df.loc[shrink_idx, f"V{vertex_idx}X"].to_numpy(dtype=float)
            for vertex_idx in range(1, 5)
        ])
        vertex_y = np.column_stack([
            kept_df.loc[shrink_idx, f"V{vertex_idx}Y"].to_numpy(dtype=float)
            for vertex_idx in range(1, 5)
        ])
        vertex_z = np.column_stack([
            kept_df.loc[shrink_idx, f"V{vertex_idx}Z"].to_numpy(dtype=float)
            for vertex_idx in range(1, 5)
        ])
        kept_df.loc[shrink_idx, "BBoxXMin"] = vertex_x.min(axis=1)
        kept_df.loc[shrink_idx, "BBoxXMax"] = vertex_x.max(axis=1)
        kept_df.loc[shrink_idx, "BBoxYMin"] = vertex_y.min(axis=1)
        kept_df.loc[shrink_idx, "BBoxYMax"] = vertex_y.max(axis=1)
        kept_df.loc[shrink_idx, "BBoxZMin"] = vertex_z.min(axis=1)
        kept_df.loc[shrink_idx, "BBoxZMax"] = vertex_z.max(axis=1)

    if progress_hook is not None:
        progress_hook(
            "断层控制区过滤进度",
            f"{total_rows}/{total_rows}, kept={len(kept_df)}, removed={removed_count}, shrunk={shrunk_count}",
        )

    stats = {
        "input_patch_count": int(len(influenced_df)),
        "kept_patch_count": int(len(kept_df)),
        "removed_patch_count": int(removed_count),
        "shrunk_patch_count": int(shrunk_count),
    }
    return kept_df, stats


def apply_fault_transition_blend_to_existing_patches(
    kept_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    fault_remove_ms: float,
    fault_transition_ms: float,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if kept_df.empty or panel_df.empty or "FaultActionText" not in kept_df.columns:
        return kept_df, {"transition_blended_original_count": 0}
    transition_mask = kept_df["FaultActionText"].astype(str).to_numpy() == "shrink"
    transition_positions = np.flatnonzero(transition_mask)
    if len(transition_positions) <= 0:
        kept_df["FaultTransitionBlendWeight"] = 0.0
        kept_df["FaultTransitionBlendApplied"] = 0
        return kept_df, {"transition_blended_original_count": 0}

    panel_lookup = {int(row["FaultPanelID"]): row for _, row in panel_df.iterrows()}
    kept_df["FaultTransitionBlendWeight"] = 0.0
    kept_df["FaultTransitionBlendApplied"] = 0

    updated_cols: dict[str, list[Any]] = {col: [] for col in PATCH_GEOMETRY_COLUMNS}
    updated_row_positions: list[int] = []
    updated_blend_weights: list[float] = []
    total_count = int(len(transition_positions))
    emit_step = max(1, total_count // 10)

    if progress_hook is not None:
        progress_hook("断层过渡带原裂缝混合进度", f"0/{total_count}, blended=0")

    for seq_idx, row_pos in enumerate(transition_positions, start=1):
        row = kept_df.iloc[int(row_pos)]
        panel_id = int(pd.to_numeric(row.get("NearestFaultPanelID"), errors="coerce") if pd.notna(row.get("NearestFaultPanelID")) else -1)
        panel_row = panel_lookup.get(panel_id)
        if panel_row is None:
            if progress_hook is not None and (seq_idx == total_count or seq_idx % emit_step == 0):
                progress_hook("断层过渡带原裂缝混合进度", f"{seq_idx}/{total_count}, blended={len(updated_row_positions)}")
            continue
        distance_ms = float(pd.to_numeric(row.get("FaultDistanceMs"), errors="coerce"))
        if not np.isfinite(distance_ms):
            if progress_hook is not None and (seq_idx == total_count or seq_idx % emit_step == 0):
                progress_hook("断层过渡带原裂缝混合进度", f"{seq_idx}/{total_count}, blended={len(updated_row_positions)}")
            continue
        fault_weight = compute_transition_fault_weight(distance_ms, fault_remove_ms, fault_transition_ms)
        if fault_weight <= 0.0:
            if progress_hook is not None and (seq_idx == total_count or seq_idx % emit_step == 0):
                progress_hook("断层过渡带原裂缝混合进度", f"{seq_idx}/{total_count}, blended={len(updated_row_positions)}")
            continue

        strike_vec = np.array(
            [
                float(panel_row.get("StrikeVecX", 1.0)),
                float(panel_row.get("StrikeVecY", 0.0)),
                float(panel_row.get("StrikeVecZ", 0.0)),
            ],
            dtype=float,
        )
        dip_vec = np.array(
            [
                float(panel_row.get("DipVecX", 0.0)),
                float(panel_row.get("DipVecY", 0.0)),
                float(panel_row.get("DipVecZ", 1.0)),
            ],
            dtype=float,
        )
        horizontal_normal = build_fault_horizontal_normal(panel_row, strike_vec)
        parallel_azimuth, parallel_dip = orientation_from_axes(strike_vec, dip_vec)
        perpendicular_u_vec, perpendicular_v_vec = build_vertical_perpendicular_axes(panel_row, strike_vec)
        perpendicular_azimuth, perpendicular_dip = orientation_from_axes(perpendicular_u_vec, perpendicular_v_vec)

        current_azimuth = float(pd.to_numeric(row.get("Azimuth"), errors="coerce"))
        current_dip = float(pd.to_numeric(row.get("Dip"), errors="coerce"))
        parallel_score = azimuth_diff_deg(current_azimuth, parallel_azimuth) + 0.35 * abs(current_dip - parallel_dip)
        perpendicular_score = azimuth_diff_deg(current_azimuth, perpendicular_azimuth) + 0.35 * abs(current_dip - perpendicular_dip)
        if parallel_score <= perpendicular_score:
            target_azimuth = parallel_azimuth
            target_dip = parallel_dip
        else:
            target_azimuth = perpendicular_azimuth
            target_dip = perpendicular_dip

        orientation_weight = float(np.clip(0.25 + 0.55 * fault_weight, 0.0, 0.85))
        blended_azimuth = blend_axial_angles_deg(target_azimuth, current_azimuth, orientation_weight)
        blended_dip = float(np.clip(
            orientation_weight * target_dip + (1.0 - orientation_weight) * current_dip,
            0.0,
            89.999,
        ))

        original_center = np.array(
            [
                float(pd.to_numeric(row.get("CenterX"), errors="coerce")),
                float(pd.to_numeric(row.get("CenterY"), errors="coerce")),
                float(pd.to_numeric(row.get("CenterTIME"), errors="coerce")),
            ],
            dtype=float,
        )
        fault_time = predict_fault_time_at_xy(panel_row, float(original_center[0]), float(original_center[1]))
        side_sign = -1.0 if float(original_center[2]) < float(fault_time) else 1.0
        panel_length = max(float(panel_row.get("PanelLength", 10.0)), 10.0)
        target_distance_ms = float(fault_remove_ms + 0.35 * max(distance_ms - float(fault_remove_ms), 0.0))
        target_center = build_fault_offset_center(
            panel_row=panel_row,
            base_point=original_center,
            fault_time=float(fault_time),
            distance_ms=target_distance_ms,
            fault_half_band_ms=float(fault_transition_ms),
            side_sign=side_sign,
            horizontal_normal=horizontal_normal,
            panel_length=panel_length,
        )
        position_weight = float(np.clip(0.12 + 0.38 * fault_weight, 0.0, 0.50))
        blended_center = (1.0 - position_weight) * original_center + position_weight * target_center
        patch_length = max(float(pd.to_numeric(row.get("PatchLength"), errors="coerce")), 1e-6)
        patch_height = max(float(pd.to_numeric(row.get("PatchHeight"), errors="coerce")), 1e-6)
        geometry = build_patch_geometry_from_orientation(
            center=blended_center,
            azimuth_deg=blended_azimuth,
            dip_deg=blended_dip,
            patch_length=patch_length,
            patch_height=patch_height,
        )
        updated_row_positions.append(int(row_pos))
        updated_blend_weights.append(float(fault_weight))
        for col in PATCH_GEOMETRY_COLUMNS:
            updated_cols[col].append(geometry[col])
        if progress_hook is not None and (seq_idx == total_count or seq_idx % emit_step == 0):
            progress_hook("断层过渡带原裂缝混合进度", f"{seq_idx}/{total_count}, blended={len(updated_row_positions)}")

    if updated_row_positions:
        for col, values in updated_cols.items():
            kept_df.loc[updated_row_positions, col] = values
        kept_df.loc[updated_row_positions, "FaultTransitionBlendWeight"] = updated_blend_weights
        kept_df.loc[updated_row_positions, "FaultTransitionBlendApplied"] = 1
        if "PatchArea3D" in kept_df.columns:
            kept_df.loc[updated_row_positions, "PatchArea3D"] = kept_df.loc[updated_row_positions, "PatchArea"].to_numpy(dtype=float)
    stats = {
        "transition_blended_original_count": int(len(updated_row_positions)),
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
    fault_remove_ms: float,
    fault_transition_ms: float,
    fault_induced_count_scale: float,
    transition_ref_lookup: dict[int, dict[str, np.ndarray]] | None = None,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(random_seed))
    parallel_rows: list[dict[str, Any]] = []
    perpendicular_rows: list[dict[str, Any]] = []
    parallel_ratio = float(np.clip(parallel_ratio, 0.0, 1.0))
    induced_count_scale = max(float(fault_induced_count_scale), 0.0)
    transition_ref_lookup = transition_ref_lookup or {}
    zone_specs = build_zone_specs(
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
    )

    total_panel_count = int(len(panel_df))
    emit_step = max(1, total_panel_count // 10) if total_panel_count > 0 else 1
    for panel_idx, (_, panel_row) in enumerate(panel_df.iterrows(), start=1):
        panel_id = int(panel_row["FaultPanelID"])
        fault_name = str(panel_row["FaultName"])
        center = np.array([float(panel_row["CenterX"]), float(panel_row["CenterY"]), float(panel_row["CenterTIME"])], dtype=float)
        strike_vec = np.array([float(panel_row["StrikeVecX"]), float(panel_row["StrikeVecY"]), float(panel_row["StrikeVecZ"])], dtype=float)
        dip_vec = np.array([float(panel_row["DipVecX"]), float(panel_row["DipVecY"]), float(panel_row["DipVecZ"])], dtype=float)
        panel_length = max(float(panel_row["PanelLength"]), 10.0)
        panel_height = max(float(panel_row["PanelHeight"]), 6.0)
        source_patch_count = max(int(panel_row.get("SourcePatchCount", 1)), 1)
        source_unit_count = max(int(panel_row.get("SourceUnitCount", 1)), 1)
        horizontal_normal = build_fault_horizontal_normal(panel_row, strike_vec)
        perpendicular_u_vec, perpendicular_v_vec = build_vertical_perpendicular_axes(panel_row, strike_vec)
        parallel_azimuth, parallel_dip = orientation_from_axes(strike_vec, dip_vec)
        perpendicular_azimuth, perpendicular_dip = orientation_from_axes(perpendicular_u_vec, perpendicular_v_vec)
        accepted_parallel_centers: list[np.ndarray] = []
        accepted_perpendicular_centers: list[np.ndarray] = []

        for zone_spec in zone_specs:
            total_count = max(1, int(round(source_patch_count * float(zone_spec["count_weight"]) * induced_count_scale)))
            parallel_count = max(1, int(round(total_count * parallel_ratio)))
            perpendicular_count = max(0, total_count - parallel_count)
            if perpendicular_count == 0:
                perpendicular_count = 1

            for _ in range(parallel_count):
                template_length, template_height, size_label = sample_patch_dimensions(panel_row, zone_spec, rng)
                min_xy_distance = max(12.0, 0.45 * template_length)
                min_time_distance = max(1.0, 0.35 * template_height)
                patch_center = None
                zone_center_distance = 0.0
                zone_weight = 0.0
                time_sign = 1.0
                for _attempt in range(16):
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, template_length, template_height, rng)
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
                        break
                if patch_center is None:
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, template_length, template_height, rng)
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
                patch_length = float(template_length)
                patch_height = float(template_height)
                patch_u_vec = strike_vec
                patch_v_vec = dip_vec
                patch_azimuth = parallel_azimuth
                patch_dip = parallel_dip
                transition_blend_weight = 0.0
                transition_blend_applied = 0
                transition_blend_source_count = 0
                confidence = float(zone_spec["confidence"])
                if int(zone_spec.get("transition_blend", 0)) == 1:
                    fault_weight = compute_transition_fault_weight(
                        zone_center_distance,
                        fault_remove_ms=fault_remove_ms,
                        fault_transition_ms=fault_transition_ms,
                    )
                    ref = sample_transition_reference(
                        ref_lookup=transition_ref_lookup,
                        panel_id=panel_id,
                        side_sign=time_sign,
                        target_distance_ms=zone_center_distance,
                        rng=rng,
                    )
                    if ref is not None:
                        patch_length = max(1e-6, float(fault_weight * template_length + (1.0 - fault_weight) * ref["PatchLength"]))
                        patch_height = max(1e-6, float(fault_weight * template_height + (1.0 - fault_weight) * ref["PatchHeight"]))
                        patch_center = (
                            fault_weight * patch_center
                            + (1.0 - fault_weight) * np.array(
                                [ref["CenterX"], ref["CenterY"], ref["CenterTIME"]],
                                dtype=float,
                            )
                        )
                        patch_azimuth = blend_axial_angles_deg(parallel_azimuth, ref["Azimuth"], fault_weight)
                        patch_dip = float(np.clip(
                            fault_weight * parallel_dip + (1.0 - fault_weight) * ref["Dip"],
                            0.0,
                            89.999,
                        ))
                        _, patch_u_vec, patch_v_vec = build_axes_from_azimuth_dip_deg(patch_azimuth, patch_dip)
                        confidence = float(fault_weight * float(zone_spec["confidence"]) + (1.0 - fault_weight) * 0.68)
                        transition_blend_weight = float(fault_weight)
                        transition_blend_applied = 1
                        transition_blend_source_count = int(ref["SourceCount"])
                accepted_parallel_centers.append(np.asarray(patch_center, dtype=float))
                parallel_rows.append(
                    make_patch_row_from_axes(
                        center=patch_center,
                        u_vec=patch_u_vec,
                        v_vec=patch_v_vec,
                        length=patch_length,
                        height=patch_height,
                        extra={
                            "Confidence": float(confidence),
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
                            "FaultTransitionBlendWeight": float(transition_blend_weight),
                            "FaultTransitionBlendApplied": int(transition_blend_applied),
                            "FaultTransitionBlendSourceCount": int(transition_blend_source_count),
                        },
                    )
                )
            for _ in range(perpendicular_count):
                template_length, template_height, size_label = sample_patch_dimensions(panel_row, zone_spec, rng)
                template_height = max(template_height, zone_spec["size_scale"] * 10.0)
                min_xy_distance = max(12.0, 0.40 * template_length)
                min_time_distance = max(1.0, 0.30 * template_height)
                patch_center = None
                zone_center_distance = 0.0
                zone_weight = 0.0
                time_sign = 1.0
                for _attempt in range(16):
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, template_length, template_height, rng)
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
                        break
                if patch_center is None:
                    strike_offset, dip_offset = sample_panel_offsets(panel_length, panel_height, template_length, template_height, rng)
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
                patch_length = float(template_length)
                patch_height = float(template_height)
                patch_u_vec = perpendicular_u_vec
                patch_v_vec = perpendicular_v_vec
                patch_azimuth = perpendicular_azimuth
                patch_dip = perpendicular_dip
                transition_blend_weight = 0.0
                transition_blend_applied = 0
                transition_blend_source_count = 0
                confidence = float(zone_spec["confidence"]) * 0.95
                if int(zone_spec.get("transition_blend", 0)) == 1:
                    fault_weight = compute_transition_fault_weight(
                        zone_center_distance,
                        fault_remove_ms=fault_remove_ms,
                        fault_transition_ms=fault_transition_ms,
                    )
                    ref = sample_transition_reference(
                        ref_lookup=transition_ref_lookup,
                        panel_id=panel_id,
                        side_sign=time_sign,
                        target_distance_ms=zone_center_distance,
                        rng=rng,
                    )
                    if ref is not None:
                        patch_length = max(1e-6, float(fault_weight * template_length + (1.0 - fault_weight) * ref["PatchLength"]))
                        patch_height = max(1e-6, float(fault_weight * template_height + (1.0 - fault_weight) * ref["PatchHeight"]))
                        patch_center = (
                            fault_weight * patch_center
                            + (1.0 - fault_weight) * np.array(
                                [ref["CenterX"], ref["CenterY"], ref["CenterTIME"]],
                                dtype=float,
                            )
                        )
                        patch_azimuth = blend_axial_angles_deg(perpendicular_azimuth, ref["Azimuth"], fault_weight)
                        patch_dip = float(np.clip(
                            fault_weight * perpendicular_dip + (1.0 - fault_weight) * ref["Dip"],
                            0.0,
                            89.999,
                        ))
                        _, patch_u_vec, patch_v_vec = build_axes_from_azimuth_dip_deg(patch_azimuth, patch_dip)
                        confidence = float(fault_weight * (float(zone_spec["confidence"]) * 0.95) + (1.0 - fault_weight) * 0.68)
                        transition_blend_weight = float(fault_weight)
                        transition_blend_applied = 1
                        transition_blend_source_count = int(ref["SourceCount"])
                accepted_perpendicular_centers.append(np.asarray(patch_center, dtype=float))
                perpendicular_rows.append(
                    make_patch_row_from_axes(
                        center=patch_center,
                        u_vec=patch_u_vec,
                        v_vec=patch_v_vec,
                        length=patch_length,
                        height=patch_height,
                        extra={
                            "Confidence": float(confidence),
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
                            "FaultTransitionBlendWeight": float(transition_blend_weight),
                            "FaultTransitionBlendApplied": int(transition_blend_applied),
                            "FaultTransitionBlendSourceCount": int(transition_blend_source_count),
                        },
                    )
                )
        if progress_hook is not None and (
            panel_idx == 1 or panel_idx == total_panel_count or panel_idx % emit_step == 0
        ):
            progress_hook(
                "断层诱导裂缝生成进度",
                (
                    f"{panel_idx}/{total_panel_count}, fault_panel_id={panel_id}, "
                    f"parallel_count={len(parallel_rows)}, perpendicular_count={len(perpendicular_rows)}"
                ),
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
        "FaultTransitionBlendWeight": 0.0,
        "FaultTransitionBlendApplied": 0,
        "FaultTransitionBlendSourceCount": 0,
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
    fault_induced_count_scale: float,
    panel_xy_buffer: float,
    parallel_ratio: float,
    random_seed: int,
) -> dict[str, Any]:
    run_dir = Path(output_root) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    emit_fault_postfusion_progress("开始断层后融合", f"input_vtk={input_vtk}")
    regional_df, scalar_types, title = read_regional_vtk_to_df(Path(input_vtk))
    emit_fault_postfusion_progress("区域 DFN 读取完成", f"input_patch_count={len(regional_df)}")
    panel_df = load_fault_panels(Path(fault_panel_csv))
    emit_fault_postfusion_progress("断层 panel 读取完成", f"fault_panel_count={len(panel_df)}")
    influenced_df = assign_fault_influence(
        df=regional_df,
        panel_df=panel_df,
        fault_half_band_ms=float(fault_half_band_ms),
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
        panel_xy_buffer=float(panel_xy_buffer),
        progress_hook=emit_fault_postfusion_progress,
    )
    emit_fault_postfusion_progress("断层影响赋值完成", f"influenced_patch_count={len(influenced_df)}")
    kept_df, filter_stats = apply_fault_filter_and_shrink(
        influenced_df=influenced_df,
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
        progress_hook=emit_fault_postfusion_progress,
    )
    emit_fault_postfusion_progress(
        "断层控制区过滤完成",
        (
            f"kept_patch_count={len(kept_df)}, removed_patch_count={filter_stats['removed_patch_count']}, "
            f"shrunk_patch_count={filter_stats['shrunk_patch_count']}"
        ),
    )
    kept_df, transition_blend_stats = apply_fault_transition_blend_to_existing_patches(
        kept_df=kept_df,
        panel_df=panel_df,
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
        progress_hook=emit_fault_postfusion_progress,
    )
    emit_fault_postfusion_progress(
        "断层过渡带原裂缝混合完成",
        f"transition_blended_original_count={transition_blend_stats['transition_blended_original_count']}",
    )
    parallel_df, perpendicular_df = build_generated_patch_rows(
        panel_df=panel_df,
        parallel_ratio=float(parallel_ratio),
        random_seed=int(random_seed),
        fault_half_band_ms=float(fault_half_band_ms),
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
        fault_induced_count_scale=float(fault_induced_count_scale),
        transition_ref_lookup=build_transition_reference_lookup(
            influenced_df=influenced_df,
            panel_df=panel_df,
            fault_remove_ms=float(fault_remove_ms),
            fault_transition_ms=float(fault_transition_ms),
        ),
        progress_hook=emit_fault_postfusion_progress,
    )
    emit_fault_postfusion_progress(
        "断层诱导裂缝生成完成",
        f"parallel_count={len(parallel_df)}, perpendicular_count={len(perpendicular_df)}",
    )
    final_df = finalize_output_dataframe(kept_df, parallel_df, perpendicular_df)
    emit_fault_postfusion_progress("断层后融合结果拼接完成", f"fracture_patch_count={len(final_df)}")

    output_csv = run_dir / "regional_dfn_fault_embedded_fractures.csv"
    fractures_vtk = run_dir / "regional_dfn_fault_embedded_fractures_raw.vtk"
    output_vtk = run_dir / "regional_dfn_fault_embedded_raw.vtk"
    write_csv_utf8(final_df, output_csv)
    write_df_to_regional_vtk(final_df, f"{title}_fault_embedded_fractures", fractures_vtk, scalar_types)
    emit_fault_postfusion_progress("断层裂缝 VTK 写出完成", f"fractures_vtk={fractures_vtk}")
    vtk_merge_stats = combine_with_fault_surface_vtk(
        fracture_vtk=fractures_vtk,
        fault_surface_vtk=Path(fault_surface_vtk) if fault_surface_vtk else None,
        output_vtk=output_vtk,
        title=f"{title}_fault_embedded",
    )
    emit_fault_postfusion_progress(
        "断层 surface 融合完成",
        (
            f"fault_surface_polygon_count={vtk_merge_stats['surface_polygon_count']}, "
            f"final_polygon_count={vtk_merge_stats['final_polygon_count']}"
        ),
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
        "fault_induced_count_scale": float(fault_induced_count_scale),
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
        "transition_blend_stats": {
            **transition_blend_stats,
            "transition_blended_generated_count": int(
                pd.to_numeric(
                    pd.concat(
                        [
                            parallel_df.get("FaultTransitionBlendApplied", pd.Series(dtype=float)),
                            perpendicular_df.get("FaultTransitionBlendApplied", pd.Series(dtype=float)),
                        ],
                        ignore_index=True,
                    ),
                    errors="coerce",
                ).fillna(0).astype(int).sum()
            ),
        },
        "origin_counts": origin_counts,
        "action_counts": action_counts,
    }
    summary_path = run_dir / "regional_dfn_fault_embedded_summary.json"
    write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    emit_fault_postfusion_progress("断层后融合结果导出完成", f"summary_json={summary_path}")
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
        fault_induced_count_scale=float(args.fault_induced_count_scale),
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
