from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
LEGACY_STEP7B_DIR = CURRENT_DIR.parent / "step7b_initial_dfn_3d"
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_step7b_medium_v4.json"
if str(LEGACY_STEP7B_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_STEP7B_DIR))

import build_initial_dfn_from_3d_density_sgy as legacy  # noqa: E402
from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    load_contract_for_mapping,
    surface_grids_from_contract,
    validate_window_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step7B medium-scale DFN from local candidate-band voxel geometry.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dfn_csv": output_dir / "medium_dfn_patches.csv",
        "raw_vtk": output_dir / "medium_dfn_raw_time.vtk",
        "candidate_components_vtk": output_dir / "medium_candidate_components_raw_time.vtk",
        "audit_csv": output_dir / "medium_generation_audit.csv",
        "component_summary_csv": output_dir / "medium_component_summary.csv",
        "summary_json": output_dir / "medium_dfn_summary.json",
    }


def load_optional_grid(path_text: str | None, source_trace_idx: np.ndarray, samples: np.ndarray) -> np.ndarray | None:
    if not path_text:
        return None
    path = Path(path_text).resolve()
    if not path.exists():
        return None
    grid, _summary = legacy.load_guidance_grid(path, source_trace_idx, samples)
    grid[np.abs(grid) > 1.0e6] = np.nan
    return grid


def layer_candidates(
    prior: np.ndarray,
    mask_grid: np.ndarray,
    grid: dict[str, Any],
    surfaces: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    component_rows: list[dict[str, Any]] = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    min_voxels = int(config.get("min_component_voxels", 45))
    min_score = float(config.get("min_prior_score", 0.45))

    for layer in legacy.ALLOWED_LAYERS:
        layer_mask = legacy.layer_mask_for_grid(layer, grid["samples"], surfaces)
        raw = layer_mask & (mask_grid > 0.5) & np.isfinite(prior) & (prior >= min_score)
        labels, count = ndimage.label(raw, structure=structure)
        if count <= 0:
            continue
        sizes = np.bincount(labels.ravel())
        for component_id in range(1, count + 1):
            size = int(sizes[component_id])
            if size < min_voxels:
                continue
            yy, xx, tt = np.where(labels == component_id)
            if yy.size == 0:
                continue
            comp_score = prior[yy, xx, tt].astype(float)
            top = surfaces["T4_TIME"][yy, xx] if layer == "沙三段" else surfaces["T6_TIME"][yy, xx]
            base = surfaces["T6_TIME"][yy, xx] if layer == "沙三段" else surfaces["T7_TIME"][yy, xx]
            frame = pd.DataFrame(
                {
                    "LayerGroup": layer,
                    "LayerCode": legacy.LAYER_CODE[layer],
                    "IY": yy.astype(np.int32),
                    "IX": xx.astype(np.int32),
                    "IT": tt.astype(np.int32),
                    "SourceTraceIdx": grid["source_trace_idx"][yy, xx].astype(np.int64),
                    "CenterTime": grid["samples"][tt].astype(float),
                    "TimeWindowMin": top.astype(float),
                    "TimeWindowMax": base.astype(float),
                    "LayerThickness": (base - top).astype(float),
                    "SourceDensity": np.maximum(comp_score, 1.0e-6),
                    "GuidedDensityScore": comp_score,
                    "CandidateScore": comp_score,
                    "SamplingWeight": comp_score,
                    "ComponentID": component_id,
                    "ComponentVoxelCount": size,
                    "LayerDensityThreshold": min_score,
                    "FractureScale": "medium",
                    "FractureScaleCode": 2,
                }
            )
            rows.append(frame)
            component_rows.append(
                {
                    "LayerGroup": layer,
                    "ComponentID": int(component_id),
                    "VoxelCount": size,
                    "ScoreMean": float(np.mean(comp_score)),
                    "ScoreMax": float(np.max(comp_score)),
                    "XExtentCells": int(xx.max() - xx.min() + 1),
                    "YExtentCells": int(yy.max() - yy.min() + 1),
                    "TimeExtentSamples": int(tt.max() - tt.min() + 1),
                    "TimeExtentMs": float(grid["samples"][tt].max() - grid["samples"][tt].min()),
                }
            )
    if not rows:
        raise RuntimeError("no medium candidate components found")
    candidates = pd.concat(rows, ignore_index=True)
    candidates = candidates[candidates["SourceTraceIdx"].ge(0) & candidates["LayerThickness"].gt(0)].reset_index(drop=True)
    return candidates, component_rows


def component_candidates_from_step6b(
    config: dict[str, Any],
    grid: dict[str, Any],
    surfaces: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    component_path = Path(config["medium_components_npz"]).resolve()
    summary_path = Path(config["medium_component_summary_csv"]).resolve()
    component_npz = np.load(component_path)
    component_flat = component_npz["medium_component_id"].astype(np.int32)
    if component_flat.shape != (grid["tracecount"], grid["sample_count"]):
        raise ValueError(f"medium component shape {component_flat.shape} does not match prior SGY")
    mapping = np.load(Path(config["trace_mapping_npz"]).resolve())
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    component_grid = np.zeros_like(grid["density"], dtype=np.int32)
    component_grid[iy, ix, :] = component_flat
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise RuntimeError("Step6B component summary is empty")
    summary["selection_score"] = (
        pd.to_numeric(summary["voxel_count"], errors="coerce").fillna(0.0)
        * pd.to_numeric(summary["score_mean"], errors="coerce").fillna(0.0)
    )
    max_components = int(config.get("max_component_count", 80))
    summary = summary.sort_values("selection_score", ascending=False)
    if max_components > 0:
        summary = summary.head(max_components)
    max_id = int(component_grid.max())
    slices = ndimage.find_objects(component_grid, max_label=max_id)
    max_points = max(int(config.get("geometry_max_points_per_component", 5000)), 100)
    min_score = float(config.get("min_prior_score", 0.65))
    rows: list[pd.DataFrame] = []
    component_rows: list[dict[str, Any]] = []
    for comp_row in summary.itertuples(index=False):
        component_id = int(comp_row.component_id)
        component_slice = slices[component_id - 1] if 0 < component_id <= len(slices) else None
        if component_slice is None:
            continue
        local_ids = component_grid[component_slice]
        yy, xx, tt = np.where(local_ids == component_id)
        yy += int(component_slice[0].start)
        xx += int(component_slice[1].start)
        tt += int(component_slice[2].start)
        scores = grid["density"][yy, xx, tt]
        valid = np.isfinite(scores) & (scores >= min_score)
        yy, xx, tt, scores = yy[valid], xx[valid], tt[valid], scores[valid]
        if len(scores) == 0:
            continue
        if len(scores) > max_points:
            keep = np.argpartition(scores, -max_points)[-max_points:]
            yy, xx, tt, scores = yy[keep], xx[keep], tt[keep], scores[keep]
        for layer in legacy.ALLOWED_LAYERS:
            top = surfaces["T4_TIME"][yy, xx] if layer == "沙三段" else surfaces["T6_TIME"][yy, xx]
            base = surfaces["T6_TIME"][yy, xx] if layer == "沙三段" else surfaces["T7_TIME"][yy, xx]
            time = grid["samples"][tt]
            in_layer = np.isfinite(top) & np.isfinite(base) & (time >= top) & (time <= base)
            if not in_layer.any():
                continue
            lyy, lxx, ltt = yy[in_layer], xx[in_layer], tt[in_layer]
            lscore, ltop, lbase = scores[in_layer], top[in_layer], base[in_layer]
            frame = pd.DataFrame(
                {
                    "LayerGroup": layer,
                    "LayerCode": legacy.LAYER_CODE[layer],
                    "IY": lyy.astype(np.int32),
                    "IX": lxx.astype(np.int32),
                    "IT": ltt.astype(np.int32),
                    "SourceTraceIdx": grid["source_trace_idx"][lyy, lxx].astype(np.int64),
                    "CenterTime": grid["samples"][ltt].astype(float),
                    "TimeWindowMin": ltop.astype(float),
                    "TimeWindowMax": lbase.astype(float),
                    "LayerThickness": (lbase - ltop).astype(float),
                    "SourceDensity": lscore.astype(float),
                    "GuidedDensityScore": lscore.astype(float),
                    "CandidateScore": lscore.astype(float),
                    "SamplingWeight": lscore.astype(float),
                    "ComponentID": component_id,
                    "ComponentVoxelCount": int(comp_row.voxel_count),
                    "GlobalComponentAzimuthDeg": float(comp_row.pca_azimuth_deg),
                    "GlobalComponentDipDeg": float(comp_row.pca_dip_deg),
                    "GlobalComponentLinearity": float(comp_row.pca_linearity),
                    "GlobalComponentXExtentM": float(comp_row.x_extent_m),
                    "GlobalComponentYExtentM": float(comp_row.y_extent_m),
                    "GlobalComponentTimeExtentMs": float(comp_row.time_extent_ms),
                    "LayerDensityThreshold": min_score,
                    "FractureScale": "medium",
                    "FractureScaleCode": 2,
                }
            )
            rows.append(frame)
            component_rows.append(
                {
                    "LayerGroup": layer,
                    "ComponentID": component_id,
                    "VoxelCount": int(comp_row.voxel_count),
                    "GeometryPointCount": int(len(frame)),
                    "ScoreMean": float(np.mean(lscore)),
                    "ScoreMax": float(np.max(lscore)),
                    "TimeExtentMs": float(frame["CenterTime"].max() - frame["CenterTime"].min()),
                    "GlobalAzimuthDeg": float(comp_row.pca_azimuth_deg),
                    "GlobalDipDeg": float(comp_row.pca_dip_deg),
                    "GlobalLinearity": float(comp_row.pca_linearity),
                    "GlobalXExtentM": float(comp_row.x_extent_m),
                    "GlobalYExtentM": float(comp_row.y_extent_m),
                    "GlobalTimeExtentMs": float(comp_row.time_extent_ms),
                }
            )
    if not rows:
        raise RuntimeError("no medium candidates remained after Step6B component reuse")
    return pd.concat(rows, ignore_index=True), component_rows


def local_geometry(
    group: pd.DataFrame,
    local_idx: int,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    x_values = grid["x_values"]
    y_values = grid["y_values"]
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))
    coords = np.column_stack(
        [
            x_values[group["IX"].to_numpy(dtype=int)],
            y_values[group["IY"].to_numpy(dtype=int)],
            group["CenterTime"].to_numpy(dtype=float) * time_scale,
        ]
    )
    center = coords[int(local_idx)]
    source_row = group.iloc[int(local_idx)]
    global_azimuth = float(source_row.get("GlobalComponentAzimuthDeg", config.get("fallback_azimuth_deg", 60.0)))
    global_dip = float(source_row.get("GlobalComponentDipDeg", config.get("fallback_dip_deg", 70.0)))
    global_xy_extent = max(
        float(source_row.get("GlobalComponentXExtentM", 0.0)),
        float(source_row.get("GlobalComponentYExtentM", 0.0)),
    )
    global_time_extent = float(source_row.get("GlobalComponentTimeExtentMs", 0.0))
    radius = float(config.get("local_pca_radius_m", 125.0))
    dist = np.linalg.norm(coords - center.reshape(1, 3), axis=1)
    local_mask = dist <= radius
    min_points = int(config.get("local_pca_min_points", 14))
    if int(local_mask.sum()) < min_points:
        order = np.argsort(dist)[: min(len(coords), max(min_points, 30))]
        local_mask = np.zeros(len(coords), dtype=bool)
        local_mask[order] = True
    local_coords = coords[local_mask]
    if len(local_coords) < 3:
        length = float(np.clip(
            float(config.get("length_base_m", 35.0))
            + float(config.get("global_extent_length_fraction", 0.16)) * global_xy_extent,
            float(config.get("min_length_m", 45.0)),
            float(config.get("max_length_m", 220.0)),
        ))
        height = float(np.clip(
            float(config.get("height_base_ms", 5.0))
            + float(config.get("global_extent_height_fraction", 0.10)) * global_time_extent,
            float(config.get("min_height_time_ms", 6.0)),
            float(config.get("max_height_time_ms", 38.0)),
        ))
        return {
            "ok": True,
            "reason": "step6_component_global_geometry_insufficient_local_points",
            "azimuth_deg": global_azimuth,
            "dip_deg": global_dip,
            "length_m": length,
            "height_time_ms": height,
            "band_width_m": 0.0,
            "band_thickness_ms": 0.0,
            "planarity": 0.0,
            "linearity": 0.0,
            "point_count": int(len(local_coords)),
        }
    weights = np.clip(group["SamplingWeight"].to_numpy(dtype=float)[local_mask], 1.0e-6, None)
    weighted_center = np.average(local_coords, axis=0, weights=weights)
    centered = local_coords - weighted_center
    cov = (centered * weights[:, None]).T @ centered / float(weights.sum())
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    axis1 = eigvecs[:, 0]
    axis2 = eigvecs[:, 1]
    normal = eigvecs[:, 2]
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-9)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])), 0.0, 1.0))))
    dip = float(np.clip(dip, float(config.get("orientation_safety_min_dip_deg", 8.0)), float(config.get("orientation_safety_max_dip_deg", 89.0))))
    strike = np.asarray([-normal[1], normal[0]], dtype=float)
    if float(np.linalg.norm(strike)) < 1.0e-8:
        azimuth = float(np.degrees(np.arctan2(axis1[1], axis1[0])) % 180.0)
    else:
        azimuth = float(np.degrees(np.arctan2(strike[1], strike[0])) % 180.0)
    total = float(eigvals.sum())
    linearity = float((eigvals[0] - eigvals[1]) / max(eigvals[0], 1.0e-12)) if total > 0 else 0.0
    planarity = float((eigvals[1] - eigvals[2]) / max(eigvals[0], 1.0e-12)) if total > 0 else 0.0
    proj1 = centered @ axis1
    proj2 = centered @ axis2
    times_ms = local_coords[:, 2] / max(time_scale, 1.0e-9)
    axis_span = float(np.quantile(proj1, 0.90) - np.quantile(proj1, 0.10))
    band_width = float(np.quantile(proj2, 0.90) - np.quantile(proj2, 0.10))
    band_thickness = float(np.quantile(times_ms, 0.90) - np.quantile(times_ms, 0.10))
    min_planarity = float(config.get("local_pca_min_planarity", 0.03))
    local_min_dip = float(config.get("local_geometry_min_dip_deg", config.get("reject_dip_below_deg", 30.0)))
    use_global_orientation = bool(
        config.get("use_global_component_orientation_fallback", True)
        and (planarity < min_planarity or dip < local_min_dip)
    )
    if use_global_orientation:
        azimuth = global_azimuth
        dip = global_dip
    score = float(group.iloc[int(local_idx)]["SamplingWeight"])
    score_factor = float(np.sqrt(np.clip(score, 0.0, 1.0)))
    effective_axis_span = max(
        axis_span,
        float(config.get("global_extent_axis_floor_fraction", 0.12)) * global_xy_extent,
    )
    effective_time_span = max(
        band_thickness,
        float(config.get("global_extent_time_floor_fraction", 0.12)) * global_time_extent,
    )
    length = (
        float(config.get("length_axis_fraction", 0.35)) * max(effective_axis_span, 0.0)
        + float(config.get("length_width_gain", 2.2)) * max(band_width, 0.0)
        + float(config.get("length_base_m", 35.0))
    ) * (1.0 + float(config.get("length_score_gain", 0.35)) * score_factor)
    height = (
        float(config.get("height_time_gain", 1.25)) * max(effective_time_span, 0.0)
        + float(config.get("height_width_time_gain", 0.035)) * max(band_width, 0.0)
        + float(config.get("height_base_ms", 5.0))
    ) * (1.0 + float(config.get("height_score_gain", 0.25)) * score_factor)
    length = float(np.clip(length, float(config.get("min_length_m", 45.0)), float(config.get("max_length_m", 220.0))))
    height = float(np.clip(height, float(config.get("min_height_time_ms", 6.0)), float(config.get("max_height_time_ms", 38.0))))
    return {
        "ok": True,
        "reason": "step6_component_global_orientation_fallback" if use_global_orientation else "local_medium_candidate_band_pca",
        "azimuth_deg": azimuth,
        "dip_deg": dip,
        "length_m": length,
        "height_time_ms": height,
        "band_width_m": max(band_width, 0.0),
        "band_thickness_ms": max(band_thickness, 0.0),
        "axis_span_m": max(axis_span, 0.0),
        "planarity": planarity,
        "linearity": linearity,
        "point_count": int(len(local_coords)),
        "used_global_orientation": int(use_global_orientation),
    }


def spatially_distributed_order(group: pd.DataFrame, time_scale: float, limit: int) -> np.ndarray:
    coords = np.column_stack(
        [
            group["IX"].to_numpy(dtype=float),
            group["IY"].to_numpy(dtype=float),
            group["CenterTime"].to_numpy(dtype=float) * time_scale / 12.5,
        ]
    )
    weights = np.clip(group["SamplingWeight"].to_numpy(dtype=float), 0.0, None)
    first = int(np.argmax(weights)) if len(weights) else 0
    selected = [first]
    remaining = np.ones(len(group), dtype=bool)
    remaining[first] = False
    min_dist = np.linalg.norm(coords - coords[first], axis=1)
    while remaining.any() and len(selected) < max(int(limit), 1):
        score_norm = weights / max(float(weights.max()), 1.0e-9)
        priority = min_dist * (0.75 + 0.25 * score_norm)
        priority[~remaining] = -np.inf
        next_idx = int(np.argmax(priority))
        selected.append(next_idx)
        remaining[next_idx] = False
        min_dist = np.minimum(min_dist, np.linalg.norm(coords - coords[next_idx], axis=1))
    return np.asarray(selected, dtype=np.int64)


def select_medium_patches(candidates: pd.DataFrame, grid: dict[str, Any], config: dict[str, Any], rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    max_total = int(config.get("target_patch_count", 1800))
    max_components = int(config.get("max_component_count", 40))
    component_voxels_per_patch = max(float(config.get("component_voxels_per_patch", 180.0)), 1.0)
    reject_low_dip = bool(config.get("reject_low_dip_patches", False))
    reject_dip_below = float(config.get("reject_dip_below_deg", -1.0))
    adjust_dip_below = float(config.get("adjust_dip_below_deg", -1.0))
    adjust_dip_to = float(config.get("adjust_dip_to_deg", adjust_dip_below))
    low_dip_rejected_count = 0
    low_dip_adjusted_count = 0
    min_center_separation = float(config.get("min_patch_center_separation_m", 25.0))
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))
    component_items = []
    for (layer, component_id), group in candidates.groupby(["LayerGroup", "ComponentID"], dropna=False):
        mass = float(group["SamplingWeight"].sum())
        component_items.append((mass, str(layer), int(component_id), group.copy()))
    component_items.sort(key=lambda item: item[0], reverse=True)
    if max_components > 0:
        component_items = component_items[:max_components]
    total_mass = max(sum(item[0] for item in component_items), 1.0e-9)
    for rank, (mass, layer, component_id, group) in enumerate(component_items, start=1):
        if max_total > 0:
            target = int(round(max_total * mass / total_mass))
        else:
            score_factor = 0.65 + 0.70 * float(np.clip(group["SamplingWeight"].mean(), 0.0, 1.0))
            target = int(round(len(group) * score_factor / component_voxels_per_patch))
        min_per_component = int(config.get("min_patches_per_component", 8))
        max_per_component = int(config.get("max_patches_per_component", 180))
        target = max(target, min_per_component)
        if max_per_component > 0:
            target = min(target, max_per_component)
        target = min(target, len(group))
        if target <= 0:
            continue
        if str(config.get("component_selection_mode", "spatial_farthest")) == "spatial_farthest":
            order = spatially_distributed_order(
                group,
                time_scale=float(config.get("orientation_time_scale_m_per_ms", 1.0)),
                limit=target,
            )
        else:
            weights = np.clip(group["SamplingWeight"].to_numpy(dtype=float), 0.0, None)
            if weights.sum() <= 0:
                order = rng.permutation(np.arange(len(group)))
            else:
                weights = weights / weights.sum()
                order = rng.choice(np.arange(len(group)), size=len(group), replace=False, p=weights)
        selected_rows = []
        selected_coords: list[np.ndarray] = []
        used_cells: set[tuple[int, int, int]] = set()
        for local_idx in order:
            if len(selected_rows) >= target:
                break
            row = group.iloc[int(local_idx)].copy()
            key = (int(row["IY"]), int(row["IX"]), int(row["IT"]))
            if key in used_cells:
                continue
            center_coord = np.asarray(
                [
                    float(grid["x_values"][int(row["IX"])]),
                    float(grid["y_values"][int(row["IY"])]),
                    float(row["CenterTime"]) * time_scale,
                ],
                dtype=float,
            )
            if selected_coords and min_center_separation > 0.0:
                distances = np.linalg.norm(np.vstack(selected_coords) - center_coord, axis=1)
                if float(np.min(distances)) < min_center_separation:
                    continue
            used_cells.add(key)
            geom = local_geometry(group, int(local_idx), grid, config)
            raw_dip = float(geom["dip_deg"])
            final_dip = raw_dip
            orientation_adjusted = 0
            if reject_low_dip and raw_dip < reject_dip_below:
                global_dip = float(row.get("GlobalComponentDipDeg", np.nan))
                if np.isfinite(global_dip) and global_dip >= reject_dip_below:
                    raw_dip = global_dip
                    final_dip = global_dip
                    geom["dip_deg"] = global_dip
                    geom["azimuth_deg"] = float(row.get("GlobalComponentAzimuthDeg", geom["azimuth_deg"]))
                    geom["reason"] = "step6_component_global_orientation_low_local_dip"
                    orientation_adjusted = 1
                    low_dip_adjusted_count += 1
                else:
                    low_dip_rejected_count += 1
                    continue
            if adjust_dip_below >= 0.0 and raw_dip < adjust_dip_below:
                final_dip = max(raw_dip, adjust_dip_to)
                orientation_adjusted = 1
                low_dip_adjusted_count += 1
            row["BandID"] = f"medium_component_{rank:04d}_{layer}_{component_id}"
            row["BandPatchOrdinal"] = 0
            row["BandContinuityMode"] = "medium_local_voxel_band_pca_v4"
            row["BandVoxelCount"] = int(len(group))
            row["BandLengthM"] = 0.0
            row["BandTimeExtentMs"] = float(group["CenterTime"].max() - group["CenterTime"].min())
            row["BandPatchSpacingM"] = 0.0
            row["BandMeanDensity"] = float(group["SourceDensity"].mean())
            row["OverrideAzimuthDeg"] = float(geom["azimuth_deg"])
            row["RawDipDeg"] = raw_dip
            row["OverrideDipDeg"] = final_dip
            row["OrientationAdjusted"] = orientation_adjusted
            row["LowDipRejected"] = 0
            row["LowDipPolicy"] = (
                f"reject_below_{reject_dip_below:g}_adjust_below_{adjust_dip_below:g}_to_{adjust_dip_to:g}"
            )
            row["OverrideLengthM"] = float(geom["length_m"])
            row["OverrideHeightTimeMs"] = float(geom["height_time_ms"])
            row["LocalBandWidthM"] = float(geom["band_width_m"])
            row["LocalBandThicknessMs"] = float(geom["band_thickness_ms"])
            row["LocalBandAxisSpanM"] = float(geom.get("axis_span_m", 0.0))
            row["LocalBandPcaPointCount"] = int(geom["point_count"])
            row["LocalBandPcaPlanarity"] = float(geom["planarity"])
            row["LocalBandPcaLinearity"] = float(geom["linearity"])
            row["PatchShapeMode"] = "rectangular_local_medium_band_pca_v4"
            row["OrientationSourceOverride"] = str(geom["reason"])
            selected_rows.append(row)
            selected_coords.append(center_coord)
        for ordinal, row in enumerate(selected_rows, start=1):
            row["BandPatchOrdinal"] = ordinal
        if selected_rows:
            selected = pd.DataFrame(selected_rows)
            parts.append(selected)
            summaries.append(
                {
                    "BandID": str(selected["BandID"].iloc[0]),
                    "LayerGroup": layer,
                    "ComponentID": component_id,
                    "CandidateVoxelCount": int(len(group)),
                    "SelectedPatchCount": int(len(selected)),
                    "ScoreMass": mass,
                    "ScoreMean": float(group["SamplingWeight"].mean()),
                    "TimeExtentMs": float(group["CenterTime"].max() - group["CenterTime"].min()),
                    "DipMedianDeg": float(selected["OverrideDipDeg"].median()),
                    "AzimuthMedianDeg": float(selected["OverrideAzimuthDeg"].median()),
                    "LengthMedianM": float(selected["OverrideLengthM"].median()),
                    "HeightMedianMs": float(selected["OverrideHeightTimeMs"].median()),
                    "LocalBandWidthMedianM": float(selected["LocalBandWidthM"].median()),
                    "LocalBandThicknessMedianMs": float(selected["LocalBandThicknessMs"].median()),
                }
            )
    if not parts:
        raise RuntimeError("no medium patches selected")
    out = pd.concat(parts, ignore_index=True)
    if max_total > 0 and len(out) > max_total:
        weights = np.clip(out["SamplingWeight"].to_numpy(dtype=float), 0.0, None)
        weights = weights / weights.sum() if weights.sum() > 0 else None
        out = out.iloc[rng.choice(np.arange(len(out)), size=max_total, replace=False, p=weights)].reset_index(drop=True)
    out["DensityCellPatchOrdinal"] = out.groupby(["SourceTraceIdx", "LayerGroup", "IT"]).cumcount() + 1
    scale = float(len(out)) / max(float(out["SamplingWeight"].sum()), 1.0e-9)
    out["ExpectedPatchCountForCell"] = out["SamplingWeight"].astype(float) * scale
    out["EffectiveCountScale"] = scale
    out["CountBasisEffectiveScale"] = scale
    out = out.reset_index(drop=True)
    out.attrs["low_dip_rejected_count"] = int(low_dip_rejected_count)
    out.attrs["low_dip_adjusted_count"] = int(low_dip_adjusted_count)
    return out, summaries


def write_candidate_components_vtk(path: Path, summaries: list[dict[str, Any]], candidates: pd.DataFrame, grid: dict[str, Any], title: str) -> None:
    points: list[tuple[float, float, float]] = []
    vertices: list[int] = []
    band_index_values: list[int] = []
    score_values: list[float] = []
    wanted = {str(item["BandID"]).split("_", 3)[-1] for item in summaries}
    for idx, row in candidates.iterrows():
        key = f"{row['LayerGroup']}_{int(row['ComponentID'])}"
        if key not in wanted:
            continue
        vertices.append(len(points))
        points.append((float(grid["x_values"][int(row["IX"])]), float(grid["y_values"][int(row["IY"])]), float(row["CenterTime"])))
        band_index_values.append(int(row["ComponentID"]))
        score_values.append(float(row["SamplingWeight"]))
        if len(points) >= int(50000):
            break
    out = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA", f"POINTS {len(points)} float"]
    out.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    out.append(f"VERTICES {len(vertices)} {len(vertices) * 2}")
    out.extend(f"1 {idx}" for idx in vertices)
    out.append(f"POINT_DATA {len(points)}")
    out.append("SCALARS ComponentID int 1")
    out.append("LOOKUP_TABLE default")
    out.extend(str(int(v)) for v in band_index_values)
    out.append("SCALARS MediumPrior float 1")
    out.append("LOOKUP_TABLE default")
    out.extend(f"{float(v):.6f}" for v in score_values)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_summary(config_path: Path, config: dict[str, Any], paths: dict[str, Path], candidates: pd.DataFrame, selected: pd.DataFrame, patch_df: pd.DataFrame, bands: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_components = candidates[["LayerGroup", "ComponentID"]].drop_duplicates()
    selected_components = selected[["LayerGroup", "ComponentID"]].drop_duplicates() if len(selected) else pd.DataFrame()
    candidate_component_count = int(len(candidate_components))
    selected_component_count = int(len(selected_components))
    upstream_summary_path = Path(config["medium_component_summary_csv"]).resolve() if config.get("medium_component_summary_csv") else None
    upstream_df = pd.read_csv(upstream_summary_path) if upstream_summary_path is not None and upstream_summary_path.exists() else pd.DataFrame()
    upstream_component_ids = set(pd.to_numeric(upstream_df.get("component_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    selected_component_ids = set(pd.to_numeric(selected.get("ComponentID", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    upstream_component_count = int(len(upstream_component_ids))
    upstream_component_coverage = float(len(upstream_component_ids & selected_component_ids) / max(upstream_component_count, 1))
    target = dict(config.get("target_block") or {})
    x_mid = 0.5 * (float(target.get("x_min", patch_df["CenterX"].min())) + float(target.get("x_max", patch_df["CenterX"].max())))
    y_mid = 0.5 * (float(target.get("y_min", patch_df["CenterY"].min())) + float(target.get("y_max", patch_df["CenterY"].max())))
    quadrant_labels = np.where(
        patch_df["CenterY"].to_numpy(dtype=float) >= y_mid,
        np.where(patch_df["CenterX"].to_numpy(dtype=float) >= x_mid, "NE", "NW"),
        np.where(patch_df["CenterX"].to_numpy(dtype=float) >= x_mid, "SE", "SW"),
    )
    quadrant_counts = {str(key): int(value) for key, value in pd.Series(quadrant_labels).value_counts().items()}
    layer_counts = {str(key): int(value) for key, value in patch_df["LayerGroup"].value_counts().items()}
    coords = np.column_stack(
        [
            patch_df["CenterX"].to_numpy(dtype=float),
            patch_df["CenterY"].to_numpy(dtype=float),
            patch_df["CenterTime"].to_numpy(dtype=float) * float(config.get("orientation_time_scale_m_per_ms", 1.0)),
        ]
    )
    if len(coords) > 1:
        nearest = cKDTree(coords).query(coords, k=2)[0][:, 1]
    else:
        nearest = np.asarray([], dtype=float)
    checks = {
        "has_patches": len(patch_df) > 0,
        "all_medium_scale": bool(patch_df["FractureScale"].astype(str).eq("medium").all()),
        "csv_exists": paths["dfn_csv"].exists(),
        "raw_vtk_exists": bool(not config.get("write_intermediate_vtk", False) or paths["raw_vtk"].exists()),
        "candidate_components_vtk_exists": bool(not config.get("write_intermediate_vtk", False) or paths["candidate_components_vtk"].exists()),
        "orientation_varies": bool(patch_df["AzimuthDeg"].round(2).nunique() > 10 and patch_df["DipDeg"].round(2).nunique() > 10),
        "size_varies": bool(patch_df["LengthM"].std(ddof=0) > 5.0 and patch_df["HeightTimeMs"].std(ddof=0) > 1.0),
        "covers_candidate_components": bool(candidate_component_count == 0 or selected_component_count / candidate_component_count >= float(config.get("min_component_coverage_fraction", 0.70))),
        "covers_upstream_components": bool(upstream_component_count == 0 or upstream_component_coverage >= float(config.get("min_upstream_component_coverage_fraction", 0.65))),
        "covers_spatial_quadrants": bool(len(quadrant_counts) >= int(config.get("min_quadrants_covered", 3))),
        "centers_respect_minimum_spacing": bool(nearest.size == 0 or float(np.quantile(nearest, 0.05)) >= 0.5 * float(config.get("min_patch_center_separation_m", 25.0))),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "generation_logic": "step7b_medium_local_candidate_band_voxel_pca_v4",
        "inputs": {
            "medium_prior_sgy": str(Path(config["medium_prior_sgy"]).resolve()),
            "medium_mask_sgy": str(Path(config["medium_mask_sgy"]).resolve()) if config.get("medium_mask_sgy") else "",
            "medium_components_npz": str(Path(config["medium_components_npz"]).resolve()) if config.get("medium_components_npz") else "",
            "trace_mapping_npz": str(Path(config["trace_mapping_npz"]).resolve()),
            "layer_dir": str(Path(config["layer_dir"]).resolve()),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "patch_count": int(len(patch_df)),
        "band_count": int(len(bands)),
        "candidate_component_count": candidate_component_count,
        "selected_component_count": selected_component_count,
        "selected_component_fraction": float(selected_component_count / max(candidate_component_count, 1)),
        "upstream_component_count": upstream_component_count,
        "upstream_selected_component_count": int(len(upstream_component_ids & selected_component_ids)),
        "upstream_component_coverage_fraction": upstream_component_coverage,
        "quadrant_counts": quadrant_counts,
        "layer_counts": layer_counts,
        "nearest_center_distance": legacy.finite_stats(nearest),
        "band_examples": bands[:30],
        "patch_stats": {
            "length_m": legacy.finite_stats(patch_df["LengthM"]),
            "height_time_ms": legacy.finite_stats(patch_df["HeightTimeMs"]),
            "area_m2": legacy.finite_stats(patch_df["PatchAreaM2"]),
            "azimuth_deg": legacy.finite_stats(patch_df["AzimuthDeg"]),
            "dip_deg": legacy.finite_stats(patch_df["DipDeg"]),
            "raw_dip_deg": legacy.finite_stats(patch_df["RawDipDeg"]) if "RawDipDeg" in patch_df else {},
            "local_band_width_m": legacy.finite_stats(patch_df["LocalBandWidthM"]) if "LocalBandWidthM" in patch_df else {},
            "local_band_thickness_ms": legacy.finite_stats(patch_df["LocalBandThicknessMs"]) if "LocalBandThicknessMs" in patch_df else {},
        },
        "orientation_source_distribution": legacy.layer_distribution(patch_df["OrientationSource"]),
        "low_dip_policy": {
            "reject_low_dip_patches": bool(config.get("reject_low_dip_patches", False)),
            "reject_dip_below_deg": float(config.get("reject_dip_below_deg", -1.0)),
            "adjust_dip_below_deg": float(config.get("adjust_dip_below_deg", -1.0)),
            "adjust_dip_to_deg": float(config.get("adjust_dip_to_deg", config.get("adjust_dip_below_deg", -1.0))),
            "selected_low_dip_adjusted_count": int(pd.to_numeric(patch_df.get("OrientationAdjusted", 0), errors="coerce").fillna(0).sum()),
            "candidate_low_dip_rejected_count": int(selected.attrs.get("low_dip_rejected_count", 0)),
        },
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)
    rng = np.random.default_rng(int(config.get("random_seed", 20260715)))

    print("[step7b-medium-v4] loading medium prior", flush=True)
    trace_mapping_path = Path(config["trace_mapping_npz"]).resolve()
    grid = legacy.load_density_grid(Path(config["medium_prior_sgy"]).resolve(), trace_mapping_path)
    print("[step7b-medium-v4] loading surfaces", flush=True)
    with np.load(trace_mapping_path) as mapping_npz:
        mapping = {key: mapping_npz[key] for key in mapping_npz.files}
    horizon_contract = load_contract_for_mapping(config, mapping)
    validate_window_contract(config, horizon_contract, grid["samples"])
    surfaces = surface_grids_from_contract(mapping, horizon_contract)
    print("[step7b-medium-v4] building candidate components", flush=True)
    if config.get("medium_components_npz"):
        candidates, component_rows = component_candidates_from_step6b(config, grid, surfaces)
    else:
        mask_grid = legacy.load_density_grid(Path(config["medium_mask_sgy"]).resolve(), Path(config["trace_mapping_npz"]).resolve())["density"]
        candidates, component_rows = layer_candidates(grid["density"], mask_grid, grid, surfaces, config)
    print("[step7b-medium-v4] selecting local band patches", flush=True)
    selected, bands = select_medium_patches(candidates, grid, config, rng)
    print(f"[step7b-medium-v4] building patches={len(selected)}", flush=True)
    patch_df, _patch_summary = legacy.build_patch_table(selected, grid["density"], grid["x_values"], grid["y_values"], config, rng)
    for column in [
        "LocalBandWidthM",
        "LocalBandThicknessMs",
        "LocalBandAxisSpanM",
        "LocalBandPcaPointCount",
        "LocalBandPcaPlanarity",
        "LocalBandPcaLinearity",
        "PatchShapeMode",
        "OrientationSourceOverride",
        "RawDipDeg",
        "OrientationAdjusted",
        "LowDipRejected",
        "LowDipPolicy",
    ]:
        if column in selected.columns:
            patch_df[column] = selected[column].to_numpy()
    patch_df["GenerationStage"] = "step7b_medium_local_candidate_band_voxel_pca_v4"
    patch_df["FractureScale"] = "medium"
    patch_df["FractureScaleCode"] = 2
    patch_df["SourceType"] = "medium_anttrack_fracture_corridor"
    patch_df["ConstraintLevel"] = "seismic_prior"
    patch_df.loc[:, "OrientationSource"] = "local_medium_candidate_band_pca"
    patch_df["Confidence"] = np.clip(pd.to_numeric(patch_df["SamplingWeight"], errors="coerce").fillna(0.0), 0.0, 1.0)
    audit_df = legacy.build_audit(patch_df)
    audit_df["ActionReason"] = "medium_scale_local_band_voxel_geometry_from_step6b"

    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(component_rows).to_csv(paths["component_summary_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(bands).to_csv(output_dir / "medium_band_summary.csv", index=False, encoding="utf-8-sig")
    geometry_time_scale = float(config.get("geometry_time_scale_m_per_ms", config.get("orientation_time_scale_m_per_ms", 1.0)))
    if bool(config.get("write_intermediate_vtk", False)):
        legacy.write_legacy_vtk(paths["raw_vtk"], patch_df, "step7b_medium_dfn_raw_time", display=False, display_z_scale=float(config.get("display_z_scale", 5.0)), geometry_time_scale_m_per_ms=geometry_time_scale)
        write_candidate_components_vtk(paths["candidate_components_vtk"], bands, candidates, grid, "medium_candidate_components_raw_time")
    summary = build_summary(config_path, config, paths, candidates, selected, patch_df, bands)
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7b-medium-v4] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7b-medium-v4] VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7b-medium-v4] status={summary['status']} patch_count={len(patch_df)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
