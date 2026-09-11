from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_preview_v1.json"
TAIGU_ROOT = CURRENT_DIR.parent
if str(TAIGU_ROOT) not in sys.path:
    sys.path.insert(0, str(TAIGU_ROOT))

from common.dfn_geometry import initial_dfn_geometry as geometry  # noqa: E402


SCALE_CODE = {"small": 1, "medium": 2, "large": 3}
NULL_ABS_LIMIT = 1.0e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the multiscale Step7B initial DFN.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_stats(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int | None]:
    return geometry.finite_stats(values)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dfn_csv": output_dir / "initial_dfn_fracture_patches.csv",
        "raw_vtk": output_dir / "initial_dfn_raw_time.vtk",
        "audit_csv": output_dir / "initial_dfn_generation_audit.csv",
        "summary_json": output_dir / "initial_dfn_summary.json",
        "band_centerline_vtk": output_dir / "fracture_band_centerlines_raw_time.vtk",
        "patch_centerline_vtk": output_dir / "fracture_patch_centerlines_raw_time.vtk",
        "band_summary_csv": output_dir / "fracture_band_summary.csv",
    }


def score_high(values: np.ndarray, valid: np.ndarray, low_q: float, high_q: float, power: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[valid & np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {"valid_count": 0}
    low = float(np.quantile(finite, low_q))
    high = float(np.quantile(finite, high_q))
    if high <= low:
        high = low + 1.0e-6
    score = np.clip((values - low) / (high - low), 0.0, 1.0)
    score = np.power(score, power).astype(np.float32)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    return score, {"valid_count": int(finite.size), "low_q": low_q, "high_q": high_q, "low": low, "high": high, "stats": finite_stats(finite)}


def score_low(values: np.ndarray, valid: np.ndarray, low_q: float, high_q: float, power: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[valid & np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {"valid_count": 0}
    low = float(np.quantile(finite, low_q))
    high = float(np.quantile(finite, high_q))
    if high <= low:
        high = low + 1.0e-6
    score = np.clip((high - values) / (high - low), 0.0, 1.0)
    score = np.power(score, power).astype(np.float32)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    return score, {"valid_count": int(finite.size), "low_q": low_q, "high_q": high_q, "low": low, "high": high, "stats": finite_stats(finite)}


def load_attribute_grids(config: dict[str, Any], grid: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    out: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    volume_paths = {key: Path(value).resolve() for key, value in dict(config["volume_paths"]).items()}
    for attr in ["Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]:
        path = volume_paths[attr]
        if not path.exists():
            raise FileNotFoundError(f"{attr} SGY not found: {path}")
        print(f"[step7b-multiscale] loading {attr}", flush=True)
        matrix, load_summary = geometry.load_guidance_grid(path, grid["source_trace_idx"], grid["samples"])
        matrix[np.abs(matrix) >= NULL_ABS_LIMIT] = np.nan
        out[attr] = matrix
        summary[attr] = load_summary
    return out, summary


def component_labels(mask: np.ndarray, min_voxels: int) -> tuple[np.ndarray, dict[int, int]]:
    labels, count = label(mask, structure=generate_binary_structure(3, 2))
    if count <= 0:
        return labels.astype(np.int32), {}
    sizes = np.bincount(labels.ravel())
    keep_ids = [int(idx) for idx, size in enumerate(sizes) if idx != 0 and int(size) >= int(min_voxels)]
    keep = np.isin(labels, keep_ids)
    labels = np.where(keep, labels, 0).astype(np.int32)
    return labels, {idx: int(sizes[idx]) for idx in keep_ids}


def voxels_to_frame(
    yy: np.ndarray,
    xx: np.ndarray,
    tt: np.ndarray,
    layer: str,
    scale: str,
    component_ids: np.ndarray,
    component_sizes: dict[int, int],
    density: np.ndarray,
    score: np.ndarray,
    attributes: dict[str, np.ndarray],
    surfaces: dict[str, np.ndarray],
    grid: dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    if yy.size == 0:
        return pd.DataFrame()
    top = surfaces["T4_TIME"][yy, xx] if layer == "沙三段" else surfaces["T6_TIME"][yy, xx]
    base = surfaces["T6_TIME"][yy, xx] if layer == "沙三段" else surfaces["T7_TIME"][yy, xx]
    out = pd.DataFrame(
        {
            "LayerGroup": layer,
            "LayerCode": geometry.LAYER_CODE[layer],
            "IY": yy.astype(np.int32),
            "IX": xx.astype(np.int32),
            "IT": tt.astype(np.int32),
            "SourceTraceIdx": grid["source_trace_idx"][yy, xx].astype(np.int64),
            "CenterTime": grid["samples"][tt].astype(float),
            "TimeWindowMin": top.astype(float),
            "TimeWindowMax": base.astype(float),
            "LayerThickness": (base - top).astype(float),
            "SourceDensity": np.maximum(density[yy, xx, tt].astype(float), 1.0e-6),
            "CoherenceValue": attributes["Coherence"][yy, xx, tt].astype(float),
            "AntTrackValue": attributes["AntTrack"][yy, xx, tt].astype(float),
            "CurvatureMaxValue": attributes["CurvatureMax"][yy, xx, tt].astype(float),
            "CurvaturePosValue": attributes["CurvaturePos"][yy, xx, tt].astype(float),
            "LowCoherenceScore": attributes["LowCoherenceScore"][yy, xx, tt].astype(float),
            "AntTrackScore": attributes["AntTrackScore"][yy, xx, tt].astype(float),
            "CurvatureScore": attributes["CurvatureScore"][yy, xx, tt].astype(float),
            "GuidedDensityScore": score[yy, xx, tt].astype(float),
            "CandidateScore": score[yy, xx, tt].astype(float),
            "SamplingWeight": score[yy, xx, tt].astype(float),
            "ComponentID": component_ids.astype(np.int32),
            "ComponentVoxelCount": np.asarray([component_sizes.get(int(cid), 1) for cid in component_ids], dtype=np.int32),
            "LayerDensityThreshold": float(threshold),
            "FractureScale": scale,
            "FractureScaleCode": SCALE_CODE[scale],
        }
    )
    return out[out["SourceTraceIdx"].ge(0) & out["LayerThickness"].gt(0)].reset_index(drop=True)


def azimuth_from_axis(axis: np.ndarray, fallback: float) -> float:
    xy = np.asarray(axis[:2], dtype=float)
    if np.linalg.norm(xy) < 1.0e-6:
        return fallback
    return float(np.degrees(np.arctan2(xy[1], xy[0])) % 180.0)


def component_stats(group: pd.DataFrame, grid: dict[str, Any], time_scale: float) -> dict[str, Any]:
    stats = geometry._component_axis_stats(group, grid["x_values"], grid["y_values"], time_scale)
    stats["azimuth_deg"] = azimuth_from_axis(stats["axis"], fallback=55.0)
    return stats


def local_axis_for_projection(group: pd.DataFrame, stats: dict[str, Any], target_projection: float, window_m: float) -> np.ndarray:
    projections = np.asarray(stats["projections"], dtype=float)
    if projections.size < 3:
        return np.asarray(stats["axis"], dtype=float)
    local_mask = np.abs(projections - float(target_projection)) <= float(window_m)
    if int(local_mask.sum()) < 6:
        order = np.argsort(np.abs(projections - float(target_projection)))[: min(len(projections), 18)]
        local_mask = np.zeros(len(projections), dtype=bool)
        local_mask[order] = True
    coords = np.asarray(stats["coords"], dtype=float)[local_mask]
    if len(coords) < 3:
        return np.asarray(stats["axis"], dtype=float)
    centered = coords - coords.mean(axis=0)
    if float(np.linalg.norm(centered)) <= 1.0e-8:
        return np.asarray(stats["axis"], dtype=float)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vh[0], dtype=float)
    if np.dot(axis, np.asarray(stats["axis"], dtype=float)) < 0:
        axis = -axis
    return axis


def dip_from_local_axis(axis: np.ndarray, row: pd.Series, scale: str, cfg: dict[str, Any]) -> float:
    horizontal = float(np.linalg.norm(np.asarray(axis[:2], dtype=float)))
    vertical = abs(float(axis[2]))
    plunge = float(np.degrees(np.arctan2(vertical, max(horizontal, 1.0e-6))))
    if scale == "large":
        evidence = float(row.get("LowCoherenceScore", 0.0))
        default_min, default_max = 66.0, 86.0
    else:
        evidence = float(row.get("AntTrackScore", 0.0))
        default_min, default_max = 60.0, 82.0
    min_dip = float(cfg.get("min_dip_deg", default_min))
    max_dip = float(cfg.get("max_dip_deg", default_max))
    plunge_score = float(np.clip(plunge / float(cfg.get("plunge_full_score_deg", 55.0)), 0.0, 1.0))
    evidence_score = float(np.clip(evidence, 0.0, 1.0))
    mix = 0.55 * evidence_score + 0.45 * plunge_score
    return float(np.clip(min_dip + (max_dip - min_dip) * mix, min_dip, max_dip))


def nearest_grid_position(
    coord: np.ndarray,
    layer: str,
    grid: dict[str, Any],
    surfaces: dict[str, np.ndarray],
    time_scale: float,
) -> tuple[int, int, int, float, float, float]:
    x_values = grid["x_values"]
    y_values = grid["y_values"]
    samples = grid["samples"]
    ix = int(np.clip(np.searchsorted(x_values, float(coord[0])), 0, len(x_values) - 1))
    if ix > 0 and abs(float(x_values[ix - 1]) - float(coord[0])) < abs(float(x_values[ix]) - float(coord[0])):
        ix -= 1
    iy = int(np.clip(np.searchsorted(y_values, float(coord[1])), 0, len(y_values) - 1))
    if iy > 0 and abs(float(y_values[iy - 1]) - float(coord[1])) < abs(float(y_values[iy]) - float(coord[1])):
        iy -= 1
    top = float(surfaces["T4_TIME"][iy, ix] if layer == "沙三段" else surfaces["T6_TIME"][iy, ix])
    base = float(surfaces["T6_TIME"][iy, ix] if layer == "沙三段" else surfaces["T7_TIME"][iy, ix])
    if not np.isfinite(top) or not np.isfinite(base) or base <= top:
        top = float(np.nanmin(samples))
        base = float(np.nanmax(samples))
    target_time = float(coord[2]) / max(float(time_scale), 1.0e-9)
    target_time = float(np.clip(target_time, top, base))
    it = int(np.clip(np.searchsorted(samples, target_time), 0, len(samples) - 1))
    if it > 0 and abs(float(samples[it - 1]) - target_time) < abs(float(samples[it]) - target_time):
        it -= 1
    return iy, ix, it, top, base, float(samples[it])


def local_band_geometry(
    group: pd.DataFrame,
    stats: dict[str, Any],
    local_idx: int,
    scale: str,
    cfg: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    coords = np.asarray(stats["coords"], dtype=float)
    if coords.size == 0:
        return {}
    center_coord = coords[int(local_idx)]
    radius = float(cfg.get("local_pca_radius_m", 180.0 if scale == "large" else 110.0))
    distances = np.linalg.norm(coords - center_coord.reshape(1, 3), axis=1)
    local_mask = distances <= radius
    min_points = int(cfg.get("local_pca_min_points", 12))
    if int(local_mask.sum()) < min_points:
        order = np.argsort(distances)[: min(len(coords), max(min_points, 24))]
        local_mask = np.zeros(len(coords), dtype=bool)
        local_mask[order] = True
    local_coords = coords[local_mask]
    local_weights = np.clip(group["SamplingWeight"].to_numpy(dtype=float)[local_mask], 1.0e-6, None)
    if len(local_coords) < 3:
        axis = np.asarray(stats["axis"], dtype=float)
        fallback_azimuth = azimuth_from_axis(axis, fallback=55.0)
        return {
            "azimuth_deg": fallback_azimuth,
            "dip_deg": float(np.clip(layer_param(config, "fallback_dip_deg", str(group["LayerGroup"].iloc[0]), 70.0), 0.0, 89.0)),
            "planarity": 0.0,
            "linearity": 0.0,
            "point_count": int(len(local_coords)),
            "length_m": float(cfg.get("fallback_length_m", 160.0 if scale == "large" else 80.0)),
            "height_time_ms": float(cfg.get("fallback_height_time_ms", 20.0 if scale == "large" else 12.0)),
            "band_width_m": 0.0,
            "band_thickness_ms": 0.0,
            "orientation_source": "fallback_insufficient_local_band_points",
        }

    center = np.average(local_coords, axis=0, weights=local_weights)
    centered = local_coords - center
    cov = (centered * local_weights[:, None]).T @ centered / float(local_weights.sum())
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum())
    axis1 = eigvecs[:, 0]
    axis2 = eigvecs[:, 1]
    normal = eigvecs[:, 2]
    if np.dot(axis1, np.asarray(stats["axis"], dtype=float)) < 0:
        axis1 = -axis1
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / normal_norm
    strike = np.asarray([-normal[1], normal[0]], dtype=float)
    if float(np.linalg.norm(strike)) < 1.0e-8:
        azimuth = azimuth_from_axis(axis1, fallback=55.0)
    else:
        azimuth = float(np.degrees(np.arctan2(strike[1], strike[0])) % 180.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])), 0.0, 1.0))))
    dip = float(np.clip(dip, float(cfg.get("orientation_safety_min_dip_deg", 0.0)), float(cfg.get("orientation_safety_max_dip_deg", 89.0))))
    linearity = float((eigvals[0] - eigvals[1]) / max(eigvals[0], 1.0e-12)) if total > 0 else 0.0
    planarity = float((eigvals[1] - eigvals[2]) / max(eigvals[0], 1.0e-12)) if total > 0 else 0.0

    proj_axis = centered @ axis1
    proj_width = centered @ axis2
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))
    local_time_ms = local_coords[:, 2] / max(time_scale, 1.0e-9)
    axis_span = float(np.quantile(proj_axis, 0.90) - np.quantile(proj_axis, 0.10))
    band_width = float(np.quantile(proj_width, 0.90) - np.quantile(proj_width, 0.10))
    band_thickness = float(np.quantile(local_time_ms, 0.90) - np.quantile(local_time_ms, 0.10))
    selected_density = float(group.iloc[int(local_idx)]["SourceDensity"])
    density_p95 = max(float(group["SourceDensity"].quantile(0.95)), 1.0e-9)
    density_factor = float(np.sqrt(np.clip(selected_density / density_p95, 0.0, float(config.get("max_density_norm", 3.0)))))

    length = (
        float(cfg.get("length_axis_fraction", 0.45)) * max(axis_span, 0.0)
        + float(cfg.get("length_width_gain", 1.8)) * max(band_width, 0.0)
    ) * (1.0 + float(cfg.get("length_density_gain", 0.25)) * density_factor)
    height = (
        float(cfg.get("height_time_gain", 1.15)) * max(band_thickness, 0.0)
        + float(cfg.get("height_width_time_gain", 0.04)) * max(band_width, 0.0)
    ) * (1.0 + float(cfg.get("height_density_gain", 0.20)) * density_factor)
    length = float(np.clip(length, float(cfg.get("min_length_m", 70.0 if scale == "large" else 35.0)), float(cfg.get("max_length_m", 360.0 if scale == "large" else 180.0))))
    height = float(np.clip(height, float(cfg.get("min_height_time_ms", 8.0 if scale == "large" else 5.0)), float(cfg.get("max_height_time_ms", 55.0 if scale == "large" else 32.0))))
    return {
        "azimuth_deg": azimuth,
        "dip_deg": dip,
        "planarity": planarity,
        "linearity": linearity,
        "point_count": int(len(local_coords)),
        "length_m": length,
        "height_time_ms": height,
        "band_width_m": max(band_width, 0.0),
        "band_thickness_ms": max(band_thickness, 0.0),
        "local_axis_span_m": max(axis_span, 0.0),
        "orientation_source": "local_band_candidate_pca_plane",
    }


def pick_chain_rows(
    group: pd.DataFrame,
    stats: dict[str, Any],
    scale: str,
    band_id: str,
    count: int,
    config: dict[str, Any],
    grid: dict[str, Any],
    surfaces: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if group.empty or count <= 0:
        return pd.DataFrame(), {}
    cfg = dict(config.get(f"{scale}_selection", {}))
    projections = np.asarray(stats["projections"], dtype=float)
    if projections.size == 0:
        return pd.DataFrame(), {}
    axis_length = max(float(stats["axis_length_m"]), 1.0)
    requested_spacing = float(cfg.get("patch_spacing_m", 55.0 if scale == "large" else 38.0))
    spacing = max(requested_spacing, 1.0)
    spacing_count = int(round(axis_length / spacing)) + 1
    count = min(max(spacing_count, int(cfg.get("min_patch_count", 3))), int(cfg.get("max_patch_count", count)))
    count = min(count, max(1, len(group)))
    positions = np.linspace(float(np.nanmin(projections)), float(np.nanmax(projections)), count)
    selected = []
    used: set[int] = set()
    used_cells: set[tuple[int, int, int]] = set()
    density = group["SourceDensity"].to_numpy(dtype=float)
    coords = np.asarray(stats["coords"], dtype=float)
    axis = np.asarray(stats["axis"], dtype=float)
    center = np.asarray(stats["center"], dtype=float)
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))
    local_window = float(cfg.get("local_orientation_window_m", max(spacing * 2.5, 100.0)))
    spacing_to_length = float(cfg.get("center_spacing_to_length_ratio", 0.55 if scale == "large" else 0.62))
    target_length = spacing / max(spacing_to_length, 0.10)
    target_height = float(cfg.get("height_time_ms", 18.0 if scale == "large" else 13.0))
    for ordinal, pos in enumerate(positions, start=1):
        target_coord = center + float(pos) * axis
        local_axis = local_axis_for_projection(group, stats, float(pos), local_window)
        rank = np.linalg.norm(coords - target_coord.reshape(1, 3), axis=1) - 1.0e-4 * density
        for local_idx in np.argsort(rank):
            original_idx = int(group.index[int(local_idx)])
            if original_idx in used:
                continue
            template = group.iloc[int(local_idx)].copy()
            layer = str(template["LayerGroup"])
            iy, ix, it, top, base, center_time = nearest_grid_position(
                target_coord,
                layer=layer,
                grid=grid,
                surfaces=surfaces,
                time_scale=time_scale,
            )
            cell_key = (iy, ix, it)
            if cell_key in used_cells:
                continue
            source_trace_idx = int(grid["source_trace_idx"][iy, ix])
            if source_trace_idx < 0:
                continue
            used.add(original_idx)
            used_cells.add(cell_key)
            row = template
            row["IY"] = int(iy)
            row["IX"] = int(ix)
            row["IT"] = int(it)
            row["SourceTraceIdx"] = source_trace_idx
            row["CenterTime"] = center_time
            row["TimeWindowMin"] = top
            row["TimeWindowMax"] = base
            row["LayerThickness"] = float(base - top)
            row["BandID"] = band_id
            row["BandPatchOrdinal"] = ordinal
            row["BandContinuityMode"] = f"{scale}_continuous_centerline_v2"
            row["BandVoxelCount"] = int(len(group))
            row["BandLengthM"] = float(stats["axis_length_m"])
            row["BandTimeExtentMs"] = float(stats["time_extent_ms"])
            row["BandPatchSpacingM"] = float(spacing)
            row["BandMeanDensity"] = float(group["SourceDensity"].mean())
            row["OverrideAzimuthDeg"] = azimuth_from_axis(local_axis, fallback=float(stats["azimuth_deg"]))
            row["OverrideDipDeg"] = dip_from_local_axis(local_axis, row, scale, cfg)
            row["OverrideLengthM"] = float(target_length * (1.0 + 0.18 * np.clip(float(row.get("SamplingWeight", 0.0)), 0.0, 1.0)))
            row["OverrideHeightTimeMs"] = float(target_height * (1.0 + 0.12 * np.clip(float(row.get("SamplingWeight", 0.0)), 0.0, 1.0)))
            selected.append(row)
            break
    out = pd.DataFrame(selected).reset_index(drop=True)
    centerline_points = [
        (
            float(grid["x_values"][int(row["IX"])]),
            float(grid["y_values"][int(row["IY"])]),
            float(row["CenterTime"]),
        )
        for _, row in out.sort_values("BandPatchOrdinal").iterrows()
    ]
    summary = {
        "band_id": band_id,
        "scale": scale,
        "candidate_voxels": int(len(group)),
        "selected_patch_count": int(len(out)),
        "axis_length_m": float(stats["axis_length_m"]),
        "time_extent_ms": float(stats["time_extent_ms"]),
        "azimuth_deg": float(out["OverrideAzimuthDeg"].median()) if "OverrideAzimuthDeg" in out else float(stats["azimuth_deg"]),
        "dip_deg": float(out["OverrideDipDeg"].median()) if "OverrideDipDeg" in out else None,
        "target_spacing_m": float(spacing),
        "target_length_m": float(target_length),
        "mean_density": float(group["SourceDensity"].mean()),
        "mean_score": float(group["SamplingWeight"].mean()),
        "centerline_points": centerline_points,
    }
    return out, summary


def pick_chain_rows_v3(
    group: pd.DataFrame,
    stats: dict[str, Any],
    scale: str,
    band_id: str,
    count: int,
    config: dict[str, Any],
    grid: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if group.empty or count <= 0:
        return pd.DataFrame(), {}
    cfg = dict(config.get(f"{scale}_selection", {}))
    projections = np.asarray(stats["projections"], dtype=float)
    coords = np.asarray(stats["coords"], dtype=float)
    if projections.size == 0 or coords.size == 0:
        return pd.DataFrame(), {}
    axis = np.asarray(stats["axis"], dtype=float)
    center = np.asarray(stats["center"], dtype=float)
    axis_length = max(float(stats["axis_length_m"]), 1.0)
    spacing = float(cfg.get("patch_spacing_m", 90.0 if scale == "large" else 55.0))
    spacing = max(spacing, 1.0)
    spacing_count = int(round(axis_length / spacing)) + 1
    count = min(max(spacing_count, int(cfg.get("min_patch_count", 3))), int(cfg.get("max_patch_count", count)))
    count = min(count, len(group))
    positions = np.linspace(float(np.nanmin(projections)), float(np.nanmax(projections)), count)
    density = group["SourceDensity"].to_numpy(dtype=float)
    score = group["SamplingWeight"].to_numpy(dtype=float)
    selected = []
    used: set[int] = set()
    group_indices = group.index.to_numpy(dtype=int)
    target_centerline = []
    for ordinal, position in enumerate(positions, start=1):
        target_coord = center + float(position) * axis
        target_centerline.append((float(target_coord[0]), float(target_coord[1]), float(target_coord[2])))
        orthogonal = np.linalg.norm((coords - target_coord.reshape(1, 3)) - np.outer((coords - target_coord.reshape(1, 3)) @ axis, axis), axis=1)
        projection_distance = np.abs(projections - float(position))
        rank = projection_distance + float(cfg.get("orthogonal_distance_weight", 0.65)) * orthogonal - 1.0e-4 * density - 1.0e-3 * score
        for local_idx in np.argsort(rank):
            original_idx = int(group_indices[int(local_idx)])
            if original_idx in used:
                continue
            row = group.iloc[int(local_idx)].copy()
            geom = local_band_geometry(group, stats, int(local_idx), scale, cfg, config)
            if not geom:
                continue
            used.add(original_idx)
            row["BandID"] = band_id
            row["BandPatchOrdinal"] = ordinal
            row["BandContinuityMode"] = f"{scale}_local_band_pca_v3"
            row["BandVoxelCount"] = int(len(group))
            row["BandLengthM"] = float(stats["axis_length_m"])
            row["BandTimeExtentMs"] = float(stats["time_extent_ms"])
            row["BandPatchSpacingM"] = float(spacing)
            row["BandMeanDensity"] = float(group["SourceDensity"].mean())
            row["ObjectBandAzimuthDeg"] = float(stats["azimuth_deg"])
            row["ObjectBandDipDeg"] = float(geom["dip_deg"])
            row["ObjectBandLengthM"] = float(axis_length)
            row["ObjectBandCenterSpacingM"] = float(spacing)
            row["ObjectBandOverlapRatio"] = float(max(0.0, 1.0 - spacing / max(float(geom["length_m"]), 1.0)))
            row["OverrideAzimuthDeg"] = float(geom["azimuth_deg"])
            row["OverrideDipDeg"] = float(geom["dip_deg"])
            row["OverrideLengthM"] = float(geom["length_m"])
            row["OverrideHeightTimeMs"] = float(geom["height_time_ms"])
            row["LocalBandWidthM"] = float(geom["band_width_m"])
            row["LocalBandThicknessMs"] = float(geom["band_thickness_ms"])
            row["LocalBandAxisSpanM"] = float(geom.get("local_axis_span_m", 0.0))
            row["LocalBandPcaPointCount"] = int(geom["point_count"])
            row["LocalBandPcaPlanarity"] = float(geom["planarity"])
            row["LocalBandPcaLinearity"] = float(geom["linearity"])
            row["PatchShapeMode"] = "rectangular_local_band_pca_v3"
            selected.append(row)
            break
    out = pd.DataFrame(selected).reset_index(drop=True)
    patch_centerline = [
        (
            float(grid["x_values"][int(row["IX"])]),
            float(grid["y_values"][int(row["IY"])]),
            float(row["CenterTime"]),
        )
        for _, row in out.sort_values("BandPatchOrdinal").iterrows()
    ]
    summary = {
        "band_id": band_id,
        "scale": scale,
        "candidate_voxels": int(len(group)),
        "selected_patch_count": int(len(out)),
        "axis_length_m": float(axis_length),
        "time_extent_ms": float(stats["time_extent_ms"]),
        "azimuth_deg": float(out["OverrideAzimuthDeg"].median()) if "OverrideAzimuthDeg" in out else float(stats["azimuth_deg"]),
        "dip_deg": float(out["OverrideDipDeg"].median()) if "OverrideDipDeg" in out else None,
        "target_spacing_m": float(spacing),
        "mean_density": float(group["SourceDensity"].mean()),
        "mean_score": float(group["SamplingWeight"].mean()),
        "mean_local_band_width_m": float(out["LocalBandWidthM"].mean()) if "LocalBandWidthM" in out else None,
        "mean_local_band_thickness_ms": float(out["LocalBandThicknessMs"].mean()) if "LocalBandThicknessMs" in out else None,
        "centerline_points": target_centerline,
        "patch_centerline_points": patch_centerline,
    }
    return out, summary


def allocate_band_counts(components: list[dict[str, Any]], target_count: int, scale: str, config: dict[str, Any]) -> list[int]:
    if not components or target_count <= 0:
        return []
    cfg = dict(config.get(f"{scale}_selection", {}))
    min_count = int(cfg.get("min_patch_count", 3))
    max_count = int(cfg.get("max_patch_count", 80))
    mass = np.asarray([max(float(item["score_mass"]), 0.0) for item in components], dtype=float)
    if mass.sum() <= 0:
        raw = np.full(len(components), target_count / len(components), dtype=float)
    else:
        raw = target_count * mass / mass.sum()
    counts = np.clip(np.rint(raw).astype(int), min_count, max_count)
    while counts.sum() > target_count and np.any(counts > min_count):
        idx = int(np.argmax(counts - raw))
        if counts[idx] > min_count:
            counts[idx] -= 1
        else:
            break
    while counts.sum() < target_count and np.any(counts < max_count):
        idx = int(np.argmax(raw - counts))
        if counts[idx] < max_count:
            counts[idx] += 1
        else:
            break
    return [int(v) for v in counts]


def select_scale_bands(
    candidates: pd.DataFrame,
    scale: str,
    target_count: int,
    grid: dict[str, Any],
    surfaces: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if candidates.empty or target_count <= 0:
        return pd.DataFrame(), []
    cfg = dict(config.get(f"{scale}_selection", {}))
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))
    items: list[dict[str, Any]] = []
    for _, group in candidates.groupby(["LayerGroup", "ComponentID"], dropna=False):
        if group.empty:
            continue
        stats = component_stats(group, grid, time_scale)
        if stats["axis_length_m"] < float(cfg.get("min_axis_length_m", 100.0)):
            continue
        if stats["time_extent_ms"] < float(cfg.get("min_time_extent_ms", 0.0)):
            continue
        items.append({"group": group, "stats": stats, "score_mass": float(group["SamplingWeight"].sum())})
    items.sort(key=lambda item: item["score_mass"], reverse=True)
    items = items[: int(cfg.get("max_band_count", 20))]
    counts = allocate_band_counts(items, target_count, scale, config)
    selected_parts = []
    band_summaries: list[dict[str, Any]] = []
    for idx, (item, count) in enumerate(zip(items, counts), start=1):
        band_id = f"{scale}_band_{idx:04d}"
        if str(config.get("generation_mode", "")).endswith("_v3"):
            selected, summary = pick_chain_rows_v3(
                item["group"],
                item["stats"],
                scale,
                band_id,
                count,
                config,
                grid=grid,
            )
        else:
            selected, summary = pick_chain_rows(
                item["group"],
                item["stats"],
                scale,
                band_id,
                count,
                config,
                grid=grid,
                surfaces=surfaces,
            )
        if not selected.empty:
            selected_parts.append(selected)
            band_summaries.append(summary)
    if not selected_parts:
        return pd.DataFrame(), band_summaries
    return pd.concat(selected_parts, ignore_index=True), band_summaries


def build_scale_candidates(
    grid: dict[str, Any],
    surfaces: dict[str, np.ndarray],
    attributes: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    density = grid["density"]
    valid = np.isfinite(density) & (density > float(config.get("min_density", 1.0e-6)))
    for layer in geometry.ALLOWED_LAYERS:
        valid |= geometry.layer_mask_for_grid(layer, grid["samples"], surfaces)
    valid_attr = valid & np.isfinite(attributes["Coherence"]) & np.isfinite(attributes["AntTrack"])
    lowcoh, low_summary = score_low(attributes["Coherence"], valid_attr, 0.05, 0.95, power=1.2)
    ant, ant_summary = score_high(attributes["AntTrack"], valid_attr, 0.50, 0.95, power=1.0)
    curv_max, curvmax_summary = score_high(attributes["CurvatureMax"], valid_attr & np.isfinite(attributes["CurvatureMax"]), 0.50, 0.95, power=1.0)
    curv_pos, curvpos_summary = score_high(attributes["CurvaturePos"], valid_attr & np.isfinite(attributes["CurvaturePos"]), 0.50, 0.95, power=1.0)
    curv = np.maximum(curv_max, curv_pos).astype(np.float32)
    attributes["LowCoherenceScore"] = lowcoh
    attributes["AntTrackScore"] = ant
    attributes["CurvatureScore"] = curv

    scale_frames = {"small": [], "medium": [], "large": []}
    summaries: dict[str, Any] = {
        "score_summaries": {
            "low_coherence": low_summary,
            "anttrack_high": ant_summary,
            "curvaturemax_high": curvmax_summary,
            "curvaturepos_high": curvpos_summary,
        },
        "layer_summary": {},
    }
    medium_cfg = dict(config.get("medium_prior", {}))
    large_cfg = dict(config.get("large_prior", {}))
    small_cfg = dict(config.get("small_prior", {}))

    for layer in geometry.ALLOWED_LAYERS:
        layer_mask = geometry.layer_mask_for_grid(layer, grid["samples"], surfaces) & valid_attr
        if not layer_mask.any():
            continue
        layer_density = density[layer_mask]
        d_q95 = max(float(np.quantile(layer_density[layer_density > 0], 0.95)), 1.0e-6) if np.any(layer_density > 0) else 1.0
        density_score = np.clip(density / d_q95, 0.0, 1.0).astype(np.float32)
        medium_score = ant * (0.55 + 0.30 * lowcoh + 0.15 * curv)
        large_score = lowcoh * (0.75 + 0.25 * ant)
        small_score = density_score * np.clip(1.0 - 0.45 * np.maximum(medium_score, large_score), 0.15, 1.0)

        medium_mask = (
            layer_mask
            & (ant >= float(medium_cfg.get("min_ant_score", 0.62)))
            & (medium_score >= float(medium_cfg.get("min_medium_score", 0.48)))
        )
        large_mask = layer_mask & (lowcoh >= float(large_cfg.get("min_lowcoh_score", 0.70))) & (large_score >= float(large_cfg.get("min_large_score", 0.62)))
        small_threshold = float(np.quantile(small_score[layer_mask], float(small_cfg.get("score_quantile", 0.92))))
        small_mask = layer_mask & (small_score >= small_threshold) & (~medium_mask) & (~large_mask)

        for scale, mask, score, cfg in [
            ("medium", medium_mask, medium_score, medium_cfg),
            ("large", large_mask, large_score, large_cfg),
            ("small", small_mask, small_score, small_cfg),
        ]:
            labels, sizes = component_labels(mask, int(cfg.get("min_component_voxels", 12)))
            yy, xx, tt = np.where(labels > 0)
            frame = voxels_to_frame(
                yy=yy,
                xx=xx,
                tt=tt,
                layer=layer,
                scale=scale,
                component_ids=labels[yy, xx, tt],
                component_sizes=sizes,
                density=density,
                score=score,
                attributes=attributes,
                surfaces=surfaces,
                grid=grid,
                threshold=small_threshold if scale == "small" else float(cfg.get(f"min_{scale}_score", 0.0)),
            )
            scale_frames[scale].append(frame)
            summaries["layer_summary"].setdefault(layer, {})[scale] = {
                "raw_voxel_count": int(mask.sum()),
                "kept_voxel_count": int(len(frame)),
                "component_count": int(len(sizes)),
                "score_stats": finite_stats(score[mask]) if mask.any() else finite_stats([]),
            }

    out = {
        scale: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for scale, parts in scale_frames.items()
    }
    return out, summaries


def sample_small(candidates: pd.DataFrame, target_count: int, rng: np.random.Generator) -> pd.DataFrame:
    if candidates.empty or target_count <= 0:
        return pd.DataFrame()
    n = min(int(target_count), len(candidates))
    weights = np.clip(candidates["SamplingWeight"].to_numpy(dtype=float), 0.0, None)
    if weights.sum() <= 0:
        weights = None
    else:
        weights = weights / weights.sum()
    chosen = rng.choice(candidates.index.to_numpy(), size=n, replace=False, p=weights)
    out = candidates.loc[chosen].copy()
    out["BandID"] = ""
    out["BandPatchOrdinal"] = 0
    out["BandContinuityMode"] = "small_background_density_sampling"
    out["BandVoxelCount"] = out["ComponentVoxelCount"].astype(int)
    out["BandLengthM"] = 0.0
    out["BandTimeExtentMs"] = 0.0
    out["BandPatchSpacingM"] = 0.0
    out["BandMeanDensity"] = out["SourceDensity"].astype(float)
    return out.reset_index(drop=True)


def assign_patch_ordinals(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.reset_index(drop=True).copy()
    out["DensityCellPatchOrdinal"] = out.groupby(["SourceTraceIdx", "LayerGroup", "IT"]).cumcount() + 1
    total_weight = max(float(out["SamplingWeight"].sum()), 1.0e-9)
    scale = float(len(out)) / total_weight
    out["ExpectedPatchCountForCell"] = out["SamplingWeight"].to_numpy(dtype=float) * scale
    out["EffectiveCountScale"] = scale
    out["CountBasisEffectiveScale"] = scale
    return out


def write_centerline_vtk(path: Path, band_summaries: list[dict[str, Any]], point_key: str, title: str) -> None:
    points: list[tuple[float, float, float]] = []
    line_cells: list[list[int]] = []
    scalars: dict[str, list[float]] = {
        "BandIndex": [],
        "ScaleCode": [],
        "BandLengthM": [],
        "SelectedPatchCount": [],
        "AzimuthDeg": [],
        "DipDeg": [],
    }
    for band_idx, band in enumerate(band_summaries, start=1):
        centerline = band.get(point_key, [])
        if len(centerline) < 2:
            continue
        start = len(points)
        for x, y, z in centerline:
            points.append((float(x), float(y), float(z)))
        line_cells.append(list(range(start, start + len(centerline))))
        scalars["BandIndex"].append(float(band_idx))
        scalars["ScaleCode"].append(3.0 if str(band.get("scale")) == "large" else 2.0)
        scalars["BandLengthM"].append(float(band.get("axis_length_m", 0.0)))
        scalars["SelectedPatchCount"].append(float(band.get("selected_patch_count", 0)))
        scalars["AzimuthDeg"].append(float(band.get("azimuth_deg", 0.0)))
        scalars["DipDeg"].append(float(band.get("dip_deg", 0.0) or 0.0))
    total_line_size = sum(len(cell) + 1 for cell in line_cells)
    out = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    out.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    out.append(f"LINES {len(line_cells)} {total_line_size}")
    out.extend(f"{len(cell)} {' '.join(str(idx) for idx in cell)}" for cell in line_cells)
    out.append(f"CELL_DATA {len(line_cells)}")
    for name, values in scalars.items():
        out.append(f"SCALARS {name} float 1")
        out.append("LOOKUP_TABLE default")
        safe = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        out.extend(f"{float(value):.6f}" for value in safe)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    grid: dict[str, Any],
    scale_candidates: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    patch_df: pd.DataFrame,
    candidate_summary: dict[str, Any],
    band_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "has_patches": len(patch_df) > 0,
        "has_medium_or_large": bool((patch_df["FractureScale"].astype(str) != "small").any()),
        "raw_vtk_exists": paths["raw_vtk"].exists(),
        "csv_exists": paths["dfn_csv"].exists(),
        "shared_geometry_module": True,
    }
    continuity_rows: list[dict[str, Any]] = []
    band_patch_df = patch_df[patch_df["FractureScale"].astype(str).isin(["large", "medium"])].copy()
    for band_id, group in band_patch_df.groupby("BandID", dropna=False):
        if len(group) < 2:
            continue
        group = group.sort_values("BandPatchOrdinal")
        coords = group[["CenterX", "CenterY", "CenterTime"]].to_numpy(dtype=float)
        dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        length = group["LengthM"].to_numpy(dtype=float)
        local_len = 0.5 * (length[:-1] + length[1:])
        ratio = dist / np.maximum(local_len, 1.0e-6)
        continuity_rows.append(
            {
                "BandID": str(band_id),
                "FractureScale": str(group["FractureScale"].iloc[0]),
                "PatchCount": int(len(group)),
                "StepDistanceMedianM": float(np.median(dist)),
                "StepDistanceP90M": float(np.quantile(dist, 0.90)),
                "StepToLengthRatioMedian": float(np.median(ratio)),
                "StepToLengthRatioP90": float(np.quantile(ratio, 0.90)),
                "AzimuthUniqueCount": int(group["AzimuthDeg"].round(3).nunique()),
                "DipUniqueCount": int(group["DipDeg"].round(3).nunique()),
            }
        )
    continuity_df = pd.DataFrame(continuity_rows)
    continuity_summary: dict[str, Any] = {
        "band_count": int(len(continuity_df)),
        "rows": continuity_rows[:30],
    }
    if not continuity_df.empty:
        continuity_summary["by_scale"] = {}
        for scale, group in continuity_df.groupby("FractureScale"):
            continuity_summary["by_scale"][str(scale)] = {
                "band_count": int(len(group)),
                "step_to_length_ratio_median": float(group["StepToLengthRatioMedian"].median()),
                "step_to_length_ratio_p90_median": float(group["StepToLengthRatioP90"].median()),
                "step_distance_median_m": float(group["StepDistanceMedianM"].median()),
                "azimuth_unique_count_median": float(group["AzimuthUniqueCount"].median()),
                "dip_unique_count_median": float(group["DipUniqueCount"].median()),
            }
    checks["band_step_to_length_ratio_reasonable"] = bool(
        not continuity_df.empty and float(continuity_df["StepToLengthRatioMedian"].median()) <= 0.90
    )
    checks["band_orientation_not_fixed"] = bool(
        not band_patch_df.empty
        and band_patch_df.groupby("FractureScale")["DipDeg"].nunique().reindex(["large", "medium"]).fillna(0).min() > 1
    )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "generation_logic": "multiscale_preview_small_density_medium_anttrack_large_lowcoh",
        "note": "Preview only. It uses current density/attribute volumes until Step6A-D formal multiscale bundle is ready.",
        "inputs": {
            "density_sgy": str(Path(config["density_sgy"]).resolve()),
            "trace_mapping_npz": str(Path(config["trace_mapping_npz"]).resolve()),
            "layer_dir": str(Path(config["layer_dir"]).resolve()),
            "volume_paths": config["volume_paths"],
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "grid": {
            "shape_y_x_t": [int(v) for v in grid["density"].shape],
            "x_range": [float(np.min(grid["x_values"])), float(np.max(grid["x_values"]))],
            "y_range": [float(np.min(grid["y_values"])), float(np.max(grid["y_values"]))],
            "time_range_ms": [float(grid["samples"][0]), float(grid["samples"][-1])],
        },
        "candidate_summary": candidate_summary,
        "scale_candidate_counts": {scale: int(len(df)) for scale, df in scale_candidates.items()},
        "selected_scale_counts": {str(k): int(v) for k, v in selected["FractureScale"].value_counts(dropna=False).items()},
        "patch_count": int(len(patch_df)),
        "patch_scale_counts": {str(k): int(v) for k, v in patch_df["FractureScale"].value_counts(dropna=False).items()},
        "patch_stats": {
            "length_m": finite_stats(patch_df["LengthM"]),
            "height_time_ms": finite_stats(patch_df["HeightTimeMs"]),
            "area_m2": finite_stats(patch_df["PatchAreaM2"]),
            "source_density": finite_stats(patch_df["SourceDensity"]),
            "dip_deg": finite_stats(patch_df["DipDeg"]),
        },
        "band_count": int(len(band_summaries)),
        "band_examples": [{k: v for k, v in item.items() if k != "centerline_points"} for item in band_summaries[:20]],
        "band_continuity_qc": continuity_summary,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)
    rng = np.random.default_rng(int(config.get("random_seed", 20260714)))

    print("[step7b-multiscale] loading density", flush=True)
    grid = geometry.load_density_grid(Path(config["density_sgy"]).resolve(), Path(config["trace_mapping_npz"]).resolve())
    print("[step7b-multiscale] loading attributes", flush=True)
    attributes, attr_summary = load_attribute_grids(config, grid)
    print("[step7b-multiscale] loading surfaces", flush=True)
    surfaces = geometry.attach_surface_grids(Path(config["layer_dir"]).resolve(), grid["x_values"], grid["y_values"])
    print("[step7b-multiscale] building scale candidates", flush=True)
    scale_candidates, candidate_summary = build_scale_candidates(grid, surfaces, attributes, config)
    candidate_summary["attribute_load"] = attr_summary

    scale_counts = dict(config.get("target_patch_count", {"small": 2500, "medium": 2200, "large": 900}))
    print("[step7b-multiscale] selecting medium/large bands", flush=True)
    medium_selected, medium_bands = select_scale_bands(scale_candidates["medium"], "medium", int(scale_counts.get("medium", 0)), grid, surfaces, config)
    large_selected, large_bands = select_scale_bands(scale_candidates["large"], "large", int(scale_counts.get("large", 0)), grid, surfaces, config)
    small_selected = sample_small(scale_candidates["small"], int(scale_counts.get("small", 0)), rng)
    selected_parts = [df for df in [large_selected, medium_selected, small_selected] if not df.empty]
    if not selected_parts:
        raise RuntimeError("no multiscale candidates selected")
    selected = assign_patch_ordinals(pd.concat(selected_parts, ignore_index=True))

    print(f"[step7b-multiscale] building patches={len(selected)}", flush=True)
    patch_df, patch_build_summary = geometry.build_patch_table(
        selected=selected,
        density=grid["density"],
        x_values=grid["x_values"],
        y_values=grid["y_values"],
        config=config,
        rng=rng,
    )
    for column in [
        "AntTrackValue",
        "AntTrackScore",
        "CurvatureMaxValue",
        "CurvaturePosValue",
        "CurvatureScore",
        "LocalBandWidthM",
        "LocalBandThicknessMs",
        "LocalBandAxisSpanM",
        "LocalBandPcaPointCount",
        "LocalBandPcaPlanarity",
        "LocalBandPcaLinearity",
        "PatchShapeMode",
    ]:
        if column in selected.columns:
            patch_df[column] = selected[column].to_numpy()
    if "PatchShapeMode" in patch_df.columns:
        mask = patch_df["PatchShapeMode"].astype(str).eq("rectangular_local_band_pca_v3")
        patch_df.loc[mask, "OrientationSource"] = "local_band_candidate_pca_plane"
    patch_df["GenerationStage"] = "multiscale_step7b_preview"
    audit_df = geometry.build_audit(patch_df)
    audit_df["ActionReason"] = "multiscale_preview_from_density_anttrack_coherence_curvature"

    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    geometry_time_scale = float(config.get("geometry_time_scale_m_per_ms", config.get("orientation_time_scale_m_per_ms", 1.0)))
    geometry.write_patch_vtk(
        paths["raw_vtk"],
        patch_df,
        "multiscale_initial_dfn_preview_raw_time",
        display=False,
        display_z_scale=float(config.get("display_z_scale", 5.0)),
        geometry_time_scale_m_per_ms=geometry_time_scale,
    )
    band_summaries = large_bands + medium_bands
    if band_summaries:
        write_centerline_vtk(paths["band_centerline_vtk"], band_summaries, "centerline_points", "fracture_band_candidate_centerlines_raw_time")
        write_centerline_vtk(paths["patch_centerline_vtk"], band_summaries, "patch_centerline_points", "fracture_patch_centerlines_raw_time")
        pd.DataFrame([{k: v for k, v in item.items() if k not in {"centerline_points", "patch_centerline_points"}} for item in band_summaries]).to_csv(
            paths["band_summary_csv"], index=False, encoding="utf-8-sig"
        )
    summary = build_summary(
        config_path=config_path,
        config=config,
        paths=paths,
        grid=grid,
        scale_candidates=scale_candidates,
        selected=selected,
        patch_df=patch_df,
        candidate_summary={**candidate_summary, "patch_build_summary": patch_build_summary},
        band_summaries=band_summaries,
    )
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7b-multiscale] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7b-multiscale] VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7b-multiscale] summary: {paths['summary_json']}", flush=True)
    print(f"[step7b-multiscale] status={summary['status']} patch_count={len(patch_df)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
