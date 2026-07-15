from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage

from build_multiscale_density_bundle import (
    ensure_dir,
    finite_stats,
    flat_to_grid,
    grid_to_flat,
    high_score,
    load_mapping,
    load_trace_matrix,
    low_score,
    valid_values,
    write_sgy_like,
)


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "output/candidate_cheye1_multiscale_rebalance_v1/step6c_large"
DEFAULT_INPUT_QC_DIR = CURRENT_DIR / "output/candidate_cheye1_multiscale_rebalance_v1/input_qc"
DEFAULT_FAULT_PATCH_ROOT = REPO_ROOT / "小范围DFN生成/断层裂缝片生成/断层划分切割/fault_patches_out/patches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6C large fault/fault-zone prior.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-qc-dir", type=Path, default=DEFAULT_INPUT_QC_DIR)
    parser.add_argument("--fault-patch-root", type=Path, default=DEFAULT_FAULT_PATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coherence-low-quantile", type=float, default=0.20)
    parser.add_argument("--large-candidate-quantile", type=float, default=0.85)
    parser.add_argument("--min-component-voxels", type=int, default=40)
    parser.add_argument("--max-component-voxels-before-split", type=int, default=25000)
    parser.add_argument("--split-tile-cells", type=int, default=32)
    parser.add_argument("--split-time-samples", type=int, default=20)
    parser.add_argument("--inferred-extraction-mode", choices=["component_tiles", "surface_ransac"], default="component_tiles")
    parser.add_argument("--orientation-time-scale-m-per-ms", type=float, default=2.0)
    parser.add_argument("--min-vertical-extent-ms", type=float, default=24.0)
    parser.add_argument("--min-dip-deg", type=float, default=45.0)
    parser.add_argument("--min-horizontal-extent-m", type=float, default=120.0)
    parser.add_argument("--horizontal-max-time-extent-ms", type=float, default=12.0)
    parser.add_argument("--horizontal-min-extent-m", type=float, default=700.0)
    parser.add_argument("--layer-like-max-dip-deg", type=float, default=18.0)
    parser.add_argument("--layer-like-max-time-extent-ms", type=float, default=80.0)
    parser.add_argument("--layer-like-min-horizontal-extent-m", type=float, default=600.0)
    parser.add_argument("--min-ant-or-curv-support", type=float, default=0.08)
    parser.add_argument("--support-neighborhood-cells", type=int, default=1)
    parser.add_argument("--fault-time-padding-ms", type=float, default=20.0)
    parser.add_argument("--fault-xy-padding-m", type=float, default=25.0)
    parser.add_argument("--vtk-max-points", type=int, default=250000)
    parser.add_argument("--surface-ransac-iterations", type=int, default=180)
    parser.add_argument("--surface-ransac-distance-m", type=float, default=45.0)
    parser.add_argument("--surface-ransac-max-points", type=int, default=60000)
    parser.add_argument("--surface-ransac-max-surfaces-per-component", type=int, default=5)
    parser.add_argument("--surface-ransac-min-inlier-voxels", type=int, default=800)
    parser.add_argument("--surface-ransac-min-inlier-fraction", type=float, default=0.04)
    parser.add_argument("--surface-ransac-max-raw-components", type=int, default=20)
    parser.add_argument("--surface-ransac-min-cluster-voxels", type=int, default=350)
    parser.add_argument("--surface-max-panel-length-m", type=float, default=1200.0)
    parser.add_argument("--surface-max-panel-height-ms", type=float, default=360.0)
    parser.add_argument("--surface-max-horizontal-extent-m", type=float, default=1800.0)
    parser.add_argument("--surface-max-time-extent-ms", type=float, default=520.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def grid_axis_values(mapping: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    x_values = np.zeros(nx, dtype=np.float64)
    y_values = np.zeros(ny, dtype=np.float64)
    for idx in range(nx):
        x_values[idx] = float(np.median(mapping["x"][ix == idx]))
    for idx in range(ny):
        y_values[idx] = float(np.median(mapping["y"][iy == idx]))
    return x_values, y_values


def load_fault_overlap(input_qc_dir: Path) -> pd.DataFrame:
    path = input_qc_dir / "fault_patch_demo_overlap.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage 0 fault overlap CSV not found: {path}")
    return pd.read_csv(path)


def vtp_path_for_row(root: Path, row: pd.Series) -> Path:
    fault_name = str(row["fault_name"])
    return root / fault_name / f"{fault_name}__i{int(row['cell_i'])}_j{int(row['cell_j'])}.vtp"


def merge_fault_vtps(selected: pd.DataFrame, root: Path, output_path: Path) -> dict[str, Any]:
    meshes = []
    missing = 0
    for _, row in selected.iterrows():
        path = vtp_path_for_row(root, row)
        if not path.exists():
            missing += 1
            continue
        mesh = pv.read(path)
        if mesh.n_points > 0:
            mesh = mesh.copy()
            mesh["FaultPatchArea"] = np.full(mesh.n_points, float(row.get("area_3d", 0.0)), dtype=np.float32)
            meshes.append(mesh)
    if not meshes:
        pv.PolyData().save(output_path)
        return {"selected_patch_count": int(len(selected)), "written_patch_count": 0, "missing_patch_count": int(missing)}
    merged = meshes[0]
    for mesh in meshes[1:]:
        merged = merged.merge(mesh)
    merged.save(output_path)
    return {
        "selected_patch_count": int(len(selected)),
        "written_patch_count": int(len(meshes)),
        "missing_patch_count": int(missing),
        "n_points": int(merged.n_points),
        "n_cells": int(merged.n_cells),
    }


def rasterize_original_faults(
    selected: pd.DataFrame,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_values, y_values = grid_axis_values(mapping)
    ny = len(y_values)
    nx = len(x_values)
    nt = len(samples)
    prior = np.zeros((ny, nx, nt), dtype=np.float32)
    mask = np.zeros((ny, nx, nt), dtype=bool)
    audit_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        x0 = float(row["bbox_xmin"]) - float(args.fault_xy_padding_m)
        x1 = float(row["bbox_xmax"]) + float(args.fault_xy_padding_m)
        y0 = float(row["bbox_ymin"]) - float(args.fault_xy_padding_m)
        y1 = float(row["bbox_ymax"]) + float(args.fault_xy_padding_m)
        t0 = float(row["bbox_zmin"]) - float(args.fault_time_padding_ms)
        t1 = float(row["bbox_zmax"]) + float(args.fault_time_padding_ms)
        xx = np.where((x_values >= x0) & (x_values <= x1))[0]
        yy = np.where((y_values >= y0) & (y_values <= y1))[0]
        tt = np.where((samples >= t0) & (samples <= t1))[0]
        if len(xx) == 0 or len(yy) == 0 or len(tt) == 0:
            voxels = 0
        else:
            yy_grid, xx_grid, tt_grid = np.meshgrid(yy, xx, tt, indexing="ij")
            mask[yy_grid, xx_grid, tt_grid] = True
            prior[yy_grid, xx_grid, tt_grid] = np.maximum(prior[yy_grid, xx_grid, tt_grid], 1.0)
            voxels = int(yy_grid.size)
        audit_rows.append(
            {
                "fault_name": str(row["fault_name"]),
                "cell_i": int(row["cell_i"]),
                "cell_j": int(row["cell_j"]),
                "area_3d": float(row["area_3d"]),
                "dip_deg": float(row["dip_deg"]),
                "rasterized_voxel_count": voxels,
            }
        )
    return prior, mask, pd.DataFrame(audit_rows)


def pca_orientation(points: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if points.shape[0] < 3:
        return None, None, None
    centered = points - points.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    main = vh[0]
    normal = vh[-1]
    azimuth = float((np.degrees(np.arctan2(main[0], main[1])) + 360.0) % 180.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1.0e-9), 0.0, 1.0))))
    linearity = float(s[0] / max(s[1], 1.0e-9)) if len(s) > 1 else None
    return azimuth, dip, linearity


def normalize_vector(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm > 1.0e-12:
        return arr / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    fb = np.asarray(fallback, dtype=np.float64)
    fb_norm = float(np.linalg.norm(fb))
    return fb / max(fb_norm, 1.0e-12)


def strike_dip_from_normal(normal: np.ndarray) -> tuple[float, float]:
    n = normalize_vector(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    strike = np.array([-n[1], n[0], 0.0], dtype=np.float64)
    strike_norm = float(np.linalg.norm(strike))
    if strike_norm <= 1.0e-12:
        azimuth = 0.0
    else:
        strike /= strike_norm
        azimuth = float((np.degrees(np.arctan2(strike[0], strike[1])) + 360.0) % 180.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(n[2])), 0.0, 1.0))))
    return azimuth, dip


def plane_axes_from_points(points_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    centered = np.asarray(points_scaled, dtype=np.float64) - np.asarray(points_scaled, dtype=np.float64).mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    axis1 = normalize_vector(vh[0], np.array([1.0, 0.0, 0.0]))
    normal = normalize_vector(vh[-1], np.array([0.0, 0.0, 1.0]))
    axis2 = normalize_vector(np.cross(normal, axis1), np.array([0.0, 0.0, 1.0]))
    if axis2[2] < 0:
        axis2 = -axis2
    azimuth, dip = strike_dip_from_normal(normal)
    planarity = float((s[1] - s[2]) / max(s[0], 1.0e-9)) if len(s) >= 3 else 0.0
    return axis1, axis2, normal, azimuth, dip, planarity


def quad_area(vertices: np.ndarray) -> float:
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.shape != (4, 3):
        return 0.0
    area1 = 0.5 * float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
    area2 = 0.5 * float(np.linalg.norm(np.cross(pts[3] - pts[0], pts[2] - pts[0])))
    return area1 + area2


def surface_vertices_from_inliers(
    points_unscaled: np.ndarray,
    points_scaled: np.ndarray,
    time_scale: float,
    max_length_m: float,
    max_height_ms: float,
) -> tuple[np.ndarray, dict[str, float]]:
    center_scaled = points_scaled.mean(axis=0)
    axis1, axis2, normal, azimuth, dip, planarity = plane_axes_from_points(points_scaled)
    local = np.column_stack(
        [
            (points_scaled - center_scaled) @ axis1,
            (points_scaled - center_scaled) @ axis2,
        ]
    )
    u0, u1 = np.percentile(local[:, 0], [3.0, 97.0])
    v0, v1 = np.percentile(local[:, 1], [3.0, 97.0])
    length = float(np.clip(u1 - u0, 1.0, max_length_m))
    # `axis2` includes scaled time, so convert its vertical component back to ms.
    height_scaled = float(np.clip(v1 - v0, 1.0, max_height_ms * max(time_scale, 1.0e-6)))
    half_l = 0.5 * length
    half_h = 0.5 * height_scaled
    center_unscaled = np.asarray([center_scaled[0], center_scaled[1], center_scaled[2] / max(time_scale, 1.0e-6)], dtype=np.float64)
    axis1_unscaled = np.asarray([axis1[0], axis1[1], axis1[2] / max(time_scale, 1.0e-6)], dtype=np.float64)
    axis2_unscaled = np.asarray([axis2[0], axis2[1], axis2[2] / max(time_scale, 1.0e-6)], dtype=np.float64)
    vertices = np.asarray(
        [
            center_unscaled - half_l * axis1_unscaled - half_h * axis2_unscaled,
            center_unscaled + half_l * axis1_unscaled - half_h * axis2_unscaled,
            center_unscaled + half_l * axis1_unscaled + half_h * axis2_unscaled,
            center_unscaled - half_l * axis1_unscaled + half_h * axis2_unscaled,
        ],
        dtype=np.float64,
    )
    height_ms = float(vertices[:, 2].max() - vertices[:, 2].min())
    stats = {
        "center_x": float(center_unscaled[0]),
        "center_y": float(center_unscaled[1]),
        "center_time_ms": float(center_unscaled[2]),
        "length_m": float(length),
        "height_time_ms": float(height_ms),
        "area_m2": float(quad_area(vertices)),
        "azimuth_deg": float(azimuth),
        "dip_deg": float(dip),
        "planarity": float(planarity),
        "x_min": float(points_unscaled[:, 0].min()),
        "x_max": float(points_unscaled[:, 0].max()),
        "y_min": float(points_unscaled[:, 1].min()),
        "y_max": float(points_unscaled[:, 1].max()),
        "time_min_ms": float(points_unscaled[:, 2].min()),
        "time_max_ms": float(points_unscaled[:, 2].max()),
    }
    return vertices, stats


def build_local_support_grid(ant_grid: np.ndarray, curv_grid: np.ndarray, radius: int) -> np.ndarray:
    support = np.maximum(ant_grid, curv_grid).astype(np.float32)
    if radius <= 0:
        return support
    size = 2 * int(radius) + 1
    return ndimage.maximum_filter(support, size=(size, size, size), mode="nearest").astype(np.float32)


def extract_inferred_faults(
    score_grid: np.ndarray,
    candidate_grid: np.ndarray,
    support_grid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(candidate_grid, structure=structure)
    x_values, y_values = grid_axis_values(mapping)
    component_id_grid = np.zeros(candidate_grid.shape, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    time_scale = float(args.orientation_time_scale_m_per_ms)
    rejected_small = 0
    rejected_horizontal = 0
    rejected_low_dip = 0
    rejected_thin = 0
    rejected_short_xy = 0
    rejected_layer_like = 0
    rejected_low_support = 0
    split_raw_count = 0
    sizes = np.bincount(labels.ravel())
    next_id = 1
    for component_id in range(1, count + 1):
        voxel_count = int(sizes[component_id])
        if voxel_count <= 0:
            continue
        yy, xx, tt = np.where(labels == component_id)
        if yy.size > int(args.max_component_voxels_before_split):
            split_raw_count += 1
            tile = max(int(args.split_tile_cells), 1)
            time_tile = max(int(args.split_time_samples), 1)
            tile_key = (xx // tile) + 10000 * (yy // tile) + 100000000 * (tt // time_tile)
            groups = [np.where(tile_key == key)[0] for key in np.unique(tile_key)]
        else:
            groups = [np.arange(yy.size)]
        for group in groups:
            gyy = yy[group]
            gxx = xx[group]
            gtt = tt[group]
            voxel_count = int(len(group))
            if voxel_count < int(args.min_component_voxels):
                rejected_small += 1
                continue
            x_extent = float(x_values[gxx].max() - x_values[gxx].min())
            y_extent = float(y_values[gyy].max() - y_values[gyy].min())
            t_extent = float(samples[gtt].max() - samples[gtt].min())
            horizontal_extent = max(x_extent, y_extent)
            horizontal_like = bool(
                t_extent <= float(args.horizontal_max_time_extent_ms)
                and horizontal_extent >= float(args.horizontal_min_extent_m)
            )
            if horizontal_like:
                rejected_horizontal += 1
                continue
            if t_extent < float(args.min_vertical_extent_ms):
                rejected_thin += 1
                continue
            if horizontal_extent < float(args.min_horizontal_extent_m):
                rejected_short_xy += 1
                continue
            support_mean = float(np.mean(support_grid[gyy, gxx, gtt]))
            support_max = float(np.max(support_grid[gyy, gxx, gtt]))
            if support_max < float(args.min_ant_or_curv_support):
                rejected_low_support += 1
                continue
            points = np.column_stack([x_values[gxx], y_values[gyy], samples[gtt] * time_scale]).astype(np.float64)
            azimuth, dip, linearity = pca_orientation(points)
            layer_like = bool(
                dip is not None
                and dip <= float(args.layer_like_max_dip_deg)
                and horizontal_extent >= float(args.layer_like_min_horizontal_extent_m)
                and t_extent <= float(args.layer_like_max_time_extent_ms)
            )
            if layer_like:
                rejected_layer_like += 1
                continue
            if dip is None or dip < float(args.min_dip_deg):
                rejected_low_dip += 1
                continue
            component_id_grid[gyy, gxx, gtt] = next_id
            rows.append(
                {
                    "component_id": int(next_id),
                    "raw_component_id": int(component_id),
                    "voxel_count": voxel_count,
                    "score_mean": float(np.mean(score_grid[gyy, gxx, gtt])),
                    "score_max": float(np.max(score_grid[gyy, gxx, gtt])),
                    "support_mean": support_mean,
                    "support_max": support_max,
                    "x_min": float(x_values[gxx].min()),
                    "x_max": float(x_values[gxx].max()),
                    "y_min": float(y_values[gyy].min()),
                    "y_max": float(y_values[gyy].max()),
                    "time_min_ms": float(samples[gtt].min()),
                    "time_max_ms": float(samples[gtt].max()),
                    "x_extent_m": x_extent,
                    "y_extent_m": y_extent,
                    "time_extent_ms": t_extent,
                    "pca_azimuth_deg": azimuth,
                    "pca_dip_deg": dip,
                    "pca_linearity": linearity,
                    "layer_like": layer_like,
                }
            )
            next_id += 1
    summary = {
        "raw_component_count": int(count),
        "kept_component_count": int(next_id - 1),
        "split_raw_component_count": int(split_raw_count),
        "rejected_small_component_count": int(rejected_small),
        "rejected_horizontal_component_count": int(rejected_horizontal),
        "rejected_thin_component_count": int(rejected_thin),
        "rejected_short_xy_component_count": int(rejected_short_xy),
        "rejected_low_support_component_count": int(rejected_low_support),
        "rejected_layer_like_component_count": int(rejected_layer_like),
        "rejected_low_dip_component_count": int(rejected_low_dip),
        "max_component_voxels_before_split": int(args.max_component_voxels_before_split),
        "split_tile_cells": int(args.split_tile_cells),
        "split_time_samples": int(args.split_time_samples),
        "orientation_time_scale_m_per_ms": time_scale,
    }
    return component_id_grid, pd.DataFrame(rows), summary


def fit_plane_ransac(
    points_scaled: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    distance_threshold: float,
    max_points: int,
    min_dip_deg: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points_scaled) < 3:
        return None
    points = np.asarray(points_scaled, dtype=np.float64)
    if len(points) > int(max_points):
        sample_idx = rng.choice(np.arange(len(points)), size=int(max_points), replace=False)
        fit_points = points[sample_idx]
    else:
        fit_points = points
    best_inliers_fit: np.ndarray | None = None
    best_count = 0
    for _ in range(max(int(iterations), 1)):
        ids = rng.choice(np.arange(len(fit_points)), size=3, replace=False)
        p0, p1, p2 = fit_points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-9:
            continue
        normal = normal / normal_norm
        _azimuth, dip = strike_dip_from_normal(normal)
        if dip < float(min_dip_deg):
            continue
        distances = np.abs((fit_points - p0) @ normal)
        inliers = distances <= float(distance_threshold)
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers_fit = inliers
    if best_inliers_fit is None or best_count < 3:
        return None
    coarse_points = fit_points[best_inliers_fit]
    _, _, normal, _, _, _ = plane_axes_from_points(coarse_points)
    center = coarse_points.mean(axis=0)
    distances_all = np.abs((points - center) @ normal)
    all_inliers = distances_all <= float(distance_threshold)
    return normal, all_inliers


def extract_inferred_fault_surfaces(
    score_grid: np.ndarray,
    candidate_grid: np.ndarray,
    support_grid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], list[np.ndarray]]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(candidate_grid, structure=structure)
    x_values, y_values = grid_axis_values(mapping)
    surface_id_grid = np.zeros(candidate_grid.shape, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    surface_vertices: list[np.ndarray] = []
    time_scale = float(args.orientation_time_scale_m_per_ms)
    sizes = np.bincount(labels.ravel())
    raw_ids = [idx for idx in range(1, count + 1) if int(sizes[idx]) >= int(args.min_component_voxels)]
    raw_ids.sort(key=lambda idx: int(sizes[idx]), reverse=True)
    raw_ids = raw_ids[: int(args.surface_ransac_max_raw_components)]

    rejected_small = 0
    rejected_low_support = 0
    rejected_thin = 0
    rejected_short_xy = 0
    rejected_low_dip = 0
    rejected_layer_like = 0
    rejected_low_inlier = 0
    rejected_too_broad = 0
    next_surface_id = 1

    for raw_component_id in raw_ids:
        yy, xx, tt = np.where(labels == raw_component_id)
        if len(yy) < int(args.min_component_voxels):
            rejected_small += 1
            continue
        remaining = np.ones(len(yy), dtype=bool)
        for surface_ord in range(int(args.surface_ransac_max_surfaces_per_component)):
            active_idx = np.where(remaining)[0]
            if len(active_idx) < int(args.surface_ransac_min_inlier_voxels):
                break
            ayy = yy[active_idx]
            axx = xx[active_idx]
            att = tt[active_idx]
            points_unscaled = np.column_stack([x_values[axx], y_values[ayy], samples[att]]).astype(np.float64)
            points_scaled = np.column_stack([points_unscaled[:, 0], points_unscaled[:, 1], points_unscaled[:, 2] * time_scale]).astype(np.float64)
            fit = fit_plane_ransac(
                points_scaled,
                rng=rng,
                iterations=int(args.surface_ransac_iterations),
                distance_threshold=float(args.surface_ransac_distance_m),
                max_points=int(args.surface_ransac_max_points),
                min_dip_deg=float(args.min_dip_deg),
            )
            if fit is None:
                break
            _normal, inliers_local = fit
            inlier_count = int(inliers_local.sum())
            if inlier_count < int(args.surface_ransac_min_inlier_voxels):
                rejected_low_inlier += 1
                break
            if inlier_count / max(len(active_idx), 1) < float(args.surface_ransac_min_inlier_fraction):
                rejected_low_inlier += 1
                break
            iyy = ayy[inliers_local]
            ixx = axx[inliers_local]
            itt = att[inliers_local]
            inlier_points_unscaled = points_unscaled[inliers_local]
            inlier_points_scaled = points_scaled[inliers_local]
            support_mean = float(np.mean(support_grid[iyy, ixx, itt]))
            support_max = float(np.max(support_grid[iyy, ixx, itt]))
            if support_max < float(args.min_ant_or_curv_support):
                rejected_low_support += 1
                remaining[active_idx[inliers_local]] = False
                continue
            x_extent = float(inlier_points_unscaled[:, 0].max() - inlier_points_unscaled[:, 0].min())
            y_extent = float(inlier_points_unscaled[:, 1].max() - inlier_points_unscaled[:, 1].min())
            t_extent = float(inlier_points_unscaled[:, 2].max() - inlier_points_unscaled[:, 2].min())
            horizontal_extent = max(x_extent, y_extent)
            if horizontal_extent > float(args.surface_max_horizontal_extent_m) or t_extent > float(args.surface_max_time_extent_ms):
                rejected_too_broad += 1
                remaining[active_idx[inliers_local]] = False
                continue
            if t_extent < float(args.min_vertical_extent_ms):
                rejected_thin += 1
                remaining[active_idx[inliers_local]] = False
                continue
            if horizontal_extent < float(args.min_horizontal_extent_m):
                rejected_short_xy += 1
                remaining[active_idx[inliers_local]] = False
                continue
            vertices, geom = surface_vertices_from_inliers(
                inlier_points_unscaled,
                inlier_points_scaled,
                time_scale=time_scale,
                max_length_m=float(args.surface_max_panel_length_m),
                max_height_ms=float(args.surface_max_panel_height_ms),
            )
            layer_like = bool(
                geom["dip_deg"] <= float(args.layer_like_max_dip_deg)
                and horizontal_extent >= float(args.layer_like_min_horizontal_extent_m)
                and t_extent <= float(args.layer_like_max_time_extent_ms)
            )
            if layer_like:
                rejected_layer_like += 1
                remaining[active_idx[inliers_local]] = False
                continue
            if geom["dip_deg"] < float(args.min_dip_deg):
                rejected_low_dip += 1
                remaining[active_idx[inliers_local]] = False
                continue
            surface_id_grid[iyy, ixx, itt] = next_surface_id
            row = {
                "component_id": int(next_surface_id),
                "surface_id": int(next_surface_id),
                "raw_component_id": int(raw_component_id),
                "raw_component_voxel_count": int(sizes[raw_component_id]),
                "surface_ordinal_in_raw_component": int(surface_ord + 1),
                "voxel_count": int(inlier_count),
                "inlier_fraction_of_remaining": float(inlier_count / max(len(active_idx), 1)),
                "score_mean": float(np.mean(score_grid[iyy, ixx, itt])),
                "score_max": float(np.max(score_grid[iyy, ixx, itt])),
                "support_mean": support_mean,
                "support_max": support_max,
                "x_min": geom["x_min"],
                "x_max": geom["x_max"],
                "y_min": geom["y_min"],
                "y_max": geom["y_max"],
                "time_min_ms": geom["time_min_ms"],
                "time_max_ms": geom["time_max_ms"],
                "x_extent_m": x_extent,
                "y_extent_m": y_extent,
                "time_extent_ms": t_extent,
                "center_x": geom["center_x"],
                "center_y": geom["center_y"],
                "center_time_ms": geom["center_time_ms"],
                "surface_length_m": geom["length_m"],
                "surface_height_time_ms": geom["height_time_ms"],
                "surface_area_m2": geom["area_m2"],
                "pca_azimuth_deg": geom["azimuth_deg"],
                "pca_dip_deg": geom["dip_deg"],
                "pca_linearity": np.nan,
                "surface_planarity": geom["planarity"],
                "layer_like": layer_like,
            }
            for vertex_idx in range(1, 5):
                row[f"V{vertex_idx}X"] = float(vertices[vertex_idx - 1, 0])
                row[f"V{vertex_idx}Y"] = float(vertices[vertex_idx - 1, 1])
                row[f"V{vertex_idx}Z"] = float(vertices[vertex_idx - 1, 2])
            rows.append(row)
            surface_vertices.append(vertices)
            next_surface_id += 1
            remaining[active_idx[inliers_local]] = False

    summary = {
        "raw_component_count": int(count),
        "candidate_raw_component_count": int(len(raw_ids)),
        "kept_component_count": int(next_surface_id - 1),
        "kept_surface_count": int(next_surface_id - 1),
        "rejected_small_component_count": int(rejected_small),
        "rejected_low_support_surface_count": int(rejected_low_support),
        "rejected_thin_surface_count": int(rejected_thin),
        "rejected_short_xy_surface_count": int(rejected_short_xy),
        "rejected_layer_like_surface_count": int(rejected_layer_like),
        "rejected_low_dip_surface_count": int(rejected_low_dip),
        "rejected_low_inlier_surface_count": int(rejected_low_inlier),
        "rejected_too_broad_surface_count": int(rejected_too_broad),
        "surface_extraction_mode": "surface_ransac",
        "surface_ransac_iterations": int(args.surface_ransac_iterations),
        "surface_ransac_distance_m": float(args.surface_ransac_distance_m),
        "surface_ransac_max_raw_components": int(args.surface_ransac_max_raw_components),
        "surface_ransac_max_surfaces_per_component": int(args.surface_ransac_max_surfaces_per_component),
        "surface_max_horizontal_extent_m": float(args.surface_max_horizontal_extent_m),
        "surface_max_time_extent_ms": float(args.surface_max_time_extent_ms),
        "orientation_time_scale_m_per_ms": time_scale,
    }
    return surface_id_grid, pd.DataFrame(rows), summary, surface_vertices


def write_component_vtk(
    path: Path,
    component_id_grid: np.ndarray,
    score_grid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    yy, xx, tt = np.where(component_id_grid > 0)
    total = int(len(yy))
    if total == 0:
        pv.PolyData().save(path)
        return {"point_count": 0, "written_point_count": 0, "subsampled": False}
    if total > max_points:
        idx = rng.choice(np.arange(total), size=max_points, replace=False)
        yy, xx, tt = yy[idx], xx[idx], tt[idx]
        subsampled = True
    else:
        subsampled = False
    x_values, y_values = grid_axis_values(mapping)
    points = np.column_stack([x_values[xx], y_values[yy], samples[tt]]).astype(np.float32)
    cloud = pv.PolyData(points)
    cloud["ComponentID"] = component_id_grid[yy, xx, tt].astype(np.int32)
    cloud["LargePrior"] = score_grid[yy, xx, tt].astype(np.float32)
    cloud.save(path)
    return {"point_count": total, "written_point_count": int(len(points)), "subsampled": subsampled}


def write_candidate_evidence_vtk(
    path: Path,
    candidate_grid: np.ndarray,
    score_grid: np.ndarray,
    support_grid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    yy, xx, tt = np.where(candidate_grid)
    total = int(len(yy))
    if total == 0:
        pv.PolyData().save(path)
        return {"point_count": 0, "written_point_count": 0, "subsampled": False}
    if total > max_points:
        idx = rng.choice(np.arange(total), size=max_points, replace=False)
        yy, xx, tt = yy[idx], xx[idx], tt[idx]
        subsampled = True
    else:
        subsampled = False
    x_values, y_values = grid_axis_values(mapping)
    points = np.column_stack([x_values[xx], y_values[yy], samples[tt]]).astype(np.float32)
    cloud = pv.PolyData(points)
    cloud["LargePriorRaw"] = score_grid[yy, xx, tt].astype(np.float32)
    cloud["AntCurvSupport"] = support_grid[yy, xx, tt].astype(np.float32)
    cloud.save(path)
    return {"point_count": total, "written_point_count": int(len(points)), "subsampled": subsampled}


def write_surface_candidate_vtk(path: Path, surface_df: pd.DataFrame) -> dict[str, Any]:
    if surface_df.empty:
        pv.PolyData().save(path)
        return {"surface_count": 0, "polygon_count": 0}
    points: list[list[float]] = []
    faces: list[list[int]] = []
    cell_data: dict[str, list[float | int]] = {
        "SurfaceID": [],
        "RawComponentID": [],
        "VoxelCount": [],
        "ScoreMean": [],
        "SupportMean": [],
        "AzimuthDeg": [],
        "DipDeg": [],
        "SurfaceAreaM2": [],
        "SurfacePlanarity": [],
    }
    for _, row in surface_df.iterrows():
        base = len(points)
        for vertex_idx in range(1, 5):
            points.append([float(row[f"V{vertex_idx}X"]), float(row[f"V{vertex_idx}Y"]), float(row[f"V{vertex_idx}Z"])])
        faces.append([4, base, base + 1, base + 2, base + 3])
        cell_data["SurfaceID"].append(int(row["surface_id"]))
        cell_data["RawComponentID"].append(int(row["raw_component_id"]))
        cell_data["VoxelCount"].append(int(row["voxel_count"]))
        cell_data["ScoreMean"].append(float(row["score_mean"]))
        cell_data["SupportMean"].append(float(row["support_mean"]))
        cell_data["AzimuthDeg"].append(float(row["pca_azimuth_deg"]))
        cell_data["DipDeg"].append(float(row["pca_dip_deg"]))
        cell_data["SurfaceAreaM2"].append(float(row["surface_area_m2"]))
        cell_data["SurfacePlanarity"].append(float(row["surface_planarity"]))
    mesh = pv.PolyData(np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int64).ravel())
    for name, values in cell_data.items():
        dtype = np.int32 if name in {"SurfaceID", "RawComponentID", "VoxelCount"} else np.float32
        mesh.cell_data[name] = np.asarray(values, dtype=dtype)
    mesh.save(path)
    return {"surface_count": int(len(surface_df)), "polygon_count": int(mesh.n_cells), "point_count": int(mesh.n_points)}


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    rng = np.random.default_rng(int(args.random_state))

    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    volume_paths = {key: Path(value).resolve() for key, value in dict(config["volume_paths"]).items()}
    mapping = load_mapping(trace_mapping_npz)
    source_trace_idx = mapping["source_trace_idx"].astype(np.int64)
    _, samples, density_load = load_trace_matrix(input_density_sgy, None, None, "Density")

    selected_faults = load_fault_overlap(args.input_qc_dir.resolve())
    selected_faults = selected_faults[selected_faults["intersects_demo_xy_t"].astype(bool)].copy()
    original_fault_grid, original_fault_mask, fault_audit = rasterize_original_faults(selected_faults, mapping, samples, args)
    fault_audit.to_csv(output_dir / "original_fault_rasterization_audit.csv", index=False, encoding="utf-8-sig")
    original_vtk_summary = merge_fault_vtps(
        selected_faults,
        args.fault_patch_root.resolve(),
        output_dir / "original_fault_panels_selected_raw_time.vtk",
    )

    print("[step6c-large] loading seismic attributes", flush=True)
    coherence, _, coh_load = load_trace_matrix(volume_paths["Coherence"], source_trace_idx, samples, "Coherence")
    anttrack, _, ant_load = load_trace_matrix(volume_paths["AntTrack"], source_trace_idx, samples, "AntTrack")
    curvmax, _, curvmax_load = load_trace_matrix(volume_paths["CurvatureMax"], source_trace_idx, samples, "CurvatureMax")
    curvpos, _, curvpos_load = load_trace_matrix(volume_paths["CurvaturePos"], source_trace_idx, samples, "CurvaturePos")

    score_cfg = dict(config.get("score_config", {}))
    coh_cfg = dict(score_cfg.get("coherence_score", {"valid_min": 0.0, "low_quantile": 0.05, "high_quantile": 0.95}))
    ant_cfg = dict(score_cfg.get("anttrack_score", {"low_quantile": 0.20, "high_quantile": 0.96}))
    curvmax_cfg = dict(score_cfg.get("curvaturemax_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    curvpos_cfg = dict(score_cfg.get("curvaturepos_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    coh_valid = valid_values(coherence, coh_cfg)
    ant_valid = valid_values(anttrack, ant_cfg)
    curvmax_valid = valid_values(curvmax, curvmax_cfg)
    curvpos_valid = valid_values(curvpos, curvpos_cfg)
    lowcoh_score, lowcoh_summary = low_score(coherence, coh_valid, coh_cfg)
    ant_score, ant_summary = high_score(anttrack, ant_valid, ant_cfg)
    curvmax_score, curvmax_summary = high_score(curvmax, curvmax_valid, curvmax_cfg)
    curvpos_score, curvpos_summary = high_score(curvpos, curvpos_valid, curvpos_cfg)
    curv_score = np.maximum(curvmax_score, curvpos_score).astype(np.float32)
    ant_grid, _, _ = flat_to_grid(ant_score, mapping)
    curv_grid, _, _ = flat_to_grid(curv_score, mapping)
    support_grid = build_local_support_grid(ant_grid, curv_grid, int(args.support_neighborhood_cells))
    large_score = lowcoh_score * (0.75 + 0.15 * ant_score + 0.10 * curv_score)
    large_score[~coh_valid] = 0.0
    large_score = np.clip(large_score, 0.0, 1.0).astype(np.float32)
    lowcoh_cut = float(np.quantile(lowcoh_score[coh_valid], 1.0 - float(args.coherence_low_quantile)))
    large_cut = float(np.quantile(large_score[large_score > 0], float(args.large_candidate_quantile)))
    candidate_flat = (lowcoh_score >= lowcoh_cut) & (large_score >= large_cut) & coh_valid

    score_grid, _, _ = flat_to_grid(large_score, mapping)
    candidate_grid, _, _ = flat_to_grid(candidate_flat.astype(np.float32), mapping)
    candidate_grid = candidate_grid > 0.5
    evidence_vtk_summary = write_candidate_evidence_vtk(
        output_dir / "inferred_fault_evidence_points_raw_time.vtk",
        candidate_grid,
        score_grid,
        support_grid,
        mapping,
        samples,
        int(args.vtk_max_points),
        rng,
    )
    inferred_surface_vtk_summary: dict[str, Any] = {"surface_count": 0, "polygon_count": 0}
    if args.inferred_extraction_mode == "surface_ransac":
        inferred_id_grid, inferred_df, inferred_summary, _surface_vertices = extract_inferred_fault_surfaces(
            score_grid,
            candidate_grid,
            support_grid,
            mapping,
            samples,
            args,
            rng,
        )
        inferred_surface_vtk_summary = write_surface_candidate_vtk(
            output_dir / "inferred_fault_surface_candidates_raw_time.vtk",
            inferred_df,
        )
    else:
        inferred_id_grid, inferred_df, inferred_summary = extract_inferred_faults(
            score_grid,
            candidate_grid,
            support_grid,
            mapping,
            samples,
            args,
        )
        write_surface_candidate_vtk(output_dir / "inferred_fault_surface_candidates_raw_time.vtk", pd.DataFrame())
    inferred_mask = inferred_id_grid > 0
    combined_prior_grid = np.maximum(original_fault_grid, score_grid * inferred_mask.astype(np.float32))
    combined_mask_grid = original_fault_mask | inferred_mask
    large_prior_flat = grid_to_flat(combined_prior_grid.astype(np.float32), mapping)
    large_mask_flat = grid_to_flat(combined_mask_grid.astype(np.float32), mapping)
    inferred_id_flat = grid_to_flat(inferred_id_grid.astype(np.float32), mapping).astype(np.int32)

    write_sgy_like(output_dir / "large_fault_prior.sgy", input_density_sgy, large_prior_flat, samples)
    write_sgy_like(output_dir / "large_fault_mask.sgy", input_density_sgy, large_mask_flat.astype(np.float32), samples)
    inferred_df.to_csv(output_dir / "large_fault_component_summary.csv", index=False, encoding="utf-8-sig")
    inferred_vtk_summary = write_component_vtk(
        output_dir / "inferred_large_fault_candidates_raw_time.vtk",
        inferred_id_grid,
        score_grid,
        mapping,
        samples,
        int(args.vtk_max_points),
        rng,
    )
    np.savez_compressed(
        output_dir / "large_fault_prior_components.npz",
        large_prior=large_prior_flat.astype(np.float32),
        large_mask=large_mask_flat.astype(np.uint8),
        inferred_component_id=inferred_id_flat.astype(np.int32),
        samples=samples.astype(np.float32),
        x=mapping["x"].astype(np.float64),
        y=mapping["y"].astype(np.float64),
        ix=mapping["ix"].astype(np.int32),
        iy=mapping["iy"].astype(np.int32),
    )

    summary = {
        "status": "pass",
        "input_density_sgy": str(input_density_sgy),
        "output_dir": str(output_dir),
        "load": {
            "density": density_load,
            "coherence": coh_load,
            "anttrack": ant_load,
            "curvaturemax": curvmax_load,
            "curvaturepos": curvpos_load,
        },
        "original_fault": {
            "selected_patch_count": int(len(selected_faults)),
            "rasterized_voxel_count": int(original_fault_mask.sum()),
            "rasterized_voxel_fraction": float(original_fault_mask.mean()),
            "vtk": original_vtk_summary,
            "area_stats": finite_stats(selected_faults["area_3d"]) if len(selected_faults) else finite_stats([]),
            "dip_stats": finite_stats(selected_faults["dip_deg"]) if len(selected_faults) else finite_stats([]),
        },
        "inferred_fault": {
            "parameters": {
                "inferred_extraction_mode": str(args.inferred_extraction_mode),
                "coherence_low_quantile": float(args.coherence_low_quantile),
                "large_candidate_quantile": float(args.large_candidate_quantile),
                "min_component_voxels": int(args.min_component_voxels),
                "max_component_voxels_before_split": int(args.max_component_voxels_before_split),
                "split_tile_cells": int(args.split_tile_cells),
                "split_time_samples": int(args.split_time_samples),
                "orientation_time_scale_m_per_ms": float(args.orientation_time_scale_m_per_ms),
                "min_vertical_extent_ms": float(args.min_vertical_extent_ms),
                "min_dip_deg": float(args.min_dip_deg),
                "min_horizontal_extent_m": float(args.min_horizontal_extent_m),
                "horizontal_max_time_extent_ms": float(args.horizontal_max_time_extent_ms),
                "horizontal_min_extent_m": float(args.horizontal_min_extent_m),
                "layer_like_max_dip_deg": float(args.layer_like_max_dip_deg),
                "layer_like_max_time_extent_ms": float(args.layer_like_max_time_extent_ms),
                "layer_like_min_horizontal_extent_m": float(args.layer_like_min_horizontal_extent_m),
                "min_ant_or_curv_support": float(args.min_ant_or_curv_support),
                "support_neighborhood_cells": int(args.support_neighborhood_cells),
                "surface_ransac_iterations": int(args.surface_ransac_iterations),
                "surface_ransac_distance_m": float(args.surface_ransac_distance_m),
                "surface_ransac_min_inlier_voxels": int(args.surface_ransac_min_inlier_voxels),
                "surface_ransac_min_inlier_fraction": float(args.surface_ransac_min_inlier_fraction),
            },
            "lowcoh_score_threshold": lowcoh_cut,
            "large_score_threshold": large_cut,
            "raw_candidate_voxel_count": int(candidate_flat.sum()),
            "raw_candidate_voxel_fraction": float(candidate_flat.mean()),
            "kept_voxel_count": int(inferred_mask.sum()),
            "kept_voxel_fraction": float(inferred_mask.mean()),
            "component_count": int(len(inferred_df)),
            "component_summary": inferred_summary,
            "component_voxel_stats": finite_stats(inferred_df["voxel_count"]) if len(inferred_df) else finite_stats([]),
            "component_dip_stats": finite_stats(inferred_df["pca_dip_deg"]) if len(inferred_df) else finite_stats([]),
            "component_time_extent_stats": finite_stats(inferred_df["time_extent_ms"]) if len(inferred_df) else finite_stats([]),
            "vtk": inferred_vtk_summary,
            "evidence_vtk": evidence_vtk_summary,
            "surface_vtk": inferred_surface_vtk_summary,
        },
        "combined": {
            "large_prior_stats": finite_stats(large_prior_flat[large_prior_flat > 0]),
            "large_mask_voxel_count": int(large_mask_flat.sum()),
            "large_mask_voxel_fraction": float(large_mask_flat.mean()),
        },
        "attribute_score_summaries": {
            "low_coherence": lowcoh_summary,
            "anttrack": ant_summary,
            "curvaturemax": curvmax_summary,
            "curvaturepos": curvpos_summary,
        },
        "reflection": (
            "Step6C uses segmented original fault panels as hard prior and only adds inferred faults from steep, vertically continuous low-coherence components. "
            "This prevents large-scale DFN from depending only on sparse fault-stick points inside the demo."
        ),
    }
    if len(selected_faults) <= 0:
        summary["status"] = "warn"
        summary["warning"] = "no original segmented fault patches intersect demo; Step7C must rely on inferred candidates and GeoEast stick reconstruction"
    write_json(output_dir / "large_fault_qc.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
