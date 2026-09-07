from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage
from scipy.spatial import Delaunay

CURRENT_DIR = Path(__file__).resolve().parent
TAIGU_ROOT = CURRENT_DIR.parents[1]
if str(TAIGU_ROOT) not in sys.path:
    sys.path.insert(0, str(TAIGU_ROOT))
FORMAL_MAINLINE_ROOT = TAIGU_ROOT.parent / "优化阶段二" / "正式主线"
if str(FORMAL_MAINLINE_ROOT) not in sys.path:
    sys.path.append(str(FORMAL_MAINLINE_ROOT))
sys.path.append(str(FORMAL_MAINLINE_ROOT / "common"))

from common.multiscale_density.build_multiscale_density_bundle import (
    ensure_dir,
    finite_stats,
    flat_to_grid,
    grid_to_flat,
    high_score,
    load_mapping,
    load_trace_matrix,
    low_score,
    read_sgy_sample_axis,
    regular_sample_axis,
    valid_values,
    write_sgy_like,
)


FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from horizon_trace_table.horizon_contract import (  # noqa: E402
    apply_validity_inplace,
    apply_window_inplace,
    contract_summary,
    load_contract_for_mapping,
    surface_grids_from_contract,
    validate_window_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = CURRENT_DIR.parent / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR.parent / "output/default/step6c_large"
DEFAULT_INPUT_QC_DIR = CURRENT_DIR.parent / "output/default/input_qc"
DEFAULT_FAULT_PATCH_ROOT = TAIGU_ROOT / "common/fault_preprocessing_300m/patches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6C large fault/fault-zone prior.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-qc-dir", type=Path, default=DEFAULT_INPUT_QC_DIR)
    parser.add_argument("--fault-patch-root", type=Path, default=DEFAULT_FAULT_PATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coherence-low-quantile", type=float, default=0.20)
    parser.add_argument("--large-candidate-quantile", type=float, default=0.85)
    parser.add_argument("--lowcoh-weight", type=float, default=0.75)
    parser.add_argument("--anttrack-weight", type=float, default=0.15)
    parser.add_argument("--curvature-weight", type=float, default=0.10)
    parser.add_argument("--lowcoh-candidate-floor", type=float, default=0.55)
    parser.add_argument("--large-score-threshold", type=float, default=0.58)
    parser.add_argument("--min-component-voxels", type=int, default=40)
    parser.add_argument("--max-component-voxels-before-split", type=int, default=25000)
    parser.add_argument("--split-tile-cells", type=int, default=32)
    parser.add_argument("--split-time-samples", type=int, default=20)
    parser.add_argument("--inferred-extraction-mode", choices=["component_tiles", "surface_ransac"], default="surface_ransac")
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
    parser.add_argument("--surface-support-score-threshold", type=float, default=0.35)
    parser.add_argument("--surface-min-support-fraction", type=float, default=0.18)
    parser.add_argument("--surface-min-lowcoh-mean", type=float, default=0.58)
    parser.add_argument("--surface-min-planarity", type=float, default=0.08)
    parser.add_argument("--support-neighborhood-cells", type=int, default=1)
    parser.add_argument("--faultlike-filter-mode", choices=["none", "vertical_continuity"], default="vertical_continuity")
    parser.add_argument("--faultlike-xy-radius-cells", type=int, default=1)
    parser.add_argument("--faultlike-min-vertical-extent-ms", type=float, default=120.0)
    parser.add_argument("--faultlike-min-column-hits", type=int, default=8)
    parser.add_argument("--faultlike-max-horizontal-slice-fraction", type=float, default=0.35)
    parser.add_argument("--fault-time-padding-ms", type=float, default=10.0)
    parser.add_argument("--fault-xy-padding-m", type=float, default=12.5)
    parser.add_argument("--vtk-max-points", type=int, default=250000)
    parser.add_argument("--surface-ransac-iterations", type=int, default=180)
    parser.add_argument("--surface-ransac-distance-m", type=float, default=45.0)
    parser.add_argument("--surface-ransac-max-points", type=int, default=60000)
    parser.add_argument("--surface-ransac-max-surfaces-per-component", type=int, default=5)
    parser.add_argument("--surface-ransac-min-inlier-voxels", type=int, default=400)
    parser.add_argument("--surface-ransac-min-inlier-fraction", type=float, default=0.04)
    parser.add_argument("--surface-ransac-max-raw-components", type=int, default=96)
    parser.add_argument("--surface-component-tile-cells", type=int, default=32)
    parser.add_argument("--enable-lowcoh-structure-rescue", action="store_true")
    parser.add_argument("--lowcoh-rescue-min-score", type=float, default=0.78)
    parser.add_argument("--lowcoh-rescue-local-score-threshold", type=float, default=0.70)
    parser.add_argument("--lowcoh-rescue-min-local-fraction", type=float, default=0.18)
    parser.add_argument("--lowcoh-rescue-min-vertical-extent-ms", type=float, default=60.0)
    parser.add_argument("--lowcoh-rescue-max-raw-components", type=int, default=128)
    parser.add_argument("--lowcoh-rescue-spatial-tile-cells", type=int, default=32)
    parser.add_argument("--lowcoh-rescue-bridge-iterations", type=int, default=1)
    parser.add_argument("--lowcoh-rescue-bridge-min-large-score", type=float, default=0.50)
    parser.add_argument("--surface-ransac-min-cluster-voxels", type=int, default=350)
    parser.add_argument("--surface-panel-target-length-m", type=float, default=350.0)
    parser.add_argument("--surface-min-panel-length-m", type=float, default=150.0)
    parser.add_argument("--surface-max-panel-length-m", type=float, default=500.0)
    parser.add_argument("--surface-max-panel-height-ms", type=float, default=240.0)
    parser.add_argument("--surface-min-panel-voxels", type=int, default=120)
    parser.add_argument("--surface-max-horizontal-extent-m", type=float, default=1800.0)
    parser.add_argument("--surface-max-time-extent-ms", type=float, default=520.0)
    parser.add_argument("--strat-following-min-horizontal-extent-m", type=float, default=180.0)
    parser.add_argument("--strat-following-max-relative-position-std", type=float, default=0.035)
    parser.add_argument("--strat-following-max-relative-position-span", type=float, default=0.12)
    parser.add_argument("--strat-following-min-dominant-layer-fraction", type=float, default=0.80)
    parser.add_argument("--irregular-surface-max-points", type=int, default=3500)
    parser.add_argument("--irregular-surface-max-edge-m", type=float, default=220.0)
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


def geometry_hash(mesh: pv.PolyData) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(mesh.points, dtype=np.float64)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64)).tobytes())
    return digest.hexdigest()


def read_triangular_surface(path: Path) -> pv.PolyData:
    loaded = pv.read(path)
    surface = loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface(algorithm="dataset_surface")
    return surface.triangulate()


def merge_fault_vtps(
    selected: pd.DataFrame,
    root: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    meshes = []
    missing = 0
    manifest_rows: list[dict[str, Any]] = []
    fault_names = sorted(selected["fault_name"].astype(str).unique().tolist())
    fault_ids = {name: idx + 1 for idx, name in enumerate(fault_names)}
    source_point_count = 0
    source_cell_count = 0
    for unit_id, (_, row) in enumerate(selected.reset_index(drop=True).iterrows(), start=1):
        path = vtp_path_for_row(root, row)
        if not path.exists():
            missing += 1
            continue
        mesh = read_triangular_surface(path)
        if mesh.n_points > 0:
            mesh = mesh.copy()
            fault_id = int(fault_ids[str(row["fault_name"])])
            mesh.point_data["FaultPatchArea"] = np.full(mesh.n_points, float(row.get("area_3d", 0.0)), dtype=np.float32)
            mesh.cell_data["FaultID"] = np.full(mesh.n_cells, fault_id, dtype=np.int32)
            mesh.cell_data["FaultUnitID"] = np.full(mesh.n_cells, unit_id, dtype=np.int32)
            mesh.cell_data["CellI"] = np.full(mesh.n_cells, int(row["cell_i"]), dtype=np.int32)
            mesh.cell_data["CellJ"] = np.full(mesh.n_cells, int(row["cell_j"]), dtype=np.int32)
            meshes.append(mesh)
            bounds = mesh.bounds
            source_point_count += int(mesh.n_points)
            source_cell_count += int(mesh.n_cells)
            manifest_rows.append(
                {
                    "FaultUnitID": unit_id,
                    "FaultID": fault_id,
                    "FaultName": str(row["fault_name"]),
                    "CellI": int(row["cell_i"]),
                    "CellJ": int(row["cell_j"]),
                    "SourceVTP": str(path),
                    "PointCount": int(mesh.n_points),
                    "CellCount": int(mesh.n_cells),
                    "XMin": float(bounds[0]),
                    "XMax": float(bounds[1]),
                    "YMin": float(bounds[2]),
                    "YMax": float(bounds[3]),
                    "TimeMinMs": float(bounds[4]),
                    "TimeMaxMs": float(bounds[5]),
                    "GeometrySHA256": geometry_hash(mesh),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    if not meshes:
        pv.PolyData().save(output_path)
        return {"selected_patch_count": int(len(selected)), "written_patch_count": 0, "missing_patch_count": int(missing)}
    merged = pv.merge(meshes, merge_points=False)
    merged.save(output_path)
    return {
        "selected_patch_count": int(len(selected)),
        "written_patch_count": int(len(meshes)),
        "missing_patch_count": int(missing),
        "fault_count": int(len(fault_names)),
        "source_point_count": int(source_point_count),
        "source_cell_count": int(source_cell_count),
        "n_points": int(merged.n_points),
        "n_cells": int(merged.n_cells),
        "geometry_preserved_without_point_welding": bool(
            int(merged.n_points) == source_point_count and int(merged.n_cells) == source_cell_count
        ),
        "manifest_csv": str(manifest_path),
    }


def nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(axis, values, side="left")
    upper = np.clip(upper, 0, len(axis) - 1)
    lower = np.clip(upper - 1, 0, len(axis) - 1)
    choose_lower = np.abs(values - axis[lower]) <= np.abs(axis[upper] - values)
    return np.where(choose_lower, lower, upper).astype(np.int32)


def sample_triangle_points(triangle: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    scaled = triangle / spacing[None, :]
    edge_lengths = [np.linalg.norm(scaled[(idx + 1) % 3] - scaled[idx]) for idx in range(3)]
    subdivisions = max(1, int(np.ceil(max(edge_lengths))))
    weights: list[tuple[float, float, float]] = []
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            a = float(i) / subdivisions
            b = float(j) / subdivisions
            weights.append((a, b, 1.0 - a - b))
    bary = np.asarray(weights, dtype=np.float64)
    return bary @ triangle


def rasterize_original_faults(
    selected: pd.DataFrame,
    root: Path,
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
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    dt = float(np.median(np.diff(samples)))
    spacing = np.asarray([abs(dx), abs(dy), abs(dt)], dtype=np.float64)
    for _, row in selected.iterrows():
        path = vtp_path_for_row(root, row)
        sampled_point_count = 0
        triangle_count = 0
        unit_flat_indices: list[np.ndarray] = []
        if path.exists():
            mesh = read_triangular_surface(path)
            faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
            for face in faces:
                if int(face[0]) != 3:
                    continue
                triangle = np.asarray(mesh.points[face[1:4]], dtype=np.float64)
                points = sample_triangle_points(triangle, spacing)
                sampled_point_count += int(len(points))
                triangle_count += 1
                xx = nearest_axis_indices(x_values, points[:, 0])
                yy = nearest_axis_indices(y_values, points[:, 1])
                tt = nearest_axis_indices(samples, points[:, 2])
                in_volume = (
                    (points[:, 0] >= x_values[0] - 0.5 * abs(dx))
                    & (points[:, 0] <= x_values[-1] + 0.5 * abs(dx))
                    & (points[:, 1] >= y_values[0] - 0.5 * abs(dy))
                    & (points[:, 1] <= y_values[-1] + 0.5 * abs(dy))
                    & (points[:, 2] >= samples[0] - 0.5 * abs(dt))
                    & (points[:, 2] <= samples[-1] + 0.5 * abs(dt))
                )
                if in_volume.any():
                    unit_flat_indices.append(
                        np.ravel_multi_index(
                            (yy[in_volume], xx[in_volume], tt[in_volume]),
                            dims=mask.shape,
                        )
                    )
        if unit_flat_indices:
            unique_flat = np.unique(np.concatenate(unit_flat_indices))
            mask.reshape(-1)[unique_flat] = True
            voxels = int(len(unique_flat))
        else:
            voxels = 0
        audit_rows.append(
            {
                "fault_name": str(row["fault_name"]),
                "cell_i": int(row["cell_i"]),
                "cell_j": int(row["cell_j"]),
                "area_3d": float(row["area_3d"]),
                "dip_deg": float(row["dip_deg"]),
                "source_triangle_count": int(triangle_count),
                "sampled_surface_point_count": int(sampled_point_count),
                "rasterized_voxel_count": voxels,
            }
        )
    xy_radius_x = max(int(np.ceil(float(args.fault_xy_padding_m) / max(abs(dx), 1.0e-9))), 0)
    xy_radius_y = max(int(np.ceil(float(args.fault_xy_padding_m) / max(abs(dy), 1.0e-9))), 0)
    time_radius = max(int(np.ceil(float(args.fault_time_padding_ms) / max(abs(dt), 1.0e-9))), 0)
    if mask.any() and (xy_radius_x > 0 or xy_radius_y > 0 or time_radius > 0):
        structure = np.ones(
            (2 * xy_radius_y + 1, 2 * xy_radius_x + 1, 2 * time_radius + 1),
            dtype=bool,
        )
        mask = ndimage.binary_dilation(mask, structure=structure)
    prior[mask] = 1.0
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


def build_local_support_grids(
    ant_grid: np.ndarray,
    curv_grid: np.ndarray,
    radius: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.maximum(ant_grid, curv_grid).astype(np.float32)
    if radius <= 0:
        return support, (support >= float(score_threshold)).astype(np.float32)
    size = 2 * int(radius) + 1
    support_max = ndimage.maximum_filter(support, size=(size, size, size), mode="nearest").astype(np.float32)
    support_fraction = ndimage.uniform_filter(
        (support >= float(score_threshold)).astype(np.float32),
        size=(size, size, size),
        mode="nearest",
    ).astype(np.float32)
    return support_max, support_fraction


def longest_true_run_per_column(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    current = np.zeros(values.shape[:2], dtype=np.int16)
    longest = np.zeros(values.shape[:2], dtype=np.int16)
    for time_idx in range(values.shape[2]):
        current = np.where(values[:, :, time_idx], current + 1, 0).astype(np.int16, copy=False)
        longest = np.maximum(longest, current)
    return longest


def local_layer_relative_position(
    yy: np.ndarray,
    xx: np.ndarray,
    time_ms: np.ndarray,
    surfaces: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    t4 = surfaces["T4_TIME"][yy, xx].astype(np.float64)
    t6 = surfaces["T6_TIME"][yy, xx].astype(np.float64)
    t7 = surfaces["T7_TIME"][yy, xx].astype(np.float64)
    shasan = surfaces["ShasanPresent"][yy, xx].astype(bool) & (time_ms >= t4) & (time_ms <= t6)
    shasi = surfaces["ShasiPresent"][yy, xx].astype(bool) & (time_ms >= t6) & (time_ms <= t7)
    layer_code = np.zeros(len(time_ms), dtype=np.uint8)
    layer_code[shasan] = 1
    layer_code[shasi] = 2
    relative = np.full(len(time_ms), np.nan, dtype=np.float64)
    relative[shasan] = (time_ms[shasan] - t4[shasan]) / np.maximum(t6[shasan] - t4[shasan], 1.0e-6)
    relative[shasi] = (time_ms[shasi] - t6[shasi]) / np.maximum(t7[shasi] - t6[shasi], 1.0e-6)
    return layer_code, relative


def layer_relative_stats(
    yy: np.ndarray,
    xx: np.ndarray,
    time_ms: np.ndarray,
    surfaces: dict[str, np.ndarray],
) -> dict[str, Any]:
    layer_code, relative = local_layer_relative_position(yy, xx, time_ms, surfaces)
    counts = np.bincount(layer_code, minlength=3)
    dominant_code = int(np.argmax(counts[1:]) + 1) if int(counts[1:].sum()) else 0
    dominant_mask = layer_code == dominant_code
    dominant_values = relative[dominant_mask & np.isfinite(relative)]
    dominant_fraction = float(dominant_mask.mean()) if len(layer_code) else 0.0
    if len(dominant_values):
        relative_std = float(np.std(dominant_values))
        relative_span = float(np.percentile(dominant_values, 95) - np.percentile(dominant_values, 5))
    else:
        relative_std = float("nan")
        relative_span = float("nan")
    return {
        "dominant_layer": {0: "unknown", 1: "沙三段", 2: "沙四段"}[dominant_code],
        "dominant_layer_fraction": dominant_fraction,
        "relative_position_std": relative_std,
        "relative_position_span": relative_span,
    }


def filter_faultlike_evidence_grid(
    candidate_grid: np.ndarray,
    support_grid: np.ndarray,
    score_grid: np.ndarray,
    samples: np.ndarray,
    args: argparse.Namespace,
    *,
    require_attribute_support: bool = True,
    min_vertical_extent_ms: float | None = None,
    continuity_mode: str = "vertical_column",
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(candidate_grid, dtype=bool)
    if str(args.faultlike_filter_mode) == "none":
        return raw.copy(), {
            "mode": "none",
            "raw_voxel_count": int(raw.sum()),
            "filtered_voxel_count": int(raw.sum()),
            "filtered_voxel_fraction_of_raw": 1.0 if int(raw.sum()) else 0.0,
        }

    radius = max(int(args.faultlike_xy_radius_cells), 0)
    size = 2 * radius + 1
    local_candidate = ndimage.maximum_filter(raw.astype(np.uint8), size=(size, size, 1), mode="nearest") > 0
    local_support = ndimage.maximum_filter(
        (support_grid >= float(args.min_ant_or_curv_support)).astype(np.uint8),
        size=(size, size, 1),
        mode="nearest",
    ) > 0
    column_hits = local_candidate.sum(axis=2)
    longest_column_run = longest_true_run_per_column(local_candidate)
    nt = raw.shape[2]
    dt = float(np.median(np.diff(samples))) if len(samples) > 1 else 1.0
    required_extent_ms = (
        float(args.faultlike_min_vertical_extent_ms)
        if min_vertical_extent_ms is None
        else float(min_vertical_extent_ms)
    )
    min_hits_from_extent = int(np.ceil(required_extent_ms / max(abs(dt), 1.0e-6))) + 1
    min_hits = max(int(args.faultlike_min_column_hits), min_hits_from_extent)
    vertical_column = longest_column_run >= min_hits
    vertical_grid = np.repeat(vertical_column[:, :, None], nt, axis=2)

    slice_fraction = raw.mean(axis=(0, 1))
    non_layer_slice = slice_fraction <= float(args.faultlike_max_horizontal_slice_fraction)
    non_layer_grid = np.repeat(non_layer_slice[None, None, :], raw.shape[0], axis=0)
    non_layer_grid = np.repeat(non_layer_grid, raw.shape[1], axis=1)

    if continuity_mode == "connected_3d":
        exact = raw & non_layer_grid
        labels, count = ndimage.label(exact, structure=np.ones((3, 3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())
        keep_ids = np.where(sizes >= int(args.min_component_voxels))[0]
        keep_ids = keep_ids[keep_ids > 0]
        filtered = np.isin(labels, keep_ids)
        raw_count = int(raw.sum())
        filtered_count = int(filtered.sum())
        return filtered, {
            "mode": "connected_3d",
            "candidate_dilation_iterations": 0,
            "connectivity": 26,
            "require_attribute_support": False,
            "raw_voxel_count": raw_count,
            "exact_prefilter_voxel_count": int(exact.sum()),
            "raw_component_count_after_prefilter": int(count),
            "kept_component_count_after_size_filter": int(len(keep_ids)),
            "filtered_voxel_count": filtered_count,
            "filtered_voxel_fraction_of_raw": float(filtered_count / raw_count) if raw_count else 0.0,
            "filtered_score_stats": finite_stats(score_grid[filtered]) if filtered_count else finite_stats([]),
            "filtered_support_stats": finite_stats(support_grid[filtered]) if filtered_count else finite_stats([]),
        }

    support_gate = local_support if require_attribute_support else np.ones_like(raw, dtype=bool)
    filtered = raw & vertical_grid & support_gate & non_layer_grid
    if filtered.any():
        labels, count = ndimage.label(filtered, structure=np.ones((3, 3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())
        keep_ids = np.where(sizes >= int(args.min_component_voxels))[0]
        keep_ids = keep_ids[keep_ids > 0]
        filtered = np.isin(labels, keep_ids)
    else:
        count = 0
        keep_ids = np.asarray([], dtype=np.int64)

    raw_count = int(raw.sum())
    filtered_count = int(filtered.sum())
    summary = {
        "mode": "vertical_continuity",
        "xy_radius_cells": int(radius),
        "min_column_hits": int(min_hits),
        "min_vertical_extent_ms": required_extent_ms,
        "require_attribute_support": bool(require_attribute_support),
        "max_horizontal_slice_fraction": float(args.faultlike_max_horizontal_slice_fraction),
        "raw_voxel_count": raw_count,
        "vertical_column_count": int(vertical_column.sum()),
        "column_total_hit_stats": finite_stats(column_hits),
        "column_longest_contiguous_run_stats": finite_stats(longest_column_run),
        "column_rejected_total_hits_without_contiguous_run_count": int(((column_hits >= min_hits) & ~vertical_column).sum()),
        "removed_by_vertical_or_support_or_layer_filter": int(raw_count - int((raw & vertical_grid & support_gate & non_layer_grid).sum())),
        "raw_component_count_after_prefilter": int(count),
        "kept_component_count_after_size_filter": int(len(keep_ids)),
        "filtered_voxel_count": filtered_count,
        "filtered_voxel_fraction_of_raw": float(filtered_count / raw_count) if raw_count else 0.0,
        "filtered_score_stats": finite_stats(score_grid[filtered]) if filtered_count else finite_stats([]),
        "filtered_support_stats": finite_stats(support_grid[filtered]) if filtered_count else finite_stats([]),
    }
    return filtered, summary


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
    component_slices = ndimage.find_objects(labels, max_label=count)
    next_id = 1
    for component_id in range(1, count + 1):
        voxel_count = int(sizes[component_id])
        if voxel_count <= 0:
            continue
        component_slice = component_slices[component_id - 1]
        if component_slice is None:
            continue
        local_labels = labels[component_slice]
        yy, xx, tt = np.where(local_labels == component_id)
        yy += int(component_slice[0].start)
        xx += int(component_slice[1].start)
        tt += int(component_slice[2].start)
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


def split_inliers_into_local_panels(
    points_scaled: np.ndarray,
    target_length_m: float,
    min_voxels: int,
) -> list[np.ndarray]:
    """Split one fitted sheet along strike so Step7C receives local evidence panels."""
    points = np.asarray(points_scaled, dtype=np.float64)
    if len(points) < max(int(min_voxels), 3):
        return []
    axis1, _, _, _, _, _ = plane_axes_from_points(points)
    projection = (points - points.mean(axis=0, keepdims=True)) @ axis1
    span = float(projection.max() - projection.min())
    panel_count = max(int(np.ceil(span / max(float(target_length_m), 1.0))), 1)
    if panel_count == 1:
        return [np.arange(len(points), dtype=np.int64)]
    edges = np.linspace(float(projection.min()), float(projection.max()) + 1.0e-9, panel_count + 1)
    groups = [np.where((projection >= edges[idx]) & (projection < edges[idx + 1]))[0] for idx in range(panel_count)]
    groups = [group for group in groups if len(group)]
    merged: list[np.ndarray] = []
    pending: np.ndarray | None = None
    for group in groups:
        if pending is not None:
            group = np.concatenate([pending, group])
            pending = None
        if len(group) < int(min_voxels):
            pending = group
        else:
            merged.append(group)
    if pending is not None:
        if merged:
            merged[-1] = np.concatenate([merged[-1], pending])
        elif len(pending) >= int(min_voxels):
            merged.append(pending)
    return merged


def extract_inferred_fault_surfaces(
    score_grid: np.ndarray,
    lowcoh_grid: np.ndarray,
    anttrack_grid: np.ndarray,
    curvature_grid: np.ndarray,
    candidate_grid: np.ndarray,
    support_grid: np.ndarray,
    support_fraction_grid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    surfaces: dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(candidate_grid, structure=structure)
    x_values, y_values = grid_axis_values(mapping)
    surface_id_grid = np.zeros(candidate_grid.shape, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    surface_point_sets: list[dict[str, Any]] = []
    time_scale = float(args.orientation_time_scale_m_per_ms)
    sizes = np.bincount(labels.ravel())
    component_slices = ndimage.find_objects(labels, max_label=count)
    eligible_ids = [idx for idx in range(1, count + 1) if int(sizes[idx]) >= int(args.min_component_voxels)]
    eligible_ids.sort(key=lambda idx: int(sizes[idx]), reverse=True)
    max_raw_components = int(args.surface_ransac_max_raw_components)
    tile_cells = max(int(args.surface_component_tile_cells), 1)
    spatial_ids: list[int] = []
    seen_tiles: set[tuple[int, int]] = set()
    for raw_id in eligible_ids:
        component_slice = component_slices[raw_id - 1]
        if component_slice is None:
            continue
        tile_x = int((component_slice[1].start + component_slice[1].stop) // 2) // tile_cells
        tile_y = int((component_slice[0].start + component_slice[0].stop) // 2) // tile_cells
        tile_key = (tile_x, tile_y)
        if tile_key in seen_tiles:
            continue
        spatial_ids.append(raw_id)
        seen_tiles.add(tile_key)
        if len(spatial_ids) >= max_raw_components:
            break
    if len(spatial_ids) < max_raw_components:
        for raw_id in eligible_ids:
            if raw_id in spatial_ids:
                continue
            spatial_ids.append(raw_id)
            if len(spatial_ids) >= max_raw_components:
                break
    raw_ids = spatial_ids

    rejected_small = 0
    rejected_low_support = 0
    rejected_thin = 0
    rejected_short_xy = 0
    rejected_low_dip = 0
    rejected_layer_like = 0
    rejected_strat_following = 0
    rejected_low_inlier = 0
    rejected_too_broad = 0
    rejected_low_planarity = 0
    next_surface_id = 1

    for raw_component_id in raw_ids:
        component_slice = component_slices[raw_component_id - 1]
        if component_slice is None:
            continue
        local_labels = labels[component_slice]
        yy, xx, tt = np.where(local_labels == raw_component_id)
        yy += int(component_slice[0].start)
        xx += int(component_slice[1].start)
        tt += int(component_slice[2].start)
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
            panel_groups = split_inliers_into_local_panels(
                inlier_points_scaled,
                target_length_m=float(args.surface_panel_target_length_m),
                min_voxels=int(args.surface_min_panel_voxels),
            )
            chain_id = f"raw_{raw_component_id:05d}_sheet_{surface_ord + 1:03d}"
            for panel_ordinal, panel_idx in enumerate(panel_groups, start=1):
                pyy, pxx, ptt = iyy[panel_idx], ixx[panel_idx], itt[panel_idx]
                panel_unscaled = inlier_points_unscaled[panel_idx]
                panel_scaled = inlier_points_scaled[panel_idx]
                x_extent = float(panel_unscaled[:, 0].max() - panel_unscaled[:, 0].min())
                y_extent = float(panel_unscaled[:, 1].max() - panel_unscaled[:, 1].min())
                t_extent = float(panel_unscaled[:, 2].max() - panel_unscaled[:, 2].min())
                horizontal_extent = max(x_extent, y_extent)
                if horizontal_extent > float(args.surface_max_horizontal_extent_m) or t_extent > float(args.surface_max_time_extent_ms):
                    rejected_too_broad += 1
                    continue
                if t_extent < float(args.min_vertical_extent_ms):
                    rejected_thin += 1
                    continue
                if horizontal_extent < float(args.min_horizontal_extent_m):
                    rejected_short_xy += 1
                    continue
                vertices, geom = surface_vertices_from_inliers(
                    panel_unscaled,
                    panel_scaled,
                    time_scale=time_scale,
                    max_length_m=float(args.surface_max_panel_length_m),
                    max_height_ms=float(args.surface_max_panel_height_ms),
                )
                if geom["length_m"] < float(args.surface_min_panel_length_m):
                    rejected_short_xy += 1
                    continue
                if geom["planarity"] < float(args.surface_min_planarity):
                    rejected_low_planarity += 1
                    continue
                support_values = np.maximum(anttrack_grid[pyy, pxx, ptt], curvature_grid[pyy, pxx, ptt])
                support_fraction = float(np.mean(support_values >= float(args.surface_support_score_threshold)))
                local_support_fraction = float(np.mean(support_fraction_grid[pyy, pxx, ptt]))
                lowcoh_mean = float(np.mean(lowcoh_grid[pyy, pxx, ptt]))
                if support_fraction < float(args.surface_min_support_fraction) or lowcoh_mean < float(args.surface_min_lowcoh_mean):
                    rejected_low_support += 1
                    continue
                relative_stats = layer_relative_stats(pyy, pxx, panel_unscaled[:, 2], surfaces)
                strat_following = bool(
                    horizontal_extent >= float(args.strat_following_min_horizontal_extent_m)
                    and relative_stats["dominant_layer_fraction"] >= float(args.strat_following_min_dominant_layer_fraction)
                    and np.isfinite(relative_stats["relative_position_std"])
                    and relative_stats["relative_position_std"] <= float(args.strat_following_max_relative_position_std)
                    and relative_stats["relative_position_span"] <= float(args.strat_following_max_relative_position_span)
                )
                if strat_following:
                    rejected_strat_following += 1
                    continue
                layer_like = bool(
                    geom["dip_deg"] <= float(args.layer_like_max_dip_deg)
                    and horizontal_extent >= float(args.layer_like_min_horizontal_extent_m)
                    and t_extent <= float(args.layer_like_max_time_extent_ms)
                )
                if layer_like:
                    rejected_layer_like += 1
                    continue
                if geom["dip_deg"] < float(args.min_dip_deg):
                    rejected_low_dip += 1
                    continue
                support_mean = float(np.mean(support_values))
                support_max = float(np.max(support_values))
                _, _, panel_normal, _, _, _ = plane_axes_from_points(panel_scaled)
                panel_center_scaled = panel_scaled.mean(axis=0, keepdims=True)
                evidence_distances = np.abs((panel_scaled - panel_center_scaled) @ panel_normal)
                surface_id_grid[pyy, pxx, ptt] = next_surface_id
                row = {
                    "component_id": int(next_surface_id),
                    "surface_id": int(next_surface_id),
                    "panel_chain_id": chain_id,
                    "panel_ordinal_in_chain": int(panel_ordinal),
                    "panel_count_in_chain": int(len(panel_groups)),
                    "raw_component_id": int(raw_component_id),
                    "raw_component_voxel_count": int(sizes[raw_component_id]),
                    "surface_ordinal_in_raw_component": int(surface_ord + 1),
                    "voxel_count": int(len(panel_idx)),
                    "inlier_fraction_of_remaining": float(inlier_count / max(len(active_idx), 1)),
                    "score_mean": float(np.mean(score_grid[pyy, pxx, ptt])),
                    "score_max": float(np.max(score_grid[pyy, pxx, ptt])),
                    "lowcoh_mean": lowcoh_mean,
                    "anttrack_mean": float(np.mean(anttrack_grid[pyy, pxx, ptt])),
                    "curvature_mean": float(np.mean(curvature_grid[pyy, pxx, ptt])),
                    "support_mean": support_mean,
                    "support_max": support_max,
                    "auxiliary_support_fraction": support_fraction,
                    "local_auxiliary_support_fraction_mean": local_support_fraction,
                    "evidence_distance_mean_m": float(np.mean(evidence_distances)),
                    "evidence_distance_max_m": float(np.max(evidence_distances)),
                    "candidate_branch": "lowcoherence_weighted_local_sheet",
                    "x_min": geom["x_min"], "x_max": geom["x_max"],
                    "y_min": geom["y_min"], "y_max": geom["y_max"],
                    "time_min_ms": geom["time_min_ms"], "time_max_ms": geom["time_max_ms"],
                    "x_extent_m": x_extent, "y_extent_m": y_extent, "time_extent_ms": t_extent,
                    "center_x": geom["center_x"], "center_y": geom["center_y"], "center_time_ms": geom["center_time_ms"],
                    "surface_length_m": geom["length_m"], "surface_height_time_ms": geom["height_time_ms"],
                    "surface_area_m2": geom["area_m2"], "pca_azimuth_deg": geom["azimuth_deg"],
                    "pca_dip_deg": geom["dip_deg"], "pca_linearity": np.nan,
                    "surface_planarity": geom["planarity"], "layer_like": layer_like,
                    "strat_following": strat_following, **relative_stats,
                }
                for vertex_idx in range(1, 5):
                    row[f"V{vertex_idx}X"] = float(vertices[vertex_idx - 1, 0])
                    row[f"V{vertex_idx}Y"] = float(vertices[vertex_idx - 1, 1])
                    row[f"V{vertex_idx}Z"] = float(vertices[vertex_idx - 1, 2])
                rows.append(row)
                surface_point_sets.append(
                    {
                        "surface_id": int(next_surface_id), "raw_component_id": int(raw_component_id),
                        "points_unscaled": panel_unscaled.copy(), "points_scaled": panel_scaled.copy(),
                        "score": score_grid[pyy, pxx, ptt].astype(np.float32).copy(),
                        "support": support_values.astype(np.float32).copy(),
                        "candidate_branch": "lowcoherence_weighted_local_sheet",
                        "azimuth_deg": float(geom["azimuth_deg"]), "dip_deg": float(geom["dip_deg"]),
                        "surface_area_m2": float(geom["area_m2"]), "surface_planarity": float(geom["planarity"]),
                    }
                )
                next_surface_id += 1
            remaining[active_idx[inliers_local]] = False

    rows_by_chain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_chain.setdefault(str(row["panel_chain_id"]), []).append(row)
    for chain_rows in rows_by_chain.values():
        chain_rows.sort(key=lambda item: int(item["panel_ordinal_in_chain"]))
        for ordinal, row in enumerate(chain_rows, start=1):
            row["panel_ordinal_in_chain"] = ordinal
            row["panel_count_in_chain"] = len(chain_rows)

    summary = {
        "raw_component_count": int(count),
        "candidate_raw_component_count": int(len(raw_ids)),
        "spatially_distributed_raw_component_count": int(len(raw_ids)),
        "kept_component_count": int(next_surface_id - 1),
        "kept_surface_count": int(next_surface_id - 1),
        "rejected_small_component_count": int(rejected_small),
        "rejected_low_support_surface_count": int(rejected_low_support),
        "rejected_thin_surface_count": int(rejected_thin),
        "rejected_short_xy_surface_count": int(rejected_short_xy),
        "rejected_layer_like_surface_count": int(rejected_layer_like),
        "rejected_strat_following_surface_count": int(rejected_strat_following),
        "rejected_low_dip_surface_count": int(rejected_low_dip),
        "rejected_low_inlier_surface_count": int(rejected_low_inlier),
        "rejected_too_broad_surface_count": int(rejected_too_broad),
        "rejected_low_planarity_panel_count": int(rejected_low_planarity),
        "surface_extraction_mode": "surface_ransac",
        "surface_ransac_iterations": int(args.surface_ransac_iterations),
        "surface_ransac_distance_m": float(args.surface_ransac_distance_m),
        "surface_ransac_max_raw_components": int(args.surface_ransac_max_raw_components),
        "spatial_component_tile_cells": int(tile_cells),
        "surface_ransac_max_surfaces_per_component": int(args.surface_ransac_max_surfaces_per_component),
        "surface_max_horizontal_extent_m": float(args.surface_max_horizontal_extent_m),
        "surface_max_time_extent_ms": float(args.surface_max_time_extent_ms),
        "strat_following_min_horizontal_extent_m": float(args.strat_following_min_horizontal_extent_m),
        "strat_following_max_relative_position_std": float(args.strat_following_max_relative_position_std),
        "strat_following_max_relative_position_span": float(args.strat_following_max_relative_position_span),
        "strat_following_min_dominant_layer_fraction": float(args.strat_following_min_dominant_layer_fraction),
        "orientation_time_scale_m_per_ms": time_scale,
    }
    return surface_id_grid, pd.DataFrame(rows), summary, surface_point_sets


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


def write_irregular_surface_candidate_vtk(
    path: Path,
    surface_point_sets: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not surface_point_sets:
        pv.PolyData().save(path)
        return {"surface_count": 0, "triangle_count": 0, "point_count": 0}
    all_points: list[np.ndarray] = []
    all_faces: list[list[int]] = []
    cell_data: dict[str, list[float | int]] = {
        "SurfaceID": [],
        "RawComponentID": [],
        "ScoreMean": [],
        "SupportMean": [],
        "AzimuthDeg": [],
        "DipDeg": [],
        "SurfaceAreaM2": [],
        "SurfacePlanarity": [],
    }
    max_points = int(args.irregular_surface_max_points)
    max_edge = float(args.irregular_surface_max_edge_m)
    dropped_long_edge = 0
    dropped_degenerate = 0
    for surface in surface_point_sets:
        points = np.asarray(surface["points_unscaled"], dtype=np.float64)
        points_scaled = np.asarray(surface["points_scaled"], dtype=np.float64)
        if len(points) < 3:
            continue
        if len(points) > max_points:
            keep = rng.choice(np.arange(len(points)), size=max_points, replace=False)
            points = points[keep]
            points_scaled = points_scaled[keep]
        center = points_scaled.mean(axis=0)
        axis1, axis2, _normal, _azimuth, _dip, _planarity = plane_axes_from_points(points_scaled)
        local = np.column_stack([(points_scaled - center) @ axis1, (points_scaled - center) @ axis2])
        try:
            tri = Delaunay(local)
        except Exception:
            continue
        base = sum(len(item) for item in all_points)
        all_points.append(points.astype(np.float32))
        score_mean = float(np.mean(surface["score"])) if len(surface["score"]) else 0.0
        support_mean = float(np.mean(surface["support"])) if len(surface["support"]) else 0.0
        for simplex in tri.simplices:
            tri_local = local[simplex]
            edges = [
                float(np.linalg.norm(tri_local[0] - tri_local[1])),
                float(np.linalg.norm(tri_local[1] - tri_local[2])),
                float(np.linalg.norm(tri_local[2] - tri_local[0])),
            ]
            edge_a = tri_local[1] - tri_local[0]
            edge_b = tri_local[2] - tri_local[0]
            area = 0.5 * float(abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))
            if area <= 1.0e-6:
                dropped_degenerate += 1
                continue
            if max(edges) > max_edge:
                dropped_long_edge += 1
                continue
            all_faces.append([3, base + int(simplex[0]), base + int(simplex[1]), base + int(simplex[2])])
            cell_data["SurfaceID"].append(int(surface["surface_id"]))
            cell_data["RawComponentID"].append(int(surface["raw_component_id"]))
            cell_data["ScoreMean"].append(score_mean)
            cell_data["SupportMean"].append(support_mean)
            cell_data["AzimuthDeg"].append(float(surface["azimuth_deg"]))
            cell_data["DipDeg"].append(float(surface["dip_deg"]))
            cell_data["SurfaceAreaM2"].append(float(surface["surface_area_m2"]))
            cell_data["SurfacePlanarity"].append(float(surface["surface_planarity"]))
    if not all_points or not all_faces:
        pv.PolyData().save(path)
        return {
            "surface_count": int(len(surface_point_sets)),
            "triangle_count": 0,
            "point_count": int(sum(len(item) for item in all_points)),
            "dropped_long_edge_triangle_count": int(dropped_long_edge),
            "dropped_degenerate_triangle_count": int(dropped_degenerate),
        }
    points_arr = np.vstack(all_points).astype(np.float32)
    faces_arr = np.asarray(all_faces, dtype=np.int64).ravel()
    mesh = pv.PolyData(points_arr, faces_arr)
    for name, values in cell_data.items():
        dtype = np.int32 if name in {"SurfaceID", "RawComponentID"} else np.float32
        mesh.cell_data[name] = np.asarray(values, dtype=dtype)
    mesh.save(path)
    return {
        "surface_count": int(len(surface_point_sets)),
        "triangle_count": int(mesh.n_cells),
        "point_count": int(mesh.n_points),
        "dropped_long_edge_triangle_count": int(dropped_long_edge),
        "dropped_degenerate_triangle_count": int(dropped_degenerate),
        "max_points_per_surface": int(max_points),
        "max_edge_m": float(max_edge),
    }


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
    source_samples, density_load = read_sgy_sample_axis(input_density_sgy)
    interval_ms = float(dict(config.get("large_evidence", {})).get("sample_interval_ms", 10.0))
    samples = regular_sample_axis(source_samples, interval_ms)
    density_load["target_sample_count"] = int(len(samples))
    density_load["target_sample_interval_ms"] = interval_ms
    horizon_contract = load_contract_for_mapping(config, mapping)
    horizon_axis_qc = validate_window_contract(config, horizon_contract, samples)
    horizon_surfaces = surface_grids_from_contract(mapping, horizon_contract)

    selected_faults = load_fault_overlap(args.input_qc_dir.resolve())
    selected_faults = selected_faults[selected_faults["intersects_demo_xy"].astype(bool)].copy()
    original_vtk_summary = merge_fault_vtps(
        selected_faults,
        args.fault_patch_root.resolve(),
        output_dir / "original_fault_units_demo_raw_time.vtk",
        output_dir / "original_fault_unit_manifest.csv",
    )
    original_fault_grid, original_fault_mask, fault_audit = rasterize_original_faults(
        selected_faults,
        args.fault_patch_root.resolve(),
        mapping,
        samples,
        args,
    )
    original_fault_flat = grid_to_flat(original_fault_grid.astype(np.float32), mapping)
    original_fault_mask_flat = grid_to_flat(original_fault_mask.astype(np.uint8), mapping).astype(bool)
    original_fault_horizon_qc = apply_window_inplace(
        original_fault_flat, horizon_contract, samples, fill_value=0.0
    )
    apply_window_inplace(original_fault_mask_flat, horizon_contract, samples, fill_value=False)
    original_fault_grid, _, _ = flat_to_grid(original_fault_flat, mapping)
    original_fault_mask_grid, _, _ = flat_to_grid(original_fault_mask_flat.astype(np.float32), mapping)
    original_fault_mask = original_fault_mask_grid > 0.5
    fault_audit.to_csv(output_dir / "original_fault_rasterization_audit.csv", index=False, encoding="utf-8-sig")

    print("[step6c-large] loading seismic attributes", flush=True)
    coherence, _, coh_load = load_trace_matrix(volume_paths["Coherence"], source_trace_idx, samples, "Coherence")
    anttrack, _, ant_load = load_trace_matrix(volume_paths["AntTrack"], source_trace_idx, samples, "AntTrack")
    curvmax, _, curvmax_load = load_trace_matrix(volume_paths["CurvatureMax"], source_trace_idx, samples, "CurvatureMax")
    if "CurvaturePos" in volume_paths:
        curvpos, _, curvpos_load = load_trace_matrix(volume_paths["CurvaturePos"], source_trace_idx, samples, "CurvaturePos")
    else:
        curvpos = None
        curvpos_load = {"status": "not_configured"}

    score_cfg = dict(config.get("score_config", {}))
    coh_cfg = dict(score_cfg.get("coherence_score", {"valid_min": 0.0, "low_quantile": 0.05, "high_quantile": 0.95}))
    ant_cfg = dict(score_cfg.get("anttrack_score", {"low_quantile": 0.20, "high_quantile": 0.96}))
    curvmax_cfg = dict(score_cfg.get("curvaturemax_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    curvpos_cfg = dict(score_cfg.get("curvaturepos_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    coh_valid = valid_values(coherence, coh_cfg)
    ant_valid = valid_values(anttrack, ant_cfg)
    curvmax_valid = valid_values(curvmax, curvmax_cfg)
    curvpos_valid = valid_values(curvpos, curvpos_cfg) if curvpos is not None else None
    horizon_mask_qc = apply_validity_inplace(coh_valid, horizon_contract, samples)
    apply_validity_inplace(ant_valid, horizon_contract, samples)
    apply_validity_inplace(curvmax_valid, horizon_contract, samples)
    if curvpos_valid is not None:
        apply_validity_inplace(curvpos_valid, horizon_contract, samples)
    lowcoh_score, lowcoh_summary = low_score(coherence, coh_valid, coh_cfg)
    ant_score, ant_summary = high_score(anttrack, ant_valid, ant_cfg)
    curvmax_score, curvmax_summary = high_score(curvmax, curvmax_valid, curvmax_cfg)
    if curvpos is not None and curvpos_valid is not None:
        curvpos_score, curvpos_summary = high_score(curvpos, curvpos_valid, curvpos_cfg)
        curv_score = np.maximum(curvmax_score, curvpos_score).astype(np.float32)
    else:
        curvpos_summary = {"status": "not_configured"}
        curv_score = curvmax_score.astype(np.float32)
    ant_grid, _, _ = flat_to_grid(ant_score, mapping)
    curv_grid, _, _ = flat_to_grid(curv_score, mapping)
    support_grid, support_fraction_grid = build_local_support_grids(
        ant_grid,
        curv_grid,
        int(args.support_neighborhood_cells),
        float(args.surface_support_score_threshold),
    )
    weights = np.asarray(
        [float(args.lowcoh_weight), float(args.anttrack_weight), float(args.curvature_weight)],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Step6C evidence weights must be non-negative and have a positive sum")
    weights /= float(weights.sum())
    large_score = weights[0] * lowcoh_score + weights[1] * ant_score + weights[2] * curv_score
    large_score[~coh_valid] = 0.0
    large_score = np.clip(large_score, 0.0, 1.0).astype(np.float32)
    lowcoh_cut = float(np.quantile(lowcoh_score[coh_valid], 1.0 - float(args.coherence_low_quantile)))
    large_cut = float(args.large_score_threshold)
    effective_lowcoh_cut = max(float(args.lowcoh_candidate_floor), lowcoh_cut)
    candidate_flat = (lowcoh_score >= effective_lowcoh_cut) & (large_score >= large_cut) & coh_valid
    lowcoh_grid, _, _ = flat_to_grid(lowcoh_score, mapping)

    score_grid, _, _ = flat_to_grid(large_score, mapping)
    candidate_grid, _, _ = flat_to_grid(candidate_flat.astype(np.float32), mapping)
    candidate_grid = candidate_grid > 0.5
    write_intermediate_vtk = bool(config.get("write_intermediate_vtk", False))
    evidence_vtk_summary = (
        write_candidate_evidence_vtk(
            output_dir / "inferred_fault_evidence_points_raw_time.vtk",
            candidate_grid,
            score_grid,
            support_grid,
            mapping,
            samples,
            int(args.vtk_max_points),
            rng,
        )
        if write_intermediate_vtk
        else {"status": "disabled_by_config"}
    )
    faultlike_candidate_grid, faultlike_filter_summary = filter_faultlike_evidence_grid(
        candidate_grid,
        support_grid,
        score_grid,
        samples,
        args,
        require_attribute_support=False,
        continuity_mode="connected_3d",
    )
    faultlike_branch_code_grid = np.zeros_like(faultlike_candidate_grid, dtype=np.uint8)
    faultlike_branch_code_grid[faultlike_candidate_grid] = 1
    faultlike_evidence_vtk_summary = (
        write_candidate_evidence_vtk(
            output_dir / "inferred_faultlike_evidence_points_raw_time.vtk",
            faultlike_candidate_grid,
            score_grid,
            support_grid,
            mapping,
            samples,
            int(args.vtk_max_points),
            rng,
        )
        if write_intermediate_vtk
        else {"status": "disabled_by_config"}
    )
    inferred_surface_vtk_summary: dict[str, Any] = {"surface_count": 0, "polygon_count": 0}
    inferred_irregular_surface_vtk_summary: dict[str, Any] = {"surface_count": 0, "triangle_count": 0}
    if args.inferred_extraction_mode == "surface_ransac":
        inferred_id_grid, inferred_df, inferred_summary, surface_point_sets = extract_inferred_fault_surfaces(
            score_grid,
            lowcoh_grid,
            ant_grid,
            curv_grid,
            faultlike_candidate_grid,
            support_grid,
            support_fraction_grid,
            mapping,
            samples,
            horizon_surfaces,
            args,
            rng,
        )
        if write_intermediate_vtk:
            inferred_surface_vtk_summary = write_surface_candidate_vtk(
                output_dir / "inferred_fault_surface_candidates_raw_time.vtk",
                inferred_df,
            )
            inferred_irregular_surface_vtk_summary = write_irregular_surface_candidate_vtk(
                output_dir / "inferred_fault_surface_candidates_irregular_raw_time.vtk",
                surface_point_sets,
                args,
                rng,
            )
        else:
            inferred_surface_vtk_summary = {"status": "disabled_by_config"}
            inferred_irregular_surface_vtk_summary = {"status": "disabled_by_config"}
    else:
        inferred_id_grid, inferred_df, inferred_summary = extract_inferred_faults(
            score_grid,
            faultlike_candidate_grid,
            support_grid,
            mapping,
            samples,
            args,
        )
        if write_intermediate_vtk:
            write_surface_candidate_vtk(output_dir / "inferred_fault_surface_candidates_raw_time.vtk", pd.DataFrame())
            pv.PolyData().save(output_dir / "inferred_fault_surface_candidates_irregular_raw_time.vtk")
    inferred_mask = inferred_id_grid > 0
    combined_prior_grid = np.maximum(original_fault_grid, score_grid * inferred_mask.astype(np.float32))
    combined_mask_grid = original_fault_mask | inferred_mask
    large_prior_flat = grid_to_flat(combined_prior_grid.astype(np.float32), mapping)
    large_mask_flat = grid_to_flat(combined_mask_grid.astype(np.float32), mapping)
    inferred_id_flat = grid_to_flat(inferred_id_grid.astype(np.float32), mapping).astype(np.int32)

    write_sgy_like(output_dir / "large_fault_prior.sgy", input_density_sgy, large_prior_flat, samples)
    inferred_df.to_csv(output_dir / "large_fault_component_summary.csv", index=False, encoding="utf-8-sig")
    inferred_vtk_summary = (
        write_component_vtk(
            output_dir / "inferred_large_fault_candidates_raw_time.vtk",
            inferred_id_grid,
            score_grid,
            mapping,
            samples,
            int(args.vtk_max_points),
            rng,
        )
        if write_intermediate_vtk
        else {"status": "disabled_by_config"}
    )
    np.savez_compressed(
        output_dir / "large_fault_prior_components.npz",
        large_prior=large_prior_flat.astype(np.float32),
        large_mask=large_mask_flat.astype(np.uint8),
        original_fault_prior=grid_to_flat(original_fault_grid.astype(np.float32), mapping),
        original_fault_mask=grid_to_flat(original_fault_mask.astype(np.float32), mapping).astype(np.uint8),
        inferred_fault_prior=grid_to_flat((score_grid * inferred_mask.astype(np.float32)).astype(np.float32), mapping),
        inferred_fault_mask=grid_to_flat(inferred_mask.astype(np.float32), mapping).astype(np.uint8),
        inferred_component_id=inferred_id_flat.astype(np.int32),
        inferred_candidate_branch_code=grid_to_flat(faultlike_branch_code_grid.astype(np.float32), mapping).astype(np.uint8),
        samples=samples.astype(np.float32),
        x=mapping["x"].astype(np.float64),
        y=mapping["y"].astype(np.float64),
        ix=mapping["ix"].astype(np.int32),
        iy=mapping["iy"].astype(np.int32),
        source_trace_idx=horizon_contract.trace_idx.astype(np.int32),
        t4_time=horizon_contract.t4.astype(np.float32),
        t6_time=horizon_contract.t6.astype(np.float32),
        t7_time=horizon_contract.t7.astype(np.float32),
        shasan_present=horizon_contract.shasan_present.astype(np.uint8),
        shasi_present=horizon_contract.shasi_present.astype(np.uint8),
    )

    summary = {
        "status": "pass",
        "input_density_sgy": str(input_density_sgy),
        "output_dir": str(output_dir),
        "sample_interval_ms": interval_ms,
        "sample_count": int(len(samples)),
        "horizon_contract": contract_summary(horizon_contract),
        "horizon_axis_qc": horizon_axis_qc,
        "horizon_mask_qc": horizon_mask_qc,
        "original_fault_horizon_mask_qc": original_fault_horizon_qc,
        "output_contract": {
            "large_prior_sgy": str(output_dir / "large_fault_prior.sgy"),
            "component_npz": str(output_dir / "large_fault_prior_components.npz"),
            "original_fault_surface_vtk": str(output_dir / "original_fault_units_demo_raw_time.vtk"),
            "original_fault_manifest_csv": str(output_dir / "original_fault_unit_manifest.csv"),
            "mask_storage": "original_fault_mask, inferred_fault_mask and combined large_mask inside component_npz",
        },
        "load": {
            "density": density_load,
            "coherence": coh_load,
            "anttrack": ant_load,
            "curvaturemax": curvmax_load,
            "curvaturepos": curvpos_load,
        },
        "original_fault": {
            "selected_patch_count": int(len(selected_faults)),
            "selection_contract": "demo_xy_intersection_only; source geometry is not time or horizon clipped",
            "influence_contract": "triangle-rasterized derivative is clipped to per-trace T4-T7 before Step6D",
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
                "large_candidate_quantile_deprecated": float(args.large_candidate_quantile),
                "lowcoh_weight": float(weights[0]),
                "anttrack_weight": float(weights[1]),
                "curvature_weight": float(weights[2]),
                "lowcoh_candidate_floor": float(args.lowcoh_candidate_floor),
                "large_score_threshold": float(args.large_score_threshold),
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
                "surface_support_score_threshold": float(args.surface_support_score_threshold),
                "surface_min_support_fraction": float(args.surface_min_support_fraction),
                "surface_min_lowcoh_mean": float(args.surface_min_lowcoh_mean),
                "surface_min_planarity": float(args.surface_min_planarity),
                "support_neighborhood_cells": int(args.support_neighborhood_cells),
                "surface_ransac_iterations": int(args.surface_ransac_iterations),
                "surface_ransac_distance_m": float(args.surface_ransac_distance_m),
                "surface_ransac_min_inlier_voxels": int(args.surface_ransac_min_inlier_voxels),
                "surface_ransac_min_inlier_fraction": float(args.surface_ransac_min_inlier_fraction),
                "surface_ransac_max_raw_components": int(args.surface_ransac_max_raw_components),
                "surface_component_tile_cells": int(args.surface_component_tile_cells),
                "candidate_dilation_iterations": 0,
                "deprecated_lowcoh_rescue_flag_ignored": bool(args.enable_lowcoh_structure_rescue),
                "surface_panel_target_length_m": float(args.surface_panel_target_length_m),
                "surface_min_panel_length_m": float(args.surface_min_panel_length_m),
                "surface_max_panel_length_m": float(args.surface_max_panel_length_m),
                "surface_min_panel_voxels": int(args.surface_min_panel_voxels),
                "strat_following_min_horizontal_extent_m": float(args.strat_following_min_horizontal_extent_m),
                "strat_following_max_relative_position_std": float(args.strat_following_max_relative_position_std),
                "strat_following_max_relative_position_span": float(args.strat_following_max_relative_position_span),
                "strat_following_min_dominant_layer_fraction": float(args.strat_following_min_dominant_layer_fraction),
                "faultlike_filter_mode": str(args.faultlike_filter_mode),
                "faultlike_xy_radius_cells": int(args.faultlike_xy_radius_cells),
                "faultlike_min_vertical_extent_ms": float(args.faultlike_min_vertical_extent_ms),
                "faultlike_min_column_hits": int(args.faultlike_min_column_hits),
                "faultlike_max_horizontal_slice_fraction": float(args.faultlike_max_horizontal_slice_fraction),
                "irregular_surface_max_points": int(args.irregular_surface_max_points),
                "irregular_surface_max_edge_m": float(args.irregular_surface_max_edge_m),
            },
            "lowcoh_quantile_threshold": lowcoh_cut,
            "effective_lowcoh_score_threshold": effective_lowcoh_cut,
            "large_score_threshold": large_cut,
            "raw_candidate_voxel_count": int(candidate_flat.sum()),
            "raw_candidate_voxel_fraction": float(candidate_flat.mean()),
            "raw_candidate_branch_counts": {
                "lowcoherence_weighted_exact_candidate": int(candidate_flat.sum()),
            },
            "faultlike_filter": faultlike_filter_summary,
            "faultlike_candidate_voxel_count": int(faultlike_candidate_grid.sum()),
            "faultlike_candidate_voxel_fraction": float(faultlike_candidate_grid.mean()),
            "kept_voxel_count": int(inferred_mask.sum()),
            "kept_voxel_fraction": float(inferred_mask.mean()),
            "component_count": int(len(inferred_df)),
            "component_summary": inferred_summary,
            "component_voxel_stats": finite_stats(inferred_df["voxel_count"]) if len(inferred_df) else finite_stats([]),
            "component_dip_stats": finite_stats(inferred_df["pca_dip_deg"]) if len(inferred_df) else finite_stats([]),
            "component_time_extent_stats": finite_stats(inferred_df["time_extent_ms"]) if len(inferred_df) else finite_stats([]),
            "vtk": inferred_vtk_summary,
            "evidence_vtk": evidence_vtk_summary,
            "faultlike_evidence_vtk": faultlike_evidence_vtk_summary,
            "surface_vtk": inferred_surface_vtk_summary,
            "irregular_surface_vtk": inferred_irregular_surface_vtk_summary,
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
            "Step6C preserves original fault panels as hard prior and adds inferred faults only from exact, 26-connected, "
            "low-coherence-led sheet evidence. RANSAC sheets are split into locally supported panels before Step7C."
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
