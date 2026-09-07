from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[2]
LEGACY_STEP7B_DIR = CURRENT_DIR.parent / "step7b_initial_dfn_3d"
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_step7c_large_v1.json"
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))
if str(LEGACY_STEP7B_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_STEP7B_DIR))
import build_initial_dfn_from_3d_density_sgy as legacy  # noqa: E402
from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    HorizonSpatialLookup,
    build_spatial_lookup,
    load_contract_for_mapping,
    surface_grids_from_contract,
    validate_window_contract,
)
from common.unified_dfn_vtk import (  # noqa: E402
    geometry_fingerprint,
    unified_vtk_summary,
    write_unified_dfn_vtk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step7C large fault/fault-zone DFN from original fault sticks and Step6C prior.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def contract_layer_for_center(config: dict[str, Any], center: np.ndarray) -> str | None:
    lookup = config.get("_horizon_spatial_lookup")
    if not isinstance(lookup, HorizonSpatialLookup):
        raise RuntimeError("Step7C horizon spatial lookup is not initialized")
    return lookup.layer_group(float(center[0]), float(center[1]), float(center[2]))


def enforce_center_horizon_contract(
    frame: pd.DataFrame,
    lookup: HorizonSpatialLookup,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if frame.empty:
        return frame.copy(), {"input_count": 0, "kept_count": 0, "rejected_count": 0}
    rows: list[pd.Series] = []
    rejected = 0
    kept_center_outside_voxel_coverage = 0
    for _, source_row in frame.iterrows():
        row = source_row.copy()
        local = lookup.query(float(row["CenterX"]), float(row["CenterY"]))
        interval = lookup.interval(float(row["CenterX"]), float(row["CenterY"]), float(row["CenterTime"]))
        if interval is None:
            # Voxel-coverage validated panels: the panel body (>=90% of its
            # evidence voxels) lies inside per-trace T4-T7 windows even though
            # the panel centroid projects outside the window of its own trace.
            # Keep them so mine-scale display does not lose valid inferred
            # fault surfaces. LayerGroup was assigned from the dominant layer.
            if str(row.get("WindowValidationMode", "")).strip() == "voxel_coverage":
                row["CenterInWindow"] = 0
                kept_center_outside_voxel_coverage += 1
            else:
                rejected += 1
                continue
        else:
            row["CenterInWindow"] = 1
            row["LayerGroup"] = "沙三段" if interval == "T4->T6" else "沙四段"
            row["LayerCode"] = 1 if interval == "T4->T6" else 2
        row["SourceTraceIdx"] = int(local["TraceIdx"])
        row["LocalT4Time"] = float(local["T4"])
        row["LocalT6Time"] = float(local["T6"])
        row["LocalT7Time"] = float(local["T7"])
        row["LocalShasanPresent"] = int(local["ShasanPresent"])
        row["LocalShasiPresent"] = int(local["ShasiPresent"])
        rows.append(row)
    output = pd.DataFrame(rows, columns=list(frame.columns) + [
        name for name in [
            "SourceTraceIdx", "LocalT4Time", "LocalT6Time", "LocalT7Time", "CenterInWindow",
            "LocalShasanPresent", "LocalShasiPresent",
        ] if name not in frame.columns
    ])
    return output.reset_index(drop=True), {
        "input_count": int(len(frame)),
        "kept_count": int(len(output)),
        "rejected_count": int(rejected),
        "kept_center_outside_voxel_coverage_count": int(kept_center_outside_voxel_coverage),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read CSV: {path}") from last_error


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "original_surface_vtk": output_dir / "original_fault_units_demo_raw_time.vtk",
        "original_manifest_csv": output_dir / "original_fault_unit_manifest.csv",
        "lowcoh_csv": output_dir / "large_inferred_fault_surface_patches.csv",
        "lowcoh_vtk": output_dir / "large_inferred_fault_surfaces_raw_time.vtk",
        "duplicate_csv": output_dir / "large_inferred_known_fault_duplicates.csv",
        "duplicate_vtk": output_dir / "large_inferred_known_fault_duplicates_raw_time.vtk",
        "dfn_csv": output_dir / "large_fault_dfn_patches.csv",
        "raw_vtk": output_dir / "large_fault_dfn_raw_time.vtk",
        "audit_csv": output_dir / "large_generation_audit.csv",
        "summary_json": output_dir / "large_fault_dfn_summary.json",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vertex_columns() -> list[str]:
    return [f"V{idx}{axis}" for idx in range(1, 5) for axis in ("X", "Y", "Z")]


def has_vertex_columns(row: pd.Series) -> bool:
    return all(col in row.index and pd.notna(row.get(col)) for col in vertex_columns())


def vertices_from_row(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [[float(row[f"V{idx}X"]), float(row[f"V{idx}Y"]), float(row[f"V{idx}Z"])] for idx in range(1, 5)],
        dtype=float,
    )


def add_vertex_columns(row: dict[str, Any], vertices: np.ndarray) -> dict[str, Any]:
    verts = np.asarray(vertices, dtype=float)
    for idx in range(1, 5):
        row[f"V{idx}X"] = float(verts[idx - 1, 0])
        row[f"V{idx}Y"] = float(verts[idx - 1, 1])
        row[f"V{idx}Z"] = float(verts[idx - 1, 2])
    return row


def normalized_panel_normal(vertices: np.ndarray, time_scale_m_per_ms: float) -> np.ndarray | None:
    scaled = np.asarray(vertices, dtype=float).copy()
    scaled[:, 2] *= float(time_scale_m_per_ms)
    normal = np.cross(scaled[1] - scaled[0], scaled[3] - scaled[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-9:
        return None
    return normal / norm


def classify_inferred_panels_against_original(
    frame: pd.DataFrame,
    original_surface: Path,
    component_npz: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"input_count": 0, "formal_count": 0, "duplicate_count": 0}
    time_scale = float(config.get("original_fault_compare_time_scale_m_per_ms", 2.0))
    center_limit = float(config.get("original_fault_duplicate_center_distance_m", 60.0))
    vertex_limit = float(config.get("original_fault_duplicate_vertex_distance_m", 100.0))
    min_vertex_fraction = float(config.get("original_fault_duplicate_min_vertex_fraction", 0.75))
    max_orientation_difference = float(config.get("original_fault_duplicate_max_orientation_difference_deg", 25.0))
    branch_context_distance = float(config.get("original_fault_branch_context_distance_m", 120.0))

    npz = np.load(component_npz)
    required = {"original_fault_mask", "x", "y", "samples"}
    missing = sorted(required.difference(npz.files))
    if missing:
        raise ValueError(f"{component_npz} missing original-fault comparison arrays: {missing}")
    original_mask = npz["original_fault_mask"].astype(bool)
    trace_idx, time_idx = np.where(original_mask)
    if not len(trace_idx):
        raise RuntimeError("original fault mask is empty; cannot classify inferred fault panels")
    original_voxel_points = np.column_stack(
        [
            npz["x"][trace_idx].astype(float),
            npz["y"][trace_idx].astype(float),
            npz["samples"][time_idx].astype(float) * time_scale,
        ]
    )
    voxel_tree = cKDTree(original_voxel_points)

    original_mesh = pv.read(original_surface)
    if not isinstance(original_mesh, pv.PolyData):
        original_mesh = original_mesh.extract_surface(algorithm="dataset_surface")
    original_mesh = original_mesh.triangulate()
    faces = np.asarray(original_mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    mesh_points = np.asarray(original_mesh.points, dtype=float).copy()
    mesh_points[:, 2] *= time_scale
    triangles = mesh_points[faces]
    triangle_centers = triangles.mean(axis=1)
    triangle_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    triangle_norms = np.linalg.norm(triangle_normals, axis=1)
    valid_triangles = triangle_norms > 1.0e-9
    triangle_normals[valid_triangles] /= triangle_norms[valid_triangles, None]
    triangle_tree = cKDTree(triangle_centers)

    output = frame.copy().reset_index(drop=True)
    vertices_per_panel = [vertices_from_row(row) for _, row in output.iterrows()]
    centers = output[["CenterX", "CenterY", "CenterTime"]].to_numpy(dtype=float)
    centers_scaled = centers.copy()
    centers_scaled[:, 2] *= time_scale
    center_distances = voxel_tree.query(centers_scaled, k=1, workers=-1)[0]
    nearest_triangles = triangle_tree.query(centers_scaled, k=1, workers=-1)[1].astype(np.int64)

    relations: list[str] = []
    orientation_differences: list[float] = []
    vertex_near_fractions: list[float] = []
    vertex_min_distances: list[float] = []
    for row_idx, vertices in enumerate(vertices_per_panel):
        scaled_vertices = np.asarray(vertices, dtype=float).copy()
        scaled_vertices[:, 2] *= time_scale
        vertex_distances = voxel_tree.query(scaled_vertices, k=1, workers=-1)[0]
        vertex_fraction = float(np.mean(vertex_distances <= vertex_limit))
        vertex_min_distance = float(np.min(vertex_distances))
        panel_normal = normalized_panel_normal(vertices, time_scale)
        triangle_idx = int(nearest_triangles[row_idx])
        if panel_normal is None or not valid_triangles[triangle_idx]:
            orientation_difference = float("nan")
        else:
            cosine = float(np.clip(abs(np.dot(panel_normal, triangle_normals[triangle_idx])), 0.0, 1.0))
            orientation_difference = float(np.degrees(np.arccos(cosine)))
        duplicate = bool(
            float(center_distances[row_idx]) <= center_limit
            and np.isfinite(orientation_difference)
            and orientation_difference <= max_orientation_difference
            and vertex_fraction >= min_vertex_fraction
        )
        near_original = bool(
            float(center_distances[row_idx]) <= branch_context_distance
            or vertex_min_distance <= branch_context_distance
        )
        relations.append(
            "known_fault_duplicate"
            if duplicate
            else "known_fault_branch_or_splay"
            if near_original
            else "independent_inferred_fault"
        )
        orientation_differences.append(orientation_difference)
        vertex_near_fractions.append(vertex_fraction)
        vertex_min_distances.append(vertex_min_distance)

    output["OriginalFaultRelation"] = relations
    output["OriginalFaultCenterDistanceM"] = center_distances.astype(float)
    output["OriginalFaultVertexMinDistanceM"] = vertex_min_distances
    output["OriginalFaultVertexNearFraction"] = vertex_near_fractions
    output["OriginalFaultOrientationDifferenceDeg"] = orientation_differences
    output["OriginalFaultNearestTriangleID"] = nearest_triangles.astype(int)
    output["KeepInFormalDFN"] = output["OriginalFaultRelation"].ne("known_fault_duplicate").astype(int)
    counts = output["OriginalFaultRelation"].value_counts(dropna=False)
    return output, {
        "input_count": int(len(output)),
        "formal_count": int(output["KeepInFormalDFN"].sum()),
        "duplicate_count": int(output["OriginalFaultRelation"].eq("known_fault_duplicate").sum()),
        "relation_counts": {str(key): int(value) for key, value in counts.items()},
        "parameters": {
            "time_scale_m_per_ms": time_scale,
            "duplicate_center_distance_m": center_limit,
            "duplicate_vertex_distance_m": vertex_limit,
            "duplicate_min_vertex_fraction": min_vertex_fraction,
            "duplicate_max_orientation_difference_deg": max_orientation_difference,
            "branch_context_distance_m": branch_context_distance,
        },
        "center_distance_m": finite_stats(output["OriginalFaultCenterDistanceM"]),
        "orientation_difference_deg": finite_stats(output["OriginalFaultOrientationDifferenceDeg"]),
    }


def filter_target_block_centers(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    if frame.empty:
        return frame.copy(), {"input_count": 0, "kept_count": 0, "rejected_count": 0}
    block = dict(config.get("target_block", {}))
    required = {"x_min", "x_max", "y_min", "y_max"}
    if not required.issubset(block):
        return frame.reset_index(drop=True), {
            "input_count": int(len(frame)),
            "kept_count": int(len(frame)),
            "rejected_count": 0,
        }
    mask = (
        pd.to_numeric(frame["CenterX"], errors="coerce").between(float(block["x_min"]), float(block["x_max"]), inclusive="both")
        & pd.to_numeric(frame["CenterY"], errors="coerce").between(float(block["y_min"]), float(block["y_max"]), inclusive="both")
    )
    output = frame.loc[mask].copy().reset_index(drop=True)
    return output, {
        "input_count": int(len(frame)),
        "kept_count": int(len(output)),
        "rejected_count": int((~mask).sum()),
    }


def finite_stats(values: Any) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
    }


def infer_fault_cell_range(config: dict[str, Any]) -> dict[str, int]:
    overlap_path = Path(config["fault_patch_overlap_csv"]).resolve()
    if not overlap_path.exists():
        raise FileNotFoundError(f"fault_patch_overlap_csv not found: {overlap_path}")
    df = read_csv_flexible(overlap_path)
    required = {"cell_i", "cell_j"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{overlap_path} missing columns: {missing}")
    selected = df.copy()
    if "intersects_demo_xy_t" in selected.columns:
        selected = selected[selected["intersects_demo_xy_t"].astype(bool)].copy()
    if selected.empty and "intersects_demo_xy" in df.columns:
        selected = df[df["intersects_demo_xy"].astype(bool)].copy()
    if selected.empty:
        raise ValueError(f"no segmented fault patch overlaps target area in {overlap_path}")
    return {
        "block_x_start": int(selected["cell_i"].min()),
        "block_x_end": int(selected["cell_i"].max()),
        "block_y_start": int(selected["cell_j"].min()),
        "block_y_end": int(selected["cell_j"].max()),
        "selected_patch_count": int(len(selected)),
        "selected_fault_count": int(selected["fault_name"].nunique()) if "fault_name" in selected.columns else 0,
    }


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1.0e-9:
        return arr / norm
    fb = np.asarray(fallback, dtype=float)
    return fb / max(float(np.linalg.norm(fb)), 1.0e-9)


def strike_dip_from_normal(normal: np.ndarray) -> tuple[float, float]:
    n = normalize(normal, np.array([0.0, 0.0, 1.0]))
    strike = np.array([-n[1], n[0], 0.0], dtype=float)
    if float(np.linalg.norm(strike)) < 1.0e-9:
        azimuth = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(strike[1], strike[0])) % 180.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(n[2])), 0.0, 1.0))))
    return azimuth, dip


def plane_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(points, dtype=float)
    centered = pts - pts.mean(axis=0, keepdims=True)
    if len(pts) >= 3 and float(np.linalg.norm(centered)) > 1.0e-9:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis1 = normalize(vh[0], np.array([1.0, 0.0, 0.0]))
        axis2 = normalize(vh[1], np.array([0.0, 1.0, 0.0]))
        normal = normalize(vh[2], np.array([0.0, 0.0, 1.0]))
    elif len(pts) == 2:
        line_axis = normalize(pts[1] - pts[0], np.array([1.0, 0.0, 0.0]))
        vertical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(line_axis, vertical_axis))) > 0.95:
            vertical_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = normalize(np.cross(line_axis, vertical_axis), np.array([0.0, 1.0, 0.0]))
        axis1 = normalize(line_axis, np.array([1.0, 0.0, 0.0]))
        axis2 = normalize(np.cross(normal, axis1), np.array([0.0, 0.0, 1.0]))
    else:
        axis1 = np.array([1.0, 0.0, 0.0], dtype=float)
        axis2 = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.array([0.0, 1.0, 0.0], dtype=float)
    if axis2[2] < 0:
        axis2 = -axis2
    azimuth, dip = strike_dip_from_normal(normal)
    return axis1, axis2, normal, azimuth, dip


def make_patch(
    patch_id: str,
    center: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    length: float,
    height: float,
    azimuth: float,
    dip: float,
    layer: str,
    source_type: str,
    constraint: str,
    confidence: float,
    source_density: float,
    fault_name: str = "",
    component_id: int = 0,
    ordinal: int = 0,
) -> dict[str, Any]:
    center_arr = np.asarray(center, dtype=float)
    axis1_arr = normalize(axis1, np.array([1.0, 0.0, 0.0], dtype=float))
    axis2_arr = normalize(axis2, np.array([0.0, 0.0, 1.0], dtype=float))
    half_l = 0.5 * float(length)
    half_h = 0.5 * float(height)
    vertices = np.asarray(
        [
            center_arr - half_l * axis1_arr - half_h * axis2_arr,
            center_arr + half_l * axis1_arr - half_h * axis2_arr,
            center_arr + half_l * axis1_arr + half_h * axis2_arr,
            center_arr - half_l * axis1_arr + half_h * axis2_arr,
        ],
        dtype=float,
    )
    row = {
        "PatchID": patch_id,
        "GenerationStage": "step7c_large_fault_dfn",
        "SourceTraceIdx": -1,
        "DensityCellID": f"{source_type}_{patch_id}",
        "DensityCellPatchOrdinal": int(max(ordinal, 1)),
        "LayerGroup": layer,
        "LayerCode": legacy.LAYER_CODE.get(layer, 0),
        "CenterX": float(center[0]),
        "CenterY": float(center[1]),
        "CenterTime": float(center[2]),
        "TimeWindowMin": float(center[2] - 0.5 * height),
        "TimeWindowMax": float(center[2] + 0.5 * height),
        "LayerThickness": float(height),
        "SourceDensity": float(source_density),
        "CoherenceValue": np.nan,
        "LowCoherenceScore": float(source_density),
        "LowCoherenceWeight": 1.0,
        "GuidedDensityScore": float(source_density),
        "CandidateScore": float(source_density),
        "SamplingWeight": float(source_density),
        "DensityQuantileP95Layer": 1.0,
        "ExpectedPatchCountForCell": 1.0,
        "EffectiveCountScale": 1.0,
        "LengthM": float(length),
        "HeightTimeMs": float(height),
        "HeightM": float(height),
        "PatchAreaM2": float(length * height),
        "PatchEquivalentRadiusM": float(np.sqrt(max(length * height, 0.0) / np.pi)),
        "AzimuthDeg": float(azimuth % 180.0),
        "DipDeg": float(np.clip(dip, 0.0, 89.0)),
        "TraceGridDX": 12.5,
        "TraceGridDY": 12.5,
        "VoxelIX": -1,
        "VoxelIY": -1,
        "VoxelIT": -1,
        "ComponentID": int(component_id),
        "ComponentVoxelCount": 0,
        "LayerDensityThreshold": 0.0,
        "LocalOrientationPointCount": 0,
        "LocalDensityLinearity": 0.0,
        "LocalDensityPlanarity": 1.0,
        "LocalEigenvalue1": 0.0,
        "LocalEigenvalue2": 0.0,
        "LocalEigenvalue3": 0.0,
        "DensitySizeFactor": 1.0,
        "OrientationSource": (
            "segmented_fault_panel_geometry"
            if source_type == "large_original_fault_panel"
            else "original_fault_surface_geometry"
            if source_type == "large_original_fault_surface"
            else "large_lowcoh_component_pca"
        ),
        "SizeRule": "large_fault_surface_or_component_extent",
        "SamplingRule": "step7c_large_fault_hard_constraint_and_lowcoh_supplement",
        "FractureScale": "large",
        "FractureScaleCode": 3,
        "BandID": fault_name or f"large_component_{component_id}",
        "BandPatchOrdinal": int(max(ordinal, 1)),
        "BandContinuityMode": "large_fault_surface_panel" if source_type == "large_original_fault_surface" else "large_lowcoh_component_panel",
        "BandVoxelCount": 0,
        "BandLengthM": float(length),
        "BandTimeExtentMs": float(height),
        "BandPatchSpacingM": 0.0,
        "BandMeanDensity": float(source_density),
        "ObjectBandAzimuthDeg": float(azimuth % 180.0),
        "ObjectBandDipDeg": float(np.clip(dip, 0.0, 89.0)),
        "ObjectBandLengthM": float(length),
        "ObjectBandCenterSpacingM": 0.0,
        "ObjectBandOverlapRatio": 0.0,
        "SourceType": source_type,
        "ConstraintLevel": constraint,
        "Confidence": float(confidence),
        "FaultName": fault_name,
        "NeedsWellCorrection": 0 if constraint == "hard" else 1,
    }
    return add_vertex_columns(row, vertices)


def read_fault_sticks(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=["Line", "Trace", "X", "Y", "Time", "Flag", "FaultName"], engine="python")
    target = dict(config.get("target_block", {}))
    x_min = float(target.get("x_min", -np.inf)) - float(config.get("fault_clip_margin_m", 0.0))
    x_max = float(target.get("x_max", np.inf)) + float(config.get("fault_clip_margin_m", 0.0))
    y_min = float(target.get("y_min", -np.inf)) - float(config.get("fault_clip_margin_m", 0.0))
    y_max = float(target.get("y_max", np.inf)) + float(config.get("fault_clip_margin_m", 0.0))
    t_min = float(config.get("time_min_ms", -np.inf))
    t_max = float(config.get("time_max_ms", np.inf))
    out = df[df["X"].between(x_min, x_max) & df["Y"].between(y_min, y_max) & df["Time"].between(t_min, t_max)].copy()
    return out.reset_index(drop=True)


def build_fault_surface_panels(fault_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[tuple[np.ndarray, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    surface_polys: list[tuple[np.ndarray, dict[str, Any]]] = []
    patch_idx = 0
    min_points = int(config.get("min_fault_points_per_panel", 3))
    max_length = float(config.get("max_fault_panel_length_m", 700.0))
    max_height = float(config.get("max_fault_panel_height_ms", 360.0))
    damage_count = int(config.get("damage_zone_patch_count_per_fault", 2))
    damage_offset = float(config.get("damage_zone_offset_m", 60.0))
    for fault_name, fault_group in fault_df.groupby("FaultName", dropna=False):
        for flag, stick in fault_group.groupby("Flag", dropna=False):
            if len(stick) < min_points:
                continue
            pts = stick.sort_values("Time")[["X", "Y", "Time"]].to_numpy(dtype=float)
            axis1, axis2, normal, azimuth, dip = plane_axes(pts)
            center = pts.mean(axis=0)
            proj1 = (pts - center) @ axis1
            proj2 = (pts - center) @ axis2
            length = float(np.quantile(proj1, 0.95) - np.quantile(proj1, 0.05))
            height = float(np.quantile(pts[:, 2], 0.95) - np.quantile(pts[:, 2], 0.05))
            length = float(np.clip(length, float(config.get("min_fault_panel_length_m", 80.0)), max_length))
            height = float(np.clip(height, float(config.get("min_fault_panel_height_ms", 30.0)), max_height))
            if length <= 0 or height <= 0:
                continue
            patch_idx += 1
            patch_id = f"large_fault_surface_{patch_idx:05d}"
            rows.append(make_patch(patch_id, center, axis1, axis2, length, height, azimuth, dip, "沙三段", "large_original_fault_surface", "hard", 1.0, 1.0, str(fault_name), int(flag), patch_idx))
            half_l = 0.5 * length
            half_h = 0.5 * height
            vertices = np.asarray(
                [
                    center - half_l * axis1 - half_h * axis2,
                    center + half_l * axis1 - half_h * axis2,
                    center + half_l * axis1 + half_h * axis2,
                    center - half_l * axis1 + half_h * axis2,
                ],
                dtype=float,
            )
            surface_polys.append((vertices, {"FaultName": str(fault_name), "Flag": int(flag), "PatchID": patch_id, "AzimuthDeg": azimuth, "DipDeg": dip}))
            for damage_idx in range(damage_count):
                side = -1.0 if damage_idx % 2 == 0 else 1.0
                offset_center = center + side * damage_offset * normal
                patch_idx += 1
                rows.append(
                    make_patch(
                        f"large_fault_damage_{patch_idx:05d}",
                        offset_center,
                        axis1,
                        axis2,
                        length * float(config.get("damage_zone_length_multiplier", 0.65)),
                        height * float(config.get("damage_zone_height_multiplier", 0.80)),
                        azimuth,
                        dip,
                        "沙三段",
                        "large_original_fault_damage_zone",
                        "hard_influence",
                        0.85,
                        0.85,
                        str(fault_name),
                        int(flag),
                        patch_idx,
                    )
                )
    return pd.DataFrame(rows), surface_polys


def axis_from_azimuth_deg(azimuth_deg: float) -> np.ndarray:
    az = np.deg2rad(float(azimuth_deg))
    return normalize(np.asarray([np.cos(az), np.sin(az), 0.0], dtype=float), np.asarray([1.0, 0.0, 0.0]))


def axes_from_strike_dip(strike_deg: float, dip_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    axis1 = axis_from_azimuth_deg(strike_deg)
    dip_rad = np.deg2rad(float(np.clip(dip_deg, 1.0, 89.0)))
    horizontal_dip = normalize(np.asarray([-axis1[1], axis1[0], 0.0], dtype=float), np.asarray([0.0, 1.0, 0.0]))
    axis2 = normalize(
        np.asarray(
            [
                horizontal_dip[0] * np.cos(dip_rad),
                horizontal_dip[1] * np.cos(dip_rad),
                np.sin(dip_rad),
            ],
            dtype=float,
        ),
        np.asarray([0.0, 0.0, 1.0]),
    )
    normal = normalize(np.cross(axis1, axis2), np.asarray([0.0, 1.0, 0.0]))
    azimuth, dip = strike_dip_from_normal(normal)
    return axis1, axis2, normal, azimuth, dip


def world_to_local_points(
    points: np.ndarray,
    center: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    return np.column_stack(
        [
            rel @ normalize(axis1, np.array([1.0, 0.0, 0.0])),
            rel @ normalize(axis2, np.array([0.0, 0.0, 1.0])),
            rel @ normalize(normal, np.array([0.0, 1.0, 0.0])),
        ]
    )


def row_geometry_from_vertices(vertices: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    verts = np.asarray(vertices, dtype=float)
    center = verts.mean(axis=0)
    edge_u = verts[1] - verts[0]
    edge_v = verts[3] - verts[0]
    length = float(np.linalg.norm(edge_u))
    height = float(abs(verts[:, 2].max() - verts[:, 2].min()))
    normal = normalize(np.cross(edge_u, edge_v), np.array([0.0, 1.0, 0.0]))
    azimuth, dip = strike_dip_from_normal(normal)
    return center, length, height, azimuth, dip


def standard_original_fault_panel_row(panel_row: pd.Series, patch_idx: int) -> dict[str, Any]:
    vertices = vertices_from_row(panel_row)
    center, length, height, azimuth, dip = row_geometry_from_vertices(vertices)
    length = float(max(length, float(panel_row.get("PanelLength", 0.0)), 1.0))
    height = float(max(height, float(panel_row.get("PanelHeight", 0.0)), 1.0))
    row = make_patch(
        patch_id=f"large_original_fault_panel_{patch_idx:05d}",
        center=center,
        axis1=np.asarray([float(panel_row.get("StrikeVecX", 1.0)), float(panel_row.get("StrikeVecY", 0.0)), float(panel_row.get("StrikeVecZ", 0.0))]),
        axis2=np.asarray([float(panel_row.get("DipVecX", 0.0)), float(panel_row.get("DipVecY", 0.0)), float(panel_row.get("DipVecZ", 1.0))]),
        length=length,
        height=height,
        azimuth=float(panel_row.get("StrikeDeg", azimuth)),
        dip=float(panel_row.get("DipDeg", dip)),
        layer="沙四段",
        source_type="large_original_fault_panel",
        constraint="hard",
        confidence=1.0,
        source_density=1.0,
        fault_name=str(panel_row.get("FaultName", "")),
        component_id=int(panel_row.get("FaultPanelID", patch_idx)),
        ordinal=patch_idx,
    )
    add_vertex_columns(row, vertices)
    row.update(
        {
            "FaultPanelID": int(panel_row.get("FaultPanelID", patch_idx)),
            "SourcePatchCount": int(panel_row.get("SourcePatchCount", 1)),
            "SourceUnitCount": int(panel_row.get("SourceUnitCount", 1)),
            "SourceUnitIDs": str(panel_row.get("SourceUnitIDs", "")),
            "OrientationSource": "regional_fault_panel_vertices",
            "SizeRule": "regional_fault_panel_vertices",
            "BandContinuityMode": "original_fault_panel_geometry",
            "PatchAreaM2": float(panel_row.get("PanelArea", row["PatchAreaM2"])),
            "PatchArea": float(panel_row.get("PanelArea", row["PatchAreaM2"])),
        }
    )
    return row


def standard_surface_fragment_rows(surface_csv: Path) -> pd.DataFrame:
    if not surface_csv.exists():
        return pd.DataFrame()
    surface = read_csv_flexible(surface_csv)
    if surface.empty:
        return pd.DataFrame()
    missing_vertices = [col for col in vertex_columns() if col not in surface.columns]
    if missing_vertices:
        raise ValueError(f"{surface_csv} missing surface vertex columns: {missing_vertices}")
    rows: list[dict[str, Any]] = []
    for idx, src in enumerate(surface.itertuples(index=False), start=1):
        src_row = surface.iloc[idx - 1]
        vertices = vertices_from_row(src_row)
        center, length, height, azimuth, dip = row_geometry_from_vertices(vertices)
        out = make_patch(
            patch_id=f"large_original_fault_surface_fragment_{idx:05d}",
            center=center,
            axis1=vertices[1] - vertices[0],
            axis2=vertices[3] - vertices[0],
            length=float(src_row.get("PatchLength", length)),
            height=float(src_row.get("PatchHeight", height)),
            azimuth=float(src_row.get("Azimuth", azimuth)),
            dip=float(src_row.get("Dip", dip)),
            layer="沙四段",
            source_type="large_original_fault_surface_fragment",
            constraint="hard",
            confidence=0.98,
            source_density=1.0,
            fault_name=str(src_row.get("FaultName", "")),
            component_id=int(src_row.get("FaultSurfaceFragmentID", idx)),
            ordinal=idx,
        )
        add_vertex_columns(out, vertices)
        out.update(
            {
                "FaultSurfaceFragmentID": int(src_row.get("FaultSurfaceFragmentID", idx)),
                "SourcePatchFile": str(src_row.get("SourcePatchFile", "")),
                "SourceUnitID": str(src_row.get("SourceUnitID", "")),
                "SourceCellI": int(src_row.get("SourceCellI", -1)) if pd.notna(src_row.get("SourceCellI", np.nan)) else -1,
                "SourceCellJ": int(src_row.get("SourceCellJ", -1)) if pd.notna(src_row.get("SourceCellJ", np.nan)) else -1,
                "OrientationSource": "raw_fault_surface_fragment_vertices",
                "SizeRule": "raw_fault_surface_fragment_vertices",
                "BandContinuityMode": "original_fault_surface_fragment",
                "PatchAreaM2": float(src_row.get("PatchArea", out["PatchAreaM2"])),
                "PatchArea": float(src_row.get("PatchArea", out["PatchAreaM2"])),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def build_fault_panel_and_influence_rows(panel_csv: Path, config: dict[str, Any]) -> pd.DataFrame:
    if not panel_csv.exists():
        return pd.DataFrame()
    panel_df = read_csv_flexible(panel_csv)
    if panel_df.empty:
        return pd.DataFrame()
    missing_vertices = [col for col in vertex_columns() if col not in panel_df.columns]
    if missing_vertices:
        raise ValueError(f"{panel_csv} missing panel vertex columns: {missing_vertices}")

    rows: list[dict[str, Any]] = []
    damage_count = int(config.get("damage_zone_patch_count_per_fault", 1))
    damage_offset = float(config.get("damage_zone_offset_m", 55.0))
    length_multiplier = float(config.get("damage_zone_length_multiplier", 0.55))
    height_multiplier = float(config.get("damage_zone_height_multiplier", 0.75))
    patch_idx = 0
    for _, panel_row in panel_df.iterrows():
        patch_idx += 1
        main = standard_original_fault_panel_row(panel_row, patch_idx)
        rows.append(main)
        strike_vec = normalize(
            np.asarray([float(panel_row.get("StrikeVecX", 1.0)), float(panel_row.get("StrikeVecY", 0.0)), float(panel_row.get("StrikeVecZ", 0.0))]),
            np.array([1.0, 0.0, 0.0]),
        )
        dip_vec = normalize(
            np.asarray([float(panel_row.get("DipVecX", 0.0)), float(panel_row.get("DipVecY", 0.0)), float(panel_row.get("DipVecZ", 1.0))]),
            np.array([0.0, 0.0, 1.0]),
        )
        normal = normalize(np.cross(strike_vec, dip_vec), np.array([0.0, 1.0, 0.0]))
        center = np.asarray([float(panel_row["CenterX"]), float(panel_row["CenterY"]), float(panel_row["CenterTIME"])], dtype=float)
        for damage_idx in range(damage_count):
            side = -1.0 if damage_idx % 2 == 0 else 1.0
            patch_idx += 1
            damage = make_patch(
                patch_id=f"large_original_fault_damage_{patch_idx:05d}",
                center=center + side * damage_offset * normal,
                axis1=strike_vec,
                axis2=dip_vec,
                length=max(float(panel_row.get("PanelLength", main["LengthM"])) * length_multiplier, 1.0),
                height=max(float(panel_row.get("PanelHeight", main["HeightTimeMs"])) * height_multiplier, 1.0),
                azimuth=float(panel_row.get("StrikeDeg", main["AzimuthDeg"])),
                dip=float(panel_row.get("DipDeg", main["DipDeg"])),
                layer="沙四段",
                source_type="large_original_fault_damage_zone",
                constraint="hard_influence",
                confidence=0.85,
                source_density=0.85,
                fault_name=str(panel_row.get("FaultName", "")),
                component_id=int(panel_row.get("FaultPanelID", patch_idx)),
                ordinal=patch_idx,
            )
            damage.update(
                {
                    "FaultPanelID": int(panel_row.get("FaultPanelID", patch_idx)),
                    "SourcePatchCount": int(panel_row.get("SourcePatchCount", 1)),
                    "SourceUnitCount": int(panel_row.get("SourceUnitCount", 1)),
                    "SourceUnitIDs": str(panel_row.get("SourceUnitIDs", "")),
                    "OrientationSource": "regional_fault_panel_vertices_offset_damage_zone",
                    "SizeRule": "regional_fault_panel_scaled_damage_zone",
                    "BandContinuityMode": "original_fault_panel_damage_zone",
                    "DamageZoneSide": int(side),
                }
            )
            rows.append(damage)
    return pd.DataFrame(rows)


def build_segmented_fault_panels(config: dict[str, Any]) -> tuple[pd.DataFrame, list[tuple[np.ndarray, dict[str, Any]]]]:
    audit_path = config.get("original_fault_rasterization_audit_csv")
    overlap_path = config.get("fault_patch_overlap_csv")
    if not audit_path and not overlap_path:
        return pd.DataFrame(), []
    source_path = Path(audit_path or overlap_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"segmented fault audit/overlap CSV not found: {source_path}")
    rows_df = pd.read_csv(source_path)
    required_geometry = {"cx", "cy", "cz", "bbox_xmin", "bbox_xmax", "bbox_ymin", "bbox_ymax", "bbox_zmin", "bbox_zmax"}
    if not required_geometry.issubset(rows_df.columns):
        if not overlap_path:
            raise ValueError(f"{source_path} lacks geometry columns and no fault_patch_overlap_csv was configured")
        source_path = Path(overlap_path).resolve()
        rows_df = pd.read_csv(source_path)
        if not required_geometry.issubset(rows_df.columns):
            missing = sorted(required_geometry.difference(rows_df.columns))
            raise ValueError(f"{source_path} missing segmented fault geometry columns: {missing}")
    required_orientation = {"strike_deg", "dip_deg"}
    if "rasterized_voxel_count" in rows_df.columns:
        rows_df = rows_df[pd.to_numeric(rows_df["rasterized_voxel_count"], errors="coerce").fillna(0) > 0].copy()
    if "intersects_demo_xy_t" in rows_df.columns:
        rows_df = rows_df[rows_df["intersects_demo_xy_t"].astype(bool)].copy()
    if rows_df.empty:
        return pd.DataFrame(), []
    missing_orientation = sorted(required_orientation.difference(rows_df.columns))
    if missing_orientation:
        raise ValueError(f"{source_path} missing segmented fault orientation columns: {missing_orientation}")

    out_rows: list[dict[str, Any]] = []
    surface_polys: list[tuple[np.ndarray, dict[str, Any]]] = []
    damage_count = int(config.get("damage_zone_patch_count_per_fault", 1))
    damage_offset = float(config.get("damage_zone_offset_m", 55.0))
    patch_idx = 0
    for _, row in rows_df.iterrows():
        center = np.asarray([float(row["cx"]), float(row["cy"]), float(row["cz"])], dtype=float)
        strike = float(row["strike_deg"])
        dip_value = float(row["dip_deg"])
        axis1, axis2, normal, azimuth, dip = axes_from_strike_dip(strike, dip_value)
        x_extent = float(row["bbox_xmax"] - row["bbox_xmin"])
        y_extent = float(row["bbox_ymax"] - row["bbox_ymin"])
        t_extent = float(row["bbox_zmax"] - row["bbox_zmin"])
        length = float(np.clip(np.hypot(x_extent, y_extent), float(config.get("min_fault_panel_length_m", 80.0)), float(config.get("max_fault_panel_length_m", 700.0))))
        height = float(np.clip(t_extent, float(config.get("min_fault_panel_height_ms", 20.0)), float(config.get("max_fault_panel_height_ms", 360.0))))
        patch_idx += 1
        fault_name = str(row.get("fault_name", "segmented_fault"))
        patch_id = f"large_segmented_fault_surface_{patch_idx:05d}"
        out_rows.append(
            make_patch(
                patch_id,
                center,
                axis1,
                axis2,
                length,
                height,
                azimuth,
                dip,
                "沙三段",
                "large_original_fault_panel",
                "hard",
                1.0,
                1.0,
                fault_name,
                int(row.get("cell_i", 0)),
                patch_idx,
            )
        )
        half_l = 0.5 * length
        half_h = 0.5 * height
        vertices = np.asarray(
            [
                center - half_l * axis1 - half_h * axis2,
                center + half_l * axis1 - half_h * axis2,
                center + half_l * axis1 + half_h * axis2,
                center - half_l * axis1 + half_h * axis2,
            ],
            dtype=float,
        )
        surface_polys.append((vertices, {"FaultName": fault_name, "PatchID": patch_id, "AzimuthDeg": azimuth, "DipDeg": dip}))
        for damage_idx in range(damage_count):
            side = -1.0 if damage_idx % 2 == 0 else 1.0
            patch_idx += 1
            out_rows.append(
                make_patch(
                    f"large_segmented_fault_damage_{patch_idx:05d}",
                    center + side * damage_offset * normal,
                    axis1,
                    axis2,
                    length * float(config.get("damage_zone_length_multiplier", 0.55)),
                    height * float(config.get("damage_zone_height_multiplier", 0.75)),
                    azimuth,
                    dip,
                    "沙三段",
                    "large_original_fault_damage_zone",
                    "hard_influence",
                    0.85,
                    0.85,
                    fault_name,
                    int(row.get("cell_i", 0)),
                    patch_idx,
                )
            )
    return pd.DataFrame(out_rows), surface_polys


def build_large_lowcoh_supplements(config: dict[str, Any]) -> pd.DataFrame:
    component_summary = config.get("large_component_summary_csv")
    if component_summary:
        summary_path = Path(component_summary).resolve()
        if not summary_path.exists():
            raise FileNotFoundError(f"large component summary CSV not found: {summary_path}")
        component_df = pd.read_csv(summary_path)
        if component_df.empty:
            return pd.DataFrame()
        required = {
            "component_id",
            "voxel_count",
            "score_mean",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "time_min_ms",
            "time_max_ms",
            "pca_azimuth_deg",
            "pca_dip_deg",
        }
        missing = sorted(required.difference(component_df.columns))
        if missing:
            raise ValueError(f"{summary_path} missing large component columns: {missing}")
        max_components = int(config.get("max_lowcoh_supplement_components", 8))
        min_voxels = int(config.get("min_lowcoh_component_voxels", 80))
        min_dip = float(config.get("min_lowcoh_dip_deg", 45.0))
        component_df = component_df[
            pd.to_numeric(component_df["voxel_count"], errors="coerce").fillna(0).ge(min_voxels)
            & pd.to_numeric(component_df["pca_dip_deg"], errors="coerce").fillna(0).ge(min_dip)
        ].copy()
        if component_df.empty:
            return pd.DataFrame()
        component_df["selection_score"] = (
            pd.to_numeric(component_df["voxel_count"], errors="coerce").fillna(0).astype(float)
            * pd.to_numeric(component_df["score_mean"], errors="coerce").fillna(0).astype(float)
        )
        component_df = component_df.sort_values("selection_score", ascending=False).head(max_components)
        rows: list[dict[str, Any]] = []
        for ordinal, row in enumerate(component_df.itertuples(index=False), start=1):
            center = np.asarray(
                [
                    0.5 * (float(row.x_min) + float(row.x_max)),
                    0.5 * (float(row.y_min) + float(row.y_max)),
                    0.5 * (float(row.time_min_ms) + float(row.time_max_ms)),
                ],
                dtype=float,
            )
            axis1, axis2, _normal, azimuth, dip = axes_from_strike_dip(float(row.pca_azimuth_deg), float(row.pca_dip_deg))
            horizontal_extent = max(float(row.x_max) - float(row.x_min), float(row.y_max) - float(row.y_min))
            time_extent = float(row.time_max_ms) - float(row.time_min_ms)
            length = float(
                np.clip(
                    horizontal_extent,
                    float(config.get("min_lowcoh_length_m", 120.0)),
                    float(config.get("max_lowcoh_length_m", 480.0)),
                )
            )
            height = float(
                np.clip(
                    time_extent,
                    float(config.get("min_lowcoh_height_ms", 35.0)),
                    float(config.get("max_lowcoh_height_ms", 180.0)),
                )
            )
            layer = contract_layer_for_center(config, center)
            if layer is None:
                continue
            rows.append(
                make_patch(
                    f"large_lowcoh_supplement_{ordinal:05d}",
                    center,
                    axis1,
                    axis2,
                    length,
                    height,
                    azimuth,
                    dip,
                    layer,
                    "large_lowcoh_inferred",
                    "seismic_prior",
                    0.65,
                    float(row.score_mean),
                    f"large_lowcoh_component_{int(row.component_id)}",
                    int(row.component_id),
                    ordinal,
                )
            )
        return pd.DataFrame(rows)

    if not config.get("large_prior_sgy"):
        return pd.DataFrame()
    grid = legacy.load_density_grid(Path(config["large_prior_sgy"]).resolve(), Path(config["trace_mapping_npz"]).resolve())
    mask_grid = legacy.load_density_grid(Path(config["large_mask_sgy"]).resolve(), Path(config["trace_mapping_npz"]).resolve())["density"]
    trace_mapping_path = Path(config["trace_mapping_npz"]).resolve()
    with np.load(trace_mapping_path) as mapping_npz:
        mapping = {key: mapping_npz[key] for key in mapping_npz.files}
    horizon_contract = load_contract_for_mapping(config, mapping)
    validate_window_contract(config, horizon_contract, grid["samples"])
    surfaces = surface_grids_from_contract(mapping, horizon_contract)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(mask_grid > 0.5, structure=structure)
    rows: list[dict[str, Any]] = []
    max_components = int(config.get("max_lowcoh_supplement_components", 8))
    min_voxels = int(config.get("min_lowcoh_component_voxels", 80))
    sizes = np.bincount(labels.ravel()) if count else np.asarray([], dtype=int)
    component_ids = [idx for idx in range(1, count + 1) if int(sizes[idx]) >= min_voxels]
    component_ids.sort(key=lambda idx: float(grid["density"][labels == idx].sum()), reverse=True)
    for ordinal, component_id in enumerate(component_ids[:max_components], start=1):
        yy, xx, tt = np.where(labels == component_id)
        if yy.size < 3:
            continue
        coords = np.column_stack([grid["x_values"][xx], grid["y_values"][yy], grid["samples"][tt]]).astype(float)
        axis1, axis2, _normal, azimuth, dip = plane_axes(coords)
        center = coords.mean(axis=0)
        proj1 = (coords - center) @ axis1
        length = float(np.quantile(proj1, 0.92) - np.quantile(proj1, 0.08))
        height = float(np.quantile(coords[:, 2], 0.92) - np.quantile(coords[:, 2], 0.08))
        length = float(np.clip(length, float(config.get("min_lowcoh_length_m", 120.0)), float(config.get("max_lowcoh_length_m", 480.0))))
        height = float(np.clip(height, float(config.get("min_lowcoh_height_ms", 35.0)), float(config.get("max_lowcoh_height_ms", 180.0))))
        layer = "沙三段"
        if legacy.layer_mask_for_grid("沙四段", grid["samples"], surfaces)[yy, xx, tt].sum() > yy.size / 2:
            layer = "沙四段"
        rows.append(
            make_patch(
                f"large_lowcoh_supplement_{ordinal:05d}",
                center,
                axis1,
                axis2,
                length,
                height,
                azimuth,
                dip,
                layer,
                "large_lowcoh_inferred",
                "seismic_prior",
                0.65,
                float(np.mean(grid["density"][yy, xx, tt])),
                f"large_lowcoh_component_{component_id}",
                int(component_id),
                ordinal,
            )
        )
    return pd.DataFrame(rows)


def build_lowcoh_component_panels(config: dict[str, Any]) -> pd.DataFrame:
    component_summary = Path(config["large_component_summary_csv"]).resolve()
    component_npz = Path(config["large_prior_components_npz"]).resolve()
    if not component_summary.exists():
        raise FileNotFoundError(f"large_component_summary_csv not found: {component_summary}")
    if not component_npz.exists():
        raise FileNotFoundError(f"large_prior_components_npz not found: {component_npz}")

    summary_df = read_csv_flexible(component_summary)
    if summary_df.empty:
        return pd.DataFrame()
    required = {
        "component_id",
        "voxel_count",
        "score_mean",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "time_min_ms",
        "time_max_ms",
        "pca_azimuth_deg",
        "pca_dip_deg",
    }
    missing = sorted(required.difference(summary_df.columns))
    if missing:
        raise ValueError(f"{component_summary} missing columns: {missing}")

    npz = np.load(component_npz)
    component_id_grid = npz["inferred_component_id"].astype(np.int32)
    x_values = npz["x"].astype(float)
    y_values = npz["y"].astype(float)
    samples = npz["samples"].astype(float)
    max_components = int(config.get("max_lowcoh_supplement_components", 12))
    min_voxels = int(config.get("min_lowcoh_component_voxels", 500))
    min_dip = float(config.get("min_lowcoh_dip_deg", 45.0))
    target_length = float(config.get("lowcoh_target_panel_length_m", 180.0))
    target_height = float(config.get("lowcoh_target_panel_height_ms", 110.0))
    max_panels_per_component = int(config.get("lowcoh_max_panels_per_component", 4))
    min_panel_voxels = int(config.get("lowcoh_min_panel_voxels", max(80, min_voxels // 4)))
    overlap_ratio = float(config.get("lowcoh_panel_overlap_ratio", 0.18))
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 2.0))

    summary_df = summary_df[
        pd.to_numeric(summary_df["voxel_count"], errors="coerce").fillna(0).ge(min_voxels)
        & pd.to_numeric(summary_df["pca_dip_deg"], errors="coerce").fillna(0).ge(min_dip)
    ].copy()
    if summary_df.empty:
        return pd.DataFrame()
    summary_df["selection_score"] = (
        pd.to_numeric(summary_df["voxel_count"], errors="coerce").fillna(0).astype(float)
        * pd.to_numeric(summary_df["score_mean"], errors="coerce").fillna(0).astype(float)
    )
    summary_df = summary_df.sort_values("selection_score", ascending=False).head(max_components)

    selected_component_ids = summary_df["component_id"].astype(int).tolist()
    component_positions: dict[int, tuple[np.ndarray, ...]] = {}
    if component_id_grid.ndim == 2:
        selected_mask = np.isin(component_id_grid, np.asarray(selected_component_ids, dtype=np.int32))
        trace_idx_all, tt_all = np.where(selected_mask)
        ids_all = component_id_grid[trace_idx_all, tt_all]
        for comp_id in selected_component_ids:
            keep = ids_all == comp_id
            component_positions[comp_id] = (trace_idx_all[keep], tt_all[keep])
        x_axis = y_axis = None
    elif component_id_grid.ndim == 3:
        slices = ndimage.find_objects(component_id_grid, max_label=int(component_id_grid.max()))
        for comp_id in selected_component_ids:
            component_slice = slices[comp_id - 1] if 0 < comp_id <= len(slices) else None
            if component_slice is None:
                component_positions[comp_id] = (np.asarray([], dtype=int),) * 3
                continue
            local = component_id_grid[component_slice]
            yy, xx, tt = np.where(local == comp_id)
            yy += int(component_slice[0].start)
            xx += int(component_slice[1].start)
            tt += int(component_slice[2].start)
            component_positions[comp_id] = (yy, xx, tt)
        ix = npz["ix"].astype(np.int32)
        iy = npz["iy"].astype(np.int32)
        x_axis = np.zeros(int(ix.max()) + 1, dtype=float)
        y_axis = np.zeros(int(iy.max()) + 1, dtype=float)
        for idx in range(len(x_axis)):
            x_axis[idx] = float(np.median(x_values[ix == idx]))
        for idx in range(len(y_axis)):
            y_axis[idx] = float(np.median(y_values[iy == idx]))
    else:
        raise ValueError(f"unsupported inferred_component_id shape: {component_id_grid.shape}")

    rows: list[dict[str, Any]] = []
    patch_idx = 0
    for component_ord, comp_row in enumerate(summary_df.itertuples(index=False), start=1):
        comp_id = int(comp_row.component_id)
        if component_id_grid.ndim == 2:
            trace_idx, tt = component_positions[comp_id]
            if len(trace_idx) < min_panel_voxels:
                continue
            points = np.column_stack([x_values[trace_idx], y_values[trace_idx], samples[tt]]).astype(float)
        else:
            yy, xx, tt = component_positions[comp_id]
            if len(xx) < min_panel_voxels:
                continue
            points = np.column_stack([x_axis[xx], y_axis[yy], samples[tt]]).astype(float)
        if len(points) < min_panel_voxels:
            continue
        scaled_points = np.column_stack([points[:, 0], points[:, 1], points[:, 2] * time_scale]).astype(float)
        _, _, _, pca_azimuth, pca_dip = plane_axes(scaled_points)
        axis1, axis2, normal, azimuth, dip = axes_from_strike_dip(pca_azimuth, pca_dip)
        center = points.mean(axis=0)
        local = world_to_local_points(points, center, axis1, axis2, normal)
        strike_vals = local[:, 0]
        time_vals = points[:, 2]
        strike_extent = float(np.percentile(strike_vals, 95) - np.percentile(strike_vals, 5))
        time_extent = float(np.percentile(time_vals, 95) - np.percentile(time_vals, 5))
        n_panels = int(np.clip(np.ceil(max(strike_extent, 1.0) / max(target_length, 1.0)), 1, max_panels_per_component))
        if n_panels == 1 and len(points) >= 2 * min_panel_voxels:
            n_panels = 2
        edges = np.linspace(float(strike_vals.min()), float(strike_vals.max()), n_panels + 1)
        panel_rows: list[dict[str, Any]] = []
        for panel_idx in range(n_panels):
            left = edges[panel_idx]
            right = edges[panel_idx + 1]
            width = right - left
            pad = width * overlap_ratio
            mask = (strike_vals >= left - pad) & (strike_vals <= right + pad)
            if mask.sum() < min_panel_voxels and panel_rows:
                continue
            if mask.sum() < min_panel_voxels:
                continue
            seg_points = points[mask]
            seg_scaled = scaled_points[mask]
            seg_center = seg_points.mean(axis=0)
            seg_local = world_to_local_points(seg_points, seg_center, axis1, axis2, normal)
            seg_length = float(np.percentile(seg_local[:, 0], 95) - np.percentile(seg_local[:, 0], 5))
            seg_height = float(np.percentile(seg_points[:, 2], 95) - np.percentile(seg_points[:, 2], 5))
            seg_length = float(np.clip(seg_length, float(config.get("min_lowcoh_length_m", 120.0)), float(config.get("max_lowcoh_length_m", 620.0))))
            seg_height = float(np.clip(max(seg_height, time_extent / max(n_panels, 1)), float(config.get("min_lowcoh_height_ms", 80.0)), float(config.get("max_lowcoh_height_ms", 220.0))))
            seg_azimuth = azimuth
            seg_dip = dip
            if len(seg_scaled) >= 3:
                _, _, _, seg_azimuth, seg_dip = plane_axes(seg_scaled)
                seg_axis1, seg_axis2, _, seg_azimuth, seg_dip = axes_from_strike_dip(seg_azimuth, seg_dip)
            else:
                seg_axis1, seg_axis2 = axis1, axis2
            layer = contract_layer_for_center(config, seg_center)
            if layer is None:
                continue
            patch_idx += 1
            panel_rows.append(
                make_patch(
                    patch_id=f"large_lowcoh_panel_{patch_idx:05d}",
                    center=seg_center,
                    axis1=seg_axis1,
                    axis2=seg_axis2,
                    length=seg_length,
                    height=seg_height,
                    azimuth=float(seg_azimuth),
                    dip=float(seg_dip),
                    layer=layer,
                    source_type="large_lowcoh_inferred_panel",
                    constraint="seismic_prior",
                    confidence=0.65,
                    source_density=float(comp_row.score_mean),
                    fault_name=f"large_lowcoh_component_{comp_id}",
                    component_id=comp_id,
                    ordinal=panel_idx + 1,
                )
            )
        if not panel_rows:
            continue
        for row in panel_rows:
            row.update(
                {
                    "ComponentID": comp_id,
                    "ComponentVoxelCount": int(comp_row.voxel_count),
                    "ComponentScoreMean": float(comp_row.score_mean),
                    "ComponentPanelCount": int(len(panel_rows)),
                    "ComponentPanelOrdinal": int(row.get("BandPatchOrdinal", 1)),
                    "ComponentPanelMode": "component_slice_panel_group",
                    "ComponentPCAAzimuthDeg": float(comp_row.pca_azimuth_deg),
                    "ComponentPCADipDeg": float(comp_row.pca_dip_deg),
                    "SelectionScore": float(comp_row.selection_score),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_inferred_surface_panels(config: dict[str, Any]) -> pd.DataFrame:
    summary_path = Path(config["large_component_summary_csv"]).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"large_component_summary_csv not found: {summary_path}")
    surface_df = read_csv_flexible(summary_path)
    if surface_df.empty:
        return pd.DataFrame()
    missing = [column for column in vertex_columns() if column not in surface_df.columns]
    if missing:
        return build_lowcoh_component_panels(config)

    rows: list[dict[str, Any]] = []
    for ordinal, surface_row in surface_df.iterrows():
        vertices = vertices_from_row(surface_row)
        center, vertex_length, vertex_height, vertex_azimuth, vertex_dip = row_geometry_from_vertices(vertices)
        dominant_layer = str(surface_row.get("dominant_layer", "")).strip()
        dominant_fraction = float(surface_row.get("dominant_layer_fraction", np.nan))
        min_window_fraction = float(config.get("inferred_surface_min_window_voxel_fraction", 0.90))
        window_validation_mode = "center_fallback"
        if dominant_layer in ("沙三段", "沙四段") and np.isfinite(dominant_fraction) and dominant_fraction >= min_window_fraction:
            layer = dominant_layer
            window_validation_mode = "voxel_coverage"
        else:
            layer = contract_layer_for_center(config, center)
            if layer is None:
                continue
        surface_id = int(surface_row.get("surface_id", surface_row.get("component_id", ordinal + 1)))
        raw_component_id = int(surface_row.get("raw_component_id", surface_id))
        panel_chain_id = str(surface_row.get("panel_chain_id", f"raw_{raw_component_id:05d}"))
        panel_ordinal = int(surface_row.get("panel_ordinal_in_chain", 1))
        panel_count = int(surface_row.get("panel_count_in_chain", 1))
        length = float(max(surface_row.get("surface_length_m", vertex_length), vertex_length, 1.0))
        height = float(max(surface_row.get("surface_height_time_ms", vertex_height), vertex_height, 1.0))
        azimuth = float(surface_row.get("pca_azimuth_deg", vertex_azimuth))
        dip = float(surface_row.get("pca_dip_deg", vertex_dip))
        score_mean = float(surface_row.get("score_mean", 0.0))
        confidence = float(np.clip(0.55 + 0.35 * score_mean, 0.55, 0.90))
        out = make_patch(
            patch_id=f"large_inferred_fault_surface_{surface_id:05d}",
            center=center,
            axis1=vertices[1] - vertices[0],
            axis2=vertices[3] - vertices[0],
            length=length,
            height=height,
            azimuth=azimuth,
            dip=dip,
            layer=layer,
            source_type="large_inferred_fault_local_panel",
            constraint="seismic_prior",
            confidence=confidence,
            source_density=score_mean,
            fault_name=panel_chain_id,
            component_id=surface_id,
            ordinal=ordinal + 1,
        )
        add_vertex_columns(out, vertices)
        out.update(
            {
                "ComponentID": surface_id,
                "RawComponentID": raw_component_id,
                "ComponentVoxelCount": int(surface_row.get("voxel_count", 0)),
                "RawComponentVoxelCount": int(surface_row.get("raw_component_voxel_count", 0)),
                "SurfaceOrdinalInRawComponent": int(surface_row.get("surface_ordinal_in_raw_component", 1)),
                "PanelChainID": panel_chain_id,
                "PanelOrdinalInChain": panel_ordinal,
                "PanelCountInChain": panel_count,
                "SurfacePlanarity": float(surface_row.get("surface_planarity", np.nan)),
                "SupportMean": float(surface_row.get("support_mean", np.nan)),
                "SupportMax": float(surface_row.get("support_max", np.nan)),
                "AuxiliarySupportFraction": float(surface_row.get("auxiliary_support_fraction", np.nan)),
                "LocalAuxiliarySupportFractionMean": float(surface_row.get("local_auxiliary_support_fraction_mean", np.nan)),
                "LowCoherenceMean": float(surface_row.get("lowcoh_mean", np.nan)),
                "AntTrackMean": float(surface_row.get("anttrack_mean", np.nan)),
                "CurvatureMean": float(surface_row.get("curvature_mean", np.nan)),
                "EvidenceDistanceMeanM": float(surface_row.get("evidence_distance_mean_m", np.nan)),
                "EvidenceDistanceMaxM": float(surface_row.get("evidence_distance_max_m", np.nan)),
                "DominantLayer": str(surface_row.get("dominant_layer", layer)),
                "DominantLayerFraction": float(surface_row.get("dominant_layer_fraction", np.nan)),
                "RelativePositionStd": float(surface_row.get("relative_position_std", np.nan)),
                "RelativePositionSpan": float(surface_row.get("relative_position_span", np.nan)),
                "CandidateBranch": str(surface_row.get("candidate_branch", "lowcoherence_weighted_local_sheet")),
                "WindowValidationMode": window_validation_mode,
                "WindowVoxelFraction": float(dominant_fraction) if np.isfinite(dominant_fraction) else np.nan,
                "OrientationSource": "step6c_local_supported_sheet_panel_vertices",
                "SizeRule": "step6c_local_evidence_extent_150_to_500m",
                "BandID": panel_chain_id,
                "BandPatchOrdinal": panel_ordinal,
                "BandContinuityMode": "local_supported_panel_chain",
                "ComponentPanelCount": panel_count,
                "ComponentPanelOrdinal": panel_ordinal,
                "PatchAreaM2": float(surface_row.get("surface_area_m2", out["PatchAreaM2"])),
                "PatchArea": float(surface_row.get("surface_area_m2", out["PatchAreaM2"])),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def patch_vertices(row: pd.Series) -> list[tuple[float, float, float]]:
    if has_vertex_columns(row):
        return [(float(x), float(y), float(z)) for x, y, z in vertices_from_row(row)]
    azimuth = np.deg2rad(float(row["AzimuthDeg"]))
    dip = np.deg2rad(float(np.clip(row["DipDeg"], 1.0, 89.9)))
    half_length = 0.5 * float(row["LengthM"])
    half_height_time = 0.5 * float(row["HeightTimeMs"])
    strike = np.asarray([np.cos(azimuth), np.sin(azimuth)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(azimuth), np.cos(azimuth)], dtype=float)
    horizontal_dip_half = half_height_time / max(np.tan(dip), 1.0e-6)
    center_xy = np.asarray([float(row["CenterX"]), float(row["CenterY"])], dtype=float)
    center_t = float(row["CenterTime"])
    corners = []
    for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        xy = center_xy + strike_sign * half_length * strike + dip_sign * horizontal_dip_half * dip_horizontal
        corners.append((float(xy[0]), float(xy[1]), float(center_t + dip_sign * half_height_time)))
    return corners


def write_patch_vtk(path: Path, df: pd.DataFrame, title: str) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in df.iterrows():
        base = len(points)
        points.extend(patch_vertices(row))
        polygons.append([base, base + 1, base + 2, base + 3])
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA", f"POINTS {len(points)} float"]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {sum(len(p)+1 for p in polygons)}")
    lines.extend(f"{len(p)} {' '.join(str(i) for i in p)}" for p in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    scalar_cols = ["FractureScaleCode", "SourceDensity", "Confidence", "LengthM", "HeightTimeMs", "PatchAreaM2", "AzimuthDeg", "DipDeg"]
    for col in scalar_cols:
        if col not in df.columns:
            continue
        lines.append(f"SCALARS {col} {'int' if col == 'FractureScaleCode' else 'float'} 1")
        lines.append("LOOKUP_TABLE default")
        values = np.nan_to_num(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if col == "FractureScaleCode":
            lines.extend(str(int(v)) for v in values)
        else:
            lines.extend(f"{float(v):.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_surface_vtk(path: Path, surfaces: list[tuple[np.ndarray, dict[str, Any]]], title: str) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    azimuths: list[float] = []
    dips: list[float] = []
    for vertices, meta in surfaces:
        base = len(points)
        points.extend([(float(x), float(y), float(z)) for x, y, z in vertices])
        polygons.append([base, base + 1, base + 2, base + 3])
        azimuths.append(float(meta.get("AzimuthDeg", 0.0)))
        dips.append(float(meta.get("DipDeg", 0.0)))
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA", f"POINTS {len(points)} float"]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {sum(len(p)+1 for p in polygons)}")
    lines.extend(f"{len(p)} {' '.join(str(i) for i in p)}" for p in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    for name, values in [("AzimuthDeg", azimuths), ("DipDeg", dips)]:
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(f"{float(v):.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    horizon_lookup = build_spatial_lookup(config)
    config["_horizon_spatial_lookup"] = horizon_lookup
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)
    for legacy_path in [
        output_dir / "large_fault_only_and_influence_patches.csv",
        output_dir / "large_fault_only_and_influence_raw_time.vtk",
        output_dir / "large_fault_and_damage_patches.csv",
        output_dir / "large_fault_and_damage_raw_time.vtk",
        output_dir / "large_fault_surface_raw_time.vtk",
        output_dir / "large_fault_surface_patches.csv",
        output_dir / "large_original_fault_panel_patches.csv",
        output_dir / "large_fault_damage_zone_patches.csv",
        output_dir / "large_fault_damage_zone_raw_time.vtk",
        output_dir / "large_original_fault_merged_surface_raw_time.vtk",
        output_dir / "original_fault_units_demo_raw_time.vtp",
        output_dir / "large_inferred_fault_surfaces_raw_time.vtp",
        output_dir / "large_fault_result_raw_time.vtm",
    ]:
        legacy_path.unlink(missing_ok=True)
    shutil.rmtree(output_dir / "surface_panel_workspace", ignore_errors=True)

    source_surface_value = config.get("original_fault_surface_vtk", config.get("original_fault_merged_surface_vtk"))
    if not source_surface_value:
        raise ValueError("Step7C requires original_fault_surface_vtk")
    source_surface = Path(str(source_surface_value)).resolve()
    if not source_surface.exists():
        raise FileNotFoundError(f"original demo fault surface not found: {source_surface}")
    source_manifest_value = config.get("original_fault_manifest_csv")
    if not source_manifest_value:
        raise ValueError("Step7C requires original_fault_manifest_csv")
    source_manifest = Path(str(source_manifest_value)).resolve()
    if not source_manifest.exists():
        raise FileNotFoundError(f"original fault manifest not found: {source_manifest}")
    source_surface_hash = file_sha256(source_surface)
    shutil.copy2(source_surface, paths["original_surface_vtk"])
    shutil.copy2(source_manifest, paths["original_manifest_csv"])
    copied_surface_hash = file_sha256(paths["original_surface_vtk"])
    original_mesh = pv.read(paths["original_surface_vtk"])
    if not isinstance(original_mesh, pv.PolyData):
        original_mesh = original_mesh.extract_surface(algorithm="dataset_surface")
    original_mesh = original_mesh.triangulate()
    original_geometry_fingerprint = geometry_fingerprint(original_mesh)
    manifest_df = read_csv_flexible(paths["original_manifest_csv"])

    lowcoh_patches = build_inferred_surface_panels(config)
    if lowcoh_patches.empty:
        raise RuntimeError("no seismic-inferred local fault panels generated from Step6C")
    upstream_inferred_df = read_csv_flexible(Path(config["large_component_summary_csv"]).resolve())
    upstream_inferred_ids = set(
        pd.to_numeric(
            upstream_inferred_df.get("surface_id", upstream_inferred_df.get("component_id", pd.Series(dtype=float))),
            errors="coerce",
        ).dropna().astype(int)
    )
    lowcoh_patches, lowcoh_horizon_qc = enforce_center_horizon_contract(lowcoh_patches, horizon_lookup)
    classified_patches, inferred_classification = classify_inferred_panels_against_original(
        lowcoh_patches,
        paths["original_surface_vtk"],
        Path(config["large_prior_components_npz"]).resolve(),
        config,
    )
    duplicate_patches = classified_patches[
        classified_patches["OriginalFaultRelation"].eq("known_fault_duplicate")
    ].copy().reset_index(drop=True)
    lowcoh_patches = classified_patches[
        classified_patches["KeepInFormalDFN"].eq(1)
    ].copy().reset_index(drop=True)
    duplicate_patches.to_csv(paths["duplicate_csv"], index=False, encoding="utf-8-sig")
    write_patch_vtk(
        paths["duplicate_vtk"],
        duplicate_patches,
        "step7c_known_original_fault_duplicate_panels_raw_time",
    )
    classified_inferred_ids = set(
        pd.to_numeric(classified_patches.get("ComponentID", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)
    )
    step7c_inferred_ids = set(
        pd.to_numeric(lowcoh_patches.get("ComponentID", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)
    )
    classified_component_coverage_fraction = float(
        len(upstream_inferred_ids & classified_inferred_ids) / max(len(upstream_inferred_ids), 1)
    )
    formal_component_retention_fraction = float(
        len(upstream_inferred_ids & step7c_inferred_ids) / max(len(upstream_inferred_ids), 1)
    )
    lowcoh_patches.to_csv(paths["lowcoh_csv"], index=False, encoding="utf-8-sig")
    write_patch_vtk(paths["lowcoh_vtk"], lowcoh_patches, "step7c_large_inferred_fault_surfaces_raw_time")
    patch_df = lowcoh_patches.copy()
    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit_cols = [
        "PatchID",
        "GenerationStage",
        "SourceType",
        "ConstraintLevel",
        "FaultName",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "HeightTimeMs",
        "AzimuthDeg",
        "DipDeg",
        "PatchAreaM2",
        "OrientationSource",
        "SizeRule",
        "BandContinuityMode",
        "ComponentID",
        "ComponentPanelCount",
        "WindowValidationMode",
        "WindowVoxelFraction",
        "CenterInWindow",
        "OriginalFaultRelation",
        "OriginalFaultCenterDistanceM",
        "OriginalFaultVertexMinDistanceM",
        "OriginalFaultVertexNearFraction",
        "OriginalFaultOrientationDifferenceDeg",
    ]
    patch_df[[col for col in audit_cols if col in patch_df.columns]].to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    window_validation_retained = (
        lowcoh_patches[lowcoh_patches["CenterInWindow"].fillna(0).eq(0)].copy()
        if "CenterInWindow" in lowcoh_patches.columns
        else pd.DataFrame()
    )
    window_validation = {
        "min_window_voxel_fraction": float(config.get("inferred_surface_min_window_voxel_fraction", 0.90)),
        "mode_counts": {
            str(key): int(value)
            for key, value in lowcoh_patches.get("WindowValidationMode", pd.Series(dtype=str)).value_counts(dropna=False).items()
        },
        "center_outside_window_retained_count": int(len(window_validation_retained)),
        "center_outside_window_retained_surface_ids": sorted(
            pd.to_numeric(window_validation_retained.get("ComponentID", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        ),
    }
    unified_write = write_unified_dfn_vtk(
        paths["raw_vtk"],
        paths["lowcoh_vtk"],
        paths["original_surface_vtk"],
    )
    unified_summary = unified_vtk_summary(paths["raw_vtk"])
    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "generation_logic": "immutable_demo_original_fault_surfaces_plus_t4_t7_seismic_inferred_fault_panels",
        "inputs": {
            "original_fault_surface_vtk": str(source_surface),
            "original_fault_manifest_csv": str(source_manifest),
            "fault_patch_overlap_csv": str(Path(config["fault_patch_overlap_csv"]).resolve()),
            "large_prior_sgy": str(Path(config["large_prior_sgy"]).resolve()),
            "large_mask_sgy": str(Path(config["large_mask_sgy"]).resolve()) if config.get("large_mask_sgy") else "",
            "large_prior_components_npz": str(Path(config["large_prior_components_npz"]).resolve()),
            "large_component_summary_csv": str(Path(config["large_component_summary_csv"]).resolve()),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "original_fault_source_mode": "demo_xy_selected_complete_source_triangles; no_horizon_clip; no_patch_conversion",
        "original_fault_unit_count": int(len(manifest_df)),
        "original_fault_name_count": int(manifest_df["FaultName"].nunique()) if "FaultName" in manifest_df.columns else 0,
        "original_fault_source_sha256": source_surface_hash,
        "original_fault_copied_sha256": copied_surface_hash,
        "formal_original_fault_patch_count": 0,
        "formal_original_fault_triangle_count": int(unified_summary["original_fault_cell_count"]),
        "inferred_fault_surface_patch_count": int(len(lowcoh_patches)),
        "window_validation": window_validation,
        "inferred_known_fault_duplicate_patch_count": int(len(duplicate_patches)),
        "inferred_original_fault_classification": inferred_classification,
        "patch_count": int(len(patch_df)),
        "unified_vtk": {**unified_write, **unified_summary},
        "damage_zone_included_in_formal_dfn": False,
        "horizon_contract_qc": {
            "original_fault_surface": "not_applied_by_contract",
            "inferred_lowcoh": lowcoh_horizon_qc,
        },
        "source_type_counts": {str(k): int(v) for k, v in patch_df["SourceType"].value_counts(dropna=False).items()},
        "patch_stats": {
            "length_m": finite_stats(patch_df["LengthM"]),
            "height_time_ms": finite_stats(patch_df["HeightTimeMs"]),
            "area_m2": finite_stats(patch_df["PatchAreaM2"]),
            "azimuth_deg": finite_stats(patch_df["AzimuthDeg"]),
            "dip_deg": finite_stats(patch_df["DipDeg"]),
        },
        "inferred_surface_stats": {
            "component_count": int(lowcoh_patches["ComponentID"].nunique()) if "ComponentID" in lowcoh_patches.columns and len(lowcoh_patches) else 0,
            "panel_count": int(len(lowcoh_patches)),
            "upstream_component_count": int(len(upstream_inferred_ids)),
            "classified_upstream_component_count": int(len(upstream_inferred_ids & classified_inferred_ids)),
            "classified_upstream_component_coverage_fraction": classified_component_coverage_fraction,
            "formal_retained_upstream_component_count": int(len(upstream_inferred_ids & step7c_inferred_ids)),
            "formal_component_retention_fraction": formal_component_retention_fraction,
            "candidate_branch_counts": (
                {str(k): int(v) for k, v in lowcoh_patches["CandidateBranch"].value_counts(dropna=False).items()}
                if "CandidateBranch" in lowcoh_patches.columns
                else {}
            ),
            "panel_count_per_component": {
                str(k): int(v)
                for k, v in lowcoh_patches.groupby("ComponentID").size().items()
            }
            if "ComponentID" in lowcoh_patches.columns and len(lowcoh_patches)
            else {},
        },
        "checks": {
            "has_large_patches": len(patch_df) > 0,
            "original_surface_vtk_exists": paths["original_surface_vtk"].exists(),
            "original_manifest_csv_exists": paths["original_manifest_csv"].exists(),
            "original_surface_transport_hash_preserved": source_surface_hash == copied_surface_hash,
            "original_faults_excluded_from_patch_csv": not patch_df["SourceType"].astype(str).str.contains("original_fault").any(),
            "lowcoh_vtk_exists": paths["lowcoh_vtk"].exists(),
            "duplicate_classification_accounting_closed": bool(
                int(inferred_classification["input_count"])
                == int(inferred_classification["formal_count"]) + int(inferred_classification["duplicate_count"])
            ),
            "known_fault_duplicates_excluded_from_formal_csv": bool(
                not patch_df.get("OriginalFaultRelation", pd.Series(dtype=str)).eq("known_fault_duplicate").any()
            ),
            "raw_vtk_exists": paths["raw_vtk"].exists(),
            "csv_exists": paths["dfn_csv"].exists(),
            "unified_vtk_has_predicted_and_original_groups": bool(
                int(unified_summary["predicted_cell_count"]) == len(patch_df)
                and int(unified_summary["original_fault_cell_count"]) > 0
            ),
            "unified_vtk_preserves_original_fault_geometry": bool(
                unified_summary["original_fault_geometry_fingerprint"] == original_geometry_fingerprint
            ),
            "original_fault_render_area_uses_predicted_median": bool(
                unified_write["original_fault_render_area_policy"]
                == "predicted_patch_area_median_not_physical_triangle_area"
            ),
            "formal_dfn_excludes_damage_zone": True,
            "covers_all_step6c_inferred_components": bool(
                not upstream_inferred_ids or classified_component_coverage_fraction >= 1.0
            ),
        },
    }
    summary["status"] = "pass" if all(bool(v) for v in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7c-large] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7c-large] original surface: {paths['original_surface_vtk']}", flush=True)
    print(f"[step7c-large] unified VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7c-large] status={summary['status']} patch_count={len(patch_df)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
