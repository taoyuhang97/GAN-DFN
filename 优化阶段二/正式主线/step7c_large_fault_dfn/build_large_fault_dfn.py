from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
LEGACY_STEP7B_DIR = CURRENT_DIR.parent / "step7b_initial_dfn_3d"
OLD_FAULT_POSTFUSION_DIR = REPO_ROOT / "研究内容三/优化阶段一/单元DFN融合/区域断层后融合"
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_step7c_large_v1.json"
if str(LEGACY_STEP7B_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_STEP7B_DIR))
if str(OLD_FAULT_POSTFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_FAULT_POSTFUSION_DIR))

import build_initial_dfn_from_3d_density_sgy as legacy  # noqa: E402
from build_fault_surface_fragments_from_raw_patches import run_build_fault_surface_fragments  # noqa: E402
from build_regional_fault_panels import run_build_regional_fault_panels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step7C large fault/fault-zone DFN from original fault sticks and Step6C prior.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "fault_surface_vtk": output_dir / "large_fault_surface_raw_time.vtk",
        "fault_panel_csv": output_dir / "large_original_fault_panel_patches.csv",
        "fault_only_csv": output_dir / "large_fault_only_and_influence_patches.csv",
        "fault_only_vtk": output_dir / "large_fault_only_and_influence_raw_time.vtk",
        "lowcoh_csv": output_dir / "large_lowcoh_component_panel_patches.csv",
        "lowcoh_vtk": output_dir / "large_lowcoh_component_panels_raw_time.vtk",
        "dfn_csv": output_dir / "large_fault_and_damage_patches.csv",
        "raw_vtk": output_dir / "large_fault_and_damage_raw_time.vtk",
        "audit_csv": output_dir / "large_generation_audit.csv",
        "summary_json": output_dir / "large_fault_dfn_summary.json",
    }


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
            layer = "沙三段" if center[2] < float(config.get("shasi_time_split_ms", 2820.0)) else "沙四段"
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
    surfaces = legacy.attach_surface_grids(Path(config["layer_dir"]).resolve(), grid["x_values"], grid["y_values"])
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

    rows: list[dict[str, Any]] = []
    patch_idx = 0
    for component_ord, comp_row in enumerate(summary_df.itertuples(index=False), start=1):
        comp_id = int(comp_row.component_id)
        component_where = np.where(component_id_grid == comp_id)
        if component_id_grid.ndim == 2:
            trace_idx, tt = component_where
            if len(trace_idx) < min_panel_voxels:
                continue
            points = np.column_stack([x_values[trace_idx], y_values[trace_idx], samples[tt]]).astype(float)
        elif component_id_grid.ndim == 3:
            yy, xx, tt = component_where
            if len(xx) < min_panel_voxels:
                continue
            ix = npz["ix"].astype(np.int32)
            iy = npz["iy"].astype(np.int32)
            x_axis = np.zeros(int(ix.max()) + 1, dtype=float)
            y_axis = np.zeros(int(iy.max()) + 1, dtype=float)
            for idx in range(len(x_axis)):
                x_axis[idx] = float(np.median(x_values[ix == idx]))
            for idx in range(len(y_axis)):
                y_axis[idx] = float(np.median(y_values[iy == idx]))
            points = np.column_stack([x_axis[xx], y_axis[yy], samples[tt]]).astype(float)
        else:
            raise ValueError(f"unsupported inferred_component_id shape: {component_id_grid.shape}")
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
            layer = "沙三段" if seg_center[2] < float(config.get("shasi_time_split_ms", 2820.0)) else "沙四段"
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
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    workspace = output_dir / "surface_panel_workspace"
    ensure_dir(workspace)
    paths = output_paths(output_dir)
    fault_df = read_fault_sticks(Path(config["fault_stick_dat"]).resolve(), config)
    print(f"[step7c-large] clipped fault stick rows={len(fault_df)}", flush=True)

    fault_patches_root = Path(config["fault_patches_root"]).resolve()
    if not fault_patches_root.exists():
        raise FileNotFoundError(f"fault_patches_root not found: {fault_patches_root}")
    cell_range = infer_fault_cell_range(config)
    print(f"[step7c-large] selected segmented fault cell range={cell_range}", flush=True)
    panel_summary = run_build_regional_fault_panels(
        fault_patches_root=fault_patches_root,
        block_x_start=cell_range["block_x_start"],
        block_x_end=cell_range["block_x_end"],
        block_y_start=cell_range["block_y_start"],
        block_y_end=cell_range["block_y_end"],
        output_root=workspace,
        run_name="regional_fault_panels",
        panel_merge_xy=float(config.get("panel_merge_xy", 220.0)),
        panel_merge_time=float(config.get("panel_merge_time", 35.0)),
        panel_strike_tol=float(config.get("panel_strike_tol", 20.0)),
        panel_dip_tol=float(config.get("panel_dip_tol", 15.0)),
    )
    surface_summary = run_build_fault_surface_fragments(
        fault_patches_root=fault_patches_root,
        block_x_start=cell_range["block_x_start"],
        block_x_end=cell_range["block_x_end"],
        block_y_start=cell_range["block_y_start"],
        block_y_end=cell_range["block_y_end"],
        output_root=workspace,
        run_name="fault_surface_fragments",
        surface_max_strike=float(config.get("surface_max_strike", 70.0)),
        surface_max_dip=float(config.get("surface_max_dip", 18.0)),
        surface_min_fragment_area=float(config.get("surface_min_fragment_area", 1.0)),
        surface_elongate_ratio=float(config.get("surface_elongate_ratio", 2.8)),
        surface_normal_pad=float(config.get("surface_normal_pad", 30.0)),
        surface_gap_ratio=float(config.get("surface_gap_ratio", 0.95)),
        surface_display_offset_ms=float(config.get("surface_display_offset_ms", 0.6)),
        surface_max_fragment_area_ratio=float(config.get("surface_max_fragment_area_ratio", 1500.0)),
    )
    shutil.copy2(Path(surface_summary["surface_vtk"]), paths["fault_surface_vtk"])
    surface_fragment_df = standard_surface_fragment_rows(Path(surface_summary["summary_csv"]))
    panel_influence_df = build_fault_panel_and_influence_rows(Path(panel_summary["panel_csv"]), config)
    if panel_influence_df.empty:
        raise RuntimeError("no real segmented fault panel rows generated from raw fault patches")
    fault_panels_only = panel_influence_df[panel_influence_df["SourceType"].astype(str).eq("large_original_fault_panel")].copy()
    damage_patches = panel_influence_df[panel_influence_df["SourceType"].astype(str).eq("large_original_fault_damage_zone")].copy()
    fault_patches = pd.concat([surface_fragment_df, damage_patches], ignore_index=True)
    if fault_patches.empty:
        fault_patches = panel_influence_df.copy()
    lowcoh_patches = build_lowcoh_component_panels(config)
    fault_patches.to_csv(paths["fault_only_csv"], index=False, encoding="utf-8-sig")
    fault_panels_only.to_csv(paths["fault_panel_csv"], index=False, encoding="utf-8-sig")
    lowcoh_patches.to_csv(paths["lowcoh_csv"], index=False, encoding="utf-8-sig")
    write_patch_vtk(paths["fault_only_vtk"], fault_patches, "step7c_large_original_fault_and_influence_raw_time")
    write_patch_vtk(paths["lowcoh_vtk"], lowcoh_patches, "step7c_large_lowcoh_component_panels_raw_time")
    parts = [df for df in [fault_patches, lowcoh_patches] if not df.empty]
    if not parts:
        raise RuntimeError("no large fault/fault-zone patches generated")
    patch_df = pd.concat(parts, ignore_index=True)
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
    ]
    patch_df[[col for col in audit_cols if col in patch_df.columns]].to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    write_patch_vtk(paths["raw_vtk"], patch_df, "step7c_large_fault_and_damage_raw_time")
    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "generation_logic": "step7c_real_fault_surface_fragments_and_regional_panels_plus_lowcoh_component_panel_groups",
        "inputs": {
            "fault_stick_dat": str(Path(config["fault_stick_dat"]).resolve()),
            "fault_patches_root": str(fault_patches_root),
            "fault_patch_overlap_csv": str(Path(config["fault_patch_overlap_csv"]).resolve()),
            "large_prior_sgy": str(Path(config["large_prior_sgy"]).resolve()),
            "large_mask_sgy": str(Path(config["large_mask_sgy"]).resolve()),
            "large_prior_components_npz": str(Path(config["large_prior_components_npz"]).resolve()),
            "large_component_summary_csv": str(Path(config["large_component_summary_csv"]).resolve()),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "clipped_fault_stick_rows": int(len(fault_df)),
        "selected_fault_cell_range": cell_range,
        "original_fault_source_mode": "raw_fault_patch_surface_fragments_and_regional_panels",
        "regional_fault_panel_summary": {key: str(value) for key, value in panel_summary.items()},
        "surface_fragment_summary": {key: str(value) for key, value in surface_summary.items() if key in {"surface_vtk", "summary_csv", "fragment_count", "selected_patch_count"}},
        "regional_fault_panel_count": int(len(fault_panels_only)),
        "fault_damage_zone_patch_count": int(len(damage_patches)),
        "fault_only_patch_count": int(len(fault_patches)),
        "raw_fault_surface_fragment_count": int(len(surface_fragment_df)),
        "lowcoh_supplement_patch_count": int(len(lowcoh_patches)),
        "patch_count": int(len(patch_df)),
        "source_type_counts": {str(k): int(v) for k, v in patch_df["SourceType"].value_counts(dropna=False).items()},
        "patch_stats": {
            "length_m": finite_stats(patch_df["LengthM"]),
            "height_time_ms": finite_stats(patch_df["HeightTimeMs"]),
            "area_m2": finite_stats(patch_df["PatchAreaM2"]),
            "azimuth_deg": finite_stats(patch_df["AzimuthDeg"]),
            "dip_deg": finite_stats(patch_df["DipDeg"]),
        },
        "lowcoh_panel_group_stats": {
            "component_count": int(lowcoh_patches["ComponentID"].nunique()) if "ComponentID" in lowcoh_patches.columns and len(lowcoh_patches) else 0,
            "panel_count": int(len(lowcoh_patches)),
            "panel_count_per_component": {
                str(k): int(v)
                for k, v in lowcoh_patches.groupby("ComponentID").size().items()
            }
            if "ComponentID" in lowcoh_patches.columns and len(lowcoh_patches)
            else {},
        },
        "checks": {
            "has_large_patches": len(patch_df) > 0,
            "fault_surface_vtk_exists": paths["fault_surface_vtk"].exists(),
            "fault_only_vtk_exists": paths["fault_only_vtk"].exists(),
            "lowcoh_vtk_exists": paths["lowcoh_vtk"].exists(),
            "raw_vtk_exists": paths["raw_vtk"].exists(),
            "csv_exists": paths["dfn_csv"].exists(),
        },
    }
    summary["status"] = "pass" if all(bool(v) for v in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7c-large] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7c-large] VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7c-large] status={summary['status']} patch_count={len(patch_df)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
