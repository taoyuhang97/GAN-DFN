# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv

from build_regional_fault_panels import collect_region_fault_patch_paths
from fault_postfusion_common import (
    DEFAULT_DOCX_PATH,
    FAULT_ACTION_TEXT_TO_CODE,
    PATCH_ORIGIN_TEXT_TO_CODE,
    append_lines_to_docx,
    fit_plane_from_points,
    normalize_vector,
    parse_fault_patch_file_info,
    write_legacy_vtk_polygons_preserve_patch_area,
    write_csv_utf8,
    write_json,
)

SURFACE_FACET_NORMAL_TOL_DEG = 8.0
SURFACE_FACET_MIN_CELLS = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fault surface fragments directly from raw fault patch files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fault-patches-root", type=Path, required=True)
    parser.add_argument("--block-x-start", type=int, required=True)
    parser.add_argument("--block-x-end", type=int, required=True)
    parser.add_argument("--block-y-start", type=int, required=True)
    parser.add_argument("--block-y-end", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=f"fault_surface_fragments_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--surface-max-strike", type=float, default=70.0)
    parser.add_argument("--surface-max-dip", type=float, default=18.0)
    parser.add_argument("--surface-min-fragment-area", type=float, default=1.0)
    parser.add_argument("--surface-elongate-ratio", type=float, default=2.8)
    parser.add_argument("--surface-normal-pad", type=float, default=30.0)
    parser.add_argument("--surface-gap-ratio", type=float, default=0.95)
    parser.add_argument("--surface-display-offset-ms", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def ensure_fault_polydata(path: Path) -> pv.PolyData:
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.clean().triangulate()
    if not isinstance(mesh, pv.PolyData) or mesh.n_cells <= 0:
        raise ValueError(f"invalid fault patch mesh: {path}")
    return mesh


def parse_faces_to_polygons(faces: np.ndarray) -> list[list[int]]:
    arr = np.asarray(faces, dtype=int).ravel()
    polygons: list[list[int]] = []
    idx = 0
    while idx < len(arr):
        vertex_count = int(arr[idx])
        idx += 1
        polygon = [int(value) for value in arr[idx: idx + vertex_count]]
        idx += vertex_count
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"vectors must have shape (n, 3), got {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return arr / norms


def build_cell_edge_adjacency(polygons: list[list[int]]) -> list[list[int]]:
    edge_to_cells: dict[tuple[int, int], list[int]] = {}
    for cell_idx, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        vertex_count = len(polygon)
        for vertex_idx in range(vertex_count):
            left = int(polygon[vertex_idx])
            right = int(polygon[(vertex_idx + 1) % vertex_count])
            edge = (left, right) if left < right else (right, left)
            edge_to_cells.setdefault(edge, []).append(cell_idx)

    adjacency: list[list[int]] = [[] for _ in range(len(polygons))]
    for linked_cells in edge_to_cells.values():
        if len(linked_cells) < 2:
            continue
        for pos, cell_idx in enumerate(linked_cells):
            for other_idx in linked_cells[pos + 1:]:
                adjacency[cell_idx].append(int(other_idx))
                adjacency[int(other_idx)].append(int(cell_idx))
    return adjacency


def split_patch_into_facet_components(
    local_poly: pv.PolyData,
    normal_tol_deg: float = SURFACE_FACET_NORMAL_TOL_DEG,
    min_component_cells: int = SURFACE_FACET_MIN_CELLS,
) -> list[np.ndarray]:
    tri_poly = local_poly.extract_surface().clean().triangulate()
    if not isinstance(tri_poly, pv.PolyData) or tri_poly.n_cells <= 0:
        return []

    polygons = parse_faces_to_polygons(np.asarray(tri_poly.faces, dtype=int))
    if not polygons:
        return []
    if len(polygons) == 1:
        return [np.asarray([0], dtype=int)]

    normals_poly = tri_poly.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=False,
        consistent_normals=False,
        inplace=False,
    )
    cell_normals = normalize_rows(np.asarray(normals_poly.cell_data["Normals"], dtype=float))
    adjacency = build_cell_edge_adjacency(polygons)
    cos_tol = float(np.cos(np.deg2rad(float(normal_tol_deg))))
    visited = np.zeros(len(polygons), dtype=bool)
    components: list[np.ndarray] = []

    for start_idx in range(len(polygons)):
        if visited[start_idx]:
            continue
        queue = [int(start_idx)]
        visited[start_idx] = True
        component_cells: list[int] = []
        while queue:
            cell_idx = int(queue.pop())
            component_cells.append(cell_idx)
            base_normal = cell_normals[cell_idx]
            for nb_idx in adjacency[cell_idx]:
                nb = int(nb_idx)
                if visited[nb]:
                    continue
                dot_value = abs(float(np.dot(base_normal, cell_normals[nb])))
                if dot_value < cos_tol:
                    continue
                visited[nb] = True
                queue.append(nb)
        if len(component_cells) >= int(min_component_cells):
            components.append(np.asarray(sorted(component_cells), dtype=int))

    if components:
        return components

    return [np.arange(len(polygons), dtype=int)]


def world_to_local_points(
    points: np.ndarray,
    center: np.ndarray,
    strike_vec: np.ndarray,
    dip_vec: np.ndarray,
    normal_vec: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    return np.column_stack(
        [
            rel @ np.asarray(strike_vec, dtype=float),
            rel @ np.asarray(dip_vec, dtype=float),
            rel @ np.asarray(normal_vec, dtype=float),
        ]
    )


def local_to_world_points(
    local_points: np.ndarray,
    center: np.ndarray,
    strike_vec: np.ndarray,
    dip_vec: np.ndarray,
    normal_vec: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_points, dtype=float)
    return (
        np.asarray(center, dtype=float)
        + local[:, [0]] * np.asarray(strike_vec, dtype=float)
        + local[:, [1]] * np.asarray(dip_vec, dtype=float)
        + local[:, [2]] * np.asarray(normal_vec, dtype=float)
    )


def transform_mesh_to_local(
    poly: pv.PolyData,
    center: np.ndarray,
    strike_vec: np.ndarray,
    dip_vec: np.ndarray,
    normal_vec: np.ndarray,
) -> pv.PolyData:
    local_poly = poly.copy(deep=True)
    local_poly.points = world_to_local_points(np.asarray(poly.points, dtype=float), center, strike_vec, dip_vec, normal_vec)
    return local_poly


def transform_mesh_to_world(
    local_poly: pv.PolyData,
    center: np.ndarray,
    strike_vec: np.ndarray,
    dip_vec: np.ndarray,
    normal_vec: np.ndarray,
) -> pv.PolyData:
    world_poly = local_poly.copy(deep=True)
    world_poly.points = local_to_world_points(np.asarray(local_poly.points, dtype=float), center, strike_vec, dip_vec, normal_vec)
    return world_poly.clean().triangulate()


def shrink_local_fragment_xy(local_poly: pv.PolyData, gap_ratio: float) -> pv.PolyData:
    ratio = float(np.clip(float(gap_ratio), 0.75, 1.0))
    if ratio >= 0.999:
        return local_poly
    shrunk_poly = local_poly.copy(deep=True)
    points = np.asarray(shrunk_poly.points, dtype=float).copy()
    if len(points) == 0:
        return shrunk_poly
    center_xy = points[:, :2].mean(axis=0)
    points[:, 0] = center_xy[0] + ratio * (points[:, 0] - center_xy[0])
    points[:, 1] = center_xy[1] + ratio * (points[:, 1] - center_xy[1])
    shrunk_poly.points = points
    return shrunk_poly.clean().triangulate()


def build_fault_surface_group_seed(
    fault_name: str,
    azimuth_deg: float,
    dip_deg: float,
) -> int:
    az_bin = int(round(float(azimuth_deg) / max(float(SURFACE_FACET_NORMAL_TOL_DEG), 1e-6)))
    dip_bin = int(round(float(dip_deg) / max(float(SURFACE_FACET_NORMAL_TOL_DEG), 1e-6)))
    text_seed = sum(ord(ch) for ch in str(fault_name))
    return int(text_seed * 10007 + az_bin * 257 + dip_bin * 17)


def build_fragment_display_offset(
    normal_vec: np.ndarray,
    display_offset_ms: float,
    display_sign: float,
) -> np.ndarray:
    offset_ms = abs(float(display_offset_ms))
    if offset_ms <= 1e-8 or abs(float(display_sign)) <= 1e-8:
        return np.zeros(3, dtype=float)
    normal = normalize_vector(np.asarray(normal_vec, dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    if abs(float(normal[2])) <= 1e-6:
        return np.array([0.0, 0.0, float(display_sign) * offset_ms], dtype=float)
    local_normal_shift = offset_ms / abs(float(normal[2]))
    return np.array([0.0, 0.0, float(display_sign) * local_normal_shift], dtype=float)


def apply_fragment_display_transform(
    local_poly: pv.PolyData,
    gap_ratio: float,
    display_offset_ms: float,
    fragment_seed: int,
    normal_vec: np.ndarray,
) -> pv.PolyData:
    transformed = local_poly.copy(deep=True)
    parity = int(fragment_seed) % 2
    display_sign = 1.0 if parity == 0 else -1.0
    offset = build_fragment_display_offset(
        normal_vec=np.asarray(normal_vec, dtype=float),
        display_offset_ms=float(display_offset_ms),
        display_sign=display_sign,
    )
    if float(np.linalg.norm(offset)) <= 1e-8:
        return transformed
    shifted_poly = transformed.copy(deep=True)
    shifted_poly.points = np.asarray(shifted_poly.points, dtype=float) + offset
    return shifted_poly.clean().triangulate()


def extract_fragment_meshes(
    patch_path: Path,
    surface_max_strike: float,
    surface_max_dip: float,
    surface_min_fragment_area: float,
    surface_elongate_ratio: float,
    surface_normal_pad: float,
    surface_gap_ratio: float,
    surface_display_offset_ms: float,
) -> list[dict[str, Any]]:
    poly = ensure_fault_polydata(Path(patch_path))
    points = np.asarray(poly.points, dtype=float)
    plane = fit_plane_from_points(points)
    center = np.asarray(plane["center"], dtype=float)
    strike_vec = normalize_vector(np.asarray(plane["strike_vec"], dtype=float), fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    dip_vec = normalize_vector(np.asarray(plane["dip_vec"], dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    normal_vec = normalize_vector(np.asarray(plane["normal"], dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))

    local_poly = transform_mesh_to_local(poly, center, strike_vec, dip_vec, normal_vec)
    patch_info = parse_fault_patch_file_info(Path(patch_path))
    component_cell_ids = split_patch_into_facet_components(local_poly)
    facet_component_count = int(len(component_cell_ids))
    split_applied = int(facet_component_count > 1)
    fragment_records: list[dict[str, Any]] = []

    tri_local_poly = local_poly.extract_surface().clean().triangulate()
    if not isinstance(tri_local_poly, pv.PolyData) or tri_local_poly.n_cells <= 0:
        return []

    for component_idx, cell_ids in enumerate(component_cell_ids):
        component = tri_local_poly.extract_cells(np.asarray(cell_ids, dtype=int))
        component = component.extract_surface().clean().triangulate()
        if not isinstance(component, pv.PolyData) or component.n_cells <= 0:
            continue
        if float(component.area) < float(surface_min_fragment_area):
            continue

        component_world_raw = transform_mesh_to_world(component, center, strike_vec, dip_vec, normal_vec)
        component_plane_raw = fit_plane_from_points(np.asarray(component_world_raw.points, dtype=float))
        fragment_seed = build_fault_surface_group_seed(
            fault_name=str(patch_info["fault_name"]),
            azimuth_deg=float(component_plane_raw["strike_deg"]),
            dip_deg=float(component_plane_raw["dip_deg"]),
        )
        display_local_fragment = apply_fragment_display_transform(
            local_poly=component,
            gap_ratio=float(surface_gap_ratio),
            display_offset_ms=float(surface_display_offset_ms),
            fragment_seed=int(fragment_seed),
            normal_vec=normal_vec,
        )
        world_fragment = transform_mesh_to_world(display_local_fragment, center, strike_vec, dip_vec, normal_vec)
        fragment_points = np.asarray(world_fragment.points, dtype=float)
        if len(fragment_points) < 3 or world_fragment.n_cells <= 0:
            continue

        fragment_plane = fit_plane_from_points(fragment_points)
        fragment_area = float(world_fragment.area)
        display_offset_sign = 1 if int(fragment_seed) % 2 == 0 else -1
        fragment_records.append(
            {
                "mesh": world_fragment,
                "meta": {
                    "FaultName": str(patch_info["fault_name"]),
                    "SourcePatchFile": str(Path(patch_path).name),
                    "SourcePatchPath": str(Path(patch_path)),
                    "SourceUnitID": f"BX{int(patch_info['cell_i'])}_BY{int(patch_info['cell_j'])}",
                    "SourceCellI": int(patch_info["cell_i"]),
                    "SourceCellJ": int(patch_info["cell_j"]),
                    "FaultSurfaceFragmentID": int(component_idx + 1),
                    "FragmentGridI": int(component_idx),
                    "FragmentGridJ": 0,
                    "SourceStrikeCount": int(facet_component_count),
                    "SourceDipCount": 1,
                    "FaultSurfaceSplitApplied": int(split_applied),
                    "CenterX": float(fragment_plane["center"][0]),
                    "CenterY": float(fragment_plane["center"][1]),
                    "CenterTIME": float(fragment_plane["center"][2]),
                    "Azimuth": float(fragment_plane["strike_deg"]),
                    "Dip": float(fragment_plane["dip_deg"]),
                    "PatchLength": float(max(fragment_plane["panel_length"], 1e-6)),
                    "PatchHeight": float(max(fragment_plane["panel_height"], 1e-6)),
                    "PatchArea": float(fragment_area),
                    "PatchArea3D": float(fragment_area),
                    "PolygonCellCount": int(world_fragment.n_cells),
                    "DisplayGapRatio": 1.0,
                    "DisplayOffsetMs": float(surface_display_offset_ms),
                    "DisplayOffsetSign": int(display_offset_sign),
                    "BBoxXMin": float(fragment_points[:, 0].min()),
                    "BBoxXMax": float(fragment_points[:, 0].max()),
                    "BBoxYMin": float(fragment_points[:, 1].min()),
                    "BBoxYMax": float(fragment_points[:, 1].max()),
                    "BBoxZMin": float(fragment_points[:, 2].min()),
                    "BBoxZMax": float(fragment_points[:, 2].max()),
                },
            }
        )
    return fragment_records


def build_surface_vtk_payload(fragment_records: list[dict[str, Any]]) -> dict[str, Any]:
    merged_points: list[list[float]] = []
    merged_polygons: list[list[int]] = []
    cell_data_lists: dict[str, list[Any]] = {
        "Azimuth": [],
        "Dip": [],
        "PatchLength": [],
        "PatchHeight": [],
        "PatchArea": [],
        "Confidence": [],
        "PatchOriginCode": [],
        "FaultActionCode": [],
        "FaultDistanceMs": [],
        "FaultInfluenceWeight": [],
        "FaultPanelID": [],
        "NearestFaultPanelID": [],
        "SourcePatchCount": [],
        "SourceUnitCount": [],
        "ParentPatchCount": [],
        "ReliabilityLevel": [],
        "ConnectionType": [],
        "AggregationMode": [],
        "ScaleClass": [],
        "CorridorSupport": [],
        "IsSupplemented": [],
        "FractureSet": [],
        "FaultSurfaceFragmentID": [],
        "SourceCellI": [],
        "SourceCellJ": [],
        "FaultSurfaceSplitApplied": [],
        "DisplayGapRatio": [],
        "DisplayOffsetMs": [],
        "DisplayOffsetSign": [],
    }
    scalar_types = {
        "Azimuth": "float",
        "Dip": "float",
        "PatchLength": "float",
        "PatchHeight": "float",
        "PatchArea": "float",
        "Confidence": "float",
        "PatchOriginCode": "int",
        "FaultActionCode": "int",
        "FaultDistanceMs": "float",
        "FaultInfluenceWeight": "float",
        "FaultPanelID": "int",
        "NearestFaultPanelID": "int",
        "SourcePatchCount": "int",
        "SourceUnitCount": "int",
        "ParentPatchCount": "int",
        "ReliabilityLevel": "int",
        "ConnectionType": "int",
        "AggregationMode": "int",
        "ScaleClass": "int",
        "CorridorSupport": "int",
        "IsSupplemented": "int",
        "FractureSet": "int",
        "FaultSurfaceFragmentID": "int",
        "SourceCellI": "int",
        "SourceCellJ": "int",
        "FaultSurfaceSplitApplied": "int",
        "DisplayGapRatio": "float",
        "DisplayOffsetMs": "float",
        "DisplayOffsetSign": "int",
    }

    for record in fragment_records:
        mesh = record["mesh"]
        meta = record["meta"]
        points = np.asarray(mesh.points, dtype=float)
        polygons = parse_faces_to_polygons(np.asarray(mesh.faces, dtype=int))
        if len(points) == 0 or not polygons:
            continue
        point_offset = len(merged_points)
        merged_points.extend(points.tolist())
        merged_polygons.extend([[point_offset + int(vertex_idx) for vertex_idx in polygon] for polygon in polygons])
        repeat_count = len(polygons)
        field_values = {
            "Azimuth": float(meta["Azimuth"]),
            "Dip": float(meta["Dip"]),
            "PatchLength": float(meta["PatchLength"]),
            "PatchHeight": float(meta["PatchHeight"]),
            "PatchArea": float(meta["PatchArea"]),
            "Confidence": 0.98,
            "PatchOriginCode": int(PATCH_ORIGIN_TEXT_TO_CODE["fault_surface"]),
            "FaultActionCode": int(FAULT_ACTION_TEXT_TO_CODE["surface"]),
            "FaultDistanceMs": 0.0,
            "FaultInfluenceWeight": 1.0,
            "FaultPanelID": -1,
            "NearestFaultPanelID": -1,
            "SourcePatchCount": 1,
            "SourceUnitCount": 1,
            "ParentPatchCount": 1,
            "ReliabilityLevel": 2,
            "ConnectionType": 0,
            "AggregationMode": 0,
            "ScaleClass": 2,
            "CorridorSupport": 1,
            "IsSupplemented": 0,
            "FractureSet": -1,
            "FaultSurfaceFragmentID": int(meta["FaultSurfaceFragmentID"]),
            "SourceCellI": int(meta["SourceCellI"]),
            "SourceCellJ": int(meta["SourceCellJ"]),
            "FaultSurfaceSplitApplied": int(meta["FaultSurfaceSplitApplied"]),
            "DisplayGapRatio": float(meta["DisplayGapRatio"]),
            "DisplayOffsetMs": float(meta["DisplayOffsetMs"]),
            "DisplayOffsetSign": int(meta["DisplayOffsetSign"]),
        }
        for name, value in field_values.items():
            cell_data_lists[name].extend([value] * repeat_count)

    cell_data = {
        name: np.asarray(values, dtype=int if scalar_types[name] == "int" else float)
        for name, values in cell_data_lists.items()
    }
    return {
        "points": np.asarray(merged_points, dtype=float),
        "polygons": merged_polygons,
        "cell_data": cell_data,
        "scalar_types": scalar_types,
    }


def run_build_fault_surface_fragments(
    fault_patches_root: Path,
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
    output_root: Path,
    run_name: str,
    surface_max_strike: float,
    surface_max_dip: float,
    surface_min_fragment_area: float,
    surface_elongate_ratio: float,
    surface_normal_pad: float,
    surface_gap_ratio: float,
    surface_display_offset_ms: float,
) -> dict[str, Any]:
    run_dir = Path(output_root) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    patch_paths = collect_region_fault_patch_paths(
        fault_patches_root=Path(fault_patches_root),
        block_x_start=int(block_x_start),
        block_x_end=int(block_x_end),
        block_y_start=int(block_y_start),
        block_y_end=int(block_y_end),
    )
    fragment_records: list[dict[str, Any]] = []
    for patch_path in patch_paths:
        fragment_records.extend(
            extract_fragment_meshes(
                patch_path=Path(patch_path),
                surface_max_strike=float(surface_max_strike),
                surface_max_dip=float(surface_max_dip),
                surface_min_fragment_area=float(surface_min_fragment_area),
                surface_elongate_ratio=float(surface_elongate_ratio),
                surface_normal_pad=float(surface_normal_pad),
                surface_gap_ratio=float(surface_gap_ratio),
                surface_display_offset_ms=float(surface_display_offset_ms),
            )
        )
    for global_idx, record in enumerate(fragment_records, start=1):
        record["meta"]["FaultSurfaceFragmentID"] = int(global_idx)

    summary_rows = [record["meta"] for record in fragment_records]
    summary_df = pd.DataFrame(summary_rows)
    output_csv = run_dir / "fault_surface_fragments_summary.csv"
    output_vtk = run_dir / "fault_surface_fragments_raw.vtk"
    write_csv_utf8(summary_df, output_csv)

    payload = build_surface_vtk_payload(fragment_records)
    write_legacy_vtk_polygons_preserve_patch_area(
        path=output_vtk,
        title="regional_fault_surface_fragments",
        points=payload["points"],
        polygons=payload["polygons"],
        cell_data=payload["cell_data"],
        scalar_types=payload["scalar_types"],
    )

    summary = {
        "fault_patches_root": str(fault_patches_root),
        "selected_patch_count": int(len(patch_paths)),
        "fragment_count": int(len(fragment_records)),
        "surface_polygon_count": int(len(payload["polygons"])),
        "surface_fragment_mode": "raw_patch_facet_components",
        "surface_internal_facet_split_enabled": 1,
        "surface_facet_normal_tol_deg": float(SURFACE_FACET_NORMAL_TOL_DEG),
        "surface_facet_min_cells": int(SURFACE_FACET_MIN_CELLS),
        "surface_xy_gap_enabled": 0,
        "surface_vtk": str(output_vtk),
        "summary_csv": str(output_csv),
        "surface_max_strike": float(surface_max_strike),
        "surface_max_dip": float(surface_max_dip),
        "surface_min_fragment_area": float(surface_min_fragment_area),
        "surface_elongate_ratio": float(surface_elongate_ratio),
        "surface_normal_pad": float(surface_normal_pad),
        "surface_gap_ratio": float(surface_gap_ratio),
        "surface_gap_ratio_effective": 1.0,
        "surface_display_offset_ms": float(surface_display_offset_ms),
    }
    summary_path = run_dir / "fault_surface_fragments_summary.json"
    write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / str(args.run_name)
    if run_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output run dir already exists: {run_dir}")
    summary = run_build_fault_surface_fragments(
        fault_patches_root=Path(args.fault_patches_root),
        block_x_start=int(args.block_x_start),
        block_x_end=int(args.block_x_end),
        block_y_start=int(args.block_y_start),
        block_y_end=int(args.block_y_end),
        output_root=Path(args.output_root),
        run_name=str(args.run_name),
        surface_max_strike=float(args.surface_max_strike),
        surface_max_dip=float(args.surface_max_dip),
        surface_min_fragment_area=float(args.surface_min_fragment_area),
        surface_elongate_ratio=float(args.surface_elongate_ratio),
        surface_normal_pad=float(args.surface_normal_pad),
        surface_gap_ratio=float(args.surface_gap_ratio),
        surface_display_offset_ms=float(args.surface_display_offset_ms),
    )
    append_lines_to_docx(
        docx_path=args.docx_path,
        title=f"fault surface fragments {args.run_name}",
        lines=[
            f"fault_patches_root: {args.fault_patches_root}",
            f"block_x_range: {min(args.block_x_start, args.block_x_end)}-{max(args.block_x_start, args.block_x_end)}",
            f"block_y_range: {min(args.block_y_start, args.block_y_end)}-{max(args.block_y_start, args.block_y_end)}",
            f"selected_patch_count: {summary['selected_patch_count']}",
            f"fragment_count: {summary['fragment_count']}",
            f"surface_polygon_count: {summary['surface_polygon_count']}",
            f"surface_fragment_mode: {summary['surface_fragment_mode']}",
            f"surface_facet_normal_tol_deg: {summary['surface_facet_normal_tol_deg']}",
            f"surface_gap_ratio: {summary['surface_gap_ratio']}",
            f"surface_gap_ratio_effective: {summary['surface_gap_ratio_effective']}",
            f"surface_display_offset_ms: {summary['surface_display_offset_ms']}",
            f"surface_vtk: {summary['surface_vtk']}",
            f"summary_csv: {summary['summary_csv']}",
        ],
    )
    print(f"surface_vtk: {summary['surface_vtk']}")
    print(f"fragment_count: {summary['fragment_count']}")
    print(f"surface_polygon_count: {summary['surface_polygon_count']}")


if __name__ == "__main__":
    main()
