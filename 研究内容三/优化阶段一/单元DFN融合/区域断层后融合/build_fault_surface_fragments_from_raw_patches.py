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
    make_patch_row_from_axes,
    normalize_vector,
    parse_fault_patch_file_info,
    write_legacy_vtk_polygons_preserve_patch_area,
    write_csv_utf8,
    write_json,
)

SURFACE_FACET_NORMAL_TOL_DEG = 8.0
SURFACE_FACET_MIN_CELLS = 1


def extract_surface_clean_tri(mesh: pv.DataSet) -> pv.PolyData:
    surface = mesh.extract_surface(algorithm="dataset_surface")
    surface = surface.clean().triangulate()
    if not isinstance(surface, pv.PolyData):
        raise ValueError("extract_surface_clean_tri did not produce PolyData")
    return surface


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
    parser.add_argument("--surface-max-fragment-area-ratio", type=float, default=1500.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def ensure_fault_polydata(path: Path) -> pv.PolyData:
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = extract_surface_clean_tri(mesh)
    else:
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
    tri_poly = extract_surface_clean_tri(local_poly)
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


def extract_local_submesh(tri_local_poly: pv.PolyData, cell_ids: np.ndarray) -> pv.PolyData | None:
    if len(cell_ids) <= 0:
        return None
    submesh = tri_local_poly.extract_cells(np.asarray(cell_ids, dtype=int))
    submesh = extract_surface_clean_tri(submesh)
    if not isinstance(submesh, pv.PolyData) or submesh.n_cells <= 0:
        return None
    return submesh


def split_local_component_by_area_cap(component_local: pv.PolyData, area_cap: float) -> list[pv.PolyData]:
    tri_local_poly = extract_surface_clean_tri(component_local)
    if not isinstance(tri_local_poly, pv.PolyData) or tri_local_poly.n_cells <= 0:
        return []
    if float(area_cap) <= 0.0 or float(tri_local_poly.area) <= float(area_cap):
        return [tri_local_poly]

    cell_centers = tri_local_poly.cell_centers().points
    if cell_centers is None or len(cell_centers) != tri_local_poly.n_cells:
        return [tri_local_poly]
    cell_centers = np.asarray(cell_centers, dtype=float)

    pending: list[np.ndarray] = [np.arange(tri_local_poly.n_cells, dtype=int)]
    leaves: list[pv.PolyData] = []
    while pending:
        current_ids = pending.pop()
        current_mesh = extract_local_submesh(tri_local_poly, current_ids)
        if current_mesh is None:
            continue
        current_area = float(current_mesh.area)
        if current_area <= float(area_cap) or len(current_ids) <= 1:
            leaves.append(current_mesh)
            continue

        current_points = np.asarray(current_mesh.points, dtype=float)
        if len(current_points) == 0:
            leaves.append(current_mesh)
            continue
        span_x = float(current_points[:, 0].max() - current_points[:, 0].min())
        span_y = float(current_points[:, 1].max() - current_points[:, 1].min())
        axis_candidates = [0, 1] if span_x >= span_y else [1, 0]
        split_done = False
        for axis in axis_candidates:
            axis_values = cell_centers[current_ids, axis]
            if len(axis_values) <= 1:
                continue
            for split_value in (float(np.median(axis_values)), float(0.5 * (axis_values.min() + axis_values.max()))):
                left_ids = current_ids[axis_values <= split_value]
                right_ids = current_ids[axis_values > split_value]
                if len(left_ids) == 0 or len(right_ids) == 0:
                    continue
                if len(left_ids) == len(current_ids) or len(right_ids) == len(current_ids):
                    continue
                pending.append(np.asarray(left_ids, dtype=int))
                pending.append(np.asarray(right_ids, dtype=int))
                split_done = True
                break
            if split_done:
                break
        if not split_done:
            leaves.append(current_mesh)
    return leaves


def build_planar_patch_row_from_mesh(
    world_mesh: pv.PolyData,
    display_offset_ms: float,
    fragment_seed: int,
) -> dict[str, Any] | None:
    world_points = np.asarray(world_mesh.points, dtype=float)
    if len(world_points) < 3:
        return None
    plane = fit_plane_from_points(world_points)
    center = np.asarray(plane["center"], dtype=float)
    strike_vec = normalize_vector(np.asarray(plane["strike_vec"], dtype=float), fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    dip_vec = normalize_vector(np.asarray(plane["dip_vec"], dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    normal_vec = normalize_vector(np.asarray(plane["normal"], dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    local_points = world_to_local_points(world_points, center, strike_vec, dip_vec, normal_vec)
    if len(local_points) == 0:
        return None

    strike_min = float(local_points[:, 0].min())
    strike_max = float(local_points[:, 0].max())
    dip_min = float(local_points[:, 1].min())
    dip_max = float(local_points[:, 1].max())
    bbox_length = max(float(strike_max - strike_min), 1e-6)
    bbox_height = max(float(dip_max - dip_min), 1e-6)
    bbox_area = max(bbox_length * bbox_height, 1e-6)
    target_area = max(float(world_mesh.area), 1e-6)
    scale_ratio = float(np.sqrt(np.clip(target_area / bbox_area, 1e-6, 1.0)))
    patch_length = max(bbox_length * scale_ratio, 1e-6)
    patch_height = max(bbox_height * scale_ratio, 1e-6)
    local_center = np.array(
        [
            0.5 * (strike_min + strike_max),
            0.5 * (dip_min + dip_max),
            0.0,
        ],
        dtype=float,
    )
    display_sign = 1.0 if int(fragment_seed) % 2 == 0 else -1.0
    local_center = local_center + build_fragment_display_offset(
        normal_vec=normal_vec,
        display_offset_ms=float(display_offset_ms),
        display_sign=float(display_sign),
    )
    patch_center = local_to_world_points(
        local_center.reshape(1, 3),
        center,
        strike_vec,
        dip_vec,
        normal_vec,
    )[0]
    row = make_patch_row_from_axes(
        center=patch_center,
        u_vec=strike_vec,
        v_vec=dip_vec,
        length=patch_length,
        height=patch_height,
    )
    row["PatchArea"] = float(target_area)
    row["PatchArea3D"] = float(target_area)
    row["PlanarBBoxArea"] = float(bbox_area)
    row["PlanarAreaScaleRatio"] = float(scale_ratio)
    row["DisplayOffsetSign"] = int(display_sign)
    return row


def extract_patch_component_bundles(patch_path: Path) -> list[dict[str, Any]]:
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

    tri_local_poly = extract_surface_clean_tri(local_poly)
    if not isinstance(tri_local_poly, pv.PolyData) or tri_local_poly.n_cells <= 0:
        return []

    bundles: list[dict[str, Any]] = []
    for component_idx, cell_ids in enumerate(component_cell_ids):
        component = tri_local_poly.extract_cells(np.asarray(cell_ids, dtype=int))
        component = extract_surface_clean_tri(component)
        if not isinstance(component, pv.PolyData) or component.n_cells <= 0:
            continue
        bundles.append(
            {
                "component_local": component,
                "center": center,
                "strike_vec": strike_vec,
                "dip_vec": dip_vec,
                "normal_vec": normal_vec,
                "patch_info": patch_info,
                "patch_path": str(Path(patch_path)),
                "facet_component_count": facet_component_count,
                "facet_component_idx": int(component_idx),
                "facet_split_applied": split_applied,
            }
        )
    return bundles


def build_fragment_records_from_bundle(
    bundle: dict[str, Any],
    surface_min_fragment_area: float,
    surface_display_offset_ms: float,
    surface_area_cap: float,
) -> list[dict[str, Any]]:
    fragment_records: list[dict[str, Any]] = []
    component_local = bundle["component_local"]
    center = np.asarray(bundle["center"], dtype=float)
    strike_vec = np.asarray(bundle["strike_vec"], dtype=float)
    dip_vec = np.asarray(bundle["dip_vec"], dtype=float)
    normal_vec = np.asarray(bundle["normal_vec"], dtype=float)
    patch_info = dict(bundle["patch_info"])
    patch_path = str(bundle["patch_path"])
    facet_component_count = int(bundle["facet_component_count"])
    facet_component_idx = int(bundle["facet_component_idx"])
    facet_split_applied = int(bundle["facet_split_applied"])
    area_split_applied = int(float(component_local.area) > float(surface_area_cap) > 0.0)

    for split_idx, submesh_local in enumerate(split_local_component_by_area_cap(component_local, float(surface_area_cap))):
        if not isinstance(submesh_local, pv.PolyData) or submesh_local.n_cells <= 0:
            continue
        fragment_area = float(submesh_local.area)
        if fragment_area < float(surface_min_fragment_area):
            continue
        submesh_world = transform_mesh_to_world(submesh_local, center, strike_vec, dip_vec, normal_vec)
        if not isinstance(submesh_world, pv.PolyData) or submesh_world.n_cells <= 0:
            continue
        fragment_points = np.asarray(submesh_world.points, dtype=float)
        if len(fragment_points) < 3:
            continue
        fragment_plane = fit_plane_from_points(fragment_points)
        fragment_seed = build_fault_surface_group_seed(
            fault_name=str(patch_info["fault_name"]),
            azimuth_deg=float(fragment_plane["strike_deg"]),
            dip_deg=float(fragment_plane["dip_deg"]),
        ) + int(split_idx)
        planar_row = build_planar_patch_row_from_mesh(
            world_mesh=submesh_world,
            display_offset_ms=float(surface_display_offset_ms),
            fragment_seed=int(fragment_seed),
        )
        if planar_row is None:
            continue
        planar_vertices = np.asarray(
            [
                [float(planar_row[f"V{vertex_idx}X"]), float(planar_row[f"V{vertex_idx}Y"]), float(planar_row[f"V{vertex_idx}Z"])]
                for vertex_idx in range(1, 5)
            ],
            dtype=float,
        )
        fragment_records.append(
            {
                "meta": {
                    **planar_row,
                    "FaultName": str(patch_info["fault_name"]),
                    "SourcePatchFile": str(Path(patch_path).name),
                    "SourcePatchPath": str(patch_path),
                    "SourceUnitID": f"BX{int(patch_info['cell_i'])}_BY{int(patch_info['cell_j'])}",
                    "SourceCellI": int(patch_info["cell_i"]),
                    "SourceCellJ": int(patch_info["cell_j"]),
                    "FaultSurfaceFragmentID": int(split_idx + 1),
                    "FragmentGridI": int(facet_component_idx),
                    "FragmentGridJ": int(split_idx),
                    "SourceStrikeCount": int(facet_component_count),
                    "SourceDipCount": 1,
                    "FaultSurfaceSplitApplied": int(facet_split_applied),
                    "FaultSurfaceAreaSplitApplied": int(area_split_applied),
                    "CenterX": float(planar_row["CenterX"]),
                    "CenterY": float(planar_row["CenterY"]),
                    "CenterTIME": float(planar_row["CenterTIME"]),
                    "Azimuth": float(fragment_plane["strike_deg"]),
                    "Dip": float(fragment_plane["dip_deg"]),
                    "PatchArea": float(fragment_area),
                    "PatchArea3D": float(fragment_area),
                    "PolygonCellCount": int(submesh_world.n_cells),
                    "PlanarVertexCount": 4,
                    "DisplayGapRatio": 1.0,
                    "DisplayOffsetMs": float(surface_display_offset_ms),
                    "DisplayOffsetSign": int(planar_row["DisplayOffsetSign"]),
                    "BBoxXMin": float(planar_vertices[:, 0].min()),
                    "BBoxXMax": float(planar_vertices[:, 0].max()),
                    "BBoxYMin": float(planar_vertices[:, 1].min()),
                    "BBoxYMax": float(planar_vertices[:, 1].max()),
                    "BBoxZMin": float(planar_vertices[:, 2].min()),
                    "BBoxZMax": float(planar_vertices[:, 2].max()),
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
        "FaultSurfaceAreaSplitApplied": [],
        "PlanarBBoxArea": [],
        "PlanarAreaScaleRatio": [],
        "PlanarVertexCount": [],
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
        "FaultSurfaceAreaSplitApplied": "int",
        "PlanarBBoxArea": "float",
        "PlanarAreaScaleRatio": "float",
        "PlanarVertexCount": "int",
        "DisplayGapRatio": "float",
        "DisplayOffsetMs": "float",
        "DisplayOffsetSign": "int",
    }

    for record in fragment_records:
        meta = record["meta"]
        points = np.asarray(
            [
                [float(meta[f"V{vertex_idx}X"]), float(meta[f"V{vertex_idx}Y"]), float(meta[f"V{vertex_idx}Z"])]
                for vertex_idx in range(1, 5)
            ],
            dtype=float,
        )
        polygons = [list(range(4))]
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
            "FaultSurfaceAreaSplitApplied": int(meta["FaultSurfaceAreaSplitApplied"]),
            "PlanarBBoxArea": float(meta["PlanarBBoxArea"]),
            "PlanarAreaScaleRatio": float(meta["PlanarAreaScaleRatio"]),
            "PlanarVertexCount": int(meta["PlanarVertexCount"]),
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
    surface_max_fragment_area_ratio: float,
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
    component_bundles: list[dict[str, Any]] = []
    component_areas: list[float] = []
    for patch_path in patch_paths:
        bundles = extract_patch_component_bundles(Path(patch_path))
        component_bundles.extend(bundles)
        for bundle in bundles:
            component_area = float(bundle["component_local"].area)
            if component_area >= float(surface_min_fragment_area):
                component_areas.append(component_area)
    surface_area_cap_base = float(min(component_areas)) if component_areas else float(surface_min_fragment_area)
    surface_area_cap = max(float(surface_area_cap_base) * float(surface_max_fragment_area_ratio), float(surface_min_fragment_area))

    fragment_records: list[dict[str, Any]] = []
    for bundle in component_bundles:
        fragment_records.extend(
            build_fragment_records_from_bundle(
                bundle=bundle,
                surface_min_fragment_area=float(surface_min_fragment_area),
                surface_display_offset_ms=float(surface_display_offset_ms),
                surface_area_cap=float(surface_area_cap),
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
        "component_bundle_count": int(len(component_bundles)),
        "fragment_count": int(len(fragment_records)),
        "surface_polygon_count": int(len(payload["polygons"])),
        "surface_fragment_mode": "facet_components_area_capped_planar_patches",
        "surface_internal_facet_split_enabled": 1,
        "surface_internal_area_cap_split_enabled": 1,
        "surface_facet_normal_tol_deg": float(SURFACE_FACET_NORMAL_TOL_DEG),
        "surface_facet_min_cells": int(SURFACE_FACET_MIN_CELLS),
        "surface_xy_gap_enabled": 0,
        "surface_planarized": 1,
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
        "surface_max_fragment_area_ratio": float(surface_max_fragment_area_ratio),
        "surface_area_cap_base": float(surface_area_cap_base),
        "surface_area_cap": float(surface_area_cap),
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
        surface_max_fragment_area_ratio=float(args.surface_max_fragment_area_ratio),
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
            f"surface_max_fragment_area_ratio: {summary['surface_max_fragment_area_ratio']}",
            f"surface_area_cap_base: {summary['surface_area_cap_base']}",
            f"surface_area_cap: {summary['surface_area_cap']}",
            f"surface_vtk: {summary['surface_vtk']}",
            f"summary_csv: {summary['summary_csv']}",
        ],
    )
    print(f"surface_vtk: {summary['surface_vtk']}")
    print(f"fragment_count: {summary['fragment_count']}")
    print(f"surface_polygon_count: {summary['surface_polygon_count']}")


if __name__ == "__main__":
    main()
