# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv

from build_regional_fault_panels import collect_region_fault_patch_paths
from fault_postfusion_common import (
    parse_fault_patch_file_info,
    write_csv_utf8,
    write_json,
    write_legacy_vtk_polygons_preserve_patch_area,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export raw regional fault surfaces as display VTK without planarization or display offsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fault-patches-root", type=Path, required=True)
    parser.add_argument("--block-x-start", type=int, required=True)
    parser.add_argument("--block-x-end", type=int, required=True)
    parser.add_argument("--block-y-start", type=int, required=True)
    parser.add_argument("--block-y-end", type=int, required=True)
    parser.add_argument("--output-vtk", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def parse_faces(faces: np.ndarray) -> list[list[int]]:
    values = np.asarray(faces, dtype=int).ravel()
    polygons: list[list[int]] = []
    pos = 0
    while pos < len(values):
        vertex_count = int(values[pos])
        pos += 1
        polygon = [int(item) for item in values[pos: pos + vertex_count]]
        pos += vertex_count
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def read_fault_polydata(path: Path) -> pv.PolyData:
    mesh = pv.read(str(path))
    if isinstance(mesh, pv.PolyData):
        poly = mesh
    else:
        poly = mesh.extract_surface(algorithm="dataset_surface")
    if not isinstance(poly, pv.PolyData):
        raise ValueError(f"failed to read fault surface as PolyData: {path}")
    poly = poly.clean()
    if poly.n_cells <= 0 or poly.n_points <= 0:
        raise ValueError(f"empty fault surface: {path}")
    return poly


def cell_areas(poly: pv.PolyData) -> np.ndarray:
    sized = poly.compute_cell_sizes(length=False, area=True, volume=False)
    values = np.asarray(sized.cell_data.get("Area", []), dtype=float)
    if len(values) == poly.n_cells:
        return values
    fallback = float(poly.area) / max(int(poly.n_cells), 1)
    return np.full(poly.n_cells, fallback, dtype=float)


def export_raw_fault_surface(
    fault_patches_root: Path,
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
    output_vtk: Path,
    output_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    patch_paths = collect_region_fault_patch_paths(
        fault_patches_root=Path(fault_patches_root),
        block_x_start=int(block_x_start),
        block_x_end=int(block_x_end),
        block_y_start=int(block_y_start),
        block_y_end=int(block_y_end),
    )
    if not patch_paths:
        raise FileNotFoundError(
            "no fault patch files selected for "
            f"BX{block_x_start}-{block_x_end}, BY{block_y_start}-{block_y_end} under {fault_patches_root}"
        )

    fault_name_to_code: dict[str, int] = {}
    merged_points: list[list[float]] = []
    merged_polygons: list[list[int]] = []
    cell_rows: list[dict[str, Any]] = []

    for patch_idx, patch_path in enumerate(patch_paths, start=1):
        info = parse_fault_patch_file_info(Path(patch_path))
        fault_name = str(info["fault_name"])
        if fault_name not in fault_name_to_code:
            fault_name_to_code[fault_name] = len(fault_name_to_code) + 1
        fault_code = int(fault_name_to_code[fault_name])

        poly = read_fault_polydata(Path(patch_path))
        points = np.asarray(poly.points, dtype=float)
        polygons = parse_faces(np.asarray(poly.faces, dtype=int))
        if len(polygons) != poly.n_cells:
            poly = poly.extract_surface(algorithm="dataset_surface").clean().triangulate()
            points = np.asarray(poly.points, dtype=float)
            polygons = parse_faces(np.asarray(poly.faces, dtype=int))
        areas = cell_areas(poly)
        patch_area = float(np.sum(areas))

        point_offset = len(merged_points)
        merged_points.extend(points.tolist())
        for local_cell_idx, polygon in enumerate(polygons):
            global_polygon = [point_offset + int(point_id) for point_id in polygon]
            merged_polygons.append(global_polygon)
            polygon_points = points[np.asarray(polygon, dtype=int)]
            center = polygon_points.mean(axis=0)
            cell_rows.append(
                {
                    "PatchArea": float(areas[local_cell_idx]) if local_cell_idx < len(areas) else 0.0,
                    "SourcePatchArea": patch_area,
                    "FaultNameCode": fault_code,
                    "SourcePatchIndex": int(patch_idx),
                    "SourceCellI": int(info["cell_i"]),
                    "SourceCellJ": int(info["cell_j"]),
                    "SourceLocalCellID": int(local_cell_idx),
                    "CenterX": float(center[0]),
                    "CenterY": float(center[1]),
                    "CenterTIME": float(center[2]),
                    "VertexCount": int(len(polygon)),
                    "FaultName": fault_name,
                    "SourcePatchFile": Path(patch_path).name,
                    "SourcePatchPath": str(patch_path),
                }
            )

    cell_df = pd.DataFrame(cell_rows)
    cell_data = {
        "PatchArea": cell_df["PatchArea"].to_numpy(dtype=float),
        "SourcePatchArea": cell_df["SourcePatchArea"].to_numpy(dtype=float),
        "FaultNameCode": cell_df["FaultNameCode"].to_numpy(dtype=int),
        "SourcePatchIndex": cell_df["SourcePatchIndex"].to_numpy(dtype=int),
        "SourceCellI": cell_df["SourceCellI"].to_numpy(dtype=int),
        "SourceCellJ": cell_df["SourceCellJ"].to_numpy(dtype=int),
        "SourceLocalCellID": cell_df["SourceLocalCellID"].to_numpy(dtype=int),
        "CenterX": cell_df["CenterX"].to_numpy(dtype=float),
        "CenterY": cell_df["CenterY"].to_numpy(dtype=float),
        "CenterTIME": cell_df["CenterTIME"].to_numpy(dtype=float),
        "VertexCount": cell_df["VertexCount"].to_numpy(dtype=int),
    }
    scalar_types = {
        "PatchArea": "float",
        "SourcePatchArea": "float",
        "FaultNameCode": "int",
        "SourcePatchIndex": "int",
        "SourceCellI": "int",
        "SourceCellJ": "int",
        "SourceLocalCellID": "int",
        "CenterX": "float",
        "CenterY": "float",
        "CenterTIME": "float",
        "VertexCount": "int",
    }
    write_legacy_vtk_polygons_preserve_patch_area(
        path=Path(output_vtk),
        title="raw_fault_surface_no_planarization",
        points=np.asarray(merged_points, dtype=float),
        polygons=merged_polygons,
        cell_data=cell_data,
        scalar_types=scalar_types,
        recompute_patch_area=False,
    )

    if output_csv is not None:
        write_csv_utf8(cell_df, Path(output_csv))

    summary = {
        "fault_patches_root": str(fault_patches_root),
        "block_x_range": [int(min(block_x_start, block_x_end)), int(max(block_x_start, block_x_end))],
        "block_y_range": [int(min(block_y_start, block_y_end)), int(max(block_y_start, block_y_end))],
        "selected_patch_count": int(len(patch_paths)),
        "fault_name_count": int(len(fault_name_to_code)),
        "fault_name_to_code": fault_name_to_code,
        "point_count": int(len(merged_points)),
        "polygon_count": int(len(merged_polygons)),
        "total_area": float(cell_df["PatchArea"].sum()) if not cell_df.empty else 0.0,
        "output_vtk": str(output_vtk),
        "output_csv": str(output_csv) if output_csv is not None else None,
    }
    if summary_json is not None:
        write_json(Path(summary_json), summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = args.output_vtk.with_suffix(".csv")
    summary_json = args.summary_json
    if summary_json is None:
        summary_json = args.output_vtk.with_suffix(".summary.json")
    summary = export_raw_fault_surface(
        fault_patches_root=Path(args.fault_patches_root),
        block_x_start=int(args.block_x_start),
        block_x_end=int(args.block_x_end),
        block_y_start=int(args.block_y_start),
        block_y_end=int(args.block_y_end),
        output_vtk=Path(args.output_vtk),
        output_csv=Path(output_csv),
        summary_json=Path(summary_json),
    )
    print(f"output_vtk: {summary['output_vtk']}")
    print(f"selected_patch_count: {summary['selected_patch_count']}")
    print(f"polygon_count: {summary['polygon_count']}")
    print(f"total_area: {summary['total_area']:.6f}")


if __name__ == "__main__":
    main()
