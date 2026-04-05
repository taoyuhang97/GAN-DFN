# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_json
from merge_unit_dfn_vtks import read_legacy_vtk_polygons, write_legacy_vtk_polygons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uniformly scale fracture polygons in an existing merged DFN VTK file.")
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--scale-factor", type=float, required=True)
    parser.add_argument("--output-vtk", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def format_scale_suffix(scale_factor: float) -> str:
    text = f"{float(scale_factor):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def build_default_output_vtk(input_vtk: Path, scale_factor: float) -> Path:
    suffix = format_scale_suffix(scale_factor)
    return input_vtk.with_name(f"{input_vtk.stem}_uniform_scaled_x{suffix}{input_vtk.suffix}")


def scale_polygons_uniform(
    points: np.ndarray,
    polygons: list[list[int]],
    scale_factor: float,
) -> tuple[np.ndarray, list[list[int]]]:
    factor = float(scale_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"scale_factor must be a finite positive number, got {scale_factor}")

    scaled_points: list[list[float]] = []
    scaled_polygons: list[list[int]] = []
    for polygon in polygons:
        polygon_points = np.asarray([points[int(point_idx)] for point_idx in polygon], dtype=float)
        center = polygon_points.mean(axis=0)
        scaled_polygon_points = center + factor * (polygon_points - center)
        start_idx = len(scaled_points)
        scaled_points.extend(scaled_polygon_points.tolist())
        scaled_polygons.append(list(range(start_idx, start_idx + len(polygon))))
    return np.asarray(scaled_points, dtype=float), scaled_polygons


def scale_cell_data(cell_data: dict[str, np.ndarray], scale_factor: float) -> dict[str, np.ndarray]:
    factor = float(scale_factor)
    result: dict[str, np.ndarray] = {}
    for name, values in cell_data.items():
        array = np.asarray(values).copy()
        if name in {"PatchLength", "PatchHeight"}:
            array = array.astype(float) * factor
        result[name] = array
    return result


def main() -> None:
    args = build_parser().parse_args()
    input_vtk = Path(args.input_vtk)
    if not input_vtk.exists():
        raise FileNotFoundError(f"input_vtk not found: {input_vtk}")
    if not np.isfinite(float(args.scale_factor)) or float(args.scale_factor) <= 0.0:
        raise ValueError(f"scale_factor must be a finite positive number, got {args.scale_factor}")

    output_vtk = Path(args.output_vtk) if args.output_vtk else build_default_output_vtk(input_vtk, float(args.scale_factor))
    if output_vtk.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output_vtk already exists: {output_vtk}")

    payload = read_legacy_vtk_polygons(input_vtk)
    input_points = np.asarray(payload["points"], dtype=float)
    input_polygons = payload["polygons"]
    input_cell_data = payload["cell_data"]
    scalar_types = payload["scalar_types"]

    scaled_points, scaled_polygons = scale_polygons_uniform(
        points=input_points,
        polygons=input_polygons,
        scale_factor=float(args.scale_factor),
    )
    scaled_cell_data = scale_cell_data(input_cell_data, float(args.scale_factor))
    title = f"{payload['title']}_uniform_scaled_x{format_scale_suffix(float(args.scale_factor))}"
    write_legacy_vtk_polygons(
        path=output_vtk,
        title=title,
        points=scaled_points,
        polygons=scaled_polygons,
        cell_data=scaled_cell_data,
        scalar_types=scalar_types,
    )

    summary = {
        "input_vtk": str(input_vtk),
        "output_vtk": str(output_vtk),
        "scale_factor": float(args.scale_factor),
        "polygon_count": int(len(input_polygons)),
        "input_point_count": int(len(input_points)),
        "output_point_count": int(len(scaled_points)),
        "input_x_min": float(input_points[:, 0].min()) if len(input_points) else None,
        "input_x_max": float(input_points[:, 0].max()) if len(input_points) else None,
        "input_y_min": float(input_points[:, 1].min()) if len(input_points) else None,
        "input_y_max": float(input_points[:, 1].max()) if len(input_points) else None,
        "input_z_min": float(input_points[:, 2].min()) if len(input_points) else None,
        "input_z_max": float(input_points[:, 2].max()) if len(input_points) else None,
        "output_x_min": float(scaled_points[:, 0].min()) if len(scaled_points) else None,
        "output_x_max": float(scaled_points[:, 0].max()) if len(scaled_points) else None,
        "output_y_min": float(scaled_points[:, 1].min()) if len(scaled_points) else None,
        "output_y_max": float(scaled_points[:, 1].max()) if len(scaled_points) else None,
        "output_z_min": float(scaled_points[:, 2].min()) if len(scaled_points) else None,
        "output_z_max": float(scaled_points[:, 2].max()) if len(scaled_points) else None,
    }
    summary_path = output_vtk.with_suffix(".summary.json")
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN 合并VTK裂缝片放大",
        lines=[
            f"input_vtk: {input_vtk}",
            f"output_vtk: {output_vtk}",
            f"scale_factor: {args.scale_factor}",
            f"polygon_count: {summary['polygon_count']}",
            f"input_point_count: {summary['input_point_count']}",
            f"output_point_count: {summary['output_point_count']}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"output_vtk: {output_vtk}")
    print(f"polygon_count: {summary['polygon_count']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
