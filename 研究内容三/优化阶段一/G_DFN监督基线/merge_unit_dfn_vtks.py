# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_csv_utf8, write_json


UNIT_ID_PATTERN = re.compile(r"^BX(?P<block_x>\d+)_BY(?P<block_y>\d+)$", flags=re.IGNORECASE)
SCALARS_PATTERN = re.compile(r"^SCALARS\s+(?P<name>\S+)\s+(?P<dtype>\S+)\s+1$", flags=re.IGNORECASE)


def emit_merge_progress(stage: str, detail: str | None = None) -> None:
    if detail:
        print(f"[merge] {stage} | {detail}", flush=True)
    else:
        print(f"[merge] {stage}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge rectangle-selected unit DFN VTK files into one regional VTK.")
    parser.add_argument("--units-root", type=Path, required=True, help="Path to the run's units directory.")
    parser.add_argument("--block-x-start", type=int, required=True)
    parser.add_argument("--block-x-end", type=int, required=True)
    parser.add_argument("--block-y-start", type=int, required=True)
    parser.add_argument("--block-y-end", type=int, required=True)
    parser.add_argument("--vtk-name", type=str, default="predicted_patches_raw_time.vtk")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name", type=str, default=f"merge_unit_dfn_vtks_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--output-vtk-name", type=str)
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def parse_unit_id(unit_id: str) -> tuple[int, int]:
    matched = UNIT_ID_PATTERN.match(str(unit_id).strip())
    if not matched:
        raise ValueError(f"invalid UnitID: {unit_id}")
    return int(matched.group("block_x")), int(matched.group("block_y"))


def build_target_unit_ids(args: argparse.Namespace) -> list[str]:
    block_x_start = int(min(args.block_x_start, args.block_x_end))
    block_x_end = int(max(args.block_x_start, args.block_x_end))
    block_y_start = int(min(args.block_y_start, args.block_y_end))
    block_y_end = int(max(args.block_y_start, args.block_y_end))
    unit_ids = [
        f"BX{block_x}_BY{block_y}"
        for block_x in range(block_x_start, block_x_end + 1)
        for block_y in range(block_y_start, block_y_end + 1)
    ]
    return sorted(unit_ids, key=parse_unit_id)


def validate_required_unit_vtks(units_root: Path, unit_ids: list[str], vtk_name: str) -> list[Path]:
    missing_unit_dirs: list[str] = []
    missing_vtk_files: list[str] = []
    vtk_paths: list[Path] = []
    for unit_id in unit_ids:
        unit_dir = units_root / unit_id
        if not unit_dir.exists():
            missing_unit_dirs.append(unit_id)
            continue
        vtk_path = unit_dir / vtk_name
        if not vtk_path.exists():
            missing_vtk_files.append(str(vtk_path))
            continue
        vtk_paths.append(vtk_path)
    if missing_unit_dirs or missing_vtk_files:
        parts: list[str] = []
        if missing_unit_dirs:
            preview = ", ".join(missing_unit_dirs[:20])
            parts.append(f"missing_unit_dirs({len(missing_unit_dirs)}): {preview}")
        if missing_vtk_files:
            preview = ", ".join(missing_vtk_files[:10])
            parts.append(f"missing_vtk_files({len(missing_vtk_files)}): {preview}")
        raise FileNotFoundError("required units/vtks missing for requested rectangle; " + " | ".join(parts))
    if not vtk_paths:
        raise FileNotFoundError("no unit VTK files found for requested rectangle")
    return vtk_paths


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_legacy_vtk_polygons(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() != ""]
    _expect(len(lines) >= 5, f"invalid vtk file, too few lines: {path}")
    _expect(lines[0].startswith("# vtk DataFile"), f"unsupported vtk header: {path}")
    _expect(lines[2] == "ASCII", f"only ASCII VTK is supported: {path}")
    _expect(lines[3] == "DATASET POLYDATA", f"only POLYDATA is supported: {path}")

    idx = 4
    point_header = lines[idx].split()
    _expect(len(point_header) >= 3 and point_header[0] == "POINTS", f"POINTS block missing: {path}")
    point_count = int(point_header[1])
    idx += 1
    points: list[list[float]] = []
    for _ in range(point_count):
        parts = lines[idx].split()
        _expect(len(parts) >= 3, f"invalid point row in {path}: {lines[idx]}")
        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        idx += 1

    polygon_header = lines[idx].split()
    _expect(len(polygon_header) >= 3 and polygon_header[0] == "POLYGONS", f"POLYGONS block missing: {path}")
    polygon_count = int(polygon_header[1])
    idx += 1
    polygons: list[list[int]] = []
    for _ in range(polygon_count):
        parts = lines[idx].split()
        _expect(len(parts) >= 2, f"invalid polygon row in {path}: {lines[idx]}")
        vertex_count = int(parts[0])
        polygon = [int(value) for value in parts[1:]]
        _expect(len(polygon) == vertex_count, f"polygon vertex count mismatch in {path}: {lines[idx]}")
        polygons.append(polygon)
        idx += 1

    scalar_types: dict[str, str] = {}
    cell_data: dict[str, np.ndarray] = {}
    if idx < len(lines):
        cell_header = lines[idx].split()
        _expect(len(cell_header) >= 2 and cell_header[0] == "CELL_DATA", f"unexpected tail block in {path}: {lines[idx]}")
        cell_count = int(cell_header[1])
        _expect(cell_count == polygon_count, f"CELL_DATA count mismatch in {path}: {cell_count} != {polygon_count}")
        idx += 1
        while idx < len(lines):
            matched = SCALARS_PATTERN.match(lines[idx])
            _expect(matched is not None, f"SCALARS block expected in {path}: {lines[idx]}")
            name = str(matched.group("name"))
            dtype = str(matched.group("dtype")).lower()
            scalar_types[name] = dtype
            idx += 1
            _expect(idx < len(lines) and lines[idx] == "LOOKUP_TABLE default", f"LOOKUP_TABLE missing in {path} for {name}")
            idx += 1
            values: list[float | int] = []
            for _ in range(cell_count):
                raw = lines[idx]
                if dtype == "int":
                    values.append(int(float(raw)))
                else:
                    values.append(float(raw))
                idx += 1
            array_dtype = int if dtype == "int" else float
            cell_data[name] = np.asarray(values, dtype=array_dtype)

    return {
        "title": lines[1],
        "points": np.asarray(points, dtype=float),
        "polygons": polygons,
        "cell_data": cell_data,
        "scalar_types": scalar_types,
    }


def write_vtk_scalar_block(lines: list[str], name: str, values: np.ndarray, scalar_type: str) -> None:
    dtype = str(scalar_type).lower()
    if dtype == "int":
        formatter = lambda value: str(int(value))
    else:
        dtype = "float"
        formatter = lambda value: f"{float(value):.6f}"
    lines.append(f"SCALARS {name} {dtype} 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(formatter(value) for value in values.tolist())


def compute_polygon_areas(points: np.ndarray, polygons: list[list[int]]) -> np.ndarray:
    areas = np.zeros(len(polygons), dtype=float)
    for polygon_idx, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        polygon_points = np.asarray([points[int(point_idx)] for point_idx in polygon], dtype=float)
        origin = polygon_points[0]
        area = 0.0
        for vertex_idx in range(1, len(polygon_points) - 1):
            vec1 = polygon_points[vertex_idx] - origin
            vec2 = polygon_points[vertex_idx + 1] - origin
            area += 0.5 * float(np.linalg.norm(np.cross(vec1, vec2)))
        areas[polygon_idx] = area
    return areas


def write_legacy_vtk_polygons(
    path: Path,
    title: str,
    points: np.ndarray,
    polygons: list[list[int]],
    cell_data: dict[str, np.ndarray],
    scalar_types: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local_cell_data = {name: np.asarray(values) for name, values in cell_data.items()}
    local_scalar_types = dict(scalar_types)
    if polygons:
        local_cell_data["PatchArea"] = compute_polygon_areas(np.asarray(points, dtype=float), polygons)
        local_scalar_types["PatchArea"] = "float"

    total_polygon_size = sum(len(polygon) + 1 for polygon in polygons)
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f}" for point in points)
    lines.append(f"POLYGONS {len(polygons)} {total_polygon_size}")
    lines.extend(f"{len(polygon)} {' '.join(str(int(idx)) for idx in polygon)}" for polygon in polygons)
    if local_cell_data:
        lines.append(f"CELL_DATA {len(polygons)}")
        for name, values in local_cell_data.items():
            write_vtk_scalar_block(lines, name, values, local_scalar_types.get(name, "float"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_vtk_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged_points: list[list[float]] = []
    merged_polygons: list[list[int]] = []
    scalar_types: dict[str, str] = {}
    for payload in payloads:
        scalar_types.update(payload["scalar_types"])

    all_scalar_names = sorted(scalar_types.keys())
    merged_cell_lists: dict[str, list[np.ndarray]] = {name: [] for name in all_scalar_names}

    point_offset = 0
    for payload in payloads:
        points = np.asarray(payload["points"], dtype=float)
        polygons = payload["polygons"]
        cell_count = len(polygons)
        merged_points.extend(points.tolist())
        merged_polygons.extend([[int(vertex_idx) + point_offset for vertex_idx in polygon] for polygon in polygons])
        point_offset += len(points)

        payload_cell_data = payload["cell_data"]
        for name in all_scalar_names:
            dtype = scalar_types.get(name, "float")
            if name in payload_cell_data:
                merged_cell_lists[name].append(np.asarray(payload_cell_data[name]))
            else:
                fill_value = -9999 if dtype == "int" else -9999.0
                fill_dtype = int if dtype == "int" else float
                merged_cell_lists[name].append(np.full(cell_count, fill_value, dtype=fill_dtype))

    merged_cell_data = {
        name: np.concatenate(arrays).astype(int if scalar_types.get(name, "float") == "int" else float)
        for name, arrays in merged_cell_lists.items()
    }
    return {
        "points": np.asarray(merged_points, dtype=float),
        "polygons": merged_polygons,
        "cell_data": merged_cell_data,
        "scalar_types": scalar_types,
    }


def main() -> None:
    args = build_parser().parse_args()
    units_root = Path(args.units_root)
    output_root = Path(args.output_root) if args.output_root else units_root.parent / "merged_regions"
    run_dir = output_root / str(args.run_name)

    target_unit_ids = build_target_unit_ids(args)
    vtk_paths = validate_required_unit_vtks(units_root=units_root, unit_ids=target_unit_ids, vtk_name=str(args.vtk_name))
    emit_merge_progress(
        "开始区域单元 VTK 合并",
        f"selected_unit_count={len(target_unit_ids)}, vtk_name={args.vtk_name}",
    )

    payloads: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    cumulative_polygon_count = 0
    total_units = len(target_unit_ids)
    for unit_idx, (unit_id, vtk_path) in enumerate(zip(target_unit_ids, vtk_paths), start=1):
        block_x, block_y = parse_unit_id(unit_id)
        payload = read_legacy_vtk_polygons(vtk_path)
        payloads.append(payload)
        points = np.asarray(payload["points"], dtype=float)
        cumulative_polygon_count += int(len(payload["polygons"]))
        source_rows.append(
            {
                "UnitID": str(unit_id),
                "BlockX": int(block_x),
                "BlockY": int(block_y),
                "SourceVTK": str(vtk_path),
                "PointCount": int(len(points)),
                "PolygonCount": int(len(payload["polygons"])),
                "XMin": float(points[:, 0].min()) if len(points) else None,
                "XMax": float(points[:, 0].max()) if len(points) else None,
                "YMin": float(points[:, 1].min()) if len(points) else None,
                "YMax": float(points[:, 1].max()) if len(points) else None,
                "ZMin": float(points[:, 2].min()) if len(points) else None,
                "ZMax": float(points[:, 2].max()) if len(points) else None,
            }
        )
        if unit_idx == 1 or unit_idx == total_units or unit_idx % 50 == 0:
            emit_merge_progress(
                "读取单元 VTK 进度",
                (
                    f"{unit_idx}/{total_units}, unit={unit_id}, "
                    f"cumulative_polygon_count={cumulative_polygon_count}"
                ),
            )

    emit_merge_progress("开始合并 VTK payload", f"payload_count={len(payloads)}")
    merged = merge_vtk_payloads(payloads)
    output_vtk_name = (
        str(args.output_vtk_name)
        if args.output_vtk_name
        else f"merged_bx{min(args.block_x_start, args.block_x_end)}_{max(args.block_x_start, args.block_x_end)}"
        f"_by{min(args.block_y_start, args.block_y_end)}_{max(args.block_y_start, args.block_y_end)}"
        f"_{Path(args.vtk_name).name}"
    )
    output_vtk_path = run_dir / output_vtk_name
    title = f"merged_{Path(args.vtk_name).stem}"
    emit_merge_progress(
        "开始写出合并后 VTK",
        f"merged_polygon_count={len(merged['polygons'])}, output={output_vtk_path}",
    )
    write_legacy_vtk_polygons(
        path=output_vtk_path,
        title=title,
        points=merged["points"],
        polygons=merged["polygons"],
        cell_data=merged["cell_data"],
        scalar_types=merged["scalar_types"],
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    source_df = pd.DataFrame(source_rows)
    source_csv = run_dir / "merged_source_units.csv"
    write_csv_utf8(source_df, source_csv)

    points = np.asarray(merged["points"], dtype=float)
    summary = {
        "run_name": str(args.run_name),
        "units_root": str(units_root),
        "vtk_name": str(args.vtk_name),
        "output_vtk": str(output_vtk_path),
        "selected_unit_count": int(len(target_unit_ids)),
        "merged_point_count": int(len(merged["points"])),
        "merged_polygon_count": int(len(merged["polygons"])),
        "x_min": float(points[:, 0].min()) if len(points) else None,
        "x_max": float(points[:, 0].max()) if len(points) else None,
        "y_min": float(points[:, 1].min()) if len(points) else None,
        "y_max": float(points[:, 1].max()) if len(points) else None,
        "z_min": float(points[:, 2].min()) if len(points) else None,
        "z_max": float(points[:, 2].max()) if len(points) else None,
        "source_csv": str(source_csv),
    }
    summary_json = run_dir / "merge_summary.json"
    write_json(summary_json, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN 单元区域 VTK 合并",
        lines=[
            f"units_root: {units_root}",
            f"vtk_name: {args.vtk_name}",
            f"block_x_range: {min(args.block_x_start, args.block_x_end)}-{max(args.block_x_start, args.block_x_end)}",
            f"block_y_range: {min(args.block_y_start, args.block_y_end)}-{max(args.block_y_start, args.block_y_end)}",
            f"selected_unit_count: {summary['selected_unit_count']}",
            f"merged_point_count: {summary['merged_point_count']}",
            f"merged_polygon_count: {summary['merged_polygon_count']}",
            f"output_vtk: {output_vtk_path}",
            f"source_csv: {source_csv}",
            f"summary_json: {summary_json}",
        ],
    )

    print(f"output_vtk: {output_vtk_path}")
    print(f"selected_unit_count: {summary['selected_unit_count']}")
    print(f"merged_polygon_count: {summary['merged_polygon_count']}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
