# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import mmap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_csv_utf8, write_json
from infer_production_units_baseline import (
    DEFAULT_LAYER_BOTTOM_BOUNDARY_MS,
    DEFAULT_LAYER_TOP_BOUNDARY_MS,
    DEFAULT_SURFACE_DIR,
    DEFAULT_TRACE_HEADER_CSV,
    DEFAULT_Z_STEP_MS,
    build_auto_layers_from_surface_rows,
    extend_layers_to_seismic_bounds,
    load_surface_nearest_lookups,
    load_trace_header,
    query_surface_nearest,
    resolve_unit_center_trace_xy,
)


LAYER_ORDER = [
    "TOP_1100MS->T1",
    "T1->T2",
    "T2->T3",
    "T3->T4",
    "T4->T5",
    "T5->T6",
    "T6->T7",
    "T7->BOTTOM_3800MS",
]
SELECTED_SCALARS = ("BlockX", "BlockY", "PatchArea", "PatchLength", "PatchHeight")


@dataclass
class MinimalRegionalPayload:
    title: str
    point_count: int
    polygon_count: int
    center_time: np.ndarray
    block_x: np.ndarray
    block_y: np.ndarray
    patch_area: np.ndarray


def emit_progress(stage: str, detail: str | None = None) -> None:
    if detail:
        print(f"[regional-layer-density] {stage} | {detail}", flush=True)
    else:
        print(f"[regional-layer-density] {stage}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按当前层位规则对区域 DFN 重新赋层，并统计各层裂缝密度。")
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        type=str,
        default=f"regional_layer_density_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--top-boundary-ms", type=float, default=DEFAULT_LAYER_TOP_BOUNDARY_MS)
    parser.add_argument("--bottom-boundary-ms", type=float, default=DEFAULT_LAYER_BOTTOM_BOUNDARY_MS)
    parser.add_argument("--z-step-ms", type=float, default=DEFAULT_Z_STEP_MS)
    return parser


def sort_layers(df: pd.DataFrame, column: str = "LayerSurfacePairKey") -> pd.DataFrame:
    work = df.copy()
    order_map = {name: idx for idx, name in enumerate(LAYER_ORDER)}
    work["_layer_order"] = work[column].map(order_map).fillna(9999)
    work = work.sort_values(["_layer_order", column], kind="stable").drop(columns=["_layer_order"])
    return work.reset_index(drop=True)


def _read_line(mm: mmap.mmap, pos: int) -> tuple[bytes, int]:
    if pos >= len(mm):
        return b"", pos
    end = mm.find(b"\n", pos)
    if end == -1:
        end = len(mm)
        next_pos = end
    else:
        next_pos = end + 1
    return mm[pos:end].strip(), next_pos


def _read_next_nonempty_line(mm: mmap.mmap, pos: int) -> tuple[bytes, int]:
    while pos < len(mm):
        line, next_pos = _read_line(mm, pos)
        pos = next_pos
        if line:
            return line, pos
    raise EOFError("unexpected end of vtk file while reading line")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_header_count(line: bytes, expected_prefix: bytes) -> int:
    parts = line.split()
    _expect(len(parts) >= 2 and parts[0] == expected_prefix, f"invalid header line: {line.decode('utf-8', errors='ignore')}")
    return int(parts[1])


def load_minimal_regional_payload(input_vtk: Path) -> MinimalRegionalPayload:
    input_vtk = Path(input_vtk)
    start_time = time.perf_counter()
    with input_vtk.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        header_1, pos = _read_next_nonempty_line(mm, 0)
        title_line, pos = _read_next_nonempty_line(mm, pos)
        header_3, pos = _read_next_nonempty_line(mm, pos)
        header_4, pos = _read_next_nonempty_line(mm, pos)
        _expect(header_1.startswith(b"# vtk DataFile"), f"unsupported vtk header: {input_vtk}")
        _expect(header_3 == b"ASCII", f"only ASCII VTK is supported: {input_vtk}")
        _expect(header_4 == b"DATASET POLYDATA", f"only POLYDATA is supported: {input_vtk}")

        point_header, pos = _read_next_nonempty_line(mm, pos)
        point_count = _parse_header_count(point_header, b"POINTS")
        polygon_header_start = mm.find(b"\nPOLYGONS ", pos)
        _expect(polygon_header_start != -1, f"POLYGONS block not found: {input_vtk}")
        emit_progress("开始解析 VTK 点坐标", f"point_count={point_count}")
        point_values = np.fromstring(mm[pos:polygon_header_start], sep=" ", dtype=np.float32)
        _expect(point_values.size == point_count * 3, f"POINTS parse size mismatch: {point_values.size} != {point_count * 3}")
        point_z = point_values[2::3].copy()
        del point_values

        polygon_header, polygon_data_start = _read_next_nonempty_line(mm, polygon_header_start + 1)
        polygon_header_parts = polygon_header.split()
        _expect(len(polygon_header_parts) >= 3 and polygon_header_parts[0] == b"POLYGONS", f"invalid POLYGONS header: {polygon_header}")
        polygon_count = int(polygon_header_parts[1])
        total_polygon_size = int(polygon_header_parts[2])
        cell_header_start = mm.find(b"\nCELL_DATA ", polygon_data_start)
        _expect(cell_header_start != -1, f"CELL_DATA block not found: {input_vtk}")
        emit_progress("开始解析 VTK 多边形", f"polygon_count={polygon_count}")
        polygon_values = np.fromstring(mm[polygon_data_start:cell_header_start], sep=" ", dtype=np.int64)
        _expect(
            polygon_values.size == total_polygon_size,
            f"POLYGONS parse size mismatch: {polygon_values.size} != {total_polygon_size}",
        )
        row_width = total_polygon_size // max(polygon_count, 1)
        _expect(row_width * polygon_count == total_polygon_size, "POLYGONS rows are not fixed-width; current parser only supports fixed-width polygons")
        polygon_matrix = polygon_values.reshape(polygon_count, row_width)
        vertex_counts = polygon_matrix[:, 0]
        _expect(np.all(vertex_counts == vertex_counts[0]), "polygon vertex count is not uniform")
        vertex_count = int(vertex_counts[0])
        _expect(vertex_count >= 3, f"invalid polygon vertex_count={vertex_count}")
        polygon_indices = polygon_matrix[:, 1 : 1 + vertex_count].astype(np.int64, copy=False)
        center_time = point_z[polygon_indices].mean(axis=1, dtype=np.float32)
        del polygon_values
        del polygon_matrix

        cell_header, cell_data_start = _read_next_nonempty_line(mm, cell_header_start + 1)
        cell_count = _parse_header_count(cell_header, b"CELL_DATA")
        _expect(cell_count == polygon_count, f"CELL_DATA count mismatch: {cell_count} != {polygon_count}")

        selected_data: dict[str, np.ndarray] = {}
        pos = cell_data_start
        emit_progress("开始解析所需标量字段", f"selected={','.join(SELECTED_SCALARS)}")
        while pos < len(mm):
            scalar_pos = mm.find(b"SCALARS ", pos)
            if scalar_pos == -1:
                break
            scalar_header, after_header = _read_next_nonempty_line(mm, scalar_pos)
            header_parts = scalar_header.split()
            _expect(len(header_parts) >= 4 and header_parts[0] == b"SCALARS", f"invalid SCALARS header: {scalar_header}")
            scalar_name = header_parts[1].decode("utf-8", errors="ignore")
            lookup_line, after_lookup = _read_next_nonempty_line(mm, after_header)
            _expect(lookup_line == b"LOOKUP_TABLE default", f"LOOKUP_TABLE missing for scalar={scalar_name}")
            next_scalar_pos = mm.find(b"\nSCALARS ", after_lookup)
            data_end = len(mm) if next_scalar_pos == -1 else next_scalar_pos
            if scalar_name in SELECTED_SCALARS:
                scalar_values = np.fromstring(mm[after_lookup:data_end], sep=" ", dtype=np.float32)
                _expect(
                    scalar_values.size == cell_count,
                    f"scalar parse size mismatch for {scalar_name}: {scalar_values.size} != {cell_count}",
                )
                selected_data[scalar_name] = scalar_values
                emit_progress("已解析标量字段", f"{scalar_name} ({len(selected_data)}/{len(SELECTED_SCALARS)})")
            pos = data_end + 1

    for required_name in ("BlockX", "BlockY"):
        _expect(required_name in selected_data, f"required scalar missing in vtk: {required_name}")
    if "PatchArea" not in selected_data:
        _expect(
            "PatchLength" in selected_data and "PatchHeight" in selected_data,
            "PatchArea missing and PatchLength/PatchHeight unavailable",
        )
        selected_data["PatchArea"] = selected_data["PatchLength"] * selected_data["PatchHeight"]

    duration = time.perf_counter() - start_time
    emit_progress(
        "VTK 最小载荷解析完成",
        f"polygon_count={polygon_count}, elapsed_sec={duration:.1f}",
    )
    return MinimalRegionalPayload(
        title=title_line.decode("utf-8", errors="ignore"),
        point_count=int(point_count),
        polygon_count=int(polygon_count),
        center_time=np.asarray(center_time, dtype=np.float32),
        block_x=np.rint(selected_data["BlockX"]).astype(np.int32),
        block_y=np.rint(selected_data["BlockY"]).astype(np.int32),
        patch_area=np.asarray(selected_data["PatchArea"], dtype=np.float32),
    )


def build_unit_layers(
    unit_pairs: np.ndarray,
    surface_dir: Path,
    trace_header_csv: Path,
    top_boundary_ms: float,
    bottom_boundary_ms: float,
    z_step_ms: float,
) -> tuple[dict[tuple[int, int], pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    emit_progress("开始加载层位与道头", f"surface_dir={surface_dir}, trace_header_csv={trace_header_csv}")
    trace_df, unique_x, unique_y = load_trace_header(trace_header_csv)
    del trace_df
    surface_lookups = load_surface_nearest_lookups(surface_dir)
    surface_codes = sorted(surface_lookups.keys())
    unit_layer_map: dict[tuple[int, int], pd.DataFrame] = {}
    layer_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, Any]] = []

    for block_x, block_y in tqdm(unit_pairs.tolist(), desc="resolve-unit-layers", unit="unit"):
        bx = int(block_x)
        by = int(block_y)
        unit_id = f"BX{bx}_BY{by}"
        try:
            center_x, center_y = resolve_unit_center_trace_xy(unique_x, unique_y, bx, by)
            surface_rows = [
                query_surface_nearest(center_x=center_x, center_y=center_y, lookup=surface_lookups[code], z_step_ms=z_step_ms)
                for code in surface_codes
            ]
            resolved_surface_df = pd.DataFrame(surface_rows)
            layers_df = build_auto_layers_from_surface_rows(
                unit_id=unit_id,
                block_x=bx,
                block_y=by,
                resolved_surface_df=resolved_surface_df,
                top_boundary_time_ms=float(top_boundary_ms),
                bottom_boundary_time_ms=float(bottom_boundary_ms),
                z_step_ms=float(z_step_ms),
            )
            layers_df = extend_layers_to_seismic_bounds(
                layers_df=layers_df,
                unit_id=unit_id,
                block_x=bx,
                block_y=by,
                top_boundary_time_ms=float(top_boundary_ms),
                bottom_boundary_time_ms=float(bottom_boundary_ms),
                z_step_ms=float(z_step_ms),
            )
            if layers_df.empty:
                raise ValueError("empty layers after surface resolution")
            layers_df = layers_df.copy().reset_index(drop=True)
            layers_df["UnitCenterX"] = float(center_x)
            layers_df["UnitCenterY"] = float(center_y)
            layers_df["LayerTopMs"] = pd.to_numeric(layers_df["TopTime"], errors="coerce")
            layers_df["LayerBaseMs"] = pd.to_numeric(layers_df["BaseTime"], errors="coerce")
            layers_df["LayerThicknessMs"] = layers_df["LayerBaseMs"] - layers_df["LayerTopMs"]
            layers_df["LayerSurfacePairKey"] = layers_df["StrataName"].astype(str)
            layers_df = sort_layers(layers_df, column="LayerSurfacePairKey")
            unit_layer_map[(bx, by)] = layers_df
            layer_frames.append(layers_df)
        except Exception as exc:  # pragma: no cover - defensive path for bad units
            skipped_rows.append(
                {
                    "UnitID": unit_id,
                    "BlockX": bx,
                    "BlockY": by,
                    "Reason": str(exc),
                }
            )

    if not layer_frames:
        raise RuntimeError("failed to build layers for all units")
    all_layers_df = pd.concat(layer_frames, ignore_index=True)
    skipped_df = pd.DataFrame(skipped_rows)
    emit_progress(
        "单元层位构建完成",
        f"unit_count={len(unit_layer_map)}, skipped_unit_count={len(skipped_df)}",
    )
    return unit_layer_map, all_layers_df, skipped_df


def assign_patches_to_layers_and_summarize(
    payload: MinimalRegionalPayload,
    unit_layer_map: dict[tuple[int, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_key = payload.block_x.astype(np.int64) * 100000 + payload.block_y.astype(np.int64)
    order = np.argsort(unit_key, kind="stable")
    sorted_unit_key = unit_key[order]
    sorted_block_x = payload.block_x[order]
    sorted_block_y = payload.block_y[order]
    sorted_center_time = payload.center_time[order]
    sorted_patch_area = np.nan_to_num(payload.patch_area[order], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    unique_keys, start_indices, counts = np.unique(sorted_unit_key, return_index=True, return_counts=True)
    unit_rows: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []

    for unit_key_value, start_idx, count in tqdm(
        zip(unique_keys.tolist(), start_indices.tolist(), counts.tolist()),
        total=len(unique_keys),
        desc="summarize-unit-layer-density",
        unit="unit",
    ):
        end_idx = start_idx + count
        bx = int(sorted_block_x[start_idx])
        by = int(sorted_block_y[start_idx])
        unit_id = f"BX{bx}_BY{by}"
        unit_layers = unit_layer_map.get((bx, by))
        if unit_layers is None or unit_layers.empty:
            unassigned_rows.append(
                {
                    "UnitID": unit_id,
                    "BlockX": bx,
                    "BlockY": by,
                    "Reason": "missing_unit_layers",
                    "PatchCount": int(count),
                }
            )
            continue

        layer_top = pd.to_numeric(unit_layers["LayerTopMs"], errors="coerce").to_numpy(dtype=float)
        layer_base = pd.to_numeric(unit_layers["LayerBaseMs"], errors="coerce").to_numpy(dtype=float)
        valid_layer_mask = np.isfinite(layer_top) & np.isfinite(layer_base) & (layer_base > layer_top)
        work_layers = unit_layers.loc[valid_layer_mask].reset_index(drop=True)
        if work_layers.empty:
            unassigned_rows.append(
                {
                    "UnitID": unit_id,
                    "BlockX": bx,
                    "BlockY": by,
                    "Reason": "empty_valid_layers",
                    "PatchCount": int(count),
                }
            )
            continue

        layer_top = pd.to_numeric(work_layers["LayerTopMs"], errors="coerce").to_numpy(dtype=float)
        layer_base = pd.to_numeric(work_layers["LayerBaseMs"], errors="coerce").to_numpy(dtype=float)
        layer_count = len(work_layers)
        unit_center_time = sorted_center_time[start_idx:end_idx].astype(np.float64, copy=False)
        unit_patch_area = sorted_patch_area[start_idx:end_idx].astype(np.float64, copy=False)
        clipped_idx = np.clip(np.searchsorted(layer_base, unit_center_time, side="right"), 0, layer_count - 1)
        is_last_layer = clipped_idx == (layer_count - 1)
        within_base = (unit_center_time < layer_base[clipped_idx]) | (is_last_layer & np.isclose(unit_center_time, layer_base[clipped_idx], atol=1e-6))
        assigned_mask = np.isfinite(unit_center_time) & (unit_center_time >= layer_top[clipped_idx]) & within_base

        if assigned_mask.any():
            assigned_idx = clipped_idx[assigned_mask]
            count_by_layer = np.bincount(assigned_idx, minlength=layer_count)
            area_by_layer = np.bincount(assigned_idx, weights=unit_patch_area[assigned_mask], minlength=layer_count)
            time_min_by_layer = np.full(layer_count, np.nan, dtype=float)
            time_max_by_layer = np.full(layer_count, np.nan, dtype=float)
            assigned_times = unit_center_time[assigned_mask]
            for layer_idx in range(layer_count):
                layer_time_mask = assigned_idx == layer_idx
                if np.any(layer_time_mask):
                    time_min_by_layer[layer_idx] = float(np.min(assigned_times[layer_time_mask]))
                    time_max_by_layer[layer_idx] = float(np.max(assigned_times[layer_time_mask]))
        else:
            count_by_layer = np.zeros(layer_count, dtype=int)
            area_by_layer = np.zeros(layer_count, dtype=float)
            time_min_by_layer = np.full(layer_count, np.nan, dtype=float)
            time_max_by_layer = np.full(layer_count, np.nan, dtype=float)

        unassigned_count = int((~assigned_mask).sum())
        if unassigned_count > 0:
            unassigned_rows.append(
                {
                    "UnitID": unit_id,
                    "BlockX": bx,
                    "BlockY": by,
                    "Reason": "outside_layer_bounds",
                    "PatchCount": unassigned_count,
                }
            )

        for layer_idx, layer_row in work_layers.iterrows():
            thickness_ms = float(layer_row["LayerThicknessMs"])
            patch_count = int(count_by_layer[layer_idx])
            patch_area_sum = float(area_by_layer[layer_idx])
            unit_rows.append(
                {
                    "UnitID": unit_id,
                    "BlockX": bx,
                    "BlockY": by,
                    "UnitCenterX": float(layer_row.get("UnitCenterX", np.nan)),
                    "UnitCenterY": float(layer_row.get("UnitCenterY", np.nan)),
                    "GeoIntervalKey": str(layer_row.get("GeoIntervalKey", "")),
                    "LayerSurfacePairKey": str(layer_row.get("LayerSurfacePairKey", "")),
                    "TopSurfaceCode": str(layer_row.get("TopSurfaceCode", "")),
                    "BaseSurfaceCode": str(layer_row.get("BaseSurfaceCode", "")),
                    "LayerTopMs": float(layer_row["LayerTopMs"]),
                    "LayerBaseMs": float(layer_row["LayerBaseMs"]),
                    "LayerThicknessMs": thickness_ms,
                    "PatchCount": patch_count,
                    "PatchAreaSum": patch_area_sum,
                    "PatchDensityPer100Ms": float(patch_count) / max(thickness_ms, 1e-9) * 100.0,
                    "PatchAreaPer100Ms": patch_area_sum / max(thickness_ms, 1e-9) * 100.0,
                    "AssignedCenterTimeMinMs": float(time_min_by_layer[layer_idx]) if np.isfinite(time_min_by_layer[layer_idx]) else np.nan,
                    "AssignedCenterTimeMaxMs": float(time_max_by_layer[layer_idx]) if np.isfinite(time_max_by_layer[layer_idx]) else np.nan,
                }
            )

    unit_layer_df = pd.DataFrame(unit_rows)
    unassigned_df = pd.DataFrame(unassigned_rows)
    if not unit_layer_df.empty:
        unit_layer_df = sort_layers(unit_layer_df, column="LayerSurfacePairKey")
    return unit_layer_df, unassigned_df


def build_regional_summary(unit_layer_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        unit_layer_df.groupby("LayerSurfacePairKey", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "CoveredUnitCount": int(g["UnitID"].nunique()),
                    "ActiveUnitCount": int(g.loc[pd.to_numeric(g["PatchCount"], errors="coerce").fillna(0) > 0, "UnitID"].nunique()),
                    "LayerSegmentCount": int(len(g)),
                    "TotalThicknessMs": float(pd.to_numeric(g["LayerThicknessMs"], errors="coerce").fillna(0.0).sum()),
                    "PatchCount": int(pd.to_numeric(g["PatchCount"], errors="coerce").fillna(0).sum()),
                    "PatchAreaSum": float(pd.to_numeric(g["PatchAreaSum"], errors="coerce").fillna(0.0).sum()),
                }
            )
        )
        .reset_index()
    )
    grouped["PatchDensityPer100Ms"] = grouped["PatchCount"] / grouped["TotalThicknessMs"].replace(0.0, np.nan) * 100.0
    grouped["PatchAreaPer100Ms"] = grouped["PatchAreaSum"] / grouped["TotalThicknessMs"].replace(0.0, np.nan) * 100.0
    grouped = sort_layers(grouped, column="LayerSurfacePairKey")
    return grouped


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_vtk = Path(args.input_vtk).resolve()
    run_dir = Path(args.output_dir).resolve() / str(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = load_minimal_regional_payload(input_vtk)
    unit_pairs = np.unique(np.stack([payload.block_x, payload.block_y], axis=1), axis=0)
    emit_progress("识别区域单元完成", f"unit_count={len(unit_pairs)}")

    unit_layer_map, all_layers_df, skipped_units_df = build_unit_layers(
        unit_pairs=unit_pairs,
        surface_dir=Path(args.surface_dir).resolve(),
        trace_header_csv=Path(args.trace_header_csv).resolve(),
        top_boundary_ms=float(args.top_boundary_ms),
        bottom_boundary_ms=float(args.bottom_boundary_ms),
        z_step_ms=float(args.z_step_ms),
    )
    unit_layer_df, unassigned_df = assign_patches_to_layers_and_summarize(payload, unit_layer_map)
    regional_summary_df = build_regional_summary(unit_layer_df)

    unit_layers_csv = run_dir / "regional_unit_layers_resolved.csv"
    unit_layer_density_csv = run_dir / "regional_unit_layer_density.csv"
    regional_layer_density_csv = run_dir / "regional_layer_density_summary.csv"
    skipped_units_csv = run_dir / "regional_layer_density_skipped_units.csv"
    unassigned_csv = run_dir / "regional_layer_density_unassigned_counts.csv"

    write_csv_utf8(sort_layers(all_layers_df, column="LayerSurfacePairKey"), unit_layers_csv)
    write_csv_utf8(unit_layer_df, unit_layer_density_csv)
    write_csv_utf8(regional_summary_df, regional_layer_density_csv)
    if skipped_units_df.empty:
        skipped_units_df = pd.DataFrame(columns=["UnitID", "BlockX", "BlockY", "Reason"])
    write_csv_utf8(skipped_units_df, skipped_units_csv)
    if unassigned_df.empty:
        unassigned_df = pd.DataFrame(columns=["UnitID", "BlockX", "BlockY", "Reason", "PatchCount"])
    write_csv_utf8(unassigned_df, unassigned_csv)

    summary = {
        "input_vtk": str(input_vtk),
        "vtk_title": str(payload.title),
        "point_count": int(payload.point_count),
        "polygon_count": int(payload.polygon_count),
        "unit_count": int(len(unit_pairs)),
        "resolved_unit_count": int(len(unit_layer_map)),
        "skipped_unit_count": int(len(skipped_units_df)),
        "unit_layer_row_count": int(len(unit_layer_df)),
        "regional_layer_row_count": int(len(regional_summary_df)),
        "assigned_patch_count": int(pd.to_numeric(unit_layer_df["PatchCount"], errors="coerce").fillna(0).sum()) if not unit_layer_df.empty else 0,
        "unassigned_patch_count": int(pd.to_numeric(unassigned_df["PatchCount"], errors="coerce").fillna(0).sum()) if not unassigned_df.empty else 0,
        "unit_layers_csv": str(unit_layers_csv),
        "unit_layer_density_csv": str(unit_layer_density_csv),
        "regional_layer_density_csv": str(regional_layer_density_csv),
        "skipped_units_csv": str(skipped_units_csv),
        "unassigned_csv": str(unassigned_csv),
    }
    summary_json = run_dir / "regional_layer_density_summary.json"
    write_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)

    append_lines_to_docx(
        docx_path=Path(args.docx_path),
        title=f"regional layer density {args.run_name}",
        lines=[
            f"input_vtk: {input_vtk}",
            f"polygon_count: {payload.polygon_count}",
            f"unit_count: {len(unit_pairs)}",
            f"resolved_unit_count: {len(unit_layer_map)}",
            f"assigned_patch_count: {summary['assigned_patch_count']}",
            f"unassigned_patch_count: {summary['unassigned_patch_count']}",
            f"regional_layer_density_csv: {regional_layer_density_csv}",
        ],
    )
    emit_progress("区域 DFN 分层密度统计完成", f"summary_json={summary_json}")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_analysis(args)
    print(f"regional_layer_density_csv={summary['regional_layer_density_csv']}")
    print(f"assigned_patch_count={summary['assigned_patch_count']}")
    print(f"unassigned_patch_count={summary['unassigned_patch_count']}")


if __name__ == "__main__":
    main()
