# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.collections import LineCollection

from build_all_area_section_visualization import (
    INTERVAL_LABELS,
    ProjectionSegment,
    SurfaceSectionCurve,
    configure_matplotlib_fonts,
    interval_for_center as legacy_interval_for_center,
    next_nonempty,
    load_surface_lookups,
    polygon_area,
    read_points,
    representative_line_2d,
    scan_vtk,
    select_demo_well,
)
from build_well_attribute_section_visualization import (
    AttributeSection,
    axis_edges,
    build_trace_grid,
    build_trace_tree,
    draw_well_trajectory,
    finite_quantile_bounds,
    open_volume_context,
    sample_attribute_section,
    select_time_samples,
)
from trace_horizon_section import build_trace_horizon_section_curves, resolve_horizon_trace_table

FORMAL_ROOT = Path(__file__).resolve().parent.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))
from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    HorizonSpatialLookup,
    build_spatial_lookup,
)
from common.unified_dfn_vtk import (  # noqa: E402
    ORIGINAL_FAULT_GEOMETRY_GROUP_CODE,
    extract_geometry_group,
    geometry_group_counts,
)


WELL_NAME = "车页1导眼"
COHERENCE_CMAP = "gray"
DFN_INTERVAL_COLORS = {
    "T4->T6": "#ff7a00",
    "T6->T7": "#00a6ff",
}
DFN_HALO_COLOR = "#111827"
FRACTURE_LABEL_COLOR = "#ff00e6"
FRACTURE_LABEL_EDGE = "#111827"
FRACTURE_LABEL_ALPHA = 0.48
FRACTURE_ORIENTATION_ALPHA = 0.58
IMAGING_PATCH_COLOR = "#ff2bd6"
IMAGING_PATCH_HALO = "#3b0764"
IMAGING_SEGMENT_COLOR = "#00f5ff"
IMAGING_SEGMENT_GLOW = "#083344"
FAULT_TRACE_COLOR = "#ffe600"
FAULT_TRACE_HALO = "#111827"
FAULT_TRACE_WIDTH = 2.8
FAULT_TRACE_HALO_WIDTH = 5.2
FAULT_TRACE_OVERVIEW_WIDTH = 1.25
FAULT_TRACE_OVERVIEW_HALO_WIDTH = 2.4
DFN_WIDTH_MIN = 0.45
DFN_WIDTH_MAX = 3.0
DFN_WIDTH_POWER = 0.80
DFN_SEGMENT_SCALE_MIN = 0.70
DFN_SEGMENT_SCALE_MAX = 1.85
DFN_SEGMENT_SCALE_POWER = 0.90
FRACTURE_LABEL_SIZE = 28.0
IMAGING_PATCH_LENGTH_MIN_M = 25.0
IMAGING_PATCH_LENGTH_MAX_M = 110.0
IMAGING_PATCH_ASPECT_RATIO = 1.5
IMAGING_PATCH_SIZE_POWER = 1.25
IMAGING_PATCH_LINE_WIDTH = 2.2
IMAGING_PATCH_GEOMETRY_TIME_SCALE_M_PER_MS = 1.0


def interval_for_center(center: np.ndarray, surfaces: Any) -> str | None:
    if isinstance(surfaces, HorizonSpatialLookup):
        return surfaces.interval(float(center[0]), float(center[1]), float(center[2]))
    return legacy_interval_for_center(center, surfaces)
SECTION_INTERSECTION_EPS_M = 1.0e-6


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 8 retained Cheye1 pilot-well DFN/coherence section PNGs."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def path_from_config(config: dict[str, Any], key: str) -> Path:
    return Path(str(config[key])).resolve()


def build_namespace(config: dict[str, Any], half_width: float) -> SimpleNamespace:
    input_vtk = path_from_config(config, "input_vtk")
    group_counts = geometry_group_counts(input_vtk)
    return SimpleNamespace(
        input_vtk=input_vtk,
        predicted_cell_count=int(group_counts["predicted_cell_count"]),
        original_fault_cell_count=int(group_counts["original_fault_cell_count"]),
        combined_cell_count=int(group_counts["combined_cell_count"]),
        well_trajectory_csv=None,
        surface_dir=path_from_config(config, "surface_dir"),
        output_dir=path_from_config(config, "output_dir"),
        real_well_samples_root=path_from_config(config, "real_well_samples_root"),
        well_name=str(config.get("well_name", WELL_NAME)),
        exclude_wells=set(str(item) for item in config.get("exclude_wells", [])),
        target_block=dict(config.get("target_block") or {}),
        half_width=float(half_width),
        max_polygons=int(config.get("max_polygons", 0)),
        max_selected=int(config.get("max_selected", 0)),
        min_patch_area=float(config.get("min_patch_area", 0.0)),
        max_patch_area=float(config.get("max_patch_area", 0.0)),
        fig_width=float(config.get("fig_width", 15.5)),
        fig_height=float(config.get("fig_height", 7.8)),
        dpi=int(config.get("dpi", 240)),
        title_prefix=str(config.get("title_prefix", "candidate_cheye1 过车页1导眼剖面")),
        z_label=str(config.get("z_label", "TWT / ms")),
        surface_samples=int(config.get("surface_samples", 500)),
        progress_interval=int(config.get("progress_interval", 100000)),
        no_svg=True,
        dfn_patch_metadata=None,
        small_projection_enabled=bool(config.get("small_projection_enabled", False)),
        small_projection_half_width=float(config.get("small_projection_half_width_m", half_width)),
        small_projection_include_well_control=bool(config.get("small_projection_include_well_control", True)),
    )


def infer_dfn_patch_csv(config: dict[str, Any]) -> Path | None:
    if config.get("dfn_patch_csv"):
        return path_from_config(config, "dfn_patch_csv")
    input_vtk = path_from_config(config, "input_vtk")
    candidate = input_vtk.with_name("well_corrected_dfn_fracture_patches.csv")
    return candidate if candidate.exists() else None


def load_dfn_patch_metadata(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "FractureScale" not in df.columns:
        return None
    return df.reset_index(drop=True)


def patch_scale_for_index(metadata: pd.DataFrame | None, polygon_index: int) -> str:
    if metadata is None or polygon_index < 0 or polygon_index >= len(metadata):
        return ""
    return str(metadata.iloc[int(polygon_index)].get("FractureScale", "")).strip().lower()


def patch_is_well_control_for_index(metadata: pd.DataFrame | None, polygon_index: int) -> bool:
    if metadata is None or polygon_index < 0 or polygon_index >= len(metadata):
        return False
    if "IsWellControlPatch" not in metadata.columns:
        return False
    value = pd.to_numeric(pd.Series([metadata.iloc[int(polygon_index)].get("IsWellControlPatch")]), errors="coerce").iloc[0]
    return bool(pd.notna(value) and int(value) == 1)


def patch_window_mode_for_index(metadata: pd.DataFrame | None, polygon_index: int) -> str:
    if metadata is None or polygon_index < 0 or polygon_index >= len(metadata):
        return ""
    return str(metadata.iloc[int(polygon_index)].get("WindowValidationMode", "")).strip().lower()


def patch_layer_for_index(metadata: pd.DataFrame | None, polygon_index: int) -> str:
    if metadata is None or polygon_index < 0 or polygon_index >= len(metadata):
        return ""
    return str(metadata.iloc[int(polygon_index)].get("LayerGroup", "")).strip()


def validate_inputs(config: dict[str, Any]) -> None:
    required_paths = [
        "input_vtk",
        "trace_header_csv",
        "real_well_samples_root",
        "surface_dir",
        "coherence_volume_path",
        "step3_imaging_group_csvs",
    ]
    for key in required_paths:
        if key == "step3_imaging_group_csvs":
            for value in config.get(key, []):
                path = Path(str(value)).resolve()
                if not path.exists():
                    raise FileNotFoundError(f"{key} item not found: {path}")
            continue
        path = path_from_config(config, key)
        if not path.exists():
            raise FileNotFoundError(f"{key} not found: {path}")
    if config.get("fault_surface_csv"):
        path = path_from_config(config, "fault_surface_csv")
        if not path.exists():
            raise FileNotFoundError(f"fault_surface_csv not found: {path}")
    if config.get("dfn_patch_csv"):
        path = path_from_config(config, "dfn_patch_csv")
        if not path.exists():
            raise FileNotFoundError(f"dfn_patch_csv not found: {path}")


def reset_output_images(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("*.png", "*.svg"):
        for path in output_dir.glob(suffix):
            path.unlink()


def load_step3_imaging_fracture_labels(paths: list[Path], well_name: str, target_block: dict[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {"X", "Y", "TIME", "GT_POINT_FLAG"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Step3 imaging group csv missing columns {missing}: {path}")
        work = df[pd.to_numeric(df["GT_POINT_FLAG"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
        work["WellName"] = well_name
        work["Step3GroupCSV"] = str(path)
        parts.append(work)
    if not parts:
        raise RuntimeError("no Step3 imaging group csvs configured")
    work = pd.concat(parts, ignore_index=True)
    for column in ["X", "Y", "TIME", "Density", "Frac_Azimuth", "Frac_Dip"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["X", "Y", "TIME"]).copy()
    if target_block:
        work = work[
            work["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
            & work["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
        ].copy()
    if work.empty:
        raise RuntimeError(f"no Step3 GT_POINT_FLAG fracture labels found for {well_name} inside target block")
    layer_col = "StrataName" if "StrataName" in work.columns else "LayerGroup" if "LayerGroup" in work.columns else None
    keep = ["WellName", "X", "Y", "TIME"]
    for column in [layer_col, "Density", "SampleID", "GT_POINT_FLAG", "Frac_Azimuth", "Frac_Dip", "Step3GroupCSV"]:
        if column and column in work.columns and column not in keep:
            keep.append(column)
    out = work[keep].sort_values("TIME").reset_index(drop=True)
    if layer_col and layer_col != "LayerGroup":
        out = out.rename(columns={layer_col: "LayerGroup"})
    return out


def load_step3_imaging_segment_track(paths: list[Path], well_name: str, target_block: dict[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {"X", "Y", "TIME"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Step3 imaging group csv missing columns {missing}: {path}")
        work = df.copy()
        work["WellName"] = well_name
        work["Step3GroupCSV"] = str(path)
        for column in ["X", "Y", "TIME", "DEPT", "TVD"]:
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(subset=["X", "Y", "TIME"]).copy()
        parts.append(work)
    if not parts:
        raise RuntimeError("no Step3 imaging group csvs configured")
    out = pd.concat(parts, ignore_index=True)
    if target_block:
        out = out[
            out["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
            & out["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
        ].copy()
    if out.empty:
        raise RuntimeError(f"no Step3 imaging segment samples found for {well_name} inside target block")
    keep = ["WellName", "X", "Y", "TIME"]
    for column in ["StrataName", "LayerGroup", "SampleID", "DEPT", "TVD", "Step3GroupCSV"]:
        if column in out.columns and column not in keep:
            keep.append(column)
    return out[keep].sort_values("TIME").reset_index(drop=True)


def finite_bounds_from_curves(curves: list[SurfaceSectionCurve], padding_ms: float) -> tuple[float, float]:
    values = np.concatenate([curve.z[np.isfinite(curve.z)] for curve in curves if np.isfinite(curve.z).any()])
    if values.size == 0:
        raise RuntimeError("surface section curves contain no finite times")
    return float(np.nanmin(values) - padding_ms), float(np.nanmax(values) + padding_ms)


def aligned_local_axis(trace_df: pd.DataFrame, column: str, low: float, high: float) -> np.ndarray:
    values = np.sort(trace_df.loc[trace_df[column].between(low, high), column].unique()).astype(np.float64)
    if values.size < 2:
        raise RuntimeError(f"not enough trace coordinates for local {column} axis: {low}..{high}")
    return values


def make_summary(display: dict[str, float], time_min: float, time_max: float) -> dict[str, float]:
    return {
        "display_x_min": float(display["display_x_min"]),
        "display_x_max": float(display["display_x_max"]),
        "display_y_min": float(display["display_y_min"]),
        "display_y_max": float(display["display_y_max"]),
        "display_time_min": float(time_min),
        "display_time_max": float(time_max),
    }


def segment_lines(
    segments: list[ProjectionSegment], projection: str
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[str], list[float]]:
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    colors: list[str] = []
    widths: list[float] = []
    selected = [segment for segment in segments if segment.projection == projection]
    areas = np.asarray([max(float(segment.patch_area), 1.0) for segment in selected], dtype=float)
    if areas.size > 1 and np.isfinite(areas).all() and float(np.nanmax(areas)) > float(np.nanmin(areas)):
        lo = float(np.nanquantile(areas, 0.05))
        hi = float(np.nanquantile(areas, 0.95))
        if hi <= lo:
            lo = float(np.nanmin(areas))
            hi = float(np.nanmax(areas))
    else:
        lo = hi = 1.0
    for segment in selected:
        if segment.projection != projection:
            continue
        colors.append(DFN_INTERVAL_COLORS.get(segment.interval, "#ef4444"))
        if hi > lo:
            area_norm = float(np.clip((max(float(segment.patch_area), 1.0) - lo) / (hi - lo), 0.0, 1.0))
            width = DFN_WIDTH_MIN + (DFN_WIDTH_MAX - DFN_WIDTH_MIN) * (area_norm ** DFN_WIDTH_POWER)
            scale = DFN_SEGMENT_SCALE_MIN + (DFN_SEGMENT_SCALE_MAX - DFN_SEGMENT_SCALE_MIN) * (area_norm ** DFN_SEGMENT_SCALE_POWER)
        else:
            width = 0.5 * (DFN_WIDTH_MIN + DFN_WIDTH_MAX)
            scale = 1.0
        h_mid = 0.5 * (float(segment.h1) + float(segment.h2))
        z_mid = 0.5 * (float(segment.z1) + float(segment.z2))
        h_half = 0.5 * (float(segment.h2) - float(segment.h1)) * scale
        z_half = 0.5 * (float(segment.z2) - float(segment.z1)) * scale
        lines.append(((h_mid - h_half, z_mid - z_half), (h_mid + h_half, z_mid + z_half)))
        widths.append(float(width))
    return lines, colors, widths


def append_segment_with_mode(segments: list[ProjectionSegment], segment: ProjectionSegment, mode: str) -> None:
    # ProjectionSegment is imported from the shared section module and has no
    # display-mode field; attach it dynamically for Step9-specific styling.
    setattr(segment, "display_mode", mode)
    segments.append(segment)


def plane_polygon_intersection_line(vertices: np.ndarray, axis: int, value: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the line segment where a convex polygon crosses X=value or Y=value."""
    points: list[np.ndarray] = []
    count = len(vertices)
    if count < 3:
        return None
    for idx in range(count):
        p1 = vertices[idx]
        p2 = vertices[(idx + 1) % count]
        d1 = float(p1[axis] - value)
        d2 = float(p2[axis] - value)
        if abs(d1) <= SECTION_INTERSECTION_EPS_M:
            points.append(p1.copy())
        if d1 * d2 < 0.0:
            ratio = abs(d1) / (abs(d1) + abs(d2))
            points.append(p1 + ratio * (p2 - p1))
        elif abs(d2) <= SECTION_INTERSECTION_EPS_M:
            points.append(p2.copy())
    if len(points) < 2:
        return None
    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) <= 1.0e-5 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    arr = np.asarray(unique, dtype=float)
    if axis == 1:  # XZ section: horizontal axis is X.
        coords = arr[:, [0, 2]]
    else:  # YZ section: horizontal axis is Y.
        coords = arr[:, [1, 2]]
    if len(coords) == 2:
        p1, p2 = coords[0], coords[1]
    else:
        line = representative_line_2d(coords)
        if line is None:
            return None
        return line
    if float(np.linalg.norm(p2 - p1)) <= 1.0e-8:
        return None
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def projected_polygon_line(vertices: np.ndarray, projection: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if projection == "XZ":
        coords = vertices[:, [0, 2]]
    elif projection == "YZ":
        coords = vertices[:, [1, 2]]
    else:
        return None
    return representative_line_2d(coords)


def scan_vtk_intersections(args: SimpleNamespace, surfaces: dict[str, Any], well_df: pd.DataFrame) -> tuple[list[ProjectionSegment], dict[str, float | int | str]]:
    well_time = well_df["TIME"].to_numpy(dtype=float)
    well_x = well_df["X"].to_numpy(dtype=float)
    well_y = well_df["Y"].to_numpy(dtype=float)
    segments: list[ProjectionSegment] = []
    skipped_by_curve_time = 0
    skipped_by_interval = 0
    voxel_coverage_interval_exemptions = 0
    skipped_by_area = 0
    skipped_by_geometry = 0
    skipped_by_scale = 0
    skipped_by_surface_distance = 0
    skipped_by_surface_distance_xz = 0
    skipped_by_surface_distance_yz = 0
    skipped_by_no_intersection_xz = 0
    skipped_by_no_intersection_yz = 0
    selected_intersection_xz = 0
    selected_intersection_yz = 0
    selected_small_projection_xz = 0
    selected_small_projection_yz = 0
    selected_patch_count = 0
    processed_polygon_count = 0
    skipped_small_projection_well_control_xz = 0
    skipped_small_projection_well_control_yz = 0
    scanned_scale_counts: dict[str, int] = {}
    allowed_scales = getattr(args, "allowed_scales", None)
    allowed_scale_set = {str(value).strip().lower() for value in allowed_scales} if allowed_scales else None

    with args.input_vtk.open("r", encoding="utf-8", errors="ignore") as handle:
        header = [next_nonempty(handle) for _ in range(4)]
        if not header[0].startswith("# vtk DataFile") or header[2] != "ASCII" or header[3] != "DATASET POLYDATA":
            raise ValueError(f"unsupported ASCII legacy POLYDATA VTK: {args.input_vtk}")
        point_header = next_nonempty(handle).split()
        if len(point_header) < 3 or point_header[0] != "POINTS":
            raise ValueError(f"POINTS block missing: {args.input_vtk}")
        point_count = int(point_header[1])
        print(f"[cheye1-section] reading points for intersection scan: {point_count}", flush=True)
        points = read_points(handle, point_count)
        bounds_min = np.nanmin(points, axis=0)
        bounds_max = np.nanmax(points, axis=0)
        polygon_header = next_nonempty(handle).split()
        if len(polygon_header) < 3 or polygon_header[0] != "POLYGONS":
            raise ValueError(f"POLYGONS block missing: {args.input_vtk}")
        combined_polygon_count = int(polygon_header[1])
        predicted_polygon_count = int(getattr(args, "predicted_cell_count", combined_polygon_count))
        if predicted_polygon_count > combined_polygon_count:
            raise ValueError(f"predicted cell count exceeds VTK polygon count: {args.input_vtk}")
        total_to_scan = (
            min(predicted_polygon_count, int(args.max_polygons))
            if int(args.max_polygons) > 0
            else predicted_polygon_count
        )
        print(
            f"[cheye1-section] intersection scanning predicted polygons: "
            f"{total_to_scan}/{predicted_polygon_count}; combined={combined_polygon_count}",
            flush=True,
        )

        for polygon_index in range(total_to_scan):
            processed_polygon_count += 1
            parts = next_nonempty(handle).split()
            vertex_count = int(parts[0])
            vertex_indices = [int(value) for value in parts[1:]]
            if len(vertex_indices) != vertex_count or vertex_count < 3:
                skipped_by_geometry += 1
                continue
            vertices = points[np.asarray(vertex_indices, dtype=np.int64)]
            if not np.all(np.isfinite(vertices)):
                skipped_by_geometry += 1
                continue
            center = vertices.mean(axis=0)
            center_time = float(center[2])
            area = polygon_area(vertices)
            if area < float(args.min_patch_area) or (float(args.max_patch_area) > 0.0 and area > float(args.max_patch_area)):
                skipped_by_area += 1
                continue
            interval = interval_for_center(center, surfaces)
            if interval is None:
                window_mode = patch_window_mode_for_index(getattr(args, "dfn_patch_metadata", None), polygon_index)
                layer = patch_layer_for_index(getattr(args, "dfn_patch_metadata", None), polygon_index)
                if window_mode == "voxel_coverage" and layer in ("沙三段", "沙四段"):
                    interval = "T4->T6" if layer == "沙三段" else "T6->T7"
                    voxel_coverage_interval_exemptions += 1
                else:
                    skipped_by_interval += 1
                    continue
            patch_scale = patch_scale_for_index(getattr(args, "dfn_patch_metadata", None), polygon_index)
            patch_is_well_control = patch_is_well_control_for_index(getattr(args, "dfn_patch_metadata", None), polygon_index)
            scanned_scale_counts[patch_scale or "unknown"] = int(scanned_scale_counts.get(patch_scale or "unknown", 0)) + 1
            if allowed_scale_set is not None and (patch_scale or "unknown") not in allowed_scale_set:
                skipped_by_scale += 1
                continue

            y_on_well_curve = float(np.interp(center_time, well_time, well_y))
            x_on_well_curve = float(np.interp(center_time, well_time, well_x))
            xz_surface_distance = abs(float(center[1]) - y_on_well_curve)
            yz_surface_distance = abs(float(center[0]) - x_on_well_curve)
            selected_this_patch = False
            xz_added = False
            yz_added = False

            if xz_surface_distance <= float(args.half_width):
                line = plane_polygon_intersection_line(vertices, axis=1, value=y_on_well_curve)
                if line is None:
                    skipped_by_no_intersection_xz += 1
                else:
                    selected_this_patch = True
                    xz_added = True
                    selected_intersection_xz += 1
                    append_segment_with_mode(
                        segments,
                        ProjectionSegment(
                            polygon_index=int(polygon_index),
                            projection="XZ",
                            interval=interval,
                            center_x=float(center[0]),
                            center_y=float(center[1]),
                            center_z=center_time,
                            surface_distance=xz_surface_distance,
                            patch_area=float(area),
                            h1=float(line[0][0]),
                            z1=float(line[0][1]),
                            h2=float(line[1][0]),
                            z2=float(line[1][1]),
                        ),
                        "intersection",
                    )
            else:
                skipped_by_surface_distance_xz += 1
            if (
                bool(getattr(args, "small_projection_enabled", False))
                and not xz_added
                and patch_scale == "small"
                and xz_surface_distance <= float(getattr(args, "small_projection_half_width", args.half_width))
            ):
                if patch_is_well_control and not bool(getattr(args, "small_projection_include_well_control", True)):
                    skipped_small_projection_well_control_xz += 1
                else:
                    line = projected_polygon_line(vertices, "XZ")
                    if line is not None:
                        selected_this_patch = True
                        selected_small_projection_xz += 1
                        append_segment_with_mode(
                            segments,
                            ProjectionSegment(
                                polygon_index=int(polygon_index),
                                projection="XZ",
                                interval=interval,
                                center_x=float(center[0]),
                                center_y=float(center[1]),
                                center_z=center_time,
                                surface_distance=xz_surface_distance,
                                patch_area=float(area),
                                h1=float(line[0][0]),
                                z1=float(line[0][1]),
                                h2=float(line[1][0]),
                                z2=float(line[1][1]),
                            ),
                            "small_projection",
                        )

            if yz_surface_distance <= float(args.half_width):
                line = plane_polygon_intersection_line(vertices, axis=0, value=x_on_well_curve)
                if line is None:
                    skipped_by_no_intersection_yz += 1
                else:
                    selected_this_patch = True
                    yz_added = True
                    selected_intersection_yz += 1
                    append_segment_with_mode(
                        segments,
                        ProjectionSegment(
                            polygon_index=int(polygon_index),
                            projection="YZ",
                            interval=interval,
                            center_x=float(center[0]),
                            center_y=float(center[1]),
                            center_z=center_time,
                            surface_distance=yz_surface_distance,
                            patch_area=float(area),
                            h1=float(line[0][0]),
                            z1=float(line[0][1]),
                            h2=float(line[1][0]),
                            z2=float(line[1][1]),
                        ),
                        "intersection",
                    )
            else:
                skipped_by_surface_distance_yz += 1
            if (
                bool(getattr(args, "small_projection_enabled", False))
                and not yz_added
                and patch_scale == "small"
                and yz_surface_distance <= float(getattr(args, "small_projection_half_width", args.half_width))
            ):
                if patch_is_well_control and not bool(getattr(args, "small_projection_include_well_control", True)):
                    skipped_small_projection_well_control_yz += 1
                else:
                    line = projected_polygon_line(vertices, "YZ")
                    if line is not None:
                        selected_this_patch = True
                        selected_small_projection_yz += 1
                        append_segment_with_mode(
                            segments,
                            ProjectionSegment(
                                polygon_index=int(polygon_index),
                                projection="YZ",
                                interval=interval,
                                center_x=float(center[0]),
                                center_y=float(center[1]),
                                center_z=center_time,
                                surface_distance=yz_surface_distance,
                                patch_area=float(area),
                                h1=float(line[0][0]),
                                z1=float(line[0][1]),
                                h2=float(line[1][0]),
                                z2=float(line[1][1]),
                            ),
                            "small_projection",
                        )

            if not selected_this_patch:
                skipped_by_surface_distance += 1
            else:
                selected_patch_count += 1
            if int(args.max_selected) > 0 and len(segments) >= int(args.max_selected):
                print(f"[cheye1-section] stopped early by max_selected={args.max_selected}", flush=True)
                break
            if args.progress_interval > 0 and (polygon_index + 1) % args.progress_interval == 0:
                print(f"[cheye1-section] intersection scanned {polygon_index + 1}/{total_to_scan}, selected={len(segments)}", flush=True)

    summary = {
        "input_vtk": str(args.input_vtk),
        "well_trajectory_csv": str(args.well_trajectory_csv),
        "surface_dir": str(args.surface_dir),
        "half_width": float(args.half_width),
        "section_geometry_mode": "polygon_plane_intersection",
        "small_projection_enabled": bool(getattr(args, "small_projection_enabled", False)),
        "small_projection_half_width_m": float(getattr(args, "small_projection_half_width", args.half_width)),
        "small_projection_include_well_control": bool(getattr(args, "small_projection_include_well_control", True)),
        "point_count": int(point_count),
        "polygon_count": int(predicted_polygon_count),
        "combined_polygon_count": int(combined_polygon_count),
        "original_fault_polygon_count": int(getattr(args, "original_fault_cell_count", 0)),
        "scanned_polygon_count": int(total_to_scan),
        "processed_polygon_count": int(processed_polygon_count),
        "selected_patch_count": int(selected_patch_count),
        "selected_segment_count": int(len(segments)),
        "selected_segment_count_xz": int(sum(segment.projection == "XZ" for segment in segments)),
        "selected_segment_count_yz": int(sum(segment.projection == "YZ" for segment in segments)),
        "selected_intersection_count": int(selected_intersection_xz + selected_intersection_yz),
        "selected_intersection_count_xz": int(selected_intersection_xz),
        "selected_intersection_count_yz": int(selected_intersection_yz),
        "selected_small_projection_count": int(selected_small_projection_xz + selected_small_projection_yz),
        "selected_small_projection_count_xz": int(selected_small_projection_xz),
        "selected_small_projection_count_yz": int(selected_small_projection_yz),
        "skipped_small_projection_well_control_count": int(skipped_small_projection_well_control_xz + skipped_small_projection_well_control_yz),
        "skipped_small_projection_well_control_count_xz": int(skipped_small_projection_well_control_xz),
        "skipped_small_projection_well_control_count_yz": int(skipped_small_projection_well_control_yz),
        "scanned_scale_counts_after_time_interval_filter": {str(key): int(value) for key, value in sorted(scanned_scale_counts.items())},
        "skipped_by_surface_distance": int(skipped_by_surface_distance),
        "skipped_by_surface_distance_xz": int(skipped_by_surface_distance_xz),
        "skipped_by_surface_distance_yz": int(skipped_by_surface_distance_yz),
        "skipped_by_no_intersection_xz": int(skipped_by_no_intersection_xz),
        "skipped_by_no_intersection_yz": int(skipped_by_no_intersection_yz),
        "skipped_by_curve_time": int(skipped_by_curve_time),
        "skipped_by_interval": int(skipped_by_interval),
        "voxel_coverage_interval_exemption_count": int(voxel_coverage_interval_exemptions),
        "skipped_by_area": int(skipped_by_area),
        "skipped_by_geometry": int(skipped_by_geometry),
        "skipped_by_scale": int(skipped_by_scale),
        "patch_accounting_total": int(
            selected_patch_count + skipped_by_surface_distance + skipped_by_interval + skipped_by_area + skipped_by_geometry + skipped_by_scale
        ),
        "patch_accounting_closed": bool(
            processed_polygon_count
            == selected_patch_count + skipped_by_surface_distance + skipped_by_interval + skipped_by_area + skipped_by_geometry + skipped_by_scale
        ),
        "bounds_x_min": float(bounds_min[0]),
        "bounds_x_max": float(bounds_max[0]),
        "bounds_y_min": float(bounds_min[1]),
        "bounds_y_max": float(bounds_max[1]),
        "bounds_z_min": float(bounds_min[2]),
        "bounds_z_max": float(bounds_max[2]),
    }
    return segments, summary


def scan_fault_surface_csv_intersections(
    fault_csv: Path | None,
    surfaces: dict[str, Any],
    well_df: pd.DataFrame,
    half_width: float,
    *,
    max_polygons: int = 0,
) -> tuple[list[ProjectionSegment], dict[str, Any]]:
    if fault_csv is None:
        return [], {"enabled": False, "segment_count": 0}
    df = pd.read_csv(fault_csv, encoding="utf-8-sig", low_memory=False)
    if "SourceType" in df.columns:
        df = df[df["SourceType"].astype(str).eq("fault_surface")].copy()
    required = [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"fault_surface_csv missing vertex columns: {missing}")
    if int(max_polygons) > 0:
        df = df.head(int(max_polygons)).copy()
    well_time = well_df["TIME"].to_numpy(dtype=float)
    well_x = well_df["X"].to_numpy(dtype=float)
    well_y = well_df["Y"].to_numpy(dtype=float)
    segments: list[ProjectionSegment] = []
    skipped_by_time = 0
    skipped_by_interval = 0
    skipped_by_distance = 0
    skipped_by_no_intersection = 0
    skipped_by_geometry = 0
    for polygon_index, row in df.reset_index(drop=True).iterrows():
        vertices = np.asarray(
            [
                [float(row[f"V{vertex_idx}X"]), float(row[f"V{vertex_idx}Y"]), float(row[f"V{vertex_idx}Z"])]
                for vertex_idx in range(1, 5)
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(vertices)):
            skipped_by_geometry += 1
            continue
        center = vertices.mean(axis=0)
        center_time = float(center[2])
        interval = interval_for_center(center, surfaces)
        if interval is None:
            skipped_by_interval += 1
            continue
        area = polygon_area(vertices)
        y_on_well_curve = float(np.interp(center_time, well_time, well_y))
        x_on_well_curve = float(np.interp(center_time, well_time, well_x))
        selected_this_patch = False
        xz_surface_distance = abs(float(center[1]) - y_on_well_curve)
        if xz_surface_distance <= float(half_width):
            line = plane_polygon_intersection_line(vertices, axis=1, value=y_on_well_curve)
            if line is None:
                line = representative_line_2d(vertices[:, [0, 2]])
            if line is None:
                skipped_by_no_intersection += 1
            else:
                selected_this_patch = True
                segments.append(
                    ProjectionSegment(
                        polygon_index=int(polygon_index),
                        projection="XZ",
                        interval=interval,
                        center_x=float(center[0]),
                        center_y=float(center[1]),
                        center_z=center_time,
                        surface_distance=xz_surface_distance,
                        patch_area=float(area),
                        h1=float(line[0][0]),
                        z1=float(line[0][1]),
                        h2=float(line[1][0]),
                        z2=float(line[1][1]),
                    )
                )
        yz_surface_distance = abs(float(center[0]) - x_on_well_curve)
        if yz_surface_distance <= float(half_width):
            line = plane_polygon_intersection_line(vertices, axis=0, value=x_on_well_curve)
            if line is None:
                line = representative_line_2d(vertices[:, [1, 2]])
            if line is None:
                skipped_by_no_intersection += 1
            else:
                selected_this_patch = True
                segments.append(
                    ProjectionSegment(
                        polygon_index=int(polygon_index),
                        projection="YZ",
                        interval=interval,
                        center_x=float(center[0]),
                        center_y=float(center[1]),
                        center_z=center_time,
                        surface_distance=yz_surface_distance,
                        patch_area=float(area),
                        h1=float(line[0][0]),
                        z1=float(line[0][1]),
                        h2=float(line[1][0]),
                        z2=float(line[1][1]),
                    )
                )
        if not selected_this_patch:
            skipped_by_distance += 1
    return segments, {
        "enabled": True,
        "fault_trace_source": "step7c_fault_surface_csv",
        "fault_surface_csv": str(fault_csv),
        "fault_surface_patch_count": int(len(df)),
        "half_width_m": float(half_width),
        "segment_count": int(len(segments)),
        "segment_count_xz": int(sum(segment.projection == "XZ" for segment in segments)),
        "segment_count_yz": int(sum(segment.projection == "YZ" for segment in segments)),
        "skipped_by_time": int(skipped_by_time),
        "skipped_by_interval": int(skipped_by_interval),
        "skipped_by_distance": int(skipped_by_distance),
        "skipped_by_no_intersection": int(skipped_by_no_intersection),
        "skipped_by_geometry": int(skipped_by_geometry),
    }


def curved_section_triangle_line(
    vertices: np.ndarray,
    projection: str,
    well_time: np.ndarray,
    well_x: np.ndarray,
    well_y: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if projection == "XZ":
        signed = vertices[:, 1] - np.interp(vertices[:, 2], well_time, well_y)
        horizontal_axis = 0
    else:
        signed = vertices[:, 0] - np.interp(vertices[:, 2], well_time, well_x)
        horizontal_axis = 1
    points: list[np.ndarray] = []
    for idx in range(len(vertices)):
        p1 = vertices[idx]
        p2 = vertices[(idx + 1) % len(vertices)]
        d1 = float(signed[idx])
        d2 = float(signed[(idx + 1) % len(vertices)])
        if abs(d1) <= SECTION_INTERSECTION_EPS_M:
            points.append(p1.copy())
        if d1 * d2 < 0.0:
            ratio = abs(d1) / (abs(d1) + abs(d2))
            points.append(p1 + ratio * (p2 - p1))
        elif abs(d2) <= SECTION_INTERSECTION_EPS_M:
            points.append(p2.copy())
    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) <= 1.0e-5 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    coords = np.asarray(unique, dtype=float)[:, [horizontal_axis, 2]]
    if len(coords) == 2:
        p1, p2 = coords
    else:
        line = representative_line_2d(coords)
        if line is None:
            return None
        p1, p2 = np.asarray(line[0], dtype=float), np.asarray(line[1], dtype=float)
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def scan_original_fault_surface_intersections(
    fault_vtk: Path | pv.PolyData | None,
    well_df: pd.DataFrame,
    display: dict[str, float],
) -> tuple[list[ProjectionSegment], dict[str, Any]]:
    if fault_vtk is None:
        return [], {"enabled": False, "segment_count": 0}
    loaded = pv.read(fault_vtk) if isinstance(fault_vtk, Path) else fault_vtk
    mesh = loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface(algorithm="dataset_surface")
    mesh = mesh.triangulate()
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
    well_time = well_df["TIME"].to_numpy(dtype=float)
    well_x = well_df["X"].to_numpy(dtype=float)
    well_y = well_df["Y"].to_numpy(dtype=float)
    segments: list[ProjectionSegment] = []
    counts = {"XZ": 0, "YZ": 0}
    for cell_index, face in enumerate(faces):
        if int(face[0]) != 3:
            continue
        vertices = np.asarray(mesh.points[face[1:4]], dtype=float)
        center = vertices.mean(axis=0)
        area = polygon_area(vertices)
        for projection in ("XZ", "YZ"):
            line = curved_section_triangle_line(vertices, projection, well_time, well_x, well_y)
            if line is None:
                continue
            h_min = float(display["display_x_min"] if projection == "XZ" else display["display_y_min"])
            h_max = float(display["display_x_max"] if projection == "XZ" else display["display_y_max"])
            if max(line[0][0], line[1][0]) < h_min or min(line[0][0], line[1][0]) > h_max:
                continue
            segments.append(
                ProjectionSegment(
                    polygon_index=int(cell_index),
                    projection=projection,
                    interval="original_fault",
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    center_z=float(center[2]),
                    surface_distance=0.0,
                    patch_area=float(area),
                    h1=float(line[0][0]),
                    z1=float(line[0][1]),
                    h2=float(line[1][0]),
                    z2=float(line[1][1]),
                )
            )
            counts[projection] += 1
    return segments, {
        "enabled": True,
        "fault_trace_source": "step8_unified_dfn_original_fault_triangles",
        "original_fault_surface_vtk": str(fault_vtk) if isinstance(fault_vtk, Path) else "embedded_in_step8_unified_vtk",
        "surface_point_count": int(mesh.n_points),
        "surface_triangle_count": int(mesh.n_cells),
        "segment_count": int(len(segments)),
        "segment_count_xz": int(counts["XZ"]),
        "segment_count_yz": int(counts["YZ"]),
        "horizon_filter_applied": False,
        "intersection_mode": "triangle_vs_well_curved_section",
    }


def scan_unified_original_fault_intersections(
    unified_vtk: Path,
    well_df: pd.DataFrame,
    display: dict[str, float],
) -> tuple[list[ProjectionSegment], dict[str, Any]]:
    original_mesh = extract_geometry_group(
        unified_vtk,
        ORIGINAL_FAULT_GEOMETRY_GROUP_CODE,
    ).triangulate()
    segments, summary = scan_original_fault_surface_intersections(original_mesh, well_df, display)
    summary["unified_dfn_vtk"] = str(unified_vtk)
    return segments, summary


def load_original_fault_sticks(path: Path, target_block: dict[str, Any], context_padding_m: float = 10000.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    {
                        "Line": int(float(parts[0])),
                        "Trace": int(float(parts[1])),
                        "X": float(parts[2]),
                        "Y": float(parts[3]),
                        "TIME": float(parts[4]),
                        "Flag": int(float(parts[5])),
                        "FaultName": str(parts[6]),
                    }
                )
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no fault stick rows parsed from {path}")
    if target_block:
        pad = float(context_padding_m)
        df = df[
            df["X"].between(float(target_block["x_min"]) - pad, float(target_block["x_max"]) + pad)
            & df["Y"].between(float(target_block["y_min"]) - pad, float(target_block["y_max"]) + pad)
        ].copy()
    return df.reset_index(drop=True)


def raw_fault_segments_for_projection(
    fault_df: pd.DataFrame,
    projection: str,
    well_df: pd.DataFrame,
    half_width: float,
    display: dict[str, float],
    surfaces: Any,
) -> tuple[list[ProjectionSegment], dict[str, int]]:
    well_time = well_df["TIME"].to_numpy(dtype=float)
    well_x = well_df["X"].to_numpy(dtype=float)
    well_y = well_df["Y"].to_numpy(dtype=float)
    work = fault_df.copy()
    if isinstance(surfaces, HorizonSpatialLookup):
        inside = np.asarray(
            [
                surfaces.interval(float(x), float(y), float(time)) is not None
                for x, y, time in work[["X", "Y", "TIME"]].to_numpy(dtype=float)
            ],
            dtype=bool,
        )
        work = work[inside].copy()
    if work.empty:
        return [], {"selected_point_count": 0, "segment_count": 0, "selected_group_count": 0}
    if projection == "XZ":
        work["SectionDistance"] = np.abs(work["Y"].to_numpy(dtype=float) - np.interp(work["TIME"].to_numpy(dtype=float), well_time, well_y))
        work["H"] = work["X"]
        h_min = float(display["display_x_min"])
        h_max = float(display["display_x_max"])
    else:
        work["SectionDistance"] = np.abs(work["X"].to_numpy(dtype=float) - np.interp(work["TIME"].to_numpy(dtype=float), well_time, well_x))
        work["H"] = work["Y"]
        h_min = float(display["display_y_min"])
        h_max = float(display["display_y_max"])
    selected = work[(work["SectionDistance"] <= float(half_width)) & work["H"].between(h_min, h_max)].copy()
    if selected.empty:
        return [], {"selected_point_count": 0, "segment_count": 0, "selected_group_count": 0}
    segments: list[ProjectionSegment] = []
    group_cols = ["FaultName", "Line", "Flag"]
    selected_group_count = 0
    selected_group_count = 0
    selected_keys = {tuple(row) for row in selected[group_cols].to_numpy()}
    for key, group in work.sort_values(group_cols + ["TIME"]).groupby(group_cols, dropna=False):
        key_tuple = tuple(key if isinstance(key, tuple) else (key,))
        if key_tuple not in selected_keys:
            continue
        if len(group) < 2:
            continue
        arr = group[["H", "TIME", "X", "Y", "SectionDistance"]].to_numpy(dtype=float)
        group_has_segment = False
        for idx in range(len(arr) - 1):
            h1, z1, x1, y1, d1 = arr[idx]
            h2, z2, x2, y2, d2 = arr[idx + 1]
            if not np.all(np.isfinite([h1, z1, h2, z2])):
                continue
            if max(h1, h2) < h_min or min(h1, h2) > h_max:
                continue
            if abs(z2 - z1) > 450.0:
                continue
            # Retain only segments touching the projection band. This keeps a
            # nearby fault visible without drawing its entire remote stick.
            if min(d1, d2) > float(half_width):
                continue
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            center_z = 0.5 * (z1 + z2)
            segments.append(
                ProjectionSegment(
                    polygon_index=int(len(segments)),
                    projection=projection,
                    interval="original_fault",
                    center_x=float(center_x),
                    center_y=float(center_y),
                    center_z=float(center_z),
                    surface_distance=float(0.5 * (d1 + d2)),
                    patch_area=0.0,
                    h1=float(h1),
                    z1=float(z1),
                    h2=float(h2),
                    z2=float(z2),
                )
            )
            group_has_segment = True
        if group_has_segment:
            selected_group_count += 1
    return segments, {
        "selected_point_count": int(len(selected)),
        "segment_count": int(len(segments)),
        "selected_group_count": int(selected_group_count),
    }


def scan_original_fault_stick_traces(
    fault_dat: Path | None,
    target_block: dict[str, Any],
    well_df: pd.DataFrame,
    half_width: float,
    display: dict[str, float],
    surfaces: Any,
    *,
    context_padding_m: float = 10000.0,
) -> tuple[list[ProjectionSegment], dict[str, Any]]:
    if fault_dat is None:
        return [], {"enabled": False, "segment_count": 0}
    fault_df = load_original_fault_sticks(fault_dat, target_block, context_padding_m=context_padding_m)
    xz_segments, xz_summary = raw_fault_segments_for_projection(fault_df, "XZ", well_df, half_width, display, surfaces)
    yz_segments, yz_summary = raw_fault_segments_for_projection(fault_df, "YZ", well_df, half_width, display, surfaces)
    segments = xz_segments + yz_segments
    return segments, {
        "enabled": True,
        "fault_trace_source": "original_fault_stick_dat",
        "original_fault_stick_dat": str(fault_dat),
        "raw_point_count_in_target_with_padding": int(len(fault_df)),
        "context_padding_m": float(context_padding_m),
        "half_width_m": float(half_width),
        "segment_count": int(len(segments)),
        "segment_count_xz": int(len(xz_segments)),
        "segment_count_yz": int(len(yz_segments)),
        "xz": xz_summary,
        "yz": yz_summary,
    }


def add_fault_trace_segments(
    ax,
    segments: list[ProjectionSegment],
    projection: str,
    *,
    overview: bool = False,
) -> int:
    lines = [
        [(segment.h1, segment.z1), (segment.h2, segment.z2)]
        for segment in segments
        if segment.projection == projection
    ]
    if not lines:
        return 0
    halo_width = FAULT_TRACE_OVERVIEW_HALO_WIDTH if overview else FAULT_TRACE_HALO_WIDTH
    line_width = FAULT_TRACE_OVERVIEW_WIDTH if overview else FAULT_TRACE_WIDTH
    line_style = (0, (5, 3)) if overview else "-"
    ax.add_collection(LineCollection(lines, colors=FAULT_TRACE_HALO, linewidths=halo_width, alpha=0.70 if overview else 0.86, zorder=9, linestyles=line_style))
    ax.add_collection(LineCollection(lines, colors=FAULT_TRACE_COLOR, linewidths=line_width, alpha=0.78 if overview else 0.96, zorder=10, linestyles=line_style))
    ax.plot([], [], color=FAULT_TRACE_COLOR, linewidth=line_width, linestyle=line_style, label="断层轨迹")
    return len(lines)


def add_dfn_segments(ax, segments: list[ProjectionSegment], projection: str, *, overlay: bool) -> int:
    intersection_segments = [
        segment
        for segment in segments
        if segment.projection == projection and getattr(segment, "display_mode", "intersection") != "small_projection"
    ]
    projected_segments = [
        segment
        for segment in segments
        if segment.projection == projection and getattr(segment, "display_mode", "intersection") == "small_projection"
    ]
    intersection_lines, intersection_colors, intersection_widths = segment_lines(intersection_segments, projection)
    projected_lines, projected_colors, projected_widths = segment_lines(projected_segments, projection)
    if not intersection_lines and not projected_lines:
        ax.text(0.5, 0.5, "无T4-T7裂缝片段", transform=ax.transAxes, ha="center", va="center")
        return 0
    if projected_lines:
        projected_widths = [max(0.35, width * 0.62) for width in projected_widths]
        if overlay:
            halo_widths = [width + 0.55 for width in projected_widths]
            ax.add_collection(LineCollection(projected_lines, colors=DFN_HALO_COLOR, linewidths=halo_widths, alpha=0.22, zorder=12.6))
        ax.add_collection(LineCollection(projected_lines, colors=projected_colors, linewidths=projected_widths, alpha=0.48, zorder=12.8 if overlay else 1.8))
        ax.plot([], [], color="#6b7280", linewidth=1.4, alpha=0.58, label="小尺度裂缝投影")
    if not intersection_lines:
        return len(projected_lines)
    if overlay:
        halo_widths = [width + 1.35 for width in intersection_widths]
        ax.add_collection(LineCollection(intersection_lines, colors=DFN_HALO_COLOR, linewidths=halo_widths, alpha=0.72, zorder=13.0))
    ax.add_collection(LineCollection(intersection_lines, colors=intersection_colors, linewidths=intersection_widths, alpha=0.92, zorder=13.2 if overlay else 2))
    return len(intersection_lines) + len(projected_lines)


def density_size_factor(values: pd.Series) -> pd.Series:
    density = pd.to_numeric(values, errors="coerce").fillna(float(pd.to_numeric(values, errors="coerce").median()) if pd.to_numeric(values, errors="coerce").notna().any() else 0.0).clip(lower=0.0)
    if float(density.max()) <= float(density.min()):
        return pd.Series(0.5, index=values.index)
    lo = float(density.quantile(0.05))
    hi = float(density.quantile(0.95))
    if hi <= lo:
        lo = float(density.min())
        hi = float(density.max())
    return ((density - lo) / (hi - lo)).clip(0.0, 1.0) ** IMAGING_PATCH_SIZE_POWER


def imaging_patch_vertices(row: pd.Series, length_m: float) -> np.ndarray | None:
    azimuth = pd.to_numeric(pd.Series([row.get("Frac_Azimuth")]), errors="coerce").iloc[0]
    dip = pd.to_numeric(pd.Series([row.get("Frac_Dip")]), errors="coerce").iloc[0]
    x = pd.to_numeric(pd.Series([row.get("X")]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([row.get("Y")]), errors="coerce").iloc[0]
    time_value = pd.to_numeric(pd.Series([row.get("TIME")]), errors="coerce").iloc[0]
    if not np.isfinite([azimuth, dip, x, y, time_value]).all():
        return None
    # Keep Step9 consistent with Step8: Frac_Azimuth is written to AzimuthDeg
    # and used as the patch long-edge direction in VTK geometry.
    theta = np.deg2rad(float(azimuth) % 180.0)
    dip_rad = np.deg2rad(float(np.clip(dip, 1.0, 89.0)))
    strike = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(theta), np.cos(theta)], dtype=float)
    half_length = 0.5 * float(length_m)
    half_short_edge_m = 0.5 * float(length_m) / IMAGING_PATCH_ASPECT_RATIO
    half_dip_xy = half_short_edge_m * float(np.cos(dip_rad))
    half_height = half_short_edge_m * float(np.sin(dip_rad)) / max(IMAGING_PATCH_GEOMETRY_TIME_SCALE_M_PER_MS, 1.0e-6)
    vertices: list[tuple[float, float, float]] = []
    for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        xy = np.asarray([float(x), float(y)], dtype=float) + strike_sign * half_length * strike + dip_sign * half_dip_xy * dip_horizontal
        z = float(time_value) + dip_sign * half_height
        vertices.append((float(xy[0]), float(xy[1]), float(z)))
    return np.asarray(vertices, dtype=float)


def build_imaging_fracture_patch_segments(fracture_df: pd.DataFrame) -> list[ProjectionSegment]:
    if fracture_df.empty or "Density" not in fracture_df.columns:
        return []
    factors = density_size_factor(fracture_df["Density"])
    segments: list[ProjectionSegment] = []
    for idx, row in fracture_df.iterrows():
        factor = float(factors.loc[idx])
        length = IMAGING_PATCH_LENGTH_MIN_M + (IMAGING_PATCH_LENGTH_MAX_M - IMAGING_PATCH_LENGTH_MIN_M) * factor
        vertices = imaging_patch_vertices(row, length_m=length)
        if vertices is None:
            continue
        area = polygon_area(vertices)
        for projection, coords in [("XZ", vertices[:, [0, 2]]), ("YZ", vertices[:, [1, 2]])]:
            line = representative_line_2d(coords)
            if line is None:
                continue
            segments.append(
                ProjectionSegment(
                    polygon_index=int(idx),
                    projection=projection,
                    interval="Step3GT",
                    center_x=float(row["X"]),
                    center_y=float(row["Y"]),
                    center_z=float(row["TIME"]),
                    surface_distance=0.0,
                    patch_area=float(area),
                    h1=float(line[0][0]),
                    z1=float(line[0][1]),
                    h2=float(line[1][0]),
                    z2=float(line[1][1]),
                )
            )
    return segments


def add_imaging_fracture_patch_segments(ax, segments: list[ProjectionSegment], projection: str) -> int:
    selected = [segment for segment in segments if segment.projection == projection]
    if not selected:
        return 0
    lines = [((segment.h1, segment.z1), (segment.h2, segment.z2)) for segment in selected]
    ax.add_collection(LineCollection(lines, colors=IMAGING_PATCH_HALO, linewidths=IMAGING_PATCH_LINE_WIDTH + 1.9, alpha=0.46, zorder=8.1))
    ax.add_collection(LineCollection(lines, colors=IMAGING_PATCH_COLOR, linewidths=IMAGING_PATCH_LINE_WIDTH, alpha=0.78, zorder=8.2, label="Step3真实成像裂缝解释片"))
    return int(len(selected))


def draw_surfaces(ax, surface_curves: list[SurfaceSectionCurve], projection: str) -> None:
    for curve in surface_curves:
        if curve.projection != projection:
            continue
        finite = np.isfinite(curve.h) & np.isfinite(curve.z)
        if int(finite.sum()) < 2:
            continue
        ax.plot(curve.h[finite], curve.z[finite], color="#374151", linestyle="--", linewidth=1.05, alpha=0.82, zorder=4)


def draw_imaging_segment_trajectory(ax, imaging_df: pd.DataFrame, projection: str) -> None:
    if imaging_df.empty:
        return
    h_col = "X" if projection == "XZ" else "Y"
    work = imaging_df.dropna(subset=[h_col, "TIME"]).sort_values("TIME")
    if len(work) < 2:
        return
    ax.plot(work[h_col], work["TIME"], color=IMAGING_SEGMENT_GLOW, linewidth=5.2, alpha=0.78, zorder=7)
    ax.plot(work[h_col], work["TIME"], color=IMAGING_SEGMENT_COLOR, linewidth=2.8, alpha=0.96, zorder=8, label="Step3成像测井井段轨迹")


def set_axes(
    ax,
    projection: str,
    well_df: pd.DataFrame,
    imaging_df: pd.DataFrame,
    summary: dict[str, float],
    z_label: str,
) -> None:
    if projection == "XZ":
        draw_well_trajectory(ax, well_df["X"], well_df["TIME"], label=f"{WELL_NAME}井轨迹")
        ax.set_xlim(summary["display_x_min"], summary["display_x_max"])
        ax.set_xlabel("X / m")
    else:
        draw_well_trajectory(ax, well_df["Y"], well_df["TIME"], label=f"{WELL_NAME}井轨迹")
        ax.set_xlim(summary["display_y_min"], summary["display_y_max"])
        ax.set_xlabel("Y / m")
    draw_imaging_segment_trajectory(ax, imaging_df, projection)
    ax.set_ylim(summary["display_time_min"], summary["display_time_max"])
    ax.invert_yaxis()
    ax.set_ylabel(z_label)
    ax.grid(True, linewidth=0.3, alpha=0.25)


def draw_fracture_labels(ax, fracture_df: pd.DataFrame, projection: str, summary: dict[str, float]) -> int:
    if fracture_df.empty:
        return 0
    h_col = "X" if projection == "XZ" else "Y"
    h_min = summary["display_x_min"] if projection == "XZ" else summary["display_y_min"]
    h_max = summary["display_x_max"] if projection == "XZ" else summary["display_y_max"]
    mask = (
        fracture_df[h_col].between(h_min, h_max)
        & fracture_df["TIME"].between(summary["display_time_min"], summary["display_time_max"])
    )
    labels = fracture_df.loc[mask].copy()
    if labels.empty:
        return 0
    ax.scatter(
        labels[h_col],
        labels["TIME"],
        s=FRACTURE_LABEL_SIZE,
        marker="o",
        facecolors=FRACTURE_LABEL_COLOR,
        edgecolors=FRACTURE_LABEL_EDGE,
        linewidths=0.7,
        alpha=FRACTURE_LABEL_ALPHA,
        label="Step3原始成像测井裂缝点",
        zorder=9,
    )
    return int(len(labels))


def draw_fracture_orientation_symbols(ax, labels: pd.DataFrame, projection: str) -> None:
    required = {"Frac_Azimuth", "Frac_Dip", "TIME"}
    if not required.issubset(labels.columns):
        return
    h_col = "X" if projection == "XZ" else "Y"
    section_azimuth = 90.0 if projection == "XZ" else 0.0
    time_scale_m_per_ms = 2.0
    half_len_m = 34.0 if projection == "XZ" else 34.0
    max_half_dt_ms = 22.0
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for _, row in labels.iterrows():
        azimuth = pd.to_numeric(pd.Series([row.get("Frac_Azimuth")]), errors="coerce").iloc[0]
        dip = pd.to_numeric(pd.Series([row.get("Frac_Dip")]), errors="coerce").iloc[0]
        h = pd.to_numeric(pd.Series([row.get(h_col)]), errors="coerce").iloc[0]
        t = pd.to_numeric(pd.Series([row.get("TIME")]), errors="coerce").iloc[0]
        if not np.isfinite([azimuth, dip, h, t]).all():
            continue
        dip = float(np.clip(dip, 0.0, 89.0))
        directional_factor = float(np.cos(np.deg2rad(section_azimuth - azimuth)))
        dt_half = half_len_m * float(np.tan(np.deg2rad(dip))) * abs(directional_factor) / time_scale_m_per_ms
        dt_half = float(np.clip(dt_half, 2.0, max_half_dt_ms))
        sign = 1.0 if directional_factor >= 0.0 else -1.0
        # TIME increases downward, so the positive-dip side has larger time.
        lines.append(((float(h) - half_len_m, float(t) - sign * dt_half), (float(h) + half_len_m, float(t) + sign * dt_half)))
    if not lines:
        return
    ax.add_collection(LineCollection(lines, colors=FRACTURE_LABEL_EDGE, linewidths=2.2, alpha=0.42, zorder=8))
    ax.add_collection(LineCollection(lines, colors=FRACTURE_LABEL_COLOR, linewidths=1.0, alpha=FRACTURE_ORIENTATION_ALPHA, zorder=8.5, label="Step3裂缝倾向/倾角"))


def visible_fracture_label_count(fracture_df: pd.DataFrame, projection: str, summary: dict[str, float]) -> int:
    h_col = "X" if projection == "XZ" else "Y"
    h_min = summary["display_x_min"] if projection == "XZ" else summary["display_y_min"]
    h_max = summary["display_x_max"] if projection == "XZ" else summary["display_y_max"]
    return int(
        (
            fracture_df[h_col].between(h_min, h_max)
            & fracture_df["TIME"].between(summary["display_time_min"], summary["display_time_max"])
        ).sum()
    )


def plot_dfn(
    segments: list[ProjectionSegment],
    fault_segments: list[ProjectionSegment],
    imaging_patch_segments: list[ProjectionSegment],
    well_df: pd.DataFrame,
    surface_curves: list[SurfaceSectionCurve],
    summary: dict[str, float],
    projection: str,
    output_path: Path,
    args: SimpleNamespace,
    fracture_df: pd.DataFrame,
    imaging_df: pd.DataFrame,
) -> int:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.set_facecolor("#f8fafc")
    count = add_dfn_segments(ax, segments, projection, overlay=False)
    fault_count = add_fault_trace_segments(ax, fault_segments, projection)
    draw_surfaces(ax, surface_curves, projection)
    set_axes(ax, projection, well_df, imaging_df, summary, args.z_label)
    imaging_patch_count = add_imaging_fracture_patch_segments(ax, imaging_patch_segments, projection)
    label_count = draw_fracture_labels(ax, fracture_df, projection, summary)
    for interval, color in DFN_INTERVAL_COLORS.items():
        ax.plot([], [], color=color, linewidth=2.2, label=INTERVAL_LABELS.get(interval, interval))
    ax.legend(loc="upper right")
    ax.set_title(f"{args.title_prefix} | DFN裂缝剖面+断层轨迹+真实成像解释片 | {projection} | DFN={count} | 断层={fault_count} | 解释片={imaging_patch_count} | 交点={label_count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    return count


def plot_coherence(
    section: AttributeSection,
    fault_segments: list[ProjectionSegment],
    imaging_patch_segments: list[ProjectionSegment],
    well_df: pd.DataFrame,
    surface_curves: list[SurfaceSectionCurve],
    summary: dict[str, float],
    output_path: Path,
    args: SimpleNamespace,
    vmin: float,
    vmax: float,
    fracture_df: pd.DataFrame,
    imaging_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    mesh = ax.pcolormesh(
        axis_edges(section.h),
        axis_edges(section.time),
        section.values,
        shading="auto",
        cmap=COHERENCE_CMAP,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    fault_count = add_fault_trace_segments(ax, fault_segments, section.projection)
    draw_surfaces(ax, surface_curves, section.projection)
    set_axes(ax, section.projection, well_df, imaging_df, summary, args.z_label)
    imaging_patch_count = add_imaging_fracture_patch_segments(ax, imaging_patch_segments, section.projection)
    label_count = draw_fracture_labels(ax, fracture_df, section.projection, summary)
    fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.94, label="相干体")
    ax.legend(loc="upper right")
    ax.set_title(f"{args.title_prefix} | 相干体剖面+断层轨迹+真实成像解释片 | {section.projection} | 断层={fault_count} | 解释片={imaging_patch_count} | 交点={label_count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def plot_overlay(
    section: AttributeSection,
    segments: list[ProjectionSegment],
    fault_segments: list[ProjectionSegment],
    imaging_patch_segments: list[ProjectionSegment],
    well_df: pd.DataFrame,
    surface_curves: list[SurfaceSectionCurve],
    summary: dict[str, float],
    output_path: Path,
    args: SimpleNamespace,
    vmin: float,
    vmax: float,
    title_suffix: str,
    fracture_df: pd.DataFrame,
    imaging_df: pd.DataFrame,
) -> int:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    mesh = ax.pcolormesh(
        axis_edges(section.h),
        axis_edges(section.time),
        section.values,
        shading="auto",
        cmap=COHERENCE_CMAP,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    draw_surfaces(ax, surface_curves, section.projection)
    count = add_dfn_segments(ax, segments, section.projection, overlay=True)
    fault_count = add_fault_trace_segments(ax, fault_segments, section.projection)
    set_axes(ax, section.projection, well_df, imaging_df, summary, args.z_label)
    imaging_patch_count = add_imaging_fracture_patch_segments(ax, imaging_patch_segments, section.projection)
    label_count = draw_fracture_labels(ax, fracture_df, section.projection, summary)
    for interval, color in DFN_INTERVAL_COLORS.items():
        ax.plot([], [], color=color, linewidth=2.4, label=INTERVAL_LABELS.get(interval, interval))
    ax.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.94, label="相干体")
    ax.set_title(f"{args.title_prefix} | {title_suffix}+断层轨迹+真实成像解释片 | {section.projection} | DFN={count} | 断层={fault_count} | 解释片={imaging_patch_count} | 交点={label_count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    return count


def main() -> None:
    cli = parse_args()
    config = read_json(cli.config.resolve())
    validate_inputs(config)
    configure_matplotlib_fonts()

    output_dir = path_from_config(config, "output_dir")
    reset_output_images(output_dir)

    standard_args = build_namespace(config, float(config.get("dfn_half_width_m", 50.0)))
    local_args = build_namespace(config, float(config.get("local_dfn_half_width_m", 200.0)))
    local_args.small_projection_half_width = float(
        config.get(
            "local_small_projection_half_width_m",
            config.get("small_projection_half_width_m", local_args.half_width),
        )
    )
    dfn_patch_csv = infer_dfn_patch_csv(config)
    dfn_patch_metadata = load_dfn_patch_metadata(dfn_patch_csv)
    standard_args.dfn_patch_metadata = dfn_patch_metadata
    local_args.dfn_patch_metadata = dfn_patch_metadata
    selected_well = select_demo_well(standard_args)
    standard_args.well_trajectory_csv = selected_well.path
    local_args.well_trajectory_csv = selected_well.path
    standard_args.selected_well_name = selected_well.well_name
    local_args.selected_well_name = selected_well.well_name
    well_df = selected_well.trajectory.copy()
    step3_group_paths = [Path(str(value)).resolve() for value in config.get("step3_imaging_group_csvs", [])]
    fracture_df = load_step3_imaging_fracture_labels(
        step3_group_paths,
        selected_well.well_name,
        dict(config.get("target_block") or {}),
    )
    imaging_patch_segments = build_imaging_fracture_patch_segments(fracture_df)
    imaging_segment_df = load_step3_imaging_segment_track(
        step3_group_paths,
        selected_well.well_name,
        dict(config.get("target_block") or {}),
    )

    trace_df = build_trace_grid(path_from_config(config, "trace_header_csv"), dict(config.get("target_block") or {}))
    trace_tree, trace_ids = build_trace_tree(trace_df)
    surfaces = build_spatial_lookup(config)
    horizon_trace_table_path = resolve_horizon_trace_table(config)
    x_values = np.sort(trace_df["X"].unique()).astype(np.float64)
    y_values = np.sort(trace_df["Y"].unique()).astype(np.float64)

    full_display = {
        "display_x_min": float(trace_df["X"].min()),
        "display_x_max": float(trace_df["X"].max()),
        "display_y_min": float(trace_df["Y"].min()),
        "display_y_max": float(trace_df["Y"].max()),
    }
    full_summary_base = make_summary(full_display, 0.0, 1.0)
    full_surface_curves, full_horizon_summary = build_trace_horizon_section_curves(
        horizon_trace_table_path,
        well_df,
        trace_tree,
        trace_ids,
        x_values,
        y_values,
        iteration_count=int(config.get("horizon_curve_iteration_count", 12)),
    )
    full_time_min, full_time_max = finite_bounds_from_curves(full_surface_curves, float(config.get("time_padding_ms", 20.0)))
    full_summary = make_summary(full_display, full_time_min, full_time_max)

    local_radius = float(config.get("local_axis_radius_m", 200.0))
    x_local = aligned_local_axis(trace_df, "X", float(well_df["X"].min()) - local_radius, float(well_df["X"].max()) + local_radius)
    y_local = aligned_local_axis(trace_df, "Y", float(well_df["Y"].min()) - local_radius, float(well_df["Y"].max()) + local_radius)
    local_display = {
        "display_x_min": float(x_local.min()),
        "display_x_max": float(x_local.max()),
        "display_y_min": float(y_local.min()),
        "display_y_max": float(y_local.max()),
    }
    local_summary_base = make_summary(local_display, 0.0, 1.0)
    local_surface_curves, local_horizon_summary = build_trace_horizon_section_curves(
        horizon_trace_table_path,
        well_df,
        trace_tree,
        trace_ids,
        x_local,
        y_local,
        iteration_count=int(config.get("horizon_curve_iteration_count", 12)),
    )
    local_time_min, local_time_max = finite_bounds_from_curves(local_surface_curves, float(config.get("time_padding_ms", 20.0)))
    local_summary = make_summary(local_display, local_time_min, local_time_max)

    print("[cheye1-section] scanning DFN standard band as section intersections + small-scale band projection", flush=True)
    standard_segments, standard_scan = scan_vtk_intersections(standard_args, surfaces, well_df)
    print("[cheye1-section] scanning DFN local 200m band as section intersections + small-scale band projection", flush=True)
    local_segments, local_scan = scan_vtk_intersections(local_args, surfaces, well_df)
    print("[cheye1-section] scanning fault surface traces", flush=True)
    standard_fault_segments, standard_fault_scan = scan_unified_original_fault_intersections(
        standard_args.input_vtk, well_df, full_display
    )
    local_fault_segments, local_fault_scan = scan_unified_original_fault_intersections(
        standard_args.input_vtk, well_df, local_display
    )

    coherence_path = path_from_config(config, "coherence_volume_path")
    handle, samples, trace_at = open_volume_context(coherence_path)
    try:
        attr_time_min = max(full_time_min, local_time_min, float(np.nanmin(samples)))
        attr_time_max = min(full_time_max, local_time_max, float(np.nanmax(samples)))
        time_values = select_time_samples(samples, attr_time_min, attr_time_max)
        print("[cheye1-section] sampling full coherence XZ/YZ", flush=True)
        full_xz = sample_attribute_section("XZ", x_values, time_values, well_df, trace_tree, trace_ids, samples, trace_at, int(config.get("progress_interval", 100000)))
        full_yz = sample_attribute_section("YZ", y_values, time_values, well_df, trace_tree, trace_ids, samples, trace_at, int(config.get("progress_interval", 100000)))
        print("[cheye1-section] sampling local 200m coherence XZ/YZ", flush=True)
        local_xz = sample_attribute_section("XZ", x_local, time_values, well_df, trace_tree, trace_ids, samples, trace_at, int(config.get("progress_interval", 100000)))
        local_yz = sample_attribute_section("YZ", y_local, time_values, well_df, trace_tree, trace_ids, samples, trace_at, int(config.get("progress_interval", 100000)))
    finally:
        handle.close()

    full_xz_section = AttributeSection("Coherence", "XZ", x_values, time_values, full_xz)
    full_yz_section = AttributeSection("Coherence", "YZ", y_values, time_values, full_yz)
    local_xz_section = AttributeSection("Coherence", "XZ", x_local, time_values, local_xz)
    local_yz_section = AttributeSection("Coherence", "YZ", y_local, time_values, local_yz)
    vmin, vmax = finite_quantile_bounds([full_xz, full_yz, local_xz, local_yz])

    image_counts: dict[str, int | None] = {}
    image_counts["dfn_xz"] = plot_dfn(standard_segments, standard_fault_segments, imaging_patch_segments, well_df, full_surface_curves, full_summary, "XZ", output_dir / "01_dfn_section_xz_t4_t7.png", standard_args, fracture_df, imaging_segment_df)
    image_counts["dfn_yz"] = plot_dfn(standard_segments, standard_fault_segments, imaging_patch_segments, well_df, full_surface_curves, full_summary, "YZ", output_dir / "02_dfn_section_yz_t4_t7.png", standard_args, fracture_df, imaging_segment_df)
    plot_coherence(full_xz_section, standard_fault_segments, imaging_patch_segments, well_df, full_surface_curves, full_summary, output_dir / "03_coherence_section_xz_t4_t7.png", standard_args, vmin, vmax, fracture_df, imaging_segment_df)
    plot_coherence(full_yz_section, standard_fault_segments, imaging_patch_segments, well_df, full_surface_curves, full_summary, output_dir / "04_coherence_section_yz_t4_t7.png", standard_args, vmin, vmax, fracture_df, imaging_segment_df)
    image_counts["overlay_xz"] = plot_overlay(full_xz_section, standard_segments, standard_fault_segments, imaging_patch_segments, well_df, full_surface_curves, full_summary, output_dir / "05_coherence_dfn_overlay_xz_t4_t7.png", standard_args, vmin, vmax, "相干体上的DFN裂缝分布", fracture_df, imaging_segment_df)
    image_counts["overlay_yz"] = plot_overlay(full_yz_section, standard_segments, standard_fault_segments, imaging_patch_segments, well_df, full_surface_curves, full_summary, output_dir / "06_coherence_dfn_overlay_yz_t4_t7.png", standard_args, vmin, vmax, "相干体上的DFN裂缝分布", fracture_df, imaging_segment_df)
    image_counts["local_overlay_xz"] = plot_overlay(local_xz_section, local_segments, local_fault_segments, imaging_patch_segments, well_df, local_surface_curves, local_summary, output_dir / "07_coherence_dfn_overlay_200m_xz_t4_t7.png", local_args, vmin, vmax, "井周200m相干体上的DFN裂缝分布", fracture_df, imaging_segment_df)
    image_counts["local_overlay_yz"] = plot_overlay(local_yz_section, local_segments, local_fault_segments, imaging_patch_segments, well_df, local_surface_curves, local_summary, output_dir / "08_coherence_dfn_overlay_200m_yz_t4_t7.png", local_args, vmin, vmax, "井周200m相干体上的DFN裂缝分布", fracture_df, imaging_segment_df)

    png_files = sorted(path.name for path in output_dir.glob("*.png"))
    summary = {
        "status": "pass" if len(png_files) == 8 and selected_well.well_name == WELL_NAME else "fail",
        "config_path": str(cli.config.resolve()),
        "input_vtk": str(standard_args.input_vtk),
        "output_dir": str(output_dir),
        "retained_png_count": len(png_files),
        "retained_png_files": png_files,
        "selected_well_name": selected_well.well_name,
        "selected_well_csv": str(selected_well.path),
        "selected_well_inside_target_rows": int(selected_well.inside_rows),
        "selected_well_total_rows": int(selected_well.total_rows),
        "selected_well_inside_target_ratio": float(selected_well.inside_ratio),
        "target_block": dict(config.get("target_block") or {}),
        "coherence_volume_path": str(coherence_path),
        "step3_imaging_group_csvs": [str(path) for path in step3_group_paths],
        "step3_label_definition": "GT_POINT_FLAG=1 from Step3 formal_rebuild groups; original imaging-log fracture points mapped to sample grid",
        "step3_imaging_segment_sample_count": int(len(imaging_segment_df)),
        "step3_imaging_segment_time_min": float(imaging_segment_df["TIME"].min()),
        "step3_imaging_segment_time_max": float(imaging_segment_df["TIME"].max()),
        "step3_imaging_fracture_label_count": int(len(fracture_df)),
        "step3_imaging_fracture_label_layer_counts": {
            str(key): int(value) for key, value in fracture_df.get("LayerGroup", pd.Series(dtype=object)).value_counts().sort_index().items()
        },
        "step3_orientation_fields": {
            "azimuth_column": "Frac_Azimuth",
            "dip_column": "Frac_Dip",
            "azimuth_non_null_count": int(fracture_df["Frac_Azimuth"].notna().sum()) if "Frac_Azimuth" in fracture_df.columns else 0,
            "dip_non_null_count": int(fracture_df["Frac_Dip"].notna().sum()) if "Frac_Dip" in fracture_df.columns else 0,
            "projection_rule": "construct temporary Step3 imaging fracture patches centered at original well intersections, keep Frac_Azimuth handling consistent with Step8 AzimuthDeg as patch long-edge direction, constrain in-plane aspect ratio near 3:2, then project patch polygons to XZ/YZ",
        },
        "step3_imaging_fracture_patch_projection": {
            "segment_count_total": int(len(imaging_patch_segments)),
            "segment_count_xz": int(sum(segment.projection == "XZ" for segment in imaging_patch_segments)),
            "segment_count_yz": int(sum(segment.projection == "YZ" for segment in imaging_patch_segments)),
            "center_policy": "patch centers remain on original Step3 well-intersection coordinates",
            "long_edge_length_m_range": [IMAGING_PATCH_LENGTH_MIN_M, IMAGING_PATCH_LENGTH_MAX_M],
            "in_plane_aspect_ratio_long_to_short": IMAGING_PATCH_ASPECT_RATIO,
            "density_size_power": IMAGING_PATCH_SIZE_POWER,
            "density_controls": "long_edge_length",
            "line_width": IMAGING_PATCH_LINE_WIDTH,
            "geometry_time_scale_m_per_ms": IMAGING_PATCH_GEOMETRY_TIME_SCALE_M_PER_MS,
        },
        "visible_fracture_label_counts": {
            "full_xz": visible_fracture_label_count(fracture_df, "XZ", full_summary),
            "full_yz": visible_fracture_label_count(fracture_df, "YZ", full_summary),
            "local_200m_xz": visible_fracture_label_count(fracture_df, "XZ", local_summary),
            "local_200m_yz": visible_fracture_label_count(fracture_df, "YZ", local_summary),
        },
        "coherence_display": {"colormap": COHERENCE_CMAP, "vmin_q02": float(vmin), "vmax_q98": float(vmax)},
        "horizon_display": {
            "horizon_trace_table_path": str(horizon_trace_table_path),
            "full": full_horizon_summary,
            "local_200m": local_horizon_summary,
        },
        "dfn_patch_csv": str(dfn_patch_csv) if dfn_patch_csv is not None else None,
        "dfn_patch_metadata_loaded": dfn_patch_metadata is not None,
        "dfn_patch_metadata_count": int(len(dfn_patch_metadata)) if dfn_patch_metadata is not None else 0,
        "dfn_section_geometry_mode": "polygon_plane_intersection_plus_small_band_projection"
        if bool(config.get("small_projection_enabled", False))
        else "polygon_plane_intersection",
        "dfn_standard_half_width_m": float(standard_args.half_width),
        "dfn_local_half_width_m": float(local_args.half_width),
        "small_projection_enabled": bool(config.get("small_projection_enabled", False)),
        "small_projection_half_width_m": float(standard_args.small_projection_half_width),
        "local_small_projection_half_width_m": float(local_args.small_projection_half_width),
        "local_axis_radius_m": local_radius,
        "trace_count_in_candidate": int(len(trace_df)),
        "full_axis": {"x_count": int(len(x_values)), "y_count": int(len(y_values))},
        "local_axis": {"x_count": int(len(x_local)), "y_count": int(len(y_local)), **local_display},
        "time_sample_count": int(len(time_values)),
        "standard_scan": standard_scan,
        "local_scan": local_scan,
        "standard_fault_trace_scan": standard_fault_scan,
        "local_fault_trace_scan": local_fault_scan,
        "fault_trace_visible_segment_count": int(standard_fault_scan.get("segment_count", 0)) + int(local_fault_scan.get("segment_count", 0)),
        "plotted_segment_counts": image_counts,
        "checks": {
            "exactly_8_png_images": len(png_files) == 8,
            "selected_well_is_cheye1": selected_well.well_name == WELL_NAME,
            "well_fully_inside_candidate": int(selected_well.inside_rows) == int(selected_well.total_rows),
            "standard_has_xz_yz_segments": int(image_counts["dfn_xz"] or 0) > 0 and int(image_counts["dfn_yz"] or 0) > 0,
            "local_has_xz_yz_segments": int(image_counts["local_overlay_xz"] or 0) > 0 and int(image_counts["local_overlay_yz"] or 0) > 0,
            "fault_trace_loaded": bool(
                not standard_fault_scan.get("enabled", False)
                or int(
                    standard_fault_scan.get(
                        "surface_triangle_count",
                        standard_fault_scan.get(
                            "raw_point_count_in_target_with_padding",
                            standard_fault_scan.get("fault_surface_patch_count", 0),
                        ),
                    )
                )
                > 0
            ),
            "coherence_samples_nonempty": int(np.isfinite(full_xz).sum()) > 0 and int(np.isfinite(full_yz).sum()) > 0,
            "step3_gt_point_labels_loaded": int(len(fracture_df)) > 0,
            "step3_imaging_segment_loaded": int(len(imaging_segment_df)) > 0,
            "step3_orientation_fields_complete": "Frac_Azimuth" in fracture_df.columns
            and "Frac_Dip" in fracture_df.columns
            and int(fracture_df["Frac_Azimuth"].notna().sum()) == int(len(fracture_df))
            and int(fracture_df["Frac_Dip"].notna().sum()) == int(len(fracture_df)),
            "all_labels_visible_on_full_sections": visible_fracture_label_count(fracture_df, "XZ", full_summary) == int(len(fracture_df))
            and visible_fracture_label_count(fracture_df, "YZ", full_summary) == int(len(fracture_df)),
        },
    }
    if not all(bool(value) for value in summary["checks"].values()):
        summary["status"] = "fail"
    (output_dir / "section_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cheye1-section] output_dir={output_dir}", flush=True)
    print(f"[cheye1-section] retained_png_count={len(png_files)}", flush=True)
    print(f"[cheye1-section] status={summary['status']}", flush=True)


if __name__ == "__main__":
    main()
