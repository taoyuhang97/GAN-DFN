# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gan_dfn_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import LineCollection

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


CHINESE_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

SURFACE_FILE_MAP = {
    "T5": "T5（沙三中底）.dat",
    "T6": "T6（沙三下底）.dat",
    "T7": "T7（沙四上底）.dat",
}

INTERVAL_COLORS = {
    "T5->T6": "#d97706",
    "T6->T7": "#2563eb",
}

SURFACE_LINE_STYLES = {
    "T5": ("#111827", "-"),
    "T6": ("#059669", "--"),
    "T7": ("#7c3aed", "-.") ,
}


@dataclass
class SurfaceLookup:
    code: str
    path: Path
    xy: np.ndarray
    z: np.ndarray
    tree: object | None


@dataclass
class ProjectionSegment:
    polygon_index: int
    projection: str
    interval: str
    center_x: float
    center_y: float
    center_z: float
    surface_distance: float
    patch_area: float
    h1: float
    z1: float
    h2: float
    z2: float


@dataclass
class SurfaceSectionCurve:
    projection: str
    surface: str
    h: np.ndarray
    z: np.ndarray


def configure_matplotlib_fonts() -> None:
    for font_path in CHINESE_FONT_CANDIDATES:
        path = Path(font_path)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        font_name = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return
    plt.rcParams["axes.unicode_minus"] = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export mine-scale XZ/YZ projection sections for fractures near one well trajectory, "
            "restricted to the T5-T7 interval."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-vtk", type=Path, required=True, help="Input ASCII legacy VTK POLYDATA file.")
    parser.add_argument(
        "--well-trajectory-csv",
        type=Path,
        default=Path("/data/shared/project-oil/wx数据/砂砾岩/研究内容一/井斜/测井-地震时窗/车页1导眼_around_data.csv"),
        help="CSV with X/Y/TIME columns for the well trajectory.",
    )
    parser.add_argument(
        "--surface-dir",
        type=Path,
        default=Path("/data/shared/project-oil/wx数据/砂砾岩/层位"),
        help="Directory containing T5/T6/T7 horizon files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--half-width", type=float, default=50.0, help="XY observation-band half width around the well trajectory, in meters.")
    parser.add_argument("--max-polygons", type=int, default=0, help="Optional debug cap for scanned polygons. 0 means all polygons.")
    parser.add_argument("--max-selected", type=int, default=0, help="Optional cap for selected T5-T7 patches. 0 means unlimited.")
    parser.add_argument("--min-patch-area", type=float, default=0.0)
    parser.add_argument("--max-patch-area", type=float, default=0.0, help="0 disables the upper area filter.")
    parser.add_argument("--fig-width", type=float, default=18.0)
    parser.add_argument("--fig-height", type=float, default=8.2)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--title-prefix", type=str, default="车页1导眼 T5-T7 井轨迹观察带裂缝投影")
    parser.add_argument("--z-label", type=str, default="TWT / ms")
    parser.add_argument("--surface-samples", type=int, default=650, help="Number of horizon samples used to draw T5/T6/T7 lines on each section.")
    parser.add_argument("--no-svg", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100000)
    return parser


def next_nonempty(handle) -> str:
    for line in handle:
        stripped = line.strip()
        if stripped:
            return stripped
    raise ValueError("unexpected end of VTK file")


def read_points(handle, point_count: int) -> np.ndarray:
    points = np.empty((point_count, 3), dtype=np.float64)
    cursor = 0
    buffer: list[float] = []
    while cursor < point_count:
        buffer.extend(float(value) for value in next_nonempty(handle).split())
        while len(buffer) >= 3 and cursor < point_count:
            points[cursor] = (buffer[0], buffer[1], buffer[2])
            del buffer[:3]
            cursor += 1
    return points


def polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    origin = vertices[0]
    area = 0.0
    for idx in range(1, len(vertices) - 1):
        area += 0.5 * float(np.linalg.norm(np.cross(vertices[idx] - origin, vertices[idx + 1] - origin)))
    return area


def representative_line_2d(coords_2d: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(coords_2d) < 2 or not np.all(np.isfinite(coords_2d)):
        return None
    center = coords_2d.mean(axis=0)
    centered = coords_2d - center
    if float(np.linalg.norm(centered)) <= 1e-8:
        return None
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vh[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return None
    direction = direction / norm
    projections = centered @ direction
    min_proj = float(np.nanmin(projections))
    max_proj = float(np.nanmax(projections))
    if max_proj - min_proj <= 1e-8:
        return None
    p1 = center + min_proj * direction
    p2 = center + max_proj * direction
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def read_surface_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no XYZ rows found in surface file: {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, :2].copy(), arr[:, 2].copy()


def load_surface_lookups(surface_dir: Path) -> dict[str, SurfaceLookup]:
    if cKDTree is None:
        raise ImportError("scipy is required for horizon and well-neighborhood lookup")
    lookups: dict[str, SurfaceLookup] = {}
    for code, filename in SURFACE_FILE_MAP.items():
        path = surface_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"surface file not found for {code}: {path}")
        xy, z = read_surface_points(path)
        lookups[code] = SurfaceLookup(code=code, path=path, xy=xy, z=z, tree=cKDTree(xy))
        print(f"[well-section] loaded surface {code}: points={len(z)} path={path}", flush=True)
    return lookups


def query_surface_time(lookup: SurfaceLookup, xy: np.ndarray) -> float:
    assert lookup.tree is not None
    _, index = lookup.tree.query(np.asarray([xy], dtype=float), k=1)
    return float(lookup.z[int(np.asarray(index).reshape(-1)[0])])


def interval_for_center(center: np.ndarray, surfaces: dict[str, SurfaceLookup]) -> str | None:
    xy = center[:2]
    t5 = query_surface_time(surfaces["T5"], xy)
    t6 = query_surface_time(surfaces["T6"], xy)
    t7 = query_surface_time(surfaces["T7"], xy)
    # Keep the same practical inversion handling used elsewhere: upper inverted
    # boundaries are moved upward to the next lower boundary for classification.
    if np.isfinite(t5) and np.isfinite(t6) and t5 > t6:
        t5 = t6
    if np.isfinite(t6) and np.isfinite(t7) and t6 > t7:
        t6 = t7
    z = float(center[2])
    if np.isfinite(t5) and np.isfinite(t6) and t5 <= z < t6:
        return "T5->T6"
    if np.isfinite(t6) and np.isfinite(t7) and t6 <= z <= t7:
        return "T6->T7"
    return None


def load_well_trajectory(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"X", "Y", "TIME"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"well trajectory csv missing columns {missing}: {path}")
    work = df[["X", "Y", "TIME"]].copy()
    for col in ["X", "Y", "TIME"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna().sort_values("TIME").reset_index(drop=True)
    if work.empty:
        raise ValueError(f"well trajectory csv has no valid X/Y/TIME rows: {path}")
    return work


def scan_vtk(args: argparse.Namespace, surfaces: dict[str, SurfaceLookup], well_df: pd.DataFrame) -> tuple[list[ProjectionSegment], dict[str, float | int | str]]:
    well_time = well_df["TIME"].to_numpy(dtype=float)
    well_x = well_df["X"].to_numpy(dtype=float)
    well_y = well_df["Y"].to_numpy(dtype=float)
    well_time_min = float(np.nanmin(well_time))
    well_time_max = float(np.nanmax(well_time))
    segments: list[ProjectionSegment] = []
    skipped_by_surface_distance = 0
    skipped_by_surface_distance_xz = 0
    skipped_by_surface_distance_yz = 0
    skipped_by_curve_time = 0
    skipped_by_interval = 0
    skipped_by_area = 0
    skipped_by_geometry = 0

    with args.input_vtk.open("r", encoding="utf-8", errors="ignore") as handle:
        header = [next_nonempty(handle) for _ in range(4)]
        if not header[0].startswith("# vtk DataFile") or header[2] != "ASCII" or header[3] != "DATASET POLYDATA":
            raise ValueError(f"unsupported ASCII legacy POLYDATA VTK: {args.input_vtk}")
        point_header = next_nonempty(handle).split()
        if len(point_header) < 3 or point_header[0] != "POINTS":
            raise ValueError(f"POINTS block missing: {args.input_vtk}")
        point_count = int(point_header[1])
        print(f"[well-section] reading points: {point_count}", flush=True)
        points = read_points(handle, point_count)
        bounds_min = np.nanmin(points, axis=0)
        bounds_max = np.nanmax(points, axis=0)

        polygon_header = next_nonempty(handle).split()
        if len(polygon_header) < 3 or polygon_header[0] != "POLYGONS":
            raise ValueError(f"POLYGONS block missing: {args.input_vtk}")
        polygon_count = int(polygon_header[1])
        total_to_scan = min(polygon_count, int(args.max_polygons)) if int(args.max_polygons) > 0 else polygon_count
        print(f"[well-section] scanning polygons: {total_to_scan}/{polygon_count}", flush=True)

        iterator = range(total_to_scan)
        if tqdm is not None:
            iterator = tqdm(iterator, total=total_to_scan, desc="scan T5-T7 curved surfaces", unit="patch")
        for polygon_index in iterator:
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
            if center_time < well_time_min or center_time > well_time_max:
                skipped_by_curve_time += 1
                continue

            area = polygon_area(vertices)
            if area < float(args.min_patch_area) or (float(args.max_patch_area) > 0.0 and area > float(args.max_patch_area)):
                skipped_by_area += 1
                continue
            interval = interval_for_center(center, surfaces)
            if interval is None:
                skipped_by_interval += 1
                continue

            xz_line = representative_line_2d(vertices[:, [0, 2]])
            yz_line = representative_line_2d(vertices[:, [1, 2]])
            if xz_line is None or yz_line is None:
                skipped_by_geometry += 1
                continue

            y_on_well_curve = float(np.interp(center_time, well_time, well_y))
            x_on_well_curve = float(np.interp(center_time, well_time, well_x))
            xz_surface_distance = abs(float(center[1]) - y_on_well_curve)
            yz_surface_distance = abs(float(center[0]) - x_on_well_curve)
            selected_this_patch = False

            if xz_surface_distance <= float(args.half_width):
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
                        h1=xz_line[0][0],
                        z1=xz_line[0][1],
                        h2=xz_line[1][0],
                        z2=xz_line[1][1],
                    )
                )
            else:
                skipped_by_surface_distance_xz += 1

            if yz_surface_distance <= float(args.half_width):
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
                        h1=yz_line[0][0],
                        z1=yz_line[0][1],
                        h2=yz_line[1][0],
                        z2=yz_line[1][1],
                    )
                )
            else:
                skipped_by_surface_distance_yz += 1

            if not selected_this_patch:
                skipped_by_surface_distance += 1
            if int(args.max_selected) > 0 and len(segments) >= int(args.max_selected):
                print(f"[well-section] stopped early by --max-selected={args.max_selected}", flush=True)
                break
            if tqdm is None and args.progress_interval > 0 and (polygon_index + 1) % args.progress_interval == 0:
                print(f"[well-section] scanned {polygon_index + 1}/{total_to_scan}, selected={len(segments)}", flush=True)

    summary = {
        "input_vtk": str(args.input_vtk),
        "well_trajectory_csv": str(args.well_trajectory_csv),
        "surface_dir": str(args.surface_dir),
        "half_width": float(args.half_width),
        "point_count": int(point_count),
        "polygon_count": int(polygon_count),
        "scanned_polygon_count": int(total_to_scan),
        "selected_segment_count": int(len(segments)),
        "selected_segment_count_xz": int(sum(segment.projection == "XZ" for segment in segments)),
        "selected_segment_count_yz": int(sum(segment.projection == "YZ" for segment in segments)),
        "skipped_by_surface_distance": int(skipped_by_surface_distance),
        "skipped_by_surface_distance_xz": int(skipped_by_surface_distance_xz),
        "skipped_by_surface_distance_yz": int(skipped_by_surface_distance_yz),
        "skipped_by_curve_time": int(skipped_by_curve_time),
        "skipped_by_interval": int(skipped_by_interval),
        "skipped_by_area": int(skipped_by_area),
        "skipped_by_geometry": int(skipped_by_geometry),
        "bounds_x_min": float(bounds_min[0]),
        "bounds_x_max": float(bounds_max[0]),
        "bounds_y_min": float(bounds_min[1]),
        "bounds_y_max": float(bounds_max[1]),
        "bounds_z_min": float(bounds_min[2]),
        "bounds_z_max": float(bounds_max[2]),
    }
    return segments, summary


def write_segments_csv(path: Path, segments: list[ProjectionSegment]) -> None:
    fieldnames = list(ProjectionSegment.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow({field: getattr(segment, field) for field in fieldnames})


def write_well_projection_csv(path: Path, well_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    well_df.to_csv(path, index=False, encoding="utf-8-sig")


def make_line_collection(lines: list[tuple[tuple[float, float], tuple[float, float]]], colors: list[str], linewidths: list[float]) -> LineCollection:
    return LineCollection(lines, colors=colors, linewidths=linewidths, alpha=0.76)


def build_surface_section_curves(
    surfaces: dict[str, SurfaceLookup],
    well_df: pd.DataFrame,
    summary: dict[str, float | int | str],
    args: argparse.Namespace,
) -> list[SurfaceSectionCurve]:
    """Sample horizon intersections with the two well-controlled curved sections.

    XZ section: Y is controlled by the well trajectory curve Y_w(TIME), while X
    spans the whole mine. YZ section is symmetric with X_w(TIME).
    """
    well_time = well_df["TIME"].to_numpy(dtype=float)
    well_x = well_df["X"].to_numpy(dtype=float)
    well_y = well_df["Y"].to_numpy(dtype=float)
    sample_count = max(50, int(args.surface_samples))
    x_values = np.linspace(float(summary["bounds_x_min"]), float(summary["bounds_x_max"]), sample_count)
    y_values = np.linspace(float(summary["bounds_y_min"]), float(summary["bounds_y_max"]), sample_count)
    curves: list[SurfaceSectionCurve] = []

    for code, lookup in surfaces.items():
        xz_times: list[float] = []
        for x_value in x_values:
            z_guess = float(np.nanmedian(lookup.z))
            for _ in range(5):
                y_on_curve = float(np.interp(z_guess, well_time, well_y))
                z_next = query_surface_time(lookup, np.asarray([x_value, y_on_curve], dtype=float))
                if not np.isfinite(z_next) or abs(z_next - z_guess) < 1e-3:
                    z_guess = z_next
                    break
                z_guess = z_next
            xz_times.append(float(z_guess))
        curves.append(SurfaceSectionCurve(projection="XZ", surface=code, h=x_values.copy(), z=np.asarray(xz_times, dtype=float)))

        yz_times: list[float] = []
        for y_value in y_values:
            z_guess = float(np.nanmedian(lookup.z))
            for _ in range(5):
                x_on_curve = float(np.interp(z_guess, well_time, well_x))
                z_next = query_surface_time(lookup, np.asarray([x_on_curve, y_value], dtype=float))
                if not np.isfinite(z_next) or abs(z_next - z_guess) < 1e-3:
                    z_guess = z_next
                    break
                z_guess = z_next
            yz_times.append(float(z_guess))
        curves.append(SurfaceSectionCurve(projection="YZ", surface=code, h=y_values.copy(), z=np.asarray(yz_times, dtype=float)))
    return curves


def write_surface_curves_csv(path: Path, curves: list[SurfaceSectionCurve]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["projection", "surface", "h", "time"])
        writer.writeheader()
        for curve in curves:
            for h_value, z_value in zip(curve.h, curve.z):
                writer.writerow(
                    {
                        "projection": curve.projection,
                        "surface": curve.surface,
                        "h": float(h_value),
                        "time": float(z_value),
                    }
                )


def draw_surface_curves(ax, curves: list[SurfaceSectionCurve], projection: str) -> None:
    for curve in curves:
        if curve.projection != projection:
            continue
        finite = np.isfinite(curve.h) & np.isfinite(curve.z)
        if int(finite.sum()) < 2:
            continue
        color, linestyle = SURFACE_LINE_STYLES.get(curve.surface, ("#334155", "--"))
        ax.plot(
            curve.h[finite],
            curve.z[finite],
            color=color,
            linestyle=linestyle,
            linewidth=1.65,
            alpha=0.9,
            label=f"{curve.surface}层位界面",
            zorder=3,
        )


def plot_projection(
    segments: list[ProjectionSegment],
    well_df: pd.DataFrame,
    surface_curves: list[SurfaceSectionCurve],
    summary: dict[str, float | int | str],
    output_stem: Path,
    projection: str,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    colors: list[str] = []
    widths: list[float] = []
    for segment in segments:
        if segment.projection != projection:
            continue
        line = ((segment.h1, segment.z1), (segment.h2, segment.z2))
        lines.append(line)
        colors.append(INTERVAL_COLORS.get(segment.interval, "#475569"))
        widths.append(float(np.clip(0.35 + 0.18 * math.log10(max(segment.patch_area, 1.0)), 0.35, 1.25)))

    if lines:
        ax.add_collection(make_line_collection(lines, colors, widths))
    else:
        ax.text(0.5, 0.5, "无 T5-T7 观察带裂缝", transform=ax.transAxes, ha="center", va="center")

    draw_surface_curves(ax, surface_curves, projection)

    if projection == "XZ":
        ax.plot(well_df["X"], well_df["TIME"], color="#dc2626", linewidth=2.4, label="车页1导眼井轨迹", zorder=4)
        ax.set_xlim(float(summary["bounds_x_min"]), float(summary["bounds_x_max"]))
        ax.set_xlabel("X / m")
    else:
        ax.plot(well_df["Y"], well_df["TIME"], color="#dc2626", linewidth=2.4, label="车页1导眼井轨迹", zorder=4)
        ax.set_xlim(float(summary["bounds_y_min"]), float(summary["bounds_y_max"]))
        ax.set_xlabel("Y / m")
    ax.set_ylim(float(summary["bounds_z_min"]), float(summary["bounds_z_max"]))
    ax.invert_yaxis()
    ax.set_ylabel(args.z_label)
    ax.grid(True, linewidth=0.35, alpha=0.32)
    for interval, color in INTERVAL_COLORS.items():
        ax.plot([], [], color=color, linewidth=2.0, label=interval)
    ax.legend(loc="upper right")
    ax.set_title(f"{args.title_prefix} | {projection} 曲面投影 | T5-T7 | 线段数={len(lines)} | half-width={args.half_width:g}m")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=args.dpi)
    if not args.no_svg:
        fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_combined(
    segments: list[ProjectionSegment],
    well_df: pd.DataFrame,
    surface_curves: list[SurfaceSectionCurve],
    summary: dict[str, float | int | str],
    output_stem: Path,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width * 1.35, args.fig_height), sharey=True)
    for ax, projection in zip(axes, ["XZ", "YZ"]):
        lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
        colors: list[str] = []
        widths: list[float] = []
        for segment in segments:
            if segment.projection != projection:
                continue
            line = ((segment.h1, segment.z1), (segment.h2, segment.z2))
            lines.append(line)
            colors.append(INTERVAL_COLORS.get(segment.interval, "#475569"))
            widths.append(float(np.clip(0.35 + 0.18 * math.log10(max(segment.patch_area, 1.0)), 0.35, 1.25)))
        if lines:
            ax.add_collection(make_line_collection(lines, colors, widths))
        else:
            ax.text(0.5, 0.5, "无 T5-T7 观察带裂缝", transform=ax.transAxes, ha="center", va="center")
        draw_surface_curves(ax, surface_curves, projection)
        if projection == "XZ":
            ax.plot(well_df["X"], well_df["TIME"], color="#dc2626", linewidth=2.4, label="车页1导眼井轨迹", zorder=4)
            ax.set_xlim(float(summary["bounds_x_min"]), float(summary["bounds_x_max"]))
            ax.set_xlabel("X / m")
        else:
            ax.plot(well_df["Y"], well_df["TIME"], color="#dc2626", linewidth=2.4, label="车页1导眼井轨迹", zorder=4)
            ax.set_xlim(float(summary["bounds_y_min"]), float(summary["bounds_y_max"]))
            ax.set_xlabel("Y / m")
        ax.set_ylim(float(summary["bounds_z_min"]), float(summary["bounds_z_max"]))
        ax.invert_yaxis()
        ax.grid(True, linewidth=0.35, alpha=0.32)
        ax.set_title(f"{projection} 曲面投影 | 线段数={len(lines)}")
    axes[0].set_ylabel(args.z_label)
    for interval, color in INTERVAL_COLORS.items():
        axes[1].plot([], [], color=color, linewidth=2.0, label=interval)
    for surface, (color, linestyle) in SURFACE_LINE_STYLES.items():
        axes[1].plot([], [], color=color, linestyle=linestyle, linewidth=1.65, label=f"{surface}层位界面")
    axes[1].plot([], [], color="#dc2626", linewidth=2.4, label="车页1导眼井轨迹")
    axes[1].legend(loc="upper right")
    fig.suptitle(f"{args.title_prefix} | T5-T7 曲面观察带 | half-width={args.half_width:g}m | 线段数={len(segments)}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_stem.with_suffix(".png"), dpi=args.dpi)
    if not args.no_svg:
        fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    configure_matplotlib_fonts()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    well_df = load_well_trajectory(args.well_trajectory_csv)
    surfaces = load_surface_lookups(args.surface_dir)
    segments, summary = scan_vtk(args, surfaces, well_df)
    surface_curves = build_surface_section_curves(surfaces, well_df, summary, args)

    summary.update(
        {
            "well_point_count": int(len(well_df)),
            "well_x_min": float(well_df["X"].min()),
            "well_x_max": float(well_df["X"].max()),
            "well_y_min": float(well_df["Y"].min()),
            "well_y_max": float(well_df["Y"].max()),
            "well_time_min": float(well_df["TIME"].min()),
            "well_time_max": float(well_df["TIME"].max()),
            "interval_counts": {interval: int(sum(s.interval == interval for s in segments)) for interval in sorted(INTERVAL_COLORS)},
            "interval_counts_xz": {interval: int(sum(s.projection == "XZ" and s.interval == interval for s in segments)) for interval in sorted(INTERVAL_COLORS)},
            "interval_counts_yz": {interval: int(sum(s.projection == "YZ" and s.interval == interval for s in segments)) for interval in sorted(INTERVAL_COLORS)},
        }
    )

    write_segments_csv(args.output_dir / "t5_t7_projection_segments.csv", segments)
    write_well_projection_csv(args.output_dir / "well_trajectory_projected.csv", well_df)
    write_surface_curves_csv(args.output_dir / "t5_t7_surface_section_curves.csv", surface_curves)
    (args.output_dir / "t5_t7_projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_projection(segments, well_df, surface_curves, summary, args.output_dir / "section_xz_t5_t7", "XZ", args)
    plot_projection(segments, well_df, surface_curves, summary, args.output_dir / "section_yz_t5_t7", "YZ", args)
    plot_combined(segments, well_df, surface_curves, summary, args.output_dir / "section_xz_yz_t5_t7_combined", args)

    print(f"[well-section] output_dir={args.output_dir}", flush=True)
    print(f"[well-section] selected_segment_count={summary['selected_segment_count']}", flush=True)
    print(f"[well-section] interval_counts={summary['interval_counts']}", flush=True)


if __name__ == "__main__":
    main()
