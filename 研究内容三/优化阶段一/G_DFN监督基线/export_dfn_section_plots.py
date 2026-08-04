# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gan_dfn_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib import font_manager

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for portability.
    tqdm = None


CHINESE_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def configure_matplotlib_fonts() -> str | None:
    """Prefer a CJK-capable font so Chinese plot titles render correctly."""
    for font_path in CHINESE_FONT_CANDIDATES:
        path = Path(font_path)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        font_name = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return font_name
    plt.rcParams["axes.unicode_minus"] = False
    return None


SELECTED_FONT_NAME = configure_matplotlib_fonts()


@dataclass
class SectionSegment:
    section: str
    mode: str
    polygon_index: int
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    patch_area: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export X-Z and Y-Z fracture section plots from an ASCII legacy VTK POLYDATA file. "
            "The X-section fixes Y=y0; the Y-section fixes X=x0."
        )
    )
    parser.add_argument("--vtk", type=Path, required=True, help="Input ASCII legacy VTK POLYDATA file.")
    parser.add_argument("--x", type=float, required=True, help="X coordinate of section center.")
    parser.add_argument("--y", type=float, required=True, help="Y coordinate of section center.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for section plots and CSV files.")
    parser.add_argument(
        "--half-width",
        type=float,
        default=25.0,
        help="Half width used by near-projection mode, in the same XY unit as VTK coordinates.",
    )
    parser.add_argument(
        "--along-half-width",
        type=float,
        default=None,
        help=(
            "Optional crop distance along each section. For fixed-Y section it limits X around --x; "
            "for fixed-X section it limits Y around --y. Omit to draw the full section."
        ),
    )
    parser.add_argument("--mode", choices=["exact", "near", "both"], default="both")
    parser.add_argument(
        "--near-render-mode",
        choices=["centerline", "projected_edges"],
        default="centerline",
        help=(
            "How to draw polygons inside the half-width observation band but not cut by the exact section. "
            "centerline draws one representative line per patch; projected_edges keeps the old projected polygon outline."
        ),
    )
    parser.add_argument(
        "--max-near-patches-per-section",
        type=int,
        default=100000,
        help="Safety cap for near-projected polygons per section to avoid unreadable plots.",
    )
    parser.add_argument("--max-polygons", type=int, default=None, help="Optional debug cap for scanned polygons.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--fig-width", type=float, default=14.0)
    parser.add_argument("--fig-height", type=float, default=8.0)
    parser.add_argument("--z-label", type=str, default="Z / Time")
    parser.add_argument("--title-prefix", type=str, default="DFN section")
    parser.add_argument("--invert-z", dest="invert_z", action="store_true", default=True)
    parser.add_argument("--no-invert-z", dest="invert_z", action="store_false")
    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Only write PNG figures. By default both PNG and SVG are written.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100000,
        help="Progress print interval when tqdm is unavailable.",
    )
    return parser


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
        line = next_nonempty(handle)
        buffer.extend(float(value) for value in line.split())
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


def unique_points(points: Iterable[np.ndarray], tol: float = 1e-7) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - existing)) <= tol for existing in unique):
            unique.append(np.asarray(point, dtype=float))
    return unique


def farthest_pair(points: list[np.ndarray], horizontal_axis: int) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    best_pair: tuple[np.ndarray, np.ndarray] | None = None
    best_dist = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            vec = np.asarray([points[i][horizontal_axis] - points[j][horizontal_axis], points[i][2] - points[j][2]])
            dist = float(np.linalg.norm(vec))
            if dist > best_dist:
                best_dist = dist
                best_pair = (points[i], points[j])
    return best_pair


def intersect_polygon_with_axis_plane(vertices: np.ndarray, axis: int, value: float) -> tuple[np.ndarray, np.ndarray] | None:
    coords = vertices[:, axis]
    eps = 1e-8
    if np.nanmin(coords) > value + eps or np.nanmax(coords) < value - eps:
        return None

    intersections: list[np.ndarray] = []
    for start, end in zip(vertices, np.roll(vertices, shift=-1, axis=0)):
        d0 = float(start[axis] - value)
        d1 = float(end[axis] - value)
        if abs(d0) <= eps and abs(d1) <= eps:
            intersections.extend([start, end])
        elif abs(d0) <= eps:
            intersections.append(start)
        elif abs(d1) <= eps:
            intersections.append(end)
        elif d0 * d1 < 0.0:
            ratio = abs(d0) / (abs(d0) + abs(d1))
            intersections.append(start + ratio * (end - start))

    unique = unique_points(intersections)
    horizontal_axis = 0 if axis == 1 else 1
    return farthest_pair(unique, horizontal_axis)


def edge_segments_for_projection(vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(start, end) for start, end in zip(vertices, np.roll(vertices, shift=-1, axis=0))]


def representative_centerline_on_section(
    vertices: np.ndarray,
    fixed_axis: int,
    fixed_value: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Represent a nearby patch by one principal line on the section plane.

    Nearby patches are selected from a finite observation band. Drawing the whole
    projected polygon outline makes the section look like flattened 3D patches,
    so display-only sections use the dominant X-Z or Y-Z footprint instead.
    """
    horizontal_axis = 0 if fixed_axis == 1 else 1
    coords_2d = np.column_stack([vertices[:, horizontal_axis], vertices[:, 2]]).astype(float)
    if len(coords_2d) < 2 or not np.all(np.isfinite(coords_2d)):
        return None

    center = coords_2d.mean(axis=0)
    centered = coords_2d - center
    if float(np.linalg.norm(centered)) <= 1e-8:
        return None

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
    except np.linalg.LinAlgError:
        return None
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return None
    direction = direction / norm

    projections = centered @ direction
    span = float(np.nanmax(projections) - np.nanmin(projections))
    if span <= 1e-8:
        return None
    endpoint_1_2d = center + float(np.nanmin(projections)) * direction
    endpoint_2_2d = center + float(np.nanmax(projections)) * direction

    p1 = np.zeros(3, dtype=float)
    p2 = np.zeros(3, dtype=float)
    p1[fixed_axis] = fixed_value
    p2[fixed_axis] = fixed_value
    p1[horizontal_axis], p1[2] = endpoint_1_2d[0], endpoint_1_2d[1]
    p2[horizontal_axis], p2[2] = endpoint_2_2d[0], endpoint_2_2d[1]
    return p1, p2


def bbox_intersects(bounds_min: float, bounds_max: float, center: float, half_width: float | None) -> bool:
    if half_width is None:
        return True
    return bounds_min <= center + half_width and bounds_max >= center - half_width


def add_segment(
    output: list[SectionSegment],
    section: str,
    mode: str,
    polygon_index: int,
    p1: np.ndarray,
    p2: np.ndarray,
    patch_area: float,
) -> None:
    if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
        return
    if float(np.linalg.norm(p2 - p1)) <= 1e-8:
        return
    output.append(
        SectionSegment(
            section=section,
            mode=mode,
            polygon_index=polygon_index,
            x1=float(p1[0]),
            y1=float(p1[1]),
            z1=float(p1[2]),
            x2=float(p2[0]),
            y2=float(p2[1]),
            z2=float(p2[2]),
            patch_area=float(max(patch_area, 0.0)),
        )
    )


def scan_vtk_sections(args: argparse.Namespace) -> tuple[list[SectionSegment], dict[str, float | int | str]]:
    x_segments: list[SectionSegment] = []
    y_segments: list[SectionSegment] = []
    x_near_patch_count = 0
    y_near_patch_count = 0
    scanned = 0

    with args.vtk.open("r", encoding="utf-8") as handle:
        header = [next_nonempty(handle) for _ in range(4)]
        expect(header[0].startswith("# vtk DataFile"), f"unsupported VTK header: {args.vtk}")
        expect(header[2] == "ASCII", f"only ASCII VTK is supported: {args.vtk}")
        expect(header[3] == "DATASET POLYDATA", f"only POLYDATA VTK is supported: {args.vtk}")

        point_header = next_nonempty(handle).split()
        expect(len(point_header) >= 3 and point_header[0] == "POINTS", f"POINTS block missing: {args.vtk}")
        point_count = int(point_header[1])
        print(f"[section] reading points: {point_count}", flush=True)
        points = read_points(handle, point_count)

        polygon_header = next_nonempty(handle).split()
        expect(len(polygon_header) >= 3 and polygon_header[0] == "POLYGONS", f"POLYGONS block missing: {args.vtk}")
        polygon_count = int(polygon_header[1])
        total_to_scan = min(polygon_count, int(args.max_polygons)) if args.max_polygons else polygon_count
        print(f"[section] scanning polygons: {total_to_scan}/{polygon_count}", flush=True)

        polygon_iter = range(total_to_scan)
        if tqdm is not None:
            polygon_iter = tqdm(polygon_iter, total=total_to_scan, desc="scan polygons", unit="patch")

        for polygon_index in polygon_iter:
            line = next_nonempty(handle)
            parts = line.split()
            expect(len(parts) >= 2, f"invalid polygon row at index {polygon_index}")
            vertex_count = int(parts[0])
            vertex_indices = [int(value) for value in parts[1:]]
            expect(len(vertex_indices) == vertex_count, f"polygon vertex count mismatch at index {polygon_index}")
            vertices = points[np.asarray(vertex_indices, dtype=np.int64)]
            patch_area = polygon_area(vertices)
            min_x, min_y, min_z = np.nanmin(vertices, axis=0)
            max_x, max_y, max_z = np.nanmax(vertices, axis=0)

            x_along_ok = bbox_intersects(min_x, max_x, float(args.x), args.along_half_width)
            y_along_ok = bbox_intersects(min_y, max_y, float(args.y), args.along_half_width)

            exact_x = None
            if x_along_ok and min_y <= args.y <= max_y and args.mode in {"exact", "both"}:
                exact_x = intersect_polygon_with_axis_plane(vertices, axis=1, value=float(args.y))
                if exact_x is not None:
                    add_segment(x_segments, "fixed_y_xz", "exact", polygon_index, exact_x[0], exact_x[1], patch_area)

            exact_y = None
            if y_along_ok and min_x <= args.x <= max_x and args.mode in {"exact", "both"}:
                exact_y = intersect_polygon_with_axis_plane(vertices, axis=0, value=float(args.x))
                if exact_y is not None:
                    add_segment(y_segments, "fixed_x_yz", "exact", polygon_index, exact_y[0], exact_y[1], patch_area)

            if args.mode in {"near", "both"} and float(args.half_width) > 0.0:
                near_x = x_along_ok and min_y <= args.y + args.half_width and max_y >= args.y - args.half_width
                if near_x and (args.mode == "near" or exact_x is None) and x_near_patch_count < args.max_near_patches_per_section:
                    x_near_patch_count += 1
                    if args.near_render_mode == "projected_edges":
                        projected = vertices.copy()
                        projected[:, 1] = float(args.y)
                        for p1, p2 in edge_segments_for_projection(projected):
                            add_segment(x_segments, "fixed_y_xz", "near_projected", polygon_index, p1, p2, patch_area)
                    else:
                        centerline = representative_centerline_on_section(vertices, fixed_axis=1, fixed_value=float(args.y))
                        if centerline is not None:
                            add_segment(x_segments, "fixed_y_xz", "near_centerline", polygon_index, centerline[0], centerline[1], patch_area)

                near_y = y_along_ok and min_x <= args.x + args.half_width and max_x >= args.x - args.half_width
                if near_y and (args.mode == "near" or exact_y is None) and y_near_patch_count < args.max_near_patches_per_section:
                    y_near_patch_count += 1
                    if args.near_render_mode == "projected_edges":
                        projected = vertices.copy()
                        projected[:, 0] = float(args.x)
                        for p1, p2 in edge_segments_for_projection(projected):
                            add_segment(y_segments, "fixed_x_yz", "near_projected", polygon_index, p1, p2, patch_area)
                    else:
                        centerline = representative_centerline_on_section(vertices, fixed_axis=0, fixed_value=float(args.x))
                        if centerline is not None:
                            add_segment(y_segments, "fixed_x_yz", "near_centerline", polygon_index, centerline[0], centerline[1], patch_area)

            scanned += 1
            if tqdm is None and args.progress_interval > 0 and scanned % args.progress_interval == 0:
                print(f"[section] scanned {scanned}/{total_to_scan} polygons", flush=True)

        if args.max_polygons and args.max_polygons < polygon_count:
            print(f"[section] stopped early by --max-polygons={args.max_polygons}", flush=True)

    segments = x_segments + y_segments
    summary = {
        "vtk": str(args.vtk),
        "x": float(args.x),
        "y": float(args.y),
        "mode": str(args.mode),
        "near_render_mode": str(args.near_render_mode),
        "half_width": float(args.half_width),
        "along_half_width": "" if args.along_half_width is None else float(args.along_half_width),
        "point_count": int(point_count),
        "polygon_count": int(polygon_count),
        "scanned_polygon_count": int(scanned),
        "fixed_y_xz_segment_count": int(len(x_segments)),
        "fixed_x_yz_segment_count": int(len(y_segments)),
        "fixed_y_xz_near_patch_count": int(x_near_patch_count),
        "fixed_x_yz_near_patch_count": int(y_near_patch_count),
    }
    return segments, summary


def segment_to_2d(segment: SectionSegment) -> tuple[tuple[float, float], tuple[float, float]]:
    if segment.section == "fixed_y_xz":
        return (segment.x1, segment.z1), (segment.x2, segment.z2)
    return (segment.y1, segment.z1), (segment.y2, segment.z2)


def plot_section(
    segments: list[SectionSegment],
    section: str,
    output_stem: Path,
    args: argparse.Namespace,
    horizontal_label: str,
    fixed_label: str,
) -> None:
    section_segments = [segment for segment in segments if segment.section == section]
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    if section_segments:
        lines = [segment_to_2d(segment) for segment in section_segments]
        values = np.asarray([max(segment.patch_area, 1e-9) for segment in section_segments], dtype=float)
        if np.nanmax(values) > np.nanmin(values) and np.nanmin(values) > 0.0:
            norm = LogNorm(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
        else:
            norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(values)), 1.0))
        collection = LineCollection(lines, cmap="viridis", norm=norm, linewidths=0.55, alpha=0.88)
        collection.set_array(values)
        ax.add_collection(collection)
        ax.autoscale()
        colorbar = fig.colorbar(collection, ax=ax, pad=0.01)
        colorbar.set_label("PatchArea")
    else:
        ax.text(0.5, 0.5, "No fracture segments selected", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel(horizontal_label)
    ax.set_ylabel(args.z_label)
    ax.set_title(f"{args.title_prefix} | {fixed_label} | segments={len(section_segments)}")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    if args.invert_z:
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=args.dpi)
    if not args.no_svg:
        fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def write_segments_csv(path: Path, segments: list[SectionSegment]) -> None:
    fieldnames = [
        "Section",
        "Mode",
        "PolygonIndex",
        "X1",
        "Y1",
        "Z1",
        "X2",
        "Y2",
        "Z2",
        "PatchArea",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "Section": segment.section,
                    "Mode": segment.mode,
                    "PolygonIndex": segment.polygon_index,
                    "X1": f"{segment.x1:.6f}",
                    "Y1": f"{segment.y1:.6f}",
                    "Z1": f"{segment.z1:.6f}",
                    "X2": f"{segment.x2:.6f}",
                    "Y2": f"{segment.y2:.6f}",
                    "Z2": f"{segment.z2:.6f}",
                    "PatchArea": f"{segment.patch_area:.6f}",
                }
            )


def write_summary_csv(path: Path, summary: dict[str, float | int | str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Key", "Value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"Key": key, "Value": value})


def safe_coord(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.3f}".replace("-", "m").replace(".", "p")
    return "nan"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments, summary = scan_vtk_sections(args)
    coord_tag = f"x{safe_coord(args.x)}_y{safe_coord(args.y)}"
    plot_section(
        segments,
        section="fixed_y_xz",
        output_stem=args.output_dir / f"section_fixed_y_xz_{coord_tag}",
        args=args,
        horizontal_label="X",
        fixed_label=f"Y={args.y:.3f}, half_width={args.half_width:.3f}",
    )
    plot_section(
        segments,
        section="fixed_x_yz",
        output_stem=args.output_dir / f"section_fixed_x_yz_{coord_tag}",
        args=args,
        horizontal_label="Y",
        fixed_label=f"X={args.x:.3f}, half_width={args.half_width:.3f}",
    )
    write_segments_csv(args.output_dir / f"section_segments_{coord_tag}.csv", segments)
    write_summary_csv(args.output_dir / f"section_summary_{coord_tag}.csv", summary)
    print(f"[section] output_dir={args.output_dir}", flush=True)
    print(f"[section] fixed_y_xz_segment_count={summary['fixed_y_xz_segment_count']}", flush=True)
    print(f"[section] fixed_x_yz_segment_count={summary['fixed_x_yz_segment_count']}", flush=True)


if __name__ == "__main__":
    main()
