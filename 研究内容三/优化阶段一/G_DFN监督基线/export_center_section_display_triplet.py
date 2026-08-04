# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gan_dfn_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize


THIS_DIR = Path(__file__).resolve().parent
SECTION_SCRIPT = THIS_DIR / "export_dfn_section_plots.py"
CONNECTED_SCRIPT = THIS_DIR / "export_connected_display_dfn.py"

CHINESE_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def configure_matplotlib_fonts() -> str | None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate original, connected-display and density-background section figures for one DFN VTK.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--half-width", type=float, default=25.0)
    parser.add_argument(
        "--along-half-width",
        type=float,
        default=100000.0,
        help="Crop distance along sections. Large default keeps the full local region.",
    )
    parser.add_argument("--title-prefix", type=str, default="BX33-35_BY33-35 区域中心剖面")
    parser.add_argument("--z-label", type=str, default="TWT / ms")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--fig-width", type=float, default=18.0)
    parser.add_argument("--fig-height", type=float, default=7.2)
    parser.add_argument("--density-x-bins", type=int, default=90)
    parser.add_argument("--density-z-bins", type=int, default=70)
    parser.add_argument("--density-weight", choices=["count", "area"], default="count")
    parser.add_argument("--max-connect-distance", type=float, default=150.0)
    parser.add_argument("--max-z-gap", type=float, default=40.0)
    parser.add_argument("--minor-offset-limit", type=float, default=45.0)
    parser.add_argument("--azimuth-tol", type=float, default=25.0)
    parser.add_argument("--dip-tol", type=float, default=20.0)
    parser.add_argument("--min-along-ratio", type=float, default=0.68)
    parser.add_argument("--max-bridges", type=int, default=1800)
    parser.add_argument("--max-bridges-per-patch", type=int, default=2)
    parser.add_argument("--bridge-height-scale", type=float, default=0.75)
    parser.add_argument("--max-bridge-length", type=float, default=180.0)
    parser.add_argument("--no-svg", action="store_true")
    return parser


def run_command(command: list[str]) -> None:
    print("[triplet] run:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def safe_coord(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.3f}".replace("-", "m").replace(".", "p")
    return "nan"


def read_segments_csv(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "Section": str(row["Section"]),
                        "Mode": str(row.get("Mode", "")),
                        "PolygonIndex": int(float(row.get("PolygonIndex", 0))),
                        "X1": float(row["X1"]),
                        "Y1": float(row["Y1"]),
                        "Z1": float(row["Z1"]),
                        "X2": float(row["X2"]),
                        "Y2": float(row["Y2"]),
                        "Z2": float(row["Z2"]),
                        "PatchArea": float(row.get("PatchArea", 0.0)),
                    }
                )
            except Exception:
                continue
    return rows


def section_segments_to_lines(rows: list[dict[str, float | int | str]], section: str) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], np.ndarray]:
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    values: list[float] = []
    for row in rows:
        if row["Section"] != section:
            continue
        if section == "fixed_y_xz":
            p1 = (float(row["X1"]), float(row["Z1"]))
            p2 = (float(row["X2"]), float(row["Z2"]))
        else:
            p1 = (float(row["Y1"]), float(row["Z1"]))
            p2 = (float(row["Y2"]), float(row["Z2"]))
        if not all(math.isfinite(v) for v in (*p1, *p2)):
            continue
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= 1e-8:
            continue
        lines.append((p1, p2))
        values.append(max(float(row["PatchArea"]), 1e-9))
    return lines, np.asarray(values, dtype=float)


def add_line_collection(ax, lines, values, *, linewidth: float, alpha: float, cmap: str = "viridis"):
    if not lines:
        ax.text(0.5, 0.5, "无剖面裂缝线段", transform=ax.transAxes, ha="center", va="center")
        return None
    if len(values) and float(np.nanmax(values)) > float(np.nanmin(values)) and float(np.nanmin(values)) > 0.0:
        norm = LogNorm(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    else:
        norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(values)) if len(values) else 1.0, 1.0))
    collection = LineCollection(lines, cmap=cmap, norm=norm, linewidths=linewidth, alpha=alpha)
    collection.set_array(values)
    ax.add_collection(collection)
    ax.autoscale()
    return collection


def plot_line_triplet(rows: list[dict[str, float | int | str]], output_png: Path, output_svg: Path | None, title: str, args: argparse.Namespace) -> dict[str, int]:
    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height), sharey=False)
    counts: dict[str, int] = {}
    last_collection = None
    for ax, section, horizontal_label, fixed_text in [
        (axes[0], "fixed_y_xz", "X", f"固定 Y={args.y:.3f}"),
        (axes[1], "fixed_x_yz", "Y", f"固定 X={args.x:.3f}"),
    ]:
        lines, values = section_segments_to_lines(rows, section)
        counts[section] = len(lines)
        last_collection = add_line_collection(ax, lines, values, linewidth=0.62, alpha=0.88)
        ax.set_title(f"{fixed_text} | 线段数={len(lines)}")
        ax.set_xlabel(horizontal_label)
        ax.set_ylabel(args.z_label)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.invert_yaxis()
    if last_collection is not None:
        colorbar = fig.colorbar(last_collection, ax=axes.ravel().tolist(), pad=0.01)
        colorbar.set_label("PatchArea")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.94))
    fig.savefig(output_png, dpi=args.dpi)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)
    return counts


def midpoint_density(rows: list[dict[str, float | int | str]], section: str, args: argparse.Namespace):
    xs: list[float] = []
    zs: list[float] = []
    weights: list[float] = []
    for row in rows:
        if row["Section"] != section:
            continue
        if section == "fixed_y_xz":
            h1, h2 = float(row["X1"]), float(row["X2"])
        else:
            h1, h2 = float(row["Y1"]), float(row["Y2"])
        z1, z2 = float(row["Z1"]), float(row["Z2"])
        if not all(math.isfinite(v) for v in (h1, h2, z1, z2)):
            continue
        xs.append((h1 + h2) / 2.0)
        zs.append((z1 + z2) / 2.0)
        weights.append(max(float(row["PatchArea"]), 1e-9) if args.density_weight == "area" else 1.0)
    if not xs:
        return None
    x_arr = np.asarray(xs, dtype=float)
    z_arr = np.asarray(zs, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    hist, x_edges, z_edges = np.histogram2d(
        x_arr,
        z_arr,
        bins=[args.density_x_bins, args.density_z_bins],
        weights=w_arr,
    )
    return hist.T, x_edges, z_edges, len(xs)


def plot_density_triplet(rows: list[dict[str, float | int | str]], output_png: Path, output_svg: Path | None, title: str, args: argparse.Namespace) -> dict[str, int]:
    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height), sharey=False)
    counts: dict[str, int] = {}
    last_mesh = None
    for ax, section, horizontal_label, fixed_text in [
        (axes[0], "fixed_y_xz", "X", f"固定 Y={args.y:.3f}"),
        (axes[1], "fixed_x_yz", "Y", f"固定 X={args.x:.3f}"),
    ]:
        density = midpoint_density(rows, section, args)
        lines, values = section_segments_to_lines(rows, section)
        counts[section] = len(lines)
        if density is None:
            ax.text(0.5, 0.5, "无密度统计样本", transform=ax.transAxes, ha="center", va="center")
        else:
            hist, x_edges, z_edges, sample_count = density
            positive = hist[hist > 0]
            norm = LogNorm(vmin=max(float(np.nanmin(positive)), 1e-9), vmax=float(np.nanmax(positive))) if len(positive) and float(np.nanmax(positive)) > float(np.nanmin(positive)) else Normalize(vmin=0.0, vmax=max(float(np.nanmax(hist)), 1.0))
            last_mesh = ax.pcolormesh(x_edges, z_edges, hist, cmap="magma", norm=norm, shading="auto", alpha=0.82)
            ax.set_title(f"{fixed_text} | 密度样本={sample_count}")
        add_line_collection(ax, lines, values, linewidth=0.22, alpha=0.25, cmap="Greys")
        ax.set_xlabel(horizontal_label)
        ax.set_ylabel(args.z_label)
        ax.grid(True, linewidth=0.25, alpha=0.25)
        ax.invert_yaxis()
    if last_mesh is not None:
        colorbar = fig.colorbar(last_mesh, ax=axes.ravel().tolist(), pad=0.01)
        colorbar.set_label("裂缝密度" if args.density_weight == "count" else "裂缝面积密度")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.94))
    fig.savefig(output_png, dpi=args.dpi)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)
    return counts


def find_segment_csv(section_dir: Path, x: float, y: float) -> Path:
    coord_tag = f"x{safe_coord(x)}_y{safe_coord(y)}"
    path = section_dir / f"section_segments_{coord_tag}.csv"
    if path.exists():
        return path
    candidates = sorted(section_dir.glob("section_segments_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"section segment csv not found in: {section_dir}")
    return candidates[0]


def main() -> None:
    args = build_parser().parse_args()
    configure_matplotlib_fonts()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_section_dir = args.output_dir / "01_original_section_files"
    bridge_section_dir = args.output_dir / "02_connected_section_files"
    bridge_vtk = args.output_dir / "connected_display_center_dfn.vtk"

    common_section_args = [
        "--x", str(args.x),
        "--y", str(args.y),
        "--half-width", str(args.half_width),
        "--along-half-width", str(args.along_half_width),
        "--mode", "both",
        "--z-label", args.z_label,
        "--dpi", str(args.dpi),
        "--fig-width", "14",
        "--fig-height", "8",
    ]

    run_command([
        sys.executable,
        str(SECTION_SCRIPT),
        "--vtk", str(args.input_vtk),
        "--output-dir", str(raw_section_dir),
        "--title-prefix", f"{args.title_prefix} 原始剖面",
        *common_section_args,
    ])

    run_command([
        sys.executable,
        str(CONNECTED_SCRIPT),
        "--input-vtk", str(args.input_vtk),
        "--output-vtk", str(bridge_vtk),
        "--x", str(args.x),
        "--y", str(args.y),
        "--half-width", str(args.half_width),
        "--along-half-width", str(args.along_half_width),
        "--max-connect-distance", str(args.max_connect_distance),
        "--max-z-gap", str(args.max_z_gap),
        "--minor-offset-limit", str(args.minor_offset_limit),
        "--azimuth-tol", str(args.azimuth_tol),
        "--dip-tol", str(args.dip_tol),
        "--min-along-ratio", str(args.min_along_ratio),
        "--max-bridges", str(args.max_bridges),
        "--max-bridges-per-patch", str(args.max_bridges_per_patch),
        "--bridge-height-scale", str(args.bridge_height_scale),
        "--max-bridge-length", str(args.max_bridge_length),
    ])

    run_command([
        sys.executable,
        str(SECTION_SCRIPT),
        "--vtk", str(bridge_vtk),
        "--output-dir", str(bridge_section_dir),
        "--title-prefix", f"{args.title_prefix} 桥接增强剖面",
        *common_section_args,
    ])

    raw_rows = read_segments_csv(find_segment_csv(raw_section_dir, args.x, args.y))
    bridge_rows = read_segments_csv(find_segment_csv(bridge_section_dir, args.x, args.y))

    raw_counts = plot_line_triplet(
        raw_rows,
        args.output_dir / "01_original_sections_combined.png",
        None if args.no_svg else args.output_dir / "01_original_sections_combined.svg",
        f"{args.title_prefix}：原始剖面",
        args,
    )
    bridge_counts = plot_line_triplet(
        bridge_rows,
        args.output_dir / "02_connected_bridge_sections_combined.png",
        None if args.no_svg else args.output_dir / "02_connected_bridge_sections_combined.svg",
        f"{args.title_prefix}：桥接增强展示剖面",
        args,
    )
    density_counts = plot_density_triplet(
        raw_rows,
        args.output_dir / "03_density_background_sections_combined.png",
        None if args.no_svg else args.output_dir / "03_density_background_sections_combined.svg",
        f"{args.title_prefix}：裂缝密度背景 + 原始线段",
        args,
    )

    summary = {
        "input_vtk": str(args.input_vtk),
        "x": float(args.x),
        "y": float(args.y),
        "half_width": float(args.half_width),
        "along_half_width": float(args.along_half_width),
        "bridge_vtk": str(bridge_vtk),
        "raw_section_dir": str(raw_section_dir),
        "bridge_section_dir": str(bridge_section_dir),
        "original_combined_png": str(args.output_dir / "01_original_sections_combined.png"),
        "connected_combined_png": str(args.output_dir / "02_connected_bridge_sections_combined.png"),
        "density_combined_png": str(args.output_dir / "03_density_background_sections_combined.png"),
        "raw_counts": raw_counts,
        "bridge_counts": bridge_counts,
        "density_counts": density_counts,
        "bridge_params": {
            "max_connect_distance": float(args.max_connect_distance),
            "max_z_gap": float(args.max_z_gap),
            "minor_offset_limit": float(args.minor_offset_limit),
            "azimuth_tol": float(args.azimuth_tol),
            "dip_tol": float(args.dip_tol),
            "min_along_ratio": float(args.min_along_ratio),
            "max_bridges": int(args.max_bridges),
            "max_bridges_per_patch": int(args.max_bridges_per_patch),
        },
    }
    (args.output_dir / "display_triplet_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[triplet] output_dir={args.output_dir}", flush=True)
    print(f"[triplet] original_counts={raw_counts}", flush=True)
    print(f"[triplet] connected_counts={bridge_counts}", flush=True)
    print(f"[triplet] density_counts={density_counts}", flush=True)


if __name__ == "__main__":
    main()
