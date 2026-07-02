# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
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
import segyio
from scipy.spatial import cKDTree

from build_all_area_section_visualization import (
    SURFACE_LINE_STYLES,
    build_surface_section_curves,
    configure_matplotlib_fonts,
    load_surface_lookups,
    select_demo_well,
)


ATTRIBUTE_LABELS = {
    "Coherence": "相干体",
    "AntTrack": "蚂蚁体",
}

ATTRIBUTE_CMAPS = {
    # High-coherence continuous depositional background is shown as white,
    # while low-coherence discontinuities are shown as black to dark gray.
    "Coherence": "gray",
    # Current display convention: -1 = weak development = white, 1 = strong
    # development = black.
    "AntTrack": "gray_r",
}

DEFAULT_ATTRIBUTES = ["Coherence", "AntTrack"]
NULL_THRESHOLD = -1.0e6
WELL_TRAJECTORY_COLOR = "#ffea00"
WELL_TRAJECTORY_GLOW = "#5b4b00"


@dataclass
class AttributeSection:
    attribute: str
    projection: str
    h: np.ndarray
    time: np.ndarray
    values: np.ndarray


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config = read_json(args.config)
    path_keys = {
        "trace_header_csv",
        "surface_dir",
        "output_root",
        "real_well_samples_root",
        "well_trajectory_csv",
    }
    for key in [
        "trace_header_csv",
        "surface_dir",
        "output_root",
        "real_well_samples_root",
        "well_trajectory_csv",
        "well_name",
        "attribute_names",
        "section_tag",
        "title_prefix",
        "z_label",
        "time_padding_ms",
        "axis_sample_count",
        "surface_samples",
        "progress_interval",
        "fig_width",
        "fig_height",
        "dpi",
        "no_svg",
    ]:
        current = getattr(args, key, None)
        if current is None and key in config:
            value = config[key]
            if key in path_keys and value is not None:
                value = Path(str(value))
            setattr(args, key, value)
    args.target_block = config.get("target_block") or {}
    args.exclude_wells = set(str(item) for item in config.get("exclude_wells", ["车页1导眼"]))
    args.volume_paths = {str(key): Path(str(value)) for key, value in (config.get("volume_paths") or {}).items()}
    return args


def set_default_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "surface_dir": Path("/data/shared/project-oil/wx数据/砂砾岩/层位"),
        "output_root": Path("优化阶段二/正式主线/step9_section_visualize/output/attribute_sections"),
        "attribute_names": DEFAULT_ATTRIBUTES,
        "section_tag": "anttrack_coherence_t4_t7",
        "title_prefix": "candidate A demo 过井属性体剖面",
        "z_label": "TWT / ms",
        "time_padding_ms": 20.0,
        "axis_sample_count": 0,
        "surface_samples": 500,
        "progress_interval": 50,
        "fig_width": 15.5,
        "fig_height": 7.8,
        "dpi": 240,
    }
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if getattr(args, "no_svg", None) is None:
        args.no_svg = False
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build AntTrack/Coherence XZ and YZ attribute sections passing a selected well, "
            "restricted to the formal T4-T7 interval."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-header-csv", type=Path, default=None)
    parser.add_argument("--surface-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--real-well-samples-root", type=Path, default=None)
    parser.add_argument("--well-trajectory-csv", type=Path, default=None)
    parser.add_argument("--well-name", type=str, default=None)
    parser.add_argument("--section-tag", type=str, default=None)
    parser.add_argument("--title-prefix", type=str, default=None)
    parser.add_argument("--z-label", type=str, default=None)
    parser.add_argument("--time-padding-ms", type=float, default=None)
    parser.add_argument("--axis-sample-count", type=int, default=None, help="Optional lateral resampling count per projection axis. 0 keeps unique trace coordinates.")
    parser.add_argument("--surface-samples", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--fig-width", type=float, default=None)
    parser.add_argument("--fig-height", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--no-svg", action="store_true", default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    required_paths = {
        "trace_header_csv": args.trace_header_csv,
        "surface_dir": args.surface_dir,
        "output_root": args.output_root,
    }
    if args.well_trajectory_csv is None:
        required_paths["real_well_samples_root"] = args.real_well_samples_root
    for name, path in required_paths.items():
        if path is None:
            raise ValueError(f"{name} is required")
        if not path.exists() and name != "output_root":
            raise FileNotFoundError(f"{name} not found: {path}")
    if not args.volume_paths:
        raise ValueError("volume_paths is required in config")
    missing_attributes = [name for name in args.attribute_names if name not in args.volume_paths]
    if missing_attributes:
        raise ValueError(f"missing volume paths for attributes: {missing_attributes}")
    for attr, path in args.volume_paths.items():
        if attr in args.attribute_names and not path.exists():
            raise FileNotFoundError(f"volume file not found for {attr}: {path}")


def file_safe_label(text: str) -> str:
    return str(text).strip().replace("/", "_").replace(" ", "_")


def build_output_dir(args: argparse.Namespace, well_name: str) -> Path:
    return Path(args.output_root) / file_safe_label(well_name) / file_safe_label(args.section_tag)


def build_trace_grid(trace_header_csv: Path, target_block: dict[str, object]) -> pd.DataFrame:
    df = pd.read_csv(trace_header_csv, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    for column in ["TraceIdx", "X", "Y"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["TraceIdx", "X", "Y"]).copy()
    if target_block:
        df = df[
            df["X"].between(float(target_block["x_min"]), float(target_block["x_max"]), inclusive="both")
            & df["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]), inclusive="both")
        ].copy()
    if df.empty:
        raise ValueError("no trace headers remain after target-block filtering")
    return df.sort_values(["X", "Y", "TraceIdx"]).drop_duplicates("TraceIdx").reset_index(drop=True)


def build_display_bounds(trace_df: pd.DataFrame, target_block: dict[str, object]) -> dict[str, float | str]:
    scope = "target_block_trace_extent" if target_block else "mine_trace_extent"
    return {
        "display_scope": scope,
        "display_x_min": float(trace_df["X"].min()),
        "display_x_max": float(trace_df["X"].max()),
        "display_y_min": float(trace_df["Y"].min()),
        "display_y_max": float(trace_df["Y"].max()),
    }


def build_axis_values(trace_df: pd.DataFrame, column: str, axis_sample_count: int) -> np.ndarray:
    values = np.sort(trace_df[column].unique()).astype(np.float64)
    if axis_sample_count > 1 and len(values) > axis_sample_count:
        return np.linspace(float(values[0]), float(values[-1]), int(axis_sample_count), dtype=np.float64)
    return values


def build_trace_tree(trace_df: pd.DataFrame) -> tuple[cKDTree, np.ndarray]:
    xy = trace_df[["X", "Y"]].to_numpy(dtype=np.float64)
    trace_ids = trace_df["TraceIdx"].to_numpy(dtype=np.int64)
    return cKDTree(xy), trace_ids


def open_volume_context(volume_path: Path):
    handle = segyio.open(str(volume_path), "r", ignore_geometry=True)
    handle.mmap()
    samples = np.asarray(handle.samples, dtype=np.float64)
    trace_cache: dict[int, np.ndarray] = {}

    def trace_at(trace_idx: int) -> np.ndarray:
        key = int(trace_idx)
        if key not in trace_cache:
            arr = np.asarray(handle.trace[key], dtype=np.float32)
            arr = arr.astype(np.float64, copy=False)
            arr[arr <= NULL_THRESHOLD] = np.nan
            trace_cache[key] = arr
        return trace_cache[key]

    return handle, samples, trace_at


def sample_trace_at_time(trace_data: np.ndarray, samples: np.ndarray, time_ms: float) -> float:
    if not np.isfinite(time_ms):
        return np.nan
    frac = np.interp(time_ms, samples, np.arange(len(samples), dtype=np.float64), left=np.nan, right=np.nan)
    if not np.isfinite(frac):
        return np.nan
    i0 = int(np.floor(frac))
    if i0 < 0 or i0 >= len(trace_data):
        return np.nan
    if i0 + 1 >= len(trace_data):
        return float(trace_data[i0]) if np.isfinite(trace_data[i0]) else np.nan
    v0 = trace_data[i0]
    v1 = trace_data[i0 + 1]
    if not np.isfinite(v0) and not np.isfinite(v1):
        return np.nan
    if not np.isfinite(v0):
        return float(v1)
    if not np.isfinite(v1):
        return float(v0)
    weight = float(frac - i0)
    return float((1.0 - weight) * v0 + weight * v1)


def select_time_samples(samples: np.ndarray, time_min: float, time_max: float) -> np.ndarray:
    mask = (samples >= time_min) & (samples <= time_max)
    selected = samples[mask]
    if selected.size >= 2:
        return selected.astype(np.float64, copy=True)
    return np.linspace(time_min, time_max, num=200, dtype=np.float64)


def sample_attribute_section(
    projection: str,
    h_values: np.ndarray,
    time_values: np.ndarray,
    well_df: pd.DataFrame,
    trace_tree: cKDTree,
    trace_ids: np.ndarray,
    samples: np.ndarray,
    trace_at,
    progress_interval: int,
) -> np.ndarray:
    well_time = well_df["TIME"].to_numpy(dtype=np.float64)
    well_x = well_df["X"].to_numpy(dtype=np.float64)
    well_y = well_df["Y"].to_numpy(dtype=np.float64)
    values = np.full((len(time_values), len(h_values)), np.nan, dtype=np.float64)
    for row_idx, time_ms in enumerate(time_values):
        if projection == "XZ":
            curve_coord = float(np.interp(time_ms, well_time, well_y))
            points = np.column_stack([h_values, np.full(len(h_values), curve_coord, dtype=np.float64)])
        else:
            curve_coord = float(np.interp(time_ms, well_time, well_x))
            points = np.column_stack([np.full(len(h_values), curve_coord, dtype=np.float64), h_values])
        nearest_idx = np.asarray(trace_tree.query(points, k=1)[1], dtype=np.int64).reshape(-1)
        row = np.full(len(h_values), np.nan, dtype=np.float64)
        for tree_pos in np.unique(nearest_idx):
            trace_idx = int(trace_ids[int(tree_pos)])
            trace_value = sample_trace_at_time(trace_at(trace_idx), samples, time_ms)
            row[nearest_idx == tree_pos] = trace_value
        values[row_idx, :] = row
        if progress_interval > 0 and (row_idx + 1) % progress_interval == 0:
            print(f"[attribute-section] projection={projection} sampled {row_idx + 1}/{len(time_values)} time rows", flush=True)
    return values


def axis_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], mid, [last]]).astype(np.float64, copy=False)


def finite_quantile_bounds(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if np.isfinite(arr).any()])
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(finite, 0.02))
    high = float(np.quantile(finite, 0.98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def finite_values(arrays: list[np.ndarray]) -> np.ndarray:
    parts = [arr[np.isfinite(arr)] for arr in arrays if np.isfinite(arr).any()]
    if not parts:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(parts).astype(np.float64, copy=False)


def attribute_display_settings(attribute: str, arrays: list[np.ndarray]) -> tuple[float, float, float | None, object | None]:
    if attribute == "AntTrack":
        return -1.0, 1.0, 0.0, None
    if attribute == "Coherence":
        low, high = finite_quantile_bounds(arrays)
        return low, high, 0.5 * (low + high), None
    low, high = finite_quantile_bounds(arrays)
    return low, high, 0.5 * (low + high), None


def draw_surface_curves(ax, curves, projection: str) -> None:
    for curve in curves:
        if curve.projection != projection:
            continue
        finite = np.isfinite(curve.h) & np.isfinite(curve.z)
        if int(finite.sum()) < 2:
            continue
        color, linestyle = SURFACE_LINE_STYLES.get(curve.surface, ("#334155", "--"))
        ax.plot(curve.h[finite], curve.z[finite], color=color, linestyle=linestyle, linewidth=1.55, alpha=0.9, zorder=3)


def draw_well_trajectory(ax, h_values: pd.Series, time_values: pd.Series, label: str | None = None) -> None:
    # A dark underlay keeps the bright-yellow well track visible on both dark and light backgrounds.
    ax.plot(h_values, time_values, color=WELL_TRAJECTORY_GLOW, linewidth=3.6, zorder=4)
    ax.plot(h_values, time_values, color=WELL_TRAJECTORY_COLOR, linewidth=2.2, zorder=5, label=label)


def plot_attribute_projection(
    section: AttributeSection,
    well_df: pd.DataFrame,
    surface_curves,
    summary: dict[str, object],
    output_stem: Path,
    args: argparse.Namespace,
    vmin: float,
    vmax: float,
    norm,
) -> None:
    label = ATTRIBUTE_LABELS.get(section.attribute, section.attribute)
    cmap = ATTRIBUTE_CMAPS.get(section.attribute, "viridis")
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    mesh_kwargs = {"shading": "auto", "cmap": cmap, "zorder": 1}
    if norm is None:
        mesh_kwargs.update({"vmin": vmin, "vmax": vmax})
    else:
        mesh_kwargs["norm"] = norm
    mesh = ax.pcolormesh(axis_edges(section.h), axis_edges(section.time), section.values, **mesh_kwargs)
    draw_surface_curves(ax, surface_curves, section.projection)
    if section.projection == "XZ":
        draw_well_trajectory(ax, well_df["X"], well_df["TIME"], label=f"{args.selected_well_name}井轨迹")
        ax.set_xlabel("X / m")
        ax.set_xlim(float(summary["display_x_min"]), float(summary["display_x_max"]))
    else:
        draw_well_trajectory(ax, well_df["Y"], well_df["TIME"], label=f"{args.selected_well_name}井轨迹")
        ax.set_xlabel("Y / m")
        ax.set_xlim(float(summary["display_y_min"]), float(summary["display_y_max"]))
    ax.set_ylim(float(summary["display_time_min"]), float(summary["display_time_max"]))
    ax.invert_yaxis()
    ax.set_ylabel(args.z_label)
    ax.grid(True, linewidth=0.3, alpha=0.28)
    fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.95, label=label)
    ax.set_title(f"{args.title_prefix} | {args.selected_well_name} | {label} | {section.projection} | T4-T7")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=args.dpi)
    if not args.no_svg:
        fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_attribute_combined(
    sections: list[AttributeSection],
    well_df: pd.DataFrame,
    surface_curves,
    summary: dict[str, object],
    output_stem: Path,
    args: argparse.Namespace,
    vmin: float,
    vmax: float,
    norm,
) -> None:
    first = sections[0]
    label = ATTRIBUTE_LABELS.get(first.attribute, first.attribute)
    cmap = ATTRIBUTE_CMAPS.get(first.attribute, "viridis")
    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width * 1.4, args.fig_height), sharey=True)
    colorbar_mappable = None
    for ax, projection in zip(axes, ["XZ", "YZ"]):
        section = next(item for item in sections if item.projection == projection)
        mesh_kwargs = {"shading": "auto", "cmap": cmap, "zorder": 1}
        if norm is None:
            mesh_kwargs.update({"vmin": vmin, "vmax": vmax})
        else:
            mesh_kwargs["norm"] = norm
        mesh = ax.pcolormesh(axis_edges(section.h), axis_edges(section.time), section.values, **mesh_kwargs)
        colorbar_mappable = mesh
        draw_surface_curves(ax, surface_curves, projection)
        if projection == "XZ":
            draw_well_trajectory(ax, well_df["X"], well_df["TIME"])
            ax.set_xlim(float(summary["display_x_min"]), float(summary["display_x_max"]))
            ax.set_xlabel("X / m")
        else:
            draw_well_trajectory(ax, well_df["Y"], well_df["TIME"])
            ax.set_xlim(float(summary["display_y_min"]), float(summary["display_y_max"]))
            ax.set_xlabel("Y / m")
        ax.set_ylim(float(summary["display_time_min"]), float(summary["display_time_max"]))
        ax.invert_yaxis()
        ax.grid(True, linewidth=0.3, alpha=0.28)
        ax.set_title(f"{projection}")
    axes[0].set_ylabel(args.z_label)
    fig.suptitle(f"{args.title_prefix} | {args.selected_well_name} | {label} | T4-T7")
    if colorbar_mappable is not None:
        fig.colorbar(colorbar_mappable, ax=axes, pad=0.02, shrink=0.92, label=label)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_stem.with_suffix(".png"), dpi=args.dpi)
    if not args.no_svg:
        fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def write_section_csv(path: Path, section: AttributeSection) -> None:
    hh, tt = np.meshgrid(section.h, section.time)
    out_df = pd.DataFrame(
        {
            "attribute": section.attribute,
            "projection": section.projection,
            "h": hh.reshape(-1),
            "time": tt.reshape(-1),
            "value": section.values.reshape(-1),
        }
    )
    out_df.to_csv(path, index=False, encoding="utf-8-sig")


def write_well_projection_csv(path: Path, well_df: pd.DataFrame) -> None:
    well_df.to_csv(path, index=False, encoding="utf-8-sig")


def write_surface_curves_csv(path: Path, curves) -> None:
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


def main() -> None:
    args = apply_config(build_parser().parse_args())
    args = set_default_args(args)
    validate_args(args)
    configure_matplotlib_fonts()

    selected_well = select_demo_well(args)
    args.well_trajectory_csv = selected_well.path
    args.selected_well_name = selected_well.well_name
    well_df = selected_well.trajectory.copy()

    output_dir = build_output_dir(args, selected_well.well_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_df = build_trace_grid(args.trace_header_csv, args.target_block)
    trace_tree, trace_ids = build_trace_tree(trace_df)
    surfaces = load_surface_lookups(args.surface_dir)
    summary: dict[str, object] = {
        "selected_well_name": selected_well.well_name,
        "selected_well_csv": str(selected_well.path),
        "selected_well_inside_target_rows": int(selected_well.inside_rows),
        "selected_well_total_rows": int(selected_well.total_rows),
        "selected_well_inside_target_ratio": float(selected_well.inside_ratio),
        "output_dir": str(output_dir),
        "trace_header_csv": str(args.trace_header_csv),
        "surface_dir": str(args.surface_dir),
        "volume_paths": {attr: str(args.volume_paths[attr]) for attr in args.attribute_names},
        "target_block": args.target_block,
        "attribute_names": list(args.attribute_names),
        "excluded_wells": sorted(args.exclude_wells),
        "section_tag": str(args.section_tag),
        "trace_count_in_block": int(len(trace_df)),
        "trace_unique_x": int(trace_df["X"].nunique()),
        "trace_unique_y": int(trace_df["Y"].nunique()),
        "section_mode": "well_curve_attribute_resampling_t4_t7",
        "xy_projection_logic": {
            "XZ": "sample attribute at (x, y_well(time), time)",
            "YZ": "sample attribute at (x_well(time), y, time)",
        },
    }
    summary.update(build_display_bounds(trace_df, args.target_block))

    surface_curves = build_surface_section_curves(surfaces, well_df, summary, args)
    curve_times = np.concatenate([curve.z[np.isfinite(curve.z)] for curve in surface_curves if np.isfinite(curve.z).any()])
    time_min = float(np.nanmin(curve_times)) - float(args.time_padding_ms)
    time_max = float(np.nanmax(curve_times)) + float(args.time_padding_ms)
    well_time_min = float(well_df["TIME"].min())
    well_time_max = float(well_df["TIME"].max())
    summary.update(
        {
            "display_time_min": float(time_min),
            "display_time_max": float(time_max),
            "well_time_min": well_time_min,
            "well_time_max": well_time_max,
        }
    )

    x_values = build_axis_values(trace_df, "X", int(args.axis_sample_count))
    y_values = build_axis_values(trace_df, "Y", int(args.axis_sample_count))
    summary.update(
        {
            "axis_sample_count_config": int(args.axis_sample_count),
            "section_x_sample_count": int(len(x_values)),
            "section_y_sample_count": int(len(y_values)),
        }
    )
    all_sections: list[AttributeSection] = []
    attribute_summary: dict[str, object] = {}

    for attribute in args.attribute_names:
        print(f"[attribute-section] sampling attribute={attribute}", flush=True)
        handle, samples, trace_at = open_volume_context(args.volume_paths[attribute])
        try:
            attr_time_min = max(time_min, float(np.nanmin(samples)))
            attr_time_max = min(time_max, float(np.nanmax(samples)))
            time_values = select_time_samples(samples, attr_time_min, attr_time_max)
            xz_values = sample_attribute_section(
                projection="XZ",
                h_values=x_values,
                time_values=time_values,
                well_df=well_df,
                trace_tree=trace_tree,
                trace_ids=trace_ids,
                samples=samples,
                trace_at=trace_at,
                progress_interval=int(args.progress_interval),
            )
            yz_values = sample_attribute_section(
                projection="YZ",
                h_values=y_values,
                time_values=time_values,
                well_df=well_df,
                trace_tree=trace_tree,
                trace_ids=trace_ids,
                samples=samples,
                trace_at=trace_at,
                progress_interval=int(args.progress_interval),
            )
        finally:
            handle.close()

        xz_section = AttributeSection(attribute=attribute, projection="XZ", h=x_values.copy(), time=time_values.copy(), values=xz_values)
        yz_section = AttributeSection(attribute=attribute, projection="YZ", h=y_values.copy(), time=time_values.copy(), values=yz_values)
        all_sections.extend([xz_section, yz_section])
        vmin, vmax, vcenter, norm = attribute_display_settings(attribute, [xz_values, yz_values])

        attr_label = file_safe_label(attribute.lower())
        write_section_csv(output_dir / f"{attr_label}_section_xz_t4_t7_samples.csv", xz_section)
        write_section_csv(output_dir / f"{attr_label}_section_yz_t4_t7_samples.csv", yz_section)
        plot_attribute_projection(
            xz_section,
            well_df,
            surface_curves,
            summary,
            output_dir / f"{attr_label}_section_xz_t4_t7",
            args,
            vmin,
            vmax,
            norm,
        )
        plot_attribute_projection(
            yz_section,
            well_df,
            surface_curves,
            summary,
            output_dir / f"{attr_label}_section_yz_t4_t7",
            args,
            vmin,
            vmax,
            norm,
        )
        plot_attribute_combined(
            [xz_section, yz_section],
            well_df,
            surface_curves,
            summary,
            output_dir / f"{attr_label}_section_xz_yz_t4_t7_combined",
            args,
            vmin,
            vmax,
            norm,
        )
        attribute_summary[attribute] = {
            "display_value_min": float(vmin),
            "display_value_center": float(vcenter) if vcenter is not None else None,
            "display_value_max": float(vmax),
            "display_colormap": "grayscale low-black high-white percentile stretch" if attribute == "Coherence" else "reversed grayscale fixed -1..1",
            "time_sample_count": int(len(time_values)),
            "xz_shape": [int(xz_values.shape[0]), int(xz_values.shape[1])],
            "yz_shape": [int(yz_values.shape[0]), int(yz_values.shape[1])],
            "xz_finite_count": int(np.isfinite(xz_values).sum()),
            "yz_finite_count": int(np.isfinite(yz_values).sum()),
            "xz_nan_count": int(np.isnan(xz_values).sum()),
            "yz_nan_count": int(np.isnan(yz_values).sum()),
        }

    write_well_projection_csv(output_dir / "well_trajectory_projected.csv", well_df)
    write_surface_curves_csv(output_dir / "attribute_section_surface_curves.csv", surface_curves)
    summary.update(
        {
            "attribute_summary": attribute_summary,
            "surface_codes_drawn": sorted({curve.surface for curve in surface_curves}),
            "checks": {
                "selected_well_is_che58": selected_well.well_name == "车58",
                "has_xz_and_yz_for_each_attribute": all(
                    attribute in attribute_summary and attribute_summary[attribute]["xz_finite_count"] > 0 and attribute_summary[attribute]["yz_finite_count"] > 0
                    for attribute in args.attribute_names
                ),
                "surface_curves_include_t4_t5_t6_t7": sorted({curve.surface for curve in surface_curves}) == ["T4", "T5", "T6", "T7"],
                "output_dir_isolated_from_other_section_outputs": str(output_dir).find("attribute_sections") >= 0,
            },
        }
    )
    summary["status"] = "pass" if all(bool(value) for value in summary["checks"].values()) else "fail"
    (output_dir / "attribute_section_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[attribute-section] output_dir={output_dir}", flush=True)
    print(f"[attribute-section] selected_well={selected_well.well_name}", flush=True)
    print(f"[attribute-section] status={summary['status']}", flush=True)


if __name__ == "__main__":
    main()
