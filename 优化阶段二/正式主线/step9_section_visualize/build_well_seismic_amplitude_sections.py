# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gan_dfn_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from build_well_attribute_section_visualization import axis_edges
from well_curved_section_common import (
    AttributeSection,
    draw_surface_curves,
    draw_well_trajectory,
    prepare_geometry,
    read_json,
    sample_volume_sections,
    save_section_pair_npz,
    write_json,
)


SIGNED_RWB_STYLE = "signed_red_white_blue"
ABSOLUTE_GRAYSCALE_STYLE = "absolute_grayscale"


def resolve_amplitude_display_style(display_style: str) -> dict[str, object]:
    if display_style == SIGNED_RWB_STYLE:
        return {"cmap": "seismic", "absolute": False, "style": SIGNED_RWB_STYLE}
    if display_style == ABSOLUTE_GRAYSCALE_STYLE:
        return {"cmap": "gray_r", "absolute": True, "style": ABSOLUTE_GRAYSCALE_STYLE}
    raise ValueError(f"unsupported display style: {display_style}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build seismic variable-density and wiggle/variable-area well-curved sections.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def amplitude_limit(sections: list[AttributeSection], quantile: float) -> float:
    finite = np.concatenate([section.values[np.isfinite(section.values)] for section in sections if np.isfinite(section.values).any()])
    limit = float(np.quantile(np.abs(finite), quantile)) if finite.size else 1.0
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def figure_size_inches(section: AttributeSection, geometry, panel: bool = False) -> tuple[float, float, int]:
    dpi = int(geometry.config.get("render_dpi", geometry.args.dpi))
    if panel:
        width_px = int(geometry.config.get("panel_width_px", 3000))
    elif section.projection == "XZ":
        width_px = int(geometry.config.get("xz_total_width_px", round(geometry.args.fig_width * dpi)))
    else:
        width_px = int(geometry.config.get("yz_total_width_px", round(geometry.args.fig_width * dpi)))
    height_px = int(geometry.config.get("figure_height_px", round(geometry.args.fig_height * dpi)))
    return width_px / dpi, height_px / dpi, dpi


def save_figure(fig, output_path: Path, dpi: int, write_svg: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if write_svg:
        fig.savefig(output_path.with_suffix(".svg"))


def apply_axes(ax, section: AttributeSection, geometry, *, show_legend: bool = True) -> None:
    draw_surface_curves(ax, geometry.surface_curves, section.projection)
    if section.projection == "XZ":
        draw_well_trajectory(ax, geometry.well_df["X"], geometry.well_df["TIME"], label=f"{geometry.well_name}井轨迹")
        ax.set_xlabel("X / m")
    else:
        draw_well_trajectory(ax, geometry.well_df["Y"], geometry.well_df["TIME"], label=f"{geometry.well_name}井轨迹")
        ax.set_xlabel("Y / m")
    ax.set_xlim(float(section.h[0]), float(section.h[-1]))
    ax.set_ylim(float(geometry.summary["display_time_min"]), float(geometry.summary["display_time_max"]))
    ax.invert_yaxis()
    ax.set_ylabel(geometry.args.z_label)
    ax.grid(True, linewidth=0.25, alpha=0.22)
    if show_legend:
        ax.legend(loc="upper right")


def plot_variable_density(
    section: AttributeSection,
    geometry,
    output_path: Path,
    limit: float,
    title_suffix: str = "",
    panel: bool = False,
    *,
    display_style: str = SIGNED_RWB_STYLE,
    overlay_drawer: Callable[[Any, str], dict[str, int]] | None = None,
    scope_label: str | None = None,
    product_title: str | None = None,
) -> dict[str, int]:
    width, height, dpi = figure_size_inches(section, geometry, panel=panel)
    if overlay_drawer is not None:
        width += 4.0
    fig, ax = plt.subplots(figsize=(width, height))
    style = resolve_amplitude_display_style(display_style)
    norm = Normalize(vmin=0.0, vmax=limit) if bool(style["absolute"]) else TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    display_values = np.abs(section.values) if bool(style["absolute"]) else section.values
    mesh = ax.pcolormesh(
        axis_edges(section.h),
        axis_edges(section.time),
        display_values,
        shading="auto",
        cmap=style["cmap"],
        norm=norm,
        rasterized=True,
        zorder=1,
    )
    apply_axes(ax, section, geometry, show_legend=overlay_drawer is None)
    overlay_stats: dict[str, int] = {}
    if overlay_drawer is not None:
        overlay_stats = overlay_drawer(ax, section.projection)
        fig.subplots_adjust(left=0.06, right=0.76, bottom=0.10, top=0.88)
        cax = fig.add_axes([0.78, 0.16, 0.014, 0.66])
        fig.colorbar(mesh, cax=cax, label="地震振幅绝对值" if bool(style["absolute"]) else "地震振幅")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.81, 0.88), fontsize=8.4, framealpha=0.92)
        fig.suptitle(product_title or "车页1导眼：DFN与成像测井裂缝对比剖面", y=0.975, fontsize=13)
        ax.set_title(f"{scope_label or '剖面'} | 地震振幅变密度 | {section.projection} | T4-T7{title_suffix}", fontsize=11, pad=8)
    else:
        fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.94, label="地震振幅绝对值" if bool(style["absolute"]) else "地震振幅")
        ax.set_title(f"{geometry.config['title_prefix']} | 地震振幅变密度 | {section.projection} | T4-T7{title_suffix}")
        fig.tight_layout()
    save_figure(fig, output_path, dpi, bool(geometry.config.get("write_svg", False)))
    plt.close(fig)
    return overlay_stats


def plot_wiggle_variable_area(
    section: AttributeSection,
    geometry,
    output_path: Path,
    limit: float,
    title_suffix: str = "",
    panel: bool = False,
    *,
    overlay_drawer: Callable[[Any, str], dict[str, int]] | None = None,
    scope_label: str | None = None,
    product_title: str | None = None,
) -> int:
    width, height, dpi = figure_size_inches(section, geometry, panel=panel)
    if overlay_drawer is not None:
        width += 3.6
    fig, ax = plt.subplots(figsize=(width, height))
    max_traces = int(geometry.config.get("wiggle_max_trace_count", 0))
    if max_traces <= 0 or max_traces >= len(section.h):
        selected = np.arange(len(section.h), dtype=int)
    else:
        selected = np.linspace(0, len(section.h) - 1, max_traces, dtype=int)
        selected = np.unique(selected)
    selected_h = section.h[selected]
    spacing = float(np.median(np.diff(selected_h))) if len(selected_h) > 1 else 1.0
    swing = float(geometry.config.get("wiggle_lateral_scale_fraction", 0.42))
    lateral_scale = swing * spacing / limit
    for idx in selected:
        trace = np.nan_to_num(section.values[:, idx], nan=0.0, posinf=0.0, neginf=0.0)
        trace = np.clip(trace, -limit, limit)
        baseline = float(section.h[idx])
        displaced = baseline + trace * lateral_scale
        ax.plot(displaced, section.time, color="#111827", linewidth=0.42, alpha=0.88, zorder=1)
        ax.fill_betweenx(
            section.time,
            baseline,
            displaced,
            where=trace >= 0.0,
            facecolor="#111827",
            alpha=0.72,
            linewidth=0.0,
            zorder=1,
        )
    apply_axes(ax, section, geometry, show_legend=overlay_drawer is None)
    ax.set_facecolor("white")
    if overlay_drawer is not None:
        overlay_drawer(ax, section.projection)
        fig.subplots_adjust(left=0.06, right=0.79, bottom=0.10, top=0.88)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.81, 0.88), fontsize=8.4, framealpha=0.92)
        fig.suptitle(product_title or "车页1导眼：DFN与成像测井裂缝对比剖面", y=0.975, fontsize=13)
        ax.set_title(
            f"{scope_label or '剖面'} | 地震波形+变面积 | {section.projection} | "
            f"T4-T7 | 道数={len(selected)}{title_suffix}",
            fontsize=11,
            pad=8,
        )
    else:
        ax.set_title(
            f"{geometry.config['title_prefix']} | 地震波形+变面积 | {section.projection} | "
            f"T4-T7 | 显示道数={len(selected)} | 摆幅={swing:.2f}{title_suffix}"
        )
        fig.tight_layout()
    save_figure(fig, output_path, dpi, bool(geometry.config.get("write_svg", False)))
    plt.close(fig)
    return int(len(selected))


def split_section(section: AttributeSection, panel_count: int) -> list[AttributeSection]:
    if panel_count <= 1:
        return [section]
    index_groups = [group for group in np.array_split(np.arange(len(section.h)), panel_count) if len(group) >= 2]
    return [
        AttributeSection(
            section.attribute,
            section.projection,
            section.h[group].copy(),
            section.time.copy(),
            section.values[:, group].copy(),
        )
        for group in index_groups
    ]


def plot_section_panels(
    section: AttributeSection,
    geometry,
    output_dir: Path,
    limit: float,
    display_style: str = SIGNED_RWB_STYLE,
) -> dict[str, object]:
    panel_count = int(
        geometry.config.get("xz_panel_count" if section.projection == "XZ" else "yz_panel_count", 0)
    )
    if panel_count <= 1:
        return {"panel_count": 0, "density_pngs": [], "wiggle_pngs": [], "wiggle_trace_counts": []}
    density_dir = output_dir / "variable_density_panels"
    wiggle_dir = output_dir / "wiggle_variable_area_panels"
    density_pngs: list[str] = []
    wiggle_pngs: list[str] = []
    wiggle_counts: list[int] = []
    panels = split_section(section, panel_count)
    for panel_index, panel_section in enumerate(panels, start=1):
        suffix = f" | 分段{panel_index}/{len(panels)}"
        density_name = f"seisamp_variable_density_{section.projection.lower()}_panel_{panel_index:02d}_t4_t7.png"
        wiggle_name = f"seisamp_wiggle_variable_area_{section.projection.lower()}_panel_{panel_index:02d}_t4_t7.png"
        plot_variable_density(panel_section, geometry, density_dir / density_name, limit, suffix, panel=True, display_style=display_style)
        count = plot_wiggle_variable_area(panel_section, geometry, wiggle_dir / wiggle_name, limit, suffix, panel=True)
        density_pngs.append(str(Path("variable_density_panels") / density_name))
        wiggle_pngs.append(str(Path("wiggle_variable_area_panels") / wiggle_name))
        wiggle_counts.append(count)
    return {
        "panel_count": len(panels),
        "density_pngs": density_pngs,
        "wiggle_pngs": wiggle_pngs,
        "wiggle_trace_counts": wiggle_counts,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    geometry = prepare_geometry(config)
    output_dir = Path(str(config["output_root"])).resolve() / "seismic_amplitude_sections"
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_style = str(config.get("formal_signed_attribute_display_style", ABSOLUTE_GRAYSCALE_STYLE))
    backup_style = str(config.get("signed_attribute_backup_display_style", SIGNED_RWB_STYLE))
    backup_dir = output_dir / str(config.get("signed_attribute_backup_dir", "red_white_blue_backup"))
    seis_path = Path(str(config["volume_paths"]["SeisAmp"])).resolve()
    print("[seismic-section] sampling SeisAmp", flush=True)
    xz, yz, stats = sample_volume_sections(geometry, "SeisAmp", seis_path)
    sections = [xz, yz]
    quantile = float(config.get("amplitude_clip_quantile", 0.99))
    limit = amplitude_limit(sections, quantile)
    image_names = {
        ("density", "XZ"): "05_seisamp_variable_density_xz_t4_t7.png",
        ("density", "YZ"): "06_seisamp_variable_density_yz_t4_t7.png",
        ("wiggle", "XZ"): "07_seisamp_wiggle_variable_area_xz_t4_t7.png",
        ("wiggle", "YZ"): "08_seisamp_wiggle_variable_area_yz_t4_t7.png",
    }
    plot_variable_density(xz, geometry, output_dir / image_names[("density", "XZ")], limit, display_style=primary_style)
    plot_variable_density(yz, geometry, output_dir / image_names[("density", "YZ")], limit, display_style=primary_style)
    plot_variable_density(xz, geometry, backup_dir / image_names[("density", "XZ")], limit, display_style=backup_style)
    plot_variable_density(yz, geometry, backup_dir / image_names[("density", "YZ")], limit, display_style=backup_style)
    wiggle_xz = plot_wiggle_variable_area(xz, geometry, output_dir / image_names[("wiggle", "XZ")], limit)
    wiggle_yz = plot_wiggle_variable_area(yz, geometry, output_dir / image_names[("wiggle", "YZ")], limit)
    xz_panels = plot_section_panels(xz, geometry, output_dir, limit, primary_style)
    yz_panels = plot_section_panels(yz, geometry, output_dir, limit, primary_style)
    save_section_pair_npz(output_dir / "seisamp_section_samples.npz", xz, yz)
    images = sorted(path.name for path in output_dir.glob("*.png"))
    checks = {
        "four_pngs_generated": len(images) == 4,
        "finite_section_values": float(stats["finite_fraction"]) > 0,
        "symmetric_nonzero_display_limit": limit > 0,
        "wiggle_traces_drawn": wiggle_xz > 0 and wiggle_yz > 0,
        "signed_rwb_variable_density_backup_complete": all(
            (backup_dir / image_names[("density", projection)]).exists()
            for projection in ("XZ", "YZ")
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "section_geometry": geometry.summary,
        "seismic_amplitude": stats,
        "display": {
            "clip_quantile": quantile,
            "symmetric_limit": limit,
            "formal_variable_density_style": primary_style,
            "signed_rwb_backup_style": backup_style,
            "signed_rwb_backup_dir": str(backup_dir),
            "absolute_grayscale_rule": "zero_white_positive_and_negative_extremes_black_abs_q99",
            "wiggle_normalization": "global_section_limit_no_per_trace_normalization",
            "wiggle_trace_selection": "all_section_trace_coordinates_when_configured_max_count_is_zero",
            "wiggle_positive_fill": "black",
            "wiggle_xz_trace_count": wiggle_xz,
            "wiggle_yz_trace_count": wiggle_yz,
            "wiggle_lateral_scale_fraction": float(config.get("wiggle_lateral_scale_fraction", 0.42)),
            "write_svg": bool(config.get("write_svg", False)),
            "xz_total_width_px": int(config.get("xz_total_width_px", 0)),
            "yz_total_width_px": int(config.get("yz_total_width_px", 0)),
            "panel_width_px": int(config.get("panel_width_px", 0)),
        },
        "panels": {"XZ": xz_panels, "YZ": yz_panels},
        "png_files": images,
        "checks": checks,
    }
    write_json(output_dir / "seismic_amplitude_section_summary.json", summary)
    print(f"[seismic-section] output={output_dir}", flush=True)
    print(f"[seismic-section] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
