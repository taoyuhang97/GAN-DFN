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
from matplotlib.colors import TwoSlopeNorm

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


ATTRIBUTE_SETTINGS = {
    "AntTrack": {"label": "蚂蚁体", "cmap": "gray_r"},
    "Coherence": {"label": "相干体", "cmap": "gray"},
    "CurvatureMax": {"label": "最大曲率体", "cmap": "seismic"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build AntTrack and CurvatureMax well-curved sections.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def display_scale(attribute: str, values: list[np.ndarray]) -> tuple[float, float, object | None, dict[str, object]]:
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in values if np.isfinite(arr).any()])
    if attribute == "AntTrack":
        return -1.0, 1.0, None, {"rule": "fixed_-1_to_1_low_white_high_black"}
    if attribute == "Coherence":
        low = float(np.quantile(finite, 0.02))
        high = float(np.quantile(finite, 0.98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            high = low + 1.0
        return low, high, None, {"rule": "q02_to_q98_low_black_high_white", "low": low, "high": high}
    limit = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return -limit, limit, norm, {"rule": "signed_zero_centered_symmetric_abs_q98", "limit": limit}


def plot_section(
    section: AttributeSection,
    geometry,
    output_path: Path,
    vmin: float,
    vmax: float,
    norm,
    *,
    overlay_drawer: Callable[[Any, str], dict[str, int]] | None = None,
    scope_label: str | None = None,
    product_title: str | None = None,
) -> dict[str, int]:
    setting = ATTRIBUTE_SETTINGS[section.attribute]
    overlay_enabled = overlay_drawer is not None
    figure_width = float(geometry.args.fig_width) + (4.0 if overlay_enabled else 0.0)
    fig, ax = plt.subplots(figsize=(figure_width, geometry.args.fig_height))
    kwargs = {"shading": "auto", "cmap": setting["cmap"], "zorder": 1}
    if norm is None:
        kwargs.update({"vmin": vmin, "vmax": vmax})
    else:
        kwargs["norm"] = norm
    mesh = ax.pcolormesh(axis_edges(section.h), axis_edges(section.time), section.values, **kwargs)
    draw_surface_curves(ax, geometry.surface_curves, section.projection)
    if section.projection == "XZ":
        draw_well_trajectory(ax, geometry.well_df["X"], geometry.well_df["TIME"], label=f"{geometry.well_name}井轨迹")
        ax.set_xlim(float(geometry.summary["display_x_min"]), float(geometry.summary["display_x_max"]))
        ax.set_xlabel("X / m")
    else:
        draw_well_trajectory(ax, geometry.well_df["Y"], geometry.well_df["TIME"], label=f"{geometry.well_name}井轨迹")
        ax.set_xlim(float(geometry.summary["display_y_min"]), float(geometry.summary["display_y_max"]))
        ax.set_xlabel("Y / m")
    ax.set_ylim(float(geometry.summary["display_time_min"]), float(geometry.summary["display_time_max"]))
    ax.invert_yaxis()
    ax.set_ylabel(geometry.args.z_label)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    overlay_stats: dict[str, int] = {}
    if overlay_drawer is not None:
        overlay_stats = overlay_drawer(ax, section.projection)
        fig.subplots_adjust(left=0.06, right=0.76, bottom=0.10, top=0.88)
        cax = fig.add_axes([0.78, 0.16, 0.014, 0.66])
        fig.colorbar(mesh, cax=cax, label=setting["label"])
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.81, 0.88), fontsize=8.4, framealpha=0.92)
        title = product_title or "车页1导眼：DFN与成像测井裂缝对比剖面"
        scope = scope_label or "剖面"
        fig.suptitle(title, y=0.975, fontsize=13)
        ax.set_title(f"{scope} | {setting['label']} | {section.projection} | T4-T7", fontsize=11, pad=8)
    else:
        ax.legend(loc="upper right")
        fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.94, label=setting["label"])
        ax.set_title(f"{geometry.config['title_prefix']} | {setting['label']} | {section.projection} | T4-T7")
        fig.tight_layout()
    fig.savefig(output_path, dpi=geometry.args.dpi)
    plt.close(fig)
    return overlay_stats


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    geometry = prepare_geometry(config)
    output_dir = Path(str(config["output_root"])).resolve() / "attribute_sections"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: Path(str(path)).resolve() for name, path in dict(config["volume_paths"]).items()}
    image_names = {
        ("AntTrack", "XZ"): "01_anttrack_section_xz_t4_t7.png",
        ("AntTrack", "YZ"): "02_anttrack_section_yz_t4_t7.png",
        ("CurvatureMax", "XZ"): "03_curvaturemax_section_xz_t4_t7.png",
        ("CurvatureMax", "YZ"): "04_curvaturemax_section_yz_t4_t7.png",
        ("Coherence", "XZ"): "05_coherence_section_xz_t4_t7.png",
        ("Coherence", "YZ"): "06_coherence_section_yz_t4_t7.png",
    }
    attribute_summary: dict[str, object] = {}
    attributes = [str(name) for name in config.get("attribute_names", ["AntTrack", "CurvatureMax"])]
    unsupported = [name for name in attributes if name not in ATTRIBUTE_SETTINGS]
    if unsupported:
        raise ValueError(f"unsupported geological attributes: {unsupported}")
    for attribute in attributes:
        print(f"[geological-attribute] sampling {attribute}", flush=True)
        xz, yz, stats = sample_volume_sections(geometry, attribute, paths[attribute])
        vmin, vmax, norm, scale_info = display_scale(attribute, [xz.values, yz.values])
        plot_section(xz, geometry, output_dir / image_names[(attribute, "XZ")], vmin, vmax, norm)
        plot_section(yz, geometry, output_dir / image_names[(attribute, "YZ")], vmin, vmax, norm)
        save_section_pair_npz(output_dir / f"{attribute.lower()}_section_samples.npz", xz, yz)
        attribute_summary[attribute] = {**stats, "display": scale_info, "images": [image_names[(attribute, "XZ")], image_names[(attribute, "YZ")]]}
    images = sorted(path.name for path in output_dir.glob("*.png"))
    expected_image_count = 2 * len(attributes)
    checks = {
        "only_requested_attributes": set(attribute_summary) == set(attributes),
        "no_curvature_pos_or_combined": not any("curvaturepos" in name.lower() or "combined" in name.lower() for name in images),
        "expected_png_count_generated": len(images) == expected_image_count,
        "all_sections_have_finite_values": all(float(item["finite_fraction"]) > 0 for item in attribute_summary.values()),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "section_geometry": geometry.summary,
        "attribute_summary": attribute_summary,
        "png_files": images,
        "checks": checks,
    }
    write_json(output_dir / "attribute_section_summary.json", summary)
    print(f"[geological-attribute] output={output_dir}", flush=True)
    print(f"[geological-attribute] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
