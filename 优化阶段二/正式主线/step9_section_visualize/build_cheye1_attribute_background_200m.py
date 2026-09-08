# -*- coding: utf-8 -*-
"""Build clean Cheye-1 200 m attribute-background sections for PPT.

Only the sampled seismic attribute, axes and colorbar are rendered.  No well,
horizon, fracture, DFN or fault overlays are added so these images can be used
as standalone data-background panels.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from build_all_area_section_visualization import configure_matplotlib_fonts
from build_cheye1_dfn_coherence_sections import aligned_local_axis, finite_bounds_from_curves
from build_well_attribute_section_visualization import axis_edges
from build_well_geological_attribute_sections import display_scale
from trace_horizon_section import build_trace_horizon_section_curves, resolve_horizon_trace_table
from well_curved_section_common import CurvedSectionGeometry, prepare_geometry, read_json, sample_volume_sections, write_json


ATTRIBUTE_LABELS = {"AntTrack": "蚂蚁体", "Coherence": "相干体", "CurvatureMax": "最大曲率体"}
ATTRIBUTE_CMAPS = {"AntTrack": "gray_r", "Coherence": "gray", "CurvatureMax": "gray_r"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build clean Cheye1 200m geological attribute backgrounds.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def local_geometry(base: CurvedSectionGeometry, config: dict[str, Any]) -> CurvedSectionGeometry:
    radius = float(config.get("local_axis_radius_m", 200.0))
    x_values = aligned_local_axis(base.trace_df, "X", float(base.well_df["X"].min()) - radius, float(base.well_df["X"].max()) + radius)
    y_values = aligned_local_axis(base.trace_df, "Y", float(base.well_df["Y"].min()) - radius, float(base.well_df["Y"].max()) + radius)

    def decimate(values: np.ndarray, key: str) -> np.ndarray:
        limit = int(config.get(key, 0))
        if limit <= 1 or len(values) <= limit:
            return values
        return values[np.unique(np.linspace(0, len(values) - 1, limit, dtype=int))]

    x_values = decimate(x_values, "ppt_x_sample_count")
    y_values = decimate(y_values, "ppt_y_sample_count")
    curves, horizon_summary = build_trace_horizon_section_curves(
        resolve_horizon_trace_table(config), base.well_df, base.trace_tree, base.trace_ids,
        x_values, y_values, iteration_count=int(config.get("horizon_curve_iteration_count", 12)),
    )
    time_min, time_max = finite_bounds_from_curves(curves, float(config.get("time_padding_ms", 20.0)))
    summary = dict(base.summary)
    summary.update({
        "scope_name": "cheye1_local_200m_attribute_background",
        "display_scope": "车页1导眼轨迹外扩200m后取最近地震道",
        "display_x_min": float(x_values.min()), "display_x_max": float(x_values.max()),
        "display_y_min": float(y_values.min()), "display_y_max": float(y_values.max()),
        "display_time_min": float(time_min), "display_time_max": float(time_max),
        "section_x_sample_count": int(len(x_values)), "section_y_sample_count": int(len(y_values)),
        "horizon_display": horizon_summary,
    })
    args = SimpleNamespace(**vars(base.args))
    args.fig_width = float(config.get("fig_width", 15.0))
    args.fig_height = float(config.get("fig_height", 10.0))
    args.dpi = int(config.get("render_dpi", 300))
    return replace(base, config=dict(config), args=args, surface_curves=curves, summary=summary,
                   x_values=x_values, y_values=y_values, time_min=time_min, time_max=time_max)


def plot_background(section, geometry, output_path: Path, scale_info: dict[str, Any]) -> dict[str, Any]:
    attribute = section.attribute
    finite = section.values[np.isfinite(section.values)]
    if finite.size == 0:
        raise RuntimeError(f"{attribute} {section.projection} contains no finite samples")
    vmin = float(scale_info["vmin"])
    vmax = float(scale_info["vmax"])
    display_values = np.abs(section.values) if attribute == "CurvatureMax" else section.values
    fig, ax = plt.subplots(figsize=(geometry.args.fig_width, geometry.args.fig_height))
    mesh = ax.pcolormesh(
        axis_edges(section.h), axis_edges(section.time), display_values,
        shading="auto", cmap=ATTRIBUTE_CMAPS[attribute], vmin=vmin, vmax=vmax,
        rasterized=True, zorder=1,
    )
    ax.set_xlim(float(section.h.min()), float(section.h.max()))
    ax.set_ylim(float(geometry.summary["display_time_min"]), float(geometry.summary["display_time_max"]))
    ax.invert_yaxis()
    ax.set_xlabel("X / m" if section.projection == "XZ" else "Y / m")
    ax.set_ylabel(str(geometry.args.z_label))
    ax.grid(True, linewidth=0.25, alpha=0.20)
    ax.set_title(f"车页1导眼井周200m {ATTRIBUTE_LABELS[attribute]}背景 | {section.projection}", fontsize=12)
    colorbar_label = f"{ATTRIBUTE_LABELS[attribute]}绝对值" if attribute == "CurvatureMax" else ATTRIBUTE_LABELS[attribute]
    fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.94, label=colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=geometry.args.dpi)
    plt.close(fig)
    return {"file": output_path.name, "projection": section.projection, "attribute": attribute,
            "shape": [int(value) for value in section.values.shape], "finite_fraction": float(np.isfinite(section.values).mean()),
            "vmin": vmin, "vmax": vmax}


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    configure_matplotlib_fonts()
    base = prepare_geometry(config)
    geometry = local_geometry(base, config)
    output_dir = Path(str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: Path(str(path)).resolve() for name, path in dict(config["volume_paths"]).items()}
    attributes = [str(value) for value in config.get("attribute_names", ["AntTrack", "Coherence", "CurvatureMax"])]
    requested = set(attributes)
    if requested != set(ATTRIBUTE_LABELS):
        raise ValueError(f"attribute_names must contain exactly {sorted(ATTRIBUTE_LABELS)}")
    sections: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    scale_summary: dict[str, Any] = {}
    for attribute in attributes:
        print(f"[attribute-background] sampling {attribute}", flush=True)
        xz, yz, stats = sample_volume_sections(geometry, attribute, paths[attribute])
        # Use one shared color scale for XZ/YZ of each attribute.
        vmin, vmax, _norm, scale_info = display_scale(attribute, [xz.values, yz.values], config.get("formal_signed_attribute_display_style", "absolute_grayscale"))
        if attribute == "CurvatureMax":
            vmin = 0.0
        scale = {"vmin": float(vmin), "vmax": float(vmax), **scale_info}
        sections[attribute] = (xz, yz, stats)
        scale_summary[attribute] = scale
    files: list[dict[str, Any]] = []
    for attribute in attributes:
        xz, yz, _stats = sections[attribute]
        files.append(plot_background(xz, geometry, output_dir / f"{attribute.lower()}_background_xz_200m.png", scale_summary[attribute]))
        files.append(plot_background(yz, geometry, output_dir / f"{attribute.lower()}_background_yz_200m.png", scale_summary[attribute]))
    checks = {
        "exactly_six_background_pngs": len(files) == 6,
        "all_finite_fraction_positive": all(item["finite_fraction"] > 0 for item in files),
        "no_overlay_layers": True,
        "local_200m_geometry": geometry.summary.get("scope_name") == "cheye1_local_200m_attribute_background",
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(args.config.resolve()), "output_dir": str(output_dir),
        "section_geometry": geometry.summary, "scale_summary": scale_summary,
        "files": files, "checks": checks,
        "render_contract": "attribute raster + axes + colorbar only; no well/horizon/fracture/DFN/fault overlays",
    }
    write_json(output_dir / "attribute_background_200m_summary.json", summary)
    print(f"[attribute-background] output={output_dir}", flush=True)
    print(f"[attribute-background] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
