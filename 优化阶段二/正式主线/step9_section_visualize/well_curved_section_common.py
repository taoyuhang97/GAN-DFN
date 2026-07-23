# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from build_all_area_section_visualization import (
    configure_matplotlib_fonts,
    select_demo_well,
)
from build_well_attribute_section_visualization import (
    AttributeSection,
    build_axis_values,
    build_display_bounds,
    build_trace_grid,
    build_trace_tree,
    draw_surface_curves,
    draw_well_trajectory,
    open_volume_context,
    sample_attribute_section,
    select_time_samples,
)
from trace_horizon_section import build_trace_horizon_section_curves, resolve_horizon_trace_table


@dataclass
class CurvedSectionGeometry:
    config: dict[str, Any]
    args: SimpleNamespace
    well_name: str
    well_path: Path
    well_df: pd.DataFrame
    trace_df: pd.DataFrame
    trace_tree: Any
    trace_ids: np.ndarray
    surface_curves: list[Any]
    summary: dict[str, Any]
    x_values: np.ndarray
    y_values: np.ndarray
    time_min: float
    time_max: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def path_value(config: dict[str, Any], key: str, required: bool = True) -> Path | None:
    value = config.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing config path: {key}")
        return None
    path = Path(str(value)).resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def prepare_geometry(config: dict[str, Any]) -> CurvedSectionGeometry:
    configure_matplotlib_fonts()
    trace_header_csv = path_value(config, "trace_header_csv")
    real_well_samples_root = path_value(config, "real_well_samples_root", required=False)
    well_trajectory_csv = path_value(config, "well_trajectory_csv", required=False)
    surface_dir = path_value(config, "surface_dir", required=False)
    target_block = dict(config.get("target_block") or {})
    args = SimpleNamespace(
        trace_header_csv=trace_header_csv,
        real_well_samples_root=real_well_samples_root,
        well_trajectory_csv=well_trajectory_csv,
        surface_dir=surface_dir,
        well_name=config.get("well_name"),
        exclude_wells=set(str(item) for item in config.get("exclude_wells", [])),
        target_block=target_block,
        surface_samples=int(config.get("surface_samples", 500)),
        z_label=str(config.get("z_label", "TWT / ms")),
        fig_width=float(config.get("fig_width", 15.5)),
        fig_height=float(config.get("fig_height", 7.8)),
        dpi=int(config.get("dpi", 240)),
    )
    selected = select_demo_well(args)
    well_df = selected.trajectory.copy().sort_values("TIME").drop_duplicates("TIME").reset_index(drop=True)
    args.selected_well_name = selected.well_name
    args.well_trajectory_csv = selected.path

    trace_df = build_trace_grid(trace_header_csv, target_block)
    trace_tree, trace_ids = build_trace_tree(trace_df)
    summary: dict[str, Any] = {
        "selected_well_name": selected.well_name,
        "selected_well_csv": str(selected.path),
        "selected_well_inside_target_rows": int(selected.inside_rows),
        "selected_well_total_rows": int(selected.total_rows),
        "selected_well_inside_target_ratio": float(selected.inside_ratio),
        "trace_header_csv": str(trace_header_csv),
        "surface_dir": str(surface_dir) if surface_dir is not None else None,
        "target_block": target_block,
        "trace_count_in_block": int(len(trace_df)),
        "trace_unique_x": int(trace_df["X"].nunique()),
        "trace_unique_y": int(trace_df["Y"].nunique()),
        "section_mode": "well_trajectory_curved_surface_t4_t7",
        "xy_projection_logic": {
            "XZ": "sample volume at (x, Ywell(time), time)",
            "YZ": "sample volume at (Xwell(time), y, time)",
        },
    }
    summary.update(build_display_bounds(trace_df, target_block))
    axis_count = int(config.get("axis_sample_count", 0))
    x_values = build_axis_values(trace_df, "X", axis_count)
    y_values = build_axis_values(trace_df, "Y", axis_count)
    summary.update(
        {
            "axis_sample_count_config": axis_count,
            "section_x_sample_count": int(len(x_values)),
            "section_y_sample_count": int(len(y_values)),
        }
    )
    horizon_trace_table_path = resolve_horizon_trace_table(config)
    surface_curves, horizon_summary = build_trace_horizon_section_curves(
        horizon_trace_table_path,
        well_df,
        trace_tree,
        trace_ids,
        x_values,
        y_values,
        iteration_count=int(config.get("horizon_curve_iteration_count", 12)),
    )
    summary["horizon_display"] = horizon_summary
    finite_curve_times = [curve.z[np.isfinite(curve.z)] for curve in surface_curves if np.isfinite(curve.z).any()]
    if not finite_curve_times:
        raise RuntimeError("no finite T4-T7 surface-section curves")
    curve_times = np.concatenate(finite_curve_times)
    padding = float(config.get("time_padding_ms", 20.0))
    time_min = float(np.nanmin(curve_times)) - padding
    time_max = float(np.nanmax(curve_times)) + padding
    summary.update(
        {
            "display_time_min": time_min,
            "display_time_max": time_max,
            "well_time_min": float(well_df["TIME"].min()),
            "well_time_max": float(well_df["TIME"].max()),
            "surface_codes_drawn": sorted({curve.surface for curve in surface_curves}),
        }
    )
    return CurvedSectionGeometry(
        config=config,
        args=args,
        well_name=selected.well_name,
        well_path=selected.path,
        well_df=well_df,
        trace_df=trace_df,
        trace_tree=trace_tree,
        trace_ids=trace_ids,
        surface_curves=surface_curves,
        summary=summary,
        x_values=x_values,
        y_values=y_values,
        time_min=time_min,
        time_max=time_max,
    )


def sample_volume_sections(
    geometry: CurvedSectionGeometry,
    attribute: str,
    volume_path: Path,
) -> tuple[AttributeSection, AttributeSection, dict[str, Any]]:
    handle, samples, trace_at = open_volume_context(volume_path)
    try:
        attr_time_min = max(geometry.time_min, float(np.nanmin(samples)))
        attr_time_max = min(geometry.time_max, float(np.nanmax(samples)))
        if attr_time_max <= attr_time_min:
            raise ValueError(f"{attribute} does not overlap section time range")
        time_values = select_time_samples(samples, attr_time_min, attr_time_max)
        progress = int(geometry.config.get("progress_interval", 100))
        xz_values = sample_attribute_section(
            "XZ",
            geometry.x_values,
            time_values,
            geometry.well_df,
            geometry.trace_tree,
            geometry.trace_ids,
            samples,
            trace_at,
            progress,
        )
        yz_values = sample_attribute_section(
            "YZ",
            geometry.y_values,
            time_values,
            geometry.well_df,
            geometry.trace_tree,
            geometry.trace_ids,
            samples,
            trace_at,
            progress,
        )
    finally:
        handle.close()
    xz = AttributeSection(attribute, "XZ", geometry.x_values.copy(), time_values.copy(), xz_values)
    yz = AttributeSection(attribute, "YZ", geometry.y_values.copy(), time_values.copy(), yz_values)
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in [xz_values, yz_values] if np.isfinite(arr).any()])
    if finite.size == 0:
        raise RuntimeError(f"{attribute} curved sections contain no finite samples")
    stats = {
        "volume_path": str(volume_path),
        "source_time_min_ms": float(samples[0]),
        "source_time_max_ms": float(samples[-1]),
        "source_sample_count": int(len(samples)),
        "section_time_min_ms": float(time_values[0]),
        "section_time_max_ms": float(time_values[-1]),
        "section_time_sample_count": int(len(time_values)),
        "xz_shape": [int(v) for v in xz_values.shape],
        "yz_shape": [int(v) for v in yz_values.shape],
        "finite_fraction": float(finite.size / (xz_values.size + yz_values.size)),
        "value_quantiles": {
            "q01": float(np.quantile(finite, 0.01)),
            "q02": float(np.quantile(finite, 0.02)),
            "q50": float(np.quantile(finite, 0.50)),
            "q98": float(np.quantile(finite, 0.98)),
            "q99": float(np.quantile(finite, 0.99)),
        },
    }
    return xz, yz, stats


def save_section_pair_npz(path: Path, xz: AttributeSection, yz: AttributeSection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xz_h=xz.h.astype(np.float64),
        xz_time=xz.time.astype(np.float64),
        xz_values=xz.values.astype(np.float32),
        yz_h=yz.h.astype(np.float64),
        yz_time=yz.time.astype(np.float64),
        yz_values=yz.values.astype(np.float32),
    )


__all__ = [
    "AttributeSection",
    "CurvedSectionGeometry",
    "draw_surface_curves",
    "draw_well_trajectory",
    "prepare_geometry",
    "read_json",
    "sample_volume_sections",
    "save_section_pair_npz",
    "write_json",
]
