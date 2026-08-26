# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from build_cheye1_dfn_coherence_sections import (
    aligned_local_axis,
    build_imaging_fracture_patch_segments,
    build_namespace,
    finite_bounds_from_curves,
    infer_dfn_patch_csv,
    load_dfn_patch_metadata,
    load_step3_imaging_fracture_labels,
    load_step3_imaging_segment_track,
    path_from_config,
    reset_output_images,
    scan_unified_original_fault_intersections,
    scan_vtk_intersections,
    validate_inputs,
)
from build_well_geological_attribute_sections import display_scale, plot_section
from build_well_seismic_amplitude_sections import amplitude_limit, plot_variable_density, plot_wiggle_variable_area
from dfn_section_overlay import SectionOverlayContext, draw_section_overlays
from trace_horizon_section import build_trace_horizon_section_curves, resolve_horizon_trace_table
from well_curved_section_common import (
    AttributeSection,
    CurvedSectionGeometry,
    apply_display_t7_curve,
    prepare_geometry,
    read_json,
    sample_volume_sections,
    save_section_pair_npz,
    write_json,
)

FORMAL_ROOT = Path(__file__).resolve().parent.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))
from common.horizon_trace_table.horizon_contract import build_spatial_lookup  # noqa: E402


ATTRIBUTE_ORDER = ["AntTrack", "Coherence", "CurvatureMax"]
PRODUCT_TITLE = "车页1导眼：DFN与成像测井裂缝对比剖面"
FORMAL_DISPLAY_STYLE = "absolute_grayscale"
BACKUP_DISPLAY_STYLE = "signed_red_white_blue"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the formal 20-image Cheye1 DFN multi-background section set.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def local_geometry(base: CurvedSectionGeometry, config: dict[str, Any]) -> CurvedSectionGeometry:
    radius = float(config.get("local_axis_radius_m", 200.0))
    x_values = aligned_local_axis(
        base.trace_df,
        "X",
        float(base.well_df["X"].min()) - radius,
        float(base.well_df["X"].max()) + radius,
    )
    y_values = aligned_local_axis(
        base.trace_df,
        "Y",
        float(base.well_df["Y"].min()) - radius,
        float(base.well_df["Y"].max()) + radius,
    )
    curves, horizon_summary = build_trace_horizon_section_curves(
        resolve_horizon_trace_table(config),
        base.well_df,
        base.trace_tree,
        base.trace_ids,
        x_values,
        y_values,
        iteration_count=int(config.get("horizon_curve_iteration_count", 12)),
    )
    curves = apply_display_t7_curve(
        config,
        base.well_df,
        base.trace_tree,
        base.trace_ids,
        x_values,
        y_values,
        curves,
        iteration_count=int(config.get("horizon_curve_iteration_count", 12)),
    )
    time_min, time_max = finite_bounds_from_curves(curves, float(config.get("time_padding_ms", 20.0)))
    summary = dict(base.summary)
    summary.update(
        {
            "scope_name": "local_200m",
            "display_x_min": float(x_values.min()),
            "display_x_max": float(x_values.max()),
            "display_y_min": float(y_values.min()),
            "display_y_max": float(y_values.max()),
            "display_time_min": float(time_min),
            "display_time_max": float(time_max),
            "section_x_sample_count": int(len(x_values)),
            "section_y_sample_count": int(len(y_values)),
            "local_axis_radius_m": radius,
            "horizon_display": horizon_summary,
        }
    )
    local_args = SimpleNamespace(**vars(base.args))
    local_args.fig_width = float(config.get("local_fig_width", 15.5))
    local_args.fig_height = float(config.get("local_fig_height", 7.8))
    local_config = dict(base.config)
    for key in (
        "figure_height_px",
        "xz_total_width_px",
        "yz_total_width_px",
        "panel_width_px",
        "xz_panel_count",
        "yz_panel_count",
    ):
        local_config.pop(key, None)
    local_config["render_dpi"] = int(config.get("local_render_dpi", config.get("dpi", 240)))
    return replace(
        base,
        config=local_config,
        args=local_args,
        surface_curves=curves,
        summary=summary,
        x_values=x_values,
        y_values=y_values,
        time_min=float(time_min),
        time_max=float(time_max),
    )


def load_section_pair_npz(path: Path, attribute: str) -> tuple[AttributeSection, AttributeSection, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"overview section cache not found: {path}")
    with np.load(path) as data:
        xz = AttributeSection(
            attribute,
            "XZ",
            np.asarray(data["xz_h"], dtype=np.float64),
            np.asarray(data["xz_time"], dtype=np.float64),
            np.asarray(data["xz_values"], dtype=np.float64),
        )
        yz = AttributeSection(
            attribute,
            "YZ",
            np.asarray(data["yz_h"], dtype=np.float64),
            np.asarray(data["yz_time"], dtype=np.float64),
            np.asarray(data["yz_values"], dtype=np.float64),
        )
    finite_count = int(np.isfinite(xz.values).sum() + np.isfinite(yz.values).sum())
    return xz, yz, {
        "source": "existing_mine_scale_npz",
        "cache_path": str(path),
        "xz_shape": [int(value) for value in xz.values.shape],
        "yz_shape": [int(value) for value in yz.values.shape],
        "finite_fraction": float(finite_count / (xz.values.size + yz.values.size)),
        "section_time_min_ms": float(min(xz.time.min(), yz.time.min())),
        "section_time_max_ms": float(max(xz.time.max(), yz.time.max())),
    }


def make_overlay_drawer(context: SectionOverlayContext) -> tuple[Callable[[Any, str], dict[str, int]], dict[str, int]]:
    captured: dict[str, int] = {}

    def draw(ax: Any, projection: str) -> dict[str, int]:
        captured.clear()
        captured.update(draw_section_overlays(ax, context, projection))
        return dict(captured)

    return draw, captured


def build_overlay_contexts(
    config: dict[str, Any],
    overview: CurvedSectionGeometry,
    local: CurvedSectionGeometry,
) -> tuple[dict[str, SectionOverlayContext], dict[str, Any]]:
    standard_args = build_namespace(config, float(config.get("dfn_half_width_m", 50.0)))
    local_args = build_namespace(config, float(config.get("local_dfn_half_width_m", 200.0)))
    local_args.small_projection_half_width = float(
        config.get("local_small_projection_half_width_m", config.get("small_projection_half_width_m", local_args.half_width))
    )
    patch_csv = infer_dfn_patch_csv(config)
    metadata = load_dfn_patch_metadata(patch_csv)
    for namespace in (standard_args, local_args):
        namespace.dfn_patch_metadata = metadata
        namespace.well_trajectory_csv = overview.well_path
        namespace.selected_well_name = overview.well_name

    group_paths = [Path(str(value)).resolve() for value in config.get("step3_imaging_group_csvs", [])]
    fracture_df = load_step3_imaging_fracture_labels(group_paths, overview.well_name, dict(config.get("target_block") or {}))
    imaging_df = load_step3_imaging_segment_track(group_paths, overview.well_name, dict(config.get("target_block") or {}))
    imaging_patches = build_imaging_fracture_patch_segments(fracture_df)
    surfaces = build_spatial_lookup(config)
    print("[multi-background] scanning overview DFN", flush=True)
    standard_segments, standard_scan = scan_vtk_intersections(standard_args, surfaces, overview.well_df)
    print("[multi-background] scanning local 200m DFN", flush=True)
    local_segments, local_scan = scan_vtk_intersections(local_args, surfaces, overview.well_df)

    overview_display = {
        "display_x_min": float(overview.summary["display_x_min"]),
        "display_x_max": float(overview.summary["display_x_max"]),
        "display_y_min": float(overview.summary["display_y_min"]),
        "display_y_max": float(overview.summary["display_y_max"]),
    }
    local_display = {
        "display_x_min": float(local.summary["display_x_min"]),
        "display_x_max": float(local.summary["display_x_max"]),
        "display_y_min": float(local.summary["display_y_min"]),
        "display_y_max": float(local.summary["display_y_max"]),
    }
    standard_faults, standard_fault_scan = scan_unified_original_fault_intersections(
        standard_args.input_vtk,
        overview.well_df,
        overview_display,
    )
    local_faults, local_fault_scan = scan_unified_original_fault_intersections(
        standard_args.input_vtk,
        overview.well_df,
        local_display,
    )

    contexts = {
        "overview": SectionOverlayContext(
            "overview", standard_segments, standard_faults, imaging_patches, fracture_df, imaging_df, overview.summary
        ),
        "local_200m": SectionOverlayContext(
            "local_200m", local_segments, local_faults, imaging_patches, fracture_df, imaging_df, local.summary
        ),
    }
    summary = {
        "dfn_patch_csv": str(patch_csv) if patch_csv else None,
        "dfn_patch_count": int(len(metadata)) if metadata is not None else 0,
        "step3_group_csvs": [str(path) for path in group_paths],
        "step3_fracture_point_count": int(len(fracture_df)),
        "step3_imaging_track_sample_count": int(len(imaging_df)),
        "step3_imaging_patch_segment_count": int(len(imaging_patches)),
        "overview_scan": standard_scan,
        "local_200m_scan": local_scan,
        "overview_fault_scan": standard_fault_scan,
        "local_200m_fault_scan": local_fault_scan,
    }
    return contexts, summary


def render_all(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(config)
    presentation_root = config.get("presentation_output_root")
    scale_name = str(config.get("presentation_scale_name", ""))
    presentation_enabled = bool(presentation_root and scale_name)
    output_dir = path_from_config(config, "output_dir")
    if not presentation_enabled:
        reset_output_images(output_dir)
    overview_config = dict(config)
    overview_config["target_block"] = dict(config.get("overview_target_block") or {})
    overview = prepare_geometry(overview_config)
    overview.summary["scope_name"] = "overview"
    local = local_geometry(overview, config)
    scopes = {"overview": overview, "local_200m": local}
    # 显示下界覆盖：T7 以下延伸区（DFN 最深约 3618 ms）默认被 T7+padding 截断，
    # 用 section_time_max_override_ms 把剖面纵向范围扩到该深度，T7 层位线仍按原始位置。
    time_max_override = config.get("section_time_max_override_ms")
    if time_max_override:
        override_value = float(time_max_override)
        for geometry in scopes.values():
            geometry.time_max = max(float(geometry.time_max), override_value)
            geometry.summary["display_time_max"] = max(
                float(geometry.summary.get("display_time_max", 0.0)), override_value
            )
            geometry.summary["section_time_max_override_ms"] = override_value
    for geometry in scopes.values():
        curve_extents: dict[str, tuple[float | None, float | None]] = {}
        for projection in ("XZ", "YZ"):
            tmin: list[float] = []
            tmax: list[float] = []
            for curve in geometry.surface_curves:
                if curve.projection != projection:
                    continue
                z = np.asarray(curve.z)
                finite = np.isfinite(z)
                if finite.any():
                    tmin.append(float(z[finite].min()))
                    tmax.append(float(z[finite].max()))
            curve_extents[projection] = (
                (min(tmin), max(tmax)) if tmin else (None, None)
            )
        geometry.summary["_curve_extents"] = curve_extents
    scope_labels = {"overview": "矿区尺度 | 当前候选区DFN", "local_200m": "井周200 m"}
    overlay_contexts, overlay_summary = build_overlay_contexts(config, overview, local)

    sampled: dict[str, dict[str, tuple[Any, Any, dict[str, Any]]]] = {}
    for scope_name, geometry in scopes.items():
        sampled[scope_name] = {}
        for attribute in [*ATTRIBUTE_ORDER, "SeisAmp"]:
            cache_value = dict(config.get("overview_section_sample_paths") or {}).get(attribute) if scope_name == "overview" else None
            if cache_value:
                print(f"[multi-background] loading mine cache attribute={attribute}", flush=True)
                sampled[scope_name][attribute] = load_section_pair_npz(Path(str(cache_value)).resolve(), attribute)
            else:
                print(f"[multi-background] sampling scope={scope_name} attribute={attribute}", flush=True)
                sampled[scope_name][attribute] = sample_volume_sections(
                    geometry,
                    attribute,
                    Path(str(config["volume_paths"][attribute])).resolve(),
                )
            xz, yz, _ = sampled[scope_name][attribute]
            cache_dir = output_dir / "section_samples" / scope_name
            save_section_pair_npz(cache_dir / f"{attribute.lower()}_section_samples.npz", xz, yz)

    # 属性剖面（蚂蚁/相干/曲率）的深度范围，供振幅/波形剖面套用，保证整组剖面深度一致
    depth_pad = 10.0
    for scope_name, geometry in scopes.items():
        attribute_bounds: dict[str, tuple[float, float]] = {}
        for projection in ("XZ", "YZ"):
            proj_index = 0 if projection == "XZ" else 1
            tmin: list[float] = []
            tmax: list[float] = []
            for attribute in ATTRIBUTE_ORDER:
                section = sampled[scope_name][attribute][proj_index]
                vals = np.asarray(section.values)
                time = np.asarray(section.time)
                finite_rows = np.isfinite(vals).any(axis=1)
                if finite_rows.any():
                    tmin.append(float(time[finite_rows].min()))
                    tmax.append(float(time[finite_rows].max()))
            curve_ext = geometry.summary.get("_curve_extents", {}).get(projection)
            candidates_min = [
                value for value in (*tmin, curve_ext[0] if curve_ext else None)
                if value is not None
            ]
            candidates_max = [
                value for value in (*tmax, curve_ext[1] if curve_ext else None)
                if value is not None
            ]
            top = min(candidates_min) if candidates_min else float(geometry.time_min)
            bottom = max(candidates_max) if candidates_max else float(geometry.time_max)
            attribute_bounds[projection] = (
                max(float(geometry.time_min), top - depth_pad),
                min(float(geometry.time_max), bottom + depth_pad),
            )
        geometry.summary["_attribute_depth_bounds"] = attribute_bounds

    formal_display_style = str(config.get("formal_signed_attribute_display_style", FORMAL_DISPLAY_STYLE))
    backup_display_style = str(config.get("signed_attribute_backup_display_style", BACKUP_DISPLAY_STYLE))
    backup_dir = output_dir / str(config.get("signed_attribute_backup_dir", "red_white_blue_backup"))
    attribute_scales: dict[str, tuple[float, float, Any, dict[str, Any]]] = {}
    for attribute in ATTRIBUTE_ORDER:
        arrays = [
            sampled[scope][attribute][projection].values
            for scope in scopes
            for projection in (0, 1)
        ]
        attribute_scales[attribute] = display_scale(attribute, arrays, formal_display_style)
    curvature_arrays = [
        sampled[scope]["CurvatureMax"][projection].values
        for scope in scopes
        for projection in (0, 1)
    ]
    backup_curvature_scale = display_scale("CurvatureMax", curvature_arrays, backup_display_style)
    seis_sections = [
        sampled[scope]["SeisAmp"][projection]
        for scope in scopes
        for projection in (0, 1)
    ]
    seis_limit = amplitude_limit(seis_sections, float(config.get("amplitude_clip_quantile", 0.99)))

    image_plan = [
        (1, "overview", "AntTrack", "XZ", "attribute"),
        (2, "overview", "AntTrack", "YZ", "attribute"),
        (3, "overview", "Coherence", "XZ", "attribute"),
        (4, "overview", "Coherence", "YZ", "attribute"),
        (5, "overview", "CurvatureMax", "XZ", "attribute"),
        (6, "overview", "CurvatureMax", "YZ", "attribute"),
        (7, "overview", "SeisAmp", "XZ", "density"),
        (8, "overview", "SeisAmp", "YZ", "density"),
        (9, "overview", "SeisAmp", "XZ", "wiggle"),
        (10, "overview", "SeisAmp", "YZ", "wiggle"),
        (11, "local_200m", "AntTrack", "XZ", "attribute"),
        (12, "local_200m", "AntTrack", "YZ", "attribute"),
        (13, "local_200m", "Coherence", "XZ", "attribute"),
        (14, "local_200m", "Coherence", "YZ", "attribute"),
        (15, "local_200m", "CurvatureMax", "XZ", "attribute"),
        (16, "local_200m", "CurvatureMax", "YZ", "attribute"),
        (17, "local_200m", "SeisAmp", "XZ", "density"),
        (18, "local_200m", "SeisAmp", "YZ", "density"),
        (19, "local_200m", "SeisAmp", "XZ", "wiggle"),
        (20, "local_200m", "SeisAmp", "YZ", "wiggle"),
    ]
    requested_numbers = {int(value) for value in config.get("image_numbers", [])}
    if requested_numbers:
        image_plan = [row for row in image_plan if row[0] in requested_numbers]

    if presentation_enabled:
        root = Path(str(presentation_root)).resolve() / scale_name
        root.mkdir(parents=True, exist_ok=True)
        targets = dict(config.get("presentation_targets") or {})
        full_dir = root / "full_resolution"
        full_summary = render_pass(
            "full_resolution", full_dir, config_path, config, image_plan,
            scopes, sampled, overlay_contexts, overlay_summary,
            attribute_scales, backup_curvature_scale, seis_limit,
            scope_labels, formal_display_style, backup_display_style, requested_numbers,
        )
        ppt_sampled = build_ppt_sampled(sampled, targets)
        ppt_scopes = {
            name: make_ppt_geometry(geometry, targets.get(name, {}))
            for name, geometry in scopes.items()
        }
        ppt_dir = root / "ppt_decimated"
        ppt_summary = render_pass(
            "ppt_decimated", ppt_dir, config_path, config, image_plan,
            ppt_scopes, ppt_sampled, overlay_contexts, overlay_summary,
            attribute_scales, backup_curvature_scale, seis_limit,
            scope_labels, formal_display_style, backup_display_style, requested_numbers,
        )
        root_summary = {
            "status": "pass"
            if full_summary["status"] == "pass" and ppt_summary["status"] == "pass"
            else "fail",
            "config_path": str(config_path.resolve()),
            "scale_name": scale_name,
            "presentation_targets": targets,
            "full_resolution": {
                "dir": str(full_dir),
                "image_count": int(len(full_summary["images"])),
                "summary_json": str(full_dir / "section_summary.json"),
            },
            "ppt_decimated": {
                "dir": str(ppt_dir),
                "image_count": int(len(ppt_summary["images"])),
                "summary_json": str(ppt_dir / "section_summary.json"),
            },
            "ppt_wiggle_trace_counts": {
                f"{row['number']:02d}_{row['scope']}_{row['projection']}": int(row["overlay"].get("wiggle_trace_count", -1))
                for row in ppt_summary["images"]
                if row["renderer"] == "wiggle"
            },
        }
        write_json(root / "section_presentation_summary.json", root_summary)
        print(f"[multi-background] presentation root={root}", flush=True)
        print(f"[multi-background] status={root_summary['status']}", flush=True)
        return root_summary

    summary = render_pass(
        "default", output_dir, config_path, config, image_plan,
        scopes, sampled, overlay_contexts, overlay_summary,
        attribute_scales, backup_curvature_scale, seis_limit,
        scope_labels, formal_display_style, backup_display_style, requested_numbers,
    )
    return summary
def decimate_section(section: AttributeSection, max_traces: int, time_decim: int) -> AttributeSection:
    """按目标道数与时间采样倍数抽稀一个剖面切片。"""
    n_traces = len(section.h)
    if max_traces and max_traces > 0 and max_traces < n_traces:
        selected = np.unique(np.linspace(0, n_traces - 1, int(max_traces), dtype=int))
    else:
        selected = np.arange(n_traces, dtype=int)
    tstep = max(1, int(time_decim or 1))
    values = np.asarray(section.values)
    decimated = AttributeSection(
        section.attribute,
        section.projection,
        np.asarray(section.h)[selected],
        np.asarray(section.time)[::tstep],
        values[::tstep][:, selected],
    )
    for attr_name in ("display_time_min", "display_time_max"):
        if hasattr(section, attr_name):
            setattr(decimated, attr_name, getattr(section, attr_name))
    return decimated


def apply_section_display_bounds(section: AttributeSection, geometry: CurvedSectionGeometry) -> None:
    """按剖面自身数据范围 + 该投影层位曲线范围自适应纵轴（断层深度不参与）。"""
    vals = np.asarray(section.values)
    finite_rows = np.isfinite(vals).any(axis=1)
    time = np.asarray(section.time)
    pad = 10.0
    if finite_rows.any():
        top = float(time[finite_rows].min())
        bottom = float(time[finite_rows].max())
    else:
        top, bottom = float(geometry.time_min), float(geometry.time_max)
    curve_ext = geometry.summary.get("_curve_extents", {}).get(section.projection)
    if curve_ext and curve_ext[0] is not None:
        top = min(top, float(curve_ext[0]))
        bottom = max(bottom, float(curve_ext[1]))
    section.display_time_min = max(float(geometry.time_min), top - pad)
    section.display_time_max = min(float(geometry.time_max), bottom + pad)


def make_ppt_geometry(geometry: CurvedSectionGeometry, target: dict[str, Any]) -> CurvedSectionGeometry:
    """按 PPT 目标尺寸生成几何：dpi、画布像素、图例列宽、线宽等。"""
    dpi = int(target.get("dpi", 300))
    width_cm = float(target["width_cm"])
    height_cm = float(target["height_cm"])
    width_px = int(round(width_cm / 2.54 * dpi))
    height_px = int(round(height_cm / 2.54 * dpi))
    legend_width_in = float(target.get("legend_width_in", min(4.0, width_cm / 2.54 * 0.28)))
    args = SimpleNamespace(**vars(geometry.args))
    args.fig_width = max(1.0, width_px / dpi - legend_width_in)
    args.fig_height = height_px / dpi
    args.dpi = dpi
    config = dict(geometry.config)
    config["render_dpi"] = dpi
    config["xz_total_width_px"] = max(200, int(round(width_px - legend_width_in * dpi)))
    config["yz_total_width_px"] = max(200, int(round(width_px - legend_width_in * dpi)))
    config["figure_height_px"] = height_px
    config["overlay_legend_width_in"] = legend_width_in
    config["overlay_legend_fontsize"] = float(target.get("legend_fontsize", 8.0))
    config["wiggle_linewidth"] = float(target.get("wiggle_linewidth", 0.7))
    if target.get("wiggle_lateral_scale_fraction") is not None:
        config["wiggle_lateral_scale_fraction"] = float(target["wiggle_lateral_scale_fraction"])
    return replace(geometry, args=args, config=config)


def build_ppt_sampled(sampled: dict[str, Any], targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ppt_sampled: dict[str, Any] = {}
    for scope_name, scope_samples in sampled.items():
        target = targets.get(scope_name, {})
        max_traces_xz = int(target.get("max_traces_xz", target.get("max_traces", 0)))
        max_traces_yz = int(target.get("max_traces_yz", target.get("max_traces", 0)))
        time_decim = int(target.get("time_decim", 1))
        ppt_sampled[scope_name] = {}
        for attribute, pair in scope_samples.items():
            xz, yz, info = pair
            ppt_sampled[scope_name][attribute] = (
                decimate_section(xz, max_traces_xz, time_decim),
                decimate_section(yz, max_traces_yz, time_decim),
                info,
            )
    return ppt_sampled


def render_image_plan(
    image_plan: list[tuple[int, str, str, str, str]],
    output_dir: Path,
    scopes: dict[str, CurvedSectionGeometry],
    sampled: dict[str, Any],
    overlay_contexts: dict[str, SectionOverlayContext],
    attribute_scales: dict[str, Any],
    backup_curvature_scale: Any,
    seis_limit: float,
    scope_labels: dict[str, str],
    config: dict[str, Any],
    formal_display_style: str,
    backup_display_style: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    backup_dir = output_dir / "red_white_blue_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, Any]] = []
    mode_names = {"attribute": "dfn", "density": "density_dfn", "wiggle": "wiggle_dfn"}
    for number, scope_name, attribute, projection, renderer in image_plan:
        section_index = 0 if projection == "XZ" else 1
        section = sampled[scope_name][attribute][section_index]
        geometry = scopes[scope_name]
        apply_section_display_bounds(section, geometry)
        if attribute == "SeisAmp":
            # 振幅/波形剖面与同 scope 同投影的属性剖面共用深度范围，避免纵向不一致
            attr_bounds = geometry.summary.get("_attribute_depth_bounds", {}).get(projection)
            if attr_bounds is not None:
                section.display_time_min = float(attr_bounds[0])
                section.display_time_max = float(attr_bounds[1])
        product_title = str(
            config.get("overview_product_title", config.get("product_title", PRODUCT_TITLE))
            if scope_name == "overview"
            else config.get("local_product_title", PRODUCT_TITLE)
        )
        drawer, captured = make_overlay_drawer(overlay_contexts[scope_name])
        token = attribute.lower() if attribute != "SeisAmp" else "seisamp"
        filename = f"{number:02d}_{'overview' if scope_name == 'overview' else 'local200m'}_{token}_{mode_names[renderer]}_{projection.lower()}_t4_t7.png"
        output_path = output_dir / filename
        if renderer == "attribute":
            vmin, vmax, norm, _ = attribute_scales[attribute]
            stats = plot_section(
                section,
                geometry,
                output_path,
                vmin,
                vmax,
                norm,
                display_style=formal_display_style,
                overlay_drawer=drawer,
                scope_label=scope_labels[scope_name],
                product_title=product_title,
            )
        elif renderer == "density":
            stats = plot_variable_density(
                section,
                geometry,
                output_path,
                seis_limit,
                display_style=formal_display_style,
                overlay_drawer=drawer,
                scope_label=scope_labels[scope_name],
                product_title=product_title,
            )
        else:
            trace_count = plot_wiggle_variable_area(
                section,
                geometry,
                output_path,
                seis_limit,
                overlay_drawer=drawer,
                scope_label=scope_labels[scope_name],
                product_title=product_title,
            )
            stats = dict(captured)
            stats["wiggle_trace_count"] = int(trace_count)
        backup_file: Path | None = None
        if attribute == "CurvatureMax" and renderer == "attribute":
            vmin, vmax, norm, _ = backup_curvature_scale
            backup_file = backup_dir / filename
            backup_drawer, _ = make_overlay_drawer(overlay_contexts[scope_name])
            plot_section(
                section,
                geometry,
                backup_file,
                vmin,
                vmax,
                norm,
                display_style=backup_display_style,
                overlay_drawer=backup_drawer,
                scope_label=scope_labels[scope_name],
                product_title=product_title,
            )
        elif attribute == "SeisAmp" and renderer == "density":
            backup_file = backup_dir / filename
            backup_drawer, _ = make_overlay_drawer(overlay_contexts[scope_name])
            plot_variable_density(
                section,
                geometry,
                backup_file,
                seis_limit,
                display_style=backup_display_style,
                overlay_drawer=backup_drawer,
                scope_label=scope_labels[scope_name],
                product_title=product_title,
            )
        image_rows.append(
            {
                "number": number,
                "file": filename,
                "scope": scope_name,
                "attribute": attribute,
                "renderer": renderer,
                "projection": projection,
                "section_shape": [int(value) for value in section.values.shape],
                "overlay": stats,
                "formal_display_style": formal_display_style if renderer in ("attribute", "density") else "wiggle_black",
                "signed_rwb_backup_file": str(backup_file.relative_to(output_dir)) if backup_file else None,
            }
        )
        print(f"[multi-background] rendered {filename}", flush=True)

    png_files = sorted(path.name for path in output_dir.glob("*.png"))
    backup_png_files = sorted(path.name for path in backup_dir.glob("*.png")) if backup_dir.exists() else []
    return image_rows, png_files, backup_png_files


def render_pass(
    pass_name: str,
    pass_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    image_plan: list[tuple[int, str, str, str, str]],
    scopes: dict[str, CurvedSectionGeometry],
    sampled: dict[str, Any],
    overlay_contexts: dict[str, SectionOverlayContext],
    overlay_summary: dict[str, Any],
    attribute_scales: dict[str, Any],
    backup_curvature_scale: Any,
    seis_limit: float,
    scope_labels: dict[str, str],
    formal_display_style: str,
    backup_display_style: str,
    requested_numbers: set[int],
) -> dict[str, Any]:
    pass_dir.mkdir(parents=True, exist_ok=True)
    for path in pass_dir.glob("*.png"):
        path.unlink()
    backup_dir = pass_dir / "red_white_blue_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in backup_dir.glob("*.png"):
        path.unlink()

    image_rows, png_files, backup_png_files = render_image_plan(
        image_plan,
        pass_dir,
        scopes,
        sampled,
        overlay_contexts,
        attribute_scales,
        backup_curvature_scale,
        seis_limit,
        scope_labels,
        config,
        formal_display_style,
        backup_display_style,
    )

    scan_accounting: dict[str, dict[str, Any]] = {}
    for scan_name in ("overview_scan", "local_200m_scan"):
        scan = dict(overlay_summary[scan_name])
        rejected_before_projection = int(scan["skipped_by_interval"]) + int(scan["skipped_by_area"]) + int(scan["skipped_by_geometry"])
        scan_accounting[scan_name] = {
            "input_polygon_count": int(scan["polygon_count"]),
            "processed_polygon_count": int(scan["processed_polygon_count"]),
            "layer_and_geometry_eligible_patch_count": int(scan["processed_polygon_count"] - rejected_before_projection),
            "selected_patch_count": int(scan["selected_patch_count"]),
            "true_intersection_segment_count": int(scan["selected_intersection_count"]),
            "small_projection_segment_count": int(scan["selected_small_projection_count"]),
            "rejected_before_projection_count": rejected_before_projection,
            "rejected_by_section_distance_or_no_intersection_count": int(scan["skipped_by_surface_distance"]),
            "accounting_total": int(scan["patch_accounting_total"]),
            "accounting_closed": bool(scan["patch_accounting_closed"]),
        }
    original_fault_surface_loaded = all(
        scan.get("fault_trace_source") == "step8_unified_dfn_original_fault_triangles"
        and int(scan.get("surface_triangle_count", 0)) > 0
        for scan in (
            overlay_summary["overview_fault_scan"],
            overlay_summary["local_200m_fault_scan"],
        )
    )
    overlay_consistent = True
    for scope_name in scopes:
        for projection in ("XZ", "YZ"):
            counts = {
                int(row["overlay"].get("dfn_segment_count", -1))
                for row in image_rows
                if row["scope"] == scope_name and row["projection"] == projection
            }
            if counts:
                overlay_consistent = overlay_consistent and len(counts) == 1 and next(iter(counts), 0) > 0
    checks = {
        "expected_image_count": len(png_files) == (len(requested_numbers) if requested_numbers else 20),
        "all_planned_images_exist": all((pass_dir / row["file"]).exists() for row in image_rows),
        "signed_rwb_backup_complete": all(
            row["signed_rwb_backup_file"] is None or (pass_dir / row["signed_rwb_backup_file"]).exists()
            for row in image_rows
        ) and len(backup_png_files) == sum(row["signed_rwb_backup_file"] is not None for row in image_rows),
        "overlay_counts_consistent_across_backgrounds": bool(overlay_consistent),
        "step3_60_points_loaded": int(overlay_summary["step3_fracture_point_count"]) == 60,
        "configured_original_fault_surface_loaded": bool(original_fault_surface_loaded),
        "overview_and_local_have_xz_yz_dfn": all(
            int(overlay_summary[key][f"selected_segment_count_{projection.lower()}"]) > 0
            for key in ("overview_scan", "local_200m_scan")
            for projection in ("XZ", "YZ")
        ),
        "overview_and_local_patch_accounting_closed": all(
            bool(item["accounting_closed"]) for item in scan_accounting.values()
        ),
        "vtk_polygon_count_matches_dfn_csv": all(
            int(item["input_polygon_count"]) == int(overlay_summary["dfn_patch_count"])
            for item in scan_accounting.values()
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "pass_name": pass_name,
        "config_path": str(config_path.resolve()),
        "output_dir": str(pass_dir),
        "product_title": str(config.get("product_title", PRODUCT_TITLE)),
        "scope_product_titles": {
            "overview": str(config.get("overview_product_title", config.get("product_title", PRODUCT_TITLE))),
            "local_200m": str(config.get("local_product_title", PRODUCT_TITLE)),
        },
        "well_name": next(iter(scopes.values())).well_name,
        "scope_geometry": {name: geometry.summary for name, geometry in scopes.items()},
        "overlay_source": overlay_summary,
        "projection_accounting": scan_accounting,
        "display": {
            "attributes": {name: info for name, (*_, info) in attribute_scales.items()},
            "seismic_symmetric_limit": float(seis_limit),
            "formal_20_image_style": formal_display_style,
            "signed_rwb_backup_style": backup_display_style,
            "signed_rwb_backup_dir": str(backup_dir),
            "signed_rwb_backup_pngs": backup_png_files,
            "absolute_grayscale_rule": "zero_white_positive_and_negative_extremes_black_abs_q98",
        },
        "images": image_rows,
        "png_files": png_files,
        "checks": checks,
    }
    write_json(pass_dir / "section_summary.json", summary)
    print(f"[multi-background] pass={pass_name} output={pass_dir}", flush=True)
    print(f"[multi-background] pass_status={summary['status']}", flush=True)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    summary = render_all(args.config, config)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
