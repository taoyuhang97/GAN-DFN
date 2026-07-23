# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from build_all_area_section_visualization import load_surface_lookups
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
    scan_fault_surface_csv_intersections,
    scan_original_fault_stick_traces,
    scan_vtk_intersections,
    validate_inputs,
)
from build_well_geological_attribute_sections import display_scale, plot_section
from build_well_seismic_amplitude_sections import amplitude_limit, plot_variable_density, plot_wiggle_variable_area
from dfn_section_overlay import SectionOverlayContext, draw_section_overlays
from trace_horizon_section import build_trace_horizon_section_curves, resolve_horizon_trace_table
from well_curved_section_common import (
    CurvedSectionGeometry,
    prepare_geometry,
    read_json,
    sample_volume_sections,
    save_section_pair_npz,
    write_json,
)


ATTRIBUTE_ORDER = ["AntTrack", "Coherence", "CurvatureMax"]
PRODUCT_TITLE = "车页1导眼：DFN与成像测井裂缝对比剖面"


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
    return replace(
        base,
        args=local_args,
        surface_curves=curves,
        summary=summary,
        x_values=x_values,
        y_values=y_values,
        time_min=float(time_min),
        time_max=float(time_max),
    )


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
    surfaces = load_surface_lookups(standard_args.surface_dir)
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
    fault_dat = path_from_config(config, "original_fault_stick_dat") if config.get("original_fault_stick_dat") else None
    fault_csv = path_from_config(config, "fault_surface_csv") if config.get("fault_surface_csv") else None
    if fault_dat is not None:
        standard_faults, standard_fault_scan = scan_original_fault_stick_traces(
            fault_dat,
            dict(config.get("target_block") or {}),
            overview.well_df,
            float(config.get("fault_trace_half_width_m", standard_args.half_width)),
            overview_display,
            context_padding_m=float(config.get("original_fault_context_padding_m", 10000.0)),
        )
        local_faults, local_fault_scan = scan_original_fault_stick_traces(
            fault_dat,
            dict(config.get("target_block") or {}),
            overview.well_df,
            float(config.get("local_fault_trace_half_width_m", local_args.half_width)),
            local_display,
            context_padding_m=float(config.get("original_fault_context_padding_m", 10000.0)),
        )
    else:
        standard_faults, standard_fault_scan = scan_fault_surface_csv_intersections(
            fault_csv, surfaces, overview.well_df, float(config.get("fault_trace_half_width_m", standard_args.half_width))
        )
        local_faults, local_fault_scan = scan_fault_surface_csv_intersections(
            fault_csv, surfaces, overview.well_df, float(config.get("local_fault_trace_half_width_m", local_args.half_width))
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
    output_dir = path_from_config(config, "output_dir")
    reset_output_images(output_dir)
    overview = prepare_geometry(config)
    overview.summary["scope_name"] = "overview"
    local = local_geometry(overview, config)
    scopes = {"overview": overview, "local_200m": local}
    scope_labels = {"overview": "候选区整体", "local_200m": "井周200 m"}
    overlay_contexts, overlay_summary = build_overlay_contexts(config, overview, local)

    sampled: dict[str, dict[str, tuple[Any, Any, dict[str, Any]]]] = {}
    for scope_name, geometry in scopes.items():
        sampled[scope_name] = {}
        for attribute in [*ATTRIBUTE_ORDER, "SeisAmp"]:
            print(f"[multi-background] sampling scope={scope_name} attribute={attribute}", flush=True)
            sampled[scope_name][attribute] = sample_volume_sections(
                geometry,
                attribute,
                Path(str(config["volume_paths"][attribute])).resolve(),
            )
            xz, yz, _ = sampled[scope_name][attribute]
            cache_dir = output_dir / "section_samples" / scope_name
            save_section_pair_npz(cache_dir / f"{attribute.lower()}_section_samples.npz", xz, yz)

    attribute_scales: dict[str, tuple[float, float, Any, dict[str, Any]]] = {}
    for attribute in ATTRIBUTE_ORDER:
        arrays = [
            sampled[scope][attribute][projection].values
            for scope in scopes
            for projection in (0, 1)
        ]
        attribute_scales[attribute] = display_scale(attribute, arrays)
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
    image_rows: list[dict[str, Any]] = []
    mode_names = {"attribute": "dfn", "density": "density_dfn", "wiggle": "wiggle_dfn"}
    for number, scope_name, attribute, projection, renderer in image_plan:
        section_index = 0 if projection == "XZ" else 1
        section = sampled[scope_name][attribute][section_index]
        geometry = scopes[scope_name]
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
                overlay_drawer=drawer,
                scope_label=scope_labels[scope_name],
                product_title=PRODUCT_TITLE,
            )
        elif renderer == "density":
            stats = plot_variable_density(
                section,
                geometry,
                output_path,
                seis_limit,
                overlay_drawer=drawer,
                scope_label=scope_labels[scope_name],
                product_title=PRODUCT_TITLE,
            )
        else:
            trace_count = plot_wiggle_variable_area(
                section,
                geometry,
                output_path,
                seis_limit,
                overlay_drawer=drawer,
                scope_label=scope_labels[scope_name],
                product_title=PRODUCT_TITLE,
            )
            stats = dict(captured)
            stats["wiggle_trace_count"] = int(trace_count)
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
            }
        )
        print(f"[multi-background] rendered {filename}", flush=True)

    png_files = sorted(path.name for path in output_dir.glob("*.png"))
    overlay_consistent = True
    for scope_name in scopes:
        for projection in ("XZ", "YZ"):
            counts = {
                int(row["overlay"].get("dfn_segment_count", -1))
                for row in image_rows
                if row["scope"] == scope_name and row["projection"] == projection
            }
            overlay_consistent = overlay_consistent and len(counts) == 1 and next(iter(counts), 0) > 0
    checks = {
        "exactly_20_pngs": len(png_files) == 20,
        "all_planned_images_exist": all((output_dir / row["file"]).exists() for row in image_rows),
        "overlay_counts_consistent_across_backgrounds": bool(overlay_consistent),
        "step3_60_points_loaded": int(overlay_summary["step3_fracture_point_count"]) == 60,
        "overview_and_local_have_xz_yz_dfn": all(
            int(overlay_summary[key][f"selected_segment_count_{projection.lower()}"]) > 0
            for key in ("overview_scan", "local_200m_scan")
            for projection in ("XZ", "YZ")
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir),
        "product_title": PRODUCT_TITLE,
        "well_name": overview.well_name,
        "scope_geometry": {name: geometry.summary for name, geometry in scopes.items()},
        "overlay_source": overlay_summary,
        "display": {
            "attributes": {name: info for name, (*_, info) in attribute_scales.items()},
            "seismic_symmetric_limit": float(seis_limit),
        },
        "images": image_rows,
        "png_files": png_files,
        "checks": checks,
    }
    write_json(output_dir / "section_summary.json", summary)
    print(f"[multi-background] output={output_dir}", flush=True)
    print(f"[multi-background] status={summary['status']}", flush=True)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    summary = render_all(args.config, config)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
