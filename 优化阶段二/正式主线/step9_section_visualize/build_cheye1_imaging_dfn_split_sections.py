#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""砂砾岩车页1导眼：将 Step3 真实裂缝与 DFN 预测拆成独立剖面图。

沿用 ``formal_mine_multiscale_flow_v1`` 与 ``mine_v1/ppt_decimated`` 的
局部200 m展示口径：

* 地震波形+正振幅变面积背景；
* T4-T7层位、车页1导眼井轨迹；
* XZ 82道、YZ 98道，时间抽稀2倍；
* 两张图使用完全相同的剖面几何和坐标范围。

真实裂缝图只叠加 Step3 成像井段轨迹、真实裂缝解释片和原始裂缝点；
DFN图只叠加成像井段轨迹和三尺度DFN裂缝，不叠加真实裂缝点/解释片。

用法::

    /home/tyh/anaconda3/envs/gan-dfn/bin/python3 \
      优化阶段二/正式主线/step9_section_visualize/build_cheye1_imaging_dfn_split_sections.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
FORMAL_ROOT = SCRIPT_DIR.parent

import build_cheye1_dfn_multibackground_sections as mb
from build_cheye1_dfn_coherence_sections import (
    add_dfn_segments,
    add_imaging_fracture_patch_segments,
    build_imaging_fracture_patch_segments,
    build_namespace,
    draw_fracture_labels,
    draw_imaging_segment_trajectory,
    infer_dfn_patch_csv,
    load_dfn_patch_metadata,
    load_step3_imaging_fracture_labels,
    load_step3_imaging_segment_track,
    scan_vtk_intersections,
)
from build_well_seismic_amplitude_sections import plot_wiggle_variable_area
from dfn_section_overlay import SectionOverlayContext
from well_curved_section_common import AttributeSection, prepare_geometry, read_json, write_json


DEFAULT_CONFIG = (
    SCRIPT_DIR / "configs" / "formal_mine_multiscale_flow_v1.json"
)
DEFAULT_REFERENCE_SUMMARY = (
    FORMAL_ROOT
    / "output"
    / "step9_section_presentation"
    / "mine_v1"
    / "ppt_decimated"
    / "section_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    FORMAL_ROOT
    / "output"
    / "step9_section_presentation"
    / "mine_v1"
    / "split_imaging_dfn_v1"
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成砂砾岩真实裂缝/DFN独立剖面图")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p


def set_display_bounds(
    section: AttributeSection,
    reference_geometry: dict[str, Any],
) -> AttributeSection:
    bounds = dict(reference_geometry.get("_attribute_depth_bounds") or {})
    if section.projection in bounds:
        section.display_time_min = float(bounds[section.projection][0])
        section.display_time_max = float(bounds[section.projection][1])
    else:
        section.display_time_min = float(reference_geometry["display_time_min"])
        section.display_time_max = float(reference_geometry["display_time_max"])
    return section


def build_local_context(
    config: dict[str, Any],
    geometry,
) -> tuple[SectionOverlayContext, dict[str, Any]]:
    """扫描一次局部200 m DFN；不扫描/绘制原始断层。"""
    args = build_namespace(config, float(config.get("local_dfn_half_width_m", 200.0)))
    args.small_projection_half_width = float(
        config.get(
            "local_small_projection_half_width_m",
            config.get("small_projection_half_width_m", args.half_width),
        )
    )
    patch_csv = infer_dfn_patch_csv(config)
    metadata = load_dfn_patch_metadata(patch_csv)
    args.dfn_patch_metadata = metadata
    args.well_trajectory_csv = geometry.well_path
    args.selected_well_name = geometry.well_name

    from common.horizon_trace_table.horizon_contract import build_spatial_lookup

    surfaces = build_spatial_lookup(config)
    segments, scan = scan_vtk_intersections(args, surfaces, geometry.well_df)
    group_paths = [
        Path(str(value)).resolve()
        for value in config.get("step3_imaging_group_csvs", [])
    ]
    fracture_df = load_step3_imaging_fracture_labels(
        group_paths,
        geometry.well_name,
        dict(config.get("target_block") or {}),
    )
    imaging_df = load_step3_imaging_segment_track(
        group_paths,
        geometry.well_name,
        dict(config.get("target_block") or {}),
    )
    imaging_patches = build_imaging_fracture_patch_segments(fracture_df)
    context = SectionOverlayContext(
        "local_200m",
        segments,
        [],
        imaging_patches,
        fracture_df,
        imaging_df,
        geometry.summary,
    )
    summary = {
        "dfn_patch_csv": str(patch_csv) if patch_csv else None,
        "dfn_patch_count": int(len(metadata)) if metadata is not None else 0,
        "dfn_scan": scan,
        "step3_group_csvs": [str(path) for path in group_paths],
        "step3_fracture_point_count": int(len(fracture_df)),
        "step3_imaging_track_sample_count": int(len(imaging_df)),
        "step3_imaging_patch_segment_count": int(len(imaging_patches)),
    }
    return context, summary


def make_real_drawer(
    context: SectionOverlayContext,
) -> Callable[[Any, str], dict[str, int]]:
    def draw(ax, projection: str) -> dict[str, int]:
        draw_imaging_segment_trajectory(ax, context.imaging_df, projection)
        patch_count = add_imaging_fracture_patch_segments(
            ax, context.imaging_patch_segments, projection
        )
        point_count = draw_fracture_labels(
            ax, context.fracture_df, projection, context.display_summary
        )
        return {
            "imaging_patch_count": int(patch_count),
            "imaging_point_count": int(point_count),
            "dfn_segment_count": 0,
            "fault_segment_count": 0,
        }

    return draw


def make_dfn_drawer(
    context: SectionOverlayContext,
) -> Callable[[Any, str], dict[str, int]]:
    def draw(ax, projection: str) -> dict[str, int]:
        draw_imaging_segment_trajectory(ax, context.imaging_df, projection)
        segment_count = add_dfn_segments(
            ax, context.dfn_segments, projection, overlay=True
        )
        return {
            "dfn_segment_count": int(segment_count),
            "imaging_patch_count": 0,
            "imaging_point_count": 0,
            "fault_segment_count": 0,
        }

    return draw


def render_projection(
    output_dir: Path,
    kind: str,
    projection: str,
    section: AttributeSection,
    geometry,
    limit: float,
    drawer: Callable[[Any, str], dict[str, int]],
) -> tuple[Path, int, dict[str, int]]:
    if kind == "imaging":
        filename = (
            f"01_local200m_seisamp_wiggle_imaging_only_"
            f"{projection.lower()}_t4_t7.png"
        )
        product_title = "车页1导眼：仅Step3真实成像裂缝"
        title_suffix = " | 仅真实裂缝"
    elif kind == "dfn":
        filename = (
            f"02_local200m_seisamp_wiggle_dfn_only_"
            f"{projection.lower()}_t4_t7.png"
        )
        product_title = "车页1导眼：仅DFN预测裂缝"
        title_suffix = " | 仅DFN裂缝"
    else:
        raise ValueError(f"unsupported kind: {kind}")
    output_path = output_dir / filename
    captured: dict[str, int] = {}

    def capture(ax, current_projection: str) -> dict[str, int]:
        captured.update(drawer(ax, current_projection))
        return dict(captured)

    trace_count = plot_wiggle_variable_area(
        section,
        geometry,
        output_path,
        limit,
        title_suffix=title_suffix,
        overlay_drawer=capture,
        scope_label="井周200 m",
        product_title=product_title,
    )
    return output_path, int(trace_count), dict(captured)


def main() -> int:
    args = parser().parse_args()
    config = read_json(args.config.resolve())
    reference = read_json(args.reference_summary.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_config = dict(config)
    overview_config["target_block"] = dict(config.get("overview_target_block") or {})
    overview = prepare_geometry(overview_config)
    local = mb.local_geometry(overview, config)

    reference_geometry = dict(reference["scope_geometry"]["local_200m"])
    for key in (
        "display_x_min",
        "display_x_max",
        "display_y_min",
        "display_y_max",
        "display_time_min",
        "display_time_max",
        "_curve_extents",
        "_attribute_depth_bounds",
        "section_time_max_override_ms",
    ):
        if key in reference_geometry:
            local.summary[key] = reference_geometry[key]

    target = dict(config["presentation_targets"]["local_200m"])
    ppt_geometry = mb.make_ppt_geometry(local, target)
    cache = (
        Path(str(config["output_dir"])).resolve()
        / "section_samples"
        / "local_200m"
        / "seisamp_section_samples.npz"
    )
    if not cache.exists():
        raise FileNotFoundError(f"缺少 Step9 局部振幅缓存: {cache}")
    xz, yz, cache_info = mb.load_section_pair_npz(cache, "SeisAmp")
    max_traces = int(target.get("max_traces", 0))
    time_decim = int(target.get("time_decim", 1))
    sections = {
        "XZ": set_display_bounds(
            mb.decimate_section(xz, max_traces, time_decim), reference_geometry
        ),
        "YZ": set_display_bounds(
            mb.decimate_section(yz, max_traces, time_decim), reference_geometry
        ),
    }

    print("[split-sections] 扫描 local_200m DFN", flush=True)
    context, overlay_source = build_local_context(config, ppt_geometry)
    limit = float(reference["display"]["seismic_symmetric_limit"])
    drawers = {
        "imaging": make_real_drawer(context),
        "dfn": make_dfn_drawer(context),
    }
    images: list[dict[str, Any]] = []
    for kind in ("imaging", "dfn"):
        for projection in ("XZ", "YZ"):
            path, trace_count, overlay = render_projection(
                output_dir,
                kind,
                projection,
                sections[projection],
                ppt_geometry,
                limit,
                drawers[kind],
            )
            images.append(
                {
                    "kind": kind,
                    "projection": projection,
                    "path": str(path),
                    "section_shape": [int(value) for value in sections[projection].values.shape],
                    "wiggle_trace_count": int(trace_count),
                    "overlay": overlay,
                }
            )
            print(f"[split-sections] {path.name}", flush=True)

    counts = {
        f"{row['kind']}_{row['projection']}": dict(row["overlay"])
        for row in images
    }
    checks = {
        "four_images_generated": all(Path(row["path"]).exists() for row in images)
        and len(images) == 4,
        "local_trace_counts_match_reference": {
            row["projection"]: row["wiggle_trace_count"] for row in images
        }
        == {"XZ": 82, "YZ": 98},
        "imaging_only_has_no_dfn": all(
            row["overlay"]["dfn_segment_count"] == 0
            for row in images
            if row["kind"] == "imaging"
        ),
        "dfn_only_has_no_real_fracture": all(
            row["overlay"]["imaging_patch_count"] == 0
            and row["overlay"]["imaging_point_count"] == 0
            for row in images
            if row["kind"] == "dfn"
        ),
        "real_fractures_loaded": {
            row["projection"]: row["overlay"]["imaging_point_count"]
            for row in images
            if row["kind"] == "imaging"
        }
        == {"XZ": 60, "YZ": 60},
        "dfn_segments_loaded": all(
            row["overlay"]["dfn_segment_count"] > 0
            for row in images
            if row["kind"] == "dfn"
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config": str(args.config.resolve()),
        "reference_summary": str(args.reference_summary.resolve()),
        "output_dir": str(output_dir),
        "well_name": config.get("well_name", "车页1导眼"),
        "scope": "local_200m",
        "display_contract": {
            "background": "地震波形+正振幅变面积",
            "section_mode": "well_trajectory_curved_surface_t4_t7",
            "projections": ["XZ", "YZ"],
            "time_range_ms": [
                float(reference_geometry["display_time_min"]),
                float(reference_geometry["display_time_max"]),
            ],
            "presentation_target": target,
            "seismic_symmetric_limit": float(limit),
            "xz_values_shape": [int(value) for value in sections["XZ"].values.shape],
            "yz_values_shape": [int(value) for value in sections["YZ"].values.shape],
        },
        "seismic_cache": {
            "path": str(cache),
            **cache_info,
        },
        "overlay_source": overlay_source,
        "overlay_counts": counts,
        "images": images,
        "checks": checks,
    }
    summary_path = output_dir / "split_sections_summary.json"
    write_json(summary_path, summary)
    print(f"[split-sections] status={summary['status']} summary={summary_path}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
