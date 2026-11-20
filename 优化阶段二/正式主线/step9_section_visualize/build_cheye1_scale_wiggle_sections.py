#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""砂砾岩车页1导眼：矿区分尺度OBN波形+变面积剖面。

生成5套组合，每套XZ/YZ两图：
  1. large         原始断层 + 大尺度DFN
  2. medium        中尺度DFN
  3. small         小尺度DFN
  4. large_medium  原始断层 + 大尺度DFN + 中尺度DFN
  5. all_scales    原始断层 + 大/中/小尺度DFN

DFN使用Step8井控校正后的CSV/VTK；不叠加Step3真实裂缝或Step4井点。
背景只使用矿区尺度地震波形+正振幅变面积。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
FORMAL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FORMAL_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_window_spec = importlib.util.spec_from_file_location(
    "sand_step9_dfn_section_window",
    REPO_ROOT / "common" / "step9_dfn_section_window.py",
)
if _window_spec is None or _window_spec.loader is None:
    raise ImportError("cannot load DFN section-window helper")
section_window = importlib.util.module_from_spec(_window_spec)
_window_spec.loader.exec_module(section_window)

import build_cheye1_dfn_multibackground_sections as mb
from build_cheye1_dfn_coherence_sections import (
    add_fault_trace_segments,
    build_namespace,
    infer_dfn_patch_csv,
    load_dfn_patch_metadata,
    scan_unified_original_fault_intersections,
    scan_vtk_intersections,
)
from build_well_seismic_amplitude_sections import plot_wiggle_variable_area
from well_curved_section_common import (
    AttributeSection,
    prepare_geometry,
    read_json,
    sample_volume_sections,
    save_section_pair_npz,
    write_json,
)
_layer_spec = importlib.util.spec_from_file_location(
    "sand_layered_section_profile",
    REPO_ROOT / "common" / "layered_section_profile.py",
)
if _layer_spec is None or _layer_spec.loader is None:
    raise ImportError("cannot load layered section profile helper")
layered_profile = importlib.util.module_from_spec(_layer_spec)
_layer_spec.loader.exec_module(layered_profile)


DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "formal_mine_multiscale_flow_v1.json"
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
    / "scale_wiggle_t4_t7_v1"
)


@dataclass(frozen=True)
class ScaleSet:
    key: str
    label: str
    scales: tuple[str, ...]
    include_faults: bool
    number: int


SCALE_SETS = (
    ScaleSet("large", "大尺度", ("large",), True, 1),
    ScaleSet("medium", "中尺度", ("medium",), False, 3),
    ScaleSet("small", "小尺度", ("small",), False, 5),
    ScaleSet("large_medium", "大尺度+中尺度", ("large", "medium"), True, 7),
    ScaleSet("all_scales", "大中小尺度", ("large", "medium", "small"), True, 9),
)

# Color encodes formation; width encodes fracture scale.
SCALE_STYLES = {
    "small": {
        "label": "小尺度裂缝",
        "linewidth": 1.10,
        "alpha": 0.72,
    },
    "medium": {
        "label": "中尺度裂缝",
        "linewidth": 2.20,
        "alpha": 0.92,
    },
    "large": {
        "label": "大尺度裂缝",
        "linewidth": 3.20,
        "alpha": 0.98,
    },
}

LARGE_INTERVAL_STYLES = {
    "T4->T6": {
        "label": "沙三段（T4–T6）",
        "color": "#ff7a00",
    },
    "T6->T7": {
        "label": "沙四段（T6–T7）",
        "color": "#00a6ff",
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成砂砾岩矿区分尺度波形剖面")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dfn-patch-csv", type=Path)
    p.add_argument("--input-vtk", type=Path)
    p.add_argument("--section-window-m", type=float)
    p.add_argument("--dfn-time-padding-ms", type=float)
    p.add_argument("--section-time-mode", choices=["dfn_extent_plus_padding"])
    p.add_argument("--max-traces", type=int)
    p.add_argument("--layer-output-dir", type=Path)
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


def make_set_drawer(
    segments: list[Any],
    faults: list[Any],
    *,
    include_faults: bool,
) -> Callable[[Any, str], dict[str, int]]:
    def draw(ax, projection: str) -> dict[str, int]:
        dfn_count = add_scale_colored_dfn_segments(ax, segments, projection)
        fault_count = 0
        if include_faults:
            fault_count = add_fault_trace_segments(
                ax, faults, projection, overview=True
            )
        return {
            "dfn_segment_count": int(dfn_count),
            "fault_segment_count": int(fault_count),
            "imaging_patch_count": 0,
            "imaging_point_count": 0,
        }

    return draw


def add_scale_colored_dfn_segments(
    ax,
    segments: list[Any],
    projection: str,
) -> int:
    """Draw formation colors with scale-specific line widths."""
    count = 0
    present_scales: set[str] = set()
    present_intervals: set[str] = set()
    visible_segments = [segment for segment in segments if segment.projection == projection]
    for layer_interval, layer_style in LARGE_INTERVAL_STYLES.items():
        interval_segments = [
            segment for segment in visible_segments
            if str(getattr(segment, "interval", "")).strip() == layer_interval
        ]
        if not interval_segments:
            continue
        present_intervals.add(layer_interval)
        for scale in ("small", "medium", "large"):
            scale_segments = [
                segment for segment in interval_segments
                if str(getattr(segment, "scale", "")).strip().lower() == scale
            ]
            if not scale_segments:
                continue
            present_scales.add(scale)
            lines = [
                [(float(segment.h1), float(segment.z1)), (float(segment.h2), float(segment.z2))]
                for segment in scale_segments
            ]
            scale_style = SCALE_STYLES[scale]
            ax.add_collection(
                LineCollection(
                    lines,
                    colors=layer_style["color"],
                    linewidths=float(scale_style["linewidth"]),
                    alpha=float(scale_style["alpha"]),
                    zorder=13.2,
                )
            )
            count += len(lines)

    for interval in ("T4->T6", "T6->T7"):
        if interval in present_intervals:
            style = LARGE_INTERVAL_STYLES[interval]
            ax.add_line(
                Line2D([0], [0], color=style["color"], linewidth=3.2, label=style["label"])
            )
    for scale in ("small", "medium", "large"):
        if scale in present_scales:
            style = SCALE_STYLES[scale]
            ax.add_line(
                Line2D(
                    [0],
                    [0],
                    color="#111827",
                    linewidth=float(style["linewidth"]),
                    label=style["label"],
                )
            )
    return count


def main() -> int:
    args = parser().parse_args()
    config = read_json(args.config.resolve())
    if args.dfn_patch_csv:
        config["dfn_patch_csv"] = str(args.dfn_patch_csv.resolve())
    if args.input_vtk:
        config["input_vtk"] = str(args.input_vtk.resolve())
    if args.section_window_m is not None:
        config["section_window_m"] = float(args.section_window_m)
    if args.dfn_time_padding_ms is not None:
        config["dfn_time_padding_ms"] = float(args.dfn_time_padding_ms)
    if args.section_time_mode:
        config["section_time_mode"] = str(args.section_time_mode)
    if args.max_traces is not None:
        targets = dict(config.get("presentation_targets") or {})
        overview_target = dict(targets.get("overview") or {})
        overview_target["max_traces_xz"] = int(args.max_traces)
        overview_target["max_traces_yz"] = int(args.max_traces)
        targets["overview"] = overview_target
        config["presentation_targets"] = targets
    reference = read_json(args.reference_summary.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_config = dict(config)
    overview_config["target_block"] = dict(config.get("overview_target_block") or {})
    initial_overview = prepare_geometry(overview_config)
    section_window_m = float(config.get("section_window_m", 0.0) or 0.0)
    section_window_km = section_window_m / 1000.0
    section_window_tag = f"{section_window_km:g}km"
    section_extent_summary: dict[str, Any] | None = None
    if section_window_m > 0.0:
        block = section_window.centered_window(
            initial_overview.well_df,
            section_window_m,
            outer_bounds=dict(config.get("target_block") or {}),
            fit_to_outer_bounds=True,
            name=f"well_centered_{section_window_m:.0f}m",
        )
        config["target_block"] = block
        config["overview_target_block"] = block
        overview_config["target_block"] = block
        overview_config["overview_target_block"] = block
        overview = prepare_geometry(overview_config)
    else:
        overview = initial_overview
    overview.summary["scope_name"] = "overview"
    reference_geometry = dict(reference["scope_geometry"]["overview"])
    section_time_mode = str(config.get("section_time_mode", "")).strip()
    dfn_time_mode = section_time_mode == "dfn_extent_plus_padding"
    if dfn_time_mode:
        dfn_top, dfn_bottom, section_extent_summary = section_window.dfn_time_extent_in_block(
            infer_dfn_patch_csv(config),
            dict(config["overview_target_block"]),
            padding_ms=float(config.get("dfn_time_padding_ms", 20.0)),
            sample_interval_ms=2.0,
        )
        overview.time_min = float(dfn_top)
        overview.time_max = float(dfn_bottom)
        overview.summary["display_time_min"] = float(dfn_top)
        overview.summary["display_time_max"] = float(dfn_bottom)
        overview.summary["section_time_mode"] = section_time_mode
        overview.summary["dfn_extent"] = section_extent_summary
        config["section_scope_label"] = f"{section_window_tag}区域"
        config["figure_size_cm"] = [28.0, 12.0]
        config["include_title"] = False
        config["separate_legend"] = True
        overview.config.update({
            "figure_size_cm": [28.0, 12.0],
            "include_title": False,
            "separate_legend": True,
        })
    else:
        for key in (
            "display_x_min",
            "display_x_max",
            "display_y_min",
            "display_y_max",
            "display_time_min",
            "display_time_max",
            "_curve_extents",
            "_attribute_depth_bounds",
        ):
            if key in reference_geometry:
                overview.summary[key] = reference_geometry[key]

    target = dict(config["presentation_targets"]["overview"])
    ppt_geometry = mb.make_ppt_geometry(overview, target)
    if dfn_time_mode:
        cache = (
            output_dir
            / "section_samples"
            / f"seisamp_{section_window_tag}_dfn_pad{float(config.get('dfn_time_padding_ms', 20.0)):g}ms.npz"
        )
        if cache.exists():
            xz, yz, cache_info = mb.load_section_pair_npz(cache, "SeisAmp")
            axis_ok = (
                len(xz.h) == len(overview.x_values)
                and len(yz.h) == len(overview.y_values)
                and np.allclose(xz.h, overview.x_values, atol=1.0e-6)
                and np.allclose(yz.h, overview.y_values, atol=1.0e-6)
            )
            time_ok = (
                len(xz.time) > 1
                and abs(float(xz.time[0]) - float(overview.time_min)) <= 1.0e-6
                and abs(float(xz.time[-1]) - float(overview.time_max)) <= 1.0e-6
            )
            if not axis_ok or not time_ok:
                cache.unlink()
                xz, yz, cache_info = sample_volume_sections(
                    overview,
                    "SeisAmp",
                    Path(str(config["volume_paths"]["SeisAmp"])).resolve(),
                )
                save_section_pair_npz(cache, xz, yz)
                cache_info["source"] = f"fresh_{section_window_tag}_dfn_window_sampling"
                cache_info["cache_path"] = str(cache)
            else:
                cache_info["source"] = f"reused_{section_window_tag}_dfn_window_cache"
                cache_info["cache_path"] = str(cache)
        else:
            xz, yz, cache_info = sample_volume_sections(
                overview,
                "SeisAmp",
                Path(str(config["volume_paths"]["SeisAmp"])).resolve(),
            )
            save_section_pair_npz(cache, xz, yz)
            cache_info["source"] = f"fresh_{section_window_tag}_dfn_window_sampling"
            cache_info["cache_path"] = str(cache)
    else:
        cache = (
            Path(str(config["output_dir"])).resolve()
            / "section_samples"
            / "overview"
            / "seisamp_section_samples.npz"
        )
        if not cache.exists():
            raise FileNotFoundError(f"缺少矿区OBN波形缓存: {cache}")
        xz, yz, cache_info = mb.load_section_pair_npz(cache, "SeisAmp")
    max_xz = int(target.get("max_traces_xz", target.get("max_traces", 0)))
    max_yz = int(target.get("max_traces_yz", target.get("max_traces", 0)))
    time_decim = int(target.get("time_decim", 1))
    sections = {
        "XZ": mb.decimate_section(xz, max_xz, time_decim),
        "YZ": mb.decimate_section(yz, max_yz, time_decim),
    }
    if dfn_time_mode:
        for section in sections.values():
            section.display_time_min = float(overview.time_min)
            section.display_time_max = float(overview.time_max)
    else:
        sections = {
            "XZ": set_display_bounds(sections["XZ"], reference_geometry),
            "YZ": set_display_bounds(sections["YZ"], reference_geometry),
        }

    print(
        "[scale-wiggle-sand] 扫描Step8矿区DFN"
        "（一次扫描，按大/中/小尺度着色）",
        flush=True,
    )
    scan_args = build_namespace(
        config, float(config.get("dfn_half_width_m", 50.0))
    )
    metadata = load_dfn_patch_metadata(infer_dfn_patch_csv(config))
    scan_args.dfn_patch_metadata = metadata
    scan_args.well_trajectory_csv = overview.well_path
    scan_args.selected_well_name = overview.well_name
    from common.horizon_trace_table.horizon_contract import build_spatial_lookup

    surfaces = build_spatial_lookup(config)
    all_segments, scan = scan_vtk_intersections(
        scan_args, surfaces, overview.well_df
    )
    all_segments_before_filter = len(all_segments)
    display = {
        "display_x_min": float(overview.summary["display_x_min"]),
        "display_x_max": float(overview.summary["display_x_max"]),
        "display_y_min": float(overview.summary["display_y_min"]),
        "display_y_max": float(overview.summary["display_y_max"]),
    }
    faults, fault_scan = scan_unified_original_fault_intersections(
        scan_args.input_vtk, overview.well_df, display
    )
    faults_before_filter = len(faults)
    segments_by_scale = {
        scale: [
            segment
            for segment in all_segments
            if str(getattr(segment, "scale", "")).strip().lower() == scale
        ]
        for scale in ("small", "medium", "large")
    }
    for scale, values in segments_by_scale.items():
        print(
            f"[scale-wiggle-sand] {scale} segments={len(values)} "
            f"XZ={sum(v.projection == 'XZ' for v in values)} "
            f"YZ={sum(v.projection == 'YZ' for v in values)}",
            flush=True,
        )

    layer_manifest: dict[str, Any] | None = None
    if args.layer_output_dir:
        layer_dir = args.layer_output_dir.resolve()
        curves: dict[str, dict[str, np.ndarray]] = {}
        for curve in overview.surface_curves:
            curves.setdefault(curve.projection, {})[curve.surface] = np.asarray(
                curve.z,
                dtype=np.float64,
            )
        base_manifest = layered_profile.write_base_layer(
            layer_dir,
            region="砂砾岩",
            xz_section=sections["XZ"],
            yz_section=sections["YZ"],
            curves=curves,
            track=overview.well_df,
            faults={
                projection: [
                    segment
                    for segment in faults
                    if segment.projection == projection
                ]
                for projection in ("XZ", "YZ")
            },
            metadata={
                "config": str(args.config.resolve()),
                "scope": f"overview_{section_window_tag}",
                "section_horizontal_window_m": float(section_window_m),
                "section_block": dict(config["overview_target_block"]),
                "display_time_ms": [
                    float(sections["XZ"].display_time_min),
                    float(sections["XZ"].display_time_max),
                ],
            },
        )
        intersection_manifest = layered_profile.write_intersection_layer(
            layer_dir,
            layered_profile.projection_segment_rows(
                all_segments,
                region="砂砾岩",
                layer_by_interval={
                    "T4->T6": "沙三段",
                    "T6->T7": "沙四段",
                },
            ),
            metadata={
                "source": "Step8_VTK_polygon_plane_intersection",
                "visible_filter_enabled": False,
                "segments_before_filter": int(all_segments_before_filter),
                "segments_after_filter": int(len(all_segments)),
            },
        )
        base_drawer = lambda ax, projection: {
            "dfn_segment_count": 0,
            "fault_segment_count": int(
                add_fault_trace_segments(
                    ax,
                    faults,
                    projection,
                    overview=True,
                )
            ),
            "imaging_patch_count": 0,
            "imaging_point_count": 0,
        }
        for projection in ("XZ", "YZ"):
            plot_wiggle_variable_area(
                sections[projection],
                ppt_geometry,
                layer_dir
                / "base"
                / f"base_{projection.lower()}_t4_t7.png",
                float(reference["display"]["seismic_symmetric_limit"]),
                overlay_drawer=base_drawer,
                scope_label=str(config.get("section_scope_label", "矿区尺度")),
                product_title="车页1导眼：变面积剖面底图",
            )
        layer_manifest = {
            "directory": str(layer_dir),
            "base": base_manifest,
            "intersections": intersection_manifest,
        }

    limit = float(reference["display"]["seismic_symmetric_limit"])
    images: list[dict[str, Any]] = []
    for scale_set in SCALE_SETS:
        selected_segments = [
            segment
            for scale in scale_set.scales
            for segment in segments_by_scale[scale]
        ]
        drawer = make_set_drawer(
            selected_segments,
            faults,
            include_faults=scale_set.include_faults,
        )
        for offset, projection in enumerate(("XZ", "YZ")):
            number = scale_set.number + offset
            filename = (
                f"{number:02d}_{scale_set.key}_{projection.lower()}_t4_t7.png"
            )
            output_path = output_dir / filename
            captured: dict[str, int] = {}

            def capture(ax, current_projection: str) -> dict[str, int]:
                captured.update(drawer(ax, current_projection))
                return dict(captured)

            trace_count = plot_wiggle_variable_area(
                sections[projection],
                ppt_geometry,
                output_path,
                limit,
                overlay_drawer=capture,
                scope_label=str(config.get("section_scope_label", "矿区尺度")),
                product_title=f"车页1导眼：{scale_set.label}DFN",
            )
            images.append(
                {
                    "number": number,
                    "set": scale_set.key,
                    "set_label": scale_set.label,
                    "scales": list(scale_set.scales),
                    "include_original_faults": scale_set.include_faults,
                    "projection": projection,
                    "path": str(output_path),
                    "section_shape": [
                        int(value) for value in sections[projection].values.shape
                    ],
                    "wiggle_trace_count": int(trace_count),
                    "overlay": dict(captured),
                }
            )
            print(f"[scale-wiggle-sand] {filename}", flush=True)

    expected_images = len(SCALE_SETS) * 2
    checks = {
        "image_count_expected_10": len(images) == expected_images,
        "all_images_exist": all(Path(row["path"]).exists() for row in images),
        "dfn_has_all_three_scales": all(
            len(segments_by_scale[scale]) > 0
            for scale in ("small", "medium", "large")
        ),
        "large_sets_include_faults": all(
            row["overlay"]["fault_segment_count"] > 0
            for row in images
            if row["set"] in ("large", "large_medium", "all_scales")
        ),
        "medium_small_only_have_no_faults": all(
            row["overlay"]["fault_segment_count"] == 0
            for row in images
            if row["set"] in ("medium", "small")
        ),
        "no_real_imaging_fractures": all(
            row["overlay"]["imaging_patch_count"] == 0
            and row["overlay"]["imaging_point_count"] == 0
            for row in images
        ),
        "trace_counts_match_expected": {
            row["projection"]: row["wiggle_trace_count"] for row in images
        }
        == (
            {
                projection: min(
                    int(target.get(f"max_traces_{projection.lower()}", target.get("max_traces", 0))),
                    len(sections[projection].h),
                )
                for projection in ("XZ", "YZ")
            }
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "region": "砂砾岩",
        "well_name": overview.well_name,
        "config": str(args.config.resolve()),
        "reference_summary": str(args.reference_summary.resolve()),
        "output_dir": str(output_dir),
        "scope": f"overview_{section_window_tag}" if section_window_m > 0.0 else "overview",
        "section_horizontal_window": {
            "size_m": float(section_window_m),
            "block": dict(config.get("overview_target_block") or {}),
        },
        "section_time_extent": section_extent_summary,
        "figure_layout": {
            "size_cm": [28.0, 12.0],
            "includes_axes": True,
            "includes_title": False,
            "separate_legend": True,
        },
        "background": "OBN地震波形+正振幅变面积",
        "color_semantics": {
            "color": {
                "T4->T6": "sandstone_3_orange",
                "T6->T7": "sandstone_4_blue",
            },
            "linewidth": {
                "small": 1.10,
                "medium": 2.20,
                "large": 3.20,
            },
        },
        "dfn_source": {
            "csv": str(infer_dfn_patch_csv(config)),
            "vtk": str(scan_args.input_vtk),
            "is_step8_well_corrected": True,
            "patch_count": int(len(metadata)) if metadata is not None else 0,
            "scan": scan,
            "visible_filter_enabled": False,
            "segments_before_filter": int(all_segments_before_filter),
            "segments_after_filter": int(len(all_segments)),
            "segments_suppressed": int(all_segments_before_filter - len(all_segments)),
        },
        "fault_source": {
            **fault_scan,
            "visible_filter_enabled": False,
            "segments_before_filter": int(faults_before_filter),
            "segments_after_filter": int(len(faults)),
            "segments_suppressed": int(faults_before_filter - len(faults)),
        },
        "scale_segments": {
            scale: {
                "all": len(values),
                "XZ": sum(value.projection == "XZ" for value in values),
                "YZ": sum(value.projection == "YZ" for value in values),
            }
            for scale, values in segments_by_scale.items()
        },
        "seismic_cache": {"path": str(cache), **cache_info},
        "layer_data": layer_manifest,
        "images": images,
        "checks": checks,
    }
    summary_path = output_dir / "scale_wiggle_summary.json"
    write_json(summary_path, summary)
    print(
        f"[scale-wiggle-sand] status={summary['status']} summary={summary_path}",
        flush=True,
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
