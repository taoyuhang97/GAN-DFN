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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
FORMAL_ROOT = SCRIPT_DIR.parent

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
from well_curved_section_common import AttributeSection, prepare_geometry, read_json, write_json


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

# 与当前太古界 taigu_supp_step9_scale_wiggle_v1 的尺度和图例口径一致：
# 颜色表示尺度，不再用颜色表示地层；地层仍由 T4/T5/T6/T7 层位线表达。
SCALE_STYLES = {
    "small": {
        "label": "小尺度裂缝",
        "color": "#0891b2",
        "linewidth": 0.55,
        "alpha": 0.46,
        "legend_color": "#9ca3af",
        "legend_linewidth": 2.0,
    },
    "medium": {
        "label": "中尺度裂缝",
        "color": "#d97706",
        "linewidth": 1.25,
        "alpha": 0.72,
        "legend_color": "#6b7280",
        "legend_linewidth": 2.6,
    },
    "large": {
        "label": "大尺度裂缝",
        "color": "#dc2626",
        "linewidth": 2.15,
        "alpha": 0.88,
        "legend_color": "#374151",
        "legend_linewidth": 2.6,
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成砂砾岩矿区分尺度波形剖面")
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
    """Draw DFN segments with the same scale-based colors as Taigu."""
    count = 0
    for scale in ("small", "medium", "large"):
        lines = [
            [
                (float(segment.h1), float(segment.z1)),
                (float(segment.h2), float(segment.z2)),
            ]
            for segment in segments
            if segment.projection == projection
            and str(getattr(segment, "scale", "")).strip().lower() == scale
        ]
        if not lines:
            continue
        style = SCALE_STYLES[scale]
        ax.add_collection(
            LineCollection(
                lines,
                colors=style["color"],
                linewidths=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                zorder=13.2,
            )
        )
        ax.add_line(
            Line2D(
                [0],
                [0],
                color=style["legend_color"],
                linewidth=float(style["legend_linewidth"]),
                label=style["label"],
            )
        )
        count += len(lines)
    return count


def main() -> int:
    args = parser().parse_args()
    config = read_json(args.config.resolve())
    reference = read_json(args.reference_summary.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_config = dict(config)
    overview_config["target_block"] = dict(config.get("overview_target_block") or {})
    overview = prepare_geometry(overview_config)
    overview.summary["scope_name"] = "overview"
    reference_geometry = dict(reference["scope_geometry"]["overview"])
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
        "XZ": set_display_bounds(
            mb.decimate_section(xz, max_xz, time_decim), reference_geometry
        ),
        "YZ": set_display_bounds(
            mb.decimate_section(yz, max_yz, time_decim), reference_geometry
        ),
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
    display = {
        "display_x_min": float(overview.summary["display_x_min"]),
        "display_x_max": float(overview.summary["display_x_max"]),
        "display_y_min": float(overview.summary["display_y_min"]),
        "display_y_max": float(overview.summary["display_y_max"]),
    }
    faults, fault_scan = scan_unified_original_fault_intersections(
        scan_args.input_vtk, overview.well_df, display
    )
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
                scope_label="矿区尺度",
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
        "trace_counts_match_reference": {
            row["projection"]: row["wiggle_trace_count"] for row in images
        }
        == {"XZ": int(target.get("max_traces_xz", 510)), "YZ": int(target.get("max_traces_yz", 366))},
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "region": "砂砾岩",
        "well_name": overview.well_name,
        "config": str(args.config.resolve()),
        "reference_summary": str(args.reference_summary.resolve()),
        "output_dir": str(output_dir),
        "scope": "overview",
        "background": "OBN地震波形+正振幅变面积",
        "dfn_source": {
            "csv": str(infer_dfn_patch_csv(config)),
            "vtk": str(scan_args.input_vtk),
            "is_step8_well_corrected": True,
            "patch_count": int(len(metadata)) if metadata is not None else 0,
            "scan": scan,
        },
        "fault_source": fault_scan,
        "scale_segments": {
            scale: {
                "all": len(values),
                "XZ": sum(value.projection == "XZ" for value in values),
                "YZ": sum(value.projection == "YZ" for value in values),
            }
            for scale, values in segments_by_scale.items()
        },
        "seismic_cache": {"path": str(cache), **cache_info},
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
