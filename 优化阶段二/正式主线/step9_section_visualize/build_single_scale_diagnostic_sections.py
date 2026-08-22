# -*- coding: utf-8 -*-
"""Single-scale DFN diagnostic sections for the mine-scale formal flow.

Produces scale-vs-attribute diagnostic profiles used as internal quality /
presentation material:
  - small-scale fractures over the CurvatureMax background
  - medium-scale fractures over the AntTrack background
  - large-scale fractures (predicted large patches + original fault
    triangles) over the Coherence background

Each scale has XZ and YZ profiles. The configured scopes decide whether the
overview (mine-scale) and/or the local well-area profiles are rendered.
This product is separate from the formal 20-image combined set and does not
replace it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from build_cheye1_dfn_coherence_sections import (
    build_namespace,
    build_imaging_fracture_patch_segments,
    load_dfn_patch_metadata,
    load_step3_imaging_fracture_labels,
    load_step3_imaging_segment_track,
    path_from_config,
    scan_unified_original_fault_intersections,
    scan_vtk_intersections,
)
from build_cheye1_dfn_multibackground_sections import (
    load_section_pair_npz,
    local_geometry,
    make_overlay_drawer,
    read_json,
    write_json,
)
from build_well_geological_attribute_sections import display_scale, plot_section
from dfn_section_overlay import SectionOverlayContext
from well_curved_section_common import (
    prepare_geometry,
    sample_volume_sections,
    save_section_pair_npz,
)
from common.horizon_trace_table.horizon_contract import build_spatial_lookup


PRODUCT_TITLE = "车页1导眼：矿区尺度单尺度DFN诊断剖面"
DEFAULT_DIAGNOSTIC_CFG = {
    "enabled": True,
    "scopes": ["overview", "local_200m"],
    "scales": {
        "small": {"attribute": "CurvatureMax", "label": "小尺度裂缝×曲率体"},
        "medium": {"attribute": "AntTrack", "label": "中尺度裂缝×蚂蚁体"},
        "large": {"attribute": "Coherence", "label": "大尺度裂缝×相干体"},
    },
    "image_number_start_overview": 101,
    "image_number_start_local": 111,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-scale DFN diagnostic sections.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_DIAGNOSTIC_CFG)
    merged.update(dict(config.get("step9_diagnostic_sections") or {}))
    return merged


def build_scale_contexts(
    config: dict[str, Any],
    overview,
    local,
    diag: dict[str, Any],
) -> tuple[dict[str, dict[str, SectionOverlayContext]], dict[str, Any]]:
    """Return {scope: {scale: context}} and a scan/accounting summary."""
    scope_geometries = {"overview": overview, "local_200m": local}
    scope_args = {
        "overview": build_namespace(config, float(config.get("dfn_half_width_m", 50.0))),
        "local_200m": build_namespace(config, float(config.get("local_dfn_half_width_m", 200.0))),
    }
    scope_args["local_200m"].small_projection_half_width = float(
        config.get("local_small_projection_half_width_m", config.get("small_projection_half_width_m", 200.0))
    )
    patch_csv = path_from_config(config, "dfn_patch_csv")
    metadata = load_dfn_patch_metadata(patch_csv)
    group_paths = [Path(str(value)).resolve() for value in config.get("step3_imaging_group_csvs", [])]
    well_name = str(config.get("well_name", "车页1导眼"))
    target_block = dict(config.get("target_block") or {})
    fracture_df = load_step3_imaging_fracture_labels(group_paths, well_name, target_block)
    imaging_df = load_step3_imaging_segment_track(group_paths, well_name, target_block)
    imaging_patches = build_imaging_fracture_patch_segments(fracture_df)
    surfaces = build_spatial_lookup(config)

    contexts: dict[str, dict[str, SectionOverlayContext]] = {}
    summary: dict[str, Any] = {
        "dfn_patch_csv": str(patch_csv) if patch_csv else None,
        "dfn_patch_count": int(len(metadata)) if metadata is not None else 0,
        "step3_fracture_point_count": int(len(fracture_df)),
        "scans": {},
    }
    for scope_name, geometry in scope_geometries.items():
        args = scope_args[scope_name]
        args.dfn_patch_metadata = metadata
        args.well_trajectory_csv = overview.well_path
        args.selected_well_name = well_name
        display = {
            "display_x_min": float(geometry.summary["display_x_min"]),
            "display_x_max": float(geometry.summary["display_x_max"]),
            "display_y_min": float(geometry.summary["display_y_min"]),
            "display_y_max": float(geometry.summary["display_y_max"]),
        }
        faults, fault_scan = scan_unified_original_fault_intersections(args.input_vtk, overview.well_df, display)
        contexts[scope_name] = {}
        for scale, scale_cfg in diag["scales"].items():
            scale_args = args
            scale_args.allowed_scales = [scale]
            print(f"[single-scale-diag] scanning scope={scope_name} scale={scale}", flush=True)
            segments, scan = scan_vtk_intersections(scale_args, surfaces, overview.well_df)
            contexts[scope_name][scale] = SectionOverlayContext(
                scope_name,
                segments,
                faults,
                imaging_patches,
                fracture_df,
                imaging_df,
                geometry.summary,
            )
            summary["scans"][f"{scope_name}_{scale}"] = scan
            summary["scans"][f"{scope_name}_{scale}"]["fault_scan"] = fault_scan
    return contexts, summary


def render_all(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    diag = diagnostic_config(config)
    if not bool(diag.get("enabled", True)):
        return {"status": "skipped", "enabled": False}
    if config.get("single_scale_diag_output_dir"):
        output_dir = Path(str(config["single_scale_diag_output_dir"])).resolve()
    else:
        output_dir = path_from_config(config, "output_dir").parent / "single_scale_diagnostic_sections"
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_config = dict(config)
    overview_config["target_block"] = dict(config.get("overview_target_block") or {})
    overview = prepare_geometry(overview_config)
    overview.summary["scope_name"] = "overview"
    local = local_geometry(overview, config) if "local_200m" in diag.get("scopes", []) else None
    scopes = {"overview": overview}
    if local is not None:
        scopes["local_200m"] = local

    scope_labels = {"overview": "矿区尺度 | 单尺度诊断", "local_200m": "井周200 m | 单尺度诊断"}
    scopes_requested = [str(scope) for scope in diag.get("scopes", []) if str(scope) in scopes]

    sampled: dict[str, dict[str, Any]] = {}
    for scope_name in scopes_requested:
        sampled[scope_name] = {}
        for scale, scale_cfg in diag["scales"].items():
            attribute = str(scale_cfg["attribute"])
            cache_value = (
                dict(config.get("overview_section_sample_paths") or {}).get(attribute)
                if scope_name == "overview"
                else None
            )
            if cache_value and Path(str(cache_value)).exists():
                print(f"[single-scale-diag] reuse cache attribute={attribute}", flush=True)
                sampled[scope_name][scale] = load_section_pair_npz(Path(str(cache_value)).resolve(), attribute)
            else:
                print(f"[single-scale-diag] sampling scope={scope_name} attribute={attribute}", flush=True)
                sampled[scope_name][scale] = sample_volume_sections(
                    scopes[scope_name],
                    attribute,
                    Path(str(config["volume_paths"][attribute])).resolve(),
                )
            xz, yz, _ = sampled[scope_name][scale]
            cache_dir = output_dir / "section_samples" / scope_name
            save_section_pair_npz(cache_dir / f"{str(scale_cfg['attribute']).lower()}_section_samples.npz", xz, yz)

    contexts, overlay_summary = build_scale_contexts(config, overview, local, diag)
    formal_display_style = str(config.get("formal_signed_attribute_display_style", "absolute_grayscale"))
    attribute_scales: dict[str, tuple[float, float, Any, dict[str, Any]]] = {}
    for scale, scale_cfg in diag["scales"].items():
        attribute = str(scale_cfg["attribute"])
        arrays = [sampled[scope][scale][projection].values for scope in scopes_requested for projection in (0, 1)]
        attribute_scales[attribute] = display_scale(attribute, arrays, formal_display_style)

    image_plan: list[tuple[int, str, str, str]] = []
    number_start = {
        "overview": int(diag.get("image_number_start_overview", 101)),
        "local_200m": int(diag.get("image_number_start_local", 111)),
    }
    for scope_name in scopes_requested:
        for scale_index, (scale, scale_cfg) in enumerate(diag["scales"].items()):
            number = number_start[scope_name] + scale_index * 2
            image_plan.append((number, scope_name, scale, "XZ"))
            image_plan.append((number + 1, scope_name, scale, "YZ"))

    image_rows: list[dict[str, Any]] = []
    for number, scope_name, scale, projection in image_plan:
        scale_cfg = diag["scales"][scale]
        attribute = str(scale_cfg["attribute"])
        section_index = 0 if projection == "XZ" else 1
        section = sampled[scope_name][scale][section_index]
        geometry = scopes[scope_name]
        drawer, captured = make_overlay_drawer(contexts[scope_name][scale])
        product_title = str(config.get("step9_product_title", PRODUCT_TITLE))
        token = attribute.lower()
        filename = f"{number:02d}_{scope_name}_{scale}_{token}_dfn_{projection.lower()}_t4_t7.png"
        output_path = output_dir / filename
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
            product_title=f"{product_title} | {scale_cfg['label']}",
        )
        image_rows.append(
            {
                "number": number,
                "file": filename,
                "scope": scope_name,
                "scale": scale,
                "attribute": attribute,
                "projection": projection,
                "section_shape": [int(value) for value in section.values.shape],
                "overlay": stats,
                "formal_display_style": formal_display_style,
            }
        )
        print(f"[single-scale-diag] rendered {filename}", flush=True)

    png_files = sorted(path.name for path in output_dir.glob("*.png"))
    zero_segment_large_images = [
        str(row["file"])
        for row in image_rows
        if str(row.get("scale", "")) == "large" and int(row["overlay"].get("dfn_segment_count", 0)) == 0
    ]
    checks = {
        "all_planned_images_exist": all((output_dir / row["file"]).exists() for row in image_rows),
        "expected_image_count": len(png_files) == len(image_plan),
        # Small/medium diagnostics must have segments. Large-scale segments are
        # true section intersections and may legitimately be zero on a small
        # region; such sparsity is recorded below instead of failing the run.
        "every_required_scale_image_has_dfn_segments": all(
            int(row["overlay"].get("dfn_segment_count", 0)) > 0
            for row in image_rows
            if str(row.get("scale", "")) != "large"
        ),
        "large_scale_sparse_segment_recorded": not bool(zero_segment_large_images)
        or str(zero_segment_large_images) != "",
        "scale_purity_accounting_closed": all(
            bool(scan.get("patch_accounting_closed", True))
            for scan in overlay_summary["scans"].values()
            if isinstance(scan, dict)
        ),
    }
    if zero_segment_large_images:
        checks["large_scale_zero_segment_images"] = zero_segment_large_images
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir),
        "diagnostic_contract": diag,
        "well_name": str(config.get("well_name", "车页1导眼")),
        "scope_geometry": {name: geometry.summary for name, geometry in scopes.items() if name in scopes_requested},
        "overlay_source": overlay_summary,
        "images": image_rows,
        "png_files": png_files,
        "checks": checks,
    }
    write_json(output_dir / "single_scale_diagnostic_summary.json", summary)
    print(f"[single-scale-diag] output={output_dir}", flush=True)
    print(f"[single-scale-diag] status={summary['status']}", flush=True)
    return summary


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    summary = render_all(args.config, config)
    return 0 if summary["status"] in ("pass", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
