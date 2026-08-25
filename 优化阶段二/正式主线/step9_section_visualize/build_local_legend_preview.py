# -*- coding: utf-8 -*-
"""井周 200m 区域：不同尺度图例样式对比预览。

只扫描一次 local_200m 的 DFN 叠加，然后对 4 种"小尺度线型"各渲染
Coherence / CurvatureMax / SeisAmp变密度 / SeisAmp波形 四张剖面，
供汇报前挑选合适的图例样式。输出：
  <正式主线>/output/step9_section_presentation/图例样式对比_local200m/
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache-tyh")

import matplotlib
matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
FORMAL_ROOT = SCRIPT_DIR.parent

import build_cheye1_dfn_multibackground_sections as mb
import build_cheye1_dfn_coherence_sections as coh
from build_cheye1_dfn_coherence_sections import (
    build_imaging_fracture_patch_segments,
    build_namespace,
    infer_dfn_patch_csv,
    load_dfn_patch_metadata,
    load_step3_imaging_fracture_labels,
    load_step3_imaging_segment_track,
    scan_unified_original_fault_intersections,
    scan_vtk_intersections,
)
from common.horizon_trace_table.horizon_contract import build_spatial_lookup
from dfn_section_overlay import SectionOverlayContext
from well_curved_section_common import prepare_geometry, read_json
from build_well_geological_attribute_sections import display_scale, plot_section
from build_well_seismic_amplitude_sections import amplitude_limit, plot_variable_density, plot_wiggle_variable_area


CONFIG = SCRIPT_DIR / "configs" / "formal_demo_10km_multiscale_flow_v2.json"
CACHE_ROOT = (
    SCRIPT_DIR
    / "output" / "formal_demo_10km_multiscale_flow_v2"
    / "cheye1_dfn_multibackground_sections" / "section_samples" / "local_200m"
)
OUT_DIR = FORMAL_ROOT / "output" / "step9_section_presentation" / "图例样式对比_local200m"

FORMAL_STYLE = "absolute_grayscale"
BACKUP_STYLE = "signed_red_white_blue"

# 变体：大尺度用"虚实线"还是"描边"（小=细实线、中=中实线、颜色按层段）
VARIANTS = [
    (
        "A_大尺度虚线",
        {"small": (1.0, "-"), "medium": (1.6, "-"), "large": (2.4, (0, (6, 2))), "": (1.4, "-")},
        {"small": ("#9ca3af", "-", 1.4, "小尺度裂缝"), "medium": ("#6b7280", "-", 2.0, "中尺度裂缝"), "large": ("#374151", (0, (6, 2)), 2.8, "大尺度裂缝")},
        False,
    ),
    (
        "B_大尺度描边",
        {"small": (1.0, "-"), "medium": (1.6, "-"), "large": (2.6, "-"), "": (1.4, "-")},
        {"small": ("#9ca3af", "-", 1.4, "小尺度裂缝"), "medium": ("#6b7280", "-", 2.0, "中尺度裂缝"), "large": ("#374151", "-", 3.2, "大尺度裂缝")},
        True,
    ),
]


def main() -> int:
    config = read_json(CONFIG)
    overview_config = dict(config)
    overview_config["target_block"] = dict(config.get("overview_target_block") or {})
    overview = prepare_geometry(overview_config)
    local = mb.local_geometry(overview, config)
    print("[preview] 加载 local_200m 剖面缓存", flush=True)
    sampled: dict[str, tuple] = {}
    for attribute in ("Coherence", "CurvatureMax", "SeisAmp"):
        cache = CACHE_ROOT / f"{attribute.lower()}_section_samples.npz"
        if not cache.exists():
            raise FileNotFoundError(f"缺少缓存: {cache}")
        sampled[attribute] = mb.load_section_pair_npz(cache, attribute)

    print("[preview] 扫描 local_200m DFN 叠加", flush=True)
    local_args = build_namespace(config, float(config.get("local_dfn_half_width_m", 200.0)))
    local_args.dfn_patch_metadata = load_dfn_patch_metadata(infer_dfn_patch_csv(config))
    local_args.well_trajectory_csv = overview.well_path
    local_args.selected_well_name = overview.well_name
    surfaces = build_spatial_lookup(config)
    segments, _scan = scan_vtk_intersections(local_args, surfaces, overview.well_df)
    local_display = {
        "display_x_min": float(local.summary["display_x_min"]),
        "display_x_max": float(local.summary["display_x_max"]),
        "display_y_min": float(local.summary["display_y_min"]),
        "display_y_max": float(local.summary["display_y_max"]),
    }
    faults, _fault_scan = scan_unified_original_fault_intersections(
        local_args.input_vtk, overview.well_df, local_display
    )
    group_paths = [Path(str(value)).resolve() for value in config.get("step3_imaging_group_csvs", [])]
    fracture_df = load_step3_imaging_fracture_labels(group_paths, overview.well_name, dict(config.get("target_block") or {}))
    imaging_df = load_step3_imaging_segment_track(group_paths, overview.well_name, dict(config.get("target_block") or {}))
    imaging_patches = build_imaging_fracture_patch_segments(fracture_df)
    context = SectionOverlayContext(
        "local_200m", segments, faults, imaging_patches, fracture_df, imaging_df, local.summary
    )

    attribute_scales: dict[str, tuple] = {}
    for attribute in ("Coherence", "CurvatureMax"):
        arrays = [sampled[attribute][proj].values for proj in (0, 1)]
        attribute_scales[attribute] = display_scale(attribute, arrays, FORMAL_STYLE)
    seis_limit = amplitude_limit(
        [sampled["SeisAmp"][proj] for proj in (0, 1)],
        float(config.get("amplitude_clip_quantile", 0.99)),
    )

    preview_geometry = mb.make_ppt_geometry(
        local,
        {"width_cm": 20.0, "height_cm": 12.0, "dpi": 180,
         "legend_width_in": 2.0, "legend_fontsize": 9.0, "wiggle_linewidth": 0.7},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = [
        ("CurvatureMax", "attribute", "XZ"),
        ("SeisAmp", "density", "XZ"),
        ("SeisAmp", "wiggle", "XZ"),
    ]
    for variant_name, line_styles, legend_styles, halo in VARIANTS:
        coh.SCALE_LINE_STYLES.update(line_styles)
        coh.SCALE_LEGEND_STYLES.update(legend_styles)
        coh.SCALE_LARGE_HALO = bool(halo)
        for attribute, renderer, projection in plan:
            section = sampled[attribute][0]
            drawer, _captured = mb.make_overlay_drawer(context)
            token = attribute.lower() if attribute != "SeisAmp" else "seisamp"
            filename = f"{variant_name.replace(' ', '_')}_{token}_{renderer}_{projection.lower()}.png"
            path = OUT_DIR / filename
            if renderer == "attribute":
                vmin, vmax, norm, _ = attribute_scales[attribute]
                plot_section(
                    section, preview_geometry, path, vmin, vmax, norm,
                    display_style=FORMAL_STYLE, overlay_drawer=drawer,
                    scope_label="井周200m 图例预览", product_title="车页1导眼：DFN图例样式对比",
                )
            elif renderer == "density":
                plot_variable_density(
                    section, preview_geometry, path, seis_limit,
                    display_style=FORMAL_STYLE, overlay_drawer=drawer,
                    scope_label="井周200m 图例预览", product_title="车页1导眼：DFN图例样式对比",
                )
            else:
                plot_wiggle_variable_area(
                    section, preview_geometry, path, seis_limit,
                    overlay_drawer=drawer,
                    scope_label="井周200m 图例预览", product_title="车页1导眼：DFN图例样式对比",
                )
            print(f"[preview] {filename}", flush=True)
    print(f"[preview] 完成：{OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
