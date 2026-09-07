from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
REPO_ROOT = FORMAL_ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand the formal 10 km Step6-Step9 flow configs from reviewed templates.")
    parser.add_argument("--master-config", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    master_path = args.master_config.resolve()
    master = read_json(master_path)
    version = str(master["version"])
    block = dict(master["target_block"])
    step6_root = Path(master["output_dir"]).resolve()
    mapping = str(Path(master["trace_mapping_npz"]).resolve())
    output_root = str(master.get("output_root") or "").strip()

    def step_output_dir(legacy_dir: str, sub_dir: str) -> str:
        """Centralized version-root outputs when master defines output_root;
        otherwise keep the legacy per-step `output/<version>` layout."""
        if output_root:
            return f"{output_root.rstrip('/')}/{sub_dir}"
        return str(FORMAL_ROOT / f"{legacy_dir}/output/{version}")

    horizon_common = {
        "horizon_contract_table": str(Path(master["horizon_contract_table"]).resolve()),
        "horizon_trace_table_path": str(Path(master["horizon_contract_table"]).resolve()),
        "horizon_window_contracts": {
            str(key): str(Path(value).resolve())
            for key, value in dict(master["horizon_window_contracts"]).items()
        },
        "trace_header_csv": str(Path(master["trace_header_csv"]).resolve()),
    }

    generated: dict[str, str] = {}
    step6_config = copy.deepcopy(master)
    step6_config.pop("base_config", None)
    step6_config_path = CURRENT_DIR / "configs" / f"{version}_expanded.json"
    write_json(step6_config_path, step6_config)
    generated["step6"] = str(step6_config_path)

    step7a_template = read_json(FORMAL_ROOT / "step7a_small_scale_dfn/configs/formal_candidate_cheye1_step7a_small_final_lowcoh_vertical_v1.json")
    step7a_template.update(
        {
            "trace_mapping_npz": mapping,
            "output_dir": step_output_dir("step7a_small_scale_dfn", "step7a_small_scale"),
            "target_block": block,
            "density_sgy": str(step6_root / "step6a_small/small_background_score.sgy"),
            "small_score_sgy": str(step6_root / "step6a_small/small_background_score.sgy"),
            "small_qc_json": str(step6_root / "step6a_small/small_background_qc.json"),
            "damage_context_npz": str(step6_root / "step6d_bundle/multiscale_damage_context_10ms.npz"),
            "compact_step7_sampling": bool(master.get("compact_step7_sampling", True)),
            "compact_sampling_chunk_samples": int(master.get("step7a_compact_sampling_chunk_samples", 20)),
            "orientation_cache_xy_cells": int(master.get("step7a_orientation_cache_xy_cells", 3)),
            "orientation_cache_time_samples": int(master.get("step7a_orientation_cache_time_samples", 5)),
            "step7a_fail_fast_patch_count_limit": int(master.get("step7a_fail_fast_patch_count_limit", 120000)),
            "max_damage_selected_fraction_of_background": float(master.get("step7a_max_damage_fraction_of_background", 0.75)),
            "nearest_neighbor_time_scale_m_per_ms": float(master.get("step7a_nearest_neighbor_time_scale_m_per_ms", 2.0)),
            "write_intermediate_vtk": False,
            "orientation_window_time_samples": 15,
            **horizon_common,
        }
    )
    step7a_template.pop("small_domain_density_sgys", None)
    scale_multiplier = float(master.get("step7a_probability_scale_multiplier", 0.09))
    fail_limit = int(master.get("step7a_fail_fast_patch_count_limit", 120000))
    for domain in step7a_template["small_domain_generation"].values():
        domain["weighted_probability_scale"] = float(domain["weighted_probability_scale"]) * scale_multiplier
        domain["fail_fast_patch_count_limit"] = fail_limit
    step7a_template["small_domain_generation"]["medium_damage"].update(
        {
            "max_selected_fraction_of_background": float(master.get("step7a_medium_damage_fraction_of_background", 0.50)),
            "max_selected_count": int(master.get("step7a_medium_damage_max_count", 40000)),
        }
    )
    step7a_template["small_domain_generation"]["large_damage"].update(
        {
            "max_selected_fraction_of_background": float(master.get("step7a_large_damage_fraction_of_background", 0.25)),
            "max_selected_count": int(master.get("step7a_large_damage_max_count", 20000)),
        }
    )
    step7a_path = FORMAL_ROOT / f"step7a_small_scale_dfn/configs/{version}.json"
    write_json(step7a_path, step7a_template)
    generated["step7a"] = str(step7a_path)

    step7b = read_json(FORMAL_ROOT / "step7b_multiscale_initial_dfn/configs/formal_candidate_cheye1_step7b_medium_lowcoh_vertical_v1.json")
    step7b.update(
        {
            "medium_prior_sgy": str(step6_root / "step6b_medium/medium_corridor_prior.sgy"),
            "medium_components_npz": str(step6_root / "step6b_medium/medium_corridor_components.npz"),
            "medium_component_summary_csv": str(step6_root / "step6b_medium/medium_corridor_component_summary.csv"),
            "trace_mapping_npz": mapping,
            "output_dir": step_output_dir("step7b_multiscale_initial_dfn", "step7b_medium_scale"),
            "target_block": block,
            "min_prior_score": 0.58,
            "min_component_voxels": 80,
            "component_voxels_per_patch": 140.0,
            "max_patches_per_component": 36,
            "orientation_window_time_samples": 4,
            "component_selection_mode": "hybrid_anttrack_spatial",
            "anttrack_primary_fraction": 0.60,
            "spatial_candidate_multiplier": 4,
            "anttrack_ridge_quantile": 0.72,
            "anttrack_ridge_min_score": 0.45,
            "anttrack_ridge_nms_radius_cells": 1,
            "ridge_patch_spacing_m": 85.0,
            "min_patch_center_separation_m": 40.0,
            "min_upstream_component_coverage_fraction": 0.65,
            "min_quadrants_covered": 3,
            "spatial_coverage_tile_size_m": 250.0,
            "spatial_coverage_min_candidate_voxels_per_tile": 20,
            "min_spatial_tile_coverage_fraction": 0.60,
            "use_global_component_orientation_fallback": True,
            "local_pca_min_planarity": 0.03,
            "local_geometry_min_dip_deg": 30.0,
            "global_extent_axis_floor_fraction": 0.12,
            "global_extent_time_floor_fraction": 0.12,
            "global_extent_length_fraction": 0.16,
            "global_extent_height_fraction": 0.10,
            "write_intermediate_vtk": False,
            **horizon_common,
        }
    )
    step7b.pop("medium_mask_sgy", None)
    step7b_path = FORMAL_ROOT / f"step7b_multiscale_initial_dfn/configs/{version}.json"
    write_json(step7b_path, step7b)
    generated["step7b"] = str(step7b_path)

    step7c = read_json(FORMAL_ROOT / "step7c_large_fault_dfn/configs/formal_candidate_cheye1_step7c_large_from_step6c_faultlike_surface_v1.json")
    step7c.update(
        {
            "original_fault_rasterization_audit_csv": str(step6_root / "step6c_large/original_fault_rasterization_audit.csv"),
            "original_fault_surface_vtk": str(step6_root / "step6c_large/original_fault_units_demo_raw_time.vtk"),
            "original_fault_manifest_csv": str(step6_root / "step6c_large/original_fault_unit_manifest.csv"),
            "fault_patch_overlap_csv": str(step6_root / "input_qc/fault_patch_demo_overlap.csv"),
            "large_prior_sgy": str(step6_root / "step6c_large/large_fault_prior.sgy"),
            "large_prior_components_npz": str(step6_root / "step6c_large/large_fault_prior_components.npz"),
            "large_component_summary_csv": str(step6_root / "step6c_large/large_fault_component_summary.csv"),
            "trace_mapping_npz": mapping,
            "output_dir": step_output_dir("step7c_large_fault_dfn", "step7c_large_fault"),
            "target_block": block,
            "time_min_ms": float(master.get("step7c_time_min_ms", 1990.0)),
            "time_max_ms": float(master.get("step7c_time_max_ms", 3626.0)),
            "min_lowcoh_component_voxels": 900,
            "lowcoh_min_panel_voxels": 120,
            "inferred_geometry_mode": "step6c_surface_ransac_vertices",
            "original_fault_compare_time_scale_m_per_ms": 2.0,
            "original_fault_duplicate_center_distance_m": 60.0,
            "original_fault_duplicate_vertex_distance_m": 100.0,
            "original_fault_duplicate_min_vertex_fraction": 0.75,
            "original_fault_duplicate_max_orientation_difference_deg": 25.0,
            "original_fault_branch_context_distance_m": 120.0,
            "write_intermediate_vtk": False,
            **horizon_common,
        }
    )
    step7c.pop("large_mask_sgy", None)
    step7c_path = FORMAL_ROOT / f"step7c_large_fault_dfn/configs/{version}.json"
    write_json(step7c_path, step7c)
    generated["step7c"] = str(step7c_path)

    step7d = read_json(FORMAL_ROOT / "step7d_multiscale_fused_dfn/configs/formal_candidate_cheye1_step7d_fused_final_lowcoh_vertical_v1.json")
    step7d.update(
        {
            "small_dfn_csv": str(FORMAL_ROOT / f"step7a_small_scale_dfn/output/{version}/small_dfn_patches.csv"),
            "medium_dfn_csv": str(FORMAL_ROOT / f"step7b_multiscale_initial_dfn/output/{version}/medium_dfn_patches.csv"),
            "large_dfn_csv": str(FORMAL_ROOT / f"step7c_large_fault_dfn/output/{version}/large_fault_dfn_patches.csv"),
            "large_dfn_vtk": str(Path(step_output_dir("step7c_large_fault_dfn", "step7c_large_fault")) / "large_fault_dfn_raw_time.vtk"),
            "original_fault_manifest_csv": str(Path(step_output_dir("step7c_large_fault_dfn", "step7c_large_fault")) / "original_fault_unit_manifest.csv"),
            "output_dir": step_output_dir("step7d_multiscale_fused_dfn", "step7d_fused"),
            "write_vtk": True,
        }
    )
    step7d.pop("original_fault_surface_vtk", None)
    if output_root:
        step7d["small_dfn_csv"] = str(Path(step_output_dir("step7a_small_scale_dfn", "step7a_small_scale")) / "small_dfn_patches.csv")
        step7d["medium_dfn_csv"] = str(Path(step_output_dir("step7b_multiscale_initial_dfn", "step7b_medium_scale")) / "medium_dfn_patches.csv")
        step7d["large_dfn_csv"] = str(Path(step_output_dir("step7c_large_fault_dfn", "step7c_large_fault")) / "large_fault_dfn_patches.csv")
    step7d_path = FORMAL_ROOT / f"step7d_multiscale_fused_dfn/configs/{version}.json"
    write_json(step7d_path, step7d)
    generated["step7d"] = str(step7d_path)

    step8 = read_json(FORMAL_ROOT / "step8_dfn_well_correction/configs/formal_well_control_correction_candidate_cheye1_final_lowcoh_vertical_v1_smallwell_event_v1.json")
    step7d_dir = step_output_dir("step7d_multiscale_fused_dfn", "step7d_fused")
    step8.update(
        {
            "initial_dfn_csv": str(Path(step7d_dir) / "fused_multiscale_dfn_patches.csv"),
            "initial_dfn_vtk": str(Path(step7d_dir) / "fused_multiscale_dfn_raw_time.vtk"),
            "initial_dfn_summary_json": str(Path(step7d_dir) / "fused_multiscale_summary.json"),
            "fracture_points_csv": str(FORMAL_ROOT / "step4_expert_real_well_prediction/output/formal_six_expert_library_v3/predictions/all_wells_t4_t7_merged_fracture_points.csv"),
            "output_dir": step_output_dir("step8_dfn_well_correction", "step8_well_correction"),
            "target_block": block,
            "export_debug_step_vtks": False,
            "original_fault_manifest_csv": str(Path(step_output_dir("step7c_large_fault_dfn", "step7c_large_fault")) / "original_fault_unit_manifest.csv"),
            "enable_imaging_well_region_correction": True,
            "imaging_well_region_xy_radius_m": 150.0,
            "imaging_well_region_time_radius_ms": 40.0,
            "imaging_well_region_density_blend": 0.65,
            "imaging_well_region_min_influence": 0.05,
            **horizon_common,
        }
    )
    step8.pop("original_fault_surface_vtk", None)
    if "single_scale_dfn_export" in master:
        step8["single_scale_dfn_export"] = dict(master["single_scale_dfn_export"])
    step8_path = FORMAL_ROOT / f"step8_dfn_well_correction/configs/{version}.json"
    write_json(step8_path, step8)
    generated["step8"] = str(step8_path)

    step9 = read_json(FORMAL_ROOT / "step9_section_visualize/configs/formal_candidate_cheye1_final_lowcoh_vertical_v1_smallwell_event_v1_multibackground_sections.json")
    step9.update(
        {
            "input_vtk": str(Path(step_output_dir("step8_dfn_well_correction", "step8_well_correction")) / "well_corrected_dfn_raw_time.vtk"),
            "dfn_patch_csv": str(Path(step_output_dir("step8_dfn_well_correction", "step8_well_correction")) / "well_corrected_dfn_fracture_patches.csv"),
            "output_dir": step_output_dir("step9_section_visualize", "step9_sections/cheye1_dfn_multibackground_sections"),
            "target_block": block,
            "overview_target_block": block,
            "overview_section_sample_paths": {},
            "local_axis_radius_m": float(master.get("step9_local_axis_radius_m", 500.0)),
            "image_numbers": list(range(1, 21)),
            "title_prefix": str(master.get("step9_title_prefix", "车页1导眼10 km Demo剖面")),
            "product_title": str(master.get("step9_product_title", "车页1导眼：10 km Demo多尺度DFN")),
            "overview_product_title": str(master.get("step9_overview_product_title", "车页1导眼：10 km Demo整体DFN")),
            "local_product_title": str(master.get("step9_local_product_title", "车页1导眼：井周DFN与成像测井裂缝对比")),
            "formal_signed_attribute_display_style": "absolute_grayscale",
            "signed_attribute_backup_display_style": "signed_red_white_blue",
            "signed_attribute_backup_dir": "red_white_blue_backup",
            **horizon_common,
        }
    )
    if "step9_diagnostic_sections" in master:
        step9["step9_diagnostic_sections"] = dict(master["step9_diagnostic_sections"])
    if "single_scale_dfn_export" in master:
        step9["single_scale_dfn_export"] = dict(master["single_scale_dfn_export"])
    if master.get("single_scale_diag_output_dir"):
        step9["single_scale_diag_output_dir"] = str(master["single_scale_diag_output_dir"])
    if master.get("step9_display_horizon_table"):
        step9["step9_display_horizon_table"] = str(master["step9_display_horizon_table"])
    step9.pop("original_fault_stick_dat", None)
    step9.pop("fault_surface_csv", None)
    step9.pop("original_fault_surface_vtk", None)
    step9_path = FORMAL_ROOT / f"step9_section_visualize/configs/{version}.json"
    write_json(step9_path, step9)
    generated["step9"] = str(step9_path)

    manifest_path = step6_root / "generated_flow_configs.json"
    write_json(manifest_path, {"status": "pass", "master_config": str(master_path), "generated": generated})
    print(json.dumps({"status": "pass", "manifest": str(manifest_path), "generated": generated}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
