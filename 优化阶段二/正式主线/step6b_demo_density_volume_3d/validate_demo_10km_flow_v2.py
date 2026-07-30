from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and manifest the formal 10 km multiscale v2 flow.")
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--step6-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    formal = args.formal_root.resolve()
    if str(formal) not in sys.path:
        sys.path.insert(0, str(formal))
    from common.unified_dfn_vtk import unified_vtk_summary

    step6 = args.step6_root.resolve()
    version = str(args.version)
    summary_paths = {
        "input_qc": step6 / "input_qc/input_qc_summary.json",
        "step6a": step6 / "step6a_small/small_background_qc.json",
        "step6b": step6 / "step6b_medium/medium_corridor_qc.json",
        "step6c": step6 / "step6c_large/large_fault_qc.json",
        "step6d": step6 / "step6d_bundle/multiscale_bundle_summary.json",
        "step7a": formal / f"step7a_small_scale_dfn/output/{version}/small_dfn_summary.json",
        "step7b": formal / f"step7b_multiscale_initial_dfn/output/{version}/medium_dfn_summary.json",
        "step7c": formal / f"step7c_large_fault_dfn/output/{version}/large_fault_dfn_summary.json",
        "step7d": formal / f"step7d_multiscale_fused_dfn/output/{version}/fused_multiscale_summary.json",
        "step8": formal / f"step8_dfn_well_correction/output/{version}/well_corrected_dfn_summary.json",
        "step9": formal / f"step9_section_visualize/output/{version}/cheye1_dfn_multibackground_sections/section_summary.json",
    }
    summaries = {name: read_json(path) for name, path in summary_paths.items()}
    step7c_dir = formal / f"step7c_large_fault_dfn/output/{version}"
    original_surface_vtk = step7c_dir / "original_fault_units_demo_raw_time.vtk"
    original_manifest_csv = step7c_dir / "original_fault_unit_manifest.csv"
    step7c_unified_vtk = step7c_dir / "large_fault_dfn_raw_time.vtk"
    step7d_unified_vtk = formal / f"step7d_multiscale_fused_dfn/output/{version}/fused_multiscale_dfn_raw_time.vtk"
    step8_unified_vtk = formal / f"step8_dfn_well_correction/output/{version}/well_corrected_dfn_raw_time.vtk"
    step7c_source_surface = Path(summaries["step7c"]["inputs"]["original_fault_surface_vtk"]).resolve()
    step6c_original_count = int(summaries["step6c"]["original_fault"]["selected_patch_count"])
    step7c_original_count = int(summaries["step7c"].get("original_fault_unit_count", 0))
    step9_fault_scans = [
        summaries["step9"].get("overlay_source", {}).get("overview_fault_scan", {}),
        summaries["step9"].get("overlay_source", {}).get("local_200m_fault_scan", {}),
    ]
    checks: dict[str, bool] = {
        "all_stage_summaries_pass": all(payload.get("status", "pass") == "pass" for payload in summaries.values()),
        "step6a_is_2ms": abs(float(summaries["step6a"]["horizon_axis_qc"]["sample_interval_ms"]) - 2.0) <= 1.0e-6,
        "step6b_is_10ms": abs(float(summaries["step6b"]["sample_interval_ms"]) - 10.0) <= 1.0e-6,
        "step6c_is_10ms": abs(float(summaries["step6c"]["sample_interval_ms"]) - 10.0) <= 1.0e-6,
        "step6b_anttrack_primary": bool(
            summaries["step6b"].get("score_formula", {}).get("anttrack_polarity")
            == "high normalized AntTrack response is fracture evidence"
            and float(summaries["step6b"].get("score_formula", {}).get("anttrack_weight", 0.0))
            > float(summaries["step6b"].get("score_formula", {}).get("lowcoh_weight", 0.0))
            and float(summaries["step6b"].get("score_formula", {}).get("anttrack_weight", 0.0))
            > float(summaries["step6b"].get("score_formula", {}).get("curvature_weight", 0.0))
        ),
        "step6c_surface_ransac": summaries["step6c"]["inferred_fault"]["parameters"]["inferred_extraction_mode"] == "surface_ransac",
        "step6c_has_inferred_surfaces": int(summaries["step6c"]["inferred_fault"]["component_count"]) > 0,
        "step7b_upstream_coverage": float(summaries["step7b"]["upstream_component_coverage_fraction"]) >= 0.65,
        "step7b_hybrid_spatial_mode_applied": bool(
            summaries["step7b"].get("component_selection", {}).get("requested_mode")
            == "hybrid_anttrack_spatial"
            and summaries["step7b"].get("checks", {}).get("component_selection_mode_applied")
            and summaries["step7b"].get("checks", {}).get("covers_candidate_spatial_tiles")
        ),
        "step7c_has_inferred_surfaces": int(summaries["step7c"].get("inferred_fault_surface_patch_count", 0)) > 0,
        "step7c_original_surface_not_patchified": bool(
            int(summaries["step7c"].get("formal_original_fault_patch_count", -1)) == 0
            and not any("original_fault" in str(key) for key in summaries["step7c"]["source_type_counts"])
        ),
        "step7c_original_surface_not_horizon_clipped": bool(
            summaries["step7c"].get("horizon_contract_qc", {}).get("original_fault_surface")
            == "not_applied_by_contract"
        ),
        "step7c_original_unit_count_matches_step6c": bool(
            step6c_original_count > 0 and step7c_original_count == step6c_original_count
        ),
        "step7c_original_surface_summary_hash_preserved": bool(
            summaries["step7c"].get("original_fault_source_sha256")
            == summaries["step7c"].get("original_fault_copied_sha256")
        ),
        "step7c_damage_zone_not_in_formal_dfn": bool(
            summaries["step7c"].get("damage_zone_included_in_formal_dfn") is False
            and "large_original_fault_damage_zone" not in summaries["step7c"]["source_type_counts"]
        ),
        "step7c_known_fault_duplicate_classification_closed": bool(
            summaries["step7c"].get("checks", {}).get("duplicate_classification_accounting_closed")
            and summaries["step7c"].get("checks", {}).get("known_fault_duplicates_excluded_from_formal_csv")
        ),
        "step8_vertices_inside_contract": bool(
            summaries["step8"]["horizon_contract_qc"]["vertex_geometry"]["all_vertices_inside_local_t4_t7"]
        ),
        "step8_original_surface_carried_in_unified_vtk": bool(
            summaries["step8"].get("correction_logic", {}).get("original_fault_triangles_carried_unchanged_from_step7d_unified_vtk")
            and summaries["step8"].get("checks", {}).get("initial_unified_dfn_vtk_exists")
            and summaries["step8"].get("checks", {}).get("original_fault_manifest_passthrough_exists")
            and Path(summaries["step8"].get("initial_dfn_vtk", "")).resolve()
            == step7d_unified_vtk.resolve()
            and Path(summaries["step8"].get("original_fault_manifest_csv", "")).resolve()
            == original_manifest_csv.resolve()
        ),
        "step9_has_20_images": len(summaries["step9"]["png_files"]) == 20,
        "step9_projection_accounting_closed": all(
            bool(value["accounting_closed"]) for value in summaries["step9"]["projection_accounting"].values()
        ),
        "step9_uses_original_surface_triangles": all(
            scan.get("fault_trace_source") == "step8_unified_dfn_original_fault_triangles"
            and Path(scan.get("unified_dfn_vtk", "")).resolve() == step8_unified_vtk.resolve()
            and int(scan.get("surface_triangle_count", 0)) > 0
            for scan in step9_fault_scans
        ),
    }

    vtk_paths = [
        formal / f"step7a_small_scale_dfn/output/{version}/small_dfn_raw_time.vtk",
        formal / f"step7b_multiscale_initial_dfn/output/{version}/medium_dfn_raw_time.vtk",
        original_surface_vtk,
        formal / f"step7c_large_fault_dfn/output/{version}/large_inferred_fault_surfaces_raw_time.vtk",
        step7c_unified_vtk,
        step7d_unified_vtk,
        step8_unified_vtk,
    ]
    checks["independent_and_final_vtks_exist"] = all(path.exists() and path.stat().st_size > 0 for path in vtk_paths)
    original_transport_paths = [
        step7c_source_surface,
        original_surface_vtk,
        original_manifest_csv,
        step7c_unified_vtk,
        step7d_unified_vtk,
        step8_unified_vtk,
    ]
    checks["step7c_original_transport_files_exist"] = all(
        path.exists() and path.stat().st_size > 0 for path in original_transport_paths
    )
    if checks["step7c_original_transport_files_exist"]:
        checks["step7c_original_surface_file_hash_preserved"] = sha256(step7c_source_surface) == sha256(original_surface_vtk)
        checks["step7c_manifest_row_count_matches_summary"] = csv_data_row_count(original_manifest_csv) == step7c_original_count
        unified_summaries = {
            "step7c": unified_vtk_summary(step7c_unified_vtk),
            "step7d": unified_vtk_summary(step7d_unified_vtk),
            "step8": unified_vtk_summary(step8_unified_vtk),
        }
        checks["unified_vtks_have_both_geometry_groups"] = all(
            int(payload["predicted_cell_count"]) > 0 and int(payload["original_fault_cell_count"]) > 0
            for payload in unified_summaries.values()
        )
        original_fingerprints = {
            str(payload["original_fault_geometry_fingerprint"])
            for payload in unified_summaries.values()
        }
        checks["original_fault_geometry_fingerprint_continuous_step7c_to_step8"] = bool(
            len(original_fingerprints) == 1
        )
        checks["original_fault_render_area_policy_is_predicted_median"] = all(
            summaries[stage].get("unified_vtk", {}).get("original_fault_render_area_policy")
            == "predicted_patch_area_median_not_physical_triangle_area"
            for stage in ("step7c", "step7d", "step8")
        )
    else:
        checks["step7c_original_surface_file_hash_preserved"] = False
        checks["step7c_manifest_row_count_matches_summary"] = False
        checks["unified_vtks_have_both_geometry_groups"] = False
        checks["original_fault_geometry_fingerprint_continuous_step7c_to_step8"] = False
        checks["original_fault_render_area_policy_is_predicted_median"] = False
    image_dir = formal / f"step9_section_visualize/output/{version}/cheye1_dfn_multibackground_sections"
    png_paths = sorted(image_dir.glob("*.png"))
    checks["all_images_nontrivial"] = len(png_paths) == 20 and all(path.stat().st_size >= 10000 for path in png_paths)

    config_paths = [
        formal / f"step6b_demo_density_volume_3d/configs/{version}.json",
        formal / f"step6b_demo_density_volume_3d/configs/{version}_expanded.json",
        formal / f"step7a_small_scale_dfn/configs/{version}.json",
        formal / f"step7b_multiscale_initial_dfn/configs/{version}.json",
        formal / f"step7c_large_fault_dfn/configs/{version}.json",
        formal / f"step7d_multiscale_fused_dfn/configs/{version}.json",
        formal / f"step8_dfn_well_correction/configs/{version}.json",
        formal / f"step9_section_visualize/configs/{version}.json",
    ]
    manifest = {
        "version": version,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "summaries": {name: str(path) for name, path in summary_paths.items()},
        "configs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in config_paths
        ],
        "standalone_vtks": [{"path": str(path), "size_bytes": path.stat().st_size} for path in vtk_paths],
        "original_fault_transport": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in original_transport_paths
        ],
        "images": [{"path": str(path), "size_bytes": path.stat().st_size} for path in png_paths],
    }
    output_path = step6 / "flow_acceptance_v2.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "acceptance": str(output_path), "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
