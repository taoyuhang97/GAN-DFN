from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    args = parse_args()
    formal = args.formal_root.resolve()
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
    checks: dict[str, bool] = {
        "all_stage_summaries_pass": all(payload.get("status", "pass") == "pass" for payload in summaries.values()),
        "step6a_is_2ms": abs(float(summaries["step6a"]["horizon_axis_qc"]["sample_interval_ms"]) - 2.0) <= 1.0e-6,
        "step6b_is_10ms": abs(float(summaries["step6b"]["sample_interval_ms"]) - 10.0) <= 1.0e-6,
        "step6c_is_10ms": abs(float(summaries["step6c"]["sample_interval_ms"]) - 10.0) <= 1.0e-6,
        "step6b_anttrack_primary": "supported_rescue" in json.dumps(summaries["step6b"], ensure_ascii=False),
        "step6c_surface_ransac": summaries["step6c"]["inferred_fault"]["parameters"]["inferred_extraction_mode"] == "surface_ransac",
        "step6c_has_inferred_surfaces": int(summaries["step6c"]["inferred_fault"]["component_count"]) > 0,
        "step7b_upstream_coverage": float(summaries["step7b"]["upstream_component_coverage_fraction"]) >= 0.65,
        "step7c_has_inferred_surfaces": int(summaries["step7c"].get("inferred_fault_surface_patch_count", 0)) > 0,
        "step7c_original_fragments_are_formal_dfn_objects": int(
            summaries["step7c"]["source_type_counts"].get("large_original_fault_surface_fragment", 0)
        ) > 0,
        "step7c_damage_zone_is_diagnostic_only": bool(
            summaries["step7c"].get("damage_zone_included_in_formal_dfn") is False
            and "large_original_fault_damage_zone" not in summaries["step7c"]["source_type_counts"]
        ),
        "step8_vertices_inside_contract": bool(
            summaries["step8"]["horizon_contract_qc"]["vertex_geometry"]["all_vertices_inside_local_t4_t7"]
        ),
        "step9_has_20_images": len(summaries["step9"]["png_files"]) == 20,
        "step9_projection_accounting_closed": all(
            bool(value["accounting_closed"]) for value in summaries["step9"]["projection_accounting"].values()
        ),
    }

    vtk_paths = [
        formal / f"step7a_small_scale_dfn/output/{version}/small_dfn_raw_time.vtk",
        formal / f"step7b_multiscale_initial_dfn/output/{version}/medium_dfn_raw_time.vtk",
        formal / f"step7c_large_fault_dfn/output/{version}/large_original_fault_merged_surface_raw_time.vtk",
        formal / f"step7c_large_fault_dfn/output/{version}/large_fault_surface_raw_time.vtk",
        formal / f"step7c_large_fault_dfn/output/{version}/large_inferred_fault_surfaces_raw_time.vtk",
        formal / f"step7c_large_fault_dfn/output/{version}/large_fault_damage_zone_raw_time.vtk",
        formal / f"step7c_large_fault_dfn/output/{version}/large_fault_dfn_raw_time.vtk",
        formal / f"step8_dfn_well_correction/output/{version}/well_corrected_dfn_raw_time.vtk",
    ]
    checks["independent_and_final_vtks_exist"] = all(path.exists() and path.stat().st_size > 0 for path in vtk_paths)
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
        "images": [{"path": str(path), "size_bytes": path.stat().st_size} for path in png_paths],
    }
    output_path = step6 / "flow_acceptance_v2.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "acceptance": str(output_path), "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
