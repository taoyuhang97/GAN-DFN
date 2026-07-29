from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_large_fault_dfn as step7c


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the existing Step7C large-scale DFN CSV products as 3D VTKs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"]).resolve()
    paths = step7c.output_paths(output_dir)
    products = [
        (paths["fault_surface_csv"], paths["fault_surface_vtk"], "step7c_large_original_fault_surface_fragments_raw_time"),
        (paths["damage_csv"], paths["damage_vtk"], "step7c_large_fault_damage_zone_diagnostic_raw_time"),
        (paths["lowcoh_csv"], paths["lowcoh_vtk"], "step7c_large_inferred_fault_surfaces_raw_time"),
        (paths["dfn_csv"], paths["raw_vtk"], "step7c_large_fault_dfn_raw_time"),
    ]
    for csv_path, vtk_path, title in products:
        patch_df = pd.read_csv(csv_path, low_memory=False)
        step7c.write_patch_vtk(vtk_path, patch_df, title)
        print(f"[step7c-vtk] patches={len(patch_df)} output={vtk_path}")
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    summary.setdefault("checks", {}).update(
        {
            "fault_surface_vtk_exists": paths["fault_surface_vtk"].exists(),
            "damage_vtk_exists": paths["damage_vtk"].exists(),
            "lowcoh_vtk_exists": paths["lowcoh_vtk"].exists(),
            "raw_vtk_exists": paths["raw_vtk"].exists(),
        }
    )
    summary["standalone_3d_vtks"] = {
        "stitched_original_fault_surface": str(paths["original_merged_surface_vtk"]),
        "original_fault_surface_fragments": str(paths["fault_surface_vtk"]),
        "fault_damage_zone_diagnostic": str(paths["damage_vtk"]),
        "inferred_fault_surfaces": str(paths["lowcoh_vtk"]),
        "combined_large_dfn": str(paths["raw_vtk"]),
    }
    summary["status"] = "pass" if all(bool(value) for value in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
