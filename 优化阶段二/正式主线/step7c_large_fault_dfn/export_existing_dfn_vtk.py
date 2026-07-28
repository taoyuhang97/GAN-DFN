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
        (paths["fault_only_csv"], paths["fault_only_vtk"], "step7c_large_original_fault_and_influence_raw_time"),
        (paths["lowcoh_csv"], paths["lowcoh_vtk"], "step7c_large_lowcoh_component_panels_raw_time"),
        (paths["dfn_csv"], paths["raw_vtk"], "step7c_large_fault_and_damage_raw_time"),
    ]
    for csv_path, vtk_path, title in products:
        patch_df = pd.read_csv(csv_path, low_memory=False)
        step7c.write_patch_vtk(vtk_path, patch_df, title)
        print(f"[step7c-vtk] patches={len(patch_df)} output={vtk_path}")
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    summary.setdefault("checks", {}).update(
        {
            "fault_only_vtk_exists": paths["fault_only_vtk"].exists(),
            "lowcoh_vtk_exists": paths["lowcoh_vtk"].exists(),
            "raw_vtk_exists": paths["raw_vtk"].exists(),
        }
    )
    summary["standalone_3d_vtks"] = {
        "fault_and_influence": str(paths["fault_only_vtk"]),
        "lowcoh_supplement": str(paths["lowcoh_vtk"]),
        "combined_large_dfn": str(paths["raw_vtk"]),
        "fault_surface_fragments": str(paths["fault_surface_vtk"]),
    }
    summary["status"] = "pass" if all(bool(value) for value in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
