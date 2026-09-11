from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_medium_scale_dfn_v4 as step7b


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the existing Step7B medium-scale DFN CSV as a 3D VTK.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"]).resolve()
    paths = step7b.output_paths(output_dir)
    patch_df = pd.read_csv(paths["dfn_csv"], low_memory=False)
    if "IsWellControlPatch" not in patch_df.columns:
        patch_df["IsWellControlPatch"] = 0
    step7b.geometry.write_patch_vtk(
        paths["raw_vtk"],
        patch_df,
        "step7b_medium_dfn_raw_time",
        display=False,
        display_z_scale=float(config.get("display_z_scale", 5.0)),
        geometry_time_scale_m_per_ms=float(config.get("geometry_time_scale_m_per_ms", 1.0)),
    )
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    summary.setdefault("checks", {})["raw_vtk_exists"] = paths["raw_vtk"].exists()
    summary["standalone_3d_vtk"] = str(paths["raw_vtk"])
    summary["status"] = "pass" if all(bool(value) for value in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7b-vtk] patches={len(patch_df)} output={paths['raw_vtk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
