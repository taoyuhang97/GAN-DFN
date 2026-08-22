# -*- coding: utf-8 -*-
"""Export single-scale DFN products from the final well-corrected unified VTK.

Reads the Step8 final unified DFN (predicted patches + original fault
triangles) and the well-corrected patch CSV, then writes three standalone
DFN sets by fracture scale:
  - small: predicted cells with FractureScale == small
  - medium: predicted cells with FractureScale == medium
  - large: predicted cells with FractureScale == large plus the unchanged
    original-fault triangle group (GeometryGroupCode=2)

All outputs keep the unified legacy ASCII POLYDATA contract used by Step9, so
they can be opened with the same intersection scan code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from common.unified_dfn_vtk import (  # noqa: E402
    ORIGINAL_FAULT_GEOMETRY_GROUP_CODE,
    PREDICTED_GEOMETRY_GROUP_CODE,
    _write_ascii_legacy_polydata,
    extract_geometry_group,
    geometry_fingerprint,
)


SCALES = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export single-scale DFN VTKs from the final unified DFN.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-vtk", type=Path, default=None)
    parser.add_argument("--patch-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.config is not None:
        config = read_json(args.config.resolve())
        output_dir = Path(str(config.get("single_scale_dfn_export", {}).get("output_dir") or config["output_dir"])).resolve()
        if config.get("input_vtk") and config.get("dfn_patch_csv"):
            # Step9-style config with explicit input paths.
            input_vtk = Path(str(config["input_vtk"])).resolve()
            patch_csv = Path(str(config["dfn_patch_csv"])).resolve()
        else:
            # Step8-style config: final corrected outputs live directly in output_dir.
            input_vtk = Path(str(config["output_dir"])).resolve() / "well_corrected_dfn_raw_time.vtk"
            patch_csv = Path(str(config["output_dir"])).resolve() / "well_corrected_dfn_fracture_patches.csv"
    else:
        if args.input_vtk is None or args.patch_csv is None or args.output_dir is None:
            raise SystemExit("either --config or --input-vtk/--patch-csv/--output-dir must be provided")
        input_vtk = args.input_vtk.resolve()
        patch_csv = args.patch_csv.resolve()
        output_dir = args.output_dir.resolve()

    if not input_vtk.exists():
        raise FileNotFoundError(input_vtk)
    if not patch_csv.exists():
        raise FileNotFoundError(patch_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(patch_csv, low_memory=False)
    required = {"PatchID", "FractureScale"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"patch CSV missing columns: {missing}")
    df["_scale"] = df["FractureScale"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["_scale"].unique()) - set(SCALES))
    if unknown:
        raise ValueError(f"unexpected FractureScale values: {unknown}")

    predicted = extract_geometry_group(input_vtk, PREDICTED_GEOMETRY_GROUP_CODE)
    original = extract_geometry_group(input_vtk, ORIGINAL_FAULT_GEOMETRY_GROUP_CODE)
    if int(predicted.n_cells) != len(df):
        raise ValueError(f"predicted cell count {predicted.n_cells} != patch CSV rows {len(df)}")
    original_fingerprint = geometry_fingerprint(original)
    if original.n_cells == 0:
        raise ValueError(f"unified DFN contains no original-fault group: {input_vtk}")

    products: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        mask = df["_scale"].eq(scale).to_numpy(dtype=bool)
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise RuntimeError(f"no predicted patches with scale={scale}")
        scale_mesh = predicted.extract_cells(indices.astype(np.int64)).extract_surface(algorithm="dataset_surface")
        scale_mesh.clear_point_data()
        combined = scale_mesh
        if scale == "large":
            combined = scale_mesh.append_polydata(original)
        combined_cell_arrays = dict(combined.cell_data)
        for name, values in combined_cell_arrays.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.dtype.kind not in "biuf":
                combined.cell_data.pop(name, None)
        vtk_path = output_dir / f"{scale}_scale_dfn_raw_time.vtk"
        _write_ascii_legacy_polydata(vtk_path, combined)
        csv_path = output_dir / f"{scale}_scale_dfn_patches.csv"
        df.loc[mask].to_csv(csv_path, index=False, encoding="utf-8-sig")
        products[scale] = {
            "vtk": str(vtk_path),
            "patch_csv": str(csv_path),
            "patch_count": int(mask.sum()),
            "cell_count": int(combined.n_cells),
            "original_fault_group_included": scale == "large",
        }
        print(f"[single-scale-export] scale={scale} patches={int(mask.sum())} cells={int(combined.n_cells)} output={vtk_path}", flush=True)

    summary = {
        "status": "pass",
        "input_vtk": str(input_vtk),
        "patch_csv": str(patch_csv),
        "output_dir": str(output_dir),
        "predicted_cell_count": int(predicted.n_cells),
        "original_fault_cell_count": int(original.n_cells),
        "original_fault_geometry_fingerprint": original_fingerprint,
        "products": products,
        "checks": {
            "large_includes_original_faults": int(products["large"]["original_fault_group_included"]) == 1,
            "scale_counts_sum_to_total": sum(int(p["patch_count"]) for p in products.values()) == len(df),
        },
    }
    summary_path = output_dir / "single_scale_dfn_export_summary.json"
    write_json(summary_path, summary)
    print(f"[single-scale-export] status=pass summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
