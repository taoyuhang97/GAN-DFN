#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"上部补充层"DFN 与现有全域太古界 DFN 拼接。

两部分时间区间不相交：
  现有 DFN : [Top, Base+50]（上部复合层 + 太古界风化壳）
  新增 DFN : [TopV, Top)（上部补充层）
因此拼接是纯几何并集，不需要冲突消解。

输出（默认 ``太古界/supp_layer/output/taigu_supp_merged_v1/``）：
  merged_dfn_patches.csv
  merged_dfn_raw_time.vtk
  merged_dfn_summary.json

用法::

    python3 太古界/supp_layer/merge_supp_dfn.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import vtk


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"

FULL_CSV = TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_corrected_dfn_fracture_patches.csv"
FULL_VTK = TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_corrected_dfn_raw_time.vtk"
SUPP_CSV = TAIGU / "supp_layer/output/taigu_supp_step8_v1/step8_supp/well_corrected_dfn_supp_patches.csv"
SUPP_VTK = TAIGU / "supp_layer/output/taigu_supp_step8_v1/step8_supp/well_corrected_dfn_raw_time.vtk"
OUTPUT = TAIGU / "supp_layer/output/taigu_supp_merged_v1"


def read_polydata(path: Path):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllNormalsOn()
    reader.ReadAllTCoordsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    return reader.GetOutput()


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge supplemental-layer DFN with the existing full DFN.")
    parser.add_argument("--full-csv", type=Path, default=FULL_CSV)
    parser.add_argument("--full-vtk", type=Path, default=FULL_VTK)
    parser.add_argument("--supp-csv", type=Path, default=SUPP_CSV)
    parser.add_argument("--supp-vtk", type=Path, default=SUPP_VTK)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.full_csv, low_memory=False)
    supp = pd.read_csv(args.supp_csv, low_memory=False)
    for column in full.columns:
        if column not in supp.columns:
            supp[column] = pd.NA
    supp = supp[full.columns]
    merged = pd.concat([full, supp], ignore_index=True)
    merged_path = out_dir / "merged_dfn_patches.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"[merge] wrote {merged_path} rows={len(merged):,} (full={len(full):,}, supp={len(supp):,})")

    full_data = read_polydata(args.full_vtk)
    supp_data = read_polydata(args.supp_vtk)
    append = vtk.vtkAppendPolyData()
    append.AddInputData(full_data)
    append.AddInputData(supp_data)
    append.Update()
    merged_data = append.GetOutput()
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(str(out_dir / "merged_dfn_raw_time.vtk"))
    writer.SetInputData(merged_data)
    writer.SetFileTypeToASCII()
    writer.Write()
    print(
        f"[merge] wrote VTK points={merged_data.GetNumberOfPoints():,} "
        f"cells={merged_data.GetNumberOfCells():,}"
    )

    summary = {
        "status": "pass",
        "full_csv": str(args.full_csv.resolve()),
        "supp_csv": str(args.supp_csv.resolve()),
        "full_vtk": str(args.full_vtk.resolve()),
        "supp_vtk": str(args.supp_vtk.resolve()),
        "rows": {"full": int(len(full)), "supp": int(len(supp)), "merged": int(len(merged))},
        "vtk_cells": {"full": int(full_data.GetNumberOfCells()), "supp": int(supp_data.GetNumberOfCells()), "merged": int(merged_data.GetNumberOfCells())},
        "layer_group_counts": merged["LayerGroup"].value_counts().to_dict(),
        "layer_code_counts": {str(k): int(v) for k, v in merged["LayerCode"].value_counts(dropna=False).items()},
        "fracture_scale_counts": merged["FractureScale"].value_counts().to_dict() if "FractureScale" in merged.columns else {},
        "time_range_ms": [float(merged["CenterTime"].min()), float(merged["CenterTime"].max())],
        "outputs": {
            "patches_csv": str(merged_path),
            "vtk": str(out_dir / "merged_dfn_raw_time.vtk"),
        },
    }
    (out_dir / "merged_dfn_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["layer_group_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
