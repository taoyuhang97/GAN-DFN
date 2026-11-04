#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为太古界"上部补充层"抽取 405 的成像监督组与常规测井样本。

输入：
  * 405 的 Step2 分段（seg_002，成像段 MD 3676.6-3998.0 / TIME 2746.8-2866.2）
  * 405 的 FMI 裂缝参数 CSV（P10_SUM_裂缝密度）
  * 上部补充层契约（TopV / Top）

输出（默认 ``太古界/supp_layer/output/taigu_supp_samples_v1/``）：
  supp_imaging_group.csv    Step3 组格式，可直接给 Step4/Step8 使用
  supp_real_well_samples.csv 带 GR/RD/RS/Density/LayerGroup 的井上样本

用法::

    python3 太古界/supp_layer/build_supp_samples.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"

DEFAULT_SEGMENT = (
    TAIGU / "step2_well_log_segments/output/taigu_step2_regular_md1/埕北古斜405/埕北古斜405_seg_002.csv"
)
DEFAULT_DENSITY = Path(
    "/data/shared/project-oil/wx数据/埕岛太古界/埕岛-桩海地区成像测井/"
    "中国石化胜利油田埕北古斜405井FMI成像测井解释成果/"
    "埕北古斜405井FMI成像解释成果数据文件/CSV/埕北古斜405井_Fracture parameter_Final_裂缝参数.csv"
)
DEFAULT_CONTRACT = TAIGU / "supp_layer/output/taigu_supp_horizon_v1/supp_horizon_contract.npy"
DEFAULT_HEADER = (
    TAIGU / "common/attribute_trace_contract/output/taigu_attribute_trace_header_v1/attribute_trace_header.csv"
)
DEFAULT_OUTPUT = TAIGU / "supp_layer/output/taigu_supp_samples_v1"
LAYER_NAME = "上部补充层"
LAYER_CODE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 405 supplementary-layer samples.")
    parser.add_argument("--segment", type=Path, default=DEFAULT_SEGMENT)
    parser.add_argument("--density", type=Path, default=DEFAULT_DENSITY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--trace-header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-tolerance", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seg_path = args.segment.resolve()
    seg = pd.read_csv(seg_path, low_memory=False)
    print(f"[supp-samples] segment rows={len(seg):,}  MD={seg.MD.min():.1f}-{seg.MD.max():.1f}")

    contract = np.load(args.contract, mmap_mode="r")
    header = pd.read_csv(args.trace_header, usecols=["X", "Y"], encoding="utf-8-sig")
    tree = cKDTree(header[["X", "Y"]].to_numpy(dtype=np.float64))
    distance, nearest = tree.query(seg[["X", "Y"]].to_numpy(dtype=np.float64), k=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    top_v = contract["TopVTimeMs"][nearest].astype(np.float64)
    top = contract["TopTimeMs"][nearest].astype(np.float64)
    supp_present = contract["SuppPresent"][nearest].astype(bool)
    time = pd.to_numeric(seg["TIME"], errors="coerce").to_numpy(dtype=np.float64)
    inside = supp_present & (time >= top_v) & (time < top)
    print(
        f"[supp-samples] inside={inside.sum():,}/{len(seg):,} ({inside.mean():.1%})  "
        f"nearest_trace_max={distance.max():.1f} m  "
        f"TopV={np.nanmin(top_v[inside]):.0f}-{np.nanmax(top_v[inside]):.0f}  "
        f"Top={np.nanmin(top[inside]):.0f}-{np.nanmax(top[inside]):.0f}"
    )
    if not inside.any():
        raise RuntimeError("no segment rows fall inside the supplementary layer")

    # 密度源：按 MD 最近邻合并
    density = pd.read_csv(args.density, low_memory=False)
    density.columns = [str(c).strip().lstrip("\ufeff") for c in density.columns]
    density["MD"] = pd.to_numeric(density["MD"], errors="coerce")
    density = density.dropna(subset=["MD"]).sort_values("MD")
    density_col = "P10_SUM_裂缝密度"
    if density_col not in density.columns:
        candidates = [c for c in density.columns if "P10" in c]
        if not candidates:
            raise ValueError(f"density column not found in {args.density}")
        density_col = candidates[0]
    density["DensityRaw"] = pd.to_numeric(density[density_col], errors="coerce")
    density["DensityRaw"] = density["DensityRaw"].where(density["DensityRaw"] > -9000.0)
    print(f"[supp-samples] density rows={len(density):,}  MD={density.MD.min():.1f}-{density.MD.max():.1f}")

    work = seg.loc[inside].copy()
    work["MD"] = pd.to_numeric(work["MD"], errors="coerce")
    work = work.sort_values("MD")
    merged = pd.merge_asof(
        work,
        density[["MD", "DensityRaw"]],
        on="MD",
        direction="nearest",
        tolerance=float(args.md_tolerance),
    )
    merged["DensityRaw"] = merged["DensityRaw"].fillna(0.0)
    merged["Density"] = merged["DensityRaw"]
    merged["StrataName"] = LAYER_NAME
    merged["LayerGroup"] = LAYER_NAME
    merged["LayerCode"] = LAYER_CODE
    merged["HasFractureDensity"] = (merged["Density"] > 0).astype(int)
    merged["InImagingInterval"] = 1
    merged["InHorizonLayer"] = 1
    merged["UseCase"] = "supp_both"
    merged["LayerGroupByHorizon"] = LAYER_NAME
    merged["LayerGroupByImaging"] = LAYER_NAME

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    group_columns = [
        "GroupID", "WellName", "InputSegmentPath", "MD", "TVD", "StrataName", "Density",
        "DensityRaw", "DensityScaleFactor", "ImagingWindowID", "HasFractureDensity",
        "GT_POINT_FLAG", "RawPointCount", "FracAzimuth", "FracDip", "DensityKind",
        "FractureScope", "DensitySourcePaths", "PointSourcePaths", "DensitySupportStatus",
        "SupervisionStatus", "SupervisionTier", "InImagingInterval", "InHorizonLayer",
        "UseCase", "LayerGroupByHorizon", "LayerGroupByImaging", "ImagingTVDMin", "ImagingTVDMax",
    ]
    group = pd.DataFrame({
        "GroupID": "埕北古斜405_supp_g001",
        "WellName": "埕北古斜405",
        "InputSegmentPath": str(seg_path),
        "MD": merged["MD"],
        "TVD": merged["TVD"],
        "StrataName": LAYER_NAME,
        "Density": merged["Density"],
        "DensityRaw": merged["DensityRaw"],
        "DensityScaleFactor": 1.0,
        "ImagingWindowID": "埕北古斜405_supp_W01",
        "HasFractureDensity": merged["HasFractureDensity"],
        "GT_POINT_FLAG": 0,
        "RawPointCount": 0,
        "FracAzimuth": np.nan,
        "FracDip": np.nan,
        "DensityKind": "P10_SUM_裂缝密度",
        "FractureScope": "supp_layer_proxy",
        "DensitySourcePaths": str(Path(args.density).resolve()),
        "PointSourcePaths": "",
        "DensitySupportStatus": "ok",
        "SupervisionStatus": "supp_ready",
        "SupervisionTier": "strong",
        "InImagingInterval": 1,
        "InHorizonLayer": 1,
        "UseCase": "supp_both",
        "LayerGroupByHorizon": LAYER_NAME,
        "LayerGroupByImaging": LAYER_NAME,
        "ImagingTVDMin": float(pd.to_numeric(merged["TVD"], errors="coerce").min()),
        "ImagingTVDMax": float(pd.to_numeric(merged["TVD"], errors="coerce").max()),
    })
    group_path = out_dir / "supp_imaging_group.csv"
    group.to_csv(group_path, index=False, encoding="utf-8-sig")

    sample_columns = [
        "SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName",
        "GR", "RD", "RS", "Density", "HasFracture", "LayerGroup", "LayerCode",
    ]
    samples = pd.DataFrame({
        "SampleID": [f"405_supp_{i:06d}" for i in range(len(merged))],
        "WellName": "埕北古斜405",
        "X": merged["X"],
        "Y": merged["Y"],
        "TIME": merged["TIME"],
        "TVD": merged["TVD"],
        "DEPT": merged["MD"],
        "StrataName": LAYER_NAME,
        "GR": merged.get("GR"),
        "RD": merged.get("RD"),
        "RS": merged.get("RS"),
        "Density": merged["Density"],
        "HasFracture": (merged["Density"] > 0).astype(int),
        "LayerGroup": LAYER_NAME,
        "LayerCode": LAYER_CODE,
    })
    sample_path = out_dir / "supp_real_well_samples.csv"
    samples.to_csv(sample_path, index=False, encoding="utf-8-sig")

    summary = {
        "status": "pass",
        "segment": str(seg_path),
        "density_source": str(Path(args.density).resolve()),
        "layer": LAYER_NAME,
        "layer_code": LAYER_CODE,
        "rows_segment": int(len(seg)),
        "rows_inside_supp": int(inside.sum()),
        "inside_fraction": float(inside.mean()),
        "density_positive_rows": int((merged["Density"] > 0).sum()),
        "md_range": [float(merged["MD"].min()), float(merged["MD"].max())],
        "time_range": [float(pd.to_numeric(merged["TIME"]).min()), float(pd.to_numeric(merged["TIME"]).max())],
        "topv_range": [float(np.nanmin(top_v[inside])), float(np.nanmax(top_v[inside]))],
        "top_range": [float(np.nanmin(top[inside])), float(np.nanmax(top[inside]))],
        "outputs": {"imaging_group": str(group_path), "real_well_samples": str(sample_path)},
    }
    (out_dir / "supp_samples_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[supp-samples] wrote {group_path}")
    print(f"[supp-samples] wrote {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
