#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建"太古界上部补充层"的逐道层位契约（新增文件，不改现有契约）。

虚拟顶：
    TopV = max(TopTimeMs - 350 ms, 2650 ms)

输出（默认）::

    太古界/supp_layer/output/taigu_supp_horizon_v1/
      supp_horizon_contract.npy    # 全矿区，TraceIdx 对齐
      supp_horizon_contract.csv    # 仅 5×5 km 区块
      supp_horizon_summary.json

其中 npy 的字段与 taigu_supp_v1.json 里的 legacy_column 一致：
    TraceIdx, TopVTimeMs, TopTimeMs, MidTimeMs, BaseTimeMs,
    SurfaceOrderValid, SuppPresent, UpperPresent, CrustPresent,
    SuppThicknessMs, HorizonCorrectionCode

用法::

    python3 太古界/supp_layer/build_supp_horizon.py
    python3 太古界/supp_layer/build_supp_horizon.py --verify-well 埕北古斜405
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
FORMAL_ROOT = REPO / "优化阶段二" / "正式主线"
for path in (FORMAL_ROOT, FORMAL_ROOT / "common"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.strata_framework import load_framework_file  # noqa: E402


DEFAULT_SOURCE = (
    TAIGU / "common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.csv"
)
DEFAULT_HEADER = (
    TAIGU / "common/attribute_trace_contract/output/taigu_attribute_trace_header_v1/attribute_trace_header.csv"
)
DEFAULT_FRAMEWORK = TAIGU / "supp_layer/configs/taigu_supp_v1.json"
DEFAULT_OUTPUT = TAIGU / "supp_layer/output/taigu_supp_horizon_v1"
DEFAULT_WELL_VTK = (
    TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_trajectories_raw_time.vtk"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supplementary horizon contract (TopV above Top).")
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--trace-header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--framework-config", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--well-vtk", type=Path, default=DEFAULT_WELL_VTK)
    parser.add_argument("--verify-well", default="埕北古斜405")
    return parser.parse_args()


def load_well_polyline(path: Path, well_name: str) -> np.ndarray:
    import vtk

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    names = data.GetCellData().GetAbstractArray("WellName")
    for index in range(data.GetNumberOfCells()):
        if names is not None and str(names.GetValue(index)) == str(well_name):
            cell = data.GetCell(index)
            ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
            return np.array([data.GetPoint(j) for j in ids], dtype=np.float64)
    raise FileNotFoundError(f"well not found in VTK: {well_name}")


def main() -> int:
    args = parse_args()
    framework = load_framework_file(args.framework_config)
    virtual = dict(framework.metadata.get("virtual_top") or {})
    offset = float(virtual.get("offset_ms", 350.0))
    floor = float(virtual.get("floor_ms", 2650.0))
    block = dict(framework.metadata.get("block") or {})

    print(f"[supp-horizon] framework={framework.framework_id}")
    print(f"[supp-horizon] TopV = max(Top - {offset:.0f}, {floor:.0f})")

    source = pd.read_csv(
        args.source_contract,
        usecols=["TraceIdx", "X", "Y", "IX", "IY", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid",
                 "UpperThicknessMs", "CrustThicknessMs"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    source = source.sort_values("TraceIdx").reset_index(drop=True)
    print(f"[supp-horizon] source traces={len(source):,}")

    top = pd.to_numeric(source["TopTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    mid = pd.to_numeric(source["MidTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    base = pd.to_numeric(source["BaseTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    surface_valid = source["SurfaceValid"].fillna(0).to_numpy(dtype=bool)
    finite = np.isfinite(top) & np.isfinite(mid) & np.isfinite(base)
    valid = surface_valid & finite

    top_v = np.maximum(top - offset, floor)
    supp_thickness = top - top_v
    supp_present = valid & np.isfinite(supp_thickness) & (supp_thickness >= 1.0)

    dtype = np.dtype([
        ("TraceIdx", "<i4"),
        ("TopVTimeMs", "<f4"),
        ("TopTimeMs", "<f4"),
        ("MidTimeMs", "<f4"),
        ("BaseTimeMs", "<f4"),
        ("SurfaceOrderValid", "u1"),
        ("SuppPresent", "u1"),
        ("UpperPresent", "u1"),
        ("CrustPresent", "u1"),
        ("SuppThicknessMs", "<f4"),
        ("HorizonCorrectionCode", "u1"),
    ])
    table = np.zeros(len(source), dtype=dtype)
    table["TraceIdx"] = source["TraceIdx"].to_numpy(dtype=np.int64).astype(np.int32)
    table["TopVTimeMs"] = top_v.astype(np.float32)
    table["TopTimeMs"] = top.astype(np.float32)
    table["MidTimeMs"] = mid.astype(np.float32)
    table["BaseTimeMs"] = base.astype(np.float32)
    table["SurfaceOrderValid"] = valid.astype(np.uint8)
    table["SuppPresent"] = supp_present.astype(np.uint8)
    table["UpperPresent"] = valid.astype(np.uint8)
    table["CrustPresent"] = valid.astype(np.uint8)
    table["SuppThicknessMs"] = supp_thickness.astype(np.float32)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "supp_horizon_contract.npy"
    np.save(npy_path, table)
    print(f"[supp-horizon] wrote {npy_path} ({len(table):,} traces)")

    # 5×5 km 区块按 IX/IY 取完整的 401×401 矩形（6B 要求映射是完整矩形）。
    from scipy.spatial import cKDTree

    center_x = float(block.get("center_x", 665954.0))
    center_y = float(block.get("center_y", 4241759.0))
    xy_tree = cKDTree(source[["X", "Y"]].to_numpy(dtype=np.float64))
    _, center_pos = xy_tree.query([center_x, center_y], k=1)
    center_pos = int(np.asarray(center_pos).reshape(-1)[0])
    center_ix = int(source.loc[center_pos, "IX"])
    center_iy = int(source.loc[center_pos, "IY"])
    half_cells = int(block.get("half_cells", 200))
    ix_min = center_ix - half_cells
    ix_max = center_ix + half_cells
    iy_min = center_iy - half_cells
    iy_max = center_iy + half_cells
    block_mask = (
        (source["IX"] >= ix_min) & (source["IX"] <= ix_max)
        & (source["IY"] >= iy_min) & (source["IY"] <= iy_max)
    ).to_numpy()
    block_df = pd.DataFrame({
        "TraceIdx": source.loc[block_mask, "TraceIdx"].to_numpy(dtype=np.int64),
        "X": source.loc[block_mask, "X"].to_numpy(dtype=np.float64),
        "Y": source.loc[block_mask, "Y"].to_numpy(dtype=np.float64),
        "IX": source.loc[block_mask, "IX"].to_numpy(dtype=np.int64),
        "IY": source.loc[block_mask, "IY"].to_numpy(dtype=np.int64),
        "IXLocal": (source.loc[block_mask, "IX"] - ix_min).to_numpy(dtype=np.int64),
        "IYLocal": (source.loc[block_mask, "IY"] - iy_min).to_numpy(dtype=np.int64),
        "TopVTimeMs": top_v[block_mask],
        "TopTimeMs": top[block_mask],
        "MidTimeMs": mid[block_mask],
        "BaseTimeMs": base[block_mask],
        "SurfaceValid": valid[block_mask].astype(np.uint8),
        "SuppPresent": supp_present[block_mask].astype(np.uint8),
        "UpperPresent": valid[block_mask].astype(np.uint8),
        "CrustPresent": valid[block_mask].astype(np.uint8),
        "SuppThicknessMs": supp_thickness[block_mask],
    })
    block_csv = output_dir / "supp_horizon_contract_block.csv"
    block_df.to_csv(block_csv, index=False, encoding="utf-8-sig")
    print(f"[supp-horizon] wrote {block_csv} ({len(block_df):,} traces)")

    summary = {
        "status": "pass",
        "framework_id": framework.framework_id,
        "source_contract": str(args.source_contract.resolve()),
        "output_npy": str(npy_path),
        "output_block_csv": str(block_csv),
        "virtual_top": {"formula": f"max(TopTimeMs - {offset:g}, {floor:g})", "offset_ms": offset, "floor_ms": floor},
        "block": block,
        "block_grid": {
            "center_ix": center_ix,
            "center_iy": center_iy,
            "ix_min": ix_min,
            "ix_max": ix_max,
            "iy_min": iy_min,
            "iy_max": iy_max,
            "expected_cells": int((ix_max - ix_min + 1) * (iy_max - iy_min + 1)),
        },
        "counts": {
            "total_traces": int(len(source)),
            "surface_valid_traces": int(valid.sum()),
            "supp_present_traces": int(supp_present.sum()),
            "supp_present_fraction_of_valid": float(supp_present.sum() / max(valid.sum(), 1)),
            "block_traces": int(block_mask.sum()),
            "block_supp_present_traces": int((block_mask & supp_present).sum()),
        },
        "supp_thickness_ms": {
            "min": float(np.nanmin(supp_thickness[supp_present])) if supp_present.any() else None,
            "median": float(np.nanmedian(supp_thickness[supp_present])) if supp_present.any() else None,
            "max": float(np.nanmax(supp_thickness[supp_present])) if supp_present.any() else None,
        },
    }

    # 405 覆盖验证
    if args.verify_well and args.well_vtk.exists():
        points = load_well_polyline(args.well_vtk, args.verify_well)
        header = pd.read_csv(args.trace_header, usecols=["X", "Y"], encoding="utf-8-sig")
        from scipy.spatial import cKDTree

        tree = cKDTree(header[["X", "Y"]].to_numpy(dtype=np.float64))
        distance, nearest = tree.query(points[:, :2], k=1)
        idx = np.asarray(nearest, dtype=np.int64)
        time = points[:, 2]
        imaging = (time >= 2746.8) & (time <= 2866.2)
        margin = time[imaging] - top_v[idx[imaging]]
        inside = (
            (time[imaging] >= top_v[idx[imaging]])
            & (time[imaging] < top[idx[imaging]])
            & supp_present[idx[imaging]]
        )
        summary["well_check"] = {
            "well": args.verify_well,
            "imaging_time_ms": [2746.8, 2866.2],
            "imaging_points": int(imaging.sum()),
            "nearest_trace_distance_max_m": float(distance.max()),
            "inside_supp_layer": int(inside.sum()),
            "inside_fraction": float(inside.mean()) if imaging.any() else 0.0,
            "virtual_top_range_ms": [float(top_v[idx[imaging]].min()), float(top_v[idx[imaging]].max())],
            "top_range_ms": [float(top[idx[imaging]].min()), float(top[idx[imaging]].max())],
            "margin_ms_min": float(margin.min()) if margin.size else None,
            "margin_ms_median": float(np.median(margin)) if margin.size else None,
        }
        print(
            f"[supp-horizon] {args.verify_well}: imaging points={int(imaging.sum())}, "
            f"inside={int(inside.sum())} ({inside.mean():.1%}), "
            f"TopV={top_v[idx[imaging]].min():.0f}-{top_v[idx[imaging]].max():.0f}, "
            f"margin min/median={margin.min():.1f}/{np.median(margin):.1f} ms"
        )

    summary_path = output_dir / "supp_horizon_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[supp-horizon] wrote {summary_path}")
    return 0 if summary.get("well_check", {}).get("inside_fraction", 1.0) >= 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
