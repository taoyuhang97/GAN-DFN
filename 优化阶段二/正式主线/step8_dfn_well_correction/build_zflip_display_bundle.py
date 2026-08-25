# -*- coding: utf-8 -*-
"""批量生成 10km / 矿区两套展示文件的 Z 镜像修正版（Z 取反，Scale=1,1,-1）。

输出统一放到：
  <正式主线>/output/dfn_display_zflip/mine_v1/
  <正式主线>/output/dfn_display_zflip/10km_v2/

每个尺度目录内按展示用途扁平存放（single_scale 单独一个子目录），
并附带 README 说明文件来源与坐标系约定；井轨迹的标注 CSV 同步翻转 LabelZ。
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from apply_vtk_transform import transform_vtk


FORMAL_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = FORMAL_ROOT / "output" / "dfn_display_zflip"

MINE_SRC = FORMAL_ROOT / "output" / "formal_mine_multiscale_flow_v1"
DEMO10_SRC = FORMAL_ROOT / "step6b_demo_density_volume_3d" / "output" / "formal_demo_10km_multiscale_flow_v2"
DEMO10_SRC8 = FORMAL_ROOT / "step8_dfn_well_correction" / "output" / "formal_demo_10km_multiscale_flow_v2"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# (源文件绝对路径, 输出相对路径, 中文说明)
MINE_FILES = [
    (MINE_SRC / "step7a_small_scale/small_dfn_raw_time.vtk", "small_dfn_raw_time.vtk", "小尺度裂缝 DFN"),
    (MINE_SRC / "step7b_medium_scale/medium_dfn_raw_time.vtk", "medium_dfn_raw_time.vtk", "中尺度裂缝 DFN"),
    (MINE_SRC / "step7c_large_fault/large_fault_dfn_raw_time.vtk", "large_fault_dfn_raw_time.vtk", "大尺度断层 DFN"),
    (MINE_SRC / "step7c_large_fault/large_fault_dfn_patches_diagnostic_raw_time.vtk",
     "large_fault_dfn_patches_diagnostic_raw_time.vtk", "大尺度 DFN 面板诊断"),
    (MINE_SRC / "step7c_large_fault/large_inferred_fault_surfaces_raw_time.vtk",
     "large_inferred_fault_surfaces_raw_time.vtk", "推断断层面"),
    (MINE_SRC / "step7c_large_fault/large_inferred_known_fault_duplicates_raw_time.vtk",
     "large_inferred_known_fault_duplicates_raw_time.vtk", "已知断层重复推断面"),
    (MINE_SRC / "step7c_large_fault/original_fault_units_demo_raw_time.vtk",
     "original_fault_units_demo_raw_time.vtk", "原始断层片"),
    (MINE_SRC / "step7d_fused/fused_multiscale_dfn_raw_time.vtk",
     "fused_multiscale_dfn_raw_time.vtk", "多尺度融合 DFN（step7d）"),
    (MINE_SRC / "step8_well_correction/single_scale_dfn/large_scale_dfn_raw_time.vtk",
     "single_scale/large_scale_dfn_raw_time.vtk", "单尺度 DFN-大"),
    (MINE_SRC / "step8_well_correction/single_scale_dfn/medium_scale_dfn_raw_time.vtk",
     "single_scale/medium_scale_dfn_raw_time.vtk", "单尺度 DFN-中"),
    (MINE_SRC / "step8_well_correction/single_scale_dfn/small_scale_dfn_raw_time.vtk",
     "single_scale/small_scale_dfn_raw_time.vtk", "单尺度 DFN-小"),
    (MINE_SRC / "step8_well_correction/well_corrected_dfn_raw_time.vtk",
     "well_corrected_dfn_raw_time.vtk", "井控校正后最终 DFN（step8）"),
    (MINE_SRC / "step8_well_correction/well_trajectories_raw_time.vtk",
     "well_trajectories_raw_time.vtk", "井轨迹"),
]

DEMO10_FILES = [
    (DEMO10_SRC / "step6c_large/original_fault_panels_selected_raw_time.vtk",
     "original_fault_panels_selected_raw_time.vtk", "原始断层选中面板"),
    (DEMO10_SRC / "step6c_large/original_fault_units_demo_raw_time.vtk",
     "original_fault_units_demo_raw_time.vtk", "原始断层片"),
    (FORMAL_ROOT / "step7a_small_scale_dfn/output/formal_demo_10km_multiscale_flow_v2/small_dfn_raw_time.vtk",
     "small_dfn_raw_time.vtk", "小尺度裂缝 DFN"),
    (FORMAL_ROOT / "step7b_multiscale_initial_dfn/output/formal_demo_10km_multiscale_flow_v2/medium_dfn_raw_time.vtk",
     "medium_dfn_raw_time.vtk", "中尺度裂缝 DFN"),
    (FORMAL_ROOT / "step7c_large_fault_dfn/output/formal_demo_10km_multiscale_flow_v2/large_fault_dfn_raw_time.vtk",
     "large_fault_dfn_raw_time.vtk", "大尺度断层 DFN"),
    (FORMAL_ROOT / "step7c_large_fault_dfn/output/formal_demo_10km_multiscale_flow_v2/large_inferred_fault_surfaces_raw_time.vtk",
     "large_inferred_fault_surfaces_raw_time.vtk", "推断断层面"),
    (FORMAL_ROOT / "step7c_large_fault_dfn/output/formal_demo_10km_multiscale_flow_v2/large_inferred_known_fault_duplicates_raw_time.vtk",
     "large_inferred_known_fault_duplicates_raw_time.vtk", "已知断层重复推断面"),
    (FORMAL_ROOT / "step7c_large_fault_dfn/output/formal_demo_10km_multiscale_flow_v2/original_fault_units_demo_raw_time.vtk",
     "original_fault_units_demo_raw_time.vtk", "原始断层片（step7c）"),
    (FORMAL_ROOT / "step7d_multiscale_fused_dfn/output/formal_demo_10km_multiscale_flow_v2/fused_multiscale_dfn_raw_time.vtk",
     "fused_multiscale_dfn_raw_time.vtk", "多尺度融合 DFN（step7d）"),
    (DEMO10_SRC8 / "well_corrected_dfn_raw_time.vtk", "well_corrected_dfn_raw_time.vtk", "井控校正后最终 DFN（step8）"),
    (DEMO10_SRC8 / "well_trajectories_raw_time.vtk", "well_trajectories_raw_time.vtk", "井轨迹"),
]


def flip_annotations_csv(src_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(src_csv, encoding="utf-8-sig")
    for col in ("LabelZ", "TimeMinMs", "TimeMaxMs"):
        if col in df.columns:
            df[col] = -df[col].abs()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")


def build_scale(scale_name: str, files, src_csv: Path | None) -> dict:
    out_dir = OUT_ROOT / scale_name
    summary = []
    for src, rel, desc in files:
        if not src.exists():
            summary.append({"file": rel, "status": "MISSING", "source": str(src)})
            print("[%s] 缺失: %s" % (scale_name, src))
            continue
        out_path = out_dir / rel
        info = transform_vtk(str(src), str(out_path), scale=(1.0, 1.0, -1.0))
        summary.append({
            "file": rel,
            "desc": desc,
            "status": "ok",
            "source": str(src),
            "points": info["points"],
            "cells": info["cells"],
            "z_before": info["z_before"],
            "z_after": info["z_after"],
        })
        print("[%s] %-48s Z %7.1f~%7.1f -> %7.1f~%7.1f  (%d 点)"
              % (scale_name, rel, info["z_before"][0], info["z_before"][1],
                 info["z_after"][0], info["z_after"][1], info["points"]))
    if src_csv is not None:
        flip_annotations_csv(src_csv, out_dir / "well_trajectories_annotations.csv")
        print("[%s] 已生成翻转版井名标注 CSV（LabelZ 取反）" % scale_name)
    return {"scale": scale_name, "files": summary}


def write_readme(manifest: dict) -> None:
    lines = [
        "砂砾岩 DFN 展示文件 —— Z 镜像修正版",
        "======================================",
        "",
        "生成时间：2026-08-25",
        "变换：Z 轴取反（Scale = 1, 1, -1），即镜像修正；X/Y 不变。",
        "坐标约定：X/Y 米；Z = -|TWT| ms（负数，与原地震体负起算方向一致）。",
        "",
        "目录结构：",
        "  mine_v1/   矿区尺度（54 口井）",
        "  10km_v2/   10km demo（26 口井）",
        "",
        "文件说明：",
    ]
    seen = set()
    for scale in manifest:
        lines.append("")
        lines.append("--- %s ---" % scale["scale"])
        for item in scale["files"]:
            if item["file"] in seen:
                continue
            seen.add(item["file"])
            lines.append("  %-48s %s" % (item["file"], item.get("desc", "")))
    lines.append("")
    lines.append("well_trajectories_annotations.csv：井名标注锚点（LabelZ 已同步取反），")
    lines.append("配合 well_trajectories_raw_time.vtk 使用，标签会落在翻转后的井顶。")
    lines.append("")
    lines.append("属性保留说明：")
    lines.append("  翻转输出保留原始 VTK 的全部单元属性数组，包括：")
    lines.append("  - PatchAreaM2     裂缝片面积（m²）")
    lines.append("  - AzimuthDeg      走向方位（°）")
    lines.append("  - DipDeg          倾角（°）")
    lines.append("  - LayerCode       地层划分（沙三/沙四）")
    lines.append("  - FractureScaleCode 尺度（小/中/大）")
    lines.append("  ParaView 中可用 Coloring -> PatchAreaM2 / LayerCode / DipDeg 等按属性着色。")
    (OUT_ROOT / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []

    # 矿区 v1：step6c 与 step7c 的 original_fault_units_demo 若内容相同只保留一份
    step6 = MINE_SRC / "step6c_large/original_fault_units_demo_raw_time.vtk"
    step7 = MINE_SRC / "step7c_large_fault/original_fault_units_demo_raw_time.vtk"
    if step6.exists() and step7.exists() and md5(step6) == md5(step7):
        print("矿区 original_fault_units_demo 在 step6c/step7c 内容相同，只处理 step7c 一份")
    elif step6.exists():
        MINE_FILES.insert(0, (step6, "original_fault_units_demo_step6c_raw_time.vtk", "原始断层片（step6c 副本）"))

    manifest.append(build_scale(
        "mine_v1", MINE_FILES,
        MINE_SRC / "step8_well_correction/well_trajectories_annotations.csv",
    ))
    manifest.append(build_scale(
        "10km_v2", DEMO10_FILES,
        DEMO10_SRC8 / "well_trajectories_annotations.csv",
    ))
    write_readme(manifest)
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成，输出目录：%s" % OUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
