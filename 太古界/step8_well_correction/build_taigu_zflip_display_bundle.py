#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成太古界 quanqu1（矿区/全区）**对外展示用**的 Z 镜像修正（真实 TIME）DFN 展示包。

## 为什么需要 Z 翻转

太古界全链条的第三维是 **TIME（TWT ms，向下为正）**：层位越深，数值越大。
而 ParaView / VTK 的默认习惯是 **Z 轴朝上**，于是直接加载 `*_raw_time.vtk` 会看到
"浅层在下面、深层在上面"的倒置模型。

砂砾岩正式主线已经踩过这个坑并固化成一步：`build_zflip_display_bundle.py`
（`Scale = 1, 1, -1`，即 Z 取反，X/Y 不变），产出的文件是"展示口径"，
坐标约定变成 **Z = -|TWT| ms**，与地震体负起算方向一致，加载即正。

本脚本把同一口径套到太古界 quanqu1 的 DFN 与井轨迹上，整理成一个自包含目录，
可直接拷到 Windows 机器上用 ParaView 展示。变换实现与
`优化阶段二/正式主线/step8_dfn_well_correction/apply_vtk_transform.py` 一致
（ReadAllScalars/Fields 全开，避免面积/倾角/地层等单元属性在入口丢失）。

## 用法

    python3 太古界/step8_well_correction/build_taigu_zflip_display_bundle.py

输出目录：`太古界/dfn_display_zflip_quanqu1/`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
OUT_ROOT = TAIGU / "dfn_display_zflip_quanqu1"

STEP7A = TAIGU / "step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_quanqu1"
STEP7B = TAIGU / "step7b_multiscale_initial_dfn/output/taigu_medium_v7_anttrack_led_quanqu1"
STEP7C = TAIGU / "step7c_large_fault_dfn/output/taigu_step7c_large_v3_attribute_v3_quanqu1"
STEP7D = TAIGU / "step7d_multiscale_fused_dfn/output/taigu_step7d_fused_v1_quanqu1"
STEP8 = TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1"
PARAVIEW_SCRIPT = TAIGU / "step8_well_correction/label_taigu_well_trajectories_paraview.py"

# (源文件, 输出相对路径, 中文说明, 是否属于"轻量展示"子集)
FILES: list[tuple[Path, str, str, bool]] = [
    (STEP8 / "well_corrected_dfn_raw_time.vtk", "well_corrected_dfn_raw_time.vtk",
     "井控校正后最终多尺度 DFN（Step8，对外展示主成果）", True),
    (STEP7D / "fused_multiscale_dfn_raw_time.vtk", "fused_multiscale_dfn_raw_time.vtk",
     "多尺度融合 DFN（Step7D，井控校正前）", True),
    (STEP7B / "medium_dfn_raw_time.vtk", "medium_dfn_raw_time.vtk", "中尺度裂缝 DFN（Step7B）", True),
    (STEP7C / "large_fault_dfn_raw_time.vtk", "large_fault_dfn_raw_time.vtk", "大尺度断层 DFN（Step7C）", True),
    (STEP7C / "original_fault_units_demo_raw_time.vtk", "original_fault_units_demo_raw_time.vtk",
     "原始断层片（甲方解释，未预测）", True),
    (STEP7A / "fracture_patches.vtk", "small_dfn_raw_time.vtk", "小尺度裂缝 DFN（Step7A）", False),
    (STEP7C / "large_inferred_fault_surfaces_raw_time.vtk", "large_inferred_fault_surfaces_raw_time.vtk",
     "推断断层面（Step7C）", False),
    (STEP7C / "large_inferred_known_fault_duplicates_raw_time.vtk",
     "large_inferred_known_fault_duplicates_raw_time.vtk", "与已知断层重复的推断面（Step7C，诊断用）", False),
]

WELL_FILES = [
    (STEP8 / "well_trajectories_raw_time.vtk", "well_trajectories_raw_time.vtk", "井轨迹（每条井一条折线）", True),
    (STEP8 / "well_heads_raw_time.vtk", "well_heads_raw_time.vtk", "井顶点（每井一个点，含临时时深标记）", True),
]

FLIP_SCALE = (1.0, 1.0, -1.0)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_non_finite(data) -> int:
    """把浮点属性里的 NaN/±inf 就地替换为 0，返回替换个数。

    legacy ASCII VTK 无法用字面量表达非有限值：写出端会把 NaN 打印成字面量
    ``nan``，而读取端（含 ParaView）解析浮点数组时遇到 ``nan`` 会错位，
    进而丢掉紧随其后的 FIELD 数组并报 *Unsupported point attribute type*。
    上游个别文件确实带有全 NaN 属性（如 ``original_fault_units_demo`` 的
    ``FaultPatchArea``），所以在写 ASCII 前统一净化，保证 ParaView 打开无报错。
    """
    replaced = 0
    for dataset in (data.GetPointData(), data.GetCellData()):
        for index in range(dataset.GetNumberOfArrays()):
            array = dataset.GetArray(index)
            if array is None or array.GetDataType() not in (vtk.VTK_FLOAT, vtk.VTK_DOUBLE):
                continue
            values = vtk_to_numpy(array)
            mask = ~np.isfinite(values)
            if mask.any():
                replaced += int(mask.sum())
                values[mask] = 0.0
                array.Modified()
    return replaced


def transform_vtk(src: Path, dst: Path, scale=FLIP_SCALE) -> dict:
    """把几何变换烘焙进 legacy VTK（与砂砾岩 apply_vtk_transform.py 同实现）。"""
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(src))
    # 默认只读活动标量；面积/倾角/地层等属性必须在入口全部打开，否则会被丢掉
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllNormalsOn()
    reader.ReadAllTCoordsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    if data.GetNumberOfPoints() == 0:
        raise RuntimeError(f"输入为空：{src}")
    before = data.GetBounds()[4:6]

    transform = vtk.vtkTransform()
    transform.Scale(*scale)
    if isinstance(data, vtk.vtkPolyData):
        flt = vtk.vtkTransformPolyDataFilter()
    else:
        flt = vtk.vtkTransformFilter()
    flt.SetInputData(data)
    flt.SetTransform(transform)
    flt.Update()
    out = flt.GetOutput()
    non_finite = _sanitize_non_finite(out)

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(str(dst))
    writer.SetInputData(out)
    writer.SetFileTypeToASCII()
    writer.Write()
    after = out.GetBounds()[4:6]
    return {
        "points": int(out.GetNumberOfPoints()),
        "cells": int(out.GetNumberOfCells()),
        "non_finite_replaced": non_finite,
        "z_before": [float(before[0]), float(before[1])],
        "z_after": [float(after[0]), float(after[1])],
        "point_data": [out.GetPointData().GetArrayName(i) for i in range(out.GetPointData().GetNumberOfArrays())],
        "cell_data": [out.GetCellData().GetArrayName(i) for i in range(out.GetCellData().GetNumberOfArrays())],
    }


def flip_annotations(src: Path, dst: Path) -> pd.DataFrame:
    """井名锚点 CSV 的 Z 同步取反（LabelZ / TimeMinMs / TimeMaxMs）。"""
    frame = pd.read_csv(src, encoding="utf-8-sig")
    # 取反后"Min/Max"的数值大小关系会反过来（浅层 = 更大的负数），容易误读；
    # 因此额外保留一列原始正值口径，供人工核对（ParaView 脚本只用 LabelZ）。
    for column in ("TimeMinMs", "TimeMaxMs"):
        if column in frame.columns:
            frame[f"{column}Original"] = frame[column]
    for column in ("LabelZ", "TimeMinMs", "TimeMaxMs"):
        if column in frame.columns:
            frame[column] = -frame[column].abs()
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dst, index=False, encoding="utf-8")
    return frame


def write_readme(records: list[dict], well_frame: pd.DataFrame, out_dir: Path) -> None:
    ok = [row for row in records if row["status"] == "ok"]
    light = [row for row in ok if row["light"]]
    z_all = [row["z_after"] for row in ok]
    text = f"""# 太古界矿区（quanqu1）DFN 展示包 —— 真实 TIME 域 / Z 镜像修正版

口径：**太古界全链条**（Step7A 小尺度 → Step7B 中尺度 → Step7C 大尺度 →
Step7D 多尺度融合 → Step8 井控校正），范围 = 矿区/全区
X 658887–679500 m / Y 4231662–4247987 m（20.6 × 16.3 km，2,156,550 道）。

## 1. 坐标口径（务必先读）

| 项 | 说明 |
|---|---|
| X / Y | 投影坐标，单位米（与层位、地震体一致） |
| **Z** | **TWT 时间，单位 ms，已取反：Z = -\\|TWT\\|（负数）** |
| 为什么取反 | 原始 `*_raw_time.vtk` 的 Z 是"向下为正"的 TWT，而 ParaView 默认 Z 朝上 → 直接加载会**上下颠倒**（浅层在下）。砂砾岩正式主线固化了这一步：`Scale = (1, 1, -1)`，取反后浅层 Z 大、深层 Z 小，加载即正 |
| 变换实现 | 与 `优化阶段二/正式主线/step8_dfn_well_correction/apply_vtk_transform.py` 完全一致；PIPELINE 保留了**全部单元属性**（面积/倾向倾角/地层/尺度等） |
| 原始未翻转文件 | 留在各步骤 `output/` 目录里，命名同为 `*_raw_time.vtk`，本包不做覆盖 |

本包 Z 范围：**{min(z[0] for z in z_all):.1f} ~ {max(z[1] for z in z_all):.1f} ms**（取反后为负值）。

> 非有限值说明：个别上游文件的浮点属性含 NaN（例如 `original_fault_units_demo_raw_time.vtk`
> 的 `FaultPatchArea` 在本矿区 `step7c` 输出里全为 NaN）。legacy ASCII VTK 无法用字面量表达
> NaN，写出后会变成字面量 `nan`，读取端（含 ParaView）解析时会错位并丢掉后续 FIELD 数组。
> 因此生成本包时在写 ASCII 前已把所有非有限值统一替换为 `0`，保证 ParaView 打开无报错；
> 对可展示的属性（面积/倾角/地层/尺度等）无影响。

## 2. 文件清单

### 2.1 DFN 与断层

| 文件 | 内容 | 轻量展示 |
|---|---|---|
{chr(10).join(f"| `{row['file']}` | {row['desc']} | {'✅' if row['light'] else '—'} |" for row in ok)}

### 2.2 井轨迹与井名标注

| 文件 | 内容 |
|---|---|
| `well_trajectories_raw_time.vtk` | 井轨迹折线，单元属性 `WellID`，场数据 `WellName/WellNameASCII` |
| `well_heads_raw_time.vtk` | 井顶点（点属性 `WellID/TemporaryTimeDepth`） |
| `well_trajectories_annotations.csv` | 井名与标签锚点（`LabelZ` 已同步取反），{len(well_frame)} 口井 |

> `well_trajectories_annotations.csv` 的 `TimeMinMs/TimeMaxMs` 与砂砾岩展示包一样**同步取反**
> （保持"全表同一坐标口径"），取反后浅层是更大的负数、数值大小关系与原名相反；
> 因此另附 `TimeMinMsOriginal/TimeMaxMsOriginal` 两列保留**原始正值**，便于人工核对。
> ParaView 脚本只使用 `LabelX/Y/Z`。

本包共 **{len(well_frame)} 口井**（均在矿区网格范围内）：
{", ".join(well_frame["WellName"].astype(str).tolist())}

> 说明：太古界的成像测井井（埕北305/310/313/816、桩斜169、桩海102）井位在
> Y 4.226–4.232 M，**位于本次矿区网格南侧之外**，因此不在本展示包内；
> 埕北古斜405 在范围内，其 `temporary_neighbor_time_depth=1`（时深为临时借用）。

## 3. ParaView 怎么用（推荐流程）

1. 把本目录整个拷到 Windows（**建议英文/数字路径**，例如 `D:\\taigu_dfn\\`）。
2. `File → Open` 依次加载需要的 `.vtk`（都是 legacy ASCII POLYData，直接可读：
   `*_raw_time.vtk` 是 POLYDATA；`*_fault*` 若为 UNSTRUCTURED_GRID 也直接可读）。
3. 想看井名：`Tools → Python Shell → Load Script` 选
   `label_taigu_well_trajectories_paraview.py`，它会**自动读取同目录**的
   `well_trajectories_raw_time.vtk` + `well_trajectories_annotations.csv`，
   画出井轨迹 + 井顶点球 + ASCII 井名（中文名已转拼音，避免 3D Text 不支持中文）。
4. 着色：选中 DFN 对象 → `Coloring` 可选
   `PatchAreaM2 / DipDeg / DipAzimuthDeg / LayerCode / FractureScaleCode` 等属性。

### 轻量展示组合（约 {sum(Path(out_dir / row['file']).stat().st_size for row in light) / 1e6:.0f} MB）

机器紧张或只想讲主线时，只需拷这几件：
`well_corrected_dfn_raw_time.vtk` + `fused_multiscale_dfn_raw_time.vtk` +
`medium_dfn_raw_time.vtk` + `large_fault_dfn_raw_time.vtk` +
`original_fault_units_demo_raw_time.vtk` + 井轨迹三件 + 本脚本。

## 4. 生成方式（可复现）

```bash
# 1) 先导出井轨迹（若 Step8 目录里还没有）
python3 太古界/step8_well_correction/export_taigu_well_trajectories_vtk.py \\
  --config 太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_quanqu1.json

# 2) 生成本展示包（Z 翻转 + 打包 + README/manifest）
python3 太古界/step8_well_correction/build_taigu_zflip_display_bundle.py
```

生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Taigu quanqu1 Z-flip display bundle.")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--skip-files", action="store_true", help="只做井轨迹与标注（跳过 DFN 大文件）")
    args = parser.parse_args()

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[展示包] 输出目录：{out_root}")

    records: list[dict] = []
    todo = FILES + WELL_FILES
    for src, rel, desc, light in todo:
        is_well = (src, rel, desc, light) in WELL_FILES
        if args.skip_files and not is_well:
            continue
        if not src.exists():
            records.append({"file": rel, "desc": desc, "light": light, "status": "MISSING", "source": str(src)})
            print(f"  [缺失] {src}")
            continue
        dst = out_root / rel
        info = transform_vtk(src, dst)
        records.append({
            "file": rel, "desc": desc, "light": light, "status": "ok",
            "source": str(src.relative_to(REPO)), "source_md5": md5(src),
            "output_md5": md5(dst), "bytes": dst.stat().st_size, **info,
        })
        print(f"  [OK] {rel:52s} Z {info['z_before'][0]:8.1f}~{info['z_before'][1]:8.1f}"
              f"  ->  {info['z_after'][0]:9.1f}~{info['z_after'][1]:9.1f}  ({info['points']:,} 点)"
              + (f"  [非有限值→0: {info['non_finite_replaced']}]" if info.get("non_finite_replaced") else ""))

    print("[展示包] 翻转井名锚点 CSV")
    well_frame = flip_annotations(
        STEP8 / "well_trajectories_annotations.csv", out_root / "well_trajectories_annotations.csv"
    )
    print(f"  [OK] well_trajectories_annotations.csv（{len(well_frame)} 口井，LabelZ 已取反）")

    if PARAVIEW_SCRIPT.exists():
        shutil.copy2(PARAVIEW_SCRIPT, out_root / PARAVIEW_SCRIPT.name)
        print(f"  [OK] {PARAVIEW_SCRIPT.name}（ParaView 井名标注脚本，同目录自动读数据）")

    write_readme(records, well_frame, out_root)
    manifest = {
        "bundle": "taigu_quanqu1_zflip_display",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extent": {"x": [658887.0, 679500.0], "y": [4231662.0, 4247987.0], "trace_count": 2156550},
        "coordinate_contract": {
            "xy": "metres, projected; same as horizons/seismic",
            "z": "TWT ms, negated for display: Z = -|TWT| (see README)",
            "transform": "Scale = (1, 1, -1); mirrors 优化阶段二/.../apply_vtk_transform.py",
        },
        "well_count": int(len(well_frame)),
        "wells": well_frame["WellName"].astype(str).tolist(),
        "files": records,
        "paraview_script": PARAVIEW_SCRIPT.name,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[展示包] README.md + manifest.json 已写出；共 {len([r for r in records if r['status']=='ok'])} 个几何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
