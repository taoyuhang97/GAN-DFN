#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为太古界 Z 翻转展示包生成一张离线 3D 预览图（无需打开 ParaView 即可确认朝向）。

读展示包里的**翻转后** VTK（Z = -TWT），抽稀后画：
  * 井轨迹（黑线 + 井名）；
  * 井控校正后 DFN（按尺度上色）；
  * 原始断层片（黄色）。
俯视/侧视各一张，用来核对"浅层在上、深层在下"以及井与 DFN 的相对位置。

用法::

    python3 太古界/step8_well_correction/make_taigu_zflip_preview.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import vtk

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "太古界/dfn_display_zflip_quanqu1"


def register_chinese_font() -> str:
    """注册简体中文字面（与 Step9 同源缓存），否则图上的中文会缺字。

    Noto CJK 的 .ttc 里 face0=JP、face2=SC；Step9 已把 SC 字面抽成 otf 缓存在
    ``~/.cache/taigu_step9_fonts/``，这里直接复用；没有缓存时退回系统 ttc。
    """
    cache = sorted((Path.home() / ".cache/taigu_step9_fonts").glob("*.otf"))
    candidates = cache + [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in candidates:
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        name = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return name
    return "DejaVu Sans"


def read_polydata(path: Path):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    points = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())], dtype=float)
    return data, points


def cell_centers_and_layer(data) -> tuple[np.ndarray, np.ndarray]:
    """返回单元中心与 LayerCode。

    注：`FractureScaleCode` 在 well_corrected 的 VTK 里未写入（全为 -1），
    尺度信息在随包 CSV 的 `FractureScale` 列；预览这里用 VTK 里真实存在的
    `LayerCode`（1=上部复合层 / 2=太古界风化壳）上色，配色与 Step9 剖面图例一致。
    """
    n = data.GetNumberOfCells()
    centers = np.empty((n, 3), dtype=float)
    layers = np.zeros(n, dtype=int)
    layer_array = data.GetCellData().GetArray("LayerCode")
    for i in range(n):
        cell = data.GetCell(i)
        pts = np.array([cell.GetPoints().GetPoint(k) for k in range(cell.GetNumberOfPoints())])
        centers[i] = pts.mean(axis=0)
        if layer_array is not None:
            layers[i] = int(layer_array.GetTuple1(i))
    return centers, layers


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a 3D preview of the Z-flip bundle.")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--stride", type=int, default=40, help="DFN 单元抽稀步长")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    font_name = register_chinese_font()
    print(f"[预览] 中文字体：{font_name}", flush=True)
    dfn_path = bundle / "well_corrected_dfn_raw_time.vtk"
    fault_path = bundle / "original_fault_units_demo_raw_time.vtk"
    wells_path = bundle / "well_trajectories_raw_time.vtk"

    print(f"[预览] 读 {dfn_path.name}", flush=True)
    dfn, _ = read_polydata(dfn_path)
    centers, layers = cell_centers_and_layer(dfn)
    print(f"[预览] DFN 单元 {len(centers):,} → 抽稀 1/{args.stride}", flush=True)
    sel = slice(None, None, max(int(args.stride), 1))
    centers, layers = centers[sel], layers[sel]

    print(f"[预览] 读 {fault_path.name}", flush=True)
    fault, fault_points = read_polydata(fault_path)
    fault_points = fault_points[:: max(int(args.stride), 1)]

    print(f"[预览] 读 {wells_path.name}", flush=True)
    wells, well_points = read_polydata(wells_path)
    well_id = wells.GetCellData().GetArray("WellID")
    names = [
        wells.GetCellData().GetAbstractArray("WellName").GetValue(i)
        if wells.GetCellData().GetAbstractArray("WellName") is not None else ""
        for i in range(wells.GetNumberOfCells())
    ]

    # 与 Step9 剖面图例同色：上部复合层 = 橙、太古界风化壳 = 青
    layer_colors = {1: ("#f97316", "DFN裂缝片：上部复合层"), 2: ("#22d3ee", "DFN裂缝片：太古界风化壳")}
    fig = plt.figure(figsize=(17, 9))
    for panel, (elev, azim, title) in enumerate(
        [(22, -60, "斜视（东-北-时间）"), (2, -90, "侧视 X–Z（看深浅关系）")], start=1
    ):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        # 先画原始断层片（底图），再画 DFN（主角），否则 4.9 万个黄色点会把 DFN 盖住
        ax.scatter(fault_points[:, 0], fault_points[:, 1], fault_points[:, 2],
                   s=1.0, c="#ffe600", alpha=.35, linewidths=0, depthshade=False, label="原始断层片")
        for code, (color, label) in layer_colors.items():
            mask = layers == code
            if mask.any():
                ax.scatter(centers[mask, 0], centers[mask, 1], centers[mask, 2],
                           s=1.3, c=color, alpha=.55, linewidths=0, depthshade=False, label=label)
        other = ~np.isin(layers, list(layer_colors))
        if other.any():
            ax.scatter(centers[other, 0], centers[other, 1], centers[other, 2],
                       s=1.0, c="#94a3b8", alpha=.40, linewidths=0, depthshade=False, label="其它/井控补片")
        for i in range(wells.GetNumberOfCells()):
            ids = [wells.GetCell(i).GetPointId(k) for k in range(wells.GetCell(i).GetNumberOfPoints())]
            pts = well_points[ids]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", lw=1.1)
            top = pts[np.argmax(pts[:, 2])]
            ax.text(top[0], top[1], top[2], str(names[i]).replace("埕北", "CB"), fontsize=5.5, color="black")
        ax.set_title(f"太古界矿区 DFN（Z 已取反，浅层在上）\n{title}", fontsize=10)
        ax.set_xlabel("X / m", fontsize=8); ax.set_ylabel("Y / m", fontsize=8)
        ax.set_zlabel("Z / ms (= -TWT)", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    seen, uniq = set(), []
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen.add(label); uniq.append((handle, label))
    fig.legend([h for h, _ in uniq], [l for _, l in uniq], loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("太古界 quanqu1（矿区/全区）DFN 展示包预览 — 井控校正后多尺度 DFN + 井轨迹", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = bundle / "preview_zflip_3d.png"
    fig.savefig(out, dpi=170)
    print(f"[预览] 写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
