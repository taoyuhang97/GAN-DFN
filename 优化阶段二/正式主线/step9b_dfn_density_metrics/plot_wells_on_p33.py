# -*- coding: utf-8 -*-
"""把井位 + 中文井名叠加到成像标定 P33 密度图上（ParaView 不可用时的展示方案）。

用法：
  python3 plot_wells_on_p33.py --grid <p33_grid_300m.csv> \
      --wells <well_trajectories_annotations.csv> --out-dir <输出目录> \
      --title-prefix "砂砾岩矿区"

坐标与方向约定和 step9b 热图一致：X 向左->右增大，北在上（Y 上->下减小），
等比例显示，成像标定 P33 单位 10⁻⁴ · m³/m³，统一色标。
"""

import argparse
import csv
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache-tyh")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patheffects import withStroke
import numpy as np


LAYERS = ("沙三段", "沙四段")
LAYER_KEYS = (("沙三段", "sha3", "沙三段"), ("沙四段", "sha4", "沙四段"), ("total", "total", "整体（厚度加权）"))


def strip_parens(text):
    """去掉标题/坐标轴名称中的括号及其内容（含半角与全角括号）。"""
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"（[^）]*）", "", text)
    return re.sub(r"\s+", " ", text).strip()


def setup_cjk_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.sans-serif"] = [family]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return family
    return None


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_calibrated_maps(grid_df, x_bins, y_bins, nx, ny):
    cal_maps = {}
    thick_maps = {}
    for layer in LAYERS:
        layer_df = [r for r in grid_df if r["Layer"] == layer]
        cal = np.full((ny, nx), np.nan)
        thick = np.full((ny, nx), np.nan)
        if layer_df:
            xs = np.searchsorted(x_bins, [float(r["CellXMin"]) for r in layer_df])
            ys = np.searchsorted(y_bins, [float(r["CellYMin"]) for r in layer_df])
            cal[ys, xs] = [float(r["P33_imaging_calibrated"]) for r in layer_df]
            thick[ys, xs] = [float(r["LayerThicknessMs"]) for r in layer_df]
        cal_maps[layer] = cal
        thick_maps[layer] = thick
    h_sum = thick_maps["沙三段"] + thick_maps["沙四段"]
    total = (
        cal_maps["沙三段"] * thick_maps["沙三段"] + cal_maps["沙四段"] * thick_maps["沙四段"]
    ) / np.maximum(h_sum, 1e-9)
    cal_maps["total"] = total
    return cal_maps


def draw_overlay(values, wells, out_path, title, vmax, family):
    fig, ax = plt.subplots(figsize=(12, 9))
    display = values[::-1, :]
    # 色标统一用 10⁻⁴ · m³/m³，与 step9b 成图一致
    im = ax.imshow(
        display * 1.0e4,
        origin="upper",
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    # 井轨迹平面投影（细线）
    for pl in wells.get("polylines", []):
        ax.plot(pl[:, 0], pl[:, 1], color="white", lw=1.0, alpha=0.55, zorder=3)
    # 井位与中文井名
    for row in wells["rows"]:
        try:
            x = float(row["LabelX"])
            y = float(row["LabelY"])
        except Exception:
            continue
        ax.scatter(x, y, s=30, marker="o", facecolor="crimson",
                   edgecolor="white", linewidths=1.0, zorder=5)
        ax.annotate(
            row["WellName"],
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="#111111",
            zorder=6,
            path_effects=[withStroke(linewidth=2.5, foreground="white")],
        )
    ax.set_aspect("equal")
    ax.set_title(strip_parens(title), fontsize=14, pad=12)
    ax.set_xlabel("X(m)")
    ax.set_ylabel("Y(m)")
    fig.colorbar(im, ax=ax, label="成像标定 P33（1e-4 · m³/m³）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("已生成：%s" % out_path)


def read_vtk_polylines(path):
    """读取 legacy ASCII POLYDATA，返回每条折线 (N,3) 数组列表（按单元顺序）。"""
    with open(path, "r", encoding="utf-8") as handle:
        lines = [ln.rstrip("\n") for ln in handle]
    points = []
    polylines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("POINTS"):
            count = int(line.split()[1])
            pts = np.array(
                [[float(v) for v in lines[i + j].split()[:3]] for j in range(count)],
                dtype=np.float64,
            )
            points = pts
            i += count
        elif line.startswith("LINES"):
            _, m, _ = line.split()
            for _ in range(int(m)):
                idx = [int(v) for v in lines[i].split()[1:]]
                polylines.append(points[np.asarray(idx, dtype=int)])
                i += 1
        elif line.startswith("CELL_DATA"):
            break
    return polylines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--wells", required=True)
    parser.add_argument("--vtk", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-prefix", default="")
    parser.add_argument("--vmax", type=float, default=3.15)
    args = parser.parse_args()

    global x_bins, y_bins
    family = setup_cjk_font()
    grid_df = read_csv_rows(args.grid)
    x_bins = np.unique([float(r["CellXMin"]) for r in grid_df])
    y_bins = np.unique([float(r["CellYMin"]) for r in grid_df])
    nx, ny = len(x_bins), len(y_bins)
    cal_maps = build_calibrated_maps(grid_df, x_bins, y_bins, nx, ny)

    well_rows = read_csv_rows(args.wells)
    polylines = read_vtk_polylines(args.vtk)
    wells = {"rows": well_rows, "polylines": polylines}
    print("井数：%d，折线数：%d，网格 %d x %d，字体：%s" % (len(well_rows), len(polylines), nx, ny, family))

    os.makedirs(args.out_dir, exist_ok=True)
    for key, suffix, layer_name in LAYER_KEYS:
        title = "%s成像标定 P33 %s + 井位" % (args.title_prefix, layer_name)
        out_path = os.path.join(args.out_dir, "p33_imaging_calibrated_%s_wells.png" % suffix)
        draw_overlay(cal_maps[key], wells, out_path, title, args.vmax, family)
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
