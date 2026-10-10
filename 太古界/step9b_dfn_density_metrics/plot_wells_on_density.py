# -*- coding: utf-8 -*-
"""把井位 + 井名叠加到太古界分地层 P33 密度图上（ParaView 之外的展示方案）。

对应砂砾岩 ``plot_wells_on_p33.py``；坐标与方向约定一致：平面俯视，
X 自左向右增大，Y 自上向下减小（北在上），等比例显示。

用法::

    python3 太古界/step9b_dfn_density_metrics/plot_wells_on_density.py \
        --grid <p33_grid_300m.csv> --wells <well_trajectories_annotations.csv> \
        --vtk <well_trajectories_raw_time.vtk> --out-dir <输出目录>
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache-tyh")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

from dfn_density_metrics import (  # noqa: E402
    LAYERS,
    LAYER_SLUG,
    read_well_polylines,
    setup_cjk_font,
    strip_parens,
)


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_layer_maps(grid_rows, x_bins, y_bins, nx, ny, column="P33_mid"):
    """按地层把网格列回填成 (ny, nx) 数组；total 用厚度加权。"""
    maps, thick = {}, {}
    for layer in LAYERS:
        values = np.full((ny, nx), np.nan)
        thickness = np.full((ny, nx), np.nan)
        rows = [r for r in grid_rows if r["Layer"] == layer]
        if rows:
            xs = np.searchsorted(x_bins, [float(r["CellXMin"]) for r in rows])
            ys = np.searchsorted(y_bins, [float(r["CellYMin"]) for r in rows])
            values[ys, xs] = [float(r[column]) for r in rows]
            thickness[ys, xs] = [float(r["LayerThicknessMs"]) for r in rows]
        maps[layer], thick[layer] = values, thickness
    h_sum = thick[LAYERS[0]] + thick[LAYERS[1]]
    maps["total"] = (
        np.nan_to_num(maps[LAYERS[0]], nan=0.0) * np.nan_to_num(thick[LAYERS[0]], nan=0.0)
        + np.nan_to_num(maps[LAYERS[1]], nan=0.0) * np.nan_to_num(thick[LAYERS[1]], nan=0.0)
    ) / np.maximum(h_sum, 1e-9)
    return maps


def draw(values, wells, x_bins, y_bins, out_path, title, vmax, column_label):
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(
        values[::-1, :] * 1e6,
        origin="upper",
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
        cmap="viridis", vmin=0.0, vmax=vmax, aspect="equal",
    )
    for _, pts in wells["polylines"]:
        ax.plot(pts[:, 0], pts[:, 1], color="white", lw=1.0, alpha=0.6, zorder=3)
    for row in wells["rows"]:
        try:
            x, y = float(row["LabelX"]), float(row["LabelY"])
        except Exception:
            continue
        ax.scatter(x, y, s=30, marker="o", facecolor="crimson", edgecolor="white", linewidths=1.0, zorder=5)
        ax.annotate(row["WellName"], xy=(x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color="#111111", zorder=6,
                    path_effects=[withStroke(linewidth=2.5, foreground="white")])
    ax.set_title(strip_parens(title), fontsize=14, pad=12)
    ax.set_xlabel("X(m)")
    ax.set_ylabel("Y(m)")
    fig.colorbar(im, ax=ax, label=column_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("已生成：%s" % out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="井位叠加到太古界分地层 P33 密度图")
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--wells", type=Path, required=True)
    parser.add_argument("--vtk", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="太古界矿区")
    parser.add_argument("--column", default="P33_mid", help="用于成图的网格列（默认 P33_mid）")
    parser.add_argument("--vmax", type=float, default=None, help="统一色标上限（1e-6 单位）")
    args = parser.parse_args()

    setup_cjk_font()
    grid_rows = read_csv_rows(args.grid)
    x_bins = np.unique([float(r["CellXMin"]) for r in grid_rows])
    y_bins = np.unique([float(r["CellYMin"]) for r in grid_rows])
    nx, ny = len(x_bins), len(y_bins)
    maps = build_layer_maps(grid_rows, x_bins, y_bins, nx, ny, column=args.column)
    wells = {"rows": read_csv_rows(args.wells), "polylines": read_well_polylines(args.vtk)}
    print("井数：%d，折线数：%d，网格 %d x %d" % (len(wells["rows"]), len(wells["polylines"]), nx, ny))

    stacked = np.concatenate([maps[k][np.isfinite(maps[k]) & (maps[k] > 0)] for k in (*LAYERS, "total")])
    vmax = args.vmax if args.vmax is not None else float(np.percentile(stacked, 95)) * 1e6
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plans = [(LAYERS[0], LAYER_SLUG[LAYERS[0]], LAYERS[0]),
             (LAYERS[1], LAYER_SLUG[LAYERS[1]], LAYERS[1]),
             ("total", "total", "整体（厚度加权）")]
    for key, suffix, label in plans:
        draw(maps[key], wells, x_bins, y_bins,
             args.out_dir / f"p33_heatmap_{suffix}_wells.png",
             f"{args.title_prefix} P33 {label} + 井位", vmax,
             "P33 相对体密度（1e-6 · m³/(m²·ms)）")
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
