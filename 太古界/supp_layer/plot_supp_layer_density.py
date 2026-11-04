#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step9b（补充层）：5×5 km 区块内三层裂缝面积密度对比图。

从拼接后的 DFN patch CSV 统计每个 300 m 网格的
    FractureAreaDensity = Σ PatchAreaM2 / 300²
并输出三层并排热图（统一色标），用于和现有两层做空间对比。

输出（默认 ``太古界/supp_layer/output/taigu_supp_step9b_v1/``）：
  layer_area_density_grid_300m.csv
  layer_area_density_compare.png
  step9b_summary.json

用法::

    python3 太古界/supp_layer/plot_supp_layer_density.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
DEFAULT_INPUT = TAIGU / "supp_layer/output/taigu_supp_merged_v1/merged_dfn_patches.csv"
DEFAULT_OUTPUT = TAIGU / "supp_layer/output/taigu_supp_step9b_v1"
LAYERS = ("上部复合层", "太古界风化壳", "上部补充层")
BLOCK = (663450.0, 668450.0, 4239250.0, 4244250.0)
GRID_M = 300.0


def setup_font() -> None:
    from matplotlib import font_manager

    for path in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Step9b layer area-density comparison.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    setup_font()

    frame = pd.read_csv(
        args.input_csv,
        usecols=["LayerGroup", "CenterX", "CenterY", "PatchAreaM2"],
        low_memory=False,
    )
    x0, x1, y0, y1 = BLOCK
    x_bins = np.arange(np.floor(x0 / GRID_M) * GRID_M, np.ceil(x1 / GRID_M) * GRID_M + GRID_M, GRID_M)
    y_bins = np.arange(np.floor(y0 / GRID_M) * GRID_M, np.ceil(y1 / GRID_M) * GRID_M + GRID_M, GRID_M)
    nx, ny = len(x_bins) - 1, len(y_bins) - 1

    maps: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for layer in LAYERS:
        sub = frame[frame["LayerGroup"] == layer]
        sub = sub[
            sub["CenterX"].between(x0, x1) & sub["CenterY"].between(y0, y1)
        ]
        ix = np.digitize(sub["CenterX"].to_numpy(dtype=float), x_bins) - 1
        iy = np.digitize(sub["CenterY"].to_numpy(dtype=float), y_bins) - 1
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        area = np.zeros((ny, nx), dtype=float)
        np.add.at(area, (iy[valid], ix[valid]), sub["PatchAreaM2"].to_numpy(dtype=float)[valid])
        density = area / (GRID_M * GRID_M)
        maps[layer] = density
        for j in range(ny):
            for i in range(nx):
                if density[j, i] > 0:
                    rows.append({
                        "Layer": layer,
                        "CellXMin": float(x_bins[i]),
                        "CellXMax": float(x_bins[i + 1]),
                        "CellYMin": float(y_bins[j]),
                        "CellYMax": float(y_bins[j + 1]),
                        "FractureAreaM2": float(area[j, i]),
                        "AreaDensity": float(density[j, i]),
                    })
        print(f"[step9b] {layer}: patches={len(sub):,} cells={int((density>0).sum()):,} "
              f"mean={density[density>0].mean() if (density>0).any() else 0:.4f}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "layer_area_density_grid_300m.csv", index=False, encoding="utf-8-sig")

    stacked = np.concatenate([maps[layer][maps[layer] > 0] for layer in LAYERS if (maps[layer] > 0).any()])
    vmax = float(np.percentile(stacked, 95)) if stacked.size else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    for ax, layer in zip(axes, LAYERS):
        im = ax.imshow(
            maps[layer][::-1, :],
            origin="upper",
            extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
            cmap="viridis", vmin=0.0, vmax=vmax, aspect="equal",
        )
        ax.set_title(layer)
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
    fig.colorbar(im, ax=axes, label="裂缝面积密度 (m²/m²，300 m 网格，统一色标)", shrink=0.85)
    fig.suptitle("太古界 405 区块 分地层裂缝面积密度对比")
    fig.savefig(out_dir / "layer_area_density_compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": "pass",
        "input_csv": str(args.input_csv.resolve()),
        "block": {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1, "grid_m": GRID_M},
        "layers": {
            layer: {
                "cells_with_data": int((maps[layer] > 0).sum()),
                "mean_density": float(maps[layer][maps[layer] > 0].mean()) if (maps[layer] > 0).any() else 0.0,
                "max_density": float(maps[layer].max()),
            }
            for layer in LAYERS
        },
        "outputs": {
            "grid_csv": str(out_dir / "layer_area_density_grid_300m.csv"),
            "png": str(out_dir / "layer_area_density_compare.png"),
        },
    }
    (out_dir / "step9b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[step9b] wrote {out_dir / 'layer_area_density_compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
