# -*- coding: utf-8 -*-
"""太古界 DFN 密度量化（P10 / P33）—— 移植砂砾岩 step9b。

口径与砂砾岩 ``优化阶段二/正式主线/step9b_dfn_density_metrics/dfn_density_metrics.py``
一致，只把"地层"换成太古界的两套目标层：

  上部复合层      T4(上部复合层顶) -> T6(太古界顶)
  太古界风化壳    T6(太古界顶)     -> T7(风化壳底)

计算内容：
  * P33（体密度）：300 m 网格 × 地层 × 尺度（small/medium/large），
    开度用幂律映射 ``a = a0 + (a1*factor - a0) * SourceDensity^0.5``，
    输出 0.5 / 1 / 2 mm 三档敏感性；
  * P10（线密度）：每口井轨迹半径内、各地层时间窗内的片数 ÷ 时间厚度；
  * 图件：分地层 P33 热图（平面俯视，北在上）+ 分尺度×地层柱状图 + 井位叠加图。

坐标与方向约定（与砂砾岩一致）：平面俯视；X（东）自左向右增大，
Y（北）自上向下减小（北在上）。

注意：砂砾岩的"成像标定"用的是车页1导眼成像 P33，属于砂砾岩专有数据；
太古界成像井（埕北305/310/313/816、桩斜169、桩海102）井位在矿区网格之外，
405 的成像解释段又整体落在目标层之上，故本版**不做**成像标定，只输出
DFN 相对体密度（口径已在 README 与 metadata 中注明）。

用法::

    python3 太古界/step9b_dfn_density_metrics/dfn_density_metrics.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
STEP8 = TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1"

DEFAULT_PATCH_CSV = STEP8 / "well_corrected_dfn_fracture_patches.csv"
DEFAULT_HORIZON = (
    TAIGU / "common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.npy"
)
DEFAULT_TRACE_HEADER = (
    TAIGU / "common/attribute_trace_contract/output/taigu_attribute_trace_header_v1/attribute_trace_header.csv"
)
DEFAULT_WELL_VTK = STEP8 / "well_trajectories_raw_time.vtk"
DEFAULT_OUTPUT_DIR = TAIGU / "step9b_dfn_density_metrics/output/taigu_quanqu1_v1"

LAYERS = ("上部复合层", "太古界风化壳")
# 地层 -> (顶界层位字段, 底界层位字段)
LAYER_HORIZONS = {"上部复合层": ("T4", "T6"), "太古界风化壳": ("T6", "T7")}
LAYER_SLUG = {"上部复合层": "upper", "太古界风化壳": "crust"}
SCALES = ("small", "medium", "large")
SCALE_APERTURE_RANGE_MM = {"small": (0.1, 1.0), "medium": (0.5, 2.0), "large": (1.0, 3.0)}
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)


def strip_parens(text: str) -> str:
    """去掉标题/坐标轴名称中的括号及其内容（含半角与全角括号）。"""
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"（[^）]*）", "", text)
    return re.sub(r"\s+", " ", text).strip()


def setup_cjk_font() -> str:
    """注册可用的中文字体，避免图上的中文缺字。"""
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
            except Exception:
                pass
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="太古界 DFN density metrics (P10/P33).")
    parser.add_argument("--patch-csv", type=Path, default=DEFAULT_PATCH_CSV)
    parser.add_argument("--horizon-npy", type=Path, default=DEFAULT_HORIZON)
    parser.add_argument("--trace-header", type=Path, default=DEFAULT_TRACE_HEADER)
    parser.add_argument("--well-vtk", type=Path, default=DEFAULT_WELL_VTK,
                        help="井轨迹 VTK（用于 P10）；不存在则跳过 P10")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-m", type=float, default=300.0)
    parser.add_argument("--well-radius-m", type=float, default=150.0)
    parser.add_argument("--heatmap-vmax-percentile", type=float, default=95.0,
                        help="未指定 --calibrated-vmax 时用于统一色标的百分位")
    return parser.parse_args()


def aperture_mm(scale: str, density: np.ndarray, factor: float) -> np.ndarray:
    """幂律等效开度：a0 + (a1*factor - a0) * SourceDensity^0.5（mm）。"""
    a0, a1 = SCALE_APERTURE_RANGE_MM[scale]
    dens = np.clip(np.nan_to_num(density, nan=0.5), 0.0, 1.0)
    return np.clip(a0 + (a1 * float(factor) - a0) * np.sqrt(dens), 0.0, None)


def build_grid_bins(x_min, x_max, y_min, y_max, grid_m):
    x_bins = np.arange(np.floor(x_min / grid_m) * grid_m, np.ceil(x_max / grid_m) * grid_m + grid_m, grid_m)
    y_bins = np.arange(np.floor(y_min / grid_m) * grid_m, np.ceil(y_max / grid_m) * grid_m + grid_m, grid_m)
    return x_bins, y_bins


def read_well_polylines(path: Path):
    """用 vtk 读井轨迹折线，返回 [(WellName, ndarray[N,3]), ...]。"""
    import vtk  # 局部导入：缺 vtk 时只跳过 P10

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    names = data.GetCellData().GetAbstractArray("WellName")
    polylines = []
    for i in range(data.GetNumberOfCells()):
        cell = data.GetCell(i)
        ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
        pts = np.array([data.GetPoint(j) for j in ids], dtype=np.float64)
        name = names.GetValue(i) if names is not None else f"Well{i}"
        polylines.append((str(name), pts))
    return polylines


def main() -> int:
    args = parse_args()
    font = setup_cjk_font()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- patches
    usecols = ["FractureScale", "LayerGroup", "CenterX", "CenterY", "CenterTime", "PatchAreaM2", "SourceDensity"]
    patches = pd.read_csv(args.patch_csv, usecols=usecols, low_memory=False)
    patches = patches.dropna(subset=["CenterX", "CenterY", "PatchAreaM2"]).copy()
    patches["_scale"] = patches["FractureScale"].astype(str).str.lower().str.strip()
    patches = patches[patches["_scale"].isin(SCALES)].copy()
    patches["_layer"] = patches["LayerGroup"].astype(str).str.strip()
    patches = patches[patches["_layer"].isin(LAYERS)].copy()
    patches["_dens"] = pd.to_numeric(patches["SourceDensity"], errors="coerce").fillna(0.5)
    patches["_area"] = pd.to_numeric(patches["PatchAreaM2"], errors="coerce")
    patches = patches[patches["_area"] > 0].copy()
    print(f"[step9b] 参与统计的裂缝片：{len(patches):,}（{', '.join(LAYERS)}；尺度 {', '.join(SCALES)}）")

    # ---------------------------------------------------------------- header
    header = pd.read_csv(args.trace_header, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    header = header.sort_values("TraceIdx").reset_index(drop=True)
    # 注意：OBN 道头坐标有 ±1 m 取整抖动，唯一值差分不能直接当道距；
    # 用"外接矩形面积 / 总道数"推单道平均面积（1650×1307 规则网 → 12.5×12.5=156.25 m²）。
    x_span = float(header["X"].max() - header["X"].min())
    y_span = float(header["Y"].max() - header["Y"].min())
    bin_area_m2 = x_span * y_span / float(len(header))
    print(f"[step9b] 道头 {len(header):,} 道，范围 {x_span:.0f} × {y_span:.0f} m，"
          f"单道平均面积 {bin_area_m2:.2f} m²（等效道距 {np.sqrt(bin_area_m2):.2f} m）")

    # ---------------------------------------------------------------- horizon
    horizon = np.load(args.horizon_npy, mmap_mode="r")
    if len(header) != len(horizon):
        raise ValueError(f"道头行数({len(header)}) != 层位契约行数({len(horizon)})")
    valid = np.asarray(horizon["SurfaceOrderValid"], dtype=bool)
    fields = {}
    for name in ("T4", "T6", "T7"):
        fields[name] = np.asarray(horizon[name], dtype=np.float64)
    thickness = {}
    for layer in LAYERS:
        top_field, base_field = LAYER_HORIZONS[layer]
        thick = fields[base_field] - fields[top_field]
        thick = np.where(valid, np.clip(thick, 0.0, None), np.nan)
        thickness[layer] = thick
        print(f"[step9b] {layer}: 有效道 {np.isfinite(thick).sum():,}，平均厚度 {np.nanmean(thick):.1f} ms")

    # ---------------------------------------------------------------- grid
    grid_m = float(args.grid_m)
    x_bins, y_bins = build_grid_bins(
        float(header["X"].min()), float(header["X"].max()),
        float(header["Y"].min()), float(header["Y"].max()), grid_m,
    )
    nx, ny = len(x_bins) - 1, len(y_bins) - 1
    h_ix = np.digitize(header["X"].to_numpy(dtype=np.float64), x_bins) - 1
    h_iy = np.digitize(header["Y"].to_numpy(dtype=np.float64), y_bins) - 1
    in_grid = (h_ix >= 0) & (h_ix < nx) & (h_iy >= 0) & (h_iy < ny)

    cell_trace_count = np.zeros((ny, nx), dtype=np.int64)
    np.add.at(cell_trace_count, (h_iy[in_grid], h_ix[in_grid]), 1)
    cell_area_m2 = cell_trace_count.astype(np.float64) * bin_area_m2
    cell_thick: dict[str, np.ndarray] = {}
    for layer in LAYERS:
        values = thickness[layer]
        ok = in_grid & np.isfinite(values)
        total = np.zeros((ny, nx), dtype=np.float64)
        count = np.zeros((ny, nx), dtype=np.int64)
        np.add.at(total, (h_iy[ok], h_ix[ok]), values[ok])
        np.add.at(count, (h_iy[ok], h_ix[ok]), 1)
        cell_thick[layer] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    print(f"[step9b] 网格 {nx} x {ny}（{grid_m:.0f} m），有效单元 {int((cell_trace_count > 0).sum())}")

    # ---------------------------------------------------------------- P33
    p_ix = np.digitize(patches["CenterX"].to_numpy(dtype=np.float64), x_bins) - 1
    p_iy = np.digitize(patches["CenterY"].to_numpy(dtype=np.float64), y_bins) - 1
    p_valid = (p_ix >= 0) & (p_ix < nx) & (p_iy >= 0) & (p_iy < ny)
    patches = patches[p_valid].copy()
    p_ix, p_iy = p_ix[p_valid], p_iy[p_valid]
    p_layer = patches["_layer"].to_numpy(dtype=object)
    p_scale = patches["_scale"].to_numpy(dtype=object)
    p_area = patches["_area"].to_numpy(dtype=np.float64)
    p_dens = patches["_dens"].to_numpy(dtype=np.float64)

    patch_count_cell = {layer: {s: np.zeros((ny, nx), dtype=np.int64) for s in SCALES} for layer in LAYERS}
    fracture_area_cell = {layer: np.zeros((ny, nx), dtype=np.float64) for layer in LAYERS}
    fracture_volume_cell = {
        f: {layer: {s: np.zeros((ny, nx), dtype=np.float64) for s in SCALES} for layer in LAYERS}
        for f in SENSITIVITY_FACTORS
    }
    for layer in LAYERS:
        lm = p_layer == layer
        np.add.at(fracture_area_cell[layer], (p_iy[lm], p_ix[lm]), p_area[lm])
        for scale in SCALES:
            sel = lm & (p_scale == scale)
            np.add.at(patch_count_cell[layer][scale], (p_iy[sel], p_ix[sel]), 1)
            for factor in SENSITIVITY_FACTORS:
                vol = p_area[sel] * (aperture_mm(scale, p_dens[sel], factor) / 1000.0)
                np.add.at(fracture_volume_cell[factor][layer][scale], (p_iy[sel], p_ix[sel]), vol)

    def cell_volume(layer):
        return cell_area_m2 * cell_thick[layer]

    p33 = {
        f: {
            layer: {
                s: np.where(cell_thick[layer] > 0,
                            fracture_volume_cell[f][layer][s] / np.maximum(cell_volume(layer), 1e-30),
                            0.0)
                for s in SCALES
            }
            for layer in LAYERS
        }
        for f in SENSITIVITY_FACTORS
    }
    p33_layer = {f: {layer: sum(p33[f][layer][s] for s in SCALES) for layer in LAYERS} for f in SENSITIVITY_FACTORS}
    p33_total = {f: p33_layer[f]["上部复合层"] + p33_layer[f]["太古界风化壳"] for f in SENSITIVITY_FACTORS}

    # ---------------------------------------------------------------- grid CSV
    rows: list[dict[str, Any]] = []
    for iy in range(ny):
        for ix in range(nx):
            if cell_trace_count[iy, ix] == 0:
                continue
            base = {
                "CellXMin": float(x_bins[ix]),
                "CellXMax": float(x_bins[ix + 1]),
                "CellYMin": float(y_bins[iy]),
                "CellYMax": float(y_bins[iy + 1]),
                "CellCenterX": float((x_bins[ix] + x_bins[ix + 1]) / 2.0),
                "CellCenterY": float((y_bins[iy] + y_bins[iy + 1]) / 2.0),
                "TraceCount": int(cell_trace_count[iy, ix]),
                "CellAreaM2": float(cell_area_m2[iy, ix]),
            }
            for layer in LAYERS:
                row = dict(base)
                row["Layer"] = layer
                row["LayerThicknessMs"] = float(cell_thick[layer][iy, ix])
                row["FractureAreaM2"] = float(fracture_area_cell[layer][iy, ix])
                for scale in SCALES:
                    row[f"PatchCount_{scale}"] = int(patch_count_cell[layer][scale][iy, ix])
                row["PatchCount_total"] = int(sum(patch_count_cell[layer][s][iy, ix] for s in SCALES))
                row["FractureVolumeM3_1mm"] = float(
                    sum(fracture_volume_cell[1.0][layer][s][iy, ix] for s in SCALES)
                )
                row["P33_low"] = float(p33_layer[0.5][layer][iy, ix])
                row["P33_mid"] = float(p33_layer[1.0][layer][iy, ix])
                row["P33_high"] = float(p33_layer[2.0][layer][iy, ix])
                rows.append(row)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(output_dir / "p33_grid_300m.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------- layer × scale summary
    summary_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        for scale in SCALES:
            sel = (p_layer == layer) & (p_scale == scale)
            summary_rows.append({
                "Layer": layer,
                "Scale": scale,
                "PatchCount": int(sel.sum()),
                "FractureAreaM2": float(p_area[sel].sum()),
                "FractureVolumeM3_low": float((aperture_mm(scale, p_dens[sel], 0.5) / 1000.0 * p_area[sel]).sum()),
                "FractureVolumeM3_mid": float((aperture_mm(scale, p_dens[sel], 1.0) / 1000.0 * p_area[sel]).sum()),
                "FractureVolumeM3_high": float((aperture_mm(scale, p_dens[sel], 2.0) / 1000.0 * p_area[sel]).sum()),
            })
        sel = p_layer == layer
        summary_rows.append({
            "Layer": layer,
            "Scale": "合计",
            "PatchCount": int(sel.sum()),
            "FractureAreaM2": float(p_area[sel].sum()),
            "FractureVolumeM3_low": float(sum(
                (aperture_mm(s, p_dens[sel & (p_scale == s)], 0.5) / 1000.0 * p_area[sel & (p_scale == s)]).sum()
                for s in SCALES)),
            "FractureVolumeM3_mid": float(sum(
                (aperture_mm(s, p_dens[sel & (p_scale == s)], 1.0) / 1000.0 * p_area[sel & (p_scale == s)]).sum()
                for s in SCALES)),
            "FractureVolumeM3_high": float(sum(
                (aperture_mm(s, p_dens[sel & (p_scale == s)], 2.0) / 1000.0 * p_area[sel & (p_scale == s)]).sum()
                for s in SCALES)),
        })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "p33_summary_by_layer_scale.csv", index=False, encoding="utf-8-sig"
    )

    # ---------------------------------------------------------------- P10
    p10_rows: list[dict[str, Any]] = []
    if args.well_vtk.exists():
        polylines = read_well_polylines(args.well_vtk)
        # 只拿"层位有效"的道做邻域查询：矿区边缘有 ~12% 的道缺层位，
        # 落到这些道上会得到假的零厚度层段。
        valid_mask = valid & np.isfinite(fields["T4"]) & np.isfinite(fields["T6"]) & np.isfinite(fields["T7"])
        if valid_mask.sum() == 0:
            raise ValueError("层位契约没有任何有效道，无法计算 P10")
        valid_fields = {k: fields[k][valid_mask] for k in ("T4", "T6", "T7")}
        xy_tree = cKDTree(np.column_stack(
            [header["X"].to_numpy()[valid_mask], header["Y"].to_numpy()[valid_mask]]
        ))
        patch_xy = patches[["CenterX", "CenterY"]].to_numpy(dtype=np.float64)
        patch_time = patches["CenterTime"].to_numpy(dtype=np.float64)
        for name, pts in polylines:
            tree = cKDTree(pts[:, :2])
            dist, _ = tree.query(patch_xy, k=1)
            near = dist <= float(args.well_radius_m)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            # 取井位邻域 25 道中位数，避免单道层位噪声影响层段时间窗
            _, neighbour = xy_tree.query([cx, cy], k=min(25, xy_tree.n))
            neighbour = np.atleast_1d(neighbour)
            t4 = float(np.median(valid_fields["T4"][neighbour]))
            t6 = float(np.median(valid_fields["T6"][neighbour]))
            t7 = float(np.median(valid_fields["T7"][neighbour]))
            for layer in LAYERS:
                top_field, base_field = LAYER_HORIZONS[layer]
                t0 = {"T4": t4, "T6": t6, "T7": t7}[top_field]
                t1 = {"T4": t4, "T6": t6, "T7": t7}[base_field]
                span = max(t1 - t0, 1.0)
                in_layer = near & (patch_time >= t0) & (patch_time <= t1)
                for scale in SCALES:
                    count = int((in_layer & (p_scale == scale)).sum())
                    p10_rows.append({
                        "Well": name, "Layer": layer, "Scale": scale,
                        "PatchCountNear": count, "LayerSpanMs": float(span),
                        "P10_per100ms": float(count / span * 100.0),
                    })
                count_all = int(in_layer.sum())
                p10_rows.append({
                    "Well": name, "Layer": layer, "Scale": "合计",
                    "PatchCountNear": count_all, "LayerSpanMs": float(span),
                    "P10_per100ms": float(count_all / span * 100.0),
                })
        pd.DataFrame(p10_rows).to_csv(output_dir / "p10_well_summary.csv", index=False, encoding="utf-8-sig")
    else:
        print(f"[step9b] 未找到井轨迹 VTK（{args.well_vtk}），跳过 P10")

    # ------------------------------------------------------------ heatmaps
    def draw_heatmap(values, title, filename, vmax=None, cbar_label="P33 相对体密度（1e-6 · m³/(m²·ms)）"):
        fig, ax = plt.subplots(figsize=(12.8, 9.2))
        positive = values[values > 0]
        if vmax is None:
            vmax = float(np.percentile(positive, args.heatmap_vmax_percentile)) if positive.size else 1.0
        im = ax.imshow(values[::-1, :], origin="upper",
                       extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                       cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-12), aspect="auto")
        ax.set_title(strip_parens(title))
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    draw_heatmap(p33_total[1.0] * 1e6, "P33 总密度（上部复合层 + 太古界风化壳，1 mm 档）", "p33_heatmap_total.png")
    for layer in LAYERS:
        draw_heatmap(
            p33_layer[1.0][layer] * 1e6,
            f"P33 {layer}（1 mm 档）",
            f"p33_heatmap_{LAYER_SLUG[layer]}.png",
        )

    # 三张图统一色标（分地层可直接对比）
    layer_max = [float(np.percentile(p33_layer[1.0][layer][p33_layer[1.0][layer] > 0],
                                     args.heatmap_vmax_percentile)) if (p33_layer[1.0][layer] > 0).any() else 1.0
                 for layer in LAYERS]
    unified = max(layer_max) * 1e6
    for layer in LAYERS:
        draw_heatmap(
            p33_layer[1.0][layer] * 1e6,
            f"P33 {layer}（1 mm 档，统一色标）",
            f"p33_heatmap_{LAYER_SLUG[layer]}_unified.png",
            vmax=unified,
        )

    # ---------------------------------------------------------------- bars
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    xpos = np.arange(len(SCALES))
    for li, layer in enumerate(LAYERS):
        vals = [p33[1.0][layer][s].sum() * 1e6 for s in SCALES]
        ax.bar(xpos + li * width, vals, width, label=layer)
    ax.set_xticks(xpos + width / 2)
    ax.set_xticklabels(SCALES)
    ax.set_ylabel(strip_parens("P33 合计 (1e-6 · m³/(m²·ms))"))
    ax.set_title(strip_parens("P33 分尺度 × 分地层（1 mm 档）"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "p33_scale_layer_bars.png", dpi=160)
    plt.close(fig)

    # 厚度加权的地层对比（每层平均 P33，单位 1e-4）
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_vals = []
    for layer in LAYERS:
        thick = cell_thick[layer]
        weight = cell_area_m2 * thick
        vol = sum(fracture_volume_cell[1.0][layer][s] for s in SCALES)
        bar_vals.append(float(vol.sum() / max(weight.sum(), 1e-30) * 1e6))
    ax.bar(list(LAYERS), bar_vals, color=["#f97316", "#22d3ee"])
    ax.set_ylabel(strip_parens("体积平均 P33（1e-6 · m³/(m²·ms)，1 mm 档）"))
    ax.set_title(strip_parens("分地层体积平均 P33 对比"))
    for i, v in enumerate(bar_vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "p33_layer_compare.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- metadata
    metadata = {
        "status": "pass",
        "version": "taigu_v1",
        "ported_from": "优化阶段二/正式主线/step9b_dfn_density_metrics/dfn_density_metrics.py",
        "inputs": {
            "patch_csv": str(args.patch_csv.resolve()),
            "horizon_npy": str(args.horizon_npy.resolve()),
            "trace_header": str(args.trace_header.resolve()),
            "well_vtk": str(args.well_vtk.resolve()) if args.well_vtk.exists() else None,
        },
        "layers": {
            layer: {"top": fields_top, "base": fields_base}
            for layer, (fields_top, fields_base) in LAYER_HORIZONS.items()
        },
        "grid_m": grid_m,
        "grid_shape": [nx, ny],
        "trace_bin_area_m2": bin_area_m2,
        "aperture_mapping": {
            "formula": "aperture_mm = a0 + (a1*factor - a0) * SourceDensity^0.5",
            "scale_ranges_mm": SCALE_APERTURE_RANGE_MM,
            "sensitivity_factors": list(SENSITIVITY_FACTORS),
            "note": "等效/名义开度，非实测；口径沿用砂砾岩，便于两边对比",
        },
        "volume_contract": {
            "thickness": "TWT 时间厚度（ms），网格内按有效道（SurfaceOrderValid=1）平均",
            "p33_units": "m^3/(m^2*ms)，图中放大 1e6",
        },
        "imaging_calibration": {
            "applied": False,
            "reason": "砂砾岩标定源（车页1导眼成像 P33）为砂砾岩专有；太古界成像井均在矿区网格之外，"
                      "405 的成像解释段又整体落在目标层之上，无法做同口径层段标定。",
            "todo": "若后续提供太古界成像 P33（或井周成像密度→P33 的换算），可在此加 --calibration-csv 复用砂砾岩代码路径。",
        },
        "display_orientation": "平面俯视；X 自左向右增大，Y 自上向下减小（北在上）",
        "counts": {
            "patches_used": int(len(patches)),
            "by_layer": {layer: int((p_layer == layer).sum()) for layer in LAYERS},
            "by_layer_scale": {
                layer: {s: int(((p_layer == layer) & (p_scale == s)).sum()) for s in SCALES} for layer in LAYERS
            },
        },
        "outputs": {
            "grid_csv": str(output_dir / "p33_grid_300m.csv"),
            "layer_scale_csv": str(output_dir / "p33_summary_by_layer_scale.csv"),
            "p10_csv": str(output_dir / "p10_well_summary.csv") if p10_rows else None,
            "heatmaps": [str(output_dir / f) for f in (
                "p33_heatmap_total.png", "p33_heatmap_upper.png", "p33_heatmap_crust.png",
                "p33_heatmap_upper_unified.png", "p33_heatmap_crust_unified.png")],
            "bars": [str(output_dir / "p33_scale_layer_bars.png"), str(output_dir / "p33_layer_compare.png")],
        },
        "font": font,
    }
    (output_dir / "dfn_density_metrics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata["counts"], ensure_ascii=False, indent=2))
    print(f"[step9b] 完成，输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
