# -*- coding: utf-8 -*-
"""DFN density metrics (P10 / P21 / P33) for reporting and QC.

Computes quantitative fracture density from the Step8 final DFN:
  - P33 (volumetric): per 1 km grid cell and per layer (Sha3 / Sha4) and per
    scale (small / medium / large), using a power-law aperture mapping from
    the per-patch SourceDensity intensity. Three aperture sensitivities are
    reported (low / mid / high factors).
  - P10 (linear): along a well trajectory (patches within a horizontal radius
    divided by the layer time span).
  - P21 (areal): reserved; requires the Step9 section scan and is not produced
    in this first version.

Display orientation (per design record 2026-08-24): plan view from above,
X (easting) increases left -> right, Y (northing) decreases top -> bottom
(north at top). Heatmaps are drawn with imshow(origin="upper") and an
inverted Y axis to enforce this.

Outputs: p33_grid_1km.csv, p33_summary_by_region.csv, p10_well_summary.csv,
p33_heatmap_{total,sha3,sha4}.png, p33_scale_layer_bars.png,
p33_region_layer_compare.png, dfn_density_metrics_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FORMAL_ROOT = Path(__file__).resolve().parent.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    HorizonSpatialLookup,
)


DEFAULT_PATCH_CSV = (
    FORMAL_ROOT
    / "output/formal_mine_multiscale_flow_v1/step8_well_correction/well_corrected_dfn_fracture_patches.csv"
)
DEFAULT_HORIZON_TABLE = (
    FORMAL_ROOT
    / "common/horizon_trace_table/output/formal_horizon_trace_table_v2_mine/horizon_trace_table.npy"
)
DEFAULT_TRACE_HEADER = FORMAL_ROOT.parents[1] / "数据精简/trace_header_xy.csv"
DEFAULT_WELL_CSV = (
    FORMAL_ROOT
    / "step2_real_well_t4_t7_samples/output/formal_all_wells/车页1导眼/车页1导眼_t4_t7_real_well_main.csv"
)
DEFAULT_OUTPUT_DIR = (
    FORMAL_ROOT
    / "output/step9b_dfn_density_metrics/formal_mine_multiscale_flow_v1"
)

SCALES = ("small", "medium", "large")
LAYERS = ("沙三段", "沙四段")
SCALE_APERTURE_RANGE_MM = {"small": (0.1, 1.0), "medium": (0.5, 2.0), "large": (1.0, 3.0)}
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)
CALIBRATION_LABELED_CSV = (
    "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/"
    "LSTM/地层划分验证/exp_strata_ac_gr_v1/labeled_datasets/车页1导眼_sample_labeled.csv"
)
CALIBRATION_WELL_XY = (578803.455, 4200232.011)
CALIBRATION_INTERVALS = {"沙三段": (3500.0, 3734.0), "沙四段": (3734.0, 3750.0)}


def strip_parens(text: str) -> str:
    """去掉标题/坐标轴名称中的括号及其内容（含半角与全角括号）。"""
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"（[^）]*）", "", text)
    return re.sub(r"\s+", " ", text).strip()


def setup_cjk_font() -> None:
    """Register an available CJK font so Chinese labels render correctly."""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DFN density metrics (P10/P21/P33).")
    parser.add_argument("--patch-csv", type=Path, default=DEFAULT_PATCH_CSV)
    parser.add_argument("--horizon-table", type=Path, default=DEFAULT_HORIZON_TABLE)
    parser.add_argument("--trace-header", type=Path, default=DEFAULT_TRACE_HEADER)
    parser.add_argument("--well-csv", type=Path, default=DEFAULT_WELL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-m", type=float, default=300.0)
    parser.add_argument("--x-min", type=float, default=None,
                        help="Optional grid X lower bound (m); grid then focuses on this block only")
    parser.add_argument("--x-max", type=float, default=None,
                        help="Optional grid X upper bound (m)")
    parser.add_argument("--y-min", type=float, default=None,
                        help="Optional grid Y lower bound (m)")
    parser.add_argument("--y-max", type=float, default=None,
                        help="Optional grid Y upper bound (m)")
    parser.add_argument("--well-radius-m", type=float, default=150.0)
    parser.add_argument("--velocity-ms", type=float, default=None,
                        help="Optional nominal velocity (m/s) to convert TWT thickness to metres.")
    parser.add_argument("--calibrated-vmax", type=float, default=None,
                        help="Optional shared colorbar max (display units 1e-4) for the calibrated heatmaps.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def aperture_mm(scale: str, density: np.ndarray, factor: float) -> np.ndarray:
    """Power-law aperture: a0 + (a1*factor - a0) * density^0.5 (mm)."""
    a0, a1 = SCALE_APERTURE_RANGE_MM[scale]
    dens = np.clip(np.nan_to_num(density, nan=0.5), 0.0, 1.0)
    return np.clip(a0 + (a1 * float(factor) - a0) * np.sqrt(dens), 0.0, None)


def build_grid_bins(trace_header: pd.DataFrame, grid_m: float,
                    x_min: float | None = None, x_max: float | None = None,
                    y_min: float | None = None, y_max: float | None = None):
    x_min = float(trace_header["X"].min()) if x_min is None else float(x_min)
    x_max = float(trace_header["X"].max()) if x_max is None else float(x_max)
    y_min = float(trace_header["Y"].min()) if y_min is None else float(y_min)
    y_max = float(trace_header["Y"].max()) if y_max is None else float(y_max)
    x_bins = np.arange(np.floor(x_min / grid_m) * grid_m, np.ceil(x_max / grid_m) * grid_m + grid_m, grid_m)
    y_bins = np.arange(np.floor(y_min / grid_m) * grid_m, np.ceil(y_max / grid_m) * grid_m + grid_m, grid_m)
    return x_bins, y_bins


def cell_index(header: pd.DataFrame, x_bins: np.ndarray, y_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.digitize(header["X"].to_numpy(dtype=np.float64), x_bins) - 1
    iy = np.digitize(header["Y"].to_numpy(dtype=np.float64), y_bins) - 1
    return ix, iy


def main() -> int:
    args = parse_args()
    setup_cjk_font()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patches = pd.read_csv(args.patch_csv, low_memory=False)
    required = ["FractureScale", "LayerGroup", "CenterX", "CenterY", "CenterTime", "PatchAreaM2", "SourceDensity"]
    missing = [c for c in required if c not in patches.columns]
    if missing:
        raise ValueError(f"patch CSV missing columns: {missing}")
    patches = patches.dropna(subset=["CenterX", "CenterY", "PatchAreaM2"]).copy()
    patches["_scale"] = patches["FractureScale"].astype(str).str.lower()
    patches = patches[patches["_scale"].isin(SCALES)].copy()
    patches["_layer"] = patches["LayerGroup"].astype(str).str.strip()
    patches["_dens"] = pd.to_numeric(patches["SourceDensity"], errors="coerce").fillna(0.5)
    patches["_area"] = pd.to_numeric(patches["PatchAreaM2"], errors="coerce")
    patches = patches[patches["_area"] > 0].copy()

    header = pd.read_csv(args.trace_header, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    horizon = np.load(args.horizon_table, mmap_mode="r")
    if len(header) != len(horizon):
        raise ValueError("trace header and horizon table lengths differ")
    header["_T4"] = np.asarray(horizon["T4"], dtype=np.float64)
    header["_T6"] = np.asarray(horizon["T6"], dtype=np.float64)
    header["_T7"] = np.asarray(horizon["T7"], dtype=np.float64)
    header["_thick3"] = np.clip(header["_T6"] - header["_T4"], 0.0, None)
    header["_thick4"] = np.clip(header["_T7"] - header["_T6"], 0.0, None)

    grid_m = float(args.grid_m)
    if args.x_min is not None:
        # 聚焦指定区块：先把 trace 限定到区块内（保持原行序，与 horizon 数组对齐）
        header = header[
            header["X"].between(args.x_min, args.x_max)
            & header["Y"].between(args.y_min, args.y_max)
        ].copy()
    x_bins, y_bins = build_grid_bins(header, grid_m, args.x_min, args.x_max, args.y_min, args.y_max)
    nx = len(x_bins) - 1
    ny = len(y_bins) - 1
    h_ix, h_iy = cell_index(header, x_bins, y_bins)
    valid_cell = (h_ix >= 0) & (h_ix < nx) & (h_iy >= 0) & (h_iy < ny)
    cell_trace_count = np.zeros((ny, nx), dtype=np.int64)
    np.add.at(cell_trace_count, (h_iy[valid_cell], h_ix[valid_cell]), 1)
    cell_thick = {layer: np.zeros((ny, nx), dtype=np.float64) for layer in LAYERS}
    for layer, col in (("沙三段", "_thick3"), ("沙四段", "_thick4")):
        values = header[col].to_numpy(dtype=np.float64)
        total = np.zeros((ny, nx), dtype=np.float64)
        np.add.at(total, (h_iy[valid_cell], h_ix[valid_cell]), values[valid_cell])
        cell_thick[layer] = total / np.maximum(cell_trace_count, 1)

    cell_area_m2 = cell_trace_count.astype(np.float64) * (12.5 * 12.5)
    p_ix = np.digitize(patches["CenterX"].to_numpy(dtype=np.float64), x_bins) - 1
    p_iy = np.digitize(patches["CenterY"].to_numpy(dtype=np.float64), y_bins) - 1
    p_valid = (p_ix >= 0) & (p_ix < nx) & (p_iy >= 0) & (p_iy < ny)
    patches = patches[p_valid].copy()
    p_ix = p_ix[p_valid]
    p_iy = p_iy[p_valid]

    # P33 per cell / layer / scale / sensitivity (vectorized).
    p_layer = patches["_layer"].to_numpy(dtype=object)
    p_scale = patches["_scale"].to_numpy(dtype=object)
    p_area = patches["_area"].to_numpy(dtype=np.float64)
    p_dens = patches["_dens"].to_numpy(dtype=np.float64)
    patch_count_cell = {layer: {scale: np.zeros((ny, nx), dtype=np.int64) for scale in SCALES} for layer in LAYERS}
    fracture_volume_m3 = {
        factor: {layer: {scale: np.zeros((ny, nx), dtype=np.float64) for scale in SCALES} for layer in LAYERS}
        for factor in SENSITIVITY_FACTORS
    }
    fracture_area_m2 = {layer: np.zeros((ny, nx), dtype=np.float64) for layer in LAYERS}
    for layer in LAYERS:
        lm = p_layer == layer
        np.add.at(fracture_area_m2[layer], (p_iy[lm], p_ix[lm]), p_area[lm])
        for scale in SCALES:
            sel = lm & (p_scale == scale)
            np.add.at(patch_count_cell[layer][scale], (p_iy[sel], p_ix[sel]), 1)
            for factor in SENSITIVITY_FACTORS:
                vol = p_area[sel] * (aperture_mm(scale, p_dens[sel], factor) / 1000.0)
                np.add.at(fracture_volume_m3[factor][layer][scale], (p_iy[sel], p_ix[sel]), vol)

    p33 = {
        factor: {
            layer: {
                scale: fracture_volume_m3[factor][layer][scale] / np.maximum(cell_area_m2 * cell_thick[layer], 1e-30)
                for scale in SCALES
            }
            for layer in LAYERS
        }
        for factor in SENSITIVITY_FACTORS
    }
    for factor in SENSITIVITY_FACTORS:
        for layer in LAYERS:
            for scale in SCALES:
                p33[factor][layer][scale][cell_thick[layer] <= 0] = 0.0
    p33_total = {
        factor: {layer: sum(p33[factor][layer][s] for s in SCALES) for layer in LAYERS}
        for factor in SENSITIVITY_FACTORS
    }
    p33_all = {factor: p33_total[factor]["沙三段"] + p33_total[factor]["沙四段"] for factor in SENSITIVITY_FACTORS}

    # Grid CSV.
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
                "TraceCount": int(cell_trace_count[iy, ix]),
                "CellAreaM2": float(cell_area_m2[iy, ix]),
            }
            for layer in LAYERS:
                row = dict(base)
                row["Layer"] = layer
                row["LayerThicknessMs"] = float(cell_thick[layer][iy, ix])
                row["FractureAreaM2"] = float(fracture_area_m2[layer][iy, ix])
                for scale in SCALES:
                    row[f"PatchCount_{scale}"] = int(patch_count_cell[layer][scale][iy, ix])
                row["PatchCount_total"] = int(sum(patch_count_cell[layer][s][iy, ix] for s in SCALES))
                row["FractureVolumeM3_1mm"] = float(
                    sum(fracture_volume_m3[1.0][layer][s][iy, ix] for s in SCALES)
                )
                row["P33_low"] = float(p33_total[0.5][layer][iy, ix])
                row["P33_mid"] = float(p33_total[1.0][layer][iy, ix])
                row["P33_high"] = float(p33_total[2.0][layer][iy, ix])
                rows.append(row)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(output_dir / "p33_grid_300m.csv", index=False, encoding="utf-8-sig")

    # Imaging calibration: per-layer constant factor from the calibration well
    # (车页1导眼). f_layer = imaging P33 mean / grid well-cell P33 (mid aperture).
    calibration_rows: list[dict[str, Any]] = []
    calibrated_series = pd.Series(np.nan, index=grid_df.index)
    if Path(CALIBRATION_LABELED_CSV).exists():
        lab = pd.read_csv(CALIBRATION_LABELED_CSV, low_memory=False)
        for layer, (d0, d1) in CALIBRATION_INTERVALS.items():
            sub = lab[(lab["DEPT"] >= d0) & (lab["DEPT"] < d1)]
            imaging_p33 = float(pd.to_numeric(sub["P33"], errors="coerce").mean())
            cell = grid_df[
                (grid_df["CellXMin"] <= CALIBRATION_WELL_XY[0])
                & (grid_df["CellXMax"] > CALIBRATION_WELL_XY[0])
                & (grid_df["CellYMin"] <= CALIBRATION_WELL_XY[1])
                & (grid_df["CellYMax"] > CALIBRATION_WELL_XY[1])
                & (grid_df["Layer"] == layer)
            ]
            if len(cell) == 0:
                continue
            dfn_p33 = float(cell["P33_mid"].iloc[0])
            factor = imaging_p33 / dfn_p33 if dfn_p33 > 0 else np.nan
            mask = grid_df["Layer"] == layer
            calibrated_series.loc[mask] = grid_df.loc[mask, "P33_mid"] * factor
            calibration_rows.append(
                {
                    "CalibrationWell": "车页1导眼",
                    "Layer": layer,
                    "ImagingP33Mean": imaging_p33,
                    "DfnGridWellCellP33": dfn_p33,
                    "CalibrationFactor": factor,
                    "ImagingSamples": int(len(sub)),
                }
            )
        grid_df["P33_imaging_calibrated"] = calibrated_series
        grid_df.to_csv(output_dir / "p33_grid_300m.csv", index=False, encoding="utf-8-sig")
        cal_df = pd.DataFrame(calibration_rows)
        cal_df.to_csv(output_dir / "p33_imaging_calibration.csv", index=False, encoding="utf-8-sig")

    # Region summary.
    demo = (
        patches["CenterX"].between(571250, 581250)
        & patches["CenterY"].between(4196500, 4206500)
    ).to_numpy(dtype=bool)
    region_rows: list[dict[str, Any]] = []
    region_masks = {"全矿区": np.ones(len(patches), dtype=bool), "10km_demo块": demo, "demo块外": ~demo}
    for region, mask in region_masks.items():
        for layer in LAYERS:
            for scale in SCALES:
                sel = (p_layer == layer) & (p_scale == scale) & mask
                area = float(p_area[sel].sum())
                vols = {
                    f: float((aperture_mm(scale, p_dens[sel], f) / 1000.0 * p_area[sel]).sum())
                    for f in SENSITIVITY_FACTORS
                }
                region_rows.append(
                    {
                        "Region": region,
                        "Layer": layer,
                        "Scale": scale,
                        "PatchCount": int(sel.sum()),
                        "FractureAreaM2": area,
                        "FractureVolumeM3_low": vols[0.5],
                        "FractureVolumeM3_mid": vols[1.0],
                        "FractureVolumeM3_high": vols[2.0],
                    }
                )
            sel_all = (p_layer == layer) & mask
            vols = {
                f: float(
                    sum(
                        float(
                            (
                                aperture_mm(scale, p_dens[sel_all & (p_scale == scale)], f)
                                / 1000.0
                                * p_area[sel_all & (p_scale == scale)]
                            ).sum()
                        )
                        for scale in SCALES
                    )
                )
                for f in SENSITIVITY_FACTORS
            }
            region_rows.append(
                {
                    "Region": region,
                    "Layer": layer,
                    "Scale": "合计",
                    "PatchCount": int(sel_all.sum()),
                    "FractureAreaM2": float(p_area[sel_all].sum()),
                    "FractureVolumeM3_low": vols[0.5],
                    "FractureVolumeM3_mid": vols[1.0],
                    "FractureVolumeM3_high": vols[2.0],
                }
            )
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(output_dir / "p33_summary_by_region.csv", index=False, encoding="utf-8-sig")

    # P10 for the configured well.
    well_rows: list[dict[str, Any]] = []
    if args.well_csv.exists():
        well = pd.read_csv(args.well_csv, low_memory=False)
        well = well.dropna(subset=["X", "Y", "TIME"]).copy()
        tree = cKDTree(well[["X", "Y"]].to_numpy(dtype=np.float64))
        dist, _ = tree.query(patches[["CenterX", "CenterY"]].to_numpy(dtype=np.float64), k=1)
        near = dist <= float(args.well_radius_m)
        lookup = HorizonSpatialLookup(args.horizon_table, args.trace_header)
        well_row = lookup.query(float(well["X"].mean()), float(well["Y"].mean()))
        t4 = float(well_row["T4"])
        t6 = float(well_row["T6"])
        t7 = float(well_row["T7"])
        for layer, (t0, t1) in (("沙三段", (t4, t6)), ("沙四段", (t6, t7))):
            span = max(t1 - t0, 1.0)
            in_layer = near & patches["CenterTime"].between(t0, t1)
            for scale in SCALES:
                count = int((in_layer & (patches["_scale"] == scale)).sum())
                well_rows.append(
                    {
                        "Well": "车页1导眼",
                        "Layer": layer,
                        "Scale": scale,
                        "PatchCountNear": count,
                        "LayerSpanMs": float(span),
                        "P10_per100ms": float(count / span * 100.0),
                    }
                )
            count_all = int(in_layer.sum())
            well_rows.append(
                {
                    "Well": "车页1导眼",
                    "Layer": layer,
                    "Scale": "合计",
                    "PatchCountNear": count_all,
                    "LayerSpanMs": float(span),
                    "P10_per100ms": float(count_all / span * 100.0),
                }
            )
    pd.DataFrame(well_rows).to_csv(output_dir / "p10_well_summary.csv", index=False, encoding="utf-8-sig")

    # Heatmaps: plan view, X increasing right, Y decreasing downward (north up).
    def draw_heatmap(values: np.ndarray, title: str, filename: str, vmax: float | None = None,
                     cbar_label: str = "P33 相对体密度 (1e-6 · m³/(m²·ms))") -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        display = values[::-1, :]
        vmax = vmax if vmax is not None else float(np.nanpercentile(values[values > 0], 95)) if (values > 0).any() else 1.0
        im = ax.imshow(display, origin="upper", extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                       cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-12), aspect="auto")
        ax.set_title(strip_parens(title))
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    p33_mid_total = p33_all[1.0] * 1.0e6
    draw_heatmap(p33_mid_total, "P33 总密度（沙三+沙四，1 mm 档）", "p33_heatmap_total.png")
    for layer in LAYERS:
        values = p33_total[1.0][layer] * 1.0e6
        suffix = "sha3" if layer == "沙三段" else "sha4"
        draw_heatmap(values, f"P33 {layer}（1 mm 档）", f"p33_heatmap_{suffix}.png")

    # Imaging-calibrated heatmaps: Sha3 / Sha4 / total, with ONE shared color
    # scale so the formations can be compared directly (displayed in 1e-4 units).
    if "P33_imaging_calibrated" in grid_df.columns:
        cal_maps: dict[str, np.ndarray] = {}
        thick_maps: dict[str, np.ndarray] = {}
        for layer in LAYERS:
            layer_df = grid_df[grid_df["Layer"] == layer]
            cal_maps[layer] = np.full((ny, nx), np.nan, dtype=np.float64)
            thick_maps[layer] = np.full((ny, nx), np.nan, dtype=np.float64)
            if layer_df.empty:
                continue
            xidx = np.searchsorted(x_bins, layer_df["CellXMin"].to_numpy())
            yidx = np.searchsorted(y_bins, layer_df["CellYMin"].to_numpy())
            cal_maps[layer][yidx, xidx] = layer_df["P33_imaging_calibrated"].to_numpy(dtype=np.float64)
            thick_maps[layer][yidx, xidx] = layer_df["LayerThicknessMs"].to_numpy(dtype=np.float64)
        h_sum = thick_maps["沙三段"] + thick_maps["沙四段"]
        total_cal = (cal_maps["沙三段"] * thick_maps["沙三段"] + cal_maps["沙四段"] * thick_maps["沙四段"]) / np.maximum(h_sum, 1e-9)
        cal_maps["total"] = total_cal
        combined = np.concatenate([cal_maps[layer][np.isfinite(cal_maps[layer]) & (cal_maps[layer] > 0)] for layer in (*LAYERS, "total")])
        unified_vmax = (
            float(args.calibrated_vmax)
            if args.calibrated_vmax is not None
            else (float(np.percentile(combined, 95)) * 1.0e4 if combined.size else 1.0)
        )
        display_plan = [
            ("沙三段", "sha3", "成像标定 P33 沙三段（车页1导眼基准）"),
            ("沙四段", "sha4", "成像标定 P33 沙四段（车页1导眼基准）"),
            ("total", "total", "成像标定 P33 整体（厚度加权）"),
        ]
        for key, suffix, title in display_plan:
            values = cal_maps[key] * 1.0e4
            draw_heatmap(
                values,
                title,
                f"p33_imaging_calibrated_{suffix}.png",
                vmax=unified_vmax,
                cbar_label="成像标定 P33（10⁻⁴ · m³/m³）",
            )

    # Bars: scale x layer (P33 mid) and region x layer.
    scales = SCALES
    layers = LAYERS
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    xpos = np.arange(len(scales))
    for li, layer in enumerate(layers):
        vals = [p33[1.0][layer][s].sum() * 1.0e6 for s in scales]
        ax.bar(xpos + li * width, vals, width, label=layer)
    ax.set_xticks(xpos + width / 2)
    ax.set_xticklabels(scales)
    ax.set_ylabel(strip_parens("P33 合计 (1e-6 · m³/(m²·ms))"))
    ax.set_title(strip_parens("P33 分尺度 × 分层段（1 mm 档）"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "p33_scale_layer_bars.png", dpi=160)
    plt.close(fig)

    region_pivot = region_df[region_df["Scale"] == "合计"].pivot_table(
        index="Region", columns="Layer", values="FractureVolumeM3_mid", aggfunc="sum"
    ).reindex(["全矿区", "10km_demo块", "demo块外"])
    fig, ax = plt.subplots(figsize=(10, 6))
    region_pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(strip_parens("裂缝体积 (m³)（1 mm 档）"))
    ax.set_title(strip_parens("区域 × 分层段 裂缝体积对比（等效开度）"))
    ax.legend(title="层段")
    fig.tight_layout()
    fig.savefig(output_dir / "p33_region_layer_compare.png", dpi=160)
    plt.close(fig)

    metadata = {
        "status": "pass",
        "version": "v1",
        "design_doc": "优化阶段二/正式主线/step9b_dfn_density_metrics/DFN密度量化指标设计_20260824.md",
        "inputs": {
            "patch_csv": str(args.patch_csv.resolve()),
            "horizon_table": str(args.horizon_table.resolve()),
            "trace_header": str(args.trace_header.resolve()),
            "well_csv": str(args.well_csv.resolve()) if args.well_csv.exists() else None,
        },
        "grid_m": grid_m,
        "grid_bounds": (
            {"x_min": args.x_min, "x_max": args.x_max,
             "y_min": args.y_min, "y_max": args.y_max}
            if args.x_min is not None else None
        ),
        "aperture_mapping": {
            "formula": "aperture_mm = a0 + (a1*factor - a0) * SourceDensity^0.5",
            "scale_ranges_mm": SCALE_APERTURE_RANGE_MM,
            "sensitivity_factors": list(SENSITIVITY_FACTORS),
            "note": "equivalent/nominal aperture, not measured",
        },
        "volume_contract": {
            "thickness": "TWT time thickness (ms), per-cell mean from horizon contract",
            "velocity_optional_ms": args.velocity_ms,
            "p33_units": "m^3/(m^2*ms), displayed scaled by 1e6 in figures",
        },
        "imaging_calibration": {
            "method": "per-layer constant factor; f_layer = imaging P33 mean (车页1导眼, Step3 DEPT interval) / grid well-cell P33_mid",
            "calibration_well": "车页1导眼",
            "factors": {str(row["Layer"]): float(row["CalibrationFactor"]) for row in calibration_rows},
            "note": "approximate point-to-field calibration; spatial pattern remains DFN-driven; report must state this",
            "table": str(output_dir / "p33_imaging_calibration.csv"),
        },
        "display_orientation": "plan view from above; X increases left->right, Y decreases top->bottom (north up)",
        "outputs": {
            "grid_csv": str(output_dir / "p33_grid_300m.csv"),
            "region_csv": str(output_dir / "p33_summary_by_region.csv"),
            "p10_csv": str(output_dir / "p10_well_summary.csv"),
            "heatmaps": [str(output_dir / f) for f in ("p33_heatmap_total.png", "p33_heatmap_sha3.png", "p33_heatmap_sha4.png")],
            "bars": [str(output_dir / "p33_scale_layer_bars.png"), str(output_dir / "p33_region_layer_compare.png")],
        },
    }
    write_json(output_dir / "dfn_density_metrics_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[dfn-metrics] total patches used={len(patches)} grid={nx}x{ny}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
