#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step9 增补：405 成像段 ±200 m 的"成像真值 / DFN 预测"拆分剖面。

输出两张图（每张含 XZ、YZ 两个面板）：
  01_imaging_trajectory_and_fractures.png
      仅井轨迹 + 真实成像裂缝（倾向/倾角按视倾角画成短线段）
  02_dfn_predicted_only.png
      仅 DFN 裂缝片（按尺度着色），不画真实成像裂缝

用法::

    python3 太古界/supp_layer/plot_supp_step9_split_sections.py
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

GROUP_CSV = TAIGU / "step3_imaging_groups/output/taigu_step3_imaging_md1/groups/埕北古斜405_g001_上部复合层.csv"
SEGMENT_CSV = TAIGU / "step2_well_log_segments/output/taigu_step2_regular_md1/埕北古斜405/埕北古斜405_seg_002.csv"
DFN_CSV = TAIGU / "supp_layer/output/taigu_supp_merged_v1/merged_dfn_patches.csv"
DEFAULT_OUTPUT = TAIGU / "supp_layer/output/taigu_supp_step9_split_v1"

HALF_WIDTH_M = 200.0
TIME_MIN_MS = 2600.0
TIME_MAX_MS = 3060.0
TIME_SCALE_M_PER_MS = 2.0


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


def load_real_fractures(group_path: Path, segment_path: Path) -> pd.DataFrame:
    group = pd.read_csv(group_path, encoding="utf-8-sig")
    flag = pd.to_numeric(group["GT_POINT_FLAG"], errors="coerce").fillna(0).astype(int)
    group = group[flag.eq(1)].copy()
    seg = pd.read_csv(segment_path, low_memory=False)
    seg = seg[["MD", "X", "Y", "TIME"]].copy()
    seg["MD"] = pd.to_numeric(seg["MD"], errors="coerce")
    group["MD"] = pd.to_numeric(group["MD"], errors="coerce")
    group = group.dropna(subset=["MD"]).sort_values("MD")
    seg = seg.dropna(subset=["MD"]).sort_values("MD")
    merged = pd.merge_asof(group, seg, on="MD", direction="nearest", tolerance=0.25)
    merged = merged.dropna(subset=["X", "Y", "TIME"])
    for column in ("FracAzimuth", "FracDip"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.dropna(subset=["FracAzimuth", "FracDip"]).reset_index(drop=True)


def apparent_dip(dip_deg: np.ndarray, azimuth_deg: np.ndarray, plane: str) -> np.ndarray:
    dip = np.deg2rad(np.clip(dip_deg, 0.0, 89.0))
    az = np.deg2rad(np.mod(azimuth_deg, 360.0))
    component = np.sin(az) if plane == "XZ" else np.cos(az)
    return np.arctan(np.tan(dip) * np.abs(component))


def plot_dip_segments(ax, x, time, dip_deg, az_deg, plane, color="#ff2d55", length_m=30.0):
    alpha = apparent_dip(np.asarray(dip_deg, float), np.asarray(az_deg, float), plane)
    x = np.asarray(x, float)
    time = np.asarray(time, float)
    half = length_m / 2.0
    dx = half * np.cos(alpha)
    dt = (half * np.sin(alpha)) / TIME_SCALE_M_PER_MS
    for i in range(len(x)):
        ax.plot([x[i] - dx[i], x[i] + dx[i]], [time[i] + dt[i], time[i] - dt[i]],
                color=color, lw=1.6, solid_capstyle="round", zorder=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Split imaging vs DFN sections.")
    parser.add_argument("--group-csv", type=Path, default=GROUP_CSV)
    parser.add_argument("--segment-csv", type=Path, default=SEGMENT_CSV)
    parser.add_argument("--dfn-csv", type=Path, default=DFN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    setup_font()

    real = load_real_fractures(args.group_csv, args.segment_csv)
    dfn = pd.read_csv(
        args.dfn_csv,
        usecols=["LayerGroup", "FractureScale", "CenterX", "CenterY", "CenterTime",
                 "LengthM", "HeightTimeMs", "AzimuthDeg", "IsWellControlPatch"],
        low_memory=False,
    )
    plane_x = float(real["X"].mean())
    plane_y = float(real["Y"].mean())
    x_lim = (float(real["X"].min()) - HALF_WIDTH_M, float(real["X"].max()) + HALF_WIDTH_M)
    y_lim = (float(real["Y"].min()) - HALF_WIDTH_M, float(real["Y"].max()) + HALF_WIDTH_M)
    print(f"[split] real fractures={len(real)}  plane_x={plane_x:.1f}  plane_y={plane_y:.1f}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 图 1：仅井轨迹 + 真实裂缝
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for ax, plane in zip(axes, ("XZ", "YZ")):
        h = real["X"] if plane == "XZ" else real["Y"]
        ax.plot(h, real["TIME"], color="#111111", lw=1.4, label="405 成像段井轨迹", zorder=3)
        plot_dip_segments(ax, h, real["TIME"], real["FracDip"], real["FracAzimuth"], plane)
        ax.scatter(h, real["TIME"], s=8, color="#ff2d55", zorder=6, label="真实成像裂缝（按倾向/倾角）")
        ax.set_xlabel("X(m)" if plane == "XZ" else "Y(m)")
        ax.set_ylabel("TIME (ms, TWT)")
        ax.set_ylim(TIME_MAX_MS, TIME_MIN_MS)
        ax.set_xlim(*(x_lim if plane == "XZ" else y_lim))
        ax.grid(alpha=0.25)
        ax.set_title(f"{plane} 剖面：405 成像段 ±{HALF_WIDTH_M:.0f} m")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("埕北古斜405 成像测井段：井轨迹 + 真实裂缝倾向/倾角")
    fig.savefig(out_dir / "01_imaging_trajectory_and_fractures.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 图 2：仅 DFN（按尺度着色），不画真实裂缝
    colors = {"small": "#22d3ee", "medium": "#f59e0b", "large": "#ef4444"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for ax, plane in zip(axes, ("XZ", "YZ")):
        sub = dfn[dfn["CenterTime"].between(TIME_MIN_MS, TIME_MAX_MS)].copy()
        if plane == "XZ":
            sub = sub[(sub["CenterY"] - plane_y).abs() <= HALF_WIDTH_M]
            sub = sub[sub["CenterX"].between(*x_lim)]
            center = sub["CenterX"].to_numpy(float)
        else:
            sub = sub[(sub["CenterX"] - plane_x).abs() <= HALF_WIDTH_M]
            sub = sub[sub["CenterY"].between(*y_lim)]
            center = sub["CenterY"].to_numpy(float)
        time = sub["CenterTime"].to_numpy(float)
        length = np.clip(pd.to_numeric(sub["LengthM"], errors="coerce").fillna(60.0).to_numpy(float), 10.0, 80.0)
        height = np.clip(pd.to_numeric(sub["HeightTimeMs"], errors="coerce").fillna(20.0).to_numpy(float), 2.0, 30.0)
        for scale, color in colors.items():
            mask = sub["FractureScale"].astype(str).eq(scale).to_numpy()
            if not mask.any():
                continue
            c = center[mask]
            t = time[mask]
            half = length[mask] / 2.0
            half_t = height[mask] / 2.0
            for i in range(len(c)):
                ax.plot([c[i] - half[i], c[i] + half[i]], [t[i] + half_t[i], t[i] - half_t[i]],
                        color=color, lw=0.6 if scale == "small" else 1.2, alpha=0.55, zorder=4)
            ax.plot([], [], color=color, lw=1.5 if scale != "small" else 0.9, label={"small": "小尺度", "medium": "中尺度", "large": "大尺度"}[scale])
        ax.set_xlabel("X(m)" if plane == "XZ" else "Y(m)")
        ax.set_ylabel("TIME (ms, TWT)")
        ax.set_ylim(TIME_MAX_MS, TIME_MIN_MS)
        ax.set_xlim(*(x_lim if plane == "XZ" else y_lim))
        ax.grid(alpha=0.25)
        ax.set_title(f"{plane} 剖面：DFN 预测（不含真实成像裂缝）")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("埕北古斜405 成像段 ±200 m：DFN 预测剖面")
    fig.savefig(out_dir / "02_dfn_predicted_only.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": "pass",
        "plane": {"x": plane_x, "y": plane_y, "half_width_m": HALF_WIDTH_M, "time_ms": [TIME_MIN_MS, TIME_MAX_MS]},
        "real_fractures": int(len(real)),
        "dfn_patches_in_window": int(dfn["CenterTime"].between(TIME_MIN_MS, TIME_MAX_MS).sum()),
        "outputs": {
            "imaging": str(out_dir / "01_imaging_trajectory_and_fractures.png"),
            "dfn": str(out_dir / "02_dfn_predicted_only.png"),
        },
    }
    (out_dir / "split_sections_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
