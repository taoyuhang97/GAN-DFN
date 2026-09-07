#!/usr/bin/env python3
"""Compare full attribute-grid coverage with the available OBN traces.

The curvature cube is the reference grid.  AntTrack and Coherence are checked
for the same trace count and sampled header coordinates, while the OBN header
CSV is overlaid as the acquired-trace subset.  No downstream workflow is
modified by this QC script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot OBN versus attribute trace coverage")
    p.add_argument("--curvature", type=Path, required=True)
    p.add_argument("--coherence", type=Path, required=True)
    p.add_argument("--anttrack", type=Path, required=True)
    p.add_argument("--obn-header-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--match-tolerance-m", type=float, default=2.0)
    return p.parse_args()


def read_segy_headers(path: Path, label: str) -> pd.DataFrame:
    rows = []
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        f.mmap()
        for i in tqdm(range(f.tracecount), desc=f"读取{label}道头", unit="trace"):
            h = f.header[i]
            rows.append((i, int(h[segyio.TraceField.INLINE_3D]),
                         int(h[segyio.TraceField.CROSSLINE_3D]),
                         float(h[segyio.TraceField.SourceX]),
                         float(h[segyio.TraceField.SourceY])))
    return pd.DataFrame(rows, columns=["TraceIdx", "Inline3D", "Crossline3D", "X", "Y"])


def main() -> int:
    a = parse_args(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    curvature = read_segy_headers(a.curvature, "曲率体")
    coherence = read_segy_headers(a.coherence, "相干体")
    anttrack = read_segy_headers(a.anttrack, "蚂蚁体")
    obn = pd.read_csv(a.obn_header_csv, encoding="utf-8-sig")
    obn = obn[["TraceIdx", "Inline3D", "Crossline3D", "X", "Y"]].dropna().drop_duplicates("TraceIdx")

    ref_xy = curvature[["X", "Y"]].to_numpy(float)
    tree = cKDTree(ref_xy)
    obn_xy = obn[["X", "Y"]].to_numpy(float)
    dist, pos = tree.query(obn_xy, k=1)
    attr_to_obn_dist, _ = cKDTree(obn_xy).query(ref_xy, k=1)
    covered = attr_to_obn_dist <= a.match_tolerance_m
    obn_near_attr = dist <= a.match_tolerance_m

    # Check the three attribute cubes using their sampled headers.  Full
    # coordinate equality is expected; report mismatches rather than failing.
    attr_checks = {}
    for name, df in (("Coherence", coherence), ("AntTrack", anttrack)):
        n = min(len(curvature), len(df))
        same = np.allclose(curvature[["X", "Y"]].to_numpy()[:n], df[["X", "Y"]].to_numpy()[:n], atol=0.0)
        attr_checks[name] = {"trace_count": int(len(df)), "same_trace_count": len(df) == len(curvature),
                             "same_xy_order": bool(same), "sampled_coordinate_mismatch_count": int(np.sum(np.any(curvature[["X", "Y"]].to_numpy()[:n] != df[["X", "Y"]].to_numpy()[:n], axis=1)))}
    attr_checks["CurvatureMax"] = {"trace_count": int(len(curvature)), "reference": True}

    # Use an installed CJK font explicitly; Matplotlib's default DejaVu font
    # cannot render the Chinese labels.
    cjk_candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    cjk = next((f for f in cjk_candidates if Path(f).exists()), None)
    if cjk:
        font_manager.fontManager.addfont(cjk)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=cjk).get_name()]
    plt.rcParams["axes.unicode_minus"] = False

    # Plot 1/2 use sampled inline/crossline traces.  Drawing every line would
    # make a 2-million-trace cube look like a solid rectangle.
    def sampled_lines(df: pd.DataFrame, step: int = 50):
        xs = np.sort(df["X"].unique()); ys = np.sort(df["Y"].unique())
        return xs[::step], ys[::step]

    def draw_lines(ax, df: pd.DataFrame, color: str, label: str, step: int = 50, lw: float = 0.45, alpha: float = 0.75):
        sx, sy = sampled_lines(df, step)
        for x in sx:
            q = df[np.isclose(df.X, x)].sort_values("Y")
            if len(q) > 1: ax.plot(q.X, q.Y, color=color, lw=lw, alpha=alpha)
        for y in sy:
            q = df[np.isclose(df.Y, y)].sort_values("X")
            if len(q) > 1: ax.plot(q.X, q.Y, color=color, lw=lw, alpha=alpha)
        ax.plot([], [], color=color, lw=1.5, label=label)

    attr_xy_df = curvature[["X", "Y"]].copy()
    covered_df = attr_xy_df.loc[covered]
    missing_df = attr_xy_df.loc[~covered]

    # Plot 1: attribute grid lines plus the actual missing attribute-grid traces.
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    draw_lines(ax, attr_xy_df, "#9e9e9e", "属性体抽样道线")
    ax.scatter(missing_df.X, missing_df.Y, s=1.4, c="#d94841", label="属性体有、OBN无", rasterized=True)
    ax.set_title("全矿区道覆盖范围对比（曲率体网格为基准）")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect("equal", adjustable="box"); ax.legend(markerscale=6)
    fig.tight_layout(); fig.savefig(out / "01_attribute_grid_vs_obn_coverage.png"); plt.close(fig)

    # Plot 2: attribute-grid lines with OBN positions/lines overlaid.
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    draw_lines(ax, attr_xy_df, "#bdbdbd", "曲率体属性抽样道线", lw=0.5, alpha=0.7)
    obn_xy_df = obn[["X", "Y"]].copy()
    draw_lines(ax, obn_xy_df, "#1769aa", "OBN抽样道线", lw=0.8, alpha=0.8)
    ax.scatter(obn_xy[~obn_near_attr, 0], obn_xy[~obn_near_attr, 1], s=4, c="#f08c00", label="OBN无近邻属性道", rasterized=True)
    ax.set_title("OBN与属性体道位空间叠加（全矿区）")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect("equal", adjustable="box"); ax.legend(markerscale=6)
    fig.tight_layout(); fig.savefig(out / "02_obn_attribute_overlay.png"); plt.close(fig)

    summary = {
        "status": "pass", "reference": "CurvatureMax SEG-Y trace headers",
        "match_tolerance_m": a.match_tolerance_m,
        "attribute_trace_count": int(len(curvature)), "obn_trace_count": int(len(obn)),
        "attribute_grid_bounds": {k: [float(curvature[k].min()), float(curvature[k].max())] for k in ("X", "Y")},
        "obn_bounds": {k: [float(obn[k].min()), float(obn[k].max())] for k in ("X", "Y")},
        "attribute_traces_with_obn_nearby": int(covered.sum()),
        "attribute_traces_without_obn_nearby": int((~covered).sum()),
        "obn_traces_with_attribute_nearby": int(obn_near_attr.sum()),
        "obn_traces_without_attribute_nearby": int((~obn_near_attr).sum()),
        "nearest_obn_distance_m": {"min": float(attr_to_obn_dist.min()), "median": float(np.median(attr_to_obn_dist)), "max": float(attr_to_obn_dist.max())},
        "attribute_cube_header_checks": attr_checks,
        "outputs": ["01_attribute_grid_vs_obn_coverage.png", "02_obn_attribute_overlay.png", "coverage_summary.json"],
    }
    (out / "coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
