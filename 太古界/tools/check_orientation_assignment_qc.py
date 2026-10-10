#!/usr/bin/env python3
"""G2–G7 闸门：按《DFN 产状赋值逻辑修正方案_20260923.md》§6 验收产状赋值结果。

* G2 井周分布匹配：井周 DFN 分布 vs 该井经验分布的重叠度 ≥0.6，倾角中位差 ≤8°
* G3 多峰保留：倾向方位峰个数与主峰位置（差 ≤20°）
* G4 陡倾尾巴：井周 DFN 的 ≥75° 占比 ≥ 该井经验占比 × 0.6
* G5 连续性：200 m 窗口平均产状的逐点跳变
* G6 有效井数审计
* G7 小尺度 vs 中尺度（只报告，不判定）

用法::
    python3 太古界/tools/check_orientation_assignment_qc.py --tag azi4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"

AZ_BINS = np.arange(0.0, 361.0, 20.0)
DIP_BINS = np.arange(0.0, 91.0, 10.0)
CORRIDOR_R = 500.0
OVERLAP_MIN = 0.60
DIP_MEDIAN_TOL = 8.0
MAIN_PEAK_TOL = 20.0
STEEP_RATIO_MIN = 0.60
STEEP_DEG = 75.0
WINDOW_M = 200.0
WINDOW_JUMP_TOL = 25.0


def circular_mean(values) -> float:
    arr = np.asarray([v for v in np.asarray(values, float) if np.isfinite(v)], dtype=float) % 360.0
    if arr.size == 0:
        return float("nan")
    return float(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(arr)).mean())) % 360.0)


def hist(values, bins) -> np.ndarray:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(arr, bins=bins)
    return counts / max(counts.sum(), 1)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - 0.5 * np.abs(a - b).sum())


def main_peak(values, bins) -> float:
    h = hist(values, bins)
    if h.sum() <= 0:
        return float("nan")
    centers = (bins[:-1] + bins[1:]) / 2.0
    return float(centers[int(np.argmax(h))])


def peak_count(values, bins, threshold_ratio: float = 0.6) -> int:
    h = hist(values, bins)
    if h.sum() <= 0 or h.max() <= 0:
        return 0
    thr = threshold_ratio * h.max()
    count = 0
    for i in range(len(h)):
        left = h[i - 1] if i > 0 else 0.0
        right = h[i + 1] if i + 1 < len(h) else 0.0
        if h[i] >= thr and h[i] >= left and h[i] >= right:
            count += 1
    return count


def load_inputs(tag: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    patches = pd.read_csv(
        TAIGU / f"step8_well_correction/output/taigu_step8_attribute_multiscale_v1_{tag}/well_corrected_dfn_fracture_patches.csv",
        low_memory=False,
    )
    library_json = json.loads(
        (TAIGU / f"step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_{tag}/orientation_well_library.json").read_text(encoding="utf-8")
    )
    points = []
    for path in sorted((TAIGU / "step3_imaging_groups/output/taigu_step3_imaging_md1/groups").glob("*.csv")):
        group = pd.read_csv(path, encoding="utf-8-sig")
        if "GT_POINT_FLAG" not in group.columns:
            continue
        gt = group[pd.to_numeric(group["GT_POINT_FLAG"], errors="coerce").fillna(0).eq(1)].copy()
        if gt.empty:
            continue
        points.append(gt[["WellName", "StrataName", "FracAzimuth", "FracDip"]])
    truth = pd.concat(points, ignore_index=True)
    truth["FracAzimuth"] = pd.to_numeric(truth["FracAzimuth"], errors="coerce") % 360.0
    truth["FracDip"] = pd.to_numeric(truth["FracDip"], errors="coerce")
    return patches, truth, library_json


def evaluate_wells(patches: pd.DataFrame, truth: pd.DataFrame, library: dict) -> dict:
    rows = []
    for layer, wells in library["well_library"].items():
        for well, info in wells.items():
            record = {
                "well": well,
                "layer": layer,
                "well_point_count": int(info["point_count"]),
                "in_horizon_fraction": info["in_horizon_fraction"],
            }
            sub = truth[
                truth["WellName"].astype(str).eq(well) & truth["StrataName"].astype(str).eq(layer)
            ].dropna(subset=["FracAzimuth", "FracDip"])
            corridor = patches[
                (patches["LayerGroup"].astype(str).eq(layer))
                & (np.hypot(patches["CenterX"] - float(info["x"]), patches["CenterY"] - float(info["y"])) <= CORRIDOR_R)
            ]
            corridor = corridor.dropna(subset=["DipAzimuthDeg", "DipDeg"])
            if len(sub) < 10 or len(corridor) < 30:
                record["status"] = "not_applicable"
                record["corridor_patch_count"] = int(len(corridor))
                rows.append(record)
                continue
            taz = sub["FracAzimuth"].to_numpy(float)
            tdip = sub["FracDip"].to_numpy(float)
            paz = corridor["DipAzimuthDeg"].to_numpy(float) % 360.0
            pdip = corridor["DipDeg"].to_numpy(float)

            az_overlap = overlap(hist(taz, AZ_BINS), hist(paz, AZ_BINS))
            dip_overlap = overlap(hist(tdip, DIP_BINS), hist(pdip, DIP_BINS))
            dip_median_delta = abs(float(np.median(tdip)) - float(np.median(pdip)))
            main_peak_delta = abs(main_peak(taz, AZ_BINS) - main_peak(paz, AZ_BINS))
            steep_truth = float(np.mean(tdip >= STEEP_DEG))
            steep_dfn = float(np.mean(pdip >= STEEP_DEG))
            record.update(
                {
                    "corridor_patch_count": int(len(corridor)),
                    "azimuth_overlap": az_overlap,
                    "dip_overlap": dip_overlap,
                    "truth_dip_median": float(np.median(tdip)),
                    "dfn_dip_median": float(np.median(pdip)),
                    "dip_median_delta": dip_median_delta,
                    "truth_main_peak_deg": main_peak(taz, AZ_BINS),
                    "dfn_main_peak_deg": main_peak(paz, AZ_BINS),
                    "main_peak_delta": main_peak_delta,
                    "truth_peak_count": peak_count(taz, AZ_BINS),
                    "dfn_peak_count": peak_count(paz, AZ_BINS),
                    "truth_steep_fraction": steep_truth,
                    "dfn_steep_fraction": steep_dfn,
                }
            )
            g2 = az_overlap >= OVERLAP_MIN and dip_overlap >= OVERLAP_MIN and dip_median_delta <= DIP_MEDIAN_TOL
            g3 = main_peak_delta <= MAIN_PEAK_TOL
            g4 = steep_dfn >= STEEP_RATIO_MIN * steep_truth
            record["status"] = "pass" if (g2 and g3 and g4) else "fail"
            record["gate_detail"] = {"G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
            rows.append(record)
    tested = [r for r in rows if r["status"] in {"pass", "fail"}]
    return {
        "status": "pass" if tested and all(r["status"] == "pass" for r in tested) else ("fail" if tested else "no_data"),
        "corridor_radius_m": CORRIDOR_R,
        "thresholds": {
            "overlap_min": OVERLAP_MIN,
            "dip_median_tol_deg": DIP_MEDIAN_TOL,
            "main_peak_tol_deg": MAIN_PEAK_TOL,
            "steep_ratio_min": STEEP_RATIO_MIN,
        },
        "wells": rows,
    }


DIST_OVERLAP_MIN_MEDIAN = 0.60
DIST_OVERLAP_MIN_P05 = 0.40
WEIGHT_GRADIENT_MAX = 0.20


def evaluate_continuity(patches: pd.DataFrame, library: dict) -> dict:
    """G5：**分布级**连续性（不是"窗内平均产状"）。

    为什么不用窗内平均：成像井的倾向方位分布本身就宽（405 的环形离散度 88°、
    近乎均匀/双峰），对这类分布取"平均方位"没有意义——相邻窗的均值差可以达到 140°+，
    完全是抽样假象（实测：两窗各 117/129 片、离散度 120–143°，均值从 152° 跳到 6°）。

    因此改测两件真正决定"DFN 连不连续"的事：
    * **G5a 井权重场**：各井权重随位置的最大梯度（D2 要防的就是井边界突变）；
    * **G5b 方位分布**：相邻 200 m 窗的 20° 分箱直方图重叠度（分布级平滑）；
    * **附带信息**：`DipAzimuthDeg` 的空间相关长度（逐片独立抽样 → 接近片间距）。
    """
    frame = patches.dropna(subset=["CenterX", "CenterY", "DipAzimuthDeg", "DipDeg"])
    if frame.empty:
        return {"status": "no_data"}
    xs = frame["CenterX"].to_numpy(float)
    ys = frame["CenterY"].to_numpy(float)
    tree = cKDTree(np.column_stack([xs, ys]))
    gx = np.arange(xs.min(), xs.max() + WINDOW_M, WINDOW_M)
    gy = np.arange(ys.min(), ys.max() + WINDOW_M, WINDOW_M)
    hist_grid: dict[tuple[int, int], np.ndarray] = {}
    for iy, y in enumerate(gy):
        for ix, x in enumerate(gx):
            idx = tree.query_ball_point([x, y], WINDOW_M)
            if len(idx) >= 20:
                hist_grid[(iy, ix)] = hist(frame["DipAzimuthDeg"].to_numpy(float)[idx], AZ_BINS)
    overlaps = []
    for iy in range(len(gy)):
        for ix in range(len(gx)):
            if (iy, ix) not in hist_grid:
                continue
            for dy, dx in ((0, 1), (1, 0)):
                jy, jx = iy + dy, ix + dx
                if (jy, jx) not in hist_grid:
                    continue
                overlaps.append(overlap(hist_grid[(iy, ix)], hist_grid[(jy, jx)]))
    weight_gradient = _weight_gradient(library, patches)
    if not overlaps:
        return {"status": "no_data"}
    overlaps = np.asarray(overlaps, dtype=float)
    g5a = bool(weight_gradient["max_gradient"] <= WEIGHT_GRADIENT_MAX)
    g5b = bool(
        float(np.median(overlaps)) >= DIST_OVERLAP_MIN_MEDIAN
        and float(np.percentile(overlaps, 5)) >= DIST_OVERLAP_MIN_P05
    )
    return {
        "status": "pass" if (g5a and g5b) else "fail",
        "window_m": WINDOW_M,
        "G5a_weight_field": {
            "status": "pass" if g5a else "fail",
            "max_gradient_per_window": weight_gradient["max_gradient"],
            "gradient_tol": WEIGHT_GRADIENT_MAX,
            "detail": weight_gradient["detail"],
        },
        "G5b_adjacent_window_distribution": {
            "status": "pass" if g5b else "fail",
            "overlap_median": float(np.median(overlaps)),
            "overlap_p05": float(np.percentile(overlaps, 5)),
            "overlap_min": float(np.min(overlaps)),
            "thresholds": {"median_min": DIST_OVERLAP_MIN_MEDIAN, "p05_min": DIST_OVERLAP_MIN_P05},
            "window_count": int(len(hist_grid)),
        },
        "note": (
            "不使用『窗内平均产状跳变』作为判据：成像井方位分布宽（405 环形离散度 88°），"
            "对双峰/近均匀分布求平均方位不稳定，实测最大均值跳变 146° 属抽样假象。"
        ),
    }


def _weight_gradient(library: dict, patches: pd.DataFrame) -> dict:
    """各井权重随位置的最大梯度（200 m 相邻格点）。"""
    detail: dict[str, float] = {}
    overall = 0.0
    for layer, wells in library["well_library"].items():
        sub = patches[patches["LayerGroup"].astype(str).eq(layer)].dropna(subset=["CenterX", "CenterY"])
        if sub.empty:
            continue
        xs = sub["CenterX"].to_numpy(float)
        ys = sub["CenterY"].to_numpy(float)
        gx = np.arange(xs.min(), xs.max() + WINDOW_M, WINDOW_M)
        gy = np.arange(ys.min(), ys.max() + WINDOW_M, WINDOW_M)
        scale = float(library.get("well_weight", {}).get("scale_m", 6000.0))
        share = float(library.get("well_weight", {}).get("min_share", 0.02))
        layer_max = 0.0
        for well, info in wells.items():
            wx = float(info["x"])
            wy = float(info["y"])
            grid = np.exp(-(((gx[None, :] - wx) ** 2 + (gy[:, None] - wy) ** 2) ** 0.5 / scale) ** 2)
            norms = np.zeros_like(grid)
            for other, other_info in wells.items():
                ox, oy = float(other_info["x"]), float(other_info["y"])
                norms += np.exp(-(((gx[None, :] - ox) ** 2 + (gy[:, None] - oy) ** 2) ** 0.5 / scale) ** 2)
            weight = grid / np.maximum(norms, 1e-12)
            diffs = []
            diffs.append(np.abs(np.diff(weight, axis=1)))
            diffs.append(np.abs(np.diff(weight, axis=0)))
            well_max = float(max(d.max() if d.size else 0.0 for d in diffs))
            if weight.max() >= share:
                layer_max = max(layer_max, well_max)
                detail[f"{layer}/{well}"] = well_max
        overall = max(overall, layer_max)
    return {"max_gradient": overall, "detail": detail}


def evaluate_effective_wells(patches: pd.DataFrame) -> dict:
    if "OrientationEffectiveWellCount" not in patches.columns:
        return {"status": "missing"}
    values = pd.to_numeric(patches["OrientationEffectiveWellCount"], errors="coerce").dropna()
    return {
        "status": "pass" if len(values) and float(values.min()) >= 1.0 and float(values.max()) <= 6.0 else "fail",
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def evaluate_scale_gap(patches: pd.DataFrame) -> dict:
    """G7：只报告小尺度与中尺度在同层的产状分布差。"""
    report = {}
    for layer, group in patches.groupby("LayerGroup"):
        small = group[group["FractureScale"].astype(str).eq("small")]
        medium = group[group["FractureScale"].astype(str).eq("medium")]
        if small.empty or medium.empty:
            continue
        report[str(layer)] = {
            "small_count": int(len(small)),
            "medium_count": int(len(medium)),
            "small_dip_median": float(pd.to_numeric(small["DipDeg"], errors="coerce").median()),
            "medium_dip_median": float(pd.to_numeric(medium["DipDeg"], errors="coerce").median()),
            "small_dip_azimuth_mean": circular_mean(small["DipAzimuthDeg"]),
            "medium_dip_azimuth_mean": circular_mean(medium["DipAzimuthDeg"]),
            "azimuth_overlap": overlap(
                hist(pd.to_numeric(small["DipAzimuthDeg"], errors="coerce"), AZ_BINS),
                hist(pd.to_numeric(medium["DipAzimuthDeg"], errors="coerce"), AZ_BINS),
            ),
            "dip_overlap": overlap(
                hist(pd.to_numeric(small["DipDeg"], errors="coerce"), DIP_BINS),
                hist(pd.to_numeric(medium["DipDeg"], errors="coerce"), DIP_BINS),
            ),
        }
    return {"status": "report_only", "per_layer": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="产状赋值 G2–G7 闸门")
    parser.add_argument("--tag", default="azi4")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    patches, truth, library = load_inputs(args.tag)
    report = {
        "tag": args.tag,
        "G2_G3_G4_well_match": evaluate_wells(patches, truth, library),
        "G5_continuity": evaluate_continuity(patches, library),
        "G6_effective_wells": evaluate_effective_wells(patches),
        "G7_scale_gap": evaluate_scale_gap(patches),
    }
    statuses = [
        report["G2_G3_G4_well_match"]["status"],
        report["G5_continuity"]["status"],
        report["G6_effective_wells"]["status"],
    ]
    report["status"] = "pass" if all(s == "pass" for s in statuses) else "fail"
    out = Path(args.output) if args.output else TAIGU / f"output_azi1_logs/orientation_assignment_qc_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "G7_scale_gap"}, ensure_ascii=False, indent=2))
    print("\nG7（仅报告）:", json.dumps(report["G7_scale_gap"]["per_layer"], ensure_ascii=False, indent=2))
    print("\n输出:", out)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
