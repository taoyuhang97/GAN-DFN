#!/usr/bin/env python3
"""S10 闸门：太古界产状口径统一的自检（orientation_convention_qc.json）。

检查项：
1. 单位换算：调用 common/orientation_frame/test_convention.py 的全部单测；
2. 无裸换算：业务脚本里不允许出现 atan2（只允许在 convention.py 中）；
3. 字段自洽：各尺度片表 AzimuthDeg == (DipAzimuthDeg - 90) % 180；
4. 井周一致性：每口成像井，井周 R 内 DFN 片的 DipAzimuthDeg 环形中位与甲方真值
   环形中位之差，以及剖面（XZ/YZ）视倾角方向差；
5. 405 因成像段在目标层外（问题记录 §0.33），记 not_applicable。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
if str(TAIGU) not in sys.path:
    sys.path.insert(0, str(TAIGU))

from common.orientation_frame import convention as orientation  # noqa: E402

ATAN2_ALLOWLIST = {"common/orientation_frame/convention.py"}

SCAN_DIRS = (
    "step6b_medium_corridor",
    "step6c_large_fault",
    "step7a_small_scale_dfn",
    "step7b_multiscale_initial_dfn",
    "step7c_large_fault_dfn",
    "step7d_multiscale_fused_dfn",
    "step8_well_correction",
    "step9_sections",
    "common/dfn_geometry",
)

WELL_CORRIDOR_RADIUS_M = 500.0
WELL_MEDIAN_TOLERANCE_DEG = 25.0
SECTION_TOLERANCE_DEG = 20.0


def circular_median_deg(values) -> float:
    arr = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)], dtype=float) % 360.0
    if arr.size == 0:
        return float("nan")
    candidates = np.linspace(0.0, 360.0, 721)[:-1]
    diffs = np.abs(((arr[None, :] - candidates[:, None] + 180.0) % 360.0) - 180.0)
    return float(candidates[int(np.argmin(diffs.sum(axis=1)))])


def check_unit_tests() -> dict:
    test_path = TAIGU / "common" / "orientation_frame" / "test_convention.py"
    namespace: dict = {"__name__": "qc_run", "__file__": str(test_path)}
    exec(compile(test_path.read_text(encoding="utf-8"), str(test_path), "exec"), namespace)  # noqa: S102
    failures = []
    for name in sorted(n for n in namespace if n.startswith("test_")):
        try:
            namespace[name]()
        except Exception as exc:  # pragma: no cover
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return {"status": "pass" if not failures else "fail",
            "test_file": str(test_path), "failures": failures}


def check_no_bare_atan2() -> dict:
    """用 AST 找**真实的** atan2 调用（不误判文档字符串/注释里的提及）。"""
    import ast

    offenders = []
    for sub in SCAN_DIRS:
        root = TAIGU / sub
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(TAIGU).as_posix()
            if rel in ATAN2_ALLOWLIST or "__pycache__" in rel:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            hits = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and node.func.attr == "atan2":
                    hits += 1
            if hits:
                offenders.append({"file": rel, "call_count": hits})
    return {"status": "pass" if not offenders else "fail", "offenders": offenders}


def check_field_consistency(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"status": "missing", "csv": str(csv_path)}
    header = pd.read_csv(csv_path, nrows=0)
    if "DipAzimuthDeg" not in header.columns:
        return {"status": "fail", "csv": str(csv_path), "reason": "缺少 DipAzimuthDeg 字段"}
    cols = [c for c in ("DipAzimuthDeg", "AzimuthDeg") if c in header.columns]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    dip_azimuth = pd.to_numeric(df["DipAzimuthDeg"], errors="coerce")
    strike = pd.to_numeric(df["AzimuthDeg"], errors="coerce")
    expected = (dip_azimuth - 90.0) % 180.0
    diff = np.abs(((strike - expected + 90.0) % 180.0) - 90.0)
    finite = np.isfinite(dip_azimuth) & np.isfinite(diff)
    max_error = float(np.nanmax(diff[finite])) if finite.any() else None
    ok = bool(finite.all() and max_error is not None and max_error <= 1e-6)
    return {
        "status": "pass" if ok else "fail",
        "csv": str(csv_path),
        "count": int(len(df)),
        "max_strike_error_deg": max_error,
        "dip_azimuth_circular_mean_deg": orientation.circular_mean_dip_azimuth(
            dip_azimuth.dropna().to_numpy()
        ) if finite.any() else None,
    }


def load_imaging_truth(groups_root: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(groups_root.glob("*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "GT_POINT_FLAG" not in df.columns:
            continue
        gt = df[pd.to_numeric(df["GT_POINT_FLAG"], errors="coerce").fillna(0).eq(1)].copy()
        if gt.empty:
            continue
        # 只保留**确实落在目标地层窗口内**的成像点：层外的成像段（如 405，见问题记录 §0.33）
        # 与层内 DFN 片不可比，必须排除，否则会得到"假 fail"。
        if "InHorizonLayer" in gt.columns:
            gt = gt[pd.to_numeric(gt["InHorizonLayer"], errors="coerce").fillna(0).eq(1)]
        if gt.empty:
            continue
        gt["WellName"] = gt["WellName"].astype(str)
        gt["StrataName"] = gt["StrataName"].astype(str)
        parts.append(gt[["WellName", "StrataName", "MD", "FracAzimuth", "FracDip"]])
    if not parts:
        return pd.DataFrame(columns=["WellName", "StrataName", "MD", "FracAzimuth", "FracDip"])
    truth = pd.concat(parts, ignore_index=True)
    truth["FracAzimuth"] = pd.to_numeric(truth["FracAzimuth"], errors="coerce") % 360.0
    truth["FracDip"] = pd.to_numeric(truth["FracDip"], errors="coerce")
    return truth.dropna(subset=["FracAzimuth", "FracDip"])


def load_well_xy(step2_segments_root: Path) -> dict:
    """从 Step2 段文件读井位（X/Y 中位）；井名取所在目录名。"""
    well_xy: dict = {}
    for path in sorted(step2_segments_root.rglob("*.csv")):
        header = pd.read_csv(path, nrows=0)
        if not {"X", "Y"}.issubset(header.columns):
            continue
        cols = ["X", "Y"] + (["WellName"] if "WellName" in header.columns else [])
        seg = pd.read_csv(path, usecols=cols, low_memory=False)
        if "WellName" in seg.columns:
            well = str(seg["WellName"].dropna().iloc[0]) if seg["WellName"].notna().any() else path.parent.name
        else:
            well = path.parent.name
        seg = seg.dropna(subset=["X", "Y"])
        if seg.empty:
            continue
        well_xy.setdefault(well, (float(seg["X"].median()), float(seg["Y"].median())))
    return well_xy


DISTRIBUTION_TOLERANCE_L1 = 0.50


def check_layer_distribution(step8_csv: Path, groups_root: Path) -> dict:
    """**分层走向分布一致性**：DFN 片 vs 成像真值（层内点）的 15° 分箱 L1 距离。

    空间井周检查在 10 km demo 区块内无井可比（唯一在块内的 405 成像段在层外，见 §0.33），
    因此增加这条**分布级**闸门：7A 的产状家族本就来自这些成像井，口径正确时两者的
    倾向方位分布应当接近。
    """
    if not step8_csv.exists():
        return {"status": "missing", "csv": str(step8_csv)}
    truth = load_imaging_truth(groups_root)
    header = pd.read_csv(step8_csv, nrows=0)
    cols = [c for c in ("LayerGroup", "DipAzimuthDeg") if c in header.columns]
    patches = pd.read_csv(step8_csv, usecols=cols, low_memory=False)
    patches["DipAzimuthDeg"] = pd.to_numeric(patches["DipAzimuthDeg"], errors="coerce")
    bins = np.arange(0.0, 360.0 + 15.0, 15.0)

    per_layer = {}
    for layer, group in truth.groupby("StrataName"):
        patch_values = patches.loc[
            patches["LayerGroup"].astype(str).eq(layer), "DipAzimuthDeg"
        ].dropna().to_numpy(dtype=float)
        if patch_values.size == 0:
            per_layer[layer] = {"status": "no_patches", "truth_points": int(len(group))}
            continue
        truth_hist, _ = np.histogram(group["FracAzimuth"].to_numpy(dtype=float), bins=bins)
        patch_hist, _ = np.histogram(patch_values, bins=bins)
        truth_share = truth_hist / max(truth_hist.sum(), 1)
        patch_share = patch_hist / max(patch_hist.sum(), 1)
        l1 = float(0.5 * np.abs(truth_share - patch_share).sum())
        truth_mean = orientation.circular_mean_dip_azimuth(group["FracAzimuth"].to_numpy(dtype=float))
        patch_mean = orientation.circular_mean_dip_azimuth(patch_values)
        delta = orientation.angular_distance_deg(truth_mean, patch_mean)
        per_layer[layer] = {
            "status": "pass" if l1 <= DISTRIBUTION_TOLERANCE_L1 else "fail",
            "truth_points": int(len(group)),
            "patch_count": int(patch_values.size),
            "histogram_l1": l1,
            "truth_circular_mean_deg": truth_mean,
            "patch_circular_mean_deg": patch_mean,
            "circular_mean_delta_deg": float(delta),
        }
    failed = [v for v in per_layer.values() if v.get("status") == "fail"]
    return {
        "status": "pass" if per_layer and not failed else ("fail" if failed else "no_data"),
        "bin_width_deg": 15.0,
        "l1_tolerance": DISTRIBUTION_TOLERANCE_L1,
        "per_layer": per_layer,
    }


def check_well_consistency(step8_csv: Path, groups_root: Path, step2_segments_root: Path) -> dict:
    if not step8_csv.exists():
        return {"status": "missing", "csv": str(step8_csv)}
    truth = load_imaging_truth(groups_root)
    header = pd.read_csv(step8_csv, nrows=0)
    cols = [c for c in ("CenterX", "CenterY", "LayerGroup", "DipAzimuthDeg", "DipDeg") if c in header.columns]
    patches = pd.read_csv(step8_csv, usecols=cols, low_memory=False)
    patches["DipAzimuthDeg"] = pd.to_numeric(patches["DipAzimuthDeg"], errors="coerce")
    patches["DipDeg"] = pd.to_numeric(patches["DipDeg"], errors="coerce")
    well_xy = load_well_xy(step2_segments_root)

    rows = []
    for (well, layer), group in truth.groupby(["WellName", "StrataName"]):
        record = {"WellName": well, "LayerGroup": layer, "truth_point_count": int(len(group))}
        if well not in well_xy:
            record.update({"status": "not_applicable_no_geometry", "note": "无井轨迹/井口坐标"})
            rows.append(record)
            continue
        x0, y0 = well_xy[well]
        near = patches[(patches["CenterX"] - x0).abs() <= WELL_CORRIDOR_RADIUS_M]
        near = near[(near["CenterY"] - y0).abs() <= WELL_CORRIDOR_RADIUS_M]
        if "LayerGroup" in near.columns:
            near = near[near["LayerGroup"].astype(str).eq(layer)]
        near = near[near["DipAzimuthDeg"].notna()]
        if near.empty:
            record.update({
                "status": "not_applicable_imaging_outside_target_layer",
                "note": "井周目标层内没有 DFN 片（成像段在层外，或该井不在 demo 区块内）",
            })
            rows.append(record)
            continue
        truth_median = circular_median_deg(group["FracAzimuth"].to_numpy(dtype=float))
        patch_median = circular_median_deg(near["DipAzimuthDeg"].to_numpy(dtype=float))
        delta = orientation.angular_distance_deg(truth_median, patch_median)
        truth_dip = float(np.median(group["FracDip"].to_numpy(dtype=float)))
        patch_dip = float(near["DipDeg"].median())
        section_delta = {}
        for proj in ("XZ", "YZ"):
            a = orientation.trace_on_section(truth_median, truth_dip, proj)
            b = orientation.trace_on_section(patch_median, patch_dip, proj)
            if a is None or b is None:
                continue
            ang_a = float(np.degrees(np.arctan2(a[1], a[0])))
            ang_b = float(np.degrees(np.arctan2(b[1], b[0])))
            raw = abs(ang_a - ang_b) % 180.0
            section_delta[proj] = float(min(raw, 180.0 - raw))
        ok = delta <= WELL_MEDIAN_TOLERANCE_DEG and all(
            v <= SECTION_TOLERANCE_DEG for v in section_delta.values()
        )
        record.update({
            "status": "pass" if ok else "fail",
            "patch_count_in_corridor": int(len(near)),
            "truth_dip_azimuth_circular_median_deg": truth_median,
            "patch_dip_azimuth_circular_median_deg": patch_median,
            "dip_azimuth_delta_deg": float(delta),
            "truth_dip_median_deg": truth_dip,
            "patch_dip_median_deg": patch_dip,
            "section_trace_delta_deg": section_delta,
        })
        rows.append(record)
    failed = [r for r in rows if r.get("status") == "fail"]
    return {
        "status": "pass" if rows and not failed else ("fail" if failed else "no_data"),
        "corridor_radius_m": WELL_CORRIDOR_RADIUS_M,
        "median_tolerance_deg": WELL_MEDIAN_TOLERANCE_DEG,
        "section_tolerance_deg": SECTION_TOLERANCE_DEG,
        "wells": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="太古界产状口径统一 QC 闸门")
    parser.add_argument("--step7a-csv", default=str(TAIGU / "step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_azi1/fracture_patches.csv"))
    parser.add_argument("--step7b-csv", default=str(TAIGU / "step7b_multiscale_initial_dfn/output/taigu_medium_v7_anttrack_led_azi1/medium_dfn_patches.csv"))
    parser.add_argument("--step7c-csv", default=str(TAIGU / "step7c_large_fault_dfn/output/taigu_step7c_large_v3_attribute_v3_azi1/large_fault_dfn_patches.csv"))
    parser.add_argument("--step8-csv", default=str(TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_azi1/well_corrected_dfn_fracture_patches.csv"))
    parser.add_argument("--groups-root", default=str(TAIGU / "step3_imaging_groups/output/taigu_step3_imaging_md1/groups"))
    parser.add_argument("--step2-segments-root", default=str(TAIGU / "step2_well_log_segments/output/taigu_step2_regular_md1"))
    parser.add_argument("--output", default=str(TAIGU / "output_azi1_logs/orientation_convention_qc.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "convention": "DipAzimuthDeg = 真倾向方位 0-360（自北顺时针）；AzimuthDeg = (D-90)%180 派生走向",
        "unit_tests": check_unit_tests(),
        "no_bare_atan2": check_no_bare_atan2(),
        "field_consistency": {
            "step7a": check_field_consistency(Path(args.step7a_csv)),
            "step7b": check_field_consistency(Path(args.step7b_csv)),
            "step7c": check_field_consistency(Path(args.step7c_csv)),
            "step8": check_field_consistency(Path(args.step8_csv)),
        },
        "well_consistency": check_well_consistency(
            Path(args.step8_csv), Path(args.groups_root), Path(args.step2_segments_root)
        ),
        "layer_distribution": check_layer_distribution(
            Path(args.step8_csv), Path(args.groups_root)
        ),
    }
    statuses = [
        report["unit_tests"]["status"],
        report["no_bare_atan2"]["status"],
        *[item["status"] for item in report["field_consistency"].values()],
        report["well_consistency"]["status"],
        report["layer_distribution"]["status"],
    ]
    report["status"] = "pass" if all(s == "pass" for s in statuses) else "fail"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "status": report["status"],
            "unit_tests": report["unit_tests"]["status"],
            "no_bare_atan2": report["no_bare_atan2"]["status"],
            "field_consistency": {k: v["status"] for k, v in report["field_consistency"].items()},
            "well_consistency": report["well_consistency"]["status"],
            "layer_distribution": {
                "status": report["layer_distribution"]["status"],
                "per_layer": {
                    k: {
                        "status": v.get("status"),
                        "histogram_l1": round(v["histogram_l1"], 3) if "histogram_l1" in v else None,
                        "truth_mean_deg": round(v["truth_circular_mean_deg"], 1) if "truth_circular_mean_deg" in v else None,
                        "patch_mean_deg": round(v["patch_circular_mean_deg"], 1) if "patch_circular_mean_deg" in v else None,
                        "mean_delta_deg": round(v["circular_mean_delta_deg"], 1) if "circular_mean_delta_deg" in v else None,
                    }
                    for k, v in report["layer_distribution"]["per_layer"].items()
                },
            },
            "wells": [
                {
                    "well": r["WellName"],
                    "layer": r["LayerGroup"],
                    "status": r["status"],
                    "corridor_patches": r.get("patch_count_in_corridor"),
                    "dip_azimuth_delta_deg": (
                        round(r["dip_azimuth_delta_deg"], 1) if "dip_azimuth_delta_deg" in r else None
                    ),
                    "section_trace_delta_deg": {
                        k: round(v, 1) for k, v in (r.get("section_trace_delta_deg") or {}).items()
                    },
                }
                for r in report["well_consistency"].get("wells", [])
            ],
            "output": str(out),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
