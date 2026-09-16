#!/usr/bin/env python3
"""只读诊断：特征窗口跨测次的实际影响。

Step4 的 `engineer_features()` 按 (WellName, StrataName) 分组计算 0.5/1/3/5 m
局部残差与 MAD，不区分 SegmentID。在重叠区，同一个窗口里会混入另一测次的样本。

本脚本对同一批样本计算两套特征：

  A. 现状：跨测次（按 井 × 地层 分组）
  B. 按段：段内计算（按 井 × 地层 × 段 分组，再拼回）

然后比较两套特征与成像密度的秩相关，以及同深度上"跨测次邻居"对特征值的扰动幅度，
用于判断这个跨段窗口是否值得修。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
TAIGU = ROOT / "太古界"
STEP4_SCRIPT = TAIGU / "step4_fracture_prediction/train_and_predict_taigu_gr_rd_rs.py"
STEP2 = TAIGU / "step2_well_log_segments/output/taigu_step2_regular_v3"
STEP3_ROOT = TAIGU / "step3_imaging_groups/output/taigu_step3_imaging_v3"

FEATURE_NAMES = [
    "GRResidual0p5MZ", "GRResidual1p0MZ", "GRResidual3p0MZ", "GRResidual5p0MZ",
    "GRMAD1MZ", "GRMAD3MZ", "GRMAD5MZ",
    "RDeepResidual0p5MZ", "RDeepResidual1p0MZ", "RDeepResidual3p0MZ",
    "RDeepMAD1MZ", "RDeepMAD3MZ",
    "GRGradientZ", "RDeepGradientZ",
]


def load_step4_module():
    spec = importlib.util.spec_from_file_location("taigu_step4", STEP4_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["taigu_step4"] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False)


def build_training_frame(well: str) -> pd.DataFrame:
    manifest = read_csv(STEP2 / "taigu_step2_segment_manifest.csv")
    groups_manifest = read_csv(STEP3_ROOT / "sample_group_manifest.csv")
    frames = []
    for _, row in groups_manifest.iterrows():
        group = read_csv(Path(row["GroupPath"]))
        if not group.empty and str(group["WellName"].iloc[0]) == well:
            frames.append(group)
    if not frames:
        return pd.DataFrame()
    labels = pd.concat(frames, ignore_index=True)
    labels["SegmentID"] = labels["InputSegmentPath"].apply(lambda p: str(Path(p).stem))
    seg_frames = []
    for _, seg in manifest[manifest["WellName"].astype(str).eq(well)].iterrows():
        df = read_csv(Path(seg["OutputFilePath"]))
        df["WellName"] = str(seg["WellName"])
        df["SegmentID"] = str(seg["SegmentID"])
        seg_frames.append(df)
    if not seg_frames:
        return pd.DataFrame()
    segments = pd.concat(seg_frames, ignore_index=True)
    training = labels.merge(
        segments[["WellName", "TVD", "SegmentID", "GR", "RD", "RS"]],
        on=["WellName", "TVD", "SegmentID"], how="left", validate="one_to_one",
    )
    return training[training["StrataName"].isin(["上部复合层", "太古界风化壳"])].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=TAIGU / "diagnostics/segment_overlap_20260916")
    parser.add_argument("--wells", nargs="*", default=["埕北古斜405", "埕北313", "埕北310"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    step4 = load_step4_module()

    report: dict = {"wells": {}, "feature_names": FEATURE_NAMES}
    for well in args.wells:
        training = build_training_frame(well)
        if training.empty:
            continue
        print(f"\n=== {well}: 训练行 {len(training)}，段 {sorted(training['SegmentID'].unique())}")
        # A. 现状：跨测次窗口
        cross = step4.engineer_features(training.copy())
        # B. 按段窗口
        parts = []
        for _, group in training.groupby(["WellName", "StrataName", "SegmentID"], dropna=False):
            parts.append(step4.engineer_features(group.copy()))
        per_seg = pd.concat(parts, ignore_index=True)

        imaging = (
            training[["TVD", "Density"]]
            .assign(TVD=lambda d: pd.to_numeric(d["TVD"], errors="coerce"),
                    Density=lambda d: pd.to_numeric(d["Density"], errors="coerce"))
            .dropna()
            .groupby("TVD", as_index=False)["Density"].mean()
        )
        imaging_map = dict(zip(imaging["TVD"], imaging["Density"]))
        well_report: dict = {"rows": int(len(training)), "features": {}}
        for feature in FEATURE_NAMES:
            if feature not in cross.columns or feature not in per_seg.columns:
                continue
            a = pd.to_numeric(cross[feature], errors="coerce")
            b = pd.to_numeric(per_seg[feature], errors="coerce")
            density = training["Density"].to_numpy(float)
            mask = np.isfinite(a.to_numpy(float)) & np.isfinite(b.to_numpy(float)) & np.isfinite(density)
            if mask.sum() < 50:
                continue
            entry = {
                "rows_compared": int(mask.sum()),
                "spearman_feature_vs_imaging_cross_segment": float(spearmanr(a.to_numpy(float)[mask], density[mask]).statistic),
                "spearman_feature_vs_imaging_per_segment": float(spearmanr(b.to_numpy(float)[mask], density[mask]).statistic),
                "feature_cross_vs_per_segment_spearman": float(spearmanr(a.to_numpy(float)[mask], b.to_numpy(float)[mask]).statistic),
                "feature_median_abs_change": float(np.median(np.abs(a.to_numpy(float)[mask] - b.to_numpy(float)[mask]))),
            }
            well_report["features"][feature] = entry
        report["wells"][well] = well_report

        print(f"{'feature':26s} {'ρ(跨段,成像)':>14s} {'ρ(段内,成像)':>14s} {'跨/段-特征ρ':>12s} {'中位变化':>10s}")
        for feature, entry in well_report["features"].items():
            print(f"{feature:26s} {entry['spearman_feature_vs_imaging_cross_segment']:14.3f} "
                  f"{entry['spearman_feature_vs_imaging_per_segment']:14.3f} "
                  f"{entry['feature_cross_vs_per_segment_spearman']:12.3f} "
                  f"{entry['feature_median_abs_change']:10.4f}")

    out = args.output / "feature_window_crossing_diagnosis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
