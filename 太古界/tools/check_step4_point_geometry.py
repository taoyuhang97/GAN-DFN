#!/usr/bin/env python3
"""自检：Step4 的井级裂缝点必须自带 X/Y/TIME（P0-4′）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--merged-curve", type=Path, default=None)
    args = parser.parse_args()

    points = pd.read_csv(args.points, encoding="utf-8-sig", low_memory=False)
    required = ["X", "Y", "TIME"]
    missing = [c for c in required if c not in points.columns]
    report: dict = {
        "points_csv": str(args.points),
        "rows": int(len(points)),
        "has_xy_time": not missing,
        "missing_columns": missing,
    }
    if not missing:
        finite = points[required].notna().all(axis=1)
        report["rows_with_geometry"] = int(finite.sum())
        report["rows_without_geometry"] = int((~finite).sum())
        if "SourceCount" in points.columns:
            report["multi_source_point_fraction"] = float(
                (pd.to_numeric(points["SourceCount"], errors="coerce") > 1).mean()
            )
        # 与合并曲线的主键一致性（若提供）
        merged_path = args.merged_curve or Path(args.points).with_name("all_wells_merged_density_prediction.csv")
        if Path(merged_path).exists():
            curve = pd.read_csv(merged_path, encoding="utf-8-sig", low_memory=False)[["WellName", "TVD", "X", "Y", "TIME"]]
            joined = points.merge(curve, on=["WellName", "TVD"], how="left", suffixes=("", "_curve"), validate="one_to_one")
            diff = (joined[["X", "Y", "TIME"]].to_numpy(float) - joined[["X_curve", "Y_curve", "TIME_curve"]].to_numpy(float))
            report["max_abs_diff_vs_merged_curve"] = float(abs(diff).max()) if len(points) else 0.0
            report["key_join_unmatched"] = int(joined["X_curve"].isna().sum())
    report["status"] = "pass" if (not missing and report.get("rows_without_geometry", 1) == 0) else "fail"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
