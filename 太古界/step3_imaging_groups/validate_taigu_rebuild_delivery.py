#!/usr/bin/env python3
"""Validate TaiGuJie Step3 imaging supervision delivery (v3).

Checks, mirroring the glutenite validator's rigor:
1. Group schema must equal LABEL_COLUMNS exactly.
2. Per group: Density non-null, HasFractureDensity == Density > 0,
   GT_POINT_FLAG in {0,1}, MD/TVD finite, StrataName in the two target strata,
   and manifest counts (Rows / DensityPositiveRows / GTPointRows) match files.
3. Per configured well: must have at least one group, unless the Step2 rejected
   table documents an upstream rejection (e.g. 埕北古7).
4. Point accounting closes: TotalPointRows == MappedUniquePoints + Dropped,
   and mapped points split into inside/outside density support.
5. Interpreted TVD coverage gaps are reported (informational, not fail).

Writes `delivery_acceptance_summary.json` and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from build_taigu_imaging_groups import LABEL_COLUMNS


VALID_STRATA = {"上部复合层", "太古界风化壳"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate TaiGuJie Step3 imaging groups (v3)")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "taigu_step3_imaging_groups.json")
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> int:
    ns = parse_args()
    cfg = json.loads(ns.config.read_text(encoding="utf-8"))
    step2_root = Path(cfg["step2_output_dir"])
    output = Path(cfg["output_dir"])
    if not step2_root.is_absolute():
        step2_root = Path(__file__).resolve().parents[2] / step2_root
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output

    errors: list[str] = []
    warnings: list[str] = []
    group_checks: list[dict[str, object]] = []

    manifest = read_csv(output / "sample_group_manifest.csv")
    point_qc = read_csv(output / "point_mapping_qc.csv")
    wells_without = read_csv(output / "wells_without_groups.csv") if (output / "wells_without_groups.csv").exists() else pd.DataFrame()
    rejected = read_csv(step2_root / "taigu_step2_rejected_wells.csv") if (step2_root / "taigu_step2_rejected_wells.csv").exists() else pd.DataFrame()
    rejected_reason = {str(r.WellName): str(r.Reason) for _, r in rejected.iterrows()} if not rejected.empty else {}

    # 1) + 2) per-group checks
    expected_columns = list(LABEL_COLUMNS)
    for _, g in manifest.iterrows():
        group_path = Path(str(g.GroupPath))
        if not group_path.exists():
            errors.append(f"[group] 文件缺失: {g.GroupID} -> {group_path.name}")
            continue
        df = read_csv(group_path)
        ok_schema = list(df.columns) == expected_columns
        density_ok = bool(df.Density.notna().all())
        has_frac_ok = bool((df.HasFractureDensity == df.Density.gt(0.0).astype(int)).all())
        gt_ok = bool(df.GT_POINT_FLAG.isin([0, 1]).all())
        finite_ok = bool(pd.to_numeric(df.MD, errors="coerce").notna().all() and pd.to_numeric(df.TVD, errors="coerce").notna().all())
        strata_ok = bool(df.StrataName.isin(VALID_STRATA).all())
        rows_ok = int(len(df)) == int(g.Rows)
        pos_ok = int(df.HasFractureDensity.sum()) == int(g.DensityPositiveRows)
        gt_count_ok = int(df.GT_POINT_FLAG.sum()) == int(g.GTPointRows)
        ok = all([ok_schema, density_ok, has_frac_ok, gt_ok, finite_ok, strata_ok, rows_ok, pos_ok, gt_count_ok])
        group_checks.append({
            "GroupID": str(g.GroupID), "Status": "pass" if ok else "fail",
            "SchemaOK": ok_schema, "DensityOK": density_ok, "HasFractureOK": has_frac_ok,
            "GTPointOK": gt_ok, "FiniteOK": finite_ok, "StrataOK": strata_ok,
            "RowsOK": rows_ok, "PositiveRowsOK": pos_ok, "GTPointRowsOK": gt_count_ok,
        })
        if not ok:
            errors.append(f"[group] 校验失败: {g.GroupID}")

    # 3) per-well enforcement with Step2 rejection exemption
    group_wells = set(manifest.WellName.astype(str)) if not manifest.empty else set()
    configured = [str(w["well_name"]) for w in cfg["wells"]]
    without_map = {str(r.WellName): str(r.Step2RejectReason) for _, r in wells_without.iterrows()} if not wells_without.empty else {}
    well_checks = []
    for well in configured:
        if well in group_wells:
            well_checks.append({"WellName": well, "Status": "pass", "Reason": ""})
            continue
        reason = without_map.get(well, "") or rejected_reason.get(well, "")
        if reason:
            well_checks.append({"WellName": well, "Status": "exempt", "Reason": f"upstream_rejected:{reason}"})
            warnings.append(f"[well] {well} 无组，但上游已放弃（{reason}）")
        else:
            well_checks.append({"WellName": well, "Status": "fail", "Reason": "configured_but_no_groups"})
            errors.append(f"[well] 配置井 {well} 无监督组且无上游放弃记录")

    # 4) point accounting
    point_checks = []
    for _, r in point_qc.iterrows():
        total_ok = int(r.TotalPointRows) == int(r.MappedUniquePoints) + int(r.DroppedPointRows)
        inside_ok = int(r.MappedUniquePoints) == int(r.MappedPointsInsideDensitySupport) + int(r.MappedPointsOutsideDensitySupport)
        support_ok = int(r.MappedPointsOutsideDensitySupport) == 0
        ok = total_ok and inside_ok and support_ok
        point_checks.append({
            "WellName": str(r.WellName), "Status": "pass" if ok else "fail",
            "TotalClosed": total_ok, "InsideOutsideClosed": inside_ok, "AllMappedInsideDensitySupport": support_ok,
        })
        if not ok:
            errors.append(f"[point] 点位账目未闭合: {r.WellName}")

    # 5) coverage gaps (informational)
    gaps = []
    step3_summary = json.loads((output / "step3_acceptance_summary.json").read_text(encoding="utf-8"))
    coverage_gaps = step3_summary.get("coverage_gaps", [])
    for gap in coverage_gaps:
        gaps.append(f"{gap['WellName']} 解释段 {gap['TVDInterval']} 未覆盖 {gap['UncoveredM']} m（{gap['UncoveredRatio']*100:.1f}%）")
    for text in gaps:
        warnings.append(f"[coverage] {text}")

    status = "pass" if not errors else "fail"
    summary = {
        "version": "taigu_step3_imaging_v3_delivery_validation",
        "status": status,
        "group_checks": group_checks,
        "well_checks": well_checks,
        "point_checks": point_checks,
        "coverage_gap_warnings": gaps,
        "errors": errors,
        "warnings": warnings,
    }
    (output / "delivery_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
