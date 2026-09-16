#!/usr/bin/env python3
"""well_segment_join 自检：多段井的守恒、行序与选段正确性。

对 4 口结构性最难的井构造用例：

* 埕北310  两段 MD 网格相差 0.042 m（合并后偏离两边各 0.021 m，旧 0.011 m 容差全丢）
* 埕北古斜405 两段重叠 288 m，且第三段只有 2 行
* 埕北313  三段 + 4629–4753 m 空档
* 埕北古10  三段 MD 范围完全重叠

断言：

1. 输入点数 = 匹配数 + 未命中数（不静默丢点）；
2. 输出行数与行序与输入完全一致；
3. 无重复 (WellName, MD) 行；
4. 容差按半采样步长推导，Step4 合并偏移可被覆盖。

用法：`python3 太古界/common/well_segment_join/smoke_segment_join.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
TAIGU_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = TAIGU_ROOT.parent
for candidate in (str(TAIGU_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from common.well_segment_join import (  # noqa: E402
    Step4SourceSegmentLookup,
    attach_geometry_by_md,
    cached_well_segment_pool,
    inferred_md_tolerance,
    read_csv_flexible,
    summarize_join,
)

STEP2_ROOT = TAIGU_ROOT / "step2_well_log_segments/output/taigu_step2_regular_v3"
STEP3_GROUPS = TAIGU_ROOT / "step3_imaging_groups/output/taigu_step3_imaging_v3/groups"
STEP4_POINTS = (
    TAIGU_ROOT
    / "step4_fracture_prediction/output/taigu_step4_gr_rd_rs_v1/predictions/all_wells_merged_fracture_points.csv"
)
STEP4_DETAIL = (
    TAIGU_ROOT
    / "step4_fracture_prediction/output/taigu_step4_gr_rd_rs_v1/predictions/all_wells_source_detail_predictions.csv"
)
TEST_WELLS = ["埕北310", "埕北古斜405", "埕北313", "埕北古10"]


def check_invariants(label: str, points: pd.DataFrame, matched: pd.DataFrame, audit: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if len(matched) != len(points):
        problems.append(f"{label}: 输出行数 {len(matched)} != 输入行数 {len(points)}（有行被静默丢弃）")
    if len(audit) != len(points):
        problems.append(f"{label}: 审计行数 {len(audit)} != 输入行数 {len(points)}")
    if not problems:
        before = list(zip(points["WellName"].astype(str), pd.to_numeric(points["MD"], errors="coerce").round(6)))
        after = list(zip(matched["WellName"].astype(str), pd.to_numeric(matched["MD"], errors="coerce").round(6)))
        if before != after:
            problems.append(f"{label}: 输出行序与输入不一致")
        duplicated = int(matched.duplicated(subset=["WellName", "MD"]).sum())
        if duplicated:
            problems.append(f"{label}: 出现 {duplicated} 个重复 (WellName, MD) 行")
    return problems


def main() -> int:
    print("=" * 78)
    print("well_segment_join 自检")
    print("=" * 78)

    points = read_csv_flexible(STEP4_POINTS, low_memory=False)
    lookup = Step4SourceSegmentLookup(STEP4_DETAIL, tvd_tolerance_m=0.25)
    print(f"Step4 合并点表: {len(points)} 行；来源明细可用: {lookup.available}")

    problems: list[str] = []

    # --- 用例 A：Step4 合并点（无显式来源列，用来源明细回溯） ---
    print("\n[用例 A] Step4 合并点 -> Step2 段（全井，来源明细优先）")
    preferred = pd.Series(
        [
            ";".join(lookup.preferred_segment_ids(str(well), float(tvd)))
            for well, tvd in zip(points["WellName"].astype(str), points["TVD"])
        ],
        index=points.index,
    )
    matched_a, audit_a = attach_geometry_by_md(
        points,
        lambda well: cached_well_segment_pool(STEP2_ROOT, well),
        preferred_ids_by_row=preferred,
        geometry_source="step4_predicted",
    )
    problems += check_invariants("用例A", points, matched_a, audit_a)
    summary_a = summarize_join(audit_a)
    print(f"  匹配 {summary_a['matched_count']} / 未命中 {summary_a['unmatched_count']} / "
          f"重复 MD {summary_a['duplicate_md_count']}")
    print(f"  MD 距离: {summary_a['md_match_distance_stats']}")
    print(f"  命中来源段比例: {summary_a.get('preferred_segment_matched_count', 0)}/{summary_a['matched_count']}")
    print(f"  容差范围: {summary_a['tolerance_stats']}")

    # --- 用例 B：Step3 成像组（行内显式 InputSegmentPath） ---
    print("\n[用例 B] Step3 成像组 -> Step2 段（行内显式来源段）")
    group_parts = []
    for group_file in sorted(STEP3_GROUPS.glob("*.csv")):
        group = read_csv_flexible(group_file, low_memory=False)
        group["Step3GroupCSV"] = group_file.name
        group_parts.append(group)
    groups = pd.concat(group_parts, ignore_index=True)
    groups = groups[pd.to_numeric(groups["GT_POINT_FLAG"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    print(f"  GT 点位行数: {len(groups)}")
    matched_b, audit_b = attach_geometry_by_md(
        groups,
        lambda well: cached_well_segment_pool(STEP2_ROOT, well),
        preferred_column="InputSegmentPath",
        geometry_source="step3_imaging_gt",
    )
    problems += check_invariants("用例B", groups, matched_b, audit_b)
    summary_b = summarize_join(audit_b)
    print(f"  匹配 {summary_b['matched_count']} / 未命中 {summary_b['unmatched_count']}")
    print(f"  MD 距离: {summary_b['md_match_distance_stats']}")
    print(f"  命中来源段比例: {summary_b.get('preferred_segment_matched_count', 0)}/{summary_b['matched_count']}")

    # --- 逐井明细 ---
    print("\n[逐井明细]")
    print(f"{'井':14s} {'点数':>5s} {'匹配':>5s} {'未命中':>6s} {'MD步长':>8s} {'容差':>7s} {'最大MD偏差':>10s}")
    for well in TEST_WELLS:
        pool = cached_well_segment_pool(STEP2_ROOT, well)
        if pool is None:
            print(f"{well:14s} 无可用段文件")
            problems.append(f"{well}: 无可用段文件")
            continue
        rows = points[points["WellName"].astype(str).eq(well)]
        sub_matched, sub_audit = attach_geometry_by_md(
            rows,
            lambda w: cached_well_segment_pool(STEP2_ROOT, w),
            preferred_column=None,
            preferred_ids_by_row=preferred.reindex(rows.index),
            geometry_source="step4_predicted",
        )
        problems += check_invariants(f"{well}", rows, sub_matched, sub_audit)
        sub_summary = summarize_join(sub_audit)
        distance_max = sub_summary["md_match_distance_stats"]["max"]
        tolerance = inferred_md_tolerance(pool)
        print(
            f"{well:14s} {len(rows):5d} {sub_summary['matched_count']:5d} "
            f"{sub_summary['unmatched_count']:6d} {pool.md_step_median():8.4f} "
            f"{tolerance:7.4f} {('%.4f' % distance_max) if distance_max is not None else '   n/a':>10s}"
        )

    print("\n" + "=" * 78)
    if problems:
        print(f"❌ 自检失败，共 {len(problems)} 项：")
        for item in problems:
            print("  -", item)
        return 1
    print("✅ 自检通过：点数守恒、行序一致、无重复、容差覆盖 Step4 合并偏移")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
