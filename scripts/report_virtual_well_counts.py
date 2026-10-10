#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计并对比"砂砾岩 / 太古界"两边构造了多少虚拟测井。

两边的虚拟井口径相同：以每口真实井对应的中心地震道为锚点，沿两个网格轴
取 dx, dy = -2..2，生成固定 **5×5 = 25** 条虚拟轨迹；每条轨迹包含该源井
全目标井段的样本（含中心那条 = 源井自身，记账时单列）。

注意砂砾岩有**两套**虚拟井产物，汇报时不要混用：
  * 正式主线 Step5A（`formal_curvature_led_v2`）——喂给 Step6 的正式训练集；
  * 早期工程骨架（`优化阶段二/虚拟测井构造/`，2026-07-01）——范围更大但非正式。

数据来源（都是各条链的正式产物，不重算）：
  砂砾岩·正式主线 优化阶段二/正式主线/step5a_single_source_virtual_wells/output/formal_curvature_led_v2/
  砂砾岩·早期骨架 优化阶段二/虚拟测井构造/outputs/
  太古界 太古界/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_v5/
            taigu_step5_acceptance_summary.json  +  wells/<井>/well_summary.json

用法::

    python3 scripts/report_virtual_well_counts.py
    python3 scripts/report_virtual_well_counts.py --csv /tmp/virtual_well_counts.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]

SAND_FORMAL_DIR = (
    REPO / "优化阶段二/正式主线/step5a_single_source_virtual_wells/output/formal_curvature_led_v2"
)
SAND_FORMAL_INDEX = SAND_FORMAL_DIR / "virtual_well_index.csv"
SAND_FORMAL_AUDIT = SAND_FORMAL_DIR / "virtual_well_build_audit.json"

SAND_PROTO_ROOT = REPO / "优化阶段二/虚拟测井构造/outputs"
SAND_PROTO_INDEX = SAND_PROTO_ROOT / "步骤1_候选虚拟井索引/virtual_well_index.csv"
SAND_PROTO_SUMMARY = SAND_PROTO_ROOT / "步骤5_训练样本汇总/virtual_well_training_summary.csv"

TAIGU_DIR = REPO / "太古界/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_v5"
TAIGU_SUMMARY = TAIGU_DIR / "taigu_step5_acceptance_summary.json"
TAIGU_WELLS = TAIGU_DIR / "wells"
TAIGU_TRAINING = TAIGU_DIR / "taigu_step5_virtual_training_samples.csv"


def collect_sand_formal() -> dict:
    index = pd.read_csv(SAND_FORMAL_INDEX, encoding="utf-8-sig")
    audit = json.loads(SAND_FORMAL_AUDIT.read_text(encoding="utf-8"))
    source_wells = int(index["SourceWellName"].nunique())
    virtual_wells = int(index["VirtualWellName"].nunique())
    center_col = "IsCenterVirtualTrace" if "IsCenterVirtualTrace" in index.columns else None
    center = int(index[center_col].sum()) if center_col else 0
    return {
        "项目": "砂砾岩·正式主线",
        "层级目录": str(SAND_FORMAL_DIR.parent.relative_to(REPO)),
        "源（真实）井数": source_wells,
        "虚拟井数（含中心源轨迹）": virtual_wells,
        "其中中心源轨迹": center,
        "纯偏移虚拟井": virtual_wells - center,
        "每口源井虚拟轨迹": virtual_wells // max(source_wells, 1),
        "虚拟训练样本行数": int(audit.get("training_sample_count", 0)) or None,
        "其中含裂缝": int(audit.get("accepted_positive_count", 0)) or None,
        "虚拟候选样本（含被剔）": None,
        "虚拟井数口径": "step5a virtual_well_index.csv 的 VirtualWellName 唯一值",
    }


def collect_sand_proto() -> dict:
    index = pd.read_csv(SAND_PROTO_INDEX, encoding="utf-8-sig")
    summary = pd.read_csv(SAND_PROTO_SUMMARY, encoding="utf-8-sig")
    source_wells = int(index["SourceWellName"].nunique())
    virtual_wells = int(index["VirtualWellName"].nunique())
    center = int((index["IsSourceTrace"] == 1).sum()) if "IsSourceTrace" in index.columns else 0
    return {
        "项目": "砂砾岩·早期骨架",
        "层级目录": str(SAND_PROTO_ROOT.relative_to(REPO)),
        "源（真实）井数": source_wells,
        "虚拟井数（含中心源轨迹）": virtual_wells,
        "其中中心源轨迹": center,
        "纯偏移虚拟井": virtual_wells - center,
        "每口源井虚拟轨迹": virtual_wells // max(source_wells, 1),
        "虚拟训练样本行数": int(summary["NumSamples"].sum()),
        "其中含裂缝": int(summary["NumFractureSamples"].sum()),
        "虚拟候选样本（含被剔）": None,
        "虚拟井数口径": "步骤1 virtual_well_index.csv 的 VirtualWellName 唯一值",
    }


def collect_taigu() -> dict:
    meta = json.loads(TAIGU_SUMMARY.read_text(encoding="utf-8"))
    virtual_wells = 0
    center = 0
    source_wells = 0
    for well_dir in sorted(p for p in TAIGU_WELLS.iterdir() if p.is_dir()):
        summary_path = well_dir / "well_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tracks = summary.get("virtual_tracks", [])
        source_wells += 1
        virtual_wells += len(tracks)
        center += sum(1 for name in tracks if name.endswith("_VW_2_2"))
    training = pd.read_csv(TAIGU_TRAINING, usecols=["PresenceLabel"], low_memory=False)
    return {
        "项目": "太古界",
        "层级目录": str(TAIGU_DIR.relative_to(REPO)),
        "源（真实）井数": source_wells,
        "虚拟井数（含中心源轨迹）": virtual_wells,
        "其中中心源轨迹": center,
        "纯偏移虚拟井": virtual_wells - center,
        "每口源井虚拟轨迹": virtual_wells // max(source_wells, 1),
        "虚拟训练样本行数": int(len(training)),
        "其中含裂缝": int((training["PresenceLabel"] == 1).sum()),
        "虚拟候选样本（含被剔）": int(meta.get("virtual_track_count", 0)),
        "虚拟井数口径": "每井 well_summary.json 的 virtual_tracks 汇总",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="统计砂砾岩/太古界虚拟测井数量")
    parser.add_argument("--csv", type=Path, default=None, help="可选：把对比表另存为 CSV")
    args = parser.parse_args()

    frame = pd.DataFrame(
        [collect_sand_formal(), collect_sand_proto(), collect_taigu()]
    ).set_index("项目")

    print("=" * 78)
    print("砂砾岩 / 太古界 虚拟测井构造规模对比")
    print("=" * 78)
    print(frame.T.to_string())
    print("-" * 78)
    print("口径：每口真实井为中心锚点生成 5×5 = 25 条虚拟轨迹；")
    print("      虚拟训练样本行数 = 可用（accepted）样本；砂砾岩取 step5 汇总，太古界取 Step5A 训练表。")
    print("注意：对外汇报砂砾岩请用【砂砾岩·正式主线】行；早期骨架行仅作历史对照。")
    print("=" * 78)
    reporting = frame.loc[[r for r in ("砂砾岩·正式主线", "太古界") if r in frame.index]]
    print(f"对外合计（砂砾岩正式主线 + 太古界）：源井 {int(reporting['源（真实）井数'].sum())} 口，"
          f"虚拟井 {int(reporting['虚拟井数（含中心源轨迹）'].sum())} 条，"
          f"虚拟训练样本 {int(reporting['虚拟训练样本行数'].sum()):,} 行")

    if args.csv is not None:
        frame.to_csv(args.csv, encoding="utf-8-sig")
        print(f"[写出] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
