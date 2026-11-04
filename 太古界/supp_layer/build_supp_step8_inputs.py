#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备 Step8（上部补充层）的输入：临时层标签的成像组 + 配置。

太古界 Step8 写死只接受 "上部复合层/太古界风化壳"，所以：
  * 初始 DFN 用**原始 7D 输出**（标签仍是上部复合层/太古界风化壳）；
  * 405 成像组复制一份，层名临时改成 "上部复合层"；
  * horizon 用 legacy CSV（T4=TopV, T6=Top-1ms, T7=Top）。
跑完后用 relabel 脚本把结果改回 "上部补充层"/LayerCode=3。

用法::

    python3 太古界/supp_layer/build_supp_step8_inputs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
SUPP = TAIGU / "supp_layer/output"

BASE_CONFIG = TAIGU / "step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_quanqu1.json"
INITIAL_CSV = SUPP / "taigu_supp_step7_v1/step7d_supp/fused_multiscale_dfn_patches.csv"
INITIAL_VTK = SUPP / "taigu_supp_step7_v1/step7d_supp/fused_multiscale_dfn_raw_time.vtk"
INITIAL_SUMMARY = SUPP / "taigu_supp_step7_v1/step7d_supp/fused_multiscale_summary.json"
SUPP_GROUP = SUPP / "taigu_supp_samples_v1/supp_imaging_group.csv"
LEGACY_CSV = SUPP / "taigu_supp_legacy_v1/supp_horizon_legacy_full.csv"
OUTPUT = SUPP / "taigu_supp_step8_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Step8 inputs for the supplementary layer.")
    parser.add_argument("--initial-csv", type=Path, default=INITIAL_CSV)
    parser.add_argument("--initial-vtk", type=Path, default=INITIAL_VTK)
    parser.add_argument("--initial-summary", type=Path, default=INITIAL_SUMMARY)
    parser.add_argument("--supp-group", type=Path, default=SUPP_GROUP)
    parser.add_argument("--legacy-csv", type=Path, default=LEGACY_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    groups_dir = out_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    group = pd.read_csv(args.supp_group, encoding="utf-8-sig")
    for column in ("StrataName", "LayerGroupByHorizon", "LayerGroupByImaging"):
        if column in group.columns:
            group[column] = "上部复合层"
    if "GroupID" in group.columns:
        group["GroupID"] = group["GroupID"].astype(str).str.replace("上部补充层", "上部复合层", regex=False)
    group_path = groups_dir / "埕北古斜405_supp_上部复合层.csv"
    group.to_csv(group_path, index=False, encoding="utf-8-sig")
    print(f"[supp-step8] wrote temp imaging group {group_path} ({len(group):,} rows)")

    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config.update({
        "version": "taigu_supp_layer_v1_8",
        "initial_dfn_csv": str(args.initial_csv.resolve()),
        "initial_dfn_vtk": str(args.initial_vtk.resolve()),
        "initial_dfn_summary_json": str(args.initial_summary.resolve()),
        "original_fault_manifest_csv": str(
            SUPP / "taigu_supp_legacy_v1/step6c_supp/original_fault_unit_manifest.csv"
        ),
        "step3_groups_root": str(groups_dir),
        "horizon_contract_table": str(args.legacy_csv.resolve()),
        "output_dir": str(out_dir / "step8_supp"),
        "_note": "上部补充层：临时使用上部复合层标签，跑完后 relabel 为上部补充层",
    })
    config_path = out_dir / "supp_8_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[supp-step8] wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
