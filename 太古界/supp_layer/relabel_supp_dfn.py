#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Step7D 融合结果的临时层标签改回"上部补充层"（LayerCode=3）。

现有 7B/7C/7D 按 T4/T6/T7 槽位运行，输出里 LayerGroup 会显示
"上部复合层/太古界风化壳"；本脚本只改 CSV 的层标签，不改几何。
VTK 中不含 LayerGroup/LayerCode 数组，无需改。

用法::

    python3 太古界/supp_layer/relabel_supp_dfn.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
DEFAULT_INPUT = TAIGU / "supp_layer/output/taigu_supp_step7_v1/step7d_supp/fused_multiscale_dfn_patches.csv"
DEFAULT_OUTPUT = TAIGU / "supp_layer/output/taigu_supp_step7_v1/step7d_supp/fused_multiscale_dfn_supp_patches.csv"
LAYER_NAME = "上部补充层"
LAYER_CODE = 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel supplementary DFN patches.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, low_memory=False)
    before = frame["LayerGroup"].value_counts().to_dict() if "LayerGroup" in frame.columns else {}
    frame["LayerGroup"] = LAYER_NAME
    frame["LayerCode"] = LAYER_CODE
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "status": "pass",
        "input_csv": str(args.input_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "rows": int(len(frame)),
        "layer_group": LAYER_NAME,
        "layer_code": LAYER_CODE,
        "before_layer_group": before,
        "fracture_scale": frame["FractureScale"].value_counts().to_dict() if "FractureScale" in frame.columns else {},
        "center_time_ms": [
            float(frame["CenterTime"].min()),
            float(frame["CenterTime"].max()),
        ] if "CenterTime" in frame.columns else None,
    }
    summary_path = args.output_csv.parent / "fused_multiscale_dfn_supp_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
