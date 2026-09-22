#!/usr/bin/env python3
"""生成 A+B 对照实验的 Step8 / Step9 配置（后缀 md2_step4ctl_*）。

实验目的（需求方 2026-09-22）：让 Step8 真的基于 Step4 常规测井预测去"移动/补片"，
使 DFN 沿井的片分布更接近 Step4 预测。两个变体：

* ``_a``  —— 只做 A：**收紧 Step4 事件聚合**（gap 6→3 ms、max_span 18→6 ms），
  md1 的弱控制门控保持不变；
* ``_ab`` —— A + B：在 A 的基础上**放宽弱控制门控**（min_probability 0.8→0.0、
  min_density_score 0.85→0.0，等价于砂砾岩"无门控"的行为）。

只写新配置，不动 md1 / v6 既有配置与产物。用法::

    python3 太古界/tools/make_md2_step4ctl_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
STEP8_SRC = "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_md1.json"
STEP9_SRC = "太古界/step9_sections/configs/taigu_step9_multibackground_v2_md1.json"


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{index}]"))
    else:
        out[prefix] = value
    return out


def build_step8(tag: str, tighten_only: bool) -> tuple[Path, dict[str, object]]:
    source = REPO / STEP8_SRC
    config = json.loads(source.read_text(encoding="utf-8"))
    config["version"] = f"taigu_step8_attribute_multiscale_v1_{tag}"
    config["output_dir"] = str(
        REPO / "太古界/step8_well_correction/output" / f"taigu_step8_attribute_multiscale_v1_{tag}"
    )
    # A：收紧事件聚合
    config["step4_event_gap_ms"] = 3.0
    config["step4_event_max_span_ms"] = 6.0
    # B：放宽弱控制门控（仅 _ab）
    gate = dict(config.get("step4_unmatched_addition", {}))
    if tighten_only:
        config["_note"] = (
            f"{tag}（2026-09-22 对照实验 A）：Step4 事件聚合收紧 gap 6→3 ms / span 18→6 ms；"
            "弱控制门控与 md1 相同（prob≥0.8 且 DensityScore≥0.85）。"
        )
    else:
        gate["enabled"] = True
        gate["min_probability"] = 0.0
        gate["min_density_score"] = 0.0
        config["step4_unmatched_addition"] = gate
        config["_note"] = (
            f"{tag}（2026-09-22 对照实验 A+B）：事件聚合收紧 gap 6→3 ms / span 18→6 ms，"
            "并放宽弱控制门控（prob≥0.0 且 DensityScore≥0.0，等价砂砾岩无门控）；"
            "其余口径与 md1 完全一致。"
        )
    target = source.with_name(f"taigu_step8_attribute_multiscale_v1_{tag}.json")
    return target, config


def build_step9(tag: str) -> tuple[Path, dict[str, object]]:
    source = REPO / STEP9_SRC
    config = json.loads(source.read_text(encoding="utf-8"))
    config["version"] = f"taigu_step9_multibackground_v2_{tag}"
    config["step8_patches_csv"] = str(
        REPO
        / "太古界/step8_well_correction/output"
        / f"taigu_step8_attribute_multiscale_v1_{tag}"
        / "well_corrected_dfn_fracture_patches.csv"
    )
    config["output_dir"] = str(
        REPO / "太古界/step9_sections/output" / f"taigu_step9_multibackground_v2_{tag}"
    )
    config["_note"] = (
        f"{tag}（2026-09-22 对照实验）：输入为 Step8 {tag} 的片表；绘制口径与 md1 完全一致。"
    )
    target = source.with_name(f"taigu_step9_multibackground_v2_{tag}.json")
    return target, config


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate md2_step4ctl experiment configs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs: list[tuple[Path, dict[str, object], Path]] = []
    for tag, tighten_only in (("md2_step4ctl_a", True), ("md2_step4ctl_ab", False)):
        target, config = build_step8(tag, tighten_only)
        jobs.append((REPO / STEP8_SRC, config, target))
        if not tighten_only:
            target9, config9 = build_step9(tag)
            jobs.append((REPO / STEP9_SRC, config9, target9))
    for source, config, target in jobs:
        before = flatten(json.loads(source.read_text(encoding="utf-8")))
        after = flatten(config)
        changes = {
            key: (before.get(key), after.get(key))
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        print(f"=== {target.relative_to(REPO)}  （{len(changes)} 处改动）")
        for key, (old, new) in changes.items():
            print(f"    {key}\n      旧: {old}\n      新: {new}")
        if not args.dry_run:
            target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
