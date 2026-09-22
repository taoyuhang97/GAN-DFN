#!/usr/bin/env python3
"""生成"关掉 Step7A 局部脊产状"对照实验的配置（后缀 md2_noridge）。

背景（2026-09-22）：405 井周目标层内的 DFN 片有 43% 来自 Step7A 的
``local_gradient_ridge`` / ``multiwell_local_blend``（走向 137–143°、倾角仅 26–31°，
与成像真值走向 87°、倾角 76° 差 40–55°），而 ``multiwell_family`` 那部分（走向 88°）
与真值一致。本实验把 ``orientation.local_high_quality`` / ``local_medium_quality``
抬到 1.01（quality 最大为 1，等于关掉局部脊分支），让 7A 全部走多井族产状。

链路：Step7A → Step7D → Step8 → Step9（7B/7C 的中/大尺度输入不变）。
只写新配置与新产物目录，不动 md1 / md2_* 既有结果。

用法::

    python3 太古界/tools/make_md2_noridge_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TAG = "md2_noridge"

JOBS: list[tuple[str, str, dict[str, object]]] = [
    (
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_md1.json",
        f"太古界/step7a_small_scale_dfn/configs/taigu_step7a_{TAG}.json",
        {"output_dir": f"太古界/step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_{TAG}"},
    ),
    (
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_md1.json",
        f"太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_{TAG}.json",
        {"output_dir": f"太古界/step7d_multiscale_fused_dfn/output/taigu_step7d_fused_v1_{TAG}"},
    ),
    (
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_md1.json",
        f"太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_{TAG}.json",
        {"output_dir": f"太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_{TAG}"},
    ),
    (
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_md1.json",
        f"太古界/step9_sections/configs/taigu_step9_multibackground_v2_{TAG}.json",
        {"output_dir": f"太古界/step9_sections/output/taigu_step9_multibackground_v2_{TAG}"},
    ),
]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate md2_noridge experiment configs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for source_name, target_name, overrides in JOBS:
        source_path = REPO / source_name
        target_path = REPO / target_name
        config = json.loads(source_path.read_text(encoding="utf-8"))
        for key, value in overrides.items():
            if key == "output_dir":
                config["output_dir"] = str(REPO / str(value))
            else:
                config[key] = value
        config["version"] = f"{config.get('version', '')}"
        if "step7a" in target_name:
            orientation = dict(config.get("orientation", {}))
            orientation["local_high_quality"] = 1.01      # >1 ⇒ 永远不满足 ⇒ 关掉 local_gradient_ridge
            orientation["local_medium_quality"] = 1.01    # >1 ⇒ 关掉 multiwell_local_blend
            config["orientation"] = orientation
            config["version"] = f"taigu_step7a_attribute_v3_regen_{TAG}"
        elif "step7d" in target_name:
            config["small_dfn_csv"] = str(
                REPO / f"太古界/step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_{TAG}/fracture_patches.csv"
            )
            config["version"] = f"taigu_step7d_fused_v1_{TAG}"
        elif "step8" in target_name:
            for key, name in (
                ("initial_dfn_csv", "fused_multiscale_dfn_patches.csv"),
                ("initial_dfn_vtk", "fused_multiscale_dfn_raw_time.vtk"),
                ("initial_dfn_summary_json", "fused_multiscale_summary.json"),
            ):
                config[key] = str(REPO / f"太古界/step7d_multiscale_fused_dfn/output/taigu_step7d_fused_v1_{TAG}" / name)
            config["version"] = f"taigu_step8_attribute_multiscale_v1_{TAG}"
        elif "step9" in target_name:
            config["step8_patches_csv"] = str(
                REPO
                / f"太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_{TAG}"
                / "well_corrected_dfn_fracture_patches.csv"
            )
            config["version"] = f"taigu_step9_multibackground_v2_{TAG}"
        config["_note"] = (
            f"{TAG}（2026-09-22 对照实验）：Step7A 关闭局部脊产状分支"
            "（local_high_quality / local_medium_quality 抬到 1.01），7A 片全部使用多井族产状；"
            "其余口径与 md1 完全一致。"
        )
        before = flatten(json.loads(source_path.read_text(encoding="utf-8")))
        after = flatten(config)
        changes = {
            key: (before.get(key), after.get(key))
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        print(f"=== {target_path.relative_to(REPO)}  （{len(changes)} 处改动）")
        for key, (old, new) in changes.items():
            print(f"    {key}\n      旧: {old}\n      新: {new}")
        if not args.dry_run:
            target_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
