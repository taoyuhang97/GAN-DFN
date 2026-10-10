#!/usr/bin/env python3
"""生成 azi2 轮次的配置（Step7A → 9）：收紧"局部几何来源"的倾角门槛。

## 为什么

azi1 跑完后剖面核对发现两类残留问题，根因是同一个：

1. DFN 里存在大量**近水平**的片（7B 中尺度 61% 小于 40°，中位 33.9°；
   7A 的 `local_gradient_ridge` 中位 31°、81% 小于 40°）；
2. 走向分布多出一撮 ~136°（正好来自这些局部脊线片，其倾向方位均值 46°）。

而**多口成像测井的统计**是：1327 个点倾角中位 64.0°，P5=30.5°、P10=37.6°，
小于 30° 仅占 4.5%、小于 40° 占 11.8%。也就是说"统计来源"（7A 的
`multiwell_family`，倾角中位 61°、0% 小于 40°）是对的，
**"局部几何来源"的门槛太松**才导致偏差。

## 改了什么

| 步 | 键 | azi1 | azi2 | 作用 |
| --- | --- | --- | --- | --- |
| 7A | `orientation.min_dip_deg` | 25.0 | **38.0** | 局部脊线倾角 <38° 判为不可用 → 退回家族统计（61°） |
| 7B | `local_geometry_min_dip_deg` | 20.0 | **38.0** | 局部带 PCA 倾角 <38° → 改用 Step6 全局分量产状（70°） |
| 7B | `orientation_safety_min_dip_deg` | 15.0 | **25.0** | 数值截断下限，避免极端浅值通过 |

38° 取自成像真值的 P10（37.6°），即"只保留真实数据里确实存在的浅倾角比例"。

用法::

    python3 太古界/tools/make_azi2_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

REWRITES: list[tuple[str, str]] = [
    ("taigu_step7a_attribute_v3_regen_full_azi1", "taigu_step7a_attribute_v3_regen_full_azi2"),
    ("taigu_medium_v7_anttrack_led_azi1", "taigu_medium_v7_anttrack_led_azi2"),
    ("taigu_step7c_large_v3_attribute_v3_azi1", "taigu_step7c_large_v3_attribute_v3_azi2"),
    ("taigu_step7d_fused_v1_azi1", "taigu_step7d_fused_v1_azi2"),
    ("taigu_step8_attribute_multiscale_v1_azi1", "taigu_step8_attribute_multiscale_v1_azi2"),
    ("taigu_step9_multibackground_v2_azi1", "taigu_step9_multibackground_v2_azi2"),
]

# 7B 的**输入**来自 Step6B；本轮不重跑 6B，因此这些键必须指回 azi1 产物。
# （输出目录仍走 azi2。）
STEP7B_STEP6B_INPUT_KEYS = (
    "medium_prior_sgy",
    "medium_components_npz",
    "medium_component_summary_csv",
    "step6b_qc_json",
)

NOTE = (
    "azi2（2026-09-22）：在 azi1 口径统一的基础上，收紧『局部几何来源』的倾角门槛"
    "（7A min_dip_deg=38、7B local_geometry_min_dip_deg=38），使浅倾角片退回"
    "多井统计来源。依据：多口成像测井 1327 点倾角中位 64.0°、P10=37.6°、<40° 仅 11.8%。"
)

JOBS: list[tuple[str, str, dict]] = [
    (
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_azi1.json",
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_azi2.json",
        {"_note": NOTE, "orientation_min_dip_deg": 38.0},
    ),
    (
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi1.json",
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi2.json",
        {
            "_note": NOTE,
            "local_geometry_min_dip_deg": 38.0,
            "orientation_safety_min_dip_deg": 25.0,
        },
    ),
    (
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi1.json",
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi2.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi1.json",
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi2.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi1.json",
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi2.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi1.json",
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi2.json",
        {"_note": NOTE},
    ),
]


def rewrite(value: object) -> object:
    if isinstance(value, str):
        out = value
        for old, new in REWRITES:
            out = out.replace(old, new)
        return out
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    return value


def apply_overrides(config: dict, extra: dict) -> dict:
    note = extra.pop("_note", None)
    if note:
        config["_note"] = note
    if "orientation_min_dip_deg" in extra:
        orientation = dict(config.get("orientation", {}))
        orientation["min_dip_deg"] = extra.pop("orientation_min_dip_deg")
        config["orientation"] = orientation
    config.update(extra)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate azi2 step configs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for source, target, extra in JOBS:
        source_path = REPO / source
        target_path = REPO / target
        if not source_path.exists():
            print(f"[skip] 源配置不存在: {source}")
            continue
        original = json.loads(source_path.read_text(encoding="utf-8"))
        updated = rewrite(original)
        if not isinstance(updated, dict):
            raise TypeError(f"config root must be an object: {source}")
        updated = apply_overrides(updated, dict(extra))
        if "step7b_multiscale_initial_dfn" in target:
            for key in STEP7B_STEP6B_INPUT_KEYS:
                if isinstance(updated.get(key), str) and "_azi2" in updated[key]:
                    updated[key] = updated[key].replace(
                        "taigu_medium_v7_anttrack_led_azi2",
                        "taigu_medium_v7_anttrack_led_azi1",
                    )
        print(f"=== {target}")
        print(f"    output_dir: {updated.get('output_dir')}")
        ori = updated.get("orientation", {})
        if "min_dip_deg" in ori:
            print(f"    orientation.min_dip_deg = {ori['min_dip_deg']}")
        for key in ("local_geometry_min_dip_deg", "orientation_safety_min_dip_deg"):
            if key in updated:
                print(f"    {key} = {updated[key]}")
        if not args.dry_run:
            target_path.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
