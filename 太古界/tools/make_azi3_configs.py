#!/usr/bin/env python3
"""生成 azi3 轮次的配置（Step7B → 9）：中尺度兜底产状改用多井统计倾角先验。

## 为什么还要 azi3

azi2 收紧了 7A/7B 的局部几何倾角门槛后：

* 小尺度修好了（<40° 从 19% → 1%，中位 57.5° → 60.5°）；
* **中尺度仍然浅**（中位 37.7°，51% 小于 40°）。

进一步定位到根源：7B 的"全局兜底"取的是 **Step6B 分量自身的 PCA 倾角**，而

| | 6B 分量 pca_dip_deg 中位 | <40° 占比 |
| --- | --- | --- |
| 砂砾岩（参考基线） | 57.3° | 0.27 |
| **太古界** | **30.6°** | **0.75** |

太古界的中尺度分量是被"压扁"的薄片（水平拉长、TIME 方向很薄），用它的形状反算平面法向，
必然得到近水平的倾角。所以兜底兜到的还是浅值。

## 改了什么

* 代码：`build_medium_scale_dfn_v4.py` 的 `local_anttrack_ridge_geometry_pca()`，
  `use_global_orientation=True` 时改为
  ``dip = max(分量 PCA 倾角, global_fallback_dip_prior_deg[层])``。
* 配置：7B 新增 `global_fallback_dip_prior_deg`（按层），取自 7A 家族倾角均值：
  上部复合层 60.0°、太古界风化壳 57.0°。

上游复用：7A 用 azi2 产物（本轮不含 7A 改动）；6B/6C 仍用 azi1 产物。

用法::

    python3 太古界/tools/make_azi3_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

NOTE = (
    "azi3（2026-09-23）：中尺度\"全局兜底产状\"改用多口成像测井的统计倾角先验"
    "（global_fallback_dip_prior_deg），因为太古界 Step6B 分量本身是压扁薄片、"
    "其 PCA 倾角中位仅 30.6°（砂砾岩同名分量 57.3°）。7A 复用 azi2 产物。"
)

# 各配置的"上游不变项"：显式写死，避免整串替换把不该换的也换掉
PINNED: dict[str, dict[str, str]] = {
    "taigu_step7b_medium_v7_anttrack_led_azi3": {
        "medium_prior_sgy": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_prior.sgy",
        "medium_components_npz": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_components.npz",
        "medium_component_summary_csv": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_component_summary.csv",
        "step6b_qc_json": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_qc.json",
    },
    "taigu_step7c_large_v3_attribute_v3_azi3": {
        "large_prior_sgy": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/large_fault_prior.sgy",
        "large_prior_components_npz": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/large_fault_components.npz",
        "large_component_summary_csv": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/large_fault_component_summary.csv",
        "original_fault_surface_vtk": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/original_fault_units_raw_time.vtk",
        "original_fault_manifest_csv": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/original_fault_unit_manifest.csv",
        "fault_patch_overlap_csv": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/original_fault_rasterization_audit.csv",
    },
    "taigu_step7d_fused_v1_azi3": {
        "small_dfn_csv": "太古界/step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_azi2/fracture_patches.csv",
        "small_dfn_vtk": "太古界/step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_azi2/fracture_patches.vtk",
    },
}

JOBS: list[tuple[str, str]] = [
    ("太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi2.json",
     "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi3.json"),
    ("太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi2.json",
     "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi3.json"),
    ("太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi2.json",
     "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi3.json"),
    ("太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi2.json",
     "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi3.json"),
    ("太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi2.json",
     "太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi3.json"),
]


def rewrite(value: object) -> object:
    if isinstance(value, str):
        return value.replace("_azi2", "_azi3")
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    return value


def apply_pins(config: dict, target_stem: str) -> dict:
    pins = PINNED.get(target_stem, {})
    for key, rel in pins.items():
        config[key] = str(REPO / rel)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate azi3 step configs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for source, target in JOBS:
        source_path, target_path = REPO / source, REPO / target
        if not source_path.exists():
            print(f"[skip] 源配置不存在: {source}")
            continue
        original = json.loads(source_path.read_text(encoding="utf-8"))
        updated = rewrite(original)
        if not isinstance(updated, dict):
            raise TypeError(f"config root must be an object: {source}")
        updated["_note"] = NOTE
        target_stem = target_path.stem
        updated = apply_pins(updated, target_stem)
        if "step7b_multiscale_initial_dfn" in target:
            updated["global_fallback_dip_prior_deg"] = {
                "上部复合层": 60.0,
                "太古界风化壳": 57.0,
            }
        print(f"=== {target}")
        print(f"    output_dir: {updated.get('output_dir')}")
        if "global_fallback_dip_prior_deg" in updated:
            print(f"    global_fallback_dip_prior_deg = {updated['global_fallback_dip_prior_deg']}")
        leftover = [k for k, v in updated.items() if isinstance(v, str) and "_azi2" in v]
        print(f"    仍指向 azi2 的键: {leftover if leftover else '无'}")
        if not args.dry_run:
            target_path.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
