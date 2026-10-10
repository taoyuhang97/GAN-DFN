#!/usr/bin/env python3
"""生成 azi4 轮次的配置（Step7A → 9）：按 2026-09-23 方案改 DFN 产状赋值逻辑。

方案：`太古界/太古界DFN产状赋值逻辑修正方案_20260923.md`
* 只改 **7A 小尺度**（D1 全部点 / D2 最近几口井加权 + 空间相关抽井 / D3 β 门控）；
* **7B/7C 沿用 azi1 配置**（决策 D4：中尺度不动，保持蚂蚁体来源）。

上游复用：6B/6C 仍用 azi1 产物；7B/7C 的输出改到 azi4（因为 7D 要按尺度分别取）。

用法::

    python3 太古界/tools/make_azi4_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

NOTE = (
    "azi4（2026-09-23）：按《DFN产状赋值逻辑修正方案》改产状赋值——7A 改为"
    "『按井经验分布重采样（D1 全点）+ 最近几口井高斯加权与空间相关抽井（D2）+ "
    "局部几何仅在走向一致时小幅融合倾角（D3）』；7B/7C 按 D4 保持蚂蚁体来源不变。"
)

#: 7A 的产状赋值参数（方案 §5.3）
AZI4_ORIENTATION_KEYS: dict[str, object] = {
    "well_weight_kernel": "gaussian",
    "well_weight_scale_m": 6000.0,
    "well_weight_min_share": 0.02,
    "well_assignment_coherence_m": 800.0,
    "well_assignment_sigma_cells": 1.0,
    "well_assignment_seed": 20260923,
    "well_library_min_points": 5,
    "well_library_min_points_for_full_weight": 50,
    "well_library_small_sample_discount": 0.5,
    "azimuth_jitter_std_deg": 8.0,
    "dip_jitter_std_deg": 5.0,
    "dip_min_deg": 5.0,
    "dip_max_deg": 89.0,
    "local_geometry_strike_tolerance_deg": 30.0,
    "local_geometry_dip_blend_max": 0.30,
    "local_geometry_min_quality": 0.5,
}

#: 每个配置里"上游不变项"的显式钉住（相对仓库根）
PINNED: dict[str, dict[str, str]] = {
    "taigu_step7b_medium_v7_anttrack_led_azi4": {
        "medium_prior_sgy": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_prior.sgy",
        "medium_components_npz": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_components.npz",
        "medium_component_summary_csv": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_component_summary.csv",
        "step6b_qc_json": "太古界/step6b_medium_corridor/output/taigu_medium_v7_anttrack_led_azi1/medium_corridor_qc.json",
    },
    "taigu_step7c_large_v3_attribute_v3_azi4": {
        "large_prior_sgy": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/large_fault_prior.sgy",
        "large_prior_components_npz": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/large_fault_prior_components.npz",
        "large_component_summary_csv": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/large_fault_component_summary.csv",
        "original_fault_surface_vtk": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/original_fault_units_demo_raw_time.vtk",
        "original_fault_manifest_csv": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/original_fault_unit_manifest.csv",
        "fault_patch_overlap_csv": "太古界/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_azi1/original_fault_rasterization_audit.csv",
    },
    "taigu_step7d_fused_v1_azi4": {
        "original_fault_manifest_csv": "太古界/step7c_large_fault_dfn/output/taigu_step7c_large_v3_attribute_v3_azi4/original_fault_unit_manifest.csv",
    },
}

JOBS: list[tuple[str, str]] = [
    ("太古界/step7a_small_scale_dfn/configs/taigu_step7a_azi1.json",
     "太古界/step7a_small_scale_dfn/configs/taigu_step7a_azi4.json"),
    ("太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi1.json",
     "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi4.json"),
    ("太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi1.json",
     "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi4.json"),
    ("太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi1.json",
     "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi4.json"),
    ("太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi1.json",
     "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi4.json"),
    ("太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi1.json",
     "太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi4.json"),
]


def rewrite(value: object) -> object:
    if isinstance(value, str):
        return value.replace("_azi1", "_azi4")
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate azi4 step configs.")
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
        for key, rel in PINNED.get(target_path.stem, {}).items():
            updated[key] = str(REPO / rel)
        if "step7a_small_scale_dfn" in target:
            orientation_cfg = dict(updated.get("orientation", {}))
            orientation_cfg.update(AZI4_ORIENTATION_KEYS)
            updated["orientation"] = orientation_cfg
        print(f"=== {target}")
        print(f"    output_dir: .../{str(updated.get('output_dir')).split('/')[-1]}")
        leftover = [k for k, v in updated.items() if isinstance(v, str) and "_azi1" in v and "step6" not in v]
        print(f"    仍指向 azi1 的键: {leftover if leftover else '无'}")
        if not args.dry_run:
            target_path.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
