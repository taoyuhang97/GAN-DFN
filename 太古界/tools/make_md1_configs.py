#!/usr/bin/env python3
"""生成 md1 轮次的各步配置（Step4 → Step9）。

只做**目录/文件名替换**：把 v5 / v6 / v8 的口径产物路径换成 md1 版本，
其余字段逐字保留（生成后会打印每个配置的改动清单，便于人工核对）。

用法::

    python3 太古界/tools/make_md1_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

# 目录片段替换表（顺序敏感：长串在前，避免子串误替换）
REWRITES: list[tuple[str, str]] = [
    ("taigu_step1_contracts_v1", "taigu_step1_contracts_md1"),
    ("taigu_step2_regular_v5", "taigu_step2_regular_md1"),
    ("taigu_step3_imaging_v5", "taigu_step3_imaging_md1"),
    ("taigu_step4_gr_rd_rs_v5", "taigu_step4_gr_rd_rs_md1"),
    ("taigu_step5_attribute_v3_common_contract_v5", "taigu_step5_attribute_v3_common_contract_md1"),
    ("taigu_step6a_attribute_v3_full_v5", "taigu_step6a_attribute_v3_full_md1"),
    ("taigu_step6a_attribute_v3_v5", "taigu_step6a_attribute_v3_md1"),
    ("taigu_step6a_small_background_v5", "taigu_step6a_small_background_md1"),
    ("taigu_medium_v7_anttrack_led_v5", "taigu_medium_v7_anttrack_led_md1"),
    ("taigu_step6c_large_v3_attribute_v3_v5", "taigu_step6c_large_v3_attribute_v3_md1"),
    ("taigu_step6d_multiscale_v5", "taigu_step6d_multiscale_md1"),
    ("taigu_step7a_attribute_v3_regen_full_v6", "taigu_step7a_attribute_v3_regen_full_md1"),
    ("taigu_step7c_large_v3_attribute_v3_v5", "taigu_step7c_large_v3_attribute_v3_md1"),
    ("taigu_step7d_fused_v1_v6", "taigu_step7d_fused_v1_md1"),
    ("taigu_step8_attribute_multiscale_v1_v6", "taigu_step8_attribute_multiscale_v1_md1"),
    ("taigu_step9_multibackground_v2_v8", "taigu_step9_multibackground_v2_md1"),
]

# (源配置, 目标配置, 版本号/备注)
JOBS: list[tuple[str, str, dict[str, str]]] = [
    (
        "太古界/step4_fracture_prediction/configs/taigu_step4_gr_rd_rs_v5.json",
        "太古界/step4_fracture_prediction/configs/taigu_step4_gr_rd_rs_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 Step2/Step3 的 MD 口径产物；算法与阈值与 v5 完全一致。"},
    ),
    (
        "太古界/step5_virtual_wells/configs/taigu_step5_attribute_v3_v5.json",
        "太古界/step5_virtual_wells/configs/taigu_step5_attribute_v3_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 Step4/Step2/Step3 的 MD 口径产物；虚拟井几何口径不变。"},
    ),
    (
        "太古界/step6a_density_volume/configs/taigu_step6a_train_attribute_v3_v5.json",
        "太古界/step6a_density_volume/configs/taigu_step6a_train_attribute_v3_md1.json",
        {"_note": "md1（2026-09-22）：训练样本改为 Step5 md1；特征与模型参数不变。"},
    ),
    (
        "太古界/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_v5.json",
        "太古界/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_md1.json",
        {"_note": "md1（2026-09-22）：模型与输出目录改为 md1；采样网格与归一化合同不变。"},
    ),
    (
        "太古界/step6a_density_volume/configs/taigu_step6a_small_background_v5.json",
        "太古界/step6a_density_volume/configs/taigu_step6a_small_background_md1.json",
        {"_note": "md1（2026-09-22）：输入密度体改为 md1。"},
    ),
    (
        "太古界/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_v5.json",
        "太古界/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_md1.json",
        {"_note": "md1（2026-09-22）：输入密度体改为 Step6A md1；中尺度参数不变。"},
    ),
    (
        "太古界/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_v5.json",
        "太古界/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_md1.json",
        {"_note": "md1（2026-09-22）：输入密度体改为 Step6A md1；大尺度参数不变。"},
    ),
    (
        "太古界/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_v5.json",
        "太古界/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_md1.json",
        {"_note": "md1（2026-09-22）：6A/6B/6C 输入与输出目录改为 md1。"},
    ),
    (
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_v6.json",
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 6D/6A/Step2/Step3 的 md1 产物；采样与产状口径不变（方位约定问题不在本轮）。"},
    ),
    (
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_v6.json",
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 Step6B md1。"},
    ),
    (
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_v6.json",
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 Step6C md1。"},
    ),
    (
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_v6.json",
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 7A/7B/7C md1。"},
    ),
    (
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_v6.json",
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 7D/7C/Step4/Step3/Step2 的 md1 产物；井控逻辑不变。"},
    ),
    (
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_v8.json",
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_md1.json",
        {"_note": "md1（2026-09-22）：输入改为 Step8/7C/Step2/Step3/Step4 的 md1 产物；绘制口径与 v8 完全一致（纵向 padding 仍 100 ms）。"},
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
    parser = argparse.ArgumentParser(description="Generate md1 step configs by path rewriting.")
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
        updated.update(extra)
        before, after = flatten(original), flatten(updated)
        changes = {
            key: (before.get(key), after.get(key))
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        print(f"=== {target}  （{len(changes)} 处改动）")
        for key, (old, new) in changes.items():
            print(f"    {key}\n      旧: {old}\n      新: {new}")
        if not args.dry_run:
            target_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
