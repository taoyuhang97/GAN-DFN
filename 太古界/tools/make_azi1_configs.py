#!/usr/bin/env python3
"""生成 azi1 轮次的各步配置（Step6B → Step9）：方位口径统一为"倾向方位 0–360 + 倾角"。

做法与 `make_md1_configs.py` 一致：从 **md1** 配置出发，只做**目录/文件名替换**
（md1 → azi1，仅限受影响的那几步），再补写少量口径键；其余字段逐字保留。

不受影响的步骤：Step1–Step5、Step6A、Step6D（它们不含方位字段）。

用法::

    python3 太古界/tools/make_azi1_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

# 只替换"受方位口径影响"的目录片段（md1 → azi1）
REWRITES: list[tuple[str, str]] = [
    ("taigu_medium_v7_anttrack_led_md1", "taigu_medium_v7_anttrack_led_azi1"),
    ("taigu_step6c_large_v3_attribute_v3_md1", "taigu_step6c_large_v3_attribute_v3_azi1"),
    ("taigu_step7a_attribute_v3_regen_full_md1", "taigu_step7a_attribute_v3_regen_full_azi1"),
    ("taigu_step7c_large_v3_attribute_v3_md1", "taigu_step7c_large_v3_attribute_v3_azi1"),
    ("taigu_step7d_fused_v1_md1", "taigu_step7d_fused_v1_azi1"),
    ("taigu_step8_attribute_multiscale_v1_md1", "taigu_step8_attribute_multiscale_v1_azi1"),
    ("taigu_step9_multibackground_v2_md1", "taigu_step9_multibackground_v2_azi1"),
]


def azi1_orientation_patch(config: dict) -> dict:
    """把 7A 的 `orientation` 块改成新口径：fallback 用 `dip_azimuth_deg`，删旧开关。"""
    orientation = dict(config.get("orientation", {}))
    orientation.pop("family_azimuth_semantics", None)
    orientation.pop("ridge_azimuth_semantics", None)
    fallback = dict(orientation.get("fallback_family", {}))
    patched = {}
    for layer, entry in fallback.items():
        entry = dict(entry)
        if "dip_azimuth_deg" not in entry:
            # 旧配置的 `azimuth_deg` 语义是"走向" → 倾向方位 = 走向 + 90°
            entry["dip_azimuth_deg"] = (float(entry.get("azimuth_deg", 0.0)) + 90.0) % 360.0
        patched[layer] = entry
    if patched:
        orientation["fallback_family"] = patched
    config["orientation"] = orientation
    return config


JOBS: list[tuple[str, str, dict[str, object]]] = [
    (
        "太古界/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_md1.json",
        "太古界/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_azi1.json",
        {
            "_note": (
                "azi1（2026-09-22）：分量产状统一为**真倾向方位 0–360**"
                "（新增 pca_dip_azimuth_deg，pca_azimuth_deg 降级为派生走向）。"
                "输入密度体与 md1 相同；中尺度几何参数不变。"
            )
        },
    ),
    (
        "太古界/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_md1.json",
        "太古界/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_azi1.json",
        {
            "_note": (
                "azi1（2026-09-22）：断层面板产状统一为真倾向方位；VTK 增 DipAzimuthDeg，"
                "AzimuthDeg 降级为派生走向。"
            )
        },
    ),
    (
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_md1.json",
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_azi1.json",
        {
            "_note": (
                "azi1（2026-09-22）：家族聚类改为**全圆倾向方位**（删除 family_azimuth_semantics / "
                "ridge_azimuth_semantics 两个旧开关）；片表新增 DipAzimuthDeg，AzimuthDeg 为派生走向。"
                "输入密度体仍取 Step6D md1（密度与方位无关）。"
            )
        },
    ),
    (
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_md1.json",
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi1.json",
        {"_note": "azi1（2026-09-22）：输入改为 Step6B azi1（含 pca_dip_azimuth_deg）；局部带 PCA 改倾向方位口径。"},
    ),
    (
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_md1.json",
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi1.json",
        {"_note": "azi1（2026-09-22）：输入改为 Step6C azi1；断层产状改倾向方位口径。"},
    ),
    (
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_md1.json",
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi1.json",
        {"_note": "azi1（2026-09-22）：输入改为 7A/7B/7C azi1；融合表透传 DipAzimuthDeg。"},
    ),
    (
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_md1.json",
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi1.json",
        {
            "_note": (
                "azi1（2026-09-22）：井控产状写 DipAzimuthDeg（成像真值直接用甲方 Frac_Azimuth）；"
                "顶点全量按口径重建（rebuild_all_vertices=true）。"
            ),
            "rebuild_all_vertices": True,
        },
    ),
    (
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_md1.json",
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi1.json",
        {
            "_note": (
                "azi1（2026-09-22）：DFN 迹线按 DipAzimuthDeg + apparent_dip_trace 绘制，"
                "默认 geometry 模式（与砂砾岩一致的『真实交线』）。"
            ),
            "dfn_draw_mode": "geometry",
            "dfn_azimuth_semantics": "dip_azimuth",
            "step8_vtk": str(REPO / "太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_azi1/well_corrected_dfn_raw_time.vtk"),
        },
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate azi1 step configs by path rewriting.")
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
        if "step7a_small_scale_dfn" in target:
            updated = azi1_orientation_patch(updated)
        updated.update(extra)
        changed = sum(
            1 for key in set(original) | set(updated) if original.get(key) != updated.get(key)
        )
        print(f"=== {target}  （{changed} 处顶层改动）")
        print(f"    output_dir: {updated.get('output_dir')}")
        if not args.dry_run:
            target_path.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
