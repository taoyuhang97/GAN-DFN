#!/usr/bin/env python3
"""生成 quanqu1（太古界**矿区/全区**）轮次的各步配置（Step6A 预测 → Step9）。

## 为什么从 Step6A 预测开始，且不需要重训

* **训练样本本来就是全区尺度的**：Step5 用的是全区 `attribute_trace_header`（2,156,550 道），
  所以 `taigu_step6a_attribute_v3_md1/models/...joblib` 这个模型可以直接复用 → **不重训**。
* **预测范围由 `demo_grid_csv` 决定**：旧配置指向 5×5 km 的 `taigu_attribute_demo_grid_v2_405center`
  （160,801 道）。本脚本改为指向新建的全区格网
  `taigu_attribute_grid_quanqu1`（1650×1307 = 2,156,550 道，X 658887–679500、Y 4231662–4247987）。
* 因此 6A 预测及其下游（6A 小尺度背景、6B/6C/6D、7A–7D、8、9）全部要按新范围重跑。

## 命名

统一后缀 **`quanqu1`**，与现有 `md1 / md2_* / md3_* / azi1..azi4 / v4..v8` 明显区分，避免混淆。

## 输入来源（口径）

* 6A 预测 / 6B / 6C / 6D：从 **md1** 配置派生（深度口径 = MD）；
* 7A–7D / 8 / 9：从 **azi4** 配置派生（产状口径 = 倾向方位 + 按井经验分布重采样 + `apparent_dip` 画法）。

用法::

    python3 太古界/tools/make_quanqu1_configs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

# 目录名替换表（顺序敏感：更具体的在前）
REWRITES: list[tuple[str, str]] = [
    # 6A 预测体（注意：不要匹配到训练产物 `taigu_step6a_attribute_v3_md1`）
    ("taigu_step6a_attribute_v3_full_md1", "taigu_step6a_attribute_v3_full_quanqu1"),
    ("taigu_step6a_small_background_md1", "taigu_step6a_small_background_quanqu1"),
    # 6B / 6C（6B 的 azi1 产物也并入 quanqu1）
    ("taigu_medium_v7_anttrack_led_md1", "taigu_medium_v7_anttrack_led_quanqu1"),
    ("taigu_medium_v7_anttrack_led_azi1", "taigu_medium_v7_anttrack_led_quanqu1"),
    ("taigu_medium_v7_anttrack_led_azi4", "taigu_medium_v7_anttrack_led_quanqu1"),
    ("taigu_step6c_large_v3_attribute_v3_md1", "taigu_step6c_large_v3_attribute_v3_quanqu1"),
    ("taigu_step6c_large_v3_attribute_v3_azi1", "taigu_step6c_large_v3_attribute_v3_quanqu1"),
    # 6D
    ("taigu_step6d_multiscale_md1", "taigu_step6d_multiscale_quanqu1"),
    # 7A–7D / 8 / 9
    ("taigu_step7a_attribute_v3_regen_full_azi4", "taigu_step7a_attribute_v3_regen_full_quanqu1"),
    ("taigu_step7c_large_v3_attribute_v3_azi4", "taigu_step7c_large_v3_attribute_v3_quanqu1"),
    ("taigu_step7d_fused_v1_azi4", "taigu_step7d_fused_v1_quanqu1"),
    ("taigu_step8_attribute_multiscale_v1_azi4", "taigu_step8_attribute_multiscale_v1_quanqu1"),
    ("taigu_step9_multibackground_v2_azi4", "taigu_step9_multibackground_v2_quanqu1"),
    # 采样格网：5 km demo 格网 → 全区格网
    ("taigu_attribute_demo_grid_v2_405center", "taigu_attribute_grid_quanqu1"),
]

NOTE = (
    "quanqu1（2026-09-23）：**太古界矿区/全区**范围（1650×1307 = 2,156,550 道，"
    "X 658887–679500 / Y 4231662–4247987）。预测范围由全区格网 taigu_attribute_grid_quanqu1 决定；"
    "模型沿用 md1 训练产物（Step5 训练样本本就是全区尺度，无需重训）。"
)

# 2026-09-26 修复：`target_block` 是"范围"字段，派生时应一并改成全区。
# 首轮生成时只替换了目录名与格网路径，漏掉了这个坐标块，导致 7C/7D/8/9 仍按老的
# 5 km（405 井中心）范围过滤 —— 7C 因此丢掉 205/556 个推断面并 `status=fail`。
TARGET_BLOCK: dict[str, object] = {
    "name": "taigu_quanqu1_full_area",
    "x_min": 658887.0,
    "x_max": 679500.0,
    "y_min": 4231662.0,
    "y_max": 4247987.0,
}

JOBS: list[tuple[str, str, dict]] = [
    (
        "太古界/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_md1.json",
        "太古界/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step6a_density_volume/configs/taigu_step6a_small_background_md1.json",
        "太古界/step6a_density_volume/configs/taigu_step6a_small_background_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_md1.json",
        "太古界/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_md1.json",
        "太古界/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_md1.json",
        "太古界/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_azi4.json",
        "太古界/step7a_small_scale_dfn/configs/taigu_step7a_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi4.json",
        "太古界/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi4.json",
        "太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_quanqu1.json",
        {
            "_note": NOTE,
            # 全区轮次：6C 的极少数推断面落在层位解释无效的道上（矿区边缘）或中心略低于 Base，
            # 7C 会丢弃并把原因写进 summary 的 upstream_component_drop_audit；覆盖率闸门放到 0.98。
            "min_upstream_component_coverage": 0.98,
        },
    ),
    (
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi4.json",
        "太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi4.json",
        "太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_quanqu1.json",
        {"_note": NOTE},
    ),
    (
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_azi4.json",
        "太古界/step9_sections/configs/taigu_step9_multibackground_v2_quanqu1.json",
        {
            "_note": NOTE,
            # 全区轮次：剖面横跨整个矿区（一张总剖面），overview 标签同步改名
            "expected_demo_trace_count": 2156550,
            "scope_labels": {"overview": "矿区全区", "local_200m": "井周200 m"},
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
    parser = argparse.ArgumentParser(description="Generate quanqu1 (mine-area) step configs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for source, target, extra in JOBS:
        source_path, target_path = REPO / source, REPO / target
        if not source_path.exists():
            print(f"[skip] 源配置不存在: {source}")
            continue
        original = json.loads(source_path.read_text(encoding="utf-8"))
        updated = rewrite(original)
        if not isinstance(updated, dict):
            raise TypeError(f"config root must be an object: {source}")
        # 范围字段统一改成全区（含 target_block 的步骤都要改）
        if "target_block" in updated:
            updated["target_block"] = dict(TARGET_BLOCK)
        updated.update(extra)
        # 一致性检查：不应残留任何 md/azi 的“范围相关”目录
        residue = [
            key
            for key, value in updated.items()
            if isinstance(value, str)
            and any(tag in value for tag in ("_md1", "_azi1", "_azi2", "_azi3", "_azi4", "_v4", "_v5", "_v6"))
            and "taigu_step6a_attribute_v3_md1/models" not in value  # 训练模型是有意复用
            and "taigu_step4_gr_rd_rs_md1" not in value  # 井上预测与范围无关
        ]
        print(f"=== {target_path.relative_to(REPO)}")
        print(f"    output_dir: .../{Path(str(updated.get('output_dir', ''))).name}")
        if "demo_grid_csv" in updated:
            print(f"    demo_grid_csv: ...{str(updated['demo_grid_csv']).split('/')[-2:]}")
        if "attribute_demo_grid_csv" in updated:
            print(f"    attribute_demo_grid_csv: ...{str(updated['attribute_demo_grid_csv']).split('/')[-2:]}")
        if "target_block" in updated:
            block = updated["target_block"]
            in_full_area = (
                float(block["x_min"]) == float(TARGET_BLOCK["x_min"])
                and float(block["x_max"]) == float(TARGET_BLOCK["x_max"])
                and float(block["y_min"]) == float(TARGET_BLOCK["y_min"])
                and float(block["y_max"]) == float(TARGET_BLOCK["y_max"])
            )
            print(f"    target_block: {block.get('name')} 全区={in_full_area}")
        print(f"    仍需确认的旧版本键: {residue if residue else '无'}")
        if not args.dry_run:
            target_path.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
