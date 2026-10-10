#!/usr/bin/env python3
"""生成 Step9「展示口径」对照小样配置（2026-09-29）。

背景：太古界矿区剖面原来把"画多少条地震道"和"画布多大"写死（15.5x7.8 in + 固定 80 道），
20.6 km 剖面上每 262 m 才有 1 条道 → 既稀疏、横向摆幅又被放大到 ±110 m；
变密度还是红白蓝。砂砾岩 formal 是相反的：**黑白变密度 + 画剖面上全部道 + 更大的画布**。

本脚本从正式 quanqu1 配置派生 3 组小样配置（只覆盖展示参数，不重采样）：

* `probeA`：全画 1650 道（100%）+ 填充 alpha 0.55 / 线宽 0.35
* `probeB`：按米抽道 25 m（825 道，50%）+ 同 A 的深色参数
* `probeC`：全画 1650 道 + 填充 alpha 0.75 / 线宽 0.45（对照"更黑"的观感）

三者都用 5280 x 1968 px 画布、黑白变密度（红白蓝另存 backup 目录），
并通过 `--reuse-samples` 复用正式 quanqu1 的 `section_samples/*.npz`，只重画图。

用法::

    python3 太古界/tools/make_step9_style_probe_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FORMAL = REPO / "太古界/step9_sections/configs/taigu_step9_multibackground_v2_quanqu1.json"
FORMAL_OUTPUT = REPO / "太古界/step9_sections/output/taigu_step9_multibackground_v2_quanqu1"
PROBE_ROOT = REPO / "太古界/step9_sections/output/_step9_style_probe"

# 画布与配色（三组共用）：对齐砂砾岩 10 km demo 的多背景口径
COMMON: dict[str, object] = {
    "seismic_density_style": "absolute_grayscale",       # 黑白：|振幅| 上 gray_r，0=白
    "seismic_density_backup_style": "signed_red_white_blue",
    "seismic_density_backup_dir": "red_white_blue_backup",
    "render_dpi": 240,
    "xz_total_width_px": 5280,
    "yz_total_width_px": 5280,
    "figure_height_px": 1968,
    "overlay_legend_width_in": 6.2,
    "overlay_legend_fontsize": 8.4,
    "wiggle_lateral_scale_fraction": 0.42,
    # 只出 overview 的地震两张（变密度 + 变面积），编号沿用正式 20 张的序号
    "render_scopes": ["overview"],
    "reuse_samples_dir": str(FORMAL_OUTPUT),
}

VARIANTS: dict[str, dict[str, object]] = {
    "probeA": {
        "wiggle_max_trace_count": 0,      # 全画
        "wiggle_linewidth": 0.35,
        "wiggle_positive_fill_alpha": 0.55,
        "render_tokens": ["seismic_density", "seismic_wiggle_area"],
        "_probe_note": "全画 1650 道（100%）+ alpha 0.55 / lw 0.35",
    },
    "probeB": {
        "wiggle_max_trace_count": None,   # 用按米抽道
        "wiggle_trace_spacing_m": 25.0,
        "wiggle_linewidth": 0.35,
        "wiggle_positive_fill_alpha": 0.55,
        "render_tokens": ["seismic_wiggle_area"],
        "_probe_note": "按米抽道 25 m（825 道，50%）+ alpha 0.55 / lw 0.35",
    },
    "probeC": {
        "wiggle_max_trace_count": 0,
        "wiggle_linewidth": 0.45,
        "wiggle_positive_fill_alpha": 0.75,
        "render_tokens": ["seismic_wiggle_area"],
        "_probe_note": "全画 1650 道 + alpha 0.75 / lw 0.45（对照更黑的观感）",
    },
}


def main() -> int:
    base = json.loads(FORMAL.read_text(encoding="utf-8"))
    for name, overrides in VARIANTS.items():
        config = dict(base)
        config.update(COMMON)
        config.update({k: v for k, v in overrides.items() if v is not None})
        config["output_dir"] = str(PROBE_ROOT / name)
        config["_note"] = f"Step9 展示口径对照小样 {name}：{overrides.get('_probe_note', '')}；只覆盖展示参数，不重采样。"
        target = REPO / f"太古界/step9_sections/configs/taigu_step9_multibackground_v2_quanqu1_{name}.json"
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"写出 {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
