# -*- coding: utf-8 -*-
"""生成剖面图例样式预览图，供汇报前挑选尺度/断层图例样式。

输出：<正式主线>/output/step9_section_presentation/图例样式预览.png
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache-tyh")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


FORMAL_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = FORMAL_ROOT / "output" / "step9_section_presentation"


def setup_cjk_font():
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [family]
            plt.rcParams["axes.unicode_minus"] = False
            return family
    return None


def legend_ax(ax, title, handles, labels, fontsize=17):
    ax.set_title(title, fontsize=20, pad=18)
    ax.axis("off")
    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        fontsize=fontsize,
        frameon=True,
        framealpha=0.95,
        handlelength=3.2,
        handletextpad=1.0,
        borderpad=1.0,
    )


def main() -> int:
    setup_cjk_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # 当前方案：三个尺度不同颜色细线 + 原始断层 + 彩色层位线
    handles_current = [
        Line2D([0], [0], color="#16a34a", linestyle="-", linewidth=2.0, label="小尺度裂缝"),
        Line2D([0], [0], color="#2563eb", linestyle="-", linewidth=2.0, label="中尺度裂缝"),
        Line2D([0], [0], color="#dc2626", linestyle="-", linewidth=2.6, label="大尺度裂缝"),
        Line2D([0], [0], color="#ffe600", linestyle="-", linewidth=3.0, label="原始断层"),
        Line2D([0], [0], color="#16a34a", linestyle="-", linewidth=1.0, alpha=0.55, label="小尺度裂缝投影"),
        Line2D([0], [0], color="#f97316", linestyle="-", linewidth=2.6, label="T4 层位界面"),
        Line2D([0], [0], color="#06b6d4", linestyle=(0, (4, 2)), linewidth=2.4, label="T5 层位界面"),
        Line2D([0], [0], color="#a855f7", linestyle="-", linewidth=2.6, label="T6 层位界面"),
        Line2D([0], [0], color="#ec4899", linestyle="-.", linewidth=2.4, label="T7 层位界面"),
    ]
    legend_ax(axes[0], "当前方案：尺度配色（绿/蓝/红细线）+ 彩色层位线", handles_current, [h.get_label() for h in handles_current])

    # 层位线候选配色对比（供挑选）
    candidates = [
        ("T4 橙 / T5 青 / T6 紫 / T7 粉★当前", "#f97316", "#06b6d4", "#a855f7", "#ec4899"),
        ("T4 红 / T5 绿 / T6 蓝 / T7 黄", "#dc2626", "#16a34a", "#2563eb", "#eab308"),
        ("T4 黄 / T5 青 / T6 蓝 / T7 红", "#eab308", "#06b6d4", "#2563eb", "#dc2626"),
    ]
    handles_cand = [
        Line2D([0], [0], color=c0, linestyle="-", linewidth=2.6, label=label)
        for label, c0, c1, c2, c3 in candidates
    ]
    legend_ax(axes[1], "层位线候选配色（T4/T5/T6/T7）", handles_cand, [h.get_label() for h in handles_cand])

    fig.tight_layout()
    out_path = OUT_DIR / "图例样式预览.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
