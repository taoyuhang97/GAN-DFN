#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build standalone legends for the two Step9 scale-wiggle figure sets.

Only the three presentation variants requested by the user are generated:
large, large+medium, and large+medium+small.  Each legend includes the well
trajectory, all interpreted horizons for that region, and the original fault
trace.  Sandstone fractures are grouped by stratigraphic interval, while the
Taigu legend explains the DFN with large/medium/small scale labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parents[1]
SAND_OUTPUT = (
    REPO
    / "优化阶段二"
    / "正式主线"
    / "output"
    / "step9_section_presentation"
    / "mine_v1"
    / "scale_wiggle_t4_t7_v1"
)
TAIGU_OUTPUT = (
    REPO
    / "太古界"
    / "supp_layer"
    / "output"
    / "taigu_supp_step9_scale_wiggle_v1"
)
TAIGU_CONFIG = (
    REPO
    / "太古界"
    / "supp_layer"
    / "configs"
    / "taigu_supp_step9_scale_wiggle.json"
)

WELL_COLOR = "#ffea00"
WELL_GLOW = "#5b4b00"
FAULT_COLOR = "#ffe600"
FAULT_GLOW = "#111827"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#475569"

@dataclass(frozen=True)
class LegendVariant:
    key: str
    title: str
    scales: tuple[str, ...]
    source_figures: tuple[str, str]


VARIANTS = (
    LegendVariant(
        "01_large",
        "大尺度DFN剖面",
        ("large",),
        ("01_large_xz_t4_t7.png", "02_large_yz_t4_t7.png"),
    ),
    LegendVariant(
        "02_large_medium",
        "大中尺度融合DFN剖面",
        ("medium", "large"),
        ("07_large_medium_xz_t4_t7.png", "08_large_medium_yz_t4_t7.png"),
    ),
    LegendVariant(
        "03_all_scales",
        "大中小尺度融合DFN剖面",
        ("small", "medium", "large"),
        ("09_all_scales_xz_t4_t7.png", "10_all_scales_yz_t4_t7.png"),
    ),
)


def register_simplified_chinese_font() -> str | None:
    """Register the SC face from Noto CJK rather than relying on TTC face 0."""
    try:
        from fontTools.ttLib import TTCollection
    except Exception:
        return None

    for candidate in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    ):
        if not candidate.exists():
            continue
        try:
            collection = TTCollection(str(candidate))
        except Exception:
            continue
        for index, face in enumerate(collection.fonts):
            try:
                family = str(face["name"].getDebugName(1) or "")
            except Exception:
                continue
            if "SC" not in family:
                continue
            target = (
                Path.home()
                / ".cache"
                / "taigu_step9_fonts"
                / f"{candidate.stem}-face{index}.otf"
            )
            try:
                if not target.exists() or target.stat().st_mtime < candidate.stat().st_mtime:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    face.save(str(target))
                font_manager.fontManager.addfont(str(target))
                name = font_manager.FontProperties(fname=str(target)).get_name()
                if name:
                    return name
            except Exception:
                continue
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sand_strata() -> list[dict[str, Any]]:
    # Mirrors SURFACE_LINE_STYLES and SURFACE_FILE_MAP in
    # build_all_area_section_visualization.py used by the sand figure set.
    return [
        {"label": "T4（沙三上底）", "color": "#f97316", "linestyle": "-", "linewidth": 1.55},
        {"label": "T5（沙三中底）", "color": "#06b6d4", "linestyle": (0, (4, 2)), "linewidth": 1.55},
        {"label": "T6（沙三下底）", "color": "#a855f7", "linestyle": "-", "linewidth": 1.55},
        {"label": "T7（沙四上底）", "color": "#ec4899", "linestyle": "-.", "linewidth": 1.55},
    ]


def taigu_strata(config: dict[str, Any]) -> list[dict[str, Any]]:
    names = {
        "TopTimeMs": "上部复合层顶（T-a-1）",
        "MidTimeMs": "太古界顶（Art_1）",
        "BaseTimeMs": "风化壳底（Art_d1-1）",
    }
    output = []
    for item in config["horizons"]:
        linestyle = item["linestyle"]
        if isinstance(linestyle, list):
            linestyle = tuple(linestyle)
            if len(linestyle) == 2 and isinstance(linestyle[1], list):
                linestyle = (linestyle[0], tuple(linestyle[1]))
        output.append(
            {
                "label": names[item["field"]],
                "color": item["color"],
                "linestyle": linestyle,
                "linewidth": 1.7,
            }
        )
    return output


def sand_fracture_layers() -> list[dict[str, Any]]:
    return [
        {
            "label": "沙三段裂缝（T4–T6）",
            "color": "#ff7a00",
            "linewidth": 2.2,
        },
        {
            "label": "沙四段裂缝（T6–T7）",
            "color": "#00a6ff",
            "linewidth": 2.2,
        },
    ]


def taigu_scale_layers(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Mirror the scale labels and legend strokes used by the Taigu sections."""
    layers = []
    for scale in ("large", "medium", "small"):
        style = config["scale_style"][scale]
        layers.append(
            {
                "label": style["label"],
                "color": style.get("legend_color", style["color"]),
                "linewidth": max(
                    float(style.get("legend_line_width", style["line_width"])),
                    0.8,
                ),
            }
        )
    return layers


def well_handle(label: str, linewidth: float) -> Line2D:
    handle = Line2D(
        [0],
        [0],
        color=WELL_COLOR,
        linewidth=linewidth,
        label=label,
    )
    return handle


def fault_handle(linewidth: float) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=FAULT_COLOR,
        linewidth=linewidth,
        linestyle=(0, (5, 3)),
        label="原始断层",
    )


def draw_group(
    ax,
    title: str,
    handles: list[Line2D],
) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        1.02,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="semibold",
        color=MUTED_COLOR,
    )
    ax.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=False,
        fontsize=13,
        handlelength=3.2,
        handletextpad=0.9,
        labelspacing=0.75,
        borderaxespad=0.0,
    )


def render_legend(
    region_name: str,
    well_name: str,
    variant: LegendVariant,
    strata: list[dict[str, Any]],
    fracture_layers: list[dict[str, Any]],
    fracture_group_title: str,
    well_linewidth: float,
    fault_linewidth: float,
    output_dir: Path,
) -> list[str]:
    title = f"{region_name}｜{well_name}：{variant.title}图例"
    fig = plt.figure(figsize=(14.5, 4.6), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.05, 1.75, 0.9),
        left=0.045,
        right=0.975,
        bottom=0.16,
        top=0.73,
        wspace=0.65,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    fracture_handles = [
        Line2D(
            [0],
            [0],
            color=item["color"],
            linewidth=item["linewidth"],
            label=item["label"],
        )
        for item in fracture_layers
    ]
    strata_handles = [
        Line2D(
            [0],
            [0],
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=item["linewidth"],
            label=item["label"],
        )
        for item in strata
    ]
    support_handles = [
        well_handle(f"{well_name}井轨迹", well_linewidth),
        fault_handle(fault_linewidth),
    ]

    draw_group(axes[0], fracture_group_title, fracture_handles)
    draw_group(axes[1], "地层界面", strata_handles)
    draw_group(axes[2], "井与构造", support_handles)

    fig.text(
        0.045,
        0.91,
        title,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="semibold",
        color=TEXT_COLOR,
    )
    fig.add_artist(
        Line2D(
            [0.045, 0.975],
            [0.84, 0.84],
            transform=fig.transFigure,
            color="#cbd5e1",
            linewidth=0.8,
        )
    )

    png_path = output_dir / f"legend_{variant.key}.png"
    svg_path = output_dir / f"legend_{variant.key}.svg"
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white",
    )
    plt.close(fig)
    return [str(png_path), str(svg_path)]


def write_manifest(
    output_dir: Path,
    region_name: str,
    well_name: str,
    variants: list[dict[str, Any]],
    strata: list[dict[str, Any]],
    fracture_layers: list[dict[str, Any]],
    fracture_group_title: str,
) -> Path:
    manifest = {
        "status": "pass",
        "region": region_name,
        "well_name": well_name,
        "scope_note": "XZ and YZ figures share the same legend",
        "variants": variants,
        "common_items": {
            "well_trajectory": f"{well_name}井轨迹",
            "original_fault": "原始断层",
        },
        "fracture_group_title": fracture_group_title,
        "strata": [
            {
                "label": item["label"],
                "color": item["color"],
                "linestyle": str(item["linestyle"]),
            }
            for item in strata
        ],
        "fracture_layers": [
            {
                "label": item["label"],
                "color": item["color"],
                "linewidth": item["linewidth"],
            }
            for item in fracture_layers
        ],
    }
    path = output_dir / "standalone_legend_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def build_region(
    region_name: str,
    well_name: str,
    output_dir: Path,
    strata: list[dict[str, Any]],
    fracture_layers: list[dict[str, Any]],
    fracture_group_title: str,
    well_linewidth: float,
    fault_linewidth: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for variant in VARIANTS:
        files = render_legend(
            region_name,
            well_name,
            variant,
            strata,
            fracture_layers,
            fracture_group_title,
            well_linewidth,
            fault_linewidth,
            output_dir,
        )
        variants.append(
            {
                "key": variant.key,
                "title": variant.title,
                "scales": list(variant.scales),
                "source_figures": list(variant.source_figures),
                "files": files,
            }
        )
        print(f"[standalone-legend] {region_name}: {variant.key}", flush=True)
    manifest = write_manifest(
        output_dir,
        region_name,
        well_name,
        variants,
        strata,
        fracture_layers,
        fracture_group_title,
    )
    print(f"[standalone-legend] manifest={manifest}", flush=True)
    return {"output_dir": str(output_dir), "variants": variants}


def main() -> int:
    font_name = register_simplified_chinese_font()
    if font_name:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    taigu_config = load_json(TAIGU_CONFIG)
    result = {
        "status": "pass",
        "font": font_name,
        "regions": {
            "sand": build_region(
                "砂砾岩",
                "车页1导眼",
                SAND_OUTPUT,
                sand_strata(),
                sand_fracture_layers(),
                "地层裂缝",
                well_linewidth=2.2,
                fault_linewidth=2.5,
            ),
            "taigu": build_region(
                "太古界",
                str(taigu_config["profile_well"]),
                TAIGU_OUTPUT,
                taigu_strata(taigu_config),
                taigu_scale_layers(taigu_config),
                "裂缝尺度",
                well_linewidth=2.2,
                fault_linewidth=float(taigu_config["fault_style"]["line_width"]),
            ),
        },
    }
    print(f"[standalone-legend] status=pass regions={list(result['regions'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
