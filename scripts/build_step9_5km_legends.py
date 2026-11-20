#!/usr/bin/env python3
"""Build standalone legends for the unified DFN sections."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import matplotlib
from matplotlib import font_manager, patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_imaging_log_spec = importlib.util.spec_from_file_location(
    "legend_step9_imaging_log_overlay",
    REPO_ROOT / "common" / "step9_imaging_log_overlay.py",
)
if _imaging_log_spec is None or _imaging_log_spec.loader is None:
    raise ImportError("cannot load imaging-log overlay helper")
imaging_log_overlay = importlib.util.module_from_spec(_imaging_log_spec)
_imaging_log_spec.loader.exec_module(imaging_log_overlay)
IMAGING_LOG_STYLE = imaging_log_overlay.IMAGING_LOG_STYLE


SCALE_STYLES = {
    "small": ("小尺度", 1.10),
    "medium": ("中尺度", 2.20),
    "large": ("大尺度", 3.20),
}
SAND_LAYERS = [
    ("沙三段（T4–T6）", "#ff7a00"),
    ("沙四段（T6–T7）", "#00a6ff"),
]
TAIGU_LAYERS = [
    ("上部复合层", "#ff7a00"),
    ("太古界风化壳", "#00a6ff"),
]
LEGEND_SETS = [
    ("legend_01_large", ("large",)),
    ("legend_02_large_medium", ("large", "medium")),
    ("legend_03_all_scales", ("large", "medium", "small")),
]


def resolve_chinese_font() -> Path | None:
    """Return a Simplified-Chinese Noto face extracted from the system TTC."""

    candidates = [
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
            Path(tempfile.gettempdir())
            / "gan_dfn_fonts"
            / "NotoSansCJKsc-Medium.otf",
        ),
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path(tempfile.gettempdir())
            / "gan_dfn_fonts"
            / "NotoSansCJKsc-Regular.otf",
        ),
    ]
    for ttc_path, cache_path in candidates:
        if cache_path.exists():
            return cache_path
        if not ttc_path.exists():
            continue
        try:
            from fontTools.ttLib import TTCollection

            collection = TTCollection(str(ttc_path), lazy=False)
            for face in collection.fonts:
                if face["name"].getDebugName(1) == "Noto Sans CJK SC":
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp.otf")
                    face.save(str(temporary))
                    temporary.replace(cache_path)
                    return cache_path
        except (ImportError, KeyError, OSError):
            pass
    fallback = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    return fallback if fallback.exists() else None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成砂砾岩/太古界独立剖面图例")
    p.add_argument("--sand-output", type=Path)
    p.add_argument("--taigu-output", type=Path)
    return p


def outlined_handle(
    *,
    color: str,
    linewidth: float,
    label: str,
    linestyle: str = "-",
    halo_color: str | None = None,
    halo_width: float = 0.0,
) -> Line2D:
    handle = Line2D(
        [0],
        [0],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    if halo_color and halo_width > linewidth:
        handle.set_path_effects(
            [
                patheffects.Stroke(
                    linewidth=halo_width,
                    foreground=halo_color,
                ),
                patheffects.Normal(),
            ]
        )
    return handle


def section_legend_handles(
    *,
    layers: list[tuple[str, str]],
    scales: tuple[str, ...],
    well_label: str,
) -> list[Line2D]:
    """Build DFN, well and fault handles with the Taigu legend drawing contract."""

    handles = [
        Line2D([0], [0], color=color, linewidth=3.0, label=label)
        for label, color in layers
    ]
    for scale in scales:
        if scale == "large":
            handles.append(
                outlined_handle(
                    color="#6b7280",
                    linewidth=SCALE_STYLES[scale][1],
                    label=SCALE_STYLES[scale][0],
                    halo_color="#111827",
                    halo_width=5.2,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#111827",
                    linewidth=SCALE_STYLES[scale][1],
                    label=SCALE_STYLES[scale][0],
                )
            )
    handles.extend(
        [
            outlined_handle(
                color="#ffea00",
                linewidth=2.2,
                label=well_label,
                halo_color="#5b4b00",
                halo_width=3.6,
            ),
            outlined_handle(
                color=IMAGING_LOG_STYLE["color"],
                linewidth=IMAGING_LOG_STYLE["line_width"],
                label=IMAGING_LOG_STYLE["label"],
                halo_color=IMAGING_LOG_STYLE["halo_color"],
                halo_width=IMAGING_LOG_STYLE["halo_width"],
            ),
            outlined_handle(
                color="#ffe600",
                linewidth=2.3,
                label="原始断层",
                halo_color="#111827",
                halo_width=4.6,
            ),
        ]
    )
    return handles


def write_legends(
    output_dir: Path,
    layers: list[tuple[str, str]],
    region: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = resolve_chinese_font()
    font_names: list[str] = []
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        font_names.append(font_manager.FontProperties(fname=str(font_path)).get_name())
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [*font_names, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    well_labels = {
        "sand": "车页1导眼井轨迹",
        "taigu": "埕北古斜405井轨迹",
    }
    if region not in well_labels:
        raise ValueError(f"unsupported region: {region}")
    for filename, scales in LEGEND_SETS:
        handles = section_legend_handles(
            layers=layers,
            scales=scales,
            well_label=well_labels[region],
        )
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.axis("off")
        fig.legend(
            handles=handles,
            loc="center",
            frameon=False,
            fontsize=11.5,
            handlelength=2.8,
            labelspacing=0.8,
        )
        fig.savefig(
            output_dir / f"{filename}.png",
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.15,
        )
        plt.close(fig)


def main() -> int:
    args = parser().parse_args()
    if args.sand_output:
        write_legends(args.sand_output.resolve(), SAND_LAYERS, "sand")
    if args.taigu_output:
        write_legends(args.taigu_output.resolve(), TAIGU_LAYERS, "taigu")
    if not args.sand_output and not args.taigu_output:
        parser().error("至少指定 --sand-output 或 --taigu-output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
