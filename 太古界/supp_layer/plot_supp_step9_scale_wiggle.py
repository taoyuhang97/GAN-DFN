#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""太古界405：扩充DFN的矿区分尺度OBN波形+变面积剖面。

生成5套组合，每套XZ/YZ两图：
  large、medium、small、large_medium、all_scales。

DFN来自扩充后的三层合并CSV；使用其DipAzimuthDeg/DipDeg字段，不使用
历史V1–V4顶点。原始断层合并全矿区断层和上部补充层区块断层。
不叠加真实成像裂缝或Step4点。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
TAIGU = HERE.parent
REPO = TAIGU.parent
DEFAULT_CONFIG = HERE / "configs" / "taigu_supp_step9_scale_wiggle.json"
SPLIT_SCRIPT = HERE / "plot_supp_step9_wiggle_split.py"
ORIGINAL_STEP9 = TAIGU / "step9_sections" / "build_taigu_multibackground_sections.py"

if str(TAIGU) not in sys.path:
    sys.path.insert(0, str(TAIGU))
from common.orientation_frame import convention as orientation  # noqa: E402


@dataclass(frozen=True)
class ScaleSet:
    key: str
    label: str
    scales: tuple[str, ...]
    include_faults: bool
    number: int


SCALE_SETS = (
    ScaleSet("large", "大尺度", ("large",), True, 1),
    ScaleSet("medium", "中尺度", ("medium",), False, 3),
    ScaleSet("small", "小尺度", ("small",), False, 5),
    ScaleSet("large_medium", "大尺度+中尺度", ("large", "medium"), True, 7),
    ScaleSet("all_scales", "大中小尺度", ("large", "medium", "small"), True, 9),
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成太古界扩充DFN矿区分尺度波形剖面")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path)
    return p


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_original():
    spec = importlib.util.spec_from_file_location("taigu_step9_original_scale", ORIGINAL_STEP9)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载原Step9脚本: {ORIGINAL_STEP9}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split_helpers():
    spec = importlib.util.spec_from_file_location("taigu_split_helpers_scale", SPLIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载拆分绘图脚本: {SPLIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_wiggle(
    ax,
    values: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    config: dict[str, Any],
    original,
) -> dict[str, Any]:
    wiggle = dict(config["wiggle"])
    pick_config = {
        "wiggle_trace_spacing_m": float(wiggle.get("trace_spacing_m", 25.0)),
    }
    take, spacing = original.select_wiggle_traces(coords, pick_config, "overview")
    finite = values[np.isfinite(values)]
    limit = (
        float(
            np.percentile(
                np.abs(finite),
                float(wiggle.get("amplitude_percentile", 99.0)),
            )
        )
        if len(finite)
        else 1.0
    )
    limit = max(limit, 1.0e-9)
    scale = float(wiggle.get("lateral_scale_fraction", 0.42)) * spacing / limit
    drawn = 0
    for column_index in take:
        trace = values[:, column_index]
        if not np.isfinite(trace).any():
            continue
        clean = np.nan_to_num(trace)
        displaced = coords[column_index] + clean * scale
        ax.plot(
            displaced,
            times,
            color=str(wiggle.get("trace_color", "#111827")),
            linewidth=float(wiggle.get("line_width", 0.35)),
            zorder=0.5,
        )
        ax.fill_betweenx(
            times,
            coords[column_index],
            displaced,
            where=clean >= 0,
            color=str(wiggle.get("trace_color", "#111827")),
            alpha=float(wiggle.get("positive_fill_alpha", 0.55)),
            linewidth=0,
            zorder=0.4,
        )
        drawn += 1
    return {
        "trace_count_total": int(len(coords)),
        "trace_count_drawn": int(drawn),
        "trace_spacing_m": float(spacing),
        "amplitude_display_limit": float(limit),
    }


def normalize_linestyle(value):
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[1], list):
            return (value[0], tuple(value[1]))
        return tuple(value)
    return value


def add_horizons(
    ax,
    coords: np.ndarray,
    curves: dict[str, np.ndarray],
    config: dict[str, Any],
) -> list[Line2D]:
    handles = []
    for item in config["horizons"]:
        line, = ax.plot(
            coords,
            curves[item["field"]],
            color=item["color"],
            linestyle=normalize_linestyle(item["linestyle"]),
            linewidth=1.7,
            label=item["label"],
            zorder=1.4,
        )
        handles.append(line)
    return handles


def load_dfn(config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "LayerGroup",
        "FractureScale",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "DipAzimuthDeg",
        "DipDeg",
    ]
    frame = pd.read_csv(
        resolve_path(config["step8_patches_csv"]),
        usecols=columns,
        low_memory=False,
    )
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=[
            "CenterX",
            "CenterY",
            "CenterTime",
            "LengthM",
            "DipAzimuthDeg",
            "DipDeg",
        ]
    )
    return frame


def select_for_plane(
    frame: pd.DataFrame,
    track: pd.DataFrame,
    coords: np.ndarray,
    plane: str,
    half_width_m: float,
) -> pd.DataFrame:
    center_column = "CenterX" if plane == "XZ" else "CenterY"
    perpendicular = "Y" if plane == "XZ" else "X"
    track_time = track["TIME"].to_numpy(np.float64)
    track_perp = track[perpendicular].to_numpy(np.float64)
    well_perp = np.interp(
        frame["CenterTime"].to_numpy(np.float64), track_time, track_perp
    )
    mask = (
        frame[center_column].between(float(coords[0]), float(coords[-1]))
        & (
            np.abs(
                frame[f"Center{perpendicular}"].to_numpy(np.float64) - well_perp
            )
            <= float(half_width_m)
        )
    )
    return frame[mask].copy()


def build_scale_segments(
    frame: pd.DataFrame,
    plane: str,
    scales: tuple[str, ...],
) -> dict[str, np.ndarray]:
    center_column = "CenterX" if plane == "XZ" else "CenterY"
    output: dict[str, list[list[list[float]]]] = {
        scale: [] for scale in scales
    }
    for row in frame.itertuples(index=False):
        scale = str(row.FractureScale).strip().lower()
        if scale not in output:
            continue
        trace = orientation.trace_on_section(
            float(row.DipAzimuthDeg), float(row.DipDeg), plane
        )
        if trace is None:
            continue
        length = float(np.clip(float(row.LengthM), 10.0, 500.0))
        half = 0.5 * length
        dh, dt = trace
        center_h = float(getattr(row, center_column))
        center_t = float(row.CenterTime)
        output[scale].append(
            [
                [center_h - half * dh, center_t - half * dt],
                [center_h + half * dh, center_t + half * dt],
            ]
        )
    return {
        scale: np.asarray(lines, dtype=np.float64)
        for scale, lines in output.items()
    }


def load_fault_segments(
    config: dict[str, Any],
    track: pd.DataFrame,
    original,
) -> dict[str, list[list[list[float]]]]:
    result = {"XZ": [], "YZ": []}
    for path_value in config.get("fault_vtk_paths", []):
        path = resolve_path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        points, triangles = original.read_legacy_polydata(path)
        for projection in ("XZ", "YZ"):
            result[projection].extend(
                original.fault_section_segments(points, triangles, track, projection)
            )
        print(
            f"[scale-wiggle-taigu] fault source={path.name} "
            f"triangles={len(triangles)}",
            flush=True,
        )
    return result


def figure_legend(fig, handles: list, config: dict[str, Any]) -> None:
    figure_config = config["figure"]
    legend_width_in = float(figure_config.get("legend_width_in", 1.8))
    figure_width_in = float(figure_config["width_in"]) + legend_width_in
    right = max(0.55, 1.0 - legend_width_in / figure_width_in)
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper left",
        bbox_to_anchor=(right + 0.008, 0.88),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        fontsize=float(figure_config.get("legend_fontsize", 7.5)),
        handlelength=2.0,
        handletextpad=0.7,
        labelspacing=0.35,
        borderpad=0.45,
    )


def render_one(
    output_path: Path,
    projection: str,
    scale_set: ScaleSet,
    values: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    curves: dict[str, np.ndarray],
    dfn_by_scale: dict[str, np.ndarray],
    fault_segments: list[list[list[float]]],
    track: pd.DataFrame,
    config: dict[str, Any],
    original,
) -> tuple[dict[str, Any], list[Line2D]]:
    legend_width_in = float(config["figure"].get("legend_width_in", 1.8))
    fig, ax = plt.subplots(
        figsize=(
            float(config["figure"]["width_in"]) + legend_width_in,
            float(config["figure"]["height_in"]),
        )
    )
    background = plot_wiggle(ax, values, coords, times, config, original)
    horizon_handles = add_horizons(ax, coords, curves, config)

    scale_handles = []
    counts: dict[str, int] = {}
    for scale in ("small", "medium", "large"):
        if scale not in scale_set.scales:
            continue
        lines = dfn_by_scale.get(scale, np.empty((0, 2, 2)))
        style = config["scale_style"][scale]
        if len(lines):
            ax.add_collection(
                LineCollection(
                    lines,
                    colors=style["color"],
                    linewidths=float(style["line_width"]),
                    alpha=float(style["alpha"]),
                    zorder=2.2 if scale == "small" else 2.4,
                )
            )
        counts[scale] = int(len(lines))
        scale_handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("legend_color", style["color"]),
                linewidth=max(
                    float(style.get("legend_line_width", style["line_width"])),
                    0.8,
                ),
                label=style["label"],
            )
        )

    fault_count = 0
    fault_handle = None
    if scale_set.include_faults and fault_segments:
        style = config["fault_style"]
        ax.add_collection(
            LineCollection(
                fault_segments,
                colors=style["halo_color"],
                linewidths=float(style["halo_width"]),
                alpha=0.62,
                zorder=2.8,
            )
        )
        ax.add_collection(
            LineCollection(
                fault_segments,
                colors=style["color"],
                linewidths=float(style["line_width"]),
                alpha=float(style["alpha"]),
                zorder=2.9,
            )
        )
        fault_count = len(fault_segments)
        fault_handle = Line2D(
            [0],
            [0],
            color=style["color"],
            linewidth=float(style["line_width"]),
            label=style["label"],
        )

    _, _, well_h = original.section_queries(track, times, coords, projection)
    real_track = (
        (times >= float(track["TIME"].min()))
        & (times <= float(track["TIME"].max()))
    )
    well_style = config["well_style"]
    ax.plot(
        well_h[real_track],
        times[real_track],
        color=well_style["halo_color"],
        linewidth=float(well_style["halo_width"]),
        alpha=0.92,
        zorder=3.2,
    )
    ax.plot(
        well_h[real_track],
        times[real_track],
        color=well_style["color"],
        linewidth=float(well_style["line_width"]),
        zorder=3.3,
    )
    well_handle = Line2D(
        [0],
        [0],
        color=well_style["color"],
        linewidth=float(well_style["line_width"]),
        label=well_style["label"],
    )

    ax.set_xlim(float(coords[0]), float(coords[-1]))
    ax.set_ylim(float(times[-1]), float(times[0]))
    ax.set_xlabel("X / m" if projection == "XZ" else "Y / m")
    ax.set_ylabel("TWT / ms")
    ax.set_title(
        f"矿区尺度 | OBN地震波形+变面积 | {projection} | T4-T7 | "
        f"道数={background['trace_count_drawn']}",
        fontsize=float(config["figure"].get("title_fontsize", 11.0)),
        pad=8,
    )
    ax.grid(True, linewidth=0.25, alpha=0.22)
    figure_width_in = (
        float(config["figure"]["width_in"]) + legend_width_in
    )
    right = max(0.55, 1.0 - legend_width_in / figure_width_in)
    fig.subplots_adjust(left=0.06, right=right, bottom=0.10, top=0.88)
    fig.suptitle(
        f"{config['profile_well']}：{scale_set.label}DFN",
        fontsize=float(config["figure"].get("suptitle_fontsize", 13.0)),
        y=0.975,
    )
    handles = [well_handle, *scale_handles]
    if fault_handle is not None:
        handles.append(fault_handle)
    figure_legend(fig, handles, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=int(config["figure"]["dpi"]),
        facecolor="white",
    )
    plt.close(fig)
    return {
        **background,
        "dfn_line_counts_by_scale": counts,
        "fault_segment_count": int(fault_count),
        "imaging_patch_count": 0,
        "imaging_point_count": 0,
    }, handles


def main() -> int:
    args = parser().parse_args()
    config = read_json(args.config.resolve())
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    original = load_original()
    split = load_split_helpers()
    split.original_step9 = original

    track = original.load_track(config)
    cache_path = resolve_path(config["sample_cache"])
    with np.load(cache_path) as archive:
        times_cache = np.asarray(archive["times"], dtype=np.float64)
        axes = {
            "XZ": np.asarray(archive["xz_h"], dtype=np.float64),
            "YZ": np.asarray(archive["yz_h"], dtype=np.float64),
        }
    times_requested = np.arange(
        float(config["time_range_ms"][0]),
        float(config["time_range_ms"][1]) + 1.0e-9,
        float(config.get("section_sample_interval_ms", 2.0)),
    )
    start = int(np.searchsorted(times_cache, times_requested[0] - 1.0e-9))
    stop = int(np.searchsorted(times_cache, times_requested[-1] + 1.0e-9))
    if not np.allclose(times_cache[start:stop], times_requested, atol=1.0e-6):
        raise RuntimeError("矿区OBN缓存时间轴与配置时间窗不一致")
    times, values, sample_source = split.load_or_sample_wiggle(
        original, config, track, axes
    )
    curves, horizon_source = split.load_or_compute_horizons(
        original, config, track, axes, output_dir
    )

    dfn = load_dfn(config)
    dfn = dfn[
        dfn["CenterTime"].between(float(times[0]), float(times[-1]))
    ].copy()
    print(
        f"[scale-wiggle-taigu] merged DFN rows={len(dfn)} "
        f"layers={dfn.LayerGroup.value_counts().to_dict()}",
        flush=True,
    )
    selected_by_plane = {
        plane: select_for_plane(
            dfn,
            track,
            axes[plane],
            plane,
            float(config.get("dfn_projection_half_width_m", 50.0)),
        )
        for plane in ("XZ", "YZ")
    }
    faults = load_fault_segments(config, track, original)

    images: list[dict[str, Any]] = []
    for scale_set in SCALE_SETS:
        for offset, projection in enumerate(("XZ", "YZ")):
            number = scale_set.number + offset
            filename = (
                f"{number:02d}_{scale_set.key}_{projection.lower()}_t4_t7.png"
            )
            output_path = output_dir / filename
            dfn_by_scale = build_scale_segments(
                selected_by_plane[projection],
                projection,
                scale_set.scales,
            )
            qc, _ = render_one(
                output_path,
                projection,
                scale_set,
                values[projection],
                axes[projection],
                times,
                curves[projection],
                dfn_by_scale,
                faults[projection],
                track,
                config,
                original,
            )
            images.append(
                {
                    "number": number,
                    "set": scale_set.key,
                    "set_label": scale_set.label,
                    "scales": list(scale_set.scales),
                    "include_original_faults": scale_set.include_faults,
                    "projection": projection,
                    "path": str(output_path),
                    "overlay": qc,
                }
            )
            print(f"[scale-wiggle-taigu] {filename}", flush=True)

    checks = {
        "image_count_expected_10": len(images) == 10,
        "all_images_exist": all(Path(row["path"]).exists() for row in images),
        "merged_dfn_has_three_layers": set(dfn.LayerGroup.astype(str))
        == {"上部复合层", "太古界风化壳", "上部补充层"},
        "all_requested_scale_planes_have_segments": all(
            all(value > 0 for value in row["overlay"]["dfn_line_counts_by_scale"].values())
            for row in images
        ),
        "large_sets_include_faults": all(
            row["overlay"]["fault_segment_count"] > 0
            for row in images
            if row["set"] in ("large", "large_medium", "all_scales")
        ),
        "medium_small_only_have_no_faults": all(
            row["overlay"]["fault_segment_count"] == 0
            for row in images
            if row["set"] in ("medium", "small")
        ),
        "no_real_imaging_fractures": all(
            row["overlay"]["imaging_patch_count"] == 0
            and row["overlay"]["imaging_point_count"] == 0
            for row in images
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "region": "太古界",
        "well_name": config["profile_well"],
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "scope": "overview",
        "background": "OBN地震波形+正振幅变面积",
        "time_range_ms": [float(times[0]), float(times[-1])],
        "axis_trace_counts": {plane: int(len(axis)) for plane, axis in axes.items()},
        "dfn_source": {
            "csv": str(resolve_path(config["step8_patches_csv"])),
            "rows_in_time_window": int(len(dfn)),
            "layer_row_counts": {
                str(key): int(value)
                for key, value in dfn.LayerGroup.astype(str).value_counts().items()
            },
            "is_step8_well_corrected": True,
            "is_expanded_with_supp_layer": True,
            "geometry_policy": "CSV DipAzimuthDeg/DipDeg; no V1..V4",
        },
        "selected_patch_counts": {
            plane: int(len(frame)) for plane, frame in selected_by_plane.items()
        },
        "fault_segment_counts": {
            plane: int(len(lines)) for plane, lines in faults.items()
        },
        "sample_source": sample_source,
        "horizon_source": horizon_source,
        "images": images,
        "checks": checks,
    }
    summary_path = output_dir / "scale_wiggle_summary.json"
    summary_path.write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[scale-wiggle-taigu] status={summary['status']} summary={summary_path}",
        flush=True,
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
