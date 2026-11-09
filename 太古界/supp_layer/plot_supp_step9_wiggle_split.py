#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 OBN 地震波形+变面积底图上拆分显示成像真值与 DFN。

该脚本按 ``太古界/step9_sections/build_taigu_multibackground_sections.py``
的 OBN 采样和波形+变面积口径另行出图，不改原脚本、不覆盖正式 Step9 结果。

两张图：
  01_wiggle_area_imaging.png
      地震底图 + 405 成像段井轨迹 + 真实成像裂缝（真倾向/真倾角）
  02_wiggle_area_dfn.png
      地震底图 + 405 成像段井轨迹 + DFN 裂缝片（真倾向/真倾角），
      不叠加真实成像裂缝

DFN 不从 VTK 四边形顶点求交，而直接使用 CSV 中的
``DipAzimuthDeg``（真倾向方位）和 ``DipDeg``（真倾角）计算裂缝面与 XZ/YZ
剖面面的交线，避免当前历史 VTK 顶点几何错误污染展示角度。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAIGU = HERE.parent
REPO = TAIGU.parent
DEFAULT_CONFIG = HERE / "configs" / "taigu_supp_step9_wiggle_split.json"
ORIGINAL_STEP9 = TAIGU / "step9_sections" / "build_taigu_multibackground_sections.py"
PLANES = ("XZ", "YZ")

os.environ.setdefault("MPLCONFIGDIR", str(HERE / "output" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

if str(TAIGU) not in sys.path:
    sys.path.insert(0, str(TAIGU))
from common.orientation_frame import convention as orientation  # noqa: E402


def load_original_module():
    """只读加载原 Step9 脚本，复用其采样、轨迹回接和视倾角实现。"""
    spec = importlib.util.spec_from_file_location("taigu_step9_original", ORIGINAL_STEP9)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载原 Step9 脚本: {ORIGINAL_STEP9}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def load_local_axes(config: dict[str, Any], track: pd.DataFrame) -> dict[str, np.ndarray]:
    grid_path = resolve_path(config["attribute_demo_grid_csv"])
    grid = pd.read_csv(grid_path, usecols=["X", "Y"], low_memory=False)
    block = config["target_block"]
    grid = grid[
        grid["X"].between(float(block["x_min"]), float(block["x_max"]))
        & grid["Y"].between(float(block["y_min"]), float(block["y_max"]))
    ]
    x_all = np.sort(grid["X"].dropna().unique().astype(np.float64))
    y_all = np.sort(grid["Y"].dropna().unique().astype(np.float64))
    radius = float(config.get("local_axis_radius_m", 200.0))
    x = x_all[
        (x_all >= float(track["X"].min()) - radius)
        & (x_all <= float(track["X"].max()) + radius)
    ]
    y = y_all[
        (y_all >= float(track["Y"].min()) - radius)
        & (y_all <= float(track["Y"].max()) + radius)
    ]
    if len(x) < 2 or len(y) < 2:
        raise RuntimeError("局部剖面轴不足 2 个节点")
    return {"XZ": x, "YZ": y}


def load_or_sample_wiggle(
    original,
    config: dict[str, Any],
    track: pd.DataFrame,
    axes: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """读取原 Step9 的局部 OBN 缓存；缓存不存在时按同一合同重新采样。"""
    cache_path = resolve_path(config["sample_cache"])
    times_requested = np.arange(
        float(config["time_range_ms"][0]),
        float(config["time_range_ms"][1]) + 1.0e-9,
        float(config.get("section_sample_interval_ms", 2.0)),
    )
    if cache_path.exists():
        with np.load(cache_path) as archive:
            cached_times = np.asarray(archive["times"], dtype=np.float64)
            cached_axes = {
                "XZ": np.asarray(archive["xz_h"], dtype=np.float64),
                "YZ": np.asarray(archive["yz_h"], dtype=np.float64),
            }
            values = {
                "XZ": np.asarray(archive["xz_values"], dtype=np.float64),
                "YZ": np.asarray(archive["yz_values"], dtype=np.float64),
            }
        for plane in PLANES:
            if cached_axes[plane].shape != axes[plane].shape or not np.allclose(
                cached_axes[plane], axes[plane], atol=1.0e-6
            ):
                raise RuntimeError(f"OBN 采样缓存剖面轴与当前配置不一致: {plane}")
        start = int(np.searchsorted(cached_times, times_requested[0] - 1.0e-9))
        stop = int(np.searchsorted(cached_times, times_requested[-1] + 1.0e-9))
        cached_times = cached_times[start:stop]
        if len(cached_times) != len(times_requested) or not np.allclose(
            cached_times, times_requested, atol=1.0e-6
        ):
            raise RuntimeError("OBN 采样缓存时间轴与目标窗口不一致")
        values = {plane: values[plane][start:stop] for plane in PLANES}
        print(f"[wiggle-split] 复用 OBN 采样缓存: {cache_path}", flush=True)
        return cached_times, values, "reused_section_sample_cache"

    print(f"[wiggle-split] 未找到 OBN 缓存，重新采样: {cache_path}", flush=True)
    with original.ObnAmplitudeSampler(
        resolve_path(config["obn_segy"]),
        resolve_path(config["obn_trace_header_csv"]),
    ) as sampler:
        values: dict[str, np.ndarray] = {}
        for plane in PLANES:
            sampled, stats = original.sample_obn_section(
                sampler,
                track,
                times_requested,
                axes[plane],
                plane,
                float(config.get("obn_max_nearest_distance_m", 25.0)),
                int(config.get("progress_interval", 100)),
            )
            values[plane] = sampled
            print(
                f"[wiggle-split] {plane} OBN shape={sampled.shape} "
                f"finite={stats['finite_fraction']:.3f}",
                flush=True,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        times=times_requested,
        xz_h=axes["XZ"],
        yz_h=axes["YZ"],
        xz_values=values["XZ"],
        yz_values=values["YZ"],
    )
    return times_requested, values, "fresh_obn_sampling"


def load_or_compute_horizons(
    original,
    config: dict[str, Any],
    track: pd.DataFrame,
    axes: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    """只读取局部足迹附近的层位合同，并缓存三条层位曲线。"""
    cache_path = output_dir / "local_horizon_curves.npz"

    def cached_curves() -> dict[str, dict[str, np.ndarray]] | None:
        if not cache_path.exists():
            return None
        with np.load(cache_path) as archive:
            for plane in PLANES:
                if not np.allclose(
                    np.asarray(archive[f"{plane.lower()}_h"], dtype=np.float64),
                    axes[plane],
                    atol=1.0e-6,
                ):
                    return None
            return {
                plane: {
                    item["field"]: np.asarray(
                        archive[f"{plane.lower()}_{item['field']}"], dtype=np.float64
                    )
                    for item in config["horizons"]
                }
                for plane in PLANES
            }

    curves = cached_curves()
    if curves is not None:
        print(f"[wiggle-split] 复用层位曲线缓存: {cache_path}", flush=True)
        return curves, "reused_local_horizon_cache"

    margin = 100.0
    x_min = float(axes["XZ"].min()) - margin
    x_max = float(axes["XZ"].max()) + margin
    y_min = float(axes["YZ"].min()) - margin
    y_max = float(axes["YZ"].max()) + margin
    horizon_path = resolve_path(config["horizon_contract_csv"])
    columns = ["TraceIdx", "X", "Y", "TopTimeMs", "MidTimeMs", "BaseTimeMs"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        horizon_path,
        usecols=columns,
        chunksize=200_000,
        low_memory=False,
    ):
        for column in columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        local = chunk[
            chunk["X"].between(x_min, x_max)
            & chunk["Y"].between(y_min, y_max)
        ]
        if not local.empty:
            pieces.append(local)
    if not pieces:
        raise RuntimeError("局部足迹内没有层位合同点")
    horizon = pd.concat(pieces, ignore_index=True).dropna(
        subset=["X", "Y", "TopTimeMs", "MidTimeMs", "BaseTimeMs"]
    )
    print(f"[wiggle-split] 局部层位点={len(horizon)}", flush=True)
    curves = {
        plane: original.horizon_curves(horizon, track, axes[plane], plane)
        for plane in PLANES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for plane in PLANES:
        payload[f"{plane.lower()}_h"] = axes[plane]
        for item in config["horizons"]:
            payload[f"{plane.lower()}_{item['field']}"] = curves[plane][item["field"]]
    np.savez_compressed(cache_path, **payload)
    return curves, "computed_from_local_horizon_contract_subset"


def load_imaging(
    original,
    config: dict[str, Any],
    time_range: tuple[float, float],
) -> pd.DataFrame:
    source_config: dict[str, Any] = {
        **config,
        "output_dir": str(resolve_path(config["output_dir"])),
        "step2_segments_root": str(resolve_path(config["step2_segments_root"])),
        "step3_groups_root": str(resolve_path(config["step3_groups_root"])),
    }
    imaging = original.load_imaging(source_config)
    imaging["TIME"] = pd.to_numeric(imaging["TIME"], errors="coerce")
    imaging = imaging[imaging["TIME"].between(*time_range)].copy()
    if len(imaging) == 0:
        raise RuntimeError("目标窗口内没有成像裂缝点")
    return imaging.reset_index(drop=True)


def load_dfn(config: dict[str, Any], time_range: tuple[float, float]) -> pd.DataFrame:
    columns = [
        "LayerGroup",
        "FractureScale",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "HeightTimeMs",
        "DipAzimuthDeg",
        "DipDeg",
    ]
    dfn = pd.read_csv(
        resolve_path(config["step8_patches_csv"]),
        usecols=columns,
        low_memory=False,
    )
    for column in columns[2:]:
        dfn[column] = pd.to_numeric(dfn[column], errors="coerce")
    dfn = dfn[
        dfn["CenterTime"].between(*time_range)
        & dfn["CenterX"].notna()
        & dfn["CenterY"].notna()
        & dfn["DipAzimuthDeg"].notna()
        & dfn["DipDeg"].notna()
    ].copy()
    return dfn.reset_index(drop=True)


def select_patches_for_plane(
    dfn: pd.DataFrame,
    track: pd.DataFrame,
    axis: np.ndarray,
    plane: str,
    half_width_m: float,
) -> pd.DataFrame:
    center_column = "CenterX" if plane == "XZ" else "CenterY"
    perpendicular = "Y" if plane == "XZ" else "X"
    track_time = track["TIME"].to_numpy(np.float64)
    track_perpendicular = track[perpendicular].to_numpy(np.float64)
    well_perpendicular = np.interp(
        dfn["CenterTime"].to_numpy(np.float64),
        track_time,
        track_perpendicular,
    )
    mask = (
        dfn[center_column].between(float(axis.min()), float(axis.max()))
        & (
            np.abs(
                dfn[f"Center{perpendicular}"].to_numpy(np.float64)
                - well_perpendicular
            )
            <= float(half_width_m)
        )
    )
    return dfn[mask].reset_index(drop=True)


def panel_line_segments(
    frame: pd.DataFrame,
    plane: str,
    length_range_m: tuple[float, float],
) -> dict[str, np.ndarray]:
    center_column = "CenterX" if plane == "XZ" else "CenterY"
    grouped: dict[str, list[list[list[float]]]] = {
        "small": [],
        "medium": [],
        "large": [],
    }
    for row in frame.itertuples(index=False):
        trace = orientation.trace_on_section(
            float(row.DipAzimuthDeg),
            float(row.DipDeg),
            plane,
        )
        if trace is None:
            continue
        length = float(row.LengthM) if np.isfinite(float(row.LengthM)) else 30.0
        length = float(np.clip(length, *length_range_m))
        half = 0.5 * length
        center_h = float(getattr(row, center_column))
        center_t = float(row.CenterTime)
        dh, dt = trace
        scale = str(getattr(row, "FractureScale", "")).strip().lower()
        if scale not in grouped:
            scale = "small"
        grouped[scale].append(
            [
                [center_h - half * dh, center_t - half * dt],
                [center_h + half * dh, center_t + half * dt],
            ]
        )
    return {
        scale: np.asarray(segments, dtype=np.float64)
        for scale, segments in grouped.items()
    }


def imaging_line_segments(
    imaging: pd.DataFrame,
    plane: str,
    length_m: float,
) -> np.ndarray:
    center_column = "X" if plane == "XZ" else "Y"
    segments: list[list[list[float]]] = []
    for row in imaging.itertuples(index=False):
        trace = orientation.trace_on_section(
            float(row.FracAzimuth),
            float(row.FracDip),
            plane,
        )
        if trace is None:
            continue
        dh, dt = trace
        half = 0.5 * float(length_m)
        center_h = float(getattr(row, center_column))
        center_t = float(row.TIME)
        segments.append(
            [
                [center_h - half * dh, center_t - half * dt],
                [center_h + half * dh, center_t + half * dt],
            ]
        )
    return np.asarray(segments, dtype=np.float64)


def add_wiggle_background(
    ax,
    values: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    wiggle = dict(config["wiggle"])
    local_config = {
        "wiggle_local_max_trace_count": int(wiggle.get("local_max_trace_count", 0)),
        "wiggle_local_trace_spacing_m": float(wiggle.get("local_trace_spacing_m", 0.0)),
    }
    take, spacing_m = original_step9.select_wiggle_traces(
        coords, local_config, "local_200m"
    )
    finite = values[np.isfinite(values)]
    limit = float(
        np.percentile(
            np.abs(finite),
            float(wiggle.get("amplitude_percentile", 99.0)),
        )
    ) if len(finite) else 1.0
    limit = max(limit, 1.0e-9)
    scale = float(wiggle.get("lateral_scale_fraction", 0.42)) * spacing_m / limit
    drawn = 0
    for column_index in take:
        column = values[:, column_index]
        if not np.isfinite(column).any():
            continue
        trace = np.nan_to_num(column)
        x = coords[column_index] + trace * scale
        ax.plot(
            x,
            times,
            color=str(wiggle.get("trace_color", "#111827")),
            linewidth=float(wiggle.get("line_width", 0.32)),
            zorder=0.5,
        )
        ax.fill_betweenx(
            times,
            coords[column_index],
            x,
            where=trace >= 0,
            color=str(wiggle.get("trace_color", "#111827")),
            alpha=float(wiggle.get("positive_fill_alpha", 0.48)),
            linewidth=0,
            zorder=0.4,
        )
        drawn += 1
    return {
        "trace_count_total": int(len(coords)),
        "trace_count_drawn": int(drawn),
        "trace_spacing_m": float(spacing_m),
        "amplitude_display_limit": float(limit),
    }


def add_horizons(
    ax,
    coords: np.ndarray,
    curves: dict[str, np.ndarray],
    config: dict[str, Any],
) -> list[Line2D]:
    handles: list[Line2D] = []
    for item in config["horizons"]:
        values = np.asarray(curves[item["field"]], dtype=np.float64)
        line, = ax.plot(
            coords,
            values,
            color=item["color"],
            linestyle=matplotlib_linestyle(item["linestyle"]),
            linewidth=1.7,
            alpha=0.95,
            zorder=1.5,
        )
        handles.append(line)
    return handles


def configure_panel(ax, plane: str, coords: np.ndarray, time_range: tuple[float, float]) -> None:
    ax.set_xlim(float(coords[0]), float(coords[-1]))
    ax.set_ylim(float(time_range[1]), float(time_range[0]))
    ax.set_xlabel("X / m" if plane == "XZ" else "Y / m", fontsize=10.5)
    ax.set_ylabel("TIME / ms (TWT)", fontsize=10.5)
    ax.set_title(f"{plane} 剖面", fontsize=12)
    ax.tick_params(labelsize=9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#4b5563")


def bottom_legend(fig, handles: list, labels: list[str], config: dict[str, Any]) -> None:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(handles),
        frameon=False,
        fontsize=float(config["figure"].get("legend_fontsize", 9.0)),
        handlelength=2.3,
        columnspacing=1.8,
        handletextpad=0.6,
    )


def build_figure(
    path: Path,
    title: str,
    config: dict[str, Any],
    times: np.ndarray,
    axes: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    curves: dict[str, dict[str, np.ndarray]],
    draw_overlay,
    legend_handles: list,
    legend_labels: list[str],
) -> dict[str, Any]:
    figure_cfg = config["figure"]
    fig, plot_axes = plt.subplots(
        1,
        2,
        figsize=(float(figure_cfg["width_in"]), float(figure_cfg["height_in"])),
        sharey=True,
    )
    qc: dict[str, Any] = {}
    for ax, plane in zip(plot_axes, PLANES):
        qc[plane] = add_wiggle_background(
            ax, values[plane], axes[plane], times, config
        )
        add_horizons(ax, axes[plane], curves[plane], config)
        qc[plane].update(draw_overlay(ax, plane))
        configure_panel(ax, plane, axes[plane], (float(times[0]), float(times[-1])))
    fig.suptitle(title, fontsize=15, y=0.975)
    fig.text(
        0.5,
        0.078,
        "背景：OBN 地震波形 + 正振幅变面积；纵向比例 1 ms = 2 m；裂缝线方向均为裂缝面与剖面的交线",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.052, right=0.992, bottom=0.145, top=0.91, wspace=0.09)
    bottom_legend(fig, legend_handles, legend_labels, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=int(figure_cfg["dpi"]),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return qc


def draw_imaging_track(
    ax,
    plane: str,
    track: pd.DataFrame,
    imaging: pd.DataFrame,
    style: dict[str, Any],
    zorder: float,
) -> dict[str, Any]:
    """使用黄色主线+深色光晕绘制405成像测井段轨迹。"""
    center_column = "X" if plane == "XZ" else "Y"
    if "InImagingInterval" in track.columns:
        rows = track[
            pd.to_numeric(track["InImagingInterval"], errors="coerce")
            .fillna(0)
            .eq(1)
        ].copy()
    else:
        rows = track[
            track["TIME"].between(float(imaging["TIME"].min()), float(imaging["TIME"].max()))
        ].copy()
    rows = rows.sort_values("TIME").drop_duplicates("TIME")
    if len(rows) < 2:
        raise RuntimeError(f"{plane} 成像测井段轨迹不足 2 个采样点")
    ax.plot(
        rows[center_column],
        rows["TIME"],
        color=style["glow_color"],
        linewidth=style["glow_width"],
        alpha=style["glow_alpha"],
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.plot(
        rows[center_column],
        rows["TIME"],
        color=style["color"],
        linewidth=style["line_width"],
        alpha=style["alpha"],
        solid_capstyle="round",
        zorder=zorder + 0.1,
    )
    return {
        "imaging_track_samples": int(len(rows)),
        "imaging_track_time_range_ms": [
            float(rows["TIME"].min()),
            float(rows["TIME"].max()),
        ],
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "mean": float(np.mean(finite)),
    }


def matplotlib_linestyle(value):
    """JSON 只能存数组，matplotlib 的自定义虚线样式需要嵌套 tuple。"""
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[1], list):
            return (value[0], tuple(value[1]))
        return tuple(value)
    return value


def orientation_stats(
    imaging: pd.DataFrame,
    selected: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "imaging": {
            "FracDip": distribution(imaging["FracDip"].to_numpy(np.float64)),
            "FracAzimuth_dip_azimuth": distribution(
                imaging["FracAzimuth"].to_numpy(np.float64)
            ),
        },
        "dfn": {},
    }
    for plane in PLANES:
        frame = selected[plane]
        image_apparent = np.asarray(
            [
                value
                for value in (
                    orientation.section_apparent_dip_deg(
                        float(row.FracAzimuth), float(row.FracDip), plane
                    )
                    for row in imaging.itertuples(index=False)
                )
                if value is not None
            ],
            dtype=np.float64,
        )
        dfn_apparent = np.asarray(
            [
                value
                for value in (
                    orientation.section_apparent_dip_deg(
                        float(row.DipAzimuthDeg), float(row.DipDeg), plane
                    )
                    for row in frame.itertuples(index=False)
                )
                if value is not None
            ],
            dtype=np.float64,
        )
        result["dfn"][plane] = {
            "DipDeg": distribution(frame["DipDeg"].to_numpy(np.float64)),
            "DipAzimuthDeg": distribution(
                frame["DipAzimuthDeg"].to_numpy(np.float64)
            ),
            "apparent_dip_deg": distribution(dfn_apparent),
            "imaging_apparent_dip_deg": distribution(image_apparent),
            "scale_counts": {
                str(key): int(value)
                for key, value in frame["FractureScale"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            },
        }
    result["notes"] = [
        "成像 FracAzimuth 与 DFN DipAzimuthDeg 均按真倾向方位解释，不再把倾向当作走向。",
        "DFN 线直接由 CSV 的 DipAzimuthDeg/DipDeg 计算，不使用 Step8 VTK 的 V1..V4 顶点。",
        "历史 DFN 顶点短边实测满足 tan(几何倾角)=2*tan(DipDeg)，按几何出图会把倾角显示得过陡。",
        "DFN 是依据多井产状经验库生成的随机实现，不是 405 逐点复制，因此单片不会一一对应。",
        "倾向方位存在多峰分布；中位数接近不代表逐片角度小，必须结合分位数和视倾角比较。",
    ]
    return result


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "profile_well",
        "output_dir",
        "target_block",
        "attribute_demo_grid_csv",
        "horizon_contract_csv",
        "obn_segy",
        "obn_trace_header_csv",
        "step2_segments_root",
        "step3_groups_root",
        "step8_patches_csv",
        "time_range_ms",
        "horizons",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"配置缺少字段: {missing}")
    if len(config["time_range_ms"]) != 2 or not (
        float(config["time_range_ms"][0]) < float(config["time_range_ms"][1])
    ):
        raise ValueError("time_range_ms 必须是递增二元组")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成地震波形+变面积底图的成像/DFN拆分剖面"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    validate_config(config)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    time_range = (
        float(config["time_range_ms"][0]),
        float(config["time_range_ms"][1]),
    )

    global original_step9
    original_step9 = load_original_module()
    imaging_track_style = {
        "color": "#ffea00",
        "glow_color": original_step9.IMAGING_TRACK_GLOW,
        "line_width": 2.8,
        "glow_width": 5.2,
        "alpha": 0.96,
        "glow_alpha": 0.78,
        "label": "Step3成像井",
    }
    track = original_step9.load_track(config)
    axes = load_local_axes(config, track)
    times, values, sample_source = load_or_sample_wiggle(
        original_step9, config, track, axes
    )
    curves, horizon_source = load_or_compute_horizons(
        original_step9, config, track, axes, output_dir
    )
    imaging = load_imaging(original_step9, config, time_range)
    dfn = load_dfn(config, time_range)
    selected = {
        plane: select_patches_for_plane(
            dfn,
            track,
            axes[plane],
            plane,
            float(config.get("dfn_projection_half_width_m", 200.0)),
        )
        for plane in PLANES
    }
    print(
        f"[wiggle-split] 成像点={len(imaging)} "
        f"DFN窗口={len(dfn)} "
        f"XZ={len(selected['XZ'])} YZ={len(selected['YZ'])}",
        flush=True,
    )

    imaging_cfg = dict(config["imaging"])
    dfn_cfg = dict(config["dfn"])
    dfn_length_range = tuple(dfn_cfg["display_length_range_m"])
    common_horizon_handles = [
        Line2D(
            [0],
            [0],
            color=item["color"],
            linestyle=matplotlib_linestyle(item["linestyle"]),
            linewidth=1.7,
            label=item["label"],
        )
        for item in config["horizons"]
    ]

    def draw_imaging(ax, plane: str) -> dict[str, Any]:
        segments = imaging_line_segments(
            imaging, plane, float(imaging_cfg["line_length_m"])
        )
        if len(segments):
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=imaging_cfg["halo_color"],
                    linewidths=float(imaging_cfg["halo_width"]),
                    alpha=0.28,
                    zorder=3.0,
                )
            )
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=imaging_cfg["color"],
                    linewidths=float(imaging_cfg["line_width"]),
                    alpha=0.95,
                    zorder=3.1,
                )
            )
        track_qc = draw_imaging_track(
            ax,
            plane,
            track,
            imaging,
            imaging_track_style,
            zorder=4.0,
        )
        center_column = "X" if plane == "XZ" else "Y"
        ax.scatter(
            imaging[center_column],
            imaging["TIME"],
            s=float(imaging_cfg["point_size"]),
            facecolor="#fff7ed",
            edgecolor=imaging_cfg["color"],
            linewidth=0.45,
            zorder=4.2,
        )
        return {
            "imaging_fractures_drawn": int(len(segments)),
            "imaging_time_range_ms": [
                float(imaging["TIME"].min()),
                float(imaging["TIME"].max()),
            ],
            **track_qc,
        }

    imaging_handles = [
        Line2D(
            [0],
            [0],
            color=imaging_track_style["color"],
            linewidth=imaging_track_style["line_width"],
            label=imaging_track_style["label"],
        ),
        Line2D(
            [0],
            [0],
            color=imaging_cfg["color"],
            linewidth=float(imaging_cfg["line_width"]),
            marker="o",
            markersize=4.2,
            markerfacecolor="#fff7ed",
            markeredgecolor=imaging_cfg["color"],
            label="真实裂缝（倾向/倾角）",
        ),
        *common_horizon_handles,
    ]
    imaging_path = output_dir / "01_wiggle_area_imaging.png"
    imaging_qc = build_figure(
        imaging_path,
        "埕北古斜405 成像测井段 ±200 m：地震背景 + 成像真值",
        config,
        times,
        axes,
        values,
        curves,
        draw_imaging,
        imaging_handles,
        [handle.get_label() for handle in imaging_handles],
    )
    print(f"[wiggle-split] 已生成 {imaging_path}", flush=True)

    def draw_dfn(ax, plane: str) -> dict[str, Any]:
        grouped = panel_line_segments(
            selected[plane],
            plane,
            dfn_length_range,
        )
        drawn: dict[str, int] = {}
        for scale in ("small", "medium", "large"):
            segments = grouped.get(scale, np.empty((0, 2, 2)))
            style = dfn_cfg[scale]
            if len(segments):
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=style["color"],
                        linewidths=float(style["line_width"]),
                        alpha=float(style["alpha"]),
                        zorder=2.0 if scale == "small" else 2.2,
                    )
                )
            drawn[scale] = int(len(segments))
        track_qc = draw_imaging_track(
            ax,
            plane,
            track,
            imaging,
            imaging_track_style,
            zorder=5.0,
        )
        return {
            "dfn_line_counts_by_scale": drawn,
            **track_qc,
        }

    dfn_handles = [
        Line2D(
            [0],
            [0],
            color=dfn_cfg[scale]["color"],
            linewidth=max(float(dfn_cfg[scale]["line_width"]), 0.8),
            label=str(dfn_cfg[scale]["label"]),
        )
        for scale in ("small", "medium", "large")
    ]
    dfn_handles.append(
        Line2D(
            [0],
            [0],
            color=imaging_track_style["color"],
            linewidth=imaging_track_style["line_width"],
            label=imaging_track_style["label"],
        )
    )
    dfn_handles.extend(common_horizon_handles)
    dfn_path = output_dir / "02_wiggle_area_dfn.png"
    dfn_qc = build_figure(
        dfn_path,
        "埕北古斜405 成像测井段 ±200 m：地震背景 + DFN 预测",
        config,
        times,
        axes,
        values,
        curves,
        draw_dfn,
        dfn_handles,
        [handle.get_label() for handle in dfn_handles],
    )
    print(f"[wiggle-split] 已生成 {dfn_path}", flush=True)

    stats = orientation_stats(imaging, selected)
    summary = {
        "status": "pass",
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "profile_well": config["profile_well"],
        "time_range_ms": list(time_range),
        "half_width_m": float(config.get("dfn_projection_half_width_m", 200.0)),
        "inputs": {
            "obn_sampling": sample_source,
            "horizons": horizon_source,
            "dfn_points_in_time_window": int(len(dfn)),
            "imaging_points": int(len(imaging)),
            "dfn_selected": {
                plane: int(len(selected[plane])) for plane in PLANES
            },
        },
        "orientation_contract": {
            "imaging": "FracAzimuth = 真倾向方位；FracDip = 真倾角",
            "dfn": "DipAzimuthDeg = 真倾向方位；DipDeg = 真倾角",
            "section_rule": "裂缝面与XZ/YZ剖面面的交线",
            "time_scale_m_per_ms": float(config.get("time_scale_m_per_ms", 2.0)),
            "dfn_geometry_policy": "CSV产状字段；不使用V1..V4顶点",
        },
        "figure_qc": {
            "imaging": imaging_qc,
            "dfn": dfn_qc,
        },
        "orientation_comparison": stats,
        "outputs": {
            "imaging": str(imaging_path),
            "dfn": str(dfn_path),
        },
    }
    write_json(output_dir / "wiggle_split_summary.json", summary)
    print(
        "[wiggle-split] "
        f"imagery={imaging['FracDip'].median():.1f}°/"
        f"{imaging['FracAzimuth'].median():.1f}° "
        f"dfn_XZ={selected['XZ']['DipDeg'].median():.1f}°/"
        f"{selected['XZ']['DipAzimuthDeg'].median():.1f}°",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
