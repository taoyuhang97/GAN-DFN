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
TAIGU_LAYER_ORDER = ("上部复合层", "太古界风化壳")
TAIGU_DISPLAY_LAYER_MAP = {
    "上部复合层": "上部复合层",
    "上部补充层": "上部复合层",
    "太古界风化壳": "太古界风化壳",
}
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "output" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

if str(TAIGU) not in sys.path:
    sys.path.insert(0, str(TAIGU))
from common.orientation_frame import convention as orientation  # noqa: E402

_imaging_compare_spec = importlib.util.spec_from_file_location(
    "shared_step9_imaging_200m_compare",
    REPO / "common" / "step9_imaging_200m_compare.py",
)
if _imaging_compare_spec is None or _imaging_compare_spec.loader is None:
    raise ImportError("cannot load shared 200 m comparison contract")
shared_compare = importlib.util.module_from_spec(_imaging_compare_spec)
_imaging_compare_spec.loader.exec_module(shared_compare)
TAIGU_SCALE_WIDTHS = dict(shared_compare.DFN_SCALE_WIDTHS)


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


def load_dfn(config: dict[str, Any]) -> pd.DataFrame:
    required_columns = [
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
    optional_numeric_columns = [
        "IsDensityCompensationPatch",
        "IsImagingMatchPatch",
        "IsWellControlPatch",
        "NearestTrajectoryDistance",
        "NearestMajorDistanceM",
        "MajorInfluenceRadiusM",
    ]
    optional_text_columns = [
        "SourceType",
        "PatchID",
        "GenerationStage",
        "NearestMajorPatchID",
        "NearestMajorScale",
        "WellControlSampleID",
    ]
    available = set(pd.read_csv(
        resolve_path(config["step8_patches_csv"]),
        nrows=0,
    ).columns)
    missing = sorted(set(required_columns) - available)
    if missing:
        raise ValueError(f"Step8 DFN CSV missing required columns: {missing}")
    columns = [
        *required_columns,
        *(column for column in optional_numeric_columns if column in available),
        *(column for column in optional_text_columns if column in available),
    ]
    dfn = pd.read_csv(
        resolve_path(config["step8_patches_csv"]),
        usecols=columns,
        low_memory=False,
    )
    for column in [*required_columns[2:], *optional_numeric_columns]:
        if column in dfn.columns:
            dfn[column] = pd.to_numeric(dfn[column], errors="coerce")
    for column in optional_text_columns:
        if column not in dfn.columns:
            dfn[column] = ""
    for column in optional_numeric_columns:
        if column not in dfn.columns:
            dfn[column] = 0.0
    dfn = dfn[
        dfn["CenterX"].notna()
        & dfn["CenterY"].notna()
        & dfn["DipAzimuthDeg"].notna()
        & dfn["DipDeg"].notna()
    ].copy()
    dfn["LayerGroup"] = dfn["LayerGroup"].replace(TAIGU_DISPLAY_LAYER_MAP)
    dfn["SourceIndex"] = np.arange(len(dfn), dtype=int)
    return dfn.reset_index(drop=True)


def select_patches_for_plane(
    dfn: pd.DataFrame,
    track: pd.DataFrame,
    axis: np.ndarray,
    plane: str,
    half_width_m: float,
    time_range: tuple[float, float],
) -> pd.DataFrame:
    """Select surface-near patches whose true trace enters the local window."""

    center_column = "CenterX" if plane == "XZ" else "CenterY"
    perpendicular = "Y" if plane == "XZ" else "X"
    track_time = track["TIME"].to_numpy(np.float64)
    track_perpendicular = track[perpendicular].to_numpy(np.float64)
    well_perpendicular = np.interp(
        dfn["CenterTime"].to_numpy(np.float64),
        track_time,
        track_perpendicular,
    )
    max_length = float(shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M[1])
    time_padding_ms = max_length / float(shared_compare.TIME_SCALE_M_PER_MS)
    candidates = dfn[
        (
            np.abs(
                dfn[f"Center{perpendicular}"].to_numpy(np.float64)
                - well_perpendicular
            )
            <= float(half_width_m)
        )
        & dfn["CenterTime"].between(
            float(time_range[0]) - time_padding_ms,
            float(time_range[1]) + time_padding_ms,
        )
        & dfn[center_column].between(
            float(axis.min()) - max_length,
            float(axis.max()) + max_length,
        )
    ].copy()

    keep = np.zeros(len(candidates), dtype=bool)
    for position, row in enumerate(candidates.itertuples(index=False)):
        trace = orientation.trace_on_section(
            float(row.DipAzimuthDeg),
            float(row.DipDeg),
            plane,
        )
        if trace is None:
            continue
        length = float(
            np.clip(
                float(row.LengthM),
                *shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M,
            )
        )
        half = 0.5 * length
        center_h = float(getattr(row, center_column))
        center_t = float(row.CenterTime)
        dh, dt = trace
        h_values = (center_h - half * dh, center_h + half * dh)
        t_values = (center_t - half * dt, center_t + half * dt)
        keep[position] = (
            max(h_values) >= float(axis.min())
            and min(h_values) <= float(axis.max())
            and max(t_values) >= float(time_range[0])
            and min(t_values) <= float(time_range[1])
        )
    return candidates[keep].reset_index(drop=True)


def _trace_geometry(
    row,
    plane: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    trace = orientation.trace_on_section(
        float(row.DipAzimuthDeg),
        float(row.DipDeg),
        plane,
    )
    if trace is None:
        return None
    length = float(
        np.clip(
            float(row.LengthM),
            *shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M,
        )
    )
    half = 0.5 * length
    center_column = "CenterX" if plane == "XZ" else "CenterY"
    center_h = float(getattr(row, center_column))
    center_t = float(row.CenterTime)
    dh, dt = trace
    line = np.asarray(
        [
            [center_h - half * dh, center_t - half * dt],
            [center_h + half * dh, center_t + half * dt],
        ],
        dtype=np.float64,
    )
    return line, np.asarray(trace, dtype=np.float64)


def _point_to_segment_distance_m(
    point: tuple[float, float],
    line: np.ndarray,
) -> float:
    x1, y1 = float(line[0, 0]), 2.0 * float(line[0, 1])
    x2, y2 = float(line[1, 0]), 2.0 * float(line[1, 1])
    dx = x2 - x1
    dy = y2 - y1
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-12:
        return float(np.hypot(point[0] - x1, point[1] - y1))
    fraction = float(
        np.clip(
            ((point[0] - x1) * dx + (point[1] - y1) * dy) / denominator,
            0.0,
            1.0,
        )
    )
    return float(
        np.hypot(
            point[0] - (x1 + fraction * dx),
            point[1] - (y1 + fraction * dy),
        )
    )


def _line_to_line_distance_m(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        min(
            _point_to_segment_distance_m(
                (float(first[0, 0]), 2.0 * float(first[0, 1])),
                second,
            ),
            _point_to_segment_distance_m(
                (float(first[1, 0]), 2.0 * float(first[1, 1])),
                second,
            ),
            _point_to_segment_distance_m(
                (float(second[0, 0]), 2.0 * float(second[0, 1])),
                first,
            ),
            _point_to_segment_distance_m(
                (float(second[1, 0]), 2.0 * float(second[1, 1])),
                first,
            ),
        )
    )


def _line_crosses_well_xz(
    line: np.ndarray,
    track: pd.DataFrame,
    time_range: tuple[float, float],
) -> tuple[bool, float | None]:
    times = track["TIME"].to_numpy(dtype=float)
    x_values = track["X"].to_numpy(dtype=float)
    low = max(float(line[:, 1].min()), float(time_range[0]))
    high = min(float(line[:, 1].max()), float(time_range[1]))
    if high <= low:
        return False, None
    sampled_t = np.linspace(low, high, 2001)
    sampled_h = np.interp(sampled_t, line[:, 1], line[:, 0])
    sampled_well = np.interp(sampled_t, times, x_values)
    delta = sampled_h - sampled_well
    crossings = np.flatnonzero(
        np.signbit(delta[:-1]) != np.signbit(delta[1:])
    )
    if len(crossings):
        return True, float(sampled_t[int(crossings[0])])
    nearest = int(np.argmin(np.abs(delta)))
    return False, float(sampled_t[nearest])


def _line_well_min_distance_xz(
    line: np.ndarray,
    track: pd.DataFrame,
    time_range: tuple[float, float],
) -> float:
    times = track["TIME"].to_numpy(dtype=float)
    x_values = track["X"].to_numpy(dtype=float)
    low = max(float(line[:, 1].min()), float(time_range[0]))
    high = min(float(line[:, 1].max()), float(time_range[1]))
    if high <= low:
        return float("inf")
    sampled_t = np.linspace(low, high, 2001)
    sampled_h = np.interp(sampled_t, line[:, 1], line[:, 0])
    sampled_well = np.interp(sampled_t, times, x_values)
    return float(np.min(np.abs(sampled_h - sampled_well)))


def condition_small_dfn_near_xz_large_fractures(
    dfn: pd.DataFrame,
    selected_xz: pd.DataFrame,
    track: pd.DataFrame,
    imaging_events: pd.DataFrame,
    time_range: tuple[float, float],
    protected_imaging_ids: set[str] | None = None,
    protected_imaging_events: pd.DataFrame | None = None,
    *,
    move_fraction: float = 0.45,
    anchor_radius_m: float = 250.0,
    max_cluster_offset_m: float = 300.0,
    halo_distance_power: float = 0.75,
    longitudinal_jitter_fraction: float = 0.01,
    anchor_lateral_ranges_m: list[float] | None = None,
    sparse_band_half_width_m: float = 20.0,
    sparse_band_density_ratio: float = 0.9,
    sparse_band_quota_multiplier: float = 1.0,
    horizontal_balance_band_ms: list[float] | None = None,
    horizontal_balance_flank_ranges_ms: list[list[float]] | None = None,
    horizontal_balance_ratio: float = 1.0,
    well_clearance_m: float = 10.0,
    orientation_sigma_deg: float = 22.0,
    neighbor_max_orientation_weight: float = 0.30,
    random_seed: int = 405,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Redistribute eligible expansion patches around well-crossing large XZ traces."""

    work = dfn.copy()
    work["LargeAnchorConditioned"] = 0
    work["LargeAnchorID"] = -1
    work["LargeAnchorDistanceM"] = np.nan
    work["LargeAnchorOrientationWeight"] = 0.0
    work["OriginalLargeAnchorCenterX"] = work["CenterX"]
    work["OriginalLargeAnchorCenterY"] = work["CenterY"]
    work["OriginalLargeAnchorCenterTime"] = work["CenterTime"]
    work["OriginalLargeAnchorDipDeg"] = work["DipDeg"]
    work["OriginalLargeAnchorDipAzimuthDeg"] = work["DipAzimuthDeg"]
    work["ImagingMatchEventLocked"] = 0

    aligned_imaging_patch_count = 0
    if protected_imaging_events is not None and len(protected_imaging_events):
        event_frame = protected_imaging_events.copy()
        for column in ("TIME", "Frac_Azimuth", "Frac_Dip"):
            event_frame[column] = pd.to_numeric(
                event_frame[column],
                errors="coerce",
            )
        event_frame = event_frame.dropna(
            subset=["WellControlSampleID", "TIME", "Frac_Azimuth", "Frac_Dip"]
        )
        track_times = track["TIME"].to_numpy(dtype=float)
        for event in event_frame.itertuples(index=False):
            sample_id = str(event.WellControlSampleID).strip()
            matches = work[
                work["WellControlSampleID"].astype(str).str.strip().eq(sample_id)
                & work["FractureScale"].eq("small")
            ]
            if matches.empty:
                continue
            source_index = int(matches.index[0])
            event_time = float(event.TIME)
            work.at[source_index, "CenterX"] = float(
                np.interp(event_time, track_times, track["X"].to_numpy(dtype=float))
            )
            work.at[source_index, "CenterY"] = float(
                np.interp(event_time, track_times, track["Y"].to_numpy(dtype=float))
            )
            work.at[source_index, "CenterTime"] = event_time
            work.at[source_index, "DipAzimuthDeg"] = float(event.Frac_Azimuth)
            work.at[source_index, "DipDeg"] = float(event.Frac_Dip)
            work.at[source_index, "ImagingMatchEventLocked"] = 1
            aligned_imaging_patch_count += 1

    anchor_records: list[dict[str, Any]] = []
    for _, row in selected_xz[selected_xz["FractureScale"].eq("large")].iterrows():
        geometry = _trace_geometry(row, "XZ")
        if geometry is None:
            continue
        line, _ = geometry
        crosses, crossing_time = _line_crosses_well_xz(line, track, time_range)
        if not crosses:
            continue
        anchor_records.append(
            {
                "SourceIndex": int(row["SourceIndex"]),
                "line": line,
                "layer": str(row["LayerGroup"]),
                "dip_azimuth_deg": float(row["DipAzimuthDeg"]),
                "dip_deg": float(row["DipDeg"]),
                "crossing_time_ms": float(crossing_time),
            }
        )
    if not anchor_records:
        return work, {
            "enabled": False,
            "reason": "no large XZ traces cross the imaging well trajectory",
        }
    anchor_records.sort(key=lambda record: record["crossing_time_ms"])
    for index, record in enumerate(anchor_records):
        line = record["line"]
        low_t = max(min(float(line[0, 1]), float(line[1, 1])), float(time_range[0]))
        high_t = min(max(float(line[0, 1]), float(line[1, 1])), float(time_range[1]))
        if high_t <= low_t:
            raise ValueError(f"large anchor {index} has no visible local trace")
        low_h = float(np.interp(low_t, line[:, 1], line[:, 0]))
        high_h = float(np.interp(high_t, line[:, 1], line[:, 0]))
        record["visible_time_range_ms"] = [low_t, high_t]
        record["visible_length_m"] = float(
            np.hypot(high_h - low_h, 2.0 * (high_t - low_t))
        )
        record["lateral_range_m"] = float(
            anchor_lateral_ranges_m[index]
            if anchor_lateral_ranges_m
            and index < len(anchor_lateral_ranges_m)
            else max_cluster_offset_m
        )
    anchor_display_lines = [
        np.asarray(
            [
                [record["line"][0, 0], 2.0 * record["line"][0, 1]],
                [record["line"][1, 0], 2.0 * record["line"][1, 1]],
            ],
            dtype=float,
        )
        for record in anchor_records
    ]
    directions = []
    for line in anchor_display_lines:
        direction = line[1] - line[0]
        directions.append(direction / np.linalg.norm(direction))
    mean_direction = np.mean(directions, axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    sparse_normal = np.asarray(
        [-mean_direction[1], mean_direction[0]],
        dtype=float,
    )
    center_points = np.asarray(
        [line.mean(axis=0) for line in anchor_display_lines],
        dtype=float,
    )
    sparse_center = center_points.mean(axis=0)
    sparse_center_offset = float(np.dot(sparse_center, sparse_normal))
    sparse_band_angle_deg = float(
        np.degrees(np.arctan2(mean_direction[1], mean_direction[0])) % 180.0
    )

    def sparse_band_distance_m(h_value: float, time_value: float) -> float:
        point = np.asarray([h_value, 2.0 * time_value], dtype=float)
        return float(abs(np.dot(point, sparse_normal) - sparse_center_offset))

    source_type_control = (
        work["SourceType"]
        .astype(str)
        .str.strip()
        .eq("small_step3_imaging_well_control")
    )
    if protected_imaging_ids:
        imaging_source = (
            work["WellControlSampleID"]
            .astype(str)
            .str.strip()
            .isin(protected_imaging_ids)
        )
    else:
        imaging_source = source_type_control
    expansion_source = ~imaging_source
    protected_source_indices = [
        int(value)
        for value in selected_xz.loc[
            selected_xz["SourceIndex"].isin(
                work.loc[imaging_source & work["FractureScale"].eq("small"), "SourceIndex"]
            ),
            "SourceIndex",
        ]
    ]
    eligible = selected_xz[
        selected_xz["FractureScale"].eq("small")
        & expansion_source.reindex(selected_xz["SourceIndex"]).to_numpy()
    ].copy()
    anchor_layers = {record["layer"] for record in anchor_records}
    eligible = eligible[eligible["LayerGroup"].isin(anchor_layers)].copy()

    candidate_records: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        geometry = _trace_geometry(row, "XZ")
        if geometry is None:
            continue
        line, _ = geometry
        distances = [
            _line_to_line_distance_m(line, record["line"])
            for record in anchor_records
        ]
        nearest_index = int(np.argmin(distances))
        crosses_well, _ = _line_crosses_well_xz(line, track, time_range)
        sparse_distance = sparse_band_distance_m(
            float(row.CenterX),
            float(row.CenterTime),
        )
        candidate_records.append(
            {
                "SourceIndex": int(row["SourceIndex"]),
                "distance_m": float(distances[nearest_index]),
                "anchor_index": nearest_index,
                "crosses_well": bool(crosses_well),
                "in_sparse_band": bool(
                    sparse_distance < float(sparse_band_half_width_m)
                ),
            }
        )
    candidates = pd.DataFrame(candidate_records)
    if candidates.empty:
        return work, {
            "enabled": False,
            "reason": "no eligible small expansion patches",
        }

    candidates = candidates.sort_values(
        ["crosses_well", "in_sparse_band", "distance_m", "SourceIndex"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    crossing_expansion_count = int(candidates["crosses_well"].sum())
    move_count = min(
        max(
            int(round(float(move_fraction) * len(candidates))),
            crossing_expansion_count,
        ),
        len(candidates),
    )
    moving = candidates.head(move_count).copy()
    if horizontal_balance_band_ms and len(horizontal_balance_band_ms) == 2:
        central_low, central_high = (
            float(value) for value in horizontal_balance_band_ms
        )
        baseline_small_times = selected_xz[
            selected_xz["FractureScale"].eq("small")
        ]["CenterTime"].to_numpy(dtype=float)
        baseline_central_count = int(
            np.count_nonzero(
                (baseline_small_times >= central_low)
                & (baseline_small_times < central_high)
            )
        )
        flank_densities: list[float] = []
        for flank in horizontal_balance_flank_ranges_ms or []:
            flank_low, flank_high = float(flank[0]), float(flank[1])
            flank_width = max(flank_high - flank_low, 1.0e-9)
            flank_count = int(
                np.count_nonzero(
                    (baseline_small_times >= flank_low)
                    & (baseline_small_times < flank_high)
                )
            )
            flank_densities.append(flank_count / flank_width)
        target_horizontal_density = (
            float(np.mean(flank_densities)) if flank_densities else 0.0
        )
        target_central_count = int(
            round(
                target_horizontal_density
                * (central_high - central_low)
                * float(horizontal_balance_ratio)
            )
        )
        horizontal_band_extra_target = max(
            target_central_count - baseline_central_count,
            0,
        )
        horizontal_band_extra_target = min(
            horizontal_band_extra_target,
            len(moving),
        )
        central_rng = np.random.default_rng(int(random_seed) + 1)
        central_moved_ordinals = set(
            int(value)
            for value in central_rng.choice(
                np.arange(len(moving)),
                size=horizontal_band_extra_target,
                replace=False,
            )
        )
    else:
        central_low = central_high = None
        baseline_central_count = 0
        target_horizontal_density = 0.0
        target_central_count = 0
        horizontal_band_extra_target = 0
        central_moved_ordinals: set[int] = set()
    visible_lengths = np.asarray(
        [
            max(
                float(record["visible_length_m"])
                * float(record["lateral_range_m"]),
                1.0e-9,
            )
            for record in anchor_records
        ],
        dtype=float,
    )
    raw_anchor_counts = float(len(moving)) * visible_lengths / visible_lengths.sum()
    anchor_counts = np.floor(raw_anchor_counts).astype(int)
    remainder = int(len(moving) - anchor_counts.sum())
    if remainder:
        order = np.argsort(-(raw_anchor_counts - anchor_counts))
        for index in order[:remainder]:
            anchor_counts[int(index)] += 1
    anchor_normal_offsets = [
        float(np.dot(center_points[index], sparse_normal))
        for index in range(len(anchor_records))
    ]
    influence_low = min(
        anchor_normal_offsets[index] - float(anchor_records[index]["lateral_range_m"])
        for index in range(len(anchor_records))
    )
    influence_high = max(
        anchor_normal_offsets[index] + float(anchor_records[index]["lateral_range_m"])
        for index in range(len(anchor_records))
    )
    influence_width_m = max(influence_high - influence_low, 1.0e-9)
    sparse_band_width_m = 2.0 * float(sparse_band_half_width_m)
    sparse_band_area_fraction = min(
        sparse_band_width_m / influence_width_m,
        1.0,
    )
    sparse_band_moved_target = int(
        round(
            float(len(moving))
            * sparse_band_area_fraction
            * float(sparse_band_density_ratio)
            * float(sparse_band_quota_multiplier)
        )
    )
    if sparse_band_moved_target:
        sparse_band_moved_ordinals = set(
            int(value)
            for value in np.linspace(
                0,
                max(len(moving) - 1, 0),
                sparse_band_moved_target,
            ).astype(int)
        )
    else:
        sparse_band_moved_ordinals = set()
    assigned_anchor_indices = [
        anchor_index
        for anchor_index, count in enumerate(anchor_counts)
        for _ in range(int(count))
    ]
    anchor_cursors = {index: 0 for index in range(len(anchor_records))}
    anchor_half_counts = {
        index: [int(count) // 2, int(count) - int(count) // 2]
        for index, count in enumerate(anchor_counts)
    }
    rng = np.random.default_rng(int(random_seed))
    moved_offsets: list[float] = []
    moved_by_anchor = {index: 0 for index in range(len(anchor_records))}
    successfully_moved_sources: set[int] = set()
    moved_time_ranges = {
        index: [float("inf"), float("-inf")]
        for index in range(len(anchor_records))
    }
    moved_times_by_anchor = {
        index: [] for index in range(len(anchor_records))
    }
    moved_points_by_anchor = {
        index: [] for index in range(len(anchor_records))
    }
    for ordinal, row in enumerate(moving.itertuples(index=False)):
        anchor_index = int(assigned_anchor_indices[ordinal])
        rank = anchor_cursors[anchor_index]
        anchor_cursors[anchor_index] += 1
        anchor = anchor_records[anchor_index]
        line = anchor["line"]
        visible_low_t, visible_high_t = (
            float(value) for value in anchor["visible_time_range_ms"]
        )
        z1, z2 = float(line[0, 1]), float(line[1, 1])
        low_t = max(min(z1, z2), float(time_range[0]))
        high_t = min(max(z1, z2), float(time_range[1]))
        if high_t <= low_t:
            continue
        if abs(z2 - z1) <= 1.0e-12:
            continue
        dx = float(line[1, 0] - line[0, 0])
        dy = 2.0 * float(line[1, 1] - line[0, 1])
        norm = float(np.hypot(dx, dy))
        if norm <= 1.0e-12:
            continue
        source_index = int(row.SourceIndex)
        placed = None
        upper_half_count = max(int(anchor_half_counts[anchor_index][0]), 1)
        for _ in range(100):
            if rank < anchor_half_counts[anchor_index][0]:
                half_fraction = (rank + 0.5) / upper_half_count
                lower_t = visible_low_t
                upper_t = 0.5 * (visible_low_t + visible_high_t)
            else:
                lower_rank = rank - anchor_half_counts[anchor_index][0]
                lower_half_count = max(
                    int(anchor_half_counts[anchor_index][1]),
                    1,
                )
                half_fraction = (lower_rank + 0.5) / lower_half_count
                lower_t = 0.5 * (visible_low_t + visible_high_t)
                upper_t = visible_high_t
            if ordinal in central_moved_ordinals and central_low is not None:
                anchor_mid_t = 0.5 * (visible_low_t + visible_high_t)
                central_mid_t = 0.5 * (central_low + central_high)
                if rank < anchor_half_counts[anchor_index][0]:
                    lower_t = max(central_low, visible_low_t)
                    upper_t = min(central_mid_t, anchor_mid_t)
                else:
                    lower_t = max(central_mid_t, anchor_mid_t)
                    upper_t = min(central_high, visible_high_t)
            half_fraction = float(
                np.clip(
                    half_fraction
                    + float(
                        rng.normal(
                            0.0,
                            float(longitudinal_jitter_fraction),
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
            sampled_t = float(lower_t + half_fraction * (upper_t - lower_t))
            sampled_t = float(np.clip(sampled_t, low_t, high_t))
            u = float(np.clip((sampled_t - z1) / (z2 - z1), 0.0, 1.0))
            distance_quantile = (
                (rank + 0.5) * 0.6180339887498949
                + 0.137 * (_ + 1)
            ) % 1.0
            offset_magnitude = float(anchor["lateral_range_m"]) * (
                distance_quantile ** float(halo_distance_power)
            )
            offset = offset_magnitude if (rank + _) % 2 == 0 else -offset_magnitude
            base_h = float(line[0, 0] + u * (line[1, 0] - line[0, 0]))
            base_t = float(line[0, 1] + u * (line[1, 1] - line[0, 1]))
            moved_h = base_h + offset * (-dy / norm)
            moved_t = base_t + offset * (dx / norm) / 2.0
            zone_low_t, zone_high_t = anchor["visible_time_range_ms"]
            if not (
                float(zone_low_t) <= moved_t <= float(zone_high_t)
                and float(lower_t) <= moved_t <= float(upper_t)
            ):
                continue
            sparse_distance = sparse_band_distance_m(moved_h, moved_t)
            target_sparse = ordinal in sparse_band_moved_ordinals
            if target_sparse and sparse_distance >= float(
                sparse_band_half_width_m
            ):
                continue
            if not target_sparse and sparse_distance < float(
                sparse_band_half_width_m
            ):
                continue
            azimuth = (
                anchor["dip_azimuth_deg"]
                + float(rng.normal(0.0, float(orientation_sigma_deg)))
            ) % 360.0
            dip = float(
                np.clip(
                    anchor["dip_deg"]
                    + float(
                        rng.normal(0.0, float(orientation_sigma_deg))
                    ),
                    1.0,
                    89.0,
                )
            )
            trace = orientation.trace_on_section(azimuth, dip, "XZ")
            if trace is None:
                continue
            length = float(
                np.clip(
                    float(work.at[source_index, "LengthM"]),
                    *shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M,
                )
            )
            half = 0.5 * length
            dh, dt = trace
            candidate_line = np.asarray(
                [
                    [moved_h - half * dh, moved_t - half * dt],
                    [moved_h + half * dh, moved_t + half * dt],
                ],
                dtype=float,
            )
            if (
                _line_well_min_distance_xz(
                    candidate_line,
                    track,
                    time_range,
                )
                < float(well_clearance_m)
            ):
                continue
            placed = (moved_h, moved_t, azimuth, dip, abs(offset))
            break
        if placed is None:
            continue
        moved_h, moved_t, azimuth, dip, offset_abs = placed
        work.at[source_index, "CenterX"] = moved_h
        work.at[source_index, "CenterY"] = float(
            np.interp(
                moved_t,
                track["TIME"].to_numpy(dtype=float),
                track["Y"].to_numpy(dtype=float),
            )
        )
        work.at[source_index, "CenterTime"] = moved_t
        work.at[source_index, "DipAzimuthDeg"] = azimuth
        work.at[source_index, "DipDeg"] = dip
        work.at[source_index, "LargeAnchorConditioned"] = 1
        work.at[source_index, "LargeAnchorID"] = int(anchor["SourceIndex"])
        work.at[source_index, "LargeAnchorDistanceM"] = offset_abs
        work.at[source_index, "LargeAnchorOrientationWeight"] = 1.0
        moved_offsets.append(offset_abs)
        successfully_moved_sources.add(source_index)
        moved_by_anchor[anchor_index] += 1
        moved_time_ranges[anchor_index][0] = min(
            moved_time_ranges[anchor_index][0], moved_t
        )
        moved_time_ranges[anchor_index][1] = max(
            moved_time_ranges[anchor_index][1], moved_t
        )
        moved_times_by_anchor[anchor_index].append(moved_t)
        moved_points_by_anchor[anchor_index].append((moved_h, moved_t))

    adjusted_neighbors = 0
    for _, row in eligible.iterrows():
        source_index = int(row["SourceIndex"])
        if source_index in successfully_moved_sources:
            continue
        geometry = _trace_geometry(row, "XZ")
        if geometry is None:
            continue
        line, _ = geometry
        distances = [
            _line_to_line_distance_m(line, record["line"])
            for record in anchor_records
        ]
        anchor_index = int(np.argmin(distances))
        distance = float(distances[anchor_index])
        if distance > float(anchor_radius_m):
            continue
        anchor = anchor_records[anchor_index]
        zone_low, zone_high = anchor["visible_time_range_ms"]
        if not float(zone_low) <= float(row["CenterTime"]) <= float(zone_high):
            continue
        weight = float(neighbor_max_orientation_weight) * (
            1.0 - distance / float(anchor_radius_m)
        ) ** 2
        if (
            sparse_band_distance_m(
                float(row["CenterX"]),
                float(row["CenterTime"]),
            )
            < float(sparse_band_half_width_m)
        ):
            weight *= float(sparse_band_density_ratio)
        azimuth_delta = (
            float(anchor["dip_azimuth_deg"])
            - float(work.at[source_index, "DipAzimuthDeg"])
            + 180.0
        ) % 360.0 - 180.0
        adjusted_azimuth = (
            float(work.at[source_index, "DipAzimuthDeg"])
            + weight * azimuth_delta
        ) % 360.0
        adjusted_dip = (
            (1.0 - weight) * float(work.at[source_index, "DipDeg"])
            + weight * float(anchor["dip_deg"])
        )
        adjusted_trace = orientation.trace_on_section(
            adjusted_azimuth,
            adjusted_dip,
            "XZ",
        )
        if adjusted_trace is None:
            continue
        length = float(
            np.clip(
                float(work.at[source_index, "LengthM"]),
                *shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M,
            )
        )
        half = 0.5 * length
        c_h = float(work.at[source_index, "CenterX"])
        c_t = float(work.at[source_index, "CenterTime"])
        dh, dt = adjusted_trace
        adjusted_line = np.asarray(
            [
                [c_h - half * dh, c_t - half * dt],
                [c_h + half * dh, c_t + half * dt],
            ],
            dtype=float,
        )
        if (
            _line_well_min_distance_xz(adjusted_line, track, time_range)
            < float(well_clearance_m)
        ):
            continue
        work.at[source_index, "DipAzimuthDeg"] = adjusted_azimuth
        work.at[source_index, "DipDeg"] = adjusted_dip
        work.at[source_index, "LargeAnchorConditioned"] = 2
        work.at[source_index, "LargeAnchorID"] = int(anchor["SourceIndex"])
        work.at[source_index, "LargeAnchorDistanceM"] = distance
        work.at[source_index, "LargeAnchorOrientationWeight"] = weight
        adjusted_neighbors += 1

    scale_before = {
        str(key): int(value)
        for key, value in dfn["FractureScale"].astype(str).value_counts().items()
    }
    scale_after = {
        str(key): int(value)
        for key, value in work["FractureScale"].astype(str).value_counts().items()
    }
    moved_in_sparse_band = int(
        sum(
            sparse_band_distance_m(h_value, time_value)
            < float(sparse_band_half_width_m)
            for points in moved_points_by_anchor.values()
            for h_value, time_value in points
        )
    )
    moved_outside_sparse_band = int(len(moved_offsets) - moved_in_sparse_band)
    influence_outside_width_m = max(
        influence_width_m - sparse_band_width_m,
        1.0e-9,
    )
    achieved_sparse_band_density_ratio = float(
        (
            moved_in_sparse_band / max(sparse_band_width_m, 1.0e-9)
        )
        / max(
            moved_outside_sparse_band / influence_outside_width_m,
            1.0e-12,
        )
    )
    if central_low is not None and central_high is not None:
        selected_small_indices = selected_xz.loc[
            selected_xz["FractureScale"].eq("small"),
            "SourceIndex",
        ]
        post_small_times = work.loc[
            work["SourceIndex"].isin(selected_small_indices),
            "CenterTime",
        ].to_numpy(dtype=float)
        actual_central_count = int(
            np.count_nonzero(
                (post_small_times >= central_low)
                & (post_small_times < central_high)
            )
        )
        actual_central_density = actual_central_count / max(
            central_high - central_low,
            1.0e-9,
        )
    else:
        actual_central_count = 0
        actual_central_density = 0.0
    summary = {
        "enabled": True,
        "method": "small_expansion_reallocation_near_well_crossing_large_xz_traces",
        "imaging_match_event_locked_patch_count": int(
            aligned_imaging_patch_count
        ),
        "scale_counts_before": scale_before,
        "scale_counts_after": scale_after,
        "scale_counts_unchanged": scale_before == scale_after,
        "large_anchor_source_indices": [
            int(record["SourceIndex"]) for record in anchor_records
        ],
        "large_anchor_crossing_times_ms": [
            float(record["crossing_time_ms"]) for record in anchor_records
        ],
        "eligible_expansion_patch_count": int(len(candidates)),
        "eligible_expansion_patch_count_crossing_well_before": int(
            crossing_expansion_count
        ),
        "moved_expansion_patch_count": int(len(moved_offsets)),
        "neighbor_orientation_adjusted_count": int(adjusted_neighbors),
        "move_fraction": float(move_fraction),
        "anchor_radius_m": float(anchor_radius_m),
        "max_cluster_offset_m": float(max_cluster_offset_m),
        "halo_distance_power": float(halo_distance_power),
        "longitudinal_jitter_fraction": float(longitudinal_jitter_fraction),
        "anchor_lateral_ranges_m": (
            [float(value) for value in anchor_lateral_ranges_m]
            if anchor_lateral_ranges_m
            else None
        ),
        "sparse_band": {
            "type": "diagonal_parallel_to_mean_large_fracture_trace",
            "mean_trace_angle_deg": float(sparse_band_angle_deg),
            "half_width_m": float(sparse_band_half_width_m),
            "target_density_ratio": float(sparse_band_density_ratio),
            "quota_multiplier": float(sparse_band_quota_multiplier),
            "achieved_density_ratio": achieved_sparse_band_density_ratio,
            "influence_width_m": float(influence_width_m),
            "band_area_fraction": float(sparse_band_area_fraction),
            "center_h_m": float(sparse_center[0]),
            "center_time_ms": float(sparse_center[1] / 2.0),
            "normal_vector_xz": [
                float(sparse_normal[0]),
                float(sparse_normal[1]),
            ],
            "moved_patch_count_inside": moved_in_sparse_band,
            "target_moved_patch_count_inside": int(
                sparse_band_moved_target
            ),
            "moved_patch_count_outside": moved_outside_sparse_band,
        },
        "horizontal_time_balance": {
            "band_ms": (
                [float(central_low), float(central_high)]
                if central_low is not None and central_high is not None
                else None
            ),
            "flank_ranges_ms": horizontal_balance_flank_ranges_ms,
            "baseline_count": int(baseline_central_count),
            "target_count": int(target_central_count),
            "extra_moved_target": int(horizontal_band_extra_target),
            "actual_count_after": int(actual_central_count),
            "target_density_per_ms": float(target_horizontal_density),
            "actual_density_per_ms": float(actual_central_density),
            "density_ratio_to_target": float(
                actual_central_density
                / max(target_horizontal_density, 1.0e-12)
            ),
        },
        "well_clearance_m": float(well_clearance_m),
        "orientation_sigma_deg": float(orientation_sigma_deg),
        "neighbor_max_orientation_weight": float(
            neighbor_max_orientation_weight
        ),
        "anchor_zones": [
            {
                "anchor_index": index,
                "large_anchor_source_index": int(record["SourceIndex"]),
                "crossing_time_ms": float(record["crossing_time_ms"]),
                "visible_time_range_ms": [
                    float(value) for value in record["visible_time_range_ms"]
                ],
                "visible_length_m": float(record["visible_length_m"]),
                "lateral_range_m": float(record["lateral_range_m"]),
                "target_moved_patch_count": int(anchor_counts[index]),
                "moved_patch_count": int(moved_by_anchor[index]),
                "moved_upper_half_count": int(
                    sum(
                        value
                        <= 0.5
                        * (
                            float(record["visible_time_range_ms"][0])
                            + float(record["visible_time_range_ms"][1])
                        )
                        for value in moved_times_by_anchor[index]
                    )
                ),
                "moved_lower_half_count": int(
                    sum(
                        value
                        > 0.5
                        * (
                            float(record["visible_time_range_ms"][0])
                            + float(record["visible_time_range_ms"][1])
                        )
                        for value in moved_times_by_anchor[index]
                    )
                ),
                "moved_time_range_ms": (
                    [
                        float(moved_time_ranges[index][0]),
                        float(moved_time_ranges[index][1]),
                    ]
                    if moved_by_anchor[index]
                    else None
                ),
            }
            for index, record in enumerate(anchor_records)
        ],
        "moved_offset_m": {
            "median": float(np.median(moved_offsets)) if moved_offsets else None,
            "p90": float(np.percentile(moved_offsets, 90)) if moved_offsets else None,
            "max": float(max(moved_offsets)) if moved_offsets else None,
        },
        "protected_source_type": "small_step3_imaging_well_control",
        "protected_imaging_source_indices": protected_source_indices,
        "eligible_source_types": sorted(
            str(value)
            for value in eligible["SourceType"].dropna().unique()
        ),
        "random_seed": int(random_seed),
    }
    return work.reset_index(drop=True), summary


def summarize_small_xz_anchor_relation(
    selected_xz: pd.DataFrame,
    anchor_source_indices: list[int],
) -> dict[str, Any]:
    anchors = selected_xz[
        selected_xz["SourceIndex"].isin(anchor_source_indices)
        & selected_xz["FractureScale"].eq("large")
    ]
    small = selected_xz[selected_xz["FractureScale"].eq("small")]
    if anchors.empty or small.empty:
        return {"small_count": int(len(small)), "anchor_count": int(len(anchors))}
    anchor_geometries = [
        _trace_geometry(row, "XZ")
        for _, row in anchors.iterrows()
    ]
    anchor_geometries = [value for value in anchor_geometries if value is not None]
    records: list[dict[str, float]] = []
    for _, row in small.iterrows():
        geometry = _trace_geometry(row, "XZ")
        if geometry is None:
            continue
        line, _ = geometry
        distances = [
            _line_to_line_distance_m(line, anchor[0])
            for anchor in anchor_geometries
        ]
        anchor_index = int(np.argmin(distances))
        distance = float(distances[anchor_index])
        anchor_line, _ = anchor_geometries[anchor_index]
        dx = float(anchor_line[1, 0] - anchor_line[0, 0])
        dy = 2.0 * float(anchor_line[1, 1] - anchor_line[0, 1])
        line_dx = float(line[1, 0] - line[0, 0])
        line_dy = 2.0 * float(line[1, 1] - line[0, 1])
        anchor_angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        line_angle = float(np.degrees(np.arctan2(line_dy, line_dx)) % 180.0)
        angle_difference = abs((line_angle - anchor_angle + 90.0) % 180.0 - 90.0)
        records.append(
            {
                "distance_m": distance,
                "angle_difference_deg": angle_difference,
            }
        )
    frame = pd.DataFrame(records)
    return {
        "small_count": int(len(small)),
        "anchor_count": int(len(anchors)),
        "within_5m": int(frame["distance_m"].le(5.0).sum()),
        "within_10m": int(frame["distance_m"].le(10.0).sum()),
        "within_20m": int(frame["distance_m"].le(20.0).sum()),
        "within_30m": int(frame["distance_m"].le(30.0).sum()),
        "within_60m": int(frame["distance_m"].le(60.0).sum()),
        "nearest_distance_median_m": float(frame["distance_m"].median()),
        "angle_difference_median_deg": float(
            frame["angle_difference_deg"].median()
        ),
        "angle_difference_within_30m_median_deg": float(
            frame.loc[
                frame["distance_m"].le(30.0),
                "angle_difference_deg",
            ].median()
        ),
    }


def summarize_well_crossings(
    selected: pd.DataFrame,
    track: pd.DataFrame,
    time_range: tuple[float, float],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scale in ("small", "medium", "large"):
        count = 0
        for _, row in selected[selected["FractureScale"].eq(scale)].iterrows():
            geometry = _trace_geometry(row, "XZ")
            if geometry is None:
                continue
            crosses, _ = _line_crosses_well_xz(
                geometry[0],
                track,
                time_range,
            )
            count += int(crosses)
        counts[scale] = count
    return counts


def _subdivide_by_mid(
    line: np.ndarray,
    plane_coords: np.ndarray,
    mid_times: np.ndarray,
) -> list[np.ndarray]:
    params = np.linspace(0.0, 1.0, 17)
    sampled_h = np.interp(params, [0.0, 1.0], line[:, 0])
    sampled_t = np.interp(params, [0.0, 1.0], line[:, 1])
    sampled_mid = np.interp(
        sampled_h,
        np.asarray(plane_coords, dtype=np.float64),
        np.asarray(mid_times, dtype=np.float64),
    )
    delta = sampled_t - sampled_mid
    split_params = [0.0, 1.0]
    for index in np.flatnonzero(delta[:-1] * delta[1:] < 0.0):
        left = float(delta[index])
        right = float(delta[index + 1])
        fraction = abs(left) / max(abs(left) + abs(right), 1.0e-12)
        split_params.append(
            float(params[index] + fraction * (params[index + 1] - params[index]))
        )
    split_params = sorted(set(split_params))
    subdivided = []
    for start, stop in zip(split_params[:-1], split_params[1:]):
        if stop - start <= 1.0e-9:
            continue
        subdivided.append(
            np.asarray(
                [
                    [
                        np.interp(start, params, sampled_h),
                        np.interp(start, params, sampled_t),
                    ],
                    [
                        np.interp(stop, params, sampled_h),
                        np.interp(stop, params, sampled_t),
                    ],
                ],
                dtype=np.float64,
            )
        )
    return subdivided or [line]


def panel_trace_entries(
    frame: pd.DataFrame,
    plane: str,
    plane_coords: np.ndarray,
    mid_times: np.ndarray,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for patch_id, row in enumerate(frame.itertuples(index=False)):
        geometry = _trace_geometry(row, plane)
        if geometry is None:
            continue
        line, _ = geometry
        scale = str(getattr(row, "FractureScale", "")).strip().lower()
        if scale not in ("small", "medium", "large"):
            scale = "small"
        if int(getattr(row, "IsDensityCompensationPatch", 0) or 0) == 1:
            category = "density_compensation"
        elif int(getattr(row, "IsImagingMatchPatch", 0) or 0) == 1:
            category = "imaging_match"
        elif int(getattr(row, "IsWellControlPatch", 0) or 0) == 1:
            category = "well_control"
        else:
            category = "base_prediction"
        entries.append(
            {
                "patch_id": int(patch_id),
                "line": line,
                "subdivisions": _subdivide_by_mid(
                    line,
                    plane_coords,
                    mid_times,
                ),
                "scale": scale,
                "category": category,
            }
        )
    return entries


def panel_line_segments(
    entries: list[dict[str, Any]],
    plane_coords: np.ndarray,
    mid_times: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[list[list[float]]]] = {
        (layer, scale): []
        for layer in TAIGU_LAYER_ORDER
        for scale in ("small", "medium", "large")
    }
    for entry in entries:
        for segment in entry["subdivisions"]:
            segment_center_h = 0.5 * float(segment[0, 0] + segment[1, 0])
            segment_center_t = 0.5 * float(segment[0, 1] + segment[1, 1])
            local_mid = float(
                np.interp(
                    segment_center_h,
                    np.asarray(plane_coords, dtype=np.float64),
                    np.asarray(mid_times, dtype=np.float64),
                )
            )
            layer = "太古界风化壳" if segment_center_t >= local_mid else "上部复合层"
            grouped[(layer, entry["scale"])].append(segment.tolist())
    return {
        key: np.asarray(segments, dtype=np.float64)
        for key, segments in grouped.items()
    }


def panel_comparison_rows(
    entries: list[dict[str, Any]],
    plane: str,
    plane_coords: np.ndarray,
    mid_times: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        h1, z1 = (float(value) for value in entry["line"][0])
        h2, z2 = (float(value) for value in entry["line"][1])
        center_t = 0.5 * (z1 + z2)
        center_h = 0.5 * (h1 + h2)
        local_mid = float(np.interp(center_h, plane_coords, mid_times))
        layer = "太古界风化壳" if center_t >= local_mid else "上部复合层"
        line_length, apparent_dip, line_angle = shared_compare.line_metrics(
            h1,
            z1,
            h2,
            z2,
        )
        rows.append(
            {
                "region": "太古界",
                "projection": plane,
                "source": "step8_final_dfn",
                "category": entry["category"],
                "layer": layer,
                "scale": entry["scale"],
                "patch_id": int(entry["patch_id"]),
                "h1": h1,
                "z1": z1,
                "h2": h2,
                "z2": z2,
                "line_length_m": line_length,
                "apparent_dip_deg": apparent_dip,
                "line_angle_deg": line_angle,
            }
        )
    return rows


def imaging_comparison_rows(
    segments: list[np.ndarray],
    plane: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(segments):
        h1, z1 = (float(value) for value in line[0])
        h2, z2 = (float(value) for value in line[1])
        line_length, apparent_dip, line_angle = shared_compare.line_metrics(
            h1, z1, h2, z2
        )
        rows.append(
            {
                "region": "太古界",
                "projection": plane,
                "source": "step3_imaging_truth",
                "category": "imaging_truth",
                "layer": "成像测井",
                "scale": "imaging",
                "patch_id": int(index),
                "h1": h1,
                "z1": z1,
                "h2": h2,
                "z2": z2,
                "line_length_m": line_length,
                "apparent_dip_deg": apparent_dip,
                "line_angle_deg": line_angle,
            }
        )
    return rows


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
    *,
    planes: tuple[str, ...] = PLANES,
) -> dict[str, Any]:
    figure_cfg = config["figure"]
    fig, plot_axes = plt.subplots(
        1,
        len(planes),
        figsize=(float(figure_cfg["width_in"]), float(figure_cfg["height_in"])),
        sharey=len(planes) > 1,
        squeeze=False,
    )
    plot_axes = plot_axes.reshape(-1)
    qc: dict[str, Any] = {}
    for ax, plane in zip(plot_axes, planes):
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
    parser.add_argument("--dfn-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overlay-only", action="store_true")
    parser.add_argument("--preserve-config-time-range", action="store_true")
    parser.add_argument(
        "--disable-large-anchor-conditioning",
        action="store_true",
    )
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if args.dfn_csv:
        config["step8_patches_csv"] = str(args.dfn_csv.resolve())
    if args.disable_large_anchor_conditioning:
        config["large_fracture_anchor_conditioning"] = {
            **dict(config.get("large_fracture_anchor_conditioning") or {}),
            "enabled": False,
        }
    validate_config(config)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    broad_time_range = (
        float(config["time_range_ms"][0]),
        float(config["time_range_ms"][1]),
    )

    global original_step9
    original_step9 = load_original_module()
    track = original_step9.load_track(config)
    axes = load_local_axes(config, track)
    imaging = load_imaging(original_step9, config, broad_time_range)
    if args.preserve_config_time_range:
        time_range = broad_time_range
    else:
        time_range = shared_compare.tight_imaging_window(
            imaging["TIME"],
            padding_ms=20.0,
            sample_interval_ms=float(
                config.get("section_sample_interval_ms", 2.0)
            ),
        )
    config["time_range_ms"] = [float(time_range[0]), float(time_range[1])]
    imaging = imaging[
        imaging["TIME"].between(float(time_range[0]), float(time_range[1]))
    ].reset_index(drop=True)
    times, values, sample_source = load_or_sample_wiggle(
        original_step9, config, track, axes
    )
    curves, horizon_source = load_or_compute_horizons(
        original_step9, config, track, axes, output_dir
    )
    dfn_source = load_dfn(config)
    selected_before_anchor_conditioning = {
        plane: select_patches_for_plane(
            dfn_source,
            track,
            axes[plane],
            plane,
            float(config.get("dfn_projection_half_width_m", 200.0)),
            time_range,
        )
        for plane in PLANES
    }
    anchor_cfg = dict(config.get("large_fracture_anchor_conditioning") or {})
    imaging_match_selection = {
        "audit_csv": None,
        "selected_patch_count": 0,
        "selected_weighted_point_count": 0,
    }
    protected_imaging_ids: set[str] = set()
    protected_imaging_events: pd.DataFrame | None = None
    audit_value = config.get("imaging_match_selection_audit_csv")
    if audit_value:
        audit_path = resolve_path(audit_value)
        if audit_path.exists():
            audit = pd.read_csv(audit_path, encoding="utf-8-sig")
            selected_audit = audit[
                pd.to_numeric(
                    audit["ImagingMatchSelected"],
                    errors="coerce",
                ).fillna(0).astype(int).eq(1)
            ].copy()
            protected_imaging_ids = set(
                selected_audit["WellControlSampleID"].astype(str).str.strip()
            )
            protected_imaging_events = selected_audit
            weights = pd.to_numeric(
                selected_audit.get("RawPointWeight", 1),
                errors="coerce",
            ).fillna(1)
            imaging_match_selection = {
                "audit_csv": str(audit_path),
                "selected_patch_count": int(len(selected_audit)),
                "selected_weighted_point_count": int(
                    round(float(weights.sum()))
                ),
            }
    if bool(anchor_cfg.get("enabled", True)):
        dfn, large_anchor_conditioning = (
            condition_small_dfn_near_xz_large_fractures(
                dfn_source,
                selected_before_anchor_conditioning["XZ"],
                track,
                imaging,
                time_range,
                protected_imaging_ids=protected_imaging_ids,
                protected_imaging_events=protected_imaging_events,
                move_fraction=float(anchor_cfg.get("move_fraction", 0.45)),
                anchor_radius_m=float(anchor_cfg.get("anchor_radius_m", 250.0)),
                max_cluster_offset_m=float(
                    anchor_cfg.get("max_cluster_offset_m", 120.0)
                ),
                halo_distance_power=float(
                    anchor_cfg.get("halo_distance_power", 0.75)
                ),
                longitudinal_jitter_fraction=float(
                    anchor_cfg.get("longitudinal_jitter_fraction", 0.01)
                ),
                anchor_lateral_ranges_m=(
                    [
                        float(value)
                        for value in anchor_cfg.get(
                            "anchor_lateral_ranges_m",
                            [],
                        )
                    ]
                    or None
                ),
                sparse_band_half_width_m=float(
                    anchor_cfg.get("sparse_band_half_width_m", 20.0)
                ),
                sparse_band_density_ratio=float(
                    anchor_cfg.get("sparse_band_density_ratio", 0.6)
                ),
                sparse_band_quota_multiplier=float(
                    anchor_cfg.get("sparse_band_quota_multiplier", 1.0)
                ),
                horizontal_balance_band_ms=(
                    [
                        float(value)
                        for value in anchor_cfg.get(
                            "horizontal_balance_band_ms",
                            [],
                        )
                    ]
                    or None
                ),
                horizontal_balance_flank_ranges_ms=(
                    [
                        [float(value) for value in flank]
                        for flank in anchor_cfg.get(
                            "horizontal_balance_flank_ranges_ms",
                            [],
                        )
                    ]
                    or None
                ),
                horizontal_balance_ratio=float(
                    anchor_cfg.get("horizontal_balance_ratio", 1.0)
                ),
                well_clearance_m=float(
                    anchor_cfg.get("well_clearance_m", 10.0)
                ),
                orientation_sigma_deg=float(
                    anchor_cfg.get("orientation_sigma_deg", 22.0)
                ),
                neighbor_max_orientation_weight=float(
                    anchor_cfg.get(
                        "neighbor_max_orientation_weight",
                        0.35,
                    )
                ),
                random_seed=int(anchor_cfg.get("random_seed", 405)),
            )
        )
    else:
        dfn = dfn_source.copy()
        large_anchor_conditioning = {
            "enabled": False,
            "reason": "disabled by config",
        }
    large_anchor_conditioning["imaging_match_selection"] = (
        imaging_match_selection
    )
    anchor_source_indices = [
        int(value)
        for value in large_anchor_conditioning.get(
            "large_anchor_source_indices", []
        )
    ]
    if anchor_source_indices:
        large_anchor_conditioning["small_relation_before"] = (
            summarize_small_xz_anchor_relation(
                selected_before_anchor_conditioning["XZ"],
                anchor_source_indices,
            )
        )
    selected = {
        plane: select_patches_for_plane(
            dfn,
            track,
            axes[plane],
            plane,
            float(config.get("dfn_projection_half_width_m", 200.0)),
            time_range,
        )
        for plane in PLANES
    }
    large_anchor_conditioning["well_crossings_before"] = (
        summarize_well_crossings(
            selected_before_anchor_conditioning["XZ"],
            track,
            time_range,
        )
    )
    large_anchor_conditioning["well_crossings_after"] = (
        summarize_well_crossings(
            selected["XZ"],
            track,
            time_range,
        )
    )
    if anchor_source_indices:
        large_anchor_conditioning["small_relation_after"] = (
            summarize_small_xz_anchor_relation(
                selected["XZ"],
                anchor_source_indices,
            )
        )
    local_source_indices = sorted(
        {
            int(value)
            for plane in PLANES
            for value in selected[plane]["SourceIndex"].tolist()
        }
    )
    local_conditioned_patches_path = (
        output_dir / "local_conditioned_dfn_patches.csv"
    )
    dfn[dfn["SourceIndex"].isin(local_source_indices)].to_csv(
        local_conditioned_patches_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"[wiggle-split] 成像点={len(imaging)} "
        f"DFN候选={len(dfn_source)} "
        f"XZ={len(selected['XZ'])} YZ={len(selected['YZ'])}",
        flush=True,
    )
    print(
        "[wiggle-split] "
        f"large-anchor moved={large_anchor_conditioning.get('moved_expansion_patch_count', 0)} "
        f"neighbor-adjusted={large_anchor_conditioning.get('neighbor_orientation_adjusted_count', 0)}",
        flush=True,
    )
    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_payload: dict[str, np.ndarray] = {
        "times": times,
        "well_x": track["X"].to_numpy(dtype=float),
        "well_y": track["Y"].to_numpy(dtype=float),
        "well_time": track["TIME"].to_numpy(dtype=float),
    }
    for plane in PLANES:
        base_payload[f"{plane.lower()}_h"] = axes[plane]
        base_payload[f"{plane.lower()}_values"] = values[plane]
        for item in config["horizons"]:
            base_payload[f"{plane.lower()}_{item['field']}"] = curves[plane][
                item["field"]
            ]
    np.savez_compressed(base_dir / "base_section_data.npz", **base_payload)

    imaging_cfg = dict(config["imaging"])
    imaging_cfg.update(
        {
            "color": shared_compare.IMAGING_COLOR,
            "halo_color": shared_compare.IMAGING_HALO_COLOR,
            "line_width": shared_compare.IMAGING_LINE_WIDTH,
            "halo_width": shared_compare.IMAGING_HALO_WIDTH,
        }
    )
    dfn_cfg = dict(config["dfn"])
    panel_entries = {
        plane: panel_trace_entries(
            selected[plane],
            plane,
            axes[plane],
            curves[plane]["MidTimeMs"],
        )
        for plane in PLANES
    }
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
    base_image = base_dir / "base_imaging_window.png"
    build_figure(
        base_image,
        "埕北古斜405 成像段 ±200 m：底板",
        config,
        times,
        axes,
        values,
        curves,
        lambda ax, plane: {},
        common_horizon_handles,
        [handle.get_label() for handle in common_horizon_handles],
    )
    write_json(
        base_dir / "base_manifest.json",
        {
            "status": "pass",
            "data": str(base_dir / "base_section_data.npz"),
            "image": str(base_image),
            "scope": "local_200m_imaging_window",
            "time_window_ms": [float(time_range[0]), float(time_range[1])],
            "imaging_time_padding_ms": 20.0,
        },
    )

    def draw_well_tracks(ax, plane: str) -> dict[str, Any]:
        return shared_compare.draw_well_context(
            ax,
            plane,
            track,
        )

    def draw_imaging_fractures(ax, plane: str) -> dict[str, Any]:
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
                    zorder=3.50,
                )
            )
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=imaging_cfg["color"],
                    linewidths=float(imaging_cfg["line_width"]),
                    alpha=0.95,
                    zorder=3.60,
                )
            )
        center_column = "X" if plane == "XZ" else "Y"
        ax.scatter(
            imaging[center_column],
            imaging["TIME"],
            s=float(imaging_cfg["point_size"]),
            facecolor="#fff7ed",
            edgecolor=imaging_cfg["color"],
            linewidth=0.45,
            zorder=3.70,
        )
        return {
            "imaging_fractures_drawn": int(len(segments)),
            "imaging_time_range_ms": [
                float(imaging["TIME"].min()),
                float(imaging["TIME"].max()),
            ],
            "fracture_zorder": 3.60,
        }

    def draw_imaging(ax, plane: str) -> dict[str, Any]:
        track_qc = draw_well_tracks(ax, plane)
        fracture_qc = draw_imaging_fractures(ax, plane)
        return {**track_qc, **fracture_qc}

    conventional_track_handle = Line2D(
        [0],
        [0],
        color=shared_compare.CONVENTIONAL_TRACK_COLOR,
        linewidth=shared_compare.CONVENTIONAL_TRACK_LINE_WIDTH,
        label="常规测井轨迹",
    )
    imaging_track_handle = Line2D(
        [0],
        [0],
        color=shared_compare.IMAGING_LOG_COLOR,
        linewidth=shared_compare.IMAGING_LOG_LINE_WIDTH,
        label="成像测井段",
    )
    imaging_fracture_handle = Line2D(
        [0],
        [0],
        color=imaging_cfg["color"],
        linewidth=float(imaging_cfg["line_width"]),
        marker="o",
        markersize=4.2,
        markerfacecolor="#fff7ed",
        markeredgecolor=imaging_cfg["color"],
        label="真实裂缝（倾向/倾角）",
    )
    imaging_handles = [
        conventional_track_handle,
        imaging_track_handle,
        imaging_fracture_handle,
        *common_horizon_handles,
    ]
    imaging_path = output_dir / "01_wiggle_area_imaging.png"
    imaging_qc = None
    if not args.overlay_only:
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
            panel_entries[plane],
            axes[plane],
            curves[plane]["MidTimeMs"],
        )
        drawn: dict[str, int] = {}
        drawn_by_layer: dict[str, int] = {}
        for scale in ("small", "medium", "large"):
            drawn[scale] = 0
            for layer in TAIGU_LAYER_ORDER:
                segments = grouped.get(
                    (layer, scale), np.empty((0, 2, 2))
                )
                if len(segments):
                    ax.add_collection(
                        LineCollection(
                            segments,
                            colors=shared_compare.DFN_SCALE_COLORS[scale],
                            linewidths=shared_compare.DFN_SCALE_WIDTHS[scale],
                            alpha=float(dfn_cfg[scale]["alpha"]),
                            zorder=1.8 if scale == "small" else 2.0 if scale == "medium" else 2.2,
                        )
                    )
                count = int(len(segments))
                drawn[scale] += count
                drawn_by_layer[layer] = drawn_by_layer.get(layer, 0) + count
        track_qc = draw_well_tracks(ax, plane)
        return {
            "dfn_line_counts_by_scale": drawn,
            "dfn_line_counts_by_layer": drawn_by_layer,
            **track_qc,
        }

    layer_scale_handles = []
    for scale in ("small", "medium", "large"):
        handle = Line2D(
            [0],
            [0],
            color=shared_compare.DFN_SCALE_COLORS[scale],
            linewidth=shared_compare.DFN_SCALE_WIDTHS[scale],
            label={
                "small": "小尺度",
                "medium": "中尺度",
                "large": "大尺度",
            }[scale],
        )
        layer_scale_handles.append(handle)
    dfn_handles = [
        *layer_scale_handles,
        conventional_track_handle,
        imaging_track_handle,
    ]
    dfn_handles.extend(common_horizon_handles)
    dfn_path = output_dir / "02_wiggle_area_dfn.png"
    dfn_qc = None
    if not args.overlay_only:
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

    fusion_path = output_dir / "03_wiggle_area_dfn_imaging_fusion.png"

    def draw_overlay(ax, plane: str) -> dict[str, Any]:
        dfn_overlay_qc = draw_dfn(ax, plane)
        imaging_overlay_qc = draw_imaging_fractures(ax, plane)
        return {
            **dfn_overlay_qc,
            **imaging_overlay_qc,
        }

    overlay_handles = [
        *layer_scale_handles,
        conventional_track_handle,
        imaging_track_handle,
        imaging_fracture_handle,
        *common_horizon_handles,
    ]
    overlay_qc = build_figure(
        fusion_path,
        "埕北古斜405 成像测井段 ±200 m：地震背景 + 多尺度DFN + 成像真值",
        config,
        times,
        axes,
        values,
        curves,
        draw_overlay,
        overlay_handles,
        [handle.get_label() for handle in overlay_handles],
    )
    print(f"[wiggle-split] 已生成 {fusion_path}", flush=True)

    xz_only_config = dict(config)
    xz_only_config["figure"] = {
        **config["figure"],
        "width_in": 11.5,
        "height_in": float(config["figure"].get("height_in", 8.4)),
    }
    xz_dfn_path = (
        output_dir
        / "04_taigu_xz_dfn_large_fracture_anchored_small_clusters.png"
    )
    xz_dfn_qc = build_figure(
        xz_dfn_path,
        "埕北古斜405 成像测井段 ±200 m：XZ小尺度DFN大裂缝锚定",
        xz_only_config,
        times,
        axes,
        values,
        curves,
        draw_dfn,
        dfn_handles,
        [handle.get_label() for handle in dfn_handles],
        planes=("XZ",),
    )
    print(f"[wiggle-split] 已生成 {xz_dfn_path}", flush=True)

    xz_overlay_path = (
        output_dir
        / "05_taigu_xz_dfn_imaging_large_fracture_anchored_small_clusters.png"
    )
    xz_overlay_qc = build_figure(
        xz_overlay_path,
        "埕北古斜405 成像测井段 ±200 m：XZ锚定小尺度DFN与成像裂缝对比",
        xz_only_config,
        times,
        axes,
        values,
        curves,
        draw_overlay,
        overlay_handles,
        [handle.get_label() for handle in overlay_handles],
        planes=("XZ",),
    )
    print(f"[wiggle-split] 已生成 {xz_overlay_path}", flush=True)

    xz_imaging_only_path = (
        output_dir
        / "06_taigu_xz_imaging_only.png"
    )
    xz_imaging_only_qc = build_figure(
        xz_imaging_only_path,
        "埕北古斜405 成像测井段 ±200 m：XZ成像测井裂缝",
        xz_only_config,
        times,
        axes,
        values,
        curves,
        draw_imaging,
        imaging_handles,
        [handle.get_label() for handle in imaging_handles],
        planes=("XZ",),
    )
    print(f"[wiggle-split] 已生成 {xz_imaging_only_path}", flush=True)

    imaging_rows: list[dict[str, Any]] = []
    dfn_rows: list[dict[str, Any]] = []
    for plane in PLANES:
        imaging_rows.extend(
            imaging_comparison_rows(
                imaging_line_segments(
                    imaging,
                    plane,
                    float(imaging_cfg["line_length_m"]),
                ),
                plane,
            )
        )
        dfn_rows.extend(
            panel_comparison_rows(
                panel_entries[plane],
                plane,
                axes[plane],
                curves[plane]["MidTimeMs"],
            )
        )
    imaging_rows_df = pd.DataFrame(imaging_rows)
    dfn_rows_df = pd.DataFrame(dfn_rows)
    intersection_manifest = shared_compare.write_intersection_layer(
        output_dir,
        region="太古界",
        imaging_rows=imaging_rows,
        dfn_rows=dfn_rows,
        metadata={
            "dfn_geometry_source": "STEP8_CSV_DIP_AZIMUTH_TRACE",
            "time_window_ms": [float(time_range[0]), float(time_range[1])],
            "imaging_time_padding_ms": 20.0,
            "half_width_m": float(
                config.get("dfn_projection_half_width_m", 200.0)
            ),
            "dfn_length_policy_m": list(
                shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M
            ),
            "selection_policy": "surface_near_patch_and_trace_intersects_local_window",
        },
    )
    comparison_statistics = shared_compare.comparison_stats(
        imaging_rows_df,
        dfn_rows_df,
        time_min=float(time_range[0]),
        time_max=float(time_range[1]),
    )
    output_paths = (
        {
            "fusion": str(fusion_path),
            "xz_dfn": str(xz_dfn_path),
            "xz_overlay": str(xz_overlay_path),
            "xz_imaging_only": str(xz_imaging_only_path),
        }
        if args.overlay_only
        else {
            "imaging": str(imaging_path),
            "dfn": str(dfn_path),
            "fusion": str(fusion_path),
            "xz_dfn": str(xz_dfn_path),
            "xz_overlay": str(xz_overlay_path),
            "xz_imaging_only": str(xz_imaging_only_path),
        }
    )
    anchor_enabled = bool(
        large_anchor_conditioning.get("enabled", False)
    )
    checks = {
        "base_image_exists": bool(base_image.exists()),
        "all_output_images_exist": all(Path(path).exists() for path in output_paths.values()),
        "imaging_intersections_nonempty": bool(imaging_rows),
        "dfn_intersections_nonempty": bool(dfn_rows),
        "imaging_time_window_padded_20ms": bool(
            float(time_range[0]) <= float(imaging["TIME"].min()) - 20.0
            and float(time_range[1]) >= float(imaging["TIME"].max()) + 20.0
        ),
        "track_layer_order": all(
            value.get("track_zorders", {}).get("conventional")
            < value.get("track_zorders", {}).get("imaging_log")
            for value in (
                imaging_qc.values()
                if imaging_qc is not None
                else [overlay_qc]
            )
        ),
        "imaging_fractures_above_tracks": all(
            value.get("fracture_zorder", -1.0)
            > value.get("track_zorders", {}).get("imaging_log", 1.0e9)
            for value in (
                imaging_qc.values()
                if imaging_qc is not None
                else [overlay_qc]
            )
        ),
        "dfn_geometry_uses_full_length_policy": all(
            float(row["line_length_m"])
            <= float(shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M[1]) + 1.0e-6
            for row in dfn_rows
        ),
        "large_fracture_anchor_conditioning_applied": bool(
            not anchor_enabled
            or large_anchor_conditioning.get("moved_expansion_patch_count", 0) > 0
        ),
        "well_small_crossings_match_imaging_selection": bool(
            not anchor_enabled
            or (
                large_anchor_conditioning.get("well_crossings_after", {}).get(
                    "small", -1
                )
                == large_anchor_conditioning.get(
                    "imaging_match_selection", {}
                ).get("selected_patch_count", -2)
            )
        ),
        "well_small_crossings_within_277_weighted_points": bool(
            large_anchor_conditioning.get("well_crossings_after", {}).get(
                "small", 10**9
            )
            <= large_anchor_conditioning.get(
                "imaging_match_selection", {}
            ).get("selected_weighted_point_count", -1)
        ),
        "large_anchor_density_similar": bool(
            not anchor_enabled
            or len(large_anchor_conditioning.get("anchor_zones", [])) >= 2
            and (
                max(
                    float(zone["moved_patch_count"])
                    / max(
                        float(zone["visible_length_m"])
                        * float(zone["lateral_range_m"]),
                        1.0e-9,
                    )
                    for zone in large_anchor_conditioning["anchor_zones"]
                )
                / max(
                    min(
                        float(zone["moved_patch_count"])
                        / max(
                            float(zone["visible_length_m"])
                            * float(zone["lateral_range_m"]),
                            1.0e-9,
                        )
                        for zone in large_anchor_conditioning["anchor_zones"]
                    ),
                    1.0e-12,
                )
                <= 1.15
            )
        ),
        "large_anchor_halves_balanced": bool(
            not anchor_enabled
            or (
                large_anchor_conditioning.get("anchor_zones")
            and all(
                max(
                    float(zone["moved_upper_half_count"]),
                    float(zone["moved_lower_half_count"]),
                )
                / max(
                    min(
                        float(zone["moved_upper_half_count"]),
                        float(zone["moved_lower_half_count"]),
                    ),
                    1.0,
                )
                <= 1.25
                for zone in large_anchor_conditioning["anchor_zones"]
            )
            )
        ),
        "sparse_band_density_ratio_matches": bool(
            not anchor_enabled
            or abs(
                float(
                    large_anchor_conditioning.get("sparse_band", {}).get(
                        "achieved_density_ratio",
                        0.0,
                    )
                )
                - float(
                    large_anchor_conditioning.get("sparse_band", {}).get(
                        "target_density_ratio",
                        0.35,
                    )
                )
            )
            <= 0.15
        ),
        "horizontal_time_band_not_sparse": bool(
            not anchor_enabled
            or float(
                large_anchor_conditioning.get(
                    "horizontal_time_balance",
                    {},
                ).get("density_ratio_to_target", 0.0)
            )
            >= 0.8
        ),
        "orientation_conditioning_softened": bool(
            not anchor_enabled
            or float(
                large_anchor_conditioning.get(
                    "neighbor_max_orientation_weight",
                    1.0,
                )
            )
            <= 0.4
            and float(
                large_anchor_conditioning.get("orientation_sigma_deg", 0.0)
            )
            >= 15.0
        ),
        "scale_counts_unchanged": bool(
            not anchor_enabled
            or large_anchor_conditioning.get("scale_counts_unchanged", False)
        ),
        "xz_only_results_exist": bool(
            xz_dfn_path.exists()
            and xz_overlay_path.exists()
            and xz_imaging_only_path.exists()
        ),
        "local_conditioned_patch_export_exists": bool(
            local_conditioned_patches_path.exists()
            and len(local_source_indices) > 0
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "profile_well": config["profile_well"],
        "time_range_ms": list(time_range),
        "half_width_m": float(config.get("dfn_projection_half_width_m", 200.0)),
        "inputs": {
            "obn_sampling": sample_source,
            "horizons": horizon_source,
            "dfn_loaded_rows": int(len(dfn)),
            "dfn_center_rows_in_display_window": int(
                dfn["CenterTime"].between(*time_range).sum()
            ),
            "imaging_points": int(len(imaging)),
            "dfn_selected": {
                plane: int(len(selected[plane])) for plane in PLANES
            },
            "dfn_panel_entries": {
                plane: int(len(panel_entries[plane])) for plane in PLANES
            },
        },
        "local_conditioned_patches": {
            "path": str(local_conditioned_patches_path),
            "patch_count": int(len(local_source_indices)),
        },
        "large_fracture_anchor_conditioning": large_anchor_conditioning,
        "orientation_contract": {
            "imaging": "FracAzimuth = 真倾向方位；FracDip = 真倾角",
            "dfn": "DipAzimuthDeg = 真倾向方位；DipDeg = 真倾角",
            "section_rule": "裂缝面与XZ/YZ剖面面的交线",
            "time_scale_m_per_ms": float(config.get("time_scale_m_per_ms", 2.0)),
            "dfn_geometry_policy": "CSV产状字段；不使用V1..V4顶点",
            "dfn_length_policy_m": list(
                shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M
            ),
        },
        "color_semantics": {
            "color_by_scale": dict(shared_compare.DFN_SCALE_COLORS),
            "linewidth": TAIGU_SCALE_WIDTHS,
            "large_scale_halo": {
                "enabled": False,
            },
            "supp_layer_display_merged_into": "上部复合层",
            "track_bottom_to_top": [
                "conventional_well_track",
                "imaging_log_interval",
                "imaging_fractures",
            ],
            "conventional_track_color": shared_compare.CONVENTIONAL_TRACK_COLOR,
            "imaging_log_color": shared_compare.IMAGING_LOG_COLOR,
            "imaging_fracture_color": shared_compare.IMAGING_COLOR,
        },
        "base_layer": {
            "directory": str(base_dir),
            "data": str(base_dir / "base_section_data.npz"),
            "image": str(base_image),
        },
        "intersection_layer": intersection_manifest,
        "figure_qc": {
            **({"imaging": imaging_qc} if imaging_qc is not None else {}),
            **({"dfn": dfn_qc} if dfn_qc is not None else {}),
            **({"overlay": overlay_qc} if overlay_qc is not None else {}),
            "xz_dfn": xz_dfn_qc,
            "xz_overlay": xz_overlay_qc,
            "xz_imaging_only": xz_imaging_only_qc,
        },
        "comparison_statistics": comparison_statistics,
        "checks": checks,
        "outputs": output_paths,
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
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
