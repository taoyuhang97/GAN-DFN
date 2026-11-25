#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""砂砾岩车页1导眼：将 Step3 真实裂缝与 DFN 预测拆成独立剖面图。

支持在当前 Step8 DFN 上生成井周局部对比剖面：

* 地震波形+正振幅变面积背景；
* T4-T7层位、车页1导眼井轨迹；
* 局部范围、道数和时间抽稀由配置控制；
* 两张图使用完全相同的剖面几何和坐标范围。

真实裂缝图只叠加 Step3 成像井段轨迹、真实裂缝解释片和原始裂缝点；
DFN图只叠加成像井段轨迹和三尺度DFN裂缝，不叠加真实裂缝点/解释片。

用法::

    /home/tyh/anaconda3/envs/gan-dfn/bin/python3 \
      优化阶段二/正式主线/step9_section_visualize/build_cheye1_imaging_dfn_split_sections.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
FORMAL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FORMAL_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_cheye1_dfn_multibackground_sections as mb
from build_cheye1_dfn_coherence_sections import (
    add_dfn_segments,
    add_imaging_fracture_patch_segments,
    build_imaging_fracture_patch_segments,
    build_namespace,
    draw_fracture_labels,
    infer_dfn_patch_csv,
    load_dfn_patch_metadata,
    load_step3_imaging_fracture_labels,
    load_step3_imaging_segment_track,
    scan_vtk_intersections,
)
from build_all_area_section_visualization import ProjectionSegment
from build_well_seismic_amplitude_sections import plot_wiggle_variable_area
from dfn_section_overlay import SectionOverlayContext
from 太古界.common.orientation_frame import convention as orientation
from well_curved_section_common import (
    AttributeSection,
    prepare_geometry,
    read_json,
    sample_volume_sections,
    save_section_pair_npz,
    write_json,
)

_imaging_compare_spec = importlib.util.spec_from_file_location(
    "sand_step9_imaging_200m_compare",
    REPO_ROOT / "common" / "step9_imaging_200m_compare.py",
)
if _imaging_compare_spec is None or _imaging_compare_spec.loader is None:
    raise ImportError("cannot load shared 200 m comparison contract")
shared_compare = importlib.util.module_from_spec(_imaging_compare_spec)
_imaging_compare_spec.loader.exec_module(shared_compare)
comparison_stats = shared_compare.comparison_stats
draw_well_context = shared_compare.draw_well_context
segment_rows = shared_compare.segment_rows
tight_imaging_window = shared_compare.tight_imaging_window
write_intersection_layer = shared_compare.write_intersection_layer

DFN_200M_SCALE_LINE_STYLES = {
    scale: (float(width), "-")
    for scale, width in shared_compare.DFN_SCALE_WIDTHS.items()
}
DFN_200M_SCALE_LINE_STYLES[""] = (1.4, "-")
DFN_200M_LARGE_HALO_ENABLED = False
DFN_200M_SCALE_LEGEND_STYLES = {
    scale: (
        shared_compare.DFN_SCALE_COLORS[scale],
        "-",
        float(shared_compare.DFN_SCALE_WIDTHS[scale]),
        label,
    )
    for scale, label in (
        ("small", "小尺度"),
        ("medium", "中尺度"),
        ("large", "大尺度"),
    )
}


DEFAULT_CONFIG = (
    SCRIPT_DIR / "configs" / "formal_mine_multiscale_flow_v1.json"
)
DEFAULT_REFERENCE_SUMMARY = (
    FORMAL_ROOT
    / "output"
    / "step9_section_presentation"
    / "mine_v1"
    / "ppt_decimated"
    / "section_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    FORMAL_ROOT
    / "output"
    / "step9_section_presentation"
    / "mine_v1"
    / "split_imaging_dfn_v1"
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成砂砾岩真实裂缝/DFN独立剖面图")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dfn-patch-csv", type=Path)
    p.add_argument("--input-vtk", type=Path)
    p.add_argument("--local-axis-radius-m", type=float)
    p.add_argument("--legend-width-in", type=float)
    p.add_argument("--overlay-only", action="store_true")
    p.add_argument(
        "--dfn-geometry-source",
        choices=("vtk", "csv"),
        default="csv",
    )
    p.add_argument("--preserve-local-time-window", action="store_true")
    p.add_argument(
        "--disable-well-orientation-calibration",
        action="store_true",
    )
    return p


def set_display_bounds(
    section: AttributeSection,
    local_summary: dict[str, Any],
) -> AttributeSection:
    section.display_time_min = float(local_summary["display_time_min"])
    section.display_time_max = float(local_summary["display_time_max"])
    return section


def build_local_context(
    config: dict[str, Any],
    geometry,
) -> tuple[SectionOverlayContext, dict[str, Any]]:
    """扫描一次局部200 m DFN；不扫描/绘制原始断层。"""
    args = build_namespace(config, float(config.get("local_dfn_half_width_m", 200.0)))
    args.small_projection_half_width = float(
        config.get(
            "local_small_projection_half_width_m",
            config.get("small_projection_half_width_m", args.half_width),
        )
    )
    patch_csv = infer_dfn_patch_csv(config)
    metadata = load_dfn_patch_metadata(patch_csv)
    args.dfn_patch_metadata = metadata
    args.well_trajectory_csv = geometry.well_path
    args.selected_well_name = geometry.well_name

    from common.horizon_trace_table.horizon_contract import build_spatial_lookup

    surfaces = build_spatial_lookup(config)
    segments, scan = scan_vtk_intersections(args, surfaces, geometry.well_df)
    group_paths = [
        Path(str(value)).resolve()
        for value in config.get("step3_imaging_group_csvs", [])
    ]
    fracture_df = load_step3_imaging_fracture_labels(
        group_paths,
        geometry.well_name,
        dict(config.get("target_block") or {}),
    )
    imaging_df = load_step3_imaging_segment_track(
        group_paths,
        geometry.well_name,
        dict(config.get("target_block") or {}),
    )
    imaging_patches = build_imaging_fracture_patch_segments(fracture_df)
    context = SectionOverlayContext(
        "local_200m",
        segments,
        [],
        imaging_patches,
        fracture_df,
        imaging_df,
        geometry.summary,
    )
    summary = {
        "dfn_patch_csv": str(patch_csv) if patch_csv else None,
        "dfn_patch_count": int(len(metadata)) if metadata is not None else 0,
        "dfn_scan": scan,
        "step3_group_csvs": [str(path) for path in group_paths],
        "step3_fracture_point_count": int(len(fracture_df)),
        "step3_imaging_track_sample_count": int(len(imaging_df)),
        "step3_imaging_patch_segment_count": int(len(imaging_patches)),
    }
    return context, summary


def calibrate_small_dfn_well_orientations(
    frame: pd.DataFrame,
    imaging_events: pd.DataFrame,
    *,
    neighbor_radius_m: float = 60.0,
    neighbor_max_weight: float = 0.75,
    max_match_distance_m: float = 75.0,
    time_scale_m_per_ms: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply event-level imaging constraints without changing fracture scale."""

    required = {
        "FractureScale",
        "CenterX",
        "CenterY",
        "CenterTime",
        "DipAzimuthDeg",
        "DipDeg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"DFN frame missing calibration columns: {missing}")
    event_required = {"X", "Y", "TIME", "Frac_Azimuth", "Frac_Dip"}
    missing = sorted(event_required - set(imaging_events.columns))
    if missing:
        raise ValueError(f"imaging events missing calibration columns: {missing}")

    work = frame.reset_index(drop=True).copy()
    scale_before = {
        str(key): int(value)
        for key, value in work["FractureScale"].astype(str).str.lower().value_counts().items()
    }
    small_mask = work["FractureScale"].astype(str).str.lower().eq("small")
    candidate_positions = np.flatnonzero(small_mask.to_numpy())
    events = imaging_events.copy()
    for column in event_required:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    events = events.dropna(subset=list(event_required)).sort_values("TIME").reset_index(drop=True)
    if not len(events):
        raise RuntimeError("no imaging events available for small-DFN calibration")
    if not len(candidate_positions):
        raise RuntimeError("no small DFN patches available for well calibration")

    candidates = work.iloc[candidate_positions]
    dx = candidates["CenterX"].to_numpy(dtype=float)[:, None] - events["X"].to_numpy(dtype=float)[None, :]
    dy = candidates["CenterY"].to_numpy(dtype=float)[:, None] - events["Y"].to_numpy(dtype=float)[None, :]
    dt = (
        candidates["CenterTime"].to_numpy(dtype=float)[:, None]
        - events["TIME"].to_numpy(dtype=float)[None, :]
    ) * float(time_scale_m_per_ms)
    distance = np.sqrt(dx * dx + dy * dy + dt * dt)
    candidate_rows, event_rows = linear_sum_assignment(distance)

    work["WellOrientationCalibrated"] = 0
    work["WellCalibrationEventID"] = -1
    work["WellCalibrationDistanceM"] = np.nan
    work["WellCalibrationWeight"] = 0.0
    work["OriginalDipDeg"] = work["DipDeg"]
    work["OriginalDipAzimuthDeg"] = work["DipAzimuthDeg"]

    matched_positions: set[int] = set()
    matched_distances: list[float] = []
    for candidate_row, event_row in zip(candidate_rows, event_rows):
        match_distance = float(distance[candidate_row, event_row])
        if match_distance > float(max_match_distance_m):
            continue
        position = int(candidate_positions[candidate_row])
        event = events.iloc[int(event_row)]
        work.at[position, "CenterX"] = float(event["X"])
        work.at[position, "CenterY"] = float(event["Y"])
        work.at[position, "CenterTime"] = float(event["TIME"])
        work.at[position, "DipAzimuthDeg"] = float(event["Frac_Azimuth"])
        work.at[position, "DipDeg"] = float(event["Frac_Dip"])
        work.at[position, "NearestTrajectoryDistance"] = 0.0
        work.at[position, "WellOrientationCalibrated"] = 1
        work.at[position, "WellCalibrationEventID"] = int(event_row)
        work.at[position, "WellCalibrationDistanceM"] = match_distance
        work.at[position, "WellCalibrationWeight"] = 1.0
        matched_positions.add(position)
        matched_distances.append(match_distance)

    neighbor_count = 0
    for candidate_row, position in enumerate(candidate_positions):
        position = int(position)
        if position in matched_positions:
            continue
        event_row = int(np.argmin(distance[candidate_row]))
        nearest_distance = float(distance[candidate_row, event_row])
        if nearest_distance > float(neighbor_radius_m):
            continue
        event = events.iloc[event_row]
        weight = float(neighbor_max_weight) * (
            1.0 - nearest_distance / float(neighbor_radius_m)
        ) ** 2
        azimuth_delta = (
            float(event["Frac_Azimuth"]) - float(work.at[position, "DipAzimuthDeg"])
            + 180.0
        ) % 360.0 - 180.0
        work.at[position, "DipAzimuthDeg"] = (
            float(work.at[position, "DipAzimuthDeg"]) + weight * azimuth_delta
        ) % 360.0
        work.at[position, "DipDeg"] = (
            (1.0 - weight) * float(work.at[position, "DipDeg"])
            + weight * float(event["Frac_Dip"])
        )
        work.at[position, "WellOrientationCalibrated"] = 2
        work.at[position, "WellCalibrationEventID"] = event_row
        work.at[position, "WellCalibrationDistanceM"] = nearest_distance
        work.at[position, "WellCalibrationWeight"] = weight
        neighbor_count += 1

    scale_after = {
        str(key): int(value)
        for key, value in work["FractureScale"].astype(str).str.lower().value_counts().items()
    }
    summary = {
        "enabled": True,
        "method": "one_to_one_time_xy_match_then_imaging_true_orientation",
        "small_only": True,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "scale_counts_unchanged": scale_before == scale_after,
        "imaging_event_count": int(len(events)),
        "matched_event_count": int(len(matched_positions)),
        "matched_patch_count": int(len(matched_positions)),
        "neighbor_patch_count": int(neighbor_count),
        "max_match_distance_m": float(max_match_distance_m),
        "matched_distance_m": {
            "min": float(min(matched_distances)) if matched_distances else None,
            "median": float(np.median(matched_distances)) if matched_distances else None,
            "max": float(max(matched_distances)) if matched_distances else None,
        },
        "neighbor_radius_m": float(neighbor_radius_m),
        "neighbor_max_weight": float(neighbor_max_weight),
        "time_scale_m_per_ms": float(time_scale_m_per_ms),
        "orientation_policy": (
            "matched small DFN patches copy their paired imaging event's "
            "position and true azimuth/dip; neighbors receive a bounded blend"
        ),
    }
    return work.reset_index(drop=True), summary


def build_csv_orientation_context(
    config: dict[str, Any],
    geometry,
) -> tuple[SectionOverlayContext, dict[str, Any]]:
    """Render DFN orientations from Step8 CSV instead of stale VTK vertices."""
    patch_csv = infer_dfn_patch_csv(config)
    if patch_csv is None:
        raise FileNotFoundError("Step8 DFN CSV is required for CSV orientation rendering")
    columns = [
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "PatchAreaM2",
        "DipAzimuthDeg",
        "DipDeg",
        "AzimuthDeg",
        "LayerGroup",
        "FractureScale",
        "NearestTrajectoryDistance",
        "IsDensityCompensationPatch",
        "IsImagingMatchPatch",
        "IsWellControlPatch",
    ]
    available = pd.read_csv(patch_csv, nrows=0).columns
    columns = [column for column in columns if column in available]
    frame = pd.read_csv(patch_csv, usecols=columns, low_memory=False)
    for column in columns:
        if column not in {"LayerGroup", "FractureScale"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["CenterTime"].between(
            float(geometry.summary["display_time_min"]),
            float(geometry.summary["display_time_max"]),
        )
        & frame.get(
            "NearestTrajectoryDistance",
            pd.Series(0.0, index=frame.index),
        ).le(float(config.get("local_dfn_half_width_m", 200.0)))
    ].copy()
    group_paths = [
        Path(str(value)).resolve()
        for value in config.get("step3_imaging_group_csvs", [])
    ]
    fracture_df = load_step3_imaging_fracture_labels(
        group_paths,
        geometry.well_name,
        dict(config.get("target_block") or {}),
    )
    imaging_df = load_step3_imaging_segment_track(
        group_paths,
        geometry.well_name,
        dict(config.get("target_block") or {}),
    )
    imaging_patches = build_imaging_fracture_patch_segments(fracture_df)
    if bool(config.get("well_orientation_calibration_enabled", True)):
        frame, orientation_calibration = calibrate_small_dfn_well_orientations(
            frame,
            fracture_df,
        )
    else:
        orientation_calibration = {
            "enabled": False,
            "reason": "disabled for v3 density-balance compatibility",
        }
    frame["SourceRowID"] = np.arange(len(frame), dtype=int)
    local_conditioned_patches_path = (
        Path(str(config["output_dir"]))
        / "local_conditioned_dfn_patches.csv"
    )
    frame.to_csv(
        local_conditioned_patches_path,
        index=False,
        encoding="utf-8-sig",
    )

    interval_by_layer = {
        "沙三段": "T4->T6",
        "沙四段": "T6->T7",
    }
    segments: list[ProjectionSegment] = []
    for projection in ("XZ", "YZ"):
        center_column = f"Center{'X' if projection == 'XZ' else 'Y'}"
        coordinate_values = (
            geometry.x_values if projection == "XZ" else geometry.y_values
        )
        for row in frame.itertuples(index=False):
            center_h = float(getattr(row, center_column))
            if not (
                float(coordinate_values[0])
                <= center_h
                <= float(coordinate_values[-1])
            ):
                continue
            azimuth = float(getattr(row, "DipAzimuthDeg", np.nan))
            if not np.isfinite(azimuth):
                legacy_angle = float(getattr(row, "AzimuthDeg", np.nan))
                if not np.isfinite(legacy_angle):
                    continue
                azimuth = orientation.dip_azimuth_from_strike(
                    orientation.strike_from_xy_line_angle(legacy_angle)
                )
            dip = float(getattr(row, "DipDeg", np.nan))
            if not np.isfinite(dip):
                continue
            trace = orientation.trace_on_section(azimuth, dip, projection)
            if trace is None:
                continue
            length = float(np.clip(float(row.LengthM), 10.0, 500.0))
            half = 0.5 * length
            dh, dt = trace
            center_t = float(row.CenterTime)
            segments.append(
                ProjectionSegment(
                    polygon_index=int(getattr(row, "SourceRowID", -1)),
                    projection=projection,
                    interval=interval_by_layer.get(str(row.LayerGroup), ""),
                    center_x=float(row.CenterX),
                    center_y=float(row.CenterY),
                    center_z=center_t,
                    surface_distance=float(
                        getattr(row, "NearestTrajectoryDistance", 0.0)
                    ),
                    patch_area=float(getattr(row, "PatchAreaM2", 1.0)),
                    h1=center_h - half * dh,
                    z1=center_t - half * dt,
                    h2=center_h + half * dh,
                    z2=center_t + half * dt,
                    scale=str(row.FractureScale).strip().lower(),
                )
            )
            if int(getattr(row, "IsDensityCompensationPatch", 0) or 0) == 1:
                category = "density_compensation"
            elif int(getattr(row, "IsImagingMatchPatch", 0) or 0) == 1:
                category = "imaging_match"
            elif int(getattr(row, "IsWellControlPatch", 0) or 0) == 1:
                category = "well_control"
            else:
                category = "base_prediction"
            setattr(segments[-1], "source_category", category)
    context = SectionOverlayContext(
        "local_200m",
        segments,
        [],
        imaging_patches,
        fracture_df,
        imaging_df,
        geometry.summary,
    )
    summary = {
        "dfn_geometry_source": "STEP8_CSV_DIP_AZIMUTH_TRACE",
        "dfn_patch_csv": str(patch_csv),
        "dfn_patch_count": int(len(frame)),
        "dfn_segment_count": int(len(segments)),
        "step3_group_csvs": [str(path) for path in group_paths],
        "step3_fracture_point_count": int(len(fracture_df)),
        "step3_imaging_track_sample_count": int(len(imaging_df)),
        "step3_imaging_patch_segment_count": int(len(imaging_patches)),
        "small_dfn_well_orientation_calibration": orientation_calibration,
        "local_conditioned_patches": {
            "path": str(local_conditioned_patches_path),
            "patch_count": int(len(frame)),
        },
    }
    return context, summary


def make_real_drawer(
    context: SectionOverlayContext,
    conventional_track: pd.DataFrame,
) -> Callable[[Any, str], dict[str, int]]:
    def draw(ax, projection: str) -> dict[str, int]:
        track_qc = draw_well_context(
            ax,
            projection,
            conventional_track,
            context.imaging_df,
        )
        patch_count = add_imaging_fracture_patch_segments(
            ax, context.imaging_patch_segments, projection
        )
        point_count = draw_fracture_labels(
            ax, context.fracture_df, projection, context.display_summary
        )
        return {
            **track_qc,
            "imaging_patch_count": int(patch_count),
            "imaging_point_count": int(point_count),
            "dfn_segment_count": 0,
            "fault_segment_count": 0,
        }

    return draw


def make_dfn_drawer(
    context: SectionOverlayContext,
    conventional_track: pd.DataFrame,
) -> Callable[[Any, str], dict[str, int]]:
    def draw(ax, projection: str) -> dict[str, int]:
        segment_count = add_dfn_segments(
            ax,
            context.dfn_segments,
            projection,
            overlay=True,
            large_section_style=True,
            add_scale_legend=False,
            scale_line_styles=DFN_200M_SCALE_LINE_STYLES,
            scale_color_overrides=shared_compare.DFN_SCALE_COLORS,
            scale_halo_enabled=DFN_200M_LARGE_HALO_ENABLED,
            zorder_base=2.2,
        )
        add_dfn_legend_handles(
            ax,
            scale_legend_styles=DFN_200M_SCALE_LEGEND_STYLES,
        )
        track_qc = draw_well_context(
            ax,
            projection,
            conventional_track,
            context.imaging_df,
        )
        return {
            **track_qc,
            "dfn_segment_count": int(segment_count),
            "imaging_patch_count": 0,
            "imaging_point_count": 0,
            "fault_segment_count": 0,
        }

    return draw


def make_overlay_drawer(
    context: SectionOverlayContext,
    conventional_track: pd.DataFrame,
) -> Callable[[Any, str], dict[str, int]]:
    def draw(ax, projection: str) -> dict[str, int]:
        segment_count = add_dfn_segments(
            ax,
            context.dfn_segments,
            projection,
            overlay=True,
            large_section_style=True,
            add_scale_legend=False,
            scale_line_styles=DFN_200M_SCALE_LINE_STYLES,
            scale_color_overrides=shared_compare.DFN_SCALE_COLORS,
            scale_halo_enabled=DFN_200M_LARGE_HALO_ENABLED,
            zorder_base=2.2,
        )
        add_dfn_legend_handles(
            ax,
            scale_legend_styles=DFN_200M_SCALE_LEGEND_STYLES,
        )
        track_qc = draw_well_context(
            ax,
            projection,
            conventional_track,
            context.imaging_df,
        )
        patch_count = add_imaging_fracture_patch_segments(
            ax, context.imaging_patch_segments, projection
        )
        point_count = draw_fracture_labels(
            ax, context.fracture_df, projection, context.display_summary
        )
        return {
            **track_qc,
            "dfn_segment_count": int(segment_count),
            "imaging_patch_count": int(patch_count),
            "imaging_point_count": int(point_count),
            "fault_segment_count": 0,
        }

    return draw


def add_dfn_legend_handles(
    ax,
    *,
    scale_legend_styles: dict[str, tuple] | None = None,
) -> None:
    for scale_key in ("small", "medium", "large"):
        color, linestyle, width, label = (
            scale_legend_styles or DFN_200M_SCALE_LEGEND_STYLES
        )[scale_key]
        handle = ax.plot(
            [],
            [],
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=label,
        )[0]

def orientation_comparison(
    context: SectionOverlayContext,
    time_scale_m_per_ms: float,
) -> dict[str, Any]:
    def values(segments, projection):
        dips, line_angles = [], []
        for segment in segments:
            if segment.projection != projection:
                continue
            horizontal = abs(float(segment.h2) - float(segment.h1))
            vertical = (
                abs(float(segment.z2) - float(segment.z1))
                * float(time_scale_m_per_ms)
            )
            if horizontal + vertical <= 1.0e-12:
                continue
            dips.append(np.degrees(np.arctan2(vertical, horizontal)))
            line_angles.append(
                np.degrees(
                    np.arctan2(
                        float(segment.z2) - float(segment.z1),
                        float(segment.h2) - float(segment.h1),
                    )
                )
                % 180.0
            )
        return np.asarray(dips), np.asarray(line_angles)

    def overlap(left: np.ndarray, right: np.ndarray) -> float | None:
        if not len(left) or not len(right):
            return None
        bins = np.arange(0.0, 181.0, 10.0)
        left_hist, _ = np.histogram(left, bins=bins)
        right_hist, _ = np.histogram(right, bins=bins)
        left_hist = left_hist / left_hist.sum()
        right_hist = right_hist / right_hist.sum()
        return float(np.minimum(left_hist, right_hist).sum())

    result: dict[str, Any] = {}
    for projection in ("XZ", "YZ"):
        imaging_dip, imaging_line = values(
            context.imaging_patch_segments, projection
        )
        dfn_dip, dfn_line = values(context.dfn_segments, projection)
        result[projection] = {
            "imaging_count": int(len(imaging_dip)),
            "dfn_count": int(len(dfn_dip)),
            "imaging_apparent_dip_median_deg": float(np.median(imaging_dip)),
            "dfn_apparent_dip_median_deg": float(np.median(dfn_dip)),
            "imaging_line_angle_median_deg": float(np.median(imaging_line)),
            "dfn_line_angle_median_deg": float(np.median(dfn_line)),
            "apparent_dip_histogram_overlap": overlap(
                imaging_dip, dfn_dip
            ),
            "line_angle_histogram_overlap": overlap(
                imaging_line, dfn_line
            ),
        }
    return result


def render_projection(
    output_dir: Path,
    kind: str,
    projection: str,
    section: AttributeSection,
    geometry,
    limit: float,
    drawer: Callable[[Any, str], dict[str, int]],
) -> tuple[Path, int, dict[str, int]]:
    if kind == "imaging":
        filename = (
            f"01_local200m_seisamp_wiggle_imaging_only_"
            f"{projection.lower()}_t4_t7.png"
        )
        product_title = "车页1导眼：仅Step3真实成像裂缝"
        title_suffix = " | 仅真实裂缝"
    elif kind == "dfn":
        filename = (
            f"02_local200m_seisamp_wiggle_dfn_only_"
            f"{projection.lower()}_t4_t7.png"
        )
        product_title = "车页1导眼：仅DFN预测裂缝"
        title_suffix = " | 仅DFN裂缝"
    elif kind == "overlay":
        filename = (
            f"03_local200m_fusion_df_and_imaging_"
            f"{projection.lower()}_t4_t7.png"
        )
        product_title = "车页1导眼：井周200 m多尺度DFN与真实成像裂缝"
        title_suffix = " | 多尺度DFN+真实成像"
    else:
        raise ValueError(f"unsupported kind: {kind}")
    output_path = output_dir / filename
    captured: dict[str, int] = {}

    def capture(ax, current_projection: str) -> dict[str, int]:
        captured.update(drawer(ax, current_projection))
        return dict(captured)

    trace_count = plot_wiggle_variable_area(
        section,
        geometry,
        output_path,
        limit,
        title_suffix=title_suffix,
        overlay_drawer=capture,
        scope_label="井周200 m",
        product_title=product_title,
        draw_well=False,
    )
    return output_path, int(trace_count), dict(captured)


def main() -> int:
    args = parser().parse_args()
    config = read_json(args.config.resolve())
    if args.dfn_patch_csv:
        config["dfn_patch_csv"] = str(args.dfn_patch_csv.resolve())
    if args.input_vtk:
        config["input_vtk"] = str(args.input_vtk.resolve())
    if args.local_axis_radius_m is not None:
        config["local_axis_radius_m"] = float(args.local_axis_radius_m)
        config["local_dfn_half_width_m"] = float(args.local_axis_radius_m)
        config["local_small_projection_half_width_m"] = float(args.local_axis_radius_m)
    if args.disable_well_orientation_calibration:
        config["well_orientation_calibration_enabled"] = False
    reference = read_json(args.reference_summary.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)

    overview_config = dict(config)
    overview_config["target_block"] = dict(config.get("overview_target_block") or {})
    overview = prepare_geometry(overview_config)
    local = mb.local_geometry(overview, config)

    imaging_group_paths = [
        Path(str(value)).resolve()
        for value in config.get("step3_imaging_group_csvs", [])
    ]
    imaging_window_df = load_step3_imaging_fracture_labels(
        imaging_group_paths,
        local.well_name,
        dict(config.get("target_block") or {}),
    )
    if args.preserve_local_time_window:
        time_min = float(local.time_min)
        time_max = float(local.time_max)
    else:
        time_min, time_max = tight_imaging_window(
            imaging_window_df["TIME"],
            padding_ms=20.0,
            sample_interval_ms=float(
                config.get("section_sample_interval_ms", 2.0)
            ),
        )
    local.time_min = float(time_min)
    local.time_max = float(time_max)
    local.summary["display_time_min"] = float(time_min)
    local.summary["display_time_max"] = float(time_max)
    local.summary["imaging_time_padding_ms"] = 20.0
    local_geometry_summary = dict(local.summary)

    target = dict(config["presentation_targets"]["local_200m"])
    if args.legend_width_in is not None:
        target["legend_width_in"] = float(args.legend_width_in)
    ppt_geometry = mb.make_ppt_geometry(local, target)
    cache = (
        output_dir
        / "section_samples"
        / "local_200m"
        / "seisamp_section_samples.npz"
    )
    cache_valid = False
    if cache.exists():
        xz, yz, cache_info = mb.load_section_pair_npz(cache, "SeisAmp")
        expected_times = np.arange(
            float(time_min),
            float(time_max) + 1.0e-9,
            float(config.get("section_sample_interval_ms", 2.0)),
        )
        cache_valid = (
            len(xz.h) == len(ppt_geometry.x_values)
            and len(yz.h) == len(ppt_geometry.y_values)
            and np.allclose(xz.h, ppt_geometry.x_values, atol=1.0e-6)
            and np.allclose(yz.h, ppt_geometry.y_values, atol=1.0e-6)
            and len(xz.time) == len(expected_times)
            and np.allclose(xz.time, expected_times, atol=1.0e-6)
        )
    if not cache_valid:
        xz, yz, cache_info = sample_volume_sections(
            ppt_geometry,
            "SeisAmp",
            Path(str(config["volume_paths"]["SeisAmp"])).resolve(),
        )
        save_section_pair_npz(cache, xz, yz)
        cache_info = {
            **cache_info,
            "source": "fresh_local_200m_density_balance_v2",
            "cache_path": str(cache),
        }
    max_traces = int(target.get("max_traces", 0))
    time_decim = int(target.get("time_decim", 1))
    sections = {
        "XZ": set_display_bounds(
            mb.decimate_section(xz, max_traces, time_decim),
            local_geometry_summary,
        ),
        "YZ": set_display_bounds(
            mb.decimate_section(yz, max_traces, time_decim),
            local_geometry_summary,
        ),
    }
    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        base_dir / "base_section_data.npz",
        xz_h=sections["XZ"].h,
        xz_time=sections["XZ"].time,
        xz_values=sections["XZ"].values,
        yz_h=sections["YZ"].h,
        yz_time=sections["YZ"].time,
        yz_values=sections["YZ"].values,
        well_x=local.well_df["X"].to_numpy(dtype=float),
        well_y=local.well_df["Y"].to_numpy(dtype=float),
        well_time=local.well_df["TIME"].to_numpy(dtype=float),
    )
    base_limit = float(reference["display"]["seismic_symmetric_limit"])
    base_images: dict[str, str] = {}
    for projection in ("XZ", "YZ"):
        base_path = base_dir / f"base_{projection.lower()}_imaging_window.png"
        plot_wiggle_variable_area(
            sections[projection],
            ppt_geometry,
            base_path,
            base_limit,
            scope_label="井周200 m成像段",
        )
        base_images[projection] = str(base_path)
    write_json(
        base_dir / "base_manifest.json",
        {
            "status": "pass",
            "data": str(base_dir / "base_section_data.npz"),
            "images": base_images,
            "scope": "local_200m_imaging_window",
            "time_window_ms": [float(time_min), float(time_max)],
            "imaging_time_padding_ms": 20.0,
        },
    )

    if args.dfn_geometry_source == "csv":
        print("[split-sections] 使用 Step8 CSV 产状构建 local_200m DFN", flush=True)
        context, overlay_source = build_csv_orientation_context(
            config, ppt_geometry
        )
    else:
        print("[split-sections] 扫描 local_200m DFN VTK", flush=True)
        context, overlay_source = build_local_context(config, ppt_geometry)
    imaging_rows = segment_rows(
        context.imaging_patch_segments,
        region="砂砾岩",
        source="step3_imaging_truth",
        category="imaging_truth",
    )
    dfn_rows = segment_rows(
        context.dfn_segments,
        region="砂砾岩",
        source="step8_final_dfn",
        category="final_dfn",
        layer_by_interval={
            "T4->T6": "沙三段",
            "T6->T7": "沙四段",
        },
    )
    imaging_rows_df = pd.DataFrame(imaging_rows)
    dfn_rows_df = pd.DataFrame(dfn_rows)
    intersection_manifest = write_intersection_layer(
        output_dir,
        region="砂砾岩",
        imaging_rows=imaging_rows,
        dfn_rows=dfn_rows,
        metadata={
            "dfn_geometry_source": overlay_source["dfn_geometry_source"],
            "time_window_ms": [float(time_min), float(time_max)],
            "imaging_time_padding_ms": 20.0,
            "half_width_m": float(config.get("local_dfn_half_width_m", 200.0)),
            "dfn_length_policy_m": list(
                shared_compare.DFN_GEOMETRY_LENGTH_RANGE_M
            ),
        },
    )
    comparison_statistics = comparison_stats(
        imaging_rows_df,
        dfn_rows_df,
        time_min=float(time_min),
        time_max=float(time_max),
    )
    limit = base_limit
    drawers = {
        "imaging": make_real_drawer(context, ppt_geometry.well_df),
        "dfn": make_dfn_drawer(context, ppt_geometry.well_df),
        "overlay": make_overlay_drawer(context, ppt_geometry.well_df),
    }
    images: list[dict[str, Any]] = []
    kinds = ("overlay",) if args.overlay_only else ("imaging", "dfn", "overlay")
    for kind in kinds:
        for projection in ("XZ", "YZ"):
            path, trace_count, overlay = render_projection(
                output_dir,
                kind,
                projection,
                sections[projection],
                ppt_geometry,
                limit,
                drawers[kind],
            )
            images.append(
                {
                    "kind": kind,
                    "projection": projection,
                    "path": str(path),
                    "section_shape": [int(value) for value in sections[projection].values.shape],
                    "wiggle_trace_count": int(trace_count),
                    "overlay": overlay,
                }
            )
            print(f"[split-sections] {path.name}", flush=True)

    yz_only_path = (
        output_dir
        / "04_local200m_yz_seisamp_wiggle_dfn_only_t4_t7.png"
    )
    yz_only_stats: dict[str, int] = {}

    def capture_yz_only(ax, projection: str) -> dict[str, int]:
        if projection != "YZ":
            raise ValueError(f"unexpected projection for YZ-only figure: {projection}")
        yz_only_stats.update(drawers["dfn"](ax, projection))
        return dict(yz_only_stats)

    yz_only_trace_count = plot_wiggle_variable_area(
        sections["YZ"],
        ppt_geometry,
        yz_only_path,
        limit,
        title_suffix=" | 单独YZ小尺度DFN（成像井段约束产状）",
        overlay_drawer=capture_yz_only,
        scope_label="井周200 m",
        product_title="车页1导眼：井周200 m YZ 小尺度DFN预测",
        draw_well=False,
    )
    print(f"[split-sections] {yz_only_path.name}", flush=True)

    yz_overlay_path = (
        output_dir
        / "05_local200m_yz_seisamp_wiggle_dfn_imaging_overlay_t4_t7.png"
    )
    yz_overlay_stats: dict[str, int] = {}

    def capture_yz_overlay(ax, projection: str) -> dict[str, int]:
        if projection != "YZ":
            raise ValueError(
                f"unexpected projection for YZ overlay figure: {projection}"
            )
        yz_overlay_stats.update(drawers["overlay"](ax, projection))
        return dict(yz_overlay_stats)

    yz_overlay_trace_count = plot_wiggle_variable_area(
        sections["YZ"],
        ppt_geometry,
        yz_overlay_path,
        limit,
        title_suffix=" | 单独YZ小尺度DFN与成像裂缝对比",
        overlay_drawer=capture_yz_overlay,
        scope_label="井周200 m",
        product_title="车页1导眼：井周200 m YZ DFN与成像测井裂缝对比",
        draw_well=False,
    )
    print(f"[split-sections] {yz_overlay_path.name}", flush=True)

    counts = {
        f"{row['kind']}_{row['projection']}": dict(row["overlay"])
        for row in images
    }
    checks = {
        "expected_images_generated": all(Path(row["path"]).exists() for row in images)
        and len(images) == len(kinds) * 2,
        "base_stage_complete": all(
            Path(path).exists() for path in base_images.values()
        )
        and (base_dir / "base_section_data.npz").exists(),
        "intersection_stage_complete": Path(
            intersection_manifest["imaging_intersections"]
        ).exists()
        and Path(intersection_manifest["dfn_intersections"]).exists(),
        "imaging_time_window_padded_20ms": bool(
            float(time_min) <= float(imaging_window_df["TIME"].min()) - 20.0
            and float(time_max) >= float(imaging_window_df["TIME"].max()) + 20.0
        ),
        "local_trace_counts_positive": all(
            row["wiggle_trace_count"] > 0 for row in images
        ),
        "local_trace_counts_match": {
            row["projection"]: row["wiggle_trace_count"] for row in images
        }
        == {
            projection: len(sections[projection].h)
            for projection in ("XZ", "YZ")
        },
        "imaging_only_has_no_dfn": all(
            row["overlay"]["dfn_segment_count"] == 0
            for row in images
            if row["kind"] == "imaging"
        ),
        "dfn_only_has_no_real_fracture": all(
            row["overlay"]["imaging_patch_count"] == 0
            and row["overlay"]["imaging_point_count"] == 0
            for row in images
            if row["kind"] == "dfn"
        ),
        "overlay_has_dfn_and_real_fracture": all(
            row["overlay"]["dfn_segment_count"] > 0
            and row["overlay"]["imaging_point_count"] > 0
            for row in images
            if row["kind"] == "overlay"
        ),
        "real_fractures_loaded": all(
            row["overlay"]["imaging_point_count"]
            == int(overlay_source["step3_fracture_point_count"])
            for row in images
            if row["kind"] == "imaging"
        ),
        "dfn_segments_loaded": all(
            row["overlay"]["dfn_segment_count"] > 0
            for row in images
            if row["kind"] == "dfn"
        ),
        "conventional_and_imaging_tracks_present": all(
            row["overlay"]["conventional_track_samples"] > 0
            and row["overlay"]["imaging_track_samples"] > 0
            for row in images
        ),
        "track_and_fracture_layer_order": bool(
            shared_compare.CONVENTIONAL_TRACK_ZORDER
            < shared_compare.IMAGING_LOG_ZORDER
            < 8.1
        ),
        "scale_width_contract_matches_large_sections": bool(
            all(
                abs(
                    DFN_200M_SCALE_LINE_STYLES[scale][0]
                    - float(shared_compare.DFN_SCALE_WIDTHS[scale])
                )
                <= 1.0e-9
                for scale in ("small", "medium", "large")
            )
        ),
        "scale_color_contract_matches_large_sections": bool(
            all(
                DFN_200M_SCALE_LEGEND_STYLES[scale][0]
                == shared_compare.DFN_SCALE_COLORS[scale]
                for scale in ("small", "medium", "large")
            )
        ),
        "large_scale_halo_disabled": not DFN_200M_LARGE_HALO_ENABLED,
        "small_well_orientation_calibration_complete": bool(
            not overlay_source["small_dfn_well_orientation_calibration"].get(
                "enabled", True
            )
            or (
                overlay_source["small_dfn_well_orientation_calibration"].get(
                    "matched_event_count", -1
                )
                == overlay_source["small_dfn_well_orientation_calibration"].get(
                    "imaging_event_count", -2
                )
            )
        ),
        "no_medium_or_large_added": bool(
            (
                not overlay_source[
                    "small_dfn_well_orientation_calibration"
                ].get("enabled", True)
                or overlay_source[
                    "small_dfn_well_orientation_calibration"
                ].get("scale_counts_unchanged", False)
            )
            and not any(
                scale in {"medium", "large"}
                for scale in overlay_source[
                    "small_dfn_well_orientation_calibration"
                ].get("scale_after", {"small": 1})
            )
        ),
        "single_yz_output_exists": bool(yz_only_path.exists()),
        "single_yz_has_dfn": bool(yz_only_stats.get("dfn_segment_count", 0) > 0),
        "single_yz_overlay_exists": bool(yz_overlay_path.exists()),
        "single_yz_overlay_has_dfn_and_imaging": bool(
            yz_overlay_stats.get("dfn_segment_count", 0) > 0
            and yz_overlay_stats.get("imaging_point_count", 0) > 0
        ),
        "local_conditioned_patch_export_exists": bool(
            Path(overlay_source["local_conditioned_patches"]["path"]).exists()
            and overlay_source["local_conditioned_patches"]["patch_count"] > 0
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config": str(args.config.resolve()),
        "reference_summary": str(args.reference_summary.resolve()),
        "output_dir": str(output_dir),
        "well_name": config.get("well_name", "车页1导眼"),
        "scope": "local_200m",
        "display_contract": {
            "background": "地震波形+正振幅变面积",
            "section_mode": "well_trajectory_curved_surface_t4_t7",
            "projections": ["XZ", "YZ"],
            "time_range_ms": [
                float(time_min),
                float(time_max),
            ],
            "time_window_policy": "imaging_extent_plus_20ms",
            "imaging_raw_time_range_ms": [
                float(imaging_window_df["TIME"].min()),
                float(imaging_window_df["TIME"].max()),
            ],
            "presentation_target": target,
            "seismic_symmetric_limit": float(limit),
            "xz_values_shape": [int(value) for value in sections["XZ"].values.shape],
            "yz_values_shape": [int(value) for value in sections["YZ"].values.shape],
        },
        "local_geometry": {
            "axis_radius_m": float(config.get("local_axis_radius_m", 500.0)),
            "x_range_m": [
                float(local_geometry_summary["display_x_min"]),
                float(local_geometry_summary["display_x_max"]),
            ],
            "y_range_m": [
                float(local_geometry_summary["display_y_min"]),
                float(local_geometry_summary["display_y_max"]),
            ],
        },
        "seismic_cache": {
            "path": str(cache),
            **cache_info,
        },
        "base_layer": {
            "directory": str(base_dir),
            "manifest": str(base_dir / "base_manifest.json"),
        },
        "intersection_layer": intersection_manifest,
        "overlay_source": overlay_source,
        "single_projection_output": {
            "projection": "YZ",
            "path": str(yz_only_path),
            "wiggle_trace_count": int(yz_only_trace_count),
            "overlay": dict(yz_only_stats),
            "content": "seismic_wiggle_background_plus_small_scale_dfn_prediction",
        },
        "single_projection_overlay_output": {
            "projection": "YZ",
            "path": str(yz_overlay_path),
            "wiggle_trace_count": int(yz_overlay_trace_count),
            "overlay": dict(yz_overlay_stats),
            "content": (
                "seismic_wiggle_background_plus_small_scale_dfn_"
                "and_imaging_fractures"
            ),
        },
        "overlay_counts": counts,
        "comparison_statistics": comparison_statistics,
        "dfn_line_style_contract": {
            "source": "shared_200m_latest_large_section_legend",
            "plot_styles": {
                scale: {
                    "linewidth": float(width),
                    "linestyle": linestyle,
                }
                for scale, (width, linestyle) in DFN_200M_SCALE_LINE_STYLES.items()
                if scale
            },
            "legend_styles": {
                scale: {
                    "color": color,
                    "linestyle": linestyle,
                    "linewidth": float(width),
                    "label": label,
                }
                for scale, (color, linestyle, width, label) in DFN_200M_SCALE_LEGEND_STYLES.items()
            },
            "large_scale_halo": {
                "enabled": DFN_200M_LARGE_HALO_ENABLED,
            },
        },
        "well_track_contract": {
            "bottom_to_top": [
                "conventional_well_track",
                "imaging_log_interval",
                "imaging_fractures",
            ],
            "conventional_color": shared_compare.CONVENTIONAL_TRACK_COLOR,
            "imaging_log_color": shared_compare.IMAGING_LOG_COLOR,
            "imaging_fracture_color": shared_compare.IMAGING_COLOR,
        },
        "images": images,
        "checks": checks,
    }
    summary_path = output_dir / "split_sections_summary.json"
    write_json(summary_path, summary)
    print(f"[split-sections] status={summary['status']} summary={summary_path}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
