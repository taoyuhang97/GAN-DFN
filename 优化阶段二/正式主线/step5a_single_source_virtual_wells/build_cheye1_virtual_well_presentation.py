# -*- coding: utf-8 -*-
"""Render the Step5 virtual-well construction evidence around Cheye-1 pilot well.

This is deliberately independent from the DFN pipeline.  It only visualizes
the Step3 observed imaging fractures, Step4 conventional-log predictions, and
the Step5 weak-label transfer events on the same local seismic geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

CURRENT_DIR = Path(__file__).resolve().parent
SECTION_DIR = CURRENT_DIR.parent / "step9_section_visualize"
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))

from build_all_area_section_visualization import configure_matplotlib_fonts
from build_cheye1_dfn_coherence_sections import (
    aligned_local_axis,
    build_imaging_fracture_patch_segments,
    finite_bounds_from_curves,
    load_step3_imaging_fracture_labels,
    load_step3_imaging_segment_track,
)
from build_well_attribute_section_visualization import axis_edges
from trace_horizon_section import build_trace_horizon_section_curves, resolve_horizon_trace_table
from well_curved_section_common import (
    AttributeSection,
    CurvedSectionGeometry,
    draw_surface_curves,
    draw_well_trajectory,
    prepare_geometry,
    read_json,
    sample_volume_sections,
    write_json,
)

WELL_COLOR = "#FFD400"
IMAGING_COLOR = "#00D9FF"
PREDICTED_COLOR = "#EC4899"
VIRTUAL_COLOR = "#9B5DE5"
CENTER_VIRTUAL_COLOR = "#F97316"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Step5 Cheye1 virtual-well presentation figures.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def local_geometry(base: CurvedSectionGeometry, config: dict[str, Any]) -> CurvedSectionGeometry:
    radius = float(config.get("local_axis_radius_m", 200.0))
    x_values = aligned_local_axis(base.trace_df, "X", float(base.well_df["X"].min()) - radius, float(base.well_df["X"].max()) + radius)
    y_values = aligned_local_axis(base.trace_df, "Y", float(base.well_df["Y"].min()) - radius, float(base.well_df["Y"].max()) + radius)
    # Keep the full-resolution geometry available to the sampler, but use a
    # stable, presentation-sized lateral grid for the rendered PNGs.
    def decimate(values: np.ndarray, key: str) -> np.ndarray:
        limit = int(config.get(key, 0))
        if limit <= 1 or len(values) <= limit:
            return values
        return values[np.unique(np.linspace(0, len(values) - 1, limit, dtype=int))]

    x_values = decimate(x_values, "ppt_x_sample_count")
    y_values = decimate(y_values, "ppt_y_sample_count")
    curves, horizon_summary = build_trace_horizon_section_curves(
        resolve_horizon_trace_table(config), base.well_df, base.trace_tree, base.trace_ids, x_values, y_values,
        iteration_count=int(config.get("horizon_curve_iteration_count", 12)),
    )
    time_min, time_max = finite_bounds_from_curves(curves, float(config.get("time_padding_ms", 20.0)))
    summary = dict(base.summary)
    summary.update({
        "scope_name": "well_local_200m_ppt", "display_scope": "车页1导眼轨迹外扩200m后取最近地震道，并按PPT道数抽稀",
        "display_x_min": float(x_values.min()), "display_x_max": float(x_values.max()),
        "display_y_min": float(y_values.min()), "display_y_max": float(y_values.max()),
        "display_time_min": float(time_min), "display_time_max": float(time_max),
        "section_x_sample_count": int(len(x_values)), "section_y_sample_count": int(len(y_values)),
        "ppt_target_x_sample_count": int(config.get("ppt_x_sample_count", 0)),
        "ppt_target_y_sample_count": int(config.get("ppt_y_sample_count", 0)),
        "horizon_display": horizon_summary,
    })
    args = SimpleNamespace(**vars(base.args))
    args.fig_width = float(config.get("fig_width", 15.0))
    args.fig_height = float(config.get("fig_height", 10.0))
    args.dpi = int(config.get("render_dpi", 300))
    return replace(base, config=dict(config), args=args, surface_curves=curves, summary=summary, x_values=x_values, y_values=y_values, time_min=time_min, time_max=time_max)


def positive_events(frame: pd.DataFrame, *, group_cols: list[str], gap_ms: float) -> pd.DataFrame:
    """Collapse contiguous positive sample points to one honest position-only event."""
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "X", "Y", "TIME", "StrataName", "SampleCount"])
    work = frame.copy()
    work["TIME"] = pd.to_numeric(work["TIME"], errors="coerce")
    work["X"] = pd.to_numeric(work["X"], errors="coerce")
    work["Y"] = pd.to_numeric(work["Y"], errors="coerce")
    work = work.dropna(subset=["X", "Y", "TIME"]).sort_values([*group_cols, "TIME"])
    if work.empty:
        return pd.DataFrame(columns=[*group_cols, "X", "Y", "TIME", "StrataName", "SampleCount"])
    delta = work.groupby(group_cols, dropna=False)["TIME"].diff()
    work["_event"] = (delta.isna() | delta.gt(gap_ms)).groupby([work[column] for column in group_cols], dropna=False).cumsum()
    grouped = work.groupby([*group_cols, "_event"], dropna=False, as_index=False)
    return grouped.agg(X=("X", "median"), Y=("Y", "median"), TIME=("TIME", "median"), StrataName=("StrataName", "first"), SampleCount=("TIME", "size"))


def read_step4_events(config: dict[str, Any], imaging_track: pd.DataFrame) -> pd.DataFrame:
    path = Path(str(config["step4_prediction_csv"])).resolve()
    parts: list[pd.DataFrame] = []
    imaging_ranges = [(float(group.TIME.min()), float(group.TIME.max())) for _, group in imaging_track.groupby("Step3GroupCSV") if not group.empty]
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=["WellName", "X", "Y", "TIME", "StrataName", "HasFracture", "PredictionValid"], chunksize=int(config.get("step4_csv_chunksize", 250000))):
        selected = chunk[(chunk["WellName"].astype(str) == str(config["well_name"])) & (pd.to_numeric(chunk["HasFracture"], errors="coerce") == 1) & (pd.to_numeric(chunk["PredictionValid"], errors="coerce") == 1)].copy()
        if imaging_ranges and not selected.empty:
            time = pd.to_numeric(selected["TIME"], errors="coerce")
            covered = np.zeros(len(selected), dtype=bool)
            for low, high in imaging_ranges:
                covered |= time.between(low, high).to_numpy()
            selected = selected.loc[~covered]
        if not selected.empty:
            parts.append(selected)
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return positive_events(raw, group_cols=["WellName", "StrataName"], gap_ms=float(config.get("event_gap_ms", 1.0)))


def read_virtual_events(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(str(config["virtual_well_training_csv"])).resolve()
    well_name = str(config["well_name"])
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=["SourceWellName", "VirtualWellName", "X", "Y", "TIME", "StrataName", "PresenceLabel"], chunksize=int(config.get("virtual_csv_chunksize", 250000))):
        selected = chunk[(chunk["SourceWellName"].astype(str) == well_name) & (pd.to_numeric(chunk["PresenceLabel"], errors="coerce") == 1)].copy()
        if not selected.empty:
            parts.append(selected)
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    events = positive_events(raw, group_cols=["VirtualWellName", "StrataName"], gap_ms=float(config.get("event_gap_ms", 1.0)))
    return raw, events


def draw_common(ax: Any, geometry: CurvedSectionGeometry, section: AttributeSection) -> None:
    draw_surface_curves(ax, geometry.surface_curves, section.projection)
    values = geometry.well_df["X"] if section.projection == "XZ" else geometry.well_df["Y"]
    draw_well_trajectory(ax, values, geometry.well_df["TIME"], label="车页1导眼真实井轨迹")
    ax.set_xlim(float(section.h.min()), float(section.h.max()))
    ax.set_ylim(float(geometry.summary["display_time_min"]), float(geometry.summary["display_time_max"]))
    ax.invert_yaxis()
    ax.set_xlabel("X / m" if section.projection == "XZ" else "Y / m")
    ax.set_ylabel(str(geometry.args.z_label))
    ax.grid(True, linewidth=0.25, alpha=0.22)


def draw_imaging(ax: Any, segments: list[Any], projection: str) -> int:
    selected = [item for item in segments if item.projection == projection]
    if not selected:
        return 0
    lines = [((item.h1, item.z1), (item.h2, item.z2)) for item in selected]
    ax.add_collection(LineCollection(lines, colors="#0F172A", linewidths=4.2, alpha=0.65, zorder=9))
    ax.add_collection(LineCollection(lines, colors=IMAGING_COLOR, linewidths=2.3, alpha=0.96, zorder=10, label="成像测井真实裂缝姿态投影"))
    return len(selected)


def draw_events(ax: Any, events: pd.DataFrame, projection: str, color: str, marker: str, label: str, *, size: float, alpha: float = 0.85, center_name: str | None = None) -> int:
    if events.empty:
        return 0
    h = "X" if projection == "XZ" else "Y"
    ax.scatter(events[h], events["TIME"], s=size, marker=marker, c=color, edgecolors="#111827", linewidths=0.42, alpha=alpha, zorder=11, label=label)
    if center_name:
        center = events[events["VirtualWellName"].astype(str) == center_name]
        if not center.empty:
            ax.scatter(center[h], center["TIME"], s=size * 1.8, marker=marker, c=CENTER_VIRTUAL_COLOR, edgecolors="#111827", linewidths=0.55, alpha=0.95, zorder=12, label="中心虚拟井裂缝位置")
    return int(len(events))


def plot_section(section: AttributeSection, geometry: CurvedSectionGeometry, output_path: Path, *, background: str, imaging_segments: list[Any], step4_events: pd.DataFrame, virtual_events: pd.DataFrame, mode: str, limit: float | None = None) -> dict[str, int]:
    fig, ax = plt.subplots(figsize=(geometry.args.fig_width, geometry.args.fig_height))
    if background == "amplitude":
        value_limit = limit if limit else 1.0
        ax.pcolormesh(axis_edges(section.h), axis_edges(section.time), np.abs(section.values), shading="auto", cmap="gray_r", vmin=0.0, vmax=value_limit, rasterized=True, zorder=1)
        background_label = "地震振幅绝对值"
    else:
        finite = section.values[np.isfinite(section.values)]
        value_limit = float(np.quantile(np.abs(finite), 0.98)) if len(finite) else 1.0
        ax.pcolormesh(axis_edges(section.h), axis_edges(section.time), np.abs(section.values), shading="auto", cmap="gray_r", vmin=0.0, vmax=value_limit, rasterized=True, zorder=1)
        background_label = "曲率体绝对值"
    draw_common(ax, geometry, section)
    counts = {"imaging_orientation_segments": 0, "step4_position_events": 0, "virtual_position_events": 0}
    if mode in {"evidence", "combined"}:
        counts["imaging_orientation_segments"] = draw_imaging(ax, imaging_segments, section.projection)
        counts["step4_position_events"] = draw_events(ax, step4_events, section.projection, PREDICTED_COLOR, "o", "常规测井预测裂缝位置（无姿态）", size=28)
    if mode in {"virtual", "combined"}:
        counts["virtual_position_events"] = draw_events(ax, virtual_events, section.projection, VIRTUAL_COLOR, "D", "Step5虚拟测井裂缝位置（无姿态）", size=18, alpha=0.56, center_name="车页1导眼_VW_2_2")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8.2, framealpha=0.9)
    title = {"evidence": "真实井裂缝证据", "virtual": "Step5虚拟测井弱监督样本", "combined": "真实井与虚拟测井综合核对"}[mode]
    ax.set_title(f"车页1导眼 Step5 虚拟测井构造 | {title} | {background_label} | {section.projection}", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=geometry.args.dpi)
    plt.close(fig)
    return counts


def plot_plan(geometry: CurvedSectionGeometry, index_df: pd.DataFrame, output_path: Path) -> dict[str, int]:
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    ax.plot(geometry.well_df["X"], geometry.well_df["Y"], color="#111827", linewidth=5.0, zorder=3)
    ax.plot(geometry.well_df["X"], geometry.well_df["Y"], color=WELL_COLOR, linewidth=2.8, zorder=4, label="车页1导眼真实井轨迹")
    center_count = 0
    for _, row in index_df.iterrows():
        is_center = int(pd.to_numeric(row["IsCenterVirtualTrace"], errors="coerce")) == 1
        color = CENTER_VIRTUAL_COLOR if is_center else VIRTUAL_COLOR
        size = 80 if is_center else 44
        label = "中心虚拟井地震道" if is_center else None
        ax.scatter(row["VirtualAnchorX"], row["VirtualAnchorY"], marker="D", s=size, c=color, edgecolors="#111827", linewidths=0.55, zorder=5, label=label)
        center_count += int(is_center)
    ax.set_aspect("equal", adjustable="box")
    # The plan-view figure is a local diagnostic, not a mine-scale location
    # map.  Restrict it to the same 200 m display window as XZ/YZ.
    ax.set_xlim(float(geometry.summary["display_x_min"]), float(geometry.summary["display_x_max"]))
    ax.set_ylim(float(geometry.summary["display_y_min"]), float(geometry.summary["display_y_max"]))
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("车页1导眼 Step5 虚拟测井 5×5 地震道邻域")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=geometry.args.dpi)
    plt.close(fig)
    return {"virtual_well_count": int(len(index_df)), "center_virtual_well_count": center_count}


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    configure_matplotlib_fonts()
    base = prepare_geometry(config)
    geometry = local_geometry(base, config)
    output_dir = Path(str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_df = pd.read_csv(Path(str(config["virtual_well_index_csv"])).resolve(), encoding="utf-8-sig")
    index_df = index_df[index_df["SourceWellName"].astype(str) == str(config["well_name"])].copy()
    groups = [Path(str(item)).resolve() for item in config["step3_imaging_group_csvs"]]
    imaging_points = load_step3_imaging_fracture_labels(groups, str(config["well_name"]), dict(config.get("target_block") or {}))
    imaging_track = load_step3_imaging_segment_track(groups, str(config["well_name"]), dict(config.get("target_block") or {}))
    imaging_segments = build_imaging_fracture_patch_segments(imaging_points)
    step4_events = read_step4_events(config, imaging_track)
    _virtual_raw, virtual_events = read_virtual_events(config)
    print(f"[step5-presentation] virtual_wells={len(index_df)} step4_events={len(step4_events)} virtual_events={len(virtual_events)}", flush=True)
    amp_xz, amp_yz, _ = sample_volume_sections(geometry, "SeisAmp", Path(str(config["volume_paths"]["SeisAmp"])).resolve())
    curv_xz, curv_yz, _ = sample_volume_sections(geometry, "CurvatureMax", Path(str(config["volume_paths"]["CurvatureMax"])).resolve())
    finite_amp = np.concatenate([item.values[np.isfinite(item.values)] for item in (amp_xz, amp_yz) if np.isfinite(item.values).any()])
    amp_limit = float(np.quantile(np.abs(finite_amp), float(config.get("amplitude_clip_quantile", 0.99)))) if len(finite_amp) else 1.0
    outputs: dict[str, Any] = {}
    outputs["01_plan_virtual_well_grid.png"] = plot_plan(geometry, index_df, output_dir / "01_plan_virtual_well_grid.png")
    for number, section, background, mode, stem in [
        (2, amp_xz, "amplitude", "evidence", "02_xz_real_well_evidence_amplitude.png"),
        (3, amp_yz, "amplitude", "evidence", "03_yz_real_well_evidence_amplitude.png"),
        (4, curv_xz, "curvature", "virtual", "04_xz_virtual_wells_curvature.png"),
        (5, curv_yz, "curvature", "virtual", "05_yz_virtual_wells_curvature.png"),
        (6, curv_xz, "curvature", "combined", "06_xz_combined_curvature.png"),
        (7, curv_yz, "curvature", "combined", "07_yz_combined_curvature.png"),
    ]:
        outputs[stem] = plot_section(section, geometry, output_dir / stem, background=background, imaging_segments=imaging_segments, step4_events=step4_events, virtual_events=virtual_events, mode=mode, limit=amp_limit if background == "amplitude" else None)
    checks = {"exactly_25_cheye1_virtual_wells": len(index_df) == 25, "imaging_orientation_available": len(imaging_segments) > 0, "virtual_events_available": len(virtual_events) > 0, "all_pngs_written": len(list(output_dir.glob("*.png"))) == 7}
    summary = {"status": "pass" if all(checks.values()) else "fail", "config_path": str(args.config.resolve()), "output_dir": str(output_dir), "section_geometry": geometry.summary, "counts": {"step3_imaging_points": int(len(imaging_points)), "step3_orientation_projection_segments": int(len(imaging_segments)), "step4_position_events_outside_imaging": int(len(step4_events)), "step5_virtual_position_events": int(len(virtual_events))}, "files": sorted(path.name for path in output_dir.glob("*.png")), "checks": checks, "interpretation": {"yellow_line": "车页1导眼真实井轨迹", "cyan_segments": "成像测井真实裂缝姿态投影", "magenta_circles": "常规测井预测裂缝位置，无倾向倾角", "purple_diamonds": "Step5虚拟测井裂缝位置，无倾向倾角"}}
    write_json(output_dir / "virtual_well_presentation_summary.json", summary)
    print(f"[step5-presentation] output={output_dir}", flush=True)
    print(f"[step5-presentation] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
