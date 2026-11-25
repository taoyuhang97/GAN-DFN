#!/usr/bin/env python3
"""Build 200 m DFN versus conventional-log prediction sections for new wells."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from build_cheye1_dfn_coherence_sections import (  # noqa: E402
    build_namespace,
    load_dfn_patch_metadata,
    scan_unified_original_fault_intersections,
    scan_vtk_intersections,
)
from build_cheye1_dfn_multibackground_sections import local_geometry  # noqa: E402
from build_all_area_section_visualization import ProjectionSegment, configure_matplotlib_fonts  # noqa: E402
from common.horizon_trace_table.horizon_contract import build_spatial_lookup  # noqa: E402
from well_curved_section_common import (  # noqa: E402
    AttributeSection,
    prepare_geometry,
    sample_volume_sections,
)


SCALE_LABELS = {"small": "小尺度DFN", "medium": "中尺度DFN", "large": "大尺度DFN"}
LAYER_BY_CODE = {3: "沙三段", 4: "沙四段"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replace-existing-output", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_trajectory(path: Path, well_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"X", "Y", "TIME", "DEPT", "TVD"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trajectory missing columns {missing}: {path}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["WellName"] = well_name
    return (
        frame.dropna(subset=list(required))
        .sort_values("TIME")
        .drop_duplicates("TIME")
        .reset_index(drop=True)
    )


def section_segments_well_intersections(
    segments: list[ProjectionSegment],
    trajectory: pd.DataFrame,
    kind: str,
    md_min: float,
    md_max: float,
) -> pd.DataFrame:
    work = trajectory[trajectory.DEPT.between(md_min, md_max)].sort_values("DEPT").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for segment in segments:
        h_column = "X" if segment.projection == "XZ" else "Y"
        well_h = work[h_column].to_numpy(dtype=float)
        well_z = work.TIME.to_numpy(dtype=float)
        if len(work) < 2:
            continue
        sx = well_h[1:] - well_h[:-1]
        sz = well_z[1:] - well_z[:-1]
        rx = float(segment.h2 - segment.h1)
        rz = float(segment.z2 - segment.z1)
        qx = well_h[:-1] - float(segment.h1)
        qz = well_z[:-1] - float(segment.z1)
        denominator = rx * sz - rz * sx
        valid = np.abs(denominator) > 1.0e-12
        if not valid.any():
            continue
        t = np.full(len(valid), np.nan, dtype=float)
        u = np.full(len(valid), np.nan, dtype=float)
        t[valid] = (qx[valid] * sz[valid] - qz[valid] * sx[valid]) / denominator[valid]
        u[valid] = (qx[valid] * rz - qz[valid] * rx) / denominator[valid]
        hit_indices = np.flatnonzero(
            valid
            & (t >= -1.0e-8)
            & (t <= 1.0 + 1.0e-8)
            & (u >= -1.0e-8)
            & (u <= 1.0 + 1.0e-8)
        )
        for hit_index in hit_indices:
            fraction = float(np.clip(u[hit_index], 0.0, 1.0))
            md = float(
                work.DEPT.iloc[hit_index]
                + fraction * (work.DEPT.iloc[hit_index + 1] - work.DEPT.iloc[hit_index])
            )
            tvd = float(
                work.TVD.iloc[hit_index]
                + fraction * (work.TVD.iloc[hit_index + 1] - work.TVD.iloc[hit_index])
            )
            h_value = float(segment.h1 + t[hit_index] * rx)
            z_value = float(segment.z1 + t[hit_index] * rz)
            layer = "沙三段" if str(segment.interval) == "T4->T6" else "沙四段" if str(segment.interval) == "T6->T7" else ""
            scale = str(getattr(segment, "scale", "")).strip().lower()
            if not scale:
                scale = "original_fault" if kind == "original_fault" else "unknown"
            rows.append(
                {
                    "Kind": kind,
                    "Projection": segment.projection,
                    "PatchIndex": int(segment.polygon_index),
                    "FractureScaleCode": {"small": 1, "medium": 2, "large": 3}.get(scale, -1),
                    "Scale": scale,
                    "LayerCode": 3 if layer == "沙三段" else 4 if layer == "沙四段" else -1,
                    "StrataName": layer,
                    "MD": md,
                    "TVD": tvd,
                    "TIME": z_value,
                    "SectionH": h_value,
                    "DipDeg": np.nan,
                    "DipAzimuthDeg": np.nan,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return (
        result.sort_values(["Kind", "PatchIndex", "MD", "Projection"])
        .drop_duplicates(["Kind", "PatchIndex", "MD", "StrataName"])
        .reset_index(drop=True)
    )


def enrich_intersection_geometry(
    intersections: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    if intersections.empty:
        return intersections
    out = intersections.copy()
    time_values = trajectory.TIME.to_numpy(dtype=float)
    x_values = trajectory.X.to_numpy(dtype=float)
    y_values = trajectory.Y.to_numpy(dtype=float)
    out["X"] = np.interp(out.TIME.to_numpy(dtype=float), time_values, x_values)
    out["Y"] = np.interp(out.TIME.to_numpy(dtype=float), time_values, y_values)
    return out


def match_points_to_intersections(
    prediction: pd.DataFrame,
    intersections: pd.DataFrame,
    tolerance_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    positive = prediction.copy()
    if "PredHasFracture" in positive.columns:
        positive = positive[positive.PredHasFracture.astype(int).eq(1)].copy()
    positive["NearestDFNDistanceM"] = np.nan
    positive["MatchedDFN"] = 0
    if not intersections.empty:
        for strata_name, hit_group in intersections.groupby("StrataName"):
            pred_index = positive.index[positive.StrataName.astype(str).eq(str(strata_name))]
            if len(pred_index) == 0 or hit_group.empty:
                continue
            distance_values = np.abs(
                positive.loc[pred_index, "DEPT"].to_numpy(dtype=float)[:, None]
                - hit_group["MD"].to_numpy(dtype=float)[None, :]
            )
            nearest = distance_values.min(axis=1)
            positive.loc[pred_index, "NearestDFNDistanceM"] = nearest
            positive.loc[pred_index, "MatchedDFN"] = (nearest <= float(tolerance_m)).astype(int)
    hits = intersections.copy()
    if hits.empty:
        hits["NearestPredictionM"] = pd.Series(dtype=float)
        hits["MatchedPrediction"] = pd.Series(dtype=int)
    else:
        nearest_prediction: list[float] = []
        matched_prediction: list[int] = []
        for _, hit in hits.iterrows():
            candidates = positive[
                positive.StrataName.astype(str).eq(str(hit.StrataName))
            ]
            if candidates.empty:
                nearest_prediction.append(np.inf)
                matched_prediction.append(0)
                continue
            distances = np.abs(candidates.DEPT.to_numpy(dtype=float) - float(hit.MD))
            nearest = float(np.min(distances))
            nearest_prediction.append(nearest)
            matched_prediction.append(int(nearest <= float(tolerance_m)))
        hits["NearestPredictionM"] = nearest_prediction
        hits["MatchedPrediction"] = matched_prediction
    summary = {
        "tolerance_m": float(tolerance_m),
        "prediction_points": int(len(positive)),
        "matched_prediction_points": int(positive.MatchedDFN.sum()),
        "matched_prediction_fraction": float(positive.MatchedDFN.mean()) if len(positive) else 0.0,
        "dfn_intersections": int(len(hits)),
        "dfn_intersections_with_prediction": int(hits.MatchedPrediction.sum()) if len(hits) else 0,
        "dfn_intersections_without_prediction": int((1 - hits.MatchedPrediction).sum()) if len(hits) else 0,
    }
    return positive, hits, summary


def circular_mean_azimuth(values: np.ndarray, weights: np.ndarray) -> float | None:
    if len(values) == 0 or np.sum(weights) <= 0.0:
        return None
    theta = np.radians(np.asarray(values, dtype=float) * 2.0)
    x = float(np.sum(weights * np.cos(theta)))
    y = float(np.sum(weights * np.sin(theta)))
    if abs(x) + abs(y) <= 1.0e-12:
        return None
    return float((math.degrees(math.atan2(y, x)) / 2.0) % 180.0)


def assign_point_orientations(
    points: pd.DataFrame,
    hits: pd.DataFrame,
    metadata: pd.DataFrame | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = points.copy()
    for column in (
        "DipDeg",
        "DipAzimuthDeg",
        "OrientationSource",
        "OrientationConfidence",
        "NearestPatchDistanceM",
        "NeighborPatchCount",
    ):
        if column not in out.columns:
            out[column] = np.nan if column not in {"OrientationSource", "OrientationConfidence"} else ""
    if out.empty:
        return out, {}

    predicted_hits = hits[hits.Kind.eq("predicted_dfn")].copy() if not hits.empty else pd.DataFrame()
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 2.0))
    neighbor_count = int(config.get("orientation_neighbor_count", 24))
    max_distance = float(config.get("orientation_max_distance_m", 60.0))
    distance_scale = max(float(config.get("orientation_distance_scale_m", 30.0)), 1.0e-9)
    match_tolerance = float(config["match_tolerance_m"])

    trees: dict[str, dict[str, Any]] = {}
    if metadata is not None:
        layer_column = "LayerGroup" if "LayerGroup" in metadata.columns else "StrataName"
        azimuth_candidates = [
            column
            for column in ("DipAzimuthDeg", "AzimuthDeg", "FracAzimuth")
            if column in metadata.columns
        ]
        azimuth_column = (
            max(
                azimuth_candidates,
                key=lambda column: int(
                    pd.to_numeric(metadata[column], errors="coerce").notna().sum()
                ),
            )
            if azimuth_candidates
            else None
        )
        dip_column = next(
            (column for column in ("DipDeg", "FracDip") if column in metadata.columns),
            None,
        )
        if azimuth_column and dip_column:
            for layer_name, group in metadata.groupby(layer_column):
                center_columns = ["CenterX", "CenterY", "CenterTime"]
                if not set(center_columns).issubset(group.columns):
                    continue
                coordinates = group[center_columns].apply(pd.to_numeric, errors="coerce")
                coordinates["CenterTime"] *= time_scale
                valid = coordinates.notna().all(axis=1)
                if int(valid.sum()) < 2:
                    continue
                trees[str(layer_name)] = {
                    "tree": cKDTree(coordinates.loc[valid].to_numpy(dtype=float)),
                    "metadata": group.loc[valid].reset_index(drop=True),
                    "azimuth_column": azimuth_column,
                    "dip_column": dip_column,
                }

    source_counts: dict[str, int] = {}
    for index, point in out.iterrows():
        layer_name = str(point.StrataName)
        nearest_hit = None
        if not predicted_hits.empty:
            candidates = predicted_hits[predicted_hits.StrataName.astype(str).eq(layer_name)]
            if not candidates.empty:
                distances = np.abs(candidates.MD.to_numpy(dtype=float) - float(point.DEPT))
                nearest = float(np.min(distances))
                if nearest <= match_tolerance:
                    nearest_hit = candidates.iloc[int(np.argmin(distances))]
        if (
            nearest_hit is not None
            and pd.notna(nearest_hit.DipDeg)
            and pd.notna(nearest_hit.DipAzimuthDeg)
        ):
            out.loc[index, "DipDeg"] = float(nearest_hit.DipDeg)
            out.loc[index, "DipAzimuthDeg"] = float(nearest_hit.DipAzimuthDeg)
            out.loc[index, "OrientationSource"] = "DFN_exact"
            out.loc[index, "OrientationConfidence"] = "high"
            out.loc[index, "NearestPatchDistanceM"] = 0.0
            out.loc[index, "NeighborPatchCount"] = 1
            source_counts["DFN_exact"] = source_counts.get("DFN_exact", 0) + 1
            continue

        lookup = trees.get(layer_name)
        if lookup is None:
            out.loc[index, "OrientationSource"] = "none"
            source_counts["none"] = source_counts.get("none", 0) + 1
            continue
        query = np.asarray(
            [float(point.X), float(point.Y), float(point.TIME) * time_scale],
            dtype=float,
        )
        k = min(neighbor_count, len(lookup["metadata"]))
        distances, neighbour_index = lookup["tree"].query(query, k=k)
        distances = np.atleast_1d(distances).astype(float)
        neighbour_index = np.atleast_1d(neighbour_index).astype(int)
        within = distances <= max_distance
        if not within.any():
            out.loc[index, "OrientationSource"] = "none"
            source_counts["none"] = source_counts.get("none", 0) + 1
            continue
        distances = distances[within]
        neighbours = lookup["metadata"].iloc[neighbour_index[within]]
        weights = np.exp(-distances / distance_scale)
        azimuths = pd.to_numeric(neighbours[lookup["azimuth_column"]], errors="coerce").to_numpy(dtype=float)
        dips = pd.to_numeric(neighbours[lookup["dip_column"]], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(azimuths) & np.isfinite(dips)
        if not valid.any():
            out.loc[index, "OrientationSource"] = "none"
            source_counts["none"] = source_counts.get("none", 0) + 1
            continue
        azimuth = circular_mean_azimuth(azimuths[valid], weights[valid])
        dip = float(np.average(dips[valid], weights=weights[valid]))
        nearest_distance = float(distances[valid].min())
        confidence = (
            "high"
            if nearest_distance <= 5.0 and int(valid.sum()) >= 5
            else "medium"
            if nearest_distance <= 30.0
            else "low"
        )
        out.loc[index, "DipDeg"] = dip
        out.loc[index, "DipAzimuthDeg"] = azimuth
        out.loc[index, "OrientationSource"] = "DFN_local_neighbor"
        out.loc[index, "OrientationConfidence"] = confidence
        out.loc[index, "NearestPatchDistanceM"] = nearest_distance
        out.loc[index, "NeighborPatchCount"] = int(valid.sum())
        source_counts["DFN_local_neighbor"] = source_counts.get("DFN_local_neighbor", 0) + 1
    return out, {
        "orientation_source_counts": source_counts,
        "oriented_points": int((out.DipDeg.notna() & out.DipAzimuthDeg.notna()).sum()),
        "unoriented_points": int(out.DipDeg.isna().sum()),
    }


def plot_compare(
    well_name: str,
    projection: str,
    section: AttributeSection,
    geometry,
    variants: list[dict[str, Any]],
    colors: dict[str, str],
    output_path: Path,
    figure_size: tuple[float, float],
    dpi: int,
    time_scale_m_per_ms: float,
    segment_half_length_m: float,
    mode: str,
    group_title: str,
) -> None:
    h_values = section.h
    time_values = section.time
    h_min, h_max = float(h_values.min()), float(h_values.max())
    if mode not in {"dfn_only", "well_log_only", "overlay"}:
        raise ValueError(f"unsupported section plot mode: {mode}")
    panel_width = float(figure_size[0]) / len(variants) if len(variants) > 1 else float(figure_size[0]) * 0.55
    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(panel_width * len(variants), float(figure_size[1])),
        sharey=True,
        constrained_layout=True,
    )
    if len(variants) == 1:
        axes = np.asarray([axes])
    values = np.asarray(section.values, dtype=float)
    finite = values[np.isfinite(values)]
    limit = float(np.nanquantile(np.abs(finite), 0.98)) if finite.size else 1.0
    limit = max(limit, 1.0)
    spacing = float(np.median(np.diff(h_values))) if len(h_values) > 1 else 1.0
    lateral_scale = 0.42 * spacing / limit
    for ax, variant in zip(axes, variants):
        for trace_index, baseline in enumerate(h_values):
            trace = np.nan_to_num(values[:, trace_index], nan=0.0, posinf=0.0, neginf=0.0)
            trace = np.clip(trace, -limit, limit)
            displaced = float(baseline) + trace * lateral_scale
            ax.plot(displaced, time_values, color="#111827", lw=0.35, alpha=0.72, zorder=1)
            ax.fill_betweenx(
                time_values,
                float(baseline),
                displaced,
                where=trace >= 0.0,
                color="#111827",
                alpha=0.62,
                linewidth=0.0,
                zorder=1,
            )
        for curve in geometry.surface_curves:
            if curve.projection != projection:
                continue
            ax.plot(curve.h, curve.z, color="#455A64", lw=0.75, alpha=0.8)
        well_h = geometry.well_df["X" if projection == "XZ" else "Y"].to_numpy(dtype=float)
        well_t = geometry.well_df["TIME"].to_numpy(dtype=float)
        ax.plot(well_h, well_t, color=colors["well"], lw=2.0, zorder=8)
        if mode in {"dfn_only", "overlay"}:
            for segment in variant["scan_segments"]:
                if segment.projection != projection:
                    continue
                color = colors.get(str(segment.scale).lower(), "#546E7A")
                ax.plot([segment.h1, segment.h2], [segment.z1, segment.z2], color=color, lw=1.25, alpha=0.9, zorder=5)
            for segment in variant["fault_segments"]:
                if segment.projection != projection:
                    continue
                ax.plot(
                    [segment.h1, segment.h2],
                    [segment.z1, segment.z2],
                    color=colors["original_fault"],
                    lw=1.4,
                    ls="--",
                    alpha=0.95,
                    zorder=5.5,
                )
        version_points = variant["positive_points"]
        if mode in {"well_log_only", "overlay"}:
            pred_h = version_points["X" if projection == "XZ" else "Y"].to_numpy(dtype=float)
            pred_t = version_points["TIME"].to_numpy(dtype=float)
            ax.scatter(
                pred_h,
                pred_t,
                s=9,
                color=colors["prediction"],
                alpha=0.85,
                linewidths=0,
                zorder=9,
                label="常规测井细化裂缝点",
            )
            oriented = version_points[
                version_points.DipDeg.notna() & version_points.DipAzimuthDeg.notna()
            ]
            for row in oriented.itertuples(index=False):
                azimuth = math.radians(float(row.DipAzimuthDeg))
                dip = math.radians(float(row.DipDeg))
                apparent_tangent = (
                    math.tan(dip) * math.sin(azimuth)
                    if projection == "XZ"
                    else math.tan(dip) * math.cos(azimuth)
                )
                delta_time = (
                    float(segment_half_length_m)
                    * apparent_tangent
                    / max(float(time_scale_m_per_ms), 1.0e-9)
                )
                center_h = float(row.X if projection == "XZ" else row.Y)
                center_t = float(row.TIME)
                ax.plot(
                    [center_h - float(segment_half_length_m), center_h + float(segment_half_length_m)],
                    [center_t - delta_time, center_t + delta_time],
                    color=colors["prediction"],
                    lw=1.0,
                    alpha=0.9,
                    zorder=9.1,
                )
        if mode == "overlay":
            matches = version_points[version_points.MatchedDFN.eq(1)]
            if not matches.empty:
                ax.scatter(
                    matches["X" if projection == "XZ" else "Y"],
                    matches["TIME"],
                    s=34,
                    facecolors="none",
                    edgecolors=colors["match"],
                    linewidths=1.0,
                    zorder=10,
                    label="1 m内匹配",
                )
        hits = variant["intersections"]
        if mode in {"dfn_only", "overlay"} and not hits.empty:
            visible_hits = hits[
                hits.Kind.eq("predicted_dfn")
                & hits.TIME.between(float(time_values.min()), float(time_values.max()), inclusive="both")
            ]
            ax.scatter(
                visible_hits["X" if projection == "XZ" else "Y"],
                visible_hits["TIME"],
                marker="D",
                s=30,
                color=[
                    colors.get(str(value).lower(), "#546E7A")
                    for value in visible_hits["Scale"]
                ],
                edgecolors="black",
                linewidths=0.5,
                zorder=11,
                label="DFN真实穿井点",
            )
            original_hits = hits[
                hits.Kind.eq("original_fault")
                & hits.TIME.between(float(time_values.min()), float(time_values.max()), inclusive="both")
            ]
            if not original_hits.empty:
                ax.scatter(
                    original_hits["X" if projection == "XZ" else "Y"],
                    original_hits["TIME"],
                    marker="x",
                    s=34,
                    color=colors["original_fault"],
                    linewidths=1.0,
                    zorder=11,
                    label="原始断层穿井点",
                )
        counts = variant["version_counts"]
        if mode == "dfn_only":
            ax.set_title(
                f"{variant['label']} | DFN穿井 {counts['dfn_intersections']}"
            )
        elif mode == "well_log_only":
            ax.set_title(f"常规测井细化裂缝 | {counts['prediction_points']} 个点")
        else:
            ax.set_title(
                f"{variant['label']} | DFN穿井 {counts['dfn_intersections']} | "
                f"细化点 {counts['prediction_points']} | 匹配 {counts['matched_prediction_points']}"
            )
        ax.set_xlim(h_min, h_max)
        ax.set_ylim(float(time_values.max()) + 3.0, float(time_values.min()) - 3.0)
        ax.set_xlabel("X / m" if projection == "XZ" else "Y / m")
        ax.text(
            0.01,
            0.01,
            "背景：地震波形 + 正振幅变面积",
            transform=ax.transAxes,
            fontsize=7,
            color="#37474F",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            zorder=20,
        )
        ax.set_facecolor("white")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("TWT / ms")
    handles = [Line2D([0], [0], color=colors["well"], lw=2.0, label="井轨迹")]
    if mode in {"dfn_only", "overlay"}:
        handles.extend(
            [
                Line2D([0], [0], marker="D", color="none", markerfacecolor="#777777", markeredgecolor="black", markersize=5, label="DFN真实穿井点"),
                Line2D([0], [0], color=colors["small"], lw=1.4, label="小尺度DFN剖面线"),
                Line2D([0], [0], color=colors["medium"], lw=1.4, label="中尺度DFN剖面线"),
                Line2D([0], [0], color=colors["large"], lw=1.4, label="大尺度DFN剖面线"),
                Line2D([0], [0], color=colors["original_fault"], lw=1.4, ls="--", label="原始断层"),
            ]
        )
    if mode in {"well_log_only", "overlay"}:
        handles.extend(
            [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["prediction"], markersize=4, label="常规测井细化裂缝点"),
                Line2D([0], [0], color=colors["prediction"], lw=1.0, label="局部DFN产状补全"),
            ]
        )
    if mode == "overlay":
        handles.append(
            Line2D([0], [0], marker="o", color="none", markeredgecolor=colors["match"], markerfacecolor="none", markersize=7, label="1 m内匹配")
        )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.suptitle(f"{well_name} 井周200 m {projection} 剖面：{group_title}", y=1.09, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.replace_existing_output:
            raise RuntimeError(f"output exists; pass --replace-existing-output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_fonts()

    prediction_points = pd.read_csv(config["prediction_points_csv"], encoding="utf-8-sig")
    required_point_columns = {
        "WellName", "SampleID", "StrataName", "DEPT", "TVD", "TIME", "X", "Y",
        "PredFractureProb", "Density",
    }
    missing_point_columns = sorted(required_point_columns - set(prediction_points.columns))
    if missing_point_columns:
        raise ValueError(f"prediction point table missing columns: {missing_point_columns}")

    surfaces = build_spatial_lookup(config)
    version_specs = list(config["dfn_versions"])
    loaded_versions: list[dict[str, Any]] = []
    for spec in version_specs:
        vtk_path = Path(spec["input_vtk"]).resolve()
        metadata = load_dfn_patch_metadata(Path(spec["dfn_patch_csv"]).resolve())
        loaded_versions.append({**spec, "vtk_path": vtk_path, "metadata": metadata})

    all_hits: list[pd.DataFrame] = []
    all_match_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for well_cfg in config["wells"]:
        well_name = str(well_cfg["well_name"])
        trajectory_path = Path(well_cfg["trajectory_csv"]).resolve()
        trajectory = read_trajectory(trajectory_path, well_name)
        well_prediction = prediction_points[prediction_points.WellName.astype(str).eq(well_name)].copy()
        well_prediction["MatchedDFN"] = 0
        well_prediction["NearestDFNDistanceM"] = np.nan
        if well_prediction.empty:
            raise RuntimeError(f"no refined prediction points for {well_name}")

        base_config = dict(config)
        base_config.update(
            {
                "well_name": well_name,
                "well_trajectory_csv": str(trajectory_path),
            }
        )
        base = prepare_geometry(base_config)
        local = local_geometry(base, base_config)
        time_min = max(float(local.time_min), float(well_prediction.TIME.min()) - float(config.get("time_padding_ms", 20.0)))
        time_max = min(float(local.time_max), float(well_prediction.TIME.max()) + float(config.get("time_padding_ms", 20.0)))
        if time_max <= time_min:
            raise RuntimeError(f"invalid prediction time window for {well_name}")
        local = replace(local, time_min=time_min, time_max=time_max)
        xz, yz, section_qc = sample_volume_sections(
            local,
            "SeisAmp",
            Path(config["seismic_volume"]).resolve(),
        )

        display = {
            "display_x_min": float(local.x_values.min()),
            "display_x_max": float(local.x_values.max()),
            "display_y_min": float(local.y_values.min()),
            "display_y_max": float(local.y_values.max()),
        }
        plot_variants: list[dict[str, Any]] = []
        for version in loaded_versions:
            version_config = dict(base_config)
            version_config.update(
                {
                    "input_vtk": str(version["vtk_path"]),
                    "dfn_patch_csv": str(Path(version["dfn_patch_csv"]).resolve()),
                    "output_dir": str(output_dir),
                    "real_well_samples_root": str(Path(config["real_well_samples_root"]).resolve()),
                }
            )
            namespace = build_namespace(version_config, float(config["dfn_half_width_m"]))
            namespace.well_trajectory_csv = trajectory_path
            namespace.selected_well_name = well_name
            namespace.dfn_patch_metadata = version["metadata"]
            namespace.small_projection_enabled = False
            scan_segments, scan_summary = scan_vtk_intersections(namespace, surfaces, local.well_df)
            fault_segments, fault_summary = scan_unified_original_fault_intersections(
                version["vtk_path"], local.well_df, display
            )
            predicted_hits = section_segments_well_intersections(
                scan_segments,
                trajectory,
                kind="predicted_dfn",
                md_min=float(well_prediction.DEPT.min()),
                md_max=float(well_prediction.DEPT.max()),
            )
            original_hits = section_segments_well_intersections(
                fault_segments,
                trajectory,
                kind="original_fault",
                md_min=float(well_prediction.DEPT.min()),
                md_max=float(well_prediction.DEPT.max()),
            )
            hits = pd.concat([predicted_hits, original_hits], ignore_index=True)
            if not hits.empty:
                hits = enrich_intersection_geometry(hits, trajectory)
                hits.insert(0, "DFNVersion", version["name"])
                hits.insert(1, "WellName", well_name)
                for index, row in hits[hits.Kind.eq("predicted_dfn")].iterrows():
                    patch_index = int(row.PatchIndex)
                    metadata = version["metadata"]
                    if metadata is not None and 0 <= patch_index < len(metadata):
                        record = metadata.iloc[patch_index]
                        if "DipDeg" in metadata.columns:
                            hits.loc[index, "DipDeg"] = pd.to_numeric(
                                pd.Series([record.get("DipDeg")]), errors="coerce"
                            ).iloc[0]
                        for azimuth_column in ("DipAzimuthDeg", "AzimuthDeg", "FracAzimuth"):
                            if azimuth_column not in metadata.columns:
                                continue
                            azimuth_value = pd.to_numeric(
                                pd.Series([record.get(azimuth_column)]),
                                errors="coerce",
                            ).iloc[0]
                            if pd.notna(azimuth_value):
                                hits.loc[index, "DipAzimuthDeg"] = float(azimuth_value)
                                break
                hits = hits.sort_values(["TIME", "Scale", "PatchIndex"]).reset_index(drop=True)
            predicted_index = hits.index[hits.Kind.eq("predicted_dfn")] if not hits.empty else pd.Index([])
            original_index = hits.index[hits.Kind.eq("original_fault")] if not hits.empty else pd.Index([])
            positive, matched_hits, match_summary = match_points_to_intersections(
                well_prediction,
                hits.loc[predicted_index].copy() if len(predicted_index) else pd.DataFrame(),
                float(config["match_tolerance_m"]),
            )
            if not hits.empty:
                hits["NearestPredictionM"] = np.nan
                hits["MatchedPrediction"] = 0
                for source_index, row in matched_hits.iterrows():
                    target_index = predicted_index[int(source_index)]
                    hits.loc[target_index, "NearestPredictionM"] = row.NearestPredictionM
                    hits.loc[target_index, "MatchedPrediction"] = int(row.MatchedPrediction)
            positive, orientation_summary = assign_point_orientations(
                positive,
                hits,
                version["metadata"],
                config,
            )
            match_summary.update(orientation_summary)
            match_table = positive[
                ["SampleID", "WellName", "StrataName", "DEPT", "TVD", "TIME", "X", "Y",
                 "PredFractureProb", "Density", "NearestDFNDistanceM", "MatchedDFN",
                 "DipDeg", "DipAzimuthDeg", "OrientationSource", "OrientationConfidence",
                 "NearestPatchDistanceM", "NeighborPatchCount"]
            ].copy()
            match_table.insert(0, "DFNVersion", version["name"])
            all_match_rows.append(match_table)
            if not hits.empty:
                all_hits.append(hits)
            plot_variants.append(
                {
                    "name": version["name"],
                    "label": version["label"],
                    "scan_segments": scan_segments,
                    "fault_segments": fault_segments,
                    "intersections": hits,
                    "positive_points": positive,
                    "version_counts": match_summary,
                    "scan_summary": scan_summary,
                    "fault_summary": fault_summary,
                }
            )
            summary_rows.append(
                {
                    "WellName": well_name,
                    "DFNVersion": version["name"],
                    **match_summary,
                    "section_scan_segments": int(len(scan_segments)),
                    "original_fault_section_segments": int(len(fault_segments)),
                    "original_fault_well_intersections": int(len(original_index)),
                }
            )
        section_groups = (
            ("01_DFN_only", "仅DFN", "dfn_only", plot_variants),
            ("02_well_log_only", "仅测井裂缝", "well_log_only", plot_variants[:1]),
            ("03_overlay_compare", "叠加对比", "overlay", plot_variants),
        )
        for group_key, group_title, mode, group_variants in section_groups:
            group_output_dir = output_dir / well_name / group_key
            for projection, section in (("XZ", xz), ("YZ", yz)):
                plot_compare(
                    well_name,
                    projection,
                    section,
                    local,
                    group_variants,
                    dict(config["colors"]),
                    group_output_dir / f"{well_name}_{projection}_200m_{mode}.png",
                    (float(config["figure_width"]), float(config["figure_height"])),
                    int(config["dpi"]),
                    float(config.get("orientation_time_scale_m_per_ms", 2.0)),
                    float(config.get("prediction_segment_half_length_m", 12.5)),
                    mode,
                    group_title,
                )

    hits_all = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame(
        columns=[
            "DFNVersion", "WellName", "Kind", "PatchIndex", "FractureScaleCode", "Scale",
            "LayerCode", "StrataName", "Projection", "MD", "TVD", "TIME", "SectionH", "X", "Y", "DipDeg",
            "DipAzimuthDeg", "NearestPredictionM", "MatchedPrediction",
        ]
    )
    hits_all.to_csv(output_dir / "dfn_well_intersections.csv", index=False, encoding="utf-8-sig")
    matches_all = pd.concat(all_match_rows, ignore_index=True) if all_match_rows else pd.DataFrame()
    matches_all.to_csv(output_dir / "prediction_points_dfn_match.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_dir / "comparison_manifest.json",
        {
            "status": "pass",
            "version": config["version"],
            "prediction_points_csv": str(Path(config["prediction_points_csv"]).resolve()),
            "dfn_versions": [
                {
                    "name": item["name"],
                    "label": item["label"],
                    "vtk": str(item["vtk_path"]),
                    "patch_csv": str(Path(item["dfn_patch_csv"]).resolve()),
                }
                for item in loaded_versions
            ],
            "plot_count": len(config["wells"]) * 2 * 3,
            "match_tolerance_m": float(config["match_tolerance_m"]),
            "section_groups": {
                "01_DFN_only": ["XZ", "YZ"],
                "02_well_log_only": ["XZ", "YZ"],
                "03_overlay_compare": ["XZ", "YZ"],
            },
            "layout": "Each well has three group directories; every group has XZ and YZ sections.",
            "notes": [
                "Only refined conventional-log fracture points are plotted; positive samples are not shown.",
                "Point orientations are estimated from a crossing local DFN patch or a nearby same-layer DFN orientation field.",
                "DFN section lines and true well intersections are separate layers.",
                "Step7D and Step8 are shown side by side.",
                "Original faults are shown separately from predicted DFN patches.",
                "DFN-only omits conventional-log points; well-log-only omits DFN and original-fault traces; overlay contains both.",
            ],
        },
    )
    print(json.dumps({"status": "pass", "output_dir": str(output_dir), "summaries": summary_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
