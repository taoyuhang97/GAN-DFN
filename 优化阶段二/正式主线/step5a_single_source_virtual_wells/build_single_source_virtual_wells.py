from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from sklearn.neighbors import KDTree


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
STEP2_SCRIPT_DIR = REPO_ROOT / "优化阶段二/正式主线/step2_real_well_t4_t7_samples"
if str(STEP2_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_SCRIPT_DIR))

from build_real_well_t4_t7_samples import (  # noqa: E402
    GRID_SPACING,
    NEIGHBOR_OFFSETS,
    ATTRIBUTE_VALUE_RULES,
    build_trace_grid,
    build_trace_tree,
    offset_to_axis_label,
    sample_trace_at_time,
)


ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]
STAT_SUFFIXES = ["Mean", "Std", "Min", "Max", "ValidCount"]
MIN_VALID_ATTRIBUTE_COUNT = 3


@dataclass(frozen=True)
class VolumeContext:
    name: str
    handle: Any
    samples: np.ndarray
    trace_at: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build formal Step 5 single-source virtual wells.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--max-source-wells", type=int, default=0, help="Optional smoke-test cap; 0 means all source wells.")
    parser.add_argument("--max-points-per-well", type=int, default=0, help="Optional smoke-test cap per source well; 0 means all points.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N virtual wells.")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def open_volume_context(name: str, volume_path: Path) -> VolumeContext:
    handle = segyio.open(str(volume_path), "r", ignore_geometry=True)
    handle.mmap()
    samples = np.asarray(handle.samples, dtype=np.float64)
    trace_cache: dict[int, np.ndarray] = {}

    def trace_at(trace_idx: int) -> np.ndarray:
        key = int(trace_idx)
        if key not in trace_cache:
            trace_cache[key] = np.asarray(handle.trace[key], dtype=np.float32)
        return trace_cache[key]

    return VolumeContext(name=name, handle=handle, samples=samples, trace_at=trace_at)


def close_contexts(contexts: list[VolumeContext]) -> None:
    for ctx in contexts:
        ctx.handle.close()


def clean_attribute_value(name: str, value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    rule = ATTRIBUTE_VALUE_RULES.get(name)
    if rule and (value < rule["min"] or value > rule["max"]):
        return np.nan
    return float(value)


def query_trace_neighbors(
    x: float,
    y: float,
    trace_tree: KDTree,
    trace_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    distances, idx = trace_tree.query([[x, y]], k=4)
    return distances[0], trace_ids[idx[0]]


def sample_neighbor_query_at_time(
    neighbor_query: tuple[np.ndarray, np.ndarray] | None,
    time_ms: float,
    samples: np.ndarray,
    trace_at: Any,
) -> float:
    if neighbor_query is None:
        return np.nan
    distances, nearest_trace_ids = neighbor_query
    amplitudes: list[float] = []
    weights: list[float] = []
    for dist, trace_idx in zip(distances, nearest_trace_ids):
        trace_data = trace_at(int(trace_idx))
        amp = sample_trace_at_time(trace_data, samples, time_ms)
        if not np.isfinite(amp):
            continue
        if dist == 0:
            return float(amp)
        amplitudes.append(float(amp))
        weights.append(1.0 / float(dist))
    if not amplitudes:
        return np.nan
    weight_arr = np.asarray(weights, dtype=float)
    weight_arr = weight_arr / weight_arr.sum()
    return float(np.dot(np.asarray(amplitudes, dtype=float), weight_arr))


def sample_trace_at_times(trace_data: np.ndarray, samples: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    times = np.asarray(time_ms, dtype=np.float64)
    out = np.full(times.shape, np.nan, dtype=np.float64)
    valid_time = np.isfinite(times)
    if not valid_time.any():
        return out
    sample_positions = np.interp(
        times[valid_time],
        samples,
        np.arange(len(samples), dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    valid_pos = np.isfinite(sample_positions)
    if not valid_pos.any():
        return out
    valid_indices = np.flatnonzero(valid_time)[valid_pos]
    i0 = np.floor(sample_positions[valid_pos]).astype(np.int64)
    inside = (i0 >= 0) & (i0 + 1 < len(trace_data))
    if not inside.any():
        return out
    target_indices = valid_indices[inside]
    i0 = i0[inside]
    weight = sample_positions[valid_pos][inside] - i0
    out[target_indices] = (1.0 - weight) * trace_data[i0] + weight * trace_data[i0 + 1]
    return out


def query_trace_neighbors_batch(
    x: np.ndarray,
    y: np.ndarray,
    trace_tree: KDTree,
    trace_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    distances = np.full((len(x_arr), 4), np.nan, dtype=np.float64)
    neighbor_trace_ids = np.full((len(x_arr), 4), -1, dtype=np.int64)
    valid_xy = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid_xy.any():
        query_xy = np.column_stack([x_arr[valid_xy], y_arr[valid_xy]]).astype(np.float32)
        dist_part, idx_part = trace_tree.query(query_xy, k=4)
        distances[valid_xy] = dist_part
        neighbor_trace_ids[valid_xy] = trace_ids[idx_part]
    return distances, neighbor_trace_ids


def batch_sample_neighbor_query_at_time(
    neighbor_query: tuple[np.ndarray, np.ndarray],
    time_ms: np.ndarray,
    samples: np.ndarray,
    trace_at: Any,
) -> np.ndarray:
    distances, neighbor_trace_ids = neighbor_query
    times = np.asarray(time_ms, dtype=np.float64)
    exact_values = np.full(times.shape, np.nan, dtype=np.float64)
    weighted_sum = np.zeros(times.shape, dtype=np.float64)
    weight_sum = np.zeros(times.shape, dtype=np.float64)

    for col in range(distances.shape[1]):
        trace_col = neighbor_trace_ids[:, col]
        dist_col = distances[:, col]
        amp_col = np.full(times.shape, np.nan, dtype=np.float64)
        for trace_idx in np.unique(trace_col[trace_col >= 0]):
            mask = trace_col == trace_idx
            amp_col[mask] = sample_trace_at_times(trace_at(int(trace_idx)), samples, times[mask])

        exact_mask = (dist_col == 0) & np.isfinite(amp_col) & ~np.isfinite(exact_values)
        exact_values[exact_mask] = amp_col[exact_mask]

        weighted_mask = (dist_col > 0) & np.isfinite(dist_col) & np.isfinite(amp_col) & ~np.isfinite(exact_values)
        weights = np.zeros(times.shape, dtype=np.float64)
        weights[weighted_mask] = 1.0 / dist_col[weighted_mask]
        weighted_sum[weighted_mask] += amp_col[weighted_mask] * weights[weighted_mask]
        weight_sum[weighted_mask] += weights[weighted_mask]

    out = np.full(times.shape, np.nan, dtype=np.float64)
    exact_mask = np.isfinite(exact_values)
    out[exact_mask] = exact_values[exact_mask]
    weighted_mask = ~exact_mask & (weight_sum > 0)
    out[weighted_mask] = weighted_sum[weighted_mask] / weight_sum[weighted_mask]
    return out


def clean_attribute_array(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = np.nan
    rule = ATTRIBUTE_VALUE_RULES.get(name)
    if rule:
        arr[(arr < rule["min"]) | (arr > rule["max"])] = np.nan
    return arr


def stack_stats(values_by_offset: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stack = np.vstack(values_by_offset).astype(np.float64)
    valid = np.isfinite(stack)
    valid_count = valid.sum(axis=0).astype(np.int64)
    sums = np.where(valid, stack, 0.0).sum(axis=0)
    mean = np.divide(sums, valid_count, out=np.full(valid_count.shape, np.nan, dtype=np.float64), where=valid_count > 0)
    diff = np.where(valid, stack - mean, 0.0)
    std = np.sqrt(np.divide((diff * diff).sum(axis=0), valid_count, out=np.full(valid_count.shape, np.nan, dtype=np.float64), where=valid_count > 0))
    min_vals = np.where(valid, stack, np.inf).min(axis=0)
    max_vals = np.where(valid, stack, -np.inf).max(axis=0)
    min_vals[valid_count == 0] = np.nan
    max_vals[valid_count == 0] = np.nan
    return mean, std, min_vals, max_vals, valid_count


def sample_attributes_for_points(
    x: np.ndarray,
    y: np.ndarray,
    time_ms: np.ndarray,
    trace_tree: KDTree,
    trace_ids: np.ndarray,
    contexts: list[VolumeContext],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    neighbor_queries: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for dx in NEIGHBOR_OFFSETS:
        for dy in NEIGHBOR_OFFSETS:
            neighbor_queries[(float(dx), float(dy))] = query_trace_neighbors_batch(
                x=np.asarray(x, dtype=np.float64) + float(dx),
                y=np.asarray(y, dtype=np.float64) + float(dy),
                trace_tree=trace_tree,
                trace_ids=trace_ids,
            )

    center_cols: dict[str, np.ndarray] = {}
    context_cols: dict[str, np.ndarray] = {}
    stat_cols: dict[str, np.ndarray] = {}
    for ctx in contexts:
        values_by_offset: list[np.ndarray] = []
        for dx in NEIGHBOR_OFFSETS:
            for dy in NEIGHBOR_OFFSETS:
                values = batch_sample_neighbor_query_at_time(
                    neighbor_query=neighbor_queries[(float(dx), float(dy))],
                    time_ms=np.asarray(time_ms, dtype=np.float64),
                    samples=ctx.samples,
                    trace_at=ctx.trace_at,
                )
                values = clean_attribute_array(ctx.name, values)
                context_cols[f"{ctx.name}_{offset_to_axis_label(dx, 'x')}_{offset_to_axis_label(dy, 'y')}"] = values
                values_by_offset.append(values)
                if float(dx) == 0.0 and float(dy) == 0.0:
                    center_cols[ctx.name] = values
        mean, std, min_vals, max_vals, valid_count = stack_stats(values_by_offset)
        stat_cols[f"{ctx.name}Mean"] = mean
        stat_cols[f"{ctx.name}Std"] = std
        stat_cols[f"{ctx.name}Min"] = min_vals
        stat_cols[f"{ctx.name}Max"] = max_vals
        stat_cols[f"{ctx.name}ValidCount"] = valid_count
    return center_cols, context_cols, stat_cols


def build_trace_axes(trace_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_coords = np.sort(pd.to_numeric(trace_df["X"], errors="coerce").dropna().unique())
    y_coords = np.sort(pd.to_numeric(trace_df["Y"], errors="coerce").dropna().unique())
    return x_coords, y_coords


def nearest_axis_value(coords: np.ndarray, value: float) -> float:
    idx = int(np.searchsorted(coords, value))
    if idx <= 0:
        return float(coords[0])
    if idx >= len(coords):
        return float(coords[-1])
    return float(coords[idx - 1] if abs(value - coords[idx - 1]) <= abs(value - coords[idx]) else coords[idx])


def build_virtual_index(
    prediction_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    window_size: int,
    selected_wells: list[str] | None,
    max_source_wells: int,
) -> pd.DataFrame:
    x_coords, y_coords = build_trace_axes(trace_df)
    wells = sorted(prediction_df["WellName"].dropna().unique().tolist())
    if selected_wells:
        selected = set(selected_wells)
        wells = [well for well in wells if well in selected]
    if max_source_wells > 0:
        wells = wells[:max_source_wells]
    half = int(window_size) // 2
    rows: list[dict[str, Any]] = []
    for source_well in wells:
        well_df = prediction_df[prediction_df["WellName"] == source_well].copy()
        if well_df.empty:
            continue
        center_x = float(safe_numeric(well_df["X"]).median())
        center_y = float(safe_numeric(well_df["Y"]).median())
        snapped_center_x = nearest_axis_value(x_coords, center_x)
        snapped_center_y = nearest_axis_value(y_coords, center_y)
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                offset_x = dx * GRID_SPACING
                offset_y = dy * GRID_SPACING
                virtual_anchor_x = nearest_axis_value(x_coords, snapped_center_x + offset_x)
                virtual_anchor_y = nearest_axis_value(y_coords, snapped_center_y + offset_y)
                actual_offset_x = virtual_anchor_x - center_x
                actual_offset_y = virtual_anchor_y - center_y
                distance = float(math.sqrt(actual_offset_x**2 + actual_offset_y**2))
                rows.append(
                    {
                        "SourceWellName": source_well,
                        "VirtualWellName": f"{source_well}_VW_{dx + half}_{dy + half}",
                        "GridOffsetX": dx,
                        "GridOffsetY": dy,
                        "SourceAnchorX": center_x,
                        "SourceAnchorY": center_y,
                        "VirtualAnchorX": virtual_anchor_x,
                        "VirtualAnchorY": virtual_anchor_y,
                        "OffsetX": actual_offset_x,
                        "OffsetY": actual_offset_y,
                        "DistanceToSource": distance,
                        "IsSourceTrace": int(dx == 0 and dy == 0),
                    }
                )
    return pd.DataFrame(rows)


def sample_attributes_for_point(
    x: float,
    y: float,
    time_ms: float,
    trace_tree: KDTree,
    trace_ids: np.ndarray,
    contexts: list[VolumeContext],
) -> tuple[dict[str, float], dict[str, float]]:
    center: dict[str, float] = {}
    stats: dict[str, float] = {}
    neighbor_queries: dict[tuple[float, float], tuple[np.ndarray, np.ndarray] | None] = {}
    for dx in NEIGHBOR_OFFSETS:
        for dy in NEIGHBOR_OFFSETS:
            neighbor_queries[(float(dx), float(dy))] = query_trace_neighbors(
                x=x + float(dx),
                y=y + float(dy),
                trace_tree=trace_tree,
                trace_ids=trace_ids,
            )
    for ctx in contexts:
        center_val = sample_neighbor_query_at_time(
            neighbor_query=neighbor_queries[(0.0, 0.0)],
            time_ms=time_ms,
            samples=ctx.samples,
            trace_at=ctx.trace_at,
        )
        center_val = clean_attribute_value(ctx.name, center_val)
        center[ctx.name] = center_val
        vals: list[float] = []
        for dx in NEIGHBOR_OFFSETS:
            for dy in NEIGHBOR_OFFSETS:
                attr_val = sample_neighbor_query_at_time(
                    neighbor_query=neighbor_queries[(float(dx), float(dy))],
                    time_ms=time_ms,
                    samples=ctx.samples,
                    trace_at=ctx.trace_at,
                )
                attr_val = clean_attribute_value(ctx.name, attr_val)
                center[f"{ctx.name}_{offset_to_axis_label(dx, 'x')}_{offset_to_axis_label(dy, 'y')}"] = attr_val
                if np.isfinite(attr_val):
                    vals.append(float(attr_val))
        arr = np.asarray(vals, dtype=float)
        stats[f"{ctx.name}Mean"] = float(arr.mean()) if len(arr) else np.nan
        stats[f"{ctx.name}Std"] = float(arr.std(ddof=0)) if len(arr) else np.nan
        stats[f"{ctx.name}Min"] = float(arr.min()) if len(arr) else np.nan
        stats[f"{ctx.name}Max"] = float(arr.max()) if len(arr) else np.nan
        stats[f"{ctx.name}ValidCount"] = int(len(arr))
    return center, stats


def normalized_similarity(source_val: float, virtual_val: float) -> float:
    if not np.isfinite(source_val) or not np.isfinite(virtual_val):
        return np.nan
    denom = max(abs(float(source_val)), abs(float(virtual_val)), 1.0)
    return float(np.clip(1.0 - abs(float(source_val) - float(virtual_val)) / denom, 0.0, 1.0))


def attribute_continuity(source_row: pd.Series, virtual_center: dict[str, float]) -> tuple[float, int]:
    scores: list[float] = []
    for attr in ATTRIBUTE_COLUMNS:
        if attr not in source_row or attr not in virtual_center:
            continue
        score = normalized_similarity(float(source_row[attr]), float(virtual_center[attr]))
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return 0.0, 0
    return float(np.mean(scores)), int(len(scores))


def build_virtual_samples(
    prediction_df: pd.DataFrame,
    index_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    volume_paths: dict[str, str],
    max_points_per_well: int,
    min_continuity_for_density: float,
    confidence_distance_scale: float,
    min_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace_tree, trace_ids = build_trace_tree(trace_df)
    contexts = [open_volume_context(name, Path(path)) for name, path in volume_paths.items()]
    attr_rows: list[dict[str, Any]] = []
    density_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []

    try:
        for item in index_df.to_dict(orient="records"):
            source_well = item["SourceWellName"]
            well_df = prediction_df[prediction_df["WellName"] == source_well].copy()
            if max_points_per_well > 0 and len(well_df) > max_points_per_well:
                idx = np.linspace(0, len(well_df) - 1, max_points_per_well, dtype=int)
                well_df = well_df.iloc[idx].copy()
            for row in well_df.itertuples(index=False):
                source_row = pd.Series(row._asdict())
                virtual_x = float(source_row["X"]) + float(item["OffsetX"])
                virtual_y = float(source_row["Y"]) + float(item["OffsetY"])
                time_ms = float(source_row["TIME"])
                center, stats = sample_attributes_for_point(
                    x=virtual_x,
                    y=virtual_y,
                    time_ms=time_ms,
                    trace_tree=trace_tree,
                    trace_ids=trace_ids,
                    contexts=contexts,
                )
                continuity, valid_count = attribute_continuity(source_row, center)
                source_density = float(source_row["Density"]) if np.isfinite(float(source_row["Density"])) else np.nan
                if np.isfinite(source_density) and valid_count >= MIN_VALID_ATTRIBUTE_COUNT and continuity >= min_continuity_for_density:
                    density = max(source_density * continuity, 0.0)
                    density_status = "single_source_attribute_continuity"
                else:
                    density = 0.0
                    density_status = "attribute_continuity_below_threshold_or_invalid_source"
                distance_weight = math.exp(-float(item["DistanceToSource"]) / max(confidence_distance_scale, 1e-6))
                point_confidence = float(np.clip(max(min_confidence, distance_weight) * max(continuity, 0.0), 0.0, 1.0))
                sample_id = f"{item['VirtualWellName']}_{source_row['SampleID']}"
                common = {
                    "SampleID": sample_id,
                    "SourceSampleID": source_row["SampleID"],
                    "SourceWellName": source_well,
                    "VirtualWellName": item["VirtualWellName"],
                    "X": virtual_x,
                    "Y": virtual_y,
                    "TIME": time_ms,
                    "TVD": source_row.get("TVD", np.nan),
                    "DEPT": source_row.get("DEPT", np.nan),
                    "StrataName": source_row.get("StrataName", pd.NA),
                    "DistanceToSource": float(item["DistanceToSource"]),
                    "AttributeContinuity": continuity,
                    "ValidContinuityAttributeCount": valid_count,
                }
                attr_rows.append({**common, **{k: v for k, v in center.items() if k in ATTRIBUTE_COLUMNS}, **stats})
                context_rows.append({**common, **center})
                density_rows.append(
                    {
                        **common,
                        "SourceDensity": source_density,
                        "Density": density,
                        "HasFracture": int(density > 0.0),
                        "DensitySourceLogic": density_status,
                    }
                )
                confidence_rows.append(
                    {
                        **common,
                        "DistanceConfidenceWeight": distance_weight,
                        "PointConfidence": point_confidence,
                        "ConfidenceLogic": "distance_controls_confidence_only_attribute_continuity_controls_density",
                    }
                )
    finally:
        close_contexts(contexts)

    return (
        pd.DataFrame(attr_rows),
        pd.DataFrame(context_rows),
        pd.DataFrame(density_rows),
        pd.DataFrame(confidence_rows),
    )


def build_virtual_samples_for_item(
    prediction_df: pd.DataFrame,
    item: dict[str, Any],
    trace_tree: KDTree,
    trace_ids: np.ndarray,
    contexts: list[VolumeContext],
    max_points_per_well: int,
    min_continuity_for_density: float,
    confidence_distance_scale: float,
    min_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_well = item["SourceWellName"]
    well_df = prediction_df[prediction_df["WellName"] == source_well].copy()
    if max_points_per_well > 0 and len(well_df) > max_points_per_well:
        idx = np.linspace(0, len(well_df) - 1, max_points_per_well, dtype=int)
        well_df = well_df.iloc[idx].copy()
    if well_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    virtual_x = safe_numeric(well_df["X"]).to_numpy(dtype=np.float64) + float(item["OffsetX"])
    virtual_y = safe_numeric(well_df["Y"]).to_numpy(dtype=np.float64) + float(item["OffsetY"])
    time_ms = safe_numeric(well_df["TIME"]).to_numpy(dtype=np.float64)
    center_cols, context_cols, stat_cols = sample_attributes_for_points(
        x=virtual_x,
        y=virtual_y,
        time_ms=time_ms,
        trace_tree=trace_tree,
        trace_ids=trace_ids,
        contexts=contexts,
    )

    score_sum = np.zeros(len(well_df), dtype=np.float64)
    score_count = np.zeros(len(well_df), dtype=np.int64)
    for attr in ATTRIBUTE_COLUMNS:
        if attr not in well_df.columns or attr not in center_cols:
            continue
        source_vals = safe_numeric(well_df[attr]).to_numpy(dtype=np.float64)
        virtual_vals = center_cols[attr]
        valid = np.isfinite(source_vals) & np.isfinite(virtual_vals)
        if not valid.any():
            continue
        denom = np.maximum.reduce([np.abs(source_vals[valid]), np.abs(virtual_vals[valid]), np.ones(int(valid.sum()), dtype=np.float64)])
        scores = np.clip(1.0 - np.abs(source_vals[valid] - virtual_vals[valid]) / denom, 0.0, 1.0)
        score_sum[valid] += scores
        score_count[valid] += 1
    continuity = np.divide(score_sum, score_count, out=np.zeros(len(well_df), dtype=np.float64), where=score_count > 0)

    source_density = safe_numeric(well_df["Density"]).to_numpy(dtype=np.float64)
    density_mask = (
        np.isfinite(source_density)
        & (score_count >= MIN_VALID_ATTRIBUTE_COUNT)
        & (continuity >= float(min_continuity_for_density))
    )
    density = np.where(density_mask, np.maximum(source_density * continuity, 0.0), 0.0)
    density_status = np.where(
        density_mask,
        "single_source_attribute_continuity",
        "attribute_continuity_below_threshold_or_invalid_source",
    )
    distance_weight = math.exp(-float(item["DistanceToSource"]) / max(confidence_distance_scale, 1e-6))
    point_confidence = np.clip(max(min_confidence, distance_weight) * np.maximum(continuity, 0.0), 0.0, 1.0)
    sample_ids = [f"{item['VirtualWellName']}_{sample_id}" for sample_id in well_df["SampleID"].astype(str).tolist()]

    common = {
        "SampleID": sample_ids,
        "SourceSampleID": well_df["SampleID"].to_numpy(),
        "SourceWellName": np.full(len(well_df), source_well, dtype=object),
        "VirtualWellName": np.full(len(well_df), item["VirtualWellName"], dtype=object),
        "X": virtual_x,
        "Y": virtual_y,
        "TIME": time_ms,
        "TVD": safe_numeric(well_df["TVD"]).to_numpy(dtype=np.float64) if "TVD" in well_df.columns else np.full(len(well_df), np.nan),
        "DEPT": safe_numeric(well_df["DEPT"]).to_numpy(dtype=np.float64) if "DEPT" in well_df.columns else np.full(len(well_df), np.nan),
        "StrataName": well_df["StrataName"].to_numpy() if "StrataName" in well_df.columns else np.full(len(well_df), pd.NA, dtype=object),
        "DistanceToSource": np.full(len(well_df), float(item["DistanceToSource"]), dtype=np.float64),
        "AttributeContinuity": continuity,
        "ValidContinuityAttributeCount": score_count,
    }

    attr_df = pd.DataFrame({**common, **{k: center_cols[k] for k in ATTRIBUTE_COLUMNS if k in center_cols}, **stat_cols})
    context_df = pd.DataFrame({**common, **context_cols})
    density_df = pd.DataFrame(
        {
            **common,
            "SourceDensity": source_density,
            "Density": density,
            "HasFracture": (density > 0.0).astype(int),
            "DensitySourceLogic": density_status,
        }
    )
    confidence_df = pd.DataFrame(
        {
            **common,
            "DistanceConfidenceWeight": np.full(len(well_df), distance_weight, dtype=np.float64),
            "PointConfidence": point_confidence,
            "ConfidenceLogic": np.full(
                len(well_df),
                "distance_controls_confidence_only_attribute_continuity_controls_density",
                dtype=object,
            ),
        }
    )

    return attr_df, context_df, density_df, confidence_df


def build_training_package(attr_df: pd.DataFrame, density_df: pd.DataFrame, confidence_df: pd.DataFrame) -> pd.DataFrame:
    if density_df.empty:
        return pd.DataFrame()
    merge_keys = ["SampleID", "SourceSampleID", "SourceWellName", "VirtualWellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName", "DistanceToSource", "AttributeContinuity", "ValidContinuityAttributeCount"]
    out = density_df.merge(
        confidence_df[[col for col in [*merge_keys, "DistanceConfidenceWeight", "PointConfidence", "ConfidenceLogic"] if col in confidence_df.columns]],
        on=[col for col in merge_keys if col in density_df.columns and col in confidence_df.columns],
        how="left",
    )
    attr_keep = [col for col in [*merge_keys, *ATTRIBUTE_COLUMNS, *[f"{a}{s}" for a in ATTRIBUTE_COLUMNS for s in STAT_SUFFIXES]] if col in attr_df.columns]
    out = out.merge(attr_df[attr_keep], on=[col for col in merge_keys if col in out.columns and col in attr_df.columns], how="left")
    out = out.sort_values(["SourceWellName", "VirtualWellName", "TIME"]).reset_index(drop=True)
    return out


def write_summary(training_df: pd.DataFrame, output_csv: Path) -> None:
    if training_df.empty:
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig")
        return
    summary = (
        training_df.groupby(["SourceWellName", "VirtualWellName"], as_index=False)
        .agg(
            NumSamples=("TIME", "size"),
            NumFractureSamples=("HasFracture", "sum"),
            MeanDensity=("Density", "mean"),
            MaxDensity=("Density", "max"),
            MeanPointConfidence=("PointConfidence", "mean"),
            MeanAttributeContinuity=("AttributeContinuity", "mean"),
            UniqueSourceWellCount=("SourceWellName", "nunique"),
        )
    )
    summary["SingleSourceCheck"] = np.where(summary["UniqueSourceWellCount"] == 1, "pass", "fail")
    summary.to_csv(output_csv, index=False, encoding="utf-8-sig")


def append_csv_chunk(df: pd.DataFrame, output_csv: Path, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    encoding = "utf-8-sig" if write_header else "utf-8"
    df.to_csv(output_csv, mode=mode, header=write_header, index=False, encoding=encoding)


def summarize_training_chunk(training_df: pd.DataFrame) -> pd.DataFrame:
    if training_df.empty:
        return pd.DataFrame()
    summary = (
        training_df.groupby(["SourceWellName", "VirtualWellName"], as_index=False)
        .agg(
            NumSamples=("TIME", "size"),
            NumFractureSamples=("HasFracture", "sum"),
            MeanDensity=("Density", "mean"),
            MaxDensity=("Density", "max"),
            MeanPointConfidence=("PointConfidence", "mean"),
            MeanAttributeContinuity=("AttributeContinuity", "mean"),
            UniqueSourceWellCount=("SourceWellName", "nunique"),
        )
    )
    summary["SingleSourceCheck"] = np.where(summary["UniqueSourceWellCount"] == 1, "pass", "fail")
    return summary


def finalize_tmp_outputs(tmp_paths: dict[str, Path], final_paths: dict[str, Path]) -> None:
    for key, tmp_path in tmp_paths.items():
        tmp_path.replace(final_paths[key])


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(Path(args.config))
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)

    prediction_csv = Path(config["real_well_prediction_csv"])
    trace_header_csv = Path(config["trace_header_csv"])
    prediction_df = pd.read_csv(prediction_csv)
    required_prediction_cols = {"SampleID", "WellName", "X", "Y", "TIME", "Density"}
    missing = sorted(required_prediction_cols - set(prediction_df.columns))
    if missing:
        raise RuntimeError(f"Step 4 prediction csv missing required columns: {missing}")
    prediction_df = prediction_df[prediction_df["Density"].notna()].copy()
    selected_wells = config.get("selected_wells")
    trace_df = build_trace_grid(trace_header_csv)

    index_df = build_virtual_index(
        prediction_df=prediction_df,
        trace_df=trace_df,
        window_size=int(config.get("virtual_grid_window_size", 5)),
        selected_wells=selected_wells,
        max_source_wells=args.max_source_wells,
    )
    final_paths = {
        "index": output_dir / "virtual_well_index.csv",
        "attributes": output_dir / "virtual_well_point_attributes.csv",
        "context": output_dir / "virtual_well_3x3_context.csv",
        "density": output_dir / "virtual_well_density_weaklabel.csv",
        "confidence": output_dir / "virtual_well_confidence.csv",
        "training": output_dir / "virtual_well_training_samples.csv",
        "summary": output_dir / "virtual_well_training_summary.csv",
        "audit": output_dir / "virtual_well_build_audit.json",
    }
    tmp_paths = {key: path.with_name(f"{path.name}.tmp") for key, path in final_paths.items()}
    index_df.to_csv(tmp_paths["index"], index=False, encoding="utf-8-sig")

    trace_tree, trace_ids = build_trace_tree(trace_df)
    contexts = [open_volume_context(name, Path(path)) for name, path in dict(config["volume_paths"]).items()]
    write_headers = {
        "attributes": True,
        "context": True,
        "density": True,
        "confidence": True,
        "training": True,
    }
    summary_chunks: list[pd.DataFrame] = []
    training_sample_count = 0
    density_logic_counts: dict[str, int] = {}
    point_confidence_sum = 0.0
    point_confidence_count = 0
    continuity_sum = 0.0
    continuity_count = 0
    attribute_non_null_counts = {attr: 0 for attr in ATTRIBUTE_COLUMNS}
    progress_every = max(int(args.progress_every), 1)

    try:
        for item_idx, item in enumerate(index_df.to_dict(orient="records"), start=1):
            attr_df, context_df, density_df, confidence_df = build_virtual_samples_for_item(
                prediction_df=prediction_df,
                item=item,
                trace_tree=trace_tree,
                trace_ids=trace_ids,
                contexts=contexts,
                max_points_per_well=args.max_points_per_well,
                min_continuity_for_density=float(config.get("min_continuity_for_density", 0.35)),
                confidence_distance_scale=float(config.get("confidence_distance_scale", 75.0)),
                min_confidence=float(config.get("min_confidence", 0.05)),
            )
            training_df = build_training_package(attr_df=attr_df, density_df=density_df, confidence_df=confidence_df)
            for key, frame in [
                ("attributes", attr_df),
                ("context", context_df),
                ("density", density_df),
                ("confidence", confidence_df),
                ("training", training_df),
            ]:
                append_csv_chunk(frame, tmp_paths[key], write_headers[key])
                write_headers[key] = False

            summary_chunk = summarize_training_chunk(training_df)
            if not summary_chunk.empty:
                summary_chunks.append(summary_chunk)

            training_sample_count += int(len(training_df))
            if "DensitySourceLogic" in density_df.columns:
                for logic, count in density_df["DensitySourceLogic"].value_counts(dropna=False).items():
                    density_logic_counts[str(logic)] = density_logic_counts.get(str(logic), 0) + int(count)
            if "PointConfidence" in training_df.columns:
                confidence_vals = pd.to_numeric(training_df["PointConfidence"], errors="coerce").dropna()
                point_confidence_sum += float(confidence_vals.sum())
                point_confidence_count += int(len(confidence_vals))
            if "AttributeContinuity" in training_df.columns:
                continuity_vals = pd.to_numeric(training_df["AttributeContinuity"], errors="coerce").dropna()
                continuity_sum += float(continuity_vals.sum())
                continuity_count += int(len(continuity_vals))
            for attr in ATTRIBUTE_COLUMNS:
                if attr in training_df.columns:
                    attribute_non_null_counts[attr] += int(pd.to_numeric(training_df[attr], errors="coerce").notna().sum())

            if item_idx == 1 or item_idx % progress_every == 0 or item_idx == len(index_df):
                print(
                    f"[step5] virtual_wells={item_idx}/{len(index_df)} "
                    f"training_samples={training_sample_count}",
                    flush=True,
                )
    finally:
        close_contexts(contexts)

    if summary_chunks:
        pd.concat(summary_chunks, ignore_index=True).to_csv(tmp_paths["summary"], index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(tmp_paths["summary"], index=False, encoding="utf-8-sig")

    audit = {
        "real_well_prediction_csv": str(prediction_csv),
        "trace_header_csv": str(trace_header_csv),
        "virtual_grid_window_size": int(config.get("virtual_grid_window_size", 5)),
        "density_logic": "single_source_attribute_continuity",
        "distance_role": "candidate_range_and_confidence_only",
        "source_mixing": "forbidden",
        "candidate_virtual_well_count": int(index_df["VirtualWellName"].nunique()) if not index_df.empty else 0,
        "source_well_count": int(index_df["SourceWellName"].nunique()) if not index_df.empty else 0,
        "training_sample_count": int(training_sample_count),
        "max_source_wells": int(args.max_source_wells),
        "max_points_per_well": int(args.max_points_per_well),
        "output_write_mode": "streaming_tmp_then_replace",
        "density_logic_counts": density_logic_counts,
        "mean_point_confidence": float(point_confidence_sum / point_confidence_count) if point_confidence_count else None,
        "mean_attribute_continuity": float(continuity_sum / continuity_count) if continuity_count else None,
        "attribute_non_null_counts": attribute_non_null_counts,
    }
    tmp_paths["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    finalize_tmp_outputs(tmp_paths, final_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
