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
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(STEP2_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_SCRIPT_DIR))
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from build_real_well_t4_t7_samples import (  # noqa: E402
    GRID_SPACING,
    NEIGHBOR_OFFSETS,
    ATTRIBUTE_VALUE_RULES,
    build_trace_grid,
    build_trace_tree,
    offset_to_axis_label,
    sample_trace_at_time,
)
from surface_tools import load_surface_tables  # noqa: E402


ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax"]
PRIMARY_ATTRIBUTE = "CurvatureMax"
DEFAULT_ATTRIBUTE_WEIGHTS = {
    "CurvatureMax": 0.65,
    "Coherence": 0.15,
    "AntTrack": 0.15,
    "SeisAmp": 0.05,
}
STAT_SUFFIXES = ["Mean", "Std", "Min", "Max", "ValidCount"]
VIRTUAL_TRAINING_COLUMNS = [
    "SourceSampleID",
    "SourceWellName",
    "VirtualWellName",
    "X",
    "Y",
    "TIME",
    "StrataName",
    "PresenceLabel",
    "DensityLabel",
    "PointConfidence",
    "SampleWeight",
    *ATTRIBUTE_COLUMNS,
]


@dataclass(frozen=True)
class VolumeContext:
    name: str
    handle: Any
    samples: np.ndarray
    trace_at: Any


@dataclass(frozen=True)
class SurfaceContext:
    code: str
    time: np.ndarray
    tree: KDTree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build formal Step 5 single-source virtual wells.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--max-source-wells", type=int, default=0, help="Optional smoke-test cap; 0 means all source wells.")
    parser.add_argument("--max-points-per-well", type=int, default=0, help="Optional smoke-test cap per source well; 0 means all points.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N virtual wells.")
    parser.add_argument("--output-dir", type=Path, help="Optional isolated output directory override.")
    return parser


def resolve_step4_inputs(manifest_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest = read_json(manifest_path)
    prediction_rel = manifest.get("step5_input") or manifest.get("prediction_table")
    points_rel = manifest.get("fracture_point_table")
    if not prediction_rel or not points_rel:
        raise RuntimeError("Step4 manifest must define prediction_table/step5_input and fracture_point_table")
    prediction_csv = (manifest_path.parent / str(prediction_rel)).resolve()
    points_csv = (manifest_path.parent / str(points_rel)).resolve()
    if not prediction_csv.exists() or not points_csv.exists():
        raise FileNotFoundError(f"Step4 manifest target missing: {prediction_csv} / {points_csv}")
    return prediction_csv, points_csv, manifest


def load_surface_contexts(layer_dir: Path) -> dict[str, SurfaceContext]:
    surfaces = load_surface_tables(layer_dir)
    contexts: dict[str, SurfaceContext] = {}
    for code in ("T4", "T6", "T7"):
        if code not in surfaces:
            raise RuntimeError(f"missing required surface: {code}")
        table = surfaces[code]["table"]
        xy = table[["X", "Y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        time = pd.to_numeric(table["Z"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(time)
        contexts[code] = SurfaceContext(code=code, time=time[valid], tree=KDTree(xy[valid]))
    return contexts


def query_surface_time(context: SurfaceContext, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = np.column_stack([x, y]).astype(float)
    out = np.full(len(xy), np.nan, dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    if valid.any():
        _, idx = context.tree.query(xy[valid], k=1)
        out[valid] = context.time[idx[:, 0]]
    return out


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


def sample_center_attributes_for_points(
    x: np.ndarray,
    y: np.ndarray,
    time_ms: np.ndarray,
    trace_tree: KDTree,
    trace_ids: np.ndarray,
    contexts: list[VolumeContext],
) -> dict[str, np.ndarray]:
    query = query_trace_neighbors_batch(x, y, trace_tree, trace_ids)
    return {
        ctx.name: clean_attribute_array(
            ctx.name,
            batch_sample_neighbor_query_at_time(query, time_ms, ctx.samples, ctx.trace_at),
        )
        for ctx in contexts
    }


def robust_scale(values: np.ndarray, floor: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(floor)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return max(1.4826 * mad, float(floor))


def prepare_source_cache(
    well_df: pd.DataFrame,
    trace_tree: KDTree,
    trace_ids: np.ndarray,
    contexts: list[VolumeContext],
    surfaces: dict[str, SurfaceContext],
    scale_floors: dict[str, float],
) -> dict[str, Any]:
    source_x = safe_numeric(well_df["X"]).to_numpy(dtype=float)
    source_y = safe_numeric(well_df["Y"]).to_numpy(dtype=float)
    source_time = safe_numeric(well_df["TIME"]).to_numpy(dtype=float)
    attrs = sample_center_attributes_for_points(
        source_x, source_y, source_time, trace_tree, trace_ids, contexts
    )
    surface_times = {
        code: query_surface_time(context, source_x, source_y)
        for code, context in surfaces.items()
    }
    strata = well_df["StrataName"].astype(str).to_numpy()
    top = np.where(strata == "沙三段", surface_times["T4"], surface_times["T6"])
    base = np.where(strata == "沙三段", surface_times["T6"], surface_times["T7"])
    thickness = base - top
    relative = np.divide(
        source_time - top,
        thickness,
        out=np.full(len(well_df), np.nan, dtype=float),
        where=np.isfinite(thickness) & (thickness > 0),
    )
    relative[(relative < 0) | (relative > 1)] = np.nan
    scales = {
        attr: robust_scale(values, float(scale_floors.get(attr, 1.0e-6)))
        for attr, values in attrs.items()
    }
    return {
        "attrs": attrs,
        "scales": scales,
        "relative": relative,
        "source_time": source_time,
    }


def remap_virtual_time(
    strata: np.ndarray,
    relative: np.ndarray,
    virtual_x: np.ndarray,
    virtual_y: np.ndarray,
    surfaces: dict[str, SurfaceContext],
) -> np.ndarray:
    surface_times = {
        code: query_surface_time(context, virtual_x, virtual_y)
        for code, context in surfaces.items()
    }
    top = np.where(strata == "沙三段", surface_times["T4"], surface_times["T6"])
    base = np.where(strata == "沙三段", surface_times["T6"], surface_times["T7"])
    thickness = base - top
    return top + relative * thickness


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
    exclude_source_trace: bool,
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
                if exclude_source_trace and dx == 0 and dy == 0:
                    continue
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
                        "IsCenterVirtualTrace": int(dx == 0 and dy == 0),
                    }
                )
    return pd.DataFrame(rows)


def build_virtual_samples_for_item(
    prediction_df: pd.DataFrame,
    item: dict[str, Any],
    trace_tree: KDTree,
    trace_ids: np.ndarray,
    contexts: list[VolumeContext],
    surfaces: dict[str, SurfaceContext],
    source_cache: dict[str, Any],
    max_points_per_well: int,
    attribute_weights: dict[str, float],
    primary_min_similarity: float,
    min_support_attribute_count: int,
    similarity_scale_multiplier: float,
    max_virtual_time_shift_ms: float,
    confidence_distance_scale: float,
    min_label_confidence: float,
    virtual_weight_divisor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_well = item["SourceWellName"]
    well_df = prediction_df[prediction_df["WellName"] == source_well].copy()
    cache = source_cache
    if max_points_per_well > 0 and len(well_df) > max_points_per_well:
        idx = np.linspace(0, len(well_df) - 1, max_points_per_well, dtype=int)
        well_df = well_df.iloc[idx].copy()
        cache = {
            **source_cache,
            "attrs": {key: np.asarray(value)[idx] for key, value in source_cache["attrs"].items()},
            "relative": np.asarray(source_cache["relative"])[idx],
            "source_time": np.asarray(source_cache["source_time"])[idx],
        }
    if well_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    virtual_x = safe_numeric(well_df["X"]).to_numpy(dtype=np.float64) + float(item["OffsetX"])
    virtual_y = safe_numeric(well_df["Y"]).to_numpy(dtype=np.float64) + float(item["OffsetY"])
    source_time = np.asarray(cache["source_time"], dtype=float)
    relative_time = np.asarray(cache["relative"], dtype=float)
    strata = well_df["StrataName"].astype(str).to_numpy()
    time_ms = remap_virtual_time(strata, relative_time, virtual_x, virtual_y, surfaces)
    center_cols, context_cols, stat_cols = sample_attributes_for_points(
        x=virtual_x,
        y=virtual_y,
        time_ms=time_ms,
        trace_tree=trace_tree,
        trace_ids=trace_ids,
        contexts=contexts,
    )

    score_sum = np.zeros(len(well_df), dtype=np.float64)
    score_weight = np.zeros(len(well_df), dtype=np.float64)
    score_count = np.zeros(len(well_df), dtype=np.int64)
    primary_similarity = np.full(len(well_df), np.nan, dtype=float)
    for attr in ATTRIBUTE_COLUMNS:
        if attr not in cache["attrs"] or attr not in center_cols:
            continue
        source_vals = np.asarray(cache["attrs"][attr], dtype=float)
        virtual_vals = center_cols[attr]
        valid = np.isfinite(source_vals) & np.isfinite(virtual_vals)
        if not valid.any():
            continue
        scale = max(
            float(cache["scales"].get(attr, 1.0e-6)) * float(similarity_scale_multiplier),
            1.0e-9,
        )
        scores = np.exp(-np.abs(source_vals[valid] - virtual_vals[valid]) / scale)
        weight = float(attribute_weights.get(attr, 0.0))
        score_sum[valid] += weight * scores
        score_weight[valid] += weight
        score_count[valid] += 1
        if attr == PRIMARY_ATTRIBUTE:
            primary_similarity[valid] = scores
    continuity = np.divide(
        score_sum,
        score_weight,
        out=np.full(len(well_df), np.nan, dtype=np.float64),
        where=score_weight > 0,
    )

    source_density = safe_numeric(well_df["Density"]).to_numpy(dtype=np.float64)
    source_presence = safe_numeric(well_df["HasFracture"]).to_numpy(dtype=float)
    time_shift = time_ms - source_time
    support_count = score_count - np.isfinite(primary_similarity).astype(int)
    distance_weight = math.exp(-float(item["DistanceToSource"]) / max(confidence_distance_scale, 1e-6))
    label_confidence = distance_weight * np.nan_to_num(continuity, nan=0.0)
    label_mask = (
        np.isfinite(source_density)
        & np.isfinite(source_presence)
        & np.isfinite(relative_time)
        & np.isfinite(time_ms)
        & np.isfinite(primary_similarity)
        & (primary_similarity >= float(primary_min_similarity))
        & (support_count >= int(min_support_attribute_count))
        & (np.abs(time_shift) <= float(max_virtual_time_shift_ms))
        & (label_confidence >= float(min_label_confidence))
    )
    presence_label = np.where(label_mask, (source_presence > 0).astype(float), np.nan)
    positive_mask = label_mask & (source_presence > 0)
    density_label = np.where(positive_mask, np.maximum(source_density * continuity, 0.0), np.nan)
    density = np.where(label_mask & (source_presence <= 0), 0.0, density_label)
    density_status = np.where(label_mask, "curvature_led_transfer", "unlabelled_low_confidence")
    point_confidence = np.where(label_mask, np.clip(label_confidence, 0.0, 1.0), np.nan)
    sample_weight = np.where(
        label_mask,
        np.clip(label_confidence, 0.0, 1.0) / max(float(virtual_weight_divisor), 1.0),
        np.nan,
    )
    sample_ids = [f"{item['VirtualWellName']}_{sample_id}" for sample_id in well_df["SampleID"].astype(str).tolist()]

    common = {
        "SampleID": sample_ids,
        "SourceSampleID": well_df["SampleID"].to_numpy(),
        "SourceWellName": np.full(len(well_df), source_well, dtype=object),
        "VirtualWellName": np.full(len(well_df), item["VirtualWellName"], dtype=object),
        "X": virtual_x,
        "Y": virtual_y,
        "TIME": time_ms,
        "SourceTIME": source_time,
        "VirtualTIME": time_ms,
        "TimeShiftMs": time_shift,
        "RelativeTimeInLayer": relative_time,
        "TVD": safe_numeric(well_df["TVD"]).to_numpy(dtype=np.float64) if "TVD" in well_df.columns else np.full(len(well_df), np.nan),
        "DEPT": safe_numeric(well_df["DEPT"]).to_numpy(dtype=np.float64) if "DEPT" in well_df.columns else np.full(len(well_df), np.nan),
        "StrataName": well_df["StrataName"].to_numpy() if "StrataName" in well_df.columns else np.full(len(well_df), pd.NA, dtype=object),
        "DistanceToSource": np.full(len(well_df), float(item["DistanceToSource"]), dtype=np.float64),
        "AttributeContinuity": continuity,
        "PrimaryCurvatureSimilarity": primary_similarity,
        "ValidContinuityAttributeCount": score_count,
    }

    attr_df = pd.DataFrame({**common, **{k: center_cols[k] for k in ATTRIBUTE_COLUMNS if k in center_cols}, **stat_cols})
    context_df = pd.DataFrame({**common, **context_cols})
    density_df = pd.DataFrame(
        {
            **common,
            "SourceDensity": source_density,
            "Density": density,
            "DensityLabel": density_label,
            "PresenceLabel": presence_label,
            "HasFracture": presence_label,
            "LabelStatus": density_status,
            "DensitySourceLogic": density_status,
        }
    )
    confidence_df = pd.DataFrame(
        {
            **common,
            "DistanceConfidenceWeight": np.full(len(well_df), distance_weight, dtype=np.float64),
            "PointConfidence": point_confidence,
            "LabelConfidence": point_confidence,
            "SampleWeight": sample_weight,
            "ConfidenceLogic": np.full(
                len(well_df),
                "curvature_required_weighted_attribute_transfer",
                dtype=object,
            ),
        }
    )

    return attr_df, context_df, density_df, confidence_df


def build_training_package(attr_df: pd.DataFrame, density_df: pd.DataFrame, confidence_df: pd.DataFrame) -> pd.DataFrame:
    if density_df.empty:
        return pd.DataFrame()
    # These frames are produced from one source array in the same order.  An
    # index-aligned join avoids floating-point coordinate merge keys and keeps
    # duplicate sample IDs impossible to introduce accidentally.
    out = density_df.reset_index(drop=True).copy()
    for frame in (confidence_df, attr_df):
        extra = frame.reset_index(drop=True).drop(columns=[c for c in out.columns if c in frame.columns], errors="ignore")
        out = pd.concat([out, extra], axis=1)
    training_mask = out["LabelStatus"].eq("curvature_led_transfer") & out["PresenceLabel"].notna()
    out = out.loc[training_mask].copy()
    missing = [column for column in VIRTUAL_TRAINING_COLUMNS if column not in out.columns]
    if missing:
        raise RuntimeError(f"compact virtual training contract missing columns: {missing}")
    out = out[VIRTUAL_TRAINING_COLUMNS]
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
            NumFractureSamples=("PresenceLabel", "sum"),
            MeanPositiveDensity=("DensityLabel", "mean"),
            MaxPositiveDensity=("DensityLabel", "max"),
            MeanPointConfidence=("PointConfidence", "mean"),
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
            NumFractureSamples=("PresenceLabel", "sum"),
            MeanPositiveDensity=("DensityLabel", "mean"),
            MaxPositiveDensity=("DensityLabel", "max"),
            MeanPointConfidence=("PointConfidence", "mean"),
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
    output_dir = args.output_dir or Path(config["output_dir"])
    ensure_dir(output_dir)

    manifest_path = Path(config["step4_manifest"]).resolve()
    prediction_csv, _points_csv, manifest = resolve_step4_inputs(manifest_path)
    trace_header_csv = Path(config["trace_header_csv"])
    prediction_df = pd.read_csv(prediction_csv)
    required_prediction_cols = {
        "SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName",
        "Density", "HasFracture", "PredictionValid",
    }
    missing = sorted(required_prediction_cols - set(prediction_df.columns))
    if missing:
        raise RuntimeError(f"Step 4 prediction csv missing required columns: {missing}")
    prediction_df = prediction_df[
        prediction_df["PredictionValid"].eq(1)
        & prediction_df["Density"].notna()
        & prediction_df["HasFracture"].notna()
    ].copy()
    if prediction_df.duplicated(["WellName", "DEPT"]).any():
        raise RuntimeError("Step5 requires the Step4 merged table: duplicate WellName+DEPT rows found")
    selected_wells = config.get("selected_wells")
    trace_df = build_trace_grid(trace_header_csv)

    index_df = build_virtual_index(
        prediction_df=prediction_df,
        trace_df=trace_df,
        window_size=int(config.get("virtual_grid_window_size", 5)),
        selected_wells=selected_wells,
        max_source_wells=args.max_source_wells,
        exclude_source_trace=bool(config.get("exclude_source_trace", False)),
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
    surfaces = load_surface_contexts(Path(config["layer_dir"]))
    source_caches = {
        well: prepare_source_cache(
            prediction_df[prediction_df["WellName"] == well].reset_index(drop=True),
            trace_tree, trace_ids, contexts, surfaces,
            dict(config.get("similarity_scale_floors", {})),
        )
        for well in index_df["SourceWellName"].drop_duplicates().tolist()
    }
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
    candidate_sample_count = 0
    accepted_positive_count = 0
    accepted_negative_count = 0
    unknown_sample_count = 0
    time_shift_rejection_count = 0
    curvature_rejection_count = 0
    progress_every = max(int(args.progress_every), 1)

    try:
        for item_idx, item in enumerate(index_df.to_dict(orient="records"), start=1):
            attr_df, context_df, density_df, confidence_df = build_virtual_samples_for_item(
                prediction_df=prediction_df,
                item=item,
                trace_tree=trace_tree,
                trace_ids=trace_ids,
                contexts=contexts,
                surfaces=surfaces,
                source_cache=source_caches[item["SourceWellName"]],
                max_points_per_well=args.max_points_per_well,
                attribute_weights=dict(config.get("attribute_weights", DEFAULT_ATTRIBUTE_WEIGHTS)),
                primary_min_similarity=float(config.get("primary_curvature_min_similarity", 0.5)),
                min_support_attribute_count=int(config.get("min_support_attribute_count", 2)),
                similarity_scale_multiplier=float(config.get("similarity_scale_multiplier", 2.0)),
                max_virtual_time_shift_ms=float(config.get("max_virtual_time_shift_ms", 30.0)),
                confidence_distance_scale=float(config.get("confidence_distance_scale", 75.0)),
                min_label_confidence=float(config.get("min_label_confidence", 0.2)),
                virtual_weight_divisor=float(config.get("virtual_weight_divisor", 25.0)),
            )
            training_df = build_training_package(attr_df=attr_df, density_df=density_df, confidence_df=confidence_df)
            candidate_sample_count += int(len(density_df))
            unknown_sample_count += int(density_df["PresenceLabel"].isna().sum())
            accepted_positive_count += int(pd.to_numeric(training_df.get("PresenceLabel"), errors="coerce").eq(1).sum())
            accepted_negative_count += int(pd.to_numeric(training_df.get("PresenceLabel"), errors="coerce").eq(0).sum())
            shifts = pd.to_numeric(density_df.get("TimeShiftMs"), errors="coerce")
            time_shift_rejection_count += int((shifts.isna() | shifts.abs().gt(float(config.get("max_virtual_time_shift_ms", 30.0)))).sum())
            primary = pd.to_numeric(density_df.get("PrimaryCurvatureSimilarity"), errors="coerce")
            curvature_rejection_count += int((primary.isna() | primary.lt(float(config.get("primary_curvature_min_similarity", 0.5)))).sum())
            for key, frame in [
                ("attributes", attr_df),
                ("context", context_df),
                ("density", density_df),
                ("confidence", confidence_df),
                ("training", training_df),
            ]:
                if not frame.empty and (bool(config.get("write_intermediate_tables", False)) or key == "training"):
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
        "step4_manifest": str(manifest_path),
        "step4_manifest_version": manifest.get("version"),
        "real_well_prediction_csv": str(prediction_csv),
        "trace_header_csv": str(trace_header_csv),
        "virtual_grid_window_size": int(config.get("virtual_grid_window_size", 5)),
        "density_logic": "curvature_led_transfer_with_presence_and_conditional_density_labels",
        "distance_role": "candidate_range_and_confidence_only",
        "source_mixing": "forbidden",
        "candidate_virtual_well_count": int(index_df["VirtualWellName"].nunique()) if not index_df.empty else 0,
        "source_well_count": int(index_df["SourceWellName"].nunique()) if not index_df.empty else 0,
        "training_sample_count": int(training_sample_count),
        "candidate_sample_count": int(candidate_sample_count),
        "accepted_positive_count": int(accepted_positive_count),
        "accepted_negative_count": int(accepted_negative_count),
        "unknown_rejected_count": int(unknown_sample_count),
        "time_shift_rejection_count": int(time_shift_rejection_count),
        "curvature_rejection_count": int(curvature_rejection_count),
        "max_source_wells": int(args.max_source_wells),
        "max_points_per_well": int(args.max_points_per_well),
        "output_write_mode": "streaming_tmp_then_replace",
        "density_logic_counts": density_logic_counts,
        "mean_point_confidence": float(point_confidence_sum / point_confidence_count) if point_confidence_count else None,
        "mean_attribute_continuity": float(continuity_sum / continuity_count) if continuity_count else None,
        "attribute_non_null_counts": attribute_non_null_counts,
        "primary_attribute": PRIMARY_ATTRIBUTE,
        "curvature_pos_formal": False,
    }
    tmp_paths["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    publish_keys = ["index", "training", "summary", "audit"]
    if bool(config.get("write_intermediate_tables", False)):
        publish_keys.extend(["attributes", "context", "density", "confidence"])
    finalize_tmp_outputs(
        {key: tmp_paths[key] for key in publish_keys},
        {key: final_paths[key] for key in publish_keys},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
