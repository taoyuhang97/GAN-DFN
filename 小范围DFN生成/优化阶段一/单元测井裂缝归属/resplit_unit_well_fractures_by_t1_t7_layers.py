from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - fallback path
    cKDTree = None


DEFAULT_INPUT_UNITS_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/单元测井裂缝/units"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/单元测井裂缝_新层位重拆分"
)
DEFAULT_TRACE_HEADER_CSV = Path(r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv")
DEFAULT_SURFACE_DIR = Path(r"/data/shared/project-oil/wx数据/砂砾岩/层位")

SURFACE_FILE_MAP = {
    "T1": "T1(馆陶底）.dat",
    "T2": "T2（沙一下特殊岩性顶）.dat",
    "T3": "T3（沙二底）.dat",
    "T4": "T4（沙三上底）.dat",
    "T5": "T5（沙三中底）.dat",
    "T6": "T6（沙三下底）.dat",
    "T7": "T7（沙四上底）.dat",
}
SURFACE_CODES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]

UNIT_ID_PATTERN = re.compile(r"^BX(?P<block_x>\d+)_BY(?P<block_y>\d+)$", flags=re.IGNORECASE)
BLOCK_SIZE = 25
CENTER_OFFSET = BLOCK_SIZE // 2
LAYER_COLUMNS = [
    "BlockX",
    "BlockY",
    "GeoIntervalKey",
    "StrataName",
    "TopSurfaceCode",
    "BaseSurfaceCode",
    "TopDepth",
    "BaseDepth",
    "TopTime",
    "BaseTime",
    "RealPointSeedCount",
    "RealSegmentSeedCount",
    "VirtualPointSeedCount",
    "VirtualSegmentSeedCount",
    "RealSeedCount",
    "VirtualSeedCount",
    "PreferredSource",
]
SEED_LAYER_COLUMNS = ["GeoIntervalKey", "StrataName", "TopSurfaceCode", "BaseSurfaceCode"]
UNIT_CATALOG_COLUMNS = [
    "UnitID",
    "BlockX",
    "BlockY",
    "ReliabilityClass",
    "HasRealData",
    "HasVirtualData",
    "DataMode",
    "RealLayerCount",
    "VirtualLayerCount",
    "LayerCount",
    "CoveredIntervalCount",
    "SeedCount",
    "RealSeedCount",
    "VirtualSeedCount",
    "RealPointSeedCount",
    "RealSegmentSeedCount",
    "VirtualPointSeedCount",
    "VirtualSegmentSeedCount",
    "TopDepth",
    "BaseDepth",
    "TopTime",
    "BaseTime",
    "PackageDir",
]


@dataclass
class SurfaceNearestLookup:
    surface_code: str
    surface_name: str
    filepath: Path
    points_xy: np.ndarray
    values_z: np.ndarray
    tree: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-split unit well-fracture seeds using the confirmed short-name T1-T7 surfaces."
    )
    parser.add_argument("--input-units-root", type=Path, default=DEFAULT_INPUT_UNITS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=f"t1_t7_resplit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--unit-id", nargs="+")
    parser.add_argument("--limit-units", type=int, default=0)
    parser.add_argument("--top-boundary-time-ms", type=float, default=1100.0)
    parser.add_argument("--bottom-boundary-time-ms", type=float, default=3800.0)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--min-layer-thickness-ms", type=float, default=0.2)
    return parser.parse_args()


def write_csv_utf8(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_unit_id(unit_id: str) -> tuple[int, int]:
    match = UNIT_ID_PATTERN.match(str(unit_id).strip())
    if match is None:
        raise ValueError(f"invalid unit id: {unit_id}")
    return int(match.group("block_x")), int(match.group("block_y"))


def load_trace_header_unique_xy(trace_header_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    if not trace_header_csv.exists():
        raise FileNotFoundError(f"trace header csv not found: {trace_header_csv}")
    header_df = pd.read_csv(trace_header_csv, encoding="utf-8-sig")
    if "X" not in header_df.columns or "Y" not in header_df.columns:
        raise ValueError(f"trace header csv missing X/Y columns: {trace_header_csv}")
    unique_x = np.sort(pd.to_numeric(header_df["X"], errors="coerce").dropna().unique().astype(float))
    unique_y = np.sort(pd.to_numeric(header_df["Y"], errors="coerce").dropna().unique().astype(float))
    if unique_x.size == 0 or unique_y.size == 0:
        raise ValueError(f"trace header csv has empty X/Y axes: {trace_header_csv}")
    return unique_x, unique_y


def resolve_unit_center_trace_xy(unique_x: np.ndarray, unique_y: np.ndarray, block_x: int, block_y: int) -> tuple[float, float]:
    start_x = int(block_x) * (BLOCK_SIZE - 1)
    start_y = int(block_y) * (BLOCK_SIZE - 1)
    center_x_idx = start_x + CENTER_OFFSET
    center_y_idx = start_y + CENTER_OFFSET
    if center_x_idx >= len(unique_x) or center_y_idx >= len(unique_y):
        raise ValueError(f"unit center exceeds trace header extent: BX{block_x}_BY{block_y}")
    return float(unique_x[center_x_idx]), float(unique_y[center_y_idx])


def read_dat_surface_points(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no valid XYZ rows found in surface file: {filepath}")
    data = np.asarray(rows, dtype=float)
    return data[:, :2].copy(), data[:, 2].copy()


def load_surface_nearest_lookups(surface_dir: Path) -> dict[str, SurfaceNearestLookup]:
    lookups: dict[str, SurfaceNearestLookup] = {}
    for surface_code in SURFACE_CODES:
        filename = SURFACE_FILE_MAP[surface_code]
        filepath = Path(surface_dir) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"surface file not found for {surface_code}: {filepath}")
        points_xy, values_z = read_dat_surface_points(filepath)
        tree = cKDTree(points_xy) if cKDTree is not None else None
        lookups[surface_code] = SurfaceNearestLookup(
            surface_code=surface_code,
            surface_name=filepath.stem,
            filepath=filepath,
            points_xy=points_xy,
            values_z=values_z,
            tree=tree,
        )
    return lookups


def snap_to_interval(value: float, z_step_ms: float) -> float:
    step = float(z_step_ms)
    if not np.isfinite(value):
        return float("nan")
    if step <= 0:
        return float(value)
    return float(np.round(float(value) / step) * step)


def query_surface_nearest(center_x: float, center_y: float, lookup: SurfaceNearestLookup, z_step_ms: float) -> dict[str, Any]:
    query_xy = np.asarray([[float(center_x), float(center_y)]], dtype=float)
    if lookup.tree is not None:
        distance_arr, index_arr = lookup.tree.query(query_xy, k=1)
        nearest_idx = int(index_arr.reshape(-1)[0])
        nearest_distance = float(distance_arr.reshape(-1)[0])
    else:
        deltas = lookup.points_xy - query_xy[0]
        dist2 = np.sum(deltas * deltas, axis=1)
        nearest_idx = int(np.argmin(dist2))
        nearest_distance = float(np.sqrt(dist2[nearest_idx]))
    nearest_time = float(lookup.values_z[nearest_idx])
    return {
        "SurfaceCode": lookup.surface_code,
        "SurfaceName": lookup.surface_name,
        "SurfaceFile": str(lookup.filepath),
        "QueryX": float(center_x),
        "QueryY": float(center_y),
        "NearestX": float(lookup.points_xy[nearest_idx, 0]),
        "NearestY": float(lookup.points_xy[nearest_idx, 1]),
        "DistanceXY": nearest_distance,
        "RawTime": nearest_time,
        "SnappedTime": snap_to_interval(nearest_time, z_step_ms),
    }


def build_surface_rows(unit_id: str, center_x: float, center_y: float, lookups: dict[str, SurfaceNearestLookup], z_step_ms: float) -> pd.DataFrame:
    rows = []
    for surface_code in SURFACE_CODES:
        rows.append(query_surface_nearest(center_x, center_y, lookups[surface_code], z_step_ms=z_step_ms))
    surface_df = pd.DataFrame(rows)
    surface_df.insert(0, "UnitID", str(unit_id))
    return surface_df


def adjust_surface_times_for_inversion(surface_df: pd.DataFrame) -> pd.DataFrame:
    work = surface_df.copy()
    raw_times = pd.to_numeric(work["SnappedTime"], errors="coerce").to_numpy(dtype=float)
    adjusted = raw_times.copy()
    inversion_flags = np.zeros(len(adjusted), dtype=bool)
    for idx in range(len(adjusted) - 2, -1, -1):
        next_value = adjusted[idx + 1]
        current_value = adjusted[idx]
        if np.isfinite(current_value) and np.isfinite(next_value) and current_value > next_value:
            adjusted[idx] = next_value
            inversion_flags[idx] = True
    work["AdjustedTime"] = adjusted
    work["AdjustedUpward"] = inversion_flags
    work["IntervalToNextValid"] = True
    for idx in range(len(work) - 1):
        top_time = adjusted[idx]
        base_time = adjusted[idx + 1]
        work.at[idx, "IntervalToNextValid"] = bool(np.isfinite(top_time) and np.isfinite(base_time) and base_time > top_time)
    if len(work) > 0:
        work.at[len(work) - 1, "IntervalToNextValid"] = True
    return work


def clamp_surface_times_to_window(
    surface_df: pd.DataFrame,
    top_boundary_time_ms: float,
    bottom_boundary_time_ms: float,
    z_step_ms: float,
) -> pd.DataFrame:
    work = surface_df.copy()
    lower = snap_to_interval(min(float(top_boundary_time_ms), float(bottom_boundary_time_ms)), z_step_ms)
    upper = snap_to_interval(max(float(top_boundary_time_ms), float(bottom_boundary_time_ms)), z_step_ms)
    adjusted = pd.to_numeric(work["AdjustedTime"], errors="coerce").to_numpy(dtype=float)
    clamped = adjusted.copy()
    finite_mask = np.isfinite(clamped)
    clamped[finite_mask] = np.clip(clamped[finite_mask], lower, upper)
    work["WindowAdjustedTime"] = clamped
    work["ClampedToWindow"] = finite_mask & (~np.isclose(clamped, adjusted, equal_nan=True))
    work["IntervalToNextValid"] = True
    for idx in range(len(work) - 1):
        top_time = clamped[idx]
        base_time = clamped[idx + 1]
        work.at[idx, "IntervalToNextValid"] = bool(np.isfinite(top_time) and np.isfinite(base_time) and base_time > top_time)
    if len(work) > 0:
        work.at[len(work) - 1, "IntervalToNextValid"] = True
    return work


def build_layers_from_adjusted_surfaces(
    unit_id: str,
    block_x: int,
    block_y: int,
    adjusted_surface_df: pd.DataFrame,
    top_boundary_time_ms: float,
    bottom_boundary_time_ms: float,
    min_layer_thickness_ms: float,
    z_step_ms: float,
) -> pd.DataFrame:
    boundary_rows: list[tuple[str, float]] = [("TOP_1100MS", snap_to_interval(top_boundary_time_ms, z_step_ms))]
    time_col = "WindowAdjustedTime" if "WindowAdjustedTime" in adjusted_surface_df.columns else "AdjustedTime"
    for row in adjusted_surface_df.itertuples(index=False):
        boundary_rows.append((str(row.SurfaceCode), float(getattr(row, time_col))))
    boundary_rows.append(("BOTTOM_3800MS", snap_to_interval(bottom_boundary_time_ms, z_step_ms)))

    rows: list[dict[str, Any]] = []
    interval_idx = 1
    min_thickness = float(max(min_layer_thickness_ms, 0.0))
    for idx in range(len(boundary_rows) - 1):
        top_code, top_time = boundary_rows[idx]
        base_code, base_time = boundary_rows[idx + 1]
        if not np.isfinite(top_time) or not np.isfinite(base_time):
            continue
        if (float(base_time) - float(top_time)) < min_thickness:
            continue
        rows.append(
            {
                "BlockX": float(block_x),
                "BlockY": float(block_y),
                "GeoIntervalKey": f"{interval_idx:03d}_interval_{interval_idx:03d}",
                "StrataName": f"{top_code}->{base_code}",
                "TopSurfaceCode": str(top_code),
                "BaseSurfaceCode": str(base_code),
                "TopDepth": float(top_time),
                "BaseDepth": float(base_time),
                "TopTime": float(top_time),
                "BaseTime": float(base_time),
                "RealPointSeedCount": 0,
                "RealSegmentSeedCount": 0,
                "VirtualPointSeedCount": 0,
                "VirtualSegmentSeedCount": 0,
                "RealSeedCount": 0,
                "VirtualSeedCount": 0,
                "PreferredSource": "",
            }
        )
        interval_idx += 1
    return pd.DataFrame(rows, columns=LAYER_COLUMNS)


def load_unit_inputs(unit_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    seeds_path = unit_dir / "fracture_seeds.csv"
    meta_path = unit_dir / "unit_meta.json"
    if not seeds_path.exists():
        raise FileNotFoundError(f"missing fracture_seeds.csv: {seeds_path}")
    seeds_df = pd.read_csv(seeds_path, encoding="utf-8-sig")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return seeds_df, meta


def sort_layers_df(layers_df: pd.DataFrame) -> pd.DataFrame:
    if layers_df.empty:
        return layers_df.copy()
    work = layers_df.copy()
    work["TopTime"] = pd.to_numeric(work["TopTime"], errors="coerce")
    work["BaseTime"] = pd.to_numeric(work["BaseTime"], errors="coerce")
    return work.sort_values(["TopTime", "BaseTime", "GeoIntervalKey"]).reset_index(drop=True)


def resolve_seed_time_range(seed_row: pd.Series) -> tuple[float, float]:
    start_time = pd.to_numeric(seed_row.get("TimeStart"), errors="coerce")
    end_time = pd.to_numeric(seed_row.get("TimeEnd"), errors="coerce")
    center_time = pd.to_numeric(seed_row.get("CenterTime"), errors="coerce")
    if pd.notna(start_time) and pd.notna(end_time):
        return float(min(start_time, end_time)), float(max(start_time, end_time))
    if pd.notna(center_time):
        center = float(center_time)
        return center, center
    if pd.notna(start_time):
        value = float(start_time)
        return value, value
    if pd.notna(end_time):
        value = float(end_time)
        return value, value
    return float("nan"), float("nan")


def match_point_layer(time_value: float, layers_df: pd.DataFrame) -> pd.Series | None:
    if not np.isfinite(time_value) or layers_df.empty:
        return None
    work = sort_layers_df(layers_df)
    for idx, layer in work.iterrows():
        top_time = float(layer["TopTime"])
        base_time = float(layer["BaseTime"])
        is_last = idx == len(work) - 1
        if top_time <= float(time_value) < base_time:
            return layer
        if is_last and top_time <= float(time_value) <= base_time:
            return layer
    return None


def clip_value_by_ratio(start_value: float, end_value: float, orig_start: float, orig_end: float, clipped_time: float) -> float:
    if not all(np.isfinite(value) for value in [start_value, end_value, orig_start, orig_end, clipped_time]):
        return float("nan")
    denom = float(orig_end) - float(orig_start)
    if abs(denom) <= 1e-8:
        return float((float(start_value) + float(end_value)) / 2.0)
    ratio = (float(clipped_time) - float(orig_start)) / denom
    return float(float(start_value) + ratio * (float(end_value) - float(start_value)))


def split_seed_row_by_layers(seed_row: pd.Series, layers_df: pd.DataFrame, z_step_ms: float) -> list[dict[str, Any]]:
    lower_time, upper_time = resolve_seed_time_range(seed_row)
    if not np.isfinite(lower_time) or not np.isfinite(upper_time):
        return []

    seed_type = str(seed_row.get("SeedType", "")).strip().lower()
    duration = float(upper_time - lower_time)
    is_point_like = seed_type == "point" or duration <= max(float(z_step_ms) / 2.0, 1e-8)
    if is_point_like:
        center_time = pd.to_numeric(seed_row.get("CenterTime"), errors="coerce")
        point_time = float(center_time) if pd.notna(center_time) else float(lower_time)
        matched = match_point_layer(point_time, layers_df)
        if matched is None:
            return []
        row = seed_row.to_dict()
        for col in SEED_LAYER_COLUMNS:
            row[col] = matched[col]
        return [row]

    orig_start_time = float(pd.to_numeric(seed_row.get("TimeStart"), errors="coerce"))
    orig_end_time = float(pd.to_numeric(seed_row.get("TimeEnd"), errors="coerce"))
    orig_start_depth = float(pd.to_numeric(seed_row.get("DepthStart"), errors="coerce"))
    orig_end_depth = float(pd.to_numeric(seed_row.get("DepthEnd"), errors="coerce"))
    center_x = pd.to_numeric(seed_row.get("CenterX"), errors="coerce")
    center_y = pd.to_numeric(seed_row.get("CenterY"), errors="coerce")
    center_depth = pd.to_numeric(seed_row.get("CenterDepth"), errors="coerce")
    density_weight = pd.to_numeric(seed_row.get("DensityWeight"), errors="coerce")
    length_weight = pd.to_numeric(seed_row.get("LengthWeight"), errors="coerce")

    split_rows: list[dict[str, Any]] = []
    overlap_index = 0
    for layer in sort_layers_df(layers_df).itertuples(index=False):
        layer_top = float(layer.TopTime)
        layer_base = float(layer.BaseTime)
        overlap_top = max(lower_time, layer_top)
        overlap_base = min(upper_time, layer_base)
        overlap_span = float(overlap_base - overlap_top)
        if overlap_span <= max(float(z_step_ms) / 2.0, 1e-8):
            continue
        overlap_index += 1
        row = seed_row.to_dict()
        if overlap_index > 1:
            row["SeedID"] = f"{seed_row.get('SeedID', '')}::SUB{overlap_index:02d}"
        for col in SEED_LAYER_COLUMNS:
            row[col] = getattr(layer, col)

        row["TimeStart"] = float(overlap_top)
        row["TimeEnd"] = float(overlap_base)
        row["CenterTime"] = float((overlap_top + overlap_base) / 2.0)
        row["DepthStart"] = clip_value_by_ratio(orig_start_depth, orig_end_depth, orig_start_time, orig_end_time, overlap_top)
        row["DepthEnd"] = clip_value_by_ratio(orig_start_depth, orig_end_depth, orig_start_time, orig_end_time, overlap_base)
        row["CenterDepth"] = clip_value_by_ratio(orig_start_depth, orig_end_depth, orig_start_time, orig_end_time, row["CenterTime"])
        if pd.notna(length_weight) and duration > 1e-8:
            row["LengthWeight"] = float(float(length_weight) * overlap_span / duration)
        if pd.notna(density_weight):
            row["DensityWeight"] = float(density_weight)
        if pd.notna(center_x):
            row["CenterX"] = float(center_x)
        if pd.notna(center_y):
            row["CenterY"] = float(center_y)
        if pd.isna(center_depth) and np.isfinite(row["CenterDepth"]):
            row["CenterDepth"] = float(row["CenterDepth"])
        split_rows.append(row)
    return split_rows


def resplit_unit_seeds(seeds_df: pd.DataFrame, layers_df: pd.DataFrame, z_step_ms: float) -> tuple[pd.DataFrame, int]:
    if seeds_df.empty:
        return seeds_df.copy(), 0
    work = seeds_df.copy()
    for col in SEED_LAYER_COLUMNS:
        if col not in work.columns:
            work[col] = ""
    output_rows: list[dict[str, Any]] = []
    split_segment_count = 0
    for _, seed_row in work.iterrows():
        pieces = split_seed_row_by_layers(seed_row, layers_df, z_step_ms=z_step_ms)
        if not pieces:
            continue
        if len(pieces) > 1:
            split_segment_count += 1
        output_rows.extend(pieces)
    output_df = pd.DataFrame(output_rows, columns=work.columns)
    if output_df.empty:
        return output_df, split_segment_count
    output_df["CenterDepth"] = pd.to_numeric(output_df.get("CenterDepth"), errors="coerce")
    output_df["CenterTime"] = pd.to_numeric(output_df.get("CenterTime"), errors="coerce")
    output_df["TimeStart"] = pd.to_numeric(output_df.get("TimeStart"), errors="coerce")
    output_df["TimeEnd"] = pd.to_numeric(output_df.get("TimeEnd"), errors="coerce")
    output_df = output_df.sort_values(["CenterDepth", "CenterTime", "SeedID"], na_position="last").reset_index(drop=True)
    return output_df, split_segment_count


def fill_layer_seed_counts(layers_df: pd.DataFrame, seeds_df: pd.DataFrame) -> pd.DataFrame:
    if layers_df.empty:
        return layers_df.copy()
    work = sort_layers_df(layers_df)
    if seeds_df.empty:
        return work

    seed_work = seeds_df.copy()
    seed_work["SourceKind"] = seed_work.get("SourceKind", pd.Series(index=seed_work.index, dtype=object)).fillna("").astype(str).str.lower()
    seed_work["SeedType"] = seed_work.get("SeedType", pd.Series(index=seed_work.index, dtype=object)).fillna("").astype(str).str.lower()
    group_cols = ["GeoIntervalKey", "TopSurfaceCode", "BaseSurfaceCode"]

    def count_mask(source_kind: str, seed_type: str) -> pd.DataFrame:
        mask = seed_work["SourceKind"].eq(source_kind) & seed_work["SeedType"].eq(seed_type)
        return (
            seed_work.loc[mask]
            .groupby(group_cols, as_index=False)
            .size()
            .rename(columns={"size": f"{source_kind}_{seed_type}_count"})
        )

    real_point = count_mask("real", "point")
    real_segment = count_mask("real", "segment")
    virtual_point = count_mask("virtual", "point")
    virtual_segment = count_mask("virtual", "segment")

    for df_count in [real_point, real_segment, virtual_point, virtual_segment]:
        work = work.merge(df_count, on=group_cols, how="left")

    work["RealPointSeedCount"] = pd.to_numeric(work.pop("real_point_count"), errors="coerce").fillna(0).astype(int)
    work["RealSegmentSeedCount"] = pd.to_numeric(work.pop("real_segment_count"), errors="coerce").fillna(0).astype(int)
    work["VirtualPointSeedCount"] = pd.to_numeric(work.pop("virtual_point_count"), errors="coerce").fillna(0).astype(int)
    work["VirtualSegmentSeedCount"] = pd.to_numeric(work.pop("virtual_segment_count"), errors="coerce").fillna(0).astype(int)
    work["RealSeedCount"] = work["RealPointSeedCount"] + work["RealSegmentSeedCount"]
    work["VirtualSeedCount"] = work["VirtualPointSeedCount"] + work["VirtualSegmentSeedCount"]

    preferred_source = []
    for row in work.itertuples(index=False):
        if int(row.RealSeedCount) > 0 and int(row.RealSeedCount) >= int(row.VirtualSeedCount):
            preferred_source.append("real")
        elif int(row.VirtualSeedCount) > 0:
            preferred_source.append("virtual")
        else:
            preferred_source.append("")
    work["PreferredSource"] = preferred_source
    return work[LAYER_COLUMNS]


def derive_data_mode(has_real_data: bool, has_virtual_data: bool) -> str:
    if has_real_data and has_virtual_data:
        return "real_virtual"
    if has_real_data:
        return "real_only"
    if has_virtual_data:
        return "virtual_only"
    return ""


def build_unit_catalog_row(
    unit_id: str,
    block_x: int,
    block_y: int,
    unit_meta: dict[str, Any],
    original_meta: dict[str, Any],
    layers_df: pd.DataFrame,
) -> dict[str, Any]:
    real_seed_count = int(unit_meta.get("RealSeedCount", 0) or 0)
    virtual_seed_count = int(unit_meta.get("VirtualSeedCount", 0) or 0)
    has_real_data = bool(original_meta.get("HasRealData", False)) or real_seed_count > 0
    has_virtual_data = bool(original_meta.get("HasVirtualData", False)) or virtual_seed_count > 0
    layer_work = layers_df.copy()
    real_layer_count = int((pd.to_numeric(layer_work.get("RealSeedCount"), errors="coerce").fillna(0) > 0).sum()) if not layer_work.empty else 0
    virtual_layer_count = int((pd.to_numeric(layer_work.get("VirtualSeedCount"), errors="coerce").fillna(0) > 0).sum()) if not layer_work.empty else 0
    data_mode = str(original_meta.get("DataMode", "")).strip() or derive_data_mode(has_real_data, has_virtual_data)
    reliability_class = str(original_meta.get("ReliabilityClass", "")).strip()
    return {
        "UnitID": str(unit_id),
        "BlockX": int(block_x),
        "BlockY": int(block_y),
        "ReliabilityClass": reliability_class,
        "HasRealData": bool(has_real_data),
        "HasVirtualData": bool(has_virtual_data),
        "DataMode": data_mode,
        "RealLayerCount": int(real_layer_count),
        "VirtualLayerCount": int(virtual_layer_count),
        "LayerCount": int(len(layer_work)),
        "CoveredIntervalCount": int(layer_work["GeoIntervalKey"].astype(str).nunique()) if not layer_work.empty else 0,
        "SeedCount": int(unit_meta.get("ResplitSeedCount", 0) or 0),
        "RealSeedCount": int(real_seed_count),
        "VirtualSeedCount": int(virtual_seed_count),
        "RealPointSeedCount": int(pd.to_numeric(layer_work.get("RealPointSeedCount"), errors="coerce").fillna(0).sum()) if not layer_work.empty else 0,
        "RealSegmentSeedCount": int(pd.to_numeric(layer_work.get("RealSegmentSeedCount"), errors="coerce").fillna(0).sum()) if not layer_work.empty else 0,
        "VirtualPointSeedCount": int(pd.to_numeric(layer_work.get("VirtualPointSeedCount"), errors="coerce").fillna(0).sum()) if not layer_work.empty else 0,
        "VirtualSegmentSeedCount": int(pd.to_numeric(layer_work.get("VirtualSegmentSeedCount"), errors="coerce").fillna(0).sum()) if not layer_work.empty else 0,
        "TopDepth": float(pd.to_numeric(layer_work["TopDepth"], errors="coerce").min()) if not layer_work.empty else np.nan,
        "BaseDepth": float(pd.to_numeric(layer_work["BaseDepth"], errors="coerce").max()) if not layer_work.empty else np.nan,
        "TopTime": float(pd.to_numeric(layer_work["TopTime"], errors="coerce").min()) if not layer_work.empty else np.nan,
        "BaseTime": float(pd.to_numeric(layer_work["BaseTime"], errors="coerce").max()) if not layer_work.empty else np.nan,
        "PackageDir": f"units/{unit_id}",
    }


def build_unit_meta(
    unit_id: str,
    block_x: int,
    block_y: int,
    center_x: float,
    center_y: float,
    original_seed_count: int,
    resplit_seed_df: pd.DataFrame,
    layers_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    split_segment_count: int,
) -> dict[str, Any]:
    real_seed_count = int((resplit_seed_df.get("SourceKind", pd.Series(dtype=object)).fillna("").astype(str).str.lower() == "real").sum()) if not resplit_seed_df.empty else 0
    virtual_seed_count = int((resplit_seed_df.get("SourceKind", pd.Series(dtype=object)).fillna("").astype(str).str.lower() == "virtual").sum()) if not resplit_seed_df.empty else 0
    inversion_surface_count = int(pd.to_numeric(surface_df.get("AdjustedUpward"), errors="coerce").fillna(0).astype(bool).sum()) if not surface_df.empty else 0
    clamped_surface_count = int(pd.to_numeric(surface_df.get("ClampedToWindow"), errors="coerce").fillna(0).astype(bool).sum()) if not surface_df.empty else 0
    skipped_interval_count = int((~pd.Series(surface_df.get("IntervalToNextValid", []), dtype=bool)).sum()) if not surface_df.empty else 0
    return {
        "UnitID": str(unit_id),
        "BlockX": int(block_x),
        "BlockY": int(block_y),
        "UnitCenterX": float(center_x),
        "UnitCenterY": float(center_y),
        "OriginalSeedCount": int(original_seed_count),
        "ResplitSeedCount": int(len(resplit_seed_df)),
        "SplitSegmentCount": int(split_segment_count),
        "LayerCount": int(len(layers_df)),
        "RealSeedCount": int(real_seed_count),
        "VirtualSeedCount": int(virtual_seed_count),
        "TopTime": float(pd.to_numeric(layers_df["TopTime"], errors="coerce").min()) if not layers_df.empty else None,
        "BaseTime": float(pd.to_numeric(layers_df["BaseTime"], errors="coerce").max()) if not layers_df.empty else None,
        "InversionAdjustedSurfaceCount": inversion_surface_count,
        "WindowClampedSurfaceCount": clamped_surface_count,
        "SkippedCollapsedIntervalCount": skipped_interval_count,
        "Files": ["unit_layers.csv", "fracture_seeds.csv", "resolved_surfaces.csv"],
    }


def resolve_unit_ids(input_units_root: Path, requested_unit_ids: list[str] | None, limit_units: int) -> list[str]:
    if requested_unit_ids:
        unit_ids = [str(item).strip() for item in requested_unit_ids if str(item).strip()]
    else:
        unit_ids = sorted(path.name for path in Path(input_units_root).iterdir() if path.is_dir())
    if limit_units > 0:
        unit_ids = unit_ids[: int(limit_units)]
    if not unit_ids:
        raise ValueError("no unit ids selected")
    return unit_ids


def process_unit(
    unit_id: str,
    input_units_root: Path,
    output_units_root: Path,
    unique_x: np.ndarray,
    unique_y: np.ndarray,
    lookups: dict[str, SurfaceNearestLookup],
    top_boundary_time_ms: float,
    bottom_boundary_time_ms: float,
    min_layer_thickness_ms: float,
    z_step_ms: float,
) -> dict[str, Any]:
    unit_dir = Path(input_units_root) / unit_id
    seeds_df, original_meta = load_unit_inputs(unit_dir)
    block_x, block_y = parse_unit_id(unit_id)
    center_x, center_y = resolve_unit_center_trace_xy(unique_x, unique_y, block_x, block_y)
    surface_df = build_surface_rows(unit_id, center_x, center_y, lookups, z_step_ms=z_step_ms)
    adjusted_surface_df = adjust_surface_times_for_inversion(surface_df)
    adjusted_surface_df = clamp_surface_times_to_window(
        adjusted_surface_df,
        top_boundary_time_ms=top_boundary_time_ms,
        bottom_boundary_time_ms=bottom_boundary_time_ms,
        z_step_ms=z_step_ms,
    )
    new_layers = build_layers_from_adjusted_surfaces(
        unit_id=unit_id,
        block_x=block_x,
        block_y=block_y,
        adjusted_surface_df=adjusted_surface_df,
        top_boundary_time_ms=top_boundary_time_ms,
        bottom_boundary_time_ms=bottom_boundary_time_ms,
        min_layer_thickness_ms=min_layer_thickness_ms,
        z_step_ms=z_step_ms,
    )
    resplit_seed_df, split_segment_count = resplit_unit_seeds(seeds_df, new_layers, z_step_ms=z_step_ms)
    new_layers = fill_layer_seed_counts(new_layers, resplit_seed_df)
    unit_meta = build_unit_meta(
        unit_id=unit_id,
        block_x=block_x,
        block_y=block_y,
        center_x=center_x,
        center_y=center_y,
        original_seed_count=int(len(seeds_df)),
        resplit_seed_df=resplit_seed_df,
        layers_df=new_layers,
        surface_df=adjusted_surface_df,
        split_segment_count=split_segment_count,
    )

    output_unit_dir = Path(output_units_root) / unit_id
    write_csv_utf8(new_layers, output_unit_dir / "unit_layers.csv")
    write_csv_utf8(resplit_seed_df, output_unit_dir / "fracture_seeds.csv")
    write_csv_utf8(adjusted_surface_df, output_unit_dir / "resolved_surfaces.csv")
    write_json(unit_meta, output_unit_dir / "unit_meta.json")
    unit_catalog_row = build_unit_catalog_row(
        unit_id=unit_id,
        block_x=block_x,
        block_y=block_y,
        unit_meta=unit_meta,
        original_meta=original_meta,
        layers_df=new_layers,
    )

    summary_row = {
        "UnitID": unit_id,
        "BlockX": int(block_x),
        "BlockY": int(block_y),
        "UnitCenterX": float(center_x),
        "UnitCenterY": float(center_y),
        "OriginalSeedCount": int(len(seeds_df)),
        "ResplitSeedCount": int(len(resplit_seed_df)),
        "LayerCount": int(len(new_layers)),
        "SplitSegmentCount": int(split_segment_count),
        "InversionAdjustedSurfaceCount": int(pd.to_numeric(adjusted_surface_df["AdjustedUpward"], errors="coerce").fillna(0).astype(bool).sum()),
        "WindowClampedSurfaceCount": int(pd.to_numeric(adjusted_surface_df.get("ClampedToWindow"), errors="coerce").fillna(0).astype(bool).sum()),
        "CollapsedIntervalCount": int((~adjusted_surface_df["IntervalToNextValid"].astype(bool)).sum()) if not adjusted_surface_df.empty else 0,
        "TopTime": float(pd.to_numeric(new_layers["TopTime"], errors="coerce").min()) if not new_layers.empty else np.nan,
        "BaseTime": float(pd.to_numeric(new_layers["BaseTime"], errors="coerce").max()) if not new_layers.empty else np.nan,
    }
    return {
        "summary_row": summary_row,
        "catalog_row": unit_catalog_row,
    }


def main() -> int:
    args = parse_args()
    unit_ids = resolve_unit_ids(args.input_units_root, args.unit_id, args.limit_units)
    output_root = Path(args.output_root) / str(args.run_name)
    output_units_root = output_root / "units"
    output_root.mkdir(parents=True, exist_ok=True)

    unique_x, unique_y = load_trace_header_unique_xy(args.trace_header_csv)
    lookups = load_surface_nearest_lookups(args.surface_dir)

    summary_rows = []
    catalog_rows = []
    for unit_id in unit_ids:
        result = process_unit(
            unit_id=unit_id,
            input_units_root=args.input_units_root,
            output_units_root=output_units_root,
            unique_x=unique_x,
            unique_y=unique_y,
            lookups=lookups,
            top_boundary_time_ms=float(args.top_boundary_time_ms),
            bottom_boundary_time_ms=float(args.bottom_boundary_time_ms),
            min_layer_thickness_ms=float(args.min_layer_thickness_ms),
            z_step_ms=float(args.z_step_ms),
        )
        summary_rows.append(result["summary_row"])
        catalog_rows.append(result["catalog_row"])

    summary_df = pd.DataFrame(summary_rows)
    unit_catalog_df = pd.DataFrame(catalog_rows, columns=UNIT_CATALOG_COLUMNS)
    if not unit_catalog_df.empty:
        unit_catalog_df = unit_catalog_df.sort_values(["BlockX", "BlockY", "UnitID"]).reset_index(drop=True)
    write_csv_utf8(summary_df, output_root / "unit_summary.csv")
    write_csv_utf8(unit_catalog_df, output_root / "unit_catalog.csv")
    write_json(
        {
            "run_name": str(args.run_name),
            "input_units_root": str(args.input_units_root),
            "output_root": str(output_root),
            "surface_dir": str(args.surface_dir),
            "trace_header_csv": str(args.trace_header_csv),
            "unit_count": int(len(summary_df)),
            "unit_ids": [str(unit_id) for unit_id in unit_ids],
            "unit_catalog_csv": str(output_root / "unit_catalog.csv"),
            "top_boundary_time_ms": float(args.top_boundary_time_ms),
            "bottom_boundary_time_ms": float(args.bottom_boundary_time_ms),
            "z_step_ms": float(args.z_step_ms),
            "min_layer_thickness_ms": float(args.min_layer_thickness_ms),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        output_root / "run_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
