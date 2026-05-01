from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TRACE_HEADER_CSV = Path(r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv")
DEFAULT_AGGREGATE_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/现有常规测井裂缝预测/汇总结果"
)
DEFAULT_INCLINED_AROUND_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/井斜/测井-地震时窗"
)
DEFAULT_VERTICAL_AROUND_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/测井/测井-地震时窗"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/虚拟测井构建/单元裂缝归属"
)

POINT_NUMERIC_COLS = [
    "GeoRangeOrder",
    "GeoDepthMin",
    "GeoDepthMax",
    "TVD",
    "TIME",
    "X",
    "Y",
    "Segment_ID",
    "Point_ID_In_Segment",
    "SegStartDepth",
    "SegEndDepth",
    "SegLength",
    "PredPointCount",
    "PointDensityMassPerLengthAllocated",
    "PointDensityMassAllocated",
    "PointAzimuth",
    "PointDip",
    "PredOrientationConfidence",
]

SEGMENT_NUMERIC_COLS = [
    "GeoRangeOrder",
    "GeoDepthMin",
    "GeoDepthMax",
    "Segment_ID",
    "SegStartDepth",
    "SegEndDepth",
    "SegLength",
    "PredPointCount",
    "PredDensityStrength",
    "PredP10MassPerLength",
    "PredP10Mass",
    "PredOrientationConfidence",
    "PredAzimuth",
    "PredDip",
]

INTERVAL_NUMERIC_COLS = [
    "GeoRangeOrder",
    "GeoDepthMin",
    "GeoDepthMax",
]

TRAJECTORY_NUMERIC_COLS = ["TVD", "DEPT", "TIME", "X", "Y"]

STRING_FILL_COLS = [
    "WellName",
    "SourceGroup",
    "StrataName",
    "GeoSegmentID",
    "GeoIntervalKey",
    "TopSurfaceCode",
    "TopSurfaceName",
    "BaseSurfaceCode",
    "BaseSurfaceName",
]


@dataclass(frozen=True)
class GridConfig:
    trace_header_csv: Path
    block_size: int = 25
    stride: int = 24
    chunksize: int = 200_000


@dataclass
class GridIndex:
    x_coords: np.ndarray
    y_coords: np.ndarray
    block_size: int
    stride: int

    @property
    def num_blocks_x(self) -> int:
        return max(0, math.floor((len(self.x_coords) - self.block_size) / self.stride) + 1)

    @property
    def num_blocks_y(self) -> int:
        return max(0, math.floor((len(self.y_coords) - self.block_size) / self.stride) + 1)

    def unit_id(self, block_x: int, block_y: int) -> str:
        return f"BX{int(block_x)}_BY{int(block_y)}"

    def nearest_indices(self, coords: np.ndarray, values: Iterable[float]) -> np.ndarray:
        values_arr = np.asarray(list(values), dtype=float)
        if len(coords) == 0:
            raise ValueError("Grid coordinates are empty.")
        if len(coords) == 1:
            return np.zeros(len(values_arr), dtype=int)
        idx = np.searchsorted(coords, values_arr)
        idx = np.clip(idx, 1, len(coords) - 1)
        before = coords[idx - 1]
        after = coords[idx]
        use_before = np.abs(values_arr - before) <= np.abs(values_arr - after)
        return np.where(use_before, idx - 1, idx).astype(int)

    def locate_blocks_xy(self, x_values: Iterable[float], y_values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        x_idx = self.nearest_indices(self.x_coords, x_values)
        y_idx = self.nearest_indices(self.y_coords, y_values)
        block_x = np.clip(x_idx // self.stride, 0, self.num_blocks_x - 1).astype(int)
        block_y = np.clip(y_idx // self.stride, 0, self.num_blocks_y - 1).astype(int)
        return block_x, block_y

    def locate_block(self, x: float, y: float) -> tuple[int, int]:
        block_x, block_y = self.locate_blocks_xy([x], [y])
        return int(block_x[0]), int(block_y[0])

    def block_bounds(self, block_x: int, block_y: int) -> dict[str, float]:
        start_x = block_x * self.stride
        end_x = min(start_x + self.block_size - 1, len(self.x_coords) - 1)
        start_y = block_y * self.stride
        end_y = min(start_y + self.block_size - 1, len(self.y_coords) - 1)
        x_slice = self.x_coords[start_x : end_x + 1]
        y_slice = self.y_coords[start_y : end_y + 1]
        return {
            "BlockX": int(block_x),
            "BlockY": int(block_y),
            "UnitID": self.unit_id(block_x, block_y),
            "XMin": float(x_slice.min()),
            "XMax": float(x_slice.max()),
            "YMin": float(y_slice.min()),
            "YMax": float(y_slice.max()),
            "XCenter": float((x_slice.min() + x_slice.max()) / 2.0),
            "YCenter": float((y_slice.min() + y_slice.max()) / 2.0),
            "TraceCountX": int(len(x_slice)),
            "TraceCountY": int(len(y_slice)),
        }


def read_csv_utf8(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv_utf8(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fill_string_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def coalesce_series(df: pd.DataFrame, candidates: list[str], default: float | str | None = np.nan) -> pd.Series:
    series: pd.Series | None = None
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col] if series is None else series.combine_first(df[col])
    if series is None:
        return pd.Series(default, index=df.index)
    return series.fillna(default)


def parse_well_names(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_segment_key(df: pd.DataFrame) -> pd.Series:
    segment_id = coalesce_series(df, ["Segment_ID"], default=-1).fillna(-1)
    return (
        df["WellName"].astype(str)
        + "::"
        + df["GeoIntervalKey"].astype(str)
        + "::SEG"
        + segment_id.astype(int).astype(str)
    )


def build_interval_key(df: pd.DataFrame) -> pd.Series:
    return df["WellName"].astype(str) + "::" + df["GeoIntervalKey"].astype(str)


def build_grid_index(config: GridConfig) -> GridIndex:
    if not config.trace_header_csv.exists():
        raise FileNotFoundError(f"Trace header CSV not found: {config.trace_header_csv}")

    x_values: set[float] = set()
    y_values: set[float] = set()
    for chunk in pd.read_csv(
        config.trace_header_csv,
        usecols=["X", "Y"],
        chunksize=config.chunksize,
        encoding="utf-8-sig",
    ):
        x = pd.to_numeric(chunk["X"], errors="coerce").dropna().astype(float).unique()
        y = pd.to_numeric(chunk["Y"], errors="coerce").dropna().astype(float).unique()
        x_values.update(map(float, x.tolist()))
        y_values.update(map(float, y.tolist()))

    if not x_values or not y_values:
        raise ValueError(f"Failed to extract grid coordinates from: {config.trace_header_csv}")

    return GridIndex(
        x_coords=np.array(sorted(x_values), dtype=float),
        y_coords=np.array(sorted(y_values), dtype=float),
        block_size=config.block_size,
        stride=config.stride,
    )


def build_unit_index(
    grid: GridIndex,
    block_x_start: int | None,
    block_x_end: int | None,
    block_y_start: int | None,
    block_y_end: int | None,
) -> pd.DataFrame:
    x_start = 0 if block_x_start is None else max(0, int(block_x_start))
    x_end = grid.num_blocks_x - 1 if block_x_end is None else min(grid.num_blocks_x - 1, int(block_x_end))
    y_start = 0 if block_y_start is None else max(0, int(block_y_start))
    y_end = grid.num_blocks_y - 1 if block_y_end is None else min(grid.num_blocks_y - 1, int(block_y_end))

    rows: list[dict[str, float]] = []
    for block_x in range(x_start, x_end + 1):
        for block_y in range(y_start, y_end + 1):
            rows.append(grid.block_bounds(block_x, block_y))
    return pd.DataFrame(rows)


def map_xy_to_units(df: pd.DataFrame, grid: GridIndex, x_col: str = "X", y_col: str = "Y") -> pd.DataFrame:
    work = df.copy()
    valid = work[x_col].notna() & work[y_col].notna()
    work["BlockX"] = np.nan
    work["BlockY"] = np.nan
    work["UnitID"] = ""
    if valid.any():
        block_x, block_y = grid.locate_blocks_xy(work.loc[valid, x_col], work.loc[valid, y_col])
        work.loc[valid, "BlockX"] = block_x
        work.loc[valid, "BlockY"] = block_y
        work.loc[valid, "UnitID"] = [grid.unit_id(int(x), int(y)) for x, y in zip(block_x, block_y)]
    return work


def load_aggregate_tables(aggregate_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points_path = aggregate_dir / "all_final_fracture_points.csv"
    segments_path = aggregate_dir / "all_final_fracture_segments.csv"
    intervals_path = aggregate_dir / "all_final_strata_segmentation.csv"
    if not points_path.exists() or not segments_path.exists() or not intervals_path.exists():
        raise FileNotFoundError(f"Aggregate directory is incomplete: {aggregate_dir}")

    points = ensure_numeric(read_csv_utf8(points_path), POINT_NUMERIC_COLS).drop_duplicates()
    segments = ensure_numeric(read_csv_utf8(segments_path), SEGMENT_NUMERIC_COLS).drop_duplicates()
    intervals = ensure_numeric(read_csv_utf8(intervals_path), INTERVAL_NUMERIC_COLS).drop_duplicates()

    points = fill_string_columns(points, STRING_FILL_COLS)
    segments = fill_string_columns(segments, STRING_FILL_COLS)
    intervals = fill_string_columns(intervals, STRING_FILL_COLS)
    return points, segments, intervals


def resolve_target_wells(
    points_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    well_names: list[str],
    max_wells: int,
) -> list[str]:
    discovered = sorted(
        {
            *points_df["WellName"].dropna().astype(str).tolist(),
            *segments_df["WellName"].dropna().astype(str).tolist(),
            *intervals_df["WellName"].dropna().astype(str).tolist(),
        }
    )
    if well_names:
        return [item for item in well_names if item in discovered]
    if max_wells > 0:
        return discovered[:max_wells]
    return discovered


def filter_frames_by_wells(
    points_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    wells: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    well_set = set(wells)
    points = points_df[points_df["WellName"].isin(well_set)].copy()
    segments = segments_df[segments_df["WellName"].isin(well_set)].copy()
    intervals = intervals_df[intervals_df["WellName"].isin(well_set)].copy()
    return points, segments, intervals


def build_trajectory_catalog(
    inclined_dir: Path,
    vertical_dir: Path,
    wells: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    selected = set(wells)
    seen: set[str] = set()
    for source_group, folder in [("inclined", inclined_dir), ("vertical", vertical_dir)]:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*_around_data.csv")):
            well_name = path.name.removesuffix("_around_data.csv")
            if well_name not in selected or well_name in seen:
                continue
            rows.append(
                {
                    "WellName": well_name,
                    "SourceGroup": source_group,
                    "TrajectoryPath": str(path),
                }
            )
            seen.add(well_name)
    return pd.DataFrame(rows)


def load_trajectories(catalog_df: pd.DataFrame, grid: GridIndex) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    trajectories: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []

    for item in catalog_df.itertuples(index=False):
        path = Path(item.TrajectoryPath)
        raw = ensure_numeric(read_csv_utf8(path), TRAJECTORY_NUMERIC_COLS)
        raw["TVD"] = coalesce_series(raw, ["TVD", "DEPT"])
        raw = raw.dropna(subset=["TVD", "TIME", "X", "Y"]).copy()
        if raw.empty:
            continue
        raw = raw.sort_values("TVD").drop_duplicates(subset=["TVD"], keep="first").reset_index(drop=True)
        raw["WellName"] = item.WellName
        raw["SourceGroup"] = item.SourceGroup
        raw = map_xy_to_units(raw, grid)
        raw["SampleIndex"] = np.arange(1, len(raw) + 1, dtype=int)
        trajectories[item.WellName] = raw

        covered_units = sorted(raw["UnitID"].dropna().astype(str).unique().tolist())
        summaries.append(
            {
                "WellName": item.WellName,
                "SourceGroup": item.SourceGroup,
                "TrajectoryPath": str(path),
                "SampleCount": int(len(raw)),
                "TVDMin": float(raw["TVD"].min()),
                "TVDMax": float(raw["TVD"].max()),
                "TIMEMin": float(raw["TIME"].min()),
                "TIMEMax": float(raw["TIME"].max()),
                "XStart": float(raw["X"].iloc[0]),
                "YStart": float(raw["Y"].iloc[0]),
                "XEnd": float(raw["X"].iloc[-1]),
                "YEnd": float(raw["Y"].iloc[-1]),
                "CoveredUnitCount": int(len(covered_units)),
                "CoveredUnits": ",".join(covered_units),
            }
        )

    return trajectories, pd.DataFrame(summaries)


def azimuth_dip_to_normal(azimuth_deg: np.ndarray, dip_deg: np.ndarray) -> np.ndarray:
    azimuth_rad = np.deg2rad(azimuth_deg)
    dip_rad = np.deg2rad(dip_deg)
    return np.column_stack(
        [
            np.sin(dip_rad) * np.sin(azimuth_rad),
            np.sin(dip_rad) * np.cos(azimuth_rad),
            np.cos(dip_rad),
        ]
    )


def weighted_orientation(
    df: pd.DataFrame,
    azimuth_col: str,
    dip_col: str,
    weight_col: str,
) -> tuple[float, float]:
    work = df.copy()
    work[azimuth_col] = pd.to_numeric(work[azimuth_col], errors="coerce")
    work[dip_col] = pd.to_numeric(work[dip_col], errors="coerce")
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    valid = work[azimuth_col].notna() & work[dip_col].notna() & work[weight_col].notna() & (work[weight_col] > 0)
    if not valid.any():
        return float("nan"), float("nan")
    normals = azimuth_dip_to_normal(
        work.loc[valid, azimuth_col].to_numpy(),
        work.loc[valid, dip_col].to_numpy(),
    )
    normal = np.average(normals, axis=0, weights=work.loc[valid, weight_col].to_numpy())
    norm = np.linalg.norm(normal)
    if norm <= 1e-8:
        return float("nan"), float("nan")
    normal = normal / norm
    azimuth = math.degrees(math.atan2(normal[0], normal[1])) % 360.0
    dip = math.degrees(math.acos(np.clip(normal[2], -1.0, 1.0)))
    return float(azimuth), float(dip)


def interpolate_trajectory_point(trajectory_df: pd.DataFrame, depth: float, grid: GridIndex) -> dict[str, object] | None:
    if trajectory_df is None or trajectory_df.empty or not np.isfinite(depth):
        return None
    tvd = trajectory_df["TVD"].to_numpy(dtype=float)
    if depth < tvd.min() or depth > tvd.max():
        return None
    x = float(np.interp(depth, tvd, trajectory_df["X"].to_numpy(dtype=float)))
    y = float(np.interp(depth, tvd, trajectory_df["Y"].to_numpy(dtype=float)))
    time = float(np.interp(depth, tvd, trajectory_df["TIME"].to_numpy(dtype=float)))
    block_x, block_y = grid.locate_block(x, y)
    return {
        "TVD": float(depth),
        "TIME": time,
        "X": x,
        "Y": y,
        "BlockX": int(block_x),
        "BlockY": int(block_y),
        "UnitID": grid.unit_id(block_x, block_y),
    }


def extract_interval_path(
    trajectory_df: pd.DataFrame | None,
    depth_start: float,
    depth_end: float,
    grid: GridIndex,
) -> pd.DataFrame:
    if trajectory_df is None or trajectory_df.empty:
        return pd.DataFrame(columns=["TVD", "TIME", "X", "Y", "BlockX", "BlockY", "UnitID"])
    if not np.isfinite(depth_start) or not np.isfinite(depth_end):
        return pd.DataFrame(columns=["TVD", "TIME", "X", "Y", "BlockX", "BlockY", "UnitID"])

    start = min(float(depth_start), float(depth_end))
    end = max(float(depth_start), float(depth_end))
    lower = float(trajectory_df["TVD"].min())
    upper = float(trajectory_df["TVD"].max())
    if end < lower or start > upper:
        return pd.DataFrame(columns=["TVD", "TIME", "X", "Y", "BlockX", "BlockY", "UnitID"])

    start = max(start, lower)
    end = min(end, upper)
    start_row = interpolate_trajectory_point(trajectory_df, start, grid)
    end_row = interpolate_trajectory_point(trajectory_df, end, grid)
    if start_row is None or end_row is None:
        return pd.DataFrame(columns=["TVD", "TIME", "X", "Y", "BlockX", "BlockY", "UnitID"])

    interior = trajectory_df[(trajectory_df["TVD"] > start) & (trajectory_df["TVD"] < end)][
        ["TVD", "TIME", "X", "Y", "BlockX", "BlockY", "UnitID"]
    ].copy()
    path = pd.concat([pd.DataFrame([start_row]), interior, pd.DataFrame([end_row])], ignore_index=True)
    path = path.sort_values("TVD").drop_duplicates(subset=["TVD"], keep="first").reset_index(drop=True)
    return path


def build_well_source_group_map(
    points_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    trajectory_summary_df: pd.DataFrame,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for df in [points_df, segments_df, trajectory_summary_df]:
        if df.empty or "WellName" not in df.columns or "SourceGroup" not in df.columns:
            continue
        subset = df[["WellName", "SourceGroup"]].dropna().copy()
        for row in subset.itertuples(index=False):
            well_name = str(row.WellName)
            source_group = str(row.SourceGroup)
            if well_name and source_group and well_name not in mapping:
                mapping[well_name] = source_group
    return mapping


def prepare_point_master(
    points_df: pd.DataFrame,
    grid: GridIndex,
    well_source_map: dict[str, str],
) -> pd.DataFrame:
    points = points_df.copy()
    points["SourceGroup"] = points["SourceGroup"].replace("", np.nan)
    points["SourceGroup"] = points["SourceGroup"].fillna(points["WellName"].map(well_source_map)).fillna("")
    points["SegmentKey"] = build_segment_key(points)
    points["IntervalKey"] = build_interval_key(points)
    points["FracturePointKey"] = (
        points["SegmentKey"]
        + "::PT"
        + coalesce_series(points, ["Point_ID_In_Segment"], default=-1).fillna(-1).astype(int).astype(str)
    )
    points = map_xy_to_units(points, grid)
    points["OrientationWeight"] = coalesce_series(
        points,
        ["PointDensityMassAllocated", "PointDensityMassPerLengthAllocated"],
        default=1.0,
    )
    points["RecordType"] = "fracture_point"
    return points


def build_point_based_hints(
    point_master_df: pd.DataFrame,
    key_col: str,
    point_id_col: str,
) -> pd.DataFrame:
    if point_master_df.empty:
        return pd.DataFrame()
    group_cols = [key_col, "UnitID", "BlockX", "BlockY"]
    hints = (
        point_master_df.groupby(group_cols, as_index=False)
        .agg(
            HintDepthMin=("TVD", "min"),
            HintDepthMax=("TVD", "max"),
            HintTimeMin=("TIME", "min"),
            HintTimeMax=("TIME", "max"),
            HintCenterDepth=("TVD", "mean"),
            HintCenterTime=("TIME", "mean"),
            HintCenterX=("X", "mean"),
            HintCenterY=("Y", "mean"),
            HintPointCount=(point_id_col, "count"),
        )
        .sort_values([key_col, "HintPointCount"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return hints


def prepare_segment_master(
    segments_df: pd.DataFrame,
    trajectory_map: dict[str, pd.DataFrame],
    point_hints_df: pd.DataFrame,
    grid: GridIndex,
    well_source_map: dict[str, str],
) -> pd.DataFrame:
    segments = segments_df.copy()
    segments["SourceGroup"] = segments["SourceGroup"].replace("", np.nan)
    segments["SourceGroup"] = segments["SourceGroup"].fillna(segments["WellName"].map(well_source_map)).fillna("")
    segments["SegmentKey"] = build_segment_key(segments)
    segments["IntervalKey"] = build_interval_key(segments)
    segments["MidDepth"] = (segments["SegStartDepth"] + segments["SegEndDepth"]) / 2.0
    segments["MidTime"] = np.nan
    segments["MidX"] = np.nan
    segments["MidY"] = np.nan
    segments["MidBlockX"] = np.nan
    segments["MidBlockY"] = np.nan
    segments["MidUnitID"] = ""
    segments["TrajectoryAnchorSource"] = "none"
    hint_lookup = {
        key: group.reset_index(drop=True)
        for key, group in point_hints_df.groupby("SegmentKey")
    } if not point_hints_df.empty else {}

    for idx, row in segments.iterrows():
        anchor = interpolate_trajectory_point(trajectory_map.get(row["WellName"]), row["MidDepth"], grid)
        if anchor is not None:
            segments.at[idx, "MidTime"] = anchor["TIME"]
            segments.at[idx, "MidX"] = anchor["X"]
            segments.at[idx, "MidY"] = anchor["Y"]
            segments.at[idx, "MidBlockX"] = anchor["BlockX"]
            segments.at[idx, "MidBlockY"] = anchor["BlockY"]
            segments.at[idx, "MidUnitID"] = anchor["UnitID"]
            segments.at[idx, "TrajectoryAnchorSource"] = "trajectory"
            continue
        hint_rows = hint_lookup.get(row["SegmentKey"])
        if hint_rows is not None and not hint_rows.empty:
            hint = hint_rows.iloc[0]
            segments.at[idx, "MidTime"] = hint["HintCenterTime"]
            segments.at[idx, "MidX"] = hint["HintCenterX"]
            segments.at[idx, "MidY"] = hint["HintCenterY"]
            segments.at[idx, "MidBlockX"] = hint["BlockX"]
            segments.at[idx, "MidBlockY"] = hint["BlockY"]
            segments.at[idx, "MidUnitID"] = hint["UnitID"]
            segments.at[idx, "TrajectoryAnchorSource"] = "point_hint"
    segments["RecordType"] = "fracture_segment"
    return segments


def prepare_interval_master(
    intervals_df: pd.DataFrame,
    trajectory_map: dict[str, pd.DataFrame],
    point_hints_df: pd.DataFrame,
    grid: GridIndex,
    well_source_map: dict[str, str],
) -> pd.DataFrame:
    intervals = intervals_df.copy()
    intervals["SourceGroup"] = intervals["SourceGroup"].replace("", np.nan) if "SourceGroup" in intervals.columns else np.nan
    intervals["SourceGroup"] = coalesce_series(intervals, ["SourceGroup"], default=np.nan)
    intervals["SourceGroup"] = intervals["SourceGroup"].fillna(intervals["WellName"].map(well_source_map)).fillna("")
    intervals["IntervalKey"] = build_interval_key(intervals)
    intervals["MidDepth"] = (intervals["GeoDepthMin"] + intervals["GeoDepthMax"]) / 2.0
    intervals["MidTime"] = np.nan
    intervals["MidX"] = np.nan
    intervals["MidY"] = np.nan
    intervals["MidBlockX"] = np.nan
    intervals["MidBlockY"] = np.nan
    intervals["MidUnitID"] = ""
    intervals["TrajectoryAnchorSource"] = "none"
    hint_lookup = {
        key: group.reset_index(drop=True)
        for key, group in point_hints_df.groupby("IntervalKey")
    } if not point_hints_df.empty else {}

    for idx, row in intervals.iterrows():
        anchor = interpolate_trajectory_point(trajectory_map.get(row["WellName"]), row["MidDepth"], grid)
        if anchor is not None:
            intervals.at[idx, "MidTime"] = anchor["TIME"]
            intervals.at[idx, "MidX"] = anchor["X"]
            intervals.at[idx, "MidY"] = anchor["Y"]
            intervals.at[idx, "MidBlockX"] = anchor["BlockX"]
            intervals.at[idx, "MidBlockY"] = anchor["BlockY"]
            intervals.at[idx, "MidUnitID"] = anchor["UnitID"]
            intervals.at[idx, "TrajectoryAnchorSource"] = "trajectory"
            continue
        hint_rows = hint_lookup.get(row["IntervalKey"])
        if hint_rows is not None and not hint_rows.empty:
            hint = hint_rows.iloc[0]
            intervals.at[idx, "MidTime"] = hint["HintCenterTime"]
            intervals.at[idx, "MidX"] = hint["HintCenterX"]
            intervals.at[idx, "MidY"] = hint["HintCenterY"]
            intervals.at[idx, "MidBlockX"] = hint["BlockX"]
            intervals.at[idx, "MidBlockY"] = hint["BlockY"]
            intervals.at[idx, "MidUnitID"] = hint["UnitID"]
            intervals.at[idx, "TrajectoryAnchorSource"] = "point_hint"
    intervals["RecordType"] = "well_interval"
    return intervals


def build_fragments_from_path(
    path_df: pd.DataFrame,
    base_row: dict[str, object],
    fragment_id_prefix: str,
    total_depth_span: float,
    assignment_source: str,
) -> list[dict[str, object]]:
    if path_df.empty:
        return []
    path = path_df.copy()
    path["RunID"] = path["UnitID"].ne(path["UnitID"].shift()).cumsum()
    fragment_rows: list[dict[str, object]] = []
    raw_spans: list[float] = []
    grouped_rows: list[tuple[int, pd.DataFrame]] = list(path.groupby("RunID", sort=False))
    for _, group in grouped_rows:
        raw_spans.append(max(float(group["TVD"].iloc[-1]) - float(group["TVD"].iloc[0]), 0.0))
    raw_span_sum = sum(raw_spans)
    if raw_span_sum <= 1e-8:
        raw_spans = [1.0] * len(grouped_rows)
        raw_span_sum = float(len(grouped_rows))

    for fragment_index, ((_, group), raw_span) in enumerate(zip(grouped_rows, raw_spans), start=1):
        ratio = raw_span / raw_span_sum if raw_span_sum > 0 else 1.0 / len(grouped_rows)
        assigned_depth_span = float(total_depth_span * ratio) if np.isfinite(total_depth_span) else float(raw_span)
        assigned_time_span = max(float(group["TIME"].iloc[-1]) - float(group["TIME"].iloc[0]), 0.0)
        fragment_rows.append(
            {
                **base_row,
                "FragmentID": f"{fragment_id_prefix}::F{fragment_index:03d}",
                "AssignmentSource": assignment_source,
                "BlockX": int(group["BlockX"].iloc[0]),
                "BlockY": int(group["BlockY"].iloc[0]),
                "UnitID": str(group["UnitID"].iloc[0]),
                "AssignedDepthStart": float(group["TVD"].iloc[0]),
                "AssignedDepthEnd": float(group["TVD"].iloc[-1]),
                "AssignedDepthSpan": assigned_depth_span,
                "AssignedDepthSpanRaw": float(raw_span),
                "AssignedTimeStart": float(group["TIME"].iloc[0]),
                "AssignedTimeEnd": float(group["TIME"].iloc[-1]),
                "AssignedTimeSpan": assigned_time_span,
                "CenterDepth": float(group["TVD"].mean()),
                "CenterTime": float(group["TIME"].mean()),
                "CenterX": float(group["X"].mean()),
                "CenterY": float(group["Y"].mean()),
                "PathSampleCount": int(len(group)),
                "AssignedLengthRatio": float(ratio),
            }
        )
    return fragment_rows


def build_fragments_from_hints(
    hint_df: pd.DataFrame,
    base_row: dict[str, object],
    fragment_id_prefix: str,
    total_depth_span: float,
) -> list[dict[str, object]]:
    if hint_df is None or hint_df.empty:
        return []
    total_points = float(hint_df["HintPointCount"].sum())
    if total_points <= 0:
        total_points = float(len(hint_df))
    fragment_rows: list[dict[str, object]] = []
    for fragment_index, hint in enumerate(hint_df.itertuples(index=False), start=1):
        ratio = (float(hint.HintPointCount) / total_points) if total_points > 0 else 1.0 / len(hint_df)
        fragment_rows.append(
            {
                **base_row,
                "FragmentID": f"{fragment_id_prefix}::F{fragment_index:03d}",
                "AssignmentSource": "point_hint",
                "BlockX": int(hint.BlockX),
                "BlockY": int(hint.BlockY),
                "UnitID": str(hint.UnitID),
                "AssignedDepthStart": float(hint.HintDepthMin),
                "AssignedDepthEnd": float(hint.HintDepthMax),
                "AssignedDepthSpan": float(total_depth_span * ratio) if np.isfinite(total_depth_span) else np.nan,
                "AssignedDepthSpanRaw": max(float(hint.HintDepthMax) - float(hint.HintDepthMin), 0.0),
                "AssignedTimeStart": float(hint.HintTimeMin),
                "AssignedTimeEnd": float(hint.HintTimeMax),
                "AssignedTimeSpan": max(float(hint.HintTimeMax) - float(hint.HintTimeMin), 0.0),
                "CenterDepth": float(hint.HintCenterDepth),
                "CenterTime": float(hint.HintCenterTime),
                "CenterX": float(hint.HintCenterX),
                "CenterY": float(hint.HintCenterY),
                "PathSampleCount": int(hint.HintPointCount),
                "AssignedLengthRatio": float(ratio),
            }
        )
    return fragment_rows


def split_segments_to_units(
    segment_master_df: pd.DataFrame,
    trajectory_map: dict[str, pd.DataFrame],
    point_hints_df: pd.DataFrame,
    grid: GridIndex,
) -> pd.DataFrame:
    hint_lookup = {
        key: group.reset_index(drop=True)
        for key, group in point_hints_df.groupby("SegmentKey")
    } if not point_hints_df.empty else {}
    fragments: list[dict[str, object]] = []

    for row in segment_master_df.itertuples(index=False):
        base = row._asdict()
        total_depth_span = float(base.get("SegLength") or 0.0)
        fragment_prefix = str(base["SegmentKey"])
        path = extract_interval_path(
            trajectory_map.get(str(base["WellName"])),
            float(base["SegStartDepth"]),
            float(base["SegEndDepth"]),
            grid,
        )
        if not path.empty:
            fragment_rows = build_fragments_from_path(path, base, fragment_prefix, total_depth_span, "trajectory")
        else:
            fragment_rows = build_fragments_from_hints(hint_lookup.get(fragment_prefix), base, fragment_prefix, total_depth_span)
        for fragment in fragment_rows:
            ratio = float(fragment["AssignedLengthRatio"])
            pred_point_count = pd.to_numeric(base.get("PredPointCount"), errors="coerce")
            pred_p10_mass = pd.to_numeric(base.get("PredP10Mass"), errors="coerce")
            fragment["AssignedSegLength"] = float(fragment["AssignedDepthSpan"])
            fragment["AssignedPredPointCountFloat"] = float(pred_point_count * ratio) if np.isfinite(pred_point_count) else np.nan
            fragment["AssignedPredP10Mass"] = float(pred_p10_mass * ratio) if np.isfinite(pred_p10_mass) else np.nan
            fragment["AssignedPredP10MassPerLength"] = base.get("PredP10MassPerLength")
            fragment["AssignedPredDensityStrength"] = base.get("PredDensityStrength")
            fragment["OrientationWeight"] = (
                fragment["AssignedPredP10Mass"]
                if np.isfinite(fragment["AssignedPredP10Mass"])
                else fragment["AssignedSegLength"]
            )
        fragments.extend(fragment_rows)

    work = pd.DataFrame(fragments)
    if not work.empty:
        work = work.rename(columns={"FragmentID": "SegmentFragmentID"})
    return work


def split_intervals_to_units(
    interval_master_df: pd.DataFrame,
    trajectory_map: dict[str, pd.DataFrame],
    point_hints_df: pd.DataFrame,
    grid: GridIndex,
) -> pd.DataFrame:
    hint_lookup = {
        key: group.reset_index(drop=True)
        for key, group in point_hints_df.groupby("IntervalKey")
    } if not point_hints_df.empty else {}
    fragments: list[dict[str, object]] = []

    for row in interval_master_df.itertuples(index=False):
        base = row._asdict()
        total_depth_span = max(float(base["GeoDepthMax"]) - float(base["GeoDepthMin"]), 0.0)
        fragment_prefix = str(base["IntervalKey"])
        path = extract_interval_path(
            trajectory_map.get(str(base["WellName"])),
            float(base["GeoDepthMin"]),
            float(base["GeoDepthMax"]),
            grid,
        )
        if not path.empty:
            fragment_rows = build_fragments_from_path(path, base, fragment_prefix, total_depth_span, "trajectory")
        else:
            fragment_rows = build_fragments_from_hints(hint_lookup.get(fragment_prefix), base, fragment_prefix, total_depth_span)
        for fragment in fragment_rows:
            fragment["AssignedIntervalThickness"] = float(fragment["AssignedDepthSpan"])
        fragments.extend(fragment_rows)

    work = pd.DataFrame(fragments)
    if not work.empty:
        work = work.rename(columns={"FragmentID": "IntervalFragmentID"})
    return work


def join_unique_strings(values: pd.Series) -> str:
    items = sorted({str(item).strip() for item in values if str(item).strip()})
    return ",".join(items)


def merge_string_lists(*values: object) -> str:
    items: set[str] = set()
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        for token in str(value).split(","):
            token = token.strip()
            if token:
                items.add(token)
    return ",".join(sorted(items))


def ensure_column(df: pd.DataFrame, col: str, default: object) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    else:
        df[col] = df[col].fillna(default)
    return df


def build_unit_layer_seed_library(
    unit_points_df: pd.DataFrame,
    unit_segments_df: pd.DataFrame,
    unit_intervals_df: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = ["UnitID", "BlockX", "BlockY", "GeoIntervalKey", "StrataName", "TopSurfaceCode", "BaseSurfaceCode"]

    point_stats = pd.DataFrame()
    if not unit_points_df.empty:
        point_stats = (
            unit_points_df.groupby(group_cols, as_index=False)
            .agg(
                PointRecordCount=("FracturePointKey", "count"),
                PointWellCount=("WellName", "nunique"),
                PointDensityMassSum=("PointDensityMassAllocated", "sum"),
                PointDepthMin=("TVD", "min"),
                PointDepthMax=("TVD", "max"),
                PointTimeMin=("TIME", "min"),
                PointTimeMax=("TIME", "max"),
                PointSupportWells=("WellName", join_unique_strings),
            )
        )

    segment_stats = pd.DataFrame()
    if not unit_segments_df.empty:
        segment_stats = (
            unit_segments_df.groupby(group_cols, as_index=False)
            .agg(
                SegmentFragmentCount=("SegmentFragmentID", "count"),
                SegmentWellCount=("WellName", "nunique"),
                SegmentAssignedLengthSum=("AssignedSegLength", "sum"),
                SegmentP10MassSum=("AssignedPredP10Mass", "sum"),
                SegmentDepthMin=("AssignedDepthStart", "min"),
                SegmentDepthMax=("AssignedDepthEnd", "max"),
                SegmentTimeMin=("AssignedTimeStart", "min"),
                SegmentTimeMax=("AssignedTimeEnd", "max"),
                SegmentSupportWells=("WellName", join_unique_strings),
            )
        )

    interval_stats = pd.DataFrame()
    if not unit_intervals_df.empty:
        interval_stats = (
            unit_intervals_df.groupby(group_cols, as_index=False)
            .agg(
                IntervalFragmentCount=("IntervalFragmentID", "count"),
                IntervalWellCount=("WellName", "nunique"),
                IntervalThicknessSum=("AssignedIntervalThickness", "sum"),
                IntervalDepthMin=("AssignedDepthStart", "min"),
                IntervalDepthMax=("AssignedDepthEnd", "max"),
                IntervalTimeMin=("AssignedTimeStart", "min"),
                IntervalTimeMax=("AssignedTimeEnd", "max"),
                IntervalSupportWells=("WellName", join_unique_strings),
            )
        )

    orient_frames: list[pd.DataFrame] = []
    if not unit_points_df.empty:
        point_orient = unit_points_df[group_cols + ["PointAzimuth", "PointDip", "OrientationWeight"]].copy()
        point_orient = point_orient.rename(
            columns={"PointAzimuth": "Azimuth", "PointDip": "Dip", "OrientationWeight": "Weight"}
        )
        orient_frames.append(point_orient)
    if not unit_segments_df.empty:
        segment_orient = unit_segments_df[group_cols + ["PredAzimuth", "PredDip", "OrientationWeight"]].copy()
        segment_orient = segment_orient.rename(
            columns={"PredAzimuth": "Azimuth", "PredDip": "Dip", "OrientationWeight": "Weight"}
        )
        orient_frames.append(segment_orient)

    orientation_stats = pd.DataFrame(columns=group_cols + ["MeanAzimuth", "MeanDip", "OrientationSampleCount"])
    if orient_frames:
        orient_df = pd.concat(orient_frames, ignore_index=True)
        rows: list[dict[str, object]] = []
        for key, group in orient_df.groupby(group_cols, dropna=False):
            azimuth, dip = weighted_orientation(group, "Azimuth", "Dip", "Weight")
            rows.append(
                {
                    **dict(zip(group_cols, key)),
                    "MeanAzimuth": azimuth,
                    "MeanDip": dip,
                    "OrientationSampleCount": int(len(group)),
                }
            )
        orientation_stats = pd.DataFrame(rows)

    merged = point_stats.copy()
    for frame in [segment_stats, interval_stats, orientation_stats]:
        if merged.empty:
            merged = frame.copy()
        elif not frame.empty:
            merged = merged.merge(frame, on=group_cols, how="outer")

    if merged.empty:
        return merged

    for col in group_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    for col in [
        "PointRecordCount",
        "SegmentFragmentCount",
        "IntervalFragmentCount",
        "PointWellCount",
        "SegmentWellCount",
        "IntervalWellCount",
        "OrientationSampleCount",
    ]:
        merged = ensure_column(merged, col, 0)
        merged[col] = merged[col].astype(int)
    for col in [
        "PointDensityMassSum",
        "SegmentAssignedLengthSum",
        "SegmentP10MassSum",
        "IntervalThicknessSum",
    ]:
        merged = ensure_column(merged, col, np.nan)
    merged["DepthMin"] = merged[
        [col for col in ["IntervalDepthMin", "PointDepthMin", "SegmentDepthMin"] if col in merged.columns]
    ].min(axis=1, skipna=True)
    merged["DepthMax"] = merged[
        [col for col in ["IntervalDepthMax", "PointDepthMax", "SegmentDepthMax"] if col in merged.columns]
    ].max(axis=1, skipna=True)
    merged["TimeMin"] = merged[
        [col for col in ["IntervalTimeMin", "PointTimeMin", "SegmentTimeMin"] if col in merged.columns]
    ].min(axis=1, skipna=True)
    merged["TimeMax"] = merged[
        [col for col in ["IntervalTimeMax", "PointTimeMax", "SegmentTimeMax"] if col in merged.columns]
    ].max(axis=1, skipna=True)
    merged["SupportWells"] = merged.apply(
        lambda row: merge_string_lists(
            row.get("PointSupportWells"),
            row.get("SegmentSupportWells"),
            row.get("IntervalSupportWells"),
        ),
        axis=1,
    )
    merged["SeedEvidenceClass"] = "interval_only"
    merged.loc[(merged["PointRecordCount"] > 0) & (merged["SegmentFragmentCount"] == 0), "SeedEvidenceClass"] = "point_only"
    merged.loc[(merged["PointRecordCount"] == 0) & (merged["SegmentFragmentCount"] > 0), "SeedEvidenceClass"] = "segment_only"
    merged.loc[(merged["PointRecordCount"] > 0) & (merged["SegmentFragmentCount"] > 0), "SeedEvidenceClass"] = "point_segment"
    return merged.sort_values(["BlockX", "BlockY", "GeoIntervalKey"]).reset_index(drop=True)


def compute_nearest_support_distance(unit_index_df: pd.DataFrame, support_unit_ids: set[str]) -> np.ndarray:
    if unit_index_df.empty:
        return np.array([], dtype=float)
    if not support_unit_ids:
        return np.full(len(unit_index_df), np.nan, dtype=float)
    support_xy = (
        unit_index_df[unit_index_df["UnitID"].isin(support_unit_ids)][["XCenter", "YCenter"]]
        .drop_duplicates()
        .to_numpy(dtype=float)
    )
    target_xy = unit_index_df[["XCenter", "YCenter"]].to_numpy(dtype=float)
    distances = np.empty(len(target_xy), dtype=float)
    for idx, point in enumerate(target_xy):
        delta = support_xy - point
        distances[idx] = float(np.sqrt((delta * delta).sum(axis=1)).min())
    return distances


def build_reliability_mask(
    unit_index_df: pd.DataFrame,
    trajectory_map: dict[str, pd.DataFrame],
    unit_points_df: pd.DataFrame,
    unit_segments_df: pd.DataFrame,
    unit_intervals_df: pd.DataFrame,
    near_support_distance_m: float,
) -> pd.DataFrame:
    reliability = unit_index_df.copy()
    fracture_units = set(unit_points_df["UnitID"].dropna().astype(str).tolist()) | set(
        unit_segments_df["UnitID"].dropna().astype(str).tolist()
    )
    interval_units = set(unit_intervals_df["UnitID"].dropna().astype(str).tolist())
    trajectory_units: set[str] = set()
    for trajectory_df in trajectory_map.values():
        trajectory_units.update(trajectory_df["UnitID"].dropna().astype(str).tolist())
    well_units = interval_units | trajectory_units

    reliability["HasRealFracture"] = reliability["UnitID"].isin(fracture_units)
    reliability["HasWellCoverage"] = reliability["UnitID"].isin(well_units)
    reliability["NearestWellDistanceM"] = compute_nearest_support_distance(reliability, well_units)
    reliability["NearestFractureDistanceM"] = compute_nearest_support_distance(reliability, fracture_units)
    reliability["ReliabilityClass"] = "far_from_well"
    reliability["ReliabilityRank"] = 1
    reliability.loc[
        ~reliability["HasRealFracture"]
        & ~reliability["HasWellCoverage"]
        & reliability["NearestWellDistanceM"].notna()
        & (reliability["NearestWellDistanceM"] <= float(near_support_distance_m)),
        ["ReliabilityClass", "ReliabilityRank"],
    ] = ["near_well_supported", 2]
    reliability.loc[reliability["HasWellCoverage"], ["ReliabilityClass", "ReliabilityRank"]] = ["well_controlled", 3]
    reliability.loc[reliability["HasRealFracture"], ["ReliabilityClass", "ReliabilityRank"]] = ["real_controlled", 4]
    return reliability


def build_compact_unit_layers(unit_layer_seed_library_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "UnitID",
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
    if unit_layer_seed_library_df.empty:
        return pd.DataFrame(columns=columns)

    layers = unit_layer_seed_library_df.copy()
    layers["RealPointSeedCount"] = coalesce_series(layers, ["PointRecordCount"], default=0).fillna(0).astype(int)
    layers["RealSegmentSeedCount"] = coalesce_series(layers, ["SegmentFragmentCount"], default=0).fillna(0).astype(int)
    layers["VirtualPointSeedCount"] = 0
    layers["VirtualSegmentSeedCount"] = 0
    layers["RealSeedCount"] = layers["RealPointSeedCount"] + layers["RealSegmentSeedCount"]
    layers["VirtualSeedCount"] = 0
    layers["TopDepth"] = coalesce_series(layers, ["DepthMin"])
    layers["BaseDepth"] = coalesce_series(layers, ["DepthMax"])
    layers["TopTime"] = coalesce_series(layers, ["TimeMin"])
    layers["BaseTime"] = coalesce_series(layers, ["TimeMax"])
    layers["PreferredSource"] = np.where(layers["RealSeedCount"] > 0, "real", "seismic_fill")
    return layers[columns].sort_values(["BlockX", "BlockY", "GeoIntervalKey"]).reset_index(drop=True)


def build_compact_fracture_seeds(
    unit_points_df: pd.DataFrame,
    unit_segments_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "UnitID",
        "BlockX",
        "BlockY",
        "GeoIntervalKey",
        "StrataName",
        "TopSurfaceCode",
        "BaseSurfaceCode",
        "SeedID",
        "SourceKind",
        "SourceName",
        "SeedType",
        "CenterX",
        "CenterY",
        "CenterDepth",
        "CenterTime",
        "DepthStart",
        "DepthEnd",
        "TimeStart",
        "TimeEnd",
        "Azimuth",
        "Dip",
        "DensityWeight",
        "LengthWeight",
        "Confidence",
    ]
    frames: list[pd.DataFrame] = []

    if not unit_points_df.empty:
        point_cols = [
            "UnitID",
            "BlockX",
            "BlockY",
            "GeoIntervalKey",
            "StrataName",
            "TopSurfaceCode",
            "BaseSurfaceCode",
            "FracturePointKey",
            "WellName",
            "X",
            "Y",
            "TVD",
            "TIME",
            "SegStartDepth",
            "SegEndDepth",
            "PointAzimuth",
            "PointDip",
            "PointDensityMassAllocated",
            "PointDensityMassPerLengthAllocated",
            "PredOrientationConfidence",
        ]
        points = unit_points_df[point_cols].copy()
        points = points.rename(
            columns={
                "FracturePointKey": "SeedID",
                "WellName": "SourceName",
                "X": "CenterX",
                "Y": "CenterY",
                "TVD": "CenterDepth",
                "TIME": "CenterTime",
                "SegStartDepth": "DepthStart",
                "SegEndDepth": "DepthEnd",
                "PointAzimuth": "Azimuth",
                "PointDip": "Dip",
                "PointDensityMassAllocated": "DensityWeight",
                "PointDensityMassPerLengthAllocated": "LengthWeight",
                "PredOrientationConfidence": "Confidence",
            }
        )
        points["TimeStart"] = points["CenterTime"]
        points["TimeEnd"] = points["CenterTime"]
        points["SourceKind"] = "real"
        points["SeedType"] = "point"
        frames.append(points[columns])

    if not unit_segments_df.empty:
        segment_cols = [
            "UnitID",
            "BlockX",
            "BlockY",
            "GeoIntervalKey",
            "StrataName",
            "TopSurfaceCode",
            "BaseSurfaceCode",
            "SegmentFragmentID",
            "WellName",
            "CenterX",
            "CenterY",
            "CenterDepth",
            "CenterTime",
            "AssignedDepthStart",
            "AssignedDepthEnd",
            "AssignedTimeStart",
            "AssignedTimeEnd",
            "PredAzimuth",
            "PredDip",
            "AssignedPredP10Mass",
            "AssignedSegLength",
            "PredOrientationConfidence",
        ]
        segments = unit_segments_df[segment_cols].copy()
        segments = segments.rename(
            columns={
                "SegmentFragmentID": "SeedID",
                "WellName": "SourceName",
                "AssignedDepthStart": "DepthStart",
                "AssignedDepthEnd": "DepthEnd",
                "AssignedTimeStart": "TimeStart",
                "AssignedTimeEnd": "TimeEnd",
                "PredAzimuth": "Azimuth",
                "PredDip": "Dip",
                "AssignedPredP10Mass": "DensityWeight",
                "AssignedSegLength": "LengthWeight",
                "PredOrientationConfidence": "Confidence",
            }
        )
        segments["SourceKind"] = "real"
        segments["SeedType"] = "segment"
        frames.append(segments[columns])

    if not frames:
        return pd.DataFrame(columns=columns)
    seeds = pd.concat(frames, ignore_index=True)
    return seeds.sort_values(["BlockX", "BlockY", "GeoIntervalKey", "SeedType", "SeedID"]).reset_index(drop=True)


def build_unit_catalog(
    unit_index_df: pd.DataFrame,
    unit_reliability_mask_df: pd.DataFrame,
    compact_unit_layers_df: pd.DataFrame,
    compact_fracture_seeds_df: pd.DataFrame,
) -> pd.DataFrame:
    catalog = unit_index_df.copy()
    reliability_cols = [
        "UnitID",
        "ReliabilityClass",
        "ReliabilityRank",
        "HasRealFracture",
        "HasWellCoverage",
        "NearestWellDistanceM",
        "NearestFractureDistanceM",
    ]
    if not unit_reliability_mask_df.empty:
        catalog = catalog.merge(unit_reliability_mask_df[reliability_cols], on="UnitID", how="left")

    if compact_unit_layers_df.empty:
        layer_stats = pd.DataFrame(
            columns=["UnitID", "LayerCount", "CoveredIntervalCount", "TopDepth", "BaseDepth", "TopTime", "BaseTime"]
        )
    else:
        layer_stats = (
            compact_unit_layers_df.groupby("UnitID", as_index=False)
            .agg(
                LayerCount=("GeoIntervalKey", "count"),
                CoveredIntervalCount=("GeoIntervalKey", "nunique"),
                TopDepth=("TopDepth", "min"),
                BaseDepth=("BaseDepth", "max"),
                TopTime=("TopTime", "min"),
                BaseTime=("BaseTime", "max"),
            )
        )
    catalog = catalog.merge(layer_stats, on="UnitID", how="left")

    if compact_fracture_seeds_df.empty:
        seed_stats = pd.DataFrame(
            columns=[
                "UnitID",
                "SeedCount",
                "RealSeedCount",
                "VirtualSeedCount",
                "RealPointSeedCount",
                "RealSegmentSeedCount",
                "VirtualPointSeedCount",
                "VirtualSegmentSeedCount",
            ]
        )
    else:
        seeds = compact_fracture_seeds_df.copy()
        seeds["RealSeedCount"] = seeds["SourceKind"].eq("real").astype(int)
        seeds["VirtualSeedCount"] = seeds["SourceKind"].eq("virtual").astype(int)
        seeds["RealPointSeedCount"] = ((seeds["SourceKind"] == "real") & (seeds["SeedType"] == "point")).astype(int)
        seeds["RealSegmentSeedCount"] = ((seeds["SourceKind"] == "real") & (seeds["SeedType"] == "segment")).astype(int)
        seeds["VirtualPointSeedCount"] = ((seeds["SourceKind"] == "virtual") & (seeds["SeedType"] == "point")).astype(int)
        seeds["VirtualSegmentSeedCount"] = ((seeds["SourceKind"] == "virtual") & (seeds["SeedType"] == "segment")).astype(int)
        seed_stats = (
            seeds.groupby("UnitID", as_index=False)
            .agg(
                SeedCount=("SeedID", "count"),
                RealSeedCount=("RealSeedCount", "sum"),
                VirtualSeedCount=("VirtualSeedCount", "sum"),
                RealPointSeedCount=("RealPointSeedCount", "sum"),
                RealSegmentSeedCount=("RealSegmentSeedCount", "sum"),
                VirtualPointSeedCount=("VirtualPointSeedCount", "sum"),
                VirtualSegmentSeedCount=("VirtualSegmentSeedCount", "sum"),
            )
        )
    catalog = catalog.merge(seed_stats, on="UnitID", how="left")

    for col in [
        "LayerCount",
        "CoveredIntervalCount",
        "SeedCount",
        "RealSeedCount",
        "VirtualSeedCount",
        "RealPointSeedCount",
        "RealSegmentSeedCount",
        "VirtualPointSeedCount",
        "VirtualSegmentSeedCount",
    ]:
        catalog = ensure_column(catalog, col, 0)
        catalog[col] = catalog[col].astype(int)
    for col in ["TopDepth", "BaseDepth", "TopTime", "BaseTime"]:
        catalog = ensure_column(catalog, col, np.nan)

    catalog["HasRealData"] = (catalog["LayerCount"] > 0) | (catalog["RealSeedCount"] > 0)
    catalog["HasVirtualData"] = catalog["VirtualSeedCount"] > 0
    catalog["DataMode"] = "empty"
    catalog.loc[catalog["HasRealData"] & ~catalog["HasVirtualData"], "DataMode"] = "real_only"
    catalog.loc[~catalog["HasRealData"] & catalog["HasVirtualData"], "DataMode"] = "virtual_only"
    catalog.loc[catalog["HasRealData"] & catalog["HasVirtualData"], "DataMode"] = "real_virtual"
    catalog["PackageRequired"] = catalog["HasRealData"] | catalog["HasVirtualData"]
    catalog["PackageDir"] = np.where(catalog["PackageRequired"], "units/" + catalog["UnitID"].astype(str), "")
    return catalog.sort_values(["BlockX", "BlockY"]).reset_index(drop=True)


def materialize_unit_packages(
    run_output_dir: Path,
    unit_catalog_df: pd.DataFrame,
    compact_unit_layers_df: pd.DataFrame,
    compact_fracture_seeds_df: pd.DataFrame,
) -> int:
    units_root = run_output_dir / "units"
    packaged_count = 0
    layer_columns = [col for col in compact_unit_layers_df.columns if col != "UnitID"]
    seed_columns = [col for col in compact_fracture_seeds_df.columns if col != "UnitID"]
    layer_groups = (
        {unit_id: group.drop(columns=["UnitID"]).reset_index(drop=True) for unit_id, group in compact_unit_layers_df.groupby("UnitID")}
        if not compact_unit_layers_df.empty
        else {}
    )
    seed_groups = (
        {unit_id: group.drop(columns=["UnitID"]).reset_index(drop=True) for unit_id, group in compact_fracture_seeds_df.groupby("UnitID")}
        if not compact_fracture_seeds_df.empty
        else {}
    )

    for row in unit_catalog_df[unit_catalog_df["PackageRequired"]].itertuples(index=False):
        unit_id = str(row.UnitID)
        unit_dir = units_root / unit_id
        unit_dir.mkdir(parents=True, exist_ok=True)
        layer_df = layer_groups.get(unit_id, pd.DataFrame(columns=layer_columns))
        seed_df = seed_groups.get(unit_id, pd.DataFrame(columns=seed_columns))

        write_csv_utf8(layer_df, unit_dir / "unit_layers.csv")
        write_csv_utf8(seed_df, unit_dir / "fracture_seeds.csv")

        meta = {
            "UnitID": unit_id,
            "BlockX": int(row.BlockX),
            "BlockY": int(row.BlockY),
            "ReliabilityClass": str(row.ReliabilityClass),
            "HasRealData": bool(row.HasRealData),
            "HasVirtualData": bool(row.HasVirtualData),
            "DataMode": str(row.DataMode),
            "LayerCount": int(row.LayerCount),
            "CoveredIntervalCount": int(row.CoveredIntervalCount),
            "SeedCount": int(row.SeedCount),
            "RealSeedCount": int(row.RealSeedCount),
            "VirtualSeedCount": int(row.VirtualSeedCount),
            "RealPointSeedCount": int(row.RealPointSeedCount),
            "RealSegmentSeedCount": int(row.RealSegmentSeedCount),
            "VirtualPointSeedCount": int(row.VirtualPointSeedCount),
            "VirtualSegmentSeedCount": int(row.VirtualSegmentSeedCount),
            "TopDepth": None if pd.isna(row.TopDepth) else float(row.TopDepth),
            "BaseDepth": None if pd.isna(row.BaseDepth) else float(row.BaseDepth),
            "TopTime": None if pd.isna(row.TopTime) else float(row.TopTime),
            "BaseTime": None if pd.isna(row.BaseTime) else float(row.BaseTime),
            "Files": ["unit_layers.csv", "fracture_seeds.csv"],
        }
        (unit_dir / "unit_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        packaged_count += 1

    return packaged_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real-well fracture-to-unit assignment tables for stage 2 DFN.")
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--inclined-around-dir", type=Path, default=DEFAULT_INCLINED_AROUND_DIR)
    parser.add_argument("--vertical-around-dir", type=Path, default=DEFAULT_VERTICAL_AROUND_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="real_well_unit_assignment")
    parser.add_argument("--well-names", type=str, default="")
    parser.add_argument("--max-wells", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--block-x-start", type=int, default=None)
    parser.add_argument("--block-x-end", type=int, default=None)
    parser.add_argument("--block-y-start", type=int, default=None)
    parser.add_argument("--block-y-end", type=int, default=None)
    parser.add_argument("--near-support-distance-m", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_output_dir = args.output_root / args.run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    global_output_dir = run_output_dir / "00_global"
    global_output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid_index(GridConfig(args.trace_header_csv, block_size=args.block_size, stride=args.stride))
    unit_index = build_unit_index(grid, args.block_x_start, args.block_x_end, args.block_y_start, args.block_y_end)
    selected_unit_ids = set(unit_index["UnitID"].astype(str).tolist())

    all_points, all_segments, all_intervals = load_aggregate_tables(args.aggregate_dir)
    target_wells = resolve_target_wells(
        all_points,
        all_segments,
        all_intervals,
        parse_well_names(args.well_names),
        args.max_wells,
    )
    points, segments, intervals = filter_frames_by_wells(all_points, all_segments, all_intervals, target_wells)

    trajectory_catalog = build_trajectory_catalog(args.inclined_around_dir, args.vertical_around_dir, target_wells)
    trajectory_map, trajectory_summary = load_trajectories(trajectory_catalog, grid)
    well_source_map = build_well_source_group_map(points, segments, trajectory_summary)

    point_master = prepare_point_master(points, grid, well_source_map)
    segment_point_hints = build_point_based_hints(point_master, "SegmentKey", "FracturePointKey")
    interval_point_hints = build_point_based_hints(point_master, "IntervalKey", "FracturePointKey")
    segment_master = prepare_segment_master(segments, trajectory_map, segment_point_hints, grid, well_source_map)
    interval_master = prepare_interval_master(intervals, trajectory_map, interval_point_hints, grid, well_source_map)

    unit_points = point_master[point_master["UnitID"].isin(selected_unit_ids)].copy()
    unit_segments = split_segments_to_units(segment_master, trajectory_map, segment_point_hints, grid)
    unit_intervals = split_intervals_to_units(interval_master, trajectory_map, interval_point_hints, grid)
    if not unit_segments.empty:
        unit_segments = unit_segments[unit_segments["UnitID"].isin(selected_unit_ids)].reset_index(drop=True)
    if not unit_intervals.empty:
        unit_intervals = unit_intervals[unit_intervals["UnitID"].isin(selected_unit_ids)].reset_index(drop=True)

    unit_layer_seed_library = build_unit_layer_seed_library(unit_points, unit_segments, unit_intervals)
    unit_reliability_mask = build_reliability_mask(
        unit_index,
        trajectory_map,
        unit_points,
        unit_segments,
        unit_intervals,
        args.near_support_distance_m,
    )
    compact_unit_layers = build_compact_unit_layers(unit_layer_seed_library)
    compact_fracture_seeds = build_compact_fracture_seeds(unit_points, unit_segments)
    unit_catalog = build_unit_catalog(unit_index, unit_reliability_mask, compact_unit_layers, compact_fracture_seeds)
    packaged_unit_count = materialize_unit_packages(
        run_output_dir,
        unit_catalog,
        compact_unit_layers,
        compact_fracture_seeds,
    )

    write_csv_utf8(unit_index, global_output_dir / "unit_index.csv")
    write_csv_utf8(trajectory_summary, global_output_dir / "well_trajectory_summary.csv")
    write_csv_utf8(point_master, global_output_dir / "fracture_point_master.csv")
    write_csv_utf8(segment_master, global_output_dir / "fracture_segment_master.csv")
    write_csv_utf8(interval_master, global_output_dir / "well_interval_master.csv")
    write_csv_utf8(unit_points, global_output_dir / "unit_fracture_points_real.csv")
    write_csv_utf8(unit_segments, global_output_dir / "unit_fracture_segments_real.csv")
    write_csv_utf8(unit_intervals, global_output_dir / "unit_well_intervals_real.csv")
    write_csv_utf8(unit_layer_seed_library, global_output_dir / "unit_layer_seed_library.csv")
    write_csv_utf8(unit_reliability_mask, global_output_dir / "unit_reliability_mask.csv")
    write_csv_utf8(compact_unit_layers, global_output_dir / "compact_unit_layers.csv")
    write_csv_utf8(compact_fracture_seeds, global_output_dir / "compact_fracture_seeds.csv")
    write_csv_utf8(unit_catalog, run_output_dir / "unit_catalog.csv")

    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trace_header_csv": str(args.trace_header_csv),
        "aggregate_dir": str(args.aggregate_dir),
        "inclined_around_dir": str(args.inclined_around_dir),
        "vertical_around_dir": str(args.vertical_around_dir),
        "output_dir": str(run_output_dir),
        "well_count_requested": len(target_wells),
        "trajectory_count_loaded": int(len(trajectory_map)),
        "unit_count": int(len(unit_index)),
        "fracture_point_count": int(len(point_master)),
        "fracture_segment_count": int(len(segment_master)),
        "interval_count": int(len(interval_master)),
        "unit_point_count": int(len(unit_points)),
        "unit_segment_fragment_count": int(len(unit_segments)),
        "unit_interval_fragment_count": int(len(unit_intervals)),
        "seed_library_count": int(len(unit_layer_seed_library)),
        "reliability_mask_count": int(len(unit_reliability_mask)),
        "compact_layer_count": int(len(compact_unit_layers)),
        "compact_seed_count": int(len(compact_fracture_seeds)),
        "unit_catalog_count": int(len(unit_catalog)),
        "packaged_unit_count": int(packaged_unit_count),
        "args": {
            "well_names": args.well_names,
            "max_wells": args.max_wells,
            "block_x_start": args.block_x_start,
            "block_x_end": args.block_x_end,
            "block_y_start": args.block_y_start,
            "block_y_end": args.block_y_end,
            "near_support_distance_m": args.near_support_distance_m,
        },
    }
    (run_output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
