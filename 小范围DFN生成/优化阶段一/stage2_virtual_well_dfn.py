from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (
            (candidate / "README.md").exists() and (candidate / "模型训练").exists()
        ):
            return candidate
    raise FileNotFoundError(f"Failed to locate project root from: {current}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_TRACE_HEADER_CSV = PROJECT_ROOT / "数据精简" / "trace_header_xy.csv"


POINT_NUMERIC_COLS = [
    "TVD",
    "NearestSampleDepth",
    "X",
    "Y",
    "TIME",
    "SegStartDepth",
    "SegEndDepth",
    "SegLength",
    "PredPointCountFloat",
    "PredPointCount",
    "PredProbMean",
    "PredProbMax",
    "StrataDepthMin",
    "StrataDepthMax",
    "SegmentDensityMassBudget",
    "SegmentDensityMassPerLength",
    "PointSupportLeftDepth",
    "PointSupportRightDepth",
    "PointSupportLength",
    "PointDensityWeightRaw",
    "PointDensityWeightNorm",
    "PointDensityMassAllocated",
    "PointDensityMassPerLengthAllocated",
    "PointOrientationTemplateValue",
    "PointOrientationConfidenceScale",
    "PointAzimuthStdRefDeg",
    "PointDipStdRefDeg",
    "PointAzimuthOffsetDeg",
    "PointDipOffsetDeg",
    "PointAzimuth",
    "PointDip",
    "PredDensityStrength",
    "PredP10Mean",
    "PredP10MassPerLength",
    "PredP10Mass",
    "PredDensitySizeScale",
    "PredOrientationConfidence",
    "PredAzimuth",
    "PredDip",
]


SEGMENT_NUMERIC_COLS = [
    "SegStartDepth",
    "SegEndDepth",
    "SegLength",
    "ProbMean",
    "ProbMax",
    "AC_MeanInPredSegment",
    "GR_MeanInPredSegment",
    "StrataDepthMin",
    "StrataDepthMax",
    "PredPointCountFloat",
    "PredDensityStrength",
    "PredP10Mean",
    "PredP10MassPerLength",
    "PredP10Mass",
    "PredDensitySizeScale",
    "PredOrientationConfidence",
    "PredAzimuth",
    "PredDip",
    "PredPointCount",
]


STRATA_NUMERIC_COLS = ["DepthMin", "DepthMax", "StrataTop", "StrataBase"]


@dataclass(frozen=True)
class GridConfig:
    trace_header_csv: Path
    block_size: int = 25
    stride: int = 24
    cache_json: Path | None = None
    chunksize: int = 200_000


@dataclass(frozen=True)
class VirtualWellConfig:
    unit_neighbor_steps: int = 1
    depth_step: float = 5.0
    search_radius: float = 350.0
    max_neighbors: int = 6
    relative_window: float = 0.12
    include_covered_units: bool = False
    max_target_units: int = 0


@dataclass(frozen=True)
class PatchConfig:
    density_mode: str = "hybrid"
    base_length: float = 8.0
    base_height: float = 4.0
    density_size_gain: float = 0.35
    density_count_scale: float = 1.25
    max_replicates: int = 3
    size_scale_min: float = 0.45
    size_scale_max: float = 3.5
    plane_jitter_fraction: float = 0.18


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

    @property
    def dx(self) -> float:
        return float(np.median(np.diff(self.x_coords))) if len(self.x_coords) > 1 else float("nan")

    @property
    def dy(self) -> float:
        return float(np.median(np.diff(self.y_coords))) if len(self.y_coords) > 1 else float("nan")

    def nearest_index(self, coords: np.ndarray, value: float) -> int:
        idx = int(np.searchsorted(coords, value))
        if idx <= 0:
            return 0
        if idx >= len(coords):
            return len(coords) - 1
        before = coords[idx - 1]
        after = coords[idx]
        return idx - 1 if abs(value - before) <= abs(value - after) else idx

    def locate_block(self, x: float, y: float) -> tuple[int, int]:
        x_idx = self.nearest_index(self.x_coords, x)
        y_idx = self.nearest_index(self.y_coords, y)
        bx = min(self.num_blocks_x - 1, max(0, x_idx // self.stride))
        by = min(self.num_blocks_y - 1, max(0, y_idx // self.stride))
        return bx, by

    def unit_id(self, block_x: int, block_y: int) -> str:
        return f"BX{block_x:03d}_BY{block_y:03d}"

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


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coalesce_series(df: pd.DataFrame, candidates: list[str], default: float | str | None = np.nan) -> pd.Series:
    series: pd.Series | None = None
    for col in candidates:
        if col not in df.columns:
            continue
        if series is None:
            series = df[col]
        else:
            series = series.combine_first(df[col])
    if series is None:
        return pd.Series(default, index=df.index)
    return series.fillna(default)


def compute_relative_position(depth: pd.Series, top: pd.Series, base: pd.Series) -> pd.Series:
    result = pd.Series(0.5, index=depth.index, dtype=float)
    span = base - top
    valid = span.notna() & (span.abs() > 1e-8) & depth.notna() & top.notna()
    result.loc[valid] = ((depth.loc[valid] - top.loc[valid]) / span.loc[valid]).clip(0.0, 1.0)
    return result


def build_grid_index(config: GridConfig) -> GridIndex:
    cache_json = config.cache_json
    if cache_json and cache_json.exists():
        payload = json.loads(cache_json.read_text(encoding="utf-8"))
        return GridIndex(
            x_coords=np.array(payload["x_coords"], dtype=float),
            y_coords=np.array(payload["y_coords"], dtype=float),
            block_size=int(payload["block_size"]),
            stride=int(payload["stride"]),
        )

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
        raise ValueError(f"Failed to build grid index from: {config.trace_header_csv}")

    grid = GridIndex(
        x_coords=np.array(sorted(x_values), dtype=float),
        y_coords=np.array(sorted(y_values), dtype=float),
        block_size=config.block_size,
        stride=config.stride,
    )
    if cache_json:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(
            json.dumps(
                {
                    "trace_header_csv": str(config.trace_header_csv),
                    "block_size": config.block_size,
                    "stride": config.stride,
                    "x_coords": grid.x_coords.tolist(),
                    "y_coords": grid.y_coords.tolist(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return grid


def resolve_deployment_paths(deployment_dir: Path) -> tuple[Path, Path, Path | None]:
    points_csv = deployment_dir / "whole_well_pred_fracture_points.csv"
    segments_csv = deployment_dir / "whole_well_pred_segment_summary.csv"
    strata_csv = deployment_dir / "holdout_segments" / "holdout_strata_range.csv"
    if not points_csv.exists():
        raise FileNotFoundError(f"Missing points csv: {points_csv}")
    if not segments_csv.exists():
        raise FileNotFoundError(f"Missing segments csv: {segments_csv}")
    return points_csv, segments_csv, strata_csv if strata_csv.exists() else None


def load_deployments(
    deployment_dirs: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_points: list[pd.DataFrame] = []
    all_segments: list[pd.DataFrame] = []
    all_strata: list[pd.DataFrame] = []

    for deployment_dir in deployment_dirs:
        points_csv, segments_csv, strata_csv = resolve_deployment_paths(deployment_dir)

        points_df = ensure_numeric(read_csv_utf8(points_csv), POINT_NUMERIC_COLS)
        points_df["SourceDeploymentDir"] = str(deployment_dir)
        all_points.append(points_df)

        segments_df = ensure_numeric(read_csv_utf8(segments_csv), SEGMENT_NUMERIC_COLS)
        segments_df["SourceDeploymentDir"] = str(deployment_dir)
        all_segments.append(segments_df)

        if strata_csv is not None:
            strata_df = ensure_numeric(read_csv_utf8(strata_csv), STRATA_NUMERIC_COLS)
            strata_df["SourceDeploymentDir"] = str(deployment_dir)
            all_strata.append(strata_df)

    points = pd.concat(all_points, ignore_index=True).drop_duplicates()
    segments = pd.concat(all_segments, ignore_index=True).drop_duplicates()
    if all_strata:
        strata = pd.concat(all_strata, ignore_index=True).drop_duplicates()
    else:
        strata = pd.DataFrame(columns=["WellName", "StrataName", "DepthMin", "DepthMax", "SourceDeploymentDir"])
    return points, segments, strata


def prepare_source_frames(
    points_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    strata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = points_df.copy()
    segments = segments_df.copy()
    strata = strata_df.copy()

    if "TVD" not in points.columns:
        points["TVD"] = coalesce_series(points, ["NearestSampleDepth"])
    if "StrataDepthMin" not in points.columns:
        points["StrataDepthMin"] = np.nan
    if "StrataDepthMax" not in points.columns:
        points["StrataDepthMax"] = np.nan
    points["RelPos"] = compute_relative_position(points["TVD"], points["StrataDepthMin"], points["StrataDepthMax"])

    well_anchor = (
        points.groupby("WellName", as_index=False)
        .agg(AnchorX=("X", "mean"), AnchorY=("Y", "mean"))
        .dropna(subset=["AnchorX", "AnchorY"])
    )

    strata_from_points = (
        points.groupby(["WellName", "StrataName"], as_index=False)
        .agg(
            AnchorX=("X", "mean"),
            AnchorY=("Y", "mean"),
            TimeMin=("TIME", "min"),
            TimeMax=("TIME", "max"),
            DepthMinFromPoints=("TVD", "min"),
            DepthMaxFromPoints=("TVD", "max"),
            NumPointSources=("Segment_ID", "count"),
        )
    )

    if strata.empty:
        strata = (
            points.groupby(["WellName", "StrataName", "SourceDeploymentDir"], as_index=False)
            .agg(
                DepthMin=("StrataDepthMin", "min"),
                DepthMax=("StrataDepthMax", "max"),
            )
        )

    strata = strata.merge(strata_from_points, on=["WellName", "StrataName"], how="outer")
    strata["DepthMin"] = coalesce_series(strata, ["DepthMin", "StrataTop", "DepthMinFromPoints"])
    strata["DepthMax"] = coalesce_series(strata, ["DepthMax", "StrataBase", "DepthMaxFromPoints"])
    strata["AnchorX"] = coalesce_series(strata, ["AnchorX"])
    strata["AnchorY"] = coalesce_series(strata, ["AnchorY"])
    strata["StrataThickness"] = strata["DepthMax"] - strata["DepthMin"]

    segments = segments.merge(well_anchor, on="WellName", how="left")
    if "StrataDepthMin" not in segments.columns:
        segments["StrataDepthMin"] = np.nan
    if "StrataDepthMax" not in segments.columns:
        segments["StrataDepthMax"] = np.nan
    segments["SegMidDepth"] = (segments["SegStartDepth"] + segments["SegEndDepth"]) / 2.0
    segments["RelPos"] = compute_relative_position(
        segments["SegMidDepth"],
        segments["StrataDepthMin"],
        segments["StrataDepthMax"],
    )

    return points, segments, strata


def map_points_to_units(points_df: pd.DataFrame, grid: GridIndex) -> pd.DataFrame:
    points = points_df.copy()
    block_xy = points.apply(lambda row: grid.locate_block(float(row["X"]), float(row["Y"])), axis=1)
    points["BlockX"] = [item[0] for item in block_xy]
    points["BlockY"] = [item[1] for item in block_xy]
    points["UnitID"] = [grid.unit_id(int(bx), int(by)) for bx, by in zip(points["BlockX"], points["BlockY"])]
    return points


def collect_target_units(
    mapped_points_df: pd.DataFrame,
    grid: GridIndex,
    config: VirtualWellConfig,
) -> pd.DataFrame:
    covered_units = {
        (int(row.BlockX), int(row.BlockY))
        for row in mapped_points_df[["BlockX", "BlockY"]].drop_duplicates().itertuples(index=False)
    }
    target_units: set[tuple[int, int]] = set()
    for block_x, block_y in covered_units:
        for dx in range(-config.unit_neighbor_steps, config.unit_neighbor_steps + 1):
            for dy in range(-config.unit_neighbor_steps, config.unit_neighbor_steps + 1):
                bx = block_x + dx
                by = block_y + dy
                if 0 <= bx < grid.num_blocks_x and 0 <= by < grid.num_blocks_y:
                    target_units.add((bx, by))

    rows = []
    for block_x, block_y in sorted(target_units):
        row = grid.block_bounds(block_x, block_y)
        row["CoveredByRealPoint"] = int((block_x, block_y) in covered_units)
        rows.append(row)
    unit_df = pd.DataFrame(rows)
    if config.max_target_units and len(unit_df) > config.max_target_units:
        uncovered = unit_df[unit_df["CoveredByRealPoint"] == 0]
        covered = unit_df[unit_df["CoveredByRealPoint"] == 1]
        keep = pd.concat(
            [
                uncovered.head(config.max_target_units),
                covered.head(max(0, config.max_target_units - len(uncovered.head(config.max_target_units)))),
            ],
            ignore_index=True,
        )
        unit_df = keep.drop_duplicates(subset=["UnitID"]).reset_index(drop=True)
    return unit_df


def build_candidate_weights(
    df: pd.DataFrame,
    target_x: float,
    target_y: float,
    rel_pos: float | None,
    max_neighbors: int,
    radius: float,
    relative_window: float,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    work = df.copy()
    work = work.dropna(subset=[x_col, y_col])
    if work.empty:
        return work

    work["XYDistance"] = np.sqrt((work[x_col] - target_x) ** 2 + (work[y_col] - target_y) ** 2)
    within_radius = work[work["XYDistance"] <= radius]
    if not within_radius.empty:
        work = within_radius

    if rel_pos is not None and "RelPos" in work.columns:
        work["RelDistance"] = (work["RelPos"] - rel_pos).abs()
        within_rel = work[work["RelDistance"] <= relative_window]
        if not within_rel.empty:
            work = within_rel
        else:
            work = work.nsmallest(max_neighbors * 4, "RelDistance")
    else:
        work["RelDistance"] = 0.0

    work = work.nsmallest(max_neighbors, ["XYDistance", "RelDistance"]).copy()
    work["Weight"] = 1.0 / np.power(work["XYDistance"] + 1.0, 2.0) / np.power(work["RelDistance"] + 0.05, 1.0)
    return work


def weighted_numeric(candidates: pd.DataFrame, value_col: str) -> float:
    if value_col not in candidates.columns:
        return float("nan")
    values = pd.to_numeric(candidates[value_col], errors="coerce")
    valid = values.notna() & candidates["Weight"].notna()
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=candidates.loc[valid, "Weight"]))


def azimuth_dip_to_normal(azimuth_deg: np.ndarray, dip_deg: np.ndarray) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    dip = np.deg2rad(dip_deg)
    return np.column_stack(
        [
            np.sin(dip) * np.sin(az),
            np.sin(dip) * np.cos(az),
            np.cos(dip),
        ]
    )


def weighted_orientation(
    candidates: pd.DataFrame,
    azimuth_col: str = "PredAzimuth",
    dip_col: str = "PredDip",
) -> tuple[float, float]:
    if azimuth_col not in candidates.columns or dip_col not in candidates.columns:
        return float("nan"), float("nan")
    az = pd.to_numeric(candidates[azimuth_col], errors="coerce")
    dip = pd.to_numeric(candidates[dip_col], errors="coerce")
    valid = az.notna() & dip.notna() & candidates["Weight"].notna()
    if not valid.any():
        return float("nan"), float("nan")
    normals = azimuth_dip_to_normal(az.loc[valid].to_numpy(), dip.loc[valid].to_numpy())
    normal = np.average(normals, axis=0, weights=candidates.loc[valid, "Weight"].to_numpy())
    norm = np.linalg.norm(normal)
    if norm <= 1e-8:
        return float("nan"), float("nan")
    normal = normal / norm
    azimuth = math.degrees(math.atan2(normal[0], normal[1])) % 360.0
    dip_angle = math.degrees(math.acos(np.clip(normal[2], -1.0, 1.0)))
    return float(azimuth), float(dip_angle)


def interpolate_strata_interval(
    strata_df: pd.DataFrame,
    target_x: float,
    target_y: float,
    strata_name: str,
    config: VirtualWellConfig,
) -> dict[str, float] | None:
    candidates = strata_df[strata_df["StrataName"] == strata_name].copy()
    candidates = candidates.dropna(subset=["AnchorX", "AnchorY", "DepthMin", "DepthMax"])
    if candidates.empty:
        return None

    candidates["XYDistance"] = np.sqrt((candidates["AnchorX"] - target_x) ** 2 + (candidates["AnchorY"] - target_y) ** 2)
    within_radius = candidates[candidates["XYDistance"] <= config.search_radius]
    if not within_radius.empty:
        candidates = within_radius
    candidates = candidates.nsmallest(config.max_neighbors, "XYDistance").copy()
    candidates["Weight"] = 1.0 / np.power(candidates["XYDistance"] + 1.0, 2.0)

    depth_min = weighted_numeric(candidates, "DepthMin")
    depth_max = weighted_numeric(candidates, "DepthMax")
    time_min = weighted_numeric(candidates, "TimeMin")
    time_max = weighted_numeric(candidates, "TimeMax")
    thickness = depth_max - depth_min if np.isfinite(depth_min) and np.isfinite(depth_max) else float("nan")
    if not np.isfinite(thickness) or thickness <= 0:
        return None

    return {
        "DepthMin": float(depth_min),
        "DepthMax": float(depth_max),
        "TimeMin": float(time_min) if np.isfinite(time_min) else float("nan"),
        "TimeMax": float(time_max) if np.isfinite(time_max) else float("nan"),
        "SupportCount": int(len(candidates)),
        "SupportWells": ",".join(sorted(candidates["WellName"].dropna().astype(str).unique().tolist())),
    }


def interpolate_virtual_well_samples(
    target_units_df: pd.DataFrame,
    strata_df: pd.DataFrame,
    points_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    config: VirtualWellConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strata_names = [item for item in sorted(strata_df["StrataName"].dropna().astype(str).unique().tolist()) if item]

    for unit in target_units_df.itertuples(index=False):
        if unit.CoveredByRealPoint and not config.include_covered_units:
            continue

        virtual_well_name = f"VIRT_{unit.UnitID}"
        for strata_name in strata_names:
            interval = interpolate_strata_interval(strata_df, unit.XCenter, unit.YCenter, strata_name, config)
            if interval is None:
                continue

            depth_samples = np.arange(interval["DepthMin"], interval["DepthMax"] + config.depth_step / 2.0, config.depth_step)
            if len(depth_samples) == 0:
                continue

            span = interval["DepthMax"] - interval["DepthMin"]
            for sample_idx, sample_depth in enumerate(depth_samples, start=1):
                rel_pos = 0.5 if span <= 1e-8 else float(np.clip((sample_depth - interval["DepthMin"]) / span, 0.0, 1.0))
                point_candidates = build_candidate_weights(
                    points_df[points_df["StrataName"] == strata_name],
                    unit.XCenter,
                    unit.YCenter,
                    rel_pos,
                    config.max_neighbors,
                    config.search_radius,
                    config.relative_window,
                    x_col="X",
                    y_col="Y",
                )
                segment_candidates = build_candidate_weights(
                    segments_df[segments_df["StrataName"] == strata_name],
                    unit.XCenter,
                    unit.YCenter,
                    rel_pos,
                    config.max_neighbors,
                    config.search_radius,
                    config.relative_window,
                    x_col="AnchorX",
                    y_col="AnchorY",
                )

                ac_value = weighted_numeric(segment_candidates, "AC_MeanInPredSegment")
                gr_value = weighted_numeric(segment_candidates, "GR_MeanInPredSegment")
                density_strength = weighted_numeric(point_candidates, "PredDensityStrength")
                if not np.isfinite(density_strength):
                    density_strength = weighted_numeric(segment_candidates, "PredDensityStrength")
                density_size_scale = weighted_numeric(point_candidates, "PredDensitySizeScale")
                if not np.isfinite(density_size_scale):
                    density_size_scale = weighted_numeric(segment_candidates, "PredDensitySizeScale")
                orientation_conf = weighted_numeric(point_candidates, "PredOrientationConfidence")
                if not np.isfinite(orientation_conf):
                    orientation_conf = weighted_numeric(segment_candidates, "PredOrientationConfidence")
                azimuth, dip = weighted_orientation(point_candidates)
                if not np.isfinite(azimuth) or not np.isfinite(dip):
                    azimuth, dip = weighted_orientation(segment_candidates)

                time_value = float("nan")
                if np.isfinite(interval["TimeMin"]) and np.isfinite(interval["TimeMax"]):
                    time_value = float(interval["TimeMin"] + rel_pos * (interval["TimeMax"] - interval["TimeMin"]))

                source_wells = sorted(
                    set(point_candidates.get("WellName", pd.Series(dtype=str)).dropna().astype(str).tolist())
                    | set(segment_candidates.get("WellName", pd.Series(dtype=str)).dropna().astype(str).tolist())
                )
                rows.append(
                    {
                        "VirtualWellName": virtual_well_name,
                        "UnitID": unit.UnitID,
                        "BlockX": int(unit.BlockX),
                        "BlockY": int(unit.BlockY),
                        "CoveredByRealPoint": int(unit.CoveredByRealPoint),
                        "X": float(unit.XCenter),
                        "Y": float(unit.YCenter),
                        "StrataName": strata_name,
                        "SampleSeq": int(sample_idx),
                        "SampleTVD": float(sample_depth),
                        "SampleTIME": time_value,
                        "SampleRelPos": rel_pos,
                        "VirtualAC": ac_value,
                        "VirtualGR": gr_value,
                        "PredDensityStrength": density_strength,
                        "PredDensitySizeScale": density_size_scale,
                        "PredOrientationConfidence": orientation_conf,
                        "PredAzimuth": azimuth,
                        "PredDip": dip,
                        "SourceStrataSupportCount": int(interval["SupportCount"]),
                        "SourcePointSupportCount": int(len(point_candidates)),
                        "SourceSegmentSupportCount": int(len(segment_candidates)),
                        "SourceWellNames": ",".join(source_wells),
                        "SourceStrataWells": interval["SupportWells"],
                    }
                )

    return pd.DataFrame(rows)


def build_control_seed_points(points_df: pd.DataFrame) -> pd.DataFrame:
    control = points_df.copy()
    control["SourceKind"] = "real_point"
    control["SourceName"] = control["WellName"].astype(str)
    control["DepthValue"] = coalesce_series(control, ["TVD", "NearestSampleDepth"])
    control["DensityStrength"] = coalesce_series(
        control,
        ["PredDensityStrength", "PointDensityMassPerLengthAllocated", "PredP10MassPerLength"],
        default=0.0,
    )
    control["DensitySizeScale"] = coalesce_series(control, ["PredDensitySizeScale"], default=1.0)
    control["OrientationConfidence"] = coalesce_series(
        control,
        ["PredOrientationConfidence", "PointOrientationConfidenceScale"],
        default=0.0,
    )
    control["Azimuth"] = coalesce_series(control, ["PredAzimuth", "PointAzimuth"])
    control["Dip"] = coalesce_series(control, ["PredDip", "PointDip"])
    control["AC"] = np.nan
    control["GR"] = np.nan
    control["SupportCount"] = 1
    keep_cols = [
        "SourceKind",
        "SourceName",
        "WellName",
        "UnitID",
        "BlockX",
        "BlockY",
        "X",
        "Y",
        "StrataName",
        "DepthValue",
        "TIME",
        "RelPos",
        "DensityStrength",
        "DensitySizeScale",
        "OrientationConfidence",
        "Azimuth",
        "Dip",
        "AC",
        "GR",
        "PredPointCount",
        "PredPointCountFloat",
        "SupportCount",
        "PredOrientationFamily",
        "PredDensityStrengthLevel",
    ]
    return control[keep_cols].rename(columns={"TIME": "TimeValue"})


def build_virtual_seed_points(virtual_df: pd.DataFrame) -> pd.DataFrame:
    if virtual_df.empty:
        return pd.DataFrame(
            columns=[
                "SourceKind",
                "SourceName",
                "WellName",
                "UnitID",
                "BlockX",
                "BlockY",
                "X",
                "Y",
                "StrataName",
                "DepthValue",
                "TimeValue",
                "RelPos",
                "DensityStrength",
                "DensitySizeScale",
                "OrientationConfidence",
                "Azimuth",
                "Dip",
                "AC",
                "GR",
                "PredPointCount",
                "PredPointCountFloat",
                "SupportCount",
                "PredOrientationFamily",
                "PredDensityStrengthLevel",
            ]
        )

    virtual = virtual_df.copy()
    virtual["SourceKind"] = "virtual_sample"
    virtual["SourceName"] = virtual["VirtualWellName"].astype(str)
    virtual["WellName"] = virtual["VirtualWellName"].astype(str)
    virtual["DepthValue"] = virtual["SampleTVD"]
    virtual["TimeValue"] = virtual["SampleTIME"]
    virtual["RelPos"] = virtual["SampleRelPos"]
    virtual["DensityStrength"] = coalesce_series(virtual, ["PredDensityStrength"], default=0.0)
    virtual["DensitySizeScale"] = coalesce_series(virtual, ["PredDensitySizeScale"], default=1.0)
    virtual["OrientationConfidence"] = coalesce_series(virtual, ["PredOrientationConfidence"], default=0.0)
    virtual["Azimuth"] = coalesce_series(virtual, ["PredAzimuth"])
    virtual["Dip"] = coalesce_series(virtual, ["PredDip"])
    virtual["AC"] = virtual["VirtualAC"]
    virtual["GR"] = virtual["VirtualGR"]
    virtual["PredPointCount"] = np.nan
    virtual["PredPointCountFloat"] = np.nan
    virtual["SupportCount"] = virtual["SourcePointSupportCount"] + virtual["SourceSegmentSupportCount"]
    virtual["PredOrientationFamily"] = np.nan
    virtual["PredDensityStrengthLevel"] = pd.cut(
        virtual["DensityStrength"],
        bins=[-np.inf, 0.5, 1.0, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    keep_cols = [
        "SourceKind",
        "SourceName",
        "WellName",
        "UnitID",
        "BlockX",
        "BlockY",
        "X",
        "Y",
        "StrataName",
        "DepthValue",
        "TimeValue",
        "RelPos",
        "DensityStrength",
        "DensitySizeScale",
        "OrientationConfidence",
        "Azimuth",
        "Dip",
        "AC",
        "GR",
        "PredPointCount",
        "PredPointCountFloat",
        "SupportCount",
        "PredOrientationFamily",
        "PredDensityStrengthLevel",
    ]
    return virtual[keep_cols]


def build_unit_constraint_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()
    summary = (
        seed_df.groupby(["UnitID", "BlockX", "BlockY", "StrataName", "SourceKind"], as_index=False)
        .agg(
            SeedCount=("SourceName", "count"),
            MeanDensityStrength=("DensityStrength", "mean"),
            MeanDensitySizeScale=("DensitySizeScale", "mean"),
            MeanAzimuth=("Azimuth", "mean"),
            MeanDip=("Dip", "mean"),
            MeanOrientationConfidence=("OrientationConfidence", "mean"),
            MeanAC=("AC", "mean"),
            MeanGR=("GR", "mean"),
            DepthMin=("DepthValue", "min"),
            DepthMax=("DepthValue", "max"),
            TimeMin=("TimeValue", "min"),
            TimeMax=("TimeValue", "max"),
        )
    )
    return summary


def plane_basis_from_orientation(azimuth_deg: float, dip_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = math.radians(float(azimuth_deg))
    dip = math.radians(float(dip_deg))
    normal = np.array(
        [
            math.sin(dip) * math.sin(az),
            math.sin(dip) * math.cos(az),
            math.cos(dip),
        ],
        dtype=float,
    )
    norm = np.linalg.norm(normal)
    if norm <= 1e-8:
        raise ValueError("Invalid orientation normal")
    normal = normal / norm
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.allclose(np.abs(np.dot(normal, up)), 1.0, atol=1e-6):
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    u_vec = np.cross(normal, up)
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    return normal, u_vec, v_vec


def build_patch_vertices(center: np.ndarray, u_vec: np.ndarray, v_vec: np.ndarray, length: float, height: float) -> np.ndarray:
    half_u = (length / 2.0) * u_vec
    half_v = (height / 2.0) * v_vec
    return np.array(
        [
            center - half_u - half_v,
            center + half_u - half_v,
            center + half_u + half_v,
            center - half_u + half_v,
        ],
        dtype=float,
    )


def determine_patch_geometry(
    seed_row: pd.Series,
    unit_row: pd.Series,
    config: PatchConfig,
) -> tuple[float, float, int]:
    size_scale = float(seed_row.get("DensitySizeScale", 1.0))
    if not np.isfinite(size_scale):
        size_scale = 1.0
    size_scale = float(np.clip(size_scale, config.size_scale_min, config.size_scale_max))

    strength = float(seed_row.get("DensityStrength", 0.0))
    if not np.isfinite(strength):
        strength = 0.0

    unit_span = max(1.0, min(float(unit_row["XMax"] - unit_row["XMin"]), float(unit_row["YMax"] - unit_row["YMin"])))
    length = config.base_length * size_scale
    height = config.base_height * math.sqrt(size_scale)

    if config.density_mode in {"size", "hybrid"}:
        length *= 1.0 + config.density_size_gain * max(0.0, strength)
        height *= 1.0 + 0.5 * config.density_size_gain * max(0.0, strength)

    length = min(length, unit_span * 0.9)
    height = min(height, unit_span * 0.75)

    replicate_count = 1
    if config.density_mode in {"count", "hybrid"}:
        replicate_count = max(1, min(config.max_replicates, int(round(max(0.0, strength) * config.density_count_scale))))

    return float(length), float(height), int(replicate_count)


def build_dfn_patches(
    seed_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    config: PatchConfig,
) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()

    unit_lookup = unit_df.set_index("UnitID")
    patch_rows: list[dict[str, object]] = []
    patch_counter = 1

    for seed in seed_df.itertuples(index=False):
        azimuth = float(seed.Azimuth) if pd.notna(seed.Azimuth) else float("nan")
        dip = float(seed.Dip) if pd.notna(seed.Dip) else float("nan")
        if not np.isfinite(azimuth) or not np.isfinite(dip):
            continue
        if seed.UnitID not in unit_lookup.index:
            continue
        unit_row = unit_lookup.loc[seed.UnitID]

        length, height, replicate_count = determine_patch_geometry(pd.Series(seed._asdict()), unit_row, config)
        normal, u_vec, v_vec = plane_basis_from_orientation(azimuth, dip)
        center = np.array([float(seed.X), float(seed.Y), float(seed.TimeValue)], dtype=float)

        for replicate_idx in range(replicate_count):
            offset_scale = 0.0
            if replicate_count > 1:
                offset_scale = (
                    replicate_idx - (replicate_count - 1) / 2.0
                ) * config.plane_jitter_fraction * min(length, height)
            shifted_center = center + offset_scale * u_vec
            vertices = build_patch_vertices(shifted_center, u_vec, v_vec, length, height)

            row = {
                "PatchID": f"PATCH_{patch_counter:06d}",
                "UnitID": seed.UnitID,
                "BlockX": int(seed.BlockX),
                "BlockY": int(seed.BlockY),
                "SourceKind": seed.SourceKind,
                "SourceName": seed.SourceName,
                "WellName": seed.WellName,
                "StrataName": seed.StrataName,
                "CenterX": float(shifted_center[0]),
                "CenterY": float(shifted_center[1]),
                "CenterTIME": float(shifted_center[2]),
                "DepthValue": float(seed.DepthValue) if pd.notna(seed.DepthValue) else float("nan"),
                "Azimuth": azimuth,
                "Dip": dip,
                "OrientationConfidence": float(seed.OrientationConfidence) if pd.notna(seed.OrientationConfidence) else float("nan"),
                "DensityStrength": float(seed.DensityStrength) if pd.notna(seed.DensityStrength) else float("nan"),
                "DensitySizeScale": float(seed.DensitySizeScale) if pd.notna(seed.DensitySizeScale) else float("nan"),
                "PatchLength": length,
                "PatchHeight": height,
                "ReplicateIndex": replicate_idx + 1,
                "ReplicateCount": replicate_count,
                "NormalX": float(normal[0]),
                "NormalY": float(normal[1]),
                "NormalZ": float(normal[2]),
            }
            for vertex_idx, vertex in enumerate(vertices, start=1):
                row[f"V{vertex_idx}X"] = float(vertex[0])
                row[f"V{vertex_idx}Y"] = float(vertex[1])
                row[f"V{vertex_idx}Z"] = float(vertex[2])
            patch_rows.append(row)
            patch_counter += 1

    return pd.DataFrame(patch_rows)


def build_patch_summary(patch_df: pd.DataFrame) -> pd.DataFrame:
    if patch_df.empty:
        return pd.DataFrame()
    return (
        patch_df.groupby(["UnitID", "BlockX", "BlockY", "StrataName", "SourceKind"], as_index=False)
        .agg(
            PatchCount=("PatchID", "count"),
            MeanLength=("PatchLength", "mean"),
            MeanHeight=("PatchHeight", "mean"),
            MeanDensityStrength=("DensityStrength", "mean"),
            MeanAzimuth=("Azimuth", "mean"),
            MeanDip=("Dip", "mean"),
        )
        .sort_values(["BlockX", "BlockY", "StrataName", "SourceKind"])
    )


def save_outputs(
    output_dir: Path,
    grid: GridIndex,
    unit_df: pd.DataFrame,
    mapped_points_df: pd.DataFrame,
    virtual_well_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    unit_constraint_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    patch_summary_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    unit_df.to_csv(output_dir / "unit_index.csv", index=False, encoding="utf-8-sig")
    mapped_points_df.to_csv(output_dir / "mapped_control_points.csv", index=False, encoding="utf-8-sig")
    virtual_well_df.to_csv(output_dir / "virtual_well_samples.csv", index=False, encoding="utf-8-sig")
    seed_df.to_csv(output_dir / "dfn_seed_points.csv", index=False, encoding="utf-8-sig")
    unit_constraint_df.to_csv(output_dir / "unit_constraint_summary.csv", index=False, encoding="utf-8-sig")
    patch_df.to_csv(output_dir / "unit_dfn_patches.csv", index=False, encoding="utf-8-sig")
    patch_summary_df.to_csv(output_dir / "unit_dfn_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "deployment_dirs": [str(Path(item).resolve()) for item in args.deployment_dir],
        "trace_header_csv": str(Path(args.trace_header_csv).resolve()),
        "block_size": int(args.block_size),
        "stride": int(args.stride),
        "num_unique_x": int(len(grid.x_coords)),
        "num_unique_y": int(len(grid.y_coords)),
        "num_blocks_x": int(grid.num_blocks_x),
        "num_blocks_y": int(grid.num_blocks_y),
        "dx_median": float(grid.dx),
        "dy_median": float(grid.dy),
        "num_target_units": int(len(unit_df)),
        "num_control_points": int(len(mapped_points_df)),
        "num_virtual_samples": int(len(virtual_well_df)),
        "num_seed_points": int(len(seed_df)),
        "num_dfn_patches": int(len(patch_df)),
        "density_mode": args.density_mode,
        "virtual_depth_step": float(args.virtual_depth_step),
        "unit_neighbor_steps": int(args.unit_neighbor_steps),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于第一部分输出构建虚拟测井样本，并生成单元级 DFN 裂缝片。",
    )
    parser.add_argument(
        "--deployment-dir",
        action="append",
        required=True,
        help="第一部分部署结果目录，可重复传入多个目录。",
    )
    parser.add_argument(
        "--trace-header-csv",
        default=str(DEFAULT_TRACE_HEADER_CSV),
        help="trace_header_xy.csv 路径。",
    )
    parser.add_argument("--output-dir", required=True, help="输出目录。")
    parser.add_argument("--grid-cache-json", default="", help="可选的网格缓存 json。")
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--unit-neighbor-steps", type=int, default=1)
    parser.add_argument("--virtual-depth-step", type=float, default=5.0)
    parser.add_argument("--search-radius", type=float, default=350.0)
    parser.add_argument("--max-neighbors", type=int, default=6)
    parser.add_argument("--relative-window", type=float, default=0.12)
    parser.add_argument("--include-covered-units", action="store_true")
    parser.add_argument("--max-target-units", type=int, default=0)
    parser.add_argument("--density-mode", choices=["size", "count", "hybrid"], default="hybrid")
    parser.add_argument("--base-length", type=float, default=8.0)
    parser.add_argument("--base-height", type=float, default=4.0)
    parser.add_argument("--density-size-gain", type=float, default=0.35)
    parser.add_argument("--density-count-scale", type=float, default=1.25)
    parser.add_argument("--max-replicates", type=int, default=3)
    parser.add_argument("--plane-jitter-fraction", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    deployment_dirs = [Path(item).resolve() for item in args.deployment_dir]
    trace_header_csv = Path(args.trace_header_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    grid_cache_json = Path(args.grid_cache_json).resolve() if args.grid_cache_json else output_dir / "trace_header_grid_summary.json"

    grid = build_grid_index(
        GridConfig(
            trace_header_csv=trace_header_csv,
            block_size=args.block_size,
            stride=args.stride,
            cache_json=grid_cache_json,
        )
    )

    points_df, segments_df, strata_df = load_deployments(deployment_dirs)
    points_df, segments_df, strata_df = prepare_source_frames(points_df, segments_df, strata_df)
    mapped_points_df = map_points_to_units(points_df, grid)

    virtual_config = VirtualWellConfig(
        unit_neighbor_steps=args.unit_neighbor_steps,
        depth_step=args.virtual_depth_step,
        search_radius=args.search_radius,
        max_neighbors=args.max_neighbors,
        relative_window=args.relative_window,
        include_covered_units=args.include_covered_units,
        max_target_units=args.max_target_units,
    )
    unit_df = collect_target_units(mapped_points_df, grid, virtual_config)
    virtual_well_df = interpolate_virtual_well_samples(unit_df, strata_df, mapped_points_df, segments_df, virtual_config)

    control_seed_df = build_control_seed_points(mapped_points_df)
    virtual_seed_df = build_virtual_seed_points(virtual_well_df)
    seed_df = pd.concat([control_seed_df, virtual_seed_df], ignore_index=True, sort=False)
    unit_constraint_df = build_unit_constraint_summary(seed_df)

    patch_config = PatchConfig(
        density_mode=args.density_mode,
        base_length=args.base_length,
        base_height=args.base_height,
        density_size_gain=args.density_size_gain,
        density_count_scale=args.density_count_scale,
        max_replicates=args.max_replicates,
        plane_jitter_fraction=args.plane_jitter_fraction,
    )
    patch_df = build_dfn_patches(seed_df, unit_df, patch_config)
    patch_summary_df = build_patch_summary(patch_df)

    save_outputs(
        output_dir=output_dir,
        grid=grid,
        unit_df=unit_df,
        mapped_points_df=mapped_points_df,
        virtual_well_df=virtual_well_df,
        seed_df=seed_df,
        unit_constraint_df=unit_constraint_df,
        patch_df=patch_df,
        patch_summary_df=patch_summary_df,
        args=args,
    )

    print(f"[OK] output dir: {output_dir}")
    print(f"[OK] target units: {len(unit_df)}")
    print(f"[OK] control points: {len(mapped_points_df)}")
    print(f"[OK] virtual samples: {len(virtual_well_df)}")
    print(f"[OK] dfn patches: {len(patch_df)}")


if __name__ == "__main__":
    main()
