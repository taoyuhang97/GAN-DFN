import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from sklearn.neighbors import KDTree, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

try:
    import segyio
except ImportError:
    segyio = None

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    OrdinaryKriging = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


DEFAULT_TRACE_HEADER = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"
)
DEFAULT_SURFACE_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\层位"
)
DEFAULT_ZONE_DATA_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\虚拟测井构建\相对深度重采样数据"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\虚拟测井构建\虚拟测井批量生成"
)
DEFAULT_OUTPUT_FILES = (
    "virtual_well_index.csv",
    "virtual_well_zone_samples.csv",
    "virtual_well_curve.csv",
    "virtual_well_around_data.csv",
    "virtual_well_strata_segmentation.csv",
    "run_summary.json",
)

BLOCK_SIZE = 25
BLOCK_STEP = BLOCK_SIZE - 1
CENTER_OFFSET = BLOCK_SIZE // 2
RESAMPLE_INTERVAL = 0.2
DEFAULT_NEIGHBORS = 12
DEFAULT_METHOD = "knn"
SUPPORTED_METHODS = ("knn", "xgboost", "kriging")
DEFAULT_SGY_FILE = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\psdm_final_time.sgy"
)
DEFAULT_SEISMIC_GRID_SPACING = 12.5
DEFAULT_SAMPLING_START_MS = 1100.0
DEFAULT_SAMPLING_INTERVAL_MS = 1.0
DEFAULT_SEISMIC_WINDOW_XY_OFFSETS = (-12.5, 0.0, 12.5)
DEFAULT_SEISMIC_WINDOW_T_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
FIRST_STEP_SURFACE_CODES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
FIRST_STEP_SURFACE_PREFERRED_NAMES = {
    "T1": "T1(馆陶底)",
    "T2": "T2(沙一下特殊岩性顶)",
    "T3": "T3(沙二底)",
    "T4": "T4(沙三上底)",
    "T5": "T5_20240715sm2+DM_Sm(沙三下顶面)",
    "T6": "T6_20240715AtoInt+DM_Sm(沙三下底面)",
    "T7": "T7_gljmAto地震+DM_Sm(沙四上底面)",
}
FIRST_STEP_SURFACE_CODE_SET = set(FIRST_STEP_SURFACE_CODES)
ZONE_NAME_ALIAS_CODES_RAW = {
    "Between_T5_20240715sm2+DM_Sm(沙三": ("between", "T5", "T6"),
}


def normalize_name(text: str) -> str:
    value = str(text).strip()
    value = value.replace("（", "(").replace("）", ")")
    value = value.replace("【", "[").replace("】", "]")
    value = value.replace("，", ",").replace("：", ":")
    value = value.replace("  ", " ")
    return value


def first_not_null(series: pd.Series):
    for value in series:
        if pd.notna(value):
            return value
    return None


def extract_surface_code(surface_name: Optional[str]) -> str:
    if not surface_name:
        return ""
    normalized = normalize_name(surface_name)
    match = re.match(r"^(T[1-7])(?:\b|[_(].*)?$", normalized)
    return match.group(1) if match else ""


def build_interval_key(top_code: str, base_code: str) -> str:
    return f"{top_code}->{base_code}"


def build_range_id(top_code: str, base_code: str) -> str:
    if not top_code and base_code:
        return f"above_{base_code}"
    if top_code and not base_code:
        return f"below_{top_code}"
    return f"between_{top_code}_{base_code}"


def snap_to_resample_interval(value: Optional[float], interval: float = RESAMPLE_INTERVAL) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(round(value / interval) * interval, 6)


def is_first_step_relevant_interval(top_code: str, base_code: str) -> bool:
    if top_code and top_code not in FIRST_STEP_SURFACE_CODE_SET:
        return False
    if base_code and base_code not in FIRST_STEP_SURFACE_CODE_SET:
        return False
    if top_code and base_code and top_code == base_code:
        return False
    return bool(top_code or base_code)


@dataclass
class SurfaceInfo:
    original_name: str
    normalized_name: str
    filepath: Path
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    interpolator: NearestNDInterpolator


@dataclass
class ZoneInfo:
    zone_name: str
    zone_path: Path
    zone_type: str
    top_surface_name: Optional[str]
    base_surface_name: Optional[str]
    top_surface_code: str
    base_surface_code: str
    interval_key: str
    range_id: str
    strata_name: str


@dataclass
class GenerationContext:
    trace_df: pd.DataFrame
    unique_x: np.ndarray
    unique_y: np.ndarray
    surface_catalog: Dict[str, SurfaceInfo]
    selected_zones: Dict[str, ZoneInfo]


def build_unit_id(block_x: int, block_y: int) -> str:
    return f"BX{block_x}_BY{block_y}"


def get_unit_output_dir(output_root: Path, block_x: int, block_y: int) -> Path:
    return output_root / build_unit_id(block_x, block_y)


def has_complete_virtual_well_output(output_root: Path, block_x: int, block_y: int) -> bool:
    unit_output_dir = get_unit_output_dir(output_root, block_x, block_y)
    if not unit_output_dir.exists():
        return False
    return all((unit_output_dir / filename).exists() for filename in DEFAULT_OUTPUT_FILES)


def read_dat_surface(filepath: Path) -> pd.DataFrame:
    data_lines: List[Tuple[float, float, float]] = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                data_lines.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue

    if not data_lines:
        raise ValueError(f"No valid XYZ rows found in surface file: {filepath}")
    return pd.DataFrame(data_lines, columns=["X", "Y", "Z"])


def load_surface_catalog(surface_dir: Path) -> Dict[str, SurfaceInfo]:
    catalog: Dict[str, SurfaceInfo] = {}
    for filepath in sorted(surface_dir.glob("*.dat")):
        if filepath.name.lower().startswith("faultstick"):
            continue
        surface_df = read_dat_surface(filepath)
        normalized_name = normalize_name(filepath.stem)
        catalog[normalized_name] = SurfaceInfo(
            original_name=filepath.stem,
            normalized_name=normalized_name,
            filepath=filepath,
            x_min=float(surface_df["X"].min()),
            x_max=float(surface_df["X"].max()),
            y_min=float(surface_df["Y"].min()),
            y_max=float(surface_df["Y"].max()),
            interpolator=NearestNDInterpolator(
                surface_df[["X", "Y"]].values,
                surface_df["Z"].values,
            ),
        )
    if not catalog:
        raise FileNotFoundError(f"No surface DAT files found under: {surface_dir}")
    return catalog


def resolve_surface_name(raw_name: Optional[str], surface_catalog: Dict[str, SurfaceInfo]) -> Optional[str]:
    if raw_name is None:
        return None
    normalized = normalize_name(raw_name)
    if normalized in surface_catalog:
        return normalized

    for candidate in surface_catalog:
        if candidate == normalized:
            return candidate
        if candidate in normalized or normalized in candidate:
            return candidate
    return None


def parse_zone_name(zone_name: str, surface_catalog: Dict[str, SurfaceInfo]) -> Optional[ZoneInfo]:
    normalized_zone_name = normalize_name(zone_name)
    surface_names = sorted(surface_catalog.keys(), key=len, reverse=True)

    for alias_name, (zone_type, top_code, base_code) in ZONE_NAME_ALIAS_CODES_RAW.items():
        if normalized_zone_name != normalize_name(alias_name):
            continue
        top_surface_name = resolve_surface_name(
            FIRST_STEP_SURFACE_PREFERRED_NAMES.get(top_code, top_code),
            surface_catalog,
        )
        base_surface_name = resolve_surface_name(
            FIRST_STEP_SURFACE_PREFERRED_NAMES.get(base_code, base_code),
            surface_catalog,
        )
        if zone_type == "between" and top_surface_name and base_surface_name:
            return ZoneInfo(
                zone_name=normalized_zone_name,
                zone_path=Path(),
                zone_type=zone_type,
                top_surface_name=top_surface_name,
                base_surface_name=base_surface_name,
                top_surface_code=top_code,
                base_surface_code=base_code,
                interval_key=build_interval_key(top_code, base_code),
                range_id=build_range_id(top_code, base_code),
                strata_name="",
            )

    if normalized_zone_name.startswith("Above_"):
        bottom_raw = normalized_zone_name[len("Above_") :]
        bottom_name = resolve_surface_name(bottom_raw, surface_catalog)
        if bottom_name is None:
            return None
        base_code = extract_surface_code(bottom_name)
        return ZoneInfo(
            zone_name=normalized_zone_name,
            zone_path=Path(),
            zone_type="above",
            top_surface_name=None,
            base_surface_name=bottom_name,
            top_surface_code="",
            base_surface_code=base_code,
            interval_key=build_interval_key("", base_code),
            range_id=build_range_id("", base_code),
            strata_name="",
        )

    if normalized_zone_name.startswith("Below_"):
        top_raw = normalized_zone_name[len("Below_") :]
        top_name = resolve_surface_name(top_raw, surface_catalog)
        if top_name is None:
            return None
        top_code = extract_surface_code(top_name)
        return ZoneInfo(
            zone_name=normalized_zone_name,
            zone_path=Path(),
            zone_type="below",
            top_surface_name=top_name,
            base_surface_name=None,
            top_surface_code=top_code,
            base_surface_code="",
            interval_key=build_interval_key(top_code, ""),
            range_id=build_range_id(top_code, ""),
            strata_name="",
        )

    if normalized_zone_name.startswith("Between_"):
        body = normalized_zone_name[len("Between_") :]
        for top_candidate in surface_names:
            prefix = f"{top_candidate}_"
            if not body.startswith(prefix):
                continue
            bottom_raw = body[len(prefix) :]
            bottom_candidate = resolve_surface_name(bottom_raw, surface_catalog)
            if bottom_candidate is None:
                continue
            top_code = extract_surface_code(top_candidate)
            base_code = extract_surface_code(bottom_candidate)
            return ZoneInfo(
                zone_name=normalized_zone_name,
                zone_path=Path(),
                zone_type="between",
                top_surface_name=top_candidate,
                base_surface_name=bottom_candidate,
                top_surface_code=top_code,
                base_surface_code=base_code,
                interval_key=build_interval_key(top_code, base_code),
                range_id=build_range_id(top_code, base_code),
                strata_name="",
            )

    return None


def load_zone_catalog(zone_data_dir: Path, surface_catalog: Dict[str, SurfaceInfo]) -> List[ZoneInfo]:
    zone_catalog: List[ZoneInfo] = []
    for filepath in sorted(zone_data_dir.glob("*_relative_depth.csv")):
        zone_name = normalize_name(filepath.stem.replace("_relative_depth", ""))
        parsed = parse_zone_name(zone_name, surface_catalog)
        if parsed is None:
            continue
        parsed.zone_path = filepath
        zone_catalog.append(parsed)
    if not zone_catalog:
        raise FileNotFoundError(f"No parsable zone CSV files found under: {zone_data_dir}")
    return zone_catalog


def build_grid_index(trace_header_file: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    trace_df = pd.read_csv(trace_header_file)
    required_cols = {"TraceIdx", "X", "Y"}
    missing_cols = required_cols.difference(trace_df.columns)
    if missing_cols:
        raise ValueError(f"Trace header is missing columns: {sorted(missing_cols)}")
    unique_x = np.sort(trace_df["X"].unique())
    unique_y = np.sort(trace_df["Y"].unique())
    if len(unique_x) < BLOCK_SIZE or len(unique_y) < BLOCK_SIZE:
        raise ValueError("Trace header does not contain enough traces for one unit.")
    return trace_df, unique_x, unique_y


def get_num_blocks(unique_coords: np.ndarray) -> int:
    return int((len(unique_coords) - 1) // BLOCK_STEP)


def locate_unit_center(
    trace_df: pd.DataFrame,
    unique_x: np.ndarray,
    unique_y: np.ndarray,
    block_x: int,
    block_y: int,
) -> Dict[str, float]:
    num_blocks_x = get_num_blocks(unique_x)
    num_blocks_y = get_num_blocks(unique_y)
    if not (0 <= block_x < num_blocks_x):
        raise ValueError(f"block_x={block_x} out of range [0, {num_blocks_x - 1}]")
    if not (0 <= block_y < num_blocks_y):
        raise ValueError(f"block_y={block_y} out of range [0, {num_blocks_y - 1}]")

    start_x_idx = block_x * BLOCK_STEP
    start_y_idx = block_y * BLOCK_STEP
    center_x_idx = start_x_idx + CENTER_OFFSET
    center_y_idx = start_y_idx + CENTER_OFFSET

    center_x = float(unique_x[center_x_idx])
    center_y = float(unique_y[center_y_idx])

    center_trace = trace_df[
        (trace_df["X"] == center_x) & (trace_df["Y"] == center_y)
    ].copy()
    if center_trace.empty:
        raise ValueError("Could not find the center trace in trace_header_xy.csv.")

    center_trace_idx = int(center_trace.iloc[0]["TraceIdx"])
    return {
        "block_x": int(block_x),
        "block_y": int(block_y),
        "unit_id": f"BX{block_x}_BY{block_y}",
        "center_x_idx": int(center_x_idx),
        "center_y_idx": int(center_y_idx),
        "center_x": center_x,
        "center_y": center_y,
        "center_trace_idx": center_trace_idx,
        "num_blocks_x": int(num_blocks_x),
        "num_blocks_y": int(num_blocks_y),
    }


def build_trace_spatial_index(trace_df: pd.DataFrame) -> Tuple[KDTree, np.ndarray]:
    trace_xy = trace_df[["X", "Y"]].to_numpy(dtype=np.float64)
    traceidx_array = trace_df["TraceIdx"].to_numpy(dtype=np.int64)
    tree = KDTree(trace_xy, leaf_size=64)
    return tree, traceidx_array


def interpolate_seismic_amplitude(
    x: float,
    y: float,
    t: float,
    tree: KDTree,
    traceidx_array: np.ndarray,
    segy_handle,
    sampling_start_ms: float,
    sampling_interval_ms: float,
    nsamp: int,
    trace_cache: Dict[int, np.ndarray],
    grid_spacing: float,
) -> float:
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(t):
        return float("nan")
    if sampling_interval_ms <= 0:
        return float("nan")

    sample_pos = (float(t) - float(sampling_start_ms)) / float(sampling_interval_ms)
    sample_idx = int(math.floor(sample_pos))
    sample_weight = sample_pos - sample_idx
    if sample_idx < 0 or sample_idx + 1 >= nsamp:
        return float("nan")

    x0 = math.floor(float(x) / grid_spacing) * grid_spacing
    y0 = math.floor(float(y) / grid_spacing) * grid_spacing
    x1 = x0 + grid_spacing
    y1 = y0 + grid_spacing
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]

    corner_amps: List[float] = []
    for corner_x, corner_y in corners:
        _, corner_idx = tree.query(np.array([[corner_x, corner_y]], dtype=np.float64), k=1)
        trace_idx = int(traceidx_array[int(corner_idx[0][0])])
        if trace_idx not in trace_cache:
            trace_cache[trace_idx] = np.asarray(segy_handle.trace[trace_idx], dtype=np.float64)
        trace_data = trace_cache[trace_idx]
        amp = (1.0 - sample_weight) * trace_data[sample_idx] + sample_weight * trace_data[sample_idx + 1]
        corner_amps.append(float(amp))

    a00, a10, a01, a11 = corner_amps
    alpha = (float(x) - x0) / grid_spacing
    beta = (float(y) - y0) / grid_spacing
    amp = (
        (1.0 - alpha) * (1.0 - beta) * a00
        + alpha * (1.0 - beta) * a10
        + (1.0 - alpha) * beta * a01
        + alpha * beta * a11
    )
    return float(amp)


def extract_seismic_window(
    x: float,
    y: float,
    t: float,
    tree: KDTree,
    traceidx_array: np.ndarray,
    segy_handle,
    sampling_start_ms: float,
    sampling_interval_ms: float,
    nsamp: int,
    trace_cache: Dict[int, np.ndarray],
    grid_spacing: float,
    xy_offsets: Tuple[float, ...],
    time_offsets: Tuple[int, ...],
) -> Tuple[float, List[float]]:
    seis_true = interpolate_seismic_amplitude(
        x=x,
        y=y,
        t=t,
        tree=tree,
        traceidx_array=traceidx_array,
        segy_handle=segy_handle,
        sampling_start_ms=sampling_start_ms,
        sampling_interval_ms=sampling_interval_ms,
        nsamp=nsamp,
        trace_cache=trace_cache,
        grid_spacing=grid_spacing,
    )
    if not np.isfinite(seis_true):
        return float("nan"), []

    window_values: List[float] = []
    for dx in xy_offsets:
        for dy in xy_offsets:
            for dt in time_offsets:
                amp = interpolate_seismic_amplitude(
                    x=x + float(dx),
                    y=y + float(dy),
                    t=t + float(dt) * float(sampling_interval_ms),
                    tree=tree,
                    traceidx_array=traceidx_array,
                    segy_handle=segy_handle,
                    sampling_start_ms=sampling_start_ms,
                    sampling_interval_ms=sampling_interval_ms,
                    nsamp=nsamp,
                    trace_cache=trace_cache,
                    grid_spacing=grid_spacing,
                )
                if not np.isfinite(amp):
                    return float("nan"), []
                window_values.append(float(amp))
    return float(seis_true), window_values


def attach_seismic_window_features(
    curve_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    sgy_file: Path,
    sampling_start_ms: float,
    sampling_interval_ms: float,
    grid_spacing: float,
    xy_offsets: Tuple[float, ...] = DEFAULT_SEISMIC_WINDOW_XY_OFFSETS,
    time_offsets: Tuple[int, ...] = DEFAULT_SEISMIC_WINDOW_T_OFFSETS,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if segyio is None:
        raise ImportError("segyio is not installed. Install it before extracting seismic windows.")
    if not sgy_file.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {sgy_file}")
    if curve_df.empty:
        return curve_df.copy(), {
            "input_curve_rows": 0,
            "seismic_curve_rows": 0,
            "seismic_skipped_rows": 0,
        }

    tree, traceidx_array = build_trace_spatial_index(trace_df)
    seismic_rows: List[dict] = []
    skipped_rows = 0
    expected_window_size = len(xy_offsets) * len(xy_offsets) * len(time_offsets)

    with segyio.open(str(sgy_file), "r", ignore_geometry=True) as segy_handle:
        segy_handle.mmap()
        nsamp = int(segy_handle.samples.size)
        trace_cache: Dict[int, np.ndarray] = {}

        for row in curve_df.to_dict(orient="records"):
            seis_true, seis_window = extract_seismic_window(
                x=float(row["X"]),
                y=float(row["Y"]),
                t=float(row["TIME"]),
                tree=tree,
                traceidx_array=traceidx_array,
                segy_handle=segy_handle,
                sampling_start_ms=sampling_start_ms,
                sampling_interval_ms=sampling_interval_ms,
                nsamp=nsamp,
                trace_cache=trace_cache,
                grid_spacing=grid_spacing,
                xy_offsets=xy_offsets,
                time_offsets=time_offsets,
            )
            if not np.isfinite(seis_true) or len(seis_window) != expected_window_size:
                skipped_rows += 1
                continue

            seismic_row = dict(row)
            seismic_row["SEIS_TRUE"] = float(seis_true)
            for idx, amp in enumerate(seis_window):
                seismic_row[f"SEIS_{idx}"] = float(amp)
            seismic_rows.append(seismic_row)

    seismic_df = pd.DataFrame(seismic_rows)
    seismic_meta = {
        "input_curve_rows": int(len(curve_df)),
        "seismic_curve_rows": int(len(seismic_df)),
        "seismic_skipped_rows": int(skipped_rows),
        "seismic_window_size": int(expected_window_size),
        "seismic_grid_spacing": float(grid_spacing),
        "sampling_start_ms": float(sampling_start_ms),
        "sampling_interval_ms": float(sampling_interval_ms),
    }
    return seismic_df, seismic_meta


def build_stage1_compatible_virtual_well(
    center_info: Dict[str, float],
    seismic_curve_df: pd.DataFrame,
) -> pd.DataFrame:
    if seismic_curve_df.empty:
        raise ValueError("No seismic-enhanced virtual-well rows remain for around-data export.")

    well_name = f"virtual_{center_info['unit_id']}"
    around_df = seismic_curve_df.copy()
    if "VirtualWellName" not in around_df.columns:
        around_df.insert(0, "VirtualWellName", well_name)
    around_df["WellName"] = well_name
    around_df["ROW_IN_WELL"] = np.arange(len(around_df), dtype=np.int64)
    if "SampleIndex" in around_df.columns:
        around_df["SampleIndex"] = np.arange(1, len(around_df) + 1, dtype=np.int64)

    time_axis = pd.to_numeric(around_df["TIME"], errors="coerce")
    around_df["DEPT"] = time_axis
    around_df["TVD"] = time_axis
    around_df["DepthForStrata"] = time_axis
    around_df["DepthAxis"] = "TIME_MS"
    around_df["StrataTop"] = around_df.get("TopSurfaceCode", pd.Series(index=around_df.index, dtype=object))
    around_df["StrataBase"] = around_df.get("BaseSurfaceCode", pd.Series(index=around_df.index, dtype=object))

    seismic_cols = ["SEIS_TRUE"] + [f"SEIS_{idx}" for idx in range(63)]
    preferred_cols = [
        "TVD",
        "TIME",
        "X",
        "Y",
        *seismic_cols,
        "DEPT",
        "AC",
        "GR",
        "ROW_IN_WELL",
        "WellName",
        "DepthForStrata",
        "StrataName",
        "StrataTop",
        "StrataBase",
        "VirtualWellName",
        "UnitID",
        "BlockX",
        "BlockY",
        "TraceIdx",
        "RangeId",
        "IntervalKey",
        "ZoneName",
        "ZoneType",
        "TopSurfaceCode",
        "TopSurfaceName",
        "BaseSurfaceCode",
        "BaseSurfaceName",
        "TopTime",
        "BaseTime",
        "DepthAxis",
        "PredictMethod",
        "SampleIndex",
    ]
    ordered_cols = [col for col in preferred_cols if col in around_df.columns]
    ordered_cols.extend(col for col in around_df.columns if col not in ordered_cols)
    return around_df[ordered_cols].reset_index(drop=True)


def build_virtual_well_strata_segmentation(
    center_info: Dict[str, float],
    actual_surface_order: List[Tuple[str, str, float]],
) -> pd.DataFrame:
    rows: List[dict] = []
    well_name = f"virtual_{center_info['unit_id']}"
    if len(actual_surface_order) < 2:
        return pd.DataFrame(
            columns=[
                "VirtualWellName",
                "LayerOrder",
                "TopSurfaceCode",
                "TopSurfaceName",
                "TopDepth",
                "BaseSurfaceCode",
                "BaseSurfaceName",
                "BaseDepth",
            ]
        )

    for idx in range(len(actual_surface_order) - 1):
        top_code, top_name, top_depth = actual_surface_order[idx]
        base_code, base_name, base_depth = actual_surface_order[idx + 1]
        rows.append(
            {
                "VirtualWellName": well_name,
                "LayerOrder": int(idx + 1),
                "TopSurfaceCode": top_code,
                "TopSurfaceName": top_name,
                "TopDepth": float(top_depth),
                "BaseSurfaceCode": base_code,
                "BaseSurfaceName": base_name,
                "BaseDepth": float(base_depth),
            }
        )

    return pd.DataFrame(rows)


def reorder_output_columns(df: pd.DataFrame, preferred_cols: List[str]) -> pd.DataFrame:
    ordered_cols = [col for col in preferred_cols if col in df.columns]
    ordered_cols.extend(col for col in df.columns if col not in ordered_cols)
    return df[ordered_cols].reset_index(drop=True)


def point_in_surface_extent(x: float, y: float, surface: SurfaceInfo) -> bool:
    return surface.x_min <= x <= surface.x_max and surface.y_min <= y <= surface.y_max


def get_surface_time(x: float, y: float, surface_name: str, surface_catalog: Dict[str, SurfaceInfo]) -> Optional[float]:
    surface = surface_catalog.get(surface_name)
    if surface is None:
        return None
    try:
        value = float(surface.interpolator(x, y))
    except Exception:
        return None
    if math.isnan(value):
        return None
    return value


def get_surface_time_on_sampling_grid(
    x: float,
    y: float,
    surface_name: str,
    surface_catalog: Dict[str, SurfaceInfo],
) -> Optional[float]:
    return snap_to_resample_interval(get_surface_time(x, y, surface_name, surface_catalog))


def get_zone_boundary_times(
    x: float,
    y: float,
    zone: ZoneInfo,
    surface_catalog: Dict[str, SurfaceInfo],
) -> Tuple[Optional[float], Optional[float]]:
    top_time = None
    base_time = None
    if zone.top_surface_name is not None:
        top_time = get_surface_time_on_sampling_grid(x, y, zone.top_surface_name, surface_catalog)
    if zone.base_surface_name is not None:
        base_time = get_surface_time_on_sampling_grid(x, y, zone.base_surface_name, surface_catalog)
    return top_time, base_time


def get_applicable_zones(
    x: float,
    y: float,
    zone_catalog: List[ZoneInfo],
    surface_catalog: Dict[str, SurfaceInfo],
) -> List[ZoneInfo]:
    applicable: List[ZoneInfo] = []
    for zone in zone_catalog:
        top_ok = True
        base_ok = True
        if zone.top_surface_name is not None:
            top_surface = surface_catalog[zone.top_surface_name]
            top_ok = point_in_surface_extent(x, y, top_surface)
        if zone.base_surface_name is not None:
            base_surface = surface_catalog[zone.base_surface_name]
            base_ok = point_in_surface_extent(x, y, base_surface)
        if top_ok and base_ok:
            applicable.append(zone)
    return applicable


def resolve_preferred_surface_name(surface_code: str, surface_catalog: Dict[str, SurfaceInfo]) -> Optional[str]:
    if not surface_code:
        return None
    preferred_name = FIRST_STEP_SURFACE_PREFERRED_NAMES.get(surface_code, surface_code)
    resolved = resolve_surface_name(preferred_name, surface_catalog)
    if resolved is not None:
        return resolved
    for candidate in surface_catalog:
        if candidate.startswith(surface_code):
            return candidate
    return None


def score_zone_candidate(zone: ZoneInfo) -> Tuple[int, int, str]:
    score = 0
    joined = " ".join([zone.zone_name, zone.top_surface_name or "", zone.base_surface_name or ""])
    if "2024" in joined:
        score += 4
    if "DM_Sm" in joined:
        score += 4
    if "gljmAto" in joined:
        score += 2
    if zone.zone_name.startswith(("Above_T", "Between_T", "Below_T")):
        score += 1
    code_count = int(bool(zone.top_surface_code)) + int(bool(zone.base_surface_code))
    return (score, code_count, zone.zone_name)


def select_first_step_relevant_zones(
    zone_catalog: List[ZoneInfo],
    surface_catalog: Dict[str, SurfaceInfo],
) -> Dict[str, ZoneInfo]:
    selected: Dict[str, ZoneInfo] = {}
    for zone in zone_catalog:
        if not is_first_step_relevant_interval(zone.top_surface_code, zone.base_surface_code):
            continue
        interval_key = zone.interval_key
        existing = selected.get(interval_key)
        if existing is None or score_zone_candidate(zone) > score_zone_candidate(existing):
            selected[interval_key] = ZoneInfo(
                zone_name=zone.zone_name,
                zone_path=zone.zone_path,
                zone_type=zone.zone_type,
                top_surface_name=resolve_preferred_surface_name(zone.top_surface_code, surface_catalog),
                base_surface_name=resolve_preferred_surface_name(zone.base_surface_code, surface_catalog),
                top_surface_code=zone.top_surface_code,
                base_surface_code=zone.base_surface_code,
                interval_key=interval_key,
                range_id=zone.range_id,
                strata_name="",
            )
    return selected


def build_actual_surface_order(
    x: float,
    y: float,
    surface_catalog: Dict[str, SurfaceInfo],
) -> List[Tuple[str, str, float]]:
    ordered_surfaces: List[Tuple[str, str, float, float]] = []
    for surface_code in FIRST_STEP_SURFACE_CODES:
        surface_name = resolve_preferred_surface_name(surface_code, surface_catalog)
        if not surface_name:
            continue
        raw_surface_time = get_surface_time(x, y, surface_name, surface_catalog)
        snapped_surface_time = get_surface_time_on_sampling_grid(x, y, surface_name, surface_catalog)
        if raw_surface_time is None or not np.isfinite(raw_surface_time):
            continue
        if snapped_surface_time is None or not np.isfinite(snapped_surface_time):
            continue
        ordered_surfaces.append(
            (
                surface_code,
                surface_name,
                float(raw_surface_time),
                float(snapped_surface_time),
            )
        )
    ordered_surfaces.sort(key=lambda item: (item[2], item[0]))
    return [(code, name, snapped_time) for code, name, _raw_time, snapped_time in ordered_surfaces]


def build_first_step_interval_plan(
    x: float,
    y: float,
    selected_zones: Dict[str, ZoneInfo],
    surface_catalog: Dict[str, SurfaceInfo],
) -> List[ZoneInfo]:
    ordered_surfaces = build_actual_surface_order(x, y, surface_catalog)
    if not ordered_surfaces:
        return []

    planned: List[ZoneInfo] = []
    seen_interval_keys: set[str] = set()

    shallowest_code = ordered_surfaces[0][0]
    above_key = build_interval_key("", shallowest_code)
    above_zone = selected_zones.get(above_key)
    if above_zone is not None:
        planned.append(above_zone)
        seen_interval_keys.add(above_key)

    for idx in range(len(ordered_surfaces) - 1):
        top_code = ordered_surfaces[idx][0]
        base_code = ordered_surfaces[idx + 1][0]
        interval_key = build_interval_key(top_code, base_code)
        zone = selected_zones.get(interval_key)
        if zone is None or interval_key in seen_interval_keys:
            continue
        planned.append(zone)
        seen_interval_keys.add(interval_key)

    deepest_code = ordered_surfaces[-1][0]
    below_key = build_interval_key(deepest_code, "")
    below_zone = selected_zones.get(below_key)
    if below_zone is not None and below_key not in seen_interval_keys:
        planned.append(below_zone)

    return planned


def create_target_relative_depths(
    zone_df: pd.DataFrame,
    zone: ZoneInfo,
    top_time: Optional[float],
    base_time: Optional[float],
) -> np.ndarray:
    eps = 1e-9
    rel_min = float(zone_df["RELATIVE_DEPTH"].min())
    rel_max = float(zone_df["RELATIVE_DEPTH"].max())

    if zone.zone_type == "above":
        rel_max = min(rel_max, -RESAMPLE_INTERVAL)
    elif zone.zone_type == "between" and top_time is not None and base_time is not None:
        rel_min = 0.0
        rel_max = float(base_time) - float(top_time) - RESAMPLE_INTERVAL
    elif zone.zone_type == "below":
        rel_min = max(rel_min, 0.0)

    start = math.ceil((rel_min - eps) / RESAMPLE_INTERVAL) * RESAMPLE_INTERVAL
    end = math.floor((rel_max + eps) / RESAMPLE_INTERVAL) * RESAMPLE_INTERVAL
    if start > end:
        return np.array([], dtype=float)
    count = int(round((end - start) / RESAMPLE_INTERVAL)) + 1
    return np.round(np.linspace(start, end, count), 3)


def fit_knn_model(
    zone_df: pd.DataFrame,
    target_col: str,
    neighbors: int,
) -> Tuple[Optional[KNeighborsRegressor], Optional[StandardScaler], int]:
    train_df = zone_df.dropna(subset=["X", "Y", "RELATIVE_DEPTH", target_col]).copy()
    train_df = train_df[np.isfinite(train_df[target_col].astype(float))]
    sample_count = len(train_df)
    if sample_count < 3:
        return None, None, sample_count

    features = train_df[["X", "Y", "RELATIVE_DEPTH"]].astype(float).values
    target = train_df[target_col].astype(float).values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = KNeighborsRegressor(
        n_neighbors=min(neighbors, sample_count),
        weights="distance",
    )
    model.fit(scaled_features, target)
    return model, scaler, sample_count


def fit_xgboost_model(
    zone_df: pd.DataFrame,
    target_col: str,
) -> Tuple[Optional[object], Optional[StandardScaler], int]:
    if xgb is None:
        raise ImportError("xgboost is not installed. Install it before using --method xgboost.")

    train_df = zone_df.dropna(subset=["X", "Y", "RELATIVE_DEPTH", target_col]).copy()
    train_df = train_df[np.isfinite(train_df[target_col].astype(float))]
    sample_count = len(train_df)
    if sample_count < 10:
        return None, None, sample_count

    features = train_df[["X", "Y", "RELATIVE_DEPTH"]].astype(float).values
    target = train_df[target_col].astype(float).values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(scaled_features, target)
    return model, scaler, sample_count


def fit_kriging_model(
    zone_df: pd.DataFrame,
    target_col: str,
) -> Tuple[Optional[object], Optional[StandardScaler], int]:
    if OrdinaryKriging is None:
        raise ImportError("pykrige is not installed. Install it before using --method kriging.")

    train_df = zone_df.dropna(subset=["X", "Y", target_col]).copy()
    train_df = train_df[np.isfinite(train_df[target_col].astype(float))]
    sample_count = len(train_df)
    if sample_count < 10:
        return None, None, sample_count

    x_values = train_df["X"].astype(float).values
    y_values = train_df["Y"].astype(float).values
    z_values = train_df[target_col].astype(float).values
    try:
        model = OrdinaryKriging(
            x_values,
            y_values,
            z_values,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )
    except Exception:
        return None, None, sample_count
    return model, None, sample_count


def fit_predictor(
    zone_df: pd.DataFrame,
    target_col: str,
    method: str,
    neighbors: int,
) -> Tuple[Optional[object], Optional[StandardScaler], int]:
    if method == "knn":
        return fit_knn_model(zone_df, target_col, neighbors)
    if method == "xgboost":
        return fit_xgboost_model(zone_df, target_col)
    if method == "kriging":
        return fit_kriging_model(zone_df, target_col)
    raise ValueError(f"Unsupported prediction method: {method}")


def predict_with_model(
    model: Optional[object],
    scaler: Optional[StandardScaler],
    features: np.ndarray,
    method: str,
) -> np.ndarray:
    if model is None:
        return np.full(len(features), np.nan, dtype=float)

    if method in {"knn", "xgboost"}:
        if scaler is None:
            return np.full(len(features), np.nan, dtype=float)
        return np.asarray(model.predict(scaler.transform(features)), dtype=float)

    if method == "kriging":
        try:
            x_values = features[:, 0].astype(float)
            y_values = features[:, 1].astype(float)
            pred, _ = model.execute("points", x_values, y_values)
            return np.asarray(pred, dtype=float)
        except Exception:
            return np.full(len(features), np.nan, dtype=float)

    raise ValueError(f"Unsupported prediction method: {method}")


def predict_zone_curve(
    x: float,
    y: float,
    zone: ZoneInfo,
    method: str,
    neighbors: int,
    surface_catalog: Dict[str, SurfaceInfo],
) -> Optional[pd.DataFrame]:
    zone_df = pd.read_csv(zone.zone_path)
    required_cols = {"X", "Y", "RELATIVE_DEPTH", "AC", "GR"}
    missing_cols = required_cols.difference(zone_df.columns)
    if missing_cols:
        raise ValueError(f"{zone.zone_path} missing columns: {sorted(missing_cols)}")

    top_time, base_time = get_zone_boundary_times(x, y, zone, surface_catalog)
    target_depths = create_target_relative_depths(zone_df, zone, top_time, base_time)
    if len(target_depths) == 0:
        return None

    features = np.column_stack(
        [
            np.full(len(target_depths), x, dtype=float),
            np.full(len(target_depths), y, dtype=float),
            target_depths.astype(float),
        ]
    )

    ac_model, ac_scaler, ac_samples = fit_predictor(zone_df, "AC", method, neighbors)
    gr_model, gr_scaler, gr_samples = fit_predictor(zone_df, "GR", method, neighbors)

    ac_pred = predict_with_model(ac_model, ac_scaler, features, method)
    gr_pred = predict_with_model(gr_model, gr_scaler, features, method)

    result_df = pd.DataFrame(
        {
            "ZoneName": zone.zone_name,
            "PredictMethod": method,
            "RangeId": zone.range_id,
            "IntervalKey": zone.interval_key,
            "ZoneType": zone.zone_type,
            "StrataName": zone.strata_name,
            "TopSurfaceCode": zone.top_surface_code,
            "TopSurfaceName": zone.top_surface_name,
            "BaseSurfaceCode": zone.base_surface_code,
            "BaseSurfaceName": zone.base_surface_name,
            "RELATIVE_DEPTH": target_depths,
            "AC": ac_pred,
            "GR": gr_pred,
            "TrainSampleCountAC": ac_samples,
            "TrainSampleCountGR": gr_samples,
        }
    )
    result_df = result_df.dropna(subset=["AC", "GR"], how="all").reset_index(drop=True)
    if result_df.empty:
        return None
    return result_df


def convert_zone_to_absolute_time(
    zone_df: pd.DataFrame,
    x: float,
    y: float,
    zone: ZoneInfo,
    surface_catalog: Dict[str, SurfaceInfo],
) -> Optional[pd.DataFrame]:
    if zone_df.empty:
        return None

    top_time, base_time = get_zone_boundary_times(x, y, zone, surface_catalog)

    if zone.zone_type == "above":
        if base_time is None:
            return None
        time_values = base_time + zone_df["RELATIVE_DEPTH"].astype(float).values
    else:
        if top_time is None:
            return None
        time_values = top_time + zone_df["RELATIVE_DEPTH"].astype(float).values

    result_df = zone_df.copy()
    result_df["TIME"] = np.round(time_values, 6)
    result_df["TopTime"] = top_time
    result_df["BaseTime"] = base_time
    time_series = pd.to_numeric(result_df["TIME"], errors="coerce")
    if zone.zone_type == "above" and base_time is not None:
        result_df = result_df[time_series < float(base_time) - 1e-9].copy()
    elif zone.zone_type == "between" and top_time is not None and base_time is not None:
        result_df = result_df[
            (time_series >= float(top_time) - 1e-9)
            & (time_series < float(base_time) - 1e-9)
        ].copy()
    elif zone.zone_type == "below" and top_time is not None:
        result_df = result_df[time_series >= float(top_time) - 1e-9].copy()
    result_df = result_df.reset_index(drop=True)
    if result_df.empty:
        return None
    return result_df


def build_virtual_well_outputs(
    center_info: Dict[str, float],
    zone_samples: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not zone_samples:
        raise ValueError("No zone samples were generated for this unit.")

    zone_df = pd.concat(zone_samples, ignore_index=True)
    zone_df.insert(0, "VirtualWellName", f"virtual_{center_info['unit_id']}")
    zone_df.insert(1, "UnitID", center_info["unit_id"])
    zone_df.insert(2, "BlockX", center_info["block_x"])
    zone_df.insert(3, "BlockY", center_info["block_y"])
    zone_df.insert(4, "TraceIdx", center_info["center_trace_idx"])
    zone_df.insert(5, "X", center_info["center_x"])
    zone_df.insert(6, "Y", center_info["center_y"])
    zone_df = zone_df.sort_values(["TIME", "RangeId", "RELATIVE_DEPTH"]).reset_index(drop=True)

    merged_df = zone_df.groupby("TIME", as_index=False).agg(
        {
            "VirtualWellName": "first",
            "UnitID": "first",
            "PredictMethod": "first",
            "BlockX": "first",
            "BlockY": "first",
            "TraceIdx": "first",
            "X": "first",
            "Y": "first",
            "AC": "mean",
            "GR": "mean",
            "RangeId": first_not_null,
            "IntervalKey": first_not_null,
            "ZoneName": first_not_null,
            "ZoneType": first_not_null,
            "StrataName": first_not_null,
            "TopSurfaceCode": first_not_null,
            "TopSurfaceName": first_not_null,
            "BaseSurfaceCode": first_not_null,
            "BaseSurfaceName": first_not_null,
            "TopTime": first_not_null,
            "BaseTime": first_not_null,
        }
    )
    merged_df = merged_df.sort_values("TIME").reset_index(drop=True)
    merged_df.insert(7, "SampleIndex", np.arange(1, len(merged_df) + 1))
    return zone_df, merged_df


def save_outputs(
    output_root: Path,
    center_info: Dict[str, float],
    zone_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    around_df: pd.DataFrame,
    strata_df: pd.DataFrame,
    seismic_meta: Dict[str, float],
    plan_meta: Dict[str, object],
) -> Path:
    unit_output_dir = output_root / center_info["unit_id"]
    unit_output_dir.mkdir(parents=True, exist_ok=True)

    zone_df = reorder_output_columns(
        zone_df,
        [
            "VirtualWellName",
            "UnitID",
            "PredictMethod",
            "BlockX",
            "BlockY",
            "TraceIdx",
            "X",
            "Y",
            "RangeId",
            "IntervalKey",
            "ZoneName",
            "ZoneType",
            "StrataName",
            "TopSurfaceCode",
            "TopSurfaceName",
            "BaseSurfaceCode",
            "BaseSurfaceName",
            "TopTime",
            "BaseTime",
            "TIME",
            "RELATIVE_DEPTH",
            "AC",
            "GR",
            "TrainSampleCountAC",
            "TrainSampleCountGR",
        ],
    )
    merged_df = reorder_output_columns(
        merged_df,
        [
            "VirtualWellName",
            "UnitID",
            "PredictMethod",
            "BlockX",
            "BlockY",
            "TraceIdx",
            "SampleIndex",
            "X",
            "Y",
            "TIME",
            "AC",
            "GR",
            "RangeId",
            "IntervalKey",
            "ZoneName",
            "ZoneType",
            "StrataName",
            "TopSurfaceCode",
            "TopSurfaceName",
            "BaseSurfaceCode",
            "BaseSurfaceName",
            "TopTime",
            "BaseTime",
        ],
    )
    strata_df = reorder_output_columns(
        strata_df,
        [
            "VirtualWellName",
            "LayerOrder",
            "TopSurfaceCode",
            "TopSurfaceName",
            "TopDepth",
            "BaseSurfaceCode",
            "BaseSurfaceName",
            "BaseDepth",
        ],
    )

    index_df = pd.DataFrame([center_info])
    index_df.insert(0, "VirtualWellName", f"virtual_{center_info['unit_id']}")
    if "PredictMethod" in zone_df.columns and not zone_df.empty:
        index_df.insert(1, "PredictMethod", str(zone_df["PredictMethod"].iloc[0]))
    index_df.to_csv(unit_output_dir / "virtual_well_index.csv", index=False, encoding="utf-8-sig")
    zone_df.to_csv(unit_output_dir / "virtual_well_zone_samples.csv", index=False, encoding="utf-8-sig")
    merged_df.to_csv(unit_output_dir / "virtual_well_curve.csv", index=False, encoding="utf-8-sig")
    around_df.to_csv(unit_output_dir / "virtual_well_around_data.csv", index=False, encoding="utf-8-sig")
    strata_df.to_csv(unit_output_dir / "virtual_well_strata_segmentation.csv", index=False, encoding="utf-8-sig")

    summary = {
        "virtual_well_name": f"virtual_{center_info['unit_id']}",
        "unit_id": center_info["unit_id"],
        "predict_method": str(zone_df["PredictMethod"].iloc[0]) if "PredictMethod" in zone_df.columns and not zone_df.empty else "",
        "block_x": center_info["block_x"],
        "block_y": center_info["block_y"],
        "center_trace_idx": center_info["center_trace_idx"],
        "center_x": center_info["center_x"],
        "center_y": center_info["center_y"],
        "zone_sample_rows": int(len(zone_df)),
        "curve_rows": int(len(merged_df)),
        "around_data_rows": int(len(around_df)),
        "strata_segmentation_rows": int(len(strata_df)),
        "zone_count": int(zone_df["RangeId"].nunique()),
        "zone_names": sorted(zone_df["RangeId"].dropna().unique().tolist()),
    }
    summary.update(seismic_meta)
    summary.update(plan_meta)
    (unit_output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return unit_output_dir


def build_generation_context(
    trace_header_file: Path,
    surface_dir: Path,
    zone_data_dir: Path,
) -> GenerationContext:
    trace_df, unique_x, unique_y = build_grid_index(trace_header_file)
    surface_catalog = load_surface_catalog(surface_dir)
    zone_catalog = load_zone_catalog(zone_data_dir, surface_catalog)
    selected_zones = select_first_step_relevant_zones(
        zone_catalog,
        surface_catalog,
    )
    return GenerationContext(
        trace_df=trace_df,
        unique_x=unique_x,
        unique_y=unique_y,
        surface_catalog=surface_catalog,
        selected_zones=selected_zones,
    )


def normalize_block_range(start: int, end: int, block_count: int, axis_name: str) -> List[int]:
    lower = min(start, end)
    upper = max(start, end)
    if lower < 0 or upper >= block_count:
        raise ValueError(
            f"{axis_name} range [{start}, {end}] is out of bounds. "
            f"Valid {axis_name} indices are 0..{block_count - 1}."
        )
    return list(range(lower, upper + 1))


def build_batch_summary_paths(
    output_root: Path,
    block_x_values: List[int],
    block_y_values: List[int],
) -> Tuple[Path, Path]:
    x_min = min(block_x_values)
    x_max = max(block_x_values)
    y_min = min(block_y_values)
    y_max = max(block_y_values)
    stem = f"batch_run_summary_BX{x_min}_{x_max}_BY{y_min}_{y_max}"
    return output_root / f"{stem}.csv", output_root / f"{stem}.json"


def save_batch_summary(
    output_root: Path,
    block_x_values: List[int],
    block_y_values: List[int],
    rows: List[dict],
) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = build_batch_summary_paths(output_root, block_x_values, block_y_values)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "block_x_values": block_x_values,
        "block_y_values": block_y_values,
        "requested_unit_count": int(len(block_x_values) * len(block_y_values)),
        "success_count": int(sum(1 for row in rows if row["Status"] == "success")),
        "skipped_count": int(sum(1 for row in rows if row["Status"] == "skipped")),
        "failed_count": int(sum(1 for row in rows if row["Status"] == "failed")),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def run_generation_with_context(
    block_x: int,
    block_y: int,
    context: GenerationContext,
    sgy_file: Path,
    output_root: Path,
    method: str,
    neighbors: int,
    sampling_start_ms: float,
    sampling_interval_ms: float,
    grid_spacing: float,
) -> Path:
    center_info = locate_unit_center(context.trace_df, context.unique_x, context.unique_y, block_x, block_y)

    actual_surface_order = build_actual_surface_order(
        center_info["center_x"],
        center_info["center_y"],
        context.surface_catalog,
    )
    applicable_zones = build_first_step_interval_plan(
        center_info["center_x"],
        center_info["center_y"],
        context.selected_zones,
        context.surface_catalog,
    )
    if not applicable_zones:
        raise ValueError("No first-step-style strata intervals were found for the target unit center.")

    zone_samples: List[pd.DataFrame] = []
    for zone in applicable_zones:
        relative_curve = predict_zone_curve(
            center_info["center_x"],
            center_info["center_y"],
            zone,
            method,
            neighbors,
            context.surface_catalog,
        )
        if relative_curve is None:
            continue
        absolute_curve = convert_zone_to_absolute_time(
            relative_curve,
            center_info["center_x"],
            center_info["center_y"],
            zone,
            context.surface_catalog,
        )
        if absolute_curve is None:
            continue
        zone_samples.append(absolute_curve)

    if not zone_samples:
        raise ValueError("All applicable zones failed during virtual-well generation.")

    zone_df, merged_df = build_virtual_well_outputs(center_info, zone_samples)
    seismic_curve_df, seismic_meta = attach_seismic_window_features(
        curve_df=merged_df,
        trace_df=context.trace_df,
        sgy_file=sgy_file,
        sampling_start_ms=sampling_start_ms,
        sampling_interval_ms=sampling_interval_ms,
        grid_spacing=grid_spacing,
    )
    around_df = build_stage1_compatible_virtual_well(center_info, seismic_curve_df)
    strata_df = build_virtual_well_strata_segmentation(center_info, actual_surface_order)
    plan_meta = {
        "actual_surface_order_codes": [code for code, _name, _time in actual_surface_order],
        "actual_surface_order_names": [name for _code, name, _time in actual_surface_order],
        "actual_surface_order_times": [float(time_value) for _code, _name, time_value in actual_surface_order],
        "actual_strata_interval_keys": [
            build_interval_key(actual_surface_order[idx][0], actual_surface_order[idx + 1][0])
            for idx in range(max(len(actual_surface_order) - 1, 0))
        ],
        "planned_interval_keys": [zone.interval_key for zone in applicable_zones],
        "planned_range_ids": [zone.range_id for zone in applicable_zones],
    }
    return save_outputs(output_root, center_info, zone_df, merged_df, around_df, strata_df, seismic_meta, plan_meta)


def run_generation(
    block_x: int,
    block_y: int,
    trace_header_file: Path,
    sgy_file: Path,
    surface_dir: Path,
    zone_data_dir: Path,
    output_root: Path,
    method: str,
    neighbors: int,
    sampling_start_ms: float,
    sampling_interval_ms: float,
    grid_spacing: float,
    skip_existing: bool = False,
) -> Path:
    if skip_existing and has_complete_virtual_well_output(output_root, block_x, block_y):
        return get_unit_output_dir(output_root, block_x, block_y)

    context = build_generation_context(trace_header_file, surface_dir, zone_data_dir)
    return run_generation_with_context(
        block_x=block_x,
        block_y=block_y,
        context=context,
        sgy_file=sgy_file,
        output_root=output_root,
        method=method,
        neighbors=neighbors,
        sampling_start_ms=sampling_start_ms,
        sampling_interval_ms=sampling_interval_ms,
        grid_spacing=grid_spacing,
    )


def run_generation_batch(
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
    trace_header_file: Path,
    sgy_file: Path,
    surface_dir: Path,
    zone_data_dir: Path,
    output_root: Path,
    method: str,
    neighbors: int,
    sampling_start_ms: float,
    sampling_interval_ms: float,
    grid_spacing: float,
    skip_existing: bool,
    fail_fast: bool,
) -> Dict[str, object]:
    context = build_generation_context(trace_header_file, surface_dir, zone_data_dir)
    block_x_values = normalize_block_range(
        block_x_start,
        block_x_end,
        get_num_blocks(context.unique_x),
        "BlockX",
    )
    block_y_values = normalize_block_range(
        block_y_start,
        block_y_end,
        get_num_blocks(context.unique_y),
        "BlockY",
    )

    total = len(block_x_values) * len(block_y_values)
    rows: List[dict] = []
    outputs: List[str] = []
    counter = 0
    for block_y in block_y_values:
        for block_x in block_x_values:
            counter += 1
            unit_id = build_unit_id(block_x, block_y)
            existing_output_dir = get_unit_output_dir(output_root, block_x, block_y)
            if skip_existing and has_complete_virtual_well_output(output_root, block_x, block_y):
                outputs.append(str(existing_output_dir))
                rows.append(
                    {
                        "BlockX": block_x,
                        "BlockY": block_y,
                        "UnitID": unit_id,
                        "Status": "skipped",
                        "OutputDir": str(existing_output_dir),
                        "Error": "",
                    }
                )
                print(f"[{counter}/{total}] Skipped {unit_id} -> {existing_output_dir}")
                continue
            try:
                output_dir = run_generation_with_context(
                    block_x=block_x,
                    block_y=block_y,
                    context=context,
                    sgy_file=sgy_file,
                    output_root=output_root,
                    method=method,
                    neighbors=neighbors,
                    sampling_start_ms=sampling_start_ms,
                    sampling_interval_ms=sampling_interval_ms,
                    grid_spacing=grid_spacing,
                )
                outputs.append(str(output_dir))
                rows.append(
                    {
                        "BlockX": block_x,
                        "BlockY": block_y,
                        "UnitID": unit_id,
                        "Status": "success",
                        "OutputDir": str(output_dir),
                        "Error": "",
                    }
                )
                print(f"[{counter}/{total}] Generated {unit_id} -> {output_dir}")
            except Exception as exc:
                error_text = str(exc)
                rows.append(
                    {
                        "BlockX": block_x,
                        "BlockY": block_y,
                        "UnitID": unit_id,
                        "Status": "failed",
                        "OutputDir": "",
                        "Error": error_text,
                    }
                )
                print(f"[{counter}/{total}] Failed {unit_id}: {error_text}")
                if fail_fast:
                    csv_path, json_path = save_batch_summary(output_root, block_x_values, block_y_values, rows)
                    raise RuntimeError(
                        f"Batch virtual-well generation stopped at {unit_id} because --fail-fast was enabled. "
                        f"Summary CSV: {csv_path}; Summary JSON: {json_path}; Error: {error_text}"
                    ) from exc

    csv_path, json_path = save_batch_summary(output_root, block_x_values, block_y_values, rows)
    success_count = sum(1 for row in rows if row["Status"] == "success")
    skipped_count = sum(1 for row in rows if row["Status"] == "skipped")
    failed_count = sum(1 for row in rows if row["Status"] == "failed")
    completed_count = success_count + skipped_count
    if completed_count == 0:
        raise RuntimeError(
            f"Batch virtual-well generation failed for all requested units. "
            f"Summary: {csv_path}"
        )
    return {
        "requested_unit_count": total,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "outputs": outputs,
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate virtual wells for one unit or a BlockX/BlockY range.",
    )
    parser.add_argument("--block-x", type=int, help="Unit block index in X direction for single-unit mode.")
    parser.add_argument("--block-y", type=int, help="Unit block index in Y direction for single-unit mode.")
    parser.add_argument("--block-x-start", type=int, help="Start BlockX index for batch mode.")
    parser.add_argument("--block-x-end", type=int, help="End BlockX index for batch mode.")
    parser.add_argument("--block-y-start", type=int, help="Start BlockY index for batch mode.")
    parser.add_argument("--block-y-end", type=int, help="End BlockY index for batch mode.")
    parser.add_argument(
        "--trace-header",
        type=Path,
        default=DEFAULT_TRACE_HEADER,
        help=f"Path to trace_header_xy.csv. Default: {DEFAULT_TRACE_HEADER}",
    )
    parser.add_argument(
        "--surface-dir",
        type=Path,
        default=DEFAULT_SURFACE_DIR,
        help=f"Directory with layer surface DAT files. Default: {DEFAULT_SURFACE_DIR}",
    )
    parser.add_argument(
        "--zone-data-dir",
        type=Path,
        default=DEFAULT_ZONE_DATA_DIR,
        help=f"Directory with *_relative_depth.csv files. Default: {DEFAULT_ZONE_DATA_DIR}",
    )
    parser.add_argument(
        "--sgy-file",
        type=Path,
        default=DEFAULT_SGY_FILE,
        help=f"SEG-Y file used for seismic window extraction. Default: {DEFAULT_SGY_FILE}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=list(SUPPORTED_METHODS),
        help=f"Prediction method for virtual AC/GR generation. Default: {DEFAULT_METHOD}",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=DEFAULT_NEIGHBORS,
        help=f"KNN neighbors for AC/GR interpolation when --method knn. Default: {DEFAULT_NEIGHBORS}",
    )
    parser.add_argument(
        "--sampling-start-ms",
        type=float,
        default=DEFAULT_SAMPLING_START_MS,
        help=f"Seismic sampling start time in ms. Default: {DEFAULT_SAMPLING_START_MS}",
    )
    parser.add_argument(
        "--sampling-interval-ms",
        type=float,
        default=DEFAULT_SAMPLING_INTERVAL_MS,
        help=f"Seismic sampling interval in ms. Default: {DEFAULT_SAMPLING_INTERVAL_MS}",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=DEFAULT_SEISMIC_GRID_SPACING,
        help=f"Seismic trace spacing in meters. Default: {DEFAULT_SEISMIC_GRID_SPACING}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip units whose output directory already contains a complete virtual-well result set.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="In batch mode, stop immediately after the first failed unit and write the partial batch summary.",
    )
    return parser.parse_args()


def resolve_run_mode(args: argparse.Namespace) -> str:
    single_mode = args.block_x is not None or args.block_y is not None
    batch_mode = any(
        value is not None
        for value in (args.block_x_start, args.block_x_end, args.block_y_start, args.block_y_end)
    )

    if single_mode and batch_mode:
        raise ValueError("Use either single-unit mode (--block-x/--block-y) or batch mode (--block-x-start/--block-x-end/--block-y-start/--block-y-end), not both.")

    if single_mode:
        if args.block_x is None or args.block_y is None:
            raise ValueError("Single-unit mode requires both --block-x and --block-y.")
        return "single"

    if batch_mode:
        required = {
            "--block-x-start": args.block_x_start,
            "--block-x-end": args.block_x_end,
            "--block-y-start": args.block_y_start,
            "--block-y-end": args.block_y_end,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Batch mode is missing required arguments: {', '.join(missing)}")
        return "batch"

    raise ValueError(
        "No target unit was specified. "
        "Use --block-x/--block-y for one unit, or use "
        "--block-x-start/--block-x-end/--block-y-start/--block-y-end for batch mode."
    )


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)
    if run_mode == "single":
        preexisting = args.skip_existing and has_complete_virtual_well_output(args.output_root, args.block_x, args.block_y)
        output_dir = run_generation(
            block_x=args.block_x,
            block_y=args.block_y,
            trace_header_file=args.trace_header,
            sgy_file=args.sgy_file,
            surface_dir=args.surface_dir,
            zone_data_dir=args.zone_data_dir,
            output_root=args.output_root,
            method=str(args.method),
            neighbors=args.neighbors,
            sampling_start_ms=args.sampling_start_ms,
            sampling_interval_ms=args.sampling_interval_ms,
            grid_spacing=args.grid_spacing,
            skip_existing=args.skip_existing,
        )
        if preexisting:
            print(f"Virtual well skipped because output already exists: {output_dir}")
        else:
            print(f"Virtual well generated under: {output_dir}")
        return

    batch_summary = run_generation_batch(
        block_x_start=args.block_x_start,
        block_x_end=args.block_x_end,
        block_y_start=args.block_y_start,
        block_y_end=args.block_y_end,
        trace_header_file=args.trace_header,
        sgy_file=args.sgy_file,
        surface_dir=args.surface_dir,
        zone_data_dir=args.zone_data_dir,
        output_root=args.output_root,
        method=str(args.method),
        neighbors=args.neighbors,
        sampling_start_ms=args.sampling_start_ms,
        sampling_interval_ms=args.sampling_interval_ms,
        grid_spacing=args.grid_spacing,
        skip_existing=args.skip_existing,
        fail_fast=args.fail_fast,
    )
    print(
        "Batch virtual-well generation completed: "
        f"{batch_summary['success_count']} succeeded, "
        f"{batch_summary['skipped_count']} skipped, "
        f"{batch_summary['failed_count']} failed. "
        f"Summary CSV: {batch_summary['summary_csv']}"
    )


if __name__ == "__main__":
    main()
