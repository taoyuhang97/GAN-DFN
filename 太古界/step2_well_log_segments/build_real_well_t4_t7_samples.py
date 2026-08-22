from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree


STANDARD_LOG_CURVES = [
    "AC",
    "CAL",
    "CNL",
    "DEN",
    "GR",
    "RFOC",
    "RILD",
    "RILM",
    "SP",
]
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")
INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)
GRID_SPACING = 12.5
NEIGHBOR_OFFSETS = (-GRID_SPACING, 0.0, GRID_SPACING)
CURRENT_DIR = Path(__file__).resolve().parent
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_strata_contracts"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import assign_surface_times, classify_time_domain, load_surface_tables, validate_surface_order  # noqa: E402


@dataclass(frozen=True)
class VolumeSpec:
    output_prefix: str
    path: Path


LOG_ALIAS_MAP = {
    "DEPT": "DEPT",
    "DEPTH": "DEPT",
    "MD": "DEPT",
    "TVD": "TVD",
    "TIME": "TIME",
    "AC": "AC",
    "CAL": "CAL",
    "CALI": "CAL",
    "CNL": "CNL",
    "NPHI": "CNL",
    "DEN": "DEN",
    "RHOB": "DEN",
    "GR": "GR",
    "RFOC": "RFOC",
    "RILD": "RILD",
    "RILM": "RILM",
    "SP": "SP",
}
WELL_ALIAS_MAP = {
    "导眼井": "车页1导眼",
    "车页1HF": "车页1导眼",
}
TIME_CLASS_TO_LAYER = {
    "sha3_t4_t6": "沙三段",
    "sha4_t6_t7": "沙四段",
}
ATTRIBUTE_VALUE_RULES = {
    "SeisAmp": {"min": -1.0e5, "max": 1.0e5},
    "Coherence": {"min": 0.0, "max": 1.0e3},
    "AntTrack": {"min": -10.0, "max": 10.0},
    "CurvatureMax": {"min": -10.0, "max": 10.0},
    "CurvaturePos": {"min": -10.0, "max": 10.0},
}
MIN_VALID_NEIGHBORS_FOR_MODEL = 5
MAIN_ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]
MAIN_LOGIC_COLUMNS = ["SampleUsableForModel", "SampleUsableStatus"]
SUMMARY_STAT_SUFFIXES = ["Mean", "Std", "Min", "Max", "ValidCount"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build T4-T7-only real-well formal sample tables using true XY/TIME points and interpolated 3x3 neighborhood."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--max-points-per-well", type=int, default=0, help="Optional cap for demo/self-check runs; 0 means all target points.")
    parser.add_argument("--max-log-rows-per-well", type=int, default=0, help="Optional cap before point building; 0 means all raw log rows.")
    parser.add_argument("--max-workers", type=int, default=0, help="Number of wells to process in parallel; 0 means sequential.")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_lines(path: Path) -> list[str]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, errors="ignore") as handle:
                return handle.readlines()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read text file: {path}") from last_error


def normalize_name(name: str) -> str:
    norm = unicodedata.normalize("NFKD", str(name))
    norm = norm.upper().replace("'", "").replace(".", "").strip()
    return re.sub(r"\s+", "", norm)


def canonicalize_well_name(well_name: str) -> str:
    return WELL_ALIAS_MAP.get(str(well_name).strip(), str(well_name).strip())


def clean_numeric_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    out = out.mask(out.abs() >= 1e6)
    return out


def read_las_ascii_block(file_path: Path) -> pd.DataFrame:
    lines = read_text_lines(file_path)
    columns: list[str] = []
    in_columns_section = False
    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("~curve") or lower.startswith("~c"):
            in_columns_section = True
            continue
        if in_columns_section:
            if line.startswith("~") or lower.startswith("~parameter"):
                break
            if "." in line:
                name = line.split(".")[0].strip()
                if name:
                    columns.append(name)

    start_idx = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("~ascii"))
    data: list[list[float]] = []
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        values = []
        for item in stripped.split():
            try:
                values.append(float(item))
            except Exception:
                values.append(np.nan)
        data.append(values)
    if not columns:
        raise ValueError(f"no LAS curve columns found: {file_path}")
    return pd.DataFrame(data, columns=columns)


def normalize_las_curves(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for col in out.columns:
        norm = normalize_name(col)
        if norm in LOG_ALIAS_MAP:
            rename_map[col] = LOG_ALIAS_MAP[norm]
    out = out.rename(columns=rename_map)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    if "DEPT" not in out.columns:
        for candidate in ["TVD", "DEPTH", "MD"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "DEPT"})
                break
    if "DEPT" not in out.columns:
        raise ValueError(f"LAS missing depth column: {df.columns.tolist()}")

    out["DEPT"] = pd.to_numeric(out["DEPT"], errors="coerce")
    out = out[out["DEPT"].notna()].copy()
    out = out.sort_values("DEPT").reset_index(drop=True)

    if "TVD" not in out.columns:
        out["TVD"] = out["DEPT"]
    out["TVD"] = pd.to_numeric(out["TVD"], errors="coerce")

    for curve in STANDARD_LOG_CURVES:
        if curve not in out.columns:
            out[curve] = np.nan
            continue
        out[curve] = clean_numeric_series(out[curve])

    return out


def thin_rows_by_stride(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def read_well_xy_from_las(file_path: Path) -> tuple[float | None, float | None]:
    lines = read_text_lines(file_path)
    in_parameter_block = False
    x = y = None
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("~parameter") or lower.startswith("~p"):
            in_parameter_block = True
            continue
        if stripped.startswith("~") and in_parameter_block:
            break
        if in_parameter_block:
            upper = stripped.upper()
            if "XCRD" in upper:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", stripped)
                if nums:
                    x = float(nums[0])
            elif "YCRD" in upper:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", stripped)
                if nums:
                    y = float(nums[0])
    return x, y


def read_well_track(file_path: Path) -> pd.DataFrame:
    lines = read_text_lines(file_path)
    if len(lines) < 3:
        raise ValueError(f"invalid well-track file: {file_path}")
    header = re.split(r"\s+", lines[1].strip("#").strip())
    data = [re.split(r"\s+", line.strip()) for line in lines[2:] if line.strip()]
    df = pd.DataFrame(data, columns=header)
    for col in ["MD", "TVD", "X", "Y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["MD", "TVD", "X", "Y"]].dropna().sort_values("MD").drop_duplicates("MD").reset_index(drop=True)


def read_timedepth_file(file_path: Path) -> pd.DataFrame:
    lines = read_text_lines(file_path)
    header_line_index = -1
    for idx, line in enumerate(lines):
        if not line.strip().startswith("#") and len(re.split(r"\s+", line.strip())) >= 5:
            header_line_index = idx - 1
            break
    if header_line_index < 0:
        raise ValueError(f"invalid timedepth file: {file_path}")
    header_line = lines[header_line_index].lstrip("#").strip()
    column_names = re.split(r"\s+", header_line)
    data = [re.split(r"\s+", line.strip()) for line in lines[header_line_index + 1 :] if line.strip()]
    df = pd.DataFrame(data, columns=column_names)

    renamed = []
    seen = set()
    for col in df.columns:
        norm = normalize_name(col)
        if norm not in seen:
            renamed.append(norm)
            seen.add(norm)
        else:
            renamed.append(col)
    df.columns = renamed
    for col in ["MD", "TVD", "TIME"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["MD", "TVD", "TIME"]].dropna(subset=["TVD", "TIME"]).sort_values("TVD").drop_duplicates("TVD").reset_index(drop=True)


def build_time_mapper(timedepth_df: pd.DataFrame, depth_axis: str = "TVD") -> interp1d:
    mapping_df = (
        timedepth_df[[depth_axis, "TIME"]]
        .dropna()
        .sort_values(depth_axis)
        .drop_duplicates(depth_axis)
    )
    if len(mapping_df) < 2:
        raise ValueError(f"timedepth table needs at least two valid {depth_axis}/TIME rows")
    return interp1d(
        mapping_df[depth_axis].to_numpy(dtype=float),
        mapping_df["TIME"].to_numpy(dtype=float),
        bounds_error=False,
        fill_value="extrapolate",
    )


def build_xy_mapper(track_df: pd.DataFrame, depth_axis: str = "TVD"):
    if len(track_df) <= 1:
        x0 = float(track_df["X"].iloc[0])
        y0 = float(track_df["Y"].iloc[0])

        def fx(values):
            arr = np.asarray(values, dtype=float)
            return np.full(arr.shape, x0, dtype=float)

        def fy(values):
            arr = np.asarray(values, dtype=float)
            return np.full(arr.shape, y0, dtype=float)

        return fx, fy

    mapping_df = track_df[[depth_axis, "X", "Y"]].dropna().sort_values(depth_axis).drop_duplicates(depth_axis)
    fx = interp1d(mapping_df[depth_axis].to_numpy(dtype=float), mapping_df["X"].to_numpy(dtype=float), bounds_error=False, fill_value="extrapolate")
    fy = interp1d(mapping_df[depth_axis].to_numpy(dtype=float), mapping_df["Y"].to_numpy(dtype=float), bounds_error=False, fill_value="extrapolate")
    return fx, fy


def build_tvd_mapper(track_df: pd.DataFrame) -> interp1d:
    mapping_df = track_df[["MD", "TVD"]].dropna().sort_values("MD").drop_duplicates("MD")
    if len(mapping_df) < 2:
        raise ValueError("well-track table needs at least two valid MD/TVD rows")
    return interp1d(
        mapping_df["MD"].to_numpy(dtype=float),
        mapping_df["TVD"].to_numpy(dtype=float),
        bounds_error=False,
        fill_value="extrapolate",
    )


def build_trace_grid(trace_header_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trace_header_csv)
    for col in ["TraceIdx", "X", "Y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["TraceIdx", "X", "Y"]].dropna().reset_index(drop=True)


def build_trace_tree(trace_df: pd.DataFrame) -> tuple[KDTree, np.ndarray]:
    xy = trace_df[["X", "Y"]].to_numpy(dtype=np.float32)
    trace_ids = trace_df["TraceIdx"].to_numpy(dtype=np.int64)
    return KDTree(xy, leaf_size=64), trace_ids


def open_volume_context(volume_path: Path):
    handle = segyio.open(str(volume_path), "r", ignore_geometry=True)
    handle.mmap()
    samples = np.asarray(handle.samples, dtype=np.float64)
    trace_cache: dict[int, np.ndarray] = {}

    def trace_at(trace_idx: int) -> np.ndarray:
        key = int(trace_idx)
        if key not in trace_cache:
            trace_cache[key] = np.asarray(handle.trace[key], dtype=np.float32)
        return trace_cache[key]

    return handle, samples, trace_at


def sample_trace_at_time(trace_data: np.ndarray, samples: np.ndarray, time_ms: float) -> float:
    if not np.isfinite(time_ms):
        return np.nan
    it = np.interp(time_ms, samples, np.arange(len(samples), dtype=np.float64), left=np.nan, right=np.nan)
    if not np.isfinite(it):
        return np.nan
    i0 = int(np.floor(it))
    if i0 < 0 or i0 + 1 >= len(trace_data):
        return np.nan
    weight = float(it - i0)
    return float((1.0 - weight) * trace_data[i0] + weight * trace_data[i0 + 1])


def inverse_distance_xy_sample(
    x: float,
    y: float,
    time_ms: float,
    tree: KDTree,
    trace_ids: np.ndarray,
    samples: np.ndarray,
    trace_at,
) -> float:
    if not np.isfinite(x) or not np.isfinite(y):
        return np.nan
    distances, idx = tree.query([[x, y]], k=4)
    distances = distances[0]
    idx = idx[0]
    amplitudes: list[float] = []
    weights: list[float] = []
    for dist, pos in zip(distances, idx):
        trace_idx = int(trace_ids[pos])
        trace_data = trace_at(trace_idx)
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


def build_volume_specs(config: dict[str, str]) -> list[VolumeSpec]:
    return [VolumeSpec(output_prefix=name, path=Path(path_str).resolve()) for name, path_str in config.items()]


def offset_to_axis_label(offset: float, axis: str) -> str:
    mapping = {
        -GRID_SPACING: f"{axis}0",
        0.0: f"{axis}1",
        GRID_SPACING: f"{axis}2",
    }
    return mapping[float(offset)]


def discover_las_wells(log_dir: Path) -> list[str]:
    names = []
    seen = set()
    for path in sorted(log_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".las":
            continue
        name = canonicalize_well_name(path.stem)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def resolve_las_path(log_dir: Path, well_name: str) -> Path | None:
    canonical = canonicalize_well_name(well_name)
    for candidate in [log_dir / f"{canonical}.Las", log_dir / f"{canonical}.las", log_dir / f"{well_name}.Las", log_dir / f"{well_name}.las"]:
        if candidate.exists():
            return candidate
    return None


def build_point_rows_for_well(
    well_name: str,
    las_df: pd.DataFrame,
    track_df: pd.DataFrame,
    time_df: pd.DataFrame,
    surface_tables,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_kind = "vertical" if len(track_df) <= 1 else "deviated"
    if source_kind == "deviated":
        tvd_mapper = build_tvd_mapper(track_df)
        fx, fy = build_xy_mapper(track_df, depth_axis="MD")
        time_mapper = build_time_mapper(time_df, depth_axis="MD")
    else:
        tvd_mapper = None
        fx, fy = build_xy_mapper(track_df)
        time_mapper = build_time_mapper(time_df)

    records = []
    for row_idx, row in las_df.iterrows():
        md = float(row["DEPT"])
        if source_kind == "deviated":
            tvd = float(tvd_mapper(md))
            time_ms = float(time_mapper(md))
            x = float(fx(md))
            y = float(fy(md))
        else:
            tvd = float(row["TVD"]) if np.isfinite(row["TVD"]) else md
            time_ms = float(time_mapper(tvd))
            x = float(fx(tvd))
            y = float(fy(tvd))
        if not np.isfinite(time_ms):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        record = {
            "SampleID": f"{well_name}_{row_idx}",
            "WellName": well_name,
            "DEPT": md,
            "TVD": tvd,
            "TIME": time_ms,
            "X": x,
            "Y": y,
        }
        for curve in STANDARD_LOG_CURVES:
            record[curve] = row.get(curve, np.nan)
        records.append(record)

    out_df = pd.DataFrame(records)
    if out_df.empty:
        return out_df, {"WellName": well_name, "SourceKind": source_kind, "RawLogRows": int(len(las_df)), "PointRows": 0, "InTargetRows": 0}

    out_df = assign_surface_times(out_df, surface_tables)
    out_df = classify_time_domain(out_df)
    out_df = validate_surface_order(out_df, min_thickness=1.0)
    out_df["LayerGroup"] = out_df["TimeDomainClass"].map(TIME_CLASS_TO_LAYER).fillna("OUT_OF_TARGET")
    out_df["InTargetT4T7"] = out_df["TimeDomainClass"].isin(TIME_CLASS_TO_LAYER.keys()) & out_df["Check_All"].fillna(False)
    out_df["SurfaceOrderValid"] = out_df["Check_Order_T4_T5_T6_T7"].fillna(False)
    out_df["T4Time"] = pd.to_numeric(out_df["T4_TIME"], errors="coerce")
    out_df["T5Time"] = pd.to_numeric(out_df["T5_TIME"], errors="coerce")
    out_df["T6Time"] = pd.to_numeric(out_df["T6_TIME"], errors="coerce")
    out_df["T7Time"] = pd.to_numeric(out_df["T7_TIME"], errors="coerce")

    summary = {
        "WellName": well_name,
        "SourceKind": source_kind,
        "RawLogRows": int(len(las_df)),
        "PointRows": int(len(out_df)),
        "InTargetRows": int(out_df["InTargetT4T7"].sum()),
    }
    return out_df, summary


def downsample_points_by_time(point_df: pd.DataFrame, max_points_per_well: int) -> pd.DataFrame:
    if max_points_per_well <= 0 or len(point_df) <= max_points_per_well:
        return point_df.copy()
    idx = np.linspace(0, len(point_df) - 1, num=max_points_per_well, dtype=int)
    return point_df.iloc[idx].reset_index(drop=True)


def build_interval_table(point_df: pd.DataFrame) -> pd.DataFrame:
    if point_df.empty:
        return pd.DataFrame(
            columns=[
                "WellName",
                "CanonicalWellName",
                "SourceKind",
                "CoverageClass",
                "T4_DEPT",
                "T4_TVD",
                "T4_TIME",
                "T5_DEPT",
                "T5_TVD",
                "T5_TIME",
                "T6_DEPT",
                "T6_TVD",
                "T6_TIME",
                "T7_DEPT",
                "T7_TVD",
                "T7_TIME",
            ]
        )

    def nearest_row(surface_col: str) -> pd.Series:
        work_df = point_df[["DEPT", "TVD", "TIME", surface_col]].copy()
        work_df["TIME"] = pd.to_numeric(work_df["TIME"], errors="coerce")
        work_df[surface_col] = pd.to_numeric(work_df[surface_col], errors="coerce")
        work_df = work_df.dropna(subset=["TIME", surface_col])
        if work_df.empty:
            return pd.Series(dtype=object)
        dist = (work_df["TIME"] - work_df[surface_col]).abs()
        idx = dist.idxmin()
        return point_df.loc[idx]

    t4_row = nearest_row("T4Time")
    t5_row = nearest_row("T5Time")
    t6_row = nearest_row("T6Time")
    t7_row = nearest_row("T7Time")
    coverage_class = "FULL_T4_T7" if set(point_df["LayerGroup"].dropna().astype(str)) >= {"沙三段", "沙四段"} else "PARTIAL_T4_T7"

    def get_boundary_value(row: pd.Series, row_col: str, surface_col: str) -> float:
        if row.empty:
            return np.nan
        if row_col == "TIME":
            return float(pd.to_numeric(row.get(surface_col), errors="coerce"))
        return float(pd.to_numeric(row.get(row_col), errors="coerce"))

    row = {
        "WellName": str(point_df["WellName"].iloc[0]),
        "CanonicalWellName": str(point_df["WellName"].iloc[0]),
        "SourceKind": str(point_df["SourceKind"].iloc[0]) if "SourceKind" in point_df.columns else "",
        "CoverageClass": coverage_class,
        "T4_DEPT": get_boundary_value(t4_row, "DEPT", "T4Time"),
        "T4_TVD": get_boundary_value(t4_row, "TVD", "T4Time"),
        "T4_TIME": get_boundary_value(t4_row, "TIME", "T4Time"),
        "T5_DEPT": get_boundary_value(t5_row, "DEPT", "T5Time"),
        "T5_TVD": get_boundary_value(t5_row, "TVD", "T5Time"),
        "T5_TIME": get_boundary_value(t5_row, "TIME", "T5Time"),
        "T6_DEPT": get_boundary_value(t6_row, "DEPT", "T6Time"),
        "T6_TVD": get_boundary_value(t6_row, "TVD", "T6Time"),
        "T6_TIME": get_boundary_value(t6_row, "TIME", "T6Time"),
        "T7_DEPT": get_boundary_value(t7_row, "DEPT", "T7Time"),
        "T7_TVD": get_boundary_value(t7_row, "TVD", "T7Time"),
        "T7_TIME": get_boundary_value(t7_row, "TIME", "T7Time"),
    }
    return pd.DataFrame([row])


def select_target_interval_points(point_df: pd.DataFrame) -> pd.DataFrame:
    work_df = point_df.copy()
    time_series = pd.to_numeric(work_df["TIME"], errors="coerce")
    t4 = pd.to_numeric(work_df["T4Time"], errors="coerce")
    t7 = pd.to_numeric(work_df["T7Time"], errors="coerce")
    in_window = time_series.ge(t4) & time_series.le(t7)
    in_domain = work_df["TimeDomainClass"].isin(TIME_CLASS_TO_LAYER.keys())
    work_df["InTargetWindow"] = in_window.fillna(False)
    work_df["InTargetDomain"] = in_domain.fillna(False)
    work_df["InTargetT4T7"] = work_df["InTargetWindow"] & work_df["InTargetDomain"]
    selected = work_df[work_df["InTargetT4T7"]].copy()
    if selected.empty:
        return selected

    selected["DEPT"] = pd.to_numeric(selected["DEPT"], errors="coerce")
    selected = selected.sort_values("DEPT").reset_index(drop=True)
    dept_diff = selected["DEPT"].diff().fillna(0.0)
    new_segment = dept_diff.gt(0.126)
    selected["SegmentID"] = new_segment.cumsum().astype(int)
    segment_sizes = selected.groupby("SegmentID").size().sort_values(ascending=False)
    best_segment = int(segment_sizes.index[0])
    selected = selected[selected["SegmentID"] == best_segment].copy().reset_index(drop=True)
    return selected


def trim_edge_invalid_points(point_df: pd.DataFrame, usable_mask: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    point_df = point_df.reset_index(drop=True).copy()
    usable_mask = usable_mask.reset_index(drop=True).fillna(False).astype(bool)
    if point_df.empty:
        return point_df, usable_mask
    usable_indices = usable_mask[usable_mask].index.to_list()
    if not usable_indices:
        return point_df.iloc[0:0].copy(), usable_mask.iloc[0:0].copy()
    start_idx = int(usable_indices[0])
    end_idx = int(usable_indices[-1])
    trimmed_df = point_df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    trimmed_mask = usable_mask.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    return trimmed_df, trimmed_mask


def build_qc_summary_row(
    well_name: str,
    canonical_name: str,
    source_kind: str,
    raw_target_df: pd.DataFrame,
    final_point_df: pd.DataFrame,
    final_core_df: pd.DataFrame,
    usable_mask_before_trim: pd.Series,
) -> pd.DataFrame:
    raw_target_df = raw_target_df.reset_index(drop=True)
    final_point_df = final_point_df.reset_index(drop=True)
    usable_mask_before_trim = usable_mask_before_trim.reset_index(drop=True).fillna(False).astype(bool)

    usable_indices = usable_mask_before_trim[usable_mask_before_trim].index.to_list()
    if usable_indices:
        kept_start = int(usable_indices[0])
        kept_end = int(usable_indices[-1])
    else:
        kept_start = -1
        kept_end = -1

    raw_target_rows = int(len(raw_target_df))
    final_rows = int(len(final_point_df))
    trimmed_head_rows = int(max(0, kept_start))
    trimmed_tail_rows = int(max(0, raw_target_rows - kept_end - 1)) if kept_end >= 0 else int(raw_target_rows)
    mid_invalid_rows = int((~usable_mask_before_trim).sum() - trimmed_head_rows - trimmed_tail_rows)
    final_usable_rows = int(final_core_df["SampleUsableForModel"].fillna(False).sum()) if "SampleUsableForModel" in final_core_df.columns else final_rows

    row = {
        "WellName": well_name,
        "CanonicalWellName": canonical_name,
        "SourceKind": source_kind,
        "RawTargetRows": raw_target_rows,
        "FinalMainRows": final_rows,
        "FinalUsableRows": final_usable_rows,
        "TrimmedHeadRows": trimmed_head_rows,
        "TrimmedTailRows": trimmed_tail_rows,
        "InteriorInvalidRows": int(max(0, mid_invalid_rows)),
        "HasInteriorInvalid": bool(mid_invalid_rows > 0),
        "IsContinuousAfterTrim": True,
    }
    return pd.DataFrame([row])


def sanitize_attribute_value(value: float, attr_name: str) -> float:
    if not np.isfinite(value):
        return np.nan
    rule = ATTRIBUTE_VALUE_RULES.get(attr_name)
    if rule is None:
        return float(value)
    if value < float(rule["min"]) or value > float(rule["max"]):
        return np.nan
    return float(value)


def annotate_attribute_qc(core_df: pd.DataFrame, context_df: pd.DataFrame, volume_specs: list[VolumeSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_df = core_df.reset_index(drop=True).copy()
    context_df = context_df.reset_index(drop=True).copy()

    sample_usable = pd.Series(True, index=core_df.index, dtype=bool)
    center_invalid_any = pd.Series(False, index=core_df.index, dtype=bool)
    neighborhood_empty_any = pd.Series(False, index=core_df.index, dtype=bool)
    neighborhood_sparse_any = pd.Series(False, index=core_df.index, dtype=bool)

    for spec in volume_specs:
        attr_name = spec.output_prefix
        center_series = pd.to_numeric(core_df[attr_name], errors="coerce")
        center_series = center_series.apply(lambda v: sanitize_attribute_value(v, attr_name))
        core_df[attr_name] = center_series
        center_valid = center_series.notna()

        context_cols = [f"{attr_name}_{offset_to_axis_label(dx, 'x')}_{offset_to_axis_label(dy, 'y')}" for dx in NEIGHBOR_OFFSETS for dy in NEIGHBOR_OFFSETS]
        context_matrix = []
        for context_col in context_cols:
            if context_col not in context_df.columns:
                context_df[context_col] = np.nan
            cleaned = pd.to_numeric(context_df[context_col], errors="coerce").apply(lambda v: sanitize_attribute_value(v, attr_name))
            context_df[context_col] = cleaned
            context_matrix.append(cleaned.to_numpy(dtype=float))

        context_values = np.column_stack(context_matrix) if context_matrix else np.empty((len(core_df), 0), dtype=float)
        valid_mask = np.isfinite(context_values)
        valid_count = valid_mask.sum(axis=1).astype(int)

        mean_values = np.full(len(core_df), np.nan, dtype=float)
        std_values = np.full(len(core_df), np.nan, dtype=float)
        min_values = np.full(len(core_df), np.nan, dtype=float)
        max_values = np.full(len(core_df), np.nan, dtype=float)
        for idx in range(len(core_df)):
            valid_vals = context_values[idx][valid_mask[idx]]
            if valid_vals.size == 0:
                continue
            mean_values[idx] = float(valid_vals.mean())
            std_values[idx] = float(valid_vals.std(ddof=0))
            min_values[idx] = float(valid_vals.min())
            max_values[idx] = float(valid_vals.max())

        core_df[f"{attr_name}Mean"] = mean_values
        core_df[f"{attr_name}Std"] = std_values
        core_df[f"{attr_name}Min"] = min_values
        core_df[f"{attr_name}Max"] = max_values
        core_df[f"{attr_name}ValidCount"] = valid_count
        core_df[f"{attr_name}CenterValid"] = center_valid
        core_df[f"{attr_name}WindowComplete"] = valid_count == len(context_cols)
        core_df[f"{attr_name}ModelReady"] = center_valid & (valid_count >= MIN_VALID_NEIGHBORS_FOR_MODEL)

        center_invalid_any |= ~center_valid
        neighborhood_empty_any |= pd.Series(valid_count == 0, index=core_df.index)
        neighborhood_sparse_any |= pd.Series(valid_count < MIN_VALID_NEIGHBORS_FOR_MODEL, index=core_df.index)
        sample_usable &= core_df[f"{attr_name}ModelReady"]

    sample_status = np.where(
        sample_usable,
        "usable",
        np.where(
            center_invalid_any,
            "center_invalid",
            np.where(neighborhood_empty_any, "neighborhood_empty", np.where(neighborhood_sparse_any, "neighborhood_sparse", "qc_issue")),
        ),
    )
    core_df["SampleUsableForModel"] = sample_usable
    core_df["SampleUsableStatus"] = sample_status
    return core_df, context_df


def extract_attribute_tables_for_points(point_df: pd.DataFrame, volume_specs: list[VolumeSpec], trace_df: pd.DataFrame):
    core_df = point_df[["SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", *STANDARD_LOG_CURVES]].copy()
    interval_df = build_interval_table(point_df)

    trace_tree, trace_ids = build_trace_tree(trace_df)
    context_rows: list[dict[str, Any]] = []
    volume_contexts = []
    for spec in volume_specs:
        handle, samples, trace_at = open_volume_context(spec.path)
        volume_contexts.append((spec, handle, samples, trace_at))

    try:
        context_map: dict[str, dict[str, Any]] = {
            str(row.SampleID): {
                "SampleID": str(row.SampleID),
                "WellName": str(row.WellName),
                "X": float(row.X),
                "Y": float(row.Y),
                "TIME": float(row.TIME),
            }
            for row in core_df.itertuples(index=False)
        }
        for spec, _, samples, trace_at in volume_contexts:
            center_values = []

            for row in core_df.itertuples(index=False):
                center_val = inverse_distance_xy_sample(
                    x=float(row.X),
                    y=float(row.Y),
                    time_ms=float(row.TIME),
                    tree=trace_tree,
                    trace_ids=trace_ids,
                    samples=samples,
                    trace_at=trace_at,
                )
                center_values.append(center_val)

                neighbor_vals: list[float] = []
                for dx in NEIGHBOR_OFFSETS:
                    for dy in NEIGHBOR_OFFSETS:
                        nx = float(row.X) + dx
                        ny = float(row.Y) + dy
                        attr_val = inverse_distance_xy_sample(
                            x=nx,
                            y=ny,
                            time_ms=float(row.TIME),
                            tree=trace_tree,
                            trace_ids=trace_ids,
                            samples=samples,
                            trace_at=trace_at,
                        )
                        x_label = offset_to_axis_label(dx, "x")
                        y_label = offset_to_axis_label(dy, "y")
                        context_map[str(row.SampleID)][f"{spec.output_prefix}_{x_label}_{y_label}"] = attr_val
                        neighbor_vals.append(attr_val)

            core_df[spec.output_prefix] = center_values
    finally:
        for _, handle, _, _ in volume_contexts:
            handle.close()

    context_rows = [context_map[key] for key in core_df["SampleID"].astype(str).tolist()]
    context_df = pd.DataFrame(context_rows)
    core_df, context_df = annotate_attribute_qc(core_df=core_df, context_df=context_df, volume_specs=volume_specs)
    return core_df, context_df, interval_df


def process_single_well(
    well_name: str,
    log_dir: Path,
    track_dir: Path,
    time_dir: Path,
    surface_tables,
    volume_specs: list[VolumeSpec],
    trace_df: pd.DataFrame,
    out_dir: Path,
    max_points_per_well: int,
    max_log_rows_per_well: int,
) -> dict[str, Any]:
    canonical_name = canonicalize_well_name(well_name)
    las_path = resolve_las_path(log_dir, well_name)
    if las_path is None:
        return {"WellName": well_name, "CanonicalWellName": canonical_name, "Status": "missing_las"}

    time_path = time_dir / f"{well_name}.dat"
    if not time_path.exists():
        time_path = time_dir / f"{canonical_name}.dat"
    if not time_path.exists():
        return {"WellName": well_name, "CanonicalWellName": canonical_name, "Status": "missing_timedepth"}

    track_path = track_dir / f"{well_name}.dat"
    if not track_path.exists():
        track_path = track_dir / f"{canonical_name}.dat"
    track_df = read_well_track(track_path) if track_path.exists() else None
    if track_df is None or track_df.empty:
        x, y = read_well_xy_from_las(las_path)
        if x is None or y is None:
            return {"WellName": well_name, "CanonicalWellName": canonical_name, "Status": "missing_xy"}
        track_df = pd.DataFrame([{"MD": 0.0, "TVD": 0.0, "X": x, "Y": y}])

    las_df = normalize_las_curves(read_las_ascii_block(las_path))
    las_df = thin_rows_by_stride(las_df, max_rows=max_log_rows_per_well)
    time_df = read_timedepth_file(time_path)
    point_df, build_summary = build_point_rows_for_well(
        well_name=canonical_name,
        las_df=las_df,
        track_df=track_df,
        time_df=time_df,
        surface_tables=surface_tables,
    )
    if point_df.empty:
        return {"WellName": well_name, "CanonicalWellName": canonical_name, "Status": "empty_points"}

    point_df = select_target_interval_points(point_df)
    if point_df.empty:
        return {"WellName": well_name, "CanonicalWellName": canonical_name, "Status": "empty_t4_t7"}
    point_df = downsample_points_by_time(point_df, max_points_per_well=max_points_per_well)
    raw_target_point_df = point_df.copy()

    point_df["SourceKind"] = build_summary["SourceKind"]
    core_df, context_df, interval_df = extract_attribute_tables_for_points(point_df=point_df, volume_specs=volume_specs, trace_df=trace_df)
    usable_mask_before_trim = core_df["SampleUsableForModel"].copy()
    trimmed_point_df, trimmed_usable_mask = trim_edge_invalid_points(
        point_df=point_df,
        usable_mask=core_df["SampleUsableForModel"],
    )
    if trimmed_point_df.empty:
        return {"WellName": well_name, "CanonicalWellName": canonical_name, "Status": "empty_after_edge_trim"}
    keep_ids = set(trimmed_point_df["SampleID"].astype(str).tolist())
    core_df = core_df[core_df["SampleID"].astype(str).isin(keep_ids)].copy().reset_index(drop=True)
    context_df = context_df[context_df["SampleID"].astype(str).isin(keep_ids)].copy().reset_index(drop=True)
    point_df = trimmed_point_df.copy()
    core_df["SampleUsableForModel"] = trimmed_usable_mask.to_numpy()
    core_df["SampleUsableStatus"] = np.where(core_df["SampleUsableForModel"], "usable", "interior_invalid")
    interval_df = build_interval_table(point_df)
    qc_df = build_qc_summary_row(
        well_name=well_name,
        canonical_name=canonical_name,
        source_kind=build_summary["SourceKind"],
        raw_target_df=raw_target_point_df,
        final_point_df=point_df,
        final_core_df=core_df,
        usable_mask_before_trim=usable_mask_before_trim,
    )

    main_stat_cols = []
    for attr_name in MAIN_ATTRIBUTE_COLUMNS:
        for suffix in SUMMARY_STAT_SUFFIXES:
            col = f"{attr_name}{suffix}"
            if col in core_df.columns:
                main_stat_cols.append(col)
    keep_main_cols = ["SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", *STANDARD_LOG_CURVES, *MAIN_ATTRIBUTE_COLUMNS, *main_stat_cols, *MAIN_LOGIC_COLUMNS]
    core_df = core_df[[col for col in keep_main_cols if col in core_df.columns]].copy()

    core_csv = out_dir / f"{canonical_name}_t4_t7_real_well_main.csv"
    context_csv = out_dir / f"{canonical_name}_t4_t7_real_well_3x3_context.csv"
    horizon_csv = out_dir / f"{canonical_name}_t4_t7_real_well_interval.csv"
    qc_csv = out_dir / f"{canonical_name}_t4_t7_real_well_qc_summary.csv"
    legacy_horizon_map_csv = out_dir / f"{canonical_name}_t4_t7_real_well_horizon_map.csv"
    ensure_parent(core_csv)
    if legacy_horizon_map_csv.exists():
        legacy_horizon_map_csv.unlink()
    core_df.to_csv(core_csv, index=False, encoding="utf-8-sig")
    context_df.to_csv(context_csv, index=False, encoding="utf-8-sig")
    interval_df.to_csv(horizon_csv, index=False, encoding="utf-8-sig")
    qc_df.to_csv(qc_csv, index=False, encoding="utf-8-sig")

    return {
        "WellName": well_name,
        "CanonicalWellName": canonical_name,
        "Status": "ok",
        "SourceKind": build_summary["SourceKind"],
        "LogRowsUsed": int(len(las_df)),
        "TargetPointRows": int(len(point_df)),
        "MainRows": int(len(core_df)),
        "ContextRows": int(len(context_df)),
        "IntervalRows": int(len(interval_df)),
        "ModelUsableRows": int(core_df["SampleUsableForModel"].fillna(False).sum()),
        "ModelUnusableRows": int((~core_df["SampleUsableForModel"].fillna(False)).sum()),
        "MaxPointsPerWell": int(max_points_per_well),
        "MaxLogRowsPerWell": int(max_log_rows_per_well),
        "MainCsv": str(core_csv),
        "ContextCsv": str(context_csv),
        "IntervalCsv": str(horizon_csv),
        "QcCsv": str(qc_csv),
    }


def process_single_well_from_args(
    well_name: str,
    log_dir: str,
    track_dir: str,
    time_dir: str,
    layer_dir: str,
    volume_specs_cfg: list[tuple[str, str]],
    trace_header_csv: str,
    out_dir: str,
    max_points_per_well: int,
    max_log_rows_per_well: int,
) -> dict[str, Any]:
    volume_specs = [VolumeSpec(output_prefix=name, path=Path(path)) for name, path in volume_specs_cfg]
    trace_df = build_trace_grid(Path(trace_header_csv))
    surface_tables = load_surface_tables(Path(layer_dir))
    return process_single_well(
        well_name=well_name,
        log_dir=Path(log_dir),
        track_dir=Path(track_dir),
        time_dir=Path(time_dir),
        surface_tables=surface_tables,
        volume_specs=volume_specs,
        trace_df=trace_df,
        out_dir=Path(out_dir),
        max_points_per_well=max_points_per_well,
        max_log_rows_per_well=max_log_rows_per_well,
    )


def resolve_well_list(config: dict[str, Any], log_dir: Path) -> list[str]:
    wells = [str(item).strip() for item in config.get("wells", []) if str(item).strip()]
    if wells:
        return wells
    if config.get("discover_wells_from_log_dir", False):
        return discover_las_wells(log_dir)
    raise ValueError("config must provide non-empty wells or set discover_wells_from_log_dir=true")


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(Path(args.config).resolve())

    log_dir = Path(config["log_dir"]).resolve()
    track_dir = Path(config["track_dir"]).resolve()
    time_dir = Path(config["time_dir"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    trace_header_csv = Path(config["trace_header_csv"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    summary_csv = Path(config["summary_csv"]).resolve()
    wells = resolve_well_list(config, log_dir=log_dir)
    volume_specs = build_volume_specs(config["volume_paths"])
    volume_specs_cfg = [(spec.output_prefix, str(spec.path)) for spec in volume_specs]

    trace_df = build_trace_grid(trace_header_csv)
    surface_tables = load_surface_tables(layer_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    max_workers = int(args.max_workers)
    if max_workers > 1 and len(wells) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    process_single_well_from_args,
                    well_name,
                    str(log_dir),
                    str(track_dir),
                    str(time_dir),
                    str(layer_dir),
                    volume_specs_cfg,
                    str(trace_header_csv),
                    str(output_root / canonicalize_well_name(well_name)),
                    int(args.max_points_per_well),
                    int(args.max_log_rows_per_well),
                ): well_name
                for well_name in wells
            }
            for future in as_completed(future_map):
                well_name = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"WellName": well_name, "CanonicalWellName": canonicalize_well_name(well_name), "Status": "failed", "Reason": str(exc)}
                summary_rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
    else:
        for well_name in wells:
            try:
                row = process_single_well(
                    well_name=well_name,
                    log_dir=log_dir,
                    track_dir=track_dir,
                    time_dir=time_dir,
                    surface_tables=surface_tables,
                    volume_specs=volume_specs,
                    trace_df=trace_df,
                    out_dir=output_root / canonicalize_well_name(well_name),
                    max_points_per_well=int(args.max_points_per_well),
                    max_log_rows_per_well=int(args.max_log_rows_per_well),
                )
            except Exception as exc:
                row = {"WellName": well_name, "CanonicalWellName": canonicalize_well_name(well_name), "Status": "failed", "Reason": str(exc)}
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    summary_df = pd.DataFrame(summary_rows)
    ensure_parent(summary_csv)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
