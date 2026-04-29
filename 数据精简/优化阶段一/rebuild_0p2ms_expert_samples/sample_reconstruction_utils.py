from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)


@dataclass
class ResampleStats:
    well_name: str
    input_csv: Path
    output_csv: Path
    original_row_count: int
    valid_input_row_count: int
    output_row_count: int
    dropped_invalid_time_count: int
    duplicate_time_count: int
    time_min: float
    time_max: float


def read_csv_flexible(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read csv: {csv_path}") from last_error


def infer_well_name_from_path(csv_path: Path) -> str:
    stem = csv_path.stem
    suffix = "_around_data"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def discover_around_data_csvs(input_dir: Path) -> list[Path]:
    csv_paths = []
    for csv_path in sorted(input_dir.glob("*_around_data.csv")):
        name_lower = csv_path.name.lower()
        if "_rebuild" in name_lower:
            continue
        csv_paths.append(csv_path)
    return csv_paths


def _first_valid_value(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def _normalize_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    numeric_cols: list[str] = []
    non_numeric_cols: list[str] = []

    for col in out.columns:
        if col == "TIME":
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            numeric_cols.append(col)
            continue

        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().any():
            out[col] = converted
            numeric_cols.append(col)
        else:
            non_numeric_cols.append(col)

    return out, numeric_cols, non_numeric_cols


def _replace_invalid_sentinels(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col].isin(INVALID_SENTINELS), col] = np.nan
    return out


def _build_uniform_grid(time_values: pd.Series, interval: float) -> np.ndarray:
    if interval <= 0:
        raise ValueError(f"interval must be > 0, got: {interval}")

    if time_values.empty:
        return np.array([], dtype=np.float64)

    eps = 1e-9
    time_min = float(time_values.min())
    time_max = float(time_values.max())
    start = np.ceil((time_min - eps) / interval) * interval
    end = np.floor((time_max + eps) / interval) * interval
    if start > end:
        return np.array([], dtype=np.float64)

    count = int(round((end - start) / interval)) + 1
    return np.round(np.linspace(start, end, count), 6)


def _aggregate_duplicate_times(
    df: pd.DataFrame,
    numeric_cols: list[str],
    non_numeric_cols: list[str],
) -> pd.DataFrame:
    agg_map: dict[str, object] = {}
    for col in numeric_cols:
        agg_map[col] = "mean"
    for col in non_numeric_cols:
        agg_map[col] = _first_valid_value
    return df.groupby("TIME", as_index=False, sort=True).agg(agg_map)


def _resample_numeric_columns(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_grid: np.ndarray,
) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(index=pd.Index(target_grid, name="TIME"))

    source_indexed = df.set_index("TIME")[numeric_cols].sort_index()
    target_index = pd.Index(target_grid, name="TIME")
    union_index = source_indexed.index.union(target_index).sort_values()

    out = source_indexed.reindex(union_index)
    out = out.interpolate(method="index", limit_direction="both")
    return out.reindex(target_index)


def _resample_non_numeric_columns(
    df: pd.DataFrame,
    non_numeric_cols: list[str],
    target_grid: np.ndarray,
) -> pd.DataFrame:
    if not non_numeric_cols:
        return pd.DataFrame(index=pd.Index(target_grid, name="TIME"))

    source_indexed = df.set_index("TIME")[non_numeric_cols].sort_index()
    target_index = pd.Index(target_grid, name="TIME")
    union_index = source_indexed.index.union(target_index).sort_values()

    out = source_indexed.reindex(union_index)
    out = out.ffill().bfill()
    return out.reindex(target_index)


def resample_around_data_df(df: pd.DataFrame, interval: float) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if "TIME" not in df.columns:
        raise ValueError("Missing required column: TIME")

    original_columns = list(df.columns)
    original_row_count = int(len(df))
    time_numeric = pd.to_numeric(df["TIME"], errors="coerce")
    valid_mask = time_numeric.notna()
    dropped_invalid_time_count = int((~valid_mask).sum())

    work_df = df.loc[valid_mask].copy()
    if work_df.empty:
        raise ValueError("No valid rows remain after TIME cleaning")

    work_df["TIME"] = time_numeric.loc[valid_mask].to_numpy(dtype=np.float64)
    work_df = work_df.sort_values("TIME").reset_index(drop=True)
    work_df, numeric_cols, non_numeric_cols = _normalize_numeric_columns(work_df)
    work_df = _replace_invalid_sentinels(work_df, numeric_cols)

    valid_input_row_count = int(len(work_df))
    unique_time_count = int(work_df["TIME"].nunique())
    duplicate_time_count = valid_input_row_count - unique_time_count
    if duplicate_time_count > 0:
        work_df = _aggregate_duplicate_times(
            df=work_df,
            numeric_cols=numeric_cols,
            non_numeric_cols=non_numeric_cols,
        )

    target_grid = _build_uniform_grid(work_df["TIME"], interval)
    if target_grid.size == 0:
        raise ValueError("Failed to build target TIME grid")

    numeric_resampled = _resample_numeric_columns(work_df, numeric_cols, target_grid)
    non_numeric_resampled = _resample_non_numeric_columns(work_df, non_numeric_cols, target_grid)

    output_parts: dict[str, object] = {}
    for col in original_columns:
        if col == "TIME":
            output_parts[col] = target_grid
        elif col in numeric_resampled.columns:
            output_parts[col] = numeric_resampled[col].to_numpy()
        elif col in non_numeric_resampled.columns:
            output_parts[col] = non_numeric_resampled[col].to_numpy()
        else:
            output_parts[col] = np.full(len(target_grid), np.nan)

    out_df = pd.DataFrame(output_parts, columns=original_columns)
    stats = {
        "original_row_count": original_row_count,
        "valid_input_row_count": valid_input_row_count,
        "output_row_count": int(len(out_df)),
        "dropped_invalid_time_count": dropped_invalid_time_count,
        "duplicate_time_count": duplicate_time_count,
        "time_min": float(target_grid[0]),
        "time_max": float(target_grid[-1]),
    }
    return out_df, stats


def resample_around_data_csv(
    input_csv: Path,
    output_csv: Path,
    interval: float,
    well_name: str | None = None,
) -> ResampleStats:
    df = read_csv_flexible(input_csv)
    out_df, stats = resample_around_data_df(df, interval=interval)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return ResampleStats(
        well_name=str(well_name).strip() if well_name is not None else infer_well_name_from_path(output_csv),
        input_csv=input_csv,
        output_csv=output_csv,
        original_row_count=int(stats["original_row_count"]),
        valid_input_row_count=int(stats["valid_input_row_count"]),
        output_row_count=int(stats["output_row_count"]),
        dropped_invalid_time_count=int(stats["dropped_invalid_time_count"]),
        duplicate_time_count=int(stats["duplicate_time_count"]),
        time_min=float(stats["time_min"]),
        time_max=float(stats["time_max"]),
    )
