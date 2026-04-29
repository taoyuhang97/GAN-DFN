from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_WELL_FILES,
    EXPECTED_TIME_STEP,
    MANUAL_STRATA_CONFIG,
    TIME_STEP_TOLERANCE,
    canonicalize_well_name,
    resolve_depth_column,
)


def assign_manual_strata(df: pd.DataFrame, well_name: str, boundary_tolerance: float) -> pd.DataFrame:
    canonical_well_name = canonicalize_well_name(well_name)
    if canonical_well_name not in MANUAL_STRATA_CONFIG:
        raise ValueError(f"Missing manual strata config for well: {canonical_well_name}")

    out = df.copy()
    depth_col = resolve_depth_column(out)
    depth = pd.to_numeric(out[depth_col], errors="coerce")
    intervals = MANUAL_STRATA_CONFIG[canonical_well_name]

    out["DepthForStrata"] = depth
    out["StrataName"] = pd.NA
    out["StrataTop"] = np.nan
    out["StrataBase"] = np.nan

    for idx, interval in enumerate(intervals):
        lower = -np.inf if interval["top"] is None else float(interval["top"])
        if idx == 0 and np.isfinite(lower):
            lower -= float(boundary_tolerance)

        if idx < len(intervals) - 1:
            next_top = intervals[idx + 1]["top"]
            upper = np.inf if next_top is None else float(next_top)
            mask = depth >= lower
            if np.isfinite(upper):
                mask &= depth < upper
        else:
            upper = np.inf if interval["base"] is None else float(interval["base"])
            if np.isfinite(upper):
                upper += float(boundary_tolerance)
            mask = depth >= lower
            if np.isfinite(upper):
                mask &= depth <= upper

        out.loc[mask, "StrataName"] = interval["name"]
        out.loc[mask, "StrataTop"] = interval["top"]
        out.loc[mask, "StrataBase"] = interval["base"]

    return out


def validate_uniform_time_sampling(df: pd.DataFrame, well_name: str) -> None:
    if "TIME" not in df.columns:
        raise ValueError(f"{well_name} sample missing TIME column")

    time_series = pd.to_numeric(df["TIME"], errors="coerce")
    if time_series.isna().any():
        bad_count = int(time_series.isna().sum())
        raise ValueError(f"{well_name} sample has {bad_count} invalid TIME rows")

    time_values = time_series.to_numpy(dtype=np.float64)
    if time_values.size <= 1:
        return

    time_diff = np.diff(time_values)
    bad_mask = np.abs(time_diff - EXPECTED_TIME_STEP) > TIME_STEP_TOLERANCE
    if np.any(bad_mask):
        bad_indices = np.flatnonzero(bad_mask)[:5]
        examples = [
            f"{idx}:{time_values[idx]:.6f}->{time_values[idx + 1]:.6f},diff={time_diff[idx]:.6f}"
            for idx in bad_indices
        ]
        raise ValueError(
            f"{well_name} sample TIME step is not uniform {EXPECTED_TIME_STEP} ms; "
            f"examples={examples}"
        )


def normalize_well_sample_df(df: pd.DataFrame, well_name: str) -> pd.DataFrame:
    out = df.copy()
    depth_col = resolve_depth_column(out)
    out[depth_col] = pd.to_numeric(out[depth_col], errors="coerce")
    if out[depth_col].isna().any():
        bad_count = int(out[depth_col].isna().sum())
        raise ValueError(f"{well_name} sample has {bad_count} invalid {depth_col} rows")

    if "TIME" not in out.columns:
        raise ValueError(f"{well_name} sample missing TIME column")
    out["TIME"] = pd.to_numeric(out["TIME"], errors="coerce")

    sort_cols: list[str] = ["TIME", depth_col]
    if "ROW_IN_WELL" in out.columns:
        out["ROW_IN_WELL"] = pd.to_numeric(out["ROW_IN_WELL"], errors="coerce")
        sort_cols.append("ROW_IN_WELL")

    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    validate_uniform_time_sampling(out, well_name=well_name)
    out["ROW_IN_WELL"] = np.arange(len(out), dtype=np.int64)
    return out


def build_labeled_dataset(
    data_dir: Path,
    selected_wells: list[str],
    boundary_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    labeled_parts = []
    source_columns = {}

    for requested_well_name in selected_wells:
        well_name = canonicalize_well_name(requested_well_name)
        file_name = DEFAULT_WELL_FILES.get(well_name)
        if not file_name:
            raise ValueError(f"Unsupported well name: {requested_well_name}")
        csv_path = data_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing sample csv: {csv_path}")

        df = pd.read_csv(csv_path).copy()
        df = normalize_well_sample_df(df, well_name=well_name)
        df["WellName"] = well_name
        labeled_df = assign_manual_strata(df, well_name, boundary_tolerance)
        source_columns[well_name] = list(labeled_df.columns)
        labeled_parts.append(labeled_df)

    combined = pd.concat(labeled_parts, ignore_index=True)
    combined = combined.sort_values(["WellName", "ROW_IN_WELL"]).reset_index(drop=True)
    return combined, source_columns


def export_filtered_strata_dataset(
    labeled_df: pd.DataFrame,
    target_strata: str,
    selected_wells: list[str],
    source_columns: dict[str, list[str]],
    output_dir: Path,
    summary_filename: str,
    summary_builder,
) -> tuple[pd.DataFrame, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    valid_wells = []

    for requested_well_name in selected_wells:
        well_name = canonicalize_well_name(requested_well_name)
        sub = labeled_df[
            (labeled_df["WellName"] == well_name) & (labeled_df["StrataName"] == target_strata)
        ].copy()
        if "ROW_IN_WELL" not in sub.columns:
            sub["ROW_IN_WELL"] = np.arange(len(sub))

        output_cols = [col for col in source_columns[well_name] if col in sub.columns]
        if not output_cols:
            output_cols = list(sub.columns)

        save_path = output_dir / DEFAULT_WELL_FILES[well_name]
        if sub.empty:
            pd.DataFrame(columns=output_cols).to_csv(save_path, index=False, encoding="utf-8-sig")
        else:
            sub = sub.sort_values("ROW_IN_WELL").reset_index(drop=True)
            sub[output_cols].to_csv(save_path, index=False, encoding="utf-8-sig")
            valid_wells.append(well_name)

        summary_rows.append(summary_builder(well_name=well_name, target_strata=target_strata, sub=sub))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / summary_filename, index=False, encoding="utf-8-sig")
    return summary_df, valid_wells
