from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_WELL_FILES,
    MANUAL_STRATA_CONFIG,
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
        if "ROW_IN_WELL" not in df.columns:
            df["ROW_IN_WELL"] = np.arange(len(df))
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
