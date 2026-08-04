from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.append(str(WORKFLOW_ROOT))

from common.config import STEP1_DIR, STEP2_DIR
from common.io_utils import read_csv_flexible, write_csv_utf8


ATTRIBUTE_MAP = {
    "SEIS_TRUE": ["SEIS_TRUE", "AMPLITUDE_CENTER"],
    "COHERENCE": ["车西_相干体T4_T7_CENTER", "COHERENCE_CENTER"],
    "ANT_TRACK": ["车西_蚂蚁体T4_T7_CENTER", "ANT_CENTER"],
    "CURVATURE_MAX": ["车西_最大曲率T4_T7_CENTER", "CURVATURE_CENTER"],
    "CURVATURE_POS": ["车西_最大正曲率T4_T7_CENTER"],
    "FRACTURE_INV": ["FRACTURE_INVERSION_CENTER"],
}

WINDOW_PREFIX_MAP = {
    "COHERENCE": ["车西_相干体T4_T7", "COHERENCE"],
    "ANT_TRACK": ["车西_蚂蚁体T4_T7", "ANT"],
    "CURVATURE_MAX": ["车西_最大曲率T4_T7", "CURVATURE"],
    "CURVATURE_POS": ["车西_最大正曲率T4_T7"],
    "SEIS_TRUE": ["SEIS"],
}

INVALID_SENTINELS = {-999.25, -9999.0, -99999.0, 9999.0, 99999.0, -9999999.0}


def coalesce_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    result = pd.Series(index=df.index, dtype=object)
    for col in candidates:
        if col in df.columns:
            if result.isna().all():
                result = df[col]
            else:
                result = result.combine_first(df[col])
    return result


def clean_center_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        numeric = numeric.mask(np.isclose(numeric, sentinel, equal_nan=False))
    large_invalid = numeric.abs() >= 1e6
    numeric = numeric.mask(large_invalid)
    return numeric


def window_cols_for_prefix(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        for prefix in prefixes:
            if re.fullmatch(rf"{re.escape(prefix)}_W\d{{2}}", str(col)):
                cols.append(col)
                break
    return sorted(cols)


def clean_numeric_window(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    win = df[cols].apply(pd.to_numeric, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        win = win.mask(np.isclose(win, sentinel, equal_nan=False))
    win = win.mask(win.abs() >= 1e6)
    return win


def attach_window_stats(out: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    for target_name, prefixes in WINDOW_PREFIX_MAP.items():
        cols = window_cols_for_prefix(log_df, prefixes)
        if not cols:
            continue
        win = clean_numeric_window(log_df, cols)
        out[f"{target_name}_WIN_MEAN"] = win.mean(axis=1, skipna=True)
        out[f"{target_name}_WIN_STD"] = win.std(axis=1, skipna=True)
        out[f"{target_name}_WIN_MIN"] = win.min(axis=1, skipna=True)
        out[f"{target_name}_WIN_MAX"] = win.max(axis=1, skipna=True)
        out[f"{target_name}_WIN_VALID_COUNT"] = win.notna().sum(axis=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", type=Path, default=STEP1_DIR / "virtual_well_index.csv")
    parser.add_argument("--output-csv", type=Path, default=STEP2_DIR / "virtual_well_attributes.csv")
    args = parser.parse_args()

    index_df = read_csv_flexible(args.index_csv)
    rows: list[pd.DataFrame] = []
    for item in index_df.to_dict(orient="records"):
        source_log_csv = Path(str(item["SourceFinalLogCsv"]))
        if not source_log_csv.exists():
            continue
        log_df = read_csv_flexible(source_log_csv)
        out = pd.DataFrame(
            {
                "SourceWellName": item["SourceWellName"],
                "VirtualWellName": item["VirtualWellName"],
                "TIME": log_df["TIME"] if "TIME" in log_df.columns else None,
                "TVD": log_df["TVD"] if "TVD" in log_df.columns else log_df.get("DEPT"),
                "SourceX": item["SourceX"],
                "SourceY": item["SourceY"],
                "VirtualX": item["VirtualX"],
                "VirtualY": item["VirtualY"],
                "DistanceToSource": item["DistanceToSource"],
            }
        )
        for target_col, source_candidates in ATTRIBUTE_MAP.items():
            out[target_col] = clean_center_series(coalesce_column(log_df, source_candidates))
        out = attach_window_stats(out, log_df)
        rows.append(out)

    result_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_csv_utf8(result_df, args.output_csv)
    print(args.output_csv)
    print(f"rows={len(result_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
