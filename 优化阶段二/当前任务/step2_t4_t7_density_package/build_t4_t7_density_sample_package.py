#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step0_t4_t7_surface_tools"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import (  # noqa: E402
    DEFAULT_MIN_THICKNESS,
    assign_surface_times,
    choose_surface_files,
    classify_time_domain,
    load_surface_tables,
    validate_surface_order,
)


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
REQUIRED_COLUMNS = ("WellName", "X", "Y", "TIME", "Density", "HasFracture")
CANONICAL_ORDER = [
    "SourceFile",
    "WellName",
    "X",
    "Y",
    "TIME",
    "TVD",
    "DEPT",
    "Density",
    "DensityRaw",
    "HasFracture",
    "LayerGroup",
    "LayerGroupCode",
    "TimeDomainClass",
    "InTargetT4T7",
    "SurfaceOrderValid",
    "InputRowValid",
    "TargetRowValid",
    "DensityImputedFromHasFracture",
    "T4Time",
    "T5Time",
    "T6Time",
    "T7Time",
    "T4_MANHATTAN_DISTANCE",
    "T5_MANHATTAN_DISTANCE",
    "T6_MANHATTAN_DISTANCE",
    "T7_MANHATTAN_DISTANCE",
]
CLASS_TO_LAYER_GROUP = {
    "sha3_t4_t6": ("沙三段", "T4-T6"),
    "sha4_t6_t7": ("沙四段", "T6-T7"),
}
FILENAME_WELLNAME_SUFFIX = "_连续密度曲线_含井周属性"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an official multi-well T4-T7 density training sample package from 01 real-well density curves."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def normalize_bool_flag(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    mask = numeric.notna()
    out.loc[mask] = (numeric.loc[mask] > 0).astype("int64")
    return out


def normalize_density(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DensityRaw"] = safe_numeric(out["Density"])
    out["HasFracture"] = normalize_bool_flag(out["HasFracture"])
    impute_mask = out["DensityRaw"].isna() & out["HasFracture"].fillna(0).eq(0)
    out["Density"] = out["DensityRaw"]
    out.loc[impute_mask, "Density"] = 0.0
    out["DensityImputedFromHasFracture"] = impute_mask
    return out


def derive_well_name(file_path: Path, df: pd.DataFrame) -> str:
    if "WellName" in df.columns:
        non_empty = (
            df["WellName"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        non_empty = non_empty[non_empty.ne("")]
        if not non_empty.empty:
            return str(non_empty.iloc[0])
    stem = file_path.stem
    if stem.endswith(FILENAME_WELLNAME_SUFFIX):
        return stem[: -len(FILENAME_WELLNAME_SUFFIX)]
    return stem


def scan_input_files(input_dir: Path, file_glob: str) -> list[Path]:
    files = sorted([path for path in input_dir.glob(file_glob) if path.is_file()])
    if not files:
        raise FileNotFoundError(f"no files matched {file_glob} under {input_dir}")
    return files


def enrich_with_surfaces(df: pd.DataFrame, surfaces: dict[str, dict[str, object]], min_thickness: float) -> pd.DataFrame:
    work_df = df.copy()
    for column in ("X", "Y", "TIME"):
        work_df[column] = safe_numeric(work_df[column])
    work_df = assign_surface_times(work_df, surfaces)
    work_df = classify_time_domain(work_df)
    work_df = validate_surface_order(work_df, min_thickness=min_thickness)
    work_df["LayerGroup"] = work_df["TimeDomainClass"].map(lambda value: CLASS_TO_LAYER_GROUP.get(value, ("OUT_OF_TARGET", "OUT_OF_TARGET"))[0])
    work_df["LayerGroupCode"] = work_df["TimeDomainClass"].map(lambda value: CLASS_TO_LAYER_GROUP.get(value, ("OUT_OF_TARGET", "OUT_OF_TARGET"))[1])
    work_df["SurfaceOrderValid"] = work_df["Check_Order_T4_T5_T6_T7"].fillna(False)
    work_df["InputRowValid"] = ~work_df[["X", "Y", "TIME"]].isna().any(axis=1)
    work_df["InTargetT4T7"] = work_df["TimeDomainClass"].isin(CLASS_TO_LAYER_GROUP.keys()) & work_df["Check_All"].fillna(False)
    work_df["TargetRowValid"] = (
        work_df["InTargetT4T7"]
        & work_df["Density"].notna()
        & work_df["HasFracture"].notna()
    )
    for code in ("T4", "T5", "T6", "T7"):
        work_df[f"{code}Time"] = safe_numeric(work_df[f"{code}_TIME"])
    return work_df


def prepare_input_frame(file_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_df, encoding = read_csv_flexible(file_path)
    require_columns(raw_df, REQUIRED_COLUMNS, file_path.name)
    work_df = raw_df.copy()
    work_df["WellName"] = work_df["WellName"].astype(str).str.strip()
    inferred_name = derive_well_name(file_path, work_df)
    blank_name_mask = work_df["WellName"].eq("") | work_df["WellName"].eq("nan")
    work_df.loc[blank_name_mask, "WellName"] = inferred_name
    work_df["SourceFile"] = file_path.name
    work_df = normalize_density(work_df)
    file_info = {
        "source_file": file_path.name,
        "well_name": inferred_name,
        "encoding": encoding,
    }
    return work_df, file_info


def summarize_density(series: pd.Series) -> dict[str, float | int | None]:
    numeric = safe_numeric(series).dropna()
    if numeric.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "sum": None,
        }
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": None if numeric.count() <= 1 else float(numeric.std(ddof=1)),
        "sum": float(numeric.sum()),
    }


def build_summary(
    *,
    config_path: Path,
    input_dir: Path,
    layer_dir: Path,
    output_csv: Path,
    summary_json: Path,
    min_thickness: float,
    selected_files: list[Path],
    all_rows_df: pd.DataFrame,
    well_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    used_well_names = sorted([
        well_name
        for well_name, payload in well_summaries.items()
        if int(payload.get("records_output", 0)) > 0
    ])
    zero_output_well_names = sorted([
        well_name
        for well_name, payload in well_summaries.items()
        if int(payload.get("records_output", 0)) == 0
    ])
    layer_counts = {
        str(key): int(value)
        for key, value in all_rows_df["LayerGroup"].value_counts(dropna=False).sort_index().items()
    }
    selected_surface_files = {}
    for choice in choose_surface_files(layer_dir):
        selected_surface_files[choice.surface_code] = {
            "surface_name": choice.surface_name,
            "surface_file": str(choice.surface_path),
            "selection_reason": choice.selection_reason,
        }

    density_stats = summarize_density(all_rows_df["Density"])
    manhattan_stats = {}
    for code in ("T4", "T5", "T6", "T7"):
        manhattan_stats[code] = summarize_density(all_rows_df[f"{code}_MANHATTAN_DISTANCE"])

    return {
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "layer_dir": str(layer_dir),
        "output_csv": str(output_csv),
        "summary_json": str(summary_json),
        "scanned_file_count": int(len(selected_files)),
        "used_well_count": int(len(used_well_names)),
        "used_wells": used_well_names,
        "zero_output_well_count": int(len(zero_output_well_names)),
        "zero_output_wells": zero_output_well_names,
        "record_count": int(len(all_rows_df)),
        "min_thickness": float(min_thickness),
        "density_stats": density_stats,
        "layer_group_distribution": layer_counts,
        "records_per_well": {
            well_name: int(well_summaries[well_name]["records_output"])
            for well_name in used_well_names
        },
        "per_well_summary": well_summaries,
        "surface_match_manhattan_distance_stats": manhattan_stats,
        "selected_surface_files": selected_surface_files,
    }


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [column for column in CANONICAL_ORDER if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining]


def run_pipeline(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    input_dir = Path(config["input_dir"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    output_csv = Path(config["output_csv"]).resolve()
    summary_json = Path(config.get("summary_json", output_csv.with_name(f"{output_csv.stem}_summary.json"))).resolve()
    file_glob = str(config.get("file_glob", "*.csv"))
    min_thickness = float(config.get("min_thickness", DEFAULT_MIN_THICKNESS))

    input_files = scan_input_files(input_dir, file_glob=file_glob)
    surfaces = load_surface_tables(layer_dir)

    input_frames: list[pd.DataFrame] = []
    file_infos: list[dict[str, Any]] = []
    for file_path in input_files:
        input_df, file_info = prepare_input_frame(file_path)
        input_frames.append(input_df)
        file_infos.append(file_info)

    all_input_df = pd.concat(input_frames, ignore_index=True)
    all_work_df = enrich_with_surfaces(all_input_df, surfaces=surfaces, min_thickness=min_thickness)
    missing_positive_density = all_work_df["HasFracture"].fillna(0).eq(1) & all_work_df["Density"].isna()
    result_df = all_work_df.loc[~missing_positive_density & all_work_df["TargetRowValid"]].copy()
    if result_df.empty:
        raise RuntimeError("no target T4-T7 rows were produced from the scanned files")

    well_summaries: dict[str, dict[str, Any]] = {}
    for file_info in file_infos:
        source_file = str(file_info["source_file"])
        source_mask = all_work_df["SourceFile"].eq(source_file)
        well_all_df = all_work_df.loc[source_mask].copy()
        well_out_df = result_df.loc[result_df["SourceFile"].eq(source_file)].copy()
        if well_all_df.empty:
            continue
        well_name = str(well_all_df["WellName"].iloc[0]).strip()
        well_missing_positive_density = missing_positive_density.loc[source_mask]
        well_summaries[well_name] = {
            "source_file": source_file,
            "records_total": int(len(well_all_df)),
            "records_in_target": int(well_all_df["InTargetT4T7"].sum()),
            "records_output": int(len(well_out_df)),
            "density_non_null_before_filter": int(well_all_df["Density"].notna().sum()),
            "density_imputed_zero_count": int(well_all_df["DensityImputedFromHasFracture"].sum()),
            "has_fracture_positive_count": int(well_all_df["HasFracture"].fillna(0).eq(1).sum()),
            "dropped_missing_density_positive_count": int(well_missing_positive_density.sum()),
            "layer_group_distribution": {
                str(key): int(value)
                for key, value in well_out_df["LayerGroup"].value_counts(dropna=False).sort_index().items()
            },
            "encoding": str(file_info["encoding"]),
        }

    result_df = order_columns(result_df)

    ensure_parent(output_csv)
    ensure_parent(summary_json)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = build_summary(
        config_path=config_path,
        input_dir=input_dir,
        layer_dir=layer_dir,
        output_csv=output_csv,
        summary_json=summary_json,
        min_thickness=min_thickness,
        selected_files=input_files,
        all_rows_df=result_df,
        well_summaries=well_summaries,
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    summary = run_pipeline(config_path)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
