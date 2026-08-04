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


REQUIRED_INPUT_COLUMNS = ["X", "Y", "TIME"]
TIME_OUTPUT_MAP = {
    "T4_TIME": "T4Time",
    "T5_TIME": "T5Time",
    "T6_TIME": "T6Time",
    "T7_TIME": "T7Time",
}
CLASS_TO_LAYER_GROUP = {
    "sha3_t4_t6": "沙三段",
    "sha4_t6_t7": "沙四段",
    "out_of_target": "OUT_OF_TARGET",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach T4-T7 surface times and LayerGroup to an existing point-level sample CSV."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def enrich_samples(input_df: pd.DataFrame, layer_dir: Path, min_thickness: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    work_df = input_df.copy()
    require_columns(work_df, REQUIRED_INPUT_COLUMNS, "input_csv")
    for column in REQUIRED_INPUT_COLUMNS:
        work_df[column] = safe_numeric(work_df[column])

    surfaces = load_surface_tables(layer_dir)
    work_df = assign_surface_times(work_df, surfaces)
    work_df = classify_time_domain(work_df)
    work_df = validate_surface_order(work_df, min_thickness=min_thickness)

    work_df["LayerGroup"] = work_df["TimeDomainClass"].map(CLASS_TO_LAYER_GROUP).fillna("OUT_OF_TARGET")
    work_df["InTargetT4T7"] = work_df["TimeDomainClass"].isin(["sha3_t4_t6", "sha4_t6_t7"]) & work_df["Check_All"].fillna(False)
    work_df["SurfaceOrderValid"] = work_df["Check_Order_T4_T5_T6_T7"].fillna(False)
    work_df["InputRowValid"] = ~work_df[REQUIRED_INPUT_COLUMNS].isna().any(axis=1)

    for source_col, target_col in TIME_OUTPUT_MAP.items():
        work_df[target_col] = safe_numeric(work_df[source_col])

    selected_surface_files = {}
    for choice in choose_surface_files(layer_dir):
        selected_surface_files[choice.surface_code] = {
            "surface_name": choice.surface_name,
            "surface_file": str(choice.surface_path),
            "selection_reason": choice.selection_reason,
        }

    summary = {
        "layer_dir": str(layer_dir),
        "selected_surface_files": selected_surface_files,
        "row_count": int(len(work_df)),
        "input_required_column_valid_rows": int(work_df["InputRowValid"].sum()),
        "input_required_column_invalid_rows": int((~work_df["InputRowValid"]).sum()),
        "surface_order_valid_rows": int(work_df["SurfaceOrderValid"].sum()),
        "in_target_t4_t7_rows": int(work_df["InTargetT4T7"].sum()),
        "time_domain_counts": {
            str(key): int(value)
            for key, value in work_df["TimeDomainClass"].value_counts(dropna=False).sort_index().items()
        },
        "layer_group_distribution": {
            str(key): int(value)
            for key, value in work_df["LayerGroup"].value_counts(dropna=False).sort_index().items()
        },
    }
    return work_df, summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)

    input_path = Path(config["input_csv"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    output_path = Path(config["output_csv"]).resolve()
    summary_path = Path(config.get("summary_json", output_path.with_name(f"{output_path.stem}_summary.json"))).resolve()
    min_thickness = float(config.get("min_thickness", DEFAULT_MIN_THICKNESS))

    input_df = read_csv(input_path)
    result_df, summary = enrich_samples(input_df=input_df, layer_dir=layer_dir, min_thickness=min_thickness)

    preferred_columns = REQUIRED_INPUT_COLUMNS + [
        "T4Time",
        "T5Time",
        "T6Time",
        "T7Time",
        "LayerGroup",
        "InTargetT4T7",
        "SurfaceOrderValid",
        "InputRowValid",
        "TimeDomainClass",
        "T4_MANHATTAN_DISTANCE",
        "T5_MANHATTAN_DISTANCE",
        "T6_MANHATTAN_DISTANCE",
        "T7_MANHATTAN_DISTANCE",
    ]
    ordered_columns = [column for column in preferred_columns if column in result_df.columns]
    remaining_columns = [column for column in result_df.columns if column not in ordered_columns]
    result_df = result_df[ordered_columns + remaining_columns]

    ensure_parent(output_path)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary.update(
        {
            "input_csv": str(input_path),
            "output_csv": str(output_path),
            "summary_json": str(summary_path),
            "min_thickness": min_thickness,
            "surface_time_map": {
                surface: (
                    None
                    if result_df[target_col].isna().all()
                    else float(result_df[target_col].dropna().iloc[0])
                )
                for surface, target_col in [("T4", "T4Time"), ("T5", "T5Time"), ("T6", "T6Time"), ("T7", "T7Time")]
            },
        }
    )

    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Attached horizon fields: {output_path}")
    print(f"Rows={len(result_df)} Cols={len(result_df.columns)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
