from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_TASK_DIR = CURRENT_DIR.parent
SAMPLE_ATTACH_DIR = PROJECT_TASK_DIR / "sample_horizon_attach"
if str(SAMPLE_ATTACH_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_ATTACH_DIR))

from attach_sample_horizons import enrich_samples  # noqa: E402


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
DEFAULT_LABEL_COLUMN = "DensityLabel"
DEFAULT_RAW_LABEL_COLUMN = "DensityRaw"
DEFAULT_POSITIVE_FLAG_COLUMN = "IsPositiveDensity"
INVALID_VALUE_THRESHOLD = -1.0e4
REAL_COLUMN_RENAMES = {
    "车西_相干体T4_T7_CENTER": "COHERENCE",
    "车西_蚂蚁体T4_T7_CENTER": "ANT_TRACK",
    "车西_最大曲率T4_T7_CENTER": "CURVATURE_MAX",
    "车西_最大正曲率T4_T7_CENTER": "CURVATURE_POS",
}
FEATURE_COLUMNS_TO_CLEAN = [
    "SEIS_TRUE",
    "COHERENCE",
    "ANT_TRACK",
    "CURVATURE_MAX",
    "CURVATURE_POS",
    "FRACTURE_INV",
    "COHERENCE_WIN_MEAN",
    "COHERENCE_WIN_STD",
    "COHERENCE_WIN_MIN",
    "COHERENCE_WIN_MAX",
    "COHERENCE_WIN_VALID_COUNT",
    "ANT_TRACK_WIN_MEAN",
    "ANT_TRACK_WIN_STD",
    "ANT_TRACK_WIN_MIN",
    "ANT_TRACK_WIN_MAX",
    "ANT_TRACK_WIN_VALID_COUNT",
    "CURVATURE_MAX_WIN_MEAN",
    "CURVATURE_MAX_WIN_STD",
    "CURVATURE_MAX_WIN_MIN",
    "CURVATURE_MAX_WIN_MAX",
    "CURVATURE_MAX_WIN_VALID_COUNT",
    "CURVATURE_POS_WIN_MEAN",
    "CURVATURE_POS_WIN_STD",
    "CURVATURE_POS_WIN_MIN",
    "CURVATURE_POS_WIN_MAX",
    "CURVATURE_POS_WIN_VALID_COUNT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build formal multi-well T4-T7 density training samples from official stage-2 well files."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    cleaned_counts: dict[str, int] = {}
    for column in FEATURE_COLUMNS_TO_CLEAN:
        if column not in out.columns:
            continue
        numeric = safe_numeric(out[column])
        invalid_mask = numeric <= float(INVALID_VALUE_THRESHOLD)
        cleaned_counts[column] = int(invalid_mask.sum())
        out[column] = numeric.mask(invalid_mask, np.nan)
    return out, cleaned_counts


def well_name_from_file(path: Path) -> str:
    return path.name.split("_")[0].strip()


def normalize_selected_wells(config: dict[str, Any]) -> list[str]:
    selected = config.get("selected_wells") or []
    return [str(item).strip() for item in selected if str(item).strip()]


def block_contains(x: float, y: float, block: dict[str, float]) -> bool:
    return (
        math.isfinite(x)
        and math.isfinite(y)
        and float(block["x_min"]) <= float(x) <= float(block["x_max"])
        and float(block["y_min"]) <= float(y) <= float(block["y_max"])
    )


def resolve_input_files(input_dir: Path, selected_wells: list[str]) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))
    if not selected_wells:
        return files
    selected_set = set(selected_wells)
    return [path for path in files if well_name_from_file(path) in selected_set]


def normalize_real_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {source: target for source, target in REAL_COLUMN_RENAMES.items() if source in df.columns}
    out = df.rename(columns=rename_map).copy()
    out["SourceWellName"] = out["WellName"].astype(str)
    out["TrackWellName"] = out["WellName"].astype(str)
    out["SourceKind"] = "real_well"
    return out


def normalize_virtual_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "VirtualWellName" not in out.columns:
        out["VirtualWellName"] = out["SourceWellName"].astype(str) if "SourceWellName" in out.columns else ""
    out["WellName"] = out["SourceWellName"].astype(str)
    out["TrackWellName"] = out["VirtualWellName"].astype(str)
    out["SourceKind"] = "virtual_well"
    return out


def apply_row_block_filter(df: pd.DataFrame, block: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["X"] = safe_numeric(out["X"])
    out["Y"] = safe_numeric(out["Y"])
    mask = (
        out["X"].ge(float(block["x_min"]))
        & out["X"].le(float(block["x_max"]))
        & out["Y"].ge(float(block["y_min"]))
        & out["Y"].le(float(block["y_max"]))
    )
    return out[mask].copy()


def build_density_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_density = safe_numeric(out.get("Density", pd.Series(np.nan, index=out.index)))
    has_fracture = safe_numeric(out.get("HasFracture", pd.Series(np.nan, index=out.index))).fillna(0.0)
    out[DEFAULT_RAW_LABEL_COLUMN] = raw_density
    out[DEFAULT_LABEL_COLUMN] = raw_density.fillna(0.0)
    out.loc[(has_fracture <= 0) & raw_density.isna(), DEFAULT_LABEL_COLUMN] = 0.0
    out[DEFAULT_POSITIVE_FLAG_COLUMN] = (out[DEFAULT_LABEL_COLUMN] > 0).astype(int)
    return out


def summarize_density(series: pd.Series) -> dict[str, float | int | None]:
    numeric = safe_numeric(series).dropna()
    if numeric.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p90": float(numeric.quantile(0.90)),
        "p95": float(numeric.quantile(0.95)),
    }


def ordered_columns(df: pd.DataFrame, model_input_fields: list[str], auxiliary_fields: list[str]) -> list[str]:
    preferred = [
        "SourceKind",
        "SourceWellName",
        "TrackWellName",
        "WellName",
        "SourceFile",
        "X",
        "Y",
        "TIME",
        "TVD",
        "DEPT",
        "LayerGroup",
        "TimeDomainClass",
        "T4Time",
        "T5Time",
        "T6Time",
        "T7Time",
        DEFAULT_RAW_LABEL_COLUMN,
        DEFAULT_LABEL_COLUMN,
        DEFAULT_POSITIVE_FLAG_COLUMN,
        "HasFracture",
    ]
    preferred.extend(model_input_fields)
    preferred.extend(auxiliary_fields)
    unique = []
    seen: set[str] = set()
    for column in preferred:
        if column in df.columns and column not in seen:
            unique.append(column)
            seen.add(column)
    for column in df.columns:
        if column not in seen:
            unique.append(column)
            seen.add(column)
    return unique


def build_summary(
    sample_df: pd.DataFrame,
    selected_files: list[Path],
    selected_wells: list[str],
    block: dict[str, Any],
    model_input_fields: list[str],
    auxiliary_fields: list[str],
    cleaning_stats: dict[str, int],
    output_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    per_well_counts = (
        sample_df.groupby(["SourceKind", "SourceWellName"], as_index=False)
        .agg(
            SampleRows=("SourceWellName", "size"),
            PositiveRows=(DEFAULT_POSITIVE_FLAG_COLUMN, "sum"),
            DensityMean=(DEFAULT_LABEL_COLUMN, "mean"),
            X=("X", "median"),
            Y=("Y", "median"),
        )
        .sort_values(["SourceKind", "SourceWellName"])
    )
    layer_counts = (
        sample_df.groupby(["SourceKind", "SourceWellName", "LayerGroup"], dropna=False)
        .size()
        .reset_index(name="Rows")
        .sort_values(["SourceKind", "SourceWellName", "LayerGroup"])
    )
    overall_density = summarize_density(sample_df[DEFAULT_LABEL_COLUMN])
    positive_density = summarize_density(sample_df.loc[sample_df[DEFAULT_LABEL_COLUMN] > 0, DEFAULT_LABEL_COLUMN])
    per_source_kind = {}
    for source_kind, group in sample_df.groupby("SourceKind"):
        per_source_kind[str(source_kind)] = {
            "row_count": int(len(group)),
            "positive_density_row_count": int((group[DEFAULT_LABEL_COLUMN] > 0).sum()),
            "density_label_stats": summarize_density(group[DEFAULT_LABEL_COLUMN]),
            "used_source_well_count": int(group["SourceWellName"].nunique()),
            "used_track_well_count": int(group["TrackWellName"].nunique()),
        }

    return {
        "input_file_count": int(len(selected_files)),
        "used_wells": selected_wells,
        "target_block": block,
        "row_count": int(len(sample_df)),
        "positive_density_row_count": int((sample_df[DEFAULT_LABEL_COLUMN] > 0).sum()),
        "zero_density_row_count": int((sample_df[DEFAULT_LABEL_COLUMN] == 0).sum()),
        "source_kind_distribution": {
            str(key): int(value)
            for key, value in sample_df["SourceKind"].value_counts(dropna=False).sort_index().items()
        },
        "layer_group_distribution": {
            str(key): int(value)
            for key, value in sample_df["LayerGroup"].value_counts(dropna=False).sort_index().items()
        },
        "used_source_well_count": int(sample_df["SourceWellName"].nunique()),
        "used_track_well_count": int(sample_df["TrackWellName"].nunique()),
        "density_label_stats": {
            "overall": overall_density,
            "positive_only": positive_density,
            "by_source_kind": per_source_kind,
        },
        "model_input_fields": model_input_fields,
        "auxiliary_fields": auxiliary_fields,
        "cleaning_stats": cleaning_stats,
        "label_field": DEFAULT_LABEL_COLUMN,
        "label_raw_field": DEFAULT_RAW_LABEL_COLUMN,
        "well_row_summary": per_well_counts.to_dict(orient="records"),
        "well_layer_summary": layer_counts.to_dict(orient="records"),
        "output_csv": str(output_csv),
        "summary_json": str(summary_json),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)

    input_dir = Path(config["input_dir"]).resolve()
    virtual_input_dir = Path(config["virtual_input_dir"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    output_csv = Path(config["output_csv"]).resolve()
    summary_json = Path(config["summary_json"]).resolve()
    min_thickness = float(config.get("min_thickness", 1.0))
    block = dict(config["target_block"])
    selected_wells = normalize_selected_wells(config)
    model_input_fields = list(config.get("model_input_fields", []))
    auxiliary_fields = list(config.get("auxiliary_fields", []))

    real_input_files = resolve_input_files(input_dir=input_dir, selected_wells=selected_wells)
    virtual_input_files = resolve_input_files(input_dir=virtual_input_dir, selected_wells=selected_wells)
    input_files = real_input_files + virtual_input_files
    if not input_files:
        raise FileNotFoundError(f"no input well files resolved under {input_dir} and {virtual_input_dir}")

    all_frames: list[pd.DataFrame] = []
    used_wells: list[str] = []
    cleaning_counter: dict[str, int] = {}
    for path in real_input_files:
        base_df = read_csv_flexible(path)
        if base_df.empty:
            continue
        well_name = (
            str(base_df["WellName"].dropna().astype(str).iloc[0]).strip()
            if "WellName" in base_df.columns and base_df["WellName"].dropna().shape[0] > 0
            else well_name_from_file(path)
        )
        base_df = normalize_real_columns(base_df)
        for column in ("X", "Y", "TIME", "TVD", "DEPT", "Density", "HasFracture"):
            if column in base_df.columns:
                base_df[column] = safe_numeric(base_df[column])
        rep_x = float(base_df["X"].median()) if "X" in base_df.columns else math.nan
        rep_y = float(base_df["Y"].median()) if "Y" in base_df.columns else math.nan
        if not block_contains(rep_x, rep_y, block):
            continue
        base_df = apply_row_block_filter(base_df, block)
        if base_df.empty:
            continue
        enriched_df, _ = enrich_samples(input_df=base_df, layer_dir=layer_dir, min_thickness=min_thickness)
        enriched_df, cleaned_counts = clean_feature_columns(enriched_df)
        for key, value in cleaned_counts.items():
            cleaning_counter[key] = cleaning_counter.get(key, 0) + int(value)
        enriched_df = build_density_label(enriched_df)
        enriched_df["SourceFile"] = path.name
        enriched_df = enriched_df[
            enriched_df["InputRowValid"].fillna(False)
            & enriched_df["SurfaceOrderValid"].fillna(False)
            & enriched_df["InTargetT4T7"].fillna(False)
        ].copy()
        if enriched_df.empty:
            continue
        all_frames.append(enriched_df)
        used_wells.append(well_name)

    for path in virtual_input_files:
        base_df = read_csv_flexible(path)
        if base_df.empty:
            continue
        well_name = (
            str(base_df["SourceWellName"].dropna().astype(str).iloc[0]).strip()
            if "SourceWellName" in base_df.columns and base_df["SourceWellName"].dropna().shape[0] > 0
            else well_name_from_file(path)
        )
        if selected_wells and well_name not in selected_wells:
            continue
        base_df = normalize_virtual_columns(base_df)
        for column in ("X", "Y", "TIME", "TVD", "DEPT", "Density", "HasFracture"):
            if column in base_df.columns:
                base_df[column] = safe_numeric(base_df[column])
        rep_x = float(base_df["X"].median()) if "X" in base_df.columns else math.nan
        rep_y = float(base_df["Y"].median()) if "Y" in base_df.columns else math.nan
        if not block_contains(rep_x, rep_y, block):
            # 仍允许有部分虚拟井轨迹切入区块，所以只在整文件完全偏离时再依赖行过滤判定
            pass
        base_df = apply_row_block_filter(base_df, block)
        if base_df.empty:
            continue
        enriched_df, _ = enrich_samples(input_df=base_df, layer_dir=layer_dir, min_thickness=min_thickness)
        enriched_df, cleaned_counts = clean_feature_columns(enriched_df)
        for key, value in cleaned_counts.items():
            cleaning_counter[key] = cleaning_counter.get(key, 0) + int(value)
        enriched_df = build_density_label(enriched_df)
        enriched_df["SourceFile"] = path.name
        enriched_df = enriched_df[
            enriched_df["InputRowValid"].fillna(False)
            & enriched_df["SurfaceOrderValid"].fillna(False)
            & enriched_df["InTargetT4T7"].fillna(False)
        ].copy()
        if enriched_df.empty:
            continue
        all_frames.append(enriched_df)
        used_wells.append(well_name)

    if not all_frames:
        raise RuntimeError("no rows retained after target block and T4-T7 filtering")

    sample_df = pd.concat(all_frames, ignore_index=True)
    sample_df = sample_df.sort_values(["WellName", "TIME", "TVD"], kind="stable").reset_index(drop=True)
    sample_df = sample_df[ordered_columns(sample_df, model_input_fields, auxiliary_fields)]

    ensure_parent(output_csv)
    sample_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    used_wells_sorted = sorted(dict.fromkeys(used_wells))
    summary = build_summary(
        sample_df=sample_df,
        selected_files=input_files,
        selected_wells=used_wells_sorted,
        block=block,
        model_input_fields=model_input_fields,
        auxiliary_fields=auxiliary_fields,
        cleaning_stats=cleaning_counter,
        output_csv=output_csv,
        summary_json=summary_json,
    )
    ensure_parent(summary_json)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Output CSV: {output_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Wells: {len(used_wells_sorted)}")
    print(f"Rows: {len(sample_df)}")


if __name__ == "__main__":
    main()
