# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import defaultdict
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layer_model_registry import (
    LAYER_SURFACE_PAIR_KEY_COL,
    UNIT_LAYER_SEGMENT_KEY_COL,
    ensure_layer_surface_pair_key_column,
    ensure_unit_layer_segment_key_column,
    resolve_layer_density_calibration_config,
    resolve_layer_density_prior,
)

THIS_DIR = Path(__file__).resolve().parent
OPT_STAGE_DIR = THIS_DIR.parent
INSTANCE_DIR = OPT_STAGE_DIR / "DFN实例表达互转实验"
if str(INSTANCE_DIR) not in sys.path:
    sys.path.append(str(INSTANCE_DIR))

from instance_roundtrip_common import (  # type: ignore
    DEFAULT_DOCX_PATH,
    DEFAULT_UNIT_DFN_ROOT,
    GridSpec,
    VtkPatchExportConfig,
    build_grid_spec,
    build_patch_match_metrics,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    decode_instance_label_to_patches,
    export_patch_comparison_vtk,
    export_patch_vtk_files,
    load_layer_table,
    load_patch_table,
    load_unit_summary,
    open_or_create_doc,
    read_csv_utf8,
    write_csv_utf8,
    write_json,
)


DEFAULT_DATASET_RUN_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/GAN训练准备/训练样本打包/phase1_t128_sparse_v1_full_fix1_20260330"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线"
)
DEFAULT_SLOTS_PER_VOXEL = 16
DEFAULT_XY_RESOLUTION = 24
DEFAULT_Z_STEP_MS = 0.2
DEFAULT_WINDOW_SIZE = 128
DEFAULT_INPUT_CHANNELS = [
    "seismic_amp",
    "grad_x",
    "grad_y",
    "grad_z",
    "rel_depth_in_interval",
]
GEOM_CHANNELS = [
    "offset_x",
    "offset_y",
    "offset_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "u_dir_x",
    "u_dir_y",
    "u_dir_z",
    "length",
    "height",
    "confidence",
]
DEFAULT_LAYER_DENSITY_SOURCE = "robust"
LAYER_DENSITY_SOURCE_CHOICES = ("aggregate", "unit_mean", "unit_median", "unit_p75", "robust")


def normalize_layer_density_source(value: Any) -> str:
    text = str(value or DEFAULT_LAYER_DENSITY_SOURCE).strip().lower()
    if text == "mean":
        text = "unit_mean"
    if text == "median":
        text = "unit_median"
    if text == "p75":
        text = "unit_p75"
    if text in {"robust_density", "robust-median", "robust_median"}:
        text = "robust"
    if text not in LAYER_DENSITY_SOURCE_CHOICES:
        return DEFAULT_LAYER_DENSITY_SOURCE
    return text


def scalar_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    value_float = float(numeric)
    if not np.isfinite(value_float):
        return float(default)
    return value_float


def scalar_int(value: Any, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return int(default)
    value_float = float(numeric)
    if not np.isfinite(value_float):
        return int(default)
    return int(round(value_float))


def build_unit_layer_segment_table(layers_df: pd.DataFrame) -> pd.DataFrame:
    work = ensure_unit_layer_segment_key_column(
        ensure_layer_surface_pair_key_column(layers_df.copy(), key_col=LAYER_SURFACE_PAIR_KEY_COL),
        key_col=UNIT_LAYER_SEGMENT_KEY_COL,
        pair_key_col=LAYER_SURFACE_PAIR_KEY_COL,
    )
    if work.empty:
        return pd.DataFrame(
            columns=[
                "UnitID",
                "GeoIntervalKey",
                LAYER_SURFACE_PAIR_KEY_COL,
                UNIT_LAYER_SEGMENT_KEY_COL,
                "TopTime",
                "BaseTime",
                "LayerThicknessMs",
            ]
        )
    work["TopTime"] = pd.to_numeric(work.get("TopTime"), errors="coerce")
    work["BaseTime"] = pd.to_numeric(work.get("BaseTime"), errors="coerce")
    work["LayerThicknessMs"] = (work["BaseTime"] - work["TopTime"]).abs()
    work = work.sort_values(["TopTime", "BaseTime", "GeoIntervalKey"], na_position="last").reset_index(drop=True)
    return work


def build_unit_layer_density_rows(
    unit_dir: Path,
    patch_filename: str = "unit_dfn_patches.csv",
) -> pd.DataFrame:
    unit_dir = Path(unit_dir)
    layers_path = unit_dir / "unit_layers_input.csv"
    if not layers_path.exists():
        raise FileNotFoundError(f"unit layer table not found: {layers_path}")
    raw_layers_df = read_csv_utf8(layers_path)
    if "UnitID" not in raw_layers_df.columns:
        raw_layers_df["UnitID"] = str(unit_dir.name)
    else:
        unit_id_series = raw_layers_df["UnitID"].fillna("").astype(str).str.strip()
        if unit_id_series.eq("").all():
            raw_layers_df["UnitID"] = str(unit_dir.name)
    layers_df = build_unit_layer_segment_table(raw_layers_df)
    patch_path = unit_dir / patch_filename
    if patch_path.exists():
        patch_df = ensure_unit_layer_segment_key_column(
            ensure_layer_surface_pair_key_column(read_csv_utf8(patch_path), key_col=LAYER_SURFACE_PAIR_KEY_COL),
            key_col=UNIT_LAYER_SEGMENT_KEY_COL,
            pair_key_col=LAYER_SURFACE_PAIR_KEY_COL,
        )
    else:
        patch_df = pd.DataFrame()

    if not patch_df.empty:
        patch_df["GeoIntervalKey"] = patch_df.get("GeoIntervalKey", pd.Series(dtype="object")).fillna("").astype(str)
        patch_df[LAYER_SURFACE_PAIR_KEY_COL] = patch_df.get(LAYER_SURFACE_PAIR_KEY_COL, pd.Series(dtype="object")).fillna("").astype(str)
        patch_df[UNIT_LAYER_SEGMENT_KEY_COL] = patch_df.get(UNIT_LAYER_SEGMENT_KEY_COL, pd.Series(dtype="object")).fillna("").astype(str)

    rows: list[dict[str, Any]] = []
    for _, layer_row in layers_df.iterrows():
        unit_id = str(layer_row.get("UnitID", "")).strip() or str(unit_dir.name)
        geo_interval_key = str(layer_row.get("GeoIntervalKey", "")).strip()
        layer_surface_pair_key = str(layer_row.get(LAYER_SURFACE_PAIR_KEY_COL, "")).strip()
        unit_layer_segment_key = str(layer_row.get(UNIT_LAYER_SEGMENT_KEY_COL, "")).strip()
        thickness_ms = scalar_float(layer_row.get("LayerThicknessMs"), default=0.0)
        if patch_df.empty:
            patch_count = 0
        elif geo_interval_key and "GeoIntervalKey" in patch_df.columns:
            patch_count = int((patch_df["GeoIntervalKey"] == geo_interval_key).sum())
        elif unit_layer_segment_key and UNIT_LAYER_SEGMENT_KEY_COL in patch_df.columns:
            patch_count = int((patch_df[UNIT_LAYER_SEGMENT_KEY_COL] == unit_layer_segment_key).sum())
        elif layer_surface_pair_key and LAYER_SURFACE_PAIR_KEY_COL in patch_df.columns:
            patch_count = int((patch_df[LAYER_SURFACE_PAIR_KEY_COL] == layer_surface_pair_key).sum())
        else:
            patch_count = 0
        rows.append(
            {
                "UnitID": unit_id,
                "GeoIntervalKey": geo_interval_key,
                LAYER_SURFACE_PAIR_KEY_COL: layer_surface_pair_key,
                UNIT_LAYER_SEGMENT_KEY_COL: unit_layer_segment_key,
                "TopTime": scalar_float(layer_row.get("TopTime"), default=0.0),
                "BaseTime": scalar_float(layer_row.get("BaseTime"), default=0.0),
                "LayerThicknessMs": thickness_ms,
                "PatchCount": int(patch_count),
                "PatchDensity": float(patch_count / thickness_ms) if thickness_ms > 0.0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_layer_density_priors(
    unit_dfn_root: Path,
    unit_ids: list[str],
    patch_filename: str = "unit_dfn_patches.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_dfn_root = Path(unit_dfn_root)
    density_rows: list[pd.DataFrame] = []
    missing_units: list[str] = []
    for unit_id in [str(value).strip() for value in unit_ids if str(value).strip()]:
        unit_dir = unit_dfn_root / unit_id
        if not unit_dir.exists():
            missing_units.append(unit_id)
            continue
        density_rows.append(build_unit_layer_density_rows(unit_dir=unit_dir, patch_filename=patch_filename))
    if missing_units:
        preview = ", ".join(missing_units[:12])
        raise FileNotFoundError(f"missing unit dirs for density priors ({len(missing_units)}): {preview}")
    detail_df = pd.concat(density_rows, ignore_index=True, sort=False) if density_rows else pd.DataFrame()
    if detail_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    LAYER_SURFACE_PAIR_KEY_COL,
                    "UnitCount",
                    "SegmentCount",
                    "TotalPatchCount",
                    "TotalThicknessMs",
                    "AggregateDensity",
                    "UnitMeanDensity",
                    "UnitMedianDensity",
                    "UnitP25Density",
                    "UnitP10Density",
                    "UnitP75Density",
                    "UnitMaxDensity",
                    "NonZeroSegmentCount",
                    "NonZeroSegmentFraction",
                    "RobustDensity",
                ]
            ),
            detail_df,
        )
    summary_df = (
        detail_df.groupby(LAYER_SURFACE_PAIR_KEY_COL, dropna=False)
        .agg(
            UnitCount=("UnitID", lambda values: int(pd.Series(values).astype(str).nunique())),
            SegmentCount=("UnitID", "count"),
            TotalPatchCount=("PatchCount", "sum"),
            TotalThicknessMs=("LayerThicknessMs", "sum"),
            AggregateDensity=("PatchDensity", lambda values: float("nan")),
            UnitMeanDensity=("PatchDensity", "mean"),
            UnitMedianDensity=("PatchDensity", "median"),
            UnitP25Density=("PatchDensity", lambda values: float(pd.Series(values).quantile(0.25))),
            UnitP10Density=("PatchDensity", lambda values: float(pd.Series(values).quantile(0.10))),
            UnitP75Density=("PatchDensity", lambda values: float(pd.Series(values).quantile(0.75))),
            UnitMaxDensity=("PatchDensity", "max"),
            NonZeroSegmentCount=("PatchCount", lambda values: int((pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0) > 0.0).sum())),
        )
        .reset_index()
    )
    total_patch_counts = detail_df.groupby(LAYER_SURFACE_PAIR_KEY_COL, dropna=False)["PatchCount"].sum()
    total_thickness = detail_df.groupby(LAYER_SURFACE_PAIR_KEY_COL, dropna=False)["LayerThicknessMs"].sum()
    summary_df["AggregateDensity"] = [
        float(total_patch_counts.get(key, 0) / total_thickness.get(key, 1.0))
        if float(total_thickness.get(key, 0.0)) > 0.0
        else 0.0
        for key in summary_df[LAYER_SURFACE_PAIR_KEY_COL].astype(str)
    ]
    summary_df["NonZeroSegmentFraction"] = [
        float(nonzero_count / segment_count) if int(segment_count) > 0 else 0.0
        for nonzero_count, segment_count in zip(
            summary_df["NonZeroSegmentCount"],
            summary_df["SegmentCount"],
        )
    ]
    robust_density_rows: list[float] = []
    for _, row in summary_df.iterrows():
        unit_median_density = scalar_float(row.get("UnitMedianDensity"), default=0.0)
        unit_mean_density = scalar_float(row.get("UnitMeanDensity"), default=0.0)
        aggregate_density = scalar_float(row.get("AggregateDensity"), default=0.0)
        unit_p10_density = scalar_float(row.get("UnitP10Density"), default=0.0)
        if unit_median_density > 0.0:
            robust_density = unit_median_density
        elif unit_mean_density > 0.0:
            robust_density = min(unit_mean_density, aggregate_density) if aggregate_density > 0.0 else unit_mean_density
        else:
            robust_density = max(unit_p10_density, aggregate_density)
        robust_density_rows.append(float(max(robust_density, 0.0)))
    summary_df["RobustDensity"] = robust_density_rows
    summary_df = summary_df.sort_values(LAYER_SURFACE_PAIR_KEY_COL).reset_index(drop=True)
    return summary_df, detail_df


def build_layer_density_prior_payload(summary_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if summary_df.empty:
        return {}
    payload: dict[str, dict[str, Any]] = {}
    for _, row in summary_df.iterrows():
        layer_key = str(row.get(LAYER_SURFACE_PAIR_KEY_COL, "")).strip()
        if not layer_key:
            continue
        payload[layer_key] = {
            "aggregate_density": scalar_float(row.get("AggregateDensity"), default=0.0),
            "unit_mean_density": scalar_float(row.get("UnitMeanDensity"), default=0.0),
            "unit_median_density": scalar_float(row.get("UnitMedianDensity"), default=0.0),
            "unit_p25_density": scalar_float(row.get("UnitP25Density"), default=0.0),
            "unit_p10_density": scalar_float(row.get("UnitP10Density"), default=0.0),
            "unit_p75_density": scalar_float(row.get("UnitP75Density"), default=0.0),
            "unit_max_density": scalar_float(row.get("UnitMaxDensity"), default=0.0),
            "robust_density": scalar_float(row.get("RobustDensity"), default=0.0),
            "unit_count": scalar_int(row.get("UnitCount"), default=0),
            "segment_count": scalar_int(row.get("SegmentCount"), default=0),
            "nonzero_segment_count": scalar_int(row.get("NonZeroSegmentCount"), default=0),
            "total_patch_count": scalar_int(row.get("TotalPatchCount"), default=0),
            "total_thickness_ms": scalar_float(row.get("TotalThicknessMs"), default=0.0),
            "nonzero_segment_fraction": scalar_float(row.get("NonZeroSegmentFraction"), default=0.0),
        }
    return payload


def resolve_target_density_from_prior(
    prior_payload: dict[str, Any] | None,
    density_source: str,
) -> float | None:
    if not prior_payload:
        return None
    normalized_source = normalize_layer_density_source(density_source)
    if normalized_source == "robust":
        direct_robust_density = pd.to_numeric(prior_payload.get("robust_density"), errors="coerce")
        if not pd.isna(direct_robust_density):
            value_float = float(direct_robust_density)
            if np.isfinite(value_float):
                return max(value_float, 0.0)
        unit_median_density = scalar_float(prior_payload.get("unit_median_density"), default=0.0)
        unit_mean_density = scalar_float(prior_payload.get("unit_mean_density"), default=0.0)
        aggregate_density = scalar_float(prior_payload.get("aggregate_density"), default=0.0)
        unit_p10_density = scalar_float(prior_payload.get("unit_p10_density"), default=0.0)
        if unit_median_density > 0.0:
            return unit_median_density
        if unit_mean_density > 0.0:
            return min(unit_mean_density, aggregate_density) if aggregate_density > 0.0 else unit_mean_density
        return max(unit_p10_density, aggregate_density, 0.0)
    field_name = {
        "aggregate": "aggregate_density",
        "unit_mean": "unit_mean_density",
        "unit_median": "unit_median_density",
        "unit_p75": "unit_p75_density",
    }[normalized_source]
    value = pd.to_numeric(prior_payload.get(field_name), errors="coerce")
    if pd.isna(value):
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return max(value_float, 0.0)


def resolve_target_nonzero_fraction_from_prior(
    prior_payload: dict[str, Any] | None,
) -> float | None:
    if not prior_payload:
        return None
    value = pd.to_numeric(prior_payload.get("nonzero_segment_fraction"), errors="coerce")
    if pd.isna(value):
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return float(np.clip(value_float, 0.0, 1.0))


def load_layer_density_calibration_payload(
    calibration_json: Path | None,
) -> dict[str, dict[str, Any]]:
    if calibration_json is None:
        return {}
    calibration_path = Path(calibration_json)
    if not calibration_path.exists():
        return {}
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    layers_payload = payload.get("layers", payload)
    if not isinstance(layers_payload, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, raw_value in layers_payload.items():
        layer_key = str(key).strip()
        if not layer_key or not isinstance(raw_value, dict):
            continue
        normalized[layer_key] = dict(raw_value)
    return normalized


def resolve_layer_density_calibration_entry(
    calibration_payload: dict[str, dict[str, Any]] | None,
    layer_surface_pair_key: str,
) -> dict[str, Any]:
    if not calibration_payload:
        return {}
    layer_key = str(layer_surface_pair_key).strip()
    raw_layer_payload = calibration_payload.get(layer_key, {})
    return dict(raw_layer_payload) if isinstance(raw_layer_payload, dict) else {}


def resolve_layer_density_calibration_bounds(
    registry_payload: dict[str, Any] | None,
    layer_surface_pair_key: str,
    default_min_scale: float = 0.25,
    default_max_scale: float = 4.0,
) -> tuple[float, float]:
    layer_config = resolve_layer_density_calibration_config(registry_payload, layer_surface_pair_key)
    min_scale = scalar_float(layer_config.get("min_scale"), default=default_min_scale)
    max_scale = scalar_float(layer_config.get("max_scale"), default=default_max_scale)
    min_scale = float(min(min_scale, max_scale))
    max_scale = float(max(min_scale, max_scale))
    return min_scale, max_scale


def resolve_layer_activation_suppression_config(
    registry_payload: dict[str, Any] | None,
    layer_surface_pair_key: str,
) -> dict[str, Any]:
    layer_config = resolve_layer_density_calibration_config(registry_payload, layer_surface_pair_key)
    mode_text = str(layer_config.get("activation_suppression_mode", "hard")).strip().lower()
    if mode_text not in {"hard", "soft"}:
        mode_text = "hard"
    soft_min_fraction = float(
        np.clip(
            scalar_float(layer_config.get("soft_activation_min_fraction"), default=0.0),
            0.0,
            1.0,
        )
    )
    soft_exponent = max(
        scalar_float(layer_config.get("soft_activation_exponent"), default=1.0),
        1e-6,
    )
    soft_min_keep_count = max(
        scalar_int(layer_config.get("soft_activation_min_keep_count"), default=1),
        0,
    )
    return {
        "mode": mode_text,
        "soft_min_fraction": soft_min_fraction,
        "soft_exponent": soft_exponent,
        "soft_min_keep_count": soft_min_keep_count,
    }


def resolve_layer_density_calibration_scale(
    calibration_payload: dict[str, dict[str, Any]] | None,
    layer_surface_pair_key: str,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> float:
    if not calibration_payload:
        return 1.0
    layer_key = str(layer_surface_pair_key).strip()
    raw_layer_payload = calibration_payload.get(layer_key, {})
    if not isinstance(raw_layer_payload, dict) or not raw_layer_payload:
        return 1.0
    raw_scale = raw_layer_payload.get("suggested_density_scale_clipped", raw_layer_payload.get("suggested_density_scale"))
    scale_value = pd.to_numeric(raw_scale, errors="coerce")
    if pd.isna(scale_value):
        return 1.0
    scale_float = float(scale_value)
    if not np.isfinite(scale_float):
        return 1.0
    min_scale = float(min(min_scale, max_scale))
    max_scale = float(max(min_scale, max_scale))
    return float(np.clip(scale_float, min_scale, max_scale))


def resolve_layer_activation_threshold(
    prior_payload: dict[str, Any] | None,
    calibration_payload: dict[str, dict[str, Any]] | None,
    layer_surface_pair_key: str,
    target_density: float | None,
) -> tuple[float | None, str, float | None]:
    calibration_entry = resolve_layer_density_calibration_entry(calibration_payload, layer_surface_pair_key)
    calibration_nonzero_fraction = pd.to_numeric(
        calibration_entry.get("ground_truth_nonzero_segment_fraction"),
        errors="coerce",
    ) if calibration_entry else np.nan
    target_nonzero_fraction = (
        float(np.clip(float(calibration_nonzero_fraction), 0.0, 1.0))
        if not pd.isna(calibration_nonzero_fraction) and np.isfinite(float(calibration_nonzero_fraction))
        else resolve_target_nonzero_fraction_from_prior(prior_payload)
    )

    if calibration_entry:
        raw_threshold = pd.to_numeric(
            calibration_entry.get("suggested_activation_score_threshold"),
            errors="coerce",
        )
        if not pd.isna(raw_threshold):
            threshold_float = float(raw_threshold)
            if np.isfinite(threshold_float):
                return max(threshold_float, 0.0), "calibration_activation_score", target_nonzero_fraction

    if target_density is None:
        return None, "activation_threshold_missing_target_density", target_nonzero_fraction
    target_density = max(float(target_density), 0.0)
    if target_density <= 0.0:
        return None, "activation_threshold_zero_target_density", target_nonzero_fraction
    if target_nonzero_fraction is None:
        return 0.0, "activation_threshold_prior_fraction_missing", None
    if target_nonzero_fraction <= 0.0:
        return None, "activation_threshold_zero_target_fraction", 0.0
    if target_nonzero_fraction >= 1.0:
        return 0.0, "activation_threshold_full_target_fraction", 1.0
    positive_density_proxy = max(float(target_density) / float(target_nonzero_fraction), 0.0)
    return positive_density_proxy, "activation_threshold_prior_positive_density_proxy", target_nonzero_fraction


def resolve_second_pass_decode_config(
    decode_runtime_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(decode_runtime_config, dict):
        return {}
    raw_second_pass = decode_runtime_config.get("second_pass")
    if not isinstance(raw_second_pass, dict):
        return {}
    config: dict[str, Any] = {}
    config["enabled"] = bool(raw_second_pass.get("enabled", False))
    config["center_threshold"] = scalar_float(
        raw_second_pass.get("center_threshold"),
        default=scalar_float(decode_runtime_config.get("center_threshold"), default=0.7),
    )
    config["count_activation_threshold"] = scalar_float(
        raw_second_pass.get("count_activation_threshold"),
        default=scalar_float(decode_runtime_config.get("count_activation_threshold"), default=0.5),
    )
    config["min_count_if_active"] = max(
        scalar_int(
            raw_second_pass.get("min_count_if_active"),
            default=scalar_int(decode_runtime_config.get("min_count_if_active"), default=1),
        ),
        1,
    )
    decode_mode = str(raw_second_pass.get("decode_mode", decode_runtime_config.get("decode_mode", "strict"))).strip().lower()
    config["decode_mode"] = decode_mode if decode_mode in {"strict", "relaxed"} else "strict"
    config["relaxed_min_count"] = max(
        scalar_int(
            raw_second_pass.get("relaxed_min_count"),
            default=scalar_int(decode_runtime_config.get("relaxed_min_count"), default=1),
        ),
        1,
    )
    max_total_patches = scalar_int(
        raw_second_pass.get("max_total_patches_per_window"),
        default=scalar_int(decode_runtime_config.get("max_total_patches_per_window"), default=0),
    )
    config["max_total_patches_per_window"] = int(max_total_patches) if int(max_total_patches) > 0 else 0
    config["primary_patch_count_threshold"] = max(
        scalar_int(raw_second_pass.get("primary_patch_count_threshold"), default=0),
        0,
    )
    config["min_calibration_scale"] = max(
        scalar_float(raw_second_pass.get("min_calibration_scale"), default=0.0),
        0.0,
    )
    config["trigger_without_calibration"] = bool(raw_second_pass.get("trigger_without_calibration", False))
    return config if config.get("enabled", False) else {}


def should_trigger_second_pass_decode(
    second_pass_config: dict[str, Any] | None,
    primary_patch_count: int,
    layer_surface_pair_key: str,
    calibration_payload: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, str, float | None]:
    config = dict(second_pass_config or {})
    if not config:
        return False, "second_pass_disabled", None
    primary_threshold = max(scalar_int(config.get("primary_patch_count_threshold"), default=0), 0)
    if int(primary_patch_count) > primary_threshold:
        return False, "primary_patch_count_above_threshold", None

    min_calibration_scale = max(scalar_float(config.get("min_calibration_scale"), default=0.0), 0.0)
    calibration_entry = resolve_layer_density_calibration_entry(calibration_payload, layer_surface_pair_key)
    if calibration_entry:
        ground_truth_density = scalar_float(calibration_entry.get("ground_truth_density"), default=0.0)
        predicted_density = scalar_float(calibration_entry.get("predicted_density"), default=0.0)
        if ground_truth_density > 0.0 and predicted_density <= 0.0:
            return True, "calibration_zero_density_recall", None
        calibration_scale = resolve_layer_density_calibration_scale(
            calibration_payload,
            layer_surface_pair_key,
            min_scale=0.0,
            max_scale=max(min_calibration_scale, 16.0),
        )
        if calibration_scale + 1e-8 < min_calibration_scale:
            return False, "calibration_scale_below_min", calibration_scale
        return True, "calibration_scale_trigger", calibration_scale

    if bool(config.get("trigger_without_calibration", False)):
        return True, "primary_patch_threshold_trigger", None
    return False, "calibration_missing", None


def apply_layer_density_budget(
    patch_df: pd.DataFrame,
    layers_df: pd.DataFrame,
    registry_payload: dict[str, Any] | None,
    density_source: str = DEFAULT_LAYER_DENSITY_SOURCE,
    density_scale: float = 1.0,
    calibration_payload: dict[str, dict[str, Any]] | None = None,
    calibration_min_scale: float = 0.25,
    calibration_max_scale: float = 4.0,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_layers_df = build_unit_layer_segment_table(layers_df)
    normalized_source = normalize_layer_density_source(density_source)
    patch_work = ensure_unit_layer_segment_key_column(
        ensure_layer_surface_pair_key_column(patch_df.copy(), key_col=LAYER_SURFACE_PAIR_KEY_COL),
        key_col=UNIT_LAYER_SEGMENT_KEY_COL,
        pair_key_col=LAYER_SURFACE_PAIR_KEY_COL,
    )

    if patch_work.empty:
        summary_rows = []
        for _, layer_row in normalized_layers_df.iterrows():
            layer_key = str(layer_row.get(LAYER_SURFACE_PAIR_KEY_COL, "")).strip()
            prior_payload = resolve_layer_density_prior(registry_payload, layer_key) if enabled else {}
            base_density = resolve_target_density_from_prior(prior_payload, normalized_source) if enabled else None
            base_nonzero_fraction = resolve_target_nonzero_fraction_from_prior(prior_payload) if enabled else None
            activation_suppression_config = resolve_layer_activation_suppression_config(
                registry_payload,
                layer_key,
            ) if enabled else {"mode": "hard", "soft_min_fraction": 0.0, "soft_exponent": 1.0, "soft_min_keep_count": 1}
            layer_min_scale, layer_max_scale = resolve_layer_density_calibration_bounds(
                registry_payload,
                layer_key,
                default_min_scale=calibration_min_scale,
                default_max_scale=calibration_max_scale,
            ) if enabled else (1.0, 1.0)
            calibration_scale = resolve_layer_density_calibration_scale(
                calibration_payload,
                layer_key,
                min_scale=layer_min_scale,
                max_scale=layer_max_scale,
            ) if enabled else 1.0
            thickness_ms = scalar_float(layer_row.get("LayerThicknessMs"), default=0.0)
            effective_density_scale = float(density_scale) * float(calibration_scale)
            target_density = (base_density * effective_density_scale) if base_density is not None else None
            target_patch_count = int(round(target_density * thickness_ms)) if target_density is not None else None
            activation_threshold, activation_threshold_source, target_nonzero_fraction = resolve_layer_activation_threshold(
                prior_payload=prior_payload,
                calibration_payload=calibration_payload if enabled else None,
                layer_surface_pair_key=layer_key,
                target_density=target_density,
            ) if enabled else (None, "activation_threshold_disabled", base_nonzero_fraction)
            summary_rows.append(
                {
                    "UnitID": str(layer_row.get("UnitID", "")),
                    "GeoIntervalKey": str(layer_row.get("GeoIntervalKey", "")),
                    LAYER_SURFACE_PAIR_KEY_COL: layer_key,
                    UNIT_LAYER_SEGMENT_KEY_COL: str(layer_row.get(UNIT_LAYER_SEGMENT_KEY_COL, "")),
                    "LayerThicknessMs": thickness_ms,
                    "BaseDensity": base_density,
                    "BaseNonZeroSegmentFraction": base_nonzero_fraction,
                    "CalibrationScale": float(calibration_scale),
                    "EffectiveDensityScale": float(effective_density_scale),
                    "TargetDensity": target_density,
                    "TargetNonZeroSegmentFraction": target_nonzero_fraction,
                    "TargetPatchCount": target_patch_count,
                    "PredictedPatchCountBefore": 0,
                    "PredictedPatchCountAfter": 0,
                    "PredictedDensityBefore": 0.0,
                    "PredictedDensityAfter": 0.0,
                    "ActivationScoreBeforeDensityControl": 0.0,
                    "ActivationScoreThreshold": activation_threshold,
                    "ActivationThresholdSource": activation_threshold_source,
                    "ActivationSuppressionModeConfigured": str(activation_suppression_config.get("mode", "hard")),
                    "ActivationSuppressionModeApplied": "none",
                    "ActivationSoftKeepFraction": None,
                    "ActivationTargetPatchCountAfterThreshold": target_patch_count,
                    "PredictedNonZeroBeforeDensityControl": False,
                    "PredictedNonZeroAfterDensityControl": False,
                    "SuppressedByActivationThreshold": False,
                    "DensityControlEnabled": bool(enabled),
                    "DensitySource": normalized_source,
                    "DensityScale": float(density_scale),
                }
            )
        return patch_work, pd.DataFrame(summary_rows)

    patch_work["ConfidenceSort"] = pd.to_numeric(patch_work.get("Confidence"), errors="coerce").fillna(0.0)
    patch_work["PatchLengthSort"] = pd.to_numeric(patch_work.get("PatchLength"), errors="coerce").fillna(0.0)
    patch_work["PatchHeightSort"] = pd.to_numeric(patch_work.get("PatchHeight"), errors="coerce").fillna(0.0)

    kept_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    handled_segment_keys: set[str] = set()

    for _, layer_row in normalized_layers_df.iterrows():
        segment_key = str(layer_row.get(UNIT_LAYER_SEGMENT_KEY_COL, "")).strip()
        layer_key = str(layer_row.get(LAYER_SURFACE_PAIR_KEY_COL, "")).strip()
        thickness_ms = scalar_float(layer_row.get("LayerThicknessMs"), default=0.0)
        segment_df = patch_work[patch_work[UNIT_LAYER_SEGMENT_KEY_COL].astype(str) == segment_key].copy()
        before_count = int(len(segment_df))
        prior_payload = resolve_layer_density_prior(registry_payload, layer_key) if enabled else {}
        base_density = resolve_target_density_from_prior(prior_payload, normalized_source) if enabled else None
        base_nonzero_fraction = resolve_target_nonzero_fraction_from_prior(prior_payload) if enabled else None
        activation_suppression_config = resolve_layer_activation_suppression_config(
            registry_payload,
            layer_key,
        ) if enabled else {"mode": "hard", "soft_min_fraction": 0.0, "soft_exponent": 1.0, "soft_min_keep_count": 1}
        layer_min_scale, layer_max_scale = resolve_layer_density_calibration_bounds(
            registry_payload,
            layer_key,
            default_min_scale=calibration_min_scale,
            default_max_scale=calibration_max_scale,
        ) if enabled else (1.0, 1.0)
        calibration_scale = resolve_layer_density_calibration_scale(
            calibration_payload,
            layer_key,
            min_scale=layer_min_scale,
            max_scale=layer_max_scale,
        ) if enabled else 1.0
        effective_density_scale = float(density_scale) * float(calibration_scale)
        target_density = (base_density * effective_density_scale) if base_density is not None else None
        target_patch_count = None
        if target_density is not None:
            target_patch_count = max(int(round(target_density * thickness_ms)), 0)
        activation_score_before = 0.0
        if before_count > 0:
            max_confidence_before = float(segment_df["ConfidenceSort"].max())
            mean_confidence_before = float(segment_df["ConfidenceSort"].mean())
            activation_score_before = float(
                (before_count / thickness_ms) if thickness_ms > 0.0 else before_count
            ) + (max_confidence_before * 1e-6) + (mean_confidence_before * 1e-9)
        else:
            max_confidence_before = 0.0
            mean_confidence_before = 0.0
        activation_threshold, activation_threshold_source, target_nonzero_fraction = resolve_layer_activation_threshold(
            prior_payload=prior_payload,
            calibration_payload=calibration_payload if enabled else None,
            layer_surface_pair_key=layer_key,
            target_density=target_density,
        ) if enabled else (None, "activation_threshold_disabled", base_nonzero_fraction)
        predicted_nonzero_before = before_count > 0
        suppressed_by_activation_threshold = False
        activation_suppression_mode_applied = "none"
        activation_soft_keep_fraction: float | None = None
        activation_target_patch_count_after_threshold = target_patch_count
        if (
            target_nonzero_fraction is not None
            and enabled
            and float(target_nonzero_fraction) <= 0.0
        ):
            if before_count > 0:
                segment_df = segment_df.iloc[0:0].copy()
                suppressed_by_activation_threshold = True
                activation_suppression_mode_applied = "hard_zero"
                activation_target_patch_count_after_threshold = 0
        elif target_patch_count is not None and target_patch_count <= 0:
            if before_count > 0:
                segment_df = segment_df.iloc[0:0].copy()
                suppressed_by_activation_threshold = True
                activation_suppression_mode_applied = "hard_zero"
                activation_target_patch_count_after_threshold = 0
        elif (
            enabled
            and before_count > 0
            and activation_threshold is not None
            and activation_score_before + 1e-12 < float(activation_threshold)
        ):
            if (
                str(activation_suppression_config.get("mode", "hard")) == "soft"
                and target_patch_count is not None
                and int(target_patch_count) > 0
            ):
                threshold_value = max(float(activation_threshold), 1e-12)
                raw_ratio = float(np.clip(float(activation_score_before) / threshold_value, 0.0, 1.0))
                soft_keep_fraction = float(
                    np.clip(
                        raw_ratio ** float(activation_suppression_config.get("soft_exponent", 1.0)),
                        0.0,
                        1.0,
                    )
                )
                soft_keep_fraction = max(
                    soft_keep_fraction,
                    float(activation_suppression_config.get("soft_min_fraction", 0.0)),
                )
                soft_min_keep_count = max(
                    int(activation_suppression_config.get("soft_min_keep_count", 1)),
                    0,
                )
                soft_keep_count = int(round(float(target_patch_count) * soft_keep_fraction))
                if soft_keep_fraction > 0.0:
                    soft_keep_count = max(soft_keep_count, soft_min_keep_count)
                activation_soft_keep_fraction = float(soft_keep_fraction)
                activation_target_patch_count_after_threshold = max(
                    min(soft_keep_count, int(target_patch_count)),
                    0,
                )
                activation_suppression_mode_applied = "soft_scale"
            else:
                segment_df = segment_df.iloc[0:0].copy()
                suppressed_by_activation_threshold = True
                activation_suppression_mode_applied = "hard_zero"
                activation_target_patch_count_after_threshold = 0
        effective_target_patch_count = activation_target_patch_count_after_threshold
        if target_patch_count is not None:
            effective_target_patch_count = min(
                int(target_patch_count),
                int(activation_target_patch_count_after_threshold)
                if activation_target_patch_count_after_threshold is not None
                else int(target_patch_count),
            )
        if effective_target_patch_count is not None and len(segment_df) > effective_target_patch_count:
            segment_df = (
                segment_df.sort_values(
                    ["ConfidenceSort", "PatchLengthSort", "PatchHeightSort"],
                    ascending=[False, False, False],
                )
                .head(int(effective_target_patch_count))
                .copy()
            )
        kept_frames.append(segment_df)
        summary_rows.append(
            {
                "UnitID": str(layer_row.get("UnitID", "")),
                "GeoIntervalKey": str(layer_row.get("GeoIntervalKey", "")),
                LAYER_SURFACE_PAIR_KEY_COL: layer_key,
                UNIT_LAYER_SEGMENT_KEY_COL: segment_key,
                "LayerThicknessMs": thickness_ms,
                "BaseDensity": base_density,
                "BaseNonZeroSegmentFraction": base_nonzero_fraction,
                "CalibrationScale": float(calibration_scale),
                "EffectiveDensityScale": float(effective_density_scale),
                "TargetDensity": target_density,
                "TargetNonZeroSegmentFraction": target_nonzero_fraction,
                "TargetPatchCount": target_patch_count,
                "PredictedPatchCountBefore": before_count,
                "PredictedPatchCountAfter": int(len(segment_df)),
                "PredictedDensityBefore": float(before_count / thickness_ms) if thickness_ms > 0.0 else 0.0,
                "PredictedDensityAfter": float(len(segment_df) / thickness_ms) if thickness_ms > 0.0 else 0.0,
                "ActivationScoreBeforeDensityControl": activation_score_before,
                "ActivationScoreThreshold": activation_threshold,
                "ActivationThresholdSource": activation_threshold_source,
                "ActivationSuppressionModeConfigured": str(activation_suppression_config.get("mode", "hard")),
                "ActivationSuppressionModeApplied": activation_suppression_mode_applied,
                "ActivationSoftKeepFraction": activation_soft_keep_fraction,
                "ActivationTargetPatchCountAfterThreshold": effective_target_patch_count,
                "PredictedNonZeroBeforeDensityControl": bool(predicted_nonzero_before),
                "PredictedNonZeroAfterDensityControl": bool(len(segment_df) > 0),
                "SuppressedByActivationThreshold": bool(suppressed_by_activation_threshold),
                "DensityControlEnabled": bool(enabled),
                "DensitySource": normalized_source,
                "DensityScale": float(density_scale),
            }
        )
        handled_segment_keys.add(segment_key)

    if handled_segment_keys:
        orphan_df = patch_work[~patch_work[UNIT_LAYER_SEGMENT_KEY_COL].astype(str).isin(handled_segment_keys)].copy()
    else:
        orphan_df = patch_work.copy()
    if not orphan_df.empty:
        kept_frames.append(orphan_df)
        for segment_key, orphan_group in orphan_df.groupby(UNIT_LAYER_SEGMENT_KEY_COL, dropna=False):
            summary_rows.append(
                {
                    "UnitID": str(orphan_group.get("UnitID", pd.Series([""])).iloc[0]),
                    "GeoIntervalKey": str(orphan_group.get("GeoIntervalKey", pd.Series([""])).iloc[0]),
                    LAYER_SURFACE_PAIR_KEY_COL: str(orphan_group.get(LAYER_SURFACE_PAIR_KEY_COL, pd.Series([""])).iloc[0]),
                    UNIT_LAYER_SEGMENT_KEY_COL: str(segment_key),
                    "LayerThicknessMs": None,
                    "BaseDensity": None,
                    "BaseNonZeroSegmentFraction": None,
                    "CalibrationScale": 1.0,
                    "EffectiveDensityScale": float(density_scale),
                    "TargetDensity": None,
                    "TargetNonZeroSegmentFraction": None,
                    "TargetPatchCount": None,
                    "PredictedPatchCountBefore": int(len(orphan_group)),
                    "PredictedPatchCountAfter": int(len(orphan_group)),
                    "PredictedDensityBefore": None,
                    "PredictedDensityAfter": None,
                    "ActivationScoreBeforeDensityControl": None,
                    "ActivationScoreThreshold": None,
                    "ActivationThresholdSource": "activation_threshold_orphan_segment",
                    "ActivationSuppressionModeConfigured": "hard",
                    "ActivationSuppressionModeApplied": "none",
                    "ActivationSoftKeepFraction": None,
                    "ActivationTargetPatchCountAfterThreshold": None,
                    "PredictedNonZeroBeforeDensityControl": bool(len(orphan_group) > 0),
                    "PredictedNonZeroAfterDensityControl": bool(len(orphan_group) > 0),
                    "SuppressedByActivationThreshold": False,
                    "DensityControlEnabled": bool(enabled),
                    "DensitySource": normalized_source,
                    "DensityScale": float(density_scale),
                }
            )

    result_df = pd.concat(kept_frames, ignore_index=True, sort=False) if kept_frames else patch_work.iloc[0:0].copy()
    result_df = result_df.drop(columns=["ConfidenceSort", "PatchLengthSort", "PatchHeightSort"], errors="ignore")
    result_df = result_df.reset_index(drop=True)
    return result_df, pd.DataFrame(summary_rows)


def load_sample_manifest(dataset_run_dir: Path) -> pd.DataFrame:
    manifest_path = Path(dataset_run_dir) / "aggregated" / "sample_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"sample_manifest.csv not found: {manifest_path}")
    df = read_csv_utf8(manifest_path)
    for col in [
        "BlockX",
        "BlockY",
        "WindowIndex",
        "WindowValidZCount",
        "PatchCount",
        "InstanceCount",
        "PackageBytes",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return ensure_layer_surface_pair_key_column(df, key_col=LAYER_SURFACE_PAIR_KEY_COL)


def build_unit_level_split(
    manifest_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    train_unit_count: int | None = None,
    val_unit_count: int | None = None,
    test_unit_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if manifest_df.empty:
        raise ValueError("manifest_df is empty")
    units = sorted(manifest_df["UnitID"].astype(str).unique().tolist())
    rng = random.Random(int(seed))
    rng.shuffle(units)
    unit_count = len(units)
    if unit_count < 3:
        raise ValueError(f"Need at least 3 units to split, got {unit_count}")

    explicit_count_mode = any(value is not None for value in [train_unit_count, val_unit_count, test_unit_count])
    if explicit_count_mode:
        if train_unit_count is None or test_unit_count is None:
            raise ValueError("explicit split mode requires train_unit_count and test_unit_count")
        n_train = int(train_unit_count)
        n_test = int(test_unit_count)
        n_val = unit_count - n_train - n_test if val_unit_count is None else int(val_unit_count)
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                "invalid explicit unit split counts: "
                f"train={n_train}, val={n_val}, test={n_test}, total_units={unit_count}"
            )
        if (n_train + n_val + n_test) != unit_count:
            raise ValueError(
                "explicit unit split counts do not cover all units: "
                f"train={n_train}, val={n_val}, test={n_test}, total_units={unit_count}"
            )
    else:
        n_train = max(1, int(round(unit_count * float(train_ratio))))
        n_val = max(1, int(round(unit_count * float(val_ratio))))
        if n_train + n_val >= unit_count:
            n_val = max(1, min(n_val, unit_count - 2))
            n_train = max(1, unit_count - n_val - 1)
        n_test = unit_count - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            else:
                n_val -= 1

    split_map = {
        "train": units[:n_train],
        "val": units[n_train:n_train + n_val],
        "test": units[n_train + n_val:],
    }
    rows: list[dict[str, Any]] = []
    for split_name, split_units in split_map.items():
        for unit_id in split_units:
            unit_rows = manifest_df[manifest_df["UnitID"].astype(str) == str(unit_id)]
            rows.append(
                {
                    "UnitID": str(unit_id),
                    "Split": split_name,
                    "WindowCount": int(len(unit_rows)),
                    "PositiveWindowCount": int(unit_rows["IsPositiveWindow"].sum()) if "IsPositiveWindow" in unit_rows.columns else None,
                    "PatchCount": int(unit_rows["PatchCount"].sum()) if "PatchCount" in unit_rows.columns else None,
                }
            )
    return pd.DataFrame(rows), split_map


def build_dense_targets_from_sparse_npz(
    npz_path: Path,
    slots_per_voxel: int = DEFAULT_SLOTS_PER_VOXEL,
) -> dict[str, np.ndarray]:
    sparse_arrays = load_sparse_npz_arrays(npz_path)
    return build_dense_targets_from_sparse_arrays(sparse_arrays, slots_per_voxel=slots_per_voxel)


def load_sparse_npz_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        sparse_arrays = {
            "input_features": np.asarray(data["input_features"]),
            "valid_z_mask": np.asarray(data["valid_z_mask"]),
            "instance_ijk": np.asarray(data["instance_ijk"]),
            "instance_geom": np.asarray(data["instance_geom"]),
            "instance_weight": np.asarray(data["instance_weight"]),
        }
        if "instance_source" in data.files:
            sparse_arrays["instance_source"] = np.asarray(data["instance_source"])
        if "instance_patch_index" in data.files:
            sparse_arrays["instance_patch_index"] = np.asarray(data["instance_patch_index"])
        if "count_volume" in data.files:
            sparse_arrays["count_volume"] = np.asarray(data["count_volume"])
    return sparse_arrays


def build_dense_targets_from_sparse_arrays(
    sparse_arrays: dict[str, np.ndarray],
    slots_per_voxel: int = DEFAULT_SLOTS_PER_VOXEL,
) -> dict[str, np.ndarray]:
    input_features = np.asarray(sparse_arrays["input_features"], dtype=np.float32)
    valid_z_mask = np.asarray(sparse_arrays["valid_z_mask"], dtype=np.float32)
    instance_ijk = np.asarray(sparse_arrays["instance_ijk"], dtype=np.int32)
    instance_geom = np.asarray(sparse_arrays["instance_geom"], dtype=np.float32)
    instance_weight = np.asarray(sparse_arrays["instance_weight"], dtype=np.float32)
    if "count_volume" in sparse_arrays:
        count_volume = np.asarray(sparse_arrays["count_volume"], dtype=np.float32)
    else:
        count_volume = np.zeros(input_features.shape[1:], dtype=np.float32)

    if instance_ijk.ndim == 1 and instance_ijk.size == 0:
        instance_ijk = instance_ijk.reshape(0, 3)
    if instance_geom.ndim == 1 and instance_geom.size == 0:
        instance_geom = instance_geom.reshape(0, len(GEOM_CHANNELS))
    if instance_weight.ndim == 0 and instance_weight.size == 1:
        instance_weight = instance_weight.reshape(1)

    if instance_ijk.ndim != 2 or instance_ijk.shape[1] < 3:
        raise ValueError(f"instance_ijk has invalid shape: {instance_ijk.shape}")
    if instance_geom.ndim != 2 or instance_geom.shape[1] != len(GEOM_CHANNELS):
        raise ValueError(f"instance_geom has invalid shape: {instance_geom.shape}")
    if len(instance_ijk) != len(instance_geom) or len(instance_ijk) != len(instance_weight):
        raise ValueError(
            "sparse instance arrays have inconsistent lengths: "
            f"ijk={len(instance_ijk)}, geom={len(instance_geom)}, weight={len(instance_weight)}"
        )

    _, nx, ny, nz = input_features.shape
    slots_per_voxel = int(slots_per_voxel)
    center_target = np.zeros((slots_per_voxel, nx, ny, nz), dtype=np.float32)
    geom_target = np.zeros((slots_per_voxel, len(GEOM_CHANNELS), nx, ny, nz), dtype=np.float32)
    weight_target = np.zeros((slots_per_voxel, nx, ny, nz), dtype=np.float32)

    if len(instance_ijk) > 0:
        i_idx = instance_ijk[:, 0]
        j_idx = instance_ijk[:, 1]
        k_idx = instance_ijk[:, 2]
        valid_mask = (
            (i_idx >= 0) & (i_idx < nx) &
            (j_idx >= 0) & (j_idx < ny) &
            (k_idx >= 0) & (k_idx < nz)
        )
        if np.any(valid_mask):
            valid_ijk = instance_ijk[valid_mask, :3]
            valid_geom = instance_geom[valid_mask]
            valid_weight = instance_weight[valid_mask]

            linear_index = (
                valid_ijk[:, 0].astype(np.int64) * (ny * nz) +
                valid_ijk[:, 1].astype(np.int64) * nz +
                valid_ijk[:, 2].astype(np.int64)
            )
            stable_order = np.argsort(linear_index, kind="mergesort")
            sorted_linear_index = linear_index[stable_order]
            sorted_ijk = valid_ijk[stable_order]
            sorted_geom = valid_geom[stable_order]
            sorted_weight = valid_weight[stable_order]

            position_index = np.arange(sorted_linear_index.shape[0], dtype=np.int32)
            group_start_mask = np.empty(sorted_linear_index.shape[0], dtype=bool)
            group_start_mask[0] = True
            group_start_mask[1:] = sorted_linear_index[1:] != sorted_linear_index[:-1]
            group_start_positions = np.maximum.accumulate(
                np.where(group_start_mask, position_index, 0)
            )
            slot_index = position_index - group_start_positions
            keep_mask = slot_index < slots_per_voxel

            kept_slots = slot_index[keep_mask]
            kept_ijk = sorted_ijk[keep_mask]
            kept_geom = sorted_geom[keep_mask]
            kept_weight = sorted_weight[keep_mask]

            center_target[kept_slots, kept_ijk[:, 0], kept_ijk[:, 1], kept_ijk[:, 2]] = 1.0
            geom_target[kept_slots, :, kept_ijk[:, 0], kept_ijk[:, 1], kept_ijk[:, 2]] = kept_geom
            weight_target[kept_slots, kept_ijk[:, 0], kept_ijk[:, 1], kept_ijk[:, 2]] = kept_weight
            overflow_instance_count = int((~keep_mask).sum())
        else:
            overflow_instance_count = 0
    else:
        overflow_instance_count = 0

    count_target = np.maximum(count_volume, 0.0).astype(np.float32, copy=False)[None, ...]
    return {
        "input_features": input_features,
        "valid_z_mask": valid_z_mask,
        "center_target": center_target,
        "geom_target": geom_target,
        "weight_target": weight_target,
        "count_target": count_target,
        "overflow_instance_count": np.asarray([overflow_instance_count], dtype=np.int32),
    }


def prepare_unit_context(
    unit_id: str,
    unit_dfn_root: Path = DEFAULT_UNIT_DFN_ROOT,
    xy_resolution: int = DEFAULT_XY_RESOLUTION,
    z_step_ms: float = DEFAULT_Z_STEP_MS,
) -> dict[str, Any]:
    unit_dir = Path(unit_dfn_root) / str(unit_id)
    patch_csv = unit_dir / "unit_dfn_patches.csv"
    patch_df = load_patch_table(patch_csv)
    layers_df = load_layer_table(unit_dir)
    unit_summary = load_unit_summary(unit_dir)
    if patch_df.empty:
        raise ValueError(f"empty patch table: {patch_csv}")
    block_x = int(unit_summary.get("BlockX", patch_df["BlockX"].iloc[0]))
    block_y = int(unit_summary.get("BlockY", patch_df["BlockY"].iloc[0]))
    x_min, x_max, y_min, y_max = compute_unit_bounds_from_patch_table(patch_df)
    z_min, z_max = compute_time_bounds(patch_df, layers_df, z_padding_ms=0.0)
    grid = build_grid_spec(
        unit_id=str(unit_id),
        block_x=block_x,
        block_y=block_y,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
        z_bounds=(z_min, z_max),
        xy_resolution=int(xy_resolution),
        z_step_ms=float(z_step_ms),
    )
    return {
        "unit_id": str(unit_id),
        "unit_dir": unit_dir,
        "patch_df": patch_df,
        "layers_df": layers_df,
        "grid": grid,
        "block_x": block_x,
        "block_y": block_y,
        "x_bounds": (x_min, x_max),
        "y_bounds": (y_min, y_max),
    }


def build_window_grid(
    unit_context: dict[str, Any],
    window_top: float,
    window_size: int = DEFAULT_WINDOW_SIZE,
    z_step_ms: float = DEFAULT_Z_STEP_MS,
) -> GridSpec:
    x_min, x_max = unit_context["x_bounds"]
    y_min, y_max = unit_context["y_bounds"]
    window_base = float(window_top) + float(window_size) * float(z_step_ms)
    return build_grid_spec(
        unit_id=str(unit_context["unit_id"]),
        block_x=int(unit_context["block_x"]),
        block_y=int(unit_context["block_y"]),
        x_bounds=(float(x_min), float(x_max)),
        y_bounds=(float(y_min), float(y_max)),
        z_bounds=(float(window_top), float(window_base)),
        xy_resolution=DEFAULT_XY_RESOLUTION,
        z_step_ms=float(z_step_ms),
    )


def normalize_vectors_last_dim(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norm = np.where(norm > float(eps), norm, 1.0)
    return vectors / norm


def prediction_to_label_payload(
    center_probs: np.ndarray,
    count_pred: np.ndarray,
    geom_pred: np.ndarray,
    valid_z_mask: np.ndarray,
    threshold: float,
    max_slots_per_voxel: int,
    max_total_patches: int | None = None,
    decode_mode: str = "strict",
    relaxed_min_count: int = 1,
    count_activation_threshold: float = 0.5,
    min_count_if_active: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    center_probs = np.asarray(center_probs, dtype=np.float32)
    count_pred = np.asarray(count_pred, dtype=np.float32)
    geom_pred = np.asarray(geom_pred, dtype=np.float32)
    valid_z_mask = np.asarray(valid_z_mask, dtype=np.float32)
    center_probs = center_probs * valid_z_mask.reshape(1, 1, 1, -1)
    count_pred = np.squeeze(count_pred).astype(np.float32)
    if count_pred.ndim == 4 and count_pred.shape[0] == 1:
        count_pred = count_pred[0]
    if count_pred.ndim != 3:
        raise ValueError(f"count_pred must be 3D after squeeze, got shape={count_pred.shape}")
    count_pred = count_pred * valid_z_mask.reshape(1, 1, -1)

    slot_count = int(center_probs.shape[0])
    geom_swapped = np.moveaxis(geom_pred, 1, -1)
    order = np.argsort(-center_probs, axis=0)
    sorted_center_probs = np.take_along_axis(center_probs, order, axis=0)
    sorted_geom = np.take_along_axis(geom_swapped, order[..., None], axis=0)

    count_limit = np.rint(count_pred).astype(np.int16)
    count_activation_mask = count_pred >= float(count_activation_threshold)
    count_limit = np.where(
        count_activation_mask,
        np.maximum(count_limit, int(max(min_count_if_active, 1))),
        0,
    ).astype(np.int16)
    count_limit = np.clip(count_limit, 0, min(int(max_slots_per_voxel), slot_count))
    forced_voxel_mask = np.zeros_like(count_limit, dtype=bool)
    if str(decode_mode).lower() == "relaxed":
        relaxed_min_count = int(max(relaxed_min_count, 1))
        top_slot_mask = sorted_center_probs[0] >= float(threshold)
        forced_voxel_mask = top_slot_mask & (count_limit < relaxed_min_count)
        count_limit = np.where(forced_voxel_mask, relaxed_min_count, count_limit).astype(np.int16)
        count_limit = np.clip(count_limit, 0, min(int(max_slots_per_voxel), slot_count))
    keep_mask = (np.arange(slot_count, dtype=np.int16).reshape(-1, 1, 1, 1) < count_limit.reshape(1, *count_limit.shape))
    keep_mask &= sorted_center_probs >= float(threshold)

    raw_candidate_count = int(np.count_nonzero(keep_mask))
    if max_total_patches is not None and raw_candidate_count > int(max_total_patches):
        flat_keep = keep_mask.reshape(-1)
        flat_scores = sorted_center_probs.reshape(-1)
        active_indices = np.flatnonzero(flat_keep)
        rank_order = np.argsort(flat_scores[active_indices])[::-1][: int(max_total_patches)]
        limited_indices = active_indices[rank_order]
        limited_flat_keep = np.zeros_like(flat_keep, dtype=bool)
        limited_flat_keep[limited_indices] = True
        keep_mask = limited_flat_keep.reshape(keep_mask.shape)

    center_probs_kept = sorted_center_probs * keep_mask.astype(np.float32)
    center_heatmap = np.transpose(center_probs_kept, (1, 2, 3, 0)).astype(np.float32)
    center_count = keep_mask.sum(axis=0).astype(np.int16)
    normals = normalize_vectors_last_dim(np.transpose(sorted_geom[:, :, :, :, 3:6], (1, 2, 3, 0, 4)))
    u_dirs = normalize_vectors_last_dim(np.transpose(sorted_geom[:, :, :, :, 6:9], (1, 2, 3, 0, 4)))
    payload = {
        "center_heatmap": center_heatmap,
        "center_count": center_count,
        "offsets": np.transpose(sorted_geom[:, :, :, :, 0:3], (1, 2, 3, 0, 4)).astype(np.float32),
        "normals": normals.astype(np.float32),
        "u_dirs": u_dirs.astype(np.float32),
        "lengths": np.transpose(np.maximum(sorted_geom[:, :, :, :, 9], 1e-6), (1, 2, 3, 0)).astype(np.float32),
        "heights": np.transpose(np.maximum(sorted_geom[:, :, :, :, 10], 1e-6), (1, 2, 3, 0)).astype(np.float32),
        "confidence": np.transpose(np.clip(sorted_geom[:, :, :, :, 11], 0.0, 1.0), (1, 2, 3, 0)).astype(np.float32),
        "source_patch_index": np.zeros(center_heatmap.shape, dtype=np.int32),
    }
    stats = {
        "decode_mode": str(decode_mode).lower(),
        "raw_candidate_count": int(raw_candidate_count),
        "kept_candidate_count": int(np.count_nonzero(keep_mask)),
        "mean_pred_count": float(np.mean(count_limit)),
        "max_pred_count": int(np.max(count_limit)) if count_limit.size else 0,
        "forced_voxel_count": int(np.count_nonzero(forced_voxel_mask)),
        "count_activation_threshold": float(count_activation_threshold),
        "min_count_if_active": int(max(min_count_if_active, 1)),
    }
    return payload, stats


def decode_window_predictions_with_optional_second_pass(
    center_probs: np.ndarray,
    count_pred: np.ndarray,
    geom_pred: np.ndarray,
    valid_z_mask: np.ndarray,
    grid: GridSpec,
    layers_df: pd.DataFrame,
    decode_runtime_config: dict[str, Any],
    max_slots_per_voxel: int,
    dedupe_xy_tol_m: float,
    dedupe_time_tol_ms: float,
    dedupe_azimuth_tol_deg: float,
    dedupe_dip_tol_deg: float,
    layer_surface_pair_key: str = "",
    calibration_payload: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    primary_threshold = scalar_float(decode_runtime_config.get("center_threshold"), default=0.7)
    primary_max_total = scalar_int(decode_runtime_config.get("max_total_patches_per_window"), default=0)
    primary_label_payload, primary_filter_stats = prediction_to_label_payload(
        center_probs=center_probs,
        count_pred=count_pred,
        geom_pred=geom_pred,
        valid_z_mask=valid_z_mask,
        threshold=float(primary_threshold),
        max_slots_per_voxel=int(max_slots_per_voxel),
        max_total_patches=int(primary_max_total) if int(primary_max_total) > 0 else None,
        decode_mode=str(decode_runtime_config.get("decode_mode", "strict")),
        relaxed_min_count=int(scalar_int(decode_runtime_config.get("relaxed_min_count"), default=1)),
        count_activation_threshold=float(
            scalar_float(decode_runtime_config.get("count_activation_threshold"), default=0.5)
        ),
        min_count_if_active=int(scalar_int(decode_runtime_config.get("min_count_if_active"), default=1)),
    )
    primary_patch_df, primary_decoded_df, primary_summary = decode_instance_label_to_patches(
        label_payload=primary_label_payload,
        grid=grid,
        layers_df=layers_df,
        threshold=float(primary_threshold),
    )
    if not primary_patch_df.empty:
        primary_patch_df["DecodePass"] = "primary"
    if not primary_decoded_df.empty:
        primary_decoded_df["DecodePass"] = "primary"

    merged_patch_df = primary_patch_df.copy()
    merged_decoded_df = primary_decoded_df.copy()
    merged_summary = dict(primary_summary)
    second_pass_config = resolve_second_pass_decode_config(decode_runtime_config)
    trigger_second_pass, trigger_reason, calibration_scale = should_trigger_second_pass_decode(
        second_pass_config=second_pass_config,
        primary_patch_count=int(len(primary_patch_df)),
        layer_surface_pair_key=layer_surface_pair_key,
        calibration_payload=calibration_payload,
    )
    second_filter_stats: dict[str, Any] = {}
    second_summary: dict[str, Any] = {}
    second_patch_df = pd.DataFrame()
    second_decoded_df = pd.DataFrame()

    if trigger_second_pass:
        second_threshold = scalar_float(second_pass_config.get("center_threshold"), default=primary_threshold)
        second_max_total = scalar_int(second_pass_config.get("max_total_patches_per_window"), default=0)
        second_label_payload, second_filter_stats = prediction_to_label_payload(
            center_probs=center_probs,
            count_pred=count_pred,
            geom_pred=geom_pred,
            valid_z_mask=valid_z_mask,
            threshold=float(second_threshold),
            max_slots_per_voxel=int(max_slots_per_voxel),
            max_total_patches=int(second_max_total) if int(second_max_total) > 0 else None,
            decode_mode=str(second_pass_config.get("decode_mode", "relaxed")),
            relaxed_min_count=int(scalar_int(second_pass_config.get("relaxed_min_count"), default=1)),
            count_activation_threshold=float(
                scalar_float(second_pass_config.get("count_activation_threshold"), default=0.5)
            ),
            min_count_if_active=int(scalar_int(second_pass_config.get("min_count_if_active"), default=1)),
        )
        second_patch_df, second_decoded_df, second_summary = decode_instance_label_to_patches(
            label_payload=second_label_payload,
            grid=grid,
            layers_df=layers_df,
            threshold=float(second_threshold),
        )
        if not second_patch_df.empty:
            second_patch_df["DecodePass"] = "second_pass"
        if not second_decoded_df.empty:
            second_decoded_df["DecodePass"] = "second_pass"
        merged_patch_df = pd.concat([primary_patch_df, second_patch_df], ignore_index=True, sort=False)
        if not merged_patch_df.empty:
            merged_patch_df = dedupe_patch_df(
                merged_patch_df,
                xy_tol_m=float(dedupe_xy_tol_m),
                time_tol_ms=float(dedupe_time_tol_ms),
                azimuth_tol_deg=float(dedupe_azimuth_tol_deg),
                dip_tol_deg=float(dedupe_dip_tol_deg),
            )
        merged_decoded_df = pd.concat([primary_decoded_df, second_decoded_df], ignore_index=True, sort=False)
        merged_summary["decoded_patch_count"] = int(len(merged_patch_df))
        merged_summary["active_slot_count"] = int(len(merged_patch_df))

    merged_stats = {
        **primary_filter_stats,
        "primary_patch_count": int(len(primary_patch_df)),
        "primary_active_slot_count": int(primary_summary.get("active_slot_count", len(primary_patch_df))),
        "second_pass_triggered": bool(trigger_second_pass),
        "second_pass_reason": str(trigger_reason),
        "second_pass_calibration_scale": calibration_scale,
        "second_pass_raw_candidate_count": int(second_filter_stats.get("raw_candidate_count", 0)),
        "second_pass_kept_candidate_count": int(second_filter_stats.get("kept_candidate_count", 0)),
        "second_pass_patch_count": int(len(second_patch_df)),
        "second_pass_active_slot_count": int(second_summary.get("active_slot_count", len(second_patch_df))),
        "merged_patch_count": int(len(merged_patch_df)),
    }
    return merged_patch_df, merged_decoded_df, merged_summary, merged_stats


def dedupe_patch_df(
    patch_df: pd.DataFrame,
    xy_tol_m: float = 6.25,
    time_tol_ms: float = 0.4,
    azimuth_tol_deg: float = 20.0,
    dip_tol_deg: float = 12.0,
) -> pd.DataFrame:
    if patch_df.empty:
        return patch_df.copy()
    work = ensure_unit_layer_segment_key_column(
        patch_df.copy(),
        key_col=UNIT_LAYER_SEGMENT_KEY_COL,
        pair_key_col=LAYER_SURFACE_PAIR_KEY_COL,
    )
    work["ConfidenceSort"] = pd.to_numeric(work["Confidence"], errors="coerce").fillna(0.0)
    work = work.sort_values(["ConfidenceSort", "PatchLength", "PatchHeight"], ascending=[False, False, False]).reset_index(drop=True)
    row_count = len(work)

    def _numeric_array(column_name: str) -> np.ndarray:
        if column_name not in work.columns:
            return np.full(row_count, np.nan, dtype=float)
        return pd.to_numeric(work[column_name], errors="coerce").to_numpy(dtype=float)

    center_x = _numeric_array("CenterX")
    center_y = _numeric_array("CenterY")
    center_t = _numeric_array("CenterTIME")
    azimuth = _numeric_array("Azimuth")
    dip = _numeric_array("Dip")
    interval_keys = np.full(row_count, "", dtype=object)
    for candidate_col in (UNIT_LAYER_SEGMENT_KEY_COL, LAYER_SURFACE_PAIR_KEY_COL, "GeoIntervalKey"):
        if candidate_col not in work.columns:
            continue
        values = work[candidate_col].fillna("").astype(str).to_numpy(dtype=object)
        empty_mask = np.array([not str(value).strip() or str(value).strip().lower() == "nan" for value in interval_keys], dtype=bool)
        if empty_mask.any():
            interval_keys[empty_mask] = values[empty_mask]

    xy_tol = float(xy_tol_m)
    time_tol = float(time_tol_ms)
    azimuth_tol = float(azimuth_tol_deg)
    dip_tol = float(dip_tol_deg)
    xy_bucket_size = xy_tol if xy_tol > 0.0 else 1e-6
    time_bucket_size = time_tol if time_tol > 0.0 else 1e-6
    xy_neighbor_offsets = (-1, 0, 1) if xy_tol > 0.0 else (0,)
    time_neighbor_offsets = (-1, 0, 1) if time_tol > 0.0 else (0,)

    kept_indices: list[int] = []
    bucket_to_kept: dict[tuple[str, int, int, int], list[int]] = defaultdict(list)

    for row_idx in range(row_count):
        row_center_x = center_x[row_idx]
        row_center_y = center_y[row_idx]
        row_center_t = center_t[row_idx]
        row_azimuth = azimuth[row_idx]
        row_dip = dip[row_idx]
        interval_key = str(interval_keys[row_idx])

        is_indexable = (
            np.isfinite(row_center_x)
            and np.isfinite(row_center_y)
            and np.isfinite(row_center_t)
            and np.isfinite(row_azimuth)
            and np.isfinite(row_dip)
        )
        duplicated = False

        if is_indexable:
            bucket_x = int(np.floor(row_center_x / xy_bucket_size))
            bucket_y = int(np.floor(row_center_y / xy_bucket_size))
            bucket_t = int(np.floor(row_center_t / time_bucket_size))
            for offset_x in xy_neighbor_offsets:
                if duplicated:
                    break
                for offset_y in xy_neighbor_offsets:
                    if duplicated:
                        break
                    for offset_t in time_neighbor_offsets:
                        candidate_key = (interval_key, bucket_x + offset_x, bucket_y + offset_y, bucket_t + offset_t)
                        candidate_indices = bucket_to_kept.get(candidate_key)
                        if not candidate_indices:
                            continue
                        for kept_idx in candidate_indices:
                            dx = center_x[kept_idx] - row_center_x
                            dy = center_y[kept_idx] - row_center_y
                            dt = center_t[kept_idx] - row_center_t
                            xy_dist = float(np.hypot(dx, dy))
                            az_diff = abs(azimuth[kept_idx] - row_azimuth) % 360.0
                            az_diff = min(az_diff, 360.0 - az_diff)
                            dip_diff = abs(dip[kept_idx] - row_dip)
                            if (
                                xy_dist <= xy_tol
                                and abs(dt) <= time_tol
                                and az_diff <= azimuth_tol
                                and dip_diff <= dip_tol
                            ):
                                duplicated = True
                                break

        if duplicated:
            continue

        kept_indices.append(row_idx)
        if is_indexable:
            bucket_key = (interval_key, bucket_x, bucket_y, bucket_t)
            bucket_to_kept[bucket_key].append(row_idx)

    result = work.iloc[kept_indices].reset_index(drop=True)
    if "ConfidenceSort" in result.columns:
        result = result.drop(columns=["ConfidenceSort"])
    return result


def append_lines_to_docx(docx_path: Path, title: str, lines: list[str]) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    body = [f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] + list(lines)
    doc.add_paragraph("\n".join(body))
    doc.add_paragraph("")
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))
