# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pandas as pd


LAYER_SURFACE_PAIR_KEY_COL = "LayerSurfacePairKey"
UNIT_LAYER_SEGMENT_KEY_COL = "UnitLayerSegmentKey"
LAYER_DENSITY_PRIOR_METADATA_KEY = "layer_density_priors"
LAYER_DENSITY_CALIBRATION_METADATA_KEY = "layer_density_calibration_settings"
SURFACE_CODE_ALIASES = {
    "SEIS_TOP": "TOP_1100MS",
    "SEIS_BASE": "BOTTOM_3800MS",
}
LAYER_DECODE_OVERRIDES = {
    "T2->T3": {
        "second_pass": {
            "enabled": True,
            "center_threshold": 0.25,
            "count_activation_threshold": 0.1,
            "min_count_if_active": 1,
            "decode_mode": "relaxed",
            "relaxed_min_count": 1,
            "max_total_patches_per_window": 16,
            "primary_patch_count_threshold": 2,
            "min_calibration_scale": 1.5,
            "trigger_without_calibration": False,
        }
    },
    "T7->BOTTOM_3800MS": {
        "center_threshold": 0.4,
        "count_activation_threshold": 0.5,
        "min_count_if_active": 1,
        "decode_mode": "strict",
        "relaxed_min_count": 1,
        "max_total_patches_per_window": 64,
        "second_pass": {
            "enabled": True,
            "center_threshold": 0.1,
            "count_activation_threshold": 0.0,
            "min_count_if_active": 1,
            "decode_mode": "relaxed",
            "relaxed_min_count": 1,
            "max_total_patches_per_window": 64,
            "primary_patch_count_threshold": 4,
            "min_calibration_scale": 0.0,
            "trigger_without_calibration": True,
        },
    }
}
LAYER_DENSITY_CALIBRATION_OVERRIDES = {
    "T1->T2": {
        "activation_suppression_mode": "soft",
        "soft_activation_min_fraction": 0.45,
    },
    "T3->T4": {
        "min_scale": 0.10,
        "activation_suppression_mode": "soft",
        "soft_activation_min_fraction": 0.55,
    },
    "T4->T5": {
        "activation_suppression_mode": "soft",
        "soft_activation_min_fraction": 0.45,
    },
    "T5->T6": {
        "activation_suppression_mode": "soft",
        "soft_activation_min_fraction": 0.55,
    },
    "T6->T7": {
        "min_scale": 0.0,
        "activation_suppression_mode": "soft",
        "soft_activation_min_fraction": 0.50,
    },
    "TOP_1100MS->T1": {
        "activation_suppression_mode": "soft",
        "soft_activation_min_fraction": 0.35,
    },
}


def normalize_surface_code(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return SURFACE_CODE_ALIASES.get(text.upper(), text)


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def normalize_unit_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def build_unit_id_from_block_coords(block_x: Any, block_y: Any) -> str:
    block_x_value = pd.to_numeric(block_x, errors="coerce")
    block_y_value = pd.to_numeric(block_y, errors="coerce")
    if pd.isna(block_x_value) or pd.isna(block_y_value):
        return ""
    return f"BX{int(round(float(block_x_value)))}_BY{int(round(float(block_y_value)))}"


def resolve_unit_id_from_row(row: Any) -> str:
    if not hasattr(row, "get"):
        return ""
    unit_id = normalize_unit_id(row.get("UnitID", ""))
    if unit_id:
        return unit_id
    return build_unit_id_from_block_coords(
        row.get("BlockX", ""),
        row.get("BlockY", ""),
    )


def build_layer_surface_pair_key(top_surface_code: Any, base_surface_code: Any) -> str:
    return f"{normalize_surface_code(top_surface_code)}->{normalize_surface_code(base_surface_code)}"


def build_layer_surface_pair_key_from_row(row: Any, key_col: str = LAYER_SURFACE_PAIR_KEY_COL) -> str:
    if hasattr(row, "get"):
        existing = normalize_surface_code(row.get(key_col, ""))
        if existing:
            return existing
        return build_layer_surface_pair_key(
            row.get("TopSurfaceCode", ""),
            row.get("BaseSurfaceCode", ""),
        )
    return ""


def build_unit_layer_segment_key(
    unit_id: Any,
    geo_interval_key: Any,
    layer_surface_pair_key: Any,
) -> str:
    unit_text = normalize_unit_id(unit_id) or "UNKNOWN_UNIT"
    interval_text = str(geo_interval_key).strip() or "UNKNOWN_INTERVAL"
    pair_text = str(layer_surface_pair_key).strip() or "UNKNOWN_LAYER_PAIR"
    return f"{unit_text}__{interval_text}__{pair_text}"


def build_unit_layer_segment_key_from_row(
    row: Any,
    key_col: str = UNIT_LAYER_SEGMENT_KEY_COL,
    pair_key_col: str = LAYER_SURFACE_PAIR_KEY_COL,
) -> str:
    if not hasattr(row, "get"):
        return ""
    existing = str(row.get(key_col, "")).strip()
    if existing and existing.lower() != "nan":
        return existing
    layer_surface_pair_key = build_layer_surface_pair_key_from_row(row, key_col=pair_key_col)
    return build_unit_layer_segment_key(
        unit_id=resolve_unit_id_from_row(row),
        geo_interval_key=row.get("GeoIntervalKey", ""),
        layer_surface_pair_key=layer_surface_pair_key,
    )


def sanitize_layer_surface_pair_key(layer_surface_pair_key: str) -> str:
    text = str(layer_surface_pair_key).strip()
    text = text.replace("->", "__TO__")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "UNSPECIFIED_LAYER_PAIR"


def ensure_layer_surface_pair_key_column(
    manifest_df: pd.DataFrame,
    key_col: str = LAYER_SURFACE_PAIR_KEY_COL,
) -> pd.DataFrame:
    work = manifest_df.copy()
    if work.empty:
        if key_col not in work.columns:
            work[key_col] = pd.Series(dtype="object")
        return work

    derived_keys = [
        build_layer_surface_pair_key(top_surface_code, base_surface_code)
        for top_surface_code, base_surface_code in zip(
            work.get("TopSurfaceCode", pd.Series([""] * len(work))),
            work.get("BaseSurfaceCode", pd.Series([""] * len(work))),
        )
    ]
    if key_col not in work.columns:
        work[key_col] = derived_keys
        return work

    normalized_existing = work[key_col].apply(normalize_surface_code)
    fill_mask = normalized_existing.eq("")
    work[key_col] = normalized_existing
    if fill_mask.any():
        work.loc[fill_mask, key_col] = pd.Series(derived_keys, index=work.index).loc[fill_mask]
    return work


def ensure_unit_id_column(
    manifest_df: pd.DataFrame,
    key_col: str = "UnitID",
) -> pd.DataFrame:
    work = manifest_df.copy()
    if work.empty:
        if key_col not in work.columns:
            work[key_col] = pd.Series(dtype="object")
        return work

    if key_col in work.columns:
        existing = work[key_col].apply(normalize_unit_id)
    else:
        existing = pd.Series([""] * len(work), index=work.index, dtype="object")
    fill_mask = existing.eq("")
    work[key_col] = existing
    if fill_mask.any():
        derived_keys = [
            build_unit_id_from_block_coords(block_x, block_y)
            for block_x, block_y in zip(
                work.get("BlockX", pd.Series([""] * len(work))),
                work.get("BlockY", pd.Series([""] * len(work))),
            )
        ]
        work.loc[fill_mask, key_col] = pd.Series(derived_keys, index=work.index).loc[fill_mask]
    return work


def ensure_unit_layer_segment_key_column(
    manifest_df: pd.DataFrame,
    key_col: str = UNIT_LAYER_SEGMENT_KEY_COL,
    pair_key_col: str = LAYER_SURFACE_PAIR_KEY_COL,
) -> pd.DataFrame:
    work = ensure_unit_id_column(
        ensure_layer_surface_pair_key_column(manifest_df, key_col=pair_key_col),
        key_col="UnitID",
    )
    if work.empty:
        if key_col not in work.columns:
            work[key_col] = pd.Series(dtype="object")
        return work

    derived_keys = [
        build_unit_layer_segment_key(
            unit_id=unit_id,
            geo_interval_key=geo_interval_key,
            layer_surface_pair_key=layer_surface_pair_key,
        )
        for unit_id, geo_interval_key, layer_surface_pair_key in zip(
            work.get("UnitID", pd.Series([""] * len(work))),
            work.get("GeoIntervalKey", pd.Series([""] * len(work))),
            work.get(pair_key_col, pd.Series([""] * len(work))),
        )
    ]
    if key_col not in work.columns:
        work[key_col] = derived_keys
        return work

    existing = work[key_col].fillna("").astype(str).str.strip()
    fill_mask = existing.eq("") | existing.str.lower().eq("nan")
    work[key_col] = existing
    if fill_mask.any():
        work.loc[fill_mask, key_col] = pd.Series(derived_keys, index=work.index).loc[fill_mask]
    return work


def filter_manifest_by_layer_surface_pair_key(
    manifest_df: pd.DataFrame,
    layer_surface_pair_key: str,
    key_col: str = LAYER_SURFACE_PAIR_KEY_COL,
) -> pd.DataFrame:
    normalized_key = str(layer_surface_pair_key).strip()
    work = ensure_layer_surface_pair_key_column(manifest_df, key_col=key_col)
    return work[work[key_col].astype(str) == normalized_key].copy()


def summarize_manifest_by_layer_surface_pair(
    manifest_df: pd.DataFrame,
    key_col: str = LAYER_SURFACE_PAIR_KEY_COL,
) -> pd.DataFrame:
    work = ensure_unit_id_column(
        ensure_layer_surface_pair_key_column(manifest_df, key_col=key_col),
        key_col="UnitID",
    )
    if work.empty:
        return pd.DataFrame(
            columns=[
                key_col,
                "TopSurfaceCode",
                "BaseSurfaceCode",
                "UnitCount",
                "WindowCount",
                "PositiveWindowCount",
                "PatchCount",
                "InstanceCount",
            ]
        )

    for col in ("TopSurfaceCode", "BaseSurfaceCode", "UnitID"):
        if col not in work.columns:
            work[col] = ""
    grouped = (
        work.groupby([key_col, "TopSurfaceCode", "BaseSurfaceCode"], dropna=False)
        .agg(
            UnitCount=("UnitID", lambda values: int(pd.Series(values).astype(str).nunique())),
            WindowCount=("UnitID", "count"),
            PositiveWindowCount=("IsPositiveWindow", "sum") if "IsPositiveWindow" in work.columns else ("UnitID", "count"),
            PatchCount=("PatchCount", "sum") if "PatchCount" in work.columns else ("UnitID", "count"),
            InstanceCount=("InstanceCount", "sum") if "InstanceCount" in work.columns else ("UnitID", "count"),
        )
        .reset_index()
    )
    if "PositiveWindowCount" in grouped.columns:
        grouped["PositiveWindowCount"] = pd.to_numeric(grouped["PositiveWindowCount"], errors="coerce").fillna(0).astype(int)
    if "PatchCount" in grouped.columns:
        grouped["PatchCount"] = pd.to_numeric(grouped["PatchCount"], errors="coerce").fillna(0).astype(int)
    if "InstanceCount" in grouped.columns:
        grouped["InstanceCount"] = pd.to_numeric(grouped["InstanceCount"], errors="coerce").fillna(0).astype(int)
    grouped["WindowCount"] = pd.to_numeric(grouped["WindowCount"], errors="coerce").fillna(0).astype(int)
    grouped["UnitCount"] = pd.to_numeric(grouped["UnitCount"], errors="coerce").fillna(0).astype(int)
    grouped = grouped.sort_values(["WindowCount", key_col], ascending=[False, True]).reset_index(drop=True)
    return grouped


def _load_python_module(config_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_layer_model_registry_{config_path.stem}", str(config_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load layer model registry: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_checkpoint_path(raw_value: Any, base_dir: Path) -> str:
    text = normalize_surface_code(raw_value)
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _normalize_layer_decode_config(raw_decode: Any) -> dict[str, Any]:
    if not isinstance(raw_decode, dict):
        return {}
    config: dict[str, Any] = {}
    if "center_threshold" in raw_decode:
        try:
            config["center_threshold"] = float(raw_decode["center_threshold"])
        except (TypeError, ValueError):
            pass
    if "count_activation_threshold" in raw_decode:
        try:
            config["count_activation_threshold"] = float(raw_decode["count_activation_threshold"])
        except (TypeError, ValueError):
            pass
    if "min_count_if_active" in raw_decode:
        try:
            config["min_count_if_active"] = int(raw_decode["min_count_if_active"])
        except (TypeError, ValueError):
            pass
    if "decode_mode" in raw_decode:
        text = str(raw_decode["decode_mode"]).strip().lower()
        if text in {"strict", "relaxed"}:
            config["decode_mode"] = text
    if "relaxed_min_count" in raw_decode:
        try:
            config["relaxed_min_count"] = int(raw_decode["relaxed_min_count"])
        except (TypeError, ValueError):
            pass
    if "max_total_patches_per_window" in raw_decode:
        try:
            value = int(raw_decode["max_total_patches_per_window"])
            if value > 0:
                config["max_total_patches_per_window"] = value
        except (TypeError, ValueError):
            pass
    if "score" in raw_decode:
        try:
            config["score"] = float(raw_decode["score"])
        except (TypeError, ValueError):
            pass
    second_pass = _normalize_second_pass_decode_config(raw_decode.get("second_pass"))
    if second_pass:
        config["second_pass"] = second_pass
    return config


def _normalize_second_pass_decode_config(raw_second_pass: Any) -> dict[str, Any]:
    if not isinstance(raw_second_pass, dict):
        return {}
    config: dict[str, Any] = {}
    if "enabled" in raw_second_pass:
        config["enabled"] = _normalize_bool(raw_second_pass.get("enabled"), default=False)
    if "center_threshold" in raw_second_pass:
        try:
            config["center_threshold"] = float(raw_second_pass["center_threshold"])
        except (TypeError, ValueError):
            pass
    if "count_activation_threshold" in raw_second_pass:
        try:
            config["count_activation_threshold"] = float(raw_second_pass["count_activation_threshold"])
        except (TypeError, ValueError):
            pass
    if "min_count_if_active" in raw_second_pass:
        try:
            config["min_count_if_active"] = int(raw_second_pass["min_count_if_active"])
        except (TypeError, ValueError):
            pass
    if "decode_mode" in raw_second_pass:
        text = str(raw_second_pass["decode_mode"]).strip().lower()
        if text in {"strict", "relaxed"}:
            config["decode_mode"] = text
    if "relaxed_min_count" in raw_second_pass:
        try:
            config["relaxed_min_count"] = int(raw_second_pass["relaxed_min_count"])
        except (TypeError, ValueError):
            pass
    if "max_total_patches_per_window" in raw_second_pass:
        try:
            value = int(raw_second_pass["max_total_patches_per_window"])
            if value > 0:
                config["max_total_patches_per_window"] = value
        except (TypeError, ValueError):
            pass
    if "primary_patch_count_threshold" in raw_second_pass:
        try:
            config["primary_patch_count_threshold"] = max(int(raw_second_pass["primary_patch_count_threshold"]), 0)
        except (TypeError, ValueError):
            pass
    if "min_calibration_scale" in raw_second_pass:
        try:
            config["min_calibration_scale"] = float(raw_second_pass["min_calibration_scale"])
        except (TypeError, ValueError):
            pass
    if "trigger_without_calibration" in raw_second_pass:
        config["trigger_without_calibration"] = _normalize_bool(
            raw_second_pass.get("trigger_without_calibration"),
            default=False,
        )
    return config


def _normalize_layer_density_prior(raw_prior: Any) -> dict[str, Any]:
    if not isinstance(raw_prior, dict):
        return {}
    config: dict[str, Any] = {}
    float_fields = {
        "aggregate_density",
        "unit_mean_density",
        "unit_median_density",
        "unit_p25_density",
        "unit_p10_density",
        "unit_p75_density",
        "unit_max_density",
        "robust_density",
        "total_thickness_ms",
        "nonzero_segment_fraction",
    }
    int_fields = {
        "unit_count",
        "segment_count",
        "total_patch_count",
        "nonzero_segment_count",
    }
    for field in float_fields:
        if field not in raw_prior:
            continue
        try:
            config[field] = float(raw_prior[field])
        except (TypeError, ValueError):
            continue
    for field in int_fields:
        if field not in raw_prior:
            continue
        try:
            config[field] = int(raw_prior[field])
        except (TypeError, ValueError):
            continue
    return config


def _normalize_layer_density_calibration_config(raw_config: Any) -> dict[str, Any]:
    if not isinstance(raw_config, dict):
        return {}
    config: dict[str, Any] = {}
    if "min_scale" in raw_config:
        try:
            config["min_scale"] = float(raw_config["min_scale"])
        except (TypeError, ValueError):
            pass
    if "max_scale" in raw_config:
        try:
            config["max_scale"] = float(raw_config["max_scale"])
        except (TypeError, ValueError):
            pass
    if "activation_suppression_mode" in raw_config:
        mode_text = str(raw_config["activation_suppression_mode"]).strip().lower()
        if mode_text in {"hard", "soft"}:
            config["activation_suppression_mode"] = mode_text
    if "soft_activation_min_fraction" in raw_config:
        try:
            config["soft_activation_min_fraction"] = float(raw_config["soft_activation_min_fraction"])
        except (TypeError, ValueError):
            pass
    if "soft_activation_exponent" in raw_config:
        try:
            config["soft_activation_exponent"] = float(raw_config["soft_activation_exponent"])
        except (TypeError, ValueError):
            pass
    if "soft_activation_min_keep_count" in raw_config:
        try:
            config["soft_activation_min_keep_count"] = int(raw_config["soft_activation_min_keep_count"])
        except (TypeError, ValueError):
            pass
    return config


def _normalize_registry_models(raw_models: dict[str, Any], base_dir: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    models: dict[str, str] = {}
    layer_decode_settings: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in raw_models.items():
        model_key = str(raw_key).strip()
        if not model_key:
            continue
        checkpoint_value = raw_value
        raw_decode = None
        if isinstance(raw_value, dict):
            checkpoint_value = raw_value.get("checkpoint") or raw_value.get("path") or raw_value.get("best_checkpoint")
            raw_decode = raw_value.get("decode") or raw_value.get("decode_config")
        checkpoint_path = _resolve_checkpoint_path(checkpoint_value, base_dir=base_dir)
        if checkpoint_path:
            models[model_key] = checkpoint_path
        decode_config = _normalize_layer_decode_config(raw_decode)
        if decode_config:
            layer_decode_settings[model_key] = decode_config
    return models, layer_decode_settings


def apply_layer_decode_overrides(
    layer_decode_settings: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    work = {
        str(key).strip(): _normalize_layer_decode_config(value)
        for key, value in (layer_decode_settings or {}).items()
        if str(key).strip()
    }
    for layer_key, raw_override in LAYER_DECODE_OVERRIDES.items():
        normalized_override = _normalize_layer_decode_config(raw_override)
        if not normalized_override:
            continue
        merged = dict(work.get(layer_key, {}))
        merged.update(normalized_override)
        work[layer_key] = merged
    return {key: value for key, value in work.items() if value}


def apply_layer_density_calibration_overrides(
    layer_density_calibration_settings: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    work = {
        str(key).strip(): _normalize_layer_density_calibration_config(value)
        for key, value in (layer_density_calibration_settings or {}).items()
        if str(key).strip()
    }
    for layer_key, raw_override in LAYER_DENSITY_CALIBRATION_OVERRIDES.items():
        normalized_override = _normalize_layer_density_calibration_config(raw_override)
        if not normalized_override:
            continue
        merged = dict(work.get(layer_key, {}))
        merged.update(normalized_override)
        work[layer_key] = merged
    return {key: value for key, value in work.items() if value}


def load_layer_model_registry(config_path: Path) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    module = _load_python_module(resolved_path)
    raw_registry = (
        getattr(module, "LAYER_MODEL_REGISTRY", None)
        or getattr(module, "MODEL_REGISTRY", None)
        or getattr(module, "REGISTRY", None)
    )
    if raw_registry is None:
        raise ValueError(
            f"layer model registry not found in {resolved_path}; "
            "expected LAYER_MODEL_REGISTRY / MODEL_REGISTRY / REGISTRY"
        )

    base_dir = resolved_path.parent
    default_checkpoint = getattr(module, "DEFAULT_CHECKPOINT", None)
    metadata = getattr(module, "REGISTRY_METADATA", {})

    if isinstance(raw_registry, dict) and ("models" in raw_registry or "default_checkpoint" in raw_registry):
        raw_models = raw_registry.get("models", {})
        default_checkpoint = raw_registry.get("default_checkpoint", default_checkpoint)
        metadata = raw_registry.get("metadata", metadata)
    elif isinstance(raw_registry, dict):
        raw_models = raw_registry
    else:
        raise TypeError(f"layer model registry must be dict-like, got: {type(raw_registry)}")

    models, model_level_decode_settings = _normalize_registry_models(raw_models, base_dir=base_dir)
    metadata = metadata if isinstance(metadata, dict) else {}
    metadata_decode_settings = {
        str(key).strip(): _normalize_layer_decode_config(value)
        for key, value in (metadata.get("layer_decode_settings", {}) or {}).items()
    }
    metadata_density_priors = {
        str(key).strip(): _normalize_layer_density_prior(value)
        for key, value in (metadata.get(LAYER_DENSITY_PRIOR_METADATA_KEY, {}) or {}).items()
        if str(key).strip()
    }
    metadata_density_calibration_settings = {
        str(key).strip(): _normalize_layer_density_calibration_config(value)
        for key, value in (metadata.get(LAYER_DENSITY_CALIBRATION_METADATA_KEY, {}) or {}).items()
        if str(key).strip()
    }
    layer_decode_settings = apply_layer_decode_overrides(
        {
            key: value
            for key, value in {**metadata_decode_settings, **model_level_decode_settings}.items()
            if value
        }
    )
    return {
        "config_path": str(resolved_path),
        "models": models,
        "default_checkpoint": _resolve_checkpoint_path(default_checkpoint, base_dir=base_dir),
        "metadata": metadata,
        "layer_decode_settings": layer_decode_settings,
        "layer_density_priors": {
            key: value
            for key, value in metadata_density_priors.items()
            if value
        },
        "layer_density_calibration_settings": apply_layer_density_calibration_overrides(
            metadata_density_calibration_settings
        ),
    }


def resolve_layer_checkpoint(
    registry_payload: dict[str, Any],
    layer_surface_pair_key: str,
    fallback_checkpoint: Path | str | None = None,
    require_match: bool = True,
) -> str:
    normalized_key = str(layer_surface_pair_key).strip()
    models = registry_payload.get("models", {}) or {}
    if normalized_key in models:
        return str(models[normalized_key])

    if fallback_checkpoint is not None and str(fallback_checkpoint).strip():
        return str(Path(str(fallback_checkpoint)).resolve())

    default_checkpoint = normalize_surface_code(registry_payload.get("default_checkpoint", ""))
    if default_checkpoint:
        return default_checkpoint

    if require_match:
        available = ", ".join(sorted(models.keys())[:12])
        raise KeyError(
            f"no checkpoint configured for layer_surface_pair_key={normalized_key}; "
            f"available keys: {available}"
        )
    return ""


def resolve_layer_decode_config(
    registry_payload: dict[str, Any] | None,
    layer_surface_pair_key: str,
) -> dict[str, Any]:
    if registry_payload is None:
        return {}
    normalized_key = str(layer_surface_pair_key).strip()
    decode_settings = registry_payload.get("layer_decode_settings", {}) or {}
    raw_config = decode_settings.get(normalized_key, {})
    return _normalize_layer_decode_config(raw_config)


def resolve_layer_density_prior(
    registry_payload: dict[str, Any] | None,
    layer_surface_pair_key: str,
) -> dict[str, Any]:
    if registry_payload is None:
        return {}
    normalized_key = str(layer_surface_pair_key).strip()
    density_priors = registry_payload.get("layer_density_priors", {}) or {}
    raw_config = density_priors.get(normalized_key, {})
    return _normalize_layer_density_prior(raw_config)


def resolve_layer_density_calibration_config(
    registry_payload: dict[str, Any] | None,
    layer_surface_pair_key: str,
) -> dict[str, Any]:
    normalized_key = str(layer_surface_pair_key).strip()
    if registry_payload is None:
        raw_config = LAYER_DENSITY_CALIBRATION_OVERRIDES.get(normalized_key, {})
        return _normalize_layer_density_calibration_config(raw_config)
    settings = registry_payload.get("layer_density_calibration_settings", {}) or {}
    raw_config = settings.get(normalized_key, LAYER_DENSITY_CALIBRATION_OVERRIDES.get(normalized_key, {}))
    return _normalize_layer_density_calibration_config(raw_config)


def write_layer_model_registry_py(
    output_path: Path,
    models: dict[str, str],
    default_checkpoint: str | None = None,
    metadata: dict[str, Any] | None = None,
    layer_decode_settings: dict[str, dict[str, Any]] | None = None,
    layer_density_priors: dict[str, dict[str, Any]] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    normalized_decode_settings = apply_layer_decode_overrides(layer_decode_settings)
    if normalized_decode_settings:
        metadata["layer_decode_settings"] = normalized_decode_settings
    normalized_density_priors = {
        str(key).strip(): _normalize_layer_density_prior(value)
        for key, value in (layer_density_priors or {}).items()
        if str(key).strip()
    }
    normalized_density_priors = {
        key: value
        for key, value in normalized_density_priors.items()
        if value
    }
    if normalized_density_priors:
        metadata[LAYER_DENSITY_PRIOR_METADATA_KEY] = normalized_density_priors
    normalized_density_calibration_settings = apply_layer_density_calibration_overrides(
        metadata.get(LAYER_DENSITY_CALIBRATION_METADATA_KEY, {})
    )
    if normalized_density_calibration_settings:
        metadata[LAYER_DENSITY_CALIBRATION_METADATA_KEY] = normalized_density_calibration_settings
    lines = [
        "# -*- coding: utf-8 -*-",
        "LAYER_MODEL_REGISTRY = {",
        '    "models": {',
    ]
    for key in sorted(models.keys()):
        checkpoint_path = str(Path(models[key]).resolve())
        lines.append(f'        "{key}": "{checkpoint_path}",')
    lines.extend(
        [
            "    },",
            f'    "default_checkpoint": {repr(str(Path(default_checkpoint).resolve())) if default_checkpoint else "None"},',
            f'    "metadata": {repr(metadata)},',
            "}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
