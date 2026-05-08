# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from baseline_common import compute_file_sha256
from layer_model_registry import (
    LAYER_SURFACE_PAIR_KEY_COL,
    ensure_layer_surface_pair_key_column,
    normalize_surface_code,
)


CANONICAL_SURFACE_CODES = [
    "TOP_1100MS",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "BOTTOM_3800MS",
]
CANONICAL_LAYER_SURFACE_PAIR_KEYS = [
    f"{top}->{base}"
    for top, base in zip(CANONICAL_SURFACE_CODES[:-1], CANONICAL_SURFACE_CODES[1:])
]
SUPPORTED_LAYER_TRAINING_OVERRIDE_KEYS = {
    "slots_per_voxel",
    "base_channels",
    "batch_size",
    "num_workers",
    "torch_num_threads",
    "torch_num_interop_threads",
    "epochs",
    "learning_rate",
    "weight_decay",
    "seed",
    "train_limit_samples",
    "val_limit_samples",
    "max_train_steps",
    "max_val_steps",
    "disable_amp",
    "no_cache_raw_packages",
    "no_progress",
    "center_positive_weight",
    "center_negative_weight",
    "center_focal_gamma",
    "count_positive_weight",
    "count_negative_weight",
    "center_loss_weight",
    "count_loss_weight",
    "calibration_center_thresholds",
    "calibration_count_thresholds",
    "calibration_window_weight",
}

_INT_OVERRIDE_KEYS = {
    "slots_per_voxel",
    "base_channels",
    "batch_size",
    "num_workers",
    "torch_num_threads",
    "torch_num_interop_threads",
    "epochs",
    "seed",
    "train_limit_samples",
    "val_limit_samples",
    "max_train_steps",
    "max_val_steps",
}
_FLOAT_OVERRIDE_KEYS = {
    "learning_rate",
    "weight_decay",
    "center_positive_weight",
    "center_negative_weight",
    "center_focal_gamma",
    "count_positive_weight",
    "count_negative_weight",
    "center_loss_weight",
    "count_loss_weight",
    "calibration_window_weight",
}
_BOOL_OVERRIDE_KEYS = {
    "disable_amp",
    "no_cache_raw_packages",
    "no_progress",
}
_FLOAT_LIST_OVERRIDE_KEYS = {
    "calibration_center_thresholds",
    "calibration_count_thresholds",
}


def split_layer_surface_pair_key(layer_surface_pair_key: Any) -> tuple[str, str]:
    text = str(layer_surface_pair_key or "").strip()
    if "->" not in text:
        return "", ""
    top_text, base_text = text.split("->", 1)
    return normalize_surface_code(top_text), normalize_surface_code(base_text)


def canonicalize_layer_surface_pair_key(
    layer_surface_pair_key: Any = "",
    *,
    top_surface_code: Any = "",
    base_surface_code: Any = "",
) -> str:
    top_code = normalize_surface_code(top_surface_code)
    base_code = normalize_surface_code(base_surface_code)
    if not top_code and not base_code:
        top_code, base_code = split_layer_surface_pair_key(layer_surface_pair_key)

    if not top_code and not base_code:
        return ""

    if top_code and top_code not in CANONICAL_SURFACE_CODES:
        return ""
    if base_code and base_code not in CANONICAL_SURFACE_CODES:
        return ""

    if top_code and base_code:
        try:
            top_index = CANONICAL_SURFACE_CODES.index(top_code)
            base_index = CANONICAL_SURFACE_CODES.index(base_code)
        except ValueError:
            return ""
        if base_index - top_index != 1:
            return ""
        return f"{top_code}->{base_code}"

    if not top_code and base_code:
        base_index = CANONICAL_SURFACE_CODES.index(base_code)
        if base_index <= 0:
            return ""
        top_code = CANONICAL_SURFACE_CODES[base_index - 1]
        return f"{top_code}->{base_code}"

    if top_code and not base_code:
        top_index = CANONICAL_SURFACE_CODES.index(top_code)
        if top_index >= len(CANONICAL_SURFACE_CODES) - 1:
            return ""
        base_code = CANONICAL_SURFACE_CODES[top_index + 1]
        return f"{top_code}->{base_code}"

    return ""


def canonicalize_manifest_layer_surface_pairs(
    manifest_df: pd.DataFrame,
    key_col: str = LAYER_SURFACE_PAIR_KEY_COL,
) -> pd.DataFrame:
    work = ensure_layer_surface_pair_key_column(manifest_df, key_col=key_col).copy()
    if work.empty:
        for col in (key_col, "TopSurfaceCode", "BaseSurfaceCode"):
            if col not in work.columns:
                work[col] = pd.Series(dtype="object")
        return work

    canonical_keys: list[str] = []
    top_codes: list[str] = []
    base_codes: list[str] = []
    for row in work.to_dict("records"):
        canonical_key = canonicalize_layer_surface_pair_key(
            row.get(key_col, ""),
            top_surface_code=row.get("TopSurfaceCode", ""),
            base_surface_code=row.get("BaseSurfaceCode", ""),
        )
        canonical_keys.append(canonical_key)
        if canonical_key:
            top_code, base_code = split_layer_surface_pair_key(canonical_key)
        else:
            top_code = normalize_surface_code(row.get("TopSurfaceCode", ""))
            base_code = normalize_surface_code(row.get("BaseSurfaceCode", ""))
        top_codes.append(top_code)
        base_codes.append(base_code)

    work[key_col] = canonical_keys
    work["TopSurfaceCode"] = top_codes
    work["BaseSurfaceCode"] = base_codes
    return work


def _load_python_module(config_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_layer_training_config_{config_path.stem}", str(config_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load layer training config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def normalize_layer_training_override(raw_override: Any) -> dict[str, Any]:
    if not isinstance(raw_override, dict):
        return {}
    override: dict[str, Any] = {}
    for key, raw_value in raw_override.items():
        key_text = str(key).strip()
        if key_text not in SUPPORTED_LAYER_TRAINING_OVERRIDE_KEYS:
            continue
        if key_text in _INT_OVERRIDE_KEYS:
            try:
                override[key_text] = int(raw_value)
            except (TypeError, ValueError):
                continue
            continue
        if key_text in _FLOAT_OVERRIDE_KEYS:
            try:
                override[key_text] = float(raw_value)
            except (TypeError, ValueError):
                continue
            continue
        if key_text in _BOOL_OVERRIDE_KEYS:
            override[key_text] = _normalize_bool(raw_value)
            continue
        if key_text in _FLOAT_LIST_OVERRIDE_KEYS:
            if isinstance(raw_value, (list, tuple)):
                normalized_values: list[float] = []
                for item in raw_value:
                    try:
                        normalized_values.append(float(item))
                    except (TypeError, ValueError):
                        pass
                if normalized_values:
                    override[key_text] = normalized_values
    return override


def _normalize_allowed_layer_surface_pair_keys(raw_keys: Any) -> list[str]:
    if raw_keys is None:
        raw_keys = CANONICAL_LAYER_SURFACE_PAIR_KEYS
    normalized_keys: list[str] = []
    seen: set[str] = set()
    for raw_key in raw_keys:
        canonical_key = canonicalize_layer_surface_pair_key(raw_key)
        if canonical_key not in CANONICAL_LAYER_SURFACE_PAIR_KEYS:
            continue
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        normalized_keys.append(canonical_key)
    if normalized_keys:
        return normalized_keys
    return list(CANONICAL_LAYER_SURFACE_PAIR_KEYS)


def load_layer_training_config(config_py: Path | str | None) -> dict[str, Any]:
    config_path = Path(config_py).resolve() if config_py else None
    allowed_layer_surface_pair_keys = list(CANONICAL_LAYER_SURFACE_PAIR_KEYS)
    layer_training_overrides: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    if config_path is not None:
        module = _load_python_module(config_path)
        allowed_layer_surface_pair_keys = _normalize_allowed_layer_surface_pair_keys(
            getattr(module, "ALLOWED_LAYER_SURFACE_PAIR_KEYS", CANONICAL_LAYER_SURFACE_PAIR_KEYS)
        )
        raw_overrides = getattr(module, "LAYER_TRAINING_OVERRIDES", {})
        if isinstance(raw_overrides, dict):
            for raw_key, raw_override in raw_overrides.items():
                canonical_key = canonicalize_layer_surface_pair_key(raw_key)
                if canonical_key not in allowed_layer_surface_pair_keys:
                    continue
                normalized_override = normalize_layer_training_override(raw_override)
                if normalized_override:
                    layer_training_overrides[canonical_key] = normalized_override
        raw_metadata = getattr(module, "TRAINING_PLAN_METADATA", {})
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
    return {
        "config_py": str(config_path) if config_path is not None else "",
        "config_sha256": compute_file_sha256(config_path),
        "allowed_layer_surface_pair_keys": allowed_layer_surface_pair_keys,
        "layer_training_overrides": layer_training_overrides,
        "metadata": metadata,
    }
