from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


CURRENT_DIR = Path(__file__).resolve().parent
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import load_surface_tables, validate_surface_order  # noqa: E402


ALLOWED_LAYERS = ("沙三段", "沙四段")
LAYER_CODE = {"沙三段": 3.0, "沙四段": 4.0}
ATTRIBUTE_COLUMNS = ("SeisAmp", "Coherence", "AntTrack", "CurvatureMax")
FEATURE_COLUMNS = (
    *ATTRIBUTE_COLUMNS,
    "LayerCode",
    "RelativeTimeInLayer",
    "TimeSinceTop",
    "TimeToBase",
    "LayerThickness",
    "TimeMs",
)
NULL_THRESHOLD = -1.0e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step6 two-stage fracture-density models without volume prediction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=0, help="Smoke-test chunk cap; 0 reads the full Step5B table.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional isolated output override.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.mask(values <= NULL_THRESHOLD, np.nan)


def finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_surface_lookups(layer_dir: Path) -> dict[str, tuple[cKDTree, np.ndarray]]:
    surfaces = load_surface_tables(layer_dir)
    lookups: dict[str, tuple[cKDTree, np.ndarray]] = {}
    for code, payload in surfaces.items():
        table = payload["table"]
        xy = table[["X", "Y"]].to_numpy(dtype=np.float64)
        lookups[code] = (cKDTree(xy), table["Z"].to_numpy(dtype=np.float64))
    return lookups


def attach_layer_time_features(df: pd.DataFrame, lookups: dict[str, tuple[cKDTree, np.ndarray]]) -> pd.DataFrame:
    work = df.copy()
    for column in ("X", "Y", "TIME"):
        work[column] = clean_numeric(work[column])
    work = work.dropna(subset=["X", "Y", "TIME"]).copy()
    if work.empty:
        return work
    xy = work[["X", "Y"]].to_numpy(dtype=np.float64)
    for code, (tree, surface_time) in lookups.items():
        _, indices = tree.query(xy, k=1, p=1)
        work[f"{code}_TIME"] = surface_time[np.asarray(indices, dtype=np.int64)]
    work = validate_surface_order(work, min_thickness=1.0)
    work = work[work["Check_All"].fillna(False)].copy()
    if work.empty:
        return work
    layer = work["LayerGroup"].astype(str)
    is_sha3 = layer.eq("沙三段")
    is_sha4 = layer.eq("沙四段")
    work["LayerTopTime"] = np.where(is_sha3, work["T4_TIME"], np.where(is_sha4, work["T6_TIME"], np.nan))
    work["LayerBaseTime"] = np.where(is_sha3, work["T6_TIME"], np.where(is_sha4, work["T7_TIME"], np.nan))
    work["LayerThickness"] = work["LayerBaseTime"] - work["LayerTopTime"]
    work["TimeSinceTop"] = work["TIME"] - work["LayerTopTime"]
    work["TimeToBase"] = work["LayerBaseTime"] - work["TIME"]
    work = work[
        work["LayerThickness"].gt(0)
        & work["TimeSinceTop"].ge(0)
        & work["TimeToBase"].ge(0)
    ].copy()
    work["RelativeTimeInLayer"] = work["TimeSinceTop"] / work["LayerThickness"]
    work["LayerCode"] = layer.loc[work.index].map(LAYER_CODE)
    work["TimeMs"] = work["TIME"]
    return work


def collect_training_data(
    input_csv: Path,
    lookups: dict[str, tuple[cKDTree, np.ndarray]],
    chunksize: int,
    max_chunks: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    usecols = [
        "SourceKind",
        "SourceWellName",
        "TrackWellName",
        "X",
        "Y",
        "TIME",
        "LayerGroup",
        "PresenceLabel",
        "DensityLabel",
        "SampleWeight",
        *ATTRIBUTE_COLUMNS,
    ]
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        layer: {"x": [], "presence": [], "density": [], "weight": [], "group": []} for layer in ALLOWED_LAYERS
    }
    group_codes: dict[str, int] = {}
    scan = {
        "input_rows_seen": 0,
        "rows_after_surface_and_feature_filter": 0,
        "dropped_missing_required_feature": 0,
        "dropped_invalid_label_or_weight": 0,
        "chunks_read": 0,
        "source_kind_counts": {},
    }
    reader = pd.read_csv(input_csv, encoding="utf-8-sig", usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk_index, chunk in enumerate(reader, start=1):
        scan["chunks_read"] = chunk_index
        scan["input_rows_seen"] += int(len(chunk))
        for key, count in chunk["SourceKind"].astype(str).value_counts(dropna=False).items():
            scan["source_kind_counts"][str(key)] = scan["source_kind_counts"].get(str(key), 0) + int(count)
        work = chunk[chunk["LayerGroup"].astype(str).isin(ALLOWED_LAYERS)].copy()
        work = attach_layer_time_features(work, lookups)
        if work.empty:
            continue
        for column in (*ATTRIBUTE_COLUMNS, *FEATURE_COLUMNS[4:]):
            work[column] = clean_numeric(work[column])
        required_feature_mask = work[list(FEATURE_COLUMNS)].notna().all(axis=1)
        scan["dropped_missing_required_feature"] += int((~required_feature_mask).sum())
        work = work.loc[required_feature_mask].copy()
        work["PresenceLabel"] = clean_numeric(work["PresenceLabel"])
        work["DensityLabel"] = clean_numeric(work["DensityLabel"])
        work["SampleWeight"] = clean_numeric(work["SampleWeight"])
        valid_label = work["PresenceLabel"].isin([0.0, 1.0]) & work["SampleWeight"].gt(0)
        valid_label &= work["SourceWellName"].notna()
        positive = work["PresenceLabel"].eq(1.0)
        valid_label &= (~positive) | (work["DensityLabel"].notna() & work["DensityLabel"].ge(0))
        scan["dropped_invalid_label_or_weight"] += int((~valid_label).sum())
        work = work.loc[valid_label].copy()
        if work.empty:
            continue
        scan["rows_after_surface_and_feature_filter"] += int(len(work))
        for well in work["SourceWellName"].astype(str).unique():
            if well not in group_codes:
                group_codes[well] = len(group_codes)
        work["_GroupCode"] = work["SourceWellName"].astype(str).map(group_codes).astype(np.int16)
        for layer in ALLOWED_LAYERS:
            layer_work = work[work["LayerGroup"].astype(str).eq(layer)]
            if layer_work.empty:
                continue
            parts[layer]["x"].append(layer_work[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32))
            parts[layer]["presence"].append(layer_work["PresenceLabel"].to_numpy(dtype=np.int8))
            parts[layer]["density"].append(layer_work["DensityLabel"].to_numpy(dtype=np.float32))
            parts[layer]["weight"].append(layer_work["SampleWeight"].to_numpy(dtype=np.float32))
            parts[layer]["group"].append(layer_work["_GroupCode"].to_numpy(dtype=np.int16))
        print(
            f"[step6-train] chunk={chunk_index} input_rows={scan['input_rows_seen']} "
            f"kept_rows={scan['rows_after_surface_and_feature_filter']}",
            flush=True,
        )
        if max_chunks > 0 and chunk_index >= max_chunks:
            break

    data: dict[str, dict[str, np.ndarray]] = {}
    for layer in ALLOWED_LAYERS:
        if not parts[layer]["x"]:
            raise RuntimeError(f"no valid training rows for {layer}")
        data[layer] = {key: np.concatenate(value, axis=0) for key, value in parts[layer].items()}
    scan["source_well_count"] = int(len(group_codes))
    scan["source_well_group_codes"] = {well: int(code) for well, code in sorted(group_codes.items(), key=lambda item: item[1])}
    return data, scan


def classifier_metrics(y_true: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive_rows": int((y_true == 1).sum()),
        "positive_fraction": float(np.mean(y_true == 1)),
        "probability_stats": finite_stats(probability),
    }
    if np.unique(y_true).size == 2:
        out.update(
            {
                "roc_auc": float(roc_auc_score(y_true, probability, sample_weight=weight)),
                "average_precision": float(average_precision_score(y_true, probability, sample_weight=weight)),
                "log_loss": float(log_loss(y_true, np.clip(probability, 1.0e-7, 1.0 - 1.0e-7), sample_weight=weight)),
                "brier_score": float(brier_score_loss(y_true, probability, sample_weight=weight)),
            }
        )
    else:
        out.update({"roc_auc": None, "average_precision": None, "log_loss": None, "brier_score": None})
    return out


def regressor_metrics(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(y_true)),
        "target_stats": finite_stats(y_true),
        "prediction_stats": finite_stats(prediction),
        "mae": float(mean_absolute_error(y_true, prediction, sample_weight=weight)),
        "rmse": float(mean_squared_error(y_true, prediction, sample_weight=weight) ** 0.5),
        "r2": float(r2_score(y_true, prediction, sample_weight=weight)) if len(y_true) > 1 else None,
    }


def fit_layer(
    layer: str,
    arrays: dict[str, np.ndarray],
    classifier_params: dict[str, Any],
    regressor_params: dict[str, Any],
    validation_folds: int,
    random_state: int,
    density_cap: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    x = arrays["x"]
    presence = arrays["presence"].astype(np.int8)
    density = arrays["density"].astype(np.float64)
    weight = arrays["weight"].astype(np.float64)
    groups = arrays["group"]
    unique_groups = np.unique(groups)
    split_count = min(int(validation_folds), int(len(unique_groups)))
    if split_count < 2:
        raise RuntimeError(f"{layer} has fewer than two source-well groups")

    fold_rows: list[dict[str, Any]] = []
    combined_targets: list[np.ndarray] = []
    combined_predictions: list[np.ndarray] = []
    combined_weights: list[np.ndarray] = []
    splitter = GroupKFold(n_splits=split_count)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, presence, groups), start=1):
        started = time.time()
        classifier = HistGradientBoostingClassifier(random_state=random_state + fold, **classifier_params)
        classifier.fit(x[train_idx], presence[train_idx], sample_weight=weight[train_idx])
        probability = classifier.predict_proba(x[val_idx])[:, 1]

        positive_train = train_idx[presence[train_idx] == 1]
        if len(positive_train) == 0:
            raise RuntimeError(f"{layer} fold {fold} has no positive density rows")
        regressor = HistGradientBoostingRegressor(random_state=random_state + fold, **regressor_params)
        regressor.fit(x[positive_train], density[positive_train], sample_weight=weight[positive_train])
        conditional = np.clip(regressor.predict(x[val_idx]), 0.0, density_cap)
        final_density = np.clip(probability * conditional, 0.0, density_cap)
        target_density = np.where(presence[val_idx] == 1, density[val_idx], 0.0)
        positive_val_mask = presence[val_idx] == 1
        fold_row = {
            "layer": layer,
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "train_source_well_count": int(np.unique(groups[train_idx]).size),
            "validation_source_well_count": int(np.unique(groups[val_idx]).size),
            "source_well_overlap_count": int(len(set(groups[train_idx]).intersection(set(groups[val_idx])))),
            "classifier": classifier_metrics(presence[val_idx], probability, weight[val_idx]),
            "conditional_density": regressor_metrics(
                density[val_idx][positive_val_mask],
                conditional[positive_val_mask],
                weight[val_idx][positive_val_mask],
            ),
            "combined_density": regressor_metrics(target_density, final_density, weight[val_idx]),
            "elapsed_seconds": float(time.time() - started),
        }
        fold_rows.append(fold_row)
        combined_targets.append(target_density)
        combined_predictions.append(final_density)
        combined_weights.append(weight[val_idx])
        print(
            f"[step6-train] layer={layer} fold={fold}/{split_count} "
            f"auc={fold_row['classifier']['roc_auc']} combined_mae={fold_row['combined_density']['mae']:.6f} "
            f"elapsed_s={fold_row['elapsed_seconds']:.1f}",
            flush=True,
        )

    final_started = time.time()
    final_classifier = HistGradientBoostingClassifier(random_state=random_state, **classifier_params)
    final_classifier.fit(x, presence, sample_weight=weight)
    positive_all = presence == 1
    final_regressor = HistGradientBoostingRegressor(random_state=random_state, **regressor_params)
    final_regressor.fit(x[positive_all], density[positive_all], sample_weight=weight[positive_all])
    aggregate = regressor_metrics(
        np.concatenate(combined_targets),
        np.concatenate(combined_predictions),
        np.concatenate(combined_weights),
    )
    fold_auc = [row["classifier"]["roc_auc"] for row in fold_rows if row["classifier"]["roc_auc"] is not None]
    fold_ap = [
        row["classifier"]["average_precision"]
        for row in fold_rows
        if row["classifier"]["average_precision"] is not None
    ]
    summary = {
        "layer": layer,
        "rows": int(len(x)),
        "positive_rows": int(positive_all.sum()),
        "positive_fraction": float(positive_all.mean()),
        "source_well_count": int(unique_groups.size),
        "validation_fold_count": int(split_count),
        "feature_columns": list(FEATURE_COLUMNS),
        "sample_weight_stats": finite_stats(weight),
        "density_positive_stats": finite_stats(density[positive_all]),
        "out_of_fold_combined_density": aggregate,
        "mean_fold_roc_auc": float(np.mean(fold_auc)) if fold_auc else None,
        "mean_fold_average_precision": float(np.mean(fold_ap)) if fold_ap else None,
        "all_folds_group_disjoint": all(row["source_well_overlap_count"] == 0 for row in fold_rows),
        "final_fit_elapsed_seconds": float(time.time() - final_started),
    }
    bundle = {
        "layer": layer,
        "classifier": final_classifier,
        "conditional_density_regressor": final_regressor,
        "feature_columns": list(FEATURE_COLUMNS),
        "density_cap": float(density_cap),
        "training_summary": summary,
    }
    return bundle, fold_rows, summary


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    input_csv = Path(config["unified_samples_csv"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Step5B input not found: {input_csv}")
    if not layer_dir.exists():
        raise FileNotFoundError(f"layer directory not found: {layer_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "step6_two_stage_density_models.joblib"
    summary_path = output_dir / "step6_two_stage_training_summary.json"
    folds_path = output_dir / "step6_two_stage_grouped_validation.csv"
    for path in (artifact_path, summary_path, folds_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing Step6 training output: {path}")

    started = time.time()
    lookups = load_surface_lookups(layer_dir)
    data, scan_summary = collect_training_data(
        input_csv=input_csv,
        lookups=lookups,
        chunksize=int(config.get("training_chunksize", 250000)),
        max_chunks=int(args.max_chunks),
    )
    model_bundles: dict[str, Any] = {}
    layer_summaries: dict[str, Any] = {}
    fold_rows: list[dict[str, Any]] = []
    for offset, layer in enumerate(ALLOWED_LAYERS):
        bundle, rows, layer_summary = fit_layer(
            layer=layer,
            arrays=data[layer],
            classifier_params=dict(config.get("classifier", {})),
            regressor_params=dict(config.get("conditional_density_regressor", {})),
            validation_folds=int(config.get("validation_group_folds", 3)),
            random_state=int(config.get("random_state", 42)) + offset * 100,
            density_cap=float(config.get("density_cap", 10.0)),
        )
        model_bundles[layer] = bundle
        layer_summaries[layer] = layer_summary
        fold_rows.extend(rows)

    checks = {
        "models_for_both_layers": sorted(model_bundles) == sorted(ALLOWED_LAYERS),
        "curvature_pos_excluded": "CurvaturePos" not in FEATURE_COLUMNS,
        "ordinary_curvature_included": "CurvatureMax" in FEATURE_COLUMNS,
        "all_folds_source_well_disjoint": all(row["source_well_overlap_count"] == 0 for row in fold_rows),
        "all_layers_have_positive_density_rows": all(summary["positive_rows"] > 0 for summary in layer_summaries.values()),
    }
    status = "pass" if all(checks.values()) else "fail"
    artifact = {
        "contract_version": "step6_two_stage_curvature_led_v1",
        "model_logic": "presence_probability_times_conditional_density",
        "feature_columns": list(FEATURE_COLUMNS),
        "attribute_columns": list(ATTRIBUTE_COLUMNS),
        "models": model_bundles,
        "prediction_sample_interval_ms": float(config.get("prediction_sample_interval_ms", 2.0)),
        "training_input": str(input_csv),
        "training_uses_all_chunks": int(args.max_chunks) == 0,
        "config": config,
    }
    temporary_artifact = artifact_path.with_suffix(artifact_path.suffix + ".partial")
    joblib.dump(artifact, temporary_artifact, compress=3)
    temporary_artifact.replace(artifact_path)

    flat_fold_rows = []
    for row in fold_rows:
        flat_fold_rows.append(
            {
                "LayerGroup": row["layer"],
                "Fold": row["fold"],
                "TrainRows": row["train_rows"],
                "ValidationRows": row["validation_rows"],
                "TrainSourceWellCount": row["train_source_well_count"],
                "ValidationSourceWellCount": row["validation_source_well_count"],
                "SourceWellOverlapCount": row["source_well_overlap_count"],
                "PresenceROCAUC": row["classifier"]["roc_auc"],
                "PresenceAveragePrecision": row["classifier"]["average_precision"],
                "PresenceLogLoss": row["classifier"]["log_loss"],
                "ConditionalDensityMAE": row["conditional_density"]["mae"],
                "ConditionalDensityR2": row["conditional_density"]["r2"],
                "CombinedDensityMAE": row["combined_density"]["mae"],
                "CombinedDensityR2": row["combined_density"]["r2"],
                "ElapsedSeconds": row["elapsed_seconds"],
            }
        )
    pd.DataFrame(flat_fold_rows).to_csv(folds_path, index=False, encoding="utf-8-sig")
    summary = {
        "status": status,
        "config_path": str(config_path),
        "training_input": str(input_csv),
        "output_dir": str(output_dir),
        "output_paths": {
            "model_joblib": str(artifact_path),
            "training_summary_json": str(summary_path),
            "grouped_validation_csv": str(folds_path),
        },
        "model_logic": "presence_probability_times_conditional_density",
        "prediction_sample_interval_ms": float(config.get("prediction_sample_interval_ms", 2.0)),
        "training_scan": scan_summary,
        "layer_summaries": layer_summaries,
        "fold_details": fold_rows,
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    summary_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step6-train] model={artifact_path}", flush=True)
    print(f"[step6-train] summary={summary_path}", flush=True)
    print(f"[step6-train] status={status} elapsed_s={summary['elapsed_seconds']:.1f}", flush=True)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
