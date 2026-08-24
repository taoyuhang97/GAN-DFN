#!/usr/bin/env python3
"""Step6A (太古界): train two-stage (presence + conditional density) models.

Model logic (per layer 上部复合层 / 太古界风化壳):
  Stage1  HistGradientBoostingClassifier  on PresenceLabel (weighted)
  Stage2  HistGradientBoostingRegressor   on DensityLabel for positive rows only
  Final density = P(presence) * clip(E[density|presence], 0, density_cap)

Validation: GroupKFold on SourceWellName (real + virtual + strong rows of a
well always stay in the same fold). 上部复合层: 11 source wells / 5 folds;
太古界风化壳: 3 source wells (301/303/321) / 3-fold leave-one-well-out.
Metrics are reported per layer, per fold and pooled (OOF).

Outputs (config.output_dir):
  step6_two_stage_density_models.joblib
  step6_two_stage_training_summary.json
  step6_two_stage_grouped_validation.csv
  step6_acceptance_summary.json   status=pass
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step6A two-stage density models (太古界).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke-test cap.")
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def imaging_density_reference(groups_root: Path, strong_wells: list[str]) -> dict[str, Any]:
    """Per-well, per-strata imaging density quantiles (reference only, no mixing)."""
    out: dict[str, Any] = {}
    for well in strong_wells:
        group_files = sorted(groups_root.glob(f"{well}_*.csv"))
        parts = []
        for group_file in group_files:
            frame = pd.read_csv(group_file, encoding="utf-8-sig")
            for column in ("Density", "HasFractureDensity"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            parts.append(frame[["StrataName", "Density", "HasFractureDensity"]])
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        well_rows: dict[str, Any] = {"rows": int(len(frame))}
        for strata, sub in frame.groupby("StrataName"):
            dens = sub["Density"].dropna()
            well_rows[str(strata)] = {
                "rows": int(len(dens)),
                "positive_rows": int((sub["HasFractureDensity"].fillna(0).astype(int) == 1).sum()),
                "quantiles": {
                    str(q): float(dens.quantile(q)) if len(dens) else None
                    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
                },
            }
        out[str(well)] = well_rows
    return out


def attach_layer_features(df: pd.DataFrame, lookups: dict[str, cKDTree], payload: Any, max_distance: float) -> pd.DataFrame:
    work = df.copy()
    xy = work[["X", "Y"]].to_numpy(dtype=np.float64)
    for code in ("top", "mid", "base"):
        distance, index = lookups[code].query(xy, k=1, p=1)
        distance = np.asarray(distance, dtype=np.float64)
        times = payload[f"{code}_t"][np.asarray(index, dtype=np.int64)]
        work[f"{code}_TIME"] = np.where(distance <= max_distance, times, np.nan)
    layer = work["LayerGroup"].astype(str)
    is_upper = layer.eq("上部复合层")
    is_crust = layer.eq("太古界风化壳")
    work["LayerTopTime"] = np.where(is_upper, work["top_TIME"], np.where(is_crust, work["mid_TIME"], np.nan))
    work["LayerBaseTime"] = np.where(is_upper, work["mid_TIME"], np.where(is_crust, work["base_TIME"], np.nan))
    work["LayerCode"] = np.where(is_upper, 1.0, np.where(is_crust, 2.0, np.nan))
    work["LayerThickness"] = work["LayerBaseTime"] - work["LayerTopTime"]
    work["TimeSinceTop"] = work["TIME"] - work["LayerTopTime"]
    work["TimeToBase"] = work["LayerBaseTime"] - work["TIME"]
    work = work[
        work["LayerThickness"].gt(0)
        & work["TimeSinceTop"].ge(-0.5)
        & work["TimeToBase"].ge(-0.5)
    ].copy()
    work["RelativeTimeInLayer"] = work["TimeSinceTop"] / work["LayerThickness"]
    work["TimeMs"] = work["TIME"]
    return work


def collect_training_data(
    input_csv: Path,
    lookups: dict[str, cKDTree],
    payload: Any,
    feature_columns: list[str],
    max_distance: float,
    chunksize: int,
    max_rows: int,
) -> dict[str, dict[str, np.ndarray]]:
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
        "SeisAmp",
        "SeisAmpM4",
        "SeisAmpM2",
        "SeisAmpP2",
        "SeisAmpP4",
        "AmpMad5",
    ]
    frame = pd.read_csv(input_csv, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    if max_rows > 0:
        frame = frame.head(int(max_rows)).copy()
    for column in (
        "X",
        "Y",
        "TIME",
        "PresenceLabel",
        "DensityLabel",
        "SampleWeight",
        "SeisAmp",
        "SeisAmpM4",
        "SeisAmpM2",
        "SeisAmpP2",
        "SeisAmpP4",
        "AmpMad5",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    work = frame[frame["LayerGroup"].astype(str).isin(["上部复合层", "太古界风化壳"])].copy()
    work = attach_layer_features(work, lookups, payload, max_distance)
    if work.empty:
        return {}, {}
    for column in feature_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    valid = work[feature_columns].notna().all(axis=1)
    valid &= work["PresenceLabel"].isin([0.0, 1.0]) & work["SampleWeight"].gt(0) & work["SourceWellName"].notna()
    positive = work["PresenceLabel"].eq(1.0)
    valid &= (~positive) | (work["DensityLabel"].notna() & work["DensityLabel"].ge(0))
    work = work.loc[valid].copy()
    group_codes: dict[str, int] = {}
    for well in work["SourceWellName"].astype(str).unique():
        if well not in group_codes:
            group_codes[well] = len(group_codes)
    work["_GroupCode"] = work["SourceWellName"].astype(str).map(group_codes).astype(np.int16)
    parts: dict[str, dict[str, list[np.ndarray]]] = {}
    for layer in ("上部复合层", "太古界风化壳"):
        layer_work = work[work["LayerGroup"].astype(str).eq(layer)]
        if layer_work.empty:
            continue
        parts[layer] = {
            "x": [layer_work[feature_columns].to_numpy(dtype=np.float32)],
            "presence": [layer_work["PresenceLabel"].to_numpy(dtype=np.int8)],
            "density": [layer_work["DensityLabel"].to_numpy(dtype=np.float32)],
            "weight": [layer_work["SampleWeight"].to_numpy(dtype=np.float32)],
            "group": [layer_work["_GroupCode"].to_numpy(dtype=np.int16)],
        }
        print(f"[step6a-train] layer={layer} rows={len(layer_work)} wells={layer_work['SourceWellName'].nunique()}", flush=True)
    data: dict[str, dict[str, np.ndarray]] = {}
    for layer, part in parts.items():
        data[layer] = {key: np.concatenate(value, axis=0) for key, value in part.items()}
    return data, group_codes


def classifier_metrics(y_true: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    out = {
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
    feature_columns: list[str],
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
        if int(positive_val_mask.sum()) > 0:
            conditional_metrics = regressor_metrics(
                density[val_idx][positive_val_mask],
                conditional[positive_val_mask],
                weight[val_idx][positive_val_mask],
            )
        else:
            conditional_metrics = {
                "rows": 0,
                "target_stats": finite_stats(np.empty(0)),
                "prediction_stats": finite_stats(np.empty(0)),
                "mae": None,
                "rmse": None,
                "r2": None,
            }
        fold_row = {
            "layer": layer,
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "train_source_well_count": int(np.unique(groups[train_idx]).size),
            "validation_source_well_count": int(np.unique(groups[val_idx]).size),
            "source_well_overlap_count": int(len(set(groups[train_idx]).intersection(set(groups[val_idx])))),
            "classifier": classifier_metrics(presence[val_idx], probability, weight[val_idx]),
            "conditional_density": conditional_metrics,
            "combined_density": regressor_metrics(target_density, final_density, weight[val_idx]),
            "elapsed_seconds": float(time.time() - started),
        }
        fold_rows.append(fold_row)
        combined_targets.append(target_density)
        combined_predictions.append(final_density)
        combined_weights.append(weight[val_idx])
        print(
            f"[step6a-train] layer={layer} fold={fold}/{split_count} "
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
    fold_ap = [row["classifier"]["average_precision"] for row in fold_rows if row["classifier"]["average_precision"] is not None]
    summary = {
        "layer": layer,
        "rows": int(len(x)),
        "positive_rows": int(positive_all.sum()),
        "positive_fraction": float(positive_all.mean()),
        "source_well_count": int(unique_groups.size),
        "validation_fold_count": int(split_count),
        "feature_columns": list(feature_columns),
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
        "feature_columns": list(feature_columns),
        "density_cap": float(density_cap),
        "training_summary": summary,
    }
    return bundle, fold_rows, summary


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "step6_two_stage_density_models.joblib"
    summary_path = output_dir / "step6_two_stage_training_summary.json"
    folds_path = output_dir / "step6_two_stage_grouped_validation.csv"
    acceptance_path = output_dir / "step6_acceptance_summary.json"
    for path in (artifact_path, summary_path, folds_path, acceptance_path):
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing Step6A training output: {path}")

    started = time.time()
    payload = np.load(config["surface_lookup_cache_npz"])
    lookups: dict[str, cKDTree] = {}
    for code in ("top", "mid", "base"):
        lookups[code] = cKDTree(np.column_stack([payload[f"{code}_x"], payload[f"{code}_y"]]))
    feature_columns = list(config["feature_columns"])
    data, group_codes = collect_training_data(
        input_csv=Path(config["unified_samples_csv"]),
        lookups=lookups,
        payload=payload,
        feature_columns=feature_columns,
        max_distance=float(config["max_horizon_match_distance_m"]),
        chunksize=int(config["training_chunksize"]),
        max_rows=int(args.max_rows),
    )
    model_bundles: dict[str, Any] = {}
    layer_summaries: dict[str, Any] = {}
    fold_rows: list[dict[str, Any]] = []
    strong_wells = list(config.get("strong_supervision_wells", ["埕北古斜405"]))
    for offset, layer in enumerate(("上部复合层", "太古界风化壳")):
        if layer not in data:
            raise RuntimeError(f"no training rows available for layer {layer}; check Step5 unified samples")
        layer_config = config["layers"][layer]
        bundle, rows, layer_summary = fit_layer(
            layer=layer,
            arrays=data[layer],
            feature_columns=feature_columns,
            classifier_params=dict(config.get("classifier", {})),
            regressor_params=dict(config.get("conditional_density_regressor", {})),
            validation_folds=int(layer_config["validation_group_folds"]),
            random_state=int(config["random_state"]) + offset * 100,
            density_cap=float(config["density_cap"]),
        )
        model_bundles[layer] = bundle
        layer_summaries[layer] = layer_summary
        fold_rows.extend(rows)

    checks = {
        "models_for_both_layers": sorted(model_bundles) == sorted(("上部复合层", "太古界风化壳")),
        "window_code_not_a_feature": "WindowCode" not in feature_columns,
        "only_raw_amplitude_derived_features": all(
            name.startswith("SeisAmp") or name.startswith("AmpMad") for name in feature_columns[:6]
        ),
        "all_folds_source_well_disjoint": all(row["source_well_overlap_count"] == 0 for row in fold_rows),
        "all_layers_have_positive_density_rows": all(
            summary["positive_rows"] > 0 for summary in layer_summaries.values()
        ),
        "crust_validation_is_3_fold_loo": layer_summaries["太古界风化壳"]["validation_fold_count"] == 3,
    }
    status = "pass" if all(checks.values()) else "fail"
    artifact = {
        "contract_version": config["version"],
        "model_logic": "presence_probability_times_conditional_density",
        "feature_columns": feature_columns,
        "attribute_columns": ["SeisAmp"],
        "models": model_bundles,
        "prediction_sample_interval_ms": 2.0,
        "training_input": str(Path(config["unified_samples_csv"]).resolve()),
        "config": config,
    }
    joblib.dump(artifact, artifact_path, compress=3)

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
        "config_path": str(args.config.resolve()),
        "training_input": str(Path(config["unified_samples_csv"]).resolve()),
        "output_dir": str(output_dir),
        "output_paths": {
            "model_joblib": str(artifact_path),
            "training_summary_json": str(summary_path),
            "grouped_validation_csv": str(folds_path),
        },
        "model_logic": "presence_probability_times_conditional_density",
        "prediction_sample_interval_ms": 2.0,
        "source_well_group_codes": group_codes,
        "imaging_density_reference": imaging_density_reference(
            Path(config["step3_groups_root"]), strong_wells
        ),
        "layer_summaries": layer_summaries,
        "fold_details": fold_rows,
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    write_json(acceptance_path, {"status": status, "summary": str(summary_path), "checks": checks})
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
