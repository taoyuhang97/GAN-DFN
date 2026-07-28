"""Train and deploy the formal GR + lateral-resistivity fracture model.

The pipeline keeps raw curve provenance, aligns same-pass GR/deep/near curves
on MD with bounded interpolation, validates by leaving one imaging well out,
and writes only eligible predictions to the downstream aggregate table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import lasio
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


CURRENT_DIR = Path(__file__).resolve().parent
TARGET_STRATA = ("沙三段", "沙四段")
INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)
IDENTITY_COLUMNS = ["SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName"]
BASE_LOG_COLUMNS = ["GR", "RDeep", "RNear"]
MODEL_FEATURE_COLUMNS = [
    "GRRobustZ",
    "LogRDeepRobustZ",
    "LogRNearRobustZ",
    "DeltaLogRRobustZ",
    "AbsDeltaLogRRobustZ",
    "NormalizedContrastRobustZ",
    "GRxDeltaLogR",
    "GRxAbsDeltaLogR",
]
TRACE_COLUMNS = [
    "GR",
    "RDeep",
    "RNear",
    "LogRDeep",
    "LogRNear",
    "DeltaLogR",
    "AbsDeltaLogR",
    "NormalizedContrast",
    *MODEL_FEATURE_COLUMNS,
]
PROVENANCE_COLUMNS = [
    "PairType",
    "MeasurementFamily",
    "GRSourceCurve",
    "DeepSourceCurve",
    "NearSourceCurve",
    "LogSourcePath",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and predict with same-pass GR plus lateral resistivity.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional isolated output override.")
    parser.add_argument(
        "--replace-existing-output",
        action="store_true",
        help="Atomically replace this GR/resistivity result directory after a successful rerun.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def clean_numeric(values: pd.Series, *, positive: bool = False) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    out = out.mask(out.abs() >= 9000.0)
    if positive:
        out = out.mask(out <= 0.0)
    return out


def read_las_table(path: Path, gr_curve: str, deep_curve: str, near_curve: str) -> pd.DataFrame:
    las = lasio.read(str(path), ignore_header_errors=True)
    raw = las.df().reset_index()
    raw = raw.rename(columns={raw.columns[0]: "MD"})
    raw.columns = [str(column).upper().strip() for column in raw.columns]
    required = {"MD", gr_curve, deep_curve, near_curve}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"LAS missing required curves {sorted(missing)}: {path}")
    out = pd.DataFrame(
        {
            "MD": clean_numeric(raw["MD"]),
            "GR": clean_numeric(raw[gr_curve]),
            "RDeep": clean_numeric(raw[deep_curve], positive=True),
            "RNear": clean_numeric(raw[near_curve], positive=True),
        }
    )
    return out.dropna(subset=["MD"]).sort_values("MD").drop_duplicates("MD", keep="last").reset_index(drop=True)


def interpolate_with_gap(
    source_md: pd.Series,
    source_value: pd.Series,
    target_md: pd.Series,
    max_gap_m: float,
) -> np.ndarray:
    x = pd.to_numeric(source_md, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(source_value, errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(target_md, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.full(len(target), np.nan)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.concatenate(([True], np.diff(x) > 1.0e-9))
    x = x[keep]
    y = y[keep]
    result = np.interp(target, x, y, left=np.nan, right=np.nan)
    raw_right = np.searchsorted(x, target, side="left")
    right = np.clip(raw_right, 0, len(x) - 1)
    left = np.clip(raw_right - 1, 0, len(x) - 1)
    exact = np.isclose(x[right], target, atol=1.0e-8, rtol=0.0) | np.isclose(
        x[left], target, atol=1.0e-8, rtol=0.0
    )
    bracket_gap = x[right] - x[left]
    supported = (target >= x[0]) & (target <= x[-1]) & (exact | (bracket_gap <= max_gap_m))
    result[~supported] = np.nan
    return result


def align_log_pass(
    target: pd.DataFrame,
    source_path: Path,
    gr_curve: str,
    deep_curve: str,
    near_curve: str,
    pair_type: str,
    measurement_family: str,
    max_gap_m: float,
) -> pd.DataFrame:
    logs = read_las_table(source_path, gr_curve, deep_curve, near_curve)
    out = target.copy()
    for column in BASE_LOG_COLUMNS:
        out[column] = interpolate_with_gap(logs["MD"], logs[column], out["DEPT"], max_gap_m)
    out["PairType"] = pair_type
    out["MeasurementFamily"] = measurement_family
    out["GRSourceCurve"] = gr_curve
    out["DeepSourceCurve"] = deep_curve
    out["NearSourceCurve"] = near_curve
    out["LogSourcePath"] = str(source_path)
    out["LogInputEligible"] = out[BASE_LOG_COLUMNS].notna().all(axis=1)
    return out


def robust_center_scale(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return 0.0, 1.0
    center = float(np.median(numeric))
    scale = float(1.4826 * np.median(np.abs(numeric - center)))
    if not np.isfinite(scale) or scale < 1.0e-9:
        scale = float(np.std(numeric))
    if not np.isfinite(scale) or scale < 1.0e-9:
        scale = 1.0
    return center, scale


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["LogRDeep"] = np.log10(pd.to_numeric(out["RDeep"], errors="coerce"))
    out["LogRNear"] = np.log10(pd.to_numeric(out["RNear"], errors="coerce"))
    out["DeltaLogR"] = out["LogRDeep"] - out["LogRNear"]
    out["AbsDeltaLogR"] = out["DeltaLogR"].abs()
    denominator = out["RDeep"] + out["RNear"]
    out["NormalizedContrast"] = (out["RDeep"] - out["RNear"]) / denominator.where(denominator.abs() > 1.0e-12)
    source_columns = ["GR", "LogRDeep", "LogRNear", "DeltaLogR", "AbsDeltaLogR", "NormalizedContrast"]
    stat_rows: list[dict[str, Any]] = []
    group_columns = ["WellName", "StrataName"]
    for keys, index in out.groupby(group_columns, dropna=False).groups.items():
        well_name, strata_name = keys
        for column in source_columns:
            center, scale = robust_center_scale(out.loc[index, column])
            out.loc[index, f"{column}RobustZ"] = (pd.to_numeric(out.loc[index, column], errors="coerce") - center) / scale
            stat_rows.append(
                {
                    "WellName": well_name,
                    "StrataName": strata_name,
                    "Feature": column,
                    "Median": center,
                    "ScaleMAD": scale,
                    "RowCount": int(len(index)),
                }
            )
    out["GRxDeltaLogR"] = out["GRRobustZ"] * out["DeltaLogRRobustZ"]
    out["GRxAbsDeltaLogR"] = out["GRRobustZ"] * out["AbsDeltaLogRRobustZ"]
    out["ModelInputEligible"] = out[MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return out, pd.DataFrame(stat_rows)


def depth_bin_weights(frame: pd.DataFrame, bin_m: float) -> np.ndarray:
    work = frame[["WellName", "StrataName", "DEPT"]].copy()
    work["DepthBin"] = np.floor(pd.to_numeric(work["DEPT"], errors="coerce") / bin_m).astype("Int64")
    counts = work.groupby(["WellName", "StrataName", "DepthBin"], dropna=False)["DEPT"].transform("size")
    weights = 1.0 / counts.clip(lower=1).to_numpy(dtype=float)
    return weights * (len(weights) / weights.sum())


def build_models(config: dict[str, Any], random_state: int) -> tuple[Any, Any]:
    stage1_estimators = int(config.get("stage1_n_estimators", 240))
    stage2_estimators = int(config.get("stage2_n_estimators", 280))
    model_jobs = max(1, int(config.get("model_n_jobs", 2)))
    classifier = RandomForestClassifier(
        n_estimators=stage1_estimators,
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=model_jobs,
    )
    regressor = RandomForestRegressor(
        n_estimators=stage2_estimators,
        max_depth=12,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=model_jobs,
    )
    return classifier, regressor


def fit_models(train: pd.DataFrame, config: dict[str, Any], random_state: int) -> tuple[Any, Any]:
    x = train[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    density = pd.to_numeric(train["Density"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    flag = (density > 0.0).astype(int)
    weights = depth_bin_weights(train, float(config.get("depth_weight_bin_m", 1.0)))
    classifier, regressor = build_models(config, random_state)
    if np.unique(flag).size < 2:
        classifier = DummyClassifier(strategy="constant", constant=int(flag[0]))
    classifier.fit(x, flag, sample_weight=weights)
    regressor.fit(x, density, sample_weight=weights)
    return classifier, regressor


def predict_raw(frame: pd.DataFrame, classifier: Any, regressor: Any) -> tuple[np.ndarray, np.ndarray]:
    x = frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    probabilities = classifier.predict_proba(x)
    classes = list(getattr(classifier, "classes_", []))
    probability = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(frame), dtype=float)
    density = np.maximum(regressor.predict(x), 0.0)
    return probability, density


def select_threshold(actual_density: pd.Series, probability: pd.Series, strategy: str) -> float:
    actual = pd.to_numeric(actual_density, errors="coerce").fillna(0.0).gt(0.0).astype(int).to_numpy()
    prob = pd.to_numeric(probability, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np.unique(actual).size < 2:
        return 0.5
    candidates = np.linspace(0.10, 0.90, 81)
    scores = np.asarray([f1_score(actual, prob >= value, zero_division=0) for value in candidates])
    if strategy == "match_oof_prevalence":
        target_ratio = float(actual.mean())
        prevalence_gap = np.asarray([abs(float((prob >= value).mean()) - target_ratio) for value in candidates])
        return float(candidates[np.lexsort((-scores, prevalence_gap))[0]])
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(candidates[min(best, key=lambda idx: abs(candidates[idx] - 0.5))])


def prediction_metrics(actual_density: pd.Series, probability: np.ndarray, density_raw: np.ndarray, threshold: float) -> dict[str, Any]:
    actual = pd.to_numeric(actual_density, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    actual_flag = (actual > 0.0).astype(int)
    pred_flag = (probability >= threshold).astype(int)
    pred_density = np.where(pred_flag > 0, density_raw, 0.0)
    metrics: dict[str, Any] = {
        "RowCount": int(len(actual)),
        "ActualPositiveRows": int(actual_flag.sum()),
        "PredictedPositiveRows": int(pred_flag.sum()),
        "Accuracy": float(accuracy_score(actual_flag, pred_flag)),
        "Precision": float(precision_score(actual_flag, pred_flag, zero_division=0)),
        "Recall": float(recall_score(actual_flag, pred_flag, zero_division=0)),
        "F1": float(f1_score(actual_flag, pred_flag, zero_division=0)),
        "DensityMAE": float(mean_absolute_error(actual, pred_density)),
        "DensityRMSE": float(np.sqrt(mean_squared_error(actual, pred_density))),
        "DensityR2": float(r2_score(actual, pred_density)) if len(actual) >= 2 else np.nan,
        "ActualPositiveRatio": float(actual_flag.mean()),
        "PredictedPositiveRatio": float(pred_flag.mean()),
    }
    if np.unique(actual_flag).size >= 2:
        metrics["PresenceROC_AUC"] = float(roc_auc_score(actual_flag, probability))
        metrics["PresenceAveragePrecision"] = float(average_precision_score(actual_flag, probability))
    else:
        metrics["PresenceROC_AUC"] = np.nan
        metrics["PresenceAveragePrecision"] = np.nan
    return metrics


def training_source_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["well_name"]): row for row in config["training_sources"]}


def build_training_table(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups_dir = Path(config["step3_groups_dir"])
    sources = training_source_map(config)
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    normalization_rows: list[pd.DataFrame] = []
    for well_name, source in sources.items():
        paths = sorted(groups_dir.glob(f"{well_name}_*.csv"))
        if not paths:
            raise RuntimeError(f"No Step3 groups found for training well {well_name}")
        raw_parts = []
        for path in paths:
            part = pd.read_csv(path)
            part["WellName"] = well_name
            part["SourceGroupFile"] = str(path)
            raw_parts.append(part)
        raw = pd.concat(raw_parts, ignore_index=True)
        raw["Step3GR"] = pd.to_numeric(raw.get("GR"), errors="coerce")
        aligned = align_log_pass(
            raw,
            Path(source["log_path"]),
            str(source["gr_curve"]),
            str(source["deep_curve"]),
            str(source["near_curve"]),
            str(source["pair_type"]),
            str(source["measurement_family"]),
            float(config.get("max_interpolation_gap_m", 0.5)),
        )
        featured, norm = engineer_features(aligned)
        usable = (
            featured["StrataName"].isin(TARGET_STRATA)
            & pd.to_numeric(featured["Density"], errors="coerce").notna()
            & featured["LogInputEligible"]
            & featured["ModelInputEligible"]
        )
        featured["TrainingEligible"] = usable
        frames.append(featured[usable].copy())
        normalization_rows.append(norm.assign(DataRole="training"))
        manifest_rows.append(
            {
                "WellName": well_name,
                "PairType": source["pair_type"],
                "MeasurementFamily": source["measurement_family"],
                "SourceRows": int(len(featured)),
                "EligibleRows": int(usable.sum()),
                "DensityPositiveRows": int(pd.to_numeric(featured.loc[usable, "Density"], errors="coerce").fillna(0.0).gt(0.0).sum()),
                "PointRows": int(pd.to_numeric(featured.loc[usable, "GT_POINT_FLAG"], errors="coerce").fillna(0.0).gt(0.0).sum()),
                "Coverage": float(usable.mean()),
                "LogSourcePath": source["log_path"],
            }
        )
    training = pd.concat(frames, ignore_index=True)
    return training, pd.DataFrame(manifest_rows), pd.concat(normalization_rows, ignore_index=True)


def point_count_scales(training: pd.DataFrame) -> dict[str, float]:
    scales: dict[str, float] = {}
    for strata_name, strata in training.groupby("StrataName"):
        total_area = 0.0
        for _, well in strata.groupby("WellName"):
            ordered = well.sort_values("DEPT")
            depth = pd.to_numeric(ordered["DEPT"], errors="coerce").to_numpy(dtype=float)
            density = pd.to_numeric(ordered["Density"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            if len(depth) >= 2:
                total_area += float(np.trapezoid(density, depth))
        point_count = int(pd.to_numeric(strata["GT_POINT_FLAG"], errors="coerce").fillna(0.0).gt(0.0).sum())
        scales[str(strata_name)] = float(point_count / total_area) if total_area > 1.0e-9 else 1.0
    return scales


def train_validate_deploy(
    training: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, float]]:
    random_state = int(config.get("random_state", 42))
    wells = sorted(training["WellName"].astype(str).unique())
    oof_parts: list[pd.DataFrame] = []
    for fold_index, holdout_well in enumerate(wells):
        for strata_name in TARGET_STRATA:
            train_part = training[(training["WellName"] != holdout_well) & (training["StrataName"] == strata_name)].copy()
            test_part = training[(training["WellName"] == holdout_well) & (training["StrataName"] == strata_name)].copy()
            if train_part.empty or test_part.empty:
                continue
            classifier, regressor = fit_models(train_part, config, random_state + fold_index)
            probability, density_raw = predict_raw(test_part, classifier, regressor)
            fold = test_part[["SampleID", "WellName", "StrataName", "DEPT", "Density", "HasFracture", "GT_POINT_FLAG"]].copy()
            fold["PredFractureProb"] = probability
            fold["PredDensityRaw"] = density_raw
            fold["HoldoutWell"] = holdout_well
            oof_parts.append(fold)
    if not oof_parts:
        raise RuntimeError("Leave-one-well-out validation produced no predictions")
    oof = pd.concat(oof_parts, ignore_index=True)
    thresholds = {
        strata: select_threshold(
            part["Density"],
            part["PredFractureProb"],
            str(config.get("threshold_strategy", "match_oof_prevalence")),
        )
        for strata, part in oof.groupby("StrataName")
    }
    metric_rows: list[dict[str, Any]] = []
    for (well_name, strata_name), part in oof.groupby(["WellName", "StrataName"]):
        threshold = float(thresholds[str(strata_name)])
        metrics = prediction_metrics(part["Density"], part["PredFractureProb"].to_numpy(), part["PredDensityRaw"].to_numpy(), threshold)
        metric_rows.append({"HoldoutWell": well_name, "StrataName": strata_name, "Threshold": threshold, **metrics})
    oof["Stage1Threshold"] = oof["StrataName"].map(thresholds)
    oof["PredHasFracture"] = (oof["PredFractureProb"] >= oof["Stage1Threshold"]).astype(int)
    oof["PredDensity"] = np.where(oof["PredHasFracture"] > 0, oof["PredDensityRaw"], 0.0)

    models: dict[str, dict[str, Any]] = {}
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for strata_index, strata_name in enumerate(TARGET_STRATA):
        train_part = training[training["StrataName"] == strata_name].copy()
        if train_part.empty:
            continue
        classifier, regressor = fit_models(train_part, config, random_state + 100 + strata_index)
        stage1_path = model_dir / f"{strata_name}_stage1_presence.joblib"
        stage2_path = model_dir / f"{strata_name}_stage2_density.joblib"
        joblib.dump(classifier, stage1_path)
        joblib.dump(regressor, stage2_path)
        models[strata_name] = {
            "stage1": classifier,
            "stage2": regressor,
            "threshold": float(thresholds[strata_name]),
            "training_rows": int(len(train_part)),
            "training_wells": sorted(train_part["WellName"].astype(str).unique()),
            "stage1_path": str(Path("models") / stage1_path.name),
            "stage2_path": str(Path("models") / stage2_path.name),
        }
    scales = point_count_scales(training)
    joblib.dump(
        {
            "models": models,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "base_log_columns": BASE_LOG_COLUMNS,
            "thresholds_by_strata": thresholds,
            "point_count_scales": scales,
            "measurement_family": config["allowed_measurement_family"],
        },
        model_dir / "gr_resistivity_lateral_bundle.joblib",
    )
    return models, thresholds, oof, pd.DataFrame(metric_rows), scales


def assign_step2_strata(main: pd.DataFrame, interval: pd.Series) -> pd.DataFrame:
    out = main.copy()
    depth = pd.to_numeric(out["DEPT"], errors="coerce")
    out["StrataName"] = pd.NA
    out.loc[(depth >= float(interval["T4_DEPT"])) & (depth < float(interval["T6_DEPT"])), "StrataName"] = "沙三段"
    out.loc[(depth >= float(interval["T6_DEPT"])) & (depth <= float(interval["T7_DEPT"])), "StrataName"] = "沙四段"
    return out


def gr_curve_from_inventory(curves: str) -> str | None:
    available = set(str(curves).split(";"))
    return next((curve for curve in ("GR", "GR1", "GRSL") if curve in available), None)


def align_inventory_candidate(
    main: pd.DataFrame,
    row: pd.Series,
    max_gap_m: float,
) -> pd.DataFrame | None:
    gr_curve = gr_curve_from_inventory(str(row["Curves"]))
    if gr_curve is None:
        return None
    return align_log_pass(
        main,
        Path(str(row["SourcePath"])),
        gr_curve,
        str(row["DeepCurve"]),
        str(row["NearCurve"]),
        str(row["PairType"]),
        str(row["MeasurementFamily"]),
        max_gap_m,
    )


def select_best_pass(
    main: pd.DataFrame,
    candidates: pd.DataFrame,
    max_gap_m: float,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    best: pd.DataFrame | None = None
    best_info: dict[str, Any] = {}
    best_count = -1
    for _, row in candidates.iterrows():
        try:
            aligned = align_inventory_candidate(main, row, max_gap_m)
        except Exception as exc:
            continue
        if aligned is None:
            continue
        count = int(aligned["LogInputEligible"].sum())
        if count > best_count:
            best = aligned
            best_count = count
            best_info = {
                "PairType": row["PairType"],
                "MeasurementFamily": row["MeasurementFamily"],
                "GRSourceCurve": gr_curve_from_inventory(str(row["Curves"])),
                "DeepSourceCurve": row["DeepCurve"],
                "NearSourceCurve": row["NearCurve"],
                "LogSourcePath": row["SourcePath"],
                "AlignedLogRows": count,
            }
    return best, best_info


def cumulative_area(depth: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    out = np.zeros(len(depth), dtype=float)
    if len(depth) < 2:
        return out
    dx = np.maximum(np.diff(depth), 0.0)
    out[1:] = np.cumsum(0.5 * (intensity[1:] + intensity[:-1]) * dx)
    return out


def refine_points(predicted: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    columns = [
        "SourceSampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName",
        "Density", "HasFracture", "PredFractureProb", "SegmentStartDEPT", "SegmentEndDEPT",
        "PointOrdinalInSegment", "PointSourceLogic",
    ]
    rows: list[dict[str, Any]] = []
    for (well_name, strata_name), part in predicted.groupby(["WellName", "StrataName"]):
        part = part.sort_values("DEPT").reset_index(drop=True)
        depth = pd.to_numeric(part["DEPT"], errors="coerce").to_numpy(dtype=float)
        positive = pd.to_numeric(part["HasFracture"], errors="coerce").fillna(0).gt(0).to_numpy()
        median_step = float(np.nanmedian(np.diff(depth))) if len(depth) > 1 else 0.125
        breaks = np.ones(len(part), dtype=bool)
        if len(part) > 1:
            breaks[1:] = (~positive[:-1]) | (np.diff(depth) > max(0.5, 2.5 * median_step))
        starts = np.flatnonzero(positive & breaks)
        for start in starts:
            end = start
            while end + 1 < len(part) and positive[end + 1] and depth[end + 1] - depth[end] <= max(0.5, 2.5 * median_step):
                end += 1
            segment = part.iloc[start : end + 1]
            seg_depth = depth[start : end + 1]
            density = pd.to_numeric(segment["Density"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            prob = pd.to_numeric(segment["PredFractureProb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            intensity = np.maximum(density * prob, 0.0)
            cum = cumulative_area(seg_depth, intensity)
            scaled_count = float(scales.get(str(strata_name), 1.0)) * float(cum[-1] if len(cum) else 0.0)
            count = min(len(segment), max(1, int(round(scaled_count))))
            if len(segment) == 1 or cum[-1] <= 0.0:
                selected = [0]
            else:
                targets = (np.arange(count) + 0.5) * cum[-1] / count
                selected = np.searchsorted(cum, targets, side="left").clip(0, len(segment) - 1).tolist()
            for ordinal, local_index in enumerate(selected, start=1):
                source = segment.iloc[int(local_index)]
                rows.append(
                    {
                        "SourceSampleID": source["SampleID"],
                        "WellName": well_name,
                        "X": source.get("X", np.nan),
                        "Y": source.get("Y", np.nan),
                        "TIME": source.get("TIME", np.nan),
                        "TVD": source.get("TVD", np.nan),
                        "DEPT": source["DEPT"],
                        "StrataName": strata_name,
                        "Density": source["Density"],
                        "HasFracture": 1,
                        "PredFractureProb": source["PredFractureProb"],
                        "SegmentStartDEPT": float(seg_depth[0]),
                        "SegmentEndDEPT": float(seg_depth[-1]),
                        "PointOrdinalInSegment": ordinal,
                        "PointSourceLogic": "OOF-calibrated density integral within predicted segment",
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def predict_conventional_wells(
    config: dict[str, Any],
    models: dict[str, dict[str, Any]],
    point_scales: dict[str, float],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step2_root = Path(config["step2_output_root"])
    summary = pd.read_csv(step2_root / "real_well_t4_t7_summary.csv")
    inventory = pd.read_csv(config["resistivity_pair_inventory"])
    if "IsDuplicateContentCopy" in inventory.columns:
        inventory = inventory[~inventory["IsDuplicateContentCopy"].fillna(False)].copy()
    family = str(config["allowed_measurement_family"])
    family_inventory = inventory[inventory["MeasurementFamily"].astype(str).eq(family)].copy()
    max_gap_m = float(config.get("max_interpolation_gap_m", 0.5))
    well_root = output_dir / "real_well_predictions"
    well_root.mkdir(parents=True, exist_ok=True)
    downstream_parts: list[pd.DataFrame] = []
    point_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    for _, summary_row in summary.iterrows():
        well_name = str(summary_row["WellName"])
        status = str(summary_row["Status"])
        if status != "ok":
            coverage_rows.append({"WellName": well_name, "Status": f"step2_{status}", "TotalRows": 0, "EligibleRows": 0, "PredictedRows": 0})
            continue
        print(f"[Step4] scanning conventional well {well_name}", flush=True)
        main_path = Path(str(summary_row["MainCsv"]))
        interval_path = Path(str(summary_row["IntervalCsv"]))
        main = pd.read_csv(main_path)
        interval = pd.read_csv(interval_path).iloc[0]
        main = assign_step2_strata(main, interval)
        candidates = family_inventory[family_inventory["WellName"].astype(str).eq(well_name)]
        if candidates.empty:
            other_families = sorted(inventory.loc[inventory["WellName"].astype(str).eq(well_name), "MeasurementFamily"].dropna().astype(str).unique())
            reason = "unsupported_measurement_family" if other_families else "no_reviewed_same_pass_pair"
            coverage_rows.append(
                {
                    "WellName": well_name, "Status": reason, "TotalRows": int(len(main)), "EligibleRows": 0,
                    "PredictedRows": 0, "OtherMeasurementFamilies": ";".join(other_families),
                }
            )
            continue
        supported_gr_candidates = candidates[candidates["Curves"].map(gr_curve_from_inventory).notna()].copy()
        if supported_gr_candidates.empty:
            available_gamma_curves = sorted(
                {
                    curve
                    for curves in candidates["Curves"].astype(str)
                    for curve in curves.split(";")
                    if "GR" in curve
                }
            )
            coverage_rows.append(
                {
                    "WellName": well_name,
                    "Status": "missing_supported_gr_curve",
                    "TotalRows": int(len(main)),
                    "EligibleRows": 0,
                    "PredictedRows": 0,
                    "AvailableGammaCurves": ";".join(available_gamma_curves),
                }
            )
            continue
        aligned, pass_info = select_best_pass(main, supported_gr_candidates, max_gap_m)
        if aligned is None:
            coverage_rows.append({"WellName": well_name, "Status": "pair_read_failed", "TotalRows": int(len(main)), "EligibleRows": 0, "PredictedRows": 0})
            continue
        featured, normalization = engineer_features(aligned)
        eligible = featured["StrataName"].isin(TARGET_STRATA) & featured["LogInputEligible"] & featured["ModelInputEligible"]
        featured["PredictionStatus"] = np.where(eligible, "eligible_not_predicted", "missing_required_same_pass_logs")
        featured["PredFractureProb"] = np.nan
        featured["PredDensityRaw"] = np.nan
        featured["PredDensity"] = np.nan
        featured["PredHasFracture"] = pd.Series(pd.NA, index=featured.index, dtype="Int64")
        featured["Stage1Threshold"] = np.nan
        for strata_name in TARGET_STRATA:
            mask = eligible & featured["StrataName"].eq(strata_name)
            model = models.get(strata_name)
            if not mask.any() or model is None:
                continue
            probability, density_raw = predict_raw(featured.loc[mask], model["stage1"], model["stage2"])
            threshold = float(model["threshold"])
            flag = (probability >= threshold).astype(int)
            density = np.where(flag > 0, density_raw, 0.0)
            featured.loc[mask, "PredFractureProb"] = probability
            featured.loc[mask, "PredDensityRaw"] = density_raw
            featured.loc[mask, "PredDensity"] = density
            featured.loc[mask, "PredHasFracture"] = flag
            featured.loc[mask, "Stage1Threshold"] = threshold
            featured.loc[mask, "PredictionStatus"] = "predicted"
        featured["Density"] = featured["PredDensity"]
        featured["HasFracture"] = featured["PredHasFracture"]
        predicted = featured[featured["PredictionStatus"].eq("predicted")].copy()
        if predicted.empty:
            coverage_rows.append(
                {
                    "WellName": well_name,
                    "Status": "no_valid_rows_in_t4_t7",
                    "TotalRows": int(len(featured)),
                    "EligibleRows": 0,
                    "PredictedRows": 0,
                    "PredictionCoverage": 0.0,
                    **pass_info,
                }
            )
            continue
        predicted["HasFracture"] = pd.to_numeric(predicted["HasFracture"], errors="coerce").astype(int)
        points = refine_points(predicted, point_scales)
        out_dir = well_root / well_name
        out_dir.mkdir(parents=True, exist_ok=True)
        detail_columns = [*IDENTITY_COLUMNS, *TRACE_COLUMNS, *PROVENANCE_COLUMNS, "LogInputEligible", "ModelInputEligible", "Density", "HasFracture", "PredFractureProb", "PredDensityRaw", "PredDensity", "PredHasFracture", "Stage1Threshold", "PredictionStatus"]
        downstream_columns = [*IDENTITY_COLUMNS, *TRACE_COLUMNS, *PROVENANCE_COLUMNS, "Density", "HasFracture", "PredFractureProb", "PredDensityRaw", "PredDensity", "PredHasFracture", "Stage1Threshold", "PredictionStatus"]
        featured[[column for column in detail_columns if column in featured.columns]].to_csv(
            out_dir / f"{well_name}_t4_t7_prediction_coverage_detail.csv", index=False, encoding="utf-8-sig"
        )
        predicted[[column for column in downstream_columns if column in predicted.columns]].to_csv(
            out_dir / f"{well_name}_t4_t7_density_curve.csv", index=False, encoding="utf-8-sig"
        )
        normalization.to_csv(out_dir / f"{well_name}_normalization_stats.csv", index=False, encoding="utf-8-sig")
        points.to_csv(out_dir / f"{well_name}_t4_t7_fracture_points.csv", index=False, encoding="utf-8-sig")
        coverage = float(len(predicted) / len(featured)) if len(featured) else 0.0
        well_status = "predicted_full" if np.isclose(coverage, 1.0) else "predicted_partial"
        coverage_rows.append(
            {
                "WellName": well_name,
                "Status": well_status,
                "TotalRows": int(len(featured)),
                "EligibleRows": int(eligible.sum()),
                "PredictedRows": int(len(predicted)),
                "PredictionCoverage": coverage,
                "PredictedPositiveRows": int(pd.to_numeric(predicted["HasFracture"], errors="coerce").fillna(0).sum()),
                "PredictedPointCount": int(len(points)),
                **pass_info,
            }
        )
        downstream_parts.append(predicted[[column for column in downstream_columns if column in predicted.columns]])
        point_parts.append(points)
        print(
            f"[Step4] predicted {well_name}: {len(predicted)}/{len(featured)} rows, {len(points)} refined points",
            flush=True,
        )
    downstream = pd.concat(downstream_parts, ignore_index=True) if downstream_parts else pd.DataFrame()
    points = pd.concat(point_parts, ignore_index=True) if point_parts else refine_points(pd.DataFrame(columns=[*IDENTITY_COLUMNS, "Density", "HasFracture", "PredFractureProb"]), point_scales)
    coverage = pd.DataFrame(coverage_rows)
    downstream.to_csv(output_dir / "all_wells_t4_t7_density_prediction.csv", index=False, encoding="utf-8-sig")
    points.to_csv(output_dir / "all_wells_t4_t7_fracture_points.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "all_wells_prediction_coverage_manifest.csv", index=False, encoding="utf-8-sig")
    return downstream, points, coverage


def validate_delivery(
    config: dict[str, Any],
    training: pd.DataFrame,
    oof: pd.DataFrame,
    downstream: pd.DataFrame,
    coverage: pd.DataFrame,
    staging_dir: Path,
    published_output_dir: Path,
) -> dict[str, Any]:
    required = {"SampleID", "WellName", "X", "Y", "TIME", "Density", "HasFracture", "StrataName", *BASE_LOG_COLUMNS}
    checks = {
        "new_output_directory": published_output_dir.name == "formal_gr_resistivity_lateral_v1",
        "legacy_output_directory_not_used": published_output_dir.name != "formal_well_expert_library",
        "three_training_wells": training["WellName"].nunique() == 3,
        "training_family_allowed": training["MeasurementFamily"].eq(config["allowed_measurement_family"]).all(),
        "training_same_pass_inputs_complete": training[BASE_LOG_COLUMNS].notna().all(axis=1).all(),
        "loo_covers_all_training_wells": set(oof["HoldoutWell"].astype(str)) == set(training["WellName"].astype(str)),
        "downstream_required_columns": required.issubset(downstream.columns),
        "downstream_density_nonnull": downstream["Density"].notna().all(),
        "downstream_contains_predicted_rows_only": downstream["PredictionStatus"].eq("predicted").all(),
        "all_eligible_rows_predicted": int(coverage["EligibleRows"].fillna(0).sum()) == len(downstream),
        "no_ineligible_rows_predicted": len(downstream) == int(coverage["PredictedRows"].fillna(0).sum()),
        "prediction_family_allowed": downstream["MeasurementFamily"].eq(config["allowed_measurement_family"]).all(),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "output_dir": str(published_output_dir),
        "model_input_contract": "same-pass GR + RDeep + RNear; lateral-resistivity family only",
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "training_wells": sorted(training["WellName"].astype(str).unique()),
        "training_rows": int(len(training)),
        "training_density_positive_rows": int(pd.to_numeric(training["Density"], errors="coerce").fillna(0).gt(0).sum()),
        "loo_rows": int(len(oof)),
        "predicted_well_count": int(coverage["Status"].astype(str).str.startswith("predicted_").sum()),
        "predicted_rows": int(len(downstream)),
        "unpredicted_rows_in_predicted_well_details": int(
            (coverage["TotalRows"].fillna(0) - coverage["PredictedRows"].fillna(0)).clip(lower=0).sum()
        ),
        "checks": checks,
    }
    write_json(staging_dir / "step4_gr_resistivity_acceptance_summary.json", summary)
    return summary


def run(config: dict[str, Any], output_dir: Path, replace_existing: bool = False) -> dict[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not replace_existing:
        raise RuntimeError(f"Refusing to mix results into existing output directory: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_staging_", dir=output_dir.parent))
    try:
        training, training_manifest, training_norm = build_training_table(config)
        print(f"[Step4] prepared {len(training)} eligible training rows from {training['WellName'].nunique()} wells", flush=True)
        training.to_csv(staging / "training_feature_table.csv", index=False, encoding="utf-8-sig")
        training_manifest.to_csv(staging / "training_input_manifest.csv", index=False, encoding="utf-8-sig")
        training_norm.to_csv(staging / "training_normalization_stats.csv", index=False, encoding="utf-8-sig")
        models, thresholds, oof, metrics, point_scales = train_validate_deploy(training, config, staging)
        print(f"[Step4] completed leave-one-well-out validation and trained {len(models)} strata models", flush=True)
        oof.to_csv(staging / "leave_one_well_out_predictions.csv", index=False, encoding="utf-8-sig")
        metrics.to_csv(staging / "leave_one_well_out_metrics.csv", index=False, encoding="utf-8-sig")
        write_json(staging / "model_thresholds_and_point_scales.json", {"thresholds_by_strata": thresholds, "point_count_scales": point_scales})
        downstream, points, coverage = predict_conventional_wells(config, models, point_scales, staging)
        acceptance = validate_delivery(config, training, oof, downstream, coverage, staging, output_dir)
        if acceptance["status"] != "pass":
            raise RuntimeError(f"Step4 delivery validation failed: {acceptance['checks']}")
        write_json(staging / "run_config_snapshot.json", config)
        previous_dir: Path | None = None
        if output_dir.exists():
            previous_dir = output_dir.parent / f".{output_dir.name}_previous"
            if previous_dir.exists():
                raise RuntimeError(f"Previous-output backup path already exists: {previous_dir}")
            output_dir.rename(previous_dir)
        try:
            staging.rename(output_dir)
        except Exception:
            if previous_dir is not None and previous_dir.exists() and not output_dir.exists():
                previous_dir.rename(output_dir)
            raise
        if previous_dir is not None:
            shutil.rmtree(previous_dir)
        return acceptance
    except Exception as exc:
        print(f"[Step4] failed before publication: {type(exc).__name__}: {exc}", flush=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    summary = run(config, output_dir, replace_existing=args.replace_existing_output)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
