"""Train and deploy six Step4 GR/resistivity experts.

The library contains one expert for each StrataName x PairType combination:
沙三/沙四 x LLD_LLS/RD_RS/RILD_RILM. LLD and RD experts use direct
imaging supervision. RILD experts learn from RD teachers on reviewed bridge
wells. All six experts are published and routed only by strata and pair type.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
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
from xgboost import XGBClassifier, XGBRegressor

import train_and_predict_gr_resistivity_strata_library as v2


TARGET_STRATA = ("沙三段", "沙四段")
PAIR_TYPES = ("LLD_LLS", "RD_RS", "RILD_RILM")
DIRECT_PAIR_TYPES = ("LLD_LLS", "RD_RS")
MERGE_PAIR_PRIORITY = ("RD_RS", "LLD_LLS", "RILD_RILM")

FEATURE_COLUMNS = [
    "GRRobustZ",
    "RDeepRobustZ",
    "RReferenceRobustZ",
    "GRRank",
    "RDeepRank",
    "RReferenceRank",
    "RDeepSignedLog",
    "RReferenceSignedLog",
    "LogRDeepPositive",
    "LogRReferencePositive",
    "GRGradientZ",
    "RDeepGradientZ",
    "RReferenceGradientZ",
    "Contrast",
    "ContrastRobustZ",
    "NormalizedContrast",
    "GRxDeepResidual",
    "GRxReferenceResidual",
    "GRxNormalizedContrast",
    "DeepNonPositiveFlag",
    "ReferenceNonPositiveFlag",
]
for _prefix in ("GR", "RDeep", "RReference"):
    for _window in (0.5, 1.0, 3.0, 5.0):
        FEATURE_COLUMNS.append(f"{_prefix}Residual{str(_window).replace('.', 'p')}MZ")
    for _window in (1.0, 3.0, 5.0):
        FEATURE_COLUMNS.append(f"{_prefix}MAD{int(_window)}MZ")
for _window in (1.0, 3.0, 5.0):
    FEATURE_COLUMNS.append(f"ContrastResidual{int(_window)}MZ")

OUTPUT_COLUMNS = [
    "SampleID",
    "SourceSampleID",
    "WellName",
    "X",
    "Y",
    "TIME",
    "TVD",
    "DEPT",
    "StrataName",
    "SupportSegmentID",
    "SegmentStartDEPT",
    "SegmentEndDEPT",
    "InputPairType",
    "ReferenceType",
    "ExpertID",
    "ModelID",
    "PredFractureProb",
    "PredConditionalDensity",
    "PredDensity",
    "PredHasFracture",
    "Stage1Threshold",
    "Density",
    "HasFracture",
    "PredictionValid",
]
MERGED_OUTPUT_COLUMNS = OUTPUT_COLUMNS + [
    "AvailableExpertCount",
    "AvailablePairTypes",
    "SelectionPriority",
    "MergeRule",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Step4 six-expert GR/resistivity library.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace-existing-output", action="store_true")
    parser.add_argument("--train-only", action="store_true")
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


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def robust_center_scale(values: pd.Series) -> tuple[float, float]:
    clean = numeric(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return 0.0, 1.0
    center = float(np.median(clean))
    scale = float(1.4826 * np.median(np.abs(clean - center)))
    if not np.isfinite(scale) or scale < 1.0e-9:
        scale = float(np.std(clean))
    if not np.isfinite(scale) or scale < 1.0e-9:
        scale = 1.0
    return center, scale


def robust_z(values: pd.Series) -> pd.Series:
    center, scale = robust_center_scale(values)
    return (numeric(values) - center) / scale


def signed_log1p(values: pd.Series) -> pd.Series:
    values = numeric(values)
    return np.sign(values) * np.log1p(values.abs())


def positive_log10(values: pd.Series) -> pd.Series:
    values = numeric(values)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    positive = values.gt(0.0)
    out.loc[positive] = np.log10(values.loc[positive])
    return out


def rolling_row_count(depth: pd.Series, window_m: float) -> int:
    spacing = numeric(depth).diff().abs().replace(0.0, np.nan).dropna()
    median_step = float(spacing.median()) if not spacing.empty else window_m
    rows = max(3, int(round(window_m / max(median_step, 1.0e-6))))
    return rows + 1 if rows % 2 == 0 else rows


def rolling_residual(values: pd.Series, depth: pd.Series, window_m: float) -> pd.Series:
    values = numeric(values)
    rows = rolling_row_count(depth, window_m)
    baseline = values.rolling(rows, center=True, min_periods=max(2, rows // 3)).median()
    return values - baseline


def rolling_mad(values: pd.Series, depth: pd.Series, window_m: float) -> pd.Series:
    values = numeric(values)
    rows = rolling_row_count(depth, window_m)
    median = values.rolling(rows, center=True, min_periods=max(2, rows // 3)).median()
    deviation = (values - median).abs()
    return deviation.rolling(rows, center=True, min_periods=max(2, rows // 3)).median()


def depth_gradient(values: pd.Series, depth: pd.Series) -> pd.Series:
    value_array = numeric(values).to_numpy(dtype=float)
    depth_array = numeric(depth).to_numpy(dtype=float)
    if len(value_array) < 2:
        return pd.Series(np.nan, index=values.index, dtype=float)
    gradient = np.gradient(value_array, depth_array)
    gradient[~np.isfinite(gradient)] = np.nan
    return pd.Series(gradient, index=values.index)


def safe_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    parts: list[pd.DataFrame] = []
    stats: list[dict[str, Any]] = []
    for (well_name, strata_name, pair_type), group in frame.groupby(
        ["WellName", "StrataName", "PairType"], dropna=False
    ):
        out = group.sort_values("DEPT").copy()
        out["RDeepSignedLog"] = signed_log1p(out["RDeep"])
        out["RReferenceSignedLog"] = signed_log1p(out["RReference"])
        out["LogRDeepPositive"] = positive_log10(out["RDeep"])
        out["LogRReferencePositive"] = positive_log10(out["RReference"])
        out["GRRobustZ"] = robust_z(out["GR"])
        out["RDeepRobustZ"] = robust_z(out["RDeepSignedLog"])
        out["RReferenceRobustZ"] = robust_z(out["RReferenceSignedLog"])
        out["GRRank"] = numeric(out["GR"]).rank(method="average", pct=True)
        out["RDeepRank"] = numeric(out["RDeep"]).rank(method="average", pct=True)
        out["RReferenceRank"] = numeric(out["RReference"]).rank(method="average", pct=True)

        transformed = {
            "GR": numeric(out["GR"]),
            "RDeep": out["RDeepSignedLog"],
            "RReference": out["RReferenceSignedLog"],
        }
        for prefix, values in transformed.items():
            out[f"{prefix}GradientZ"] = robust_z(depth_gradient(values, out["DEPT"]))
            for window_m in (0.5, 1.0, 3.0, 5.0):
                suffix = str(window_m).replace(".", "p")
                out[f"{prefix}Residual{suffix}MZ"] = robust_z(
                    rolling_residual(values, out["DEPT"], window_m)
                )
            for window_m in (1.0, 3.0, 5.0):
                out[f"{prefix}MAD{int(window_m)}MZ"] = robust_z(
                    rolling_mad(values, out["DEPT"], window_m)
                )

        out["Contrast"] = out["RDeepSignedLog"] - out["RReferenceSignedLog"]
        out["ContrastRobustZ"] = robust_z(out["Contrast"])
        denominator = numeric(out["RDeep"]).abs() + numeric(out["RReference"]).abs()
        out["NormalizedContrast"] = (numeric(out["RDeep"]) - numeric(out["RReference"])) / denominator.where(
            denominator.gt(1.0e-12)
        )
        for window_m in (1.0, 3.0, 5.0):
            out[f"ContrastResidual{int(window_m)}MZ"] = robust_z(
                rolling_residual(out["Contrast"], out["DEPT"], window_m)
            )
        out["GRxDeepResidual"] = out["GRRobustZ"] * out["RDeepResidual1p0MZ"]
        out["GRxReferenceResidual"] = out["GRRobustZ"] * out["RReferenceResidual1p0MZ"]
        out["GRxNormalizedContrast"] = out["GRRobustZ"] * out["NormalizedContrast"]
        out["DeepNonPositiveFlag"] = numeric(out["RDeep"]).le(0.0).astype(int)
        out["ReferenceNonPositiveFlag"] = numeric(out["RReference"]).le(0.0).astype(int)
        out["ModelInputEligible"] = out[["GR", "RDeep", "RReference"]].notna().all(axis=1)

        for column in ("GR", "RDeep", "RReference", "RDeepSignedLog", "RReferenceSignedLog", "Contrast"):
            center, scale = robust_center_scale(out[column])
            stats.append(
                {
                    "WellName": well_name,
                    "StrataName": strata_name,
                    "PairType": pair_type,
                    "Feature": column,
                    "Median": center,
                    "ScaleMAD": scale,
                    "RowCount": int(len(out)),
                }
            )
        parts.append(out)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def nearest_point_distance(depth: pd.Series, point_mask: pd.Series) -> np.ndarray:
    values = numeric(depth).to_numpy(dtype=float)
    points = np.sort(values[point_mask.to_numpy(dtype=bool) & np.isfinite(values)])
    if points.size == 0:
        return np.full(len(values), np.inf, dtype=float)
    position = np.searchsorted(points, values)
    left = np.where(position > 0, np.abs(values - points[np.maximum(position - 1, 0)]), np.inf)
    right_index = np.minimum(position, len(points) - 1)
    right = np.where(position < len(points), np.abs(values - points[right_index]), np.inf)
    return np.minimum(left, right)


def assign_presence_labels(training: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    quantile = float(config["clear_positive_density_quantile"])
    point_radius = float(config["clear_positive_point_radius_m"])
    exclusion = float(config["clear_background_point_exclusion_m"])
    negative_max = float(config.get("clear_background_density_max", 0.0))
    parts = []
    for (_, _), group in training.groupby(["PairType", "StrataName"], sort=False):
        out = group.copy()
        density = numeric(out["Density"]).fillna(0.0).clip(lower=0.0)
        positive_density = density[density.gt(0.0)]
        density_threshold = float(positive_density.quantile(quantile)) if not positive_density.empty else 0.0
        point_mask = numeric(out.get("GT_POINT_FLAG", pd.Series(0, index=out.index))).fillna(0).gt(0)
        distances = np.full(len(out), np.inf, dtype=float)
        for _, well_index in out.groupby("WellName").groups.items():
            local = out.loc[well_index]
            local_points = numeric(local.get("GT_POINT_FLAG", pd.Series(0, index=local.index))).fillna(0).gt(0)
            distances[out.index.get_indexer(well_index)] = nearest_point_distance(local["DEPT"], local_points)
        clear_positive = density.ge(density_threshold) | point_mask | (distances <= point_radius)
        clear_background = density.le(negative_max) & (distances > exclusion) & ~clear_positive
        out["PresenceLabel"] = np.nan
        out.loc[clear_background, "PresenceLabel"] = 0.0
        out.loc[clear_positive, "PresenceLabel"] = 1.0
        out["PresenceLabelKind"] = "uncertain"
        out.loc[clear_background, "PresenceLabelKind"] = "clear_background"
        out.loc[clear_positive, "PresenceLabelKind"] = "clear_fracture"
        out["NearestGTPointDistanceM"] = distances
        out["ClearPositiveDensityThreshold"] = density_threshold
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def load_training_table(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups_dir = Path(config["step3_groups_dir"])
    frames: list[pd.DataFrame] = []
    stats: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for source in config["training_sources"]:
        well_name = str(source["well_name"])
        pair_type = str(source["pair_type"])
        paths = sorted(groups_dir.glob(f"{well_name}_*.csv"))
        if not paths:
            raise RuntimeError(f"no Step3 groups for {well_name}")
        base = pd.concat([pd.read_csv(path).assign(WellName=well_name, SourceGroupFile=str(path)) for path in paths])
        enrichment_file = v2.find_training_enrichment(config, well_name)
        selected = v2.select_pair_rows(base, pd.read_csv(enrichment_file), pair_type)
        featured, feature_stats = engineer_features(selected)
        usable = (
            featured["StrataName"].isin(TARGET_STRATA)
            & numeric(featured["Density"]).notna()
            & featured["ModelInputEligible"]
        )
        featured = featured.loc[usable].copy()
        featured["HasFracture"] = numeric(featured["Density"]).fillna(0).gt(0).astype(int)
        frames.append(featured)
        stats.append(feature_stats.assign(DataRole="training"))
        source_rows.append(
            {
                "WellName": well_name,
                "PairType": pair_type,
                "EnrichmentPath": str(enrichment_file),
                "Step3Rows": int(len(base)),
                "EligibleRows": int(len(featured)),
                "DensityPositiveRows": int(featured["HasFracture"].sum()),
            }
        )
    training = assign_presence_labels(pd.concat(frames, ignore_index=True), config)
    label_counts = (
        training.groupby(["WellName", "PairType", "StrataName", "PresenceLabelKind"]).size().unstack(fill_value=0)
    )
    manifest = pd.DataFrame(source_rows)
    if not label_counts.empty:
        summary = label_counts.groupby(["WellName", "PairType"]).sum().reset_index()
        manifest = manifest.merge(summary, on=["WellName", "PairType"], how="left")
    normalization = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()
    return training, manifest, normalization


def equalized_depth_weights(
    frame: pd.DataFrame,
    bin_m: float,
    label_column: str | None = None,
) -> np.ndarray:
    work = frame[["WellName", "StrataName", "DEPT"]].copy()
    work["DepthBin"] = np.floor(numeric(work["DEPT"]) / bin_m).astype("Int64")
    count = work.groupby(["WellName", "StrataName", "DepthBin"], dropna=False)["DEPT"].transform("size")
    weights = 1.0 / count.clip(lower=1).to_numpy(dtype=float)
    well_total = pd.Series(weights, index=frame.index).groupby(frame["WellName"].astype(str)).transform("sum")
    weights = weights / np.maximum(well_total.to_numpy(dtype=float), 1.0e-12)
    if label_column is not None:
        labels = numeric(frame[label_column]).fillna(0).astype(int).to_numpy()
        positive = max(int(labels.sum()), 1)
        negative = max(int(len(labels) - labels.sum()), 1)
        weights = weights * np.where(labels > 0, negative / positive, 1.0)
    return weights * (len(weights) / max(weights.sum(), 1.0e-12))


def xgb_classifier(config: dict[str, Any], random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=int(config.get("xgb_n_jobs", 2)),
        random_state=random_state,
        **dict(config["stage1_params"]),
    )


def xgb_regressor(config: dict[str, Any], random_state: int, transfer: bool = False) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=int(config.get("xgb_n_jobs", 2)),
        random_state=random_state,
        **dict(config["transfer_params"] if transfer else config["stage2_params"]),
    )


def fit_direct(frame: pd.DataFrame, config: dict[str, Any], random_state: int) -> tuple[Any, Any]:
    classified = frame[numeric(frame["PresenceLabel"]).notna()].copy()
    labels = numeric(classified["PresenceLabel"]).astype(int)
    if classified.empty:
        raise RuntimeError("expert has no clear Stage-1 labels")
    if labels.nunique() < 2:
        classifier: Any = DummyClassifier(strategy="constant", constant=int(labels.iloc[0]))
    else:
        classifier = xgb_classifier(config, random_state)
    classifier.fit(
        safe_features(classified),
        labels,
        sample_weight=equalized_depth_weights(
            classified, float(config["depth_weight_bin_m"]), label_column="PresenceLabel"
        ),
    )

    positive = frame[numeric(frame["Density"]).fillna(0.0).gt(0.0)].copy()
    if positive.empty:
        regressor: Any = DummyRegressor(strategy="constant", constant=0.0)
        regressor.fit(np.zeros((1, len(FEATURE_COLUMNS))), [0.0])
    else:
        regressor = xgb_regressor(config, random_state + 1)
        regressor.fit(
            safe_features(positive),
            np.log1p(numeric(positive["Density"]).clip(lower=0.0)),
            sample_weight=equalized_depth_weights(positive, float(config["depth_weight_bin_m"])),
        )
    return classifier, regressor


def predict_direct(frame: pd.DataFrame, classifier: Any, regressor: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = classifier.predict_proba(safe_features(frame))
    classes = list(getattr(classifier, "classes_", []))
    probability = matrix[:, classes.index(1)] if 1 in classes else np.zeros(len(frame), dtype=float)
    conditional = np.maximum(np.expm1(regressor.predict(safe_features(frame))), 0.0)
    return np.clip(probability, 0.0, 1.0), conditional


def select_threshold(labels: Iterable[float], probability: Iterable[float], strategy: str) -> float:
    actual = pd.to_numeric(pd.Series(labels), errors="coerce")
    prob = pd.to_numeric(pd.Series(probability), errors="coerce")
    usable = actual.notna() & prob.notna()
    actual = actual[usable].astype(int).to_numpy()
    prob = prob[usable].to_numpy(dtype=float)
    if np.unique(actual).size < 2:
        return 0.5
    candidates = np.linspace(0.05, 0.95, 91)
    scores = np.asarray([f1_score(actual, prob >= value, zero_division=0) for value in candidates])
    if strategy == "match_oof_prevalence":
        gap = np.asarray([abs(float((prob >= value).mean()) - float(actual.mean())) for value in candidates])
        return float(candidates[np.lexsort((-scores, gap))[0]])
    return float(candidates[int(np.argmax(scores))])


def validation_splits(frame: pd.DataFrame, blocks: int) -> list[tuple[str, str, pd.Index, pd.Index]]:
    wells = sorted(frame["WellName"].astype(str).unique())
    if len(wells) >= 2:
        return [
            ("leave_one_well_out", well, frame.index[frame["WellName"].astype(str).ne(well)], frame.index[frame["WellName"].astype(str).eq(well)])
            for well in wells
        ]
    ordered = frame.sort_values("DEPT")
    chunks = [chunk for chunk in np.array_split(ordered.index.to_numpy(), min(blocks, len(ordered))) if len(chunk)]
    splits = []
    for index, chunk in enumerate(chunks, start=1):
        test_index = pd.Index(chunk)
        train_index = frame.index.difference(test_index)
        splits.append(("single_well_depth_block", f"block_{index:02d}", train_index, test_index))
    return splits


def metric_row(part: pd.DataFrame, threshold: float) -> dict[str, Any]:
    actual_density = numeric(part["Density"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    expected = numeric(part["PredDensity"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    row: dict[str, Any] = {
        "RowCount": int(len(part)),
        "DensityMAE": float(mean_absolute_error(actual_density, expected)),
        "DensityRMSE": float(np.sqrt(mean_squared_error(actual_density, expected))),
        "DensityR2": float(r2_score(actual_density, expected)) if len(part) >= 2 else np.nan,
    }
    labelled = part[numeric(part["PresenceLabel"]).notna()].copy()
    actual = numeric(labelled["PresenceLabel"]).astype(int).to_numpy()
    probability = numeric(labelled["PredFractureProb"]).to_numpy(dtype=float)
    predicted = (probability >= threshold).astype(int)
    row.update(
        {
            "ClearLabelRows": int(len(labelled)),
            "ActualPositiveRows": int(actual.sum()),
            "PredictedPositiveRows": int(predicted.sum()),
            "Accuracy": float(accuracy_score(actual, predicted)) if len(labelled) else np.nan,
            "Precision": float(precision_score(actual, predicted, zero_division=0)) if len(labelled) else np.nan,
            "Recall": float(recall_score(actual, predicted, zero_division=0)) if len(labelled) else np.nan,
            "F1": float(f1_score(actual, predicted, zero_division=0)) if len(labelled) else np.nan,
            "ActualPositiveRatio": float(actual.mean()) if len(labelled) else np.nan,
            "PredictedPositiveRatio": float(predicted.mean()) if len(labelled) else np.nan,
            "PresenceROC_AUC": float(roc_auc_score(actual, probability)) if np.unique(actual).size >= 2 else np.nan,
            "PresenceAveragePrecision": float(average_precision_score(actual, probability)) if np.unique(actual).size >= 2 else np.nan,
        }
    )
    return row


def direct_expert_key(strata_name: str, pair_type: str) -> tuple[str, str]:
    return str(strata_name), str(pair_type)


def train_direct_experts(
    training: pd.DataFrame,
    config: dict[str, Any],
    model_dir: Path,
    validation_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    random_state = int(config.get("random_state", 42))
    models: dict[tuple[str, str], dict[str, Any]] = {}
    registry_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    expert_index = 0
    for pair_type in DIRECT_PAIR_TYPES:
        for strata_name in TARGET_STRATA:
            subset = training[training["PairType"].eq(pair_type) & training["StrataName"].eq(strata_name)].copy()
            if subset.empty:
                raise RuntimeError(f"missing direct training rows for {strata_name}/{pair_type}")
            expert_oof = []
            splits = validation_splits(subset, int(config["single_well_validation_blocks"]))
            for fold_index, (scheme, holdout, train_index, test_index) in enumerate(splits):
                train = subset.loc[train_index].copy()
                test = subset.loc[test_index].copy()
                if train.empty or test.empty:
                    continue
                classifier, regressor = fit_direct(train, config, random_state + expert_index * 100 + fold_index * 10)
                probability, conditional = predict_direct(test, classifier, regressor)
                fold = test[["SampleID", "WellName", "StrataName", "PairType", "DEPT", "Density", "PresenceLabel", "PresenceLabelKind"]].copy()
                fold["PredFractureProb"] = probability
                fold["PredConditionalDensity"] = conditional
                fold["PredDensity"] = probability * conditional
                fold["ValidationScheme"] = scheme
                fold["Holdout"] = holdout
                expert_oof.append(fold)
            if not expert_oof:
                raise RuntimeError(f"no validation predictions for {strata_name}/{pair_type}")
            expert_oof_frame = pd.concat(expert_oof, ignore_index=True)
            threshold = select_threshold(
                expert_oof_frame["PresenceLabel"],
                expert_oof_frame["PredFractureProb"],
                str(config["threshold_strategy"]),
            )
            actual_total = float(numeric(expert_oof_frame["Density"]).fillna(0).clip(lower=0).sum())
            predicted_total = float(numeric(expert_oof_frame["PredDensity"]).fillna(0).clip(lower=0).sum())
            density_scale = actual_total / predicted_total if predicted_total > 1.0e-12 else 1.0
            expert_oof_frame["DensityScale"] = density_scale
            expert_oof_frame["PredConditionalDensity"] *= density_scale
            expert_oof_frame["PredDensity"] = expert_oof_frame["PredFractureProb"] * expert_oof_frame["PredConditionalDensity"]
            for (scheme, holdout), part in expert_oof_frame.groupby(["ValidationScheme", "Holdout"]):
                metric_rows.append(
                    {
                        "ExpertID": f"{strata_name}_{pair_type}",
                        "StrataName": strata_name,
                        "PairType": pair_type,
                        "ValidationScheme": scheme,
                        "Holdout": holdout,
                        "Threshold": threshold,
                        **metric_row(part, threshold),
                    }
                )
            oof_parts.append(expert_oof_frame)

            classifier, regressor = fit_direct(subset, config, random_state + 1000 + expert_index * 10)
            training_probability, _ = predict_direct(subset, classifier, regressor)
            deployment_threshold = select_threshold(
                subset["PresenceLabel"], training_probability, str(config["threshold_strategy"])
            )
            expert_id = f"{strata_name}_{pair_type}"
            stage1_path = model_dir / f"{expert_id}_stage1_xgb.joblib"
            stage2_path = model_dir / f"{expert_id}_stage2_xgb.joblib"
            joblib.dump(classifier, stage1_path)
            joblib.dump(regressor, stage2_path)
            models[direct_expert_key(strata_name, pair_type)] = {
                "kind": "direct",
                "stage1": classifier,
                "stage2": regressor,
                "threshold": deployment_threshold,
                "density_scale": density_scale,
                "expert_id": expert_id,
                "model_id": f"{expert_id}_xgb_v3",
            }
            expert_metrics = [row for row in metric_rows if row["ExpertID"] == expert_id]
            registry_rows.append(
                {
                    "ExpertID": expert_id,
                    "ModelID": f"{expert_id}_xgb_v3",
                    "StrataName": strata_name,
                    "PairType": pair_type,
                    "TrainingMode": "direct_imaging_supervision",
                    "TrainingWells": ";".join(sorted(subset["WellName"].astype(str).unique())),
                    "TrainingRows": int(len(subset)),
                    "ClearFractureRows": int(subset["PresenceLabel"].eq(1).sum()),
                    "ClearBackgroundRows": int(subset["PresenceLabel"].eq(0).sum()),
                    "UncertainRows": int(subset["PresenceLabel"].isna().sum()),
                    "ValidationScheme": ";".join(sorted({str(row["ValidationScheme"]) for row in expert_metrics})),
                    "OOFStage1Threshold": threshold,
                    "DeploymentStage1Threshold": deployment_threshold,
                    "OOFDensityScale": density_scale,
                    "MeanValidationROCAUC": float(pd.Series([row["PresenceROC_AUC"] for row in expert_metrics]).mean()),
                    "MeanValidationAveragePrecision": float(pd.Series([row["PresenceAveragePrecision"] for row in expert_metrics]).mean()),
                    "Stage1ModelPath": str(Path("model_library") / stage1_path.name),
                    "Stage2ModelPath": str(Path("model_library") / stage2_path.name),
                }
            )
            expert_index += 1
    oof = pd.concat(oof_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    oof.to_csv(validation_dir / "direct_expert_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(validation_dir / "direct_expert_validation_metrics.csv", index=False, encoding="utf-8-sig")
    return models, pd.DataFrame(registry_rows), metrics


def build_rild_bridge(
    config: dict[str, Any], direct_models: dict[tuple[str, str], dict[str, Any]]
) -> pd.DataFrame:
    root = Path(config["step2_output_root"])
    parts = []
    for well_name in config["bridge_wells"]:
        well_dir = root / str(well_name)
        main = pd.read_csv(well_dir / f"{well_name}_t4_t7_real_well_main.csv")
        interval = pd.read_csv(well_dir / f"{well_name}_t4_t7_real_well_interval.csv").iloc[0]
        enrichment = pd.read_csv(v2.enrichment_path(root, str(well_name)))
        main = v2.assign_step2_strata(main, interval)
        rd = v2.select_pair_rows(main, enrichment, "RD_RS")
        rild = v2.select_pair_rows(main, enrichment, "RILD_RILM")
        common = sorted(set(rd["DEPT"]).intersection(set(rild["DEPT"])))
        if not common:
            continue
        rd_features, _ = engineer_features(rd[rd["DEPT"].isin(common)].copy())
        rild_features, _ = engineer_features(rild[rild["DEPT"].isin(common)].copy())
        teachers = []
        for strata_name in TARGET_STRATA:
            subset = rd_features[rd_features["StrataName"].eq(strata_name)].copy()
            model = direct_models.get(direct_expert_key(strata_name, "RD_RS"))
            if subset.empty or model is None:
                continue
            probability, conditional = predict_direct(subset, model["stage1"], model["stage2"])
            conditional *= float(model["density_scale"])
            teacher = subset[["WellName", "StrataName", "DEPT"]].copy()
            teacher["TeacherFractureProb"] = probability
            teacher["TeacherConditionalDensity"] = conditional
            teacher["TeacherExpectedDensity"] = probability * conditional
            teacher["TeacherPresenceLabel"] = (probability >= float(model["threshold"])).astype(int)
            teachers.append(teacher)
        if teachers:
            student = rild_features.merge(
                pd.concat(teachers, ignore_index=True),
                on=["WellName", "StrataName", "DEPT"],
                how="inner",
                validate="one_to_one",
            )
            parts.append(student)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fit_transfer(frame: pd.DataFrame, config: dict[str, Any], random_state: int) -> tuple[Any, Any]:
    weights = equalized_depth_weights(frame, float(config["depth_weight_bin_m"]))
    probability_model = xgb_regressor(config, random_state, transfer=True)
    density_model = xgb_regressor(config, random_state + 1, transfer=True)
    probability_model.fit(safe_features(frame), numeric(frame["TeacherFractureProb"]), sample_weight=weights)
    density_model.fit(
        safe_features(frame),
        np.log1p(numeric(frame["TeacherConditionalDensity"]).clip(lower=0.0)),
        sample_weight=weights,
    )
    return probability_model, density_model


def predict_transfer(frame: pd.DataFrame, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(model["probability"].predict(safe_features(frame)), 0.0, 1.0)
    conditional = np.maximum(np.expm1(model["density"].predict(safe_features(frame))), 0.0)
    return probability, conditional


def train_rild_experts(
    bridge: pd.DataFrame,
    config: dict[str, Any],
    model_dir: Path,
    validation_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    if bridge.empty:
        raise RuntimeError("RILD bridge training table is empty")
    models: dict[tuple[str, str], dict[str, Any]] = {}
    registry_rows = []
    metric_rows = []
    random_state = int(config.get("random_state", 42)) + 5000
    for expert_index, strata_name in enumerate(TARGET_STRATA):
        subset = bridge[bridge["StrataName"].eq(strata_name)].copy()
        if subset.empty:
            raise RuntimeError(f"missing RILD bridge rows for {strata_name}")
        oof_parts = []
        for fold_index, (scheme, holdout, train_index, test_index) in enumerate(
            validation_splits(subset, int(config["single_well_validation_blocks"]))
        ):
            train = subset.loc[train_index].copy()
            test = subset.loc[test_index].copy()
            probability_model, density_model = fit_transfer(
                train, config, random_state + expert_index * 100 + fold_index * 10
            )
            probability = np.clip(probability_model.predict(safe_features(test)), 0.0, 1.0)
            conditional = np.maximum(np.expm1(density_model.predict(safe_features(test))), 0.0)
            fold = test[["WellName", "StrataName", "DEPT", "TeacherFractureProb", "TeacherConditionalDensity", "TeacherExpectedDensity", "TeacherPresenceLabel"]].copy()
            fold["PredFractureProb"] = probability
            fold["PredConditionalDensity"] = conditional
            fold["PredDensity"] = probability * conditional
            fold["ValidationScheme"] = scheme
            fold["Holdout"] = holdout
            oof_parts.append(fold)
        oof = pd.concat(oof_parts, ignore_index=True)
        threshold = select_threshold(oof["TeacherPresenceLabel"], oof["PredFractureProb"], str(config["threshold_strategy"]))
        actual_total = float(numeric(oof["TeacherExpectedDensity"]).clip(lower=0).sum())
        predicted_total = float(numeric(oof["PredDensity"]).clip(lower=0).sum())
        density_scale = actual_total / predicted_total if predicted_total > 1.0e-12 else 1.0
        oof["PredConditionalDensity"] *= density_scale
        oof["PredDensity"] = oof["PredFractureProb"] * oof["PredConditionalDensity"]
        for (scheme, holdout), part in oof.groupby(["ValidationScheme", "Holdout"]):
            actual = numeric(part["TeacherPresenceLabel"]).astype(int).to_numpy()
            probability = numeric(part["PredFractureProb"]).to_numpy(dtype=float)
            predicted = (probability >= threshold).astype(int)
            metric_rows.append(
                {
                    "ExpertID": f"{strata_name}_RILD_RILM",
                    "StrataName": strata_name,
                    "PairType": "RILD_RILM",
                    "ValidationScheme": scheme,
                    "Holdout": holdout,
                    "RowCount": int(len(part)),
                    "Threshold": threshold,
                    "Accuracy": float(accuracy_score(actual, predicted)),
                    "Precision": float(precision_score(actual, predicted, zero_division=0)),
                    "Recall": float(recall_score(actual, predicted, zero_division=0)),
                    "F1": float(f1_score(actual, predicted, zero_division=0)),
                    "PresenceROC_AUC": float(roc_auc_score(actual, probability)) if np.unique(actual).size >= 2 else np.nan,
                    "PresenceAveragePrecision": float(average_precision_score(actual, probability)) if np.unique(actual).size >= 2 else np.nan,
                    "TeacherProbabilityMAE": float(mean_absolute_error(part["TeacherFractureProb"], probability)),
                    "ExpectedDensityMAE": float(mean_absolute_error(part["TeacherExpectedDensity"], part["PredDensity"])),
                    "ExpectedDensityR2": float(r2_score(part["TeacherExpectedDensity"], part["PredDensity"])) if len(part) >= 2 else np.nan,
                }
            )

        probability_model, density_model = fit_transfer(subset, config, random_state + 1000 + expert_index * 10)
        training_probability = np.clip(probability_model.predict(safe_features(subset)), 0.0, 1.0)
        deployment_threshold = select_threshold(
            subset["TeacherPresenceLabel"], training_probability, str(config["threshold_strategy"])
        )
        expert_id = f"{strata_name}_RILD_RILM"
        probability_path = model_dir / f"{expert_id}_stage1_transfer_xgb.joblib"
        density_path = model_dir / f"{expert_id}_stage2_transfer_xgb.joblib"
        joblib.dump(probability_model, probability_path)
        joblib.dump(density_model, density_path)
        models[direct_expert_key(strata_name, "RILD_RILM")] = {
            "kind": "transfer",
            "probability": probability_model,
            "density": density_model,
            "threshold": deployment_threshold,
            "density_scale": density_scale,
            "expert_id": expert_id,
            "model_id": f"{expert_id}_xgb_v3",
        }
        expert_metrics = [row for row in metric_rows if row["ExpertID"] == expert_id]
        registry_rows.append(
            {
                "ExpertID": expert_id,
                "ModelID": f"{expert_id}_xgb_v3",
                "StrataName": strata_name,
                "PairType": "RILD_RILM",
                "TrainingMode": "RD_RS_teacher_to_RILD_RILM_student",
                "TrainingWells": ";".join(sorted(subset["WellName"].astype(str).unique())),
                "TrainingRows": int(len(subset)),
                "ValidationScheme": ";".join(sorted({str(row["ValidationScheme"]) for row in expert_metrics})),
                "OOFStage1Threshold": threshold,
                "DeploymentStage1Threshold": deployment_threshold,
                "OOFDensityScale": density_scale,
                "MeanValidationROCAUC": float(pd.Series([row["PresenceROC_AUC"] for row in expert_metrics]).mean()),
                "MeanValidationAveragePrecision": float(pd.Series([row["PresenceAveragePrecision"] for row in expert_metrics]).mean()),
                "Stage1ModelPath": str(Path("model_library") / probability_path.name),
                "Stage2ModelPath": str(Path("model_library") / density_path.name),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(validation_dir / "rild_expert_bridge_validation_metrics.csv", index=False, encoding="utf-8-sig")
    return models, pd.DataFrame(registry_rows), metrics


def add_support_segments(frame: pd.DataFrame, max_gap_m: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    parts = []
    for (well_name, strata_name, pair_type), group in frame.groupby(
        ["WellName", "StrataName", "InputPairType"], sort=False
    ):
        out = group.sort_values("DEPT").copy()
        gap = numeric(out["DEPT"]).diff()
        new_segment = gap.isna() | gap.gt(max_gap_m) | gap.le(0.0)
        local_id = new_segment.cumsum().astype(int)
        out["SupportSegmentID"] = [
            f"{well_name}_{strata_name}_{pair_type}_{value:03d}" for value in local_id
        ]
        bounds = out.groupby("SupportSegmentID")["DEPT"].agg(["min", "max"])
        out["SegmentStartDEPT"] = out["SupportSegmentID"].map(bounds["min"])
        out["SegmentEndDEPT"] = out["SupportSegmentID"].map(bounds["max"])
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def add_merged_support_segments(frame: pd.DataFrame, max_gap_m: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    parts = []
    for (well_name, strata_name), group in frame.groupby(["WellName", "StrataName"], sort=False):
        out = group.sort_values("DEPT").copy()
        gap = numeric(out["DEPT"]).diff()
        pair_change = out["InputPairType"].ne(out["InputPairType"].shift())
        new_segment = gap.isna() | gap.gt(max_gap_m) | gap.le(0.0) | pair_change
        local_id = new_segment.cumsum().astype(int)
        out["SupportSegmentID"] = [
            f"{well_name}_{strata_name}_MERGED_{value:03d}" for value in local_id
        ]
        bounds = out.groupby("SupportSegmentID")["DEPT"].agg(["min", "max"])
        out["SegmentStartDEPT"] = out["SupportSegmentID"].map(bounds["min"])
        out["SegmentEndDEPT"] = out["SupportSegmentID"].map(bounds["max"])
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def merge_expert_predictions(predicted: pd.DataFrame, max_gap_m: float) -> pd.DataFrame:
    if predicted.empty:
        return pd.DataFrame(columns=MERGED_OUTPUT_COLUMNS)
    priority = {pair_type: index for index, pair_type in enumerate(MERGE_PAIR_PRIORITY)}
    work = predicted.copy()
    work["SelectionPriority"] = work["InputPairType"].map(priority)
    if work["SelectionPriority"].isna().any():
        unknown = sorted(work.loc[work["SelectionPriority"].isna(), "InputPairType"].astype(str).unique())
        raise RuntimeError(f"merge priority missing pair types: {unknown}")
    availability = (
        work.groupby(["WellName", "DEPT"], as_index=False)
        .agg(
            AvailableExpertCount=("InputPairType", "nunique"),
            AvailablePairTypes=(
                "InputPairType",
                lambda values: ";".join(
                    sorted(set(values.astype(str)), key=lambda value: priority[value])
                ),
            ),
        )
    )
    merged = (
        work.sort_values(["WellName", "DEPT", "SelectionPriority"])
        .drop_duplicates(["WellName", "DEPT"], keep="first")
        .merge(availability, on=["WellName", "DEPT"], how="left", validate="one_to_one")
    )
    merged["SelectionPriority"] = merged["SelectionPriority"].astype(int)
    merged["MergeRule"] = ">".join(MERGE_PAIR_PRIORITY)
    merged = add_merged_support_segments(merged, max_gap_m)
    return merged[MERGED_OUTPUT_COLUMNS]


def build_prediction_rows(
    main: pd.DataFrame,
    enrichment: pd.DataFrame,
    models: dict[tuple[str, str], dict[str, Any]],
    max_support_gap_m: float,
) -> pd.DataFrame:
    candidates = []
    for pair_type in PAIR_TYPES:
        selected = v2.select_pair_rows(main, enrichment, pair_type)
        featured, _ = engineer_features(selected)
        if featured.empty:
            continue
        spatial = featured[["X", "Y", "TIME", "TVD", "DEPT"]].apply(numeric).notna().all(axis=1)
        featured = featured[featured["StrataName"].isin(TARGET_STRATA) & spatial].copy()
        for strata_name in TARGET_STRATA:
            subset = featured[featured["StrataName"].eq(strata_name)].copy()
            model = models.get(direct_expert_key(strata_name, pair_type))
            if subset.empty or model is None:
                continue
            if model["kind"] == "direct":
                probability, conditional = predict_direct(subset, model["stage1"], model["stage2"])
            else:
                probability, conditional = predict_transfer(subset, model)
            conditional *= float(model["density_scale"])
            threshold = float(model["threshold"])
            subset["PredFractureProb"] = probability
            subset["PredConditionalDensity"] = conditional
            subset["PredDensity"] = probability * conditional
            subset["PredHasFracture"] = (probability >= threshold).astype(int)
            subset["Stage1Threshold"] = threshold
            subset["Density"] = subset["PredDensity"]
            subset["HasFracture"] = subset["PredHasFracture"]
            subset["InputPairType"] = pair_type
            subset["ExpertID"] = model["expert_id"]
            subset["ModelID"] = model["model_id"]
            subset["PredictionValid"] = 1
            subset["SourceSampleID"] = subset["SampleID"].astype(str)
            candidates.append(subset)
    if not candidates:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat(candidates, ignore_index=True)
    combined = combined.sort_values(["WellName", "DEPT", "InputPairType"]).drop_duplicates(
        ["WellName", "DEPT", "InputPairType"], keep="first"
    )
    return add_support_segments(combined, max_support_gap_m)


def point_count_scales(training: pd.DataFrame) -> dict[str, float]:
    scales = {}
    for strata_name, strata in training.groupby("StrataName"):
        area = 0.0
        for _, well in strata.groupby("WellName"):
            ordered = well.sort_values("DEPT")
            depth = numeric(ordered["DEPT"]).to_numpy(dtype=float)
            density = numeric(ordered["Density"]).fillna(0).clip(lower=0).to_numpy(dtype=float)
            if len(depth) >= 2:
                area += float(np.trapezoid(density, depth))
        points = int(numeric(strata.get("GT_POINT_FLAG", pd.Series(0, index=strata.index))).fillna(0).gt(0).sum())
        scales[str(strata_name)] = float(points / area) if area > 1.0e-12 else 1.0
    return scales


def refine_points(predicted: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    columns = [
        "SampleID", "SourceSampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT",
        "StrataName", "InputPairType", "ExpertID", "ModelID", "SupportSegmentID",
        "Density", "HasFracture", "PredFractureProb", "SegmentStartDEPT", "SegmentEndDEPT",
        "PointOrdinalInSegment",
    ]
    rows = []
    for support_id, support in predicted.groupby("SupportSegmentID"):
        support = support.sort_values("DEPT").reset_index(drop=True)
        positive = numeric(support["HasFracture"]).fillna(0).gt(0).to_numpy()
        starts = np.flatnonzero(positive & np.concatenate(([True], ~positive[:-1])))
        for start in starts:
            end = start
            while end + 1 < len(support) and positive[end + 1]:
                end += 1
            segment = support.iloc[start : end + 1]
            depth = numeric(segment["DEPT"]).to_numpy(dtype=float)
            density = numeric(segment["Density"]).fillna(0).to_numpy(dtype=float)
            area = float(np.trapezoid(density, depth)) if len(depth) >= 2 else float(density[0]) * 0.125
            count = min(
                len(segment),
                max(1, int(round(area * float(scales.get(str(segment["StrataName"].iloc[0]), 1.0))))),
            )
            for ordinal, local_index in enumerate(sorted(np.argsort(-density)[:count].tolist()), start=1):
                source = segment.iloc[int(local_index)]
                rows.append(
                    {
                        "SampleID": source["SampleID"],
                        "SourceSampleID": source["SourceSampleID"],
                        "WellName": source["WellName"],
                        "X": source["X"],
                        "Y": source["Y"],
                        "TIME": source["TIME"],
                        "TVD": source["TVD"],
                        "DEPT": source["DEPT"],
                        "StrataName": source["StrataName"],
                        "InputPairType": source["InputPairType"],
                        "ExpertID": source["ExpertID"],
                        "ModelID": source["ModelID"],
                        "SupportSegmentID": support_id,
                        "Density": source["Density"],
                        "HasFracture": 1,
                        "PredFractureProb": source["PredFractureProb"],
                        "SegmentStartDEPT": float(segment["DEPT"].min()),
                        "SegmentEndDEPT": float(segment["DEPT"].max()),
                        "PointOrdinalInSegment": ordinal,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def unsupported_intervals(main: pd.DataFrame, predicted: pd.DataFrame, well_name: str) -> pd.DataFrame:
    output = []
    target = main[main["StrataName"].isin(TARGET_STRATA)].copy()
    for pair_type in PAIR_TYPES:
        predicted_depth = set(numeric(predicted.loc[predicted["InputPairType"].eq(pair_type), "DEPT"]).dropna())
        rows = target[["DEPT", "StrataName"]].copy().sort_values("DEPT")
        rows["Unsupported"] = ~numeric(rows["DEPT"]).isin(predicted_depth)
        rows["Run"] = (
            rows["Unsupported"].ne(rows["Unsupported"].shift())
            | rows["StrataName"].ne(rows["StrataName"].shift())
        ).cumsum()
        for _, segment in rows[rows["Unsupported"]].groupby("Run"):
            output.append(
                {
                    "WellName": well_name,
                    "PairType": pair_type,
                    "StrataName": str(segment["StrataName"].iloc[0]),
                    "StartDEPT": float(segment["DEPT"].min()),
                    "EndDEPT": float(segment["DEPT"].max()),
                    "RowCount": int(len(segment)),
                    "Reason": "no_complete_pair_or_invalid_spatial_coordinates",
                }
            )
    return pd.DataFrame(output)


def predict_all_wells(
    config: dict[str, Any],
    models: dict[tuple[str, str], dict[str, Any]],
    training: pd.DataFrame,
    staging: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(config["step2_output_root"])
    summary = pd.read_csv(root / "real_well_t4_t7_summary.csv")
    prediction_root = staging / "predictions"
    per_well_root = prediction_root / "real_well_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    per_well_root.mkdir(parents=True, exist_ok=True)
    expert_parts = []
    merged_parts = []
    coverage_rows = []
    unsupported_parts = []
    for _, row in summary.iterrows():
        well_name = str(row["WellName"])
        status = str(row["Status"])
        if status != "ok":
            coverage_rows.append({"WellName": well_name, "Status": f"step2_{status}", "TotalRows": 0})
            continue
        main = pd.read_csv(Path(str(row["MainCsv"])))
        interval = pd.read_csv(Path(str(row["IntervalCsv"]))).iloc[0]
        main = v2.assign_step2_strata(main, interval)
        enrichment_file = v2.enrichment_path(root, well_name)
        if not enrichment_file.exists():
            coverage_rows.append({"WellName": well_name, "Status": "missing_enrichment", "TotalRows": int(len(main))})
            continue
        enrichment = pd.read_csv(enrichment_file)
        expert_predictions = build_prediction_rows(
            main,
            enrichment,
            models,
            float(config["max_support_gap_m"]),
        )
        merged = merge_expert_predictions(
            expert_predictions,
            float(config["max_support_gap_m"]),
        )
        if not expert_predictions.empty:
            well_output_dir = per_well_root / well_name
            well_output_dir.mkdir(parents=True, exist_ok=True)
            for pair_type in PAIR_TYPES:
                pair_prediction = expert_predictions[
                    expert_predictions["InputPairType"].eq(pair_type)
                ].copy()
                if pair_prediction.empty:
                    continue
                pair_prediction[OUTPUT_COLUMNS].to_csv(
                    well_output_dir / f"{well_name}_{pair_type}_expert_prediction.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            merged.to_csv(
                well_output_dir / f"{well_name}_merged_density_prediction.csv",
                index=False,
                encoding="utf-8-sig",
            )
            expert_parts.append(expert_predictions[OUTPUT_COLUMNS])
            merged_parts.append(merged)
        unsupported = unsupported_intervals(main, expert_predictions, well_name)
        if not unsupported.empty:
            unsupported_parts.append(unsupported)
        pair_counts = {
            f"{pair_type}_PredictedRows": int(
                expert_predictions["InputPairType"].eq(pair_type).sum()
            )
            for pair_type in PAIR_TYPES
        }
        coverage_rows.append(
            {
                "WellName": well_name,
                "Status": "predicted" if not merged.empty else "no_supported_prediction",
                "TotalRows": int(len(main)),
                "ExpertPredictionRows": int(len(expert_predictions)),
                "MergedPredictionRows": int(len(merged)),
                "MergedSupportSegments": int(merged["SupportSegmentID"].nunique()) if not merged.empty else 0,
                **pair_counts,
            }
        )
        print(
            f"[Step4-v3] {well_name}: expert={len(expert_predictions)} merged={len(merged)} "
            + " ".join(f"{pair}={pair_counts[f'{pair}_PredictedRows']}" for pair in PAIR_TYPES),
            flush=True,
        )
    expert_predictions = (
        pd.concat(expert_parts, ignore_index=True)
        if expert_parts
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    merged_predictions = (
        pd.concat(merged_parts, ignore_index=True)
        if merged_parts
        else pd.DataFrame(columns=MERGED_OUTPUT_COLUMNS)
    )
    points = refine_points(merged_predictions, point_count_scales(training))
    coverage = pd.DataFrame(coverage_rows)
    unsupported = pd.concat(unsupported_parts, ignore_index=True) if unsupported_parts else pd.DataFrame()
    expert_predictions.to_csv(
        prediction_root / "all_wells_t4_t7_expert_detail_prediction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    merged_predictions.to_csv(
        prediction_root / "all_wells_t4_t7_merged_density_prediction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    points.to_csv(
        prediction_root / "all_wells_t4_t7_merged_fracture_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    qc_root = staging / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(qc_root / "well_expert_prediction_coverage.csv", index=False, encoding="utf-8-sig")
    unsupported.to_csv(qc_root / "unsupported_pair_intervals.csv", index=False, encoding="utf-8-sig")
    return expert_predictions, merged_predictions, points, coverage, unsupported


def validate_delivery(
    config: dict[str, Any],
    registry: pd.DataFrame,
    expert_predictions: pd.DataFrame,
    merged_predictions: pd.DataFrame,
    points: pd.DataFrame,
    coverage: pd.DataFrame,
    staging: Path,
    output_dir: Path,
) -> dict[str, Any]:
    errors = []
    source_ok = True
    root = Path(config["step2_output_root"])
    for (well_name, pair_type), part in expert_predictions.groupby(["WellName", "InputPairType"]):
        spec = v2.PAIR_SPECS[str(pair_type)]
        source = pd.read_csv(
            v2.enrichment_path(root, str(well_name)),
            usecols=["DEPT", spec.gr_column, spec.deep_column, spec.reference_column],
        )
        joined = part[["DEPT"]].merge(source, on="DEPT", how="left", validate="many_to_one")
        complete = joined[[spec.gr_column, spec.deep_column, spec.reference_column]].notna().all(axis=1)
        if not complete.all():
            source_ok = False
            errors.append(f"{well_name}/{pair_type}:missing_source={int((~complete).sum())}")

    def support_gaps_bounded(frame: pd.DataFrame, label: str) -> bool:
        valid = True
        for support_id, part in frame.groupby("SupportSegmentID"):
            depth = np.sort(numeric(part["DEPT"]).to_numpy(dtype=float))
            if len(depth) > 1 and np.diff(depth).max() > float(config["max_support_gap_m"]) + 1.0e-9:
                valid = False
                errors.append(f"{label}/{support_id}:gap_exceeded")
        return valid

    expert_support_ok = support_gaps_bounded(expert_predictions, "expert")
    merged_support_ok = support_gaps_bounded(merged_predictions, "merged")

    priority = {pair_type: index for index, pair_type in enumerate(MERGE_PAIR_PRIORITY)}
    availability = (
        expert_predictions.groupby(["WellName", "DEPT"], as_index=False)
        .agg(
            ExpectedExpertCount=("InputPairType", "nunique"),
            ExpectedPairTypes=(
                "InputPairType",
                lambda values: ";".join(
                    sorted(set(values.astype(str)), key=lambda value: priority[value])
                ),
            ),
            ExpectedSelectedPair=(
                "InputPairType",
                lambda values: min(set(values.astype(str)), key=lambda value: priority[value]),
            ),
        )
    )
    merge_audit = merged_predictions.merge(
        availability,
        on=["WellName", "DEPT"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    merged_row_coverage_ok = merge_audit["_merge"].eq("both").all()
    deterministic_selection_ok = merged_row_coverage_ok and merge_audit["InputPairType"].eq(
        merge_audit["ExpectedSelectedPair"]
    ).all()
    availability_metadata_ok = merged_row_coverage_ok and (
        numeric(merge_audit["AvailableExpertCount"]).eq(numeric(merge_audit["ExpectedExpertCount"])).all()
        and merge_audit["AvailablePairTypes"].astype(str).eq(merge_audit["ExpectedPairTypes"].astype(str)).all()
    )
    merge_rule_ok = (
        merged_predictions["MergeRule"].eq(">".join(MERGE_PAIR_PRIORITY)).all()
        and numeric(merged_predictions["SelectionPriority"]).eq(
            merged_predictions["InputPairType"].map(priority)
        ).all()
    )

    per_well_root = staging / "predictions" / "real_well_predictions"
    predicted_wells = sorted(merged_predictions["WellName"].astype(str).unique())
    per_well_layout_ok = True
    flat_csvs = sorted(per_well_root.glob("*.csv"))
    if flat_csvs:
        per_well_layout_ok = False
        errors.append(f"flat_per_well_csvs={len(flat_csvs)}")
    actual_well_dirs = sorted(path.name for path in per_well_root.iterdir() if path.is_dir())
    if actual_well_dirs != predicted_wells:
        per_well_layout_ok = False
        errors.append("per_well_directory_set_mismatch")
    for well_name in predicted_wells:
        well_dir = per_well_root / well_name
        expected_pairs = sorted(
            expert_predictions.loc[
                expert_predictions["WellName"].astype(str).eq(well_name), "InputPairType"
            ].astype(str).unique()
        )
        expected_files = {f"{well_name}_{pair}_expert_prediction.csv" for pair in expected_pairs}
        expected_files.add(f"{well_name}_merged_density_prediction.csv")
        actual_files = {path.name for path in well_dir.glob("*.csv")}
        if actual_files != expected_files:
            per_well_layout_ok = False
            errors.append(f"{well_name}:per_well_file_set_mismatch")

    expected_experts = {f"{strata}_{pair}" for strata in TARGET_STRATA for pair in PAIR_TYPES}
    checks = {
        "six_experts_registered": set(registry["ExpertID"].astype(str)) == expected_experts,
        "registry_has_no_status_columns": not any("status" in str(column).lower() for column in registry.columns),
        "expert_predictions_nonempty": not expert_predictions.empty,
        "merged_predictions_nonempty": not merged_predictions.empty,
        "all_pair_types_predicted": set(expert_predictions["InputPairType"].astype(str)) == set(PAIR_TYPES),
        "expert_prediction_keys_unique": not expert_predictions.duplicated(
            ["WellName", "DEPT", "InputPairType"]
        ).any(),
        "merged_prediction_keys_unique": not merged_predictions.duplicated(["WellName", "DEPT"]).any(),
        "merged_rows_cover_expert_depths": merged_row_coverage_ok,
        "deterministic_pair_selection": deterministic_selection_ok,
        "merged_availability_metadata_correct": availability_metadata_ok,
        "merged_rule_metadata_correct": merge_rule_ok,
        "prediction_valid_only": (
            expert_predictions["PredictionValid"].eq(1).all()
            and merged_predictions["PredictionValid"].eq(1).all()
        ),
        "density_finite": (
            np.isfinite(numeric(expert_predictions["Density"])).all()
            and np.isfinite(numeric(merged_predictions["Density"])).all()
        ),
        "coordinates_finite": np.isfinite(
            expert_predictions[["X", "Y", "TIME", "TVD", "DEPT"]].apply(numeric).to_numpy(dtype=float)
        ).all() and np.isfinite(
            merged_predictions[["X", "Y", "TIME", "TVD", "DEPT"]].apply(numeric).to_numpy(dtype=float)
        ).all(),
        "source_triples_complete": source_ok,
        "expert_support_gaps_bounded": expert_support_ok,
        "merged_support_gaps_bounded": merged_support_ok,
        "points_inside_merged_support": points.empty or points["SupportSegmentID"].isin(
            set(merged_predictions["SupportSegmentID"])
        ).all(),
        "coverage_matches_expert_predictions": int(
            coverage.get("ExpertPredictionRows", pd.Series(dtype=float)).fillna(0).sum()
        ) == int(len(expert_predictions)),
        "coverage_matches_merged_predictions": int(
            coverage.get("MergedPredictionRows", pd.Series(dtype=float)).fillna(0).sum()
        ) == int(len(merged_predictions)),
        "per_well_directory_layout": per_well_layout_ok,
        "prediction_output_has_no_status_columns": not any(
            "status" in str(column).lower()
            for column in list(expert_predictions.columns) + list(merged_predictions.columns)
        ),
        "isolated_v3_output": output_dir.name not in {
            "formal_gr_resistivity_strata_library_v2",
            "formal_gr_resistivity_lateral_v1",
            "formal_well_expert_library",
        },
    }
    summary = {
        "status": "pass" if all(bool(value) for value in checks.values()) else "fail",
        "output_dir": str(output_dir),
        "registered_experts": sorted(registry["ExpertID"].astype(str)),
        "predicted_wells": int(coverage.get("MergedPredictionRows", pd.Series(dtype=float)).fillna(0).gt(0).sum()),
        "expert_prediction_rows": int(len(expert_predictions)),
        "merged_prediction_rows": int(len(merged_predictions)),
        "expert_prediction_rows_by_pair": expert_predictions["InputPairType"].value_counts().to_dict(),
        "merged_selected_rows_by_pair": merged_predictions["InputPairType"].value_counts().to_dict(),
        "multi_expert_well_depth_count": int(
            numeric(merged_predictions["AvailableExpertCount"]).gt(1).sum()
        ),
        "merged_fracture_points": int(len(points)),
        "source_support_errors": errors,
        "checks": checks,
    }
    write_json(staging / "step4_six_expert_acceptance_summary.json", summary)
    manifest = {
        "version": "formal_six_expert_library_v3",
        "prediction_table": "predictions/all_wells_t4_t7_merged_density_prediction.csv",
        "fracture_point_table": "predictions/all_wells_t4_t7_merged_fracture_points.csv",
        "expert_detail_table": "predictions/all_wells_t4_t7_expert_detail_prediction.csv",
        "per_well_prediction_root": "predictions/real_well_predictions",
        "model_registry": "model_library/six_expert_registry.csv",
        "prediction_unique_key": ["WellName", "DEPT"],
        "prediction_valid_column": "PredictionValid",
        "prediction_valid_value": 1,
        "pair_types": list(PAIR_TYPES),
        "expert_count": 6,
        "merge_pair_priority": list(MERGE_PAIR_PRIORITY),
        "expert_detail_unique_key": ["WellName", "DEPT", "InputPairType"],
        "step5_input": "predictions/all_wells_t4_t7_merged_density_prediction.csv",
        "qc_coverage_table": "qc/well_expert_prediction_coverage.csv",
        "acceptance_summary": "step4_six_expert_acceptance_summary.json",
    }
    write_json(staging / "six_expert_prediction_manifest.json", manifest)
    return summary


def publish(staging: Path, output_dir: Path) -> None:
    previous: Path | None = None
    if output_dir.exists():
        previous = output_dir.parent / f".{output_dir.name}_previous"
        if previous.exists():
            raise RuntimeError(f"backup path already exists: {previous}")
        output_dir.rename(previous)
    try:
        staging.rename(output_dir)
    except Exception:
        if previous is not None and previous.exists() and not output_dir.exists():
            previous.rename(output_dir)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def run(config: dict[str, Any], output_dir: Path, replace_existing: bool, train_only: bool) -> dict[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not replace_existing:
        raise RuntimeError(f"refusing to mix into existing output: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_staging_", dir=output_dir.parent))
    try:
        model_dir = staging / "model_library"
        validation_dir = staging / "validation"
        model_dir.mkdir(parents=True, exist_ok=True)
        validation_dir.mkdir(parents=True, exist_ok=True)
        training, training_manifest, normalization = load_training_table(config)
        training.to_csv(staging / "training_feature_table.csv", index=False, encoding="utf-8-sig")
        training_manifest.to_csv(staging / "training_input_manifest.csv", index=False, encoding="utf-8-sig")
        normalization.to_csv(staging / "training_normalization_stats.csv", index=False, encoding="utf-8-sig")
        print(f"[Step4-v3] direct training rows={len(training)}", flush=True)

        direct_models, direct_registry, direct_metrics = train_direct_experts(
            training, config, model_dir, validation_dir
        )
        bridge = build_rild_bridge(config, direct_models)
        bridge.to_csv(staging / "rild_bridge_teacher_student_table.csv", index=False, encoding="utf-8-sig")
        transfer_models, transfer_registry, transfer_metrics = train_rild_experts(
            bridge, config, model_dir, validation_dir
        )
        models = {**direct_models, **transfer_models}
        registry = pd.concat([direct_registry, transfer_registry], ignore_index=True)
        registry.to_csv(model_dir / "six_expert_registry.csv", index=False, encoding="utf-8-sig")
        write_json(
            staging / "expert_thresholds.json",
            {model["expert_id"]: float(model["threshold"]) for model in models.values()},
        )
        if len(models) != 6:
            raise RuntimeError(f"expected six trained experts, got {len(models)}")

        if train_only:
            summary = {
                "status": "train_only",
                "registered_experts": sorted(registry["ExpertID"].astype(str)),
                "direct_validation_rows": int(len(direct_metrics)),
                "rild_validation_rows": int(len(transfer_metrics)),
            }
            write_json(staging / "step4_six_expert_acceptance_summary.json", summary)
        else:
            expert_predictions, merged_predictions, points, coverage, _ = predict_all_wells(
                config, models, training, staging
            )
            summary = validate_delivery(
                config,
                registry,
                expert_predictions,
                merged_predictions,
                points,
                coverage,
                staging,
                output_dir,
            )
            if summary["status"] != "pass":
                raise RuntimeError(f"Step4 v3 acceptance failed: {summary['checks']}")
        write_json(staging / "run_config_snapshot.json", config)
        publish(staging, output_dir)
        return summary
    except Exception as exc:
        print(f"[Step4-v3] failed before publication: {type(exc).__name__}: {exc}", flush=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    summary = run(config, output_dir, args.replace_existing_output, args.train_only)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
