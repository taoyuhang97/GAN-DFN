"""Train and deploy the Step4 GR/resistivity strata-library v2.

Direct LLD/LLS and RD/RS models use Step3 imaging labels joined exactly to
Step2 enrichment rows. RILD/RILM is trained only as an experimental student
of the direct model on reviewed bridge wells. Formal outputs contain predicted
rows only; unsupported intervals are represented exclusively in QC tables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


CURRENT_DIR = Path(__file__).resolve().parent
TARGET_STRATA = ("沙三段", "沙四段")


@dataclass(frozen=True)
class PairSpec:
    pair_type: str
    gr_column: str
    deep_column: str
    reference_column: str
    reference_type: str
    training_mode: str


PAIR_SPECS = {
    "LLD_LLS": PairSpec("LLD_LLS", "GR_LLD_LLS", "LLD", "LLS", "SHALLOW", "direct"),
    "RD_RS": PairSpec("RD_RS", "GR_RD_RS", "RD", "RS", "SHALLOW", "direct"),
    "RILD_RILM": PairSpec("RILD_RILM", "GR_RILD_RILM", "RILD", "RILM", "MEDIUM", "transfer"),
}

IDENTITY_COLUMNS = ["SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName"]
MODEL_FEATURE_COLUMNS = [
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
    "GRResidual1MZ",
    "GRResidual3MZ",
    "RDeepResidual1MZ",
    "RDeepResidual3MZ",
    "RReferenceResidual1MZ",
    "RReferenceResidual3MZ",
    "LateralContrast",
    "InductionContrast",
    "NormalizedContrast",
    "GRxDeepResidual",
    "GRxReferenceResidual",
    "PairIsLLD",
    "PairIsRD",
    "ReferenceIsMedium",
    "DeepNonPositiveFlag",
    "ReferenceNonPositiveFlag",
]

FORMAL_OUTPUT_COLUMNS = [
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
    "ModelID",
    "ValidationStatus",
    "ConfidenceLevel",
    "PredFractureProb",
    "PredConditionalDensity",
    "PredDensity",
    "PredHasFracture",
    "Stage1Threshold",
    "Density",
    "HasFracture",
    "PredictionValid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step4 GR/resistivity strata-library v2.")
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
    mask = values > 0.0
    out.loc[mask] = np.log10(values.loc[mask])
    return out


def rolling_residual(values: pd.Series, depth: pd.Series, window_m: float) -> pd.Series:
    values = numeric(values)
    depth_values = numeric(depth)
    spacing = depth_values.diff().abs().replace(0.0, np.nan).dropna()
    median_step = float(spacing.median()) if not spacing.empty else window_m
    rows = max(3, int(round(window_m / max(median_step, 1.0e-6))))
    if rows % 2 == 0:
        rows += 1
    baseline = values.rolling(rows, center=True, min_periods=max(2, rows // 3)).median()
    return values - baseline


def enrichment_path(step2_root: Path, well_name: str) -> Path:
    return step2_root / well_name / f"{well_name}_t4_t7_real_well_gr_resistivity.csv"


def find_training_enrichment(config: dict[str, Any], well_name: str) -> Path:
    inner = enrichment_path(Path(config["step2_output_root"]), well_name)
    if inner.exists():
        return inner
    outer = enrichment_path(Path(config["step2_outer_output_root"]), well_name)
    if outer.exists():
        return outer
    raise FileNotFoundError(f"missing Step2 enrichment for training well {well_name}")


def select_pair_rows(base: pd.DataFrame, enrichment: pd.DataFrame, pair_type: str) -> pd.DataFrame:
    spec = PAIR_SPECS[pair_type]
    required = ["DEPT", spec.gr_column, spec.deep_column, spec.reference_column]
    missing = set(required) - set(enrichment.columns)
    if missing:
        raise RuntimeError(f"enrichment missing columns for {pair_type}: {sorted(missing)}")
    pair_columns = enrichment[required].rename(
        columns={
            spec.gr_column: "_PairGR",
            spec.deep_column: "_PairDeep",
            spec.reference_column: "_PairReference",
        }
    )
    merged = base.merge(pair_columns, on="DEPT", how="left", validate="many_to_one")
    complete = merged[["_PairGR", "_PairDeep", "_PairReference"]].notna().all(axis=1)
    out = merged.loc[complete].copy()
    out["GR"] = numeric(out.pop("_PairGR"))
    out["RDeep"] = numeric(out.pop("_PairDeep"))
    out["RReference"] = numeric(out.pop("_PairReference"))
    out["PairType"] = spec.pair_type
    out["ReferenceType"] = spec.reference_type
    out["SourceGRCurve"] = spec.gr_column
    out["SourceDeepCurve"] = spec.deep_column
    out["SourceReferenceCurve"] = spec.reference_column
    return out


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    parts: list[pd.DataFrame] = []
    stats: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["WellName", "StrataName", "PairType"], dropna=False):
        well_name, strata_name, pair_type = keys
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
        for source, target in (
            ("GR", "GR"),
            ("RDeepSignedLog", "RDeep"),
            ("RReferenceSignedLog", "RReference"),
        ):
            for window_m in (1.0, 3.0):
                residual = rolling_residual(out[source], out["DEPT"], window_m)
                out[f"{target}Residual{int(window_m)}MZ"] = robust_z(residual)
        contrast = out["RDeepSignedLog"] - out["RReferenceSignedLog"]
        out["LateralContrast"] = contrast.where(out["ReferenceType"].eq("SHALLOW"))
        out["InductionContrast"] = contrast.where(out["ReferenceType"].eq("MEDIUM"))
        denominator = out["RDeep"].abs() + out["RReference"].abs()
        out["NormalizedContrast"] = (out["RDeep"] - out["RReference"]) / denominator.where(
            denominator > 1.0e-12
        )
        out["GRxDeepResidual"] = out["GRRobustZ"] * out["RDeepResidual1MZ"]
        out["GRxReferenceResidual"] = out["GRRobustZ"] * out["RReferenceResidual1MZ"]
        out["PairIsLLD"] = float(pair_type == "LLD_LLS")
        out["PairIsRD"] = float(pair_type == "RD_RS")
        out["ReferenceIsMedium"] = float(out["ReferenceType"].iloc[0] == "MEDIUM")
        out["DeepNonPositiveFlag"] = numeric(out["RDeep"]).le(0.0).astype(int)
        out["ReferenceNonPositiveFlag"] = numeric(out["RReference"]).le(0.0).astype(int)
        out["ModelInputEligible"] = out[["GR", "RDeep", "RReference"]].notna().all(axis=1)
        for column in ("GR", "RDeep", "RReference", "RDeepSignedLog", "RReferenceSignedLog"):
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


def assign_step2_strata(main: pd.DataFrame, interval: pd.Series) -> pd.DataFrame:
    out = main.copy()
    depth = numeric(out["DEPT"])
    out["StrataName"] = pd.NA
    out.loc[(depth >= float(interval["T4_DEPT"])) & (depth < float(interval["T6_DEPT"])), "StrataName"] = "沙三段"
    out.loc[(depth >= float(interval["T6_DEPT"])) & (depth <= float(interval["T7_DEPT"])), "StrataName"] = "沙四段"
    return out


def load_training_table(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups_dir = Path(config["step3_groups_dir"])
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    stats: list[pd.DataFrame] = []
    for source in config["training_sources"]:
        well_name = str(source["well_name"])
        pair_type = str(source["pair_type"])
        paths = sorted(groups_dir.glob(f"{well_name}_*.csv"))
        if not paths:
            raise RuntimeError(f"no Step3 groups for {well_name}")
        parts = []
        for path in paths:
            part = pd.read_csv(path)
            part["WellName"] = well_name
            part["SourceGroupFile"] = str(path)
            parts.append(part)
        base = pd.concat(parts, ignore_index=True)
        enrichment_file = find_training_enrichment(config, well_name)
        enrichment = pd.read_csv(enrichment_file)
        selected = select_pair_rows(base, enrichment, pair_type)
        featured, feature_stats = engineer_features(selected)
        usable = (
            featured["StrataName"].isin(TARGET_STRATA)
            & numeric(featured["Density"]).notna()
            & featured["ModelInputEligible"]
        )
        featured = featured.loc[usable].copy()
        featured["HasFracture"] = numeric(featured["Density"]).fillna(0.0).gt(0.0).astype(int)
        frames.append(featured)
        stats.append(feature_stats.assign(DataRole="training"))
        manifests.append(
            {
                "WellName": well_name,
                "PairType": pair_type,
                "EnrichmentPath": str(enrichment_file),
                "Step3Rows": int(len(base)),
                "EligibleRows": int(len(featured)),
                "DensityPositiveRows": int(featured["HasFracture"].sum()),
                "PointRows": int(numeric(featured.get("GT_POINT_FLAG", pd.Series(index=featured.index))).fillna(0).gt(0).sum()),
            }
        )
    training = pd.concat(frames, ignore_index=True)
    normalization = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()
    return training, pd.DataFrame(manifests), normalization


def equalized_depth_weights(frame: pd.DataFrame, bin_m: float, balance_classes: bool = False) -> np.ndarray:
    work = frame[["WellName", "StrataName", "DEPT"]].copy()
    work["DepthBin"] = np.floor(numeric(work["DEPT"]) / bin_m).astype("Int64")
    count = work.groupby(["WellName", "StrataName", "DepthBin"], dropna=False)["DEPT"].transform("size")
    weights = 1.0 / count.clip(lower=1).to_numpy(dtype=float)
    temp = pd.DataFrame({"WellName": frame["WellName"].astype(str).to_numpy(), "Weight": weights})
    well_total = temp.groupby("WellName")["Weight"].transform("sum").to_numpy(dtype=float)
    weights = weights / np.maximum(well_total, 1.0e-12)
    if balance_classes:
        labels = numeric(frame["HasFracture"]).fillna(0).astype(int).to_numpy()
        positive = max(int(labels.sum()), 1)
        negative = max(int(len(labels) - labels.sum()), 1)
        weights = weights * np.where(labels > 0, negative / positive, 1.0)
    return weights * (len(weights) / max(weights.sum(), 1.0e-12))


def xgb_classifier(config: dict[str, Any], random_state: int) -> XGBClassifier:
    params = dict(config["stage1_params"])
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=int(config.get("xgb_n_jobs", 2)),
        random_state=random_state,
        **params,
    )


def xgb_regressor(config: dict[str, Any], random_state: int, transfer: bool = False) -> XGBRegressor:
    params = dict(config["transfer_params"] if transfer else config["stage2_params"])
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=int(config.get("xgb_n_jobs", 2)),
        random_state=random_state,
        **params,
    )


def fit_direct_models(frame: pd.DataFrame, config: dict[str, Any], random_state: int) -> tuple[Any, Any]:
    x = frame[MODEL_FEATURE_COLUMNS]
    labels = numeric(frame["HasFracture"]).fillna(0).astype(int)
    cls_weights = equalized_depth_weights(frame, float(config["depth_weight_bin_m"]), balance_classes=True)
    if labels.nunique() < 2:
        classifier: Any = DummyClassifier(strategy="constant", constant=int(labels.iloc[0]))
    else:
        classifier = xgb_classifier(config, random_state)
    classifier.fit(x, labels, sample_weight=cls_weights)

    positive = frame[numeric(frame["Density"]).fillna(0.0).gt(0.0)].copy()
    if positive.empty:
        regressor: Any = DummyRegressor(strategy="constant", constant=0.0)
        regressor.fit(np.zeros((1, len(MODEL_FEATURE_COLUMNS))), [0.0])
    else:
        regressor = xgb_regressor(config, random_state + 1)
        reg_weights = equalized_depth_weights(positive, float(config["depth_weight_bin_m"]))
        regressor.fit(
            positive[MODEL_FEATURE_COLUMNS],
            np.log1p(numeric(positive["Density"]).clip(lower=0.0)),
            sample_weight=reg_weights,
        )
    return classifier, regressor


def predict_direct(frame: pd.DataFrame, classifier: Any, regressor: Any) -> tuple[np.ndarray, np.ndarray]:
    x = frame[MODEL_FEATURE_COLUMNS]
    probability_matrix = classifier.predict_proba(x)
    classes = list(getattr(classifier, "classes_", []))
    probability = probability_matrix[:, classes.index(1)] if 1 in classes else np.zeros(len(frame), dtype=float)
    conditional = np.expm1(regressor.predict(x))
    return np.clip(probability, 0.0, 1.0), np.maximum(conditional, 0.0)


def select_threshold(actual_density: pd.Series, probability: pd.Series, strategy: str) -> float:
    actual = numeric(actual_density).fillna(0.0).gt(0.0).astype(int).to_numpy()
    prob = numeric(probability).fillna(0.0).to_numpy(dtype=float)
    if np.unique(actual).size < 2:
        return 0.5
    candidates = np.linspace(0.05, 0.95, 91)
    scores = np.asarray([f1_score(actual, prob >= value, zero_division=0) for value in candidates])
    if strategy == "match_oof_prevalence":
        target = float(actual.mean())
        gap = np.asarray([abs(float((prob >= value).mean()) - target) for value in candidates])
        return float(candidates[np.lexsort((-scores, gap))[0]])
    return float(candidates[int(np.argmax(scores))])


def metric_row(actual_density: pd.Series, probability: np.ndarray, conditional: np.ndarray, threshold: float) -> dict[str, Any]:
    actual = numeric(actual_density).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    actual_flag = (actual > 0.0).astype(int)
    pred_flag = (probability >= threshold).astype(int)
    expected = probability * conditional
    row: dict[str, Any] = {
        "RowCount": int(len(actual)),
        "ActualPositiveRows": int(actual_flag.sum()),
        "PredictedPositiveRows": int(pred_flag.sum()),
        "Accuracy": float(accuracy_score(actual_flag, pred_flag)),
        "Precision": float(precision_score(actual_flag, pred_flag, zero_division=0)),
        "Recall": float(recall_score(actual_flag, pred_flag, zero_division=0)),
        "F1": float(f1_score(actual_flag, pred_flag, zero_division=0)),
        "DensityMAE": float(mean_absolute_error(actual, expected)),
        "DensityRMSE": float(np.sqrt(mean_squared_error(actual, expected))),
        "DensityR2": float(r2_score(actual, expected)) if len(actual) >= 2 else np.nan,
        "ActualPositiveRatio": float(actual_flag.mean()),
        "PredictedPositiveRatio": float(pred_flag.mean()),
    }
    if np.unique(actual_flag).size >= 2:
        row["PresenceROC_AUC"] = float(roc_auc_score(actual_flag, probability))
        row["PresenceAveragePrecision"] = float(average_precision_score(actual_flag, probability))
    else:
        row["PresenceROC_AUC"] = np.nan
        row["PresenceAveragePrecision"] = np.nan
    return row


def train_direct_library(
    training: pd.DataFrame,
    config: dict[str, Any],
    model_dir: Path,
    validation_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], pd.DataFrame, pd.DataFrame]:
    random_state = int(config.get("random_state", 42))
    oof_parts: list[pd.DataFrame] = []
    wells = sorted(training["WellName"].astype(str).unique())
    for fold_index, holdout in enumerate(wells):
        for strata_name in TARGET_STRATA:
            train_part = training[(training["WellName"] != holdout) & training["StrataName"].eq(strata_name)].copy()
            test_part = training[(training["WellName"] == holdout) & training["StrataName"].eq(strata_name)].copy()
            if train_part.empty or test_part.empty:
                continue
            classifier, regressor = fit_direct_models(train_part, config, random_state + fold_index * 10)
            probability, conditional = predict_direct(test_part, classifier, regressor)
            fold = test_part[["SampleID", "WellName", "StrataName", "PairType", "DEPT", "Density", "HasFracture"]].copy()
            fold["PredFractureProb"] = probability
            fold["PredConditionalDensity"] = conditional
            fold["PredDensity"] = probability * conditional
            fold["HoldoutWell"] = holdout
            oof_parts.append(fold)
    if not oof_parts:
        raise RuntimeError("direct-model leave-one-well-out produced no predictions")
    oof = pd.concat(oof_parts, ignore_index=True)
    oof_thresholds = {
        str(strata): select_threshold(part["Density"], part["PredFractureProb"], str(config["threshold_strategy"]))
        for strata, part in oof.groupby("StrataName")
    }
    density_scales: dict[str, float] = {}
    for strata_name, part in oof.groupby("StrataName"):
        actual_total = float(numeric(part["Density"]).fillna(0.0).clip(lower=0.0).sum())
        predicted_total = float(numeric(part["PredDensity"]).fillna(0.0).clip(lower=0.0).sum())
        density_scales[str(strata_name)] = actual_total / predicted_total if predicted_total > 1.0e-12 else 1.0
    oof["DensityScale"] = oof["StrataName"].map(density_scales)
    oof["PredConditionalDensity"] = oof["PredConditionalDensity"] * oof["DensityScale"]
    oof["PredDensity"] = oof["PredFractureProb"] * oof["PredConditionalDensity"]
    metric_rows = []
    for (well_name, strata_name), part in oof.groupby(["WellName", "StrataName"]):
        threshold = float(oof_thresholds[str(strata_name)])
        metric_rows.append(
            {
                "HoldoutWell": well_name,
                "StrataName": strata_name,
                "PairType": str(part["PairType"].iloc[0]),
                "Threshold": threshold,
                **metric_row(
                    part["Density"],
                    part["PredFractureProb"].to_numpy(dtype=float),
                    part["PredConditionalDensity"].to_numpy(dtype=float),
                    threshold,
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    models: dict[str, dict[str, Any]] = {}
    deployment_thresholds: dict[str, float] = {}
    registry_rows: list[dict[str, Any]] = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for index, strata_name in enumerate(TARGET_STRATA):
        subset = training[training["StrataName"].eq(strata_name)].copy()
        if subset.empty:
            continue
        classifier, regressor = fit_direct_models(subset, config, random_state + 100 + index * 10)
        training_probability, _ = predict_direct(subset, classifier, regressor)
        deployment_threshold = select_threshold(
            subset["Density"], pd.Series(training_probability, index=subset.index), str(config["threshold_strategy"])
        )
        deployment_thresholds[strata_name] = float(deployment_threshold)
        stage1_path = model_dir / f"{strata_name}_direct_stage1_xgb.joblib"
        stage2_path = model_dir / f"{strata_name}_direct_stage2_xgb.joblib"
        joblib.dump(classifier, stage1_path)
        joblib.dump(regressor, stage2_path)
        model_id = f"{strata_name}_direct_lateral_xgb_v2"
        models[strata_name] = {
            "stage1": classifier,
            "stage2": regressor,
            "threshold": float(deployment_threshold),
            "oof_threshold": float(oof_thresholds[strata_name]),
            "density_scale": float(density_scales[strata_name]),
            "model_id": model_id,
            "validation_status": "DirectLimited",
            "confidence_level": "DirectLimited",
        }
        strata_metrics = metrics[metrics["StrataName"].eq(strata_name)]
        registry_rows.append(
            {
                "ModelID": model_id,
                "StrataName": strata_name,
                "PairDomain": "LLD_LLS;RD_RS",
                "TrainingMode": "direct_imaging_supervision",
                "TrainingWells": ";".join(sorted(subset["WellName"].astype(str).unique())),
                "TrainingRows": int(len(subset)),
                "PositiveRows": int(subset["HasFracture"].sum()),
                "OOFStage1Threshold": float(oof_thresholds[strata_name]),
                "DeploymentStage1Threshold": float(deployment_threshold),
                "OOFDensityScale": float(density_scales[strata_name]),
                "MeanLOOROCAUC": float(strata_metrics["PresenceROC_AUC"].mean()),
                "MeanLOOAveragePrecision": float(strata_metrics["PresenceAveragePrecision"].mean()),
                "ValidationStatus": "DirectLimited",
                "DownstreamEligible": True,
                "Stage1ModelPath": str(Path("model_library") / stage1_path.name),
                "Stage2ModelPath": str(Path("model_library") / stage2_path.name),
            }
        )
    pd.DataFrame(registry_rows).to_csv(model_dir / "direct_model_registry.csv", index=False, encoding="utf-8-sig")
    validation_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(validation_dir / "direct_leave_one_well_out_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(validation_dir / "direct_leave_one_well_out_metrics.csv", index=False, encoding="utf-8-sig")
    return models, deployment_thresholds, oof, metrics


def bridge_training_rows(config: dict[str, Any], direct_models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    root = Path(config["step2_output_root"])
    frames: list[pd.DataFrame] = []
    for well_name in config["bridge_wells"]:
        well_dir = root / str(well_name)
        main = pd.read_csv(well_dir / f"{well_name}_t4_t7_real_well_main.csv")
        interval = pd.read_csv(well_dir / f"{well_name}_t4_t7_real_well_interval.csv").iloc[0]
        enrichment = pd.read_csv(enrichment_path(root, str(well_name)))
        main = assign_step2_strata(main, interval)
        rd = select_pair_rows(main, enrichment, "RD_RS")
        rild = select_pair_rows(main, enrichment, "RILD_RILM")
        common = sorted(set(rd["DEPT"]).intersection(set(rild["DEPT"])))
        if not common:
            continue
        rd = rd[rd["DEPT"].isin(common)].copy()
        rild = rild[rild["DEPT"].isin(common)].copy()
        rd_features, _ = engineer_features(rd)
        rild_features, _ = engineer_features(rild)
        teacher_parts = []
        for strata_name in TARGET_STRATA:
            subset = rd_features[rd_features["StrataName"].eq(strata_name)].copy()
            model = direct_models.get(strata_name)
            if subset.empty or model is None:
                continue
            probability, conditional = predict_direct(subset, model["stage1"], model["stage2"])
            conditional = conditional * float(model.get("density_scale", 1.0))
            teacher = subset[["WellName", "StrataName", "DEPT"]].copy()
            teacher["TeacherFractureProb"] = probability
            teacher["TeacherConditionalDensity"] = conditional
            teacher_parts.append(teacher)
        if not teacher_parts:
            continue
        teacher = pd.concat(teacher_parts, ignore_index=True)
        student = rild_features.merge(teacher, on=["WellName", "StrataName", "DEPT"], how="inner", validate="one_to_one")
        frames.append(student)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_transfer_pair(frame: pd.DataFrame, config: dict[str, Any], random_state: int) -> tuple[Any, Any]:
    probability_model = xgb_regressor(config, random_state, transfer=True)
    density_model = xgb_regressor(config, random_state + 1, transfer=True)
    weights = equalized_depth_weights(frame, float(config["depth_weight_bin_m"]))
    probability_model.fit(frame[MODEL_FEATURE_COLUMNS], numeric(frame["TeacherFractureProb"]), sample_weight=weights)
    density_model.fit(
        frame[MODEL_FEATURE_COLUMNS],
        np.log1p(numeric(frame["TeacherConditionalDensity"]).clip(lower=0.0)),
        sample_weight=weights,
    )
    return probability_model, density_model


def train_transfer_library(
    bridge: pd.DataFrame,
    config: dict[str, Any],
    model_dir: Path,
    validation_dir: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if bridge.empty:
        return {}, pd.DataFrame()
    random_state = int(config.get("random_state", 42)) + 500
    validation_rows: list[dict[str, Any]] = []
    wells = sorted(bridge["WellName"].astype(str).unique())
    for fold_index, holdout in enumerate(wells):
        for strata_name in TARGET_STRATA:
            train = bridge[(bridge["WellName"] != holdout) & bridge["StrataName"].eq(strata_name)].copy()
            test = bridge[(bridge["WellName"] == holdout) & bridge["StrataName"].eq(strata_name)].copy()
            if len(train) < 20 or test.empty:
                continue
            prob_model, density_model = fit_transfer_pair(train, config, random_state + fold_index * 10)
            pred_prob = np.clip(prob_model.predict(test[MODEL_FEATURE_COLUMNS]), 0.0, 1.0)
            pred_density = np.maximum(np.expm1(density_model.predict(test[MODEL_FEATURE_COLUMNS])), 0.0)
            true_prob = numeric(test["TeacherFractureProb"]).to_numpy(dtype=float)
            true_density = numeric(test["TeacherConditionalDensity"]).to_numpy(dtype=float)
            validation_rows.append(
                {
                    "HoldoutBridgeWell": holdout,
                    "StrataName": strata_name,
                    "RowCount": int(len(test)),
                    "ProbabilityMAE": float(mean_absolute_error(true_prob, pred_prob)),
                    "ProbabilityR2": float(r2_score(true_prob, pred_prob)) if len(test) >= 2 else np.nan,
                    "ConditionalDensityMAE": float(mean_absolute_error(true_density, pred_density)),
                    "ConditionalDensityR2": float(r2_score(true_density, pred_density)) if len(test) >= 2 else np.nan,
                }
            )
    transfer_models: dict[str, dict[str, Any]] = {}
    registry_rows = []
    for index, strata_name in enumerate(TARGET_STRATA):
        subset = bridge[bridge["StrataName"].eq(strata_name)].copy()
        if len(subset) < 20:
            continue
        prob_model, density_model = fit_transfer_pair(subset, config, random_state + 100 + index * 10)
        prob_path = model_dir / f"{strata_name}_rild_transfer_probability_xgb.joblib"
        density_path = model_dir / f"{strata_name}_rild_transfer_density_xgb.joblib"
        joblib.dump(prob_model, prob_path)
        joblib.dump(density_model, density_path)
        model_id = f"{strata_name}_rild_rilm_transferred_xgb_v2"
        transfer_models[strata_name] = {
            "probability": prob_model,
            "density": density_model,
            "model_id": model_id,
            "validation_status": "TransferredExperimental",
            "confidence_level": "TransferredExperimental",
        }
        registry_rows.append(
            {
                "ModelID": model_id,
                "StrataName": strata_name,
                "PairDomain": "RILD_RILM",
                "TrainingMode": "RD_RS_teacher_to_RILD_RILM_student",
                "BridgeWells": ";".join(sorted(subset["WellName"].astype(str).unique())),
                "TrainingRows": int(len(subset)),
                "ValidationStatus": "TransferredExperimental",
                "DownstreamEligible": False,
                "ProbabilityModelPath": str(Path("model_library") / prob_path.name),
                "DensityModelPath": str(Path("model_library") / density_path.name),
            }
        )
    pd.DataFrame(registry_rows).to_csv(model_dir / "transfer_model_registry.csv", index=False, encoding="utf-8-sig")
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(validation_dir / "rild_transfer_bridge_validation.csv", index=False, encoding="utf-8-sig")
    return transfer_models, validation


def predict_transfer(frame: pd.DataFrame, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(model["probability"].predict(frame[MODEL_FEATURE_COLUMNS]), 0.0, 1.0)
    conditional = np.maximum(np.expm1(model["density"].predict(frame[MODEL_FEATURE_COLUMNS])), 0.0)
    return probability, conditional


def add_support_segments(frame: pd.DataFrame, max_gap_m: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    parts = []
    for (well_name, strata_name), group in frame.groupby(["WellName", "StrataName"], sort=False):
        out = group.sort_values("DEPT").copy()
        gap = numeric(out["DEPT"]).diff()
        pair_change = out["InputPairType"].ne(out["InputPairType"].shift())
        new_segment = gap.isna() | gap.gt(max_gap_m) | gap.le(0.0) | pair_change
        local_id = new_segment.cumsum().astype(int)
        out["SupportSegmentID"] = [f"{well_name}_{strata_name}_{value:03d}" for value in local_id]
        bounds = out.groupby("SupportSegmentID")["DEPT"].agg(["min", "max"])
        out["SegmentStartDEPT"] = out["SupportSegmentID"].map(bounds["min"])
        out["SegmentEndDEPT"] = out["SupportSegmentID"].map(bounds["max"])
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def build_prediction_rows(
    main: pd.DataFrame,
    enrichment: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    pair_types: list[str],
    formal: bool,
) -> pd.DataFrame:
    candidates = []
    for priority, pair_type in enumerate(pair_types):
        selected = select_pair_rows(main, enrichment, pair_type)
        featured, _ = engineer_features(selected)
        if featured.empty:
            continue
        spatial = featured[["X", "Y", "TIME", "TVD", "DEPT"]].apply(numeric).notna().all(axis=1)
        featured = featured[featured["StrataName"].isin(TARGET_STRATA) & spatial].copy()
        for strata_name in TARGET_STRATA:
            subset = featured[featured["StrataName"].eq(strata_name)].copy()
            model = models.get(strata_name)
            if subset.empty or model is None:
                continue
            if formal:
                probability, conditional = predict_direct(subset, model["stage1"], model["stage2"])
                conditional = conditional * float(model.get("density_scale", 1.0))
                threshold = float(model["threshold"])
            else:
                probability, conditional = predict_transfer(subset, model)
                threshold = 0.5
            subset["PredFractureProb"] = probability
            subset["PredConditionalDensity"] = conditional
            subset["PredDensity"] = probability * conditional
            subset["PredHasFracture"] = (probability >= threshold).astype(int)
            subset["Stage1Threshold"] = threshold
            subset["Density"] = subset["PredDensity"]
            subset["HasFracture"] = subset["PredHasFracture"]
            subset["InputPairType"] = pair_type
            subset["ModelID"] = model["model_id"]
            subset["ValidationStatus"] = model["validation_status"]
            subset["ConfidenceLevel"] = model["confidence_level"]
            subset["PredictionValid"] = 1
            subset["PairPriority"] = priority
            candidates.append(subset)
    if not candidates:
        return pd.DataFrame()
    combined = pd.concat(candidates, ignore_index=True)
    combined = combined.sort_values(["WellName", "DEPT", "PairPriority"]).drop_duplicates(
        ["WellName", "DEPT"], keep="first"
    )
    combined["SourceSampleID"] = combined["SampleID"]
    return combined


def point_count_scales(training: pd.DataFrame) -> dict[str, float]:
    scales = {}
    for strata_name, strata in training.groupby("StrataName"):
        area = 0.0
        for _, well in strata.groupby("WellName"):
            ordered = well.sort_values("DEPT")
            depth = numeric(ordered["DEPT"]).to_numpy(dtype=float)
            density = numeric(ordered["Density"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            if len(depth) >= 2:
                area += float(np.trapezoid(density, depth))
        points = int(numeric(strata.get("GT_POINT_FLAG", pd.Series(index=strata.index))).fillna(0).gt(0).sum())
        scales[str(strata_name)] = float(points / area) if area > 1.0e-9 else 1.0
    return scales


def refine_points(predicted: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    columns = [
        "SourceSampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName",
        "SupportSegmentID", "InputPairType", "ModelID", "Density", "HasFracture",
        "PredFractureProb", "SegmentStartDEPT", "SegmentEndDEPT", "PointOrdinalInSegment",
    ]
    if predicted.empty:
        return pd.DataFrame(columns=columns)
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
            density = numeric(segment["Density"]).fillna(0.0).to_numpy(dtype=float)
            if len(depth) >= 2:
                area = float(np.trapezoid(density, depth))
            else:
                area = float(density[0]) * 0.125
            count = min(len(segment), max(1, int(round(area * float(scales.get(str(segment["StrataName"].iloc[0]), 1.0))))))
            order = np.argsort(-density)[:count]
            for ordinal, local_index in enumerate(sorted(order.tolist()), start=1):
                source = segment.iloc[int(local_index)]
                rows.append(
                    {
                        "SourceSampleID": source["SourceSampleID"],
                        "WellName": source["WellName"],
                        "X": source["X"],
                        "Y": source["Y"],
                        "TIME": source["TIME"],
                        "TVD": source["TVD"],
                        "DEPT": source["DEPT"],
                        "StrataName": source["StrataName"],
                        "SupportSegmentID": support_id,
                        "InputPairType": source["InputPairType"],
                        "ModelID": source["ModelID"],
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
    depth = numeric(main["DEPT"])
    target = main["StrataName"].isin(TARGET_STRATA) & depth.notna()
    predicted_depth = set(numeric(predicted.get("DEPT", pd.Series(dtype=float))).dropna().tolist())
    unsupported = target & ~depth.isin(predicted_depth)
    if not unsupported.any():
        return pd.DataFrame()
    rows = main.loc[target, ["DEPT", "StrataName"]].copy().sort_values("DEPT")
    rows["Unsupported"] = ~numeric(rows["DEPT"]).isin(predicted_depth)
    rows["Run"] = rows["Unsupported"].ne(rows["Unsupported"].shift()).cumsum()
    output = []
    for _, segment in rows[rows["Unsupported"]].groupby("Run"):
        output.append(
            {
                "WellName": well_name,
                "StrataName": str(segment["StrataName"].iloc[0]),
                "StartDEPT": float(segment["DEPT"].min()),
                "EndDEPT": float(segment["DEPT"].max()),
                "RowCount": int(len(segment)),
                "Reason": "no_complete_approved_pair_or_invalid_spatial_coordinates",
            }
        )
    return pd.DataFrame(output)


def predict_all_wells(
    config: dict[str, Any],
    direct_models: dict[str, dict[str, Any]],
    transfer_models: dict[str, dict[str, Any]],
    training: pd.DataFrame,
    staging: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(config["step2_output_root"])
    summary = pd.read_csv(root / "real_well_t4_t7_summary.csv")
    formal_root = staging / "formal_predictions"
    experimental_root = staging / "experimental_predictions" / "rild_rilm_transferred"
    per_well_root = formal_root / "real_well_predictions"
    formal_root.mkdir(parents=True, exist_ok=True)
    experimental_root.mkdir(parents=True, exist_ok=True)
    per_well_root.mkdir(parents=True, exist_ok=True)
    formal_parts = []
    experimental_parts = []
    coverage_rows = []
    unsupported_parts = []
    scales = point_count_scales(training)

    for _, row in summary.iterrows():
        well_name = str(row["WellName"])
        status = str(row["Status"])
        if status != "ok":
            coverage_rows.append({"WellName": well_name, "Status": f"step2_{status}", "TotalRows": 0, "FormalPredictedRows": 0, "ExperimentalPredictedRows": 0})
            continue
        well_dir = root / well_name
        main = pd.read_csv(Path(str(row["MainCsv"])))
        interval = pd.read_csv(Path(str(row["IntervalCsv"]))).iloc[0]
        main = assign_step2_strata(main, interval)
        enrich_path = enrichment_path(root, well_name)
        if not enrich_path.exists():
            coverage_rows.append({"WellName": well_name, "Status": "missing_enrichment", "TotalRows": int(len(main)), "FormalPredictedRows": 0, "ExperimentalPredictedRows": 0})
            continue
        enrichment = pd.read_csv(enrich_path)
        formal = build_prediction_rows(
            main,
            enrichment,
            direct_models,
            [str(value) for value in config["formal_pair_priority"]],
            formal=True,
        )
        formal = add_support_segments(formal, float(config["max_support_gap_m"]))
        experimental = build_prediction_rows(
            main,
            enrichment,
            transfer_models,
            [str(config["experimental_pair_type"])],
            formal=False,
        )
        experimental = add_support_segments(experimental, float(config["max_support_gap_m"]))
        if not formal.empty:
            formal[FORMAL_OUTPUT_COLUMNS].to_csv(
                per_well_root / f"{well_name}_t4_t7_density_prediction.csv", index=False, encoding="utf-8-sig"
            )
            formal_parts.append(formal[FORMAL_OUTPUT_COLUMNS])
        if not experimental.empty:
            experimental[FORMAL_OUTPUT_COLUMNS].to_csv(
                experimental_root / f"{well_name}_t4_t7_rild_transfer_prediction.csv", index=False, encoding="utf-8-sig"
            )
            experimental_parts.append(experimental[FORMAL_OUTPUT_COLUMNS])
        unsupported = unsupported_intervals(main, formal, well_name)
        if not unsupported.empty:
            unsupported_parts.append(unsupported)
        pair_counts = {}
        for pair_type, spec in PAIR_SPECS.items():
            count = enrichment[[spec.gr_column, spec.deep_column, spec.reference_column]].notna().all(axis=1)
            pair_counts[f"{pair_type}_Rows"] = int(count.sum())
        coverage_rows.append(
            {
                "WellName": well_name,
                "Status": "formal_predicted" if not formal.empty else ("experimental_only" if not experimental.empty else "no_supported_prediction"),
                "TotalRows": int(len(main)),
                "FormalPredictedRows": int(len(formal)),
                "ExperimentalPredictedRows": int(len(experimental)),
                "FormalSupportSegments": int(formal["SupportSegmentID"].nunique()) if not formal.empty else 0,
                **pair_counts,
            }
        )
        print(f"[Step4-v2] {well_name}: formal={len(formal)} experimental={len(experimental)}", flush=True)

    formal_all = pd.concat(formal_parts, ignore_index=True) if formal_parts else pd.DataFrame(columns=FORMAL_OUTPUT_COLUMNS)
    experimental_all = pd.concat(experimental_parts, ignore_index=True) if experimental_parts else pd.DataFrame(columns=FORMAL_OUTPUT_COLUMNS)
    points = refine_points(formal_all, scales) if not formal_all.empty else refine_points(pd.DataFrame(), scales)
    coverage = pd.DataFrame(coverage_rows)
    unsupported = pd.concat(unsupported_parts, ignore_index=True) if unsupported_parts else pd.DataFrame(
        columns=["WellName", "StrataName", "StartDEPT", "EndDEPT", "RowCount", "Reason"]
    )
    formal_all.to_csv(formal_root / "all_wells_t4_t7_density_prediction.csv", index=False, encoding="utf-8-sig")
    points.to_csv(formal_root / "all_wells_t4_t7_fracture_points.csv", index=False, encoding="utf-8-sig")
    experimental_all.to_csv(experimental_root / "all_wells_t4_t7_rild_transfer_prediction.csv", index=False, encoding="utf-8-sig")
    qc_root = staging / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(qc_root / "well_prediction_coverage.csv", index=False, encoding="utf-8-sig")
    unsupported.to_csv(qc_root / "unsupported_interval_manifest.csv", index=False, encoding="utf-8-sig")
    return formal_all, points, experimental_all, coverage


def validate_delivery(
    config: dict[str, Any],
    training: pd.DataFrame,
    metrics: pd.DataFrame,
    formal: pd.DataFrame,
    points: pd.DataFrame,
    experimental: pd.DataFrame,
    coverage: pd.DataFrame,
    staging: Path,
    published_output: Path,
) -> dict[str, Any]:
    formal_pairs = set(str(value) for value in config["formal_pair_priority"])
    source_support_ok = True
    source_support_errors: list[str] = []
    step2_root = Path(config["step2_output_root"])
    for (well_name, pair_type), part in formal.groupby(["WellName", "InputPairType"]):
        spec = PAIR_SPECS[str(pair_type)]
        source = pd.read_csv(
            enrichment_path(step2_root, str(well_name)),
            usecols=["DEPT", spec.gr_column, spec.deep_column, spec.reference_column],
        )
        joined = part[["DEPT"]].merge(source, on="DEPT", how="left", validate="many_to_one")
        complete = joined[[spec.gr_column, spec.deep_column, spec.reference_column]].notna().all(axis=1)
        if not complete.all():
            source_support_ok = False
            source_support_errors.append(f"{well_name}:{pair_type}:missing_source_triple={int((~complete).sum())}")
    support_gap_ok = True
    for support_id, part in formal.groupby("SupportSegmentID"):
        depth = np.sort(numeric(part["DEPT"]).to_numpy(dtype=float))
        if len(depth) > 1 and np.diff(depth).max() > float(config["max_support_gap_m"]) + 1.0e-9:
            support_gap_ok = False
            source_support_errors.append(f"{support_id}:support_gap_exceeded")
    point_bounds_ok = True
    if not points.empty:
        point_bounds_ok = bool(
            (
                numeric(points["DEPT"]).ge(numeric(points["SegmentStartDEPT"]))
                & numeric(points["DEPT"]).le(numeric(points["SegmentEndDEPT"]))
            ).all()
        )
    checks = {
        "isolated_v2_output": published_output.name not in {"formal_gr_resistivity_lateral_v1", "formal_well_expert_library"},
        "training_uses_four_imaging_wells": set(training["WellName"].astype(str).unique()) == {"车页1导眼", "车151HF", "车662", "车663"},
        "formal_predictions_nonempty": not formal.empty,
        "formal_rows_unique": not formal.duplicated(["WellName", "DEPT"]).any(),
        "formal_pairs_direct_only": set(formal["InputPairType"].astype(str).unique()).issubset(formal_pairs),
        "formal_prediction_valid_only": formal["PredictionValid"].eq(1).all(),
        "formal_density_finite": np.isfinite(numeric(formal["Density"])).all(),
        "formal_coordinates_finite": np.isfinite(formal[["X", "Y", "TIME", "TVD", "DEPT"]].apply(numeric).to_numpy(dtype=float)).all(),
        "formal_support_segments_present": formal["SupportSegmentID"].astype(str).str.len().gt(0).all(),
        "formal_source_triples_complete": source_support_ok,
        "formal_support_gaps_bounded": support_gap_ok,
        "experimental_rild_only": experimental.empty or set(experimental["InputPairType"].astype(str).unique()) == {"RILD_RILM"},
        "experimental_not_in_formal": "RILD_RILM" not in set(formal["InputPairType"].astype(str).unique()),
        "points_inside_support": points.empty or points["SupportSegmentID"].isin(set(formal["SupportSegmentID"])).all(),
        "points_inside_segment_bounds": point_bounds_ok,
        "coverage_matches_formal_rows": int(coverage["FormalPredictedRows"].fillna(0).sum()) == int(len(formal)),
        "qc_has_no_prediction_targets": not any(column in coverage.columns for column in ("Density", "HasFracture")),
    }
    summary = {
        "status": "pass" if all(bool(value) for value in checks.values()) else "fail",
        "output_dir": str(published_output),
        "model_family": "two-stage XGBoost strata library with direct lateral supervision and experimental RILD transfer",
        "training_rows": int(len(training)),
        "training_wells": sorted(training["WellName"].astype(str).unique()),
        "formal_predicted_wells": int(coverage["FormalPredictedRows"].fillna(0).gt(0).sum()),
        "formal_predicted_rows": int(len(formal)),
        "formal_fracture_points": int(len(points)),
        "experimental_predicted_wells": int(coverage["ExperimentalPredictedRows"].fillna(0).gt(0).sum()),
        "experimental_predicted_rows": int(len(experimental)),
        "mean_loo_auc_by_strata": metrics.groupby("StrataName")["PresenceROC_AUC"].mean().to_dict(),
        "source_support_errors": source_support_errors,
        "checks": checks,
    }
    write_json(staging / "step4_strata_library_acceptance_summary.json", summary)
    manifest = {
        "version": "formal_gr_resistivity_strata_library_v2",
        "prediction_table": "formal_predictions/all_wells_t4_t7_density_prediction.csv",
        "fracture_point_table": "formal_predictions/all_wells_t4_t7_fracture_points.csv",
        "prediction_valid_column": "PredictionValid",
        "prediction_valid_value": 1,
        "formal_pair_types": list(config["formal_pair_priority"]),
        "experimental_results_allowed": False,
        "experimental_prediction_table": "experimental_predictions/rild_rilm_transferred/all_wells_t4_t7_rild_transfer_prediction.csv",
        "qc_coverage_table": "qc/well_prediction_coverage.csv",
        "acceptance_summary": "step4_strata_library_acceptance_summary.json",
    }
    write_json(staging / "formal_prediction_manifest.json", manifest)
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
        for name in ("model_library", "validation", "qc"):
            (staging / name).mkdir(parents=True, exist_ok=True)
        training, training_manifest, normalization = load_training_table(config)
        training.to_csv(staging / "training_feature_table.csv", index=False, encoding="utf-8-sig")
        training_manifest.to_csv(staging / "training_input_manifest.csv", index=False, encoding="utf-8-sig")
        normalization.to_csv(staging / "training_normalization_stats.csv", index=False, encoding="utf-8-sig")
        print(f"[Step4-v2] training rows={len(training)} wells={training['WellName'].nunique()}", flush=True)

        direct_models, thresholds, oof, metrics = train_direct_library(
            training, config, staging / "model_library", staging / "validation"
        )
        bridge = bridge_training_rows(config, direct_models)
        bridge.to_csv(staging / "bridge_teacher_student_training_table.csv", index=False, encoding="utf-8-sig")
        transfer_models, transfer_validation = train_transfer_library(
            bridge, config, staging / "model_library", staging / "validation"
        )
        write_json(staging / "model_thresholds.json", {"thresholds_by_strata": thresholds})

        if train_only:
            formal = pd.DataFrame(columns=FORMAL_OUTPUT_COLUMNS)
            points = refine_points(pd.DataFrame(), {})
            experimental = pd.DataFrame(columns=FORMAL_OUTPUT_COLUMNS)
            coverage = pd.DataFrame(columns=["WellName", "FormalPredictedRows", "ExperimentalPredictedRows"])
            summary = {
                "status": "train_only",
                "training_rows": int(len(training)),
                "direct_models": sorted(direct_models),
                "transfer_models": sorted(transfer_models),
            }
            write_json(staging / "step4_strata_library_acceptance_summary.json", summary)
            write_json(staging / "run_config_snapshot.json", config)
            publish(staging, output_dir)
            return summary

        formal, points, experimental, coverage = predict_all_wells(
            config, direct_models, transfer_models, training, staging
        )
        summary = validate_delivery(
            config, training, metrics, formal, points, experimental, coverage, staging, output_dir
        )
        if summary["status"] != "pass":
            raise RuntimeError(f"Step4 v2 acceptance failed: {summary['checks']}")
        write_json(staging / "run_config_snapshot.json", config)
        publish(staging, output_dir)
        return summary
    except Exception as exc:
        print(f"[Step4-v2] failed before publication: {type(exc).__name__}: {exc}", flush=True)
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
