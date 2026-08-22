#!/usr/bin/env python3
"""TaiGuJie Step4 GR/RD/RS two-stage fracture prediction (v1).

Two experts (上部复合层 / 太古界风化壳), each a two-stage XGBoost:
Stage1 presence classifier, Stage2 conditional density regressor.
Prediction is done per Step2 v3 segment row; overlapping sources are merged
per well by TVD-neighbourhood weighted averaging (equal weights by default,
configurable), and fracture points are refined from the merged curve using the
glutenite-style density-integral calibration.

See taigu_step4_design_20260812.md for the design record.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


STRATA_UPPER = "上部复合层"
STRATA_LOWER = "太古界风化壳"
TARGET_STRATA = (STRATA_UPPER, STRATA_LOWER)
FEATURE_COLUMNS = [
    "GR", "RDeep", "RReference",
    "GRRobustZ", "RDeepRobustZ", "RReferenceRobustZ",
    "GRRank", "RDeepRank", "RReferenceRank",
    "GRGradientZ", "RDeepGradientZ", "RReferenceGradientZ",
    "GRResidual0p5MZ", "GRResidual1p0MZ", "GRResidual3p0MZ", "GRResidual5p0MZ",
    "RDeepResidual0p5MZ", "RDeepResidual1p0MZ", "RDeepResidual3p0MZ", "RDeepResidual5p0MZ",
    "RReferenceResidual0p5MZ", "RReferenceResidual1p0MZ", "RReferenceResidual3p0MZ", "RReferenceResidual5p0MZ",
    "GRMAD1MZ", "GRMAD3MZ", "GRMAD5MZ",
    "RDeepMAD1MZ", "RDeepMAD3MZ", "RDeepMAD5MZ",
    "RReferenceMAD1MZ", "RReferenceMAD3MZ", "RReferenceMAD5MZ",
    "Contrast", "ContrastRobustZ", "NormalizedContrast",
    "ContrastResidual1MZ", "ContrastResidual3MZ", "ContrastResidual5MZ",
    "GRxDeepResidual", "GRxReferenceResidual", "GRxNormalizedContrast",
    "DeepNonPositiveFlag", "ReferenceNonPositiveFlag",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and predict TaiGuJie GR/RD/RS fracture experts (v1)")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "taigu_step4_gr_rd_rs_v1.json")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--replace-existing-output", action="store_true")
    return p.parse_args()


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def robust_center_scale(values: pd.Series) -> tuple[float, float]:
    center = float(values.median())
    mad = float((values - center).abs().median())
    scale = mad * 1.4826 if mad > 1.0e-12 else 1.0
    return center, scale


def robust_z(values: pd.Series) -> pd.Series:
    center, scale = robust_center_scale(values.dropna())
    return (values - center) / scale


def signed_log1p(values: pd.Series) -> pd.Series:
    v = numeric(values)
    return np.sign(v) * np.log1p(v.abs())


def positive_log10(values: pd.Series) -> pd.Series:
    v = numeric(values).clip(lower=0.0)
    return np.log10(v + 1.0)


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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gradient = np.gradient(value_array, depth_array)
    gradient[~np.isfinite(gradient)] = np.nan
    return pd.Series(gradient, index=values.index)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    parts = []
    for (well_name, strata_name), group in frame.groupby(["WellName", "StrataName"], dropna=False):
        out = group.sort_values("TVD").copy()
        out["RDeep"] = numeric(out["RD"])
        out["RReference"] = numeric(out["RS"])
        out["RDeepSignedLog"] = signed_log1p(out["RDeep"])
        out["RReferenceSignedLog"] = signed_log1p(out["RReference"])
        out["LogRDeepPositive"] = positive_log10(out["RDeep"])
        out["LogRReferencePositive"] = positive_log10(out["RReference"])
        out["GRRobustZ"] = robust_z(numeric(out["GR"]))
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
            out[f"{prefix}GradientZ"] = robust_z(depth_gradient(values, out["TVD"]))
            for window_m in (0.5, 1.0, 3.0, 5.0):
                suffix = str(window_m).replace(".", "p")
                out[f"{prefix}Residual{suffix}MZ"] = robust_z(rolling_residual(values, out["TVD"], window_m))
            for window_m in (1.0, 3.0, 5.0):
                out[f"{prefix}MAD{int(window_m)}MZ"] = robust_z(rolling_mad(values, out["TVD"], window_m))

        out["Contrast"] = out["RDeepSignedLog"] - out["RReferenceSignedLog"]
        out["ContrastRobustZ"] = robust_z(out["Contrast"])
        denominator = numeric(out["RDeep"]).abs() + numeric(out["RReference"]).abs()
        out["NormalizedContrast"] = (numeric(out["RDeep"]) - numeric(out["RReference"])) / denominator.where(denominator.gt(1.0e-12))
        for window_m in (1.0, 3.0, 5.0):
            out[f"ContrastResidual{int(window_m)}MZ"] = robust_z(rolling_residual(out["Contrast"], out["TVD"], window_m))
        out["GRxDeepResidual"] = out["GRRobustZ"] * out["RDeepResidual1p0MZ"]
        out["GRxReferenceResidual"] = out["GRRobustZ"] * out["RReferenceResidual1p0MZ"]
        out["GRxNormalizedContrast"] = out["GRRobustZ"] * out["NormalizedContrast"]
        out["DeepNonPositiveFlag"] = numeric(out["RDeep"]).le(0.0).astype(int)
        out["ReferenceNonPositiveFlag"] = numeric(out["RReference"]).le(0.0).astype(int)
        out["ModelInputEligible"] = out[["GR", "RDeep", "RReference"]].notna().all(axis=1)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


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
    for strata_name, group in training.groupby("StrataName", sort=False):
        out = group.copy()
        density = numeric(out["Density"]).fillna(0.0).clip(lower=0.0)
        positive_density = density[density.gt(0.0)]
        density_threshold = float(positive_density.quantile(quantile)) if not positive_density.empty else 0.0
        point_mask = numeric(out.get("GT_POINT_FLAG", pd.Series(0, index=out.index))).fillna(0).gt(0)
        distances = np.full(len(out), np.inf, dtype=float)
        for _, well_index in out.groupby("WellName").groups.items():
            local = out.loc[well_index]
            local_points = numeric(local.get("GT_POINT_FLAG", pd.Series(0, index=local.index))).fillna(0).gt(0)
            distances[out.index.get_indexer(well_index)] = nearest_point_distance(local["TVD"], local_points)
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


def equalized_depth_weights(frame: pd.DataFrame) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    weights = np.zeros(len(frame), dtype=float)
    for (well_name, strata_name), index in frame.groupby(["WellName", "StrataName"]).groups.items():
        weights[index] = 1.0 / len(index)
    total = weights.sum()
    if total > 0:
        weights = weights / total * len(frame)
    return weights


def xgb_classifier(config: dict[str, Any], random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_jobs=int(config["xgb_n_jobs"]), random_state=random_state,
        eval_metric="logloss", **dict(config["stage1_params"]),
    )


def xgb_regressor(config: dict[str, Any], random_state: int) -> XGBRegressor:
    return XGBRegressor(
        n_jobs=int(config["xgb_n_jobs"]), random_state=random_state,
        **dict(config["stage2_params"]),
    )


def select_threshold(labels: np.ndarray, probability: np.ndarray, strategy: str) -> float:
    if strategy == "match_oof_prevalence":
        target_rate = float(np.mean(labels))
        if len(probability) == 0:
            return 0.5
        threshold = float(np.quantile(probability, 1.0 - min(max(target_rate, 1e-3), 1 - 1e-3)))
        return float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    return 0.5


def metric_row(labels: np.ndarray, probability: np.ndarray, density_true: np.ndarray,
               density_pred: np.ndarray, threshold: float) -> dict[str, float]:
    if len(labels) == 0:
        return {}
    predicted = (probability >= threshold).astype(int)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    row: dict[str, float] = {
        "ROC_AUC": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) > 1 else np.nan,
        "AveragePrecision": float(average_precision_score(labels, probability)),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Threshold": threshold,
    }
    if len(density_true) and np.isfinite(density_true).all() and np.isfinite(density_pred).all():
        row["DensityMAE"] = float(np.abs(density_true - density_pred).mean())
        row["DensityRMSE"] = float(np.sqrt(np.mean((density_true - density_pred) ** 2)))
    return row


def load_training_table(config: dict[str, Any]) -> pd.DataFrame:
    step3_root = Path(config["step3_output_dir"])
    step2_root = Path(config["step2_output_dir"])
    if not step3_root.is_absolute():
        step3_root = Path(__file__).resolve().parents[2] / step3_root
    if not step2_root.is_absolute():
        step2_root = Path(__file__).resolve().parents[2] / step2_root
    manifest = pd.read_csv(step2_root / "taigu_step2_segment_manifest.csv", encoding="utf-8-sig")
    groups = pd.read_csv(step3_root / "sample_group_manifest.csv", encoding="utf-8-sig")
    frames = []
    for _, g in groups.iterrows():
        group = pd.read_csv(g.GroupPath, encoding="utf-8-sig")
        frames.append(group)
    labels = pd.concat(frames, ignore_index=True)
    labels["SegmentID"] = labels["InputSegmentPath"].apply(lambda p: str(Path(p).stem))
    seg_frames = []
    for _, seg in manifest.iterrows():
        df = pd.read_csv(seg.OutputFilePath, encoding="utf-8-sig")
        df["WellName"] = str(seg.WellName)
        df["SegmentID"] = str(seg.SegmentID)
        seg_frames.append(df)
    segments = pd.concat(seg_frames, ignore_index=True)
    training = labels.merge(
        segments[["WellName", "TVD", "SegmentID", "GR", "RD", "RS"]],
        on=["WellName", "TVD", "SegmentID"], how="left", validate="one_to_one",
    )
    training = training[training.StrataName.isin(TARGET_STRATA)].copy()
    return training


def train_validation_for_strata(training: pd.DataFrame, config: dict[str, Any], random_state: int,
                                auxiliary_block_cv: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (expert dict, validation frame) for one strata."""
    base = training.copy()
    strata_name = str(base.StrataName.iloc[0])
    featured = engineer_features(base)
    featured = featured[featured.ModelInputEligible].copy()
    stage1 = featured[featured.PresenceLabel.notna()].copy()
    stage2 = featured[featured.Density.gt(0.0)].copy()
    wells = sorted(featured.WellName.unique())
    oof_rows = []
    for held_out in wells:
        train_mask = featured.WellName.ne(held_out)
        if not train_mask.any():
            continue
        clf = xgb_classifier(config, random_state)
        reg = xgb_regressor(config, random_state)
        fit1 = stage1[train_mask.loc[stage1.index]]
        fit2 = stage2[train_mask.loc[stage2.index]]
        if fit1.empty or fit2.empty:
            continue
        w1 = equalized_depth_weights(fit1)
        w2 = equalized_depth_weights(fit2)
        clf.fit(fit1[FEATURE_COLUMNS], fit1.PresenceLabel, sample_weight=w1)
        reg.fit(fit2[FEATURE_COLUMNS], fit2.Density, sample_weight=w2)
        hold = featured[featured.WellName.eq(held_out)].copy()
        prob = clf.predict_proba(hold[FEATURE_COLUMNS])[:, 1]
        cond = reg.predict(hold[FEATURE_COLUMNS])
        hold["OOFProb"] = prob
        hold["OOFCond"] = cond
        hold["OOFDensity"] = prob * cond
        oof_rows.append(hold)
    oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    thresholds = {}
    pooled_metrics = {}
    fold_metrics = []
    if not oof.empty:
        labels = oof.PresenceLabel.to_numpy(float)
        valid = np.isfinite(labels)
        threshold = select_threshold(labels[valid], oof.loc[valid, "OOFProb"].to_numpy(float), config["threshold_strategy"])
        thresholds["threshold"] = threshold
        density_true = oof.loc[oof.Density.gt(0.0), "Density"].to_numpy(float)
        density_pred = oof.loc[oof.Density.gt(0.0), "OOFDensity"].to_numpy(float)
        pooled_metrics = metric_row(labels[valid], oof.loc[valid, "OOFProb"].to_numpy(float),
                                    density_true, density_pred, threshold)
        for held_out in wells:
            hold = oof[oof.WellName.eq(held_out)]
            if hold.empty:
                continue
            labels = hold.PresenceLabel.to_numpy(float)
            valid = np.isfinite(labels)
            pos = hold.Density.gt(0.0)
            fold_metrics.append({
                "StrataName": strata_name, "HeldOutWell": held_out,
                "Rows": int(len(hold)),
                **metric_row(
                    labels[valid],
                    hold.loc[valid, "OOFProb"].to_numpy(float),
                    hold.loc[pos, "Density"].to_numpy(float),
                    hold.loc[pos, "OOFDensity"].to_numpy(float),
                    threshold,
                ),
            })
    # density scale: align positive-density mean
    scale = 1.0
    if not oof.empty:
        pos = oof[oof.Density.gt(0.0)]
        if len(pos):
            mean_true = float(pos.Density.mean())
            mean_pred = float(pos.OOFDensity.mean())
            if mean_pred > 1e-12:
                scale = float(np.clip(mean_true / mean_pred, 0.1, 10.0))
    # final deployment model on all wells
    clf = xgb_classifier(config, random_state)
    reg = xgb_regressor(config, random_state)
    clf.fit(stage1[FEATURE_COLUMNS], stage1.PresenceLabel, sample_weight=equalized_depth_weights(stage1))
    reg.fit(stage2[FEATURE_COLUMNS], stage2.Density, sample_weight=equalized_depth_weights(stage2))
    expert = {
        "stage1": clf, "stage2": reg,
        "threshold": float(thresholds.get("threshold", 0.5)),
        "density_scale": float(scale),
        "training_rows": int(len(featured)),
        "stage1_rows": int(len(stage1)),
        "stage2_rows": int(len(stage2)),
        "wells": wells,
        "fold_metrics": fold_metrics,
        "pooled_metrics": pooled_metrics,
    }
    validation = oof.copy()
    if auxiliary_block_cv:
        aux_rows = []
        for well in wells:
            well_data = featured[featured.WellName.eq(well)].sort_values("TVD").reset_index(drop=True)
            if len(well_data) < 10:
                continue
            bounds = np.array_split(np.arange(len(well_data)), 5)
            for block in bounds:
                test_idx = well_data.index[block]
                train_idx = well_data.index.difference(test_idx)
                clf_b = xgb_classifier(config, random_state)
                reg_b = xgb_regressor(config, random_state)
                s1 = stage1[stage1.WellName.eq(well) & stage1.index.isin(train_idx)]
                s2 = stage2[stage2.WellName.eq(well) & stage2.index.isin(train_idx)]
                if s1.empty or s2.empty:
                    continue
                clf_b.fit(s1[FEATURE_COLUMNS], s1.PresenceLabel, sample_weight=equalized_depth_weights(s1))
                reg_b.fit(s2[FEATURE_COLUMNS], s2.Density, sample_weight=equalized_depth_weights(s2))
                test = well_data.loc[test_idx]
                prob = clf_b.predict_proba(test[FEATURE_COLUMNS])[:, 1]
                cond = reg_b.predict(test[FEATURE_COLUMNS])
                test = test.copy()
                test["OOFProb"] = prob
                test["OOFDensity"] = prob * cond
                test["AuxBlock"] = True
                aux_rows.append(test)
        if aux_rows:
            aux = pd.concat(aux_rows, ignore_index=True)
            validation = pd.concat([validation, aux], ignore_index=True)
    return expert, validation


def build_prediction_rows_for_well(well_rows: pd.DataFrame, experts: dict[str, dict[str, Any]]) -> pd.DataFrame:
    featured = engineer_features(well_rows)
    featured = featured[featured.ModelInputEligible].copy()
    parts = []
    for strata_name in TARGET_STRATA:
        subset = featured[featured.StrataName.eq(strata_name)].copy()
        expert = experts.get(strata_name)
        if subset.empty or expert is None:
            continue
        prob = expert["stage1"].predict_proba(subset[FEATURE_COLUMNS])[:, 1]
        cond = expert["stage2"].predict(subset[FEATURE_COLUMNS]) * float(expert["density_scale"])
        subset["PredFractureProb"] = prob
        subset["PredConditionalDensity"] = cond
        subset["PredDensity"] = prob * cond
        subset["PredHasFracture"] = (prob >= float(expert["threshold"])).astype(int)
        subset["Stage1Threshold"] = float(expert["threshold"])
        subset["ExpertID"] = f"{strata_name}_RD_RS"
        subset["PredictionValid"] = 1
        parts.append(subset)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def merge_well_predictions(rows: pd.DataFrame, tolerance_m: float, gap_break_m: float,
                           weight_scheme: str) -> pd.DataFrame:
    """Merge per-source rows into WellName+TVD unique rows; cross-source only."""
    rows = rows.sort_values("TVD").reset_index(drop=True)
    n = len(rows)
    if n == 0:
        return pd.DataFrame()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tv = rows["TVD"].to_numpy(float)
    seg_indices = {seg: np.asarray(list(idx), dtype=int) for seg, idx in rows.groupby("SegmentID").groups.items()}
    seg_ids = sorted(seg_indices)
    for i in range(n):
        my_seg = rows.iloc[i]["SegmentID"]
        for other in seg_ids:
            if other == my_seg:
                continue
            idxs = seg_indices[other]
            pos = int(np.searchsorted(tv[idxs], tv[i]))
            candidates = []
            for j in (pos - 1, pos):
                if 0 <= j < len(idxs):
                    candidates.append(idxs[j])
            if not candidates:
                continue
            j = min(candidates, key=lambda k: abs(tv[k] - tv[i]))
            if abs(tv[j] - tv[i]) <= tolerance_m:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    rows["ClusterID"] = [find(i) for i in range(n)]
    weights = np.ones(n, dtype=float)
    rows["MergeWeight"] = weights
    agg = rows.groupby("ClusterID").agg(
        WellName=("WellName", "first"),
        TVD=("TVD", "mean"),
        MD=("MD", "mean"),
        X=("X", "mean"),
        Y=("Y", "mean"),
        TIME=("TIME", "mean"),
        StrataName=("StrataName", "first"),
        PredFractureProb=("PredFractureProb", "mean"),
        PredConditionalDensity=("PredConditionalDensity", "mean"),
        PredDensity=("PredDensity", "mean"),
        GR=("GR", "mean"), RD=("RD", "mean"), RS=("RS", "mean"),
        SourceCount=("SegmentID", "nunique"),
        SourceSegmentIDs=("SegmentID", lambda s: ";".join(sorted(set(s)))),
        PredictionValid=("PredictionValid", "first"),
        Stage1Threshold=("Stage1Threshold", "first"),
        ExpertID=("ExpertID", "first"),
    ).reset_index(drop=True)
    agg["PredHasFracture"] = (agg.PredFractureProb >= agg.Stage1Threshold).astype(int)
    agg = agg.sort_values(["WellName", "TVD"]).reset_index(drop=True)
    # support segments
    new_seg = agg.StrataName.ne(agg.StrataName.shift()) | agg.TVD.diff().gt(gap_break_m)
    agg["SupportSegmentID"] = new_seg.cumsum()
    bounds = agg.groupby("SupportSegmentID")["TVD"].agg(["min", "max"])
    agg["SegmentStartTVD"] = agg["SupportSegmentID"].map(bounds["min"])
    agg["SegmentEndTVD"] = agg["SupportSegmentID"].map(bounds["max"])
    agg["MergeRule"] = "equal_tvd_neighbourhood"
    return agg


def point_count_scales(training: pd.DataFrame, point_wells: list[str]) -> dict[str, float]:
    scales = {}
    labeled = training[training.WellName.isin(point_wells)].copy()
    for strata_name, strata in labeled.groupby("StrataName"):
        area = 0.0
        for _, well in strata.groupby("WellName"):
            ordered = well.sort_values("TVD")
            depth = numeric(ordered["TVD"]).to_numpy(dtype=float)
            density = numeric(ordered["Density"]).fillna(0.0).clip(lower=0).to_numpy(dtype=float)
            if len(depth) >= 2:
                area += float(np.trapezoid(density, depth))
        points = int(numeric(strata.get("GT_POINT_FLAG", pd.Series(0, index=strata.index))).fillna(0).gt(0).sum())
        scales[str(strata_name)] = float(points / area) if area > 1.0e-12 else 1.0
    return scales


def refine_points(merged: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    rows = []
    for support_id, support in merged.groupby(["WellName", "SupportSegmentID"]):
        support = support.sort_values("TVD").reset_index(drop=True)
        positive = support.PredHasFracture.gt(0).to_numpy()
        starts = np.flatnonzero(positive & np.concatenate(([True], ~positive[:-1])))
        for start in starts:
            end = start
            while end + 1 < len(support) and positive[end + 1]:
                end += 1
            segment = support.iloc[start : end + 1]
            depth = segment.TVD.to_numpy(dtype=float)
            density = segment.PredDensity.to_numpy(dtype=float)
            area = float(np.trapezoid(density, depth)) if len(depth) >= 2 else float(density[0]) * 0.125
            count = min(len(segment), max(1, int(round(area * float(scales.get(str(segment.StrataName.iloc[0]), 1.0))))))
            for ordinal, local_index in enumerate(sorted(np.argsort(-density)[:count].tolist()), start=1):
                source = segment.iloc[int(local_index)]
                rows.append({
                    "WellName": source.WellName, "TVD": float(source.TVD), "MD": float(source.MD),
                    "StrataName": source.StrataName, "PredDensity": float(source.PredDensity),
                    "PredFractureProb": float(source.PredFractureProb),
                    "SupportSegmentID": int(support_id[1]), "PointOrdinalInSegment": ordinal,
                })
    return pd.DataFrame(rows)


def predict_all_wells(config: dict[str, Any], experts: dict[str, dict[str, Any]], prediction_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    step2_root = Path(config["step2_output_dir"])
    if not step2_root.is_absolute():
        step2_root = Path(__file__).resolve().parents[2] / step2_root
    manifest = pd.read_csv(step2_root / "taigu_step2_segment_manifest.csv", encoding="utf-8-sig")
    per_well_root = prediction_root / "real_well_predictions"
    per_well_root.mkdir(parents=True, exist_ok=True)
    detail_parts = []
    merged_parts = []
    coverage_rows = []
    for well in sorted(manifest.WellName.unique()):
        well_segments = manifest[manifest.WellName.eq(well)]
        frames = []
        for _, seg in well_segments.iterrows():
            df = pd.read_csv(seg.OutputFilePath, encoding="utf-8-sig")
            df["WellName"] = str(seg.WellName)
            df["SegmentID"] = str(seg.SegmentID)
            df["LogDate"] = str(seg.LogDate) if pd.notna(seg.LogDate) else ""
            frames.append(df)
        well_rows = pd.concat(frames, ignore_index=True)
        well_rows = well_rows[well_rows.StrataName.isin(TARGET_STRATA)].copy()
        detail = build_prediction_rows_for_well(well_rows, experts)
        if detail.empty:
            coverage_rows.append({"WellName": well, "Status": "no_prediction_rows", "Rows": int(len(well_rows))})
            continue
        detail_parts.append(detail)
        merged = merge_well_predictions(
            detail, float(config["merge"]["tolerance_m"]),
            float(config["merge"]["gap_break_m"]), str(config["merge"]["weight_scheme"]),
        )
        if not merged.empty:
            merged_parts.append(merged)
            well_dir = per_well_root / well
            well_dir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(well_dir / f"{well}_merged_density_prediction.csv", index=False, encoding="utf-8-sig")
            detail.to_csv(well_dir / f"{well}_source_detail_predictions.csv", index=False, encoding="utf-8-sig")
            coverage_rows.append({
                "WellName": well, "Status": "ok", "Rows": int(len(merged)),
                "SourceDetailRows": int(len(detail)),
                "TVDMin": float(merged.TVD.min()), "TVDMax": float(merged.TVD.max()),
            })
    detail_all = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    merged_all = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    detail_all.to_csv(prediction_root / "all_wells_source_detail_predictions.csv", index=False, encoding="utf-8-sig")
    merged_all.to_csv(prediction_root / "all_wells_merged_density_prediction.csv", index=False, encoding="utf-8-sig")
    coverage = pd.DataFrame(coverage_rows)
    return detail_all, merged_all, coverage_rows, coverage


def unsupported_intervals(merged: pd.DataFrame, gap_break_m: float) -> pd.DataFrame:
    rows = []
    for (well, strata_name), group in merged.groupby(["WellName", "StrataName"]):
        group = group.sort_values("TVD")
        gaps = group.TVD.diff()
        for idx in group.index[gaps.gt(gap_break_m)]:
            gap = float(gaps.loc[idx])
            rows.append({
                "WellName": well, "StrataName": strata_name,
                "StartTVD": float(group.loc[idx].TVD - gap),
                "EndTVD": float(group.loc[idx].TVD),
                "GapM": round(gap, 2),
            })
    return pd.DataFrame(rows)


def overlap_diagnostics(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (well, strata_name), group in detail.groupby(["WellName", "StrataName"]):
        segments = sorted(group.SegmentID.unique())
        if len(segments) < 2:
            continue
        for a in range(len(segments)):
            for b in range(a + 1, len(segments)):
                ga = group[group.SegmentID.eq(segments[a])].sort_values("TVD")
                gb = group[group.SegmentID.eq(segments[b])].sort_values("TVD")
                lo = max(ga.TVD.min(), gb.TVD.min())
                hi = min(ga.TVD.max(), gb.TVD.max())
                if hi - lo <= 0.1:
                    continue
                da = ga[(ga.TVD >= lo) & (ga.TVD <= hi)].PredDensity.to_numpy(float)
                db = gb[(gb.TVD >= lo) & (gb.TVD <= hi)].PredDensity.to_numpy(float)
                n = min(len(da), len(db))
                if n < 5:
                    continue
                da, db = da[:n], db[:n]
                corr = float(np.corrcoef(da, db)[0, 1]) if np.std(da) > 0 and np.std(db) > 0 else np.nan
                rows.append({
                    "WellName": well, "StrataName": strata_name,
                    "SegmentA": segments[a], "SegmentB": segments[b],
                    "OverlapTVDM": round(hi - lo, 2), "N": n,
                    "DensityCorrelation": corr,
                    "DensityMeanAbsDiff": float(np.abs(da - db).mean()),
                })
    return pd.DataFrame(rows)


def main() -> int:
    ns = parse_args()
    cfg = json.loads(ns.config.read_text(encoding="utf-8"))
    out_dir = ns.output_dir or Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[2] / out_dir
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not ns.replace_existing_output:
            raise RuntimeError(f"output exists; pass --replace-existing-output: {out_dir}")
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model_library"
    validation_dir = out_dir / "validation"
    prediction_dir = out_dir / "predictions"
    qc_dir = out_dir / "qc"
    for d in (model_dir, validation_dir, prediction_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    random_state = int(cfg["random_state"])
    training = load_training_table(cfg)
    training = assign_presence_labels(training, cfg["label"])
    experts: dict[str, dict[str, Any]] = {}
    validation_parts = []
    registry = []
    for strata_name in TARGET_STRATA:
        subset = training[training.StrataName.eq(strata_name)].copy()
        auxiliary = strata_name == STRATA_LOWER
        expert, validation = train_validation_for_strata(subset, cfg, random_state, auxiliary_block_cv=auxiliary)
        expert["strata"] = strata_name
        experts[strata_name] = expert
        validation_parts.append(validation)
        registry.append({
            "ExpertID": f"{strata_name}_RD_RS", "StrataName": strata_name, "PairType": "RD_RS",
            "Stage1Rows": expert["stage1_rows"], "Stage2Rows": expert["stage2_rows"],
            "TrainingRows": expert["training_rows"], "Wells": ";".join(expert["wells"]),
            "DeploymentThreshold": expert["threshold"], "DensityScale": expert["density_scale"],
        })
        joblib.dump(expert["stage1"], model_dir / f"{strata_name}_RD_RS_stage1_xgb.joblib")
        joblib.dump(expert["stage2"], model_dir / f"{strata_name}_RD_RS_stage2_xgb.joblib")
    pd.DataFrame(registry).to_csv(model_dir / "taigu_two_expert_registry.csv", index=False, encoding="utf-8-sig")
    thresholds = {r["ExpertID"]: r["DeploymentThreshold"] for r in registry}
    (model_dir / "expert_thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    fold_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    for strata_name in TARGET_STRATA:
        ex = experts[strata_name]
        fold_rows.extend(ex.get("fold_metrics", []))
        pooled_rows.append({"StrataName": strata_name, **ex.get("pooled_metrics", {})})
    pd.DataFrame(fold_rows).to_csv(validation_dir / "direct_expert_validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pooled_rows).to_csv(validation_dir / "pooled_validation_metrics.csv", index=False, encoding="utf-8-sig")
    validation_all = pd.concat(validation_parts, ignore_index=True) if validation_parts else pd.DataFrame()
    validation_all.to_csv(validation_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    detail_all, merged_all, coverage_rows, coverage = predict_all_wells(cfg, experts, prediction_dir)
    coverage.to_csv(qc_dir / "well_prediction_coverage.csv", index=False, encoding="utf-8-sig")
    overlap = overlap_diagnostics(detail_all)
    overlap.to_csv(qc_dir / "overlap_diagnostics.csv", index=False, encoding="utf-8-sig")
    unsupported = unsupported_intervals(merged_all, float(cfg["merge"]["gap_break_m"]))
    unsupported.to_csv(qc_dir / "unsupported_intervals.csv", index=False, encoding="utf-8-sig")

    scales = point_count_scales(training, [str(w) for w in cfg["point_wells"]])
    points = refine_points(merged_all, scales)
    points.to_csv(prediction_dir / "all_wells_merged_fracture_points.csv", index=False, encoding="utf-8-sig")

    # acceptance
    errors = []
    if len(experts) != 2:
        errors.append("expected two experts")
    if merged_all.empty or merged_all.duplicated(["WellName", "TVD"]).any():
        errors.append("merged table empty or duplicate WellName+TVD")
    if not merged_all.empty:
        bad = merged_all[~np.isfinite(merged_all[["PredFractureProb", "PredDensity"]].to_numpy(dtype=float)).all(axis=1)]
        if len(bad):
            errors.append("merged table has non-finite predictions")
    status = "pass" if not errors else "fail"
    summary = {
        "version": "taigu_step4_gr_rd_rs_v1",
        "status": status,
        "training_wells": sorted(training.WellName.unique().astype(str)),
        "training_rows": int(len(training)),
        "experts": registry,
        "merged_rows": int(len(merged_all)),
        "detail_rows": int(len(detail_all)),
        "fracture_points": int(len(points)),
        "coverage_wells": int(len(coverage)),
        "overlap_pairs": int(len(overlap)),
        "density_scales": scales,
        "errors": errors,
    }
    (out_dir / "taigu_step4_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "version": "taigu_step4_gr_rd_rs_v1",
        "all_wells_merged_prediction_table": str(prediction_dir / "all_wells_merged_density_prediction.csv"),
        "all_wells_source_detail_table": str(prediction_dir / "all_wells_source_detail_predictions.csv"),
        "all_wells_fracture_points_table": str(prediction_dir / "all_wells_merged_fracture_points.csv"),
        "per_well_predictions_root": str(prediction_dir / "real_well_predictions"),
        "registry": str(model_dir / "taigu_two_expert_registry.csv"),
        "oof_predictions": str(validation_dir / "oof_predictions.csv"),
        "acceptance": str(out_dir / "taigu_step4_acceptance_summary.json"),
    }
    (out_dir / "taigu_step4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
