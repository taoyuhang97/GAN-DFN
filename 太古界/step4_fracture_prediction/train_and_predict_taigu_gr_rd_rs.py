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


def engineer_batch_window_features(segment: pd.DataFrame) -> pd.DataFrame:
    """在**单个测井批次（测井段）内部**计算窗口类特征。

    设计口径：同一批次是一次连续观测，段内规律一致；跨批次的邻居不能进入同一个
    滑动窗口，否则局部残差/MAD 度量的是"两次测井之间的差异"而不是地质变化。
    只依赖本段样本，输出的窗口特征为未标准化原值（标准化在井×地层内统一进行）。
    """
    out = segment.sort_values("TVD").copy()
    depth = out["TVD"]
    deep_signed = signed_log1p(numeric(out["RD"]))
    reference_signed = signed_log1p(numeric(out["RS"]))
    transformed = {
        "GR": numeric(out["GR"]),
        "RDeep": deep_signed,
        "RReference": reference_signed,
    }
    for prefix, values in transformed.items():
        out[f"{prefix}Gradient"] = depth_gradient(values, depth)
        for window_m in (0.5, 1.0, 3.0, 5.0):
            suffix = str(window_m).replace(".", "p")
            out[f"{prefix}Residual{suffix}M"] = rolling_residual(values, depth, window_m)
        for window_m in (1.0, 3.0, 5.0):
            out[f"{prefix}MAD{int(window_m)}M"] = rolling_mad(values, depth, window_m)
    contrast = deep_signed - reference_signed
    for window_m in (1.0, 3.0, 5.0):
        out[f"ContrastResidual{int(window_m)}M"] = rolling_residual(contrast, depth, window_m)
    return out


def cross_segment_window_columns(max_window_m: float) -> list[str]:
    """列出"允许跨段补齐"的窗口特征名（窗口长度 ≤ max_window_m）。

    深度梯度只用相邻行，等价于最小窗口，因此一并纳入。
    """
    columns: list[str] = []
    for prefix in ("GR", "RDeep", "RReference"):
        columns.append(f"{prefix}Gradient")
        for window_m in (0.5, 1.0, 3.0, 5.0):
            if window_m <= max_window_m + 1.0e-9:
                columns.append(f"{prefix}Residual{str(window_m).replace('.', 'p')}M")
        for window_m in (1.0, 3.0, 5.0):
            if window_m <= max_window_m + 1.0e-9:
                columns.append(f"{prefix}MAD{int(window_m)}M")
    for window_m in (1.0, 3.0, 5.0):
        if window_m <= max_window_m + 1.0e-9:
            columns.append(f"ContrastResidual{int(window_m)}M")
    return columns


def add_scaled_features(out: pd.DataFrame) -> pd.DataFrame:
    """在给定分组内计算逐行派生特征与标准化特征（稳健 Z / 分位秩）。"""
    out = out.copy()
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

    for prefix in ("GR", "RDeep", "RReference"):
        out[f"{prefix}GradientZ"] = robust_z(out[f"{prefix}Gradient"])
        for window_m in (0.5, 1.0, 3.0, 5.0):
            suffix = str(window_m).replace(".", "p")
            out[f"{prefix}Residual{suffix}MZ"] = robust_z(out[f"{prefix}Residual{suffix}M"])
        for window_m in (1.0, 3.0, 5.0):
            out[f"{prefix}MAD{int(window_m)}MZ"] = robust_z(out[f"{prefix}MAD{int(window_m)}M"])

    out["Contrast"] = out["RDeepSignedLog"] - out["RReferenceSignedLog"]
    out["ContrastRobustZ"] = robust_z(out["Contrast"])
    denominator = numeric(out["RDeep"]).abs() + numeric(out["RReference"]).abs()
    out["NormalizedContrast"] = (numeric(out["RDeep"]) - numeric(out["RReference"])) / denominator.where(denominator.gt(1.0e-12))
    for window_m in (1.0, 3.0, 5.0):
        out[f"ContrastResidual{int(window_m)}MZ"] = robust_z(out[f"ContrastResidual{int(window_m)}M"])
    out["GRxDeepResidual"] = out["GRRobustZ"] * out["RDeepResidual1p0MZ"]
    out["GRxReferenceResidual"] = out["GRRobustZ"] * out["RReferenceResidual1p0MZ"]
    out["GRxNormalizedContrast"] = out["GRRobustZ"] * out["NormalizedContrast"]
    out["DeepNonPositiveFlag"] = numeric(out["RDeep"]).le(0.0).astype(int)
    out["ReferenceNonPositiveFlag"] = numeric(out["RReference"]).le(0.0).astype(int)
    out["ModelInputEligible"] = out[["GR", "RDeep", "RReference"]].notna().all(axis=1)
    return out


def engineer_features(frame: pd.DataFrame, window_policy: dict[str, Any] | None = None) -> pd.DataFrame:
    """逐行特征 + 窗口特征 + 井×地层内标准化。

    窗口特征的作用范围由 `window_policy` 控制：

    * `scope="segment"`（默认）：**窗口只在单次测井内部计算**，不跨批次；
    * `scope="well"`：窗口整条井一起算（改造前的旧口径）；
    * `cross_segment_max_window_m`：在 `scope="segment"` 下，窗口长度 ≤ 该值的特征
      允许跨段补齐（例如 0.5 m 这种只有几个采样点的小窗口），更大的窗口仍严格段内计算。
      这样既避免"大窗口跨批次"，又保留段边界处小窗口的连续性。

    标准化（稳健 Z / 分位秩）与逐行派生特征的作用范围由
    `normalization_scope` 控制：`"well_strata"`（默认）按井×地层；
    `"segment"` 按测井段——与窗口口径一致，代价是短段的统计量不稳定。
    """
    if frame.empty:
        return frame.copy()
    policy = dict(window_policy or {})
    scope = str(policy.get("scope", "segment")).strip().lower()
    cross_max = float(policy.get("cross_segment_max_window_m", 0.0) or 0.0)
    norm_scope = str(policy.get("normalization_scope", "well_strata")).strip().lower()
    parts = []
    for (well_name, strata_name), group in frame.groupby(["WellName", "StrataName"], dropna=False):
        ordered = group.sort_values("TVD").copy()
        ordered["_rowkey"] = np.arange(len(ordered))
        # --- 1) 窗口特征 ---
        if scope == "well" or "SegmentID" not in ordered.columns or not ordered["SegmentID"].notna().any():
            out = engineer_batch_window_features(ordered).sort_values("_rowkey")
        else:
            batch_parts = [
                engineer_batch_window_features(segment)
                for _, segment in ordered.groupby("SegmentID", dropna=False)
            ]
            out = pd.concat(batch_parts).sort_values("_rowkey")
            if cross_max > 0.0:
                wide = engineer_batch_window_features(ordered).set_index("_rowkey")
                out = out.set_index("_rowkey")
                for column in cross_segment_window_columns(cross_max):
                    out[column] = wide[column]
                out = out.reset_index()
        out = out.sort_values("_rowkey")

        # --- 2) 逐行派生特征与标准化 ---
        if norm_scope == "segment" and "SegmentID" in out.columns and out["SegmentID"].notna().any():
            scaled = pd.concat([add_scaled_features(segment) for _, segment in out.groupby("SegmentID", dropna=False)])
        else:
            scaled = add_scaled_features(out)
        scaled = scaled.sort_values("_rowkey").drop(columns="_rowkey").reset_index(drop=True)
        parts.append(scaled)
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


def supervision_tier(frame: pd.DataFrame) -> pd.Series:
    """监督层级：优先用 Step3 的 `SupervisionTier` 列，缺失时按 `SupervisionStatus` 回退。"""
    if "SupervisionTier" in frame.columns:
        tier = frame["SupervisionTier"].astype(str)
    elif "SupervisionStatus" in frame.columns:
        tier = frame["SupervisionStatus"].astype(str).map(
            {"supervision_ready": "strong", "candidate_scope_pending": "presence_only",
             "density_only_scope_pending": "audit_only"}
        )
    else:
        tier = pd.Series("strong", index=frame.index)
    return tier.fillna("strong")


def supervision_gate(config: dict[str, Any]) -> dict[str, list[str]]:
    gate = dict(config.get("supervision_gate") or {})
    regression_tiers = [str(v) for v in gate.get("density_regression_tiers", ["strong"])]
    return {
        "positive_tiers": [str(v) for v in gate.get("positive_tiers", ["strong", "presence_only"])],
        "negative_tiers": [str(v) for v in gate.get("negative_tiers", ["strong"])],
        "density_regression_tiers": regression_tiers,
        # 总量守恒标定用哪些井：优先用显式井清单（可解释、可审计），否则回退到层级。
        "calibration_wells": [str(v) for v in (gate.get("calibration_wells") or [])],
        "calibration_tiers": [str(v) for v in (gate.get("calibration_tiers") or regression_tiers)],
    }


def calibration_rows(featured: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """选出参与"绝对水平标定"的样本行：显式井清单优先，其次按层级。"""
    gate = supervision_gate(config)
    if gate["calibration_wells"]:
        selected = featured[featured.WellName.astype(str).isin(gate["calibration_wells"])]
        if not selected.empty:
            return selected
    return featured[featured.SupervisionTier.astype(str).isin(gate["calibration_tiers"])]


def apply_supervision_gate(training: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """把"哪一级监督能提供哪种标签"接进流程。

    - 正样本（有密度或有解释点）：`positive_tiers` 的井都可以提供；
    - 负样本（明确没有裂缝）：只有 `negative_tiers` 的井可以提供——口径未核验的井不能教模型
      "这里没有裂缝"；
    - 条件密度回归：只有 `density_regression_tiers` 的井参与（数值刻度最可信的那一批）。
    """
    gate = supervision_gate(config)
    out = training.copy()
    tier = supervision_tier(out)
    out["SupervisionTier"] = tier
    kind = out.get("PresenceLabelKind", pd.Series("", index=out.index)).astype(str)
    gated = (
        (kind.eq("clear_fracture") & ~tier.isin(gate["positive_tiers"]))
        | (kind.eq("clear_background") & ~tier.isin(gate["negative_tiers"]))
    )
    out.loc[gated, "PresenceLabel"] = np.nan
    out.loc[gated, "PresenceLabelKind"] = "gated_out"
    out["DensityRegressionEligible"] = (
        numeric(out["Density"]).fillna(0.0).gt(0.0) & tier.isin(gate["density_regression_tiers"])
    )
    return out


def depth_integral(frame: pd.DataFrame, column: str, gap_break_m: float = 0.5) -> float:
    """按井做深度积分（跨测次重叠按深度去重，缺口不跨接），用于总量守恒标定。"""
    if frame.empty or column not in frame.columns:
        return 0.0
    total = 0.0
    for _, well in frame.groupby("WellName", sort=False):
        ordered = well.sort_values("TVD").drop_duplicates("TVD", keep="first")
        depth = numeric(ordered["TVD"]).to_numpy(dtype=float)
        values = numeric(ordered[column]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        if len(depth) < 2:
            continue
        step = float(np.median(np.diff(depth)))
        breaks = np.flatnonzero(np.diff(depth) > max(3.0 * step, gap_break_m))
        for run in np.split(np.arange(len(depth)), breaks + 1):
            if len(run) >= 2:
                total += float(np.trapezoid(values[run], depth[run]))
    return total


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
    # v5：训练样本显式限定"有成像标签的窗口"（both / imaging_only）。
    # 层内但成像段外的行（UseCase=layer_only）标签为 NaN，只参与预测、不参与训练。
    if "UseCase" in training.columns:
        training = training[training["UseCase"].astype(str).isin({"both", "imaging_only"})].copy()
    return training


def train_validation_for_strata(training: pd.DataFrame, config: dict[str, Any], random_state: int,
                                auxiliary_block_cv: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (expert dict, validation frame) for one strata."""
    base = training.copy()
    strata_name = str(base.StrataName.iloc[0])
    featured = engineer_features(base, window_policy=config.get("feature_windows"))
    featured = featured[featured.ModelInputEligible].copy()
    stage1 = featured[featured.PresenceLabel.notna()].copy()
    stage2 = featured[featured.DensityRegressionEligible].copy()
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
        # 门控后可能出现"某一折的训练集只剩单一类别"（例如风化壳强监督只有 313，
        # 扣掉它以后剩下的井只有正样本）——这种折无法训练分类器，直接跳过。
        if fit1.empty or fit2.empty or fit1.PresenceLabel.nunique() < 2:
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
    # OOF 口径的标定系数（只作诊断，不再用于部署输出）
    oof_scale = 1.0
    if not oof.empty:
        pos = oof[oof.Density.gt(0.0)]
        if len(pos):
            mean_true = float(pos.Density.mean())
            mean_pred = float(pos.OOFDensity.mean())
            if mean_pred > 1.0e-12:
                oof_scale = float(mean_true / mean_pred)
    # final deployment model on all wells
    clf = xgb_classifier(config, random_state)
    reg = xgb_regressor(config, random_state)
    clf.fit(stage1[FEATURE_COLUMNS], stage1.PresenceLabel, sample_weight=equalized_depth_weights(stage1))
    reg.fit(stage2[FEATURE_COLUMNS], stage2.Density, sample_weight=equalized_depth_weights(stage2))
    # 部署标定：**总量守恒**，并且用部署模型自己的样本内预测来标定。
    # 口径是"条数"——比较的是深度积分 ∫ρdz，不是正样本均值。
    eligible = calibration_rows(featured, config)
    if eligible.empty:
        eligible = featured
    in_sample_pred = clf.predict_proba(eligible[FEATURE_COLUMNS])[:, 1] * reg.predict(eligible[FEATURE_COLUMNS])
    actual_total = depth_integral(eligible, "Density")
    predicted_total = depth_integral(eligible.assign(_PredDensity=in_sample_pred), "_PredDensity")
    scale = float(actual_total / predicted_total) if predicted_total > 1.0e-12 else 1.0
    expert = {
        "stage1": clf, "stage2": reg,
        "threshold": float(thresholds.get("threshold", 0.5)),
        "density_scale": float(scale),
        "density_scale_oof_diagnostic": float(oof_scale),
        "calibration_actual_total": float(actual_total),
        "calibration_predicted_total": float(predicted_total),
        "calibration_wells": sorted(eligible.WellName.astype(str).unique()),
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


def build_prediction_rows_for_well(
    well_rows: pd.DataFrame,
    experts: dict[str, dict[str, Any]],
    window_policy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    featured = engineer_features(well_rows, window_policy=window_policy)
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


def _select_within_clusters(rows: pd.DataFrame, gap_break_m: float,
                            priority: list[str]) -> pd.DataFrame:
    """在每个深度邻域簇内，按优先级挑一整行（不平均），并补支持段与审计字段。"""
    work = rows.copy()
    if "LogDate" in work.columns:
        work["_LogDate"] = work["LogDate"].astype(str)
    else:
        work["_LogDate"] = ""
    segment_rows = work.groupby("SegmentID")["TVD"].size().rename("_SegmentRowCount")
    work = work.merge(segment_rows, left_on="SegmentID", right_index=True, how="left")

    sort_keys: list[str] = ["ClusterID"]
    ascending: list[bool] = [True]
    for item in priority:
        key = str(item).strip().lower()
        if key == "segment_rows_desc":
            sort_keys.append("_SegmentRowCount"); ascending.append(False)
        elif key == "log_date_desc":
            sort_keys.append("_LogDate"); ascending.append(False)
        elif key == "segment_id_asc":
            sort_keys.append("SegmentID"); ascending.append(True)
        else:
            raise ValueError(f"unknown select_priority item: {item}")
    ordered = work.sort_values(sort_keys, ascending=ascending, kind="mergesort")
    selected = ordered.groupby("ClusterID", as_index=False).first()
    audit = rows.groupby("ClusterID").agg(
        SourceCount=("SegmentID", "nunique"),
        SourceSegmentIDs=("SegmentID", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()
    selected = selected.merge(audit, on="ClusterID", how="left")
    selected["SelectedSegmentID"] = selected["SegmentID"].astype(str)
    selected["PredHasFracture"] = (
        numeric(selected["PredFractureProb"]) >= numeric(selected["Stage1Threshold"])
    ).astype(int)
    selected["MergeRule"] = "select_by_segment_priority"
    selected = selected.drop(columns=["_SegmentRowCount", "_LogDate"], errors="ignore")
    selected = selected.sort_values(["WellName", "TVD"]).reset_index(drop=True)
    new_seg = selected.StrataName.ne(selected.StrataName.shift()) | selected.TVD.diff().gt(gap_break_m)
    selected["SupportSegmentID"] = new_seg.cumsum()
    bounds = selected.groupby("SupportSegmentID")["TVD"].agg(["min", "max"])
    selected["SegmentStartTVD"] = selected["SupportSegmentID"].map(bounds["min"])
    selected["SegmentEndTVD"] = selected["SupportSegmentID"].map(bounds["max"])
    columns = [
        "WellName", "TVD", "MD", "X", "Y", "TIME", "StrataName",
        "PredFractureProb", "PredConditionalDensity", "PredDensity", "GR", "RD", "RS",
        "SourceCount", "SourceSegmentIDs", "SelectedSegmentID",
        "PredictionValid", "Stage1Threshold", "ExpertID", "PredHasFracture",
        "SupportSegmentID", "SegmentStartTVD", "SegmentEndTVD", "MergeRule",
        # v5：取样窗口标记（供 Step5 只取 InHorizonLayer=1 的目标地层段）
        "InImagingInterval", "InHorizonLayer", "UseCase",
    ]
    return selected[[column for column in columns if column in selected.columns]]


def merge_well_predictions(rows: pd.DataFrame, tolerance_m: float, gap_break_m: float,
                           weight_scheme: str, rule: str = "mean",
                           select_priority: list[str] | None = None) -> pd.DataFrame:
    """把逐段预测合并成"井 + 垂直深度"唯一的井级曲线（只在不同测次之间合并）。

    `rule` 支持两种口径：

    * `"mean"`（旧口径）：对同一深度邻域内的各测次取平均（坐标、曲线、预测值都平均）；
    * `"select"`（择一口径）：**不平均**，按测次优先级在邻域内**挑一条**整行保留。
      优先级由 `select_priority` 给出，默认：段内行数多者优先 → 测井日期新者优先 → 段名字典序。
      理由与砂砾岩"不同曲线对不平均、按优先级择一"一致：不同批次的数据不可简单平均。
    """
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
    if str(rule).strip().lower() == "select":
        return _select_within_clusters(rows, gap_break_m, select_priority or [
            "segment_rows_desc", "log_date_desc", "segment_id_asc",
        ])
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
        # v5：把取样窗口标记透传到井级预测表，供 Step5 只取目标地层内（InHorizonLayer=1）
        InImagingInterval=("InImagingInterval", "max"),
        InHorizonLayer=("InHorizonLayer", "max"),
    ).reset_index(drop=True)
    if {"InImagingInterval", "InHorizonLayer"}.issubset(agg.columns):
        agg["UseCase"] = np.where(
            agg["InImagingInterval"].eq(1) & agg["InHorizonLayer"].eq(1), "both",
            np.where(agg["InImagingInterval"].eq(1), "imaging_only", "layer_only"),
        )
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
    """【审计量，已退出点数计算】每地层"解释缝条数 ÷ 密度积分"。

    这个比值以前被当作细化乘子（点数 = ∫ρdz × 该常数），等于把各家产品的数值刻度差
    当成物理规律。现在只输出到审计表，`refine_points` 不再使用它。
    """
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


REFINE_DEFAULTS = {
    "gap_break_m": 0.5,
    "min_interval_thickness_m": 1.0,
    "coverage_targets": {"上部复合层": 0.33, "太古界风化壳": 0.68},
    "density_bins": [0.5, 2.0],
    "thickness_bins": [3.0, 10.0],
}


def refine_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(REFINE_DEFAULTS)
    merged.update(dict(config.get("refine") or {}))
    merged["coverage_targets"] = dict(merged.get("coverage_targets") or {})
    return merged


def _interval_bounds(depth: np.ndarray, above: np.ndarray, gap_break_m: float,
                     min_thickness_m: float) -> list[tuple[int, int]]:
    """在"值 ≥ 地板"的布尔序列上切连续区间：缺口 ≤ gap 合并，厚度 < 下限丢弃。"""
    bounds: list[tuple[int, int]] = []
    n = len(depth)
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while True:
            k = j
            while k + 1 < n and above[k + 1] and (depth[k + 1] - depth[k]) <= gap_break_m:
                k += 1
            nxt = None
            for t in range(k + 1, n):
                if depth[t] - depth[k] > gap_break_m:
                    break
                if above[t]:
                    nxt = t
                    break
            if nxt is None:
                j = k
                break
            j = nxt
        if depth[j] - depth[i] >= min_thickness_m:
            bounds.append((i, j))
        i = j + 1
    return bounds


def _coverage_for_floor(frame: pd.DataFrame, floor: float, gap_break_m: float,
                        min_thickness_m: float) -> float:
    covered = 0.0
    total = 0.0
    for _, well in frame.groupby("WellName", sort=False):
        ordered = well.sort_values("TVD")
        depth = numeric(ordered["TVD"]).to_numpy(dtype=float)
        values = numeric(ordered["PredDensity"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        if len(depth) < 2:
            continue
        total += float(depth[-1] - depth[0])
        for i, j in _interval_bounds(depth, values >= floor, gap_break_m, min_thickness_m):
            covered += float(depth[j] - depth[i])
    return covered / total if total > 0 else 0.0


def select_density_floors(merged: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """按"区间覆盖率对齐成像井"确定地层级密度地板（每个地层一个数，不逐井调）。"""
    refine = refine_config(config)
    floors: dict[str, dict[str, float]] = {}
    for strata_name, target in refine["coverage_targets"].items():
        subset = merged[merged.StrataName.eq(strata_name)]
        if subset.empty:
            floors[strata_name] = {"floor": 0.0, "coverage": 0.0, "target": float(target), "status": "no_rows"}
            continue
        candidates = np.unique(np.quantile(subset.PredDensity.clip(lower=0.0).to_numpy(dtype=float),
                                          np.linspace(0.02, 0.98, 193)))
        best = None
        for floor in candidates:
            coverage = _coverage_for_floor(subset, float(floor), refine["gap_break_m"],
                                           refine["min_interval_thickness_m"])
            if best is None or abs(coverage - float(target)) < abs(best["coverage"] - float(target)):
                best = {"floor": float(floor), "coverage": float(coverage)}
        floors[strata_name] = {**best, "target": float(target), "status": "ok"}
    return floors


def _bin_label(value: float, edges: list[float], names: list[str]) -> str:
    for edge, name in zip(edges, names):
        if value < edge:
            return name
    return names[-1]


def build_intervals(merged: pd.DataFrame, floors: dict[str, dict[str, float]],
                    config: dict[str, Any]) -> pd.DataFrame:
    """在预测密度曲线上切"裂缝发育区间"（替代原来的阈值连通段）。"""
    refine = refine_config(config)
    density_edges = [float(v) for v in refine["density_bins"]]
    thickness_edges = [float(v) for v in refine["thickness_bins"]]
    rows: list[dict[str, object]] = []
    for (well, strata_name), group in merged.groupby(["WellName", "StrataName"], sort=False):
        ordered = group.sort_values("TVD").reset_index(drop=True)
        depth = numeric(ordered["TVD"]).to_numpy(dtype=float)
        density = numeric(ordered["PredDensity"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        floor = float(floors.get(strata_name, {}).get("floor", 0.0))
        for i, j in _interval_bounds(depth, density >= floor, refine["gap_break_m"],
                                     refine["min_interval_thickness_m"]):
            local = density[i : j + 1]
            local_depth = depth[i : j + 1]
            integral = float(np.trapezoid(local, local_depth)) if len(local) >= 2 else float(local[0]) * 0.125
            thickness = float(local_depth[-1] - local_depth[0])
            mean_density = integral / thickness if thickness > 0 else 0.0
            rows.append({
                "WellName": well, "StrataName": strata_name,
                "TVDStart": float(local_depth[0]), "TVDEnd": float(local_depth[-1]),
                "ThicknessM": round(thickness, 3), "MeanDensity": round(mean_density, 4),
                "DensityIntegral": round(integral, 3),
                "ExpectedCount": int(max(1, round(integral))),
                "DensityFloor": round(floor, 4),
                "DensityBin": _bin_label(mean_density, density_edges, [f"低<{density_edges[0]}", f"中{density_edges[0]}-{density_edges[1]}", f"高≥{density_edges[1]}"]),
                "ThicknessBin": _bin_label(thickness, thickness_edges, [f"薄<{thickness_edges[0]}m", f"中{thickness_edges[0]}-{thickness_edges[1]}m", f"厚≥{thickness_edges[1]}m"]),
            })
    intervals = pd.DataFrame(rows)
    if not intervals.empty:
        intervals.insert(0, "IntervalID", [f"INT{index:05d}" for index in range(1, len(intervals) + 1)])
    return intervals


def place_points_in_intervals(merged: pd.DataFrame, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """区间内落点：按密度累积曲线等分（点数 = ∫ρdz），不再挑密度最大的采样点。"""
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for _, interval in intervals.iterrows():
        subset = merged[(merged.WellName.eq(interval.WellName)) & (merged.StrataName.eq(interval.StrataName))
                        & (merged.TVD >= interval.TVDStart - 1.0e-9) & (merged.TVD <= interval.TVDEnd + 1.0e-9)]
        subset = subset.sort_values("TVD").reset_index(drop=True)
        depth = subset.TVD.to_numpy(dtype=float)
        density = numeric(subset.PredDensity).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        count = int(interval.ExpectedCount)
        placed = 0
        if len(depth) >= 2 and count > 0:
            cumulative = np.concatenate(([0.0], np.cumsum((density[1:] + density[:-1]) * 0.5 * np.diff(depth))))
            total = cumulative[-1]
            used: set[int] = set()
            if total > 0:
                targets = (np.arange(count) + 0.5) / count * total
                for ordinal, target in enumerate(targets, start=1):
                    index = int(np.clip(np.searchsorted(cumulative, target), 0, len(cumulative) - 1))
                    if index in used:
                        continue
                    used.add(index)
                    source = subset.iloc[index]
                    rows.append({
                        "WellName": source.WellName, "TVD": float(source.TVD), "MD": float(source.MD),
                        "X": float(source.X), "Y": float(source.Y), "TIME": float(source.TIME),
                        "StrataName": source.StrataName, "PredDensity": float(source.PredDensity),
                        "PredFractureProb": float(source.PredFractureProb),
                        "SupportSegmentID": int(source.SupportSegmentID),
                        "PointOrdinalInSegment": ordinal,
                        "IntervalID": str(interval.IntervalID),
                        "DensityBin": str(interval.DensityBin),
                        "ThicknessBin": str(interval.ThicknessBin),
                        "IntervalStartTVD": float(interval.TVDStart),
                        "IntervalEndTVD": float(interval.TVDEnd),
                        "RuleSource": "density_cumulative",
                        "SourceCount": int(source.SourceCount),
                        "SourceSegmentIDs": str(source.SourceSegmentIDs),
                    })
                    placed += 1
        audit.append({
            "WellName": interval.WellName, "StrataName": interval.StrataName, "IntervalID": interval.IntervalID,
            "DensityBin": interval.DensityBin, "ThicknessBin": interval.ThicknessBin,
            "ThicknessM": interval.ThicknessM, "MeanDensity": interval.MeanDensity,
            "DensityIntegral": interval.DensityIntegral,
            "ExpectedCount": int(interval.ExpectedCount), "PlacedCount": int(placed),
            "Shortfall": int(interval.ExpectedCount) - int(placed),
        })
    return pd.DataFrame(rows), pd.DataFrame(audit)


def build_refine_rule_library(training: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, object]]:
    """从成像井（有真实密度与真实解释点）统计"不同密度档、不同厚度档的点数与间距"。"""
    refine = refine_config(config)
    density_edges = [float(v) for v in refine["density_bins"]]
    thickness_edges = [float(v) for v in refine["thickness_bins"]]
    library: list[dict[str, object]] = []
    for strata_name, group in training.groupby("StrataName", sort=False):
        for (bin_name, thickness_name), cell in _imaging_interval_cells(group, refine, density_edges, thickness_edges):
            library.append({
                "StrataName": strata_name, "DensityBin": bin_name, "ThicknessBin": thickness_name,
                "IntervalCount": int(cell["interval_count"]),
                "WellCount": int(cell["well_count"]),
                "MeanThicknessM": round(float(cell["mean_thickness"]), 3),
                "MeanDensity": round(float(cell["mean_density"]), 4),
                "ExpectedCountFromIntegral": round(float(cell["integral"]), 2),
                "ObservedPointCount": int(cell["points"]),
                "ObservedOverIntegral": round(float(cell["points"] / cell["integral"]), 3) if cell["integral"] > 0 else None,
                "MedianSpacingM": round(float(cell["median_spacing"]), 3) if cell["median_spacing"] else None,
                "P90SpacingM": round(float(cell["p90_spacing"]), 3) if cell["p90_spacing"] else None,
                "Support": "ok" if cell["interval_count"] >= 3 else "thin_support",
            })
    return library


def _imaging_interval_cells(frame: pd.DataFrame, refine: dict[str, Any], density_edges: list[float],
                            thickness_edges: list[float]):
    cells: dict[tuple[str, str], dict[str, object]] = {}
    for (well, strata_name), group in frame.groupby(["WellName", "StrataName"], sort=False):
        ordered = group.sort_values("TVD").reset_index(drop=True)
        depth = numeric(ordered["TVD"]).to_numpy(dtype=float)
        density = numeric(ordered["Density"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        for i, j in _interval_bounds(depth, density > 0.0, refine["gap_break_m"], refine["min_interval_thickness_m"]):
            local_depth = depth[i : j + 1]
            integral = float(np.trapezoid(density[i : j + 1], local_depth))
            thickness = float(local_depth[-1] - local_depth[0])
            mean_density = integral / thickness if thickness > 0 else 0.0
            bin_name = _bin_label(mean_density, density_edges, ["低", "中", "高"])
            thickness_name = _bin_label(thickness, thickness_edges, ["薄", "中", "厚"])
            point_depths = np.sort(np.concatenate([
                numeric(group.loc[(group.TVD >= local_depth[0]) & (group.TVD <= local_depth[-1]) & group.GT_POINT_FLAG.gt(0), "TVD"]).to_numpy(dtype=float)
            ])) if "GT_POINT_FLAG" in group.columns else np.array([])
            spacings = np.diff(point_depths)
            spacings = spacings[spacings > 0]
            key = (bin_name, thickness_name)
            cell = cells.setdefault(key, {"interval_count": 0, "wells": set(), "thickness": [], "density": [],
                                          "integral": 0.0, "points": 0, "spacings": []})
            cell["interval_count"] += 1
            cell["wells"].add(well)
            cell["thickness"].append(thickness)
            cell["density"].append(mean_density)
            cell["integral"] += integral
            cell["points"] += int(len(point_depths))
            cell["spacings"].extend(spacings.tolist())
    for key, cell in cells.items():
        spacings = np.asarray(cell["spacings"], dtype=float)
        cell["mean_thickness"] = float(np.mean(cell["thickness"])) if cell["thickness"] else 0.0
        cell["mean_density"] = float(np.mean(cell["density"])) if cell["density"] else 0.0
        cell["well_count"] = len(cell["wells"])
        cell["median_spacing"] = float(np.median(spacings)) if spacings.size else None
        cell["p90_spacing"] = float(np.quantile(spacings, 0.9)) if spacings.size else None
        yield key, cell


def refine_points(merged: pd.DataFrame, config: dict[str, Any],
                  training: pd.DataFrame) -> dict[str, Any]:
    """细化：定地板 → 切区间 → 按密度累积落点。点数 = ∫ρdz，不再乘地层常数。"""
    floors = select_density_floors(merged, config)
    intervals = build_intervals(merged, floors, config)
    points, interval_audit = place_points_in_intervals(merged, intervals)
    library = build_refine_rule_library(training, config)
    return {
        "points": points, "intervals": intervals, "interval_audit": interval_audit,
        "floors": floors, "library": library,
    }


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
        detail = build_prediction_rows_for_well(well_rows, experts, window_policy=config.get("feature_windows"))
        if detail.empty:
            coverage_rows.append({"WellName": well, "Status": "no_prediction_rows", "Rows": int(len(well_rows))})
            continue
        detail_parts.append(detail)
        merged = merge_well_predictions(
            detail, float(config["merge"]["tolerance_m"]),
            float(config["merge"]["gap_break_m"]), str(config["merge"]["weight_scheme"]),
            rule=str(config["merge"].get("rule", "mean")),
            select_priority=config["merge"].get("select_priority"),
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
    training = apply_supervision_gate(training, cfg)
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
            "DensityScaleOOFDiagnostic": expert["density_scale_oof_diagnostic"],
            "CalibrationActualTotal": expert["calibration_actual_total"],
            "CalibrationPredictedTotal": expert["calibration_predicted_total"],
            "CalibrationWells": ";".join(expert["calibration_wells"]),
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
    refine = refine_points(merged_all, cfg, training)
    points = refine["points"]
    points.to_csv(prediction_dir / "all_wells_merged_fracture_points.csv", index=False, encoding="utf-8-sig")
    refine["intervals"].to_csv(prediction_dir / "all_wells_refine_intervals.csv", index=False, encoding="utf-8-sig")
    refine["interval_audit"].to_csv(qc_dir / "refine_interval_audit.csv", index=False, encoding="utf-8-sig")
    (model_dir / "refine_rule_library.json").write_text(
        json.dumps({"version": "taigu_step4_refine_rule_v1",
                    "unit": {"density": "count/m", "thickness": "m", "spacing": "m"},
                    "floors": refine["floors"], "rules": refine["library"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if not points.empty:
        point_audit = (
            points.groupby(["WellName", "StrataName"]).size().rename("PlacedPoints").reset_index()
            .merge(refine["intervals"].groupby(["WellName", "StrataName"]).agg(
                IntervalCount=("IntervalID", "size"), ExpectedPoints=("ExpectedCount", "sum"),
                IntervalThicknessM=("ThicknessM", "sum"), ExpectedIntegral=("DensityIntegral", "sum")).reset_index(),
                on=["WellName", "StrataName"], how="left")
        )
        point_audit["PlacedOverExpected"] = (
            point_audit.PlacedPoints / point_audit.ExpectedPoints.replace(0, np.nan)
        ).round(3)
        point_audit.to_csv(qc_dir / "refine_point_count_audit.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(qc_dir / "refine_point_count_audit.csv", index=False, encoding="utf-8-sig")

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
        "supervision_gate": supervision_gate(cfg),
        "supervision_tier_rows": {str(k): int(v) for k, v in training.SupervisionTier.value_counts().items()},
        "supervision_tier_wells": {str(well): str(tier) for well, tier in
                                   training.groupby("WellName").SupervisionTier.first().items()},
        "refine_floors": refine["floors"],
        "refine_interval_count": int(len(refine["intervals"])),
        "refine_rule_count": int(len(refine["library"])),
        "errors": errors,
    }
    (out_dir / "taigu_step4_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "version": "taigu_step4_gr_rd_rs_v1",
        "all_wells_merged_prediction_table": str(prediction_dir / "all_wells_merged_density_prediction.csv"),
        "all_wells_source_detail_table": str(prediction_dir / "all_wells_source_detail_predictions.csv"),
        "all_wells_fracture_points_table": str(prediction_dir / "all_wells_merged_fracture_points.csv"),
        "all_wells_refine_intervals_table": str(prediction_dir / "all_wells_refine_intervals.csv"),
        "refine_rule_library": str(model_dir / "refine_rule_library.json"),
        "refine_interval_audit": str(qc_dir / "refine_interval_audit.csv"),
        "refine_point_count_audit": str(qc_dir / "refine_point_count_audit.csv"),
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
