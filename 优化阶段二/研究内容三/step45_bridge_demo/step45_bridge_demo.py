from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None


DEFAULT_ID_COLUMNS = {
    "SampleID",
    "SourceWellName",
    "VirtualWellName",
    "VirtualWellID",
    "WellName",
    "SourceType",
}


@dataclass
class BridgeConfig:
    sample_csv: str
    relation_bins_csv: str
    relation_summary_csv: str
    output_dir: str
    layer_col: str = "LayerGroup"
    x_col: str = "X"
    y_col: str = "Y"
    time_col: str = "TIME"
    target_col: str = "FractureDensity"
    weak_target_col: str = "FractureDensityWeak"
    point_conf_col: str = "PointConfidence"
    well_conf_col: str = "WellConfidence"
    min_attr_weight: float = 0.05
    min_valid_attributes: int = 1
    device: str = "auto"


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_csv(path: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"输入文件为空: {path}")
    return table


def find_base_density_column(table: pd.DataFrame, config: BridgeConfig) -> str:
    if config.weak_target_col in table.columns:
        return config.weak_target_col
    if config.target_col in table.columns:
        return config.target_col
    raise KeyError(
        f"样本表中既没有 {config.weak_target_col}，也没有 {config.target_col}，无法构造第五步局部控制分数。"
    )


def normalize_sample_table(table: pd.DataFrame, config: BridgeConfig) -> pd.DataFrame:
    required = [config.layer_col, config.x_col, config.y_col, config.time_col]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise KeyError(f"样本表缺少必需字段: {missing}")

    normalized = table.copy()
    normalized[config.layer_col] = normalized[config.layer_col].astype(str)
    for column in [config.x_col, config.y_col, config.time_col]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if config.point_conf_col not in normalized.columns:
        normalized[config.point_conf_col] = 1.0
    normalized[config.point_conf_col] = (
        pd.to_numeric(normalized[config.point_conf_col], errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0)
    )

    if config.well_conf_col not in normalized.columns:
        normalized[config.well_conf_col] = 1.0
    normalized[config.well_conf_col] = (
        pd.to_numeric(normalized[config.well_conf_col], errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0)
    )

    base_density_col = find_base_density_column(normalized, config)
    normalized[base_density_col] = pd.to_numeric(normalized[base_density_col], errors="coerce")

    for optional_col in [config.target_col, config.weak_target_col]:
        if optional_col in normalized.columns:
            normalized[optional_col] = pd.to_numeric(normalized[optional_col], errors="coerce")

    return normalized


def normalize_relation_tables(
    relation_bins: pd.DataFrame,
    relation_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_required = {
        "layer_group",
        "attribute",
        "bin_index",
        "bin_left",
        "bin_right",
        "weighted_fracture_density_mean",
    }
    summary_required = {
        "layer_group",
        "attribute",
        "status",
        "weighted_pearson",
    }
    missing_bins = sorted(bin_required.difference(relation_bins.columns))
    missing_summary = sorted(summary_required.difference(relation_summary.columns))
    if missing_bins:
        raise KeyError(f"relation_bins.csv 缺少字段: {missing_bins}")
    if missing_summary:
        raise KeyError(f"relation_summary.csv 缺少字段: {missing_summary}")

    bins = relation_bins.copy()
    summary = relation_summary.copy()
    bins["layer_group"] = bins["layer_group"].astype(str)
    bins["attribute"] = bins["attribute"].astype(str)
    summary["layer_group"] = summary["layer_group"].astype(str)
    summary["attribute"] = summary["attribute"].astype(str)

    numeric_cols = [
        "bin_index",
        "bin_left",
        "bin_right",
        "weighted_fracture_density_mean",
        "weighted_sample_count",
    ]
    for column in numeric_cols:
        if column in bins.columns:
            bins[column] = pd.to_numeric(bins[column], errors="coerce")

    for column in ["weighted_pearson", "valid_samples", "weighted_attr_mean", "weighted_attr_std"]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    return bins, summary


def build_attribute_profiles(
    relation_bins: pd.DataFrame,
    relation_summary: pd.DataFrame,
    min_attr_weight: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    usable_summary = relation_summary[relation_summary["status"] == "ok"].copy()

    for row in usable_summary.itertuples(index=False):
        layer = str(row.layer_group)
        attribute = str(row.attribute)
        pearson = float(row.weighted_pearson) if pd.notna(row.weighted_pearson) else np.nan
        if not np.isfinite(pearson):
            continue
        attr_weight = abs(pearson)
        if attr_weight < float(min_attr_weight):
            continue

        bins = relation_bins[
            (relation_bins["layer_group"] == layer) & (relation_bins["attribute"] == attribute)
        ].copy()
        bins = bins.sort_values("bin_index")
        if bins.empty:
            continue

        profiles[(layer, attribute)] = {
            "layer_group": layer,
            "attribute": attribute,
            "pearson": pearson,
            "attr_weight": attr_weight,
            "bin_left": bins["bin_left"].to_numpy(dtype=np.float64),
            "bin_right": bins["bin_right"].to_numpy(dtype=np.float64),
            "bin_mean_density": bins["weighted_fracture_density_mean"].to_numpy(dtype=np.float64),
        }
    return profiles


def score_value_from_profile(value: float, profile: dict[str, Any]) -> float:
    if not np.isfinite(value):
        return float("nan")

    left = profile["bin_left"]
    right = profile["bin_right"]
    density = profile["bin_mean_density"]
    if left.size == 0:
        return float("nan")

    if value <= right[0]:
        return float(density[0])
    if value >= left[-1]:
        return float(density[-1])

    for index in range(len(left)):
        is_last = index == len(left) - 1
        if left[index] <= value < right[index] or (is_last and left[index] <= value <= right[index]):
            return float(density[index])
    return float("nan")


def aggregate_scores_numpy(
    score_matrix: np.ndarray,
    weight_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(score_matrix) & np.isfinite(weight_matrix) & (weight_matrix > 0)
    weighted_scores = np.where(valid_mask, score_matrix * weight_matrix, 0.0)
    effective_weights = np.where(valid_mask, weight_matrix, 0.0)
    score_sum = weighted_scores.sum(axis=1)
    weight_sum = effective_weights.sum(axis=1)
    final_score = np.full(score_sum.shape, np.nan, dtype=np.float64)
    np.divide(score_sum, weight_sum, out=final_score, where=weight_sum > 0)
    valid_attr_count = valid_mask.sum(axis=1).astype(np.int32)
    return final_score, valid_attr_count


def aggregate_scores_torch(
    score_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    tensor_device = torch.device(device)
    score_tensor = torch.as_tensor(score_matrix, dtype=torch.float32, device=tensor_device)
    weight_tensor = torch.as_tensor(weight_matrix, dtype=torch.float32, device=tensor_device)
    valid_tensor = torch.isfinite(score_tensor) & torch.isfinite(weight_tensor) & (weight_tensor > 0)
    effective_weight = torch.where(valid_tensor, weight_tensor, torch.zeros_like(weight_tensor))
    weighted_score = torch.where(valid_tensor, score_tensor * weight_tensor, torch.zeros_like(score_tensor))
    score_sum = weighted_score.sum(dim=1)
    weight_sum = effective_weight.sum(dim=1)
    final_score = torch.full_like(score_sum, float("nan"))
    valid_rows = weight_sum > 0
    final_score[valid_rows] = score_sum[valid_rows] / weight_sum[valid_rows]
    valid_attr_count = valid_tensor.sum(dim=1)
    return final_score.cpu().numpy(), valid_attr_count.cpu().numpy()


def build_scored_sample_table(
    sample_table: pd.DataFrame,
    profiles: dict[tuple[str, str], dict[str, Any]],
    config: BridgeConfig,
    device: str,
) -> tuple[pd.DataFrame, list[str]]:
    base_density_col = find_base_density_column(sample_table, config)
    working = sample_table.copy()
    used_attributes = sorted({attribute for (_, attribute) in profiles.keys()})

    score_columns: list[str] = []
    score_arrays: list[np.ndarray] = []
    weight_arrays: list[np.ndarray] = []

    for attribute in used_attributes:
        score_col = f"{attribute}_Score"
        weight_col = f"{attribute}_AttrWeight"
        score_columns.append(score_col)

        scores = np.full(len(working), np.nan, dtype=np.float64)
        weights = np.zeros(len(working), dtype=np.float64)

        if attribute not in working.columns:
            working[score_col] = scores
            working[weight_col] = weights
            score_arrays.append(scores)
            weight_arrays.append(weights)
            continue

        values = pd.to_numeric(working[attribute], errors="coerce").to_numpy(dtype=np.float64)
        layers = working[config.layer_col].astype(str).to_numpy()

        for index, (layer, value) in enumerate(zip(layers, values, strict=False)):
            profile = profiles.get((layer, attribute))
            if profile is None:
                continue
            score = score_value_from_profile(value, profile)
            if np.isfinite(score):
                scores[index] = score
                weights[index] = float(profile["attr_weight"])

        working[score_col] = scores
        working[weight_col] = weights
        score_arrays.append(scores)
        weight_arrays.append(weights)

    if score_arrays:
        score_matrix = np.column_stack(score_arrays)
        weight_matrix = np.column_stack(weight_arrays)
        use_torch = torch is not None and device.startswith("cuda")
        if use_torch:
            final_score, valid_attr_count = aggregate_scores_torch(score_matrix, weight_matrix, device)
        else:
            final_score, valid_attr_count = aggregate_scores_numpy(score_matrix, weight_matrix)
    else:
        final_score = np.full(len(working), np.nan, dtype=np.float64)
        valid_attr_count = np.zeros(len(working), dtype=np.int32)

    combined_conf = (
        working[config.point_conf_col].to_numpy(dtype=np.float64)
        * working[config.well_conf_col].to_numpy(dtype=np.float64)
    )
    base_density = working[base_density_col].to_numpy(dtype=np.float64)

    working["BaseDensity"] = base_density
    working["CombinedConfidence"] = combined_conf
    working["ValidAttributeCount"] = valid_attr_count
    working["ControlScoreRaw"] = final_score
    working["ControlScore"] = np.where(
        np.isfinite(final_score) & np.isfinite(base_density),
        0.5 * final_score + 0.5 * base_density,
        np.where(np.isfinite(final_score), final_score, base_density),
    )
    return working, score_columns


def build_local_control_field(scored_samples: pd.DataFrame, config: BridgeConfig) -> pd.DataFrame:
    required = [
        config.layer_col,
        config.x_col,
        config.y_col,
        config.time_col,
        "ControlScore",
        "CombinedConfidence",
        "ValidAttributeCount",
    ]
    field_input = scored_samples[required].copy()
    field_input["ScoreWeight"] = field_input["CombinedConfidence"].clip(lower=0.0)
    field_input["WeightedScore"] = field_input["ControlScore"] * field_input["ScoreWeight"]
    field_input["HasScore"] = field_input["ControlScore"].notna().astype(int)
    field_input["HasBaseDensity"] = scored_samples["BaseDensity"].notna().astype(int)

    grouped = (
        field_input.groupby(
            [config.layer_col, config.x_col, config.y_col, config.time_col],
            dropna=False,
            as_index=False,
        )
        .agg(
            WeightedScoreSum=("WeightedScore", "sum"),
            ScoreWeightSum=("ScoreWeight", "sum"),
            SampleCount=("HasScore", "size"),
            ScoredSampleCount=("HasScore", "sum"),
            BaseDensitySupport=("HasBaseDensity", "sum"),
            MeanValidAttributeCount=("ValidAttributeCount", "mean"),
        )
    )

    grouped["ControlScore"] = np.nan
    valid = grouped["ScoreWeightSum"] > 0
    grouped.loc[valid, "ControlScore"] = (
        grouped.loc[valid, "WeightedScoreSum"] / grouped.loc[valid, "ScoreWeightSum"]
    )

    grouped["ControlClass"] = "unknown"
    grouped.loc[grouped["ControlScore"] >= 0.65, "ControlClass"] = "high"
    grouped.loc[
        grouped["ControlScore"].between(0.45, 0.65, inclusive="left"),
        "ControlClass",
    ] = "medium"
    grouped.loc[grouped["ControlScore"] < 0.45, "ControlClass"] = "low"
    return grouped


def build_summary(
    scored_samples: pd.DataFrame,
    control_field: pd.DataFrame,
    profiles: dict[tuple[str, str], dict[str, Any]],
    config: BridgeConfig,
    device: str,
    used_score_columns: list[str],
) -> dict[str, Any]:
    layer_summary: dict[str, Any] = {}
    for layer, layer_df in control_field.groupby(config.layer_col):
        layer_summary[str(layer)] = {
            "grid_point_count": int(len(layer_df)),
            "scored_grid_point_count": int(layer_df["ControlScore"].notna().sum()),
            "high_count": int((layer_df["ControlClass"] == "high").sum()),
            "medium_count": int((layer_df["ControlClass"] == "medium").sum()),
            "low_count": int((layer_df["ControlClass"] == "low").sum()),
            "control_score_mean": (
                float(layer_df["ControlScore"].mean())
                if layer_df["ControlScore"].notna().any()
                else None
            ),
        }

    profile_records = [
        {
            "layer_group": layer,
            "attribute": attribute,
            "attr_weight": float(profile["attr_weight"]),
            "pearson": float(profile["pearson"]),
            "bin_count": int(len(profile["bin_left"])),
        }
        for (layer, attribute), profile in sorted(profiles.items())
    ]

    return {
        "sample_csv": config.sample_csv,
        "relation_bins_csv": config.relation_bins_csv,
        "relation_summary_csv": config.relation_summary_csv,
        "resolved_device": device,
        "used_attributes": sorted({attribute for (_, attribute) in profiles.keys()}),
        "used_score_columns": used_score_columns,
        "profile_count": int(len(profile_records)),
        "profiles": profile_records,
        "sample_count": int(len(scored_samples)),
        "scored_sample_count": int(scored_samples["ControlScore"].notna().sum()),
        "control_field_count": int(len(control_field)),
        "layer_summary": layer_summary,
    }


def run_step45_bridge_demo(config: BridgeConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(config.device)
    sample_table = normalize_sample_table(load_csv(config.sample_csv), config)
    relation_bins, relation_summary = normalize_relation_tables(
        load_csv(config.relation_bins_csv),
        load_csv(config.relation_summary_csv),
    )
    profiles = build_attribute_profiles(
        relation_bins=relation_bins,
        relation_summary=relation_summary,
        min_attr_weight=config.min_attr_weight,
    )
    if not profiles:
        raise ValueError("没有可用的层位-属性关系配置，无法构造第五步局部控制场。")

    scored_samples, used_score_columns = build_scored_sample_table(
        sample_table=sample_table,
        profiles=profiles,
        config=config,
        device=device,
    )
    if config.min_valid_attributes > 1:
        scored_samples.loc[
            scored_samples["ValidAttributeCount"] < int(config.min_valid_attributes),
            "ControlScore",
        ] = np.nan

    control_field = build_local_control_field(scored_samples, config)
    summary = build_summary(
        scored_samples=scored_samples,
        control_field=control_field,
        profiles=profiles,
        config=config,
        device=device,
        used_score_columns=used_score_columns,
    )

    scored_samples.to_csv(output_dir / "scored_samples.csv", index=False)
    control_field.to_csv(output_dir / "local_control_field.csv", index=False)
    with (output_dir / "control_field_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return {
        "scored_samples_path": str(output_dir / "scored_samples.csv"),
        "local_control_field_path": str(output_dir / "local_control_field.csv"),
        "control_field_summary_path": str(output_dir / "control_field_summary.json"),
        "resolved_device": device,
        "used_attributes": summary["used_attributes"],
    }
