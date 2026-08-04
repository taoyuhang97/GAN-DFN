from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - runtime fallback
    torch = None


DEFAULT_EXCLUDE_COLUMNS = {
    "SampleID",
    "SourceType",
    "WellName",
    "VirtualWellName",
    "VirtualWellID",
    "X",
    "Y",
    "T",
    "TIME",
    "Depth",
    "MD",
}


@dataclass
class DemoConfig:
    input_csv: str
    output_dir: str
    layer_col: str = "LayerGroup"
    target_col: str = "FractureDensity"
    weight_col: str = "PointConfidence"
    attribute_cols: list[str] | None = None
    bins: int = 10
    min_samples: int = 20
    device: str = "auto"
    layers: list[str] | None = None


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_input_table(path: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError("输入样本表为空，无法进行第四步 demo。")
    return table


def infer_attribute_columns(
    table: pd.DataFrame,
    layer_col: str,
    target_col: str,
    weight_col: str,
) -> list[str]:
    excluded = set(DEFAULT_EXCLUDE_COLUMNS)
    excluded.update({layer_col, target_col, weight_col})
    candidates: list[str] = []
    for column in table.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(table[column]):
            candidates.append(column)
    return candidates


def normalize_input_table(
    table: pd.DataFrame,
    layer_col: str,
    target_col: str,
    weight_col: str,
    attribute_cols: list[str],
) -> pd.DataFrame:
    required = [layer_col, target_col]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise KeyError(f"输入样本表缺少必需字段: {missing}")

    normalized = table.copy()
    normalized[layer_col] = normalized[layer_col].astype(str)
    normalized[target_col] = pd.to_numeric(normalized[target_col], errors="coerce")

    if weight_col not in normalized.columns:
        normalized[weight_col] = 1.0
    normalized[weight_col] = pd.to_numeric(normalized[weight_col], errors="coerce").fillna(1.0)

    for column in attribute_cols:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def weighted_pearson_numpy(values: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float("nan")
    mean_x = float(np.sum(values * weights) / total_weight)
    mean_y = float(np.sum(targets * weights) / total_weight)
    dx = values - mean_x
    dy = targets - mean_y
    cov = float(np.sum(weights * dx * dy) / total_weight)
    var_x = float(np.sum(weights * dx * dx) / total_weight)
    var_y = float(np.sum(weights * dy * dy) / total_weight)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return cov / np.sqrt(var_x * var_y)


def build_quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([values.min(), values.max()], dtype=np.float64)
    return edges.astype(np.float64)


def aggregate_with_torch(
    values: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tensor_device = torch.device(device)
    value_tensor = torch.as_tensor(values, dtype=torch.float32, device=tensor_device)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=tensor_device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=tensor_device)
    edge_tensor = torch.as_tensor(edges[1:-1], dtype=torch.float32, device=tensor_device)

    bucket_index = torch.bucketize(value_tensor, edge_tensor)
    bucket_count = len(edges) - 1

    count_tensor = torch.zeros(bucket_count, dtype=torch.float32, device=tensor_device)
    weight_sum_tensor = torch.zeros(bucket_count, dtype=torch.float32, device=tensor_device)
    value_sum_tensor = torch.zeros(bucket_count, dtype=torch.float32, device=tensor_device)
    target_sum_tensor = torch.zeros(bucket_count, dtype=torch.float32, device=tensor_device)
    target_sq_sum_tensor = torch.zeros(bucket_count, dtype=torch.float32, device=tensor_device)

    ones = torch.ones_like(weight_tensor)
    count_tensor.index_add_(0, bucket_index, ones)
    weight_sum_tensor.index_add_(0, bucket_index, weight_tensor)
    value_sum_tensor.index_add_(0, bucket_index, value_tensor * weight_tensor)
    target_sum_tensor.index_add_(0, bucket_index, target_tensor * weight_tensor)
    target_sq_sum_tensor.index_add_(0, bucket_index, target_tensor * target_tensor * weight_tensor)

    counts = count_tensor.cpu().numpy()
    weight_sums = weight_sum_tensor.cpu().numpy()
    value_sums = value_sum_tensor.cpu().numpy()
    target_sums = target_sum_tensor.cpu().numpy()
    target_sq_sums = target_sq_sum_tensor.cpu().numpy()
    value_means = np.zeros_like(weight_sums, dtype=np.float64)
    target_means = np.zeros_like(weight_sums, dtype=np.float64)
    target_sq_means = np.zeros_like(weight_sums, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(value_sums, weight_sums, out=value_means, where=weight_sums > 0)
        np.divide(target_sums, weight_sums, out=target_means, where=weight_sums > 0)
        np.divide(target_sq_sums, weight_sums, out=target_sq_means, where=weight_sums > 0)
    target_vars = np.maximum(target_sq_means - target_means**2, 0.0)
    target_stds = np.sqrt(target_vars)
    return counts, weight_sums, value_means, target_means, target_stds


def aggregate_with_numpy(
    values: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bucket_index = np.digitize(values, edges[1:-1], right=False)
    bucket_count = len(edges) - 1

    counts = np.bincount(bucket_index, minlength=bucket_count).astype(np.float64)
    weight_sums = np.bincount(bucket_index, weights=weights, minlength=bucket_count).astype(np.float64)
    value_weighted_sum = np.bincount(
        bucket_index, weights=values * weights, minlength=bucket_count
    ).astype(np.float64)
    target_weighted_sum = np.bincount(
        bucket_index, weights=targets * weights, minlength=bucket_count
    ).astype(np.float64)
    target_sq_weighted_sum = np.bincount(
        bucket_index, weights=targets * targets * weights, minlength=bucket_count
    ).astype(np.float64)

    value_means = np.zeros_like(weight_sums, dtype=np.float64)
    target_means = np.zeros_like(weight_sums, dtype=np.float64)
    target_sq_means = np.zeros_like(weight_sums, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(value_weighted_sum, weight_sums, out=value_means, where=weight_sums > 0)
        np.divide(target_weighted_sum, weight_sums, out=target_means, where=weight_sums > 0)
        np.divide(target_sq_weighted_sum, weight_sums, out=target_sq_means, where=weight_sums > 0)
    target_vars = np.maximum(target_sq_means - target_means**2, 0.0)
    target_stds = np.sqrt(target_vars)
    return counts, weight_sums, value_means, target_means, target_stds


def summarize_attribute(
    values: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    total_weight = float(weights.sum())
    attr_mean = float(np.sum(values * weights) / total_weight)
    target_mean = float(np.sum(targets * weights) / total_weight)
    attr_var = float(np.sum(weights * (values - attr_mean) ** 2) / total_weight)
    target_var = float(np.sum(weights * (targets - target_mean) ** 2) / total_weight)
    return {
        "weighted_attr_mean": attr_mean,
        "weighted_attr_std": float(np.sqrt(max(attr_var, 0.0))),
        "weighted_target_mean": target_mean,
        "weighted_target_std": float(np.sqrt(max(target_var, 0.0))),
        "weighted_pearson": weighted_pearson_numpy(values, targets, weights),
    }


def run_relation_demo(config: DemoConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chosen_device = choose_device(config.device)
    table = load_input_table(config.input_csv)

    attribute_cols = config.attribute_cols
    if not attribute_cols:
        attribute_cols = infer_attribute_columns(
            table=table,
            layer_col=config.layer_col,
            target_col=config.target_col,
            weight_col=config.weight_col,
        )
    if not attribute_cols:
        raise ValueError("未识别到可用属性列，请通过 --attribute-cols 手动指定。")

    normalized = normalize_input_table(
        table=table,
        layer_col=config.layer_col,
        target_col=config.target_col,
        weight_col=config.weight_col,
        attribute_cols=attribute_cols,
    )

    available_layers = sorted(normalized[config.layer_col].dropna().astype(str).unique().tolist())
    layers = config.layers if config.layers else available_layers

    relation_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for layer in layers:
        layer_table = normalized[normalized[config.layer_col] == str(layer)].copy()
        if layer_table.empty:
            continue

        for attribute in attribute_cols:
            subset = layer_table[[attribute, config.target_col, config.weight_col]].dropna()
            if len(subset) < config.min_samples:
                summary_rows.append(
                    {
                        "layer_group": layer,
                        "attribute": attribute,
                        "status": "skipped_insufficient_samples",
                        "total_samples": int(len(layer_table)),
                        "valid_samples": int(len(subset)),
                    }
                )
                continue

            values = subset[attribute].to_numpy(dtype=np.float64)
            targets = subset[config.target_col].to_numpy(dtype=np.float64)
            weights = subset[config.weight_col].to_numpy(dtype=np.float64)

            if np.nanstd(values) == 0:
                summary_rows.append(
                    {
                        "layer_group": layer,
                        "attribute": attribute,
                        "status": "skipped_constant_attribute",
                        "total_samples": int(len(layer_table)),
                        "valid_samples": int(len(subset)),
                    }
                )
                continue

            edges = build_quantile_edges(values, config.bins)
            if len(edges) < 2:
                summary_rows.append(
                    {
                        "layer_group": layer,
                        "attribute": attribute,
                        "status": "skipped_invalid_bins",
                        "total_samples": int(len(layer_table)),
                        "valid_samples": int(len(subset)),
                    }
                )
                continue

            use_torch = torch is not None and chosen_device.startswith("cuda")
            if use_torch:
                counts, weight_sums, value_means, target_means, target_stds = aggregate_with_torch(
                    values=values,
                    targets=targets,
                    weights=weights,
                    edges=edges,
                    device=chosen_device,
                )
            else:
                counts, weight_sums, value_means, target_means, target_stds = aggregate_with_numpy(
                    values=values,
                    targets=targets,
                    weights=weights,
                    edges=edges,
                )

            for bin_index in range(len(edges) - 1):
                relation_rows.append(
                    {
                        "layer_group": layer,
                        "attribute": attribute,
                        "bin_index": int(bin_index),
                        "bin_left": float(edges[bin_index]),
                        "bin_right": float(edges[bin_index + 1]),
                        "sample_count": int(counts[bin_index]),
                        "weighted_sample_count": float(weight_sums[bin_index]),
                        "weighted_attr_mean": float(value_means[bin_index]),
                        "weighted_fracture_density_mean": float(target_means[bin_index]),
                        "weighted_fracture_density_std": float(target_stds[bin_index]),
                    }
                )

            summary = summarize_attribute(values=values, targets=targets, weights=weights)
            summary_rows.append(
                {
                    "layer_group": layer,
                    "attribute": attribute,
                    "status": "ok",
                    "total_samples": int(len(layer_table)),
                    "valid_samples": int(len(subset)),
                    **summary,
                }
            )

    relation_df = pd.DataFrame(relation_rows)
    summary_df = pd.DataFrame(summary_rows)

    relation_df.to_csv(output_dir / "relation_bins.csv", index=False)
    summary_df.to_csv(output_dir / "relation_summary.csv", index=False)

    run_config = {
        "input_csv": config.input_csv,
        "output_dir": config.output_dir,
        "layer_col": config.layer_col,
        "target_col": config.target_col,
        "weight_col": config.weight_col,
        "attribute_cols": attribute_cols,
        "bins": config.bins,
        "min_samples": config.min_samples,
        "requested_device": config.device,
        "resolved_device": chosen_device,
        "torch_available": torch is not None,
        "layers": layers,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    return {
        "relation_bins_path": str(output_dir / "relation_bins.csv"),
        "relation_summary_path": str(output_dir / "relation_summary.csv"),
        "run_config_path": str(output_dir / "run_config.json"),
        "attribute_cols": attribute_cols,
        "resolved_device": chosen_device,
        "layers": layers,
    }
