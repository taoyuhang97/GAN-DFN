#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch


DEFAULT_FEATURES = [
    "SEIS_TRUE",
    "COHERENCE",
    "ANT_TRACK",
    "CURVATURE_MAX",
    "CURVATURE_POS",
    "FRACTURE_INV",
    "FaultDistance",
]


@dataclass
class LayerModel:
    layer_group: str
    intercept: float
    coefficients: Dict[str, float]
    feature_mean: Dict[str, float]
    feature_std: Dict[str, float]
    score_min: float
    score_max: float
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "第四步到第五步桥接 demo：从统一样本表估计分层关系/权重，"
            "并生成可直接作为第五步输入的裂缝发育倾向分数。"
        )
    )
    parser.add_argument("--sample-csv", required=True, help="第四步统一样本表 CSV 路径。")
    parser.add_argument(
        "--target-csv",
        help=(
            "待打分的局部目标点 CSV。若不提供，则默认对样本表自身打分，"
            "用于验证第四步到第五步的最小闭环。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录，用于保存关系权重、得分结果和运行摘要。",
    )
    parser.add_argument(
        "--feature-cols",
        help=(
            "逗号分隔的特征列名。默认自动从 "
            f"{', '.join(DEFAULT_FEATURES)} 中选择样本表实际存在的列。"
        ),
    )
    parser.add_argument("--layer-col", default="LayerGroup", help="层组列名，默认 LayerGroup。")
    parser.add_argument(
        "--target-col",
        default="FractureDensity",
        help="样本表中的裂缝密度监督列名，默认 FractureDensity。",
    )
    parser.add_argument(
        "--confidence-col",
        default="PointConfidence",
        help="样本权重列名，默认 PointConfidence；若缺失则退化为全 1。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练和推理设备，默认 auto。",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=1.0,
        help="岭回归正则系数，默认 1.0。",
    )
    parser.add_argument(
        "--min-samples-per-layer",
        type=int,
        default=8,
        help="每个层组最少样本数。低于阈值的层组会被跳过。",
    )
    parser.add_argument(
        "--score-col",
        default="FractureTendencyScore",
        help="输出裂缝发育倾向分数列名，默认 FractureTendencyScore。",
    )
    parser.add_argument(
        "--include-raw-prediction",
        action="store_true",
        help="在输出中保留线性模型原始预测值，便于调试。",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境不可用。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {csv_path}")
    return pd.read_csv(csv_path)


def resolve_feature_columns(frame: pd.DataFrame, feature_arg: str | None) -> List[str]:
    if feature_arg:
        columns = [col.strip() for col in feature_arg.split(",") if col.strip()]
    else:
        columns = [col for col in DEFAULT_FEATURES if col in frame.columns]
    if not columns:
        raise ValueError("未找到可用特征列，请通过 --feature-cols 显式指定。")
    return columns


def build_weight_tensor(frame: pd.DataFrame, confidence_col: str, device: torch.device) -> torch.Tensor:
    if confidence_col in frame.columns:
        weights = pd.to_numeric(frame[confidence_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(weights.max()) <= 0.0:
            weights = pd.Series(np.ones(len(frame), dtype=np.float32), index=frame.index)
    else:
        weights = pd.Series(np.ones(len(frame), dtype=np.float32), index=frame.index)
    return torch.as_tensor(weights.to_numpy(dtype=np.float32), device=device)


def prepare_feature_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for feature in features:
        prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")
    return prepared


def fit_layer_model(
    layer_frame: pd.DataFrame,
    layer_name: str,
    features: Sequence[str],
    target_col: str,
    confidence_col: str,
    ridge_lambda: float,
    device: torch.device,
) -> LayerModel:
    clean = prepare_feature_frame(layer_frame, features)
    clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
    required_cols = list(features) + [target_col]
    clean = clean.dropna(subset=required_cols).copy()
    if clean.empty:
        raise ValueError("层组有效样本为空。")

    feature_mean = clean[list(features)].mean()
    feature_std = clean[list(features)].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    x_np = ((clean[list(features)] - feature_mean) / feature_std).to_numpy(dtype=np.float32)
    y_np = clean[target_col].to_numpy(dtype=np.float32)

    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(y_np, device=device).unsqueeze(-1)
    ones = torch.ones((x.shape[0], 1), device=device, dtype=torch.float32)
    x_aug = torch.cat([ones, x], dim=1)

    weights = build_weight_tensor(clean, confidence_col, device).unsqueeze(-1)
    sqrt_w = torch.sqrt(weights + 1e-8)
    xw = x_aug * sqrt_w
    yw = y * sqrt_w

    reg = torch.eye(x_aug.shape[1], device=device, dtype=torch.float32)
    reg[0, 0] = 0.0
    lhs = xw.T @ xw + ridge_lambda * reg
    rhs = xw.T @ yw
    beta = torch.linalg.solve(lhs, rhs).squeeze(-1)

    raw_pred = (x_aug @ beta).detach().cpu().numpy().astype(np.float64)
    score_min = float(np.nanmin(raw_pred))
    score_max = float(np.nanmax(raw_pred))
    if np.isclose(score_min, score_max):
        score_max = score_min + 1.0

    return LayerModel(
        layer_group=layer_name,
        intercept=float(beta[0].item()),
        coefficients={feature: float(beta[idx + 1].item()) for idx, feature in enumerate(features)},
        feature_mean={feature: float(feature_mean[feature]) for feature in features},
        feature_std={feature: float(feature_std[feature]) for feature in features},
        score_min=score_min,
        score_max=score_max,
        sample_count=int(len(clean)),
    )


def apply_layer_model(
    frame: pd.DataFrame,
    model: LayerModel,
    features: Sequence[str],
    score_col: str,
    include_raw_prediction: bool,
) -> pd.DataFrame:
    scored = frame.copy()
    standardized = []
    missing_mask = pd.Series(False, index=scored.index)
    for feature in features:
        values = pd.to_numeric(scored.get(feature), errors="coerce")
        missing_mask = missing_mask | values.isna()
        mean = model.feature_mean[feature]
        std = model.feature_std[feature] if model.feature_std[feature] != 0 else 1.0
        standardized.append(((values - mean) / std).fillna(0.0).to_numpy(dtype=np.float32))

    x = np.vstack(standardized).T if standardized else np.empty((len(scored), 0), dtype=np.float32)
    beta = np.array([model.coefficients[feature] for feature in features], dtype=np.float32)
    raw = model.intercept + x @ beta
    score = (raw - model.score_min) / (model.score_max - model.score_min)
    score = np.clip(score, 0.0, 1.0)

    scored[score_col] = score.astype(np.float32)
    scored["BridgeModelLayerGroup"] = model.layer_group
    scored["BridgeModelSampleCount"] = model.sample_count
    scored["BridgeScoreMissingFeatureFlag"] = missing_mask.astype(int)
    if include_raw_prediction:
        scored["BridgeRawPrediction"] = raw.astype(np.float32)
    return scored


def build_weight_summary(models: Dict[str, LayerModel], features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for layer_group, model in models.items():
        row = {
            "LayerGroup": layer_group,
            "Intercept": model.intercept,
            "SampleCount": model.sample_count,
            "ScoreMin": model.score_min,
            "ScoreMax": model.score_max,
        }
        for feature in features:
            row[f"Weight__{feature}"] = model.coefficients[feature]
            row[f"Mean__{feature}"] = model.feature_mean[feature]
            row[f"Std__{feature}"] = model.feature_std[feature]
        rows.append(row)
    return pd.DataFrame(rows)


def save_models(models: Dict[str, LayerModel], path: Path) -> None:
    payload = {
        layer_group: {
            "layer_group": model.layer_group,
            "intercept": model.intercept,
            "coefficients": model.coefficients,
            "feature_mean": model.feature_mean,
            "feature_std": model.feature_std,
            "score_min": model.score_min,
            "score_max": model.score_max,
            "sample_count": model.sample_count,
        }
        for layer_group, model in models.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_models(
    sample_df: pd.DataFrame,
    layer_col: str,
    target_col: str,
    confidence_col: str,
    features: Sequence[str],
    ridge_lambda: float,
    min_samples_per_layer: int,
    device: torch.device,
) -> Dict[str, LayerModel]:
    if layer_col not in sample_df.columns:
        raise KeyError(f"样本表缺少层组列: {layer_col}")
    if target_col not in sample_df.columns:
        raise KeyError(f"样本表缺少目标列: {target_col}")

    models: Dict[str, LayerModel] = {}
    for layer_group, layer_frame in sample_df.groupby(layer_col, dropna=False):
        layer_name = str(layer_group)
        if len(layer_frame) < min_samples_per_layer:
            continue
        model = fit_layer_model(
            layer_frame=layer_frame,
            layer_name=layer_name,
            features=features,
            target_col=target_col,
            confidence_col=confidence_col,
            ridge_lambda=ridge_lambda,
            device=device,
        )
        model.layer_group = layer_name
        models[layer_name] = model
    if not models:
        raise ValueError("没有任何层组满足建模条件，请检查样本量或降低 --min-samples-per-layer。")
    return models


def score_targets(
    target_df: pd.DataFrame,
    models: Dict[str, LayerModel],
    layer_col: str,
    features: Sequence[str],
    score_col: str,
    include_raw_prediction: bool,
) -> pd.DataFrame:
    if layer_col not in target_df.columns:
        raise KeyError(f"目标表缺少层组列: {layer_col}")
    parts: List[pd.DataFrame] = []
    unmatched: List[pd.DataFrame] = []
    for layer_group, frame in target_df.groupby(layer_col, dropna=False):
        layer_name = str(layer_group)
        if layer_name not in models:
            missing = frame.copy()
            missing[score_col] = np.nan
            missing["BridgeModelLayerGroup"] = np.nan
            missing["BridgeModelSampleCount"] = 0
            missing["BridgeScoreMissingFeatureFlag"] = 1
            if include_raw_prediction:
                missing["BridgeRawPrediction"] = np.nan
            unmatched.append(missing)
            continue
        parts.append(apply_layer_model(frame, models[layer_name], features, score_col, include_raw_prediction))
    scored = pd.concat(parts + unmatched, ignore_index=True) if (parts or unmatched) else target_df.copy()
    return scored


def build_run_summary(
    args: argparse.Namespace,
    device: torch.device,
    features: Sequence[str],
    sample_df: pd.DataFrame,
    target_df: pd.DataFrame,
    models: Dict[str, LayerModel],
    score_col: str,
) -> Dict[str, object]:
    return {
        "device": str(device),
        "sample_csv": args.sample_csv,
        "target_csv": args.target_csv or args.sample_csv,
        "output_dir": args.output_dir,
        "feature_cols": list(features),
        "layer_col": args.layer_col,
        "target_col": args.target_col,
        "confidence_col": args.confidence_col,
        "score_col": score_col,
        "ridge_lambda": args.ridge_lambda,
        "min_samples_per_layer": args.min_samples_per_layer,
        "sample_count": int(len(sample_df)),
        "target_count": int(len(target_df)),
        "trained_layers": {
            layer: {"sample_count": model.sample_count, "coefficients": model.coefficients}
            for layer, model in models.items()
        },
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = ensure_output_dir(args.output_dir)

    sample_df = load_csv(args.sample_csv)
    target_df = load_csv(args.target_csv) if args.target_csv else sample_df.copy()
    features = resolve_feature_columns(sample_df, args.feature_cols)

    models = train_models(
        sample_df=sample_df,
        layer_col=args.layer_col,
        target_col=args.target_col,
        confidence_col=args.confidence_col,
        features=features,
        ridge_lambda=args.ridge_lambda,
        min_samples_per_layer=args.min_samples_per_layer,
        device=device,
    )

    sample_scored = score_targets(
        target_df=sample_df,
        models=models,
        layer_col=args.layer_col,
        features=features,
        score_col=args.score_col,
        include_raw_prediction=args.include_raw_prediction,
    )
    target_scored = score_targets(
        target_df=target_df,
        models=models,
        layer_col=args.layer_col,
        features=features,
        score_col=args.score_col,
        include_raw_prediction=args.include_raw_prediction,
    )

    weight_summary = build_weight_summary(models, features)
    save_models(models, output_dir / "layer_models.json")
    weight_summary.to_csv(output_dir / "layer_weight_summary.csv", index=False, encoding="utf-8-sig")
    sample_scored.to_csv(output_dir / "sample_scored.csv", index=False, encoding="utf-8-sig")
    target_scored.to_csv(output_dir / "target_scored.csv", index=False, encoding="utf-8-sig")

    field_columns = [
        col
        for col in [
            "X",
            "Y",
            "TIME",
            args.layer_col,
            args.score_col,
            "BridgeModelLayerGroup",
            "BridgeModelSampleCount",
            "BridgeScoreMissingFeatureFlag",
        ]
        if col in target_scored.columns
    ]
    target_scored[field_columns].to_csv(
        output_dir / "local_control_field.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_summary = build_run_summary(args, device, features, sample_df, target_df, models, args.score_col)
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    schema = {
        "required_sample_fields": [args.layer_col, args.target_col] + list(features),
        "optional_sample_fields": [args.confidence_col, "SourceType", "WellName", "VirtualWellName", "X", "Y", "TIME"],
        "required_target_fields": [args.layer_col] + list(features),
        "recommended_target_fields": ["X", "Y", "TIME", "FaultDistance"],
        "score_output_field": args.score_col,
    }
    (output_dir / "field_contract.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"bridge demo completed on {device}")
    print(f"trained layers: {', '.join(models.keys())}")
    print(f"outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
