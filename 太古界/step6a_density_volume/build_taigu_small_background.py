#!/usr/bin/env python3
"""Step6A 小尺度背景分数：把模型密度体与曲率体加权融合（太古界 v4）。

对齐砂砾岩正式主线 `step6b_demo_density_volume_3d/build_step6a_small_background.py`：
砂砾岩用 `small_score = 0.5 × norm(密度体) + 0.5 × norm(|曲率体|)` 作为"小尺度背景分数"，
由 Step6D/Step7A 消费；太古界此前直接把模型密度体交给 Step7A，缺少这一步，
导致模型没学到的属性关系无法由设计口径兜住（详见问题记录 0.21–0.23）。

本脚本按需求方口径使用 **密度 0.4 / 曲率 0.6** 的权重。

输出：

* `small_background_score.sgy`：融合后的小尺度背景分数（0~1），供 Step6D/Step7A 消费；
* `small_background_prior.npz`：Step6D 完整模式所需的 `density` / `small_score` / `candidate_mask`；
* `small_background_qc.json`：状态、权重、阈值、属性相关性与层内分布 QC
  （其中 `output_paths.density_sgy` 与 `model_contract_version` 供 Step7A 的输入契约校验）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TAIGU_ROOT = HERE.parent
for extra in (
    TAIGU_ROOT / "common" / "multiscale_density",
    TAIGU_ROOT / "step6d_multiscale_bundle" / "code",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from build_multiscale_density_bundle import (  # noqa: E402
    ensure_dir,
    finite_stats,
    load_mapping,
    load_trace_matrix,
    read_sgy_sample_axis,
    write_sgy_like,
)
from build_step6d_multiscale_bundle import (  # noqa: E402
    apply_taigu_window,
    contract_summary,
    load_taigu_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taigu Step6A small-scale background score (density + curvature).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def quantiles(values: np.ndarray, qs: list[float]) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.quantile(finite, q)) for q in qs}


def robust_normalize(values: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, dict[str, Any]]:
    """按分位数裁剪后线性归一化到 0~1（与砂砾岩同名函数一致）。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {"low": 0.0, "high": 0.0, "finite_count": 0}
    low = float(np.quantile(finite, low_q))
    high = float(np.quantile(finite, high_q))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1.0e-12:
        high = low + 1.0e-12
    score = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
    score[~np.isfinite(score)] = 0.0
    return score, {
        "low_quantile": low_q, "high_quantile": high_q, "low": low, "high": high,
        "finite_count": int(finite.size), "min": float(finite.min()), "max": float(finite.max()),
    }


def layer_relative_profile(small_score: np.ndarray, contract: dict[str, np.ndarray],
                           samples: np.ndarray) -> dict[str, Any]:
    """按层内相对位置四等分统计融合分数（与 Step6A 预测的同一验收口径）。"""
    top, mid, base = contract["t4"][:, None], contract["t6"][:, None], contract["t7"][:, None]
    axis = np.asarray(samples, dtype=np.float32)[None, :]
    out: dict[str, Any] = {}
    for name, lo, hi in (("上部复合层", top, mid), ("太古界风化壳", mid, base)):
        valid = np.isfinite(lo) & np.isfinite(hi) & (hi > lo) & contract["valid"][:, None]
        rel = np.where(valid, (axis - lo) / np.maximum(hi - lo, 1e-6), np.nan)
        profile = {}
        for index in range(4):
            mask = valid & (rel >= index * 0.25) & (rel < (index + 1) * 0.25)
            values = small_score[mask]
            profile[f"{index * 25}-{index * 25 + 25}%"] = {
                "count": int(mask.sum()),
                "mean": float(values.mean()) if values.size else None,
            }
        out[name] = profile
    return out


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = Path(config["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace_output:
        raise FileExistsError(f"output exists; pass --replace-output: {output_dir}")
    ensure_dir(output_dir)

    density_sgy = Path(config["density_sgy"]).resolve()
    mapping_path = Path(config["trace_mapping_npz"]).resolve()
    curvature_path = Path(config["curvature_volume_path"]).resolve()
    horizon_table = Path(config["horizon_contract_table"]).resolve()
    evidence = dict(config.get("small_evidence", {}))

    density_weight = float(evidence.get("density_weight", 0.4))
    curvature_weight = float(evidence.get("curvature_weight", 0.6))
    total = density_weight + curvature_weight
    if density_weight < 0.0 or curvature_weight < 0.0 or total <= 0.0:
        raise ValueError("small_evidence weights must be non-negative with a positive sum")
    density_weight /= total
    curvature_weight /= total
    curvature_transform = str(evidence.get("curvature_transform", "absolute"))
    density_clip = (float(evidence.get("density_clip_low_q", 0.02)), float(evidence.get("density_clip_high_q", 0.98)))
    curvature_clip = (float(evidence.get("curvature_clip_low_q", 0.50)), float(evidence.get("curvature_clip_high_q", 0.995)))
    candidate_q = float(evidence.get("candidate_quantile", 0.88))
    core_q = float(evidence.get("core_quantile", 0.95))

    # Step6A 预测写出的 trace_mapping.npz 用的是 `trace_idx` 键（同一网格顺序），
    # 这里做一次键名归一化，兼容 demo-grid 映射（source_trace_idx）。
    with np.load(mapping_path) as handle:
        mapping = {name: handle[name] for name in handle.files}
    if "source_trace_idx" not in mapping and "trace_idx" in mapping:
        mapping["source_trace_idx"] = np.asarray(mapping["trace_idx"], dtype=np.int64)
    if "source_trace_idx" not in mapping:
        raise ValueError(f"trace mapping missing source_trace_idx/trace_idx: {mapping_path}")
    density, samples, density_load = load_trace_matrix(density_sgy, None, None, "Step6A density")
    if density.shape[0] != len(mapping["x"]):
        raise ValueError("density trace count does not match the demo-grid mapping")
    contract = load_taigu_contract(horizon_table, mapping["source_trace_idx"])

    density_valid = np.isfinite(density)
    window_qc = apply_taigu_window(density_valid, contract, samples, fill_value=False)
    density = np.clip(np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None).astype(np.float32)
    density[~density_valid] = 0.0

    # 曲率体是原始属性体：它的采样轴是"原始时间"，需要按时间原点换算到绝对时间轴。
    raw_axis, _ = read_sgy_sample_axis(curvature_path)
    origin_ms = float(config.get("curvature_time_origin_ms", 1800.0))
    curvature_target = np.asarray(samples, dtype=np.float64) - origin_ms + float(raw_axis[0])
    curvature, _, curvature_load = load_trace_matrix(
        curvature_path, mapping["source_trace_idx"].astype(np.int64), curvature_target, "CurvatureMax"
    )
    curvature_valid = np.isfinite(curvature) & (np.abs(curvature) < 1.0e6)
    apply_taigu_window(curvature_valid, contract, samples, fill_value=False)
    if curvature_transform == "absolute":
        curvature_for_score = np.abs(curvature)
    elif curvature_transform == "positive":
        curvature_for_score = np.maximum(curvature, 0.0)
    elif curvature_transform == "negative_absolute":
        curvature_for_score = np.maximum(-curvature, 0.0)
    else:
        raise ValueError(f"unsupported curvature_transform: {curvature_transform}")
    curvature_for_score = np.where(curvature_valid, curvature_for_score, np.nan)

    density_score, density_norm = robust_normalize(
        np.where(density_valid, density, np.nan), density_clip[0], density_clip[1]
    )
    curvature_score, curvature_norm = robust_normalize(curvature_for_score, curvature_clip[0], curvature_clip[1])
    evidence_valid = density_valid & curvature_valid
    small_score = (density_weight * density_score + curvature_weight * curvature_score).astype(np.float32)
    small_score[~evidence_valid] = 0.0

    positive = small_score[small_score > 0]
    candidate_threshold = float(np.quantile(positive, candidate_q)) if positive.size else 1.0
    core_threshold = float(np.quantile(positive, core_q)) if positive.size else 1.0
    candidate_mask = small_score >= candidate_threshold
    core_mask = small_score >= core_threshold

    score_path = output_dir / "small_background_score.sgy"
    write_sgy_like(score_path, density_sgy, small_score, samples)
    np.savez_compressed(
        output_dir / "small_background_prior.npz",
        density=density.astype(np.float32),
        small_score=small_score,
        candidate_mask=candidate_mask.astype(np.uint8),
        core_mask=core_mask.astype(np.uint8),
        density_score=density_score,
        curvature_score=curvature_score,
        x=mapping["x"].astype(np.float64),
        y=mapping["y"].astype(np.float64),
        source_trace_idx=mapping["source_trace_idx"].astype(np.int64),
        sample_times_ms=np.asarray(samples, dtype=np.float32),
    )

    valid_pairs = evidence_valid & np.isfinite(curvature_for_score)
    correlation = None
    if valid_pairs.sum() > 100:
        a = small_score[valid_pairs].astype(np.float64)
        b = np.nan_to_num(curvature_for_score[valid_pairs]).astype(np.float64)
        if np.std(a) > 0 and np.std(b) > 0:
            correlation = float(np.corrcoef(a, b)[0, 1])

    summary: dict[str, Any] = {
        "status": "pass",
        "version": "taigu_step6a_small_background_v4",
        "model_contract_version": str(config.get("model_contract_version", "taigu_step6a_attribute_v3_v4")),
        "output_paths": {"density_sgy": str(score_path)},
        "input_density_sgy": str(density_sgy),
        "curvature_volume_path": str(curvature_path),
        "trace_mapping_npz": str(mapping_path),
        "horizon_contract_table": str(horizon_table),
        "output_dir": str(output_dir),
        "small_evidence": {
            "density_weight": density_weight,
            "curvature_weight": curvature_weight,
            "curvature_transform": curvature_transform,
            "density_clip": list(density_clip),
            "curvature_clip": list(curvature_clip),
            "curvature_time_origin_ms": origin_ms,
        },
        "trace_count": int(small_score.shape[0]),
        "sample_count": int(small_score.shape[1]),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "density_load": density_load,
        "curvature_load": curvature_load,
        "horizon_contract": contract_summary(contract),
        "horizon_mask_qc": window_qc,
        "density_stats": finite_stats(density[density_valid]),
        "curvature_stats": finite_stats(curvature_for_score[valid_pairs]) if valid_pairs.any() else {},
        "density_normalization": density_norm,
        "curvature_normalization": curvature_norm,
        "small_score_stats": finite_stats(small_score),
        "small_score_vs_curvature_correlation": correlation,
        "candidate_quantile": candidate_q,
        "candidate_threshold": candidate_threshold,
        "candidate_voxel_count": int(candidate_mask.sum()),
        "candidate_voxel_fraction": float(candidate_mask.mean()),
        "core_quantile": core_q,
        "core_threshold": core_threshold,
        "core_voxel_count": int(core_mask.sum()),
        "layer_relative_profile": layer_relative_profile(small_score, contract, samples),
        "reflection": (
            "Step6A 将模型密度体与曲率体按配置权重融合为小尺度背景分数（太古界 0.4/0.6），"
            "供 Step6D 与 Step7A 消费；这样'曲率主导小尺度'由设计口径保证，而不是依赖模型自学。"
        ),
    }
    write_json(output_dir / "small_background_qc.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "layer_relative_profile"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
