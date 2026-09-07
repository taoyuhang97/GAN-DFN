from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from build_multiscale_density_bundle import (
    ensure_dir,
    finite_stats,
    load_mapping,
    load_trace_matrix,
    write_sgy_like,
)


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    apply_validity_inplace,
    contract_summary,
    load_contract_for_mapping,
    validate_window_contract,
)

DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "output/candidate_cheye1_multiscale_rebalance_v1/step6a_small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6A small-scale background fracture density.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-low-q", type=float, default=0.02)
    parser.add_argument("--clip-high-q", type=float, default=0.995)
    parser.add_argument("--candidate-q", type=float, default=0.88)
    parser.add_argument("--core-q", type=float, default=0.95)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def quantiles(values: np.ndarray, qs: list[float]) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"q{q:g}": None for q in qs}
    return {f"q{q:g}": float(np.quantile(finite, q)) for q in qs}


def robust_normalize(values: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {"valid_count": 0}
    low = float(np.quantile(finite, low_q))
    high = float(np.quantile(finite, high_q))
    if high <= low:
        high = low + 1.0e-6
    score = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
    score[~np.isfinite(score)] = 0.0
    return score, {
        "valid_count": int(finite.size),
        "clip_low_quantile": float(low_q),
        "clip_high_quantile": float(high_q),
        "clip_low_value": low,
        "clip_high_value": high,
    }


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    mapping = load_mapping(trace_mapping_npz)
    density, samples, density_load = load_trace_matrix(input_density_sgy, None, None, "Step6A density")
    horizon_contract = load_contract_for_mapping(config, mapping)
    horizon_axis_qc = validate_window_contract(config, horizon_contract, samples)
    density_valid = np.isfinite(density)
    horizon_mask_qc = apply_validity_inplace(density_valid, horizon_contract, samples)
    density = np.clip(np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None).astype(np.float32)
    density[~density_valid] = 0.0
    if density.shape[0] != len(mapping["x"]):
        raise ValueError(f"density tracecount {density.shape[0]} != mapping rows {len(mapping['x'])}")

    evidence_cfg = dict(config.get("small_evidence", {}))
    density_weight = float(evidence_cfg.get("density_weight", 1.0))
    curvature_weight = float(evidence_cfg.get("curvature_weight", 0.0))
    weight_total = density_weight + curvature_weight
    if density_weight < 0.0 or curvature_weight < 0.0 or weight_total <= 0.0:
        raise ValueError("small_evidence weights must be non-negative and have a positive sum")
    density_weight /= weight_total
    curvature_weight /= weight_total

    density_for_score = density.copy()
    density_for_score[~density_valid] = np.nan
    density_score, density_norm_summary = robust_normalize(density_for_score, args.clip_low_q, args.clip_high_q)
    curvature_path = Path(config["volume_paths"]["CurvatureMax"]).resolve()
    curvature, _, curvature_load = load_trace_matrix(
        curvature_path,
        mapping["source_trace_idx"].astype(np.int64),
        samples,
        "Step6A CurvatureMax",
    )
    curvature_valid = np.isfinite(curvature) & (np.abs(curvature) < 1.0e6)
    apply_validity_inplace(curvature_valid, horizon_contract, samples)
    curvature_transform = str(evidence_cfg.get("curvature_transform", "absolute"))
    if curvature_transform == "absolute":
        curvature_for_score = np.abs(curvature)
    elif curvature_transform == "positive":
        curvature_for_score = np.maximum(curvature, 0.0)
    elif curvature_transform == "negative_absolute":
        curvature_for_score = np.maximum(-curvature, 0.0)
    else:
        raise ValueError(f"unsupported curvature_transform: {curvature_transform}")
    curvature_for_score[~curvature_valid] = np.nan
    curvature_score, curvature_norm_summary = robust_normalize(
        curvature_for_score,
        float(evidence_cfg.get("curvature_clip_low_q", 0.50)),
        float(evidence_cfg.get("curvature_clip_high_q", 0.995)),
    )
    evidence_valid = density_valid & curvature_valid
    small_score = (density_weight * density_score + curvature_weight * curvature_score).astype(np.float32)
    small_score[~evidence_valid] = 0.0
    positive = small_score[small_score > 0]
    candidate_threshold = float(np.quantile(positive, args.candidate_q)) if positive.size else 1.0
    core_threshold = float(np.quantile(positive, args.core_q)) if positive.size else 1.0
    candidate_mask = small_score >= candidate_threshold
    core_mask = small_score >= core_threshold

    write_sgy_like(output_dir / "small_background_score.sgy", input_density_sgy, small_score, samples)

    summary = {
        "status": "pass",
        "input_density_sgy": str(input_density_sgy),
        "trace_mapping_npz": str(trace_mapping_npz),
        "output_dir": str(output_dir),
        "output_contract": {
            "small_score_sgy": str(output_dir / "small_background_score.sgy"),
            "candidate_mask_storage": "reconstruct_from_candidate_threshold",
            "core_mask_storage": "reconstruct_from_core_threshold",
            "density_source": str(input_density_sgy),
            "trace_mapping_source": str(trace_mapping_npz),
        },
        "density_load": density_load,
        "trace_count": int(density.shape[0]),
        "sample_count": int(density.shape[1]),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "horizon_contract": contract_summary(horizon_contract),
        "horizon_axis_qc": horizon_axis_qc,
        "horizon_mask_qc": horizon_mask_qc,
        "density_stats": finite_stats(density[density_valid]),
        "density_quantiles": quantiles(density[density_valid], [0.02, 0.50, 0.88, 0.95, 0.995]),
        "small_score_stats": finite_stats(small_score),
        "small_evidence": {
            "density_weight": density_weight,
            "curvature_weight": curvature_weight,
            "curvature_transform": curvature_transform,
            "curvature_path": str(curvature_path),
            "valid_voxel_count": int(evidence_valid.sum()),
        },
        "density_normalization": density_norm_summary,
        "curvature_normalization": curvature_norm_summary,
        "curvature_load": curvature_load,
        "candidate_quantile": float(args.candidate_q),
        "candidate_threshold": candidate_threshold,
        "candidate_voxel_count": int(candidate_mask.sum()),
        "candidate_voxel_fraction": float(candidate_mask.mean()),
        "core_quantile": float(args.core_q),
        "core_threshold": core_threshold,
        "core_voxel_count": int(core_mask.sum()),
        "core_voxel_fraction": float(core_mask.mean()),
        "reflection": (
            "Step6A combines the current Step6 density and ordinary CurvatureMax using configuration weights. "
            "The fused score is the small-scale evidence consumed by Step6D and Step7A."
        ),
    }
    write_json(output_dir / "small_background_qc.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
