from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_multiscale_density_bundle import (
    copy_mapping,
    ensure_dir,
    finite_stats,
    load_mapping,
    load_trace_matrix,
    write_sgy_like,
)


CURRENT_DIR = Path(__file__).resolve().parent
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
    density = np.clip(np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None).astype(np.float32)
    if density.shape[0] != len(mapping["x"]):
        raise ValueError(f"density tracecount {density.shape[0]} != mapping rows {len(mapping['x'])}")

    small_score, norm_summary = robust_normalize(density, args.clip_low_q, args.clip_high_q)
    positive = small_score[small_score > 0]
    candidate_threshold = float(np.quantile(positive, args.candidate_q)) if positive.size else 1.0
    core_threshold = float(np.quantile(positive, args.core_q)) if positive.size else 1.0
    candidate_mask = (small_score >= candidate_threshold).astype(np.float32)
    core_mask = (small_score >= core_threshold).astype(np.float32)

    write_sgy_like(output_dir / "small_background_density.sgy", input_density_sgy, density, samples)
    write_sgy_like(output_dir / "small_background_score.sgy", input_density_sgy, small_score, samples)
    write_sgy_like(output_dir / "small_background_candidate_mask.sgy", input_density_sgy, candidate_mask, samples)
    copy_mapping(trace_mapping_npz, output_dir / "candidate_cheye1_3d_trace_mapping.npz")
    np.savez_compressed(
        output_dir / "small_background_prior.npz",
        density=density.astype(np.float32),
        small_score=small_score.astype(np.float32),
        candidate_mask=candidate_mask.astype(np.uint8),
        core_mask=core_mask.astype(np.uint8),
        samples=samples.astype(np.float32),
        x=mapping["x"].astype(np.float64),
        y=mapping["y"].astype(np.float64),
        ix=mapping["ix"].astype(np.int32),
        iy=mapping["iy"].astype(np.int32),
        source_trace_idx=mapping["source_trace_idx"].astype(np.int64),
    )

    summary = {
        "status": "pass",
        "input_density_sgy": str(input_density_sgy),
        "trace_mapping_npz": str(trace_mapping_npz),
        "output_dir": str(output_dir),
        "density_load": density_load,
        "trace_count": int(density.shape[0]),
        "sample_count": int(density.shape[1]),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "density_stats": finite_stats(density),
        "density_quantiles": quantiles(density, [0.02, 0.50, 0.88, 0.95, 0.995]),
        "small_score_stats": finite_stats(small_score),
        "normalization": norm_summary,
        "candidate_quantile": float(args.candidate_q),
        "candidate_threshold": candidate_threshold,
        "candidate_voxel_count": int(candidate_mask.sum()),
        "candidate_voxel_fraction": float(candidate_mask.mean()),
        "core_quantile": float(args.core_q),
        "core_threshold": core_threshold,
        "core_voxel_count": int(core_mask.sum()),
        "core_voxel_fraction": float(core_mask.mean()),
        "reflection": (
            "Step6A keeps the current 3D density as small-scale background only. "
            "The 0.88 candidate quantile is deliberately looser than the previous sparse Step7A path, "
            "so Step7A can recover background fractures without controlling medium/large structures."
        ),
    }
    write_json(output_dir / "small_background_qc.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
