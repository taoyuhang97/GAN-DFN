from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import ndimage

from build_multiscale_density_bundle import ensure_dir, finite_stats, flat_to_grid, grid_to_flat, load_mapping, load_trace_matrix, write_sgy_like


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_ROOT = CURRENT_DIR / "output/candidate_cheye1_multiscale_rebalance_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "step6d_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6D integrate multiscale evidence bundle.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rebalance-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--step6a-dir", type=Path, default=None)
    parser.add_argument("--step6b-dir", type=Path, default=None)
    parser.add_argument("--step6c-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--small-weight", type=float, default=0.35)
    parser.add_argument("--medium-weight", type=float, default=0.55)
    parser.add_argument("--large-weight", type=float, default=1.0)
    parser.add_argument("--medium-damage-xy-cells", type=int, default=3)
    parser.add_argument("--medium-damage-time-samples", type=int, default=2)
    parser.add_argument("--large-damage-xy-cells", type=int, default=5)
    parser.add_argument("--large-damage-time-samples", type=int, default=3)
    parser.add_argument("--medium-damage-boost", type=float, default=0.35)
    parser.add_argument("--large-damage-boost", type=float, default=0.50)
    parser.add_argument("--medium-core-attenuation", type=float, default=0.35)
    parser.add_argument("--large-core-attenuation", type=float, default=0.65)
    parser.add_argument("--final-small-candidate-quantile", type=float, default=0.82)
    parser.add_argument("--background-floor", type=float, default=0.06)
    parser.add_argument("--background-dynamic-weight", type=float, default=0.85)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return {key: data[key] for key in data.files}


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def overlap_count(a: np.ndarray, b: np.ndarray) -> int:
    return int((a.astype(bool) & b.astype(bool)).sum())


def resolve_step_dir(explicit: Path | None, root: Path, name: str) -> Path:
    path = explicit.resolve() if explicit is not None else (root / name).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_damage_shell(
    prior_flat: np.ndarray,
    mask_flat: np.ndarray,
    mapping: dict[str, np.ndarray],
    xy_cells: int,
    time_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    prior_grid, _, _ = flat_to_grid(prior_flat.astype(np.float32), mapping)
    mask_grid, _, _ = flat_to_grid(mask_flat.astype(np.float32), mapping)
    mask_grid = mask_grid > 0.5
    if not mask_grid.any():
        empty = np.zeros_like(prior_flat, dtype=np.float32)
        return empty, empty.astype(bool)
    size = (2 * int(xy_cells) + 1, 2 * int(xy_cells) + 1, 2 * int(time_samples) + 1)
    dilated = ndimage.binary_dilation(mask_grid, structure=np.ones(size, dtype=bool))
    shell = dilated & (~mask_grid)
    local_strength = ndimage.maximum_filter(prior_grid, size=size, mode="nearest")
    damage_grid = np.where(shell, local_strength, 0.0).astype(np.float32)
    return grid_to_flat(damage_grid, mapping), grid_to_flat(shell.astype(np.float32), mapping).astype(bool)


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    root = args.rebalance_root.resolve()
    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    mapping = load_mapping(trace_mapping_npz)
    _, samples, density_load = load_trace_matrix(input_density_sgy, None, None, "Density")

    step6a_dir = resolve_step_dir(args.step6a_dir, root, "step6a_small")
    step6b_dir = resolve_step_dir(args.step6b_dir, root, "step6b_medium")
    step6c_dir = resolve_step_dir(args.step6c_dir, root, "step6c_large")
    step6a = load_npz_dict(step6a_dir / "small_background_prior.npz")
    step6b = load_npz_dict(step6b_dir / "medium_corridor_components.npz")
    step6c = load_npz_dict(step6c_dir / "large_fault_prior_components.npz")

    small_density = step6a["density"].astype(np.float32)
    small_score = step6a["small_score"].astype(np.float32)
    small_mask = step6a["candidate_mask"].astype(bool)
    medium_prior = step6b["medium_prior"].astype(np.float32)
    medium_mask = step6b["medium_mask"].astype(bool)
    medium_component_id = step6b["medium_component_id"].astype(np.int32)
    large_prior = step6c["large_prior"].astype(np.float32)
    large_mask = step6c["large_mask"].astype(bool)
    large_component_id = step6c["inferred_component_id"].astype(np.int32)

    shapes = {
        "small_density": small_density.shape,
        "small_score": small_score.shape,
        "medium_prior": medium_prior.shape,
        "large_prior": large_prior.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Step6 A/B/C shapes differ: {shapes}")
    if small_density.shape[0] != len(mapping["x"]):
        raise ValueError("bundle trace count differs from mapping")

    scale_label = np.zeros(small_score.shape, dtype=np.uint8)
    scale_label[small_mask] = 1
    scale_label[medium_mask] = 2
    scale_label[large_mask] = 3
    scale_confidence = np.zeros(small_score.shape, dtype=np.float32)
    scale_confidence[small_mask] = small_score[small_mask]
    scale_confidence[medium_mask] = medium_prior[medium_mask]
    scale_confidence[large_mask] = large_prior[large_mask]
    integrated = np.clip(
        float(args.small_weight) * small_score
        + float(args.medium_weight) * medium_prior
        + float(args.large_weight) * large_prior,
        0.0,
        1.0,
    ).astype(np.float32)
    medium_damage, medium_damage_mask = build_damage_shell(
        medium_prior,
        medium_mask,
        mapping,
        xy_cells=int(args.medium_damage_xy_cells),
        time_samples=int(args.medium_damage_time_samples),
    )
    large_damage, large_damage_mask = build_damage_shell(
        large_prior,
        large_mask,
        mapping,
        xy_cells=int(args.large_damage_xy_cells),
        time_samples=int(args.large_damage_time_samples),
    )
    background_valid = small_score > 0
    background_small_density = np.where(
        background_valid,
        float(args.background_floor) + float(args.background_dynamic_weight) * small_score,
        0.0,
    ).astype(np.float32)
    background_small_density = np.clip(background_small_density, 0.0, 1.0).astype(np.float32)
    medium_damage_small_density = np.clip(float(args.medium_damage_boost) * medium_damage, 0.0, 1.0).astype(np.float32)
    large_damage_small_density = np.clip(float(args.large_damage_boost) * large_damage, 0.0, 1.0).astype(np.float32)

    attenuated_background = background_small_density.copy()
    attenuated_background = attenuated_background * (1.0 - float(args.medium_core_attenuation) * medium_mask.astype(np.float32))
    attenuated_background = attenuated_background * (1.0 - float(args.large_core_attenuation) * large_mask.astype(np.float32))
    final_small_density = np.maximum.reduce(
        [
            attenuated_background,
            medium_damage_small_density,
            large_damage_small_density,
        ]
    )
    final_small_density = np.clip(final_small_density, 0.0, 1.0).astype(np.float32)
    positive_final = final_small_density[final_small_density > 0]
    if positive_final.size:
        final_cut = float(np.quantile(positive_final, float(args.final_small_candidate_quantile)))
    else:
        final_cut = 1.0
    final_small_candidate_mask = final_small_density >= final_cut
    small_domain_label = np.zeros(small_score.shape, dtype=np.uint8)
    small_domain_label[background_small_density > 0] = 1
    small_domain_label[medium_damage_small_density > background_small_density] = 2
    small_domain_label[large_damage_small_density > np.maximum(background_small_density, medium_damage_small_density)] = 3
    small_domain_confidence = np.maximum.reduce(
        [
            background_small_density,
            medium_damage_small_density,
            large_damage_small_density,
        ]
    ).astype(np.float32)

    write_sgy_like(output_dir / "background_small_density.sgy", input_density_sgy, background_small_density, samples)
    write_sgy_like(output_dir / "medium_damage_small_density.sgy", input_density_sgy, medium_damage_small_density, samples)
    write_sgy_like(output_dir / "large_damage_small_density.sgy", input_density_sgy, large_damage_small_density, samples)
    write_sgy_like(output_dir / "small_domain_label.sgy", input_density_sgy, small_domain_label.astype(np.float32), samples)
    write_sgy_like(output_dir / "small_domain_confidence.sgy", input_density_sgy, small_domain_confidence, samples)
    write_sgy_like(output_dir / "final_small_density.sgy", input_density_sgy, final_small_density, samples)
    write_sgy_like(output_dir / "final_small_candidate_mask.sgy", input_density_sgy, final_small_candidate_mask.astype(np.float32), samples)
    write_sgy_like(output_dir / "medium_damage_density.sgy", input_density_sgy, medium_damage.astype(np.float32), samples)
    write_sgy_like(output_dir / "large_damage_density.sgy", input_density_sgy, large_damage.astype(np.float32), samples)
    write_sgy_like(output_dir / "integrated_sampling_density.sgy", input_density_sgy, integrated, samples)
    write_sgy_like(output_dir / "scale_label.sgy", input_density_sgy, scale_label.astype(np.float32), samples)
    write_sgy_like(output_dir / "scale_confidence.sgy", input_density_sgy, scale_confidence, samples)

    np.savez_compressed(
        output_dir / "multiscale_prior_bundle.npz",
        small_density=small_density,
        small_score=small_score,
        small_mask=small_mask.astype(np.uint8),
        base_small_density=small_score.astype(np.float32),
        background_small_density=background_small_density.astype(np.float32),
        medium_damage_small_density=medium_damage_small_density.astype(np.float32),
        large_damage_small_density=large_damage_small_density.astype(np.float32),
        final_small_density=final_small_density.astype(np.float32),
        final_small_candidate_mask=final_small_candidate_mask.astype(np.uint8),
        small_domain_label=small_domain_label.astype(np.uint8),
        small_domain_confidence=small_domain_confidence.astype(np.float32),
        medium_damage=medium_damage.astype(np.float32),
        medium_damage_mask=medium_damage_mask.astype(np.uint8),
        medium_prior=medium_prior,
        medium_mask=medium_mask.astype(np.uint8),
        medium_component_id=medium_component_id,
        large_damage=large_damage.astype(np.float32),
        large_damage_mask=large_damage_mask.astype(np.uint8),
        large_prior=large_prior,
        large_mask=large_mask.astype(np.uint8),
        large_component_id=large_component_id,
        integrated_sampling_density=integrated,
        scale_label=scale_label,
        scale_confidence=scale_confidence,
        x=mapping["x"].astype(np.float64),
        y=mapping["y"].astype(np.float64),
        ix=mapping["ix"].astype(np.int32),
        iy=mapping["iy"].astype(np.int32),
        source_trace_idx=mapping["source_trace_idx"].astype(np.int64),
        sample_times_ms=samples.astype(np.float32),
    )

    medium_summary_csv = root / "step6b_medium/medium_corridor_component_summary.csv"
    large_summary_csv = step6c_dir / "large_fault_component_summary.csv"
    medium_summary_csv = step6b_dir / "medium_corridor_component_summary.csv"
    medium_summary = read_optional_csv(medium_summary_csv)
    large_summary = read_optional_csv(large_summary_csv)
    total_voxels = int(scale_label.size)
    label_counts = {str(label): int((scale_label == label).sum()) for label in [0, 1, 2, 3]}
    label_fractions = {key: float(value / total_voxels) for key, value in label_counts.items()}
    summary: dict[str, Any] = {
        "status": "pass",
        "input_density_sgy": str(input_density_sgy),
        "trace_mapping_npz": str(trace_mapping_npz),
        "step6a_dir": str(step6a_dir),
        "step6b_dir": str(step6b_dir),
        "step6c_dir": str(step6c_dir),
        "output_dir": str(output_dir),
        "density_load": density_load,
        "shape": list(small_score.shape),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "weights": {
            "small": float(args.small_weight),
            "medium": float(args.medium_weight),
            "large": float(args.large_weight),
        },
        "final_small_density_rule": {
            "meaning": "small-scale/background plus derivative small fractures around medium/large structures; medium/large cores are attenuated to avoid duplicating structure bodies",
            "medium_damage_xy_cells": int(args.medium_damage_xy_cells),
            "medium_damage_time_samples": int(args.medium_damage_time_samples),
            "large_damage_xy_cells": int(args.large_damage_xy_cells),
            "large_damage_time_samples": int(args.large_damage_time_samples),
            "medium_damage_boost": float(args.medium_damage_boost),
            "large_damage_boost": float(args.large_damage_boost),
            "medium_core_attenuation": float(args.medium_core_attenuation),
            "large_core_attenuation": float(args.large_core_attenuation),
            "background_floor": float(args.background_floor),
            "background_dynamic_weight": float(args.background_dynamic_weight),
            "final_small_candidate_quantile": float(args.final_small_candidate_quantile),
            "final_small_candidate_threshold": float(final_cut),
        },
        "scale_label_meaning": {
            "0": "background/no selected evidence",
            "1": "small background fracture candidate",
            "2": "medium fracture corridor candidate",
            "3": "large original/inferred fault candidate",
        },
        "scale_label_counts": label_counts,
        "scale_label_fractions": label_fractions,
        "overlap_counts": {
            "small_medium": overlap_count(small_mask, medium_mask),
            "small_large": overlap_count(small_mask, large_mask),
            "medium_large": overlap_count(medium_mask, large_mask),
            "small_medium_large": int((small_mask & medium_mask & large_mask).sum()),
        },
        "small": {
            "candidate_voxel_count": int(small_mask.sum()),
            "candidate_voxel_fraction": float(small_mask.mean()),
            "score_stats": finite_stats(small_score[small_score > 0]),
        },
        "final_small": {
            "candidate_voxel_count": int(final_small_candidate_mask.sum()),
            "candidate_voxel_fraction": float(final_small_candidate_mask.mean()),
            "density_stats": finite_stats(final_small_density[final_small_density > 0]),
            "background_small_density_stats": finite_stats(background_small_density[background_small_density > 0]),
            "medium_damage_small_density_stats": finite_stats(medium_damage_small_density[medium_damage_small_density > 0]),
            "large_damage_small_density_stats": finite_stats(large_damage_small_density[large_damage_small_density > 0]),
            "domain_label_counts": {str(label): int((small_domain_label == label).sum()) for label in [0, 1, 2, 3]},
            "domain_label_meaning": {
                "0": "no small-domain evidence",
                "1": "background_small_fracture",
                "2": "medium_damage_small_fracture",
                "3": "large_damage_small_fracture",
            },
            "density_in_medium_core_stats": finite_stats(final_small_density[medium_mask]),
            "density_in_large_core_stats": finite_stats(final_small_density[large_mask]),
            "medium_damage_voxel_count": int(medium_damage_mask.sum()),
            "large_damage_voxel_count": int(large_damage_mask.sum()),
            "medium_damage_stats": finite_stats(medium_damage[medium_damage > 0]),
            "large_damage_stats": finite_stats(large_damage[large_damage > 0]),
        },
        "medium": {
            "candidate_voxel_count": int(medium_mask.sum()),
            "candidate_voxel_fraction": float(medium_mask.mean()),
            "component_count": int(len(medium_summary)),
            "component_voxel_stats": finite_stats(medium_summary["voxel_count"]) if len(medium_summary) else finite_stats([]),
            "prior_stats": finite_stats(medium_prior[medium_prior > 0]),
        },
        "large": {
            "candidate_voxel_count": int(large_mask.sum()),
            "candidate_voxel_fraction": float(large_mask.mean()),
            "inferred_component_count": int(len(large_summary)),
            "prior_stats": finite_stats(large_prior[large_prior > 0]),
        },
        "integrated_sampling_density_stats": finite_stats(integrated[integrated > 0]),
        "scale_confidence_stats": finite_stats(scale_confidence[scale_confidence > 0]),
        "reflection": (
            "Step6D now builds the final small-scale density for Step7A. Medium and large bodies remain separate Step7B/Step7C inputs; "
            "only their surrounding damage zones enhance small-scale derivative fractures."
        ),
    }
    write_json(output_dir / "multiscale_bundle_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
