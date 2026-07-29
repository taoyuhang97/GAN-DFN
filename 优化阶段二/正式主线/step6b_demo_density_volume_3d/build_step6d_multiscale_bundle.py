from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import ndimage

from build_multiscale_density_bundle import ensure_dir, finite_stats, flat_to_grid, grid_to_flat, load_mapping, load_trace_matrix, write_sgy_like


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    apply_window_inplace,
    contract_summary,
    load_contract_for_mapping,
    validate_window_contract,
)

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
    parser.add_argument("--medium-damage-outer-decay", type=float, default=0.45)
    parser.add_argument("--large-damage-outer-decay", type=float, default=0.35)
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
    outer_decay: float,
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
    inner_xy = max(int(np.ceil(int(xy_cells) / 2.0)), 1)
    inner_time = max(int(np.ceil(int(time_samples) / 2.0)), 1)
    inner_size = (2 * inner_xy + 1, 2 * inner_xy + 1, 2 * inner_time + 1)
    inner_shell = ndimage.binary_dilation(mask_grid, structure=np.ones(inner_size, dtype=bool)) & (~mask_grid)
    taper = np.where(inner_shell, 1.0, float(np.clip(outer_decay, 0.0, 1.0))).astype(np.float32)
    local_strength = ndimage.maximum_filter(prior_grid, size=size, mode="nearest")
    damage_grid = np.where(shell, local_strength * taper, 0.0).astype(np.float32)
    return grid_to_flat(damage_grid, mapping), grid_to_flat(shell.astype(np.float32), mapping).astype(bool)


def run_compact_context(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
    root: Path,
    mapping: dict[str, np.ndarray],
) -> int:
    step6a_dir = resolve_step_dir(args.step6a_dir, root, "step6a_small")
    step6b_dir = resolve_step_dir(args.step6b_dir, root, "step6b_medium")
    step6c_dir = resolve_step_dir(args.step6c_dir, root, "step6c_large")
    small_score_sgy = step6a_dir / "small_background_score.sgy"
    small_qc = step6a_dir / "small_background_qc.json"
    if not small_score_sgy.exists() or not small_qc.exists():
        raise FileNotFoundError("compact Step6D requires the Step6A score SGY and QC")

    step6b = load_npz_dict(step6b_dir / "medium_corridor_components.npz")
    step6c = load_npz_dict(step6c_dir / "large_fault_prior_components.npz")
    medium_samples = step6b["samples"].astype(np.float32)
    large_samples = step6c["samples"].astype(np.float32)
    if medium_samples.shape != large_samples.shape or not np.allclose(medium_samples, large_samples, atol=1.0e-6):
        raise ValueError("compact Step6D requires matching medium and large 10 ms sample axes")
    horizon_contract = load_contract_for_mapping(config, mapping)
    horizon_axis_qc = validate_window_contract(config, horizon_contract, medium_samples)

    medium_prior = step6b["medium_prior"].astype(np.float32)
    medium_mask = step6b["medium_mask"].astype(bool)
    large_prior = step6c["large_prior"].astype(np.float32)
    large_mask = step6c["large_mask"].astype(bool)
    expected_shape = (len(mapping["x"]), len(medium_samples))
    for name, values in {
        "medium_prior": medium_prior,
        "medium_mask": medium_mask,
        "large_prior": large_prior,
        "large_mask": large_mask,
    }.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape {values.shape} != {expected_shape}")
    upstream_horizon_qc = {
        "medium_prior": apply_window_inplace(medium_prior, horizon_contract, medium_samples, fill_value=0.0),
        "medium_mask": apply_window_inplace(medium_mask, horizon_contract, medium_samples, fill_value=False),
        "large_prior": apply_window_inplace(large_prior, horizon_contract, medium_samples, fill_value=0.0),
        "large_mask": apply_window_inplace(large_mask, horizon_contract, medium_samples, fill_value=False),
    }

    medium_damage, medium_damage_mask = build_damage_shell(
        medium_prior,
        medium_mask,
        mapping,
        xy_cells=int(args.medium_damage_xy_cells),
        time_samples=int(args.medium_damage_time_samples),
        outer_decay=float(args.medium_damage_outer_decay),
    )
    large_damage, large_damage_mask = build_damage_shell(
        large_prior,
        large_mask,
        mapping,
        xy_cells=int(args.large_damage_xy_cells),
        time_samples=int(args.large_damage_time_samples),
        outer_decay=float(args.large_damage_outer_decay),
    )
    damage_horizon_qc = {
        "medium_damage": apply_window_inplace(medium_damage, horizon_contract, medium_samples, fill_value=0.0),
        "medium_damage_mask": apply_window_inplace(
            medium_damage_mask, horizon_contract, medium_samples, fill_value=False
        ),
        "large_damage": apply_window_inplace(large_damage, horizon_contract, medium_samples, fill_value=0.0),
        "large_damage_mask": apply_window_inplace(
            large_damage_mask, horizon_contract, medium_samples, fill_value=False
        ),
    }
    context_path = output_dir / "multiscale_damage_context_10ms.npz"
    np.savez_compressed(
        context_path,
        samples=medium_samples,
        medium_damage=(float(args.medium_damage_boost) * medium_damage).astype(np.float16),
        medium_damage_mask=medium_damage_mask.astype(np.uint8),
        medium_core_mask=medium_mask.astype(np.uint8),
        large_damage=(float(args.large_damage_boost) * large_damage).astype(np.float16),
        large_damage_mask=large_damage_mask.astype(np.uint8),
        large_core_mask=large_mask.astype(np.uint8),
        source_trace_idx=horizon_contract.trace_idx.astype(np.int32),
        t4_time=horizon_contract.t4.astype(np.float32),
        t6_time=horizon_contract.t6.astype(np.float32),
        t7_time=horizon_contract.t7.astype(np.float32),
        shasan_present=horizon_contract.shasan_present.astype(np.uint8),
        shasi_present=horizon_contract.shasi_present.astype(np.uint8),
    )
    summary = {
        "status": "pass",
        "mode": "compact_10ms_damage_context",
        "output_dir": str(output_dir),
        "small_score_sgy": str(small_score_sgy),
        "small_qc": str(small_qc),
        "context_npz": str(context_path),
        "sample_interval_ms": float(np.median(np.diff(medium_samples))) if len(medium_samples) > 1 else None,
        "sample_count": int(len(medium_samples)),
        "trace_count": int(len(mapping["x"])),
        "horizon_contract": contract_summary(horizon_contract),
        "horizon_axis_qc": horizon_axis_qc,
        "upstream_horizon_qc": upstream_horizon_qc,
        "damage_horizon_qc": damage_horizon_qc,
        "medium": {
            "core_voxel_count": int(medium_mask.sum()),
            "damage_voxel_count": int(medium_damage_mask.sum()),
            "damage_stats": finite_stats(medium_damage[medium_damage_mask]),
        },
        "large": {
            "core_voxel_count": int(large_mask.sum()),
            "damage_voxel_count": int(large_damage_mask.sum()),
            "damage_stats": finite_stats(large_damage[large_damage_mask]),
        },
        "fusion_parameters": {
            "medium_damage_xy_cells": int(args.medium_damage_xy_cells),
            "medium_damage_time_samples": int(args.medium_damage_time_samples),
            "large_damage_xy_cells": int(args.large_damage_xy_cells),
            "large_damage_time_samples": int(args.large_damage_time_samples),
            "medium_damage_boost": float(args.medium_damage_boost),
            "large_damage_boost": float(args.large_damage_boost),
            "medium_damage_outer_decay": float(args.medium_damage_outer_decay),
            "large_damage_outer_decay": float(args.large_damage_outer_decay),
            "medium_core_attenuation": float(args.medium_core_attenuation),
            "large_core_attenuation": float(args.large_core_attenuation),
            "background_floor": float(args.background_floor),
            "background_dynamic_weight": float(args.background_dynamic_weight),
        },
        "reflection": "The compact bundle keeps medium/large damage context at 10 ms and does not expand it into duplicate 2 ms SGYs.",
    }
    write_json(output_dir / "multiscale_bundle_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    root = args.rebalance_root.resolve()
    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    mapping = load_mapping(trace_mapping_npz)
    if bool(config.get("compact_multiscale_flow", False)):
        return run_compact_context(args, config, output_dir, root, mapping)
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
        outer_decay=float(args.medium_damage_outer_decay),
    )
    large_damage, large_damage_mask = build_damage_shell(
        large_prior,
        large_mask,
        mapping,
        xy_cells=int(args.large_damage_xy_cells),
        time_samples=int(args.large_damage_time_samples),
        outer_decay=float(args.large_damage_outer_decay),
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
            "medium_damage_outer_decay": float(args.medium_damage_outer_decay),
            "large_damage_outer_decay": float(args.large_damage_outer_decay),
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
