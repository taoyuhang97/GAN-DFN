from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage

from build_multiscale_density_bundle import (
    ensure_dir,
    finite_stats,
    flat_to_grid,
    grid_to_flat,
    high_score,
    load_mapping,
    load_trace_matrix,
    low_score,
    valid_values,
    write_sgy_like,
)


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "output/candidate_cheye1_multiscale_rebalance_v1/step6b_medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6B medium-scale fracture corridor prior.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-quantile", type=float, default=0.95)
    parser.add_argument("--anttrack-high-quantile", type=float, default=0.90)
    parser.add_argument("--ant-weight-base", type=float, default=0.62)
    parser.add_argument("--lowcoh-support-weight", type=float, default=0.20)
    parser.add_argument("--curvature-support-weight", type=float, default=0.18)
    parser.add_argument("--min-direct-support-score", type=float, default=0.35)
    parser.add_argument("--support-score-threshold", type=float, default=0.35)
    parser.add_argument("--min-local-support-score", type=float, default=0.35)
    parser.add_argument("--min-local-support-fraction", type=float, default=0.20)
    parser.add_argument("--support-neighborhood-cells", type=int, default=1)
    parser.add_argument("--min-component-voxels", type=int, default=120)
    parser.add_argument("--max-component-voxels-before-split", type=int, default=12000)
    parser.add_argument("--split-tile-cells", type=int, default=16)
    parser.add_argument("--split-time-samples", type=int, default=8)
    parser.add_argument("--orientation-time-scale-m-per-ms", type=float, default=2.0)
    parser.add_argument("--min-vertical-extent-ms", type=float, default=30.0)
    parser.add_argument("--min-component-dip-deg", type=float, default=25.0)
    parser.add_argument("--min-component-linearity", type=float, default=1.20)
    parser.add_argument("--min-component-score-mean", type=float, default=0.70)
    parser.add_argument("--max-horizontal-layer-thickness-ms", type=float, default=8.0)
    parser.add_argument("--horizontal-layer-min-extent-m", type=float, default=700.0)
    parser.add_argument("--layer-like-max-dip-deg", type=float, default=18.0)
    parser.add_argument("--layer-like-min-horizontal-extent-m", type=float, default=500.0)
    parser.add_argument("--layer-like-max-time-extent-ms", type=float, default=80.0)
    parser.add_argument("--vtk-max-points", type=int, default=250000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, q))


def grid_axis_values(mapping: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    x_values = np.zeros(nx, dtype=np.float64)
    y_values = np.zeros(ny, dtype=np.float64)
    for idx in range(nx):
        x_values[idx] = float(np.median(mapping["x"][ix == idx]))
    for idx in range(ny):
        y_values[idx] = float(np.median(mapping["y"][iy == idx]))
    return x_values, y_values


def pca_orientation(points: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if points.shape[0] < 3:
        return None, None, None
    centered = points - points.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    main = vh[0]
    normal = vh[-1]
    azimuth = float((np.degrees(np.arctan2(main[0], main[1])) + 360.0) % 180.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1.0e-9), 0.0, 1.0))))
    linearity = float(s[0] / max(s[1], 1.0e-9)) if len(s) > 1 else None
    return azimuth, dip, linearity


def build_local_support_grids(
    lowcoh_grid: np.ndarray,
    curv_grid: np.ndarray,
    radius: int,
    support_score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.maximum(lowcoh_grid, curv_grid).astype(np.float32)
    support_binary = support >= float(support_score_threshold)
    if radius <= 0:
        return support, support_binary.astype(np.float32)
    size = 2 * int(radius) + 1
    support_max = ndimage.maximum_filter(support, size=(size, size, size), mode="nearest").astype(np.float32)
    support_fraction = ndimage.uniform_filter(
        support_binary.astype(np.float32),
        size=(size, size, size),
        mode="nearest",
    ).astype(np.float32)
    return support_max, support_fraction


def build_medium_components(
    mask: np.ndarray,
    score: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    raw_labels, raw_count = ndimage.label(mask, structure=structure)
    x_values, y_values = grid_axis_values(mapping)
    time_scale = float(args.orientation_time_scale_m_per_ms)
    component_id_grid = np.zeros(mask.shape, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    next_id = 1
    split_raw_count = 0
    rejected_small = 0
    rejected_horizontal = 0
    rejected_thin = 0
    rejected_layer_like = 0
    rejected_low_dip = 0
    rejected_low_linearity = 0
    rejected_low_score = 0

    sizes = np.bincount(raw_labels.ravel())
    for raw_id in range(1, raw_count + 1):
        if int(sizes[raw_id]) <= 0:
            continue
        yy, xx, tt = np.where(raw_labels == raw_id)
        if yy.size > int(args.max_component_voxels_before_split):
            split_raw_count += 1
            tile = max(int(args.split_tile_cells), 1)
            time_tile = max(int(args.split_time_samples), 1)
            tile_key = (xx // tile) + 10000 * (yy // tile) + 100000000 * (tt // time_tile)
            unique_keys = np.unique(tile_key)
            groups = [np.where(tile_key == key)[0] for key in unique_keys]
        else:
            groups = [np.arange(yy.size)]

        for group in groups:
            gyy = yy[group]
            gxx = xx[group]
            gtt = tt[group]
            voxel_count = int(len(group))
            if voxel_count < int(args.min_component_voxels):
                rejected_small += 1
                continue
            x_extent = float(x_values[gxx].max() - x_values[gxx].min()) if voxel_count else 0.0
            y_extent = float(y_values[gyy].max() - y_values[gyy].min()) if voxel_count else 0.0
            t_extent = float(samples[gtt].max() - samples[gtt].min()) if voxel_count else 0.0
            horizontal_extent = max(x_extent, y_extent)
            horizontal_like = bool(
                t_extent <= float(args.max_horizontal_layer_thickness_ms)
                and horizontal_extent >= float(args.horizontal_layer_min_extent_m)
            )
            if horizontal_like:
                rejected_horizontal += 1
                continue
            if t_extent < float(args.min_vertical_extent_ms):
                rejected_thin += 1
                continue
            points = np.column_stack([x_values[gxx], y_values[gyy], samples[gtt] * time_scale]).astype(np.float64)
            azimuth, dip, linearity = pca_orientation(points)
            score_mean = float(np.mean(score[gyy, gxx, gtt]))
            if score_mean < float(args.min_component_score_mean):
                rejected_low_score += 1
                continue
            if dip is not None and dip < float(args.min_component_dip_deg):
                rejected_low_dip += 1
                continue
            if linearity is not None and linearity < float(args.min_component_linearity):
                rejected_low_linearity += 1
                continue
            layer_like = bool(
                dip is not None
                and dip <= float(args.layer_like_max_dip_deg)
                and horizontal_extent >= float(args.layer_like_min_horizontal_extent_m)
                and t_extent <= float(args.layer_like_max_time_extent_ms)
            )
            if layer_like:
                rejected_layer_like += 1
                continue
            component_id_grid[gyy, gxx, gtt] = next_id
            rows.append(
                {
                    "component_id": next_id,
                    "raw_component_id": int(raw_id),
                    "voxel_count": voxel_count,
                    "score_mean": score_mean,
                    "score_max": float(np.max(score[gyy, gxx, gtt])),
                    "x_min": float(x_values[gxx].min()),
                    "x_max": float(x_values[gxx].max()),
                    "y_min": float(y_values[gyy].min()),
                    "y_max": float(y_values[gyy].max()),
                    "time_min_ms": float(samples[gtt].min()),
                    "time_max_ms": float(samples[gtt].max()),
                    "x_extent_m": x_extent,
                    "y_extent_m": y_extent,
                    "time_extent_ms": t_extent,
                    "pca_azimuth_deg": azimuth,
                    "pca_dip_deg": dip,
                    "pca_linearity": linearity,
                    "layer_like": layer_like,
                }
            )
            next_id += 1

    summary = {
        "raw_component_count": int(raw_count),
        "kept_component_count": int(next_id - 1),
        "split_raw_component_count": int(split_raw_count),
        "rejected_small_group_count": int(rejected_small),
        "rejected_horizontal_group_count": int(rejected_horizontal),
        "rejected_thin_group_count": int(rejected_thin),
        "rejected_layer_like_group_count": int(rejected_layer_like),
        "rejected_low_dip_group_count": int(rejected_low_dip),
        "rejected_low_linearity_group_count": int(rejected_low_linearity),
        "rejected_low_score_group_count": int(rejected_low_score),
        "max_component_voxels_before_split": int(args.max_component_voxels_before_split),
        "split_tile_cells": int(args.split_tile_cells),
        "split_time_samples": int(args.split_time_samples),
        "orientation_time_scale_m_per_ms": time_scale,
    }
    return component_id_grid, pd.DataFrame(rows), summary


def write_component_vtk(
    path: Path,
    component_id_grid: np.ndarray,
    score_grid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    yy, xx, tt = np.where(component_id_grid > 0)
    total = int(len(yy))
    if total == 0:
        pv.PolyData().save(path)
        return {"point_count": 0, "written_point_count": 0, "subsampled": False}
    if total > max_points:
        idx = rng.choice(np.arange(total), size=max_points, replace=False)
        yy, xx, tt = yy[idx], xx[idx], tt[idx]
        subsampled = True
    else:
        subsampled = False
    x_values, y_values = grid_axis_values(mapping)
    points = np.column_stack([x_values[xx], y_values[yy], samples[tt]]).astype(np.float32)
    cloud = pv.PolyData(points)
    cloud["ComponentID"] = component_id_grid[yy, xx, tt].astype(np.int32)
    cloud["MediumPrior"] = score_grid[yy, xx, tt].astype(np.float32)
    cloud.save(path)
    return {"point_count": total, "written_point_count": int(len(points)), "subsampled": subsampled}


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    rng = np.random.default_rng(int(args.random_state))

    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    volume_paths = {key: Path(value).resolve() for key, value in dict(config["volume_paths"]).items()}
    mapping = load_mapping(trace_mapping_npz)
    source_trace_idx = mapping["source_trace_idx"].astype(np.int64)
    density, samples, density_load = load_trace_matrix(input_density_sgy, None, None, "Density")

    print("[step6b-medium] loading seismic attributes", flush=True)
    coherence, _, coh_load = load_trace_matrix(volume_paths["Coherence"], source_trace_idx, samples, "Coherence")
    anttrack, _, ant_load = load_trace_matrix(volume_paths["AntTrack"], source_trace_idx, samples, "AntTrack")
    curvmax, _, curvmax_load = load_trace_matrix(volume_paths["CurvatureMax"], source_trace_idx, samples, "CurvatureMax")
    curvpos, _, curvpos_load = load_trace_matrix(volume_paths["CurvaturePos"], source_trace_idx, samples, "CurvaturePos")

    score_cfg = dict(config.get("score_config", {}))
    ant_cfg = dict(score_cfg.get("anttrack_score", {"low_quantile": 0.20, "high_quantile": 0.96}))
    coh_cfg = dict(score_cfg.get("coherence_score", {"valid_min": 0.0, "low_quantile": 0.05, "high_quantile": 0.95}))
    curvmax_cfg = dict(score_cfg.get("curvaturemax_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    curvpos_cfg = dict(score_cfg.get("curvaturepos_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    ant_valid = valid_values(anttrack, ant_cfg)
    coh_valid = valid_values(coherence, coh_cfg)
    curvmax_valid = valid_values(curvmax, curvmax_cfg)
    curvpos_valid = valid_values(curvpos, curvpos_cfg)
    valid = ant_valid & coh_valid

    ant_score, ant_summary = high_score(anttrack, ant_valid, ant_cfg)
    lowcoh_score, lowcoh_summary = low_score(coherence, coh_valid, coh_cfg)
    curvmax_score, curvmax_summary = high_score(curvmax, curvmax_valid, curvmax_cfg)
    curvpos_score, curvpos_summary = high_score(curvpos, curvpos_valid, curvpos_cfg)
    curv_score = np.maximum(curvmax_score, curvpos_score).astype(np.float32)

    lowcoh_grid, _, _ = flat_to_grid(lowcoh_score, mapping)
    curv_grid, _, _ = flat_to_grid(curv_score, mapping)
    local_support_grid, local_support_fraction_grid = build_local_support_grids(
        lowcoh_grid,
        curv_grid,
        radius=int(args.support_neighborhood_cells),
        support_score_threshold=float(args.support_score_threshold),
    )
    local_support = grid_to_flat(local_support_grid, mapping)
    local_support_fraction = grid_to_flat(local_support_fraction_grid, mapping)
    direct_support = np.maximum(lowcoh_score, curv_score).astype(np.float32)
    medium_score = ant_score * (
        float(args.ant_weight_base)
        + float(args.lowcoh_support_weight) * lowcoh_score
        + float(args.curvature_support_weight) * curv_score
    )
    medium_score[~valid] = 0.0
    medium_score = np.clip(medium_score, 0.0, 1.0).astype(np.float32)
    score_threshold = quantile(medium_score[medium_score > 0], float(args.candidate_quantile))
    ant_threshold = quantile(ant_score[ant_score > 0], float(args.anttrack_high_quantile))
    raw_mask_flat = (
        (medium_score >= score_threshold)
        & (ant_score >= ant_threshold)
        & (direct_support >= float(args.min_direct_support_score))
        & (local_support >= float(args.min_local_support_score))
        & (local_support_fraction >= float(args.min_local_support_fraction))
        & valid
    )

    medium_grid, _, _ = flat_to_grid(medium_score, mapping)
    support_grid, _, _ = flat_to_grid(local_support.astype(np.float32), mapping)
    mask_grid, _, _ = flat_to_grid(raw_mask_flat.astype(np.float32), mapping)
    mask_grid = mask_grid > 0.5
    component_id_grid, component_df, component_summary = build_medium_components(mask_grid, medium_grid, mapping, samples, args)
    kept_mask_grid = component_id_grid > 0
    kept_mask_flat = grid_to_flat(kept_mask_grid.astype(np.float32), mapping)
    component_id_flat = grid_to_flat(component_id_grid.astype(np.float32), mapping).astype(np.int32)
    medium_score_filtered = medium_score.copy()
    medium_score_filtered[kept_mask_flat <= 0.0] = 0.0

    write_sgy_like(output_dir / "medium_corridor_prior.sgy", input_density_sgy, medium_score_filtered, samples)
    write_sgy_like(output_dir / "medium_corridor_mask.sgy", input_density_sgy, kept_mask_flat.astype(np.float32), samples)
    component_df.to_csv(output_dir / "medium_corridor_component_summary.csv", index=False, encoding="utf-8-sig")
    vtk_summary = write_component_vtk(
        output_dir / "medium_corridor_components_raw_time.vtk",
        component_id_grid,
        medium_grid,
        mapping,
        samples,
        max_points=int(args.vtk_max_points),
        rng=rng,
    )
    np.savez_compressed(
        output_dir / "medium_corridor_components.npz",
        medium_prior=medium_score_filtered.astype(np.float32),
        medium_mask=kept_mask_flat.astype(np.uint8),
        medium_component_id=component_id_flat.astype(np.int32),
        local_support=local_support.astype(np.float32),
        local_support_fraction=local_support_fraction.astype(np.float32),
        direct_support=direct_support.astype(np.float32),
        samples=samples.astype(np.float32),
        x=mapping["x"].astype(np.float64),
        y=mapping["y"].astype(np.float64),
        ix=mapping["ix"].astype(np.int32),
        iy=mapping["iy"].astype(np.int32),
    )

    max_component_fraction = (
        float(component_df["voxel_count"].max() / max(int(kept_mask_grid.sum()), 1)) if len(component_df) else 0.0
    )
    summary = {
        "status": "pass",
        "input_density_sgy": str(input_density_sgy),
        "output_dir": str(output_dir),
        "load": {
            "density": density_load,
            "coherence": coh_load,
            "anttrack": ant_load,
            "curvaturemax": curvmax_load,
            "curvaturepos": curvpos_load,
        },
        "score_threshold": score_threshold,
        "ant_score_threshold": ant_threshold,
        "direct_support_threshold": float(args.min_direct_support_score),
        "local_support_threshold": float(args.min_local_support_score),
        "local_support_fraction_threshold": float(args.min_local_support_fraction),
        "score_formula": {
            "formula": "AntTrackScore * (ant_weight_base + lowcoh_support_weight * LowCoherenceScore + curvature_support_weight * CurvatureScore)",
            "ant_weight_base": float(args.ant_weight_base),
            "lowcoh_support_weight": float(args.lowcoh_support_weight),
            "curvature_support_weight": float(args.curvature_support_weight),
            "direct_support": "max(LowCoherenceScore, CurvatureScore)",
            "local_support": "max_filter(max(LowCoherenceScore, CurvatureScore))",
            "local_support_fraction": "local fraction of max(LowCoherenceScore, CurvatureScore) >= support_score_threshold",
            "support_score_threshold": float(args.support_score_threshold),
            "support_neighborhood_cells": int(args.support_neighborhood_cells),
        },
        "raw_candidate_voxel_count": int(raw_mask_flat.sum()),
        "raw_candidate_voxel_fraction": float(raw_mask_flat.mean()),
        "kept_candidate_voxel_count": int(kept_mask_grid.sum()),
        "kept_candidate_voxel_fraction": float(kept_mask_grid.mean()),
        "max_component_fraction": max_component_fraction,
        "component_summary": component_summary,
        "component_count": int(len(component_df)),
        "component_voxel_stats": finite_stats(component_df["voxel_count"]) if len(component_df) else finite_stats([]),
        "component_dip_stats": finite_stats(component_df["pca_dip_deg"]) if len(component_df) else finite_stats([]),
        "component_time_extent_stats": finite_stats(component_df["time_extent_ms"]) if len(component_df) else finite_stats([]),
        "medium_prior_stats": finite_stats(medium_score_filtered[medium_score_filtered > 0]),
        "local_support_stats_in_raw_candidates": finite_stats(local_support[raw_mask_flat]),
        "local_support_fraction_stats_in_raw_candidates": finite_stats(local_support_fraction[raw_mask_flat]),
        "direct_support_stats_in_raw_candidates": finite_stats(direct_support[raw_mask_flat]),
        "local_support_stats_in_kept_candidates": finite_stats(support_grid[kept_mask_grid]),
        "local_support_fraction_stats_in_kept_candidates": finite_stats(local_support_fraction_grid[kept_mask_grid]),
        "attribute_score_summaries": {
            "anttrack": ant_summary,
            "low_coherence": lowcoh_summary,
            "curvaturemax": curvmax_summary,
            "curvaturepos": curvpos_summary,
        },
        "vtk": vtk_summary,
        "reflection": (
            "Step6B creates explicit medium corridor components from AntTrack-led seismic evidence. "
            "Local low-coherence/curvature support gates isolated ant-track noise, and layer-like low-dip components are filtered before Step7B."
        ),
    }
    if max_component_fraction > 0.40:
        summary["status"] = "warn"
        summary["warning"] = "largest medium component still exceeds 40% of kept voxels; Step7B must handle local continuity carefully"
    write_json(output_dir / "medium_corridor_qc.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
