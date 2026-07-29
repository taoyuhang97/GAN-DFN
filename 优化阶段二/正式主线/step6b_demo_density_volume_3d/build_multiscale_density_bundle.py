from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from scipy import ndimage


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
NULL_ABS_LIMIT = 1.0e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multiscale Step6 evidence volumes for candidate demo area.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_stats(values: np.ndarray | list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def quantile_bounds(values: np.ndarray, low_q: float, high_q: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(finite, low_q))
    high = float(np.quantile(finite, high_q))
    if high <= low:
        high = low + 1.0e-6
    return low, high


def valid_values(values: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    valid = np.isfinite(values) & (values > -NULL_ABS_LIMIT) & (values < NULL_ABS_LIMIT)
    if "valid_min" in cfg:
        valid &= values >= float(cfg["valid_min"])
    if "valid_max" in cfg:
        valid &= values <= float(cfg["valid_max"])
    return valid


def high_score(values: np.ndarray, valid: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[valid]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {"valid_count": 0}
    low_q = float(config.get("low_quantile", 0.20))
    high_q = float(config.get("high_quantile", 0.95))
    low, high = quantile_bounds(finite, low_q, high_q)
    power = float(config.get("power", 1.0))
    score = np.clip((values - low) / (high - low), 0.0, 1.0)
    score = np.power(score, power).astype(np.float32)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    return score, {
        "valid_count": int(finite.size),
        "low_quantile": low_q,
        "high_quantile": high_q,
        "low_bound": low,
        "high_bound": high,
        "power": power,
        "value_stats": finite_stats(finite),
        "score_stats": finite_stats(score[valid]),
    }


def low_score(values: np.ndarray, valid: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[valid]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {"valid_count": 0}
    low_q = float(config.get("low_quantile", 0.05))
    high_q = float(config.get("high_quantile", 0.95))
    low, high = quantile_bounds(finite, low_q, high_q)
    power = float(config.get("power", 1.2))
    score = np.clip((high - values) / (high - low), 0.0, 1.0)
    score = np.power(score, power).astype(np.float32)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    return score, {
        "valid_count": int(finite.size),
        "low_quantile": low_q,
        "high_quantile": high_q,
        "low_bound": low,
        "high_bound": high,
        "power": power,
        "value_stats": finite_stats(finite),
        "score_stats": finite_stats(score[valid]),
    }


def load_mapping(path: Path) -> dict[str, np.ndarray]:
    mapping = np.load(path)
    required = {"source_trace_idx", "x", "y", "ix", "iy", "output_trace_index"}
    missing = sorted(required.difference(mapping.files))
    if missing:
        raise ValueError(f"trace mapping missing keys: {missing}")
    return {key: mapping[key] for key in mapping.files}


def read_sgy_sample_axis(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        return samples, {
            "path": str(path),
            "trace_count": int(handle.tracecount),
            "sample_count": int(len(samples)),
            "sample_interval_us": float(segyio.tools.dt(handle)),
        }


def regular_sample_axis(source_samples: np.ndarray, interval_ms: float) -> np.ndarray:
    if len(source_samples) < 1:
        raise ValueError("source sample axis is empty")
    interval = float(interval_ms)
    if interval <= 0.0:
        raise ValueError(f"sample interval must be positive: {interval}")
    start = float(source_samples[0])
    stop = float(source_samples[-1])
    count = int(np.floor((stop - start) / interval + 1.0e-9)) + 1
    return (start + np.arange(count, dtype=np.float64) * interval).astype(np.float64)


def load_trace_matrix(path: Path, source_trace_idx: np.ndarray | None, target_samples: np.ndarray | None, label: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        if source_trace_idx is None:
            matrix = np.stack([handle.trace[idx] for idx in range(handle.tracecount)]).astype(np.float32)
            return matrix, samples, {"path": str(path), "trace_count": int(handle.tracecount), "sample_count": int(len(samples))}

        if target_samples is None:
            raise ValueError("target_samples is required when source_trace_idx is provided")
        same_samples = len(samples) == len(target_samples) and np.allclose(samples, target_samples, rtol=0.0, atol=1.0e-6)
        max_trace = int(np.max(source_trace_idx)) if len(source_trace_idx) else -1
        if max_trace >= int(handle.tracecount):
            raise ValueError(f"{label} tracecount {handle.tracecount} is smaller than mapping max source trace {max_trace}")
        matrix = np.empty((len(source_trace_idx), len(target_samples)), dtype=np.float32)
        for out_idx, src_idx in enumerate(source_trace_idx):
            trace = np.asarray(handle.trace[int(src_idx)], dtype=np.float32)
            if same_samples:
                matrix[out_idx, :] = trace
            else:
                matrix[out_idx, :] = np.interp(target_samples, samples, trace, left=np.nan, right=np.nan).astype(np.float32)
            if (out_idx + 1) % 10000 == 0:
                print(f"[step6-multiscale] loaded {label} traces={out_idx + 1}/{len(source_trace_idx)}", flush=True)
        return matrix, target_samples, {
            "path": str(path),
            "source_trace_count": int(handle.tracecount),
            "loaded_trace_count": int(len(source_trace_idx)),
            "source_sample_count": int(len(samples)),
            "target_sample_count": int(len(target_samples)),
            "sample_axis_matched": bool(same_samples),
        }


def flat_to_grid(flat: np.ndarray, mapping: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    nt = int(flat.shape[1])
    grid = np.zeros((ny, nx, nt), dtype=np.float32)
    valid_grid = np.zeros((ny, nx, nt), dtype=bool)
    grid[iy, ix, :] = flat
    valid_grid[iy, ix, :] = np.isfinite(flat) & (np.abs(flat) < NULL_ABS_LIMIT)
    return grid, valid_grid, np.asarray([ny, nx, nt], dtype=np.int32)


def grid_to_flat(grid: np.ndarray, mapping: dict[str, np.ndarray]) -> np.ndarray:
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    return grid[iy, ix, :].astype(np.float32)


def filter_components(mask: np.ndarray, score: np.ndarray, config: dict[str, Any], label_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    min_voxels = int(config.get("min_component_voxels", 20))
    structure = np.ones((3, 3, 3), dtype=np.uint8) if bool(config.get("use_26_connectivity", True)) else None
    labels, count = ndimage.label(mask, structure=structure)
    if count <= 0:
        return np.zeros_like(mask, dtype=bool), {"label": label_name, "component_count": 0, "kept_component_count": 0}
    sizes = np.bincount(labels.ravel())
    keep_ids = [idx for idx in range(1, len(sizes)) if int(sizes[idx]) >= min_voxels]
    kept = np.isin(labels, keep_ids)
    examples: list[dict[str, Any]] = []
    for component_id in keep_ids[: int(config.get("summary_component_limit", 20))]:
        loc = np.where(labels == component_id)
        if loc[0].size == 0:
            continue
        examples.append(
            {
                "component_id": int(component_id),
                "voxel_count": int(sizes[component_id]),
                "score_mean": float(np.mean(score[loc])),
                "x_extent_cells": int(loc[1].max() - loc[1].min() + 1),
                "y_extent_cells": int(loc[0].max() - loc[0].min() + 1),
                "time_extent_samples": int(loc[2].max() - loc[2].min() + 1),
            }
        )
    return kept, {
        "label": label_name,
        "component_count": int(count),
        "kept_component_count": int(len(keep_ids)),
        "raw_voxel_count": int(mask.sum()),
        "kept_voxel_count": int(kept.sum()),
        "min_component_voxels": min_voxels,
        "component_examples": examples,
    }


def steep_large_mask(
    lowcoh: np.ndarray,
    valid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    threshold = float(config.get("score_threshold", 0.68))
    raw = valid & (lowcoh >= threshold)
    structure = np.ones((3, 3, 3), dtype=np.uint8) if bool(config.get("use_26_connectivity", True)) else None
    labels, count = ndimage.label(raw, structure=structure)
    if count <= 0:
        return np.zeros_like(raw, dtype=bool), np.zeros_like(lowcoh, dtype=np.float32), {"component_count": 0}

    x_values = np.sort(np.unique(mapping["x"].astype(float)))
    y_values = np.sort(np.unique(mapping["y"].astype(float)))
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))
    min_voxels = int(config.get("min_component_voxels", 120))
    min_time_extent = float(config.get("min_time_extent_ms", 35.0))
    max_layer_thickness = float(config.get("horizontal_max_time_extent_ms", 12.0))
    min_horizontal_extent = float(config.get("horizontal_min_extent_m", 700.0))
    min_dip = float(config.get("min_pca_dip_deg", 45.0))
    boost_floor = float(config.get("kept_component_floor", 0.35))
    kept = np.zeros_like(raw, dtype=bool)
    large_score = np.zeros_like(lowcoh, dtype=np.float32)
    summaries: list[dict[str, Any]] = []
    kept_count = 0
    rejected_horizontal = 0
    rejected_low_dip = 0

    sizes = np.bincount(labels.ravel())
    for component_id in range(1, count + 1):
        if int(sizes[component_id]) < min_voxels:
            continue
        yy, xx, tt = np.where(labels == component_id)
        if yy.size < 3:
            continue
        x_extent = float(x_values[xx].max() - x_values[xx].min()) if len(x_values) else float(xx.max() - xx.min())
        y_extent = float(y_values[yy].max() - y_values[yy].min()) if len(y_values) else float(yy.max() - yy.min())
        t_extent = float(samples[tt].max() - samples[tt].min())
        horizontal_extent = max(x_extent, y_extent)
        coords = np.column_stack([x_values[xx], y_values[yy], samples[tt] * time_scale]).astype(float)
        centered = coords - coords.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        dip = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1.0e-9), 0.0, 1.0))))
        horizontal_like = bool(t_extent <= max_layer_thickness and horizontal_extent >= min_horizontal_extent)
        keep = bool(t_extent >= min_time_extent and dip >= min_dip and not horizontal_like)
        if keep:
            component_mask = labels == component_id
            kept |= component_mask
            strength = np.clip((dip - min_dip) / max(85.0 - min_dip, 1.0), 0.0, 1.0)
            vertical_strength = np.clip(t_extent / max(float(config.get("strong_time_extent_ms", 120.0)), 1.0), 0.0, 1.0)
            multiplier = max(boost_floor, 0.55 * strength + 0.45 * vertical_strength)
            large_score[component_mask] = np.maximum(large_score[component_mask], lowcoh[component_mask] * multiplier)
            kept_count += 1
        elif horizontal_like:
            rejected_horizontal += 1
        else:
            rejected_low_dip += 1
        if len(summaries) < int(config.get("summary_component_limit", 30)):
            summaries.append(
                {
                    "component_id": int(component_id),
                    "voxel_count": int(sizes[component_id]),
                    "x_extent_m": x_extent,
                    "y_extent_m": y_extent,
                    "time_extent_ms": t_extent,
                    "pca_dip_deg": dip,
                    "horizontal_like": horizontal_like,
                    "kept": keep,
                }
            )

    large_score[~valid] = 0.0
    return kept, large_score.astype(np.float32), {
        "component_count": int(count),
        "kept_component_count": int(kept_count),
        "rejected_horizontal_component_count": int(rejected_horizontal),
        "rejected_low_dip_or_thin_component_count": int(rejected_low_dip),
        "threshold": threshold,
        "min_component_voxels": min_voxels,
        "min_time_extent_ms": min_time_extent,
        "min_pca_dip_deg": min_dip,
        "component_examples": summaries,
        "score_stats": finite_stats(large_score[large_score > 0]),
    }


def write_sgy_like(output_path: Path, template_path: Path, data: np.ndarray, samples: np.ndarray) -> None:
    sample_axis = np.asarray(samples, dtype=np.float64)
    if sample_axis.ndim != 1 or len(sample_axis) == 0:
        raise ValueError("SGY sample axis must be a non-empty one-dimensional array")
    if data.shape != (data.shape[0], len(sample_axis)):
        raise ValueError(f"SGY data/sample shape mismatch: data={data.shape}, samples={len(sample_axis)}")
    if len(sample_axis) > 1:
        intervals = np.diff(sample_axis)
        interval_ms = float(np.median(intervals))
        if interval_ms <= 0.0 or not np.allclose(intervals, interval_ms, atol=1.0e-6, rtol=0.0):
            raise ValueError("SGY output requires a positive regular sample axis")
    else:
        interval_ms = 1.0
    interval_us = int(round(interval_ms * 1000.0))
    delay_ms = int(round(float(sample_axis[0])))
    with segyio.open(str(template_path), "r", ignore_geometry=True) as src:
        if data.shape[0] > int(src.tracecount):
            raise ValueError(f"template trace count {src.tracecount} is smaller than output trace count {data.shape[0]}")
        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
        spec.format = int(src.bin[segyio.BinField.Format]) or 5
        spec.samples = sample_axis.astype(np.float32)
        spec.tracecount = int(data.shape[0])
        with segyio.create(str(output_path), spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin.update(src.bin)
            dst.bin[segyio.BinField.Interval] = interval_us
            dst.bin[segyio.BinField.Samples] = int(len(samples))
            dst.bin[segyio.BinField.Format] = 5
            for trace_idx in range(data.shape[0]):
                dst.header[trace_idx] = dict(src.header[trace_idx])
                dst.header[trace_idx][segyio.TraceField.TRACE_SAMPLE_INTERVAL] = interval_us
                dst.header[trace_idx][segyio.TraceField.TRACE_SAMPLE_COUNT] = int(len(sample_axis))
                dst.header[trace_idx][segyio.TraceField.DelayRecordingTime] = delay_ms
                dst.trace[trace_idx] = np.nan_to_num(data[trace_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if (trace_idx + 1) % 10000 == 0:
                    print(f"[step6-multiscale] wrote {output_path.name} traces={trace_idx + 1}/{data.shape[0]}", flush=True)
            dst.flush()

    with segyio.open(str(output_path), "r", ignore_geometry=True) as check:
        written_samples = np.asarray(check.samples, dtype=np.float64)
        binary_interval_us = int(check.bin[segyio.BinField.Interval])
        trace_intervals = {
            int(check.header[index][segyio.TraceField.TRACE_SAMPLE_INTERVAL])
            for index in sorted({0, max(0, int(check.tracecount) - 1)})
        }
        if int(check.tracecount) != int(data.shape[0]):
            raise RuntimeError(f"SGY trace count verification failed for {output_path}")
        if len(written_samples) != len(sample_axis) or not np.allclose(
            written_samples, sample_axis, atol=1.0e-6, rtol=0.0
        ):
            raise RuntimeError(
                f"SGY sample axis verification failed for {output_path}: "
                f"written=({written_samples[0]}, {written_samples[-1]}, {len(written_samples)}) "
                f"expected=({sample_axis[0]}, {sample_axis[-1]}, {len(sample_axis)})"
            )
        if binary_interval_us != interval_us or trace_intervals != {interval_us}:
            raise RuntimeError(
                f"SGY interval verification failed for {output_path}: "
                f"binary={binary_interval_us}, trace={sorted(trace_intervals)}, expected={interval_us}"
            )


def copy_mapping(src: Path, dst: Path) -> None:
    mapping = np.load(src)
    np.savez_compressed(dst, **{key: mapping[key] for key in mapping.files})


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)

    volume_paths = {key: Path(value).resolve() for key, value in dict(config["volume_paths"]).items()}
    for label, path in [("input_density_sgy", input_density_sgy), ("trace_mapping_npz", trace_mapping_npz), *volume_paths.items()]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    mapping = load_mapping(trace_mapping_npz)
    source_trace_idx = mapping["source_trace_idx"].astype(np.int64)
    density, samples, density_summary = load_trace_matrix(input_density_sgy, None, None, "Density")
    if density.shape[0] != len(source_trace_idx):
        raise ValueError(f"density tracecount {density.shape[0]} != mapping rows {len(source_trace_idx)}")
    density = np.clip(np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None).astype(np.float32)

    print("[step6-multiscale] loading attributes", flush=True)
    coherence, _, coh_load = load_trace_matrix(volume_paths["Coherence"], source_trace_idx, samples, "Coherence")
    anttrack, _, ant_load = load_trace_matrix(volume_paths["AntTrack"], source_trace_idx, samples, "AntTrack")
    curvmax, _, curvmax_load = load_trace_matrix(volume_paths["CurvatureMax"], source_trace_idx, samples, "CurvatureMax")
    if "CurvaturePos" in volume_paths:
        curvpos, _, curvpos_load = load_trace_matrix(volume_paths["CurvaturePos"], source_trace_idx, samples, "CurvaturePos")
    else:
        curvpos = np.zeros_like(curvmax, dtype=np.float32)
        curvpos_load = {"path": None, "loaded_trace_count": 0}

    score_cfg = dict(config.get("score_config", {}))
    ant_cfg = dict(score_cfg.get("anttrack_score", {}))
    coh_cfg = dict(score_cfg.get("coherence_score", {}))
    curvmax_cfg = dict(score_cfg.get("curvaturemax_score", score_cfg.get("curvature_score", {})))
    curvpos_cfg = dict(score_cfg.get("curvaturepos_score", score_cfg.get("curvature_score", {})))

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

    density_grid, valid_density_grid, _ = flat_to_grid(density, mapping)
    ant_grid, _, _ = flat_to_grid(ant_score, mapping)
    lowcoh_grid, coh_valid_grid, _ = flat_to_grid(lowcoh_score, mapping)
    curv_grid, _, _ = flat_to_grid(curv_score, mapping)
    valid_grid, _, _ = flat_to_grid(valid.astype(np.float32), mapping)
    valid_grid = valid_grid > 0.5

    medium_cfg = dict(config.get("medium_prior", {}))
    medium_prior_grid = ant_grid * (
        float(medium_cfg.get("ant_weight_base", 0.65))
        + float(medium_cfg.get("lowcoh_support_weight", 0.20)) * lowcoh_grid
        + float(medium_cfg.get("curvature_support_weight", 0.15)) * curv_grid
    )
    medium_prior_grid = np.clip(medium_prior_grid, 0.0, 1.0).astype(np.float32)
    medium_raw_mask = valid_grid & (medium_prior_grid >= float(medium_cfg.get("score_threshold", 0.55))) & (ant_grid >= float(medium_cfg.get("min_ant_score", 0.55)))
    medium_mask, medium_component_summary = filter_components(medium_raw_mask, medium_prior_grid, medium_cfg, "medium")
    medium_prior_grid[~medium_mask] *= float(medium_cfg.get("outside_mask_attenuation", 0.15))

    large_cfg = dict(config.get("large_prior", {}))
    large_mask, large_prior_grid, large_component_summary = steep_large_mask(lowcoh_grid, valid_grid & coh_valid_grid, mapping, samples, large_cfg)
    large_prior_grid = np.maximum(large_prior_grid, large_mask.astype(np.float32) * lowcoh_grid * float(large_cfg.get("mask_score_floor_multiplier", 0.65)))
    large_prior_grid = np.clip(large_prior_grid, 0.0, 1.0).astype(np.float32)

    small_cfg = dict(config.get("small_prior", {}))
    finite_density = density_grid[np.isfinite(density_grid) & (density_grid > 0)]
    d95 = float(np.quantile(finite_density, float(small_cfg.get("density_quantile_scale", 0.95)))) if finite_density.size else 1.0
    density_score_grid = np.clip(density_grid / max(d95, 1.0e-6), 0.0, 1.0).astype(np.float32)
    exclusion = np.maximum(medium_prior_grid, large_prior_grid)
    small_prior_grid = density_score_grid * np.clip(1.0 - float(small_cfg.get("large_medium_suppression", 0.55)) * exclusion, 0.10, 1.0)
    small_density_grid = density_grid * np.clip(1.0 - float(small_cfg.get("large_medium_density_attenuation", 0.35)) * exclusion, 0.20, 1.0)
    small_prior_grid = np.clip(small_prior_grid, 0.0, 1.0).astype(np.float32)
    small_density_grid = np.clip(small_density_grid, 0.0, None).astype(np.float32)

    integration_cfg = dict(config.get("integration", {}))
    integrated_density_grid = (
        float(integration_cfg.get("small_weight", 1.00)) * small_density_grid
        + float(integration_cfg.get("medium_additive_density", 4.0)) * np.power(medium_prior_grid, float(integration_cfg.get("medium_power", 1.2)))
        + float(integration_cfg.get("large_additive_density", 5.5)) * np.power(large_prior_grid, float(integration_cfg.get("large_power", 1.15)))
    )
    integrated_density_grid = np.clip(integrated_density_grid, 0.0, float(integration_cfg.get("density_cap", 10.0))).astype(np.float32)

    outputs = {
        "small_density_sgy": output_dir / str(config.get("small_density_sgy_name", "small_density.sgy")),
        "small_prior_sgy": output_dir / str(config.get("small_prior_sgy_name", "small_prior.sgy")),
        "medium_prior_sgy": output_dir / str(config.get("medium_prior_sgy_name", "medium_prior.sgy")),
        "medium_mask_sgy": output_dir / str(config.get("medium_mask_sgy_name", "medium_candidate_mask.sgy")),
        "large_prior_sgy": output_dir / str(config.get("large_prior_sgy_name", "large_prior.sgy")),
        "large_mask_sgy": output_dir / str(config.get("large_mask_sgy_name", "large_candidate_mask.sgy")),
        "integrated_density_sgy": output_dir / str(config.get("integrated_density_sgy_name", "integrated_density.sgy")),
        "mapping_npz": output_dir / str(config.get("output_trace_mapping_name", "candidate_cheye1_3d_trace_mapping.npz")),
        "bundle_npz": output_dir / str(config.get("bundle_npz_name", "multiscale_prior_bundle.npz")),
        "summary_json": output_dir / str(config.get("summary_name", "multiscale_density_bundle_summary.json")),
    }

    print("[step6-multiscale] writing SGYs", flush=True)
    write_sgy_like(outputs["small_density_sgy"], input_density_sgy, grid_to_flat(small_density_grid, mapping), samples)
    write_sgy_like(outputs["small_prior_sgy"], input_density_sgy, grid_to_flat(small_prior_grid, mapping), samples)
    write_sgy_like(outputs["medium_prior_sgy"], input_density_sgy, grid_to_flat(medium_prior_grid, mapping), samples)
    write_sgy_like(outputs["medium_mask_sgy"], input_density_sgy, grid_to_flat(medium_mask.astype(np.float32), mapping), samples)
    write_sgy_like(outputs["large_prior_sgy"], input_density_sgy, grid_to_flat(large_prior_grid, mapping), samples)
    write_sgy_like(outputs["large_mask_sgy"], input_density_sgy, grid_to_flat(large_mask.astype(np.float32), mapping), samples)
    write_sgy_like(outputs["integrated_density_sgy"], input_density_sgy, grid_to_flat(integrated_density_grid, mapping), samples)
    copy_mapping(trace_mapping_npz, outputs["mapping_npz"])

    x_values = np.sort(np.unique(mapping["x"].astype(float))).astype(np.float32)
    y_values = np.sort(np.unique(mapping["y"].astype(float))).astype(np.float32)
    np.savez_compressed(
        outputs["bundle_npz"],
        x_values=x_values,
        y_values=y_values,
        samples=samples.astype(np.float32),
        small_density=small_density_grid.astype(np.float16),
        small_prior=small_prior_grid.astype(np.float16),
        medium_prior=medium_prior_grid.astype(np.float16),
        medium_mask=medium_mask.astype(np.uint8),
        large_prior=large_prior_grid.astype(np.float16),
        large_mask=large_mask.astype(np.uint8),
        integrated_density=integrated_density_grid.astype(np.float16),
    )

    medium_top = medium_prior_grid >= float(np.quantile(medium_prior_grid[np.isfinite(medium_prior_grid)], 0.99))
    large_top = large_prior_grid >= float(np.quantile(large_prior_grid[np.isfinite(large_prior_grid)], 0.99))
    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "inputs": {
            "density": str(input_density_sgy),
            "trace_mapping": str(trace_mapping_npz),
            "volume_paths": {key: str(path) for key, path in volume_paths.items()},
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "sample_axis": {"time_min_ms": float(samples.min()), "time_max_ms": float(samples.max()), "sample_count": int(len(samples))},
        "trace_count": int(density.shape[0]),
        "attribute_load": {
            "Density": density_summary,
            "Coherence": coh_load,
            "AntTrack": ant_load,
            "CurvatureMax": curvmax_load,
            "CurvaturePos": curvpos_load,
        },
        "score_config": score_cfg,
        "score_summary": {
            "AntTrackScore": ant_summary,
            "LowCoherenceScore": lowcoh_summary,
            "CurvatureMaxScore": curvmax_summary,
            "CurvaturePosScore": curvpos_summary,
        },
        "component_summary": {
            "medium": medium_component_summary,
            "large": large_component_summary,
        },
        "stats": {
            "input_density": finite_stats(density),
            "small_density": finite_stats(small_density_grid),
            "small_prior": finite_stats(small_prior_grid),
            "medium_prior": finite_stats(medium_prior_grid),
            "large_prior": finite_stats(large_prior_grid),
            "integrated_density": finite_stats(integrated_density_grid),
        },
        "qc": {
            "medium_mask_voxels": int(medium_mask.sum()),
            "large_mask_voxels": int(large_mask.sum()),
            "medium_top1pct_in_medium_mask": float((medium_top & medium_mask).sum() / max(int(medium_top.sum()), 1)),
            "large_top1pct_in_large_mask": float((large_top & large_mask).sum() / max(int(large_top.sum()), 1)),
            "small_density_scale_p95": d95,
        },
        "checks": {
            "all_sgy_outputs_exist": all(path.exists() for key, path in outputs.items() if key.endswith("_sgy")),
            "mapping_exists": outputs["mapping_npz"].exists(),
            "bundle_exists": outputs["bundle_npz"].exists(),
            "medium_has_candidates": int(medium_mask.sum()) > 0,
            "large_checked": "component_count" in large_component_summary,
        },
    }
    summary["status"] = "pass" if all(bool(v) for v in summary["checks"].values()) else "fail"
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step6-multiscale] summary={outputs['summary_json']}", flush=True)
    print(f"[step6-multiscale] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
