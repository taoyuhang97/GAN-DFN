from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from scipy import ndimage


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_3d_density_seismic_prior_v1_postprocess.json"
NULL_ABS_LIMIT = 1.0e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step6B seismic-interpretation prior and corrected 3D density SGYs.")
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


def load_trace_mapping(path: Path) -> dict[str, np.ndarray]:
    mapping = np.load(path)
    required = {"source_trace_idx", "x", "y", "ix", "iy", "output_trace_index"}
    missing = sorted(required.difference(mapping.files))
    if missing:
        raise ValueError(f"trace mapping missing keys: {missing}")
    return {key: mapping[key] for key in mapping.files}


def valid_values(values: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    config = config or {}
    valid = np.isfinite(values) & (values > -NULL_ABS_LIMIT) & (values < NULL_ABS_LIMIT)
    if "valid_min" in config:
        valid &= values >= float(config["valid_min"])
    if "valid_max" in config:
        valid &= values <= float(config["valid_max"])
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


def steep_low_coherence_score(
    low_score_flat: np.ndarray,
    valid_flat: np.ndarray,
    mapping: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    steep_cfg = dict(config.get("steep_anomaly", {}))
    if not bool(steep_cfg.get("enabled", False)):
        return low_score_flat.astype(np.float32), {"enabled": False}

    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    _, nt = low_score_flat.shape
    score_grid = np.zeros((ny, nx, nt), dtype=np.float32)
    valid_grid = np.zeros((ny, nx, nt), dtype=bool)
    score_grid[iy, ix, :] = low_score_flat
    valid_grid[iy, ix, :] = valid_flat

    threshold = float(steep_cfg.get("component_score_threshold", 0.55))
    mask = valid_grid & np.isfinite(score_grid) & (score_grid >= threshold)
    if not mask.any():
        return np.zeros_like(low_score_flat, dtype=np.float32), {"enabled": True, "component_count": 0}

    structure = np.ones((3, 3, 3), dtype=np.uint8) if bool(steep_cfg.get("use_26_connectivity", True)) else None
    labels, component_count = ndimage.label(mask, structure=structure)
    objects = ndimage.find_objects(labels)
    multiplier_by_label = np.zeros(component_count + 1, dtype=np.float32)

    min_vertical_t = float(steep_cfg.get("min_vertical_time_samples", 6.0))
    strong_vertical_t = float(steep_cfg.get("strong_vertical_time_samples", 18.0))
    horizontal_max_t = float(steep_cfg.get("horizontal_max_time_samples", 5.0))
    horizontal_min_xy = float(steep_cfg.get("horizontal_min_xy_cells", 60.0))
    horizontal_penalty = float(steep_cfg.get("horizontal_sheet_penalty", 0.90))
    min_multiplier = float(steep_cfg.get("min_component_multiplier", 0.08))
    max_multiplier = float(steep_cfg.get("max_component_multiplier", 1.0))
    summary_limit = int(steep_cfg.get("summary_component_limit", 20))
    horizontal_component_count = 0
    steep_component_count = 0
    component_summaries: list[dict[str, Any]] = []

    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        y_slice, x_slice, t_slice = slc
        y_extent = y_slice.stop - y_slice.start
        x_extent = x_slice.stop - x_slice.start
        t_extent = t_slice.stop - t_slice.start
        xy_extent = max(x_extent, y_extent)
        vertical_score = float(np.clip((t_extent - min_vertical_t) / max(strong_vertical_t - min_vertical_t, 1.0e-6), 0.0, 1.0))
        horizontal_like = bool(t_extent <= horizontal_max_t and xy_extent >= horizontal_min_xy)
        multiplier = min_multiplier + (max_multiplier - min_multiplier) * vertical_score
        if horizontal_like:
            multiplier *= max(0.0, 1.0 - horizontal_penalty)
            horizontal_component_count += 1
        if vertical_score >= 0.75 and not horizontal_like:
            steep_component_count += 1
        multiplier_by_label[label_id] = np.float32(np.clip(multiplier, 0.0, max_multiplier))
        if len(component_summaries) < summary_limit:
            component_summaries.append(
                {
                    "label": int(label_id),
                    "x_extent_cells": int(x_extent),
                    "y_extent_cells": int(y_extent),
                    "time_extent_samples": int(t_extent),
                    "vertical_score": vertical_score,
                    "horizontal_like": horizontal_like,
                    "multiplier": float(multiplier_by_label[label_id]),
                }
            )

    steep_grid = score_grid * multiplier_by_label[labels]
    steep_flat = steep_grid[iy, ix, :].astype(np.float32)
    steep_flat[~valid_flat] = 0.0
    return steep_flat, {
        "enabled": True,
        "component_score_threshold": threshold,
        "component_count": int(component_count),
        "horizontal_component_count": int(horizontal_component_count),
        "steep_component_count": int(steep_component_count),
        "min_vertical_time_samples": min_vertical_t,
        "strong_vertical_time_samples": strong_vertical_t,
        "horizontal_max_time_samples": horizontal_max_t,
        "horizontal_min_xy_cells": horizontal_min_xy,
        "horizontal_sheet_penalty": horizontal_penalty,
        "score_stats": finite_stats(steep_flat[valid_flat]),
        "component_examples": component_summaries,
    }


def load_attribute_matrix(
    sgy_path: Path,
    source_trace_idx: np.ndarray,
    target_samples: np.ndarray,
    tracecount: int,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.empty((tracecount, len(target_samples)), dtype=np.float32)
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as handle:
        source_samples = np.asarray(handle.samples, dtype=np.float64)
        same_samples = len(source_samples) == len(target_samples) and np.allclose(source_samples, target_samples, rtol=0.0, atol=1.0e-6)
        max_trace_idx = int(np.max(source_trace_idx)) if len(source_trace_idx) else -1
        if max_trace_idx >= int(handle.tracecount):
            raise ValueError(f"{label} tracecount {handle.tracecount} is smaller than mapping max source trace {max_trace_idx}")
        for out_idx, src_idx in enumerate(source_trace_idx):
            trace = np.asarray(handle.trace[int(src_idx)], dtype=np.float32)
            if same_samples:
                matrix[out_idx, :] = trace
            else:
                matrix[out_idx, :] = np.interp(target_samples, source_samples, trace, left=np.nan, right=np.nan).astype(np.float32)
            if (out_idx + 1) % 10000 == 0:
                print(f"[step6b-seismic-prior] loaded {label} traces={out_idx + 1}/{tracecount}", flush=True)
    return matrix, {
        "path": str(sgy_path),
        "sample_axis_matched": bool(same_samples),
        "source_sample_count": int(len(source_samples)),
        "target_sample_count": int(len(target_samples)),
        "trace_count_loaded": int(tracecount),
    }


def copy_mapping(src: Path, dst: Path) -> None:
    mapping = np.load(src)
    np.savez_compressed(dst, **{key: mapping[key] for key in mapping.files})


def write_sgy_like(
    output_path: Path,
    template_handle: segyio.SegyFile,
    data: np.ndarray,
    samples: np.ndarray,
) -> None:
    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
    spec.format = int(template_handle.bin[segyio.BinField.Format]) or 5
    spec.samples = np.asarray(samples, dtype=np.float32)
    spec.tracecount = int(data.shape[0])
    with segyio.create(str(output_path), spec) as dst:
        dst.text[0] = template_handle.text[0]
        dst.bin.update(template_handle.bin)
        dst.bin[segyio.BinField.Samples] = int(len(samples))
        dst.bin[segyio.BinField.Format] = 5
        for trace_idx in range(data.shape[0]):
            dst.header[trace_idx] = dict(template_handle.header[trace_idx])
            dst.trace[trace_idx] = data[trace_idx].astype(np.float32)
            if (trace_idx + 1) % 10000 == 0:
                print(f"[step6b-seismic-prior] wrote {output_path.name} traces={trace_idx + 1}/{data.shape[0]}", flush=True)


def top_density_alignment(values: np.ndarray, masks: dict[str, np.ndarray], quantile: float = 0.99) -> dict[str, Any]:
    valid = np.isfinite(values)
    positive = values[valid]
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return {"top_quantile": quantile, "top_count": 0}
    threshold = float(np.quantile(positive, quantile))
    top = valid & (values >= threshold)
    out: dict[str, Any] = {"top_quantile": quantile, "threshold": threshold, "top_count": int(top.sum())}
    for name, mask in masks.items():
        denominator = int(top.sum())
        out[name] = float((top & mask).sum() / denominator) if denominator else None
    return out


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)

    output_density_sgy = output_dir / str(config.get("output_density_sgy_name", "candidate_cheye1_3d_predicted_density_seismic_prior_v1.sgy"))
    output_prior_sgy = output_dir / str(config.get("output_prior_sgy_name", "candidate_cheye1_3d_seismic_prior_v1.sgy"))
    output_mapping_npz = output_dir / str(config.get("output_trace_mapping_name", "candidate_cheye1_3d_trace_mapping.npz"))
    output_summary_json = output_dir / str(config.get("output_summary_name", "candidate_cheye1_3d_density_seismic_prior_v1_summary.json"))

    volume_paths = {key: Path(value).resolve() for key, value in dict(config["volume_paths"]).items()}
    required = {"Coherence", "AntTrack", "CurvatureMax"}
    missing = sorted(required.difference(volume_paths))
    if missing:
        raise ValueError(f"volume_paths missing required keys: {missing}")
    for label, path in [("input_density_sgy", input_density_sgy), ("trace_mapping_npz", trace_mapping_npz), *volume_paths.items()]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    mapping = load_trace_mapping(trace_mapping_npz)
    source_trace_idx = mapping["source_trace_idx"].astype(np.int64)

    with segyio.open(str(input_density_sgy), "r", ignore_geometry=True) as src_density:
        samples = np.asarray(src_density.samples, dtype=np.float64)
        tracecount = int(src_density.tracecount)
        if tracecount != len(source_trace_idx):
            raise ValueError(f"density tracecount {tracecount} != mapping rows {len(source_trace_idx)}")
        density = np.stack([src_density.trace[idx] for idx in range(tracecount)]).astype(np.float32)

        print("[step6b-seismic-prior] loading attributes", flush=True)
        coherence, coherence_load_summary = load_attribute_matrix(volume_paths["Coherence"], source_trace_idx, samples, tracecount, "Coherence")
        anttrack, anttrack_load_summary = load_attribute_matrix(volume_paths["AntTrack"], source_trace_idx, samples, tracecount, "AntTrack")
        curvature, curvature_load_summary = load_attribute_matrix(volume_paths["CurvatureMax"], source_trace_idx, samples, tracecount, "CurvatureMax")

        score_cfg = dict(config.get("score_config", {}))
        ant_cfg = dict(score_cfg.get("anttrack_score", {}))
        coh_cfg = dict(score_cfg.get("coherence_score", {}))
        curv_cfg = dict(score_cfg.get("curvature_score", {}))
        ant_valid = valid_values(anttrack, ant_cfg)
        coh_valid = valid_values(coherence, coh_cfg)
        curv_valid = valid_values(curvature, curv_cfg)

        ant_score, ant_summary = high_score(anttrack, ant_valid, ant_cfg)
        lowcoh_score, lowcoh_summary = low_score(coherence, coh_valid, coh_cfg)
        steep_score, steep_summary = steep_low_coherence_score(lowcoh_score, coh_valid, mapping, coh_cfg)
        curv_score, curv_summary = high_score(curvature, curv_valid, curv_cfg)

        weights = dict(config.get("prior_weights", {}))
        w_ant = float(weights.get("anttrack_high", 0.50))
        w_coh = float(weights.get("steep_low_coherence", 0.35))
        w_curv = float(weights.get("curvature_max_high", 0.15))
        weight_sum = w_ant + w_coh + w_curv
        if weight_sum <= 0.0:
            raise ValueError("prior weights must have positive sum")
        prior = (w_ant * ant_score + w_coh * steep_score + w_curv * curv_score) / weight_sum
        prior = np.clip(prior, 0.0, 1.0).astype(np.float32)

        update_cfg = dict(config.get("density_update", {}))
        gain = float(update_cfg.get("gain", 1.2))
        additive_alpha = float(update_cfg.get("additive_alpha", 0.30))
        additive_q = float(update_cfg.get("additive_density_scale_quantile", 0.95))
        additive_power = float(update_cfg.get("additive_power", 1.3))
        density_cap = float(update_cfg.get("density_cap", 10.0))
        preserve_zero_density = bool(update_cfg.get("preserve_zero_density", True))
        finite_density = density[np.isfinite(density) & (density > 0.0)]
        density_scale = float(np.quantile(finite_density, additive_q)) if finite_density.size else 1.0
        multiplicative = density * (1.0 + gain * prior)
        additive = additive_alpha * density_scale * np.power(prior, additive_power)
        output_density = multiplicative + additive
        output_density[~np.isfinite(output_density)] = 0.0
        output_density = np.clip(output_density, 0.0, density_cap).astype(np.float32)
        if preserve_zero_density:
            output_density[(density <= 0.0) & (prior <= 0.0)] = 0.0

        print("[step6b-seismic-prior] writing SGYs", flush=True)
        write_sgy_like(output_density_sgy, src_density, output_density, samples)
        write_sgy_like(output_prior_sgy, src_density, prior, samples)

    copy_mapping(trace_mapping_npz, output_mapping_npz)

    ant_q80 = float(np.quantile(anttrack[ant_valid], 0.80)) if ant_valid.any() else np.nan
    ant_q90 = float(np.quantile(anttrack[ant_valid], 0.90)) if ant_valid.any() else np.nan
    coh_q20 = float(np.quantile(coherence[coh_valid], 0.20)) if coh_valid.any() else np.nan
    curv_q80 = float(np.quantile(curvature[curv_valid], 0.80)) if curv_valid.any() else np.nan
    curv_q90 = float(np.quantile(curvature[curv_valid], 0.90)) if curv_valid.any() else np.nan
    prior_q80 = float(np.quantile(prior[np.isfinite(prior)], 0.80))
    prior_q90 = float(np.quantile(prior[np.isfinite(prior)], 0.90))
    masks = {
        "fraction_in_anttrack_ge_q80": ant_valid & (anttrack >= ant_q80),
        "fraction_in_anttrack_ge_q90": ant_valid & (anttrack >= ant_q90),
        "fraction_in_coherence_le_q20": coh_valid & (coherence <= coh_q20),
        "fraction_in_curvaturemax_ge_q80": curv_valid & (curvature >= curv_q80),
        "fraction_in_curvaturemax_ge_q90": curv_valid & (curvature >= curv_q90),
        "fraction_in_prior_ge_q80": prior >= prior_q80,
        "fraction_in_prior_ge_q90": prior >= prior_q90,
    }
    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "input_density_sgy": str(input_density_sgy),
        "input_trace_mapping_npz": str(trace_mapping_npz),
        "output_density_sgy": str(output_density_sgy),
        "output_prior_sgy": str(output_prior_sgy),
        "output_trace_mapping_npz": str(output_mapping_npz),
        "trace_count": int(density.shape[0]),
        "sample_axis": {"time_min_ms": float(samples.min()), "time_max_ms": float(samples.max()), "sample_count": int(len(samples))},
        "attribute_load": {
            "Coherence": coherence_load_summary,
            "AntTrack": anttrack_load_summary,
            "CurvatureMax": curvature_load_summary,
        },
        "score_config": score_cfg,
        "prior_weights": {"anttrack_high": w_ant, "steep_low_coherence": w_coh, "curvature_max_high": w_curv, "normalized_sum": weight_sum},
        "density_update": {
            "gain": gain,
            "additive_alpha": additive_alpha,
            "additive_density_scale_quantile": additive_q,
            "additive_power": additive_power,
            "density_scale": density_scale,
            "density_cap": density_cap,
            "preserve_zero_density": preserve_zero_density,
        },
        "scores": {
            "AntTrackHighScore": ant_summary,
            "LowCoherenceScore": lowcoh_summary,
            "SteepLowCoherenceScore": steep_summary,
            "CurvatureMaxHighScore": curv_summary,
            "SeismicPrior": {"stats": finite_stats(prior[np.isfinite(prior)]), "q80": prior_q80, "q90": prior_q90},
        },
        "attribute_thresholds": {
            "anttrack_q80": ant_q80,
            "anttrack_q90": ant_q90,
            "coherence_q20": coh_q20,
            "curvaturemax_q80": curv_q80,
            "curvaturemax_q90": curv_q90,
            "prior_q80": prior_q80,
            "prior_q90": prior_q90,
        },
        "input_density_stats": finite_stats(density[np.isfinite(density)]),
        "output_density_stats": finite_stats(output_density[np.isfinite(output_density)]),
        "density_delta_stats": finite_stats((output_density - density)[np.isfinite(output_density) & np.isfinite(density)]),
        "top_density_alignment": {
            "input_density_top1pct": top_density_alignment(density, masks, 0.99),
            "output_density_top1pct": top_density_alignment(output_density, masks, 0.99),
        },
        "checks": {
            "output_density_sgy_exists": output_density_sgy.exists(),
            "output_prior_sgy_exists": output_prior_sgy.exists(),
            "output_mapping_exists": output_mapping_npz.exists(),
            "trace_count_preserved": int(density.shape[0]) == len(source_trace_idx),
            "sample_count_preserved": int(output_density.shape[1]) == int(len(samples)),
            "prior_range_valid": float(np.nanmin(prior)) >= 0.0 and float(np.nanmax(prior)) <= 1.0,
        },
    }
    summary["status"] = "pass" if all(bool(v) for v in summary["checks"].values()) else "fail"
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step6b-seismic-prior] output_density_sgy={output_density_sgy}", flush=True)
    print(f"[step6b-seismic-prior] output_prior_sgy={output_prior_sgy}", flush=True)
    print(f"[step6b-seismic-prior] summary={output_summary_json}", flush=True)
    print(f"[step6b-seismic-prior] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

