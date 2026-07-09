from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_3d_density_lowcoh_postprocess.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process Step6B 3D density SGY with low-coherence guidance.")
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


def load_trace_mapping(path: Path) -> dict[str, np.ndarray]:
    mapping = np.load(path)
    required = {"source_trace_idx", "x", "y", "ix", "iy", "output_trace_index"}
    missing = sorted(required.difference(mapping.files))
    if missing:
        raise ValueError(f"trace mapping missing keys: {missing}")
    return {key: mapping[key] for key in mapping.files}


def low_coherence_weight(coherence: np.ndarray, valid_mask: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = valid_mask & np.isfinite(coherence)
    if "valid_min" in config:
        valid &= coherence >= float(config["valid_min"])
    if "valid_max" in config:
        valid &= coherence <= float(config["valid_max"])

    finite = coherence[valid]
    if finite.size == 0:
        score = np.zeros_like(coherence, dtype=np.float32)
        weight = np.ones_like(coherence, dtype=np.float32)
        return score, weight, {"enabled": True, "valid_count": 0}

    low_q = float(config.get("low_quantile", 0.05))
    high_q = float(config.get("high_quantile", 0.95))
    low_bound = float(np.quantile(finite, low_q))
    high_bound = float(np.quantile(finite, high_q))
    if high_bound <= low_bound:
        high_bound = low_bound + 1.0e-6

    score = np.clip((high_bound - coherence) / (high_bound - low_bound), 0.0, 1.0).astype(np.float32)
    score[~valid] = 0.0
    gain = float(config.get("gain", 1.5))
    power = float(config.get("power", 1.2))
    max_weight = float(config.get("max_weight", 3.0))
    weight = 1.0 + gain * np.power(score, power)
    weight = np.clip(weight, 1.0, max_weight).astype(np.float32)
    weight[~valid] = 1.0

    summary = {
        "enabled": True,
        "valid_count": int(finite.size),
        "invalid_or_missing_count": int(valid_mask.sum() - valid.sum()),
        "low_quantile": low_q,
        "high_quantile": high_q,
        "low_bound": low_bound,
        "high_bound": high_bound,
        "gain": gain,
        "power": power,
        "max_weight": max_weight,
        "coherence_stats": finite_stats(finite),
        "low_coherence_score_stats": finite_stats(score[valid]),
        "low_coherence_weight_stats": finite_stats(weight[valid]),
    }
    return score, weight, summary


def additive_density_compensation(
    density: np.ndarray,
    low_score: np.ndarray,
    valid_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    alpha = float(config.get("additive_alpha", 0.0))
    if alpha <= 0.0:
        zeros = np.zeros_like(density, dtype=np.float32)
        return zeros, {"enabled": False, "additive_alpha": alpha}

    finite_density = density[valid_mask & np.isfinite(density) & (density > 0.0)]
    if finite_density.size == 0:
        zeros = np.zeros_like(density, dtype=np.float32)
        return zeros, {"enabled": True, "additive_alpha": alpha, "finite_density_count": 0}

    q = float(config.get("additive_density_scale_quantile", 0.95))
    density_scale = float(np.quantile(finite_density, q))
    density_scale = max(density_scale, float(config.get("min_additive_density_scale", 1.0e-6)))
    power = float(config.get("additive_power", 1.5))
    max_additive = float(config.get("max_additive_density", alpha * density_scale))

    additive = alpha * density_scale * np.power(np.clip(low_score, 0.0, 1.0), power)
    additive = np.clip(additive, 0.0, max_additive).astype(np.float32)
    additive[~valid_mask] = 0.0
    additive[~np.isfinite(additive)] = 0.0
    return additive, {
        "enabled": True,
        "additive_alpha": alpha,
        "additive_density_scale_quantile": q,
        "density_scale": density_scale,
        "additive_power": power,
        "max_additive_density": max_additive,
        "finite_density_count": int(finite_density.size),
        "additive_density_stats": finite_stats(additive[valid_mask]),
    }


def copy_mapping(src: Path, dst: Path) -> None:
    mapping = np.load(src)
    np.savez_compressed(dst, **{key: mapping[key] for key in mapping.files})


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    coherence_sgy = Path(config["coherence_sgy"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    output_density_sgy = output_dir / str(config.get("output_density_sgy_name", "candidate_cheye1_3d_predicted_density_lowcoh.sgy"))
    output_mapping_npz = output_dir / str(config.get("output_trace_mapping_name", "candidate_cheye1_3d_trace_mapping.npz"))
    output_summary_json = output_dir / str(config.get("output_summary_name", "candidate_cheye1_3d_density_lowcoh_postprocess_summary.json"))

    for label, path in [("input_density_sgy", input_density_sgy), ("trace_mapping_npz", trace_mapping_npz), ("coherence_sgy", coherence_sgy)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    mapping = load_trace_mapping(trace_mapping_npz)
    source_trace_idx = mapping["source_trace_idx"].astype(np.int64)
    density_cap = float(config.get("density_cap", 10.0))
    preserve_zero_density = bool(config.get("preserve_zero_density", True))

    stats_parts: dict[str, list[np.ndarray]] = {
        "input_density": [],
        "output_density": [],
        "coherence": [],
        "low_score": [],
        "weight": [],
    }

    with segyio.open(str(input_density_sgy), "r", ignore_geometry=True) as src_density:
        samples = np.asarray(src_density.samples, dtype=np.float64)
        tracecount = int(src_density.tracecount)
        if tracecount != len(source_trace_idx):
            raise ValueError(f"density tracecount {tracecount} != mapping rows {len(source_trace_idx)}")

        with segyio.open(str(coherence_sgy), "r", ignore_geometry=True) as src_coh:
            coh_samples = np.asarray(src_coh.samples, dtype=np.float64)
            same_samples = len(coh_samples) == len(samples) and np.allclose(coh_samples, samples, rtol=0.0, atol=1.0e-6)
            max_trace_idx = int(np.max(source_trace_idx)) if len(source_trace_idx) else -1
            if max_trace_idx >= int(src_coh.tracecount):
                raise ValueError(f"coherence tracecount {src_coh.tracecount} is smaller than mapping max source trace {max_trace_idx}")

            density_matrix = np.stack([src_density.trace[idx] for idx in range(tracecount)]).astype(np.float32)
            coherence_matrix = np.empty_like(density_matrix, dtype=np.float32)
            for out_idx, src_idx in enumerate(source_trace_idx):
                coh_trace = np.asarray(src_coh.trace[int(src_idx)], dtype=np.float32)
                if same_samples:
                    coherence_matrix[out_idx, :] = coh_trace
                else:
                    coherence_matrix[out_idx, :] = np.interp(samples, coh_samples, coh_trace, left=np.nan, right=np.nan).astype(np.float32)
                if (out_idx + 1) % 10000 == 0:
                    print(f"[step6b-lowcoh] loaded coherence traces={out_idx + 1}/{tracecount}", flush=True)

        valid_density = np.isfinite(density_matrix) & (density_matrix > 0.0 if preserve_zero_density else density_matrix >= 0.0)
        guidance_config = dict(config.get("coherence_guidance", {}))
        mode = str(config.get("mode", guidance_config.get("mode", "multiply")))
        score, weight, guidance_summary = low_coherence_weight(coherence_matrix, valid_density, guidance_config)
        multiplicative_density = np.asarray(density_matrix, dtype=np.float32) * weight
        if mode == "multiply_plus_add":
            additive_density, additive_summary = additive_density_compensation(
                density=density_matrix,
                low_score=score,
                valid_mask=np.isfinite(coherence_matrix) & (coherence_matrix >= float(guidance_config.get("valid_min", 0.0))),
                config=guidance_config,
            )
            output_density = multiplicative_density + additive_density
        elif mode == "multiply":
            additive_density = np.zeros_like(density_matrix, dtype=np.float32)
            additive_summary = {"enabled": False, "mode": mode}
            output_density = multiplicative_density
        else:
            raise ValueError(f"unsupported postprocess mode: {mode}")
        output_density[~np.isfinite(output_density)] = 0.0
        output_density = np.clip(output_density, 0.0, density_cap).astype(np.float32)
        if preserve_zero_density:
            output_density[(density_matrix <= 0.0) & (additive_density <= 0.0)] = 0.0

        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
        spec.format = int(src_density.bin[segyio.BinField.Format]) or 5
        spec.samples = np.asarray(src_density.samples, dtype=np.float32)
        spec.tracecount = tracecount
        with segyio.create(str(output_density_sgy), spec) as dst:
            dst.text[0] = src_density.text[0]
            dst.bin.update(src_density.bin)
            dst.bin[segyio.BinField.Samples] = int(len(samples))
            dst.bin[segyio.BinField.Format] = 5
            for trace_idx in range(tracecount):
                dst.header[trace_idx] = dict(src_density.header[trace_idx])
                dst.trace[trace_idx] = output_density[trace_idx]
                if (trace_idx + 1) % 10000 == 0:
                    print(f"[step6b-lowcoh] wrote traces={trace_idx + 1}/{tracecount}", flush=True)

    copy_mapping(trace_mapping_npz, output_mapping_npz)
    valid_coh = np.isfinite(coherence_matrix) & (coherence_matrix >= float(config.get("coherence_guidance", {}).get("valid_min", 0.0)))
    stats_parts["input_density"].append(density_matrix[np.isfinite(density_matrix)])
    stats_parts["output_density"].append(output_density[np.isfinite(output_density)])
    stats_parts["coherence"].append(coherence_matrix[valid_coh])
    stats_parts["low_score"].append(score[valid_coh])
    stats_parts["weight"].append(weight[valid_coh])
    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "input_density_sgy": str(input_density_sgy),
        "input_trace_mapping_npz": str(trace_mapping_npz),
        "coherence_sgy": str(coherence_sgy),
        "output_density_sgy": str(output_density_sgy),
        "output_trace_mapping_npz": str(output_mapping_npz),
        "density_cap": density_cap,
        "preserve_zero_density": preserve_zero_density,
        "mode": mode,
        "sample_axis": {"time_min_ms": float(samples.min()), "time_max_ms": float(samples.max()), "sample_count": int(len(samples))},
        "trace_count": int(tracecount),
        "coherence_sample_axis_matched": bool(same_samples),
        "guidance": guidance_summary,
        "additive_compensation": additive_summary,
        "input_density_stats": finite_stats(np.concatenate(stats_parts["input_density"])),
        "output_density_stats": finite_stats(np.concatenate(stats_parts["output_density"])),
        "additive_density_stats": finite_stats(additive_density[np.isfinite(additive_density)]),
        "density_delta_stats": finite_stats((output_density - density_matrix)[np.isfinite(output_density) & np.isfinite(density_matrix)]),
        "checks": {
            "output_density_sgy_exists": output_density_sgy.exists(),
            "output_mapping_exists": output_mapping_npz.exists(),
            "trace_count_preserved": int(tracecount) == len(source_trace_idx),
            "sample_count_preserved": int(output_density.shape[1]) == int(len(samples)),
        },
    }
    summary["status"] = "pass" if all(summary["checks"].values()) else "fail"
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step6b-lowcoh] output_sgy={output_density_sgy}", flush=True)
    print(f"[step6b-lowcoh] summary={output_summary_json}", flush=True)
    print(f"[step6b-lowcoh] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
