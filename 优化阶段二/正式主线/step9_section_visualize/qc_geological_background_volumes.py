# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import segyio

from well_curved_section_common import prepare_geometry, read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QC Step9 geological-background SGY inputs.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def volume_qc(name: str, path: Path, trace_ids: np.ndarray, time_min: float, time_max: float) -> dict[str, object]:
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        tracecount = int(handle.tracecount)
        valid_trace_ids = trace_ids[(trace_ids >= 0) & (trace_ids < tracecount)]
        if valid_trace_ids.size:
            pick = valid_trace_ids[np.linspace(0, len(valid_trace_ids) - 1, min(128, len(valid_trace_ids)), dtype=int)]
            values = np.concatenate([np.asarray(handle.trace[int(idx)], dtype=np.float32) for idx in pick]).astype(np.float64)
            values[values <= -1.0e6] = np.nan
            finite = values[np.isfinite(values)]
        else:
            finite = np.asarray([], dtype=np.float64)
    return {
        "path": str(path),
        "trace_count": tracecount,
        "sample_count": int(len(samples)),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "sample_interval_ms": float(np.median(np.diff(samples))) if len(samples) > 1 else None,
        "max_requested_trace_idx": int(trace_ids.max()),
        "requested_trace_indices_valid": bool(len(valid_trace_ids) == len(trace_ids)),
        "covers_section_time_range": bool(float(samples[0]) <= time_min and float(samples[-1]) >= time_max),
        "sampled_finite_count": int(finite.size),
        "sampled_quantiles": {
            "q01": float(np.quantile(finite, 0.01)) if finite.size else None,
            "q50": float(np.quantile(finite, 0.50)) if finite.size else None,
            "q99": float(np.quantile(finite, 0.99)) if finite.size else None,
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    geometry = prepare_geometry(config)
    volume_paths = {name: Path(str(path)).resolve() for name, path in dict(config["volume_paths"]).items()}
    required = ["AntTrack", "CurvatureMax", "SeisAmp"]
    missing = [name for name in required if name not in volume_paths or not volume_paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"missing required background volumes: {missing}")
    volumes = {
        name: volume_qc(name, volume_paths[name], geometry.trace_ids, geometry.time_min, geometry.time_max)
        for name in required
    }
    checks = {
        "well_inside_target_block": int(geometry.summary["selected_well_inside_target_rows"]) > 0,
        "surface_curves_include_t4_t5_t6_t7": geometry.summary["surface_codes_drawn"] == ["T4", "T5", "T6", "T7"],
        "all_trace_indices_valid": all(bool(item["requested_trace_indices_valid"]) for item in volumes.values()),
        "all_volumes_cover_section_time_range": all(bool(item["covers_section_time_range"]) for item in volumes.values()),
        "all_sampled_values_nonempty": all(int(item["sampled_finite_count"]) > 0 for item in volumes.values()),
    }
    output_root = Path(str(config["output_root"])).resolve()
    output_path = output_root / "qc" / "geological_background_volume_qc.json"
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(args.config.resolve()),
        "section_geometry": geometry.summary,
        "volumes": volumes,
        "checks": checks,
    }
    write_json(output_path, payload)
    print(f"[background-qc] output={output_path}", flush=True)
    print(f"[background-qc] status={payload['status']}", flush=True)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
