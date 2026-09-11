#!/usr/bin/env python3
"""Lightweight QC for the Taigu attribute volumes."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import segyio

def read_xy(header):
    scalar = float(header[segyio.TraceField.SourceGroupScalar])
    factor = 1.0 if scalar == 0 else (scalar if scalar > 0 else 1.0 / abs(scalar))
    return (float(header[segyio.TraceField.SourceX]) * factor,
            float(header[segyio.TraceField.SourceY]) * factor)

def inspect(name, path, invalid_le, time_origin_ms, probe_count):
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n = len(f.trace); samples = np.asarray(f.samples, dtype=float)
        if n == 0 or samples.size < 2: raise ValueError(f"{name}: empty SEG-Y")
        indices = np.unique(np.linspace(0, n - 1, min(probe_count, n), dtype=int))
        xy = np.asarray([read_xy(f.header[int(i)]) for i in indices], dtype=float)
        first_values, invalid_counts, minus_one_counts = [], [], []
        for i in indices:
            trace = np.asarray(f.trace[int(i)], dtype=np.float32)
            finite = np.isfinite(trace)
            first_values.append(float(trace[finite][0]) if finite.any() else None)
            bad = ~finite
            if invalid_le is not None: bad |= trace <= invalid_le
            invalid_counts.append(int(bad.sum()))
            minus_one_counts.append(int(np.count_nonzero(np.isclose(trace, -1.0, atol=1e-6))))
        dt = float(np.median(np.diff(samples)))
        effective_samples = samples - samples[0] + float(time_origin_ms)
        return {"path": str(path.resolve()), "trace_count": int(n),
                "raw_time_min_ms": float(samples[0]), "raw_time_max_ms": float(samples[-1]),
                "time_min_ms": float(effective_samples[0]), "time_max_ms": float(effective_samples[-1]),
                "sample_interval_ms": dt, "regular_time_axis": bool(np.allclose(np.diff(samples), dt, atol=1e-6)),
                "probe_indices": [int(i) for i in indices], "probe_xy": xy.tolist(),
                "probe_first_finite_value": first_values, "probe_invalid_sample_counts": invalid_counts,
                "probe_minus_one_sample_counts": minus_one_counts,
                "invalid_le": invalid_le, "configured_time_origin_ms": time_origin_ms}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "taigu_attribute_sampling_v1.json"); ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--probe-count", type=int, default=5)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text(encoding="utf-8")); outdir = args.output_dir.resolve(); outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "attribute_sampling_smoke_summary.json"
    if summary_path.exists(): raise FileExistsError(f"refusing to overwrite existing summary: {summary_path}")
    started = time.time(); results = {}
    for name, raw_path in cfg["volume_paths"].items():
        path = Path(raw_path)
        if not path.exists(): raise FileNotFoundError(path)
        origin = cfg.get("time_origins_ms", {}).get(name)
        if origin is None: raise ValueError(f"{name}: missing configured absolute time origin")
        results[name] = inspect(name, path, cfg.get("invalid_rules", {}).get(name, {}).get("invalid_le"), origin, args.probe_count)
    names = list(results); counts_ok = len({results[n]["trace_count"] for n in names}) == 1; axes_ok = all(results[n]["regular_time_axis"] and results[n]["sample_interval_ms"] > 0 for n in names); time_ok = all(1799.0 <= results[n]["time_min_ms"] <= 1801.0 for n in names)
    ref = np.asarray(results[names[0]]["probe_xy"]); xy_ok = all(np.allclose(ref, np.asarray(results[n]["probe_xy"]), rtol=0.0, atol=1.1, equal_nan=True) for n in names[1:])
    ant_minus_one_seen = any(v > 0 for v in results.get("AntTrack", {}).get("probe_minus_one_sample_counts", []))
    checks = {"all_volumes_present": True, "trace_counts_match": counts_ok, "regular_positive_time_axes": axes_ok, "time_origin_about_1800ms": time_ok, "probe_coordinates_match": xy_ok, "anttrack_minus_one_observed_as_raw_value": ant_minus_one_seen}
    summary = {"status": "pass" if all(checks.values()) else "fail", "config": str(args.config.resolve()), "volumes": results, "checks": checks, "probe_count": args.probe_count, "elapsed_seconds": time.time() - started, "note": "Header/sample spot check only; not a full-volume coverage test."}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if summary["status"] == "pass" else 1

if __name__ == "__main__": raise SystemExit(main())
