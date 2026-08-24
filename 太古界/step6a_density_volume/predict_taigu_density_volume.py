#!/usr/bin/env python3
"""Step6A (太古界): predict the demo-area 2 ms fracture-density volume (SGY).

For every demo trace and every 2 ms sample:
  * upper window (Top..Mid): 上部复合层 model
  * crust window (Mid..Base): 太古界风化壳 model
  * ext50 window (Base..Base+50): 太古界风化壳 model extrapolation, marked
    WindowCode=2 (ext50) and reported separately in QC (low confidence).
  * traces with invalid surfaces -> all NaN.

Outputs (config.output_dir):
  predicted_fracture_density.sgy   160,801 traces x ~600 samples (2 ms)
  window_code.npz                  per-sample window code (0 none / 1 main / 2 ext50)
  prediction_block_qc.csv
  prediction_summary.json          status=pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import segyio


CURRENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = CURRENT_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(COMMON_DIR / "seismic_sampling") not in sys.path:
    sys.path.insert(0, str(COMMON_DIR / "seismic_sampling"))

from amplitude_sampling import ObnAmplitudeSampler  # noqa: E402


FEATURE_COLUMNS = [
    "SeisAmp",
    "SeisAmpM4",
    "SeisAmpM2",
    "SeisAmpP2",
    "SeisAmpP4",
    "AmpMad5",
    "LayerCode",
    "RelativeTimeInLayer",
    "TimeSinceTop",
    "TimeToBase",
    "LayerThickness",
    "TimeMs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict 2 ms fracture-density SGY for the demo area (太古界).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-x-lines", type=int, default=0, help="Smoke-test cap.")
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        self.count += int(finite.size)
        self.total += float(finite.sum())
        self.total_sq += float(np.square(finite).sum())
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": int(self.count),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "mean": float(mean),
            "std": float(variance**0.5),
        }


def build_layer_features(
    amplitude: np.ndarray,
    samples: np.ndarray,
    top: np.ndarray,
    base: np.ndarray,
    valid: np.ndarray,
    layer_code: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = samples[None, :]
    mask = valid[:, None] & (times >= top[:, None]) & (times <= base[:, None]) & np.isfinite(amplitude)
    row_idx, time_idx = np.where(mask)
    if row_idx.size == 0:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), row_idx, time_idx
    local_time = samples[time_idx].astype(np.float32)
    local_top = top[row_idx].astype(np.float32)
    local_base = base[row_idx].astype(np.float32)
    thickness = local_base - local_top
    since_top = local_time - local_top
    to_base = local_base - local_time
    center_amp = amplitude[row_idx, time_idx]
    amp_m4 = np.full(len(row_idx), np.nan, dtype=np.float32)
    amp_m2 = np.full(len(row_idx), np.nan, dtype=np.float32)
    amp_p2 = np.full(len(row_idx), np.nan, dtype=np.float32)
    amp_p4 = np.full(len(row_idx), np.nan, dtype=np.float32)
    valid_t = time_idx >= 2
    amp_m4[valid_t] = amplitude[row_idx[valid_t], time_idx[valid_t] - 2]
    amp_m2[valid_t] = amplitude[row_idx[valid_t], time_idx[valid_t] - 1]
    valid_p = time_idx < len(samples) - 2
    amp_p2[valid_p] = amplitude[row_idx[valid_p], time_idx[valid_p] + 1]
    amp_p4[valid_p] = amplitude[row_idx[valid_p], time_idx[valid_p] + 2]
    context = np.column_stack([amp_m4, amp_m2, center_amp, amp_p2, amp_p4])
    amp_mad5 = np.nanmedian(np.abs(context - center_amp[:, None]), axis=1).astype(np.float32)
    features = np.column_stack(
        [
            center_amp,
            amp_m4,
            amp_m2,
            amp_p2,
            amp_p4,
            amp_mad5,
            np.full(row_idx.size, layer_code, dtype=np.float32),
            since_top / thickness,
            since_top,
            to_base,
            thickness,
            local_time,
        ]
    ).astype(np.float32)
    return features, row_idx, time_idx


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_sgy = output_dir / "predicted_fracture_density.sgy"
    partial_sgy = output_dir / "predicted_fracture_density.sgy.partial"
    window_npz = output_dir / "window_code.npz"
    block_qc_path = output_dir / "prediction_block_qc.csv"
    summary_path = output_dir / "prediction_summary.json"
    for path in (output_sgy, partial_sgy, window_npz, block_qc_path, summary_path):
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing prediction output: {path}")

    model_path = Path(config["model_joblib"]).resolve()
    artifact = joblib.load(model_path)
    if tuple(artifact["feature_columns"]) != tuple(FEATURE_COLUMNS):
        raise ValueError(f"model feature contract mismatch: {artifact['feature_columns']}")
    interval_ms = float(config.get("sample_interval_ms", 2.0))
    ext_window_ms = float(config.get("ext_window_ms", 50.0))
    density_cap = float(config.get("density_cap", 15.0))

    grid = pd.read_csv(config["demo_grid_csv"], encoding="utf-8-sig")
    horizon = pd.read_csv(config["horizon_contract_csv"], encoding="utf-8-sig")
    for df in (grid, horizon):
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grid = grid.merge(
        horizon[["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"]],
        on="TraceIdx",
        how="left",
    )
    grid = grid.sort_values("TraceIdx").reset_index(drop=True)
    if len(grid) != int(grid["TraceIdx"].nunique()):
        raise RuntimeError("demo grid has duplicate TraceIdx")

    valid_mask = grid["SurfaceValid"].fillna(0).astype(bool).to_numpy()
    valid_top = grid.loc[valid_mask, "TopTimeMs"].to_numpy()
    valid_base = grid.loc[valid_mask, "BaseTimeMs"].to_numpy()
    axis_start = float(np.floor(valid_top.min() / interval_ms) * interval_ms)
    axis_stop = float(np.ceil((valid_base + ext_window_ms).max() / interval_ms) * interval_ms)
    sample_axis = np.arange(axis_start, axis_stop + interval_ms * 0.5, interval_ms, dtype=np.float64)

    top = grid["TopTimeMs"].to_numpy(dtype=np.float64)
    mid = grid["MidTimeMs"].to_numpy(dtype=np.float64)
    base = grid["BaseTimeMs"].to_numpy(dtype=np.float64)
    x_line_count = int(grid["IX"].max()) + 1
    block_x_lines = max(int(config.get("block_x_line_count", 20)), 1)
    model_upper = artifact["models"]["上部复合层"]
    model_crust = artifact["models"]["太古界风化壳"]
    cap = min(float(density_cap), float(model_upper["density_cap"]), float(model_crust["density_cap"]))

    block_rows: list[dict[str, Any]] = []
    stats_all = RunningStats()
    stats_main = RunningStats()
    stats_ext = RunningStats()
    layer_total = {"上部复合层": 0, "太古界风化壳": 0}
    started = time.time()

    with ObnAmplitudeSampler(Path(config["obn_segy"]), Path(config["trace_header_csv"])) as sampler:
        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
        spec.format = 5
        spec.samples = sample_axis.astype(np.float32)
        spec.tracecount = int(len(grid))
        delay_ms = int(round(float(sample_axis[0])))
        interval_us = int(round(interval_ms * 1000.0))
        source_handle = sampler.handle
        with segyio.create(str(partial_sgy), spec) as output_handle:
            output_handle.text[0] = source_handle.text[0]
            output_handle.bin.update(source_handle.bin)
            output_handle.bin[segyio.BinField.Interval] = interval_us
            output_handle.bin[segyio.BinField.Samples] = int(len(sample_axis))
            output_handle.bin[segyio.BinField.Format] = 5
            window_codes = np.zeros((len(grid), len(sample_axis)), dtype=np.uint8)
            output_cursor = 0
            for block_index, ix_start in enumerate(range(0, x_line_count, block_x_lines), start=1):
                ix_stop = min(ix_start + block_x_lines, x_line_count)
                block = grid[grid["IX"].between(ix_start, ix_stop - 1)].sort_values("TraceIdx").copy()
                source_indices = block["TraceIdx"].to_numpy(dtype=np.int64)
                amplitude = sampler.read_2ms_block(source_indices, sample_axis)
                predicted = np.full((len(block), len(sample_axis)), np.nan, dtype=np.float32)
                codes = np.zeros((len(block), len(sample_axis)), dtype=np.uint8)
                block_valid = block["SurfaceValid"].fillna(0).astype(bool).to_numpy()

                features, row_idx, time_idx = build_layer_features(
                    amplitude, sample_axis, block["TopTimeMs"].to_numpy(), block["MidTimeMs"].to_numpy(), block_valid, 1.0
                )
                if len(row_idx):
                    prob = model_upper["classifier"].predict_proba(features)[:, 1]
                    cond = model_upper["conditional_density_regressor"].predict(features)
                    predicted[row_idx, time_idx] = np.clip(prob * np.clip(cond, 0.0, cap), 0.0, cap)
                    codes[row_idx, time_idx] = 1
                    layer_total["上部复合层"] += int(len(row_idx))

                features, row_idx, time_idx = build_layer_features(
                    amplitude, sample_axis, block["MidTimeMs"].to_numpy(), block["BaseTimeMs"].to_numpy(), block_valid, 2.0
                )
                if len(row_idx):
                    prob = model_crust["classifier"].predict_proba(features)[:, 1]
                    cond = model_crust["conditional_density_regressor"].predict(features)
                    predicted[row_idx, time_idx] = np.clip(prob * np.clip(cond, 0.0, cap), 0.0, cap)
                    codes[row_idx, time_idx] = 1
                    layer_total["太古界风化壳"] += int(len(row_idx))

                features, row_idx, time_idx = build_layer_features(
                    amplitude,
                    sample_axis,
                    block["BaseTimeMs"].to_numpy(),
                    block["BaseTimeMs"].to_numpy() + ext_window_ms,
                    block_valid,
                    2.0,
                )
                if len(row_idx):
                    prob = model_crust["classifier"].predict_proba(features)[:, 1]
                    cond = model_crust["conditional_density_regressor"].predict(features)
                    predicted[row_idx, time_idx] = np.clip(prob * np.clip(cond, 0.0, cap), 0.0, cap)
                    codes[row_idx, time_idx] = 2

                stats_all.update(predicted)
                stats_main.update(np.where(codes == 1, predicted, np.nan))
                stats_ext.update(np.where(codes == 2, predicted, np.nan))
                for local_index, (_, row) in enumerate(block.iterrows()):
                    source_index = int(row["TraceIdx"])
                    header = dict(source_handle.header[source_index])
                    header[segyio.TraceField.TRACE_SEQUENCE_FILE] = output_cursor + 1
                    header[segyio.TraceField.TRACE_SEQUENCE_LINE] = output_cursor + 1
                    header[segyio.TraceField.TRACE_SAMPLE_COUNT] = int(len(sample_axis))
                    header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = interval_us
                    header[segyio.TraceField.DelayRecordingTime] = delay_ms
                    header[segyio.TraceField.SourceX] = int(round(float(row["X"])))
                    header[segyio.TraceField.SourceY] = int(round(float(row["Y"])))
                    output_handle.header[output_cursor] = header
                    output_handle.trace[output_cursor] = predicted[local_index]
                    window_codes[output_cursor] = codes[local_index]
                    output_cursor += 1
                finite_count = int(np.isfinite(predicted).sum())
                block_rows.append(
                    {
                        "BlockIndex": block_index,
                        "IXStart": ix_start,
                        "IXStopExclusive": ix_stop,
                        "TraceCount": int(len(block)),
                        "SurfaceValidTraceCount": int(block_valid.sum()),
                        "FinitePredictionCount": finite_count,
                        "NaNCount": int(predicted.size - finite_count),
                        "UpperPredictionCount": int(layer_total["上部复合层"]),
                        "CrustPredictionCount": int(layer_total["太古界风化壳"]),
                        "Ext50PredictionCount": int((codes == 2).sum()),
                        "PredictionMean": float(np.nanmean(predicted)) if finite_count else np.nan,
                        "ElapsedSeconds": 0.0,
                    }
                )
                print(
                    f"[step6a-predict] block={block_index} ix={ix_start}:{ix_stop} "
                    f"traces={output_cursor}/{len(grid)} finite={finite_count}",
                    flush=True,
                )
                if args.max_x_lines > 0 and ix_stop >= args.max_x_lines:
                    grid = grid[grid["IX"] < ix_stop].copy()
                    window_codes = window_codes[: output_cursor]
                    break
        partial_sgy.replace(output_sgy)
        np.savez_compressed(window_npz, window_code=window_codes, sample_axis=sample_axis.astype(np.float32))

    with segyio.open(str(output_sgy), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        sampled = np.stack([np.asarray(handle.trace[int(i)], dtype=np.float32) for i in np.unique(np.linspace(0, len(grid) - 1, min(32, len(grid)), dtype=np.int64))])
    checks = {
        "full_grid_is_rectangular": len(grid) == (int(grid["IX"].max()) + 1) * (int(grid["IY"].max()) + 1),
        "sample_interval_is_2ms": abs(interval_ms - 2.0) < 1.0e-9,
        "ext_window_is_50ms": abs(ext_window_ms - 50.0) < 1.0e-9,
        "upper_and_crust_predictions_exist": all(count > 0 for count in layer_total.values()),
        "ext50_predictions_exist": int((window_codes == 2).sum()) > 0,
        "output_sgy_exists": output_sgy.exists(),
        "window_code_npz_exists": window_npz.exists(),
        "sgy_trace_count_matches": int(handle.tracecount) == int(len(grid)),
        "sgy_sample_axis_matches": len(samples) == len(sample_axis) and bool(np.allclose(samples, sample_axis, atol=1.0e-6)),
        "sampled_output_contains_finite": bool(np.isfinite(sampled).any()),
        "sampled_output_contains_nan": bool(np.isnan(sampled).any()),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "model_joblib": str(model_path),
        "model_contract_version": artifact.get("contract_version"),
        "model_logic": artifact.get("model_logic"),
        "output_paths": {
            "density_sgy": str(output_sgy),
            "window_code_npz": str(window_npz),
            "prediction_block_qc_csv": str(block_qc_path),
            "prediction_summary_json": str(summary_path),
        },
        "grid": {
            "trace_count": int(len(grid)),
            "x_line_count": int(grid["IX"].max()) + 1,
            "y_line_count": int(grid["IY"].max()) + 1,
            "surface_valid_trace_count": int(valid_mask.sum()),
            "surface_valid_fraction": float(valid_mask.mean()),
        },
        "sample_axis": {
            "time_min_ms": float(sample_axis[0]),
            "time_max_ms": float(sample_axis[-1]),
            "sample_interval_ms": interval_ms,
            "sample_count": int(len(sample_axis)),
        },
        "ext_window_ms": ext_window_ms,
        "prediction_counts_by_layer": layer_total,
        "ext50_sample_count": int((window_codes == 2).sum()),
        "density_stats": {
            "all": stats_all.summary(),
            "main_window": stats_main.summary(),
            "ext50_window": stats_ext.summary(),
        },
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame(block_rows).to_csv(block_qc_path, index=False, encoding="utf-8-sig")
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
