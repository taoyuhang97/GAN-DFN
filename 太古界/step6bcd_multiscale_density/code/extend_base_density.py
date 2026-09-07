# -*- coding: utf-8 -*-
"""Append the T7-below extension band to the existing Step6 base density.

The existing mine base density was predicted only inside the original per-trace
T4-T7 windows. This script re-runs the SAME two-stage model on the extension
band (T7 .. T7'), where T7' is the per-trace attribute-data bottom from
scan_attribute_bottom.py, treating the band as the lower part of the extended
Shasi window ([T6, T7']). The original in-window density values are copied
unchanged; only the band is newly predicted.

All extension voxels are evaluated in a single batched model call (vectorized
feature assembly), so the same script is fast enough for the full mine.
Region-agnostic: only target_block / output paths change in the config.
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
FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from predict_step6_two_stage_density_volume import (  # noqa: E402
    ATTRIBUTE_COLUMNS,
    EXPECTED_FEATURE_COLUMNS,
    LAYER_CODE,
    NULL_THRESHOLD,
    read_trace_indices,
    resample_regular_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append the T7-below extension band to the base density.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_block_traces(config: dict[str, Any]) -> pd.DataFrame:
    header = pd.read_csv(config["trace_header_csv"], usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    block = config["target_block"]
    mask = (
        header["X"].between(float(block["x_min"]), float(block["x_max"]))
        & header["Y"].between(float(block["y_min"]), float(block["y_max"]))
    )
    selected = header.loc[mask, ["TraceIdx", "X", "Y"]].copy()
    selected = selected.sort_values("TraceIdx").reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("target_block contains no trace-header rows")
    return selected


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = Path(config["output_root"]).resolve() / "base_density_ext"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    selected = select_block_traces(config)
    trace_idx = selected["TraceIdx"].to_numpy(dtype=np.int64)
    n_traces = len(trace_idx)
    mapping = np.load(config["base_density_mapping"])
    map_lookup = {int(v): i for i, v in enumerate(np.asarray(mapping["source_trace_idx"], dtype=np.int64))}
    map_rows = np.asarray([map_lookup[int(ti)] for ti in trace_idx], dtype=np.int64)
    map_t6 = np.asarray(mapping["t6_time"], dtype=np.float64)[map_rows]
    map_t7 = np.asarray(mapping["t7_time"], dtype=np.float64)[map_rows]
    map_valid = np.asarray(mapping["surface_valid"], dtype=bool)[map_rows]

    bottom_npz = np.load(Path(config["output_root"]).resolve() / "attribute_bottom/attribute_bottom.npz")
    bottom_lookup = {int(ti): i for i, ti in enumerate(np.asarray(bottom_npz["TraceIdx"], dtype=np.int64))}
    bottom_rows = np.asarray([bottom_lookup[int(ti)] for ti in trace_idx], dtype=np.int64)
    t7_ext = np.asarray(bottom_npz["BottomMs"], dtype=np.float64)[bottom_rows]
    bottom_shasi = np.asarray(bottom_npz["ShasiPresent"], dtype=bool)[bottom_rows]
    bottom_t7_orig = np.asarray(bottom_npz["T7Original"], dtype=np.float64)[bottom_rows]

    interval_ms = float(config.get("sample_interval_ms", 2.0))
    density_cap = float(config.get("density_cap", 10.0))
    null_limit = float(config.get("attribute_null_abs_limit", 1.0e6))
    max_bottom_cap = float(config.get("extension_max_bottom_ms", 3674.0))

    artifact = joblib.load(config["model_joblib"])
    if tuple(artifact.get("feature_columns", ())) != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("model feature contract mismatch")
    bundle = artifact["models"]["沙四段"]

    with segyio.open(str(config["base_density_sgy"]), "r", ignore_geometry=True) as density_handle:
        density_samples = np.asarray(density_handle.samples, dtype=np.float64)
        density_matrix = read_trace_indices(density_handle, trace_idx)

    ext_start = np.maximum(map_t7, bottom_t7_orig)
    has_extension = bottom_shasi & np.isfinite(t7_ext) & np.isfinite(ext_start) & (t7_ext > ext_start + 0.5)
    t7_ext = np.minimum(np.where(has_extension, t7_ext, map_t7), max_bottom_cap)

    axis_stop = max(float(density_samples[-1]), float(np.ceil(np.nanmax(t7_ext) / interval_ms) * interval_ms))
    output_samples = np.arange(float(density_samples[0]), axis_stop + interval_ms * 0.5, interval_ms, dtype=np.float64)

    # Resample the four attribute volumes onto the output axis for the block traces.
    attributes: dict[str, np.ndarray] = {}
    with segyio.open(str(config["source_sgy_for_headers"]), "r", ignore_geometry=True) as source_handle:
        source_handle.mmap()
        for name in ATTRIBUTE_COLUMNS:
            path = Path(config["volume_paths"][name]).resolve()
            with segyio.open(str(path), "r", ignore_geometry=True) as volume_handle:
                matrix = read_trace_indices(volume_handle, trace_idx)
                volume_samples = np.asarray(volume_handle.samples, dtype=np.float64)
            matrix[~np.isfinite(matrix)] = np.nan
            matrix[matrix <= -null_limit] = np.nan
            attributes[name] = resample_regular_matrix(matrix, volume_samples, output_samples)
            print(f"[extend-density] loaded+resampled {name}: {attributes[name].shape}", flush=True)

    # Vectorized extension voxel assembly.
    cols_per_trace: list[np.ndarray] = []
    for i in range(n_traces):
        if not has_extension[i]:
            cols_per_trace.append(np.empty(0, dtype=np.int64))
            continue
        lo = int(np.searchsorted(output_samples, ext_start[i], side="right"))
        hi = int(np.searchsorted(output_samples, t7_ext[i], side="right"))
        cols_per_trace.append(np.arange(lo, hi, dtype=np.int64))
    col_offsets = np.cumsum([0] + [len(c) for c in cols_per_trace])
    total_ext = int(col_offsets[-1])
    rows = np.repeat(np.arange(n_traces, dtype=np.int64), np.diff(col_offsets))
    cols = np.concatenate(cols_per_trace) if total_ext else np.empty(0, dtype=np.int64)
    print(f"[extend-density] extension voxels={total_ext}", flush=True)

    extension_output = np.full(density_matrix.shape, np.nan, dtype=np.float32)
    if total_ext:
        times = output_samples[cols]
        t6_v = map_t6[rows]
        t7_v = t7_ext[rows]
        thickness = np.maximum(t7_v - t6_v, 1.0e-6)
        since_top = times - t6_v
        to_base = t7_v - times
        relative = since_top / thickness
        attr_matrix = np.column_stack([attributes[name][rows, cols] for name in ATTRIBUTE_COLUMNS])
        valid = np.isfinite(attr_matrix).all(axis=1)
        feature_count = int(valid.sum())
        features = np.column_stack(
            [
                *(attributes[name][rows, cols] for name in ATTRIBUTE_COLUMNS),
                np.full(total_ext, LAYER_CODE["沙四段"], dtype=np.float32),
                relative,
                since_top,
                to_base,
                thickness,
                times,
            ]
        ).astype(np.float32)
        features = features[valid]
        print(f"[extend-density] valid extension voxels={feature_count}/{total_ext}", flush=True)
        if feature_count:
            probability = bundle["classifier"].predict_proba(features)[:, 1]
            conditional = bundle["conditional_density_regressor"].predict(features)
            density = np.clip(probability * np.clip(conditional, 0.0, density_cap), 0.0, density_cap).astype(np.float32)
            valid_rows = rows[valid]
            valid_cols = cols[valid]
            extension_output[valid_rows, valid_cols] = density

    merged = density_matrix.copy()
    overwrite = np.isfinite(extension_output)
    merged[overwrite] = extension_output[overwrite]
    if len(output_samples) != density_matrix.shape[1]:
        new_matrix = np.full((n_traces, len(output_samples)), np.nan, dtype=np.float32)
        old_cols = np.searchsorted(output_samples, density_samples)
        new_matrix[:, old_cols] = density_matrix
        merged = new_matrix

    # Write extended SGY for the block traces.
    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
    spec.format = 5
    spec.samples = output_samples.astype(np.float32)
    spec.tracecount = n_traces
    delay_ms = int(round(float(output_samples[0])))
    interval_us = int(round(interval_ms * 1000.0))
    output_sgy = output_dir / "base_density_ext.sgy"
    with segyio.open(str(config["source_sgy_for_headers"]), "r", ignore_geometry=True) as source_handle:
        source_handle.mmap()
        with segyio.create(str(output_sgy), spec) as output_handle:
            output_handle.text[0] = source_handle.text[0]
            output_handle.bin.update(source_handle.bin)
            output_handle.bin[segyio.BinField.Interval] = interval_us
            output_handle.bin[segyio.BinField.Samples] = int(len(output_samples))
            output_handle.bin[segyio.BinField.Format] = 5
            for i, (_, row) in enumerate(selected.iterrows()):
                source_index = int(row["TraceIdx"])
                header = dict(source_handle.header[source_index])
                header[segyio.TraceField.TRACE_SEQUENCE_FILE] = i + 1
                header[segyio.TraceField.TRACE_SEQUENCE_LINE] = i + 1
                header[segyio.TraceField.TRACE_SAMPLE_COUNT] = int(len(output_samples))
                header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = interval_us
                header[segyio.TraceField.DelayRecordingTime] = delay_ms
                header[segyio.TraceField.SourceX] = int(round(float(row["X"])))
                header[segyio.TraceField.SourceY] = int(round(float(row["Y"])))
                output_handle.header[i] = header
                output_handle.trace[i] = merged[i]

    # Write the block trace mapping.
    x_values = np.sort(selected["X"].unique())
    y_values = np.sort(selected["Y"].unique())
    ix = np.searchsorted(x_values, selected["X"].to_numpy(dtype=np.float64)).astype(np.int32)
    iy = np.searchsorted(y_values, selected["Y"].to_numpy(dtype=np.float64)).astype(np.int32)
    np.savez_compressed(
        output_dir / "trace_mapping_ext.npz",
        output_trace_index=np.arange(n_traces, dtype=np.int32),
        source_trace_idx=trace_idx.astype(np.int64),
        x=selected["X"].to_numpy(dtype=np.float64),
        y=selected["Y"].to_numpy(dtype=np.float64),
        ix=ix,
        iy=iy,
        surface_valid=map_valid.astype(np.uint8),
        t4_time=np.asarray(mapping["t4_time"], dtype=np.float64)[map_rows].astype(np.float32),
        t6_time=map_t6.astype(np.float32),
        t7_time=t7_ext.astype(np.float32),
    )

    # QC: compare extension density with in-window Shasi density.
    in_win_shasi: list[float] = []
    for i in range(n_traces):
        if not map_valid[i]:
            continue
        lo = int(np.searchsorted(density_samples, map_t6[i], side="left"))
        hi = int(np.searchsorted(density_samples, map_t7[i], side="right"))
        vals = density_matrix[i, lo:hi]
        in_win_shasi.extend(float(v) for v in vals[np.isfinite(vals)])
    in_mean = float(np.mean(in_win_shasi)) if in_win_shasi else np.nan
    ext_values = extension_output[np.isfinite(extension_output)]
    ext_mean = float(ext_values.mean()) if ext_values.size else np.nan
    ext_max = float(ext_values.max()) if ext_values.size else np.nan
    finite_fraction = float(ext_values.size / max(total_ext, 1))
    use_density_extension = bool(
        total_ext > 0
        and finite_fraction > 0.5
        and ext_max <= density_cap
        and np.isfinite(ext_mean)
        and ext_mean >= 0.0
    )
    qc = {
        "status": "pass" if use_density_extension else "fallback_curvature",
        "decision": "use_density_extension" if use_density_extension else "fallback_curvature",
        "extension_total_samples": int(total_ext),
        "extension_valid_samples": int(ext_values.size),
        "extension_finite_fraction": float(finite_fraction),
        "extension_density": {
            "mean": ext_mean,
            "max": ext_max,
            "p50": float(np.median(ext_values)) if ext_values.size else None,
        },
        "in_window_shasi_density": {"mean": in_mean, "count": int(len(in_win_shasi))},
        "extension_to_inwindow_mean_ratio": float(ext_mean / in_mean) if np.isfinite(in_mean) and in_mean > 0 else None,
        "note": "extension band treated as the lower part of the extended Shasi window; in-window density values are unchanged",
        "outputs": {
            "density_sgy": str(output_sgy),
            "trace_mapping": str(output_dir / "trace_mapping_ext.npz"),
            "sample_axis": {
                "min_ms": float(output_samples[0]),
                "max_ms": float(output_samples[-1]),
                "count": int(len(output_samples)),
                "interval_ms": interval_ms,
            },
        },
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(output_dir / "density_extension_qc.json", qc)
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
