# -*- coding: utf-8 -*-
"""Build the T7-below extension horizon contract for the configured region.

For each trace inside target_block with original Shasi present, the new T7'
is set to the attribute-data bottom (from scan_attribute_bottom.py). The
Shasi window becomes [T6, T7']; the Shasan window is unchanged. Traces
outside the block, or without original Shasi, keep their original T7.

Outputs:
  - horizon_trace_table_ext.npy  : full-length table with T7' applied inside
                                   the block (same fields as v2_mine table).
  - horizon_windows_2ms_ext.npz / horizon_windows_10ms_ext.npz : per-trace
    windows for the block traces only (matches the region's trace mapping).

The same script is used for the 5 km demo and for the full mine; only
target_block and output paths change in the config.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABLE_DTYPE = np.dtype(
    [
        ("TraceIdx", "<i4"),
        ("T4", "<f4"),
        ("T5", "<f4"),
        ("T6", "<f4"),
        ("T7", "<f4"),
        ("SurfaceOrderValid", "u1"),
        ("ShasanPresent", "u1"),
        ("ShasiPresent", "u1"),
        ("HorizonCorrectionCode", "u1"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the T7-below extension horizon contract.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def axis_window_indices(
    samples: np.ndarray, top: np.ndarray, base: np.ndarray, present: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    start = np.searchsorted(samples, top, side="left").astype(np.int32)
    stop = np.searchsorted(samples, base, side="right").astype(np.int32)
    start = np.clip(start, 0, len(samples))
    stop = np.clip(stop, 0, len(samples))
    invalid = (~present) | (~np.isfinite(top)) | (~np.isfinite(base)) | (base <= top) | (stop <= start)
    start[invalid] = -1
    stop[invalid] = -1
    return start, stop


def write_axis_contract(path: Path, trace_idx, t4, t6, t7, t7_original, shasan, shasi, start_ms, stop_ms, interval_ms):
    count = int(np.floor((stop_ms - start_ms) / interval_ms + 1.0e-9)) + 1
    samples = start_ms + np.arange(count, dtype=np.float64) * interval_ms
    shasan_start, shasan_stop = axis_window_indices(samples, t4, t6, shasan)
    shasi_start, shasi_stop = axis_window_indices(samples, t6, t7, shasi)
    np.savez_compressed(
        path,
        TraceIdx=trace_idx.astype(np.int32),
        samples=samples.astype(np.float32),
        T4=t4.astype(np.float32),
        T6=t6.astype(np.float32),
        T7=t7.astype(np.float32),
        OriginalT7=t7_original.astype(np.float32),
        ShasanPresent=shasan.astype(np.uint8),
        ShasiPresent=shasi.astype(np.uint8),
        ShasanStartIdx=shasan_start,
        ShasanStopIdxExclusive=shasan_stop,
        ShasiStartIdx=shasi_start,
        ShasiStopIdxExclusive=shasi_stop,
    )
    return {
        "path": str(path),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "sample_interval_ms": float(interval_ms),
        "sample_count": int(len(samples)),
        "shasan_trace_count": int(np.count_nonzero(shasan_start >= 0)),
        "shasi_trace_count": int(np.count_nonzero(shasi_start >= 0)),
    }


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = Path(config["output_root"]).resolve() / "horizon_contract_ext"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_table = np.load(config["horizon_source_table"], mmap_mode="r")
    trace_header = pd.read_csv(config["trace_header_csv"], usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    if len(trace_header) != len(source_table) or not np.array_equal(
        trace_header["TraceIdx"].to_numpy(dtype=np.int64), np.asarray(source_table["TraceIdx"], dtype=np.int64)
    ):
        raise ValueError("trace header and source horizon table TraceIdx contracts differ")

    bottom_npz = np.load(Path(config["output_root"]).resolve() / "attribute_bottom/attribute_bottom.npz")
    bottom_traces = bottom_npz["TraceIdx"].astype(np.int64)
    bottom_ms = bottom_npz["BottomMs"].astype(np.float64)
    bottom_t7 = bottom_npz["T7Original"].astype(np.float64)
    bottom_shasi = bottom_npz["ShasiPresent"].astype(bool)
    bottom_lookup = {int(ti): i for i, ti in enumerate(bottom_traces)}

    t4 = np.asarray(source_table["T4"], dtype=np.float64).copy()
    t5 = np.asarray(source_table["T5"], dtype=np.float64).copy()
    t6 = np.asarray(source_table["T6"], dtype=np.float64).copy()
    t7 = np.asarray(source_table["T7"], dtype=np.float64).copy()
    shasan = np.asarray(source_table["ShasanPresent"], dtype=bool).copy()
    shasi = np.asarray(source_table["ShasiPresent"], dtype=bool).copy()

    block = config["target_block"]
    x = trace_header["X"].to_numpy(dtype=np.float64)
    y = trace_header["Y"].to_numpy(dtype=np.float64)
    selected = (
        (x >= float(block["x_min"]))
        & (x <= float(block["x_max"]))
        & (y >= float(block["y_min"]))
        & (y <= float(block["y_max"]))
    )
    if not np.any(selected):
        raise RuntimeError("target_block contains no seismic traces")

    extended_count = 0
    source_trace_idx = np.asarray(source_table["TraceIdx"], dtype=np.int64)
    for i in np.flatnonzero(selected):
        ti = int(source_trace_idx[i])
        pos = bottom_lookup.get(ti)
        if pos is None:
            continue
        if bottom_shasi[pos] and np.isfinite(bottom_ms[pos]) and bottom_ms[pos] > bottom_t7[pos]:
            t7[i] = bottom_ms[pos]
            extended_count += 1

    table_path = output_dir / "horizon_trace_table_ext.npy"
    temp = table_path.with_suffix(".tmp")
    table = np.lib.format.open_memmap(temp, mode="w+", dtype=TABLE_DTYPE, shape=(len(source_table),))
    table["TraceIdx"] = source_table["TraceIdx"]
    table["T4"] = t4.astype(np.float32)
    table["T5"] = t5.astype(np.float32)
    table["T6"] = t6.astype(np.float32)
    table["T7"] = t7.astype(np.float32)
    table["SurfaceOrderValid"] = np.asarray(source_table["SurfaceOrderValid"], dtype=np.uint8)
    table["ShasanPresent"] = shasan.astype(np.uint8)
    table["ShasiPresent"] = shasi.astype(np.uint8)
    table["HorizonCorrectionCode"] = np.asarray(source_table["HorizonCorrectionCode"], dtype=np.uint8)
    table.flush()
    del table
    os.replace(temp, table_path)

    region_indices = np.flatnonzero(selected)
    region_trace_idx = np.asarray(source_table["TraceIdx"], dtype=np.int64)[region_indices]
    region_positions = np.asarray([bottom_lookup[int(ti)] for ti in region_trace_idx], dtype=np.int64)
    if len(region_positions) != len(bottom_traces) or not np.array_equal(
        region_trace_idx, bottom_traces
    ):
        raise ValueError("bottom scan traces do not match the selected block traces")

    src_windows_2ms = np.load(config["horizon_source_windows_2ms"])
    src_axis2 = np.asarray(src_windows_2ms["samples"], dtype=np.float64)
    src_windows_10ms = np.load(config["horizon_source_windows_10ms"])
    src_axis10 = np.asarray(src_windows_10ms["samples"], dtype=np.float64)
    interval_ms = float(config.get("sample_interval_ms", 2.0))
    max_bottom = float(np.max(bottom_ms[np.isfinite(bottom_ms)])) if np.isfinite(bottom_ms).any() else src_axis2[-1]
    max_bottom = min(max_bottom, float(config.get("extension_max_bottom_ms", 3674.0)))
    axis2_stop = max(float(src_axis2[-1]), float(np.ceil(max_bottom / interval_ms) * interval_ms))
    axis10_stop = max(float(src_axis10[-1]), float(np.floor(max_bottom / 10.0) * 10.0))

    t4r = t4[region_indices]
    t6r = t6[region_indices]
    t7r = t7[region_indices]
    t7_orig_r = np.asarray(source_table["T7"], dtype=np.float64)[region_indices]
    shasan_r = shasan[region_indices]
    shasi_r = shasi[region_indices]
    mask_contracts = {
        "2ms": write_axis_contract(
            output_dir / "horizon_windows_2ms_ext.npz",
            region_trace_idx,
            t4r,
            t6r,
            t7r,
            t7_orig_r,
            shasan_r,
            shasi_r,
            float(src_axis2[0]),
            axis2_stop,
            interval_ms,
        ),
        "10ms": write_axis_contract(
            output_dir / "horizon_windows_10ms_ext.npz",
            region_trace_idx,
            t4r,
            t6r,
            t7r,
            t7_orig_r,
            shasan_r,
            shasi_r,
            float(src_axis10[0]),
            axis10_stop,
            10.0,
        ),
    }
    metadata = {
        "status": "pass",
        "version": str(config.get("version", "formal_mine_ext_demo_v1")),
        "config_path": str(args.config.resolve()),
        "target_block": block,
        "row_count": int(len(source_table)),
        "selected_trace_count": int(selected.sum()),
        "extended_t7_trace_count": int(extended_count),
        "extension_rule": "T7'=min over 3 attribute volumes' deepest valid sample, only for original-Shasi traces; capped by extension_max_bottom_ms",
        "mask_contracts": mask_contracts,
        "outputs": {
            "table": str(table_path),
            "windows_2ms": str(output_dir / "horizon_windows_2ms_ext.npz"),
            "windows_10ms": str(output_dir / "horizon_windows_10ms_ext.npz"),
        },
    }
    write_json(output_dir / "horizon_contract_ext_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
