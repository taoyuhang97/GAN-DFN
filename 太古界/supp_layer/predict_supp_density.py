#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step6A（补充层）：在 5×5 km 区块内预测"上部补充层"的 2 ms 密度体。

新层只有 405 一口成像井，按方案 A 复用"上部复合层"的两阶段模型：
  * 属性分数仍用 normalization_contract["layers"]["上部复合层"] 的限值；
  * 层窗取 [TopV, Top)，与上部复合层 [Top, Mid] 不重叠；
  * 模型特征只有 3 个属性分数，不依赖 LayerCode，因此代理预测成立。

输出（默认 ``太古界/supp_layer/output/taigu_supp_density_v1/``）：
  supp_density_block.sgy         区块道 × 2 ms
  supp_density_mapping.npz       source_trace_idx / x / y / ix / iy
  supp_density_summary.json

用法::

    python3 太古界/supp_layer/predict_supp_density.py
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


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
FORMAL_ROOT = REPO / "优化阶段二" / "正式主线"
for path in (FORMAL_ROOT, FORMAL_ROOT / "common", TAIGU / "supp_layer", TAIGU / "step6a_density_volume"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from supp_framework import load_supp_framework  # noqa: E402
from predict_taigu_density_volume import (  # noqa: E402
    MultiAttributeSampler,
    build_layer_features,
    select_model_features,
)


DEFAULT_CONFIG = TAIGU / "step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_quanqu1.json"
DEFAULT_MODEL = (
    TAIGU / "step6a_density_volume/output/taigu_step6a_attribute_v3_md1/models/step6_two_stage_density_models.joblib"
)
DEFAULT_CONTRACT = TAIGU / "supp_layer/output/taigu_supp_horizon_v1/supp_horizon_contract.npy"
DEFAULT_BLOCK = TAIGU / "supp_layer/output/taigu_supp_horizon_v1/supp_horizon_contract_block.csv"
DEFAULT_OUTPUT = TAIGU / "supp_layer/output/taigu_supp_density_v1"
LAYER_NAME = "上部补充层"
MODEL_LAYER = "上部复合层"
LAYER_CODE = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6A prediction for the Taigu supplementary layer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--block-csv", type=Path, default=DEFAULT_BLOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-chunk", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    framework = load_supp_framework()
    layer = framework.layer(LAYER_NAME)
    if layer.code != int(LAYER_CODE):
        raise ValueError("supplementary layer code mismatch")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = joblib.load(args.model)
    model = artifact["models"][MODEL_LAYER]
    normalization = json.loads(
        Path(config["attribute_normalization_contract_json"]).read_text(encoding="utf-8")
    )
    cap = float(model.get("density_cap", 15.0))
    interval = float(config.get("sample_interval_ms", 2.0))

    block = pd.read_csv(args.block_csv, encoding="utf-8-sig")
    block = block[block["SuppPresent"].astype(bool)].reset_index(drop=True)
    if block.empty:
        raise RuntimeError("supplementary block contains no valid traces")
    top_v = pd.to_numeric(block["TopVTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    top = pd.to_numeric(block["TopTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    x = pd.to_numeric(block["X"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(block["Y"], errors="coerce").to_numpy(dtype=np.float64)
    ix = block["IX"].to_numpy(dtype=np.int32)
    iy = block["IY"].to_numpy(dtype=np.int32)
    trace_idx = block["TraceIdx"].to_numpy(dtype=np.int64)

    axis_start = float(np.floor(np.nanmin(top_v) / interval) * interval)
    axis_stop = float(np.ceil(np.nanmax(top) / interval) * interval)
    sample_axis = np.arange(axis_start, axis_stop + interval * 0.5, interval, dtype=np.float64)
    base_inclusive = top - 1.0e-6  # [TopV, Top)
    print(
        f"[supp-step6a] traces={len(block):,} samples={len(sample_axis)} "
        f"axis={sample_axis[0]:.0f}-{sample_axis[-1]:.0f} ms interval={interval:g}",
        flush=True,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sgy_path = output_dir / "supp_density_block.sgy"
    mapping_path = output_dir / "supp_density_mapping.npz"
    np.savez_compressed(
        mapping_path,
        source_trace_idx=trace_idx,
        x=x,
        y=y,
        ix=ix,
        iy=iy,
    )

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
    spec.format = 5
    spec.samples = sample_axis.astype(np.float32)
    spec.tracecount = len(block)

    interval_us = int(round(interval * 1000.0))
    delay_ms = int(round(float(sample_axis[0])))
    started = time.time()
    finite_total = 0
    positive_total = 0
    density_sum = 0.0
    density_max = 0.0

    sampler = MultiAttributeSampler(
        config["attribute_volume_paths"],
        config.get("attribute_invalid_rules", {}),
        config.get("attribute_time_origins_ms", {}),
    )
    chunk_size = max(int(args.trace_chunk), 1)
    with sampler, segyio.create(str(sgy_path), spec) as handle:
        for start in range(0, len(block), chunk_size):
            stop = min(start + chunk_size, len(block))
            rows = np.arange(stop - start, dtype=np.int64)
            lo = top_v[start:stop]
            hi = base_inclusive[start:stop]
            sel_lo = np.searchsorted(sample_axis, lo, side="left")
            sel_hi = np.searchsorted(sample_axis, hi, side="right")
            counts = np.clip(sel_hi - sel_lo, 0, len(sample_axis))
            sel_rows = np.repeat(rows, counts)
            offsets = np.cumsum(counts) - counts
            sel_cols = (
                np.arange(int(counts.sum()), dtype=np.int64)
                - np.repeat(offsets, counts)
                + np.repeat(sel_lo, counts)
            )
            sel_times = sample_axis[sel_cols]
            attrs = sampler.sample_at_xy(x[start:stop][sel_rows], y[start:stop][sel_rows], sel_times)
            attr_arrays = {
                name: np.full((stop - start, len(sample_axis)), np.nan, dtype=np.float32)
                for name in ("Coherence", "AntTrack", "CurvatureMax")
            }
            for name in attr_arrays:
                attr_arrays[name][sel_rows, sel_cols] = attrs[name]

            predicted = np.full((stop - start, len(sample_axis)), np.nan, dtype=np.float32)
            features, row_idx, time_idx = build_layer_features(
                attr_arrays,
                sample_axis,
                lo,
                hi,
                np.ones(stop - start, dtype=bool),
                LAYER_CODE,
                MODEL_LAYER,
                normalization,
            )
            if row_idx.size:
                matrix = select_model_features(features, model)
                prob = model["classifier"].predict_proba(matrix)[:, 1]
                cond = model["conditional_density_regressor"].predict(matrix)
                values = np.clip(prob * np.clip(cond, 0.0, cap), 0.0, cap).astype(np.float32)
                predicted[row_idx, time_idx] = values
                finite_total += int(values.size)
                positive_total += int((values > 0).sum())
                density_sum += float(values.sum())
                density_max = max(density_max, float(values.max()) if values.size else 0.0)

            for local_index in range(stop - start):
                global_index = start + local_index
                header = handle.header[global_index]
                header[segyio.TraceField.TRACE_SEQUENCE_FILE] = global_index + 1
                header[segyio.TraceField.TRACE_SEQUENCE_LINE] = global_index + 1
                header[segyio.TraceField.TRACE_SAMPLE_COUNT] = int(len(sample_axis))
                header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = interval_us
                header[segyio.TraceField.DelayRecordingTime] = delay_ms
                header[segyio.TraceField.SourceX] = int(round(float(x[global_index])))
                header[segyio.TraceField.SourceY] = int(round(float(y[global_index])))
                handle.header[global_index] = header
                handle.trace[global_index] = predicted[local_index]
            print(f"[supp-step6a] wrote traces {stop:,}/{len(block):,}", flush=True)

    elapsed = time.time() - started
    summary: dict[str, Any] = {
        "status": "pass",
        "layer": LAYER_NAME,
        "layer_code": int(LAYER_CODE),
        "model_layer": MODEL_LAYER,
        "model_path": str(args.model.resolve()),
        "contract": str(args.contract.resolve()),
        "block_csv": str(args.block_csv.resolve()),
        "trace_count": int(len(block)),
        "sample_count": int(len(sample_axis)),
        "sample_axis_ms": [float(sample_axis[0]), float(sample_axis[-1])],
        "sample_interval_ms": interval,
        "finite_samples": int(finite_total),
        "positive_samples": int(positive_total),
        "positive_fraction_of_finite": float(positive_total / finite_total) if finite_total else 0.0,
        "density_mean": float(density_sum / finite_total) if finite_total else None,
        "density_max": float(density_max),
        "density_cap": cap,
        "elapsed_seconds": elapsed,
        "outputs": {
            "density_sgy": str(sgy_path),
            "trace_mapping_npz": str(mapping_path),
        },
    }
    (output_dir / "supp_density_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[supp-step6a] done traces={len(block):,} finite={finite_total:,} elapsed={elapsed:.1f}s")
    print(f"[supp-step6a] wrote {sgy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
