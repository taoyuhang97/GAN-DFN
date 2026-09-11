#!/usr/bin/env python3
"""Build the compact Taigu three-attribute value/normalization contract.

Only sampled statistics and representative points are saved.  No normalized
3-D attribute volume is generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio

from attribute_contract import POLARITY, robust_limits, score_attribute


LAYERS = {
    "上部复合层": ("TopTimeMs", "MidTimeMs"),
    "太古界风化壳": ("MidTimeMs", "BaseTimeMs"),
}
ATTRIBUTES = ("Coherence", "AntTrack", "CurvatureMax")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def stats(values: np.ndarray, low_q: float, high_q: float, attribute: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    limits = robust_limits(finite, low_q, high_q)
    quantiles = np.quantile(finite, [0.02, 0.05, 0.50, 0.95, 0.98])
    return {
        "finite_count": int(finite.size),
        "nan_count": int(values.size - finite.size),
        "minus_one_count": int(np.count_nonzero(finite == -1.0)) if attribute == "AntTrack" else 0,
        "minus_one_fraction": float(np.mean(finite == -1.0)) if attribute == "AntTrack" else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p02": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p98": float(quantiles[4]),
        **limits,
        "polarity": POLARITY[attribute],
    }


def absolute_axis(handle: Any, origin: float) -> np.ndarray:
    raw = np.asarray(handle.samples, dtype=np.float64)
    return raw - raw[0] + float(origin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(cfg["demo_grid_csv"], encoding="utf-8-sig")
    horizon = pd.read_csv(
        cfg["horizon_contract_csv"],
        encoding="utf-8-sig",
        usecols=["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"],
    )
    work = grid.merge(horizon, on="TraceIdx", how="left", validate="one_to_one")
    work = work[work["SurfaceValid"].fillna(0).astype(bool)].copy()
    stride = max(int(cfg.get("stats_grid_stride", 10)), 1)
    sampled = work[(work["IX"] % stride == 0) & (work["IY"] % stride == 0)].copy()
    if sampled.empty:
        raise RuntimeError("attribute contract QC selected no traces")

    handles: dict[str, Any] = {}
    axes: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    try:
        for name in ATTRIBUTES:
            handle = segyio.open(cfg["volume_paths"][name], "r", ignore_geometry=True)
            handle.mmap()
            handles[name] = handle
            axes[name] = absolute_axis(handle, cfg["time_origins_ms"][name])
            meta[name] = {
                "path": str(Path(cfg["volume_paths"][name]).resolve()),
                "trace_count": int(handle.tracecount),
                "sample_count": int(len(axes[name])),
                "time_min_ms": float(axes[name][0]),
                "time_max_ms": float(axes[name][-1]),
                "sample_interval_ms": float(np.median(np.diff(axes[name]))),
            }
        trace_counts = {meta[name]["trace_count"] for name in ATTRIBUTES}
        if len(trace_counts) != 1:
            raise RuntimeError(f"three attribute trace counts differ: {trace_counts}")

        layer_values: dict[str, dict[str, list[np.ndarray]]] = {
            layer: {name: [] for name in ATTRIBUTES} for layer in LAYERS
        }
        for row in sampled.itertuples(index=False):
            trace_idx = int(row.TraceIdx)
            for layer, (top_col, base_col) in LAYERS.items():
                top, base = float(getattr(row, top_col)), float(getattr(row, base_col))
                if not np.isfinite(top + base) or base <= top:
                    continue
                for name in ATTRIBUTES:
                    axis = axes[name]
                    mask = (axis >= top) & (axis < base)
                    if mask.any():
                        layer_values[layer][name].append(
                            np.asarray(handles[name].trace[trace_idx], dtype=np.float32)[mask]
                        )

        low_q = float(cfg["normalization_quantiles"]["low"])
        high_q = float(cfg["normalization_quantiles"]["high"])
        contract: dict[str, Any] = {
            "version": cfg["version"],
            "scope": "sampled layer statistics; no normalized volume stored",
            "reference_attribute": cfg["reference_attribute"],
            "trace_alignment_assumption": "Coherence/AntTrack/CurvatureMax share the CurvatureMax TraceIdx",
            "normalization_quantiles": {"low": low_q, "high": high_q},
            "layers": {},
        }
        for layer in LAYERS:
            contract["layers"][layer] = {}
            for name in ATTRIBUTES:
                values = np.concatenate(layer_values[layer][name])
                contract["layers"][layer][name] = stats(values, low_q, high_q, name)

        # Representative traces: four spatial corners, center and 405 location.
        probes = {
            "西南角": (float(grid.X.min()), float(grid.Y.min())),
            "西北角": (float(grid.X.min()), float(grid.Y.max())),
            "区域中心": (float(grid.X.median()), float(grid.Y.median())),
            "东南角": (float(grid.X.max()), float(grid.Y.min())),
            "东北角": (float(grid.X.max()), float(grid.Y.max())),
            "405井附近": (665425.0, 4242613.0),
        }
        xy = work[["X", "Y"]].to_numpy(dtype=np.float64)
        representative: list[dict[str, Any]] = []
        for label, (px, py) in probes.items():
            pos = int(np.argmin(np.square(xy[:, 0] - px) + np.square(xy[:, 1] - py)))
            row = work.iloc[pos]
            for layer, (top_col, base_col) in LAYERS.items():
                top, base = float(row[top_col]), float(row[base_col])
                time_ms = 0.5 * (top + base)
                record: dict[str, Any] = {
                    "Probe": label, "TraceIdx": int(row.TraceIdx), "X": float(row.X), "Y": float(row.Y),
                    "LayerGroup": layer, "TopTimeMs": top, "BaseTimeMs": base, "SampleTimeMs": time_ms,
                }
                for name in ATTRIBUTES:
                    raw = float(np.interp(time_ms, axes[name], np.asarray(handles[name].trace[int(row.TraceIdx)])))
                    record[f"{name}Raw"] = raw
                    record[f"{name}Score"] = float(score_attribute(np.array([raw]), name, contract["layers"][layer][name])[0])
                representative.append(record)
    finally:
        for handle in handles.values():
            handle.close()

    representative_df = pd.DataFrame(representative)
    representative_df.to_csv(out / "representative_trace_qc.csv", index=False, encoding="utf-8-sig")
    checks = {
        "three_trace_counts_equal": len({meta[name]["trace_count"] for name in ATTRIBUTES}) == 1,
        "three_time_contracts_cover_target": all(
            meta[name]["time_min_ms"] <= float(work.TopTimeMs.min())
            and meta[name]["time_max_ms"] >= float(work.BaseTimeMs.max()) for name in ATTRIBUTES
        ),
        "anttrack_minus_one_retained": all(
            contract["layers"][layer]["AntTrack"]["minus_one_count"] > 0 for layer in LAYERS
        ),
        "representative_values_finite": bool(np.isfinite(representative_df.filter(regex="Raw$|Score$").to_numpy()).all()),
        "representative_scores_in_0_1": bool(
            ((representative_df.filter(regex="Score$") >= 0) & (representative_df.filter(regex="Score$") <= 1)).all().all()
        ),
        "no_normalized_volume_written": True,
    }
    qc = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(args.config.resolve()),
        "sampled_trace_count": int(len(sampled)),
        "total_demo_trace_count": int(len(grid)),
        "stats_grid_stride": stride,
        "attribute_metadata": meta,
        "target_time_range_ms": {"min": float(work.TopTimeMs.min()), "max": float(work.BaseTimeMs.max())},
        "checks": checks,
    }
    write_json(out / "attribute_normalization_contract.json", contract)
    write_json(out / "attribute_value_qc.json", qc)
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    return 0 if qc["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
