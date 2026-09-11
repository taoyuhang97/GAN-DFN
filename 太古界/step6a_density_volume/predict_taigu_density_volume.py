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
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = CURRENT_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(COMMON_DIR / "seismic_sampling") not in sys.path:
    sys.path.insert(0, str(COMMON_DIR / "seismic_sampling"))

if str(COMMON_DIR / "attribute_sampling") not in sys.path:
    sys.path.insert(0, str(COMMON_DIR / "attribute_sampling"))
from attribute_sampling import MultiAttributeSampler  # noqa: E402
from attribute_contract import layer_score  # noqa: E402


class _NoAmplitudeSampler:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    @property
    def handle(self): return None
    def read_2ms_block(self, trace_indices, sample_axis):
        return np.zeros((len(np.asarray(trace_indices)), len(sample_axis)), dtype=np.float32)


FEATURE_COLUMNS = [
    "CoherenceScore",
    "AntTrackScore",
    "CurvatureMaxScore",
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


def validate_attribute_grid_contract(
    grid: pd.DataFrame,
    horizon: pd.DataFrame,
    trace_header_csv: Path,
    xy_tolerance_m: float,
) -> dict[str, Any]:
    required_grid = {"TraceIdx", "X", "Y", "IX", "IY"}
    missing = sorted(required_grid.difference(grid.columns))
    if missing:
        raise ValueError(f"attribute demo grid missing columns: {missing}")
    if grid[list(required_grid)].isna().any().any():
        raise ValueError("attribute demo grid contains null contract fields")
    if grid["TraceIdx"].duplicated().any():
        raise ValueError("attribute demo grid contains duplicate TraceIdx")
    if grid[["IX", "IY"]].duplicated().any():
        raise ValueError("attribute demo grid contains duplicate IX/IY cells")
    expected_count = int(grid["IX"].nunique() * grid["IY"].nunique())
    if len(grid) != expected_count:
        raise ValueError(f"attribute demo grid is not rectangular: {len(grid)} != {expected_count}")

    header = pd.read_csv(trace_header_csv, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    for column in ("TraceIdx", "X", "Y"):
        header[column] = pd.to_numeric(header[column], errors="coerce")
    header = header.dropna().drop_duplicates("TraceIdx").set_index("TraceIdx")
    source = header.reindex(grid["TraceIdx"].to_numpy(dtype=np.int64))
    if source[["X", "Y"]].isna().any().any():
        raise ValueError("attribute demo grid contains TraceIdx absent from attribute trace header")
    distance = np.hypot(
        source["X"].to_numpy(dtype=float) - grid["X"].to_numpy(dtype=float),
        source["Y"].to_numpy(dtype=float) - grid["Y"].to_numpy(dtype=float),
    )
    maximum_distance = float(distance.max())
    if maximum_distance > xy_tolerance_m:
        raise ValueError(
            "demo grid TraceIdx does not match attribute-header X/Y: "
            f"max_distance={maximum_distance:.3f}m > {xy_tolerance_m:.3f}m; "
            "an OBN trace index may have been supplied"
        )
    horizon_ids = set(pd.to_numeric(horizon["TraceIdx"], errors="coerce").dropna().astype(np.int64))
    missing_horizon_count = int(sum(int(value) not in horizon_ids for value in grid["TraceIdx"]))
    if missing_horizon_count:
        raise ValueError(f"attribute demo grid has {missing_horizon_count} TraceIdx absent from horizon contract")
    return {
        "status": "pass",
        "trace_count": int(len(grid)),
        "attribute_xy_tolerance_m": float(xy_tolerance_m),
        "attribute_xy_distance_max_m": maximum_distance,
        "attribute_xy_distance_median_m": float(np.median(distance)),
        "missing_horizon_trace_count": missing_horizon_count,
    }


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
    attributes: dict[str, np.ndarray],
    samples: np.ndarray,
    top: np.ndarray,
    base: np.ndarray,
    valid: np.ndarray,
    layer_code: float,
    layer_name: str,
    normalization_contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = samples[None, :]
    scores = {
        name: layer_score(values, name, normalization_contract, layer_name)
        for name, values in attributes.items()
    }
    attribute_finite = np.logical_and.reduce([np.isfinite(values) for values in scores.values()])
    mask = valid[:, None] & (times >= top[:, None]) & (times <= base[:, None]) & attribute_finite
    row_idx, time_idx = np.where(mask)
    if row_idx.size == 0:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), row_idx, time_idx
    local_time = samples[time_idx].astype(np.float32)
    local_top = top[row_idx].astype(np.float32)
    local_base = base[row_idx].astype(np.float32)
    thickness = local_base - local_top
    since_top = local_time - local_top
    to_base = local_base - local_time
    features = np.column_stack(
        [
            scores["Coherence"][row_idx, time_idx],
            scores["AntTrack"][row_idx, time_idx],
            scores["CurvatureMax"][row_idx, time_idx],
            np.full(row_idx.size, layer_code, dtype=np.float32),
            since_top / thickness,
            since_top,
            to_base,
            thickness,
            local_time,
        ]
    ).astype(np.float32)
    return features, row_idx, time_idx


def fill_missing_horizons_nearest(
    grid: pd.DataFrame, max_distance_m: float, min_thickness_ms: float
) -> pd.DataFrame:
    """Fill only nearby invalid traces from one complete valid source trace.

    Original picks are retained; filled picks are provenance-tagged for QC.
    """
    out = grid.copy()
    out["OriginalSurfaceValid"] = out["SurfaceValid"].fillna(0).astype(bool)
    for c in ("TopTimeMs", "MidTimeMs", "BaseTimeMs"):
        out[f"Original{c}"] = out[c]
    out["HorizonFillUsed"] = False
    out["HorizonFillValid"] = False
    out["HorizonFillSourceTraceIdx"] = np.nan
    out["HorizonFillDistanceM"] = np.nan
    out["FilledSurfaceValid"] = out["OriginalSurfaceValid"]
    valid = out["OriginalSurfaceValid"].to_numpy()
    valid_quality = valid & out[["TopTimeMs", "MidTimeMs", "BaseTimeMs"]].notna().all(axis=1).to_numpy()
    valid_quality &= (out["MidTimeMs"] - out["TopTimeMs"] >= min_thickness_ms).to_numpy()
    valid_quality &= (out["BaseTimeMs"] - out["MidTimeMs"] >= min_thickness_ms).to_numpy()
    bad = np.flatnonzero(~valid)
    src_idx = np.flatnonzero(valid_quality)
    if bad.size and src_idx.size:
        tree = cKDTree(out.loc[src_idx, ["X", "Y"]].to_numpy(float))
        distances, nearest = tree.query(out.loc[bad, ["X", "Y"]].to_numpy(float), k=1)
        for dest, dist, near in zip(bad, distances, nearest):
            if not np.isfinite(dist) or dist > max_distance_m:
                continue
            src = int(src_idx[int(near)])
            vals = out.loc[src, ["TopTimeMs", "MidTimeMs", "BaseTimeMs"]]
            top, mid, base = (float(vals["TopTimeMs"]), float(vals["MidTimeMs"]), float(vals["BaseTimeMs"]))
            if not (top < mid < base and mid - top >= min_thickness_ms and base - mid >= min_thickness_ms):
                continue
            out.loc[dest, ["TopTimeMs", "MidTimeMs", "BaseTimeMs"]] = [top, mid, base]
            out.loc[dest, "HorizonFillUsed"] = True
            out.loc[dest, "HorizonFillValid"] = True
            out.loc[dest, "FilledSurfaceValid"] = True
            out.loc[dest, "HorizonFillSourceTraceIdx"] = int(out.iloc[src]["TraceIdx"])
            out.loc[dest, "HorizonFillDistanceM"] = float(dist)
    return out


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_sgy = output_dir / "predicted_fracture_density.sgy"
    partial_sgy = output_dir / "predicted_fracture_density.sgy.partial"
    window_npz = output_dir / "window_code.npz"
    trace_mapping_npz = output_dir / "trace_mapping.npz"
    block_qc_path = output_dir / "prediction_block_qc.csv"
    summary_path = output_dir / "prediction_summary.json"
    horizon_qc_path = output_dir / "horizon_fill_qc.csv"
    for path in (output_sgy, partial_sgy, window_npz, trace_mapping_npz, block_qc_path, summary_path, horizon_qc_path):
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing prediction output: {path}")

    model_path = Path(config["model_joblib"]).resolve()
    artifact = joblib.load(model_path)
    if tuple(artifact["feature_columns"]) != tuple(FEATURE_COLUMNS):
        raise ValueError(f"model feature contract mismatch: {artifact['feature_columns']}")
    normalization_contract_path = Path(config["attribute_normalization_contract_json"]).resolve()
    normalization_contract = read_json(normalization_contract_path)
    model_contract = artifact.get("attribute_normalization_contract", {})
    if model_contract.get("version") != normalization_contract.get("version"):
        raise ValueError(
            f"training/prediction attribute contract mismatch: {model_contract.get('version')} "
            f"!= {normalization_contract.get('version')}"
        )
    interval_ms = float(config.get("sample_interval_ms", 2.0))
    ext_window_ms = float(config.get("ext_window_ms", 50.0))
    density_cap = float(config.get("density_cap", 15.0))

    demo_grid_path = Path(config["demo_grid_csv"]).resolve()
    trace_header_path = Path(config["trace_header_csv"]).resolve()
    horizon_contract_path = Path(config["horizon_contract_csv"]).resolve()
    grid = pd.read_csv(demo_grid_path, encoding="utf-8-sig")
    horizon = pd.read_csv(horizon_contract_path, encoding="utf-8-sig")
    for df in (grid, horizon):
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grid_contract_qc = validate_attribute_grid_contract(
        grid,
        horizon,
        trace_header_path,
        float(config.get("attribute_grid_xy_tolerance_m", 2.0)),
    )
    grid = grid.merge(
        horizon[["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"]],
        on="TraceIdx",
        how="left",
    )
    grid = grid.sort_values("TraceIdx").reset_index(drop=True)
    if len(grid) != int(grid["TraceIdx"].nunique()):
        raise RuntimeError("demo grid has duplicate TraceIdx")
    if args.max_x_lines > 0:
        grid = grid[grid["IX"] < int(args.max_x_lines)].copy().reset_index(drop=True)
        if grid.empty:
            raise RuntimeError("--max-x-lines selected no demo-grid traces")

    max_fill_distance = float(config.get("max_horizon_fill_distance_m", 25.0))
    min_thickness = float(config.get("min_horizon_thickness_ms", 1.0))
    fill_enabled = bool(config.get("horizon_fill_enabled", True))
    if fill_enabled:
        grid = fill_missing_horizons_nearest(grid, max_fill_distance, min_thickness)
    else:
        grid["OriginalSurfaceValid"] = grid["SurfaceValid"].fillna(0).astype(bool)
        grid["HorizonFillUsed"] = False; grid["HorizonFillValid"] = False
        grid["HorizonFillSourceTraceIdx"] = np.nan; grid["HorizonFillDistanceM"] = np.nan
        grid["FilledSurfaceValid"] = grid["OriginalSurfaceValid"]
    grid["SurfaceValid"] = grid["FilledSurfaceValid"]
    top = grid["TopTimeMs"].to_numpy(dtype=np.float64)
    mid = grid["MidTimeMs"].to_numpy(dtype=np.float64)
    base = grid["BaseTimeMs"].to_numpy(dtype=np.float64)
    horizon_order_valid = (np.isfinite(top) & np.isfinite(mid) & np.isfinite(base) & (top < mid) & (mid < base) & ((mid - top) >= min_thickness) & ((base - mid) >= min_thickness))
    # Defensive QC: any failed post-fill geometry remains invalid for prediction.
    grid.loc[~horizon_order_valid, "FilledSurfaceValid"] = False
    grid.loc[~horizon_order_valid, "SurfaceValid"] = False
    valid_mask = grid["FilledSurfaceValid"].to_numpy()
    if not valid_mask.any():
        raise RuntimeError("horizon contract contains no valid surfaces")
    valid_top = grid.loc[valid_mask, "TopTimeMs"].to_numpy()
    valid_base = grid.loc[valid_mask, "BaseTimeMs"].to_numpy()
    axis_start = float(np.floor(valid_top.min() / interval_ms) * interval_ms)
    axis_stop = float(np.ceil((valid_base + ext_window_ms).max() / interval_ms) * interval_ms)
    sample_axis = np.arange(axis_start, axis_stop + interval_ms * 0.5, interval_ms, dtype=np.float64)
    x_line_count = int(grid["IX"].max()) + 1
    block_x_lines = max(int(config.get("block_x_line_count", 20)), 1)
    total_blocks = (x_line_count + block_x_lines - 1) // block_x_lines
    model_upper = artifact["models"]["上部复合层"]
    model_crust = artifact["models"]["太古界风化壳"]
    cap = min(float(density_cap), float(model_upper["density_cap"]), float(model_crust["density_cap"]))

    block_rows: list[dict[str, Any]] = []
    stats_all = RunningStats()
    stats_main = RunningStats()
    stats_ext = RunningStats()
    layer_total = {"上部复合层": 0, "太古界风化壳": 0}
    anttrack_minus_one_count = 0
    anttrack_finite_count = 0
    started = time.time()

    attr_sampler = MultiAttributeSampler(
        config["attribute_volume_paths"],
        config.get("attribute_invalid_rules", {}),
        config.get("attribute_time_origins_ms", {}),
    )
    with _NoAmplitudeSampler() as sampler, attr_sampler:
        run_started = time.time()
        print(
            f"[step6a-predict] start traces={len(grid)} x_lines={x_line_count} "
            f"blocks={total_blocks} samples={len(sample_axis)} interval_ms={interval_ms}",
            flush=True,
        )
        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
        spec.format = 5
        spec.samples = sample_axis.astype(np.float32)
        spec.tracecount = int(len(grid))
        delay_ms = int(round(float(sample_axis[0])))
        interval_us = int(round(interval_ms * 1000.0))
        with segyio.create(str(partial_sgy), spec) as output_handle:
            output_handle.bin[segyio.BinField.Interval] = interval_us
            output_handle.bin[segyio.BinField.Samples] = int(len(sample_axis))
            output_handle.bin[segyio.BinField.Format] = 5
            window_codes = np.zeros((len(grid), len(sample_axis)), dtype=np.uint8)
            output_trace_idx = np.empty(len(grid), dtype=np.int64)
            output_x = np.empty(len(grid), dtype=np.float64)
            output_y = np.empty(len(grid), dtype=np.float64)
            output_cursor = 0
            for block_index, ix_start in enumerate(range(0, x_line_count, block_x_lines), start=1):
                ix_stop = min(ix_start + block_x_lines, x_line_count)
                block = grid[grid["IX"].between(ix_start, ix_stop - 1)].sort_values("TraceIdx").copy()
                source_indices = block["TraceIdx"].to_numpy(dtype=np.int64)
                bx = block["X"].to_numpy(dtype=float); by = block["Y"].to_numpy(dtype=float)
                attrs = attr_sampler.sample_at_xy(
                    np.repeat(bx, len(sample_axis)),
                    np.repeat(by, len(sample_axis)),
                    np.tile(sample_axis, len(block)),
                )
                attr_arrays = {name: attrs[name].reshape(len(block), len(sample_axis)) for name in ("Coherence", "AntTrack", "CurvatureMax")}
                anttrack = attr_arrays["AntTrack"]
                anttrack_finite_count += int(np.isfinite(anttrack).sum())
                anttrack_minus_one_count += int(np.isclose(anttrack, -1.0, equal_nan=False).sum())
                predicted = np.full((len(block), len(sample_axis)), np.nan, dtype=np.float32)
                codes = np.zeros((len(block), len(sample_axis)), dtype=np.uint8)
                block_valid = block["FilledSurfaceValid"].fillna(0).astype(bool).to_numpy()

                features, row_idx, time_idx = build_layer_features(
                    attr_arrays, sample_axis, block["TopTimeMs"].to_numpy(), block["MidTimeMs"].to_numpy(),
                    block_valid, 1.0, "上部复合层", normalization_contract
                )
                if len(row_idx):
                    prob = model_upper["classifier"].predict_proba(features)[:, 1]
                    cond = model_upper["conditional_density_regressor"].predict(features)
                    predicted[row_idx, time_idx] = np.clip(prob * np.clip(cond, 0.0, cap), 0.0, cap)
                    codes[row_idx, time_idx] = 1
                    layer_total["上部复合层"] += int(len(row_idx))

                features, row_idx, time_idx = build_layer_features(
                    attr_arrays, sample_axis, block["MidTimeMs"].to_numpy(), block["BaseTimeMs"].to_numpy(),
                    block_valid, 2.0, "太古界风化壳", normalization_contract
                )
                if len(row_idx):
                    prob = model_crust["classifier"].predict_proba(features)[:, 1]
                    cond = model_crust["conditional_density_regressor"].predict(features)
                    predicted[row_idx, time_idx] = np.clip(prob * np.clip(cond, 0.0, cap), 0.0, cap)
                    codes[row_idx, time_idx] = 1
                    layer_total["太古界风化壳"] += int(len(row_idx))

                features, row_idx, time_idx = build_layer_features(
                    attr_arrays,
                    sample_axis,
                    block["BaseTimeMs"].to_numpy(),
                    block["BaseTimeMs"].to_numpy() + ext_window_ms,
                    block_valid,
                    2.0,
                    "太古界风化壳",
                    normalization_contract,
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
                    header = {
                        segyio.TraceField.TRACE_SEQUENCE_FILE: output_cursor + 1,
                        segyio.TraceField.TRACE_SEQUENCE_LINE: output_cursor + 1,
                        segyio.TraceField.INLINE_3D: int(row.get("Inline3D", 0)),
                        segyio.TraceField.CROSSLINE_3D: int(row.get("Crossline3D", 0)),
                    }
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
                    output_trace_idx[output_cursor] = source_index
                    output_x[output_cursor] = float(row["X"])
                    output_y[output_cursor] = float(row["Y"])
                    output_cursor += 1
                finite_count = int(np.isfinite(predicted).sum())
                block_rows.append(
                    {
                        "BlockIndex": block_index,
                        "IXStart": ix_start,
                        "IXStopExclusive": ix_stop,
                        "TraceCount": int(len(block)),
                        "OriginalSurfaceValidTraceCount": int(block["OriginalSurfaceValid"].sum()),
                        "HorizonFilledTraceCount": int(block["HorizonFillValid"].sum()),
                        "FilledSurfaceValidTraceCount": int(block_valid.sum()),
                        "HorizonFillRejectedTraceCount": int((~block["OriginalSurfaceValid"] & ~block["HorizonFillValid"]).sum()),
                        "HorizonFillDistanceMeanM": float(block.loc[block["HorizonFillValid"], "HorizonFillDistanceM"].mean()) if block["HorizonFillValid"].any() else np.nan,
                        "HorizonFillDistanceMaxM": float(block.loc[block["HorizonFillValid"], "HorizonFillDistanceM"].max()) if block["HorizonFillValid"].any() else np.nan,
                        "FinitePredictionCount": finite_count,
                        "NaNCount": int(predicted.size - finite_count),
                        "UpperPredictionCount": int(layer_total["上部复合层"]),
                        "CrustPredictionCount": int(layer_total["太古界风化壳"]),
                        "Ext50PredictionCount": int((codes == 2).sum()),
                        "PredictionMean": float(np.nanmean(predicted)) if finite_count else np.nan,
                        "ElapsedSeconds": float(time.time() - run_started),
                    }
                )
                elapsed = time.time() - run_started
                completed_fraction = block_index / max(total_blocks, 1)
                eta = elapsed * (1.0 - completed_fraction) / completed_fraction if completed_fraction > 0 else np.nan
                try:
                    partial_mb = partial_sgy.stat().st_size / (1024.0 * 1024.0)
                except OSError:
                    partial_mb = np.nan
                print(
                    f"[step6a-predict] block={block_index} ix={ix_start}:{ix_stop} "
                    f"progress={block_index}/{total_blocks} ({completed_fraction:.1%}) "
                    f"traces={output_cursor}/{len(grid)} finite={finite_count} "
                    f"elapsed={elapsed/60.0:.1f}min eta={eta/60.0:.1f}min partial={partial_mb:.1f}MB",
                    flush=True,
                )
        partial_sgy.replace(output_sgy)
        np.savez_compressed(window_npz, window_code=window_codes, sample_axis=sample_axis.astype(np.float32))
        np.savez_compressed(
            trace_mapping_npz,
            trace_idx=output_trace_idx[:output_cursor],
            x=output_x[:output_cursor],
            y=output_y[:output_cursor],
            output_trace_index=np.arange(output_cursor, dtype=np.int64),
        )
        print(
            f"[step6a-predict] volume_written sgy={output_sgy} elapsed={(time.time()-run_started)/60.0:.1f}min",
            flush=True,
        )

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
        "trace_mapping_npz_exists": trace_mapping_npz.exists(),
        "sgy_trace_count_matches": int(handle.tracecount) == int(len(grid)),
        "sgy_sample_axis_matches": len(samples) == len(sample_axis) and bool(np.allclose(samples, sample_axis, atol=1.0e-6)),
        "sampled_output_contains_finite": bool(np.isfinite(sampled).any()),
        "sampled_output_contains_nan": bool(np.isnan(sampled).any()),
        "common_contract_version_matches_model": model_contract.get("version") == normalization_contract.get("version"),
        "anttrack_minus_one_retained": anttrack_minus_one_count > 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "model_joblib": str(model_path),
        "model_contract_version": artifact.get("contract_version"),
        "model_logic": artifact.get("model_logic"),
        "attribute_normalization_contract_path": str(normalization_contract_path),
        "attribute_normalization_contract_version": normalization_contract.get("version"),
        "attribute_demo_grid_csv": str(demo_grid_path),
        "attribute_grid_contract_qc": grid_contract_qc,
        "anttrack_semantics": "-1_is_valid_low_fracture_evidence",
        "attribute_sampling_audit": {
            "anttrack_finite_count": anttrack_finite_count,
            "anttrack_minus_one_count": anttrack_minus_one_count,
        },
        "output_paths": {
            "density_sgy": str(output_sgy),
            "window_code_npz": str(window_npz),
            "trace_mapping_npz": str(trace_mapping_npz),
            "prediction_block_qc_csv": str(block_qc_path),
            "prediction_summary_json": str(summary_path),
            "horizon_fill_qc_csv": str(horizon_qc_path),
        },
        "grid": {
            "trace_count": int(len(grid)),
            "x_line_count": int(grid["IX"].max()) + 1,
            "y_line_count": int(grid["IY"].max()) + 1,
            "surface_valid_trace_count": int(grid["FilledSurfaceValid"].fillna(0).astype(bool).sum()),
            "surface_valid_fraction": float(valid_mask.mean()),
            "surface_valid_original_trace_count": int(grid["OriginalSurfaceValid"].sum()),
            "horizon_filled_trace_count": int(grid["HorizonFillValid"].sum()),
            "horizon_fill_rejected_trace_count": int((~grid["OriginalSurfaceValid"] & ~grid["HorizonFillValid"]).sum()),
            "horizon_order_valid_trace_count": int(horizon_order_valid.sum()),
            "horizon_fill_enabled": fill_enabled,
            "max_horizon_fill_distance_m": max_fill_distance,
            "horizon_fill_distance_m": {
                "max": float(grid.loc[grid["HorizonFillValid"], "HorizonFillDistanceM"].max())
                if grid["HorizonFillValid"].any() else 0.0,
                "mean": float(grid.loc[grid["HorizonFillValid"], "HorizonFillDistanceM"].mean())
                if grid["HorizonFillValid"].any() else 0.0,
            },
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
    qc_columns = ["TraceIdx", "IX", "IY", "X", "Y", "OriginalSurfaceValid", "HorizonFillValid", "HorizonFillSourceTraceIdx", "HorizonFillDistanceM", "FilledSurfaceValid", "TopTimeMs", "MidTimeMs", "BaseTimeMs"]
    grid[qc_columns].to_csv(horizon_qc_path, index=False, encoding="utf-8-sig")
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
