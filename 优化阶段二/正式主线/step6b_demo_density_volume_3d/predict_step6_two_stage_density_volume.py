from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import load_surface_tables, validate_surface_order  # noqa: E402


ALLOWED_LAYERS = ("沙三段", "沙四段")
LAYER_CODE = {"沙三段": 3.0, "沙四段": 4.0}
ATTRIBUTE_COLUMNS = ("SeisAmp", "Coherence", "AntTrack", "CurvatureMax")
EXPECTED_FEATURE_COLUMNS = (
    *ATTRIBUTE_COLUMNS,
    "LayerCode",
    "RelativeTimeInLayer",
    "TimeSinceTop",
    "TimeToBase",
    "LayerThickness",
    "TimeMs",
)
NULL_THRESHOLD = -1.0e6


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a 2 ms Step6 density SGY from trained two-stage models.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-x-lines", type=int, default=0, help="Smoke-test cap; 0 predicts the full target block.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_surface_lookups(layer_dir: Path) -> dict[str, tuple[cKDTree, np.ndarray]]:
    surfaces = load_surface_tables(layer_dir)
    lookups: dict[str, tuple[cKDTree, np.ndarray]] = {}
    for code, payload in surfaces.items():
        table = payload["table"]
        xy = table[["X", "Y"]].to_numpy(dtype=np.float64)
        lookups[code] = (cKDTree(xy), table["Z"].to_numpy(dtype=np.float64))
    return lookups


def build_target_grid(trace_header_csv: Path, target_block: dict[str, float], max_x_lines: int) -> pd.DataFrame:
    trace_df = pd.read_csv(trace_header_csv, usecols=["TraceIdx", "X", "Y"])
    for column in ("TraceIdx", "X", "Y"):
        trace_df[column] = pd.to_numeric(trace_df[column], errors="coerce")
    trace_df = trace_df.dropna(subset=["TraceIdx", "X", "Y"]).copy()
    mask = (
        trace_df["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
        & trace_df["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
    )
    work = trace_df.loc[mask, ["TraceIdx", "X", "Y"]].drop_duplicates("TraceIdx").copy()
    if work.empty:
        raise RuntimeError("target block contains no trace-header rows")
    x_values = np.sort(work["X"].unique())
    y_values = np.sort(work["Y"].unique())
    if max_x_lines > 0:
        x_values = x_values[:max_x_lines]
        work = work[work["X"].isin(x_values)].copy()
    x_rank = {float(value): idx for idx, value in enumerate(x_values)}
    y_rank = {float(value): idx for idx, value in enumerate(y_values)}
    work["IX"] = work["X"].map(x_rank).astype(np.int32)
    work["IY"] = work["Y"].map(y_rank).astype(np.int32)
    work["TraceIdx"] = work["TraceIdx"].astype(np.int64)
    work = work.sort_values("TraceIdx").reset_index(drop=True)
    expected_count = int(len(x_values) * len(y_values))
    if len(work) != expected_count:
        raise RuntimeError(f"target grid is incomplete: rows={len(work)} expected={expected_count}")
    if work[["IX", "IY"]].duplicated().any():
        raise RuntimeError("target grid has duplicate IX/IY cells")
    return work


def attach_surfaces(grid: pd.DataFrame, lookups: dict[str, tuple[cKDTree, np.ndarray]]) -> pd.DataFrame:
    work = grid.copy()
    xy = work[["X", "Y"]].to_numpy(dtype=np.float64)
    for code, (tree, surface_time) in lookups.items():
        _, indices = tree.query(xy, k=1, p=1)
        work[f"{code}_TIME"] = surface_time[np.asarray(indices, dtype=np.int64)]
    work = validate_surface_order(work, min_thickness=1.0)
    work["SurfaceValid"] = work["Check_All"].fillna(False).astype(bool)
    return work


def build_sample_axis(grid: pd.DataFrame, interval_ms: float) -> np.ndarray:
    valid = grid["SurfaceValid"].fillna(False)
    if not valid.any():
        raise RuntimeError("target grid has no surface-valid traces")
    start = np.floor(float(grid.loc[valid, "T4_TIME"].min()) / interval_ms) * interval_ms
    stop = np.ceil(float(grid.loc[valid, "T7_TIME"].max()) / interval_ms) * interval_ms
    return np.arange(start, stop + interval_ms * 0.5, interval_ms, dtype=np.float64)


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return []
    boundaries = np.where(np.diff(values) != 1)[0] + 1
    chunks = np.split(values, boundaries)
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks]


def read_trace_indices(handle: segyio.SegyFile, indices: np.ndarray) -> np.ndarray:
    parts = [np.asarray(handle.trace.raw[start:stop], dtype=np.float32) for start, stop in contiguous_runs(indices)]
    if not parts:
        return np.empty((0, len(handle.samples)), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def resample_regular_matrix(matrix: np.ndarray, source_samples: np.ndarray, target_samples: np.ndarray) -> np.ndarray:
    source = np.asarray(source_samples, dtype=np.float64)
    if source.size < 2:
        raise ValueError("source sample axis must contain at least two values")
    dt = float(np.median(np.diff(source)))
    if dt <= 0 or not np.allclose(np.diff(source), dt, atol=1.0e-6, rtol=0.0):
        raise ValueError("source sample axis is not regular")
    positions = (target_samples - float(source[0])) / dt
    left = np.floor(positions).astype(np.int64)
    alpha = (positions - left).astype(np.float32)
    valid = (left >= 0) & (left < len(source) - 1)
    output = np.full((matrix.shape[0], len(target_samples)), np.nan, dtype=np.float32)
    if valid.any():
        left_valid = left[valid]
        a = alpha[valid][None, :]
        v0 = matrix[:, left_valid]
        v1 = matrix[:, left_valid + 1]
        output[:, valid] = v0 * (1.0 - a) + v1 * a
    return output


def load_attribute_block(
    handle: segyio.SegyFile,
    source_trace_indices: np.ndarray,
    target_samples: np.ndarray,
) -> np.ndarray:
    matrix = read_trace_indices(handle, source_trace_indices)
    matrix[(~np.isfinite(matrix)) | (matrix <= NULL_THRESHOLD)] = np.nan
    return resample_regular_matrix(matrix, np.asarray(handle.samples, dtype=np.float64), target_samples)


def build_layer_features(
    attributes: dict[str, np.ndarray],
    samples: np.ndarray,
    top: np.ndarray,
    base: np.ndarray,
    surface_valid: np.ndarray,
    layer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = samples[None, :]
    valid = surface_valid[:, None] & (times >= top[:, None]) & (times <= base[:, None])
    for attribute in ATTRIBUTE_COLUMNS:
        valid &= np.isfinite(attributes[attribute])
    row_idx, time_idx = np.where(valid)
    if row_idx.size == 0:
        return np.empty((0, len(EXPECTED_FEATURE_COLUMNS)), dtype=np.float32), row_idx, time_idx
    local_time = samples[time_idx].astype(np.float32)
    local_top = top[row_idx].astype(np.float32)
    local_base = base[row_idx].astype(np.float32)
    thickness = local_base - local_top
    since_top = local_time - local_top
    to_base = local_base - local_time
    features = np.column_stack(
        [
            *(attributes[name][row_idx, time_idx] for name in ATTRIBUTE_COLUMNS),
            np.full(row_idx.size, LAYER_CODE[layer], dtype=np.float32),
            since_top / thickness,
            since_top,
            to_base,
            thickness,
            local_time,
        ]
    ).astype(np.float32)
    return features, row_idx, time_idx


def predict_block(
    attributes: dict[str, np.ndarray],
    samples: np.ndarray,
    grid_block: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    density_cap: float,
) -> tuple[np.ndarray, dict[str, int]]:
    output = np.full((len(grid_block), len(samples)), np.nan, dtype=np.float32)
    layer_counts: dict[str, int] = {}
    surface_valid = grid_block["SurfaceValid"].to_numpy(dtype=bool)
    for layer, top_column, base_column in (
        ("沙三段", "T4_TIME", "T6_TIME"),
        ("沙四段", "T6_TIME", "T7_TIME"),
    ):
        features, row_idx, time_idx = build_layer_features(
            attributes=attributes,
            samples=samples,
            top=grid_block[top_column].to_numpy(dtype=np.float64),
            base=grid_block[base_column].to_numpy(dtype=np.float64),
            surface_valid=surface_valid,
            layer=layer,
        )
        layer_counts[layer] = int(len(row_idx))
        if len(row_idx) == 0:
            continue
        model_bundle = models[layer]
        probability = model_bundle["classifier"].predict_proba(features)[:, 1]
        conditional = model_bundle["conditional_density_regressor"].predict(features)
        density = np.clip(probability * np.clip(conditional, 0.0, density_cap), 0.0, density_cap)
        output[row_idx, time_idx] = density.astype(np.float32)
    return output, layer_counts


def write_mapping(path: Path, grid: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        output_trace_index=np.arange(len(grid), dtype=np.int32),
        source_trace_idx=grid["TraceIdx"].to_numpy(dtype=np.int64),
        x=grid["X"].to_numpy(dtype=np.float64),
        y=grid["Y"].to_numpy(dtype=np.float64),
        ix=grid["IX"].to_numpy(dtype=np.int32),
        iy=grid["IY"].to_numpy(dtype=np.int32),
        surface_valid=grid["SurfaceValid"].to_numpy(dtype=np.uint8),
        t4_time=grid["T4_TIME"].to_numpy(dtype=np.float32),
        t6_time=grid["T6_TIME"].to_numpy(dtype=np.float32),
        t7_time=grid["T7_TIME"].to_numpy(dtype=np.float32),
    )


def verify_output(path: Path, expected_traces: int, expected_samples: np.ndarray, sample_trace_indices: np.ndarray) -> dict[str, Any]:
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        checks = {
            "trace_count_matches": int(handle.tracecount) == int(expected_traces),
            "sample_count_matches": len(samples) == len(expected_samples),
            "sample_axis_matches": len(samples) == len(expected_samples)
            and bool(np.allclose(samples, expected_samples, atol=1.0e-6, rtol=0.0)),
        }
        sampled = np.stack([np.asarray(handle.trace[int(index)], dtype=np.float32) for index in sample_trace_indices])
    checks["sampled_output_contains_finite"] = bool(np.isfinite(sampled).any())
    checks["sampled_output_contains_nan"] = bool(np.isnan(sampled).any())
    return {"checks": checks, "sampled_trace_count": int(len(sample_trace_indices))}


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_sgy = output_dir / "predicted_fracture_density.sgy"
    partial_sgy = output_dir / "predicted_fracture_density.sgy.partial"
    mapping_path = output_dir / "trace_mapping.npz"
    block_qc_path = output_dir / "prediction_block_qc.csv"
    summary_path = output_dir / "prediction_summary.json"
    for path in (output_sgy, partial_sgy, mapping_path, block_qc_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing prediction output: {path}")

    model_path = Path(config["model_joblib"]).resolve()
    trace_header_csv = Path(config["trace_header_csv"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    source_sgy = Path(config["source_sgy_for_headers"]).resolve()
    volume_paths = {name: Path(config["volume_paths"][name]).resolve() for name in ATTRIBUTE_COLUMNS}
    for label, path in (
        ("model_joblib", model_path),
        ("trace_header_csv", trace_header_csv),
        ("layer_dir", layer_dir),
        ("source_sgy_for_headers", source_sgy),
        *((f"volume:{name}", path) for name, path in volume_paths.items()),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    artifact = joblib.load(model_path)
    if tuple(artifact.get("feature_columns", ())) != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(f"model feature contract mismatch: {artifact.get('feature_columns')}")
    if sorted(artifact.get("models", {})) != sorted(ALLOWED_LAYERS):
        raise ValueError("model artifact does not contain both 沙三段 and 沙四段")
    if "CurvaturePos" in artifact.get("feature_columns", ()):
        raise ValueError("formal predictor must not use CurvaturePos")

    started = time.time()
    grid = build_target_grid(trace_header_csv, dict(config["target_block"]), int(args.max_x_lines))
    grid = attach_surfaces(grid, load_surface_lookups(layer_dir))
    interval_ms = float(config.get("sample_interval_ms", 2.0))
    sample_axis = build_sample_axis(grid, interval_ms)
    block_x_lines = max(int(config.get("block_x_line_count", 2)), 1)
    density_cap = float(config.get("density_cap", artifact.get("models", {}).get("沙三段", {}).get("density_cap", 10.0)))
    x_line_count = int(grid["IX"].max()) + 1
    block_rows: list[dict[str, Any]] = []
    density_stats = RunningStats()
    layer_total = {layer: 0 for layer in ALLOWED_LAYERS}

    handles: dict[str, segyio.SegyFile] = {}
    try:
        for name, path in volume_paths.items():
            handle = segyio.open(str(path), "r", ignore_geometry=True)
            handle.mmap()
            if int(grid["TraceIdx"].max()) >= handle.tracecount:
                raise ValueError(f"{name} trace count is smaller than the selected source trace index")
            handles[name] = handle

        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
        spec.format = 5
        spec.samples = sample_axis.astype(np.float32)
        spec.tracecount = int(len(grid))
        delay_ms = int(round(float(sample_axis[0])))
        interval_us = int(round(interval_ms * 1000.0))
        with segyio.open(str(source_sgy), "r", ignore_geometry=True) as source_handle:
            source_handle.mmap()
            with segyio.create(str(partial_sgy), spec) as output_handle:
                output_handle.text[0] = source_handle.text[0]
                output_handle.bin.update(source_handle.bin)
                output_handle.bin[segyio.BinField.Interval] = interval_us
                output_handle.bin[segyio.BinField.Samples] = int(len(sample_axis))
                output_handle.bin[segyio.BinField.Format] = 5
                output_cursor = 0
                for block_index, ix_start in enumerate(range(0, x_line_count, block_x_lines), start=1):
                    block_started = time.time()
                    ix_stop = min(ix_start + block_x_lines, x_line_count)
                    block = grid[grid["IX"].between(ix_start, ix_stop - 1)].sort_values("TraceIdx").copy()
                    source_indices = block["TraceIdx"].to_numpy(dtype=np.int64)
                    attributes = {
                        name: load_attribute_block(handles[name], source_indices, sample_axis) for name in ATTRIBUTE_COLUMNS
                    }
                    predicted, layer_counts = predict_block(
                        attributes=attributes,
                        samples=sample_axis,
                        grid_block=block,
                        models=artifact["models"],
                        density_cap=density_cap,
                    )
                    density_stats.update(predicted)
                    for layer, count in layer_counts.items():
                        layer_total[layer] += int(count)
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
                        output_cursor += 1
                    finite_count = int(np.isfinite(predicted).sum())
                    block_rows.append(
                        {
                            "BlockIndex": block_index,
                            "IXStart": ix_start,
                            "IXStopExclusive": ix_stop,
                            "TraceCount": int(len(block)),
                            "SurfaceValidTraceCount": int(block["SurfaceValid"].sum()),
                            "FinitePredictionCount": finite_count,
                            "NaNCount": int(predicted.size - finite_count),
                            "Sha3PredictionCount": int(layer_counts["沙三段"]),
                            "Sha4PredictionCount": int(layer_counts["沙四段"]),
                            "PredictionMin": float(np.nanmin(predicted)) if finite_count else np.nan,
                            "PredictionMax": float(np.nanmax(predicted)) if finite_count else np.nan,
                            "PredictionMean": float(np.nanmean(predicted)) if finite_count else np.nan,
                            "ElapsedSeconds": float(time.time() - block_started),
                        }
                    )
                    print(
                        f"[step6-predict] block={block_index} ix={ix_start}:{ix_stop} "
                        f"traces={output_cursor}/{len(grid)} finite={finite_count} "
                        f"elapsed_s={block_rows[-1]['ElapsedSeconds']:.1f}",
                        flush=True,
                    )
        partial_sgy.replace(output_sgy)
    finally:
        for handle in handles.values():
            handle.close()

    write_mapping(mapping_path, grid)
    pd.DataFrame(block_rows).to_csv(block_qc_path, index=False, encoding="utf-8-sig")
    sample_trace_indices = np.unique(np.linspace(0, len(grid) - 1, min(32, len(grid)), dtype=np.int64))
    verification = verify_output(output_sgy, len(grid), sample_axis, sample_trace_indices)
    checks = {
        "full_grid_is_rectangular": len(grid) == (int(grid["IX"].max()) + 1) * (int(grid["IY"].max()) + 1),
        "sample_interval_is_2ms": abs(interval_ms - 2.0) < 1.0e-9,
        "curvature_pos_excluded": "CurvaturePos" not in artifact.get("feature_columns", ()),
        "both_layer_predictions_exist": all(layer_total[layer] > 0 for layer in ALLOWED_LAYERS),
        "output_sgy_exists": output_sgy.exists(),
        "mapping_exists": mapping_path.exists(),
        **verification["checks"],
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "model_joblib": str(model_path),
        "model_contract_version": artifact.get("contract_version"),
        "model_logic": artifact.get("model_logic"),
        "target_block": config["target_block"],
        "smoke_max_x_lines": int(args.max_x_lines),
        "output_paths": {
            "density_sgy": str(output_sgy),
            "trace_mapping_npz": str(mapping_path),
            "prediction_block_qc_csv": str(block_qc_path),
            "prediction_summary_json": str(summary_path),
        },
        "grid": {
            "trace_count": int(len(grid)),
            "x_line_count": int(grid["IX"].max()) + 1,
            "y_line_count": int(grid["IY"].max()) + 1,
            "surface_valid_trace_count": int(grid["SurfaceValid"].sum()),
            "surface_valid_fraction": float(grid["SurfaceValid"].mean()),
            "x_min": float(grid["X"].min()),
            "x_max": float(grid["X"].max()),
            "y_min": float(grid["Y"].min()),
            "y_max": float(grid["Y"].max()),
        },
        "sample_axis": {
            "time_min_ms": float(sample_axis[0]),
            "time_max_ms": float(sample_axis[-1]),
            "sample_interval_ms": interval_ms,
            "sample_count": int(len(sample_axis)),
        },
        "feature_columns": list(artifact.get("feature_columns", ())),
        "volume_paths": {name: str(path) for name, path in volume_paths.items()},
        "prediction_counts_by_layer": layer_total,
        "prediction_density_stats": density_stats.summary(),
        "verification": verification,
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(f"[step6-predict] output={output_sgy}", flush=True)
    print(f"[step6-predict] status={summary['status']} elapsed_s={summary['elapsed_seconds']:.1f}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
