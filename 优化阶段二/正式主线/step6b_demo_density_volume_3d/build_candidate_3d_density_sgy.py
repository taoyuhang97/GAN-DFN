from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


CURRENT_DIR = Path(__file__).resolve().parent
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import load_surface_tables, validate_surface_order  # noqa: E402


ALLOWED_LAYERS = ["沙三段", "沙四段"]
LAYER_CODE = {"沙三段": 3.0, "沙四段": 4.0}
ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]
NULL_THRESHOLD = -1.0e6
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
FEATURE_COLUMNS = [
    *ATTRIBUTE_COLUMNS,
    "LayerCode",
    "RelativeTimeInLayer",
    "TimeSinceTop",
    "TimeToBase",
    "LayerThickness",
    "TimeMs",
]


@dataclass
class SurfaceLookup:
    code: str
    xy: np.ndarray
    time: np.ndarray
    tree: cKDTree


@dataclass
class LayerModel:
    layer: str
    model: HistGradientBoostingRegressor
    medians: pd.Series
    train_rows: int
    metrics: dict[str, Any]


def make_serializable_models(models: dict[str, LayerModel]) -> dict[str, dict[str, Any]]:
    """Avoid pickling script-local dataclasses as __main__ objects."""
    return {
        layer: {
            "layer": lm.layer,
            "model": lm.model,
            "medians": lm.medians,
            "train_rows": lm.train_rows,
            "metrics": lm.metrics,
        }
        for layer, lm in models.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate 3D fracture-density SGY with time-sample model training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-train-chunks", type=int, default=0, help="Smoke-test cap for unified-sample chunks. 0 means all chunks.")
    parser.add_argument("--max-target-traces", type=int, default=0, help="Smoke-test cap for output traces. 0 means all target traces.")
    parser.add_argument("--output-suffix", type=str, default="", help="Optional suffix for smoke-test outputs.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series | Any, index: pd.Index | None = None) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(pd.Series(series, index=index), errors="coerce")


def clean_numeric(series: pd.Series | Any, index: pd.Index | None = None) -> pd.Series:
    numeric = safe_numeric(series, index=index).replace([np.inf, -np.inf], np.nan)
    return numeric.mask(numeric <= NULL_THRESHOLD, np.nan)


def finite_stats(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
    }


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def output_paths(output_dir: Path, candidate_name: str, suffix: str = "") -> dict[str, Path]:
    tag = candidate_name + suffix
    return {
        "density_sgy": output_dir / f"{tag}_3d_predicted_density.sgy",
        "trace_mapping_npz": output_dir / f"{tag}_3d_trace_mapping.npz",
        "model_joblib": output_dir / f"{tag}_3d_density_models.joblib",
        "summary_json": output_dir / f"{tag}_3d_density_sgy_summary.json",
    }


def load_surface_lookups(layer_dir: Path) -> dict[str, SurfaceLookup]:
    surfaces = load_surface_tables(layer_dir)
    lookups: dict[str, SurfaceLookup] = {}
    for code, payload in surfaces.items():
        table = payload["table"]
        xy = table[["X", "Y"]].to_numpy(dtype=np.float64)
        time = table["Z"].to_numpy(dtype=np.float64)
        lookups[code] = SurfaceLookup(code=code, xy=xy, time=time, tree=cKDTree(xy))
    return lookups


def assign_surface_times_fast(df: pd.DataFrame, lookups: dict[str, SurfaceLookup]) -> pd.DataFrame:
    out = df.copy()
    xy = out[["X", "Y"]].to_numpy(dtype=np.float64)
    valid = np.isfinite(xy).all(axis=1)
    for code, lookup in lookups.items():
        values = np.full(len(out), np.nan, dtype=np.float64)
        if valid.any():
            _, idx = lookup.tree.query(xy[valid], k=1, p=1)
            values[np.where(valid)[0]] = lookup.time[np.asarray(idx, dtype=np.int64)]
        out[f"{code}_TIME"] = values
    out = validate_surface_order(out, min_thickness=1.0)
    return out


def attach_layer_time_features(df: pd.DataFrame, lookups: dict[str, SurfaceLookup]) -> pd.DataFrame:
    work = df.copy()
    for column in ["X", "Y", "TIME"]:
        work[column] = clean_numeric(work[column])
    work = work.dropna(subset=["X", "Y", "TIME"]).copy()
    if work.empty:
        return work
    work = assign_surface_times_fast(work, lookups)
    work = work[work["Check_All"].fillna(False)].copy()
    if work.empty:
        return work
    work["LayerGroup"] = work["LayerGroup"].astype(str)
    is_sha3 = work["LayerGroup"].eq("沙三段")
    is_sha4 = work["LayerGroup"].eq("沙四段")
    work["LayerTopTime"] = np.where(is_sha3, work["T4_TIME"], np.where(is_sha4, work["T6_TIME"], np.nan))
    work["LayerBaseTime"] = np.where(is_sha3, work["T6_TIME"], np.where(is_sha4, work["T7_TIME"], np.nan))
    work["LayerThickness"] = work["LayerBaseTime"] - work["LayerTopTime"]
    work = work[work["LayerThickness"].gt(0)].copy()
    work["TimeSinceTop"] = work["TIME"] - work["LayerTopTime"]
    work["TimeToBase"] = work["LayerBaseTime"] - work["TIME"]
    work = work[work["TimeSinceTop"].ge(0) & work["TimeToBase"].ge(0)].copy()
    work["RelativeTimeInLayer"] = work["TimeSinceTop"] / work["LayerThickness"]
    work["LayerCode"] = work["LayerGroup"].map(LAYER_CODE)
    work["TimeMs"] = work["TIME"]
    return work


def prepare_feature_frame(df: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    out = pd.DataFrame(index=df.index)
    for column in FEATURE_COLUMNS:
        out[column] = clean_numeric(df[column]) if column in df.columns else np.nan
    if medians is None:
        medians = out.median(numeric_only=True)
    out = out.fillna(medians).fillna(0.0)
    return out, medians


def collect_training_arrays(
    unified_samples_csv: Path,
    lookups: dict[str, SurfaceLookup],
    chunksize: int,
    max_train_chunks: int,
) -> tuple[dict[str, list[pd.DataFrame]], dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, Any]]:
    usecols = set([
        "SourceKind",
        "SourceWellName",
        "TrackWellName",
        "X",
        "Y",
        "TIME",
        "LayerGroup",
        "DensityLabel",
        "PointConfidence",
        *ATTRIBUTE_COLUMNS,
    ])
    feature_parts: dict[str, list[pd.DataFrame]] = {layer: [] for layer in ALLOWED_LAYERS}
    y_parts: dict[str, list[np.ndarray]] = {layer: [] for layer in ALLOWED_LAYERS}
    weight_parts: dict[str, list[np.ndarray]] = {layer: [] for layer in ALLOWED_LAYERS}
    row_counts = {layer: 0 for layer in ALLOWED_LAYERS}
    total_rows = 0
    kept_rows = 0
    source_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    well_counts: dict[str, set[str]] = {"real_well": set(), "virtual_well": set()}

    reader = pd.read_csv(
        unified_samples_csv,
        encoding="utf-8-sig",
        chunksize=chunksize,
        low_memory=False,
        usecols=lambda column: column in usecols,
    )
    for chunk_idx, chunk in enumerate(reader, start=1):
        total_rows += int(len(chunk))
        work = chunk[chunk["LayerGroup"].astype(str).isin(ALLOWED_LAYERS)].copy()
        if work.empty:
            continue
        for key, value in work["SourceKind"].astype(str).value_counts(dropna=False).items():
            source_counts[str(key)] = source_counts.get(str(key), 0) + int(value)
        for key, value in work["LayerGroup"].astype(str).value_counts(dropna=False).items():
            layer_counts[str(key)] = layer_counts.get(str(key), 0) + int(value)
        for source_kind, group in work.groupby("SourceKind", dropna=False):
            source_kind_str = str(source_kind)
            if source_kind_str in well_counts:
                well_counts[source_kind_str].update(group["TrackWellName"].dropna().astype(str).unique().tolist())

        work = attach_layer_time_features(work, lookups)
        if work.empty:
            continue
        work["DensityLabel"] = clean_numeric(work["DensityLabel"]).clip(lower=0.0)
        work = work[work["DensityLabel"].notna()].copy()
        if work.empty:
            continue
        kept_rows += int(len(work))

        confidence = clean_numeric(work["PointConfidence"]).fillna(1.0).clip(lower=0.05, upper=1.0).to_numpy(dtype=np.float64)
        source_kind = work["SourceKind"].astype(str).to_numpy()
        weights = confidence * np.where(source_kind == "real_well", 3.0, 0.6)
        for layer in ALLOWED_LAYERS:
            layer_work = work[work["LayerGroup"].astype(str).eq(layer)].copy()
            if layer_work.empty:
                continue
            x_layer, _ = prepare_feature_frame(layer_work)
            idx = layer_work.index.to_numpy()
            feature_parts[layer].append(x_layer.astype(np.float32))
            y_parts[layer].append(layer_work["DensityLabel"].to_numpy(dtype=np.float32))
            weight_parts[layer].append(pd.Series(weights, index=work.index).loc[idx].to_numpy(dtype=np.float32))
            row_counts[layer] += int(len(layer_work))
        print(f"[step6b-3d] training chunk={chunk_idx} total_rows={total_rows} kept_rows={kept_rows}", flush=True)
        if max_train_chunks > 0 and chunk_idx >= max_train_chunks:
            break

    summary = {
        "input_rows_seen": int(total_rows),
        "kept_rows_after_surface_time_filter": int(kept_rows),
        "row_counts_by_layer": {key: int(value) for key, value in row_counts.items()},
        "source_kind_counts_seen": {key: int(value) for key, value in sorted(source_counts.items())},
        "layer_counts_seen": {key: int(value) for key, value in sorted(layer_counts.items())},
        "track_well_counts_seen": {key: int(len(value)) for key, value in sorted(well_counts.items())},
        "max_train_chunks": int(max_train_chunks),
    }
    return feature_parts, y_parts, weight_parts, summary


def fit_layer_models(
    feature_parts: dict[str, list[pd.DataFrame]],
    y_parts: dict[str, list[np.ndarray]],
    weight_parts: dict[str, list[np.ndarray]],
    model_params: dict[str, Any],
    random_state: int,
) -> tuple[dict[str, LayerModel], dict[str, Any]]:
    models: dict[str, LayerModel] = {}
    summary: dict[str, Any] = {}
    for layer in ALLOWED_LAYERS:
        if not feature_parts[layer]:
            raise RuntimeError(f"no training rows for layer: {layer}")
        x_raw = pd.concat(feature_parts[layer], ignore_index=True)
        y = np.concatenate(y_parts[layer]).astype(np.float64)
        weights = np.concatenate(weight_parts[layer]).astype(np.float64)
        x, medians = prepare_feature_frame(x_raw)
        rng = np.random.default_rng(random_state)
        val_size = min(max(int(0.1 * len(x)), 1), 50000) if len(x) > 20 else 0
        if val_size > 0:
            val_idx = rng.choice(len(x), size=val_size, replace=False)
            val_mask = np.zeros(len(x), dtype=bool)
            val_mask[val_idx] = True
            train_mask = ~val_mask
        else:
            train_mask = np.ones(len(x), dtype=bool)
            val_mask = np.zeros(len(x), dtype=bool)
        model = HistGradientBoostingRegressor(random_state=random_state, **model_params)
        model.fit(x.loc[train_mask], y[train_mask], sample_weight=weights[train_mask])
        train_pred = np.maximum(model.predict(x.loc[train_mask]), 0.0)
        metrics: dict[str, Any] = {
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "train_density_stats": finite_stats(y[train_mask]),
            "train_fit_mae": float(mean_absolute_error(y[train_mask], train_pred)),
            "train_fit_r2": float(r2_score(y[train_mask], train_pred)) if int(train_mask.sum()) > 1 else None,
        }
        if val_mask.any():
            val_pred = np.maximum(model.predict(x.loc[val_mask]), 0.0)
            metrics.update(
                {
                    "validation_mae": float(mean_absolute_error(y[val_mask], val_pred)),
                    "validation_r2": float(r2_score(y[val_mask], val_pred)) if int(val_mask.sum()) > 1 else None,
                    "validation_density_stats": finite_stats(y[val_mask]),
                    "validation_prediction_stats": finite_stats(val_pred),
                }
            )
        models[layer] = LayerModel(layer=layer, model=model, medians=medians, train_rows=int(len(x)), metrics=metrics)
        summary[layer] = metrics
    return models, summary


def build_target_trace_grid(trace_header_csv: Path, target_block: dict[str, Any], max_target_traces: int = 0) -> pd.DataFrame:
    df = pd.read_csv(trace_header_csv, usecols=["TraceIdx", "X", "Y"])
    for column in ["TraceIdx", "X", "Y"]:
        df[column] = clean_numeric(df[column])
    mask = (
        df["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
        & df["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
    )
    out = df.loc[mask, ["TraceIdx", "X", "Y"]].dropna().copy()
    if out.empty:
        raise RuntimeError("target block has no trace-header rows")
    out = out.drop_duplicates("TraceIdx").sort_values("TraceIdx").reset_index(drop=True)
    if max_target_traces > 0:
        out = out.head(max_target_traces).copy()
    out["TraceIdx"] = out["TraceIdx"].astype(np.int64)
    x_rank = {value: idx for idx, value in enumerate(sorted(out["X"].unique()))}
    y_rank = {value: idx for idx, value in enumerate(sorted(out["Y"].unique()))}
    out["IX"] = out["X"].map(x_rank).astype(np.int32)
    out["IY"] = out["Y"].map(y_rank).astype(np.int32)
    return out.reset_index(drop=True)


def attach_target_surfaces(trace_df: pd.DataFrame, lookups: dict[str, SurfaceLookup]) -> pd.DataFrame:
    work = trace_df.copy()
    work["TIME"] = 0.0
    work = assign_surface_times_fast(work, lookups)
    work = work[work["Check_All"].fillna(False)].copy()
    if work.empty:
        raise RuntimeError("no target traces passed T4-T7 surface validation")
    return work.reset_index(drop=True)


def build_sample_axis(target_df: pd.DataFrame, sample_interval_ms: float, padding_ms: float) -> np.ndarray:
    time_min = float(np.nanmin(target_df["T4_TIME"].to_numpy(dtype=float))) - float(padding_ms)
    time_max = float(np.nanmax(target_df["T7_TIME"].to_numpy(dtype=float))) + float(padding_ms)
    start = np.floor(time_min / sample_interval_ms) * sample_interval_ms
    stop = np.ceil(time_max / sample_interval_ms) * sample_interval_ms
    return np.arange(start, stop + 0.5 * sample_interval_ms, sample_interval_ms, dtype=np.float64)


def open_volume_context(volume_path: Path):
    handle = segyio.open(str(volume_path), "r", ignore_geometry=True)
    handle.mmap()
    samples = np.asarray(handle.samples, dtype=np.float64)

    def trace_values(trace_idx: int, target_samples: np.ndarray) -> np.ndarray:
        arr = np.asarray(handle.trace[int(trace_idx)], dtype=np.float64)
        arr[arr <= NULL_THRESHOLD] = np.nan
        return np.interp(target_samples, samples, arr, left=np.nan, right=np.nan).astype(np.float32)

    return handle, trace_values


def build_feature_rows_for_trace(
    attr_values: dict[str, np.ndarray],
    sample_axis: np.ndarray,
    layer: str,
    top_time: float,
    base_time: float,
    mask: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(index=np.where(mask)[0])
    for attr in ATTRIBUTE_COLUMNS:
        out[attr] = attr_values[attr][mask]
    layer_thickness = float(base_time - top_time)
    times = sample_axis[mask]
    out["LayerCode"] = LAYER_CODE[layer]
    out["TimeSinceTop"] = times - float(top_time)
    out["TimeToBase"] = float(base_time) - times
    out["LayerThickness"] = layer_thickness
    out["RelativeTimeInLayer"] = out["TimeSinceTop"] / layer_thickness if layer_thickness > 0 else np.nan
    out["TimeMs"] = times
    return out


def predict_trace_density(
    trace_idx: int,
    surface_row: pd.Series,
    sample_axis: np.ndarray,
    volume_trace_readers: dict[str, Callable[[int, np.ndarray], np.ndarray]],
    models: dict[str, LayerModel],
    density_cap: float,
) -> np.ndarray:
    density = np.zeros(len(sample_axis), dtype=np.float32)
    attr_values = {attr: reader(trace_idx, sample_axis) for attr, reader in volume_trace_readers.items()}
    for layer, top_col, base_col in [("沙三段", "T4_TIME", "T6_TIME"), ("沙四段", "T6_TIME", "T7_TIME")]:
        top = float(surface_row[top_col])
        base = float(surface_row[base_col])
        if not np.isfinite(top) or not np.isfinite(base) or base <= top:
            continue
        mask = (sample_axis >= top) & (sample_axis <= base)
        if not mask.any():
            continue
        features = build_feature_rows_for_trace(attr_values, sample_axis, layer, top, base, mask)
        x, _ = prepare_feature_frame(features, medians=models[layer].medians)
        pred = np.maximum(models[layer].model.predict(x), 0.0)
        pred = np.clip(pred, 0.0, density_cap)
        density[mask] = pred.astype(np.float32)
    density[~np.isfinite(density)] = 0.0
    return density


def write_density_sgy(
    output_sgy: Path,
    source_sgy: Path,
    target_df: pd.DataFrame,
    sample_axis: np.ndarray,
    volume_paths: dict[str, Path],
    models: dict[str, LayerModel],
    density_cap: float,
    sample_interval_ms: float,
) -> dict[str, Any]:
    handles = []
    try:
        volume_trace_readers = {}
        for attr, path in volume_paths.items():
            handle, reader = open_volume_context(path)
            handles.append(handle)
            volume_trace_readers[attr] = reader

        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
        spec.format = 5
        spec.samples = (sample_axis - float(sample_axis[0])).astype(np.float32)
        spec.tracecount = int(len(target_df))
        dt_us = int(round(float(sample_interval_ms) * 1000.0))
        delay_ms = int(round(float(sample_axis[0])))

        prediction_stats_parts: list[np.ndarray] = []
        with segyio.open(str(source_sgy), "r", ignore_geometry=True) as src:
            src.mmap()
            with segyio.create(str(output_sgy), spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin.update(src.bin)
                dst.bin[segyio.BinField.Interval] = dt_us
                dst.bin[segyio.BinField.Samples] = int(len(sample_axis))
                dst.bin[segyio.BinField.Format] = 5
                for out_idx, row in target_df.iterrows():
                    source_trace_idx = int(row["TraceIdx"])
                    trace = predict_trace_density(
                        trace_idx=source_trace_idx,
                        surface_row=row,
                        sample_axis=sample_axis,
                        volume_trace_readers=volume_trace_readers,
                        models=models,
                        density_cap=density_cap,
                    )
                    header = dict(src.header[source_trace_idx])
                    header[segyio.TraceField.TRACE_SEQUENCE_FILE] = int(out_idx + 1)
                    header[segyio.TraceField.TRACE_SEQUENCE_LINE] = int(out_idx + 1)
                    header[segyio.TraceField.TRACE_SAMPLE_COUNT] = int(len(sample_axis))
                    header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = dt_us
                    header[segyio.TraceField.DelayRecordingTime] = delay_ms
                    header[segyio.TraceField.SourceX] = int(round(float(row["X"])))
                    header[segyio.TraceField.SourceY] = int(round(float(row["Y"])))
                    dst.header[int(out_idx)] = header
                    dst.trace[int(out_idx)] = trace
                    finite = trace[np.isfinite(trace)]
                    if finite.size:
                        prediction_stats_parts.append(finite.copy())
                    if (out_idx + 1) % 1000 == 0:
                        print(f"[step6b-3d] wrote traces={out_idx + 1}/{len(target_df)}", flush=True)
        all_pred = np.concatenate(prediction_stats_parts) if prediction_stats_parts else np.asarray([], dtype=np.float32)
        return {"prediction_density_stats": finite_stats(all_pred), "output_trace_count": int(len(target_df))}
    finally:
        for handle in handles:
            handle.close()


def write_trace_mapping(path: Path, target_df: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        output_trace_index=np.arange(len(target_df), dtype=np.int32),
        source_trace_idx=target_df["TraceIdx"].to_numpy(dtype=np.int64),
        x=target_df["X"].to_numpy(dtype=np.float64),
        y=target_df["Y"].to_numpy(dtype=np.float64),
        ix=target_df["IX"].to_numpy(dtype=np.int32),
        iy=target_df["IY"].to_numpy(dtype=np.int32),
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    target_block = dict(config["target_block"])
    candidate_name = str(target_block.get("name", "candidate"))
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    suffix = args.output_suffix or ""
    paths = output_paths(output_dir, candidate_name, suffix=suffix)

    unified_samples_csv = Path(config["unified_samples_csv"]).resolve()
    trace_header_csv = Path(config["trace_header_csv"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    source_sgy = Path(config["source_sgy_for_headers"]).resolve()
    volume_paths = {key: Path(value).resolve() for key, value in config["volume_paths"].items() if key in ATTRIBUTE_COLUMNS}
    missing_attrs = sorted(set(ATTRIBUTE_COLUMNS) - set(volume_paths))
    if missing_attrs:
        raise ValueError(f"missing volume paths for attributes: {missing_attrs}")
    for label, path in [("unified_samples_csv", unified_samples_csv), ("trace_header_csv", trace_header_csv), ("layer_dir", layer_dir), ("source_sgy_for_headers", source_sgy), *[(f"volume:{k}", v) for k, v in volume_paths.items()]]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    lookups = load_surface_lookups(layer_dir)
    feature_parts, y_parts, weight_parts, training_scan_summary = collect_training_arrays(
        unified_samples_csv=unified_samples_csv,
        lookups=lookups,
        chunksize=int(config.get("training_chunksize", 250000)),
        max_train_chunks=int(args.max_train_chunks),
    )
    models, model_summary = fit_layer_models(
        feature_parts=feature_parts,
        y_parts=y_parts,
        weight_parts=weight_parts,
        model_params=dict(config.get("model", {})),
        random_state=int(config.get("random_state", 42)),
    )

    target_traces = build_target_trace_grid(trace_header_csv, target_block, max_target_traces=int(args.max_target_traces))
    target_surfaces = attach_target_surfaces(target_traces, lookups)
    sample_interval_ms = float(config.get("sample_interval_ms", 10.0))
    sample_axis = build_sample_axis(target_surfaces, sample_interval_ms=sample_interval_ms, padding_ms=float(config.get("time_padding_ms", 0.0)))
    density_cap = float(config.get("density_cap", 10.0))

    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "models": make_serializable_models(models),
            "model_summary": model_summary,
            "config": config,
        },
        paths["model_joblib"],
    )
    write_trace_mapping(paths["trace_mapping_npz"], target_surfaces)
    sgy_summary = write_density_sgy(
        output_sgy=paths["density_sgy"],
        source_sgy=source_sgy,
        target_df=target_surfaces,
        sample_axis=sample_axis,
        volume_paths=volume_paths,
        models=models,
        density_cap=density_cap,
        sample_interval_ms=sample_interval_ms,
    )

    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "target_block": target_block,
        "output_paths": {key: str(value) for key, value in paths.items()},
        "model_logic": "time_sample_attribute_regression_by_layer",
        "training_input": str(unified_samples_csv),
        "training_uses_all_chunks": int(args.max_train_chunks) == 0,
        "training_scan_summary": training_scan_summary,
        "model_summary": model_summary,
        "source_sgy_for_headers": str(source_sgy),
        "trace_subset_policy": "candidate_block_only_outside_traces_not_written",
        "target_trace_count": int(len(target_surfaces)),
        "source_trace_count_in_target_block": int(len(target_traces)),
        "sample_axis": {
            "time_min_ms": float(sample_axis[0]),
            "time_max_ms": float(sample_axis[-1]),
            "sample_interval_ms": sample_interval_ms,
            "sample_count": int(len(sample_axis)),
        },
        "volume_paths": {key: str(value) for key, value in volume_paths.items()},
        "density_cap": density_cap,
        **sgy_summary,
        "checks": {
            "has_target_traces": int(len(target_surfaces)) > 0,
            "has_sample_axis": int(len(sample_axis)) > 1,
            "models_for_sha3_sha4": sorted(models) == sorted(ALLOWED_LAYERS),
            "density_sgy_exists": paths["density_sgy"].exists(),
            "trace_mapping_exists": paths["trace_mapping_npz"].exists(),
            "no_point_csv_output": True,
        },
    }
    summary["status"] = "pass" if all(bool(v) for v in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step6b-3d] output_sgy={paths['density_sgy']}", flush=True)
    print(f"[step6b-3d] summary={paths['summary_json']}", flush=True)
    print(f"[step6b-3d] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
