from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_a_density_volume.json"
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import assign_surface_times, load_surface_tables, validate_surface_order  # noqa: E402


ALLOWED_LAYERS = ["沙三段", "沙四段"]
ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]
STAT_SUFFIXES = ["Mean", "Std", "Min", "Max"]
NULL_THRESHOLD = -1.0e6
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build formal candidate-A T4-T7 density volume from unified samples and seismic attributes."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    parser.add_argument("--max-traces", type=int, default=0, help="Optional smoke-test trace limit.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def safe_numeric(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_numeric(series: pd.Series | Any) -> pd.Series:
    numeric = safe_numeric(series)
    numeric = numeric.mask(numeric <= NULL_THRESHOLD, np.nan)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric


def finite_stats(series: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
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


def output_paths(output_dir: Path, candidate_name: str = "candidate_a") -> dict[str, Path]:
    prefix = candidate_name.strip() or "candidate_a"
    return {
        "target_attribute_csv": output_dir / f"{prefix}_target_trace_attributes.csv",
        "density_volume_csv": output_dir / f"{prefix}_t4_t7_predicted_density_volume.csv",
        "summary_json": output_dir / f"{prefix}_density_volume_summary.json",
        "training_table_csv": output_dir / f"{prefix}_density_training_table.csv",
    }


def feature_candidates() -> list[str]:
    cols: list[str] = []
    for attr in ATTRIBUTE_COLUMNS:
        cols.append(attr)
        cols.extend(f"{attr}{suffix}" for suffix in STAT_SUFFIXES)
    return cols


def build_target_trace_grid(
    trace_header_csv: Path,
    target_block: dict[str, Any],
    chunksize: int,
    max_traces: int = 0,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(trace_header_csv, usecols=["TraceIdx", "X", "Y"], chunksize=chunksize):
        for column in ["TraceIdx", "X", "Y"]:
            chunk[column] = safe_numeric(chunk[column])
        mask = (
            chunk["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
            & chunk["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
        )
        filtered = chunk.loc[mask, ["TraceIdx", "X", "Y"]].dropna().copy()
        if not filtered.empty:
            parts.append(filtered)
        if max_traces > 0 and sum(len(part) for part in parts) >= max_traces:
            break
    if not parts:
        raise RuntimeError("target block has no trace-header rows")
    grid = pd.concat(parts, ignore_index=True)
    grid = grid.drop_duplicates("TraceIdx").sort_values(["X", "Y", "TraceIdx"]).reset_index(drop=True)
    if max_traces > 0:
        grid = grid.head(max_traces).copy()

    x_rank = {value: idx for idx, value in enumerate(sorted(grid["X"].unique()))}
    y_rank = {value: idx for idx, value in enumerate(sorted(grid["Y"].unique()))}
    grid["IX"] = grid["X"].map(x_rank).astype(int)
    grid["IY"] = grid["Y"].map(y_rank).astype(int)
    grid["TraceRow"] = np.arange(len(grid), dtype=np.int64)
    return grid


def selected_surface_files(surfaces: dict[str, dict[str, object]]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for code, payload in surfaces.items():
        choice = payload["choice"]
        selected[code] = {
            "surface_name": str(choice.surface_name),
            "surface_file": str(choice.surface_path),
            "selection_reason": str(choice.selection_reason),
        }
    return selected


def build_target_layer_windows(
    grid_df: pd.DataFrame,
    layer_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    surfaces = load_surface_tables(layer_dir)
    surface_input = grid_df[["TraceIdx", "X", "Y", "IX", "IY", "TraceRow"]].copy()
    surface_input["TIME"] = 0.0
    assigned = assign_surface_times(surface_input, surfaces)
    assigned = validate_surface_order(assigned, min_thickness=1.0)
    valid = assigned[assigned["Check_All"].fillna(False)].copy()
    if valid.empty:
        raise RuntimeError("no target traces passed T4-T7 surface-order validation")

    for code in ["T4", "T5", "T6", "T7"]:
        valid[f"{code}Time"] = safe_numeric(valid[f"{code}_TIME"])

    common_cols = [
        "TraceIdx",
        "X",
        "Y",
        "IX",
        "IY",
        "TraceRow",
        "T4Time",
        "T5Time",
        "T6Time",
        "T7Time",
    ]
    sha3 = valid[common_cols].copy()
    sha3["LayerGroup"] = "沙三段"
    sha3["TimeWindowMin"] = sha3["T4Time"]
    sha3["TimeWindowMax"] = sha3["T6Time"]

    sha4 = valid[common_cols].copy()
    sha4["LayerGroup"] = "沙四段"
    sha4["TimeWindowMin"] = sha4["T6Time"]
    sha4["TimeWindowMax"] = sha4["T7Time"]

    out = pd.concat([sha3, sha4], ignore_index=True)
    out["LayerThickness"] = safe_numeric(out["TimeWindowMax"]) - safe_numeric(out["TimeWindowMin"])
    out = out[out["LayerThickness"].gt(0)].copy()
    out = out.sort_values(["LayerGroup", "X", "Y", "TraceIdx"]).reset_index(drop=True)

    surface_summary = {
        "layer_dir": str(layer_dir),
        "selected_surface_files": selected_surface_files(surfaces),
        "input_trace_count": int(len(grid_df)),
        "surface_valid_trace_count": int(len(valid)),
        "surface_invalid_trace_count": int(len(grid_df) - len(valid)),
        "layer_window_rows": int(len(out)),
        "surface_manhattan_distance_stats": {
            code: finite_stats(valid[f"{code}_MANHATTAN_DISTANCE"])
            for code in ["T4", "T5", "T6", "T7"]
            if f"{code}_MANHATTAN_DISTANCE" in valid.columns
        },
    }
    return out, surface_summary


def load_sgy_trace_matrix(sgy_path: Path, trace_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as handle:
        handle.mmap()
        samples = np.asarray(handle.samples, dtype=np.float64)
        matrix = np.empty((len(trace_indices), len(samples)), dtype=np.float32)
        for row_idx, trace_idx in enumerate(trace_indices.astype(np.int64)):
            matrix[row_idx, :] = np.asarray(handle.trace[int(trace_idx)], dtype=np.float32)
    return samples, matrix


def compute_layer_window_stats(
    matrix: np.ndarray,
    samples: np.ndarray,
    trace_rows: np.ndarray,
    time_min: np.ndarray,
    time_max: np.ndarray,
) -> dict[str, np.ndarray]:
    n_rows = len(trace_rows)
    mean_arr = np.full(n_rows, np.nan, dtype=np.float64)
    std_arr = np.full(n_rows, np.nan, dtype=np.float64)
    min_arr = np.full(n_rows, np.nan, dtype=np.float64)
    max_arr = np.full(n_rows, np.nan, dtype=np.float64)
    valid_count_arr = np.zeros(n_rows, dtype=np.int32)

    start_idx = np.searchsorted(samples, time_min, side="left")
    end_idx = np.searchsorted(samples, time_max, side="right")
    n_samples = len(samples)
    start_idx = np.clip(start_idx, 0, n_samples)
    end_idx = np.clip(end_idx, 0, n_samples)

    for idx in range(n_rows):
        if end_idx[idx] <= start_idx[idx]:
            continue
        window = matrix[int(trace_rows[idx]), int(start_idx[idx]) : int(end_idx[idx])].astype(np.float64)
        window[window <= NULL_THRESHOLD] = np.nan
        valid = window[np.isfinite(window)]
        if valid.size == 0:
            continue
        mean_arr[idx] = float(valid.mean())
        std_arr[idx] = float(valid.std(ddof=0))
        min_arr[idx] = float(valid.min())
        max_arr[idx] = float(valid.max())
        valid_count_arr[idx] = int(valid.size)

    return {
        "Mean": mean_arr,
        "Std": std_arr,
        "Min": min_arr,
        "Max": max_arr,
        "ValidCount": valid_count_arr.astype(np.float64),
    }


def attach_target_attributes(
    target_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    volume_paths: dict[str, Path],
) -> pd.DataFrame:
    out = target_df.copy()
    trace_indices = grid_df.sort_values("TraceRow")["TraceIdx"].to_numpy(dtype=np.int64)
    trace_rows = out["TraceRow"].to_numpy(dtype=np.int64)
    time_min = safe_numeric(out["TimeWindowMin"]).to_numpy(dtype=np.float64)
    time_max = safe_numeric(out["TimeWindowMax"]).to_numpy(dtype=np.float64)

    for attr in ATTRIBUTE_COLUMNS:
        if attr not in volume_paths:
            continue
        print(f"Extracting target SGY attributes: {attr}", flush=True)
        samples, matrix = load_sgy_trace_matrix(volume_paths[attr], trace_indices)
        stats = compute_layer_window_stats(
            matrix=matrix,
            samples=samples,
            trace_rows=trace_rows,
            time_min=time_min,
            time_max=time_max,
        )
        out[attr] = stats["Mean"]
        for suffix, values in stats.items():
            out[f"{attr}{suffix}"] = values
        del matrix
    return out


def aggregate_training_samples(
    unified_samples_csv: Path,
    chunksize: int,
    candidate_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_cols = ["LayerGroup", "SourceKind", "SourceWellName", "TrackWellName"]
    required_cols = [
        *key_cols,
        "X",
        "Y",
        "DensityLabel",
        "PointConfidence",
    ]
    usecols = set(required_cols + candidate_features)
    accumulator: pd.DataFrame | None = None
    total_rows = 0
    kept_rows = 0
    layer_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    reader = pd.read_csv(
        unified_samples_csv,
        encoding="utf-8-sig",
        chunksize=chunksize,
        low_memory=False,
        usecols=lambda column: column in usecols,
    )
    for chunk in reader:
        total_rows += int(len(chunk))
        for column in required_cols:
            if column not in chunk.columns:
                raise ValueError(f"unified samples missing required column: {column}")
        work = chunk[chunk["LayerGroup"].isin(ALLOWED_LAYERS)].copy()
        if work.empty:
            continue
        for key, value in work["LayerGroup"].value_counts(dropna=False).items():
            layer_counts[str(key)] = layer_counts.get(str(key), 0) + int(value)
        for key, value in work["SourceKind"].value_counts(dropna=False).items():
            source_counts[str(key)] = source_counts.get(str(key), 0) + int(value)

        density = clean_numeric(work["DensityLabel"]).clip(lower=0)
        confidence = clean_numeric(work["PointConfidence"]).fillna(1.0).clip(lower=0.05, upper=1.0)
        x = clean_numeric(work["X"])
        y = clean_numeric(work["Y"])
        density_valid = density.notna() & confidence.notna()
        work = work.loc[density_valid].copy()
        if work.empty:
            continue
        density = density.loc[density_valid]
        confidence = confidence.loc[density_valid]
        x = x.loc[density_valid]
        y = y.loc[density_valid]
        kept_rows += int(len(work))

        agg_work = work[key_cols].copy()
        agg_work["__row_count"] = 1.0
        agg_work["__weight"] = confidence.to_numpy(dtype=np.float64)
        agg_work["__density_weighted"] = density.to_numpy(dtype=np.float64) * agg_work["__weight"].to_numpy(dtype=np.float64)
        agg_work["__confidence_sum"] = confidence.to_numpy(dtype=np.float64)
        agg_work["__x_weighted"] = x.to_numpy(dtype=np.float64) * agg_work["__weight"].to_numpy(dtype=np.float64)
        agg_work["__y_weighted"] = y.to_numpy(dtype=np.float64) * agg_work["__weight"].to_numpy(dtype=np.float64)

        for feature in candidate_features:
            if feature not in work.columns:
                continue
            numeric = clean_numeric(work[feature])
            feature_weight = agg_work["__weight"].where(numeric.notna(), 0.0)
            agg_work[f"__{feature}_sum"] = numeric.fillna(0.0).to_numpy(dtype=np.float64) * feature_weight.to_numpy(dtype=np.float64)
            agg_work[f"__{feature}_weight"] = feature_weight.to_numpy(dtype=np.float64)

        partial = agg_work.groupby(key_cols, dropna=False).sum(numeric_only=True)
        if accumulator is None:
            accumulator = partial
        else:
            accumulator = pd.concat([accumulator, partial], axis=0).groupby(level=key_cols, dropna=False).sum(numeric_only=True)
        if total_rows and total_rows % (chunksize * 10) == 0:
            print(f"Aggregated training sample rows: {total_rows}", flush=True)

    if accumulator is None or accumulator.empty:
        raise RuntimeError("no training samples available after aggregation")

    grouped = accumulator.reset_index()
    weight = grouped["__weight"].replace(0.0, np.nan)
    grouped["DensityLabel"] = grouped["__density_weighted"] / weight
    grouped["PointConfidence"] = grouped["__confidence_sum"] / grouped["__row_count"].replace(0.0, np.nan)
    grouped["X"] = grouped["__x_weighted"] / weight
    grouped["Y"] = grouped["__y_weighted"] / weight
    grouped["TrainingPointCount"] = grouped["__row_count"].astype(np.int64)

    used_features: list[str] = []
    for feature in candidate_features:
        sum_col = f"__{feature}_sum"
        weight_col = f"__{feature}_weight"
        if sum_col not in grouped.columns or weight_col not in grouped.columns:
            continue
        feature_weight = grouped[weight_col].replace(0.0, np.nan)
        grouped[feature] = grouped[sum_col] / feature_weight
        used_features.append(feature)

    keep_cols = [
        *key_cols,
        "X",
        "Y",
        "DensityLabel",
        "PointConfidence",
        "TrainingPointCount",
        *used_features,
    ]
    grouped = grouped[keep_cols].copy()
    summary = {
        "input_rows": int(total_rows),
        "kept_rows": int(kept_rows),
        "aggregated_training_rows": int(len(grouped)),
        "layer_counts": layer_counts,
        "source_kind_counts": source_counts,
        "used_feature_candidates": used_features,
    }
    return grouped, summary


def choose_feature_columns(training_df: pd.DataFrame, target_df: pd.DataFrame, candidates: list[str]) -> list[str]:
    chosen: list[str] = []
    for column in candidates:
        if column not in training_df.columns or column not in target_df.columns:
            continue
        train_non_null = clean_numeric(training_df[column]).notna().mean()
        target_non_null = clean_numeric(target_df[column]).notna().mean()
        if train_non_null >= 0.5 and target_non_null >= 0.9:
            chosen.append(column)
    return chosen


def prepare_features(df: pd.DataFrame, feature_columns: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    out = pd.DataFrame(index=df.index)
    for column in feature_columns:
        out[column] = clean_numeric(df[column]) if column in df.columns else np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = out.median(numeric_only=True)
    out = out.fillna(medians).fillna(0.0)
    return out, medians


def holdout_metrics(
    layer_train: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    unique_tracks = np.array(sorted(layer_train["TrackWellName"].astype(str).unique()))
    if len(unique_tracks) < 5 or len(layer_train) < 20:
        return {"enabled": False, "reason": "not_enough_grouped_tracks"}

    rng = np.random.default_rng(random_state)
    shuffled = unique_tracks.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(0.2 * len(shuffled))))
    val_tracks = set(shuffled[:val_count].tolist())
    val_mask = layer_train["TrackWellName"].astype(str).isin(val_tracks)
    train_part = layer_train.loc[~val_mask].copy()
    val_part = layer_train.loc[val_mask].copy()
    if train_part.empty or val_part.empty:
        return {"enabled": False, "reason": "empty_train_or_validation_split"}

    x_train, medians = prepare_features(train_part, feature_columns)
    x_val, _ = prepare_features(val_part, feature_columns, medians=medians)
    y_train = clean_numeric(train_part["DensityLabel"]).clip(lower=0)
    y_val = clean_numeric(val_part["DensityLabel"]).clip(lower=0)
    train_weight = clean_numeric(train_part["PointConfidence"]).fillna(1.0).clip(lower=0.05, upper=1.0).to_numpy(copy=True)
    train_weight *= np.where(train_part["SourceKind"].astype(str).eq("real_well"), 3.0, 1.0)

    model = HistGradientBoostingRegressor(random_state=random_state, **model_params)
    model.fit(x_train, y_train, sample_weight=train_weight)
    pred = np.maximum(model.predict(x_val), 0.0)
    return {
        "enabled": True,
        "train_rows": int(len(train_part)),
        "validation_rows": int(len(val_part)),
        "validation_track_count": int(len(val_tracks)),
        "validation_mae": float(mean_absolute_error(y_val, pred)),
        "validation_r2": float(r2_score(y_val, pred)) if len(val_part) > 1 else None,
    }


def fit_predict_by_layer(
    training_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, Any],
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {}
    for layer in ALLOWED_LAYERS:
        layer_train = training_df[training_df["LayerGroup"] == layer].copy()
        layer_target = target_df[target_df["LayerGroup"] == layer].copy()
        layer_train["DensityLabel"] = clean_numeric(layer_train["DensityLabel"]).clip(lower=0)
        layer_train = layer_train[layer_train["DensityLabel"].notna()].copy()
        layer_train = layer_train.dropna(subset=["X", "Y"], how="any")
        if layer_train.empty or layer_target.empty:
            continue

        x_train, medians = prepare_features(layer_train, feature_columns)
        x_target, _ = prepare_features(layer_target, feature_columns, medians=medians)
        y_train = layer_train["DensityLabel"]
        sample_weight = clean_numeric(layer_train["PointConfidence"]).fillna(1.0).clip(lower=0.05, upper=1.0).to_numpy(copy=True)
        sample_weight *= np.where(layer_train["SourceKind"].astype(str).eq("real_well"), 3.0, 1.0)

        eval_metrics = holdout_metrics(
            layer_train=layer_train,
            feature_columns=feature_columns,
            model_params=model_params,
            random_state=random_state,
        )

        model = HistGradientBoostingRegressor(random_state=random_state, **model_params)
        model.fit(x_train, y_train, sample_weight=sample_weight)
        train_pred = np.maximum(model.predict(x_train), 0.0)
        pred = np.maximum(model.predict(x_target), 0.0)
        density_cap = float(max(y_train.quantile(0.995), y_train.max()))
        pred = np.clip(pred, 0.0, density_cap)

        work = layer_target.copy()
        work["PredDensity"] = pred
        work["Density"] = pred
        work["PredictionMethod"] = "attribute_driven_regression"
        work["ModelLayer"] = layer
        outputs.append(work)

        metrics[layer] = {
            "train_rows": int(len(layer_train)),
            "train_track_count": int(layer_train["TrackWellName"].nunique()),
            "train_source_well_count": int(layer_train["SourceWellName"].nunique()),
            "source_kind_counts": {
                str(key): int(value)
                for key, value in layer_train["SourceKind"].value_counts(dropna=False).to_dict().items()
            },
            "train_density_stats": finite_stats(y_train),
            "train_fit_mae": float(mean_absolute_error(y_train, train_pred)),
            "train_fit_r2": float(r2_score(y_train, train_pred)) if len(layer_train) > 1 else None,
            "holdout": eval_metrics,
            "prediction_density_stats": finite_stats(pred),
        }

    if not outputs:
        raise RuntimeError("no density-volume predictions generated")
    predicted = pd.concat(outputs, ignore_index=True)
    predicted = predicted.sort_values(["LayerGroup", "X", "Y", "TraceIdx"]).reset_index(drop=True)
    return predicted, metrics


def near_well_match_summary(
    predicted_df: pd.DataFrame,
    training_df: pd.DataFrame,
    radius_m: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in ALLOWED_LAYERS:
        layer_pred = predicted_df[predicted_df["LayerGroup"] == layer].copy()
        layer_real = training_df[
            (training_df["LayerGroup"] == layer)
            & (training_df["SourceKind"].astype(str) == "real_well")
        ].copy()
        layer_real = layer_real.dropna(subset=["X", "Y", "DensityLabel"])
        if layer_pred.empty or layer_real.empty:
            result[layer] = {"checked": False, "reason": "missing_predicted_or_real_rows"}
            continue
        tree = cKDTree(layer_real[["X", "Y"]].to_numpy(dtype=np.float64))
        distances, idx = tree.query(layer_pred[["X", "Y"]].to_numpy(dtype=np.float64), k=1)
        nearest_density = layer_real.iloc[idx]["DensityLabel"].to_numpy(dtype=np.float64)
        mask = distances <= float(radius_m)
        pred_density = layer_pred["PredDensity"].to_numpy(dtype=np.float64)
        if int(mask.sum()) > 1 and np.nanstd(nearest_density[mask]) > 0 and np.nanstd(pred_density[mask]) > 0:
            corr = float(np.corrcoef(pred_density[mask], nearest_density[mask])[0, 1])
        else:
            corr = None
        result[layer] = {
            "checked": True,
            "real_training_points": int(len(layer_real)),
            "target_rows": int(len(layer_pred)),
            "radius_m": float(radius_m),
            "target_rows_within_radius": int(mask.sum()),
            "nearest_distance_stats": finite_stats(distances),
            "mean_pred_density_within_radius": float(np.nanmean(pred_density[mask])) if mask.any() else None,
            "mean_nearest_real_density_within_radius": float(np.nanmean(nearest_density[mask])) if mask.any() else None,
            "mae_pred_vs_nearest_real_within_radius": (
                float(mean_absolute_error(nearest_density[mask], pred_density[mask])) if mask.any() else None
            ),
            "corr_pred_vs_nearest_real_within_radius": corr,
        }
    return result


def write_outputs(
    predicted_df: pd.DataFrame,
    target_attr_df: pd.DataFrame,
    training_df: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    for path in paths.values():
        ensure_dir(path.parent)

    target_attr_df.to_csv(paths["target_attribute_csv"], index=False, encoding="utf-8-sig")
    training_df.to_csv(paths["training_table_csv"], index=False, encoding="utf-8-sig")

    keep_columns = [
        "TraceIdx",
        "X",
        "Y",
        "IX",
        "IY",
        "LayerGroup",
        "TimeWindowMin",
        "TimeWindowMax",
        "LayerThickness",
        "Density",
        "PredDensity",
        "PredictionMethod",
        "ModelLayer",
        *feature_candidates(),
    ]
    keep_columns = [column for column in keep_columns if column in predicted_df.columns]
    predicted_df[keep_columns].to_csv(paths["density_volume_csv"], index=False, encoding="utf-8-sig")


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    target_grid_df: pd.DataFrame,
    target_attr_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    training_df: pd.DataFrame,
    training_summary: dict[str, Any],
    surface_summary: dict[str, Any],
    feature_columns: list[str],
    layer_metrics: dict[str, Any],
    near_well_summary: dict[str, Any],
) -> dict[str, Any]:
    density = clean_numeric(predicted_df["PredDensity"])
    layer_counts = {
        str(key): int(value)
        for key, value in predicted_df["LayerGroup"].value_counts(dropna=False).sort_index().items()
    }
    attr_non_null = {
        column: int(clean_numeric(target_attr_df[column]).notna().sum())
        for column in feature_columns
        if column in target_attr_df.columns
    }
    checks = {
        "has_rows": bool(len(predicted_df) > 0),
        "layers_limited_to_sha3_sha4": bool(set(predicted_df["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS))),
        "density_nonnegative": bool(density.notna().all() and density.min() >= 0.0),
        "density_not_all_zero": bool(density.max() > 0.0),
        "density_not_constant": bool(density.max() - density.min() > 1.0e-9),
        "attribute_driven_model": True,
        "does_not_use_spatial_distance_smoothing": True,
        "does_not_use_old_density_as_label": True,
        "feature_columns_nonempty": bool(len(feature_columns) > 0),
        "target_attribute_coverage": bool(all(value > 0 for value in attr_non_null.values())),
        "near_well_match_checked": bool(any(payload.get("checked") for payload in near_well_summary.values())),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "config_path": str(config_path),
        "unified_samples_csv": str(Path(config["unified_samples_csv"]).resolve()),
        "trace_header_csv": str(Path(config["trace_header_csv"]).resolve()),
        "target_attribute_csv": str(paths["target_attribute_csv"]),
        "density_volume_csv": str(paths["density_volume_csv"]),
        "training_table_csv": str(paths["training_table_csv"]),
        "summary_json": str(paths["summary_json"]),
        "target_block": config["target_block"],
        "model_logic": "attribute_driven_regression",
        "density_supervision": "Step 6 DensityLabel from real_well and single_source virtual_well samples",
        "forbidden_logic": {
            "spatial_smoothing_diffusion": False,
            "distance_decay_density": False,
            "old_candidate_density_volume_as_label": False,
        },
        "allowed_layers": ALLOWED_LAYERS,
        "target_trace_count": int(len(target_grid_df)),
        "target_attribute_rows": int(len(target_attr_df)),
        "predicted_volume_rows": int(len(predicted_df)),
        "layer_group_distribution": layer_counts,
        "pred_density_stats": finite_stats(density),
        "feature_columns": feature_columns,
        "target_attribute_non_null_counts": attr_non_null,
        "training_summary": training_summary,
        "surface_summary": surface_summary,
        "layer_metrics": layer_metrics,
        "near_well_match_summary": near_well_summary,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)

    target_block = dict(config["target_block"])
    output_dir = Path(config["output_dir"]).resolve()
    paths = output_paths(output_dir, str(target_block.get("name", "candidate_a")))
    trace_header_csv = Path(config["trace_header_csv"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    unified_samples_csv = Path(config["unified_samples_csv"]).resolve()
    volume_paths = {key: Path(value).resolve() for key, value in config["volume_paths"].items()}
    chunksize = int(config.get("chunksize", 250000))
    trace_chunksize = int(config.get("trace_chunksize", 500000))
    random_state = int(config.get("random_state", 42))
    model_params = dict(config.get("model", {}))
    near_well_radius_m = float(config.get("near_well_radius_m", 150.0))

    for label, path in [
        ("unified_samples_csv", unified_samples_csv),
        ("trace_header_csv", trace_header_csv),
        ("layer_dir", layer_dir),
        *[(f"volume:{key}", value) for key, value in volume_paths.items()],
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    target_grid_df = build_target_trace_grid(
        trace_header_csv=trace_header_csv,
        target_block=target_block,
        chunksize=trace_chunksize,
        max_traces=int(args.max_traces),
    )
    print(f"Target trace rows: {len(target_grid_df)}", flush=True)
    target_windows_df, surface_summary = build_target_layer_windows(
        grid_df=target_grid_df,
        layer_dir=layer_dir,
    )
    print(f"Target layer-window rows: {len(target_windows_df)}", flush=True)
    target_attr_df = attach_target_attributes(
        target_df=target_windows_df,
        grid_df=target_grid_df,
        volume_paths=volume_paths,
    )

    candidates = feature_candidates()
    training_df, training_summary = aggregate_training_samples(
        unified_samples_csv=unified_samples_csv,
        chunksize=chunksize,
        candidate_features=candidates,
    )
    print(f"Aggregated training rows: {len(training_df)}", flush=True)
    feature_columns = choose_feature_columns(training_df=training_df, target_df=target_attr_df, candidates=candidates)
    if not feature_columns:
        raise RuntimeError("no common non-null attribute feature columns selected")

    predicted_df, layer_metrics = fit_predict_by_layer(
        training_df=training_df,
        target_df=target_attr_df,
        feature_columns=feature_columns,
        model_params=model_params,
        random_state=random_state,
    )
    near_summary = near_well_match_summary(
        predicted_df=predicted_df,
        training_df=training_df,
        radius_m=near_well_radius_m,
    )

    write_outputs(
        predicted_df=predicted_df,
        target_attr_df=target_attr_df,
        training_df=training_df,
        paths=paths,
    )
    summary = build_summary(
        config_path=config_path,
        config=config,
        paths=paths,
        target_grid_df=target_grid_df,
        target_attr_df=target_attr_df,
        predicted_df=predicted_df,
        training_df=training_df,
        training_summary=training_summary,
        surface_summary=surface_summary,
        feature_columns=feature_columns,
        layer_metrics=layer_metrics,
        near_well_summary=near_summary,
    )
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Target attribute CSV: {paths['target_attribute_csv']}")
    print(f"Predicted density volume CSV: {paths['density_volume_csv']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Rows: {len(predicted_df)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
