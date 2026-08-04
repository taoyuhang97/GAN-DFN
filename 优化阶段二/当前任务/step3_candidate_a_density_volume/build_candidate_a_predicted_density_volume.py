from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
NULL_THRESHOLD = -1.0e6
FEATURE_BASES = ("SEIS_TRUE", "COHERENCE", "ANT_TRACK", "FRACTURE_INV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build predicted candidate-A fracture-density volume by supervised regression."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    numeric = numeric.mask(numeric <= NULL_THRESHOLD, np.nan)
    return numeric


def load_trace_grid(trace_grid_csv: Path) -> pd.DataFrame:
    trace_df = read_csv_flexible(trace_grid_csv)
    for column in ("TraceIdx", "X", "Y", "IX", "IY"):
        if column in trace_df.columns:
            trace_df[column] = safe_numeric(trace_df[column])
    return trace_df


def load_sgy_trace_matrix(sgy_path: Path, trace_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as handle:
        handle.mmap()
        samples = np.asarray(handle.samples, dtype=np.float64)
        matrix = np.zeros((len(trace_indices), len(samples)), dtype=np.float32)
        for row_idx, trace_idx in enumerate(trace_indices.astype(int)):
            matrix[row_idx, :] = np.asarray(handle.trace[trace_idx], dtype=np.float32)
    return samples, matrix


def compute_layer_stat_features(
    values: np.ndarray,
    samples: np.ndarray,
    time_min: np.ndarray,
    time_max: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    records: dict[str, np.ndarray] = {}
    mean_arr = np.full(len(time_min), np.nan, dtype=np.float64)
    std_arr = np.full(len(time_min), np.nan, dtype=np.float64)
    min_arr = np.full(len(time_min), np.nan, dtype=np.float64)
    max_arr = np.full(len(time_min), np.nan, dtype=np.float64)
    valid_count_arr = np.zeros(len(time_min), dtype=np.int32)

    for row_idx in range(len(time_min)):
        mask = (samples >= time_min[row_idx]) & (samples <= time_max[row_idx])
        if not np.any(mask):
            continue
        window = values[row_idx, mask].astype(np.float64)
        window[window <= NULL_THRESHOLD] = np.nan
        valid = window[np.isfinite(window)]
        if valid.size == 0:
            continue
        mean_arr[row_idx] = float(valid.mean())
        std_arr[row_idx] = float(valid.std(ddof=0))
        min_arr[row_idx] = float(valid.min())
        max_arr[row_idx] = float(valid.max())
        valid_count_arr[row_idx] = int(valid.size)

    records[f"{prefix}_mean"] = mean_arr
    records[f"{prefix}_std"] = std_arr
    records[f"{prefix}_min"] = min_arr
    records[f"{prefix}_max"] = max_arr
    records[f"{prefix}_valid_count"] = valid_count_arr.astype(np.float64)
    return pd.DataFrame(records)


def build_target_trace_attribute_table(
    trace_grid_df: pd.DataFrame,
    surface_volume_df: pd.DataFrame,
    sgy_paths: dict[str, Path],
) -> pd.DataFrame:
    layer_windows = (
        surface_volume_df.groupby(["LayerGroup", "X", "Y"], as_index=False)
        .agg(
            RepresentativeT4Time=("RepresentativeT4Time", "median"),
            RepresentativeT5Time=("RepresentativeT5Time", "median"),
            RepresentativeT6Time=("RepresentativeT6Time", "median"),
            RepresentativeT7Time=("RepresentativeT7Time", "median"),
        )
    )
    merged = trace_grid_df.merge(layer_windows, on=["X", "Y"], how="inner")
    outputs: list[pd.DataFrame] = []
    trace_indices = merged["TraceIdx"].to_numpy(dtype=np.int64)

    sgy_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feature_name, sgy_path in sgy_paths.items():
        sgy_cache[feature_name] = load_sgy_trace_matrix(sgy_path=sgy_path, trace_indices=trace_indices)

    for layer_group, layer_df in merged.groupby("LayerGroup", sort=False):
        work = layer_df.reset_index(drop=True).copy()
        if layer_group == "沙三段":
            time_min = safe_numeric(work["RepresentativeT4Time"]).to_numpy(dtype=np.float64)
            time_max = safe_numeric(work["RepresentativeT6Time"]).to_numpy(dtype=np.float64)
        elif layer_group == "沙四段":
            time_min = safe_numeric(work["RepresentativeT6Time"]).to_numpy(dtype=np.float64)
            time_max = safe_numeric(work["RepresentativeT7Time"]).to_numpy(dtype=np.float64)
        else:
            continue
        work["TimeWindowMin"] = time_min
        work["TimeWindowMax"] = time_max
        work["LayerThickness"] = time_max - time_min

        for feature_name, (samples, matrix) in sgy_cache.items():
            layer_matrix = matrix[work.index.to_numpy(dtype=int), :]
            feature_df = compute_layer_stat_features(
                values=layer_matrix,
                samples=samples,
                time_min=time_min,
                time_max=time_max,
                prefix=feature_name,
            )
            work = pd.concat([work.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        outputs.append(work)
    if not outputs:
        raise RuntimeError("failed to build target trace attribute table")
    return pd.concat(outputs, ignore_index=True)


def build_training_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    for column in ["DensityLabel", "T4Time", "T6Time", "T7Time"]:
        if column in sample_df.columns:
            sample_df[column] = safe_numeric(sample_df[column])
    for column in FEATURE_BASES + ("COHERENCE_WIN_MEAN", "COHERENCE_WIN_STD", "COHERENCE_WIN_MIN", "COHERENCE_WIN_MAX", "COHERENCE_WIN_VALID_COUNT", "ANT_TRACK_WIN_MEAN", "ANT_TRACK_WIN_STD", "ANT_TRACK_WIN_MIN", "ANT_TRACK_WIN_MAX", "ANT_TRACK_WIN_VALID_COUNT"):
        if column in sample_df.columns:
            sample_df[column] = clean_numeric_series(sample_df[column])

    group_cols = ["LayerGroup", "SourceWellName", "TrackWellName"]
    agg_map: dict[str, Any] = {
        "DensityLabel": "mean",
        "SourceKind": "first",
        "X": "median",
        "Y": "median",
        "PointConfidence": "mean",
        "WellConfidence": "mean",
        "T4Time": "median",
        "T6Time": "median",
        "T7Time": "median",
    }
    for base in FEATURE_BASES:
        if base in sample_df.columns:
            agg_map[base] = ["mean", "std", "min", "max"]
    extra_cols = [
        "COHERENCE_WIN_MEAN",
        "COHERENCE_WIN_STD",
        "COHERENCE_WIN_MIN",
        "COHERENCE_WIN_MAX",
        "COHERENCE_WIN_VALID_COUNT",
        "ANT_TRACK_WIN_MEAN",
        "ANT_TRACK_WIN_STD",
        "ANT_TRACK_WIN_MIN",
        "ANT_TRACK_WIN_MAX",
        "ANT_TRACK_WIN_VALID_COUNT",
    ]
    for column in extra_cols:
        if column in sample_df.columns:
            agg_map[column] = "mean"

    grouped = sample_df.groupby(group_cols).agg(agg_map)
    grouped.columns = [
        "_".join(part for part in col if part).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()
    grouped["LayerThickness"] = np.where(
        grouped["LayerGroup"] == "沙三段",
        grouped["T6Time_median"] - grouped["T4Time_median"],
        grouped["T7Time_median"] - grouped["T6Time_median"],
    )
    return grouped


def choose_feature_columns(training_df: pd.DataFrame) -> list[str]:
    candidates = [
        "SEIS_TRUE_mean",
        "SEIS_TRUE_std",
        "SEIS_TRUE_min",
        "SEIS_TRUE_max",
        "COHERENCE_mean",
        "COHERENCE_std",
        "COHERENCE_min",
        "COHERENCE_max",
        "ANT_TRACK_mean",
        "ANT_TRACK_std",
        "ANT_TRACK_min",
        "ANT_TRACK_max",
        "LayerThickness",
    ]
    return [column for column in candidates if column in training_df.columns]


def align_target_feature_columns(target_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = target_df.copy()
    rename_candidates = {
        "SEIS_TRUE_mean": "SEIS_TRUE_mean",
        "SEIS_TRUE_std": "SEIS_TRUE_std",
        "SEIS_TRUE_min": "SEIS_TRUE_min",
        "SEIS_TRUE_max": "SEIS_TRUE_max",
        "COHERENCE_mean": "COHERENCE_mean",
        "COHERENCE_std": "COHERENCE_std",
        "COHERENCE_min": "COHERENCE_min",
        "COHERENCE_max": "COHERENCE_max",
        "ANT_TRACK_mean": "ANT_TRACK_mean",
        "ANT_TRACK_std": "ANT_TRACK_std",
        "ANT_TRACK_min": "ANT_TRACK_min",
        "ANT_TRACK_max": "ANT_TRACK_max",
        "COHERENCE_WIN_MEAN": "COHERENCE_WIN_MEAN_mean",
        "COHERENCE_WIN_STD": "COHERENCE_WIN_STD_mean",
        "COHERENCE_WIN_MIN": "COHERENCE_WIN_MIN_mean",
        "COHERENCE_WIN_MAX": "COHERENCE_WIN_MAX_mean",
        "COHERENCE_WIN_VALID_COUNT": "COHERENCE_WIN_VALID_COUNT_mean",
        "ANT_TRACK_WIN_MEAN": "ANT_TRACK_WIN_MEAN_mean",
        "ANT_TRACK_WIN_STD": "ANT_TRACK_WIN_STD_mean",
        "ANT_TRACK_WIN_MIN": "ANT_TRACK_WIN_MIN_mean",
        "ANT_TRACK_WIN_MAX": "ANT_TRACK_WIN_MAX_mean",
        "ANT_TRACK_WIN_VALID_COUNT": "ANT_TRACK_WIN_VALID_COUNT_mean",
    }
    for source_col, target_col in rename_candidates.items():
        if source_col in out.columns and target_col not in out.columns:
            out[target_col] = out[source_col]
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def fit_predict_by_layer(
    training_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {}
    aligned_target_df = align_target_feature_columns(target_df=target_df, feature_columns=feature_columns)

    for layer_group, layer_train in training_df.groupby("LayerGroup", sort=False):
        layer_target = aligned_target_df[aligned_target_df["LayerGroup"] == layer_group].copy()
        if layer_target.empty:
            continue

        valid_train = layer_train.dropna(subset=["DensityLabel_mean"]).copy()
        valid_train = valid_train.dropna(subset=[col for col in feature_columns if col in valid_train.columns], how="any")
        X_train = valid_train[feature_columns]
        y_train = valid_train["DensityLabel_mean"]
        sample_weight = np.where(valid_train["SourceKind_first"] == "real_well", 3.0, 1.0)

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        train_pred = model.predict(X_train)
        metrics[layer_group] = {
            "train_rows": int(len(valid_train)),
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "train_r2": float(r2_score(y_train, train_pred)) if len(valid_train) > 1 else None,
            "source_well_count": int(valid_train["SourceWellName"].nunique()),
            "source_kind_counts": {
                str(key): int(value)
                for key, value in valid_train["SourceKind_first"].value_counts().to_dict().items()
            },
            "track_count": int(valid_train["TrackWellName"].nunique()),
        }

        work_target = layer_target.copy()
        for col in feature_columns:
            work_target[col] = safe_numeric(work_target[col])
        work_target["PredDensity"] = model.predict(work_target[feature_columns])
        work_target["PredDensity"] = np.maximum(work_target["PredDensity"], 0.0)
        outputs.append(work_target)

    if not outputs:
        raise RuntimeError("no layer predictions generated")
    return pd.concat(outputs, ignore_index=True), metrics


def main() -> None:
    args = parse_args()
    config = read_json(Path(args.config).resolve())

    sample_csv = Path(config["sample_csv"]).resolve()
    trace_grid_csv = Path(config["trace_grid_csv"]).resolve()
    old_volume_csv = Path(config["reference_volume_csv"]).resolve()
    output_attr_csv = Path(config["target_attribute_csv"]).resolve()
    output_volume_csv = Path(config["output_volume_csv"]).resolve()
    summary_json = Path(config["summary_json"]).resolve()
    sgy_paths = {key: Path(value).resolve() for key, value in config["sgy_paths"].items()}

    sample_df = read_csv_flexible(sample_csv)
    trace_df = load_trace_grid(trace_grid_csv=trace_grid_csv)
    ref_volume_df = read_csv_flexible(old_volume_csv)

    target_attr_df = build_target_trace_attribute_table(
        trace_grid_df=trace_df,
        surface_volume_df=ref_volume_df,
        sgy_paths=sgy_paths,
    )
    ensure_parent(output_attr_csv)
    target_attr_df.to_csv(output_attr_csv, index=False, encoding="utf-8-sig")

    training_df = build_training_table(sample_df=sample_df.copy())
    feature_columns = choose_feature_columns(training_df=training_df)
    predicted_df, metrics = fit_predict_by_layer(
        training_df=training_df,
        target_df=target_attr_df,
        feature_columns=feature_columns,
    )

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
        "PredDensity",
    ] + feature_columns
    predicted_df = predicted_df[[column for column in keep_columns if column in predicted_df.columns]].copy()
    predicted_df = predicted_df.sort_values(["LayerGroup", "X", "Y"]).reset_index(drop=True)

    ensure_parent(output_volume_csv)
    predicted_df.to_csv(output_volume_csv, index=False, encoding="utf-8-sig")

    summary = {
        "sample_csv": str(sample_csv),
        "trace_grid_csv": str(trace_grid_csv),
        "reference_volume_csv": str(old_volume_csv),
        "target_attribute_csv": str(output_attr_csv),
        "output_volume_csv": str(output_volume_csv),
        "summary_json": str(summary_json),
        "feature_columns": feature_columns,
        "training_table_rows": int(len(training_df)),
        "target_attribute_rows": int(len(target_attr_df)),
        "predicted_volume_rows": int(len(predicted_df)),
        "layer_metrics": metrics,
        "pred_density_stats": {
            "min": float(predicted_df["PredDensity"].min()),
            "max": float(predicted_df["PredDensity"].max()),
            "mean": float(predicted_df["PredDensity"].mean()),
        },
    }
    ensure_parent(summary_json)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Target attribute CSV: {output_attr_csv}")
    print(f"Predicted density volume CSV: {output_volume_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Rows: {len(predicted_df)}")


if __name__ == "__main__":
    main()
