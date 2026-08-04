from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-A local T4-T7 fracture-density volume on trace-header XY grid."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_layer_windows(sample_df: pd.DataFrame) -> pd.DataFrame:
    windows = (
        sample_df.groupby(["LayerGroup", "X", "Y"], as_index=False)
        .agg(
            TimeMin=("TIME", "min"),
            TimeMax=("TIME", "max"),
            T4Time=("T4Time", "median"),
            T5Time=("T5Time", "median"),
            T6Time=("T6Time", "median"),
            T7Time=("T7Time", "median"),
        )
        .sort_values(["LayerGroup", "X", "Y"])
        .reset_index(drop=True)
    )
    return windows


def aggregate_grid_samples(
    sample_df: pd.DataFrame,
    xy_grid_df: pd.DataFrame,
    layer_group: str,
    k_neighbors: int,
) -> pd.DataFrame:
    layer_df = sample_df[sample_df["LayerGroup"] == layer_group].copy()
    if layer_df.empty:
        return pd.DataFrame()

    sample_xy = layer_df[["X", "Y"]].to_numpy(dtype=float)
    grid_xy = xy_grid_df[["X", "Y"]].to_numpy(dtype=float)
    tree = cKDTree(sample_xy)
    distances, neighbor_idx = tree.query(grid_xy, k=min(k_neighbors, len(layer_df)), p=2)

    if distances.ndim == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]

    neighbor_density = layer_df["DensityLabel"].to_numpy(dtype=float)[neighbor_idx]
    neighbor_time = layer_df["TIME"].to_numpy(dtype=float)[neighbor_idx]
    neighbor_conf = layer_df.get("PointConfidence", pd.Series(1.0, index=layer_df.index)).to_numpy(dtype=float)[neighbor_idx]
    neighbor_source = layer_df["SourceKind"].to_numpy(dtype=object)[neighbor_idx]

    inv_distance = 1.0 / np.maximum(distances, 1.0)
    source_weight = np.where(neighbor_source == "real_well", 2.0, 1.0)
    weights = inv_distance * np.where(np.isfinite(neighbor_conf), np.maximum(neighbor_conf, 0.05), 1.0) * source_weight
    weights = np.where(np.isfinite(neighbor_density), weights, 0.0)

    density_sum = np.sum(neighbor_density * weights, axis=1)
    weight_sum = np.sum(weights, axis=1)
    density_pred = np.divide(
        density_sum,
        weight_sum,
        out=np.full_like(density_sum, np.nan, dtype=float),
        where=weight_sum > 0,
    )
    time_pred = np.divide(
        np.sum(neighbor_time * weights, axis=1),
        weight_sum,
        out=np.full(grid_xy.shape[0], np.nan, dtype=float),
        where=weight_sum > 0,
    )
    nearest_distance = distances[:, 0]
    real_neighbor_count = np.sum(neighbor_source == "real_well", axis=1).astype(int)
    positive_neighbor_count = np.sum(neighbor_density > 0, axis=1).astype(int)

    out = xy_grid_df.copy()
    out["LayerGroup"] = layer_group
    out["DensityPred"] = density_pred
    out["DensityPredNonNegative"] = np.where(np.isfinite(density_pred), np.maximum(density_pred, 0.0), np.nan)
    out["TimePred"] = time_pred
    out["NearestSampleDistance"] = nearest_distance
    out["NeighborWeightSum"] = weight_sum
    out["NeighborCount"] = int(neighbor_density.shape[1])
    out["RealNeighborCount"] = real_neighbor_count
    out["PositiveNeighborCount"] = positive_neighbor_count
    return out


def attach_representative_times(
    volume_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    windows = build_layer_windows(sample_df)
    outputs: list[pd.DataFrame] = []
    for layer_group, layer_volume in volume_df.groupby("LayerGroup", sort=False):
        layer_windows = windows[windows["LayerGroup"] == layer_group].copy()
        if layer_windows.empty:
            outputs.append(layer_volume.copy())
            continue
        tree = cKDTree(layer_windows[["X", "Y"]].to_numpy(dtype=float))
        distances, idx = tree.query(layer_volume[["X", "Y"]].to_numpy(dtype=float), k=1, p=1)
        matched = layer_windows.iloc[idx].reset_index(drop=True)
        enriched = layer_volume.reset_index(drop=True).copy()
        enriched["RepresentativeTimeMin"] = matched["TimeMin"]
        enriched["RepresentativeTimeMax"] = matched["TimeMax"]
        enriched["RepresentativeT4Time"] = matched["T4Time"]
        enriched["RepresentativeT5Time"] = matched["T5Time"]
        enriched["RepresentativeT6Time"] = matched["T6Time"]
        enriched["RepresentativeT7Time"] = matched["T7Time"]
        enriched["RepresentativeLayerWindowDistance"] = distances
        outputs.append(enriched)
    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = read_json(Path(args.config).resolve())

    sample_csv = Path(config["sample_csv"]).resolve()
    trace_xy_csv = Path(config["trace_xy_csv"]).resolve()
    output_csv = Path(config["output_csv"]).resolve()
    summary_json = Path(config["summary_json"]).resolve()
    target_block = dict(config["target_block"])
    k_neighbors = int(config.get("k_neighbors", 8))
    layer_groups = list(config.get("layer_groups", ["沙三段", "沙四段"]))

    sample_df = read_csv_flexible(sample_csv)
    trace_df = read_csv_flexible(trace_xy_csv)

    for column in ["X", "Y", "TIME", "DensityLabel", "T4Time", "T5Time", "T6Time", "T7Time"]:
        if column in sample_df.columns:
            sample_df[column] = safe_numeric(sample_df[column])
    for column in ["X", "Y"]:
        trace_df[column] = safe_numeric(trace_df[column])

    sample_df = sample_df[
        sample_df["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
        & sample_df["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
    ].copy()
    trace_df = trace_df[
        trace_df["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
        & trace_df["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
    ].copy()

    volume_parts: list[pd.DataFrame] = []
    for layer_group in layer_groups:
        part = aggregate_grid_samples(
            sample_df=sample_df,
            xy_grid_df=trace_df,
            layer_group=layer_group,
            k_neighbors=k_neighbors,
        )
        if not part.empty:
            volume_parts.append(part)

    if not volume_parts:
        raise RuntimeError("no density-volume rows generated")

    volume_df = pd.concat(volume_parts, ignore_index=True)
    volume_df = attach_representative_times(volume_df=volume_df, sample_df=sample_df)
    volume_df = volume_df.sort_values(["LayerGroup", "X", "Y"]).reset_index(drop=True)

    ensure_parent(output_csv)
    volume_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "sample_csv": str(sample_csv),
        "trace_xy_csv": str(trace_xy_csv),
        "output_csv": str(output_csv),
        "summary_json": str(summary_json),
        "target_block": target_block,
        "k_neighbors": k_neighbors,
        "layer_groups": layer_groups,
        "trace_grid_point_count": int(len(trace_df)),
        "trace_grid_unique_x": int(trace_df["X"].nunique()),
        "trace_grid_unique_y": int(trace_df["Y"].nunique()),
        "input_sample_row_count": int(len(sample_df)),
        "output_row_count": int(len(volume_df)),
        "output_density_stats": {
            "min": float(volume_df["DensityPredNonNegative"].min()),
            "max": float(volume_df["DensityPredNonNegative"].max()),
            "mean": float(volume_df["DensityPredNonNegative"].mean()),
        },
        "layer_group_distribution": {
            str(key): int(value)
            for key, value in volume_df["LayerGroup"].value_counts(dropna=False).sort_index().items()
        },
        "source_kind_distribution_in_samples": {
            str(key): int(value)
            for key, value in sample_df["SourceKind"].value_counts(dropna=False).sort_index().items()
        },
    }
    ensure_parent(summary_json)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Output CSV: {output_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Rows: {len(volume_df)}")


if __name__ == "__main__":
    main()
