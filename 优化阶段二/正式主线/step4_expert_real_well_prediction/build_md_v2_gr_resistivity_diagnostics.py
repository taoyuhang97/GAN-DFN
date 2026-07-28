"""Run comparable GR + deep/near resistivity diagnostics on MD-v2 samples."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gan_dfn_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose GR-conditioned deep/near resistivity response on MD-v2 labels.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def robust_zscore(values: pd.Series) -> pd.Series:
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - median) / scale


def add_features(frame: pd.DataFrame, gr_bins: int) -> pd.DataFrame:
    out = frame.copy().sort_values("MD").reset_index(drop=True)
    for column in ("MD", "GR", "RDeep", "RNear", "Density", "GT_POINT_FLAG"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["LogRDeep"] = np.log10(out["RDeep"])
    out["LogRNear"] = np.log10(out["RNear"])
    out["DeltaLogR"] = out["LogRDeep"] - out["LogRNear"]
    out["AbsDeltaLogR"] = out["DeltaLogR"].abs()
    denominator = out["RDeep"] + out["RNear"]
    out["NormalizedContrast"] = (out["RDeep"] - out["RNear"]) / denominator.where(denominator.abs() > 1e-12)
    out["AbsNormalizedContrast"] = out["NormalizedContrast"].abs()
    out["GRBin"] = np.nan
    out["DeltaLogRResidual"] = np.nan
    for _, idx in out.groupby("StrataName", dropna=False).groups.items():
        part = out.loc[idx]
        valid = part["GR"].notna() & part["DeltaLogR"].notna()
        if int(valid.sum()) < gr_bins:
            continue
        ranked = part.loc[valid, "GR"].rank(method="first")
        bins = pd.qcut(ranked, q=gr_bins, labels=False, duplicates="drop").astype(float)
        out.loc[bins.index, "GRBin"] = bins
        baseline = part.loc[valid].assign(_bin=bins).groupby("_bin")["DeltaLogR"].transform("median")
        out.loc[bins.index, "DeltaLogRResidual"] = part.loc[valid, "DeltaLogR"].to_numpy() - baseline.to_numpy()
    out["AbsDeltaLogRResidual"] = out["DeltaLogRResidual"].abs()
    out["HasFractureTruth"] = out["Density"].fillna(0.0).gt(0.0).astype(int)
    out["HasPointTruth"] = out["GT_POINT_FLAG"].fillna(0.0).gt(0.0).astype(int)
    for column in ("DeltaLogR", "AbsDeltaLogR", "AbsDeltaLogRResidual", "AbsNormalizedContrast"):
        out[f"{column}RobustZ"] = out.groupby("StrataName", dropna=False)[column].transform(robust_zscore)
    return out


def auc_values(y: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(score) & np.isfinite(y)
    if int(valid.sum()) < 2 or np.unique(y[valid]).size < 2:
        return {"auc": None, "best_direction_auc": None, "row_count": int(valid.sum())}
    auc = float(roc_auc_score(y[valid].astype(int), score[valid]))
    return {"auc": auc, "best_direction_auc": max(auc, 1.0 - auc), "row_count": int(valid.sum())}


def block_bootstrap_auc(
    frame: pd.DataFrame,
    score_col: str,
    truth_col: str,
    block_m: float,
    iterations: int,
    rng: np.random.Generator,
    orientation: int,
) -> dict[str, float | int | None]:
    work = frame[["MD", score_col, truth_col]].dropna().copy()
    if work.empty or work[truth_col].nunique() < 2:
        return {"median": None, "ci_low": None, "ci_high": None, "valid_iterations": 0}
    work["Block"] = np.floor((work["MD"] - work["MD"].min()) / block_m).astype(int)
    blocks = list(work.groupby("Block"))
    values: list[float] = []
    for _ in range(iterations):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        sample = pd.concat([blocks[idx][1] for idx in selected], ignore_index=True)
        if sample[truth_col].nunique() < 2:
            continue
        auc = float(roc_auc_score(sample[truth_col].astype(int), sample[score_col] * orientation))
        values.append(auc)
    if not values:
        return {"median": None, "ci_low": None, "ci_high": None, "valid_iterations": 0}
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "valid_iterations": len(values),
    }


def binned_frame(frame: pd.DataFrame, bin_m: float) -> pd.DataFrame:
    out = frame.copy()
    out["MDBin"] = np.floor(out["MD"] / bin_m).astype(int)
    aggregations = {
        "MD": "mean",
        "Density": "max",
        "HasFractureTruth": "max",
        "HasPointTruth": "max",
        "DeltaLogR": "mean",
        "AbsDeltaLogR": "mean",
        "AbsDeltaLogRResidual": "mean",
        "AbsNormalizedContrast": "mean",
    }
    return out.groupby("MDBin", as_index=False).agg(aggregations)


def metric_bundle(
    frame: pd.DataFrame,
    config: dict[str, Any],
    seed_offset: int = 0,
    truth_col: str = "HasFractureTruth",
) -> dict[str, Any]:
    score_columns = ("DeltaLogR", "AbsDeltaLogR", "AbsDeltaLogRResidual", "AbsNormalizedContrast")
    rng = np.random.default_rng(int(config.get("random_state", 42)) + seed_offset)
    row_metrics = {column: auc_values(frame[truth_col].to_numpy(float), frame[column].to_numpy(float)) for column in score_columns}
    orientations = {
        column: 1 if values["auc"] is None or float(values["auc"]) >= 0.5 else -1
        for column, values in row_metrics.items()
    }
    bootstrap = {
        column: block_bootstrap_auc(
            frame,
            column,
            truth_col,
            float(config.get("bootstrap_block_m", 10.0)),
            int(config.get("bootstrap_iterations", 1000)),
            rng,
            orientations[column],
        )
        for column in score_columns
    }
    coarse = binned_frame(frame, float(config.get("depth_bin_m", 1.0)))
    binned_metrics = {
        column: auc_values(coarse[truth_col].to_numpy(float), coarse[column].to_numpy(float) * orientations[column])
        for column in score_columns
    }
    return {
        "truth_column": truth_col,
        "fixed_orientations": orientations,
        "row_auc": row_metrics,
        "block_bootstrap_fixed_direction_auc": bootstrap,
        "depth_bin_fixed_direction_auc": binned_metrics,
        "depth_bin_count": len(coarse),
    }


def plot_full_depth(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 12), sharey=True, gridspec_kw={"wspace": 0.28})
    md = frame["MD"]
    axes[0].plot(frame["GR"], md, color="#2e7d32", linewidth=0.8)
    axes[0].set_title("Same-pass GR")
    axes[0].set_xlabel("GR")
    axes[0].set_ylabel("MD / m")
    axes[1].semilogx(frame["RDeep"], md, color="#c95f0b", linewidth=0.8, label=frame["DeepSourceCurve"].iloc[0])
    axes[1].semilogx(frame["RNear"], md, color="#1f6fb2", linewidth=0.8, label=frame["NearSourceCurve"].iloc[0])
    axes[1].set_title("Deep / near resistivity")
    axes[1].set_xlabel("Resistivity")
    axes[1].legend(fontsize=8)
    axes[2].plot(frame["DeltaLogR"], md, color="#6f42a8", linewidth=0.8)
    axes[2].axvline(0.0, color="#666666", linewidth=0.7)
    positive = frame["HasFractureTruth"].eq(1)
    axes[2].scatter(frame.loc[positive, "DeltaLogR"], md[positive], s=6, color="#c62828", alpha=0.72)
    axes[2].set_title("log10(deep / near)")
    axes[2].set_xlabel("DeltaLogR")
    axes[3].fill_betweenx(md, 0.0, frame["Density"], color="#d95f82", alpha=0.55)
    point = frame["HasPointTruth"].eq(1)
    axes[3].scatter(frame.loc[point, "Density"], md[point], s=16, facecolors="none", edgecolors="#111111", linewidths=0.7)
    axes[3].set_title("Imaging density / raw points")
    axes[3].set_xlabel("Density")
    for axis in axes:
        axis.set_ylim(float(md.max()), float(md.min()))
        axis.grid(True, alpha=0.16)
    fig.suptitle(f"{frame['WellName'].iloc[0]} | {frame['PairType'].iloc[0]} | MD-v2", fontsize=13, y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def window_centers(frame: pd.DataFrame, count: int = 3) -> list[float]:
    candidates = frame[frame["Density"].fillna(0.0).gt(0.0)].sort_values("Density", ascending=False)
    centers: list[float] = []
    for md in candidates["MD"].to_numpy(float):
        if all(abs(md - existing) >= 20.0 for existing in centers):
            centers.append(float(md))
        if len(centers) >= count:
            break
    return sorted(centers)


def plot_windows(frame: pd.DataFrame, output_path: Path) -> None:
    centers = window_centers(frame)
    if not centers:
        return
    fig, axes = plt.subplots(len(centers), 4, figsize=(18, 4.0 * len(centers)), sharey="row", gridspec_kw={"wspace": 0.28})
    if len(centers) == 1:
        axes = np.asarray([axes])
    for row_idx, center in enumerate(centers):
        part = frame[frame["MD"].between(center - 10.0, center + 10.0)]
        md = part["MD"]
        axes[row_idx, 0].plot(part["GR"], md, color="#2e7d32", linewidth=0.9)
        axes[row_idx, 1].semilogx(part["RDeep"], md, color="#c95f0b", linewidth=0.9)
        axes[row_idx, 1].semilogx(part["RNear"], md, color="#1f6fb2", linewidth=0.9)
        axes[row_idx, 2].plot(part["DeltaLogR"], md, color="#6f42a8", linewidth=0.9)
        axes[row_idx, 2].axvline(0.0, color="#666666", linewidth=0.7)
        axes[row_idx, 3].fill_betweenx(md, 0.0, part["Density"], color="#d95f82", alpha=0.55)
        point = part["HasPointTruth"].eq(1)
        axes[row_idx, 3].scatter(part.loc[point, "Density"], md[point], s=17, facecolors="none", edgecolors="#111111")
        for axis in axes[row_idx]:
            axis.set_ylim(float(md.max()), float(md.min()))
            axis.grid(True, alpha=0.16)
        axes[row_idx, 0].set_ylabel(f"MD / m\ncenter={center:.1f}")
    for axis, title in zip(axes[0], ("GR", "Deep / near", "DeltaLogR", "Density / points")):
        axis.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_response(frame: pd.DataFrame, metrics: dict[str, Any], output_path: Path) -> None:
    valid = frame.dropna(subset=["DeltaLogR", "AbsDeltaLogRResidual"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"wspace": 0.34})
    colors = np.where(valid["HasFractureTruth"].eq(1), "#c62828", "#687684")
    axes[0].scatter(valid["GR"], valid["DeltaLogR"], c=colors, s=8, alpha=0.42, linewidths=0)
    axes[0].axhline(0.0, color="#666666", linewidth=0.7)
    axes[0].set_xlabel("Same-pass GR")
    axes[0].set_ylabel("DeltaLogR")
    axes[0].set_title("GR vs separation")
    groups = [valid.loc[valid["HasFractureTruth"].eq(value), "AbsDeltaLogR"].dropna() for value in (0, 1)]
    axes[1].boxplot(groups, tick_labels=["non-fracture", "fracture"], showfliers=False)
    axes[1].set_ylabel("|DeltaLogR|")
    axes[1].set_title("Absolute separation")
    groups = [valid.loc[valid["HasFractureTruth"].eq(value), "AbsDeltaLogRResidual"].dropna() for value in (0, 1)]
    axes[2].boxplot(groups, tick_labels=["non-fracture", "fracture"], showfliers=False)
    axes[2].set_ylabel("|GR-conditioned residual|")
    axes[2].set_title("Lithology-conditioned anomaly")
    row = metrics["row_auc"]["AbsDeltaLogRResidual"]["best_direction_auc"]
    boot = metrics["block_bootstrap_fixed_direction_auc"]["AbsDeltaLogRResidual"]
    fig.suptitle(
        f"{frame['WellName'].iloc[0]} | row AUC={row:.3f} | 10 m block CI={boot['ci_low']:.3f}-{boot['ci_high']:.3f}",
        fontsize=12,
        y=0.99,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_well(frame: pd.DataFrame, config: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    metrics = metric_bundle(frame, config, seed_offset, truth_col="HasFractureTruth")
    point_metrics = metric_bundle(frame, config, seed_offset + 500, truth_col="HasPointTruth")
    strata = {}
    for idx, (name, part) in enumerate(frame.groupby("StrataName")):
        strata[str(name)] = {
            "density": metric_bundle(part, config, seed_offset + idx + 100, truth_col="HasFractureTruth"),
            "point": metric_bundle(part, config, seed_offset + idx + 600, truth_col="HasPointTruth"),
        }
    return {
        "well_name": str(frame["WellName"].iloc[0]),
        "pair_type": str(frame["PairType"].iloc[0]),
        "measurement_family": str(frame["MeasurementFamily"].iloc[0]),
        "deep_curve": str(frame["DeepSourceCurve"].iloc[0]),
        "near_curve": str(frame["NearSourceCurve"].iloc[0]),
        "row_count": int(len(frame)),
        "md_range_m": [float(frame["MD"].min()), float(frame["MD"].max())],
        "density_positive_rows": int(frame["HasFractureTruth"].sum()),
        "raw_point_rows": int(pd.to_numeric(frame["RawPointCount"], errors="coerce").fillna(0).sum()),
        "pair_valid_rows": int((frame["RDeep"].notna() & frame["RNear"].notna()).sum()),
        "metrics": metrics,
        "point_metrics": point_metrics,
        "strata_metrics": strata,
    }


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    manifest = pd.read_csv(config["input_manifest"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(manifest.itertuples(index=False)):
        frame = add_features(pd.read_csv(row.AnalysisPath), int(config.get("gr_quantile_bins", 5)))
        well_dir = output_dir / str(row.WellName)
        well_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(well_dir / f"{row.WellName}_md_v2_features.csv", index=False, encoding="utf-8-sig")
        summary = summarize_well(frame, config, idx * 1000)
        summaries.append(summary)
        (well_dir / f"{row.WellName}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_full_depth(frame, well_dir / f"01_{row.WellName}_full_depth_tracks.png")
        plot_windows(frame, well_dir / f"02_{row.WellName}_fracture_windows.png")
        plot_response(frame, summary["metrics"], well_dir / f"03_{row.WellName}_response_statistics.png")
        for truth_type, bundle in (("density", summary["metrics"]), ("raw_point", summary["point_metrics"])):
            for feature, values in bundle["row_auc"].items():
                boot = bundle["block_bootstrap_fixed_direction_auc"][feature]
                binned = bundle["depth_bin_fixed_direction_auc"][feature]
                flat_rows.append(
                    {
                        "WellName": summary["well_name"],
                        "PairType": summary["pair_type"],
                        "TruthType": truth_type,
                        "Feature": feature,
                        "FixedOrientation": bundle["fixed_orientations"][feature],
                        "RowAUC": values["auc"],
                        "RowBestDirectionAUC": values["best_direction_auc"],
                        "BlockBootstrapMedian": boot["median"],
                        "BlockBootstrapCILow": boot["ci_low"],
                        "BlockBootstrapCIHigh": boot["ci_high"],
                        "DepthBinFixedDirectionAUC": binned["auc"],
                    }
                )
    pd.DataFrame(flat_rows).to_csv(output_dir / "three_well_metric_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "three_well_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(flat_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
