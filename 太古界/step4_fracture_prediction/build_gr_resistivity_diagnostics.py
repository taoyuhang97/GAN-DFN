from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gan_dfn_matplotlib_cache")

import lasio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Keep Chinese well and formation names legible in diagnostic PNGs on Linux hosts.
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover a verified deep-near resistivity pair and diagnose its relationship to Step3 fracture labels."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_positive(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    return out.mask((out <= 0.0) | (out >= 1.0e6))


def load_las_curves(path: Path, deep_curve: str, near_curve: str, gr_curve: str) -> pd.DataFrame:
    las = lasio.read(str(path), ignore_header_errors=True)
    raw = las.df().reset_index()
    raw = raw.rename(columns={raw.columns[0]: "DEPT"})
    missing = [column for column in ("DEPT", deep_curve, near_curve, gr_curve) if column not in raw.columns]
    if missing:
        raise RuntimeError(f"LAS missing required curves {missing}: {path}")
    out = pd.DataFrame(
        {
            "DEPT": pd.to_numeric(raw["DEPT"], errors="coerce"),
            "RDeepRaw": clean_positive(raw[deep_curve]),
            "RNearRaw": clean_positive(raw[near_curve]),
            "GRFromLas": pd.to_numeric(raw[gr_curve], errors="coerce"),
        }
    ).dropna(subset=["DEPT"])
    out = out.sort_values("DEPT").drop_duplicates("DEPT", keep="last").reset_index(drop=True)
    return out


def interpolate_curve(source_depth: np.ndarray, source_value: np.ndarray, target_depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source_depth) & np.isfinite(source_value)
    if int(valid.sum()) < 2:
        return np.full(len(target_depth), np.nan, dtype=float)
    return np.interp(target_depth, source_depth[valid], source_value[valid], left=np.nan, right=np.nan)


def load_labels(group_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in group_paths:
        frame = pd.read_csv(path)
        required = {"DEPT", "Density", "GR", "StrataName"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Step3 label group missing {sorted(missing)}: {path}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["DEPT"] = pd.to_numeric(out["DEPT"], errors="coerce")
    out["Density"] = pd.to_numeric(out["Density"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["GR"] = pd.to_numeric(out["GR"], errors="coerce")
    return out.dropna(subset=["DEPT"]).sort_values("DEPT").reset_index(drop=True)


def robust_zscore(values: pd.Series) -> pd.Series:
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1.0e-9:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - median) / scale


def build_diagnostic_frame(labels: pd.DataFrame, curves: pd.DataFrame, gr_bins: int) -> pd.DataFrame:
    out = labels.copy()
    target_depth = out["DEPT"].to_numpy(dtype=float)
    source_depth = curves["DEPT"].to_numpy(dtype=float)
    for column in ("RDeepRaw", "RNearRaw", "GRFromLas"):
        out[column] = interpolate_curve(source_depth, curves[column].to_numpy(dtype=float), target_depth)
    out["LogRDeep"] = np.log10(out["RDeepRaw"])
    out["LogRNear"] = np.log10(out["RNearRaw"])
    out["DeltaLogR"] = out["LogRDeep"] - out["LogRNear"]
    out["AbsDeltaLogR"] = out["DeltaLogR"].abs()
    valid_gr = out["GR"].notna() & out["DeltaLogR"].notna()
    out["GRBin"] = np.nan
    if int(valid_gr.sum()) >= gr_bins:
        ranked = out.loc[valid_gr, "GR"].rank(method="first")
        out.loc[valid_gr, "GRBin"] = pd.qcut(ranked, q=gr_bins, labels=False, duplicates="drop").astype(float)
        baseline = out.loc[valid_gr].groupby("GRBin")["DeltaLogR"].transform("median")
        out.loc[valid_gr, "DeltaLogRResidual"] = out.loc[valid_gr, "DeltaLogR"].to_numpy() - baseline.to_numpy()
    else:
        out["DeltaLogRResidual"] = np.nan
    out["AbsDeltaLogRResidual"] = out["DeltaLogRResidual"].abs()
    out["GRNorm"] = robust_zscore(out["GR"])
    out["HasFractureTruth"] = (out["Density"] > 0.0).astype(int)
    return out


def bidirectional_auc(y: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(score)
    if int(valid.sum()) < 2 or np.unique(y[valid]).size < 2:
        return {"auc": None, "best_direction_auc": None}
    auc = float(roc_auc_score(y[valid], score[valid]))
    return {"auc": auc, "best_direction_auc": float(max(auc, 1.0 - auc))}


def median_for(mask: pd.Series, values: pd.Series) -> float | None:
    subset = values[mask & values.notna()]
    return float(subset.median()) if len(subset) else None


def response_summary(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    valid = frame[frame["DeltaLogR"].notna()].copy()
    positive = valid["HasFractureTruth"].astype(int).to_numpy()
    summary: dict[str, Any] = {
        **metadata,
        "label_row_count": int(len(frame)),
        "pair_valid_row_count": int(len(valid)),
        "pair_coverage": float(len(valid) / len(frame)) if len(frame) else 0.0,
        "fracture_positive_count": int(frame["HasFractureTruth"].sum()),
        "fracture_positive_count_with_pair": int(valid["HasFractureTruth"].sum()),
        "median_delta_log_r_fracture": median_for(valid["HasFractureTruth"].eq(1), valid["DeltaLogR"]),
        "median_delta_log_r_nonfracture": median_for(valid["HasFractureTruth"].eq(0), valid["DeltaLogR"]),
        "median_abs_delta_log_r_fracture": median_for(valid["HasFractureTruth"].eq(1), valid["AbsDeltaLogR"]),
        "median_abs_delta_log_r_nonfracture": median_for(valid["HasFractureTruth"].eq(0), valid["AbsDeltaLogR"]),
        "raw_delta_log_r_auc": bidirectional_auc(positive, valid["DeltaLogR"].to_numpy(dtype=float)),
        "abs_delta_log_r_auc": bidirectional_auc(positive, valid["AbsDeltaLogR"].to_numpy(dtype=float)),
        "abs_gr_conditioned_delta_log_r_auc": bidirectional_auc(
            positive, valid["AbsDeltaLogRResidual"].to_numpy(dtype=float)
        ),
    }
    return summary


def style_depth_axis(axis: Any, depth: pd.Series) -> None:
    axis.set_ylim(float(depth.max()), float(depth.min()))
    axis.grid(True, axis="x", alpha=0.22)
    axis.grid(True, axis="y", alpha=0.12)


def plot_full_depth(frame: pd.DataFrame, summary: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 12), sharey=True, gridspec_kw={"wspace": 0.26})
    depth = frame["DEPT"]
    axes[0].plot(frame["GR"], depth, color="#2e8b57", linewidth=0.8)
    axes[0].set_xlabel("GR")
    axes[0].set_ylabel("Depth / m")
    axes[0].set_title("GR (Step3)")
    axes[1].semilogx(frame["RDeepRaw"], depth, color="#d97706", linewidth=0.8, label="Deep")
    axes[1].semilogx(frame["RNearRaw"], depth, color="#2563eb", linewidth=0.8, label="Near")
    axes[1].set_xlabel("Resistivity")
    axes[1].set_title("Deep / near resistivity")
    axes[1].legend(fontsize=8)
    axes[2].plot(frame["DeltaLogR"], depth, color="#7c3aed", linewidth=0.8, label="DeltaLogR")
    axes[2].axvline(0.0, color="#6b7280", linewidth=0.7)
    fracture = frame["HasFractureTruth"].eq(1)
    axes[2].scatter(frame.loc[fracture, "DeltaLogR"], depth[fracture], s=5, color="#e11d48", label="Density>0", zorder=3)
    axes[2].set_xlabel("log10(Rdeep/Rnear)")
    axes[2].set_title("Resistivity separation")
    axes[2].legend(fontsize=8)
    axes[3].fill_betweenx(depth, 0.0, frame["Density"], color="#ec4899", alpha=0.58, label="Step3 Density")
    axes[3].plot(frame["Density"], depth, color="#9d174d", linewidth=0.7)
    axes[3].set_xlabel("Fracture density")
    axes[3].set_title("Imaging-log truth")
    for axis in axes:
        style_depth_axis(axis, depth)
    fig.suptitle(
        f"{summary['well_name']} | {summary['pair_type']} | GR + deep-near resistivity vs Step3 fracture density",
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        f"pair coverage={summary['pair_coverage']:.1%}; positive labels with pair={summary['fracture_positive_count_with_pair']}; "
        f"best signed-separation AUC={summary['raw_delta_log_r_auc']['best_direction_auc']}",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_window_centers(frame: pd.DataFrame, count: int) -> list[float]:
    positives = frame[frame["Density"] > 0.0].sort_values("Density", ascending=False)
    centers: list[float] = []
    for depth in positives["DEPT"].to_numpy(dtype=float):
        if all(abs(depth - existing) >= 20.0 for existing in centers):
            centers.append(float(depth))
        if len(centers) >= count:
            break
    return sorted(centers)


def plot_fracture_windows(frame: pd.DataFrame, summary: dict[str, Any], output_path: Path) -> None:
    centers = select_window_centers(frame, count=3)
    if not centers:
        return
    fig, axes = plt.subplots(len(centers), 4, figsize=(18, 4.1 * len(centers)), sharey="row", gridspec_kw={"wspace": 0.26})
    if len(centers) == 1:
        axes = np.asarray([axes])
    for row, center in enumerate(centers):
        subset = frame[frame["DEPT"].between(center - 10.0, center + 10.0)]
        depth = subset["DEPT"]
        axes[row, 0].plot(subset["GR"], depth, color="#2e8b57", linewidth=1.0)
        axes[row, 0].set_title(f"GR | {center:.1f} m")
        axes[row, 1].semilogx(subset["RDeepRaw"], depth, color="#d97706", linewidth=1.0)
        axes[row, 1].semilogx(subset["RNearRaw"], depth, color="#2563eb", linewidth=1.0)
        axes[row, 1].set_title("Deep / near")
        axes[row, 2].plot(subset["DeltaLogR"], depth, color="#7c3aed", linewidth=1.0)
        axes[row, 2].axvline(0.0, color="#6b7280", linewidth=0.7)
        axes[row, 2].set_title("DeltaLogR")
        axes[row, 3].fill_betweenx(depth, 0.0, subset["Density"], color="#ec4899", alpha=0.58)
        axes[row, 3].plot(subset["Density"], depth, color="#9d174d", linewidth=0.8)
        axes[row, 3].set_title("Step3 Density")
        for axis in axes[row]:
            style_depth_axis(axis, depth)
    axes[-1, 0].set_xlabel("GR")
    axes[-1, 1].set_xlabel("Resistivity")
    axes[-1, 2].set_xlabel("log10(Rdeep/Rnear)")
    axes[-1, 3].set_xlabel("Density")
    axes[0, 0].set_ylabel("Depth / m")
    fig.suptitle(f"{summary['well_name']} | fracture-density-centered 20 m windows", fontsize=13, y=0.998)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_response_statistics(frame: pd.DataFrame, summary: dict[str, Any], output_path: Path) -> None:
    valid = frame.dropna(subset=["DeltaLogR"]).copy()
    valid["Truth"] = np.where(valid["HasFractureTruth"].eq(1), "fracture", "non-fracture")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"wspace": 0.32})
    colors = np.where(valid["HasFractureTruth"].eq(1), "#e11d48", "#64748b")
    axes[0].scatter(valid["GR"], valid["DeltaLogR"], c=colors, s=7, alpha=0.46, linewidths=0)
    axes[0].axhline(0.0, color="#6b7280", linewidth=0.7)
    axes[0].set_xlabel("GR")
    axes[0].set_ylabel("DeltaLogR")
    axes[0].set_title("GR vs resistivity separation")
    data = [valid.loc[valid["HasFractureTruth"].eq(value), "DeltaLogR"].dropna() for value in (0, 1)]
    axes[1].boxplot(data, tick_labels=["non-fracture", "fracture"], showfliers=False)
    axes[1].axhline(0.0, color="#6b7280", linewidth=0.7)
    axes[1].set_ylabel("DeltaLogR")
    axes[1].set_title("Signed separation by truth")
    data = [valid.loc[valid["HasFractureTruth"].eq(value), "AbsDeltaLogRResidual"].dropna() for value in (0, 1)]
    axes[2].boxplot(data, tick_labels=["non-fracture", "fracture"], showfliers=False)
    axes[2].set_ylabel("|GR-conditioned DeltaLogR residual|")
    axes[2].set_title("Lithology-conditioned anomaly")
    fig.suptitle(
        f"{summary['well_name']} | signed AUC={summary['raw_delta_log_r_auc']['auc']}; "
        f"absolute AUC={summary['abs_delta_log_r_auc']['auc']}; "
        f"GR-conditioned absolute AUC={summary['abs_gr_conditioned_delta_log_r_auc']['auc']}",
        fontsize=12,
        y=0.99,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels([Path(path) for path in config["step3_group_paths"]])
    curves = load_las_curves(
        Path(config["las_path"]),
        str(config["deep_curve"]),
        str(config["near_curve"]),
        str(config.get("gr_curve", "GR")),
    )
    frame = build_diagnostic_frame(labels, curves, int(config.get("gr_quantile_bins", 5)))
    metadata = {
        "well_name": str(config["well_name"]),
        "pair_type": str(config["pair_type"]),
        "deep_curve": str(config["deep_curve"]),
        "near_curve": str(config["near_curve"]),
        "las_path": str(Path(config["las_path"]).resolve()),
        "step3_group_paths": [str(Path(path).resolve()) for path in config["step3_group_paths"]],
        "las_depth_range_m": [float(curves["DEPT"].min()), float(curves["DEPT"].max())],
        "label_depth_range_m": [float(labels["DEPT"].min()), float(labels["DEPT"].max())],
    }
    summary = response_summary(frame, metadata)
    stem = str(config.get("file_stem", "diagnostic"))
    frame.to_csv(output_dir / f"{stem}_aligned_gr_resistivity_labels.csv", index=False, encoding="utf-8-sig")
    plot_full_depth(frame, summary, output_dir / f"01_{stem}_full_depth_tracks.png")
    plot_fracture_windows(frame, summary, output_dir / f"02_{stem}_fracture_windows.png")
    plot_response_statistics(frame, summary, output_dir / f"03_{stem}_response_statistics.png")
    write_json(output_dir / f"{stem}_summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run(read_json(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
