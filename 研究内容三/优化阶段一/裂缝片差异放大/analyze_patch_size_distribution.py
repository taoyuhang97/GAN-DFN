from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/单元DFN批量生成/bx33_35_by33_35_surface_scaled_x5_20260331"
)

REQUIRED_COLUMNS = ["UnitID", "PatchLength", "PatchHeight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze patch size distributions for unit DFN outputs and export summary figures."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="Batch DFN run root containing the units directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save analysis tables and figures.")
    parser.add_argument("--bins", type=int, default=30, help="Histogram bin count.")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return parser.parse_args()


def discover_unit_patch_csvs(units_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    if not units_root.exists():
        raise FileNotFoundError(f"units directory not found: {units_root}")
    for unit_dir in sorted([path for path in units_root.iterdir() if path.is_dir()]):
        patch_csv = unit_dir / "predicted_unit_patches.csv"
        if patch_csv.exists():
            rows.append((unit_dir.name, patch_csv))
    if not rows:
        raise FileNotFoundError(f"no predicted_unit_patches.csv files found under: {units_root}")
    return rows


def load_patch_size_table(unit_id: str, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    work = df[REQUIRED_COLUMNS].copy()
    work["PatchLength"] = pd.to_numeric(work["PatchLength"], errors="coerce")
    work["PatchHeight"] = pd.to_numeric(work["PatchHeight"], errors="coerce")
    work["UnitID"] = unit_id
    work = work.dropna(subset=["PatchLength", "PatchHeight"]).reset_index(drop=True)
    return work


def build_long_table(unit_tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(unit_tables, ignore_index=True)
    length_df = merged[["UnitID", "PatchLength"]].rename(columns={"PatchLength": "Value"})
    length_df["Metric"] = "PatchLength"
    height_df = merged[["UnitID", "PatchHeight"]].rename(columns={"PatchHeight": "Value"})
    height_df["Metric"] = "PatchHeight"
    long_df = pd.concat([length_df, height_df], ignore_index=True)
    return long_df[["UnitID", "Metric", "Value"]].copy()


def summarize_metric(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "Count": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "Std": np.nan,
            "Min": np.nan,
            "P25": np.nan,
            "P75": np.nan,
            "Max": np.nan,
            "CV": np.nan,
        }
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "Count": int(len(values)),
        "Mean": mean_value,
        "Median": float(values.median()),
        "Std": std_value,
        "Min": float(values.min()),
        "P25": float(values.quantile(0.25)),
        "P75": float(values.quantile(0.75)),
        "Max": float(values.max()),
        "CV": float(std_value / mean_value) if mean_value not in (0.0, -0.0) else np.nan,
    }


def build_summary_tables(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    for unit_id, unit_df in all_df.groupby("UnitID", sort=True):
        for metric in ["PatchLength", "PatchHeight"]:
            row = {"UnitID": unit_id, "Metric": metric}
            row.update(summarize_metric(unit_df[metric]))
            summary_rows.append(row)
    summary_long = pd.DataFrame(summary_rows)
    summary_wide = summary_long.pivot(index="UnitID", columns="Metric")
    summary_wide.columns = [f"{metric}_{field}" for field, metric in summary_wide.columns]
    summary_wide = summary_wide.reset_index()
    return summary_long, summary_wide


def ensure_output_dir(run_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        resolved = output_dir.resolve()
    else:
        resolved = (run_root / "patch_size_distribution_analysis").resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def compute_grid_shape(item_count: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(item_count))
    rows = math.ceil(item_count / cols)
    return rows, cols


def plot_hist_grid(
    all_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    unit_ids = sorted(all_df["UnitID"].unique().tolist())
    values = pd.to_numeric(all_df[metric], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"no valid values found for metric: {metric}")

    rows, cols = compute_grid_shape(len(unit_ids))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.4), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    bins_edges = np.histogram_bin_edges(values.to_numpy(dtype=float), bins=int(bins))

    for ax in axes_array.ravel():
        ax.set_visible(False)

    for idx, unit_id in enumerate(unit_ids):
        ax = axes_array.ravel()[idx]
        ax.set_visible(True)
        unit_values = pd.to_numeric(
            all_df.loc[all_df["UnitID"].astype(str) == str(unit_id), metric],
            errors="coerce",
        ).dropna()
        ax.hist(unit_values, bins=bins_edges, color="#4C78A8", alpha=0.78, edgecolor="white")
        ax.axvline(float(unit_values.mean()), color="#F58518", linestyle="--", linewidth=1.6, label="Mean")
        ax.axvline(float(unit_values.median()), color="#54A24B", linestyle="-.", linewidth=1.4, label="Median")
        ax.set_title(f"{unit_id}  n={len(unit_values)}", fontsize=10)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.18, linestyle="--")
        ax.legend(fontsize=8)

    fig.suptitle(f"{metric} Distribution by Unit", fontsize=14)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(all_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    unit_ids = sorted(all_df["UnitID"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    metrics = ["PatchLength", "PatchHeight"]
    colors = ["#4C78A8", "#E45756"]

    for ax, metric, color in zip(axes, metrics, colors):
        data = [
            pd.to_numeric(
                all_df.loc[all_df["UnitID"].astype(str) == str(unit_id), metric],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            for unit_id in unit_ids
        ]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=unit_ids, showfliers=False)
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.7)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.4)
        ax.set_title(f"{metric} Boxplot")
        ax.set_xlabel("UnitID")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.18, linestyle="--")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Patch Size Comparison Across 9 Units", fontsize=14)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(all_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    unit_ids = sorted(all_df["UnitID"].unique().tolist())
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(unit_ids)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)

    for ax, metric in zip(axes, ["PatchLength", "PatchHeight"]):
        for color, unit_id in zip(colors, unit_ids):
            values = pd.to_numeric(
                all_df.loc[all_df["UnitID"].astype(str) == str(unit_id), metric],
                errors="coerce",
            ).dropna().sort_values()
            if values.empty:
                continue
            y = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
            ax.step(values.to_numpy(dtype=float), y, where="post", label=unit_id, color=color, linewidth=1.5)
        ax.set_title(f"{metric} ECDF")
        ax.set_xlabel(metric)
        ax.set_ylabel("Cumulative Probability")
        ax.grid(alpha=0.18, linestyle="--")

    axes[1].legend(loc="lower right", fontsize=8, ncol=1, frameon=True)
    fig.suptitle("Patch Size ECDF Comparison", fontsize=14)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(all_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    unit_ids = sorted(all_df["UnitID"].unique().tolist())
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(unit_ids)))
    fig, ax = plt.subplots(figsize=(8.4, 6.8), constrained_layout=True)

    for color, unit_id in zip(colors, unit_ids):
        unit_df = all_df.loc[all_df["UnitID"].astype(str) == str(unit_id)].copy()
        ax.scatter(
            pd.to_numeric(unit_df["PatchLength"], errors="coerce"),
            pd.to_numeric(unit_df["PatchHeight"], errors="coerce"),
            s=12,
            alpha=0.35,
            color=color,
            label=unit_id,
        )

    ax.set_title("PatchLength vs PatchHeight")
    ax.set_xlabel("PatchLength")
    ax.set_ylabel("PatchHeight")
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mean_std(summary_long: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)
    colors = ["#4C78A8", "#E45756"]

    for ax, metric, color in zip(axes, ["PatchLength", "PatchHeight"], colors):
        sub = summary_long.loc[summary_long["Metric"].astype(str) == metric].sort_values("UnitID").reset_index(drop=True)
        x = np.arange(len(sub), dtype=float)
        means = pd.to_numeric(sub["Mean"], errors="coerce").to_numpy(dtype=float)
        stds = pd.to_numeric(sub["Std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, capsize=4, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["UnitID"].astype(str).tolist(), rotation=30)
        ax.set_title(f"{metric} Mean ± Std")
        ax.set_xlabel("UnitID")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.18, linestyle="--", axis="y")

    fig.suptitle("Patch Size Mean and Variability", fontsize=14)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    units_root = run_root / "units"
    output_dir = ensure_output_dir(run_root, args.output_dir)

    unit_csvs = discover_unit_patch_csvs(units_root)
    unit_tables = [load_patch_size_table(unit_id, csv_path) for unit_id, csv_path in unit_csvs]
    all_df = pd.concat(unit_tables, ignore_index=True)
    long_df = build_long_table(unit_tables)
    summary_long, summary_wide = build_summary_tables(all_df)

    summary_long_path = output_dir / "patch_size_summary_long.csv"
    summary_wide_path = output_dir / "patch_size_summary_wide.csv"
    raw_table_path = output_dir / "patch_size_values.csv"
    long_table_path = output_dir / "patch_size_values_long.csv"

    all_df.to_csv(raw_table_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_table_path, index=False, encoding="utf-8-sig")
    summary_long.to_csv(summary_long_path, index=False, encoding="utf-8-sig")
    summary_wide.to_csv(summary_wide_path, index=False, encoding="utf-8-sig")

    plot_hist_grid(
        all_df=all_df,
        metric="PatchLength",
        output_path=output_dir / "patch_length_hist_grid.png",
        bins=int(args.bins),
        dpi=int(args.dpi),
    )
    plot_hist_grid(
        all_df=all_df,
        metric="PatchHeight",
        output_path=output_dir / "patch_height_hist_grid.png",
        bins=int(args.bins),
        dpi=int(args.dpi),
    )
    plot_boxplots(
        all_df=all_df,
        output_path=output_dir / "patch_size_boxplots.png",
        dpi=int(args.dpi),
    )
    plot_ecdf(
        all_df=all_df,
        output_path=output_dir / "patch_size_ecdf.png",
        dpi=int(args.dpi),
    )
    plot_scatter(
        all_df=all_df,
        output_path=output_dir / "patch_length_vs_patch_height_scatter.png",
        dpi=int(args.dpi),
    )
    plot_mean_std(
        summary_long=summary_long,
        output_path=output_dir / "patch_size_mean_std.png",
        dpi=int(args.dpi),
    )

    summary_json = {
        "run_root": str(run_root),
        "units_root": str(units_root),
        "unit_count": int(len(unit_csvs)),
        "unit_ids": [unit_id for unit_id, _ in unit_csvs],
        "total_patch_count": int(len(all_df)),
        "output_dir": str(output_dir),
        "figures": [
            "patch_length_hist_grid.png",
            "patch_height_hist_grid.png",
            "patch_size_boxplots.png",
            "patch_size_ecdf.png",
            "patch_length_vs_patch_height_scatter.png",
            "patch_size_mean_std.png",
        ],
        "tables": [
            "patch_size_values.csv",
            "patch_size_values_long.csv",
            "patch_size_summary_long.csv",
            "patch_size_summary_wide.csv",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"run_root={run_root}")
    print(f"units_analyzed={len(unit_csvs)}")
    print(f"total_patches={len(all_df)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
