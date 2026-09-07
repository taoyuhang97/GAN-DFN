from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from predict_step6_two_stage_density_volume import (  # noqa: E402
    ATTRIBUTE_COLUMNS,
    NULL_THRESHOLD,
    read_trace_indices,
    resample_regular_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build six Step6 density diagnostics for the formal 10 km demo.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def finite_limit(arrays: list[np.ndarray], quantile: float = 0.995) -> float:
    finite_parts = [array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()]
    if not finite_parts:
        return 1.0
    values = np.concatenate(finite_parts)
    return max(float(np.quantile(values, quantile)), 1.0e-6)


def read_section(
    handle: segyio.SegyFile,
    mapping: dict[str, np.ndarray],
    projection: str,
    coordinate: float,
    source_volume: bool,
    target_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if projection == "XZ":
        axis_values = mapping["y"]
        selected_axis = float(axis_values[np.argmin(np.abs(axis_values - coordinate))])
        selected = np.where(np.isclose(mapping["y"], selected_axis))[0]
        order = np.argsort(mapping["x"][selected])
        horizontal = mapping["x"][selected][order]
    else:
        axis_values = mapping["x"]
        selected_axis = float(axis_values[np.argmin(np.abs(axis_values - coordinate))])
        selected = np.where(np.isclose(mapping["x"], selected_axis))[0]
        order = np.argsort(mapping["y"][selected])
        horizontal = mapping["y"][selected][order]
    selected = selected[order]
    trace_indices = mapping["source_trace_idx"][selected] if source_volume else mapping["output_trace_index"][selected]
    matrix = read_trace_indices(handle, trace_indices.astype(np.int64))
    matrix[(~np.isfinite(matrix)) | (matrix <= NULL_THRESHOLD)] = np.nan
    if source_volume:
        matrix = resample_regular_matrix(matrix, np.asarray(handle.samples, dtype=np.float64), target_samples)
    return horizontal.astype(np.float64), matrix.astype(np.float32), selected_axis


def load_well_and_fractures(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    well = read_csv_flexible(Path(config["diagnostics"]["well_main_csv"]).resolve(), low_memory=False)
    for column in ("X", "Y", "TIME"):
        well[column] = pd.to_numeric(well[column], errors="coerce")
    well = well.dropna(subset=["X", "Y", "TIME"]).sort_values("TIME").reset_index(drop=True)
    groups = []
    for value in config["diagnostics"].get("step3_group_csvs", []):
        group = read_csv_flexible(Path(value).resolve(), low_memory=False)
        for column in ("X", "Y", "TIME", "Density", "GT_POINT_FLAG"):
            group[column] = pd.to_numeric(group[column], errors="coerce")
        groups.append(group)
    fracture = pd.concat(groups, ignore_index=True, sort=False) if groups else pd.DataFrame()
    return well, fracture


def overlay_well(ax: Any, projection: str, well: pd.DataFrame, fracture: pd.DataFrame) -> None:
    horizontal = well["X"] if projection == "XZ" else well["Y"]
    ax.plot(horizontal, well["TIME"], color="cyan", linewidth=1.3, label="Cheye1 trajectory")
    if not fracture.empty:
        points = fracture[pd.to_numeric(fracture["GT_POINT_FLAG"], errors="coerce").fillna(0).eq(1)]
        point_h = points["X"] if projection == "XZ" else points["Y"]
        ax.scatter(point_h, points["TIME"], s=16, color="magenta", edgecolors="white", linewidths=0.3, label="Imaging fracture")


def plot_density_section(
    path: Path,
    horizontal: np.ndarray,
    samples: np.ndarray,
    values: np.ndarray,
    title: str,
    projection: str,
    vmax: float,
    well: pd.DataFrame | None = None,
    fracture: pd.DataFrame | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    mesh = ax.pcolormesh(horizontal, samples, values.T, shading="auto", cmap="magma", vmin=0.0, vmax=vmax)
    if well is not None and fracture is not None:
        overlay_well(ax, projection, well, fracture)
        ax.legend(loc="upper right", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("X / m" if projection == "XZ" else "Y / m")
    ax.set_ylabel("TWT / ms")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Predicted fracture density")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_attribute_comparison(
    path: Path,
    horizontal: np.ndarray,
    samples: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    well: pd.DataFrame,
    fracture: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 14), sharex=True, constrained_layout=True)
    for ax, (name, values) in zip(axes, panels):
        finite = values[np.isfinite(values)]
        if name in {"SeisAmp", "CurvatureMax"} and finite.size:
            limit = max(float(np.quantile(np.abs(finite), 0.995)), 1.0e-6)
            cmap, vmin, vmax = "coolwarm", -limit, limit
        elif name == "Density":
            cmap, vmin, vmax = "magma", 0.0, finite_limit([values])
        else:
            low = float(np.quantile(finite, 0.01)) if finite.size else 0.0
            high = float(np.quantile(finite, 0.99)) if finite.size else 1.0
            cmap, vmin, vmax = "viridis", low, max(high, low + 1.0e-6)
        mesh = ax.pcolormesh(horizontal, samples, values.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        overlay_well(ax, "XZ", well, fracture)
        ax.invert_yaxis()
        ax.set_ylabel("TWT / ms")
        ax.set_title(name)
        fig.colorbar(mesh, ax=ax, pad=0.01)
    axes[-1].set_xlabel("X / m")
    fig.suptitle("Cheye1 XZ: Step6 density versus seismic attributes", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sample_density_along_well(
    density_path: Path,
    mapping: dict[str, np.ndarray],
    well: pd.DataFrame,
) -> pd.DataFrame:
    tree = cKDTree(np.column_stack([mapping["x"], mapping["y"]]))
    distances, output_indices = tree.query(well[["X", "Y"]].to_numpy(dtype=np.float64), k=1)
    samples_by_trace: dict[int, np.ndarray] = {}
    with segyio.open(str(density_path), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        for output_index in np.unique(output_indices.astype(np.int64)):
            samples_by_trace[int(output_index)] = np.asarray(handle.trace[int(output_index)], dtype=np.float32)
    prediction = np.full(len(well), np.nan, dtype=np.float64)
    for idx, (output_index, time_value) in enumerate(zip(output_indices, well["TIME"].to_numpy(dtype=np.float64))):
        trace = samples_by_trace[int(output_index)]
        right = int(np.searchsorted(samples, time_value, side="left"))
        if right == 0 or right >= len(samples):
            continue
        left = right - 1
        if not np.isfinite(trace[left]) or not np.isfinite(trace[right]):
            continue
        span = float(samples[right] - samples[left])
        if span <= 0.0:
            continue
        alpha = float((time_value - samples[left]) / span)
        prediction[idx] = float(trace[left] * (1.0 - alpha) + trace[right] * alpha)
    out = well[[column for column in ("WellName", "X", "Y", "TIME", "DEPT") if column in well.columns]].copy()
    out["NearestOutputTraceIndex"] = output_indices.astype(np.int64)
    out["NearestTraceDistanceM"] = distances
    out["PredictedDensity"] = prediction
    return out


def plot_well_comparison(
    path: Path,
    sampled: pd.DataFrame,
    fracture: pd.DataFrame,
    sample_interval_ms: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    ax.plot(sampled["PredictedDensity"], sampled["TIME"], color="black", linewidth=1.0, label="Step6 predicted density")
    if not fracture.empty:
        truth = fracture.sort_values("TIME")
        truth = truth.assign(
            TimeBin=np.round(truth["TIME"].to_numpy(dtype=np.float64) / sample_interval_ms) * sample_interval_ms
        )
        truth_curve = truth.groupby("TimeBin", as_index=False)["Density"].mean().sort_values("TimeBin")
        ax.plot(
            truth_curve["Density"],
            truth_curve["TimeBin"],
            color="royalblue",
            linewidth=1.2,
            alpha=0.9,
            label=f"Imaging density ({sample_interval_ms:g} ms mean)",
        )
        points = truth[truth["GT_POINT_FLAG"].fillna(0).eq(1)]
        ax.scatter(points["Density"], points["TIME"], color="magenta", s=24, label="Imaging fracture point")
    ax.invert_yaxis()
    ax.set_xlabel("Density")
    ax.set_ylabel("TWT / ms")
    ax.set_title("Cheye1 trajectory: Step6 prediction versus imaging interpretation")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    base_output = Path(config["output_dir"]).resolve()
    density_path = base_output / "predicted_fracture_density.sgy"
    mapping_path = base_output / "trace_mapping.npz"
    if not density_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("Step6 density SGY or trace mapping is missing")
    output_dir = base_output / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()

    with np.load(mapping_path) as payload:
        mapping = {key: np.asarray(payload[key]) for key in payload.files}
    well, fracture = load_well_and_fractures(config)
    center_x = 0.5 * (float(config["target_block"]["x_min"]) + float(config["target_block"]["x_max"]))
    center_y = 0.5 * (float(config["target_block"]["y_min"]) + float(config["target_block"]["y_max"]))
    well_x = float(well["X"].median())
    well_y = float(well["Y"].median())

    sections: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    with segyio.open(str(density_path), "r", ignore_geometry=True) as density_handle:
        density_handle.mmap()
        samples = np.asarray(density_handle.samples, dtype=np.float64)
        sections["center_xz"] = read_section(density_handle, mapping, "XZ", center_y, False, samples)
        sections["center_yz"] = read_section(density_handle, mapping, "YZ", center_x, False, samples)
        sections["well_xz"] = read_section(density_handle, mapping, "XZ", well_y, False, samples)
        sections["well_yz"] = read_section(density_handle, mapping, "YZ", well_x, False, samples)
    vmax = finite_limit([payload[1] for payload in sections.values()])

    images = {
        "01_center_density_xz.png": ("center_xz", "XZ", "Step6 density at demo center Y", None),
        "02_center_density_yz.png": ("center_yz", "YZ", "Step6 density at demo center X", None),
        "03_cheye1_density_xz.png": ("well_xz", "XZ", "Step6 density through Cheye1 Y", well),
        "04_cheye1_density_yz.png": ("well_yz", "YZ", "Step6 density through Cheye1 X", well),
    }
    for filename, (key, projection, title, overlay) in images.items():
        horizontal, values, actual_coordinate = sections[key]
        plot_density_section(
            output_dir / filename,
            horizontal,
            samples,
            values,
            f"{title} = {actual_coordinate:.1f}",
            projection,
            vmax,
            overlay,
            fracture if overlay is not None else None,
        )

    horizontal, density_xz, actual_y = sections["well_xz"]
    comparison_panels: list[tuple[str, np.ndarray]] = [("Density", density_xz)]
    source_indices = mapping["source_trace_idx"][np.where(np.isclose(mapping["y"], actual_y))[0]]
    order = np.argsort(mapping["x"][np.where(np.isclose(mapping["y"], actual_y))[0]])
    source_indices = source_indices[order]
    for attribute in ("SeisAmp", "CurvatureMax", "AntTrack", "Coherence"):
        path = Path(config["volume_paths"][attribute]).resolve()
        with segyio.open(str(path), "r", ignore_geometry=True) as handle:
            handle.mmap()
            matrix = read_trace_indices(handle, source_indices.astype(np.int64))
            matrix[(~np.isfinite(matrix)) | (matrix <= NULL_THRESHOLD)] = np.nan
            matrix = resample_regular_matrix(matrix, np.asarray(handle.samples, dtype=np.float64), samples)
        comparison_panels.append((attribute, matrix))
    plot_attribute_comparison(
        output_dir / "05_cheye1_xz_density_attribute_comparison.png",
        horizontal,
        samples,
        comparison_panels,
        well,
        fracture,
    )

    sampled = sample_density_along_well(density_path, mapping, well)
    sampled.to_csv(output_dir / "cheye1_well_density_samples.csv", index=False, encoding="utf-8-sig")
    plot_well_comparison(
        output_dir / "06_cheye1_well_density_comparison.png",
        sampled,
        fracture,
        float(config.get("sample_interval_ms", 2.0)),
    )

    expected_pngs = [
        "01_center_density_xz.png",
        "02_center_density_yz.png",
        "03_cheye1_density_xz.png",
        "04_cheye1_density_yz.png",
        "05_cheye1_xz_density_attribute_comparison.png",
        "06_cheye1_well_density_comparison.png",
    ]
    pngs = sorted(path.name for path in output_dir.glob("*.png"))
    checks = {
        "six_expected_pngs_exist": pngs == expected_pngs,
        "all_pngs_nonempty": all((output_dir / name).stat().st_size > 10000 for name in expected_pngs),
        "well_samples_have_prediction": bool(sampled["PredictedDensity"].notna().any()),
        "imaging_fracture_points_present": bool(not fracture.empty and fracture["GT_POINT_FLAG"].fillna(0).eq(1).any()),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "density_sgy": str(density_path),
        "trace_mapping_npz": str(mapping_path),
        "output_dir": str(output_dir),
        "images": pngs,
        "density_display_vmax": vmax,
        "well_sample_count": int(len(sampled)),
        "well_prediction_valid_count": int(sampled["PredictedDensity"].notna().sum()),
        "imaging_fracture_point_count": int(fracture["GT_POINT_FLAG"].fillna(0).eq(1).sum()) if not fracture.empty else 0,
        "checks": checks,
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[step6-diagnostics] output={output_dir}", flush=True)
    print(f"[step6-diagnostics] status={summary['status']} images={len(pngs)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
