#!/usr/bin/env python3
"""Step9 (太古界 v2): 埕北古斜405 along-well curved XZ/YZ sections.

Matches the glutenite section geometry (well_trajectory_curved_surface):
  * XZ: for each TWT sample t, the section plane passes through Ywell(t);
    image = amplitude/density at (x, Ywell(t), t) for all demo X.
  * YZ: for each t, the plane passes through Xwell(t); image = (Xwell(t), y, t).
  * vertical axis = TWT (ms), time increasing downward.

Overlays: Step8 corrected patches and imaging labels projected with a
half-width perpendicular to the section, three horizons along the well path,
and the well track curve.

Outputs (config.output_dir):
  sections/XZ_overview.png, XZ_well_local.png, YZ_overview.png, YZ_well_local.png
  step9_summary.json   status=pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio


CURRENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = CURRENT_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(COMMON_DIR / "seismic_sampling") not in sys.path:
    sys.path.insert(0, str(COMMON_DIR / "seismic_sampling"))

from amplitude_sampling import ObnAmplitudeSampler  # noqa: E402


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 埕北古斜405 along-well curved sections (太古界 Step9 v2).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_well_track(config: dict[str, Any], well: str) -> pd.DataFrame:
    segment_dir = Path(config["step2_segments_root"]) / well
    files = sorted(segment_dir.glob("*.csv")) if segment_dir.exists() else []
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        for column in ("MD", "TVD", "X", "Y", "TIME"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        frames.append(df[["MD", "TVD", "X", "Y", "TIME"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["X", "Y", "TIME"]).sort_values("TIME").reset_index(drop=True)


def load_imaging_labels(config: dict[str, Any], well: str) -> pd.DataFrame:
    group_files = sorted(Path(config["step3_groups_root"]).glob(f"{well}_*.csv"))
    parts = []
    for group_file in group_files:
        group = pd.read_csv(group_file, encoding="utf-8-sig")
        for segment_path, sub in group.groupby("InputSegmentPath"):
            segment = pd.read_csv(segment_path, encoding="utf-8-sig")
            for column in ("MD", "TVD"):
                sub[column] = pd.to_numeric(sub[column], errors="coerce")
            for column in ("MD", "TVD", "X", "Y", "TIME"):
                segment[column] = pd.to_numeric(segment[column], errors="coerce")
            merged = pd.merge_asof(
                sub.sort_values("MD"),
                segment[["MD", "X", "Y", "TIME"]].sort_values("MD"),
                on="MD",
                direction="nearest",
                tolerance=float(config["md_merge_tolerance"]),
            )
            merged = merged[merged["GT_POINT_FLAG"].fillna(0).astype(int) == 1]
            merged = merged[merged["FracAzimuth"].notna() & merged["FracDip"].notna()]
            parts.append(merged)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    for column in ("X", "Y", "TIME"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["X", "Y", "TIME"]).copy()


def interp_well(times: np.ndarray, track_t: np.ndarray, track_v: np.ndarray) -> np.ndarray:
    return np.interp(times, track_t, track_v)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    sections_dir = output_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "step9_summary.json"
    if summary_path.exists() and not args.replace_output:
        raise FileExistsError(f"refusing to overwrite existing Step9 output: {summary_path}")
    started = time.time()

    grid = pd.read_csv(config["demo_grid_csv"], encoding="utf-8-sig")
    horizon = pd.read_csv(config["horizon_contract_csv"], encoding="utf-8-sig")
    for df in (grid, horizon):
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grid = grid.merge(
        horizon[["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs"]],
        on="TraceIdx",
        how="left",
    ).sort_values("TraceIdx").reset_index(drop=True)
    grid["Row"] = np.arange(len(grid))
    x_values = np.sort(grid["X"].unique())
    y_values = np.sort(grid["Y"].unique())

    track = load_well_track(config, config["profile_well"])
    labels = load_imaging_labels(config, config["profile_well"])
    patches = pd.read_csv(config["corrected_patches_csv"], encoding="utf-8-sig")
    for column in ("X", "Y", "TIME", "Density"):
        patches[column] = pd.to_numeric(patches[column], errors="coerce")
    patches = patches.dropna(subset=["X", "Y", "TIME"]).copy()
    track_t = track["TIME"].to_numpy(dtype=np.float64)
    track_x = track["X"].to_numpy(dtype=np.float64)
    track_y = track["Y"].to_numpy(dtype=np.float64)
    half_width = float(config["projection_half_width_m"])
    time_pad = 20.0
    t_min = float(np.floor((track_t.min() - time_pad) / 2.0) * 2.0)
    t_max = float(np.ceil((track_t.max() + time_pad) / 2.0) * 2.0)
    section_times = np.arange(t_min, t_max + 1.0e-6, 2.0)

    with ObnAmplitudeSampler(Path(config["obn_segy"]), Path(config["trace_header_csv"])) as sampler:
        with segyio.open(str(config["density_sgy"]), "r", ignore_geometry=True) as handle:
            handle.mmap()

            def render(axis: str, overview: bool) -> Path:
                if axis == "X":
                    section_xy_fixed = interp_well(section_times, track_t, track_y)
                    coords = x_values
                    coord_label = "X (m)"
                    well_coord = interp_well(section_times, track_t, track_x)
                else:
                    section_xy_fixed = interp_well(section_times, track_t, track_x)
                    coords = y_values
                    coord_label = "Y (m)"
                    well_coord = interp_well(section_times, track_t, track_y)
                n_time = len(section_times)
                n_coord = len(coords)
                fixed = np.repeat(section_xy_fixed, n_coord)
                moving = np.tile(coords, n_time)
                query_x = moving if axis == "X" else fixed
                query_y = fixed if axis == "X" else moving
                trace_idx, _ = sampler.nearest_trace(query_x, query_y)
                query_times = np.repeat(section_times, n_coord)
                amp_flat = sampler.sample_at_trace(trace_idx, query_times)
                amp_image = amp_flat.reshape(n_time, n_coord)
                # density at 2 ms axis
                unique_traces, inverse = np.unique(trace_idx, return_inverse=True)
                trace_to_row = {int(r["TraceIdx"]): int(r["Row"]) for _, r in grid.iterrows()}
                dens_rows = [trace_to_row[int(t)] for t in unique_traces]
                dens_matrix = np.stack([np.asarray(handle.trace[int(r)], dtype=np.float32) for r in dens_rows])
                dens_axis = np.asarray(handle.samples, dtype=np.float64)
                dens_flat = np.array(
                    [
                        float(np.interp(query_times[i], dens_axis, dens_matrix[inverse[i]]))
                        for i in range(len(query_times))
                    ],
                    dtype=np.float32,
                )
                dens_image = dens_flat.reshape(n_time, n_coord)

                fig, ax = plt.subplots(figsize=(14, 7))
                vmax = float(np.nanpercentile(np.abs(amp_image), 98)) if np.isfinite(amp_image).any() else 1.0
                ax.imshow(
                    amp_image,
                    aspect="auto",
                    extent=[coords[0], coords[-1], t_max, t_min],
                    cmap="seismic",
                    vmin=-vmax,
                    vmax=vmax,
                    interpolation="nearest",
                )
                density_vmax = float(np.nanpercentile(dens_image, 99)) if np.isfinite(dens_image).any() else 1.0
                ax.imshow(
                    dens_image,
                    aspect="auto",
                    extent=[coords[0], coords[-1], t_max, t_min],
                    cmap="hot_r",
                    vmin=0.0,
                    vmax=density_vmax,
                    alpha=0.55,
                    interpolation="nearest",
                )
                # well path curve and horizons along the well path
                ax.plot(well_coord, section_times, color="black", lw=2.0, label=f"{config['profile_well']} track")
                well_rows = track[(track["TIME"] >= t_min) & (track["TIME"] <= t_max)]
                if len(well_rows):
                    well_trace, _ = sampler.nearest_trace(
                        well_rows["X"].to_numpy(dtype=np.float64), well_rows["Y"].to_numpy(dtype=np.float64)
                    )
                    horizon_map = horizon.set_index("TraceIdx")
                    for name, color, label in (
                        ("TopTimeMs", "cyan", "Top horizon"),
                        ("MidTimeMs", "lime", "Mid horizon"),
                        ("BaseTimeMs", "orange", "Base horizon"),
                    ):
                        values = horizon_map.loc[well_trace, name].to_numpy(dtype=np.float64)
                        ax.plot(
                            well_rows["X"].to_numpy() if axis == "X" else well_rows["Y"].to_numpy(),
                            values,
                            color=color,
                            lw=0.8,
                            label=label,
                        )
                # patches & labels projected with half width
                if axis == "X":
                    patch_proj = patches[(patches["Y"] - interp_well(patches["TIME"].to_numpy(), track_t, track_y)).abs() <= half_width]
                    label_proj = labels[(labels["Y"] - interp_well(labels["TIME"].to_numpy(), track_t, track_y)).abs() <= half_width]
                else:
                    patch_proj = patches[(patches["X"] - interp_well(patches["TIME"].to_numpy(), track_t, track_x)).abs() <= half_width]
                    label_proj = labels[(labels["X"] - interp_well(labels["TIME"].to_numpy(), track_t, track_x)).abs() <= half_width]
                if len(patch_proj):
                    ax.scatter(
                        patch_proj[coord_label.split(" ")[0]],
                        patch_proj["TIME"],
                        c=patch_proj["Density"],
                        cmap="hot_r",
                        s=8,
                        alpha=0.8,
                        label="small patches",
                    )
                if len(label_proj):
                    ax.scatter(
                        label_proj[coord_label.split(" ")[0]],
                        label_proj["TIME"],
                        marker="^",
                        color="magenta",
                        s=24,
                        alpha=0.9,
                        label="imaging labels",
                    )
                ax.set_xlabel(coord_label)
                ax.set_ylabel("TWT (ms)")
                title = (
                    f"{config['profile_well']} {'XZ' if axis == 'X' else 'YZ'} curved section "
                    f"({'demo overview' if overview else 'well local'})"
                )
                ax.set_title(title)
                ax.set_ylim(t_max, t_min)
                if not overview:
                    local_half = float(config["well_local_half_width_m"])
                    center_coord = float(np.median(well_coord))
                    ax.set_xlim(center_coord - local_half, center_coord + local_half)
                ax.legend(loc="upper right", fontsize=8)
                out = sections_dir / f"{'XZ' if axis == 'X' else 'YZ'}_{'overview' if overview else 'well_local'}.png"
                fig.savefig(out, dpi=130, bbox_inches="tight")
                plt.close(fig)
                return out

            xz_overview = render("X", True)
            xz_local = render("X", False)
            yz_overview = render("Y", True)
            yz_local = render("Y", False)

    outputs = {
        "xz_overview_png": str(xz_overview),
        "xz_well_local_png": str(xz_local),
        "yz_overview_png": str(yz_overview),
        "yz_well_local_png": str(yz_local),
    }
    checks = {
        "track_loaded": len(track) > 0,
        "imaging_labels_loaded": len(labels) > 0,
        "patches_loaded": len(patches) > 0,
        "sections_exist": all(Path(p).exists() for p in outputs.values()),
        "well_inside_demo": bool(
            (
                (track["X"] >= grid["X"].min() - half_width)
                & (track["X"] <= grid["X"].max() + half_width)
                & (track["Y"] >= grid["Y"].min() - half_width)
                & (track["Y"] <= grid["Y"].max() + half_width)
            ).all()
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "profile_well": config["profile_well"],
        "section_mode": "well_trajectory_curved_surface",
        "xy_projection_logic": {
            "XZ": "sample at (x, Ywell(time), time)",
            "YZ": "sample at (Xwell(time), y, time)",
        },
        "well_track": {
            "rows": int(len(track)),
            "time_range_ms": [float(track_t.min()), float(track_t.max())],
            "temporary_neighbor_time_depth": 1,
        },
        "sections": {
            "time_min_ms": t_min,
            "time_max_ms": t_max,
            "time_interval_ms": 2.0,
            "projection_half_width_m": half_width,
        },
        "imaging_label_count": int(len(labels)),
        "patch_count": int(len(patches)),
        "output_paths": outputs,
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
