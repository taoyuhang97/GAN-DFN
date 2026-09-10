#!/usr/bin/env python3
"""Step7A (太古界 v2): small-scale fracture patches with layered weighted
probability sampling and geologic orientation families.

Changes vs v1:
  * per-layer candidate reference quantile (no global threshold that removed
    the upper layer -> no more 'one surface' concentration);
  * occurrence probability proportional to density/reference (glutenite
    v2_weighted_probability style, no greedy global suppression);
  * orientation = layer template (from 405 imaging stats where available,
    config fallback otherwise) blended with local PCA-if-planar; dip clamped
    to [min_dip, max_dip];
  * imaging match QC: 405 well-corridor spatial bins + per-well distributional
    comparison for the other imaging wells.

Outputs (config.output_dir):
  fracture_patches.csv / fracture_patches.vtk
  step7a_imaging_match_qc.csv
  step7a_summary.json    status=pass
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small-scale fracture patches (太古界 Step7A v2).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=0, help="Smoke-test cap.")
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


def rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def normal_to_dip_azimuth(normal: np.ndarray) -> tuple[float, float]:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1.0e-12)
    dip = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
    horizontal = np.array([normal[0], normal[1]], dtype=np.float64)
    hnorm = np.linalg.norm(horizontal)
    if hnorm < 1.0e-6:
        azimuth = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(horizontal[1], horizontal[0]))) % 180.0
    return dip, azimuth


def write_legacy_vtk(path: Path, points: np.ndarray, quads: np.ndarray, cell_data: dict[str, np.ndarray]) -> None:
    lines = [
        "# vtk DataFile Version 3.0",
        "taigu_small_scale_fracture_patches_raw_time",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} double",
    ]
    lines.extend(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}" for p in points)
    lines.append(f"POLYGONS {len(quads)} {sum(4 + 1 for _ in quads)}")
    lines.extend(f"4 {' '.join(str(int(i)) for i in quad)}" for quad in quads)
    lines.append(f"CELL_DATA {len(quads)}")
    for name, values in cell_data.items():
        values = np.asarray(values)
        if values.dtype.kind in "biu":
            lines.append(f"SCALARS {name} int 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(str(int(v)) for v in values)
        else:
            lines.append(f"SCALARS {name} double 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(f"{float(v):.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def circular_mean_deg(degrees: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    radians = radians[np.isfinite(radians)]
    if radians.size == 0:
        return np.nan
    mean = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    return float(np.rad2deg(mean) % 360.0)


def imaging_orientation_prior(groups_root: Path, well: str, md_tolerance: float = 0.011) -> dict[str, dict[str, float]]:
    """Per-strata azimuth/dip prior from the imaging well's fracture points."""
    out: dict[str, dict[str, float]] = {}
    group_files = sorted(Path(groups_root).glob(f"{well}_*.csv"))
    for group_file in group_files:
        group = pd.read_csv(group_file, encoding="utf-8-sig")
        if group.empty:
            continue
        points = group[group["GT_POINT_FLAG"].fillna(0).astype(int) == 1]
        for strata, sub in points.groupby("StrataName"):
            az = pd.to_numeric(sub["FracAzimuth"], errors="coerce").dropna().to_numpy()
            dip = pd.to_numeric(sub["FracDip"], errors="coerce").dropna().to_numpy()
            if len(az) >= 5 and len(dip) >= 5:
                out[str(strata)] = {
                    "azimuth_circular_mean_deg": circular_mean_deg(az),
                    "dip_mean_deg": float(np.mean(dip)),
                    "dip_std_deg": float(np.std(dip)),
                    "n_points": int(len(az)),
                }
    return out


def load_imaging_density_profile(config: dict[str, Any], well: str) -> pd.DataFrame:
    """Imaging density vs TIME (borrowed) for QC, joining every segment path."""
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
                tolerance=float(config.get("md_merge_tolerance", 0.011)),
            )
            parts.append(merged[["X", "Y", "TIME", "StrataName", "Density"]])
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    for column in ("X", "Y", "TIME", "Density"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["X", "Y", "TIME", "Density"]).copy()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_csv = output_dir / "fracture_patches.csv"
    vtk_path = output_dir / "fracture_patches.vtk"
    qc_csv = output_dir / "step7a_imaging_match_qc.csv"
    summary_path = output_dir / "step7a_summary.json"
    for path in (patches_csv, vtk_path, qc_csv, summary_path):
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing Step7A output: {path}")

    started = time.time()
    grid = pd.read_csv(config["demo_grid_csv"], encoding="utf-8-sig")
    # SGY/window_code 的物理存储顺序是 output_trace_index；demo_grid 的
    # TraceIdx 是合同道序，二者不能直接互换。按坐标回接正式道序。
    mapping_path = config.get("trace_mapping_npz")
    if mapping_path:
        with np.load(mapping_path) as mp:
            mapping_df = pd.DataFrame({
                "X": mp["x"].astype(float), "Y": mp["y"].astype(float),
                "OutputTraceIndex": mp["output_trace_index"].astype(np.int64),
            })
        grid = grid.merge(mapping_df, on=["X", "Y"], how="left", validate="one_to_one")
        if grid["OutputTraceIndex"].isna().any():
            raise RuntimeError("demo grid contains coordinates absent from trace mapping")
        grid["OutputTraceIndex"] = grid["OutputTraceIndex"].astype(np.int64)
    else:
        # 兼容旧配置，但明确提示该模式只适用于两种道序恰好一致的输入。
        grid["OutputTraceIndex"] = np.arange(len(grid), dtype=np.int64)
    horizon = pd.read_csv(config["horizon_contract_csv"], encoding="utf-8-sig")
    for df in (grid, horizon):
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grid = grid.merge(
        horizon[["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"]],
        on="TraceIdx",
        how="left",
    ).sort_values("TraceIdx").reset_index(drop=True)
    grid["Row"] = np.arange(len(grid))
    if grid["TraceIdx"].duplicated().any():
        raise RuntimeError("demo grid has duplicate TraceIdx")
    cell_to_row = {(int(ix), int(iy)): int(row) for ix, iy, row in zip(grid["IX"], grid["IY"], grid["Row"])}
    row_to_ix = grid["IX"].to_numpy(dtype=np.int32)
    row_to_iy = grid["IY"].to_numpy(dtype=np.int32)
    row_xy = grid[["X", "Y"]].to_numpy(dtype=np.float64)

    windows = np.load(config["window_code_npz"])
    window_codes = windows["window_code"]
    sample_axis = windows["sample_axis"].astype(np.float64)
    density_sgy = Path(config["density_sgy"]).resolve()
    block_x_lines = max(int(config.get("block_x_line_count", 20)), 1)
    x_line_count = int(grid["IX"].max()) + 1
    sampling_cfg = config["sampling"]
    # 不按 demo 区域固定总片数；数量应随候选体素数量和 occurrence_rate
    # 自然增长。若工程上需要防止配置错误导致内存爆炸，只接受显式的
    # safety_max_patch_count 保护阈值，不参与正常采样和缩放。
    safety_max_patches = sampling_cfg.get("safety_max_patch_count")
    safety_max_patches = int(safety_max_patches) if safety_max_patches is not None else None
    reference_quantile = float(sampling_cfg["reference_quantile"])
    # 与砂砾岩小尺度流程一致：先按层内高分位筛掉背景体素，再以较低的
    # 出现概率抽样。这样不会把每个正密度体素都画成一块裂缝片。
    candidate_quantile = float(sampling_cfg.get("candidate_quantile", reference_quantile))
    occurrence_rate = float(sampling_cfg.get("occurrence_rate", 0.20))
    density_power = float(sampling_cfg.get("density_power", 1.0))
    length_range = [float(v) for v in config["patch_length_m"]]
    height_range = [float(v) for v in config["patch_height_ms"]]
    z_scale = float(config["display_z_scale_m_per_ms"])
    # VTK 使用原始 TIME(ms) 作为 Z；保留 z_scale 仅用于既有面积/展示口径。
    vtk_z_scale = float(config.get("vtk_z_scale_m_per_ms", 1.0))
    orient_cfg = config["orientation"]
    orient_xy_radius = int(orient_cfg["window_xy_radius"])
    orient_time_half = float(orient_cfg["time_half_span_ms"])
    orient_time_span = int(round(orient_time_half / (sample_axis[1] - sample_axis[0])))
    orient_density_fraction = float(orient_cfg["density_fraction"])
    min_planarity = float(orient_cfg["min_planarity"])
    min_local_points = int(orient_cfg["min_local_points"])
    min_dip = float(orient_cfg["min_dip_deg"])
    max_dip = float(orient_cfg["max_dip_deg"])
    az_jitter = float(orient_cfg["azimuth_jitter_std_deg"])
    dip_jitter = float(orient_cfg["dip_jitter_std_deg"])
    local_pca_share = float(orient_cfg["local_pca_share"])
    rng = np.random.default_rng(int(orient_cfg["random_seed"]))

    # per-layer positive-density samples for reference quantiles
    layer_refs: dict[str, float] = {}
    layer_positives: dict[str, list[np.ndarray]] = {"上部复合层": [], "太古界风化壳": []}
    candidate_parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "上部复合层": [],
        "太古界风化壳": [],
    }
    valid_mask = grid["SurfaceValid"].fillna(0).astype(bool).to_numpy()
    top = grid["TopTimeMs"].to_numpy(dtype=np.float64)
    mid = grid["MidTimeMs"].to_numpy(dtype=np.float64)
    base = grid["BaseTimeMs"].to_numpy(dtype=np.float64)

    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as handle:
        handle.mmap()
        for ix_start in tqdm(range(0, x_line_count, block_x_lines), desc="Step7A density blocks", unit="block"):
            ix_stop = min(ix_start + block_x_lines, x_line_count)
            block = grid[grid["IX"].between(ix_start, ix_stop - 1)].sort_values("OutputTraceIndex")
            output_indices = block["OutputTraceIndex"].to_numpy(dtype=np.int64)
            matrix = np.stack([np.asarray(handle.trace[int(r)], dtype=np.float32) for r in output_indices])
            codes = window_codes[output_indices]
            valid_block = block["SurfaceValid"].fillna(0).astype(bool).to_numpy()
            times_2d = sample_axis[None, :]
            upper_mask = (codes == 1) & (times_2d >= top[block["Row"].to_numpy(dtype=np.int64), None]) & (times_2d <= mid[block["Row"].to_numpy(dtype=np.int64), None])
            crust_mask = (codes == 1) & (times_2d > mid[block["Row"].to_numpy(dtype=np.int64), None]) & (times_2d <= base[block["Row"].to_numpy(dtype=np.int64), None])
            upper_mask &= valid_block[:, None] & np.isfinite(matrix) & (matrix > 0.0)
            crust_mask &= valid_block[:, None] & np.isfinite(matrix) & (matrix > 0.0)
            for name, mask in (("上部复合层", upper_mask), ("太古界风化壳", crust_mask)):
                if not mask.any():
                    continue
                layer_positives[name].append(matrix[mask].astype(np.float32))
                rows_m, samples_m = np.where(mask)
                candidate_parts[name].append(
                    (
                        block["Row"].to_numpy(dtype=np.int64)[rows_m],
                        samples_m.astype(np.int32),
                        matrix[mask].astype(np.float32),
                    )
                )
            print(f"[step7a] block ix={ix_start}:{ix_stop} rss_mb={rss_mb():.0f}", flush=True)
            if args.max_blocks > 0 and ix_start // block_x_lines + 1 >= args.max_blocks:
                break

    for name in ("上部复合层", "太古界风化壳"):
        positives = np.concatenate(layer_positives[name]) if layer_positives[name] else np.empty(0, dtype=np.float32)
        layer_refs[name] = float(np.quantile(positives, reference_quantile)) if positives.size else 0.0

    sampled_rows: list[np.ndarray] = []
    sampled_samples: list[np.ndarray] = []
    sampled_density: list[np.ndarray] = []
    sampled_layer: list[str] = []
    for name in ("上部复合层", "太古界风化壳"):
        if not candidate_parts[name]:
            continue
        rows = np.concatenate([p[0] for p in candidate_parts[name]])
        samples = np.concatenate([p[1] for p in candidate_parts[name]])
        density = np.concatenate([p[2] for p in candidate_parts[name]])
        # 候选阈值和参考值均在本层正值样本上计算，避免不同层厚/振幅
        # 分布差异造成某一层被异常过采样。
        candidate_ref = max(float(np.quantile(density, candidate_quantile)), 1.0e-9)
        candidate_mask = density >= candidate_ref
        rows, samples, density = rows[candidate_mask], samples[candidate_mask], density[candidate_mask]
        reference = max(float(layer_refs[name]), 1.0e-9)
        probability = occurrence_rate * np.power(np.clip(density / reference, 0.0, 2.0), density_power)
        probability = np.clip(probability, 0.0, 1.0)
        expected = float(probability.sum())
        keep = rng.random(len(rows)) < probability
        keep_pos = np.where(keep)[0]
        keep = np.zeros(len(rows), dtype=bool)
        keep[keep_pos] = True
        sampled_rows.append(rows[keep])
        sampled_samples.append(samples[keep])
        sampled_density.append(density[keep])
        if not (len(rows[keep]) == len(samples[keep]) == len(density[keep])):
            raise RuntimeError(f"sampled candidate length mismatch for layer {name}")
        sampled_layer.extend([name] * int(keep.sum()))
        print(f"[step7a] layer={name} candidate_q={candidate_quantile:.2f} threshold={candidate_ref:.3f} ref={reference:.3f} candidates={len(rows)} sampled={int(keep.sum())}", flush=True)

    accepted_rows = np.concatenate(sampled_rows) if sampled_rows else np.empty(0, dtype=np.int64)
    accepted_samples = np.concatenate(sampled_samples) if sampled_samples else np.empty(0, dtype=np.int64)
    accepted_density = np.concatenate(sampled_density) if sampled_density else np.empty(0, dtype=np.float32)
    accepted_layer = np.asarray(sampled_layer)
    if len(accepted_rows) == 0:
        raise RuntimeError("layered weighted probability sampling produced zero patches")
    print(f"[step7a] accepted_patches={len(accepted_rows)} rss_mb={rss_mb():.0f}", flush=True)

    # --- orientation: layer template blended with local PCA-if-planar ---
    prior = imaging_orientation_prior(Path(config["step3_groups_root"]), config["profile_well"])
    fallback = config["orientation"]["fallback_family"]
    template: dict[str, dict[str, float]] = {}
    for layer in ("上部复合层", "太古界风化壳"):
        if layer in prior:
            template[layer] = {
                "azimuth": float(prior[layer]["azimuth_circular_mean_deg"] % 180.0),
                "dip": float(prior[layer]["dip_mean_deg"]),
                "dip_std": float(prior[layer]["dip_std_deg"]),
            }
        else:
            template[layer] = {
                "azimuth": float(fallback[layer]["azimuth_deg"]),
                "dip": float(fallback[layer]["dip_deg"]),
                "dip_std": dip_jitter,
            }

    dips: list[float] = []
    azimuths: list[float] = []
    local_dips: list[float] = []
    local_azimuths: list[float] = []
    base_sources: list[str] = []
    families: list[str] = []
    orientation_chunk = 5000
    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as handle:
        for chunk_start in range(0, len(accepted_rows), orientation_chunk):
            chunk_end = min(chunk_start + orientation_chunk, len(accepted_rows))
            chunk_rows = accepted_rows[chunk_start:chunk_end].tolist()
            chunk_samples = accepted_samples[chunk_start:chunk_end].tolist()
            chunk_layers = accepted_layer[chunk_start:chunk_end].tolist()
            neighbor_rows: dict[int, list[int]] = {}
            needed_rows: set[int] = set()
            for row in chunk_rows:
                ix = int(row_to_ix[row])
                iy = int(row_to_iy[row])
                neighbors: list[int] = []
                for di in range(-orient_xy_radius, orient_xy_radius + 1):
                    for dj in range(-orient_xy_radius, orient_xy_radius + 1):
                        nrow = cell_to_row.get((ix + di, iy + dj))
                        if nrow is not None:
                            neighbors.append(nrow)
                            needed_rows.add(nrow)
                neighbor_rows[row] = neighbors
            dens_by_row: dict[int, np.ndarray] = {
                row: np.asarray(handle.trace[int(grid.loc[row, "OutputTraceIndex"])], dtype=np.float32) for row in needed_rows
            }
            for local_idx, (row, sample, layer) in enumerate(zip(chunk_rows, chunk_samples, chunk_layers)):
                center_density = float(accepted_density[chunk_start + local_idx])
                local_pts: list[np.ndarray] = []
                weights: list[float] = []
                for nrow in neighbor_rows[row]:
                    dens_trace = dens_by_row[nrow]
                    for ds in range(-orient_time_span, orient_time_span + 1):
                        s = sample + ds
                        if 0 <= s < len(sample_axis) and np.isfinite(dens_trace[s]):
                            value = float(dens_trace[s])
                            if value >= orient_density_fraction * center_density:
                                local_pts.append(
                                    np.array(
                                        [
                                            float(grid.loc[nrow, "X"]),
                                            float(grid.loc[nrow, "Y"]),
                                            float(sample_axis[s]) * z_scale,
                                        ]
                                    )
                                )
                                weights.append(value)
                local_dip = np.nan
                local_az = np.nan
                if len(local_pts) >= min_local_points:
                    coords = np.stack(local_pts)
                    weights_arr = np.asarray(weights, dtype=np.float64)
                    weights_arr = weights_arr / weights_arr.sum()
                    centered = coords - (coords * weights_arr[:, None]).sum(axis=0)
                    covariance = (centered * weights_arr[:, None]).T @ centered
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    evals = np.sort(eigenvalues)
                    planarity = 1.0 - evals[0] / max(evals[1], 1.0e-12) if evals[1] > 0 else 0.0
                    if planarity >= min_planarity:
                        normal = eigenvectors[:, np.argmin(eigenvalues)]
                        local_dip, local_az = normal_to_dip_azimuth(normal)
                local_dips.append(local_dip)
                local_azimuths.append(local_az)
                use_local = (
                    np.isfinite(local_dip)
                    and np.isfinite(local_az)
                    and min_dip <= local_dip <= max_dip
                    and rng.random() < local_pca_share
                )
                if use_local:
                    base_azimuth = local_az
                    base_dip = local_dip
                    base_source = "local_pca_if_planar"
                    family = "local_pca"
                else:
                    base_azimuth = float(template[layer]["azimuth"])
                    base_dip = float(template[layer]["dip"])
                    base_source = "layer_template"
                    family = "layer_template"
                azimuth = float((base_azimuth + rng.normal(0.0, az_jitter)) % 180.0)
                dip = float(np.clip(base_dip + rng.normal(0.0, float(template[layer].get("dip_std", dip_jitter))), min_dip, max_dip))
                azimuths.append(azimuth)
                dips.append(dip)
                base_sources.append(base_source)
                families.append(family)
            del dens_by_row, neighbor_rows
            gc.collect()
            print(f"[step7a] orientation chunk {chunk_start}:{chunk_end} of {len(accepted_rows)}", flush=True)

    lengths = rng.uniform(length_range[0], length_range[1], size=len(accepted_rows))
    heights = rng.uniform(height_range[0], height_range[1], size=len(accepted_rows))
    areas = lengths * heights * z_scale
    patches = pd.DataFrame(
        {
            "PatchID": [f"taigu_small_{i:07d}" for i in range(len(accepted_rows))],
            "TraceIdx": [int(grid.loc[r, "TraceIdx"]) for r in accepted_rows],
            "X": [float(grid.loc[r, "X"]) for r in accepted_rows],
            "Y": [float(grid.loc[r, "Y"]) for r in accepted_rows],
            "TIME": [float(sample_axis[s]) for s in accepted_samples],
            "LayerGroup": accepted_layer,
            "Density": accepted_density,
            "DipDeg": dips,
            "AzimuthDeg": azimuths,
            "LocalPcaDipDeg": local_dips,
            "LocalPcaAzimuthDeg": local_azimuths,
            "OrientationBaseSource": base_sources,
            "OrientationFamily": families,
            "PatchLengthM": lengths,
            "PatchHeightMs": heights,
            "PatchAreaM2": areas,
            "FractureScale": "small",
            "WindowCode": 1,
        }
    )
    patches.to_csv(patches_csv, index=False, encoding="utf-8-sig")

    points: list[np.ndarray] = []
    quads: list[np.ndarray] = []
    for i, patch in patches.iterrows():
        center = np.array([patch["X"], patch["Y"], patch["TIME"] * vtk_z_scale])
        strike_rad = np.radians(patch["AzimuthDeg"])
        strike = np.array([np.cos(strike_rad), np.sin(strike_rad), 0.0])
        dip_dir = np.array([-np.sin(strike_rad), np.cos(strike_rad), 0.0])
        dip_rad = np.radians(patch["DipDeg"])
        dip_vec = np.array([np.sin(dip_rad) * dip_dir[0], np.sin(dip_rad) * dip_dir[1], np.cos(dip_rad)])
        half_l = patch["PatchLengthM"] / 2.0
        half_h = patch["PatchHeightMs"] * vtk_z_scale / 2.0
        corners = [
            center + half_l * strike + half_h * dip_vec,
            center + half_l * strike - half_h * dip_vec,
            center - half_l * strike - half_h * dip_vec,
            center - half_l * strike + half_h * dip_vec,
        ]
        base = len(points)
        points.extend(corners)
        quads.append(np.array([base, base + 1, base + 2, base + 3], dtype=np.int64))
    cell_data = {
        "PatchID": np.arange(len(patches), dtype=np.int64),
        "Density": patches["Density"].to_numpy(dtype=np.float64),
        "DipDeg": patches["DipDeg"].to_numpy(dtype=np.float64),
        "AzimuthDeg": patches["AzimuthDeg"].to_numpy(dtype=np.float64),
        "PatchAreaM2": patches["PatchAreaM2"].to_numpy(dtype=np.float64),
        "FractureScale": np.full(len(patches), 1, dtype=np.int32),
        "WindowCode": np.full(len(patches), 1, dtype=np.int32),
    }
    write_legacy_vtk(vtk_path, np.stack(points) if points else np.empty((0, 3)), np.stack(quads) if quads else np.empty((0, 4), dtype=np.int64), cell_data)

    # --- imaging match QC ---
    qc_rows: list[dict[str, Any]] = []
    profile_well = str(config["profile_well"])
    profile = load_imaging_density_profile(config, profile_well)
    if len(profile):
        corridor = patches[
            (patches["X"] >= profile["X"].min() - 500)
            & (patches["X"] <= profile["X"].max() + 500)
            & (patches["Y"] >= profile["Y"].min() - 500)
            & (patches["Y"] <= profile["Y"].max() + 500)
        ]
        bins = np.arange(profile["TIME"].min(), profile["TIME"].max() + 20, 20)
        imaging_means = []
        patch_counts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            imaging_means.append(float(profile[(profile["TIME"] >= lo) & (profile["TIME"] < hi)]["Density"].mean()))
            patch_counts.append(int(((corridor["TIME"] >= lo) & (corridor["TIME"] < hi)).sum()))
        from scipy.stats import spearmanr
        corr = spearmanr(imaging_means, patch_counts) if len(imaging_means) >= 3 else None
        qc_rows.append(
            {
                "WellName": profile_well,
                "QCKind": "well_corridor_spatial",
                "CorridorHalfWidthM": 500.0,
                "TimeBinMs": 20.0,
                "ImagingTimeRangeMs": [float(profile["TIME"].min()), float(profile["TIME"].max())],
                "PatchCountInCorridor": int(len(corridor)),
                "SpearmanDensityVsPatches": float(corr.statistic) if corr is not None else None,
                "Note": "co-located amplitude available",
            }
        )
    strong_wells = ["埕北310", "埕北313", "埕北816", "桩斜169", "桩海102"]
    for well in strong_wells:
        img = load_imaging_density_profile(config, well)
        if img.empty:
            continue
        for strata, sub in img.groupby("StrataName"):
            dens = sub["Density"].to_numpy(dtype=np.float64)
            dfn_dens = patches.loc[patches["LayerGroup"] == strata, "Density"].to_numpy(dtype=np.float64)
            qc_rows.append(
                {
                    "WellName": well,
                    "QCKind": "distributional_reference",
                    "CorridorHalfWidthM": None,
                    "TimeBinMs": None,
                    "StrataName": str(strata),
                    "ImagingDensityP50": float(np.median(dens)),
                    "ImagingDensityP90": float(np.quantile(dens, 0.9)),
                    "DfnDensityP50": float(np.median(dfn_dens)) if len(dfn_dens) else None,
                    "DfnDensityP90": float(np.quantile(dfn_dens, 0.9)) if len(dfn_dens) else None,
                    "Note": "non co-located, distributional reference only",
                }
            )
    pd.DataFrame(qc_rows).to_csv(qc_csv, index=False, encoding="utf-8-sig")

    layer_counts = patches["LayerGroup"].value_counts().to_dict()
    upper_fraction = float(layer_counts.get("上部复合层", 0) / max(len(patches), 1))
    checks = {
        "patch_count_positive": len(patches) > 0,
        "patch_count_within_cap": safety_max_patches is None or len(patches) <= safety_max_patches,
        "both_layers_represented": upper_fraction > 0.05 and float(layer_counts.get("太古界风化壳", 0)) > 0,
        "all_patches_main_window": bool((patches["WindowCode"] == 1).all()),
        "orientation_ranges_valid": bool(patches["DipDeg"].between(0, 90).all() and patches["AzimuthDeg"].between(0, 180).all()),
        "dip_min_constraint": bool((patches["DipDeg"] >= min_dip - 1.0e-6).all()),
        "geometry_finite": bool(np.isfinite(patches[["X", "Y", "TIME", "PatchAreaM2"]]).all().all()),
        "imaging_match_qc_exists": qc_csv.exists(),
        "vtk_exists": vtk_path.exists(),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "output_paths": {
            "patches_csv": str(patches_csv),
            "patches_vtk": str(vtk_path),
            "imaging_match_qc_csv": str(qc_csv),
            "summary_json": str(summary_path),
        },
        "sampling": {
            "mode": sampling_cfg["mode"],
            "reference_quantiles": layer_refs,
            "layer_patch_counts": layer_counts,
            "upper_layer_fraction": upper_fraction,
        },
        "orientation": {
            "imaging_prior": prior,
            "template": template,
            "family_counts": patches["OrientationFamily"].value_counts().to_dict(),
            "dip_stats": {"min": float(patches["DipDeg"].min()), "max": float(patches["DipDeg"].max()), "mean": float(patches["DipDeg"].mean())},
            "azimuth_stats": {"min": float(patches["AzimuthDeg"].min()), "max": float(patches["AzimuthDeg"].max()), "mean": float(patches["AzimuthDeg"].mean())},
        },
        "accepted_patch_count": int(len(patches)),
        "imaging_match_qc_rows": int(len(qc_rows)),
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
