#!/usr/bin/env python3
"""Step7A (太古界): small-scale fracture patches with layered weighted
probability sampling, density-priority 3D spacing, and geologic orientations.

Changes vs v1:
  * per-layer candidate reference quantile (no global threshold that removed
    the upper layer -> no more 'one surface' concentration);
  * occurrence probability proportional to density/reference (glutenite
    v2_weighted_probability style, no greedy global suppression);
  * orientation = multi-well, per-layer orientation families combined with
    local density-gradient/ridge evidence; manual templates are last-resort
    fallbacks only;
  * imaging match QC: 405 well-corridor spatial bins + per-well distributional
    comparison for the other imaging wells.

Outputs (config.output_dir):
  fracture_patches.csv / fracture_patches.vtk
  orientation_families.json / orientation_families.csv
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
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small-scale fracture patches (太古界 Step7A v2).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=0, help="Smoke-test cap.")
    parser.add_argument(
        "--sampling-qc-only",
        action="store_true",
        help="Stop after candidate sampling and spatial thinning; skip orientation/VTK.",
    )
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


def layer_parameter(value: Any, layer: str) -> float:
    """Read either a scalar or a per-layer numeric configuration value."""
    if isinstance(value, dict):
        if layer not in value:
            raise KeyError(f"missing per-layer sampling parameter for {layer}")
        return float(value[layer])
    return float(value)


def thin_by_3d_spacing(
    rows: np.ndarray,
    samples: np.ndarray,
    priority: np.ndarray,
    row_xy: np.ndarray,
    sample_axis: np.ndarray,
    min_distance_m: float,
    time_scale_m_per_ms: float,
) -> np.ndarray:
    """Greedily retain stronger candidates separated in scaled XYZ space."""
    if min_distance_m <= 0.0 or len(rows) <= 1:
        return np.arange(len(rows), dtype=np.int64)
    points = np.column_stack(
        [row_xy[rows, 0], row_xy[rows, 1], sample_axis[samples] * time_scale_m_per_ms]
    )
    order = np.argsort(-priority, kind="stable")
    buckets: dict[tuple[int, int, int], list[int]] = {}
    retained: list[int] = []
    threshold_sq = min_distance_m * min_distance_m
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for candidate_index in tqdm(order, desc="Step7A 3D spacing", unit="candidate"):
        point = points[candidate_index]
        key = tuple(np.floor(point / min_distance_m).astype(np.int64))
        too_close = False
        for di, dj, dk in offsets:
            for accepted_index in buckets.get((key[0] + di, key[1] + dj, key[2] + dk), ()):
                delta = point - points[accepted_index]
                if float(delta @ delta) < threshold_sq:
                    too_close = True
                    break
            if too_close:
                break
        if not too_close:
            retained.append(int(candidate_index))
            buckets.setdefault(key, []).append(int(candidate_index))
    return np.asarray(retained, dtype=np.int64)


def normal_to_dip_azimuth(normal: np.ndarray) -> tuple[float, float]:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1.0e-12)
    dip = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
    horizontal = np.array([normal[0], normal[1]], dtype=np.float64)
    hnorm = np.linalg.norm(horizontal)
    if hnorm < 1.0e-6:
        azimuth = 0.0
    else:
        normal_azimuth = float(np.degrees(np.arctan2(horizontal[1], horizontal[0]))) % 180.0
        # The DFN geometry interprets AzimuthDeg as strike.  A plane normal's
        # horizontal projection is the dip direction, so rotate it by 90 deg.
        azimuth = (normal_azimuth + 90.0) % 180.0
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


def axial_mean_std_deg(degrees: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Mean/spread for fracture strike where theta and theta+180 are equal."""
    angle = np.asarray(degrees, dtype=np.float64)
    valid = np.isfinite(angle)
    angle = angle[valid]
    if not len(angle):
        return np.nan, np.nan
    weight = np.ones(len(angle), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)[valid]
    weight = weight / max(float(weight.sum()), 1.0e-12)
    doubled = np.deg2rad(2.0 * angle)
    vector = np.sum(weight * np.exp(1j * doubled))
    mean = float((np.rad2deg(np.angle(vector)) / 2.0) % 180.0)
    resultant = float(np.clip(abs(vector), 1.0e-12, 1.0))
    std = float(np.rad2deg(np.sqrt(max(-2.0 * np.log(resultant), 0.0))) / 2.0)
    return mean, std


def load_multiwell_orientation_points(groups_root: Path, allowed_wells: dict[str, list[str]]) -> pd.DataFrame:
    """Load valid image-log orientations and attach their interpolated well XY."""
    parts: list[pd.DataFrame] = []
    allowed = {layer: set(wells) for layer, wells in allowed_wells.items()}
    for group_file in sorted(Path(groups_root).glob("*.csv")):
        group = pd.read_csv(group_file, encoding="utf-8-sig")
        required = {"WellName", "StrataName", "GT_POINT_FLAG", "FracAzimuth", "FracDip", "MD", "InputSegmentPath"}
        if group.empty or not required.issubset(group.columns):
            continue
        points = group[group["GT_POINT_FLAG"].fillna(0).astype(int).eq(1)].copy()
        points["FracAzimuth"] = pd.to_numeric(points["FracAzimuth"], errors="coerce") % 180.0
        points["FracDip"] = pd.to_numeric(points["FracDip"], errors="coerce")
        points["MD"] = pd.to_numeric(points["MD"], errors="coerce")
        points = points.dropna(subset=["FracAzimuth", "FracDip", "MD"])
        keep = np.asarray(
            [str(well) in allowed.get(str(layer), set()) for well, layer in zip(points["WellName"], points["StrataName"])],
            dtype=bool,
        )
        points = points.loc[keep].copy()
        if points.empty:
            continue
        for segment_path, sub in points.groupby("InputSegmentPath"):
            segment = pd.read_csv(segment_path, encoding="utf-8-sig", usecols=["MD", "X", "Y"])
            for column in ("MD", "X", "Y"):
                segment[column] = pd.to_numeric(segment[column], errors="coerce")
            segment = segment.dropna().sort_values("MD")
            if segment.empty:
                continue
            merged = pd.merge_asof(
                sub.sort_values("MD"), segment, on="MD", direction="nearest", tolerance=0.05
            )
            parts.append(merged[["WellName", "StrataName", "FracAzimuth", "FracDip", "X", "Y"]])
    if not parts:
        return pd.DataFrame(columns=["WellName", "StrataName", "FracAzimuth", "FracDip", "X", "Y"])
    return pd.concat(parts, ignore_index=True).dropna()


def build_orientation_families(points: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Cluster axial strike/dip observations with approximately equal well influence."""
    families: dict[str, list[dict[str, Any]]] = {}
    random_state = int(config.get("family_random_seed", 20260822))
    dip_weight = float(config.get("family_dip_feature_weight", 0.7))
    for layer in ("上部复合层", "太古界风化壳"):
        sub = points[points["StrataName"].astype(str).eq(layer)].copy().reset_index(drop=True)
        cluster_count = min(int(config["family_count_by_layer"][layer]), len(sub))
        if cluster_count <= 0:
            families[layer] = []
            continue
        angle = np.deg2rad(2.0 * sub["FracAzimuth"].to_numpy(dtype=np.float64))
        dip = sub["FracDip"].to_numpy(dtype=np.float64)
        features = np.column_stack([np.cos(angle), np.sin(angle), dip_weight * (dip - 55.0) / 25.0])
        per_well_count = sub.groupby("WellName")["WellName"].transform("size").to_numpy(dtype=np.float64)
        fit_weight = 1.0 / np.maximum(per_well_count, 1.0)
        model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=20)
        labels = model.fit_predict(features, sample_weight=fit_weight)
        sub["FamilyLabel"] = labels
        layer_families: list[dict[str, Any]] = []
        for label in range(cluster_count):
            family_points = sub[sub["FamilyLabel"].eq(label)].copy()
            family_well_count = family_points.groupby("WellName")["WellName"].transform("size").to_numpy(dtype=np.float64)
            balanced_weight = 1.0 / np.maximum(family_well_count, 1.0)
            azimuth, azimuth_std = axial_mean_std_deg(
                family_points["FracAzimuth"].to_numpy(dtype=np.float64), balanced_weight
            )
            balanced_weight /= balanced_weight.sum()
            dip_values = family_points["FracDip"].to_numpy(dtype=np.float64)
            dip_mean = float(np.sum(balanced_weight * dip_values))
            dip_std = float(np.sqrt(np.sum(balanced_weight * np.square(dip_values - dip_mean))))
            contributions = []
            for well, well_points in family_points.groupby("WellName"):
                all_well_count = int(sub["WellName"].eq(well).sum())
                contributions.append(
                    {
                        "well": str(well),
                        "family_point_count": int(len(well_points)),
                        "well_point_count": all_well_count,
                        "within_well_fraction": float(len(well_points) / max(all_well_count, 1)),
                        "x": float(well_points["X"].median()),
                        "y": float(well_points["Y"].median()),
                    }
                )
            layer_families.append(
                {
                    "family_id": f"{layer}_family_{label + 1}",
                    "azimuth_deg": azimuth,
                    "azimuth_std_deg": azimuth_std,
                    "dip_deg": dip_mean,
                    "dip_std_deg": dip_std,
                    "point_count": int(len(family_points)),
                    "well_count": int(family_points["WellName"].nunique()),
                    "well_contributions": contributions,
                }
            )
        layer_families.sort(key=lambda item: (-item["well_count"], -item["point_count"]))
        for index, family in enumerate(layer_families, start=1):
            family["family_id"] = f"{layer}_family_{index}"
        families[layer] = layer_families
    return families


def choose_orientation_family(
    families: list[dict[str, Any]], x: float, y: float, distance_scale_m: float, rng: np.random.Generator
) -> tuple[dict[str, Any], float]:
    scores = []
    for family in families:
        score = 0.0
        for source in family["well_contributions"]:
            distance = float(np.hypot(x - source["x"], y - source["y"]))
            score += source["within_well_fraction"] * np.exp(-distance / max(distance_scale_m, 1.0))
        scores.append(score)
    probability = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(probability).all() or probability.sum() <= 0.0:
        probability = np.ones(len(families), dtype=np.float64)
    probability /= probability.sum()
    index = int(rng.choice(len(families), p=probability))
    return families[index], float(probability[index])


def blend_axial_azimuth(family_azimuth: float, local_azimuth: float, local_weight: float) -> float:
    weight = float(np.clip(local_weight, 0.0, 1.0))
    family_vector = np.exp(1j * np.deg2rad(2.0 * family_azimuth))
    local_vector = np.exp(1j * np.deg2rad(2.0 * local_azimuth))
    return float((np.rad2deg(np.angle((1.0 - weight) * family_vector + weight * local_vector)) / 2.0) % 180.0)


def local_gradient_ridge_orientation(
    row: int,
    sample: int,
    center_density: float,
    cell_to_row: dict[tuple[int, int], int],
    row_to_ix: np.ndarray,
    row_to_iy: np.ndarray,
    dens_by_row: dict[int, np.ndarray],
    sample_axis: np.ndarray,
    xy_radius: int,
    time_radius: int,
    dx_m: float,
    dy_m: float,
    time_scale_m_per_ms: float,
    density_fraction: float,
    gradient_quantile: float,
    smoothing_sigma: float,
) -> dict[str, float | int]:
    """Estimate fracture-plane normal from local density edges and ridges."""
    shape = (2 * xy_radius + 1, 2 * xy_radius + 1, 2 * time_radius + 1)
    cube = np.full(shape, np.nan, dtype=np.float64)
    center_ix, center_iy = int(row_to_ix[row]), int(row_to_iy[row])
    for xi, di in enumerate(range(-xy_radius, xy_radius + 1)):
        for yi, dj in enumerate(range(-xy_radius, xy_radius + 1)):
            neighbor = cell_to_row.get((center_ix + di, center_iy + dj))
            if neighbor is None or neighbor not in dens_by_row:
                continue
            trace = dens_by_row[neighbor]
            for zi, ds in enumerate(range(-time_radius, time_radius + 1)):
                source_sample = sample + ds
                if 0 <= source_sample < len(trace):
                    cube[xi, yi, zi] = float(trace[source_sample])
    valid = np.isfinite(cube)
    if int(valid.sum()) < 27:
        return {"dip": np.nan, "azimuth": np.nan, "anisotropy": 0.0, "gradient_strength": 0.0,
                "ridge_strength": 0.0, "quality": 0.0, "support_count": int(valid.sum())}
    fill_value = float(np.nanmedian(cube))
    filled = np.where(valid, cube, fill_value)
    smooth = gaussian_filter(filled, sigma=smoothing_sigma, mode="nearest")
    dz_m = max(float(np.median(np.diff(sample_axis))) * time_scale_m_per_ms, 1.0e-6)
    gx, gy, gz = np.gradient(smooth, dx_m, dy_m, dz_m, edge_order=1)
    gradient_magnitude = np.sqrt(gx * gx + gy * gy + gz * gz)
    finite_gradient = gradient_magnitude[valid]
    if not len(finite_gradient) or float(np.nanmax(finite_gradient)) <= 0.0:
        return {"dip": np.nan, "azimuth": np.nan, "anisotropy": 0.0, "gradient_strength": 0.0,
                "ridge_strength": 0.0, "quality": 0.0, "support_count": int(valid.sum())}
    gradient_threshold = float(np.quantile(finite_gradient, gradient_quantile))
    gxx = np.gradient(gx, dx_m, axis=0, edge_order=1)
    gyy = np.gradient(gy, dy_m, axis=1, edge_order=1)
    gzz = np.gradient(gz, dz_m, axis=2, edge_order=1)
    ridge = np.maximum(-(gxx + gyy + gzz), 0.0)
    evidence = valid & (smooth >= density_fraction * center_density) & (
        (gradient_magnitude >= gradient_threshold) | (ridge >= np.quantile(ridge[valid], gradient_quantile))
    )
    support_count = int(evidence.sum())
    if support_count < 8:
        return {"dip": np.nan, "azimuth": np.nan, "anisotropy": 0.0, "gradient_strength": 0.0,
                "ridge_strength": 0.0, "quality": 0.0, "support_count": support_count}
    vectors = np.column_stack([gx[evidence], gy[evidence], gz[evidence]])
    local_gradient = gradient_magnitude[evidence]
    local_ridge = ridge[evidence]
    ridge_norm = local_ridge / max(float(np.quantile(ridge[valid], 0.95)), 1.0e-12)
    weights = local_gradient * (1.0 + np.clip(ridge_norm, 0.0, 2.0))
    weights /= max(float(weights.sum()), 1.0e-12)
    tensor = (vectors * weights[:, None]).T @ vectors
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalues)[::-1]
    leading, second = float(eigenvalues[order[0]]), float(eigenvalues[order[1]])
    anisotropy = max((leading - second) / max(leading, 1.0e-12), 0.0)
    normal = eigenvectors[:, order[0]]
    dip, azimuth = normal_to_dip_azimuth(normal)
    p50 = float(np.quantile(finite_gradient, 0.50))
    p90 = float(np.quantile(finite_gradient, 0.90))
    gradient_strength = float(np.clip((p90 - p50) / max(p90, 1.0e-12), 0.0, 1.0))
    ridge_strength = float(np.clip(np.median(ridge_norm), 0.0, 1.0))
    quality = float(np.clip(0.65 * anisotropy + 0.25 * gradient_strength + 0.10 * ridge_strength, 0.0, 1.0))
    return {
        "dip": dip, "azimuth": azimuth, "anisotropy": anisotropy,
        "gradient_strength": gradient_strength, "ridge_strength": ridge_strength,
        "quality": quality, "support_count": support_count,
    }


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
    families_json_path = output_dir / "orientation_families.json"
    families_csv_path = output_dir / "orientation_families.csv"
    sampling_qc_path = output_dir / "step7a_sampling_qc.json"
    summary_path = output_dir / "step7a_summary.json"
    output_paths_to_check = (sampling_qc_path,) if args.sampling_qc_only else (
        patches_csv, vtk_path, qc_csv, families_json_path, families_csv_path,
        sampling_qc_path, summary_path
    )
    for path in output_paths_to_check:
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
    prediction_summary_path = Path(config["step6a_prediction_summary_json"]).resolve()
    prediction_summary = read_json(prediction_summary_path)
    if prediction_summary.get("status") != "pass":
        raise RuntimeError("Step6A prediction summary is not pass")
    if Path(prediction_summary["output_paths"]["density_sgy"]).resolve() != density_sgy:
        raise RuntimeError("Step7A density_sgy does not match the declared Step6A prediction summary")
    expected_step6a_contract = config.get("expected_step6a_model_contract_version")
    if expected_step6a_contract and prediction_summary.get("model_contract_version") != expected_step6a_contract:
        raise RuntimeError(
            f"Step6A model contract mismatch: {prediction_summary.get('model_contract_version')} "
            f"!= {expected_step6a_contract}"
        )
    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as density_handle:
        if density_handle.tracecount != len(grid):
            raise RuntimeError("Step6A density trace count does not match Step7A demo grid")
        if len(density_handle.samples) != len(sample_axis) or not np.allclose(density_handle.samples, sample_axis):
            raise RuntimeError("Step6A density sample axis does not match window_code sample axis")
    if window_codes.shape != (len(grid), len(sample_axis)):
        raise RuntimeError("window_code shape does not match Step7A grid/sample axis")
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
    candidate_quantile_cfg = sampling_cfg.get("candidate_quantile", reference_quantile)
    occurrence_rate_cfg = sampling_cfg.get("occurrence_rate", 0.20)
    density_power = float(sampling_cfg.get("density_power", 1.0))
    min_center_distance_m = float(sampling_cfg.get("min_center_distance_m", 0.0))
    spacing_time_scale = float(sampling_cfg.get("spacing_time_scale_m_per_ms", 2.0))
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
    min_dip = float(orient_cfg["min_dip_deg"])
    max_dip = float(orient_cfg["max_dip_deg"])
    az_jitter = float(orient_cfg["azimuth_jitter_std_deg"])
    dip_jitter = float(orient_cfg["dip_jitter_std_deg"])
    sampling_rng = np.random.default_rng(int(sampling_cfg["random_seed"]))
    orientation_rng = np.random.default_rng(int(orient_cfg["random_seed"]))

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
    sampled_priority: list[np.ndarray] = []
    sampled_layer: list[str] = []
    layer_sampling_audit: dict[str, dict[str, Any]] = {}
    for name in ("上部复合层", "太古界风化壳"):
        if not candidate_parts[name]:
            continue
        rows = np.concatenate([p[0] for p in candidate_parts[name]])
        samples = np.concatenate([p[1] for p in candidate_parts[name]])
        density = np.concatenate([p[2] for p in candidate_parts[name]])
        # 候选阈值和参考值均在本层正值样本上计算，避免不同层厚/振幅
        # 分布差异造成某一层被异常过采样。
        positive_count = int(len(density))
        candidate_quantile = layer_parameter(candidate_quantile_cfg, name)
        occurrence_rate = layer_parameter(occurrence_rate_cfg, name)
        candidate_ref = max(float(np.quantile(density, candidate_quantile)), 1.0e-9)
        candidate_mask = density >= candidate_ref
        rows, samples, density = rows[candidate_mask], samples[candidate_mask], density[candidate_mask]
        reference = max(float(layer_refs[name]), 1.0e-9)
        probability = occurrence_rate * np.power(np.clip(density / reference, 0.0, 2.0), density_power)
        probability = np.clip(probability, 0.0, 1.0)
        expected = float(probability.sum())
        keep = sampling_rng.random(len(rows)) < probability
        keep_pos = np.where(keep)[0]
        keep = np.zeros(len(rows), dtype=bool)
        keep[keep_pos] = True
        sampled_rows.append(rows[keep])
        sampled_samples.append(samples[keep])
        sampled_density.append(density[keep])
        sampled_priority.append((density[keep] / reference).astype(np.float32))
        if not (len(rows[keep]) == len(samples[keep]) == len(density[keep])):
            raise RuntimeError(f"sampled candidate length mismatch for layer {name}")
        sampled_layer.extend([name] * int(keep.sum()))
        layer_sampling_audit[name] = {
            "positive_voxel_count": positive_count,
            "candidate_quantile": candidate_quantile,
            "candidate_threshold": candidate_ref,
            "candidate_count": int(len(rows)),
            "reference_quantile": reference_quantile,
            "reference_density": reference,
            "occurrence_rate": occurrence_rate,
            "density_power": density_power,
            "expected_before_spacing": expected,
            "sampled_before_spacing": int(keep.sum()),
        }
        print(f"[step7a] layer={name} candidate_q={candidate_quantile:.2f} threshold={candidate_ref:.3f} ref={reference:.3f} candidates={len(rows)} sampled={int(keep.sum())}", flush=True)

    accepted_rows = np.concatenate(sampled_rows) if sampled_rows else np.empty(0, dtype=np.int64)
    accepted_samples = np.concatenate(sampled_samples) if sampled_samples else np.empty(0, dtype=np.int64)
    accepted_density = np.concatenate(sampled_density) if sampled_density else np.empty(0, dtype=np.float32)
    accepted_priority = np.concatenate(sampled_priority) if sampled_priority else np.empty(0, dtype=np.float32)
    accepted_layer = np.asarray(sampled_layer)
    if len(accepted_rows) == 0:
        raise RuntimeError("layered weighted probability sampling produced zero patches")
    pre_spacing_count = int(len(accepted_rows))
    retained = thin_by_3d_spacing(
        accepted_rows,
        accepted_samples,
        accepted_priority,
        row_xy,
        sample_axis,
        min_center_distance_m,
        spacing_time_scale,
    )
    accepted_rows = accepted_rows[retained]
    accepted_samples = accepted_samples[retained]
    accepted_density = accepted_density[retained]
    accepted_priority = accepted_priority[retained]
    accepted_layer = accepted_layer[retained]
    post_counts = pd.Series(accepted_layer).value_counts().to_dict()
    for layer, audit in layer_sampling_audit.items():
        audit["retained_after_spacing"] = int(post_counts.get(layer, 0))
    sampling_qc = {
        "status": "pass" if len(accepted_rows) > 0 else "fail",
        "config_path": str(args.config.resolve()),
        "mode": sampling_cfg["mode"],
        "layer_audit": layer_sampling_audit,
        "pre_spacing_patch_count": pre_spacing_count,
        "post_spacing_patch_count": int(len(accepted_rows)),
        "spacing_removed_count": int(pre_spacing_count - len(accepted_rows)),
        "min_center_distance_m": min_center_distance_m,
        "spacing_time_scale_m_per_ms": spacing_time_scale,
        "fixed_total_patch_cap_used": False,
    }
    write_json(sampling_qc_path, json_ready(sampling_qc))
    print(
        f"[step7a] sampled_before_spacing={pre_spacing_count} "
        f"retained_after_spacing={len(accepted_rows)} rss_mb={rss_mb():.0f}",
        flush=True,
    )
    if args.sampling_qc_only:
        print(json.dumps(json_ready(sampling_qc), ensure_ascii=False, indent=2))
        return 0 if sampling_qc["status"] == "pass" else 1

    # --- orientation: multi-well families + local gradient/ridge evidence ---
    imaging_points = load_multiwell_orientation_points(
        Path(config["step3_groups_root"]), orient_cfg["imaging_wells_by_layer"]
    )
    orientation_families = build_orientation_families(imaging_points, orient_cfg)
    write_json(families_json_path, json_ready({
        "method": "balanced_multiwell_axial_kmeans",
        "imaging_wells_by_layer": orient_cfg["imaging_wells_by_layer"],
        "point_count": int(len(imaging_points)),
        "families": orientation_families,
    }))
    family_table_rows: list[dict[str, Any]] = []
    for layer, layer_families in orientation_families.items():
        for family in layer_families:
            family_row = {key: value for key, value in family.items() if key != "well_contributions"}
            family_row["LayerGroup"] = layer
            family_row["source_wells"] = ",".join(item["well"] for item in family["well_contributions"])
            family_row["well_contributions_json"] = json.dumps(family["well_contributions"], ensure_ascii=False)
            family_table_rows.append(family_row)
    pd.DataFrame(family_table_rows).to_csv(families_csv_path, index=False, encoding="utf-8-sig")
    missing_family_layers = [
        layer for layer in ("上部复合层", "太古界风化壳") if not orientation_families[layer]
    ]
    if missing_family_layers:
        print(f"[step7a] warning: no imaging family for {missing_family_layers}; using manual fallback", flush=True)

    fallback = orient_cfg["fallback_family"]
    family_distance_scale = float(orient_cfg["family_distance_scale_m"])
    gradient_quantile = float(orient_cfg["gradient_quantile"])
    gradient_smoothing_sigma = float(orient_cfg["gradient_smoothing_sigma"])
    gradient_time_scale = float(orient_cfg["gradient_time_scale_m_per_ms"])
    local_high_quality = float(orient_cfg["local_high_quality"])
    local_medium_quality = float(orient_cfg["local_medium_quality"])
    medium_weight_min = float(orient_cfg["medium_local_weight_min"])
    medium_weight_max = float(orient_cfg["medium_local_weight_max"])
    dx_candidates = grid.groupby("IX")["X"].median().sort_index().diff().abs().dropna()
    dy_candidates = grid.groupby("IY")["Y"].median().sort_index().diff().abs().dropna()
    dx_m = float(dx_candidates[dx_candidates > 0].median())
    dy_m = float(dy_candidates[dy_candidates > 0].median())
    if not np.isfinite(dx_m) or not np.isfinite(dy_m):
        raise RuntimeError("cannot derive physical attribute-grid spacing for local orientation")

    dips: list[float] = []
    azimuths: list[float] = []
    local_dips: list[float] = []
    local_azimuths: list[float] = []
    local_gradient_strengths: list[float] = []
    local_anisotropies: list[float] = []
    local_ridge_strengths: list[float] = []
    local_support_counts: list[int] = []
    orientation_sources: list[str] = []
    selected_family_ids: list[str] = []
    orientation_confidences: list[float] = []
    family_probabilities: list[float] = []
    orientation_chunk = 5000
    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as handle:
        for chunk_start in range(0, len(accepted_rows), orientation_chunk):
            chunk_end = min(chunk_start + orientation_chunk, len(accepted_rows))
            chunk_rows = accepted_rows[chunk_start:chunk_end].tolist()
            chunk_samples = accepted_samples[chunk_start:chunk_end].tolist()
            chunk_layers = accepted_layer[chunk_start:chunk_end].tolist()
            needed_rows: set[int] = set()
            for row in chunk_rows:
                ix = int(row_to_ix[row])
                iy = int(row_to_iy[row])
                for di in range(-orient_xy_radius, orient_xy_radius + 1):
                    for dj in range(-orient_xy_radius, orient_xy_radius + 1):
                        nrow = cell_to_row.get((ix + di, iy + dj))
                        if nrow is not None:
                            needed_rows.add(nrow)
            dens_by_row: dict[int, np.ndarray] = {
                row: np.asarray(handle.trace[int(grid.loc[row, "OutputTraceIndex"])], dtype=np.float32) for row in needed_rows
            }
            for local_idx, (row, sample, layer) in enumerate(zip(chunk_rows, chunk_samples, chunk_layers)):
                center_density = float(accepted_density[chunk_start + local_idx])
                local = local_gradient_ridge_orientation(
                    row=row, sample=sample, center_density=center_density,
                    cell_to_row=cell_to_row, row_to_ix=row_to_ix, row_to_iy=row_to_iy,
                    dens_by_row=dens_by_row, sample_axis=sample_axis,
                    xy_radius=orient_xy_radius, time_radius=orient_time_span,
                    dx_m=dx_m, dy_m=dy_m, time_scale_m_per_ms=gradient_time_scale,
                    density_fraction=orient_density_fraction, gradient_quantile=gradient_quantile,
                    smoothing_sigma=gradient_smoothing_sigma,
                )
                layer_families = orientation_families[layer]
                if layer_families:
                    selected_family, family_probability = choose_orientation_family(
                        layer_families, float(grid.loc[row, "X"]), float(grid.loc[row, "Y"]),
                        family_distance_scale, orientation_rng,
                    )
                    family_id = str(selected_family["family_id"])
                    family_azimuth = float(selected_family["azimuth_deg"])
                    family_dip = float(selected_family["dip_deg"])
                    family_az_jitter = min(az_jitter, max(2.0, 0.35 * float(selected_family["azimuth_std_deg"])))
                    family_dip_jitter = min(dip_jitter, max(1.0, 0.35 * float(selected_family["dip_std_deg"])))
                else:
                    selected_family = None
                    family_probability = 1.0
                    family_id = f"{layer}_manual_fallback"
                    family_azimuth = float(fallback[layer]["azimuth_deg"])
                    family_dip = float(fallback[layer]["dip_deg"])
                    family_az_jitter, family_dip_jitter = az_jitter, dip_jitter
                local_dip = float(local["dip"])
                local_az = float(local["azimuth"])
                local_valid = np.isfinite(local_dip) and np.isfinite(local_az) and min_dip <= local_dip <= max_dip
                quality = float(local["quality"])
                if local_valid and quality >= local_high_quality:
                    base_azimuth, base_dip = local_az, local_dip
                    source, confidence = "local_gradient_ridge", quality
                    azimuth_noise = family_az_jitter * 0.25 * (1.0 - quality)
                    dip_noise = family_dip_jitter * 0.25 * (1.0 - quality)
                elif local_valid and quality >= local_medium_quality:
                    fraction = (quality - local_medium_quality) / max(local_high_quality - local_medium_quality, 1.0e-12)
                    local_weight = medium_weight_min + fraction * (medium_weight_max - medium_weight_min)
                    base_azimuth = blend_axial_azimuth(family_azimuth, local_az, local_weight)
                    base_dip = (1.0 - local_weight) * family_dip + local_weight * local_dip
                    source, confidence = "multiwell_local_blend", 0.5 * quality + 0.5 * family_probability
                    azimuth_noise = family_az_jitter * (1.0 - local_weight)
                    dip_noise = family_dip_jitter * (1.0 - local_weight)
                else:
                    base_azimuth, base_dip = family_azimuth, family_dip
                    source = "multiwell_family" if selected_family is not None else "manual_fallback"
                    confidence = family_probability
                    azimuth_noise, dip_noise = family_az_jitter, family_dip_jitter
                azimuth = float((base_azimuth + orientation_rng.normal(0.0, azimuth_noise)) % 180.0)
                dip = float(np.clip(base_dip + orientation_rng.normal(0.0, dip_noise), min_dip, max_dip))
                azimuths.append(azimuth)
                dips.append(dip)
                local_dips.append(local_dip)
                local_azimuths.append(local_az)
                local_gradient_strengths.append(float(local["gradient_strength"]))
                local_anisotropies.append(float(local["anisotropy"]))
                local_ridge_strengths.append(float(local["ridge_strength"]))
                local_support_counts.append(int(local["support_count"]))
                orientation_sources.append(source)
                selected_family_ids.append(family_id)
                orientation_confidences.append(float(confidence))
                family_probabilities.append(float(family_probability))
            del dens_by_row
            gc.collect()
            print(f"[step7a] orientation chunk {chunk_start}:{chunk_end} of {len(accepted_rows)}", flush=True)

    lengths = orientation_rng.uniform(length_range[0], length_range[1], size=len(accepted_rows))
    heights = orientation_rng.uniform(height_range[0], height_range[1], size=len(accepted_rows))
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
            "LocalGradientDipDeg": local_dips,
            "LocalGradientAzimuthDeg": local_azimuths,
            # Compatibility aliases for consumers of the previous Step7A
            # schema. They now expose gradient/ridge estimates, not PCA fits.
            "LocalPcaDipDeg": local_dips,
            "LocalPcaAzimuthDeg": local_azimuths,
            "LocalGradientStrength": local_gradient_strengths,
            "LocalAnisotropy": local_anisotropies,
            "LocalRidgeStrength": local_ridge_strengths,
            "LocalSupportCount": local_support_counts,
            "OrientationSource": orientation_sources,
            "OrientationBaseSource": orientation_sources,
            "OrientationFamily": selected_family_ids,
            "OrientationConfidence": orientation_confidences,
            "OrientationFamilyProbability": family_probabilities,
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
        "PatchLengthM": patches["PatchLengthM"].to_numpy(dtype=np.float64),
        "PatchHeightMs": patches["PatchHeightMs"].to_numpy(dtype=np.float64),
        "CenterTimeMs": patches["TIME"].to_numpy(dtype=np.float64),
        "LayerCode": patches["LayerGroup"].map({"上部复合层": 1, "太古界风化壳": 2}).to_numpy(dtype=np.int32),
        "FractureScale": np.full(len(patches), 1, dtype=np.int32),
        "WindowCode": np.full(len(patches), 1, dtype=np.int32),
        "OrientationSourceCode": patches["OrientationSource"].map({
            "local_gradient_ridge": 1,
            "multiwell_local_blend": 2,
            "multiwell_family": 3,
            "manual_fallback": 4,
        }).to_numpy(dtype=np.int32),
        "OrientationFamilyCode": pd.factorize(patches["OrientationFamily"], sort=True)[0].astype(np.int32) + 1,
        "OrientationConfidence": patches["OrientationConfidence"].to_numpy(dtype=np.float64),
        "LocalGradientStrength": patches["LocalGradientStrength"].to_numpy(dtype=np.float64),
        "LocalAnisotropy": patches["LocalAnisotropy"].to_numpy(dtype=np.float64),
        "LocalRidgeStrength": patches["LocalRidgeStrength"].to_numpy(dtype=np.float64),
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
    spacing_points = np.column_stack(
        [
            patches["X"].to_numpy(dtype=np.float64),
            patches["Y"].to_numpy(dtype=np.float64),
            patches["TIME"].to_numpy(dtype=np.float64) * spacing_time_scale,
        ]
    )
    if len(spacing_points) > 1:
        nearest_distance = cKDTree(spacing_points).query(spacing_points, k=2, workers=-1)[0][:, 1]
    else:
        nearest_distance = np.asarray([np.nan])
    cell_counts = (
        patches.assign(CellI=np.floor(patches["X"] / 300.0), CellJ=np.floor(patches["Y"] / 300.0))
        .groupby(["CellI", "CellJ"])
        .size()
        .to_numpy(dtype=np.int64)
    )
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
        "minimum_3d_spacing_respected": bool(
            min_center_distance_m <= 0.0
            or len(patches) <= 1
            or float(np.nanmin(nearest_distance)) >= min_center_distance_m - 1.0e-6
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "output_paths": {
            "patches_csv": str(patches_csv),
            "patches_vtk": str(vtk_path),
            "imaging_match_qc_csv": str(qc_csv),
            "sampling_qc_json": str(sampling_qc_path),
            "orientation_families_json": str(families_json_path),
            "orientation_families_csv": str(families_csv_path),
            "summary_json": str(summary_path),
        },
        "step6a_input": {
            "prediction_summary_json": str(prediction_summary_path),
            "model_contract_version": prediction_summary.get("model_contract_version"),
            "attribute_normalization_contract_version": prediction_summary.get("attribute_normalization_contract_version"),
        },
        "sampling": {
            "mode": sampling_cfg["mode"],
            "reference_quantiles": layer_refs,
            "layer_audit": layer_sampling_audit,
            "layer_patch_counts": layer_counts,
            "upper_layer_fraction": upper_fraction,
            "pre_spacing_patch_count": pre_spacing_count,
            "post_spacing_patch_count": int(len(patches)),
            "min_center_distance_m": min_center_distance_m,
            "spacing_time_scale_m_per_ms": spacing_time_scale,
            "nearest_neighbor_distance_m": {
                "min": float(np.nanmin(nearest_distance)),
                "median": float(np.nanmedian(nearest_distance)),
                "p90": float(np.nanquantile(nearest_distance, 0.90)),
            },
            "patches_per_300m_cell": {
                "cell_count": int(len(cell_counts)),
                "mean": float(np.mean(cell_counts)),
                "median": float(np.median(cell_counts)),
                "p90": float(np.quantile(cell_counts, 0.90)),
                "max": int(np.max(cell_counts)),
            },
        },
        "orientation": {
            "method": "multiwell_families_plus_local_gradient_ridge",
            "imaging_point_count": int(len(imaging_points)),
            "family_definitions": orientation_families,
            "source_counts": patches["OrientationSource"].value_counts().to_dict(),
            "family_counts": patches["OrientationFamily"].value_counts().to_dict(),
            "local_quality_thresholds": {"medium": local_medium_quality, "high": local_high_quality},
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
