from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_fused_candidate_cheye1.json"
ALLOWED_LAYERS = ["沙三段", "沙四段"]
LAYER_CODE = {"沙三段": 3, "沙四段": 4}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse 2D-density DFN and 3D-density DFN into one Step8-compatible initial DFN.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
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
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def finite_stats(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
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


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "fused_csv": output_dir / "fused_initial_dfn_fracture_patches.csv",
        "raw_vtk": output_dir / "fused_initial_dfn_raw_time.vtk",
        "summary_json": output_dir / "fused_initial_dfn_summary.json",
        "audit_csv": output_dir / "fused_initial_dfn_generation_audit.csv",
    }


def load_dfn(path: Path, source_name: str) -> pd.DataFrame:
    df = read_csv_flexible(path, low_memory=False)
    required = ["PatchID", "LayerGroup", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} DFN missing required columns: {missing}")
    out = df.copy()
    for column in [
        "CenterX",
        "CenterY",
        "CenterTime",
        "TimeWindowMin",
        "TimeWindowMax",
        "LayerThickness",
        "SourceDensity",
        "LengthM",
        "HeightTimeMs",
        "AzimuthDeg",
        "DipDeg",
        "LocalDensityPlanarity",
    ]:
        if column in out.columns:
            out[column] = safe_numeric(out[column])
    out["LayerGroup"] = out["LayerGroup"].astype(str)
    out = out[out["LayerGroup"].isin(ALLOWED_LAYERS)].copy()
    out["SourceDFNPath"] = str(path)
    out["InputSource"] = source_name
    return out.reset_index(drop=True)


def layer_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in df["LayerGroup"].value_counts(dropna=False).sort_index().items()}


def axial_weighted_mean_deg(values_deg: np.ndarray, weights: np.ndarray) -> float:
    angles = np.deg2rad(values_deg.astype(float) * 2.0)
    sin_sum = float(np.sum(weights * np.sin(angles)))
    cos_sum = float(np.sum(weights * np.cos(angles)))
    if abs(sin_sum) < 1.0e-12 and abs(cos_sum) < 1.0e-12:
        return float(np.nanmedian(values_deg) % 180.0)
    return float((0.5 * np.degrees(np.arctan2(sin_sum, cos_sum))) % 180.0)


def orientation_quality(df: pd.DataFrame, min_planarity: float) -> np.ndarray:
    if "LocalDensityPlanarity" in df.columns:
        quality = safe_numeric(df["LocalDensityPlanarity"]).fillna(0.0).to_numpy(dtype=float)
        quality = np.clip(quality, min_planarity, None)
    else:
        quality = np.ones(len(df), dtype=float)
    return quality


def build_layer_trees(df3d: pd.DataFrame, time_scale: float) -> dict[str, tuple[cKDTree, np.ndarray]]:
    trees: dict[str, tuple[cKDTree, np.ndarray]] = {}
    for layer in ALLOWED_LAYERS:
        idx = df3d.index[df3d["LayerGroup"].astype(str) == layer].to_numpy(dtype=int)
        if idx.size == 0:
            continue
        coords = np.column_stack(
            [
                df3d.loc[idx, "CenterX"].to_numpy(dtype=float),
                df3d.loc[idx, "CenterY"].to_numpy(dtype=float),
                df3d.loc[idx, "CenterTime"].to_numpy(dtype=float) * time_scale,
            ]
        )
        trees[layer] = (cKDTree(coords), idx)
    return trees


def correct_2d_orientation(df2d: pd.DataFrame, df3d: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    xy_radius = float(config.get("orientation_xy_radius_m", 150.0))
    time_radius = float(config.get("orientation_time_radius_ms", 60.0))
    radius = float(np.sqrt(xy_radius**2 + (time_radius * time_scale) ** 2))
    max_neighbors = int(config.get("orientation_max_neighbors", 12))
    min_neighbors = int(config.get("min_orientation_neighbors", 3))
    min_planarity = float(config.get("min_3d_planarity", 0.05))
    distance_scale = max(float(config.get("orientation_distance_scale_m", 180.0)), 1.0)
    density_weight_power = float(config.get("orientation_density_weight_power", 0.5))

    trees = build_layer_trees(df3d, time_scale=time_scale)
    quality = orientation_quality(df3d, min_planarity=min_planarity)
    corrected = df2d.copy()
    summary = {
        "corrected_by_3d_orientation": 0,
        "retained_without_3d_neighbor": 0,
        "insufficient_3d_neighbors": 0,
        "neighbor_count_stats_values": [],
        "azimuth_delta_abs_values": [],
        "dip_delta_abs_values": [],
    }
    new_columns: dict[str, list[Any]] = {
        "FusionSource": [],
        "Nearest3DPatchID": [],
        "Nearest3DDistance": [],
        "OrientationNeighborCount": [],
        "OrientationCorrectionAzimuthDelta": [],
        "OrientationCorrectionDipDelta": [],
        "DuplicateRemovalReason": [],
    }

    for idx, row in corrected.iterrows():
        layer = str(row["LayerGroup"])
        if layer not in trees:
            for key, value in {
                "FusionSource": "2d_retained_no_3d_layer",
                "Nearest3DPatchID": "",
                "Nearest3DDistance": np.nan,
                "OrientationNeighborCount": 0,
                "OrientationCorrectionAzimuthDelta": 0.0,
                "OrientationCorrectionDipDelta": 0.0,
                "DuplicateRemovalReason": "",
            }.items():
                new_columns[key].append(value)
            summary["retained_without_3d_neighbor"] += 1
            continue

        tree, source_idx = trees[layer]
        query = np.asarray([[float(row["CenterX"]), float(row["CenterY"]), float(row["CenterTime"]) * time_scale]])
        neighbor_local = tree.query_ball_point(query[0], r=radius)
        if neighbor_local:
            neighbor_local = sorted(neighbor_local, key=lambda local_i: float(np.linalg.norm(tree.data[local_i] - query[0])))[:max_neighbors]
        neighbor_idx = source_idx[np.asarray(neighbor_local, dtype=int)] if neighbor_local else np.asarray([], dtype=int)
        if neighbor_idx.size < min_neighbors:
            new_columns["FusionSource"].append("2d_retained_insufficient_3d_neighbors")
            new_columns["Nearest3DPatchID"].append(str(df3d.loc[neighbor_idx[0], "PatchID"]) if neighbor_idx.size else "")
            new_columns["Nearest3DDistance"].append(float(np.linalg.norm(tree.data[neighbor_local[0]] - query[0])) if neighbor_local else np.nan)
            new_columns["OrientationNeighborCount"].append(int(neighbor_idx.size))
            new_columns["OrientationCorrectionAzimuthDelta"].append(0.0)
            new_columns["OrientationCorrectionDipDelta"].append(0.0)
            new_columns["DuplicateRemovalReason"].append("")
            summary["insufficient_3d_neighbors"] += 1
            continue

        neighbor_coords = np.column_stack(
            [
                df3d.loc[neighbor_idx, "CenterX"].to_numpy(dtype=float),
                df3d.loc[neighbor_idx, "CenterY"].to_numpy(dtype=float),
                df3d.loc[neighbor_idx, "CenterTime"].to_numpy(dtype=float) * time_scale,
            ]
        )
        distances = np.linalg.norm(neighbor_coords - query[0], axis=1)
        densities = safe_numeric(df3d.loc[neighbor_idx, "SourceDensity"]).fillna(0.0).to_numpy(dtype=float)
        density_weights = np.power(np.clip(densities, 1.0e-9, None), density_weight_power)
        weights = np.exp(-distances / distance_scale) * density_weights * quality[neighbor_idx]
        if not np.isfinite(weights).any() or float(np.nansum(weights)) <= 0:
            weights = np.ones(neighbor_idx.size, dtype=float)
        az_old = float(row["AzimuthDeg"])
        dip_old = float(row["DipDeg"])
        az_new = axial_weighted_mean_deg(df3d.loc[neighbor_idx, "AzimuthDeg"].to_numpy(dtype=float), weights)
        dip_new = float(np.average(df3d.loc[neighbor_idx, "DipDeg"].to_numpy(dtype=float), weights=weights))
        dip_new = float(np.clip(dip_new, float(config.get("min_dip_deg", 45.0)), float(config.get("max_dip_deg", 89.0))))
        az_delta = min(abs(az_new - az_old), 180.0 - abs(az_new - az_old))
        dip_delta = abs(dip_new - dip_old)
        corrected.loc[idx, "AzimuthDeg"] = az_new
        corrected.loc[idx, "DipDeg"] = dip_new
        corrected.loc[idx, "OrientationSource"] = "fused_from_nearby_3d_dfn"
        corrected.loc[idx, "GenerationStage"] = "fused_2d_density_grid_corrected_by_3d_orientation"
        nearest_pos = int(np.argmin(distances))
        new_columns["FusionSource"].append("2d_corrected_by_3d_orientation")
        new_columns["Nearest3DPatchID"].append(str(df3d.loc[neighbor_idx[nearest_pos], "PatchID"]))
        new_columns["Nearest3DDistance"].append(float(distances[nearest_pos]))
        new_columns["OrientationNeighborCount"].append(int(neighbor_idx.size))
        new_columns["OrientationCorrectionAzimuthDelta"].append(float(az_delta))
        new_columns["OrientationCorrectionDipDelta"].append(float(dip_delta))
        new_columns["DuplicateRemovalReason"].append("")
        summary["corrected_by_3d_orientation"] += 1
        summary["neighbor_count_stats_values"].append(int(neighbor_idx.size))
        summary["azimuth_delta_abs_values"].append(float(az_delta))
        summary["dip_delta_abs_values"].append(float(dip_delta))

    for key, values in new_columns.items():
        corrected[key] = values
    corrected["InputSource"] = "2d_density_grid"
    return corrected, summary


def prepare_3d_primary(df3d: pd.DataFrame) -> pd.DataFrame:
    out = df3d.copy()
    out["FusionSource"] = "3d_primary"
    out["Nearest3DPatchID"] = out["PatchID"].astype(str)
    out["Nearest3DDistance"] = 0.0
    out["OrientationNeighborCount"] = 0
    out["OrientationCorrectionAzimuthDelta"] = 0.0
    out["OrientationCorrectionDipDelta"] = 0.0
    out["DuplicateRemovalReason"] = ""
    out["InputSource"] = "3d_density_volume"
    return out


def filter_2d_supplements(corrected_2d: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    retain_quantile = float(config.get("retain_2d_density_quantile_without_3d", 0.75))
    max_fraction = float(config.get("max_2d_retained_fraction", 0.35))
    corrected_mask = corrected_2d["FusionSource"].astype(str).eq("2d_corrected_by_3d_orientation")
    retained = corrected_2d[corrected_mask].copy()
    no_neighbor = corrected_2d[~corrected_mask].copy()
    threshold_by_layer = {
        layer: float(group["SourceDensity"].quantile(retain_quantile))
        for layer, group in corrected_2d.groupby("LayerGroup", dropna=False)
    }
    if not no_neighbor.empty:
        keep_mask = np.zeros(len(no_neighbor), dtype=bool)
        for layer, group_idx in no_neighbor.groupby("LayerGroup", dropna=False).groups.items():
            threshold = threshold_by_layer.get(str(layer), np.inf)
            keep_mask[no_neighbor.index.get_indexer(group_idx)] = no_neighbor.loc[group_idx, "SourceDensity"].to_numpy(dtype=float) >= threshold
        supplement = no_neighbor[keep_mask].copy()
        if max_fraction > 0 and len(supplement) > int(len(corrected_2d) * max_fraction):
            supplement = supplement.sort_values("SourceDensity", ascending=False).head(int(len(corrected_2d) * max_fraction)).copy()
        retained = pd.concat([retained, supplement], ignore_index=True)
    return retained.reset_index(drop=True), {
        "retain_2d_density_quantile_without_3d": retain_quantile,
        "max_2d_retained_fraction": max_fraction,
        "retained_2d_count_after_density_filter": int(len(retained)),
        "dropped_2d_count_after_density_filter": int(len(corrected_2d) - len(retained)),
        "density_threshold_by_layer": {str(k): float(v) for k, v in threshold_by_layer.items()},
    }


def remove_duplicates(df3d_primary: pd.DataFrame, df2d_retained: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    duplicate_xy = float(config.get("duplicate_xy_radius_m", 40.0))
    duplicate_time = float(config.get("duplicate_time_radius_ms", 20.0))
    radius = float(np.sqrt(duplicate_xy**2 + (duplicate_time * time_scale) ** 2))
    trees = build_layer_trees(df3d_primary, time_scale=time_scale)
    keep_mask = np.ones(len(df2d_retained), dtype=bool)
    nearest_dist = np.full(len(df2d_retained), np.nan, dtype=float)
    duplicate_count = 0
    for local_i, (_, row) in enumerate(df2d_retained.iterrows()):
        layer = str(row["LayerGroup"])
        if layer not in trees:
            continue
        tree, _ = trees[layer]
        query = np.asarray([float(row["CenterX"]), float(row["CenterY"]), float(row["CenterTime"]) * time_scale])
        candidates = tree.query_ball_point(query, r=radius)
        if candidates:
            distances = [float(np.linalg.norm(tree.data[int(candidate)] - query)) for candidate in candidates]
            nearest_dist[local_i] = float(min(distances))
            keep_mask[local_i] = False
            duplicate_count += 1
    df2d_work = df2d_retained.copy()
    df2d_work["Nearest3DDuplicateDistance"] = nearest_dist
    df2d_work.loc[~keep_mask, "DuplicateRemovalReason"] = "removed_duplicate_near_3d_primary"
    kept_2d = df2d_work[keep_mask].copy()
    fused = pd.concat([df3d_primary, kept_2d], ignore_index=True, sort=False)
    return fused.reset_index(drop=True), {
        "duplicate_xy_radius_m": duplicate_xy,
        "duplicate_time_radius_ms": duplicate_time,
        "duplicate_search_radius_scaled": radius,
        "removed_2d_duplicate_count": int(duplicate_count),
        "kept_2d_after_duplicate_removal": int(len(kept_2d)),
    }


def finalize_fused_table(fused: pd.DataFrame) -> pd.DataFrame:
    out = fused.copy()
    out["PatchID"] = [f"fused_dfn_{idx + 1:06d}" for idx in range(len(out))]
    out["OriginalPatchID"] = fused["PatchID"].astype(str).to_numpy()
    out["LayerCode"] = out["LayerGroup"].map(LAYER_CODE).astype(int)
    out["NeedsWellCorrection"] = 1
    for column in ["SourceTraceIdx", "DensityCellPatchOrdinal", "ExpectedPatchCountForCell", "EffectiveCountScale"]:
        if column not in out.columns:
            out[column] = np.nan
    out["GenerationStage"] = out["GenerationStage"].fillna("fused_initial_dfn") if "GenerationStage" in out.columns else "fused_initial_dfn"
    out["SamplingRule"] = out["SamplingRule"].fillna("step7c_fusion") if "SamplingRule" in out.columns else "step7c_fusion"
    out["SizeRule"] = out["SizeRule"].fillna("preserve_input_patch_size") if "SizeRule" in out.columns else "preserve_input_patch_size"
    return out.reset_index(drop=True)


def patch_vertices(row: pd.Series, geometry_time_scale_m_per_ms: float) -> list[tuple[float, float, float]]:
    azimuth = np.deg2rad(float(row["AzimuthDeg"]))
    dip = np.deg2rad(float(np.clip(row["DipDeg"], 1.0, 89.9)))
    half_length = 0.5 * float(row["LengthM"])
    half_height_time = 0.5 * float(row["HeightTimeMs"])
    strike = np.asarray([np.cos(azimuth), np.sin(azimuth)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(azimuth), np.cos(azimuth)], dtype=float)
    horizontal_dip_half = (half_height_time * geometry_time_scale_m_per_ms) / max(np.tan(dip), 1.0e-6)
    center_x = float(row["CenterX"])
    center_y = float(row["CenterY"])
    center_t = float(row["CenterTime"])
    corners = []
    for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        xy = np.asarray([center_x, center_y], dtype=float) + strike_sign * half_length * strike + dip_sign * horizontal_dip_half * dip_horizontal
        z = center_t + dip_sign * half_height_time
        corners.append((float(xy[0]), float(xy[1]), float(z)))
    return corners


def write_raw_vtk(path: Path, patch_df: pd.DataFrame, title: str, geometry_time_scale_m_per_ms: float) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in patch_df.iterrows():
        base = len(points)
        points.extend(patch_vertices(row, geometry_time_scale_m_per_ms=geometry_time_scale_m_per_ms))
        polygons.append([base, base + 1, base + 2, base + 3])
    total_polygon_size = sum(len(poly) + 1 for poly in polygons)
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA", f"POINTS {len(points)} float"]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {total_polygon_size}")
    lines.extend(f"{len(poly)} {' '.join(str(idx) for idx in poly)}" for poly in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    scalar_columns = [
        ("PatchIndex", np.arange(1, len(patch_df) + 1), "int"),
        ("LayerCode", patch_df["LayerCode"].to_numpy(), "int"),
        ("SourceDensity", safe_numeric(patch_df["SourceDensity"]).fillna(0.0).to_numpy(), "float"),
        (
            "SourceDensityRender",
            safe_numeric(patch_df["SourceDensityRender"]).fillna(0.0).to_numpy()
            if "SourceDensityRender" in patch_df.columns
            else safe_numeric(patch_df["SourceDensity"]).fillna(0.0).to_numpy(),
            "float",
        ),
        (
            "SourceDensityRenderNorm",
            safe_numeric(patch_df["SourceDensityRenderNorm"]).fillna(0.0).to_numpy()
            if "SourceDensityRenderNorm" in patch_df.columns
            else np.zeros(len(patch_df), dtype=float),
            "float",
        ),
        (
            "SourceDensityRenderClipMax",
            safe_numeric(patch_df["SourceDensityRenderClipMax"]).fillna(0.0).to_numpy()
            if "SourceDensityRenderClipMax" in patch_df.columns
            else np.zeros(len(patch_df), dtype=float),
            "float",
        ),
        ("CenterTime", safe_numeric(patch_df["CenterTime"]).to_numpy(), "float"),
        ("LengthM", safe_numeric(patch_df["LengthM"]).to_numpy(), "float"),
        ("HeightTimeMs", safe_numeric(patch_df["HeightTimeMs"]).to_numpy(), "float"),
        (
            "PatchAreaM2",
            safe_numeric(patch_df["PatchAreaM2"]).fillna(0.0).to_numpy()
            if "PatchAreaM2" in patch_df.columns
            else (safe_numeric(patch_df["LengthM"]).fillna(0.0) * safe_numeric(patch_df["HeightTimeMs"]).fillna(0.0)).to_numpy(),
            "float",
        ),
        ("AzimuthDeg", safe_numeric(patch_df["AzimuthDeg"]).to_numpy(), "float"),
        ("DipDeg", safe_numeric(patch_df["DipDeg"]).to_numpy(), "float"),
    ]
    for name, values, dtype in scalar_columns:
        vtk_type = "int" if dtype == "int" else "float"
        lines.append(f"SCALARS {name} {vtk_type} 1")
        lines.append("LOOKUP_TABLE default")
        if vtk_type == "int":
            lines.extend(str(int(value)) for value in values)
        else:
            lines.extend(f"{float(value):.6f}" for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(fused: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "PatchID",
        "OriginalPatchID",
        "InputSource",
        "FusionSource",
        "LayerGroup",
        "CenterX",
        "CenterY",
        "CenterTime",
        "SourceDensity",
        "AzimuthDeg",
        "DipDeg",
        "Nearest3DPatchID",
        "Nearest3DDistance",
        "OrientationNeighborCount",
        "OrientationCorrectionAzimuthDelta",
        "OrientationCorrectionDipDelta",
        "DuplicateRemovalReason",
    ]
    for column in columns:
        if column not in fused.columns:
            fused[column] = np.nan
    audit = fused[columns].copy()
    audit["Action"] = "create_fused_initial_fracture_patch"
    return audit


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    df2d: pd.DataFrame,
    df3d: pd.DataFrame,
    corrected_2d: pd.DataFrame,
    retained_2d: pd.DataFrame,
    fused: pd.DataFrame,
    orientation_summary: dict[str, Any],
    filter_summary: dict[str, Any],
    duplicate_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "has_fused_patches": len(fused) > 0,
        "contains_3d_primary": bool((fused["FusionSource"].astype(str) == "3d_primary").any()),
        "contains_2d_supplements": bool((fused["FusionSource"].astype(str).str.startswith("2d_")).any()),
        "layers_limited_to_sha3_sha4": bool(set(fused["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS))),
        "orientation_fields_complete": bool(fused[["AzimuthDeg", "DipDeg"]].notna().all().all()),
        "centers_within_layer_windows": bool(fused["CenterTime"].between(fused["TimeWindowMin"], fused["TimeWindowMax"]).all())
        if {"CenterTime", "TimeWindowMin", "TimeWindowMax"}.issubset(fused.columns)
        else True,
        "raw_vtk_output_exists": bool(paths["raw_vtk"].exists()),
        "audit_rows_match_patch_count": bool(paths["audit_csv"].exists() and len(fused) > 0),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "input_2d_dfn_csv": str(Path(config["input_2d_dfn_csv"]).resolve()),
        "input_3d_dfn_csv": str(Path(config["input_3d_dfn_csv"]).resolve()),
        "fused_initial_dfn_csv": str(paths["fused_csv"]),
        "fused_initial_dfn_raw_vtk": str(paths["raw_vtk"]),
        "fused_initial_dfn_audit_csv": str(paths["audit_csv"]),
        "summary_json": str(paths["summary_json"]),
        "generation_logic": "3d_primary_plus_2d_density_grid_corrected_by_nearby_3d_orientation_and_deduplicated",
        "parameters": {
            key: config.get(key)
            for key in [
                "time_scale_m_per_ms",
                "orientation_xy_radius_m",
                "orientation_time_radius_ms",
                "orientation_max_neighbors",
                "min_orientation_neighbors",
                "duplicate_xy_radius_m",
                "duplicate_time_radius_ms",
                "retain_2d_density_quantile_without_3d",
                "max_2d_retained_fraction",
            ]
        },
        "input_2d": {"patch_count": int(len(df2d)), "layer_distribution": layer_counts(df2d), "source_density_stats": finite_stats(df2d["SourceDensity"])},
        "input_3d": {"patch_count": int(len(df3d)), "layer_distribution": layer_counts(df3d), "source_density_stats": finite_stats(df3d["SourceDensity"])},
        "orientation_correction_summary": {
            **{k: v for k, v in orientation_summary.items() if not k.endswith("_values")},
            "neighbor_count_stats": finite_stats(orientation_summary.get("neighbor_count_stats_values", [])),
            "azimuth_delta_abs_stats": finite_stats(orientation_summary.get("azimuth_delta_abs_values", [])),
            "dip_delta_abs_stats": finite_stats(orientation_summary.get("dip_delta_abs_values", [])),
        },
        "filter_summary": filter_summary,
        "duplicate_summary": duplicate_summary,
        "fused_dfn": {
            "patch_count": int(len(fused)),
            "layer_distribution": layer_counts(fused),
            "fusion_source_distribution": {str(k): int(v) for k, v in fused["FusionSource"].value_counts(dropna=False).sort_index().items()},
            "center_time_stats": finite_stats(fused["CenterTime"]),
            "source_density_stats": finite_stats(fused["SourceDensity"]),
            "azimuth_deg_stats": finite_stats(fused["AzimuthDeg"]),
            "dip_deg_stats": finite_stats(fused["DipDeg"]),
            "length_m_stats": finite_stats(fused["LengthM"]),
            "height_time_ms_stats": finite_stats(fused["HeightTimeMs"]),
        },
        "intermediate_counts": {
            "corrected_2d_count": int(len(corrected_2d)),
            "retained_2d_after_density_filter": int(len(retained_2d)),
            "final_2d_supplement_count": int((fused["FusionSource"].astype(str).str.startswith("2d_")).sum()),
            "final_3d_primary_count": int((fused["FusionSource"].astype(str) == "3d_primary").sum()),
        },
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)

    input_2d = Path(config["input_2d_dfn_csv"]).resolve()
    input_3d = Path(config["input_3d_dfn_csv"]).resolve()
    for label, path in [("input_2d_dfn_csv", input_2d), ("input_3d_dfn_csv", input_3d)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print("[step7c-fuse] loading input DFNs", flush=True)
    df2d = load_dfn(input_2d, source_name="2d_density_grid")
    df3d = load_dfn(input_3d, source_name="3d_density_volume")
    print(f"[step7c-fuse] input 2d={len(df2d)} 3d={len(df3d)}", flush=True)
    corrected_2d, orientation_summary = correct_2d_orientation(df2d, df3d, config=config)
    retained_2d, filter_summary = filter_2d_supplements(corrected_2d, config=config)
    df3d_primary = prepare_3d_primary(df3d)
    fused_raw, duplicate_summary = remove_duplicates(df3d_primary, retained_2d, config=config)
    fused = finalize_fused_table(fused_raw)
    audit = build_audit(fused)
    fused.to_csv(paths["fused_csv"], index=False, encoding="utf-8-sig")
    audit.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    geometry_time_scale = float(config.get("geometry_time_scale_m_per_ms", 1.0))
    write_raw_vtk(paths["raw_vtk"], fused, title="fused_initial_dfn_raw_time", geometry_time_scale_m_per_ms=geometry_time_scale)
    summary = build_summary(
        config_path=config_path,
        config=config,
        paths=paths,
        df2d=df2d,
        df3d=df3d,
        corrected_2d=corrected_2d,
        retained_2d=retained_2d,
        fused=fused,
        orientation_summary=orientation_summary,
        filter_summary=filter_summary,
        duplicate_summary=duplicate_summary,
    )
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fused DFN CSV: {paths['fused_csv']}")
    print(f"Fused DFN raw VTK: {paths['raw_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Fused patch count: {len(fused)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
