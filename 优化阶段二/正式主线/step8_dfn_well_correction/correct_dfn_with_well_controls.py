from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_well_control_correction.json"
ALLOWED_LAYERS = ["沙三段", "沙四段"]
LAYER_CODE = {"沙三段": 3, "沙四段": 4}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct Step 8 initial DFN with hard real-well fracture controls.")
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
        "corrected_csv": output_dir / "well_corrected_dfn_fracture_patches.csv",
        "raw_vtk": output_dir / "well_corrected_dfn_raw_time.vtk",
        "display_vtk": output_dir / "well_corrected_dfn_display.vtk",
        "summary_json": output_dir / "well_corrected_dfn_summary.json",
        "audit_csv": output_dir / "well_control_correction_audit.csv",
    }


def load_initial_dfn(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path, low_memory=False)
    required = ["PatchID", "LayerGroup", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"initial DFN missing required columns: {missing}")
    out = df.copy()
    for column in ["CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg", "SourceDensity"]:
        if column in out.columns:
            out[column] = safe_numeric(out[column])
    out["LayerGroup"] = out["LayerGroup"].astype(str)
    out = out[out["LayerGroup"].isin(ALLOWED_LAYERS)].copy()
    out["OriginalPatchID"] = out["PatchID"].astype(str)
    out["OriginalCenterX"] = out["CenterX"]
    out["OriginalCenterY"] = out["CenterY"]
    out["OriginalCenterTime"] = out["CenterTime"]
    out["CorrectionAction"] = "unchanged_density_volume"
    out["CorrectionReason"] = "far_from_or_not_selected_by_well_control"
    out["WellControlSampleID"] = pd.NA
    out["WellControlWellName"] = pd.NA
    out["WellControlDensity"] = np.nan
    out["WellControlMatchBefore"] = np.nan
    out["WellControlMatchAfter"] = np.nan
    out["NearestTrajectoryDistance"] = np.nan
    out["IsWellControlPatch"] = 0
    return out.reset_index(drop=True)


def load_control_points(path: Path, target_block: dict[str, Any]) -> pd.DataFrame:
    df = read_csv_flexible(path, low_memory=False)
    required = ["WellName", "X", "Y", "TIME"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"fracture points missing required columns: {missing}")
    layer_col = "StrataName" if "StrataName" in df.columns else "LayerGroup" if "LayerGroup" in df.columns else None
    if layer_col is None:
        raise ValueError("fracture points missing StrataName/LayerGroup")
    id_col = "SourceSampleID" if "SourceSampleID" in df.columns else "SampleID" if "SampleID" in df.columns else None
    out = df.copy()
    for column in ["X", "Y", "TIME", "TVD", "DEPT", "Density"]:
        if column in out.columns:
            out[column] = safe_numeric(out[column])
    out["LayerGroup"] = out[layer_col].astype(str)
    if id_col is None:
        out["WellControlSampleID"] = ["well_control_%06d" % (idx + 1) for idx in range(len(out))]
    else:
        out["WellControlSampleID"] = out[id_col].astype(str)
    x_min = float(target_block["x_min"])
    x_max = float(target_block["x_max"])
    y_min = float(target_block["y_min"])
    y_max = float(target_block["y_max"])
    out = out[
        out["LayerGroup"].isin(ALLOWED_LAYERS)
        & out["X"].between(x_min, x_max)
        & out["Y"].between(y_min, y_max)
        & out["TIME"].notna()
    ].copy()
    out = out.sort_values(["WellName", "LayerGroup", "TIME", "WellControlSampleID"]).reset_index(drop=True)
    out["ControlPointID"] = ["ctrl_%06d" % (idx + 1) for idx in range(len(out))]
    return out


def load_real_well_tracks(root: Path, well_names: set[str]) -> dict[str, pd.DataFrame]:
    tracks: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob("*/*_t4_t7_real_well_main.csv")):
        well_name = path.parent.name
        if well_name not in well_names:
            continue
        df = read_csv_flexible(path, low_memory=False)
        required = ["SampleID", "WellName", "X", "Y", "TIME"]
        if any(column not in df.columns for column in required):
            continue
        for column in ["X", "Y", "TIME", "TVD", "DEPT"]:
            if column in df.columns:
                df[column] = safe_numeric(df[column])
        tracks[well_name] = df.dropna(subset=["X", "Y", "TIME"]).reset_index(drop=True)
    return tracks


def nearest_track_distances(control_df: pd.DataFrame, tracks: dict[str, pd.DataFrame], time_scale: float) -> pd.Series:
    values: list[float] = []
    tree_cache: dict[str, tuple[cKDTree, pd.DataFrame]] = {}
    for _, row in control_df.iterrows():
        well = str(row["WellName"])
        if well not in tracks or tracks[well].empty:
            values.append(np.nan)
            continue
        if well not in tree_cache:
            track = tracks[well]
            coords = np.column_stack(
                [
                    track["X"].to_numpy(dtype=float),
                    track["Y"].to_numpy(dtype=float),
                    track["TIME"].to_numpy(dtype=float) * time_scale,
                ]
            )
            tree_cache[well] = (cKDTree(coords), track)
        tree, _ = tree_cache[well]
        query = np.asarray([[float(row["X"]), float(row["Y"]), float(row["TIME"]) * time_scale]])
        dist, _ = tree.query(query, k=1)
        values.append(float(dist[0]))
    return pd.Series(values, index=control_df.index)


def build_patch_tree(df: pd.DataFrame, layer: str, time_scale: float) -> tuple[cKDTree | None, np.ndarray]:
    layer_idx = df.index[df["LayerGroup"].astype(str) == layer].to_numpy(dtype=int)
    if layer_idx.size == 0:
        return None, layer_idx
    coords = np.column_stack(
        [
            df.loc[layer_idx, "CenterX"].to_numpy(dtype=float),
            df.loc[layer_idx, "CenterY"].to_numpy(dtype=float),
            df.loc[layer_idx, "CenterTime"].to_numpy(dtype=float) * time_scale,
        ]
    )
    return cKDTree(coords), layer_idx


def nearest_patch_distances(patch_df: pd.DataFrame, control_df: pd.DataFrame, time_scale: float) -> pd.Series:
    out = pd.Series(np.nan, index=control_df.index, dtype=float)
    for layer in ALLOWED_LAYERS:
        tree, layer_idx = build_patch_tree(patch_df, layer=layer, time_scale=time_scale)
        if tree is None:
            continue
        mask = control_df["LayerGroup"].astype(str) == layer
        if not mask.any():
            continue
        query_df = control_df.loc[mask]
        coords = np.column_stack(
            [
                query_df["X"].to_numpy(dtype=float),
                query_df["Y"].to_numpy(dtype=float),
                query_df["TIME"].to_numpy(dtype=float) * time_scale,
            ]
        )
        dist, _ = tree.query(coords, k=1)
        out.loc[query_df.index] = dist.astype(float)
    return out


def choose_patch_for_control(
    initial_df: pd.DataFrame,
    trees: dict[str, tuple[cKDTree | None, np.ndarray]],
    control: pd.Series,
    used_patch_indices: set[int],
    config: dict[str, Any],
) -> tuple[int | None, float | None, float | None]:
    layer = str(control["LayerGroup"])
    tree, layer_idx = trees[layer]
    if tree is None or layer_idx.size == 0:
        return None, None, None
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    k = min(int(config.get("max_query_candidates", 30)), int(layer_idx.size))
    query = np.asarray([[float(control["X"]), float(control["Y"]), float(control["TIME"]) * time_scale]])
    distances, positions = tree.query(query, k=k)
    distances = np.atleast_1d(distances[0] if np.asarray(distances).ndim > 1 else distances)
    positions = np.atleast_1d(positions[0] if np.asarray(positions).ndim > 1 else positions)
    xy_radius = float(config.get("xy_search_radius_m", 80.0))
    time_radius = float(config.get("time_search_radius_ms", 30.0))
    for pos in positions:
        patch_idx = int(layer_idx[int(pos)])
        if patch_idx in used_patch_indices:
            continue
        patch = initial_df.loc[patch_idx]
        xy_dist = float(np.hypot(float(patch["CenterX"]) - float(control["X"]), float(patch["CenterY"]) - float(control["Y"])))
        time_dist = abs(float(patch["CenterTime"]) - float(control["TIME"]))
        if xy_dist <= xy_radius and time_dist <= time_radius:
            return patch_idx, xy_dist, time_dist
    return None, None, None


def nearest_template_patch(initial_df: pd.DataFrame, layer: str, x: float, y: float, time_value: float, time_scale: float) -> pd.Series:
    tree, layer_idx = build_patch_tree(initial_df, layer=layer, time_scale=time_scale)
    if tree is None or layer_idx.size == 0:
        raise RuntimeError(f"no template patches available for layer: {layer}")
    query = np.asarray([[x, y, time_value * time_scale]])
    _, pos = tree.query(query, k=1)
    return initial_df.loc[int(layer_idx[int(pos[0])])]


def correction_columns() -> list[str]:
    return [
        "OriginalPatchID",
        "OriginalCenterX",
        "OriginalCenterY",
        "OriginalCenterTime",
        "CorrectionAction",
        "CorrectionReason",
        "WellControlSampleID",
        "WellControlWellName",
        "WellControlDensity",
        "WellControlMatchBefore",
        "WellControlMatchAfter",
        "NearestTrajectoryDistance",
        "IsWellControlPatch",
    ]


def build_added_patch(template: pd.Series, control: pd.Series, add_index: int, before_distance: float, track_distance: float) -> dict[str, Any]:
    row = template.to_dict()
    row["PatchID"] = f"well_ctrl_dfn_{add_index:06d}"
    row["OriginalPatchID"] = ""
    row["OriginalCenterX"] = np.nan
    row["OriginalCenterY"] = np.nan
    row["OriginalCenterTime"] = np.nan
    row["CenterX"] = float(control["X"])
    row["CenterY"] = float(control["Y"])
    row["CenterTime"] = float(control["TIME"])
    row["LayerGroup"] = str(control["LayerGroup"])
    row["LayerCode"] = int(LAYER_CODE[str(control["LayerGroup"])])
    row["SourceDensity"] = float(control["Density"]) if "Density" in control and pd.notna(control["Density"]) else row.get("SourceDensity", np.nan)
    row["DensityCellID"] = f"well_control_{control['WellControlSampleID']}"
    row["CorrectionAction"] = "add_well_control_patch"
    row["CorrectionReason"] = "no_unused_initial_patch_within_search_window"
    row["WellControlSampleID"] = str(control["WellControlSampleID"])
    row["WellControlWellName"] = str(control["WellName"])
    row["WellControlDensity"] = float(control["Density"]) if "Density" in control and pd.notna(control["Density"]) else np.nan
    row["WellControlMatchBefore"] = float(before_distance) if pd.notna(before_distance) else np.nan
    row["WellControlMatchAfter"] = 0.0
    row["NearestTrajectoryDistance"] = float(track_distance) if pd.notna(track_distance) else np.nan
    row["IsWellControlPatch"] = 1
    row["NeedsWellCorrection"] = 0
    row["OrientationSource"] = "well_control_template_from_nearest_initial_patch"
    row["SamplingRule"] = "hard_well_control_addition"
    return row


def apply_well_controls(
    initial_df: pd.DataFrame,
    control_df: pd.DataFrame,
    track_distances: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    before_dist = nearest_patch_distances(initial_df, control_df, time_scale=time_scale)
    corrected = initial_df.copy()
    trees = {layer: build_patch_tree(initial_df, layer=layer, time_scale=time_scale) for layer in ALLOWED_LAYERS}
    used_patch_indices: set[int] = set()
    added_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for ctrl_idx, control in control_df.iterrows():
        before = float(before_dist.loc[ctrl_idx]) if pd.notna(before_dist.loc[ctrl_idx]) else np.nan
        track_distance = float(track_distances.loc[ctrl_idx]) if pd.notna(track_distances.loc[ctrl_idx]) else np.nan
        patch_idx, xy_dist, time_dist = choose_patch_for_control(
            initial_df=initial_df,
            trees=trees,
            control=control,
            used_patch_indices=used_patch_indices,
            config=config,
        )
        action: str
        reason: str
        if patch_idx is not None:
            used_patch_indices.add(patch_idx)
            original = corrected.loc[patch_idx].copy()
            corrected.loc[patch_idx, "CenterX"] = float(control["X"])
            corrected.loc[patch_idx, "CenterY"] = float(control["Y"])
            corrected.loc[patch_idx, "CenterTime"] = float(control["TIME"])
            corrected.loc[patch_idx, "CorrectionAction"] = "adjust_to_well_control"
            corrected.loc[patch_idx, "CorrectionReason"] = "unused_initial_patch_within_search_window"
            corrected.loc[patch_idx, "WellControlSampleID"] = str(control["WellControlSampleID"])
            corrected.loc[patch_idx, "WellControlWellName"] = str(control["WellName"])
            corrected.loc[patch_idx, "WellControlDensity"] = float(control["Density"]) if "Density" in control and pd.notna(control["Density"]) else np.nan
            corrected.loc[patch_idx, "WellControlMatchBefore"] = before
            corrected.loc[patch_idx, "WellControlMatchAfter"] = 0.0
            corrected.loc[patch_idx, "NearestTrajectoryDistance"] = track_distance
            corrected.loc[patch_idx, "IsWellControlPatch"] = 1
            corrected.loc[patch_idx, "NeedsWellCorrection"] = 0
            patch_id = str(corrected.loc[patch_idx, "PatchID"])
            action = "adjust_to_well_control"
            reason = "unused_initial_patch_within_search_window"
            original_patch_id = str(original["PatchID"])
            original_x = float(original["CenterX"])
            original_y = float(original["CenterY"])
            original_time = float(original["CenterTime"])
        else:
            template = nearest_template_patch(
                initial_df=initial_df,
                layer=str(control["LayerGroup"]),
                x=float(control["X"]),
                y=float(control["Y"]),
                time_value=float(control["TIME"]),
                time_scale=time_scale,
            )
            added = build_added_patch(
                template=template,
                control=control,
                add_index=len(added_rows) + 1,
                before_distance=before,
                track_distance=track_distance,
            )
            added_rows.append(added)
            patch_id = str(added["PatchID"])
            action = "add_well_control_patch"
            reason = "no_unused_initial_patch_within_search_window"
            original_patch_id = str(template["PatchID"])
            original_x = float(template["CenterX"])
            original_y = float(template["CenterY"])
            original_time = float(template["CenterTime"])

        audit_rows.append(
            {
                "ControlPointID": str(control["ControlPointID"]),
                "WellControlSampleID": str(control["WellControlSampleID"]),
                "WellName": str(control["WellName"]),
                "LayerGroup": str(control["LayerGroup"]),
                "ControlX": float(control["X"]),
                "ControlY": float(control["Y"]),
                "ControlTime": float(control["TIME"]),
                "ControlDensity": float(control["Density"]) if "Density" in control and pd.notna(control["Density"]) else np.nan,
                "Action": action,
                "ActionReason": reason,
                "PatchID": patch_id,
                "OriginalPatchID": original_patch_id,
                "OriginalCenterX": original_x,
                "OriginalCenterY": original_y,
                "OriginalCenterTime": original_time,
                "CorrectedCenterX": float(control["X"]),
                "CorrectedCenterY": float(control["Y"]),
                "CorrectedCenterTime": float(control["TIME"]),
                "BeforeMatchDistance": before,
                "AfterMatchDistance": 0.0,
                "NearestTrajectoryDistance": track_distance,
                "XYSearchDistance": xy_dist,
                "TimeSearchDistance": time_dist,
            }
        )

    if added_rows:
        corrected = pd.concat([corrected, pd.DataFrame(added_rows)], ignore_index=True)
    for column in correction_columns():
        if column not in corrected.columns:
            corrected[column] = np.nan
    corrected["LayerCode"] = corrected["LayerGroup"].map(LAYER_CODE).astype(int)
    return corrected.reset_index(drop=True), pd.DataFrame(audit_rows)


def write_legacy_vtk(path: Path, patch_df: pd.DataFrame, title: str, display: bool, display_z_scale: float) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in patch_df.iterrows():
        theta = np.deg2rad(float(row["AzimuthDeg"]))
        half_dx = 0.5 * float(row["LengthM"]) * np.cos(theta)
        half_dy = 0.5 * float(row["LengthM"]) * np.sin(theta)
        half_h = 0.5 * float(row["HeightTimeMs"])
        z0 = float(row["CenterTime"]) - half_h
        z1 = float(row["CenterTime"]) + half_h
        if display:
            z0 = -z0 / display_z_scale
            z1 = -z1 / display_z_scale
        base = len(points)
        points.extend(
            [
                (float(row["CenterX"]) - half_dx, float(row["CenterY"]) - half_dy, z0),
                (float(row["CenterX"]) + half_dx, float(row["CenterY"]) + half_dy, z0),
                (float(row["CenterX"]) + half_dx, float(row["CenterY"]) + half_dy, z1),
                (float(row["CenterX"]) - half_dx, float(row["CenterY"]) - half_dy, z1),
            ]
        )
        polygons.append([base, base + 1, base + 2, base + 3])

    total_polygon_size = sum(len(poly) + 1 for poly in polygons)
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {total_polygon_size}")
    lines.extend(f"{len(poly)} {' '.join(str(idx) for idx in poly)}" for poly in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    scalar_columns = [
        ("PatchIndex", np.arange(1, len(patch_df) + 1), "int"),
        ("LayerCode", patch_df["LayerCode"].to_numpy(), "int"),
        ("SourceDensity", safe_numeric(patch_df["SourceDensity"]).fillna(0.0).to_numpy(), "float"),
        ("IsWellControlPatch", safe_numeric(patch_df["IsWellControlPatch"]).fillna(0).to_numpy(), "int"),
        ("CenterTime", safe_numeric(patch_df["CenterTime"]).to_numpy(), "float"),
        ("LengthM", safe_numeric(patch_df["LengthM"]).to_numpy(), "float"),
        ("HeightTimeMs", safe_numeric(patch_df["HeightTimeMs"]).to_numpy(), "float"),
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


def layer_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in df["LayerGroup"].value_counts(dropna=False).sort_index().items()}


def load_density_mass_from_initial_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return {
        str(key): float(value)
        for key, value in payload.get("density_volume", {}).get("density_mass_by_layer", {}).items()
    }


def macro_distribution(density_mass: dict[str, float], corrected_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    total_density = float(sum(density_mass.values()))
    total_patches = int(len(corrected_df))
    patch_counts = corrected_df["LayerGroup"].value_counts().to_dict()
    result: dict[str, dict[str, float]] = {}
    for layer in ALLOWED_LAYERS:
        density_share = float(density_mass.get(layer, 0.0)) / total_density if total_density > 0 else 0.0
        patch_share = float(patch_counts.get(layer, 0)) / total_patches if total_patches > 0 else 0.0
        result[layer] = {
            "density_mass": float(density_mass.get(layer, 0.0)),
            "density_mass_share": density_share,
            "patch_count": int(patch_counts.get(layer, 0)),
            "patch_count_share": patch_share,
            "absolute_share_difference": abs(density_share - patch_share),
        }
    return result


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    initial_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    control_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    before_dist: pd.Series,
    after_dist: pd.Series,
    track_dist: pd.Series,
    density_mass: dict[str, float],
) -> dict[str, Any]:
    macro = macro_distribution(density_mass=density_mass, corrected_df=corrected_df)
    action_counts = {
        str(key): int(value)
        for key, value in audit_df["Action"].value_counts(dropna=False).to_dict().items()
    } if not audit_df.empty else {}
    changed_initial_count = int((corrected_df["CorrectionAction"].astype(str) == "adjust_to_well_control").sum())
    added_count = int((corrected_df["CorrectionAction"].astype(str) == "add_well_control_patch").sum())
    before_mean = float(before_dist.mean()) if before_dist.notna().any() else None
    after_mean = float(after_dist.mean()) if after_dist.notna().any() else None
    before_median = float(before_dist.median()) if before_dist.notna().any() else None
    after_median = float(after_dist.median()) if after_dist.notna().any() else None
    changed_fraction = float(changed_initial_count + added_count) / max(float(len(initial_df)), 1.0)
    checks = {
        "has_well_controls_in_target_block": int(len(control_df)) > 0,
        "match_mean_improved": bool(after_mean is not None and before_mean is not None and after_mean < before_mean),
        "match_median_improved": bool(after_median is not None and before_median is not None and after_median <= before_median),
        "hard_control_points_satisfied": bool(after_dist.fillna(np.inf).max() <= 1.0e-6),
        "macro_distribution_preserved": bool(all(item["absolute_share_difference"] <= 0.05 for item in macro.values())),
        "far_field_preserved": bool(changed_fraction <= 0.15),
        "audit_rows_match_control_points": bool(len(audit_df) == len(control_df)),
        "layers_limited_to_sha3_sha4": bool(set(corrected_df["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS))),
        "vtk_outputs_exist": bool(paths["raw_vtk"].exists() and paths["display_vtk"].exists()),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "initial_dfn_csv": str(Path(config["initial_dfn_csv"]).resolve()),
        "fracture_points_csv": str(Path(config["fracture_points_csv"]).resolve()),
        "real_well_samples_root": str(Path(config["real_well_samples_root"]).resolve()),
        "corrected_dfn_csv": str(paths["corrected_csv"]),
        "corrected_dfn_raw_vtk": str(paths["raw_vtk"]),
        "corrected_dfn_display_vtk": str(paths["display_vtk"]),
        "well_control_correction_audit_csv": str(paths["audit_csv"]),
        "summary_json": str(paths["summary_json"]),
        "target_block": config["target_block"],
        "correction_logic": {
            "well_points_inside_target_are_hard_controls": True,
            "adjust_existing_patch_when_local_candidate_available": True,
            "add_patch_when_no_local_candidate_available": True,
            "far_field_density_volume_patches_unchanged": True,
            "xy_search_radius_m": float(config.get("xy_search_radius_m", 80.0)),
            "time_search_radius_ms": float(config.get("time_search_radius_ms", 30.0)),
        },
        "initial_dfn": {
            "patch_count": int(len(initial_df)),
            "layer_distribution": layer_counts(initial_df),
        },
        "well_controls": {
            "control_point_count": int(len(control_df)),
            "well_count": int(control_df["WellName"].nunique()) if not control_df.empty else 0,
            "layer_distribution": layer_counts(control_df) if not control_df.empty else {},
            "nearest_trajectory_distance_stats": finite_stats(track_dist),
        },
        "correction_actions": {
            "action_counts": action_counts,
            "adjusted_initial_patch_count": changed_initial_count,
            "added_well_control_patch_count": added_count,
            "changed_or_added_fraction_of_initial": changed_fraction,
        },
        "corrected_dfn": {
            "patch_count": int(len(corrected_df)),
            "layer_distribution": layer_counts(corrected_df),
            "well_control_patch_count": int(safe_numeric(corrected_df["IsWellControlPatch"]).fillna(0).sum()),
            "center_x_stats": finite_stats(corrected_df["CenterX"]),
            "center_y_stats": finite_stats(corrected_df["CenterY"]),
            "center_time_stats": finite_stats(corrected_df["CenterTime"]),
        },
        "match_quality": {
            "before_distance_stats": finite_stats(before_dist),
            "after_distance_stats": finite_stats(after_dist),
            "mean_distance_improvement": (before_mean - after_mean) if before_mean is not None and after_mean is not None else None,
            "median_distance_improvement": (before_median - after_median) if before_median is not None and after_median is not None else None,
        },
        "macro_distribution_check": macro,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    paths = output_paths(output_dir)
    ensure_dir(output_dir)

    initial_csv = Path(config["initial_dfn_csv"]).resolve()
    fracture_csv = Path(config["fracture_points_csv"]).resolve()
    samples_root = Path(config["real_well_samples_root"]).resolve()
    initial_summary_json = Path(config.get("initial_dfn_summary_json", "")).resolve()
    for label, path in [("initial_dfn_csv", initial_csv), ("fracture_points_csv", fracture_csv), ("real_well_samples_root", samples_root)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    initial_df = load_initial_dfn(initial_csv)
    control_df = load_control_points(fracture_csv, dict(config["target_block"]))
    tracks = load_real_well_tracks(samples_root, set(control_df["WellName"].astype(str).unique()))
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    track_dist = nearest_track_distances(control_df, tracks, time_scale=time_scale)
    before_dist = nearest_patch_distances(initial_df, control_df, time_scale=time_scale)
    corrected_df, audit_df = apply_well_controls(
        initial_df=initial_df,
        control_df=control_df,
        track_distances=track_dist,
        config=config,
    )
    after_dist = nearest_patch_distances(corrected_df, control_df, time_scale=time_scale)

    corrected_df.to_csv(paths["corrected_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    display_z_scale = float(config.get("display_z_scale", 5.0))
    write_legacy_vtk(paths["raw_vtk"], corrected_df, "well_corrected_dfn_raw_time", display=False, display_z_scale=display_z_scale)
    write_legacy_vtk(paths["display_vtk"], corrected_df, "well_corrected_dfn_display", display=True, display_z_scale=display_z_scale)

    density_mass = load_density_mass_from_initial_summary(initial_summary_json)
    summary = build_summary(
        config_path=config_path,
        config=config,
        paths=paths,
        initial_df=initial_df,
        corrected_df=corrected_df,
        control_df=control_df,
        audit_df=audit_df,
        before_dist=before_dist,
        after_dist=after_dist,
        track_dist=track_dist,
        density_mass=density_mass,
    )
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Corrected DFN CSV: {paths['corrected_csv']}")
    print(f"Corrected DFN raw VTK: {paths['raw_vtk']}")
    print(f"Corrected DFN display VTK: {paths['display_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Control points: {len(control_df)} corrected patches: {len(corrected_df)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
