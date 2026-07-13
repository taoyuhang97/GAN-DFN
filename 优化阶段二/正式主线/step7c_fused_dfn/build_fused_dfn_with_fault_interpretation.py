from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_fused_candidate_cheye1_detail_preserve_v4_fault_hard.json"
ALLOWED_LAYERS = ["沙三段", "沙四段"]
LAYER_CODE = {"沙三段": 3, "沙四段": 4}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse predicted DFN with hard fault-interpretation patches.")
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
        "fused_csv": output_dir / "fused_initial_dfn_with_fault_patches.csv",
        "raw_vtk": output_dir / "fused_initial_dfn_with_fault_raw_time.vtk",
        "fault_only_csv": output_dir / "fault_interpretation_dfn_patches.csv",
        "fault_only_vtk": output_dir / "fault_interpretation_patches_raw_time.vtk",
        "audit_csv": output_dir / "fused_initial_dfn_with_fault_audit.csv",
        "summary_json": output_dir / "fused_initial_dfn_with_fault_summary.json",
    }


def load_predicted_dfn(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path, low_memory=False)
    required = ["PatchID", "LayerGroup", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"predicted DFN missing required columns: {missing}")
    out = df.copy()
    for column in [
        "CenterX", "CenterY", "CenterTime", "TimeWindowMin", "TimeWindowMax", "LayerThickness", "SourceDensity",
        "LengthM", "HeightTimeMs", "HeightM", "PatchAreaM2", "PatchEquivalentRadiusM", "AzimuthDeg", "DipDeg",
    ]:
        if column in out.columns:
            out[column] = safe_numeric(out[column])
    out = out[out["LayerGroup"].astype(str).isin(ALLOWED_LAYERS)].copy()
    out["SourceType"] = out.get("SourceType", "density_3d_predicted")
    out["ConstraintLevel"] = out.get("ConstraintLevel", "predicted")
    out["FaultName"] = out.get("FaultName", "")
    out["FaultRelation"] = out.get("FaultRelation", "background")
    out["CanModifyInStep8"] = out.get("CanModifyInStep8", 1)
    out["CanBeRemovedByDensityFilter"] = out.get("CanBeRemovedByDensityFilter", 1)
    out["CanBeJittered"] = out.get("CanBeJittered", 1)
    return out.reset_index(drop=True)


def fault_summary_to_rows(fault_summary: pd.DataFrame, predicted: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    target = dict(config.get("target_block", {}))
    x_min = float(target.get("x_min", -np.inf))
    x_max = float(target.get("x_max", np.inf))
    y_min = float(target.get("y_min", -np.inf))
    y_max = float(target.get("y_max", np.inf))
    margin = float(config.get("fault_target_margin_m", 0.0))
    fault = fault_summary.copy()
    for column in [
        "cell_i", "cell_j", "area_3d", "cx", "cy", "cz", "bbox_xmin", "bbox_xmax", "bbox_ymin", "bbox_ymax",
        "bbox_zmin", "bbox_zmax", "strike_deg", "dip_deg",
    ]:
        if column in fault.columns:
            fault[column] = safe_numeric(fault[column])
    overlap = (
        fault["bbox_xmax"].ge(x_min - margin)
        & fault["bbox_xmin"].le(x_max + margin)
        & fault["bbox_ymax"].ge(y_min - margin)
        & fault["bbox_ymin"].le(y_max + margin)
    )
    fault = fault[overlap].copy().reset_index(drop=True)
    if fault.empty:
        return pd.DataFrame()

    time_scale = float(config.get("time_scale_m_per_ms", 1.0))
    pred_coords = np.column_stack(
        [
            predicted["CenterX"].to_numpy(dtype=float),
            predicted["CenterY"].to_numpy(dtype=float),
            predicted["CenterTime"].to_numpy(dtype=float) * time_scale,
        ]
    )
    tree = cKDTree(pred_coords)
    rows: list[dict[str, Any]] = []
    for idx, row in fault.iterrows():
        center = np.asarray([float(row["cx"]), float(row["cy"]), float(row["cz"]) * time_scale], dtype=float)
        _, nearest_idx = tree.query(center, k=1)
        nearest = predicted.iloc[int(nearest_idx)]
        dx = float(row["bbox_xmax"] - row["bbox_xmin"])
        dy = float(row["bbox_ymax"] - row["bbox_ymin"])
        dz = max(float(row["bbox_zmax"] - row["bbox_zmin"]), float(config.get("min_fault_height_time_ms", 10.0)))
        horizontal_extent = max(float(np.hypot(dx, dy)), float(config.get("min_fault_length_m", 30.0)))
        height_time = dz
        area = float(row.get("area_3d", horizontal_extent * height_time))
        equivalent_radius = float(np.sqrt(max(area, 0.0) / np.pi))
        fault_name = str(row["fault_name"])
        patch_id = f"fault_interp_{fault_name}_i{int(row['cell_i'])}_j{int(row['cell_j'])}_{idx + 1:05d}"
        rows.append(
            {
                "PatchID": patch_id,
                "OriginalPatchID": patch_id,
                "GenerationStage": "fault_interpretation_hard_constraint",
                "InputSource": "fault_interpretation",
                "FusionSource": "fault_interpretation_hard_constraint",
                "SourceType": "fault_interpretation",
                "ConstraintLevel": "hard",
                "FaultRelation": "fault_core",
                "FaultName": fault_name,
                "FaultCellI": int(row["cell_i"]),
                "FaultCellJ": int(row["cell_j"]),
                "LayerGroup": str(nearest["LayerGroup"]),
                "LayerCode": int(LAYER_CODE[str(nearest["LayerGroup"])]),
                "CenterX": float(row["cx"]),
                "CenterY": float(row["cy"]),
                "CenterTime": float(row["cz"]),
                "TimeWindowMin": float(nearest.get("TimeWindowMin", np.nan)),
                "TimeWindowMax": float(nearest.get("TimeWindowMax", np.nan)),
                "LayerThickness": float(nearest.get("LayerThickness", np.nan)),
                "SourceDensity": float(config.get("fault_source_density", 999.0)),
                "LengthM": horizontal_extent,
                "HeightTimeMs": height_time,
                "HeightM": height_time * time_scale,
                "PatchAreaM2": area,
                "PatchEquivalentRadiusM": equivalent_radius,
                "AzimuthDeg": float(row["strike_deg"]) % 180.0,
                "DipDeg": float(np.clip(row["dip_deg"], float(config.get("min_fault_dip_deg", 45.0)), float(config.get("max_fault_dip_deg", 89.0)))),
                "NeedsWellCorrection": 0,
                "CanModifyInStep8": 0,
                "CanBeRemovedByDensityFilter": 0,
                "CanBeJittered": 0,
                "DistanceToFaultM": 0.0,
                "NearestFaultName": fault_name,
                "NearestFaultPatchID": patch_id,
                "SamplingRule": "geologist_fault_interpretation_xy_time_patch",
                "SizeRule": "fault_patch_bbox_and_area_preserved",
                "FractureScale": "major_fault",
                "FractureScaleCode": 4,
            }
        )
    return pd.DataFrame(rows)


def mark_and_filter_predicted(predicted: pd.DataFrame, faults: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if predicted.empty or faults.empty:
        out = predicted.copy()
        out["DistanceToFaultM"] = np.nan
        out["NearestFaultName"] = ""
        out["NearestFaultPatchID"] = ""
        return out, {"fault_relation_enabled": False, "removed_predicted_count": 0}
    time_scale = float(config.get("time_scale_m_per_ms", 1.0))
    damage_xy = float(config.get("damage_zone_xy_buffer_m", 150.0))
    damage_time = float(config.get("damage_zone_time_buffer_ms", 100.0))
    remove_xy = float(config.get("remove_small_xy_buffer_m", 80.0))
    remove_time = float(config.get("remove_small_time_buffer_ms", 60.0))
    fault_coords = np.column_stack(
        [faults["CenterX"].to_numpy(dtype=float), faults["CenterY"].to_numpy(dtype=float), faults["CenterTime"].to_numpy(dtype=float) * time_scale]
    )
    tree = cKDTree(fault_coords)
    pred_coords = np.column_stack(
        [predicted["CenterX"].to_numpy(dtype=float), predicted["CenterY"].to_numpy(dtype=float), predicted["CenterTime"].to_numpy(dtype=float) * time_scale]
    )
    dist_scaled, nearest = tree.query(pred_coords, k=1)
    nearest_fault = faults.iloc[nearest].reset_index(drop=True)
    dx = predicted["CenterX"].to_numpy(dtype=float) - nearest_fault["CenterX"].to_numpy(dtype=float)
    dy = predicted["CenterY"].to_numpy(dtype=float) - nearest_fault["CenterY"].to_numpy(dtype=float)
    dt = np.abs(predicted["CenterTime"].to_numpy(dtype=float) - nearest_fault["CenterTime"].to_numpy(dtype=float))
    dxy = np.sqrt(dx * dx + dy * dy)
    out = predicted.copy()
    out["DistanceToFaultM"] = dist_scaled.astype(float)
    out["NearestFaultName"] = nearest_fault["FaultName"].astype(str).to_numpy()
    out["NearestFaultPatchID"] = nearest_fault["PatchID"].astype(str).to_numpy()
    in_damage = (dxy <= damage_xy) & (dt <= damage_time)
    remove_small = (
        out.get("FractureScale", pd.Series([""] * len(out))).astype(str).eq("small").to_numpy()
        & (dxy <= remove_xy)
        & (dt <= remove_time)
        & bool(config.get("remove_conflicting_small_fractures", True))
    )
    out.loc[in_damage, "FaultRelation"] = "fault_damage_zone"
    out.loc[~in_damage, "FaultRelation"] = out.loc[~in_damage, "FaultRelation"].fillna("background")
    out.loc[in_damage, "ConstraintLevel"] = "guided_by_fault"
    out.loc[in_damage, "SourceType"] = "density_3d_fault_related"
    out.loc[in_damage, "CanBeRemovedByDensityFilter"] = 0
    kept = out.loc[~remove_small].copy().reset_index(drop=True)
    return kept, {
        "fault_relation_enabled": True,
        "damage_zone_xy_buffer_m": damage_xy,
        "damage_zone_time_buffer_ms": damage_time,
        "remove_small_xy_buffer_m": remove_xy,
        "remove_small_time_buffer_ms": remove_time,
        "input_predicted_count": int(len(predicted)),
        "fault_damage_zone_predicted_count": int(in_damage.sum()),
        "removed_predicted_count": int(remove_small.sum()),
        "kept_predicted_count": int(len(kept)),
    }


def finalize_table(predicted_kept: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    fused = pd.concat([faults, predicted_kept], ignore_index=True, sort=False)
    fused["PatchID"] = [f"fault_fused_dfn_{idx + 1:06d}" for idx in range(len(fused))]
    if "OriginalPatchID" not in fused.columns:
        fused["OriginalPatchID"] = fused["PatchID"].astype(str)
    for column, default in [
        ("SourceType", "density_3d_predicted"),
        ("ConstraintLevel", "predicted"),
        ("FaultRelation", "background"),
        ("FaultName", ""),
        ("NearestFaultName", ""),
        ("NearestFaultPatchID", ""),
        ("CanModifyInStep8", 1),
        ("CanBeRemovedByDensityFilter", 1),
        ("CanBeJittered", 1),
        ("NeedsWellCorrection", 1),
    ]:
        if column not in fused.columns:
            fused[column] = default
        else:
            fused[column] = fused[column].fillna(default)
    fused["LayerCode"] = fused["LayerGroup"].map(LAYER_CODE).astype(int)
    return fused.reset_index(drop=True)


def patch_vertices(row: pd.Series, geometry_time_scale_m_per_ms: float) -> list[tuple[float, float, float]]:
    azimuth = np.deg2rad(float(row["AzimuthDeg"]))
    dip = np.deg2rad(float(np.clip(row["DipDeg"], 1.0, 89.9)))
    half_length = 0.5 * float(row["LengthM"])
    half_height_time = 0.5 * float(row["HeightTimeMs"])
    strike = np.asarray([np.cos(azimuth), np.sin(azimuth)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(azimuth), np.cos(azimuth)], dtype=float)
    horizontal_dip_half = (half_height_time * geometry_time_scale_m_per_ms) / max(np.tan(dip), 1.0e-6)
    center_xy = np.asarray([float(row["CenterX"]), float(row["CenterY"])], dtype=float)
    center_t = float(row["CenterTime"])
    points = []
    for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        xy = center_xy + strike_sign * half_length * strike + dip_sign * horizontal_dip_half * dip_horizontal
        z = center_t + dip_sign * half_height_time
        points.append((float(xy[0]), float(xy[1]), float(z)))
    return points


def categorical_code(series: pd.Series, mapping: dict[str, int], default: int = 0) -> np.ndarray:
    return series.astype(str).map(mapping).fillna(default).to_numpy(dtype=int)


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
        ("CenterTime", safe_numeric(patch_df["CenterTime"]).to_numpy(), "float"),
        ("LengthM", safe_numeric(patch_df["LengthM"]).to_numpy(), "float"),
        ("HeightTimeMs", safe_numeric(patch_df["HeightTimeMs"]).to_numpy(), "float"),
        ("PatchAreaM2", safe_numeric(patch_df.get("PatchAreaM2", pd.Series([0.0] * len(patch_df)))).fillna(0.0).to_numpy(), "float"),
        ("AzimuthDeg", safe_numeric(patch_df["AzimuthDeg"]).to_numpy(), "float"),
        ("DipDeg", safe_numeric(patch_df["DipDeg"]).to_numpy(), "float"),
        ("DistanceToFaultM", safe_numeric(patch_df.get("DistanceToFaultM", pd.Series([np.nan] * len(patch_df)))).fillna(-1.0).to_numpy(), "float"),
        ("SourceTypeCode", categorical_code(patch_df["SourceType"], {"fault_interpretation": 3, "density_3d_fault_related": 2, "density_3d_predicted": 1}), "int"),
        ("ConstraintLevelCode", categorical_code(patch_df["ConstraintLevel"], {"hard": 3, "guided_by_fault": 2, "predicted": 1}), "int"),
        ("FaultRelationCode", categorical_code(patch_df["FaultRelation"], {"fault_core": 3, "fault_damage_zone": 2, "background": 1}), "int"),
        ("CanModifyInStep8", safe_numeric(patch_df["CanModifyInStep8"]).fillna(1).to_numpy(), "int"),
    ]
    for name, values, dtype in scalar_columns:
        vtk_type = "int" if dtype == "int" else "float"
        lines.append(f"SCALARS {name} {vtk_type} 1")
        lines.append("LOOKUP_TABLE default")
        safe = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if vtk_type == "int":
            lines.extend(str(int(value)) for value in safe)
        else:
            lines.extend(f"{float(value):.6f}" for value in safe)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(config_path: Path, config: dict[str, Any], paths: dict[str, Path], predicted: pd.DataFrame, faults: pd.DataFrame, fused: pd.DataFrame, relation_summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "has_fused_patches": len(fused) > 0,
        "has_fault_hard_constraints": bool((fused["ConstraintLevel"].astype(str) == "hard").any()),
        "fault_patches_not_well_corrected": bool((fused.loc[fused["ConstraintLevel"].astype(str).eq("hard"), "NeedsWellCorrection"] == 0).all()),
        "layers_limited_to_sha3_sha4": bool(set(fused["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS))),
        "orientation_fields_complete": bool(fused[["AzimuthDeg", "DipDeg"]].notna().all().all()),
        "raw_vtk_output_exists": paths["raw_vtk"].exists(),
        "fault_only_vtk_output_exists": paths["fault_only_vtk"].exists(),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "input_predicted_dfn_csv": str(Path(config["input_predicted_dfn_csv"]).resolve()),
        "fault_patches_summary_csv": str(Path(config["fault_patches_summary_csv"]).resolve()),
        "fused_initial_dfn_with_fault_csv": str(paths["fused_csv"]),
        "fused_initial_dfn_with_fault_raw_vtk": str(paths["raw_vtk"]),
        "fault_interpretation_csv": str(paths["fault_only_csv"]),
        "fault_interpretation_raw_vtk": str(paths["fault_only_vtk"]),
        "summary_json": str(paths["summary_json"]),
        "generation_logic": "predicted_dfn_plus_geologist_fault_interpretation_hard_constraints",
        "parameters": {k: config.get(k) for k in ["time_scale_m_per_ms", "damage_zone_xy_buffer_m", "damage_zone_time_buffer_ms", "remove_small_xy_buffer_m", "remove_small_time_buffer_ms"]},
        "input_predicted": {"patch_count": int(len(predicted)), "source_type_distribution": predicted.get("SourceType", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()},
        "fault_interpretation": {"patch_count": int(len(faults)), "fault_count": int(faults["FaultName"].nunique()) if not faults.empty else 0, "area_stats": finite_stats(faults["PatchAreaM2"]) if not faults.empty else finite_stats([])},
        "relation_summary": relation_summary,
        "fused_dfn": {
            "patch_count": int(len(fused)),
            "source_type_distribution": {str(k): int(v) for k, v in fused["SourceType"].value_counts(dropna=False).sort_index().items()},
            "constraint_level_distribution": {str(k): int(v) for k, v in fused["ConstraintLevel"].value_counts(dropna=False).sort_index().items()},
            "fault_relation_distribution": {str(k): int(v) for k, v in fused["FaultRelation"].value_counts(dropna=False).sort_index().items()},
            "center_time_stats": finite_stats(fused["CenterTime"]),
            "length_m_stats": finite_stats(fused["LengthM"]),
            "height_time_ms_stats": finite_stats(fused["HeightTimeMs"]),
            "patch_area_m2_stats": finite_stats(fused["PatchAreaM2"]),
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
    predicted_csv = Path(config["input_predicted_dfn_csv"]).resolve()
    fault_summary_csv = Path(config["fault_patches_summary_csv"]).resolve()
    for label, path in [("input_predicted_dfn_csv", predicted_csv), ("fault_patches_summary_csv", fault_summary_csv)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print("[step7c-fault] loading predicted DFN", flush=True)
    predicted = load_predicted_dfn(predicted_csv)
    print("[step7c-fault] loading fault patch summary", flush=True)
    fault_summary = read_csv_flexible(fault_summary_csv, low_memory=False)
    faults = fault_summary_to_rows(fault_summary, predicted, config=config)
    print(f"[step7c-fault] predicted={len(predicted)} fault_hard={len(faults)}", flush=True)
    predicted_kept, relation_summary = mark_and_filter_predicted(predicted, faults, config=config)
    fused = finalize_table(predicted_kept, faults)
    audit = fused[["PatchID", "OriginalPatchID", "SourceType", "ConstraintLevel", "FaultRelation", "FaultName", "NearestFaultName", "DistanceToFaultM", "CanModifyInStep8", "NeedsWellCorrection"]].copy()
    audit["Action"] = "fuse_fault_interpretation_hard_constraint"
    fused.to_csv(paths["fused_csv"], index=False, encoding="utf-8-sig")
    faults.to_csv(paths["fault_only_csv"], index=False, encoding="utf-8-sig")
    audit.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    geometry_time_scale = float(config.get("geometry_time_scale_m_per_ms", config.get("time_scale_m_per_ms", 1.0)))
    write_raw_vtk(paths["raw_vtk"], fused, "fused_initial_dfn_with_fault_raw_time", geometry_time_scale_m_per_ms=geometry_time_scale)
    write_raw_vtk(paths["fault_only_vtk"], faults, "fault_interpretation_patches_raw_time", geometry_time_scale_m_per_ms=geometry_time_scale)
    summary = build_summary(config_path, config, paths, predicted, faults, fused, relation_summary)
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fused DFN with fault CSV: {paths['fused_csv']}")
    print(f"Fused DFN with fault raw VTK: {paths['raw_vtk']}")
    print(f"Fault interpretation raw VTK: {paths['fault_only_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Fused patch count: {len(fused)} fault patches: {len(faults)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
