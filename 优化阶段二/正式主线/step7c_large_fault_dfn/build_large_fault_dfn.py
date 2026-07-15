from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage


CURRENT_DIR = Path(__file__).resolve().parent
LEGACY_STEP7B_DIR = CURRENT_DIR.parent / "step7b_initial_dfn_3d"
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_step7c_large_v1.json"
if str(LEGACY_STEP7B_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_STEP7B_DIR))

import build_initial_dfn_from_3d_density_sgy as legacy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step7C large fault/fault-zone DFN from original fault sticks and Step6C prior.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "fault_surface_vtk": output_dir / "large_fault_surface_raw_time.vtk",
        "dfn_csv": output_dir / "large_fault_and_damage_patches.csv",
        "raw_vtk": output_dir / "large_fault_and_damage_raw_time.vtk",
        "audit_csv": output_dir / "large_generation_audit.csv",
        "summary_json": output_dir / "large_fault_dfn_summary.json",
    }


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1.0e-9:
        return arr / norm
    fb = np.asarray(fallback, dtype=float)
    return fb / max(float(np.linalg.norm(fb)), 1.0e-9)


def strike_dip_from_normal(normal: np.ndarray) -> tuple[float, float]:
    n = normalize(normal, np.array([0.0, 0.0, 1.0]))
    strike = np.array([-n[1], n[0], 0.0], dtype=float)
    if float(np.linalg.norm(strike)) < 1.0e-9:
        azimuth = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(strike[1], strike[0])) % 180.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(n[2])), 0.0, 1.0))))
    return azimuth, dip


def plane_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(points, dtype=float)
    centered = pts - pts.mean(axis=0, keepdims=True)
    if len(pts) >= 3 and float(np.linalg.norm(centered)) > 1.0e-9:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis1 = normalize(vh[0], np.array([1.0, 0.0, 0.0]))
        axis2 = normalize(vh[1], np.array([0.0, 1.0, 0.0]))
        normal = normalize(vh[2], np.array([0.0, 0.0, 1.0]))
    elif len(pts) == 2:
        line_axis = normalize(pts[1] - pts[0], np.array([1.0, 0.0, 0.0]))
        vertical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(line_axis, vertical_axis))) > 0.95:
            vertical_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = normalize(np.cross(line_axis, vertical_axis), np.array([0.0, 1.0, 0.0]))
        axis1 = normalize(line_axis, np.array([1.0, 0.0, 0.0]))
        axis2 = normalize(np.cross(normal, axis1), np.array([0.0, 0.0, 1.0]))
    else:
        axis1 = np.array([1.0, 0.0, 0.0], dtype=float)
        axis2 = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.array([0.0, 1.0, 0.0], dtype=float)
    if axis2[2] < 0:
        axis2 = -axis2
    azimuth, dip = strike_dip_from_normal(normal)
    return axis1, axis2, normal, azimuth, dip


def make_patch(
    patch_id: str,
    center: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    length: float,
    height: float,
    azimuth: float,
    dip: float,
    layer: str,
    source_type: str,
    constraint: str,
    confidence: float,
    source_density: float,
    fault_name: str = "",
    component_id: int = 0,
    ordinal: int = 0,
) -> dict[str, Any]:
    return {
        "PatchID": patch_id,
        "GenerationStage": "step7c_large_fault_dfn",
        "SourceTraceIdx": -1,
        "DensityCellID": f"{source_type}_{patch_id}",
        "DensityCellPatchOrdinal": int(max(ordinal, 1)),
        "LayerGroup": layer,
        "LayerCode": legacy.LAYER_CODE.get(layer, 0),
        "CenterX": float(center[0]),
        "CenterY": float(center[1]),
        "CenterTime": float(center[2]),
        "TimeWindowMin": float(center[2] - 0.5 * height),
        "TimeWindowMax": float(center[2] + 0.5 * height),
        "LayerThickness": float(height),
        "SourceDensity": float(source_density),
        "CoherenceValue": np.nan,
        "LowCoherenceScore": float(source_density),
        "LowCoherenceWeight": 1.0,
        "GuidedDensityScore": float(source_density),
        "CandidateScore": float(source_density),
        "SamplingWeight": float(source_density),
        "DensityQuantileP95Layer": 1.0,
        "ExpectedPatchCountForCell": 1.0,
        "EffectiveCountScale": 1.0,
        "LengthM": float(length),
        "HeightTimeMs": float(height),
        "HeightM": float(height),
        "PatchAreaM2": float(length * height),
        "PatchEquivalentRadiusM": float(np.sqrt(max(length * height, 0.0) / np.pi)),
        "AzimuthDeg": float(azimuth % 180.0),
        "DipDeg": float(np.clip(dip, 0.0, 89.0)),
        "TraceGridDX": 12.5,
        "TraceGridDY": 12.5,
        "VoxelIX": -1,
        "VoxelIY": -1,
        "VoxelIT": -1,
        "ComponentID": int(component_id),
        "ComponentVoxelCount": 0,
        "LayerDensityThreshold": 0.0,
        "LocalOrientationPointCount": 0,
        "LocalDensityLinearity": 0.0,
        "LocalDensityPlanarity": 1.0,
        "LocalEigenvalue1": 0.0,
        "LocalEigenvalue2": 0.0,
        "LocalEigenvalue3": 0.0,
        "DensitySizeFactor": 1.0,
        "OrientationSource": "original_fault_surface_geometry" if source_type == "large_original_fault_surface" else "large_lowcoh_component_pca",
        "SizeRule": "large_fault_surface_or_component_extent",
        "SamplingRule": "step7c_large_fault_hard_constraint_and_lowcoh_supplement",
        "FractureScale": "large",
        "FractureScaleCode": 3,
        "BandID": fault_name or f"large_component_{component_id}",
        "BandPatchOrdinal": int(max(ordinal, 1)),
        "BandContinuityMode": "large_fault_surface_panel" if source_type == "large_original_fault_surface" else "large_lowcoh_component_panel",
        "BandVoxelCount": 0,
        "BandLengthM": float(length),
        "BandTimeExtentMs": float(height),
        "BandPatchSpacingM": 0.0,
        "BandMeanDensity": float(source_density),
        "ObjectBandAzimuthDeg": float(azimuth % 180.0),
        "ObjectBandDipDeg": float(np.clip(dip, 0.0, 89.0)),
        "ObjectBandLengthM": float(length),
        "ObjectBandCenterSpacingM": 0.0,
        "ObjectBandOverlapRatio": 0.0,
        "SourceType": source_type,
        "ConstraintLevel": constraint,
        "Confidence": float(confidence),
        "FaultName": fault_name,
        "NeedsWellCorrection": 0 if constraint == "hard" else 1,
    }


def read_fault_sticks(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=["Line", "Trace", "X", "Y", "Time", "Flag", "FaultName"], engine="python")
    target = dict(config.get("target_block", {}))
    x_min = float(target.get("x_min", -np.inf)) - float(config.get("fault_clip_margin_m", 0.0))
    x_max = float(target.get("x_max", np.inf)) + float(config.get("fault_clip_margin_m", 0.0))
    y_min = float(target.get("y_min", -np.inf)) - float(config.get("fault_clip_margin_m", 0.0))
    y_max = float(target.get("y_max", np.inf)) + float(config.get("fault_clip_margin_m", 0.0))
    t_min = float(config.get("time_min_ms", -np.inf))
    t_max = float(config.get("time_max_ms", np.inf))
    out = df[df["X"].between(x_min, x_max) & df["Y"].between(y_min, y_max) & df["Time"].between(t_min, t_max)].copy()
    return out.reset_index(drop=True)


def build_fault_surface_panels(fault_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[tuple[np.ndarray, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    surface_polys: list[tuple[np.ndarray, dict[str, Any]]] = []
    patch_idx = 0
    min_points = int(config.get("min_fault_points_per_panel", 3))
    max_length = float(config.get("max_fault_panel_length_m", 700.0))
    max_height = float(config.get("max_fault_panel_height_ms", 360.0))
    damage_count = int(config.get("damage_zone_patch_count_per_fault", 2))
    damage_offset = float(config.get("damage_zone_offset_m", 60.0))
    for fault_name, fault_group in fault_df.groupby("FaultName", dropna=False):
        for flag, stick in fault_group.groupby("Flag", dropna=False):
            if len(stick) < min_points:
                continue
            pts = stick.sort_values("Time")[["X", "Y", "Time"]].to_numpy(dtype=float)
            axis1, axis2, normal, azimuth, dip = plane_axes(pts)
            center = pts.mean(axis=0)
            proj1 = (pts - center) @ axis1
            proj2 = (pts - center) @ axis2
            length = float(np.quantile(proj1, 0.95) - np.quantile(proj1, 0.05))
            height = float(np.quantile(pts[:, 2], 0.95) - np.quantile(pts[:, 2], 0.05))
            length = float(np.clip(length, float(config.get("min_fault_panel_length_m", 80.0)), max_length))
            height = float(np.clip(height, float(config.get("min_fault_panel_height_ms", 30.0)), max_height))
            if length <= 0 or height <= 0:
                continue
            patch_idx += 1
            patch_id = f"large_fault_surface_{patch_idx:05d}"
            rows.append(make_patch(patch_id, center, axis1, axis2, length, height, azimuth, dip, "沙三段", "large_original_fault_surface", "hard", 1.0, 1.0, str(fault_name), int(flag), patch_idx))
            half_l = 0.5 * length
            half_h = 0.5 * height
            vertices = np.asarray(
                [
                    center - half_l * axis1 - half_h * axis2,
                    center + half_l * axis1 - half_h * axis2,
                    center + half_l * axis1 + half_h * axis2,
                    center - half_l * axis1 + half_h * axis2,
                ],
                dtype=float,
            )
            surface_polys.append((vertices, {"FaultName": str(fault_name), "Flag": int(flag), "PatchID": patch_id, "AzimuthDeg": azimuth, "DipDeg": dip}))
            for damage_idx in range(damage_count):
                side = -1.0 if damage_idx % 2 == 0 else 1.0
                offset_center = center + side * damage_offset * normal
                patch_idx += 1
                rows.append(
                    make_patch(
                        f"large_fault_damage_{patch_idx:05d}",
                        offset_center,
                        axis1,
                        axis2,
                        length * float(config.get("damage_zone_length_multiplier", 0.65)),
                        height * float(config.get("damage_zone_height_multiplier", 0.80)),
                        azimuth,
                        dip,
                        "沙三段",
                        "large_original_fault_damage_zone",
                        "hard_influence",
                        0.85,
                        0.85,
                        str(fault_name),
                        int(flag),
                        patch_idx,
                    )
                )
    return pd.DataFrame(rows), surface_polys


def build_large_lowcoh_supplements(config: dict[str, Any]) -> pd.DataFrame:
    if not config.get("large_prior_sgy"):
        return pd.DataFrame()
    grid = legacy.load_density_grid(Path(config["large_prior_sgy"]).resolve(), Path(config["trace_mapping_npz"]).resolve())
    mask_grid = legacy.load_density_grid(Path(config["large_mask_sgy"]).resolve(), Path(config["trace_mapping_npz"]).resolve())["density"]
    surfaces = legacy.attach_surface_grids(Path(config["layer_dir"]).resolve(), grid["x_values"], grid["y_values"])
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(mask_grid > 0.5, structure=structure)
    rows: list[dict[str, Any]] = []
    max_components = int(config.get("max_lowcoh_supplement_components", 8))
    min_voxels = int(config.get("min_lowcoh_component_voxels", 80))
    sizes = np.bincount(labels.ravel()) if count else np.asarray([], dtype=int)
    component_ids = [idx for idx in range(1, count + 1) if int(sizes[idx]) >= min_voxels]
    component_ids.sort(key=lambda idx: float(grid["density"][labels == idx].sum()), reverse=True)
    for ordinal, component_id in enumerate(component_ids[:max_components], start=1):
        yy, xx, tt = np.where(labels == component_id)
        if yy.size < 3:
            continue
        coords = np.column_stack([grid["x_values"][xx], grid["y_values"][yy], grid["samples"][tt]]).astype(float)
        axis1, axis2, _normal, azimuth, dip = plane_axes(coords)
        center = coords.mean(axis=0)
        proj1 = (coords - center) @ axis1
        length = float(np.quantile(proj1, 0.92) - np.quantile(proj1, 0.08))
        height = float(np.quantile(coords[:, 2], 0.92) - np.quantile(coords[:, 2], 0.08))
        length = float(np.clip(length, float(config.get("min_lowcoh_length_m", 120.0)), float(config.get("max_lowcoh_length_m", 480.0))))
        height = float(np.clip(height, float(config.get("min_lowcoh_height_ms", 35.0)), float(config.get("max_lowcoh_height_ms", 180.0))))
        layer = "沙三段"
        if legacy.layer_mask_for_grid("沙四段", grid["samples"], surfaces)[yy, xx, tt].sum() > yy.size / 2:
            layer = "沙四段"
        rows.append(
            make_patch(
                f"large_lowcoh_supplement_{ordinal:05d}",
                center,
                axis1,
                axis2,
                length,
                height,
                azimuth,
                dip,
                layer,
                "large_lowcoh_inferred",
                "seismic_prior",
                0.65,
                float(np.mean(grid["density"][yy, xx, tt])),
                f"large_lowcoh_component_{component_id}",
                int(component_id),
                ordinal,
            )
        )
    return pd.DataFrame(rows)


def patch_vertices(row: pd.Series) -> list[tuple[float, float, float]]:
    azimuth = np.deg2rad(float(row["AzimuthDeg"]))
    dip = np.deg2rad(float(np.clip(row["DipDeg"], 1.0, 89.9)))
    half_length = 0.5 * float(row["LengthM"])
    half_height_time = 0.5 * float(row["HeightTimeMs"])
    strike = np.asarray([np.cos(azimuth), np.sin(azimuth)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(azimuth), np.cos(azimuth)], dtype=float)
    horizontal_dip_half = half_height_time / max(np.tan(dip), 1.0e-6)
    center_xy = np.asarray([float(row["CenterX"]), float(row["CenterY"])], dtype=float)
    center_t = float(row["CenterTime"])
    corners = []
    for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        xy = center_xy + strike_sign * half_length * strike + dip_sign * horizontal_dip_half * dip_horizontal
        corners.append((float(xy[0]), float(xy[1]), float(center_t + dip_sign * half_height_time)))
    return corners


def write_patch_vtk(path: Path, df: pd.DataFrame, title: str) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in df.iterrows():
        base = len(points)
        points.extend(patch_vertices(row))
        polygons.append([base, base + 1, base + 2, base + 3])
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA", f"POINTS {len(points)} float"]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {sum(len(p)+1 for p in polygons)}")
    lines.extend(f"{len(p)} {' '.join(str(i) for i in p)}" for p in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    scalar_cols = ["FractureScaleCode", "SourceDensity", "Confidence", "LengthM", "HeightTimeMs", "PatchAreaM2", "AzimuthDeg", "DipDeg"]
    for col in scalar_cols:
        lines.append(f"SCALARS {col} {'int' if col == 'FractureScaleCode' else 'float'} 1")
        lines.append("LOOKUP_TABLE default")
        values = np.nan_to_num(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if col == "FractureScaleCode":
            lines.extend(str(int(v)) for v in values)
        else:
            lines.extend(f"{float(v):.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_surface_vtk(path: Path, surfaces: list[tuple[np.ndarray, dict[str, Any]]], title: str) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    azimuths: list[float] = []
    dips: list[float] = []
    for vertices, meta in surfaces:
        base = len(points)
        points.extend([(float(x), float(y), float(z)) for x, y, z in vertices])
        polygons.append([base, base + 1, base + 2, base + 3])
        azimuths.append(float(meta.get("AzimuthDeg", 0.0)))
        dips.append(float(meta.get("DipDeg", 0.0)))
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA", f"POINTS {len(points)} float"]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {sum(len(p)+1 for p in polygons)}")
    lines.extend(f"{len(p)} {' '.join(str(i) for i in p)}" for p in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    for name, values in [("AzimuthDeg", azimuths), ("DipDeg", dips)]:
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(f"{float(v):.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)
    fault_df = read_fault_sticks(Path(config["fault_stick_dat"]).resolve(), config)
    print(f"[step7c-large] clipped fault stick rows={len(fault_df)}", flush=True)
    fault_patches, fault_surfaces = build_fault_surface_panels(fault_df, config)
    lowcoh_patches = build_large_lowcoh_supplements(config)
    parts = [df for df in [fault_patches, lowcoh_patches] if not df.empty]
    if not parts:
        raise RuntimeError("no large fault/fault-zone patches generated")
    patch_df = pd.concat(parts, ignore_index=True)
    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    patch_df[["PatchID", "GenerationStage", "SourceType", "ConstraintLevel", "FaultName", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg", "PatchAreaM2"]].to_csv(
        paths["audit_csv"], index=False, encoding="utf-8-sig"
    )
    write_patch_vtk(paths["raw_vtk"], patch_df, "step7c_large_fault_and_damage_raw_time")
    write_surface_vtk(paths["fault_surface_vtk"], fault_surfaces, "step7c_large_original_fault_surface_raw_time")
    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "generation_logic": "step7c_original_fault_surface_plus_large_lowcoh_supplement",
        "inputs": {
            "fault_stick_dat": str(Path(config["fault_stick_dat"]).resolve()),
            "large_prior_sgy": str(Path(config["large_prior_sgy"]).resolve()),
            "large_mask_sgy": str(Path(config["large_mask_sgy"]).resolve()),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "clipped_fault_stick_rows": int(len(fault_df)),
        "fault_surface_patch_count": int(len(fault_patches)),
        "raw_fault_surface_panel_count": int(len(fault_surfaces)),
        "lowcoh_supplement_patch_count": int(len(lowcoh_patches)),
        "patch_count": int(len(patch_df)),
        "source_type_counts": {str(k): int(v) for k, v in patch_df["SourceType"].value_counts(dropna=False).items()},
        "patch_stats": {
            "length_m": legacy.finite_stats(patch_df["LengthM"]),
            "height_time_ms": legacy.finite_stats(patch_df["HeightTimeMs"]),
            "area_m2": legacy.finite_stats(patch_df["PatchAreaM2"]),
            "azimuth_deg": legacy.finite_stats(patch_df["AzimuthDeg"]),
            "dip_deg": legacy.finite_stats(patch_df["DipDeg"]),
        },
        "checks": {
            "has_large_patches": len(patch_df) > 0,
            "fault_surface_vtk_exists": paths["fault_surface_vtk"].exists(),
            "raw_vtk_exists": paths["raw_vtk"].exists(),
            "csv_exists": paths["dfn_csv"].exists(),
        },
    }
    summary["status"] = "pass" if all(bool(v) for v in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7c-large] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7c-large] VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7c-large] status={summary['status']} patch_count={len(patch_df)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
