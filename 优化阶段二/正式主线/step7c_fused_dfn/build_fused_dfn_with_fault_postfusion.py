from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
OLD_FAULT_POSTFUSION_DIR = REPO_ROOT / "研究内容三/优化阶段一/单元DFN融合/区域断层后融合"
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_fused_candidate_cheye1_detail_preserve_v4_fault_postfusion_v2.json"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")

if str(OLD_FAULT_POSTFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_FAULT_POSTFUSION_DIR))

from build_fault_surface_fragments_from_raw_patches import run_build_fault_surface_fragments  # noqa: E402
from build_fused_dfn_from_2d_3d import (  # noqa: E402
    build_audit as build_fracture_fusion_audit,
    correct_2d_orientation,
    filter_2d_supplements,
    finalize_fused_table,
    load_dfn as load_fracture_fusion_dfn,
    prepare_3d_primary,
    remove_duplicates,
    write_raw_vtk as write_fracture_fusion_raw_vtk,
)
from build_regional_fault_panels import run_build_regional_fault_panels  # noqa: E402
from fault_postfusion_common import make_patch_row_from_axes, write_df_to_regional_vtk, write_json  # noqa: E402
from fuse_faults_into_regional_dfn_v2 import run_fault_postfusion  # noqa: E402


LAYER_CODE = {"沙三段": 3, "沙四段": 4}
PATCH_ORIGIN_CODE_TO_SOURCE = {
    0: "density_3d_predicted",
    1: "fault_parallel",
    2: "fault_perpendicular",
    3: "fault_surface",
}
PATCH_ORIGIN_CODE_TO_RELATION = {
    0: "background",
    1: "fault_damage_zone",
    2: "fault_damage_zone",
    3: "fault_core",
}
FAULT_ACTION_CODE_TO_TEXT = {
    0: "keep",
    1: "remove",
    2: "shrink",
    3: "induced",
    4: "surface",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse Step7B DFN with real fault patches/panels using old postfusion logic.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def safe_numeric(values: Any, default: float = 0.0) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return series.fillna(default)


def finite_stats(values: Any) -> dict[str, float | int | None]:
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


def add_render_density_columns(df: pd.DataFrame, density_col: str = "SourceDensity") -> pd.DataFrame:
    out = df.copy()
    density = pd.to_numeric(out.get(density_col, pd.Series(np.zeros(len(out)))), errors="coerce").replace([np.inf, -np.inf], np.nan)
    geological_mask = pd.Series(True, index=out.index)
    if "SourceType" in out.columns:
        geological_mask &= ~out["SourceType"].astype(str).eq("fault_surface")
    if "ConstraintLevel" in out.columns:
        geological_mask &= ~out["ConstraintLevel"].astype(str).eq("hard")
    valid = density[geological_mask & density.notna()]
    if valid.empty:
        valid = density[density.notna()]
    p95 = float(valid.quantile(0.95)) if not valid.empty else 1.0
    p99 = float(valid.quantile(0.99)) if not valid.empty else max(p95, 1.0)
    clip_max = max(p99, p95, 1.0e-6)
    clipped = density.clip(lower=0.0, upper=clip_max).fillna(0.0)
    out["SourceDensityRender"] = clipped
    out["SourceDensityRenderNorm"] = (clipped / clip_max).clip(0.0, 1.0)
    out["SourceDensityRenderClipMax"] = clip_max
    return out


def compute_patch_geometry(row: pd.Series, time_scale_m_per_ms: float) -> dict[str, Any]:
    azimuth_rad = np.deg2rad(float(row["AzimuthDeg"]) % 180.0)
    dip_deg = float(np.clip(row["DipDeg"], 1.0, 89.9))
    dip_rad = np.deg2rad(dip_deg)
    strike_vec = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=float)
    dip_horizontal = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0], dtype=float)
    horizontal_per_ms = float(time_scale_m_per_ms) / max(np.tan(dip_rad), 1.0e-6)
    dip_vec = np.array(
        [
            dip_horizontal[0] * horizontal_per_ms,
            dip_horizontal[1] * horizontal_per_ms,
            1.0,
        ],
        dtype=float,
    )
    center = np.array([float(row["CenterX"]), float(row["CenterY"]), float(row["CenterTime"])], dtype=float)
    return make_patch_row_from_axes(
        center=center,
        u_vec=strike_vec,
        v_vec=dip_vec,
        length=float(row["LengthM"]),
        height=float(row["HeightTimeMs"]),
    )


def convert_step7b_csv_to_old_postfusion_df(input_csv: Path, config: dict[str, Any]) -> pd.DataFrame:
    df = read_csv_flexible(input_csv, low_memory=False)
    required = ["PatchID", "LayerGroup", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"input Step7B CSV missing required columns: {missing}")

    target = dict(config.get("target_block", {}))
    if target:
        x_min = float(target.get("x_min", -np.inf))
        x_max = float(target.get("x_max", np.inf))
        y_min = float(target.get("y_min", -np.inf))
        y_max = float(target.get("y_max", np.inf))
        df = df[
            df["CenterX"].between(x_min, x_max)
            & df["CenterY"].between(y_min, y_max)
        ].copy()
    df = df[df["LayerGroup"].astype(str).isin(LAYER_CODE)].copy().reset_index(drop=True)

    time_scale = float(config.get("geometry_time_scale_m_per_ms", config.get("time_scale_m_per_ms", 1.0)))
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        geom = compute_patch_geometry(row, time_scale_m_per_ms=time_scale)
        patch_area = float(row.get("PatchAreaM2", geom["PatchArea"]))
        out = {
            **geom,
            "OriginalRowIndex": int(idx),
            "LayerCode": int(row.get("LayerCode", LAYER_CODE.get(str(row["LayerGroup"]), 0))),
            "SourceDensity": float(row.get("SourceDensity", 0.0)),
            "CoherenceValue": float(row.get("CoherenceValue", 0.0)) if pd.notna(row.get("CoherenceValue", np.nan)) else 0.0,
            "LowCoherenceScore": float(row.get("LowCoherenceScore", 0.0)),
            "GuidedDensityScore": float(row.get("GuidedDensityScore", row.get("SourceDensity", 0.0))),
            "Confidence": float(np.clip(row.get("CandidateScore", row.get("SourceDensity", 0.5)), 0.0, 1.0))
            if float(row.get("CandidateScore", 0.5)) <= 1.0
            else 0.8,
            "PatchArea": patch_area,
            "PatchArea3D": patch_area,
            "ParentPatchCount": int(row.get("ParentPatchCount", 1)) if pd.notna(row.get("ParentPatchCount", 1)) else 1,
            "PatchOriginCode": 0,
            "FaultActionCode": 0,
            "ReliabilityLevel": "high" if str(row.get("FractureScale", "")) == "large" else "medium",
            "ConnectionType": "original",
            "AggregationMode": "original",
            "ScaleClass": "macro_core" if str(row.get("FractureScale", "")) == "large" else ("meso_link" if str(row.get("FractureScale", "")) == "medium" else "micro_bg"),
            "CorridorSupport": int(row.get("CorridorSupport", 0)) if pd.notna(row.get("CorridorSupport", 0)) else 0,
            "IsSupplemented": 0,
            "FractureSet": -1,
        }
        rows.append(out)
    return pd.DataFrame(rows)


def build_fracture_only_fusion(config: dict[str, Any], workspace: Path, output_dir: Path) -> dict[str, Any]:
    input_2d = Path(config["input_2d_dfn_csv"]).resolve()
    input_3d = Path(config["input_3d_dfn_csv"]).resolve()
    for label, path in [("input_2d_dfn_csv", input_2d), ("input_3d_dfn_csv", input_3d)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print("[step7c-postfusion] building fracture-only Step7+Step7B fusion", flush=True)
    df2d = load_fracture_fusion_dfn(input_2d, source_name="2d_density_grid")
    df3d = load_fracture_fusion_dfn(input_3d, source_name="3d_density_volume")
    corrected_2d, orientation_summary = correct_2d_orientation(df2d, df3d, config=config)
    retained_2d, filter_summary = filter_2d_supplements(corrected_2d, config=config)
    df3d_primary = prepare_3d_primary(df3d)
    fused_raw, duplicate_summary = remove_duplicates(df3d_primary, retained_2d, config=config)
    fracture_only_df = finalize_fused_table(fused_raw)
    computed_area = (
        pd.to_numeric(fracture_only_df["LengthM"], errors="coerce").fillna(0.0)
        * pd.to_numeric(fracture_only_df["HeightTimeMs"], errors="coerce").fillna(0.0)
    )
    if "PatchAreaM2" not in fracture_only_df.columns:
        fracture_only_df["PatchAreaM2"] = computed_area
    else:
        fracture_only_df["PatchAreaM2"] = pd.to_numeric(fracture_only_df["PatchAreaM2"], errors="coerce").fillna(computed_area)
    fracture_only_df = add_render_density_columns(fracture_only_df)
    audit_df = build_fracture_fusion_audit(fracture_only_df)

    fracture_only_csv = output_dir / "fracture_only_step7_step7b_fused_patches.csv"
    fracture_only_vtk = output_dir / "fracture_only_step7_step7b_fused_raw_time.vtk"
    fracture_only_audit = output_dir / "fracture_only_step7_step7b_fused_audit.csv"
    fracture_only_df.to_csv(fracture_only_csv, index=False, encoding="utf-8-sig")
    audit_df.to_csv(fracture_only_audit, index=False, encoding="utf-8-sig")
    write_fracture_fusion_raw_vtk(
        fracture_only_vtk,
        fracture_only_df,
        title="fracture_only_step7_step7b_fused_raw_time",
        geometry_time_scale_m_per_ms=float(config.get("geometry_time_scale_m_per_ms", 1.0)),
    )

    legacy_input_df = convert_step7b_csv_to_old_postfusion_df(fracture_only_csv, config=config)
    legacy_input_csv = workspace / "legacy_fracture_only_input_for_fault_postfusion.csv"
    legacy_input_vtk = workspace / "legacy_fracture_only_input_for_fault_postfusion_raw.vtk"
    legacy_input_df.to_csv(legacy_input_csv, index=False, encoding="utf-8-sig")
    write_df_to_regional_vtk(legacy_input_df, "legacy_fracture_only_input_for_fault_postfusion", legacy_input_vtk, {})

    return {
        "mode": "step7_2d_plus_step7b_3d_fracture_only_fusion",
        "input_2d_dfn_csv": str(input_2d),
        "input_3d_dfn_csv": str(input_3d),
        "fracture_only_csv": str(fracture_only_csv),
        "fracture_only_vtk": str(fracture_only_vtk),
        "fracture_only_audit_csv": str(fracture_only_audit),
        "legacy_input_csv": str(legacy_input_csv),
        "legacy_input_vtk": str(legacy_input_vtk),
        "input_2d_count": int(len(df2d)),
        "input_3d_count": int(len(df3d)),
        "fracture_only_patch_count": int(len(fracture_only_df)),
        "fracture_only_fusion_source_distribution": {
            str(k): int(v) for k, v in fracture_only_df["FusionSource"].value_counts(dropna=False).items()
        },
        "orientation_summary": {k: v for k, v in orientation_summary.items() if not str(k).endswith("_values")},
        "filter_summary": filter_summary,
        "duplicate_summary": duplicate_summary,
    }


def build_single_input_fracture_only(config: dict[str, Any], workspace: Path, output_dir: Path) -> dict[str, Any]:
    input_csv = Path(config["input_predicted_dfn_csv"]).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"input_predicted_dfn_csv not found: {input_csv}")
    print("[step7c-postfusion] using single fracture input for fault postfusion", flush=True)
    legacy_input_df = convert_step7b_csv_to_old_postfusion_df(input_csv, config=config)
    legacy_input_csv = workspace / "legacy_single_input_for_fault_postfusion.csv"
    legacy_input_vtk = workspace / "legacy_single_input_for_fault_postfusion_raw.vtk"
    fracture_only_vtk = output_dir / "fracture_only_step7_step7b_fused_raw_time.vtk"
    legacy_input_df.to_csv(legacy_input_csv, index=False, encoding="utf-8-sig")
    write_df_to_regional_vtk(legacy_input_df, "legacy_single_input_for_fault_postfusion", legacy_input_vtk, {})
    write_df_to_regional_vtk(legacy_input_df, "fracture_only_step7_step7b_fused_raw_time", fracture_only_vtk, {})
    return {
        "mode": "single_input_fracture_only",
        "input_predicted_dfn_csv": str(input_csv),
        "fracture_only_vtk": str(fracture_only_vtk),
        "legacy_input_csv": str(legacy_input_csv),
        "legacy_input_vtk": str(legacy_input_vtk),
        "fracture_only_patch_count": int(len(legacy_input_df)),
    }


def infer_fault_cell_range(fault_summary_csv: Path, config: dict[str, Any]) -> dict[str, int]:
    fault = read_csv_flexible(fault_summary_csv, low_memory=False)
    target = dict(config.get("target_block", {}))
    margin = float(config.get("fault_target_margin_m", 0.0))
    x_min = float(target.get("x_min", -np.inf)) - margin
    x_max = float(target.get("x_max", np.inf)) + margin
    y_min = float(target.get("y_min", -np.inf)) - margin
    y_max = float(target.get("y_max", np.inf)) + margin
    overlap = (
        fault["bbox_xmax"].ge(x_min)
        & fault["bbox_xmin"].le(x_max)
        & fault["bbox_ymax"].ge(y_min)
        & fault["bbox_ymin"].le(y_max)
    )
    selected = fault.loc[overlap].copy()
    if selected.empty:
        raise ValueError("no fault patch overlaps target block; cannot build fault panels")
    return {
        "block_x_start": int(selected["cell_i"].min()),
        "block_x_end": int(selected["cell_i"].max()),
        "block_y_start": int(selected["cell_j"].min()),
        "block_y_end": int(selected["cell_j"].max()),
        "overlap_patch_count": int(len(selected)),
        "overlap_fault_count": int(selected["fault_name"].nunique()),
    }


def old_postfusion_to_step7c_df(fracture_csv: Path, surface_csv: Path | None, config: dict[str, Any]) -> pd.DataFrame:
    old = read_csv_flexible(fracture_csv, low_memory=False)
    rows: list[dict[str, Any]] = []
    for idx, row in old.iterrows():
        origin_code = int(pd.to_numeric(row.get("PatchOriginCode", 0), errors="coerce") if pd.notna(row.get("PatchOriginCode", 0)) else 0)
        action_code = int(pd.to_numeric(row.get("FaultActionCode", 0), errors="coerce") if pd.notna(row.get("FaultActionCode", 0)) else 0)
        layer_code = int(pd.to_numeric(row.get("LayerCode", 0), errors="coerce") if pd.notna(row.get("LayerCode", 0)) else 0)
        layer_group = "沙三段" if layer_code == 3 else "沙四段" if layer_code == 4 else str(row.get("LayerGroup", ""))
        source_type = PATCH_ORIGIN_CODE_TO_SOURCE.get(origin_code, "density_3d_predicted")
        if action_code == 2:
            relation = "fault_damage_zone"
            constraint_level = "guided_by_fault"
        else:
            relation = PATCH_ORIGIN_CODE_TO_RELATION.get(origin_code, "background")
            constraint_level = "guided_by_fault" if origin_code in (1, 2) else "predicted"
        out_row = {
                "PatchID": f"fault_postfusion_{idx + 1:06d}",
                "OriginalPatchID": str(row.get("PatchID", f"old_{idx + 1:06d}")),
                "GenerationStage": "fault_postfusion_real_patch_panel",
                "InputSource": "step7_step7b_fracture_only_fusion",
                "FusionSource": "real_fault_patch_panel_postfusion",
                "SourceType": source_type,
                "ConstraintLevel": constraint_level,
                "FaultRelation": relation if action_code != 0 else str(row.get("FaultRelation", relation)),
                "FaultAction": FAULT_ACTION_CODE_TO_TEXT.get(action_code, "keep"),
                "FaultName": str(row.get("FaultName", row.get("NearestFaultName", ""))) if pd.notna(row.get("FaultName", "")) else "",
                "NearestFaultName": str(row.get("NearestFaultName", "")) if pd.notna(row.get("NearestFaultName", "")) else "",
                "NearestFaultPanelID": int(pd.to_numeric(row.get("NearestFaultPanelID", -1), errors="coerce") if pd.notna(row.get("NearestFaultPanelID", -1)) else -1),
                "LayerGroup": layer_group,
                "LayerCode": layer_code,
                "CenterX": float(row["CenterX"]),
                "CenterY": float(row["CenterY"]),
                "CenterTime": float(row["CenterTIME"]),
                "TimeWindowMin": np.nan,
                "TimeWindowMax": np.nan,
                "LayerThickness": np.nan,
                "SourceDensity": float(row.get("SourceDensity", 0.0)),
                "LengthM": float(row.get("PatchLength", 0.0)),
                "HeightTimeMs": float(row.get("PatchHeight", 0.0)),
                "HeightM": float(row.get("PatchHeight", 0.0)),
                "PatchAreaM2": float(row.get("PatchArea", 0.0)),
                "PatchEquivalentRadiusM": float(np.sqrt(max(float(row.get("PatchArea", 0.0)), 0.0) / np.pi)),
                "AzimuthDeg": float(row.get("Azimuth", 0.0)) % 180.0,
                "DipDeg": float(np.clip(row.get("Dip", 45.0), 1.0, 89.9)),
                "DistanceToFaultM": float(row.get("FaultDistanceMs", np.nan)) if pd.notna(row.get("FaultDistanceMs", np.nan)) else np.nan,
                "FaultInfluenceWeight": float(row.get("FaultInfluenceWeight", 0.0)),
                "CanModifyInStep8": 0 if origin_code == 3 else 1,
                "CanBeRemovedByDensityFilter": 0 if origin_code in (1, 2, 3) else 1,
                "CanBeJittered": 0 if origin_code == 3 else 1,
                "NeedsWellCorrection": 0 if origin_code == 3 else 1,
                "FractureScale": "major_fault" if origin_code == 3 else str(row.get("ScaleClass", "")),
                "FractureScaleCode": 4 if origin_code == 3 else int(layer_code > 0),
            }
        for vertex_idx in range(1, 5):
            for axis in ("X", "Y", "Z"):
                col = f"V{vertex_idx}{axis}"
                if col in row.index and pd.notna(row.get(col)):
                    out_row[col] = float(row[col])
        rows.append(out_row)

    if surface_csv is not None and Path(surface_csv).exists():
        surface = read_csv_flexible(Path(surface_csv), low_memory=False)
        for idx, row in surface.iterrows():
            out_row = {
                    "PatchID": f"fault_surface_{idx + 1:06d}",
                    "OriginalPatchID": f"fault_surface_{idx + 1:06d}",
                    "GenerationStage": "fault_surface_real_patch_panel",
                    "InputSource": "geologist_fault_interpretation",
                    "FusionSource": "real_fault_surface_fragment",
                    "SourceType": "fault_surface",
                    "ConstraintLevel": "hard",
                    "FaultRelation": "fault_core",
                    "FaultAction": "surface",
                    "FaultName": str(row.get("FaultName", "")),
                    "NearestFaultName": str(row.get("FaultName", "")),
                    "NearestFaultPanelID": -1,
                    "LayerGroup": "沙四段",
                    "LayerCode": 4,
                    "CenterX": float(row["CenterX"]),
                    "CenterY": float(row["CenterY"]),
                    "CenterTime": float(row["CenterTIME"]),
                    "TimeWindowMin": np.nan,
                    "TimeWindowMax": np.nan,
                    "LayerThickness": np.nan,
                    "SourceDensity": float(config.get("fault_source_density", 999.0)),
                    "LengthM": float(row.get("PatchLength", 0.0)),
                    "HeightTimeMs": float(row.get("PatchHeight", 0.0)),
                    "HeightM": float(row.get("PatchHeight", 0.0)),
                    "PatchAreaM2": float(row.get("PatchArea", 0.0)),
                    "PatchEquivalentRadiusM": float(np.sqrt(max(float(row.get("PatchArea", 0.0)), 0.0) / np.pi)),
                    "AzimuthDeg": float(row.get("Azimuth", 0.0)) % 180.0,
                    "DipDeg": float(np.clip(row.get("Dip", 45.0), 1.0, 89.9)),
                    "DistanceToFaultM": 0.0,
                    "FaultInfluenceWeight": 1.0,
                    "CanModifyInStep8": 0,
                    "CanBeRemovedByDensityFilter": 0,
                    "CanBeJittered": 0,
                    "NeedsWellCorrection": 0,
                    "FractureScale": "major_fault",
                    "FractureScaleCode": 4,
                }
            for vertex_idx in range(1, 5):
                for axis in ("X", "Y", "Z"):
                    col = f"V{vertex_idx}{axis}"
                    if col in row.index and pd.notna(row.get(col)):
                        out_row[col] = float(row[col])
            rows.append(out_row)
    out = pd.DataFrame(rows)
    out["PatchID"] = [f"fault_postfusion_dfn_{idx + 1:06d}" for idx in range(len(out))]
    return out


def write_step7c_compatible_vtk(df: pd.DataFrame, output_vtk: Path, config: dict[str, Any]) -> None:
    time_scale = float(config.get("geometry_time_scale_m_per_ms", config.get("time_scale_m_per_ms", 1.0)))
    rows = []
    for _, row in df.iterrows():
        vertex_cols = [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
        if all(col in row.index and pd.notna(row.get(col)) for col in vertex_cols):
            vertices = np.asarray(
                [
                    [float(row[f"V{vertex_idx}X"]), float(row[f"V{vertex_idx}Y"]), float(row[f"V{vertex_idx}Z"])]
                    for vertex_idx in range(1, 5)
                ],
                dtype=float,
            )
            geom = {
                "CenterX": float(vertices[:, 0].mean()),
                "CenterY": float(vertices[:, 1].mean()),
                "CenterTIME": float(vertices[:, 2].mean()),
                "PatchLength": float(row.get("LengthM", np.linalg.norm(vertices[1] - vertices[0]))),
                "PatchHeight": float(row.get("HeightTimeMs", np.linalg.norm(vertices[3] - vertices[0]))),
                "PatchArea": float(row.get("PatchAreaM2", 0.0)),
                "Azimuth": float(row.get("AzimuthDeg", 0.0)),
                "Dip": float(row.get("DipDeg", 0.0)),
                "NormalX": 0.0,
                "NormalY": 0.0,
                "NormalZ": 1.0,
                "BBoxXMin": float(vertices[:, 0].min()),
                "BBoxXMax": float(vertices[:, 0].max()),
                "BBoxYMin": float(vertices[:, 1].min()),
                "BBoxYMax": float(vertices[:, 1].max()),
                "BBoxZMin": float(vertices[:, 2].min()),
                "BBoxZMax": float(vertices[:, 2].max()),
            }
            normal = np.cross(vertices[1] - vertices[0], vertices[3] - vertices[0])
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 1.0e-12:
                normal = normal / normal_norm
                geom["NormalX"] = float(normal[0])
                geom["NormalY"] = float(normal[1])
                geom["NormalZ"] = float(normal[2])
            for vertex_idx in range(1, 5):
                geom[f"V{vertex_idx}X"] = float(vertices[vertex_idx - 1, 0])
                geom[f"V{vertex_idx}Y"] = float(vertices[vertex_idx - 1, 1])
                geom[f"V{vertex_idx}Z"] = float(vertices[vertex_idx - 1, 2])
        else:
            geom = compute_patch_geometry(row, time_scale_m_per_ms=time_scale)
        rows.append(
            {
                **geom,
                "LayerCode": int(row.get("LayerCode", 0)),
                "SourceDensity": float(row.get("SourceDensity", 0.0)),
                "SourceDensityRender": float(row.get("SourceDensityRender", row.get("SourceDensity", 0.0))),
                "SourceDensityRenderNorm": float(row.get("SourceDensityRenderNorm", 0.0)),
                "SourceDensityRenderClipMax": float(row.get("SourceDensityRenderClipMax", 0.0)),
                "PatchArea": float(row.get("PatchAreaM2", geom["PatchArea"])),
                "FaultInfluenceWeight": float(row.get("FaultInfluenceWeight", 0.0)),
                "DistanceToFaultM": float(row.get("DistanceToFaultM", -1.0)) if pd.notna(row.get("DistanceToFaultM", np.nan)) else -1.0,
                "SourceTypeCode": {"density_3d_predicted": 1, "fault_parallel": 2, "fault_perpendicular": 3, "fault_surface": 4}.get(str(row.get("SourceType", "")), 0),
                "ConstraintLevelCode": {"predicted": 1, "guided_by_fault": 2, "hard": 3}.get(str(row.get("ConstraintLevel", "")), 0),
                "FaultRelationCode": {"background": 1, "fault_damage_zone": 2, "fault_core": 3}.get(str(row.get("FaultRelation", "")), 0),
                "CanModifyInStep8": int(row.get("CanModifyInStep8", 1)),
            }
        )
    vtk_df = pd.DataFrame(rows)
    write_df_to_regional_vtk(vtk_df, "fused_initial_dfn_real_fault_postfusion_raw_time", output_vtk, {})


def write_fault_only_and_influence_vtk(df: pd.DataFrame, output_vtk: Path, config: dict[str, Any]) -> dict[str, Any]:
    fault_mask = (
        df["FaultRelation"].astype(str).ne("background")
        | df["FaultAction"].astype(str).isin(["shrink", "induced", "surface"])
        | df["SourceType"].astype(str).isin(["fault_parallel", "fault_perpendicular", "fault_surface"])
    )
    fault_df = df.loc[fault_mask].copy().reset_index(drop=True)
    write_step7c_compatible_vtk(fault_df, output_vtk, config=config)
    return {
        "fault_only_and_influence_vtk": str(output_vtk),
        "fault_only_and_influence_patch_count": int(len(fault_df)),
        "fault_only_and_influence_source_type_distribution": {
            str(k): int(v) for k, v in fault_df["SourceType"].value_counts(dropna=False).items()
        },
        "fault_only_and_influence_action_distribution": {
            str(k): int(v) for k, v in fault_df["FaultAction"].value_counts(dropna=False).items()
        },
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / "postfusion_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    fault_summary_csv = Path(config["fault_patches_summary_csv"]).resolve()
    fault_patches_root = Path(config["fault_patches_root"]).resolve()
    for label, path in [("fault_patches_summary_csv", fault_summary_csv), ("fault_patches_root", fault_patches_root)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if "input_2d_dfn_csv" in config and "input_3d_dfn_csv" in config:
        fracture_only_summary = build_fracture_only_fusion(config=config, workspace=workspace, output_dir=output_dir)
    else:
        fracture_only_summary = build_single_input_fracture_only(config=config, workspace=workspace, output_dir=output_dir)
    legacy_input_csv = Path(fracture_only_summary["legacy_input_csv"])
    legacy_input_vtk = Path(fracture_only_summary["legacy_input_vtk"])

    cell_range = infer_fault_cell_range(fault_summary_csv, config=config)
    print(f"[step7c-postfusion] selected fault cell range: {cell_range}", flush=True)

    panel_summary = run_build_regional_fault_panels(
        fault_patches_root=fault_patches_root,
        block_x_start=cell_range["block_x_start"],
        block_x_end=cell_range["block_x_end"],
        block_y_start=cell_range["block_y_start"],
        block_y_end=cell_range["block_y_end"],
        output_root=workspace,
        run_name="regional_fault_panels",
        panel_merge_xy=float(config.get("panel_merge_xy", 220.0)),
        panel_merge_time=float(config.get("panel_merge_time", 35.0)),
        panel_strike_tol=float(config.get("panel_strike_tol", 20.0)),
        panel_dip_tol=float(config.get("panel_dip_tol", 15.0)),
    )
    surface_summary = run_build_fault_surface_fragments(
        fault_patches_root=fault_patches_root,
        block_x_start=cell_range["block_x_start"],
        block_x_end=cell_range["block_x_end"],
        block_y_start=cell_range["block_y_start"],
        block_y_end=cell_range["block_y_end"],
        output_root=workspace,
        run_name="fault_surface_fragments",
        surface_max_strike=float(config.get("surface_max_strike", 70.0)),
        surface_max_dip=float(config.get("surface_max_dip", 18.0)),
        surface_min_fragment_area=float(config.get("surface_min_fragment_area", 1.0)),
        surface_elongate_ratio=float(config.get("surface_elongate_ratio", 2.8)),
        surface_normal_pad=float(config.get("surface_normal_pad", 30.0)),
        surface_gap_ratio=float(config.get("surface_gap_ratio", 0.95)),
        surface_display_offset_ms=float(config.get("surface_display_offset_ms", 0.6)),
        surface_max_fragment_area_ratio=float(config.get("surface_max_fragment_area_ratio", 1500.0)),
    )
    postfusion_summary = run_fault_postfusion(
        input_vtk=legacy_input_vtk,
        fault_panel_csv=Path(panel_summary["panel_csv"]),
        fault_surface_vtk=Path(surface_summary["surface_vtk"]),
        output_root=workspace,
        run_name="fault_postfusion",
        fault_half_band_ms=float(config.get("fault_half_band_ms", 100.0)),
        fault_remove_ms=float(config.get("fault_remove_ms", 50.0)),
        fault_transition_ms=float(config.get("fault_transition_ms", 100.0)),
        fault_induced_count_scale=float(config.get("fault_induced_count_scale", 5.5)),
        panel_xy_buffer=float(config.get("panel_xy_buffer", 180.0)),
        parallel_ratio=float(config.get("parallel_ratio", 0.65)),
        random_seed=int(config.get("random_seed", 42)),
    )

    print("[step7c-postfusion] exporting Step8/Step9 compatible outputs", flush=True)
    fused_csv = output_dir / "fused_initial_dfn_fracture_patches.csv"
    fused_vtk = output_dir / "fused_initial_dfn_raw_time.vtk"
    fault_only_vtk = output_dir / "fault_only_and_influence_raw_time.vtk"
    summary_json = output_dir / "fused_initial_dfn_summary.json"
    surface_csv = Path(surface_summary["summary_csv"])
    compatible_df = old_postfusion_to_step7c_df(
        fracture_csv=Path(postfusion_summary["output_csv"]),
        surface_csv=surface_csv,
        config=config,
    )
    compatible_df = add_render_density_columns(compatible_df)
    compatible_df.to_csv(fused_csv, index=False, encoding="utf-8-sig")
    write_step7c_compatible_vtk(compatible_df, fused_vtk, config=config)
    fault_only_summary = write_fault_only_and_influence_vtk(compatible_df, fault_only_vtk, config=config)

    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "generation_logic": "step7_2d_step7b_3d_fracture_only_fusion_plus_real_fault_patch_panel_postfusion",
        "fracture_only_summary": fracture_only_summary,
        "fault_patches_root": str(fault_patches_root),
        "fault_cell_range": cell_range,
        "legacy_input_csv": str(legacy_input_csv),
        "legacy_input_vtk": str(legacy_input_vtk),
        "panel_summary": panel_summary,
        "surface_summary": surface_summary,
        "postfusion_summary": postfusion_summary,
        "fused_initial_dfn_csv": str(fused_csv),
        "fused_initial_dfn_raw_vtk": str(fused_vtk),
        "fault_only_summary": fault_only_summary,
        "compatible_patch_count": int(len(compatible_df)),
        "source_type_distribution": {str(k): int(v) for k, v in compatible_df["SourceType"].value_counts(dropna=False).items()},
        "constraint_level_distribution": {str(k): int(v) for k, v in compatible_df["ConstraintLevel"].value_counts(dropna=False).items()},
        "fault_relation_distribution": {str(k): int(v) for k, v in compatible_df["FaultRelation"].value_counts(dropna=False).items()},
        "length_m_stats": finite_stats(compatible_df["LengthM"]),
        "height_time_ms_stats": finite_stats(compatible_df["HeightTimeMs"]),
        "patch_area_m2_stats": finite_stats(compatible_df["PatchAreaM2"]),
    }
    write_json(summary_json, summary)
    print(f"Fused DFN CSV: {fused_csv}", flush=True)
    print(f"Fused DFN raw VTK: {fused_vtk}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    print(f"Patch count: {len(compatible_df)} source_type={summary['source_type_distribution']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
