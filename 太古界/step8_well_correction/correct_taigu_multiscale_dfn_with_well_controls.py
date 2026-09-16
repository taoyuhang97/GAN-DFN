from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
TAIGU_ROOT = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[1]
DEFAULT_CONFIG = CURRENT_DIR / "configs/taigu_step8_attribute_multiscale_v1.json"

_vtk_path = REPO_ROOT / "优化阶段二" / "正式主线" / "common" / "unified_dfn_vtk.py"
_vtk_spec = importlib.util.spec_from_file_location("taigu_step8_unified_vtk", _vtk_path)
if _vtk_spec is None or _vtk_spec.loader is None:
    raise ImportError(f"cannot load unified VTK helper: {_vtk_path}")
_vtk = importlib.util.module_from_spec(_vtk_spec)
_vtk_spec.loader.exec_module(_vtk)
ORIGINAL_FAULT_GEOMETRY_GROUP_CODE = _vtk.ORIGINAL_FAULT_GEOMETRY_GROUP_CODE
extract_geometry_group = _vtk.extract_geometry_group
geometry_fingerprint = _vtk.geometry_fingerprint
unified_vtk_summary = _vtk.unified_vtk_summary
write_unified_dfn_vtk = _vtk.write_unified_dfn_vtk

ALLOWED_LAYERS = ["上部复合层", "太古界风化壳"]
LAYER_CODE = {"上部复合层": 1, "太古界风化壳": 2}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


class TaiguHorizonSpatialLookup:
    """Nearest-trace lookup for the Top/Mid/Base Taigu horizon contract."""

    def __init__(self, table: pd.DataFrame) -> None:
        required = {"TraceIdx", "X", "Y", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Taigu horizon contract missing columns: {missing}")
        self.table = table.reset_index(drop=True).copy()
        for column in ["TraceIdx", "X", "Y", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"]:
            self.table[column] = pd.to_numeric(self.table[column], errors="coerce")
        self.tree = cKDTree(self.table[["X", "Y"]].to_numpy(dtype=float))

    def query(self, x: float, y: float) -> dict[str, Any]:
        distance, position = self.tree.query(np.asarray([[x, y]], dtype=float), k=1)
        row = self.table.iloc[int(position[0])]
        top = float(row["TopTimeMs"])
        mid = float(row["MidTimeMs"])
        base = float(row["BaseTimeMs"])
        surface_valid = bool(int(row["SurfaceValid"]))
        return {
            "TraceIdx": int(row["TraceIdx"]),
            "X": float(row["X"]),
            "Y": float(row["Y"]),
            "DistanceM": float(distance[0]),
            "TopTimeMs": top,
            "MidTimeMs": mid,
            "BaseTimeMs": base,
            "UpperPresent": bool(surface_valid and np.isfinite(top) and np.isfinite(mid) and top < mid),
            "CrustPresent": bool(surface_valid and np.isfinite(mid) and np.isfinite(base) and mid < base),
        }

    def interval(self, x: float, y: float, time_ms: float) -> str | None:
        local = self.query(x, y)
        if local["UpperPresent"] and local["TopTimeMs"] <= time_ms <= local["MidTimeMs"]:
            return "Top->Mid"
        if local["CrustPresent"] and local["MidTimeMs"] <= time_ms <= local["BaseTimeMs"]:
            return "Mid->Base"
        return None


def build_spatial_lookup(config: dict[str, Any]) -> TaiguHorizonSpatialLookup:
    path = Path(config["horizon_contract_table"]).resolve()
    return TaiguHorizonSpatialLookup(read_csv_flexible(path, low_memory=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct the Taigu multiscale DFN with real-well controls.")
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


def patch_geometry_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["PatchID", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg"]
    canonical = frame[columns].sort_values("PatchID").to_csv(index=False, float_format="%.10g")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def robust_unit_score(values: pd.Series) -> pd.Series:
    numeric = safe_numeric(values)
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low = float(valid.quantile(0.05))
    high = float(valid.quantile(0.95))
    if high <= low:
        return pd.Series(0.5, index=values.index, dtype=float).where(numeric.notna(), np.nan)
    return ((numeric - low) / (high - low)).clip(0.0, 1.0)


def add_density_scores(patch_df: pd.DataFrame, control_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patches = patch_df.copy()
    controls = control_df.copy()
    patches["RawDensity"] = safe_numeric(patches.get("Density", patches.get("SourceDensity")))
    patches["FractureIntensityScore"] = safe_numeric(patches.get("SourceDensityRenderNorm"))
    for layer, group in patches.groupby("LayerGroup", dropna=False):
        small_idx = group.index[group["FractureScale"].astype(str).str.lower().eq("small")]
        if len(small_idx):
            patches.loc[small_idx, "FractureIntensityScore"] = robust_unit_score(
                patches.loc[small_idx, "RawDensity"]
            )
    fallback = safe_numeric(patches.get("SourceDensity")).clip(0.0, 1.0)
    patches["FractureIntensityScore"] = patches["FractureIntensityScore"].fillna(fallback).fillna(0.5)
    small = patches["FractureScale"].astype(str).str.lower().eq("small")
    patches.loc[small, "SourceDensity"] = patches.loc[small, "FractureIntensityScore"]
    patches.loc[small, "SourceDensityRender"] = patches.loc[small, "FractureIntensityScore"]
    patches.loc[small, "SourceDensityRenderNorm"] = patches.loc[small, "FractureIntensityScore"]

    controls["RawControlDensity"] = safe_numeric(controls.get("Density"))
    controls["DensityScore"] = np.nan
    source_family = np.where(
        safe_numeric(controls.get("IsImagingGroundTruth", pd.Series(0, index=controls.index))).fillna(0).astype(int).eq(1),
        "step3_imaging",
        "step4_prediction",
    )
    controls["ControlSourceFamily"] = source_family
    for _, group in controls.groupby(["LayerGroup", "ControlSourceFamily"], dropna=False):
        controls.loc[group.index, "DensityScore"] = robust_unit_score(group["RawControlDensity"])
    controls["DensityScore"] = safe_numeric(controls["DensityScore"]).fillna(0.5).clip(0.0, 1.0)
    return patches, controls


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "corrected_csv": output_dir / "well_corrected_dfn_fracture_patches.csv",
        "raw_vtk": output_dir / "well_corrected_dfn_raw_time.vtk",
        "summary_json": output_dir / "well_corrected_dfn_summary.json",
        "audit_csv": output_dir / "well_control_correction_audit.csv",
        "region_audit_csv": output_dir / "imaging_well_region_correction_audit.csv",
        "input_qc_csv": output_dir / "well_control_input_qc.csv",
        "outside_layer_audit_csv": output_dir / "well_control_outside_layer_audit.csv",
        "md_join_audit_csv": output_dir / "well_control_md_join_audit.csv",
        "md_join_unmatched_csv": output_dir / "well_control_md_join_unmatched.csv",
        "segment_coverage_csv": output_dir / "step2_segment_coverage_audit.csv",
    }


def load_initial_dfn(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path, low_memory=False)
    required = ["PatchID", "LayerGroup", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"initial DFN missing required columns: {missing}")
    out = df.copy()
    for column in ["CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg", "SourceDensity", "Density"]:
        if column in out.columns:
            out[column] = safe_numeric(out[column])
    if "SourceDensity" not in out.columns:
        out["SourceDensity"] = np.nan
    small = out.get("FractureScale", pd.Series("", index=out.index)).astype(str).str.lower().eq("small")
    if "Density" in out.columns:
        out.loc[small & out["SourceDensity"].isna(), "SourceDensity"] = out.loc[
            small & out["SourceDensity"].isna(), "Density"
        ]
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


def _inside_target(frame: pd.DataFrame, target_block: dict[str, Any]) -> pd.Series:
    return (
        frame["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
        & frame["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
    )


# ---- 公共测井段回接模块（P0-4）：见 太古界/common/well_segment_join/README.md ----
if str(TAIGU_ROOT) not in sys.path:
    sys.path.insert(0, str(TAIGU_ROOT))

from common.well_segment_join import (  # noqa: E402
    MD_JOIN_AUDIT_COLUMNS,
    WellSegmentPool,
    attach_geometry_per_segment,
    cached_well_segment_pool,
    list_segment_files as list_well_segment_files,
    summarize_join,
)


def build_control_sample_ids(prefix: str, frame: pd.DataFrame) -> list[str]:
    """Content-derived control IDs.

    Step8 draws the deterministic well-control centre offset from a hash of the
    sample ID.  Deriving that ID from the sample position would make the DFN
    geometry depend on upstream row order, so the ID is built from well plus
    depth instead and duplicates are disambiguated by depth order.
    """
    count = len(frame)
    depth = safe_numeric(frame["TVD"]).to_numpy(dtype=float) if "TVD" in frame.columns else np.full(count, np.nan)
    if "MD" in frame.columns:
        depth = np.where(np.isfinite(depth), depth, safe_numeric(frame["MD"]).to_numpy(dtype=float))
    wells = frame["WellName"].astype(str).to_numpy() if "WellName" in frame.columns else np.array([""] * count)
    ids = [
        f"{prefix}_{well}_{value:.6f}" if np.isfinite(value) else f"{prefix}_{well}_nodepth"
        for well, value in zip(wells, depth)
    ]
    groups: dict[str, list[int]] = {}
    for position, value in enumerate(ids):
        groups.setdefault(value, []).append(position)
    duplicated = {value: items for value, items in groups.items() if len(items) > 1}
    if duplicated:
        order_columns = [column for column in ["MD", "TVD", "X", "Y"] if column in frame.columns]
        order = pd.DataFrame(
            {column: safe_numeric(frame[column]).to_numpy(dtype=float) for column in order_columns}
        )
        order["_position"] = np.arange(count)
        order = order.sort_values(order_columns + ["_position"], kind="mergesort")
        ranked = order["_position"].tolist()
        for value, items in duplicated.items():
            members = set(items)
            sequence = [position for position in ranked if position in members]
            for rank, position in enumerate(sequence, start=1):
                ids[position] = f"{value}_dup{rank}"
    return ids


def load_control_points(
    path: Path,
    target_block: dict[str, Any],
    merged_density_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 Step4 的井级裂缝点（P0-4′）。

    Step4 的输出以"井"为单位，其合并密度曲线与裂缝点都直接携带 `X/Y/TIME`，
    因此这里**不需要任何按 MD 的段回接**：直接取坐标即可。
    若点表仍是旧版（缺坐标），则用 Step4 的合并密度曲线按 `WellName+TVD`
    做**精确主键连接**补齐（零容差、零歧义），并在审计里标明来源。
    """
    df = read_csv_flexible(path, low_memory=False)
    required = ["WellName", "MD", "StrataName", "PredDensity"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"fracture points missing required columns: {missing}")
    df = df.copy()
    df["MD"] = safe_numeric(df["MD"])
    df["TVD"] = safe_numeric(df["TVD"]) if "TVD" in df.columns else np.nan
    has_geometry = all(column in df.columns for column in ("X", "Y", "TIME"))
    if has_geometry:
        geometry_source = "step4_merged_point_output"
        df["X"] = safe_numeric(df["X"])
        df["Y"] = safe_numeric(df["Y"])
        df["TIME"] = safe_numeric(df["TIME"])
    else:
        if merged_density_csv is None or not Path(merged_density_csv).exists():
            raise ValueError(
                "Step4 fracture points carry no X/Y/TIME and no merged density curve is configured "
                "for the exact-key fallback; rebuild Step4 with P0-4' applied"
            )
        geometry_source = "step4_merged_curve_exact_key_join"
        curve = read_csv_flexible(Path(merged_density_csv), low_memory=False)
        curve = curve[["WellName", "TVD", "X", "Y", "TIME"]].copy()
        for column in ("TVD", "X", "Y", "TIME"):
            curve[column] = safe_numeric(curve[column])
        df = df.merge(curve, on=["WellName", "TVD"], how="left", validate="one_to_one")
    audit_df = pd.DataFrame(
        {
            "WellName": df["WellName"].astype(str),
            "MD": df["MD"],
            "TVD": df["TVD"],
            "StrataName": df["StrataName"],
            "X": df["X"],
            "Y": df["Y"],
            "MDJoinStatus": np.where(
                np.isfinite(safe_numeric(df["X"])) & np.isfinite(safe_numeric(df["Y"])) & np.isfinite(safe_numeric(df["TIME"])),
                "matched",
                "unmatched",
            ),
            "MDJoinReason": geometry_source,
            "GeometrySource": geometry_source,
            "SegmentSelectionPolicy": "not_applicable_well_level_output",
            "MDNearestDifferenceM": np.nan,
            "MDJoinDuplicateRow": 0,
        }
    )
    matched = df
    if not audit_df.empty:
        audit_df["InTargetBlock"] = _inside_target(audit_df, target_block).astype(int)
        audit_df["InAllowedLayer"] = audit_df["StrataName"].astype(str).isin(ALLOWED_LAYERS).astype(int)
        audit_df["UsedForStep8"] = (
            audit_df["MDJoinStatus"].astype(str).eq("matched")
            & audit_df["InTargetBlock"].eq(1)
            & audit_df["InAllowedLayer"].eq(1)
            & safe_numeric(audit_df["MDJoinDuplicateRow"]).fillna(0).astype(int).eq(0)
        ).astype(int)
        audit_df["ExcludedFromStep8"] = 1 - audit_df["UsedForStep8"]
    if matched.empty:
        return pd.DataFrame(), audit_df
    out = matched.copy()
    out["LayerGroup"] = out["StrataName"].astype(str)
    out["Density"] = safe_numeric(out["PredDensity"])
    out["DEPT"] = safe_numeric(out["MD"])
    out = out.dropna(subset=["X", "Y", "TIME"])
    out = out[out["LayerGroup"].isin(ALLOWED_LAYERS) & _inside_target(out, target_block)].copy()
    out["WellControlSampleID"] = build_control_sample_ids("step4", out)
    out = out.sort_values(["WellName", "LayerGroup", "TIME", "MD"]).reset_index(drop=True)
    out["ControlPointID"] = ["ctrl_%06d" % (idx + 1) for idx in range(len(out))]
    out["ControlSource"] = "step4_predicted"
    out["IsImagingGroundTruth"] = 0
    out["temporary_neighbor_time_depth"] = 0
    return out, audit_df


def load_step3_imaging_controls(
    paths: list[Path],
    samples_root: Path,
    target_block: dict[str, Any],
    tolerance: float,
    tolerance_options: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step3 成像监督组（逐段行）在其所属测井段内回接坐标。"""
    parts: list[pd.DataFrame] = []
    for path in paths:
        df = read_csv_flexible(path, low_memory=False)
        required = {"WellName", "InputSegmentPath", "MD", "StrataName", "GT_POINT_FLAG"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Step3 imaging group csv missing columns {missing}: {path}")
        work = df[pd.to_numeric(df["GT_POINT_FLAG"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
        if work.empty:
            continue
        work["MD"] = safe_numeric(work["MD"])
        work["TVD"] = safe_numeric(work["TVD"]) if "TVD" in work.columns else np.nan
        work["Step3GroupCSV"] = str(path)
        parts.append(work)
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    work = pd.concat(parts, ignore_index=True)
    matched, audit_df = attach_geometry_per_segment(
        work,
        preferred_column="InputSegmentPath",
        tolerance_m=tolerance,
        geometry_source="step3_imaging_gt",
        samples_root=samples_root,
        **(tolerance_options or {}),
    )
    if not audit_df.empty:
        audit_df["InTargetBlock"] = _inside_target(audit_df, target_block).astype(int)
        audit_df["InAllowedLayer"] = audit_df["StrataName"].astype(str).isin(ALLOWED_LAYERS).astype(int)
        audit_df["UsedForStep8"] = (
            audit_df["MDJoinStatus"].astype(str).eq("matched")
            & audit_df["InTargetBlock"].eq(1)
            & audit_df["InAllowedLayer"].eq(1)
        ).astype(int)
        audit_df["ExcludedFromStep8"] = 1 - audit_df["UsedForStep8"]
    if matched.empty:
        return pd.DataFrame(), audit_df
    work = matched.rename(columns={"FracAzimuth": "Frac_Azimuth", "FracDip": "Frac_Dip"})
    for column in ["X", "Y", "TIME", "TVD", "DEPT", "Density", "Frac_Azimuth", "Frac_Dip"]:
        if column in work.columns:
            work[column] = safe_numeric(work[column])
    work["DEPT"] = safe_numeric(work["MD"])
    work = work.dropna(subset=["X", "Y", "TIME"]).copy()
    work = work[_inside_target(work, target_block)].copy()
    work["LayerGroup"] = work["StrataName"].astype(str)
    work = work[work["LayerGroup"].isin(ALLOWED_LAYERS)].copy()
    if work.empty:
        return pd.DataFrame(), audit_df
    work["ControlSource"] = "step3_imaging_gt"
    work["IsImagingGroundTruth"] = 1
    work["temporary_neighbor_time_depth"] = 1
    work["WellControlSampleID"] = build_control_sample_ids("step3_gt", work)
    keep = [
        "WellName",
        "X",
        "Y",
        "TIME",
        "TVD",
        "DEPT",
        "LayerGroup",
        "StrataName",
        "Density",
        "WellControlSampleID",
        "ControlSource",
        "IsImagingGroundTruth",
        "Frac_Azimuth",
        "Frac_Dip",
        "Step3GroupCSV",
        "temporary_neighbor_time_depth",
        "MD",
        "TVD_Step2",
        "StrataName_Step2",
        "InputSegmentPath",
    ] + MD_JOIN_AUDIT_COLUMNS
    keep = [column for column in keep if column in work.columns]
    return work[keep].sort_values(["WellName", "LayerGroup", "TIME", "MD"]).reset_index(drop=True), audit_df


def load_step3_imaging_time_windows(controls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (well, layer), work in controls.groupby(["WellName", "LayerGroup"], dropna=False):
        rows.append(
            {
                "WellName": str(well),
                "LayerGroup": str(layer),
                "TimeMin": float(work["TIME"].min()),
                "TimeMax": float(work["TIME"].max()),
                "Step3GroupCSV": "|".join(sorted(work["Step3GroupCSV"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def build_segment_coverage_audit(
    samples_root: Path,
    wells: list[str],
    pools: dict[str, WellSegmentPool | None],
    join_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Per-segment coverage of every Taigu well: ranges, overlaps, gaps, usage."""
    manifest: pd.DataFrame | None = None
    manifest_path = samples_root / "taigu_step2_segment_manifest.csv"
    if manifest_path.exists():
        manifest = read_csv_flexible(manifest_path, low_memory=False)
    selected_counts: dict[str, int] = {}
    selected_by_source: dict[str, dict[str, int]] = {}
    if not join_audit.empty and "SelectedSegmentID" in join_audit.columns:
        selected = join_audit["SelectedSegmentID"].astype(str)
        usable = join_audit.loc[selected.ne("") & selected.ne("nan")].copy()
        selected_counts = usable["SelectedSegmentID"].astype(str).value_counts().to_dict()
        for source, group in usable.groupby("GeometrySource", dropna=False):
            selected_by_source[str(source)] = group["SelectedSegmentID"].astype(str).value_counts().to_dict()
    rows: list[dict[str, Any]] = []
    for well in wells:
        pool = pools.get(well)
        if pool is None:
            rows.append(
                {
                    "WellName": well,
                    "SegmentID": "",
                    "SegmentStatus": "no_usable_segment_csv",
                    "WellSegmentCount": 0,
                }
            )
            continue
        table = pool.table
        for segment_id, group in table.groupby("SegmentID", dropna=False):
            segment_id = str(segment_id)
            mds = np.sort(group["MD"].to_numpy(dtype=float))
            steps = np.diff(mds)
            step_median = float(np.median(steps)) if steps.size else np.nan
            gaps = steps[steps > 2.0 * step_median] if steps.size and np.isfinite(step_median) else np.array([])
            others = table[table["SegmentID"].astype(str) != segment_id]
            overlap_count = 0
            if not others.empty:
                inside = np.zeros(len(mds), dtype=bool)
                for _, other in others.groupby("SegmentID", dropna=False):
                    inside |= (mds >= float(other["MD"].min())) & (mds <= float(other["MD"].max()))
                overlap_count = int(inside.sum())
            row: dict[str, Any] = {
                "WellName": well,
                "SegmentID": segment_id,
                "SegmentStatus": "usable",
                "WellSegmentCount": int(len(pool.segment_ids_unique)),
                "SegmentPath": str(group["SegmentPath"].iloc[0]) if "SegmentPath" in group.columns else "",
                "Rows": int(len(group)),
                "MDMin": float(mds.min()) if mds.size else np.nan,
                "MDMax": float(mds.max()) if mds.size else np.nan,
                "TVDMin": float(group["TVD"].min()) if "TVD" in group.columns else np.nan,
                "TVDMax": float(group["TVD"].max()) if "TVD" in group.columns else np.nan,
                "MDStepMedianM": step_median,
                "DuplicateMDCount": int(len(mds) - np.unique(mds).size),
                "GapCount": int(len(gaps)),
                "MaxGapM": float(gaps.max()) if len(gaps) else 0.0,
                "MDCoveredByOtherSegmentCount": overlap_count,
                "SelectedControlPointCount": int(selected_counts.get(segment_id, 0)),
                "SelectedStep4ControlPointCount": int(selected_by_source.get("step4_predicted", {}).get(segment_id, 0)),
                "SelectedStep3ControlPointCount": int(selected_by_source.get("step3_imaging_gt", {}).get(segment_id, 0)),
            }
            if manifest is not None and "SegmentID" in manifest.columns:
                match = manifest[manifest["SegmentID"].astype(str).eq(segment_id)]
                for column in ["LogDate", "SourcePath", "StrataNames", "TrajectorySource", "TimeDepthSource", "BorrowedFrom"]:
                    if column in match.columns:
                        row[column] = match.iloc[0][column] if not match.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def merge_step3_imaging_controls(step4_df: pd.DataFrame, step3_df: pd.DataFrame, windows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if step3_df.empty:
        out = step4_df.copy()
        out["ControlPointID"] = ["ctrl_%06d" % (idx + 1) for idx in range(len(out))]
        return out
    filtered = step4_df.copy()
    if bool(config.get("replace_step4_inside_step3_imaging_windows", True)) and not windows.empty:
        padding = float(config.get("step3_imaging_window_time_padding_ms", 1.0))
        keep = pd.Series(True, index=filtered.index)
        for _, window in windows.iterrows():
            mask = (
                filtered["WellName"].astype(str).eq(str(window["WellName"]))
                & filtered["LayerGroup"].astype(str).eq(str(window["LayerGroup"]))
                & filtered["TIME"].between(float(window["TimeMin"]) - padding, float(window["TimeMax"]) + padding)
            )
            keep.loc[mask] = False
        filtered = filtered.loc[keep].copy()
    out = pd.concat([filtered, step3_df], ignore_index=True, sort=False)
    out = out.sort_values(["WellName", "LayerGroup", "TIME", "WellControlSampleID"]).reset_index(drop=True)
    out["ControlPointID"] = ["ctrl_%06d" % (idx + 1) for idx in range(len(out))]
    return out


def aggregate_step4_controls_to_events(step4_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Convert dense Step4 point controls into well-scale fracture events.

    Step4 points are predicted from conventional logs and can be sampled much
    more densely than any reasonable DFN patch spacing. Aggregating adjacent
    points keeps the well constraint, but avoids drawing one patch per sample.
    """
    if step4_df.empty or not bool(config.get("enable_step4_event_aggregation", False)):
        return step4_df.copy()

    gap_ms = float(config.get("step4_event_gap_ms", 6.0))
    max_span_ms = float(config.get("step4_event_max_span_ms", 18.0))
    min_points = int(config.get("step4_event_min_points", 1))
    density_rule = str(config.get("step4_event_density_aggregation", "max")).lower()
    rows: list[dict[str, Any]] = []

    for (well, layer), group in step4_df.sort_values(["WellName", "LayerGroup", "TIME"]).groupby(
        ["WellName", "LayerGroup"], dropna=False
    ):
        current_indices: list[int] = []
        event_idx = 0
        previous_time: float | None = None
        event_start_time: float | None = None

        def flush_event(indices: list[int]) -> None:
            nonlocal event_idx
            if len(indices) < min_points:
                return
            event_idx += 1
            event = group.loc[indices].copy()
            density = safe_numeric(event["Density"]) if "Density" in event.columns else pd.Series(np.nan, index=event.index)
            weights = density.clip(lower=0.0)
            if not weights.notna().any() or float(weights.fillna(0.0).sum()) <= 0.0:
                weights = pd.Series(1.0, index=event.index)
            else:
                weights = weights.fillna(float(weights[weights > 0].median()) if (weights > 0).any() else 1.0)
            weights_np = weights.to_numpy(dtype=float)
            weights_np = weights_np / max(float(weights_np.sum()), 1.0e-12)

            row = event.iloc[0].to_dict()
            for column in ["X", "Y", "TIME", "TVD", "DEPT"]:
                if column in event.columns:
                    values = safe_numeric(event[column]).to_numpy(dtype=float)
                    valid = np.isfinite(values)
                    if valid.any():
                        local_weights = weights_np.copy()
                        local_weights[~valid] = 0.0
                        if float(local_weights.sum()) > 0.0:
                            local_weights = local_weights / float(local_weights.sum())
                            row[column] = float(np.sum(values[valid] * local_weights[valid]))
                        else:
                            row[column] = float(np.nanmedian(values[valid]))
            if "Density" in event.columns:
                event_density = safe_numeric(event["Density"]).dropna()
                if event_density.empty:
                    row["Density"] = np.nan
                elif density_rule == "mean":
                    row["Density"] = float(event_density.mean())
                elif density_rule.startswith("q"):
                    quantile = float(density_rule[1:]) / 100.0
                    row["Density"] = float(event_density.quantile(np.clip(quantile, 0.0, 1.0)))
                else:
                    row["Density"] = float(event_density.max())
            if "PredFractureProb" in event.columns:
                row["PredFractureProb"] = float(safe_numeric(event["PredFractureProb"]).max())
            event_time_min = float(safe_numeric(event["TIME"]).min())
            event_time_max = float(safe_numeric(event["TIME"]).max())
            safe_well = str(well).replace("/", "_").replace(" ", "")
            safe_layer = str(layer).replace("/", "_").replace(" ", "")
            row["WellControlSampleID"] = f"step4_event_{safe_well}_{safe_layer}_{event_idx:04d}"
            row["ControlSource"] = "step4_predicted_event"
            row["IsImagingGroundTruth"] = 0
            row["EventPointCount"] = int(len(event))
            row["EventTimeMin"] = event_time_min
            row["EventTimeMax"] = event_time_max
            row["EventTimeSpanMs"] = event_time_max - event_time_min
            row["EventDensityMax"] = float(safe_numeric(event["Density"]).max()) if "Density" in event.columns else np.nan
            row["EventDensityMean"] = float(safe_numeric(event["Density"]).mean()) if "Density" in event.columns else np.nan
            rows.append(row)

        for idx, row in group.iterrows():
            time_value = float(row["TIME"])
            if not current_indices:
                current_indices = [idx]
                previous_time = time_value
                event_start_time = time_value
                continue
            gap_break = previous_time is not None and (time_value - previous_time) > gap_ms
            span_break = (
                max_span_ms > 0.0
                and event_start_time is not None
                and (time_value - event_start_time) > max_span_ms
            )
            if gap_break or span_break:
                flush_event(current_indices)
                current_indices = [idx]
                event_start_time = time_value
            else:
                current_indices.append(idx)
            previous_time = time_value
        if current_indices:
            flush_event(current_indices)

    if not rows:
        out = step4_df.head(0).copy()
    else:
        out = pd.DataFrame(rows)
    out = out.sort_values(["WellName", "LayerGroup", "TIME", "WellControlSampleID"]).reset_index(drop=True)
    out["ControlPointID"] = ["ctrl_%06d" % (idx + 1) for idx in range(len(out))]
    return out


def load_real_well_tracks(root: Path, well_names: set[str]) -> dict[str, pd.DataFrame]:
    tracks: dict[str, pd.DataFrame] = {}
    for well_name in sorted(well_names):
        for path in list_well_segment_files(root / str(well_name)):
            df = read_csv_flexible(path, low_memory=False)
            required = ["X", "Y", "TIME"]
            if any(column not in df.columns for column in required):
                continue
            for column in ["X", "Y", "TIME", "TVD", "DEPT", "MD"]:
                if column in df.columns:
                    df[column] = safe_numeric(df[column])
            part = df.dropna(subset=["X", "Y", "TIME"]).reset_index(drop=True)
            if well_name in tracks:
                part = pd.concat([tracks[well_name], part], ignore_index=True)
            tracks[well_name] = part.drop_duplicates(subset=["X", "Y", "TIME"]).reset_index(drop=True)
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


def update_layer_window_from_surfaces(
    row: pd.Series | dict[str, Any],
    surface_lookup: TaiguHorizonSpatialLookup | None,
) -> dict[str, float | int]:
    if surface_lookup is None:
        return {}
    layer = str(row["LayerGroup"])
    x = float(row["CenterX"])
    y = float(row["CenterY"])
    local = surface_lookup.query(x, y)
    if layer == "上部复合层":
        top = float(local["TopTimeMs"])
        base = float(local["MidTimeMs"])
        present = bool(local["UpperPresent"])
    elif layer == "太古界风化壳":
        top = float(local["MidTimeMs"])
        base = float(local["BaseTimeMs"])
        present = bool(local["CrustPresent"])
    else:
        return {}
    valid_window = bool(present and np.isfinite(top) and np.isfinite(base) and base > top)
    center_time = float(row["CenterTime"])
    center_inside = bool(valid_window and np.isfinite(center_time) and top <= center_time <= base)
    return {
        "TimeWindowMin": top if valid_window else np.nan,
        "TimeWindowMax": base if valid_window else np.nan,
        "LayerThickness": base - top if valid_window else np.nan,
        "LayerWindowSurfaceOrderRepaired": 0,
        "HorizonContractValid": int(valid_window),
        "HorizonCenterInside": int(center_inside),
        "SourceTraceIdx": int(local["TraceIdx"]),
        "LocalTopTimeMs": float(local["TopTimeMs"]),
        "LocalMidTimeMs": float(local["MidTimeMs"]),
        "LocalBaseTimeMs": float(local["BaseTimeMs"]),
        "LocalUpperPresent": int(local["UpperPresent"]),
        "LocalCrustPresent": int(local["CrustPresent"]),
    }


def enforce_final_center_horizon_contract(
    frame: pd.DataFrame,
    lookup: TaiguHorizonSpatialLookup,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[pd.Series] = []
    rejected = 0
    clamped = 0
    unchanged_outside = 0
    total_rows = int(len(frame))
    progress_step = max(int(total_rows / 5), 1)
    for position, (_, source_row) in enumerate(frame.iterrows(), start=1):
        if position % progress_step == 0 or position == total_rows:
            print(
                f"    [层位合同收敛] {position}/{total_rows} 片（夹取 {clamped}，层外保留 {unchanged_outside}）",
                flush=True,
            )
        row = source_row.copy()
        modified = str(row.get("CorrectionAction", "")) in {"add_well_control_patch", "adjust_to_well_control"}
        x = float(row["CenterX"])
        y = float(row["CenterY"])
        time = float(row["CenterTime"])
        local = lookup.query(x, y)
        layer = str(row["LayerGroup"])
        if layer == "上部复合层":
            top, base, present = local["TopTimeMs"], local["MidTimeMs"], local["UpperPresent"]
        elif layer == "太古界风化壳":
            top, base, present = local["MidTimeMs"], local["BaseTimeMs"], local["CrustPresent"]
        else:
            if modified:
                rejected += 1
                continue
            rows.append(row)
            continue
        valid = bool(present and np.isfinite(top) and np.isfinite(base) and top < base)
        if not modified:
            if not valid or not (float(top) <= time <= float(base)):
                unchanged_outside += 1
            rows.append(row)
            continue
        if not valid:
            rejected += 1
            continue
        adjusted_time = float(np.clip(time, float(top), float(base)))
        if abs(adjusted_time - time) > 1.0e-6:
            row["OriginalControlTimeBeforeLayerClamp"] = time
            row["CenterTime"] = adjusted_time
            row["TimeLayerConflict"] = 1
            clamped += 1
        else:
            row["TimeLayerConflict"] = 0
        row["LayerCode"] = LAYER_CODE[layer]
        row.update(update_layer_window_from_surfaces(row, lookup))
        rows.append(row)
    output = pd.DataFrame(rows)
    return output.reset_index(drop=True), {
        "input_count": int(len(frame)),
        "kept_count": int(len(output)),
        "rejected_outside_local_target_window_count": int(rejected),
        "modified_center_clamped_to_assigned_layer_count": int(clamped),
        "unchanged_center_outside_contract_count": int(unchanged_outside),
    }


def build_well_modifiable_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "CanModifyInStep8" in df.columns:
        mask &= safe_numeric(df["CanModifyInStep8"]).fillna(1).astype(int).ne(0)
    if "NeedsWellCorrection" in df.columns:
        mask &= safe_numeric(df["NeedsWellCorrection"]).fillna(1).astype(int).ne(0)
    if "ConstraintLevel" in df.columns:
        mask &= ~df["ConstraintLevel"].astype(str).eq("hard")
    if "SourceType" in df.columns:
        mask &= ~df["SourceType"].astype(str).eq("fault_surface")
    return mask


def build_patch_tree(
    df: pd.DataFrame,
    layer: str,
    time_scale: float,
    require_well_modifiable: bool = True,
    allowed_scales: set[str] | None = None,
) -> tuple[cKDTree | None, np.ndarray]:
    layer_mask = df["LayerGroup"].astype(str).eq(layer)
    if require_well_modifiable:
        layer_mask &= build_well_modifiable_mask(df)
    if allowed_scales is not None and "FractureScale" in df.columns:
        layer_mask &= df["FractureScale"].astype(str).str.lower().isin(allowed_scales)
    layer_idx = df.index[layer_mask].to_numpy(dtype=int)
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
    allowed_scales = {
        str(item).lower()
        for item in config.get("well_control_match_scales", [])
        if str(item).strip()
    }
    for pos in positions:
        patch_idx = int(layer_idx[int(pos)])
        if patch_idx in used_patch_indices:
            continue
        patch = initial_df.loc[patch_idx]
        if allowed_scales and str(patch.get("FractureScale", "")).lower() not in allowed_scales:
            continue
        xy_dist = float(np.hypot(float(patch["CenterX"]) - float(control["X"]), float(patch["CenterY"]) - float(control["Y"])))
        time_dist = abs(float(patch["CenterTime"]) - float(control["TIME"]))
        if xy_dist <= xy_radius and time_dist <= time_radius:
            return patch_idx, xy_dist, time_dist
    return None, None, None


def nearest_template_patch(
    initial_df: pd.DataFrame,
    layer: str,
    x: float,
    y: float,
    time_value: float,
    time_scale: float,
    config: dict[str, Any],
) -> pd.Series:
    allowed_scales = {
        str(item).lower()
        for item in config.get("well_control_template_scales", config.get("well_control_match_scales", []))
        if str(item).strip()
    }
    tree, layer_idx = build_patch_tree(
        initial_df,
        layer=layer,
        time_scale=time_scale,
        allowed_scales=allowed_scales or None,
    )
    if tree is None or layer_idx.size == 0:
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
        "ControlSource",
        "IsImagingGroundTruth",
        "ControlX",
        "ControlY",
        "ControlTime",
        "ControlInsidePatchEnvelope",
        "WellControlCenterOffsetM",
        "WellControlCenterOffsetTimeMs",
        "ImagingWellRegionCorrected",
        "ImagingWellRegionInfluence",
        "ImagingWellRegionControlID",
        "ImagingWellRegionOriginalDensity",
        "ImagingWellRegionCorrectedDensity",
    ]


def apply_imaging_well_region_correction(
    patch_df: pd.DataFrame,
    control_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enhance nearby small predictions from positive Step3 imaging controls only."""
    out = patch_df.copy()
    audit_columns = [
        "PatchID", "WellName", "LayerGroup", "NearestImagingControlID",
        "XYDistanceM", "TimeDistanceMs", "Influence",
        "OriginalSourceDensity", "CorrectedSourceDensity",
    ]
    numeric_defaults = {
        "ImagingWellRegionCorrected": 0,
        "ImagingWellRegionInfluence": 0.0,
        "ImagingWellRegionOriginalDensity": np.nan,
        "ImagingWellRegionCorrectedDensity": np.nan,
    }
    for column, default in numeric_defaults.items():
        if column not in out.columns:
            out[column] = default
    # This provenance field may already exist as an all-NaN float column in a
    # historical input.  Always coerce it before assigning string control IDs.
    if "ImagingWellRegionControlID" not in out.columns:
        out["ImagingWellRegionControlID"] = pd.Series("", index=out.index, dtype="object")
    else:
        out["ImagingWellRegionControlID"] = out["ImagingWellRegionControlID"].astype("object")
    if not bool(config.get("enable_imaging_well_region_correction", False)):
        return out, pd.DataFrame(columns=audit_columns)

    is_imaging = safe_numeric(
        control_df.get("IsImagingGroundTruth", pd.Series(0, index=control_df.index))
    ).fillna(0).astype(int).eq(1)
    controls = control_df.loc[is_imaging].copy()
    if controls.empty:
        return out, pd.DataFrame(columns=audit_columns)
    xy_radius = float(config.get("imaging_well_region_xy_radius_m", 150.0))
    time_radius = float(config.get("imaging_well_region_time_radius_ms", 40.0))
    blend = float(np.clip(config.get("imaging_well_region_density_blend", 0.65), 0.0, 1.0))
    min_influence = float(np.clip(config.get("imaging_well_region_min_influence", 0.05), 0.0, 1.0))
    if xy_radius <= 0.0 or time_radius <= 0.0 or blend <= 0.0:
        return out, pd.DataFrame(columns=audit_columns)

    candidate_mask = (
        out.get("FractureScale", pd.Series("", index=out.index)).astype(str).str.lower().eq("small")
        & safe_numeric(out.get("IsWellControlPatch", pd.Series(0, index=out.index))).fillna(0).astype(int).eq(0)
    )
    rows: list[dict[str, Any]] = []
    candidate_count = 0
    within_radius_count = 0
    for (well_name, layer), group in controls.groupby(["WellName", "LayerGroup"], dropna=False):
        candidates = out[candidate_mask & out["LayerGroup"].astype(str).eq(str(layer))].copy()
        if candidates.empty:
            continue
        candidate_count += int(len(candidates))
        control_coords = np.column_stack(
            [
                group["X"].to_numpy(dtype=float) / xy_radius,
                group["Y"].to_numpy(dtype=float) / xy_radius,
                group["TIME"].to_numpy(dtype=float) / time_radius,
            ]
        )
        tree = cKDTree(control_coords)
        candidate_coords = np.column_stack(
            [
                candidates["CenterX"].to_numpy(dtype=float) / xy_radius,
                candidates["CenterY"].to_numpy(dtype=float) / xy_radius,
                candidates["CenterTime"].to_numpy(dtype=float) / time_radius,
            ]
        )
        scaled_distance, positions = tree.query(candidate_coords, k=1)
        for patch_idx, distance, control_position in zip(candidates.index, scaled_distance, positions):
            if not np.isfinite(distance) or float(distance) > 1.0:
                continue
            within_radius_count += 1
            influence = float(np.exp(-0.5 * float(distance) ** 2))
            if influence < min_influence:
                continue
            control = group.iloc[int(control_position)]
            original_density = float(
                safe_numeric(pd.Series([out.at[patch_idx, "FractureIntensityScore"]])).fillna(0.0).iloc[0]
            )
            control_density = float(
                safe_numeric(pd.Series([control.get("DensityScore", np.nan)])).fillna(original_density).iloc[0]
            )
            corrected_density = max(
                original_density,
                original_density + blend * influence * max(control_density - original_density, 0.0),
            )
            out.at[patch_idx, "SourceDensity"] = corrected_density
            out.at[patch_idx, "FractureIntensityScore"] = corrected_density
            if "SourceDensityRender" in out.columns:
                out.at[patch_idx, "SourceDensityRender"] = corrected_density
            out.at[patch_idx, "ImagingWellRegionCorrected"] = 1
            out.at[patch_idx, "ImagingWellRegionInfluence"] = influence
            out.at[patch_idx, "ImagingWellRegionControlID"] = str(control["WellControlSampleID"])
            out.at[patch_idx, "ImagingWellRegionOriginalDensity"] = original_density
            out.at[patch_idx, "ImagingWellRegionCorrectedDensity"] = corrected_density
            rows.append(
                {
                    "PatchID": str(out.at[patch_idx, "PatchID"]),
                    "WellName": str(well_name),
                    "LayerGroup": str(layer),
                    "NearestImagingControlID": str(control["WellControlSampleID"]),
                    "XYDistanceM": float(np.hypot(float(out.at[patch_idx, "CenterX"]) - float(control["X"]), float(out.at[patch_idx, "CenterY"]) - float(control["Y"]))),
                    "TimeDistanceMs": abs(float(out.at[patch_idx, "CenterTime"]) - float(control["TIME"])),
                    "Influence": influence,
                    "OriginalSourceDensity": original_density,
                    "CorrectedSourceDensity": corrected_density,
                }
            )
    audit = pd.DataFrame(rows, columns=audit_columns)
    config["_imaging_well_region_eligibility"] = {
        "control_point_count": int(len(controls)),
        "candidate_patch_count": int(candidate_count),
        "within_radius_candidate_count": int(within_radius_count),
        "corrected_patch_count": int(len(audit)),
    }
    return out, audit


def stable_unit_value(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    integer = int(digest[:12], 16)
    return 2.0 * (integer / float(0xFFFFFFFFFFFF)) - 1.0


def add_control_size_columns(control_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = control_df.copy()
    size_config = dict(config.get("well_control_size", {}))
    min_length = float(size_config.get("min_length_m", 30.0))
    max_length = float(size_config.get("max_length_m", 120.0))
    min_height = float(size_config.get("min_height_time_ms", 6.0))
    max_height = float(size_config.get("max_height_time_ms", 28.0))
    imaging_multiplier = float(size_config.get("imaging_gt_size_multiplier", 1.0))
    out["WellControlSizeFactor"] = safe_numeric(out.get("DensityScore", pd.Series(0.5, index=out.index))).fillna(0.5).clip(0.0, 1.0)
    out["WellControlLengthM"] = min_length + out["WellControlSizeFactor"] * (max_length - min_length)
    out["WellControlHeightTimeMs"] = min_height + out["WellControlSizeFactor"] * (max_height - min_height)
    if "IsImagingGroundTruth" in out.columns:
        gt_mask = safe_numeric(out["IsImagingGroundTruth"]).fillna(0).astype(int).eq(1)
        out.loc[gt_mask, "WellControlLengthM"] *= imaging_multiplier
        out.loc[gt_mask, "WellControlHeightTimeMs"] *= imaging_multiplier
    out["WellControlLengthM"] = out["WellControlLengthM"].clip(lower=min_length, upper=max_length)
    out["WellControlHeightTimeMs"] = out["WellControlHeightTimeMs"].clip(lower=min_height, upper=max_height)
    return out


def config_layer_value(config: dict[str, Any], key: str, layer: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, dict):
        return float(value.get(layer, default))
    return float(value)


def control_orientation(template: pd.Series, control: pd.Series, config: dict[str, Any]) -> tuple[float, float, str]:
    is_gt = int(control.get("IsImagingGroundTruth", 0) or 0) == 1
    azimuth = pd.to_numeric(pd.Series([control.get("Frac_Azimuth")]), errors="coerce").iloc[0]
    dip = pd.to_numeric(pd.Series([control.get("Frac_Dip")]), errors="coerce").iloc[0]
    if is_gt and np.isfinite(azimuth) and np.isfinite(dip):
        return float(azimuth) % 180.0, float(np.clip(dip, 1.0, 89.0)), "step3_imaging_gt_orientation"
    template_azimuth = float(template["AzimuthDeg"]) % 180.0
    template_dip = float(np.clip(template["DipDeg"], 1.0, 89.0))
    policy = dict(config.get("step4_small_orientation_policy", {}))
    if not bool(policy.get("enabled", False)):
        return template_azimuth, template_dip, "well_control_template_from_nearest_initial_patch"
    layer = str(control.get("LayerGroup", template.get("LayerGroup", "")))
    target_dip = config_layer_value(policy, "target_dip_deg", layer, 62.0)
    min_dip = float(policy.get("min_dip_deg", 35.0))
    max_dip = float(policy.get("max_dip_deg", 78.0))
    blend = float(policy.get("template_blend_to_target", 0.45))
    steep_blend = float(policy.get("steep_template_blend_to_target", 0.65))
    az_jitter = float(policy.get("azimuth_jitter_deg", 8.0))
    dip_jitter = float(policy.get("dip_jitter_deg", 5.0))
    key = f"{control.get('WellName', '')}|{control.get('WellControlSampleID', '')}|{layer}"
    azimuth = (template_azimuth + stable_unit_value(key + "|azimuth") * az_jitter) % 180.0
    clipped_template_dip = float(np.clip(template_dip, min_dip, max_dip))
    use_blend = steep_blend if template_dip > max_dip else blend
    dip = (1.0 - use_blend) * clipped_template_dip + use_blend * target_dip
    dip += stable_unit_value(key + "|dip_soften") * dip_jitter
    dip = float(np.clip(dip, min_dip, max_dip))
    return azimuth, dip, "well_control_small_scale_template_softened_orientation"


def force_well_control_small_scale(row: dict[str, Any], control: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    if not bool(config.get("force_well_control_small_scale", False)):
        return row
    is_gt = int(control.get("IsImagingGroundTruth", 0) or 0) == 1
    row["FractureScale"] = "small"
    row["FractureScaleCode"] = 1
    row["SourceType"] = "small_step3_imaging_well_control" if is_gt else "small_step4_predicted_well_control"
    row["StructuralRelation"] = "small_well_control"
    row["ConstraintLevel"] = "well_control"
    row["GenerationStage"] = "step8_small_well_control"
    return row


def offset_center_near_control(control: pd.Series, azimuth_deg: float, dip_deg: float, length_m: float, height_time_ms: float, config: dict[str, Any]) -> dict[str, float]:
    if not bool(config.get("enable_well_control_center_offset", True)):
        return {
            "CenterX": float(control["X"]),
            "CenterY": float(control["Y"]),
            "CenterTime": float(control["TIME"]),
            "CenterOffsetM": 0.0,
            "CenterOffsetTimeMs": 0.0,
            "ControlInsidePatchEnvelope": 1,
        }
    theta = np.deg2rad(float(azimuth_deg))
    dip = np.deg2rad(float(np.clip(dip_deg, 1.0, 89.0)))
    strike = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(theta), np.cos(theta)], dtype=float)
    time_scale = float(config.get("geometry_time_scale_m_per_ms", config.get("time_scale_m_per_ms", 2.0)))
    half_length = 0.5 * float(length_m)
    half_height = 0.5 * float(height_time_ms)
    half_dip_xy = half_height * time_scale / max(float(np.tan(dip)), 1.0e-6)
    max_fraction = float(config.get("well_control_center_offset_max_fraction", 0.35))
    dip_fraction = float(config.get("well_control_center_offset_dip_fraction", max_fraction))
    target_fraction = float(config.get("well_control_center_offset_target_fraction", max_fraction))
    max_offset_m = float(config.get("well_control_center_offset_max_m", 45.0))
    min_offset_m = float(config.get("well_control_center_offset_min_m", 6.0))
    strike_limit = max(0.0, min(max_offset_m, half_length * max_fraction))
    dip_limit = max(0.0, min(max_offset_m, half_dip_xy * dip_fraction))
    key = f"{control.get('WellName', '')}|{control.get('WellControlSampleID', '')}|{control.get('LayerGroup', '')}"
    strike_raw = stable_unit_value(key + "|strike")
    dip_raw = stable_unit_value(key + "|dip")
    strike_scale = min(1.0, max(0.0, target_fraction) + (1.0 - max(0.0, target_fraction)) * abs(strike_raw))
    dip_scale = min(1.0, max(0.0, target_fraction) + (1.0 - max(0.0, target_fraction)) * abs(dip_raw))
    strike_offset = np.sign(strike_raw if strike_raw != 0.0 else 1.0) * strike_limit * strike_scale
    if abs(strike_offset) < min_offset_m and strike_limit >= min_offset_m:
        strike_offset = np.sign(strike_raw if strike_raw != 0.0 else 1.0) * min_offset_m
    dip_offset = np.sign(dip_raw if dip_raw != 0.0 else 1.0) * dip_limit * dip_scale
    xy_offset = strike_offset * strike + dip_offset * dip_horizontal
    time_offset = dip_offset * float(np.tan(dip)) / max(time_scale, 1.0e-6)
    center_x = float(control["X"]) + float(xy_offset[0])
    center_y = float(control["Y"]) + float(xy_offset[1])
    center_time = float(control["TIME"]) + float(time_offset)
    target = config.get("target_block") or {}
    if target:
        center_x = float(np.clip(center_x, float(target["x_min"]), float(target["x_max"])))
        center_y = float(np.clip(center_y, float(target["y_min"]), float(target["y_max"])))
    inside = int(abs(strike_offset) <= half_length + 1.0e-6 and abs(dip_offset) <= half_dip_xy + 1.0e-6 and abs(time_offset) <= half_height + 1.0e-6)
    return {
        "CenterX": center_x,
        "CenterY": center_y,
        "CenterTime": center_time,
        "CenterOffsetM": float(np.hypot(float(xy_offset[0]), float(xy_offset[1]))),
        "CenterOffsetTimeMs": float(abs(time_offset)),
        "ControlInsidePatchEnvelope": inside,
    }


def apply_geometry_from_control(row: pd.Series | dict[str, Any], template: pd.Series, control: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    azimuth, dip, orientation_source = control_orientation(template=template, control=control, config=config)
    length = float(control["WellControlLengthM"]) if "WellControlLengthM" in control and pd.notna(control["WellControlLengthM"]) else float(template["LengthM"])
    height = float(control["WellControlHeightTimeMs"]) if "WellControlHeightTimeMs" in control and pd.notna(control["WellControlHeightTimeMs"]) else float(template["HeightTimeMs"])
    center = offset_center_near_control(control=control, azimuth_deg=azimuth, dip_deg=dip, length_m=length, height_time_ms=height, config=config)
    out.update(center)
    out["LengthM"] = length
    out["HeightTimeMs"] = height
    out["AzimuthDeg"] = azimuth
    out["DipDeg"] = dip
    out["OrientationSource"] = orientation_source
    out["SizeRule"] = "well_control_density_scaled"
    out["ControlInsidePatchEnvelope"] = int(center["ControlInsidePatchEnvelope"])
    out["WellControlCenterOffsetM"] = float(center["CenterOffsetM"])
    out["WellControlCenterOffsetTimeMs"] = float(center["CenterOffsetTimeMs"])
    out["ControlX"] = float(control["X"])
    out["ControlY"] = float(control["Y"])
    out["ControlTime"] = float(control["TIME"])
    out["ControlSource"] = str(control.get("ControlSource", "step4_predicted"))
    out["IsImagingGroundTruth"] = int(control.get("IsImagingGroundTruth", 0) or 0)
    control_score = float(control.get("DensityScore", 0.5))
    original_score = float(pd.to_numeric(pd.Series([out.get("FractureIntensityScore", 0.5)]), errors="coerce").fillna(0.5).iloc[0])
    corrected_score = max(original_score, control_score)
    out["WellControlDensityScore"] = control_score
    out["FractureIntensityScore"] = corrected_score
    out["SourceDensity"] = corrected_score
    out["SourceDensityRender"] = corrected_score
    out["SourceDensityRenderNorm"] = corrected_score
    out["temporary_neighbor_time_depth"] = int(control.get("temporary_neighbor_time_depth", 0) or 0)
    out = force_well_control_small_scale(out, control=control, config=config)
    return out


def build_added_patch(
    template: pd.Series,
    control: pd.Series,
    add_index: int,
    before_distance: float,
    track_distance: float,
    config: dict[str, Any],
    surface_lookup: TaiguHorizonSpatialLookup | None = None,
) -> dict[str, Any]:
    row = template.to_dict()
    row["PatchID"] = f"well_ctrl_dfn_{add_index:06d}"
    row["OriginalPatchID"] = ""
    row["OriginalCenterX"] = np.nan
    row["OriginalCenterY"] = np.nan
    row["OriginalCenterTime"] = np.nan
    row["LayerGroup"] = str(control["LayerGroup"])
    row["LayerCode"] = int(LAYER_CODE[str(control["LayerGroup"])])
    row["SourceDensity"] = float(control.get("DensityScore", 0.5))
    row["RawDensity"] = float(control["Density"]) if "Density" in control and pd.notna(control["Density"]) else np.nan
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
    row["SamplingRule"] = "hard_well_control_addition"
    row = apply_geometry_from_control(row=row, template=template, control=control, config=config)
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    row["WellControlMatchAfter"] = float(
        np.sqrt(
            (float(row["CenterX"]) - float(control["X"])) ** 2
            + (float(row["CenterY"]) - float(control["Y"])) ** 2
            + ((float(row["CenterTime"]) - float(control["TIME"])) * time_scale) ** 2
        )
    )
    row.update(update_layer_window_from_surfaces(row, surface_lookup))
    return row


def apply_well_controls(
    initial_df: pd.DataFrame,
    control_df: pd.DataFrame,
    track_distances: pd.Series,
    config: dict[str, Any],
    surface_lookup: TaiguHorizonSpatialLookup | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    control_df = add_control_size_columns(control_df, config=config)
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
            updated = apply_geometry_from_control(
                row=corrected.loc[patch_idx],
                template=original,
                control=control,
                config=config,
            )
            for update_col, update_value in updated.items():
                if update_col not in corrected.columns:
                    corrected[update_col] = pd.Series([pd.NA] * len(corrected), dtype="object") if isinstance(update_value, str) else np.nan
                elif isinstance(update_value, str) and not pd.api.types.is_object_dtype(corrected[update_col].dtype):
                    corrected[update_col] = corrected[update_col].astype("object")
                corrected.loc[patch_idx, update_col] = update_value
            corrected.loc[patch_idx, "CorrectionAction"] = "adjust_to_well_control"
            corrected.loc[patch_idx, "CorrectionReason"] = "unused_initial_patch_within_search_window"
            corrected.loc[patch_idx, "WellControlSampleID"] = str(control["WellControlSampleID"])
            corrected.loc[patch_idx, "WellControlWellName"] = str(control["WellName"])
            corrected.loc[patch_idx, "WellControlDensity"] = float(control["Density"]) if "Density" in control and pd.notna(control["Density"]) else np.nan
            corrected.loc[patch_idx, "WellControlMatchBefore"] = before
            after_center_dist = float(
                np.sqrt(
                    (float(corrected.loc[patch_idx, "CenterX"]) - float(control["X"])) ** 2
                    + (float(corrected.loc[patch_idx, "CenterY"]) - float(control["Y"])) ** 2
                    + ((float(corrected.loc[patch_idx, "CenterTime"]) - float(control["TIME"])) * time_scale) ** 2
                )
            )
            corrected.loc[patch_idx, "WellControlMatchAfter"] = after_center_dist
            corrected.loc[patch_idx, "NearestTrajectoryDistance"] = track_distance
            corrected.loc[patch_idx, "IsWellControlPatch"] = 1
            corrected.loc[patch_idx, "NeedsWellCorrection"] = 0
            window_update = update_layer_window_from_surfaces(corrected.loc[patch_idx], surface_lookup)
            for update_col, update_value in window_update.items():
                corrected.loc[patch_idx, update_col] = update_value
            patch_id = str(corrected.loc[patch_idx, "PatchID"])
            action = "adjust_to_well_control"
            reason = "unused_initial_patch_within_search_window"
            original_patch_id = str(original["PatchID"])
            original_x = float(original["CenterX"])
            original_y = float(original["CenterY"])
            original_time = float(original["CenterTime"])
        else:
            is_imaging = int(control.get("IsImagingGroundTruth", 0) or 0) == 1
            weak_add = dict(config.get("step4_unmatched_addition", {}))
            probability = float(
                safe_numeric(pd.Series([control.get("PredFractureProb", np.nan)])).fillna(0.0).iloc[0]
            )
            score = float(control.get("DensityScore", 0.0))
            allow_weak_add = bool(weak_add.get("enabled", False)) and (
                probability >= float(weak_add.get("min_probability", 0.8))
                and score >= float(weak_add.get("min_density_score", 0.8))
            )
            if not is_imaging and not allow_weak_add:
                audit_rows.append(
                    {
                        "ControlPointID": str(control["ControlPointID"]),
                        "WellControlSampleID": str(control["WellControlSampleID"]),
                        "WellName": str(control["WellName"]),
                        "LayerGroup": str(control["LayerGroup"]),
                        "ControlSource": str(control.get("ControlSource", "step4_predicted")),
                        "IsImagingGroundTruth": 0,
                        "ControlX": float(control["X"]),
                        "ControlY": float(control["Y"]),
                        "ControlTime": float(control["TIME"]),
                        "ControlDensity": float(control["Density"]) if pd.notna(control.get("Density")) else np.nan,
                        "ControlDensityScore": score,
                        "PredFractureProb": probability,
                        "Action": "skip_unmatched_weak_control",
                        "ActionReason": "step4_event_does_not_meet_unmatched_addition_threshold",
                        "PatchID": "",
                        "OriginalPatchID": "",
                        "BeforeMatchDistance": before,
                        "NearestTrajectoryDistance": track_distance,
                        "XYSearchDistance": xy_dist,
                        "TimeSearchDistance": time_dist,
                    }
                )
                continue
            template = nearest_template_patch(
                initial_df=initial_df,
                layer=str(control["LayerGroup"]),
                x=float(control["X"]),
                y=float(control["Y"]),
                time_value=float(control["TIME"]),
                time_scale=time_scale,
                config=config,
            )
            added = build_added_patch(
                template=template,
                control=control,
                add_index=len(added_rows) + 1,
                before_distance=before,
                track_distance=track_distance,
                config=config,
                surface_lookup=surface_lookup,
            )
            added_rows.append(added)
            patch_id = str(added["PatchID"])
            action = "add_well_control_patch"
            reason = "no_unused_initial_patch_within_search_window"
            original_patch_id = str(template["PatchID"])
            original_x = float(template["CenterX"])
            original_y = float(template["CenterY"])
            original_time = float(template["CenterTime"])

        if patch_idx is not None:
            corrected_x = float(corrected.loc[patch_idx, "CenterX"])
            corrected_y = float(corrected.loc[patch_idx, "CenterY"])
            corrected_time = float(corrected.loc[patch_idx, "CenterTime"])
            center_offset_m = float(corrected.loc[patch_idx, "WellControlCenterOffsetM"])
            center_offset_time_ms = float(corrected.loc[patch_idx, "WellControlCenterOffsetTimeMs"])
            inside_envelope = int(corrected.loc[patch_idx, "ControlInsidePatchEnvelope"])
        else:
            corrected_x = float(added["CenterX"])
            corrected_y = float(added["CenterY"])
            corrected_time = float(added["CenterTime"])
            center_offset_m = float(added["WellControlCenterOffsetM"])
            center_offset_time_ms = float(added["WellControlCenterOffsetTimeMs"])
            inside_envelope = int(added["ControlInsidePatchEnvelope"])
        after = float(
            np.sqrt(
                (corrected_x - float(control["X"])) ** 2
                + (corrected_y - float(control["Y"])) ** 2
                + ((corrected_time - float(control["TIME"])) * time_scale) ** 2
            )
        )

        audit_rows.append(
            {
                "ControlPointID": str(control["ControlPointID"]),
                "WellControlSampleID": str(control["WellControlSampleID"]),
                "WellName": str(control["WellName"]),
                "LayerGroup": str(control["LayerGroup"]),
                "ControlSource": str(control.get("ControlSource", "step4_predicted")),
                "IsImagingGroundTruth": int(control.get("IsImagingGroundTruth", 0) or 0),
                "ControlX": float(control["X"]),
                "ControlY": float(control["Y"]),
                "ControlTime": float(control["TIME"]),
                "ControlDensity": float(control["Density"]) if "Density" in control and pd.notna(control["Density"]) else np.nan,
                "ControlSizeFactor": float(control["WellControlSizeFactor"]),
                "ControlLengthM": float(control["WellControlLengthM"]),
                "ControlHeightTimeMs": float(control["WellControlHeightTimeMs"]),
                "Action": action,
                "ActionReason": reason,
                "PatchID": patch_id,
                "OriginalPatchID": original_patch_id,
                "OriginalCenterX": original_x,
                "OriginalCenterY": original_y,
                "OriginalCenterTime": original_time,
                "CorrectedCenterX": corrected_x,
                "CorrectedCenterY": corrected_y,
                "CorrectedCenterTime": corrected_time,
                "CenterOffsetM": center_offset_m,
                "CenterOffsetTimeMs": center_offset_time_ms,
                "ControlInsidePatchEnvelope": inside_envelope,
                "BeforeMatchDistance": before,
                "AfterMatchDistance": after,
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


def patch_vertices(
    row: pd.Series,
    display: bool,
    display_z_scale: float,
    use_dip_geometry: bool,
    geometry_time_scale_m_per_ms: float,
) -> list[tuple[float, float, float]]:
    vertex_cols = [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
    preserve_vertices = all(col in row.index and pd.notna(row.get(col)) for col in vertex_cols)
    correction_action = str(row.get("CorrectionAction", "unchanged_density_volume"))
    if preserve_vertices and correction_action not in {"add_well_control_patch", "adjust_to_well_control"}:
        points = []
        for vertex_idx in range(1, 5):
            x = float(row[f"V{vertex_idx}X"])
            y = float(row[f"V{vertex_idx}Y"])
            z = float(row[f"V{vertex_idx}Z"])
            if display:
                z = -z / display_z_scale
            points.append((x, y, z))
        return points

    theta = np.deg2rad(float(row["AzimuthDeg"]))
    half_length = 0.5 * float(row["LengthM"])
    half_h = 0.5 * float(row["HeightTimeMs"])
    center_x = float(row["CenterX"])
    center_y = float(row["CenterY"])
    center_time = float(row["CenterTime"])
    strike = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)

    if use_dip_geometry:
        dip = np.deg2rad(float(np.clip(row["DipDeg"], 1.0, 89.9)))
        dip_horizontal = np.asarray([-np.sin(theta), np.cos(theta)], dtype=float)
        half_dip_xy = (half_h * geometry_time_scale_m_per_ms) / max(np.tan(dip), 1.0e-6)
        corners: list[tuple[float, float, float]] = []
        for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            xy = np.asarray([center_x, center_y], dtype=float) + strike_sign * half_length * strike + dip_sign * half_dip_xy * dip_horizontal
            z = center_time + dip_sign * half_h
            if display:
                z = -z / display_z_scale
            corners.append((float(xy[0]), float(xy[1]), float(z)))
        return corners

    half_dx = half_length * np.cos(theta)
    half_dy = half_length * np.sin(theta)
    z0 = center_time - half_h
    z1 = center_time + half_h
    if display:
        z0 = -z0 / display_z_scale
        z1 = -z1 / display_z_scale
    return [
        (center_x - half_dx, center_y - half_dy, z0),
        (center_x + half_dx, center_y + half_dy, z0),
        (center_x + half_dx, center_y + half_dy, z1),
        (center_x - half_dx, center_y - half_dy, z1),
    ]


def polygon_area(points: list[tuple[float, float, float]]) -> float:
    vertices = np.asarray(points, dtype=float)
    if vertices.shape != (4, 3):
        return 0.0
    area_1 = 0.5 * float(np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])))
    area_2 = 0.5 * float(np.linalg.norm(np.cross(vertices[2] - vertices[0], vertices[3] - vertices[0])))
    return area_1 + area_2


def refresh_well_control_vertices(
    patch_df: pd.DataFrame,
    use_dip_geometry: bool,
    geometry_time_scale_m_per_ms: float,
) -> pd.DataFrame:
    out = patch_df.copy()
    if "CorrectionAction" not in out.columns:
        return out
    modified_mask = out["CorrectionAction"].astype(str).isin(["add_well_control_patch", "adjust_to_well_control"])
    if not modified_mask.any():
        return out
    vertex_cols = [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
    for col in vertex_cols:
        if col not in out.columns:
            out[col] = np.nan
    for idx, row in out.loc[modified_mask].iterrows():
        points = patch_vertices(
            row,
            display=False,
            display_z_scale=1.0,
            use_dip_geometry=use_dip_geometry,
            geometry_time_scale_m_per_ms=geometry_time_scale_m_per_ms,
        )
        for vertex_idx, point in enumerate(points, start=1):
            out.loc[idx, f"V{vertex_idx}X"] = float(point[0])
            out.loc[idx, f"V{vertex_idx}Y"] = float(point[1])
            out.loc[idx, f"V{vertex_idx}Z"] = float(point[2])
        out.loc[idx, "PatchAreaM2"] = polygon_area(points)
    return out


def materialize_and_clip_vertices_to_horizon_contract(
    patch_df: pd.DataFrame,
    lookup: TaiguHorizonSpatialLookup,
    use_dip_geometry: bool,
    geometry_time_scale_m_per_ms: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = patch_df.reset_index(drop=True).copy()
    vertex_cols = [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
    for column in vertex_cols:
        if column not in out.columns:
            out[column] = np.nan

    all_vertices = np.empty((len(out), 4, 3), dtype=np.float64)
    for row_idx, row in out.iterrows():
        all_vertices[row_idx] = np.asarray(
            patch_vertices(
                row,
                display=False,
                display_z_scale=1.0,
                use_dip_geometry=use_dip_geometry,
                geometry_time_scale_m_per_ms=geometry_time_scale_m_per_ms,
            ),
            dtype=np.float64,
        )

    flat_vertices = all_vertices.reshape(-1, 3)
    original_vertices = flat_vertices.copy()
    _, positions = lookup.tree.query(flat_vertices[:, :2], k=1)
    horizon_rows = lookup.table.iloc[np.asarray(positions, dtype=np.int64)].reset_index(drop=True)
    vertex_layers = np.repeat(out["LayerGroup"].astype(str).to_numpy(), 4)
    is_crust = vertex_layers == "太古界风化壳"
    top = np.where(is_crust, horizon_rows["MidTimeMs"], horizon_rows["TopTimeMs"]).astype(np.float64)
    base = np.where(is_crust, horizon_rows["BaseTimeMs"], horizon_rows["MidTimeMs"]).astype(np.float64)
    surface_valid = pd.to_numeric(horizon_rows["SurfaceValid"], errors="coerce").fillna(0).to_numpy(dtype=bool)
    present = surface_valid & np.isfinite(top) & np.isfinite(base) & (top < base)
    valid = present & np.isfinite(top) & np.isfinite(base) & (base >= top)
    modified = out.get("CorrectionAction", pd.Series("", index=out.index)).astype(str).isin(
        ["add_well_control_patch", "adjust_to_well_control"]
    ).to_numpy()
    modified_vertices = modified.repeat(4)

    # A hard fault center remains fixed. If a corner reaches a local trace without a
    # If a modified corner reaches an invalid local trace, shorten that corner
    # along the center-to-corner edge until it re-enters the target layer.
    for flat_idx in np.where(modified_vertices & ~valid)[0]:
        patch_idx = int(flat_idx // 4)
        patch_layer = str(out.loc[patch_idx, "LayerGroup"])
        center_xy = out.loc[patch_idx, ["CenterX", "CenterY"]].to_numpy(dtype=np.float64)
        vertex_xy = flat_vertices[flat_idx, :2].copy()
        low = 0.0
        high = 1.0
        best: tuple[np.ndarray, dict[str, Any]] | None = None
        for _ in range(24):
            fraction = 0.5 * (low + high)
            candidate_xy = center_xy + fraction * (vertex_xy - center_xy)
            local = lookup.query(float(candidate_xy[0]), float(candidate_xy[1]))
            if patch_layer == "太古界风化壳":
                local_present = bool(local["CrustPresent"])
                local_top = float(local["MidTimeMs"])
                local_base = float(local["BaseTimeMs"])
            else:
                local_present = bool(local["UpperPresent"])
                local_top = float(local["TopTimeMs"])
                local_base = float(local["MidTimeMs"])
            local_valid = bool(local_present and np.isfinite(local_top) and np.isfinite(local_base) and local_base >= local_top)
            if local_valid:
                low = fraction
                best = (candidate_xy, local)
            else:
                high = fraction
        if best is None:
            local = lookup.query(float(center_xy[0]), float(center_xy[1]))
            best = (center_xy, local)
        candidate_xy, local = best
        flat_vertices[flat_idx, :2] = candidate_xy
        if patch_layer == "太古界风化壳":
            top[flat_idx] = float(local["MidTimeMs"])
            base[flat_idx] = float(local["BaseTimeMs"])
            local_present = bool(local["CrustPresent"])
        else:
            top[flat_idx] = float(local["TopTimeMs"])
            base[flat_idx] = float(local["MidTimeMs"])
            local_present = bool(local["UpperPresent"])
        valid[flat_idx] = bool(
            local_present
            and np.isfinite(top[flat_idx])
            and np.isfinite(base[flat_idx])
            and base[flat_idx] >= top[flat_idx]
        )
    original_z = flat_vertices[:, 2].copy()
    clipped_z = original_z.copy()
    clip_mask = modified_vertices & valid
    clipped_z[clip_mask] = np.clip(original_z[clip_mask], top[clip_mask], base[clip_mask])
    flat_vertices[:, 2] = clipped_z
    all_vertices = flat_vertices.reshape(len(out), 4, 3)
    delta = np.abs(clipped_z - original_z).reshape(len(out), 4)
    xy_delta = np.linalg.norm(flat_vertices[:, :2] - original_vertices[:, :2], axis=1).reshape(len(out), 4)

    for vertex_idx in range(1, 5):
        out[f"V{vertex_idx}X"] = all_vertices[:, vertex_idx - 1, 0]
        out[f"V{vertex_idx}Y"] = all_vertices[:, vertex_idx - 1, 1]
        out[f"V{vertex_idx}Z"] = all_vertices[:, vertex_idx - 1, 2]
    out["HorizonClippedVertexCount"] = (delta > 1.0e-6).sum(axis=1).astype(np.int16)
    out["HorizonVertexMaxClipMs"] = delta.max(axis=1)
    out["HorizonHorizontallyClippedVertexCount"] = (xy_delta > 1.0e-6).sum(axis=1).astype(np.int16)
    out["HorizonVertexMaxHorizontalClipM"] = xy_delta.max(axis=1)
    out["HorizonVerticesInsideTargetLayer"] = np.where(
        modified,
        valid.reshape(len(out), 4).all(axis=1),
        True,
    ).astype(np.uint8)
    print(f"    [顶点几何] 计算 {len(all_vertices)} 片面积", flush=True)
    out["PatchAreaM2"] = [polygon_area([tuple(point) for point in vertices]) for vertices in all_vertices]
    print(
        f"    [顶点几何] 完成：修改片 {int(modified.sum())}，裁剪顶点 {int((delta > 1.0e-6).sum())}，"
        f"水平收缩顶点 {int((xy_delta > 1.0e-6).sum())}",
        flush=True,
    )
    return out, {
        "patch_count": int(len(out)),
        "vertex_count": int(len(flat_vertices)),
        "clipped_patch_count": int((out["HorizonClippedVertexCount"] > 0).sum()),
        "clipped_vertex_count": int((delta > 1.0e-6).sum()),
        "horizontally_clipped_patch_count": int((out["HorizonHorizontallyClippedVertexCount"] > 0).sum()),
        "horizontally_clipped_vertex_count": int((xy_delta > 1.0e-6).sum()),
        "checked_modified_patch_count": int(modified.sum()),
        "invalid_local_window_vertex_count": int((modified_vertices & ~valid).sum()),
        "all_modified_vertices_inside_local_layer_window": bool(valid[modified_vertices].all()),
        "max_clip_ms": float(delta.max()) if delta.size else 0.0,
        "max_horizontal_clip_m": float(xy_delta.max()) if xy_delta.size else 0.0,
        "hard_constraint_center_or_xy_shifted": False,
    }


def write_legacy_vtk(
    path: Path,
    patch_df: pd.DataFrame,
    title: str,
    display: bool,
    display_z_scale: float,
    use_dip_geometry: bool = False,
    geometry_time_scale_m_per_ms: float = 1.0,
) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in patch_df.iterrows():
        base = len(points)
        points.extend(
            patch_vertices(
                row,
                display=display,
                display_z_scale=display_z_scale,
                use_dip_geometry=use_dip_geometry,
                geometry_time_scale_m_per_ms=geometry_time_scale_m_per_ms,
            )
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
        ("IsWellControlPatch", safe_numeric(patch_df["IsWellControlPatch"]).fillna(0).to_numpy(), "int"),
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
    optional_int_columns = ["SourceTypeCode", "ConstraintLevelCode", "FaultRelationCode", "CanModifyInStep8"]
    for column in optional_int_columns:
        if column in patch_df.columns:
            scalar_columns.append((column, safe_numeric(patch_df[column]).fillna(0).to_numpy(), "int"))
    for name, values, dtype in scalar_columns:
        vtk_type = "int" if dtype == "int" else "float"
        lines.append(f"SCALARS {name} {vtk_type} 1")
        lines.append("LOOKUP_TABLE default")
        if vtk_type == "int":
            lines.extend(str(int(value)) for value in values)
        else:
            lines.extend(f"{float(value):.6f}" for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_debug_step_vtks(
    output_dir: Path,
    initial_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    config: dict[str, Any],
    display_z_scale: float,
    use_dip_geometry: bool,
    geometry_time_scale_m_per_ms: float,
) -> dict[str, str]:
    debug_dir = output_dir / "debug_step_vtk"
    ensure_dir(debug_dir)
    modifiable_mask = build_well_modifiable_mask(initial_df)
    correction_action = corrected_df["CorrectionAction"].astype(str)
    no_added_df = corrected_df[correction_action.ne("add_well_control_patch")].copy()
    adjusted_only_df = corrected_df[correction_action.eq("adjust_to_well_control")].copy()
    added_only_df = corrected_df[correction_action.eq("add_well_control_patch")].copy()
    no_hard_fault_df = corrected_df[
        ~(
            corrected_df.get("SourceType", pd.Series("", index=corrected_df.index)).astype(str).eq("fault_surface")
            | corrected_df.get("ConstraintLevel", pd.Series("", index=corrected_df.index)).astype(str).eq("hard")
        )
    ].copy()
    steps: list[tuple[str, str, pd.DataFrame]] = [
        ("00_step7c_initial_input_all_raw_time.vtk", "step8_debug_00_initial_step7c_input_all", initial_df),
        ("01_step8_initial_modifiable_pool_raw_time.vtk", "step8_debug_01_initial_modifiable_pool", initial_df.loc[modifiable_mask].copy()),
        ("02_step8_initial_nonmodifiable_hard_or_no_well_correction_raw_time.vtk", "step8_debug_02_initial_nonmodifiable", initial_df.loc[~modifiable_mask].copy()),
        ("03_step8_after_adjust_existing_before_add_raw_time.vtk", "step8_debug_03_adjusted_existing_before_additions", no_added_df),
        ("04_step8_adjusted_existing_only_raw_time.vtk", "step8_debug_04_adjusted_existing_only", adjusted_only_df),
        ("05_step8_added_well_control_only_raw_time.vtk", "step8_debug_05_added_well_control_only", added_only_df),
        ("06_step8_final_without_hard_fault_surface_raw_time.vtk", "step8_debug_06_final_without_hard_fault_surface", no_hard_fault_df),
        ("07_step8_final_all_raw_time.vtk", "step8_debug_07_final_all", corrected_df),
    ]
    exported: dict[str, str] = {}
    for filename, title, df in steps:
        if df.empty:
            continue
        path = debug_dir / filename
        write_legacy_vtk(
            path,
            df.reset_index(drop=True),
            title,
            display=False,
            display_z_scale=display_z_scale,
            use_dip_geometry=use_dip_geometry,
            geometry_time_scale_m_per_ms=geometry_time_scale_m_per_ms,
        )
        exported[filename] = str(path)
    manifest = {
        "description": "Step8 intermediate VTK exports for diagnosing where geometry changes appear.",
        "notes": [
            "00 is the Step7C input consumed by Step8.",
            "01 is the initial patch pool eligible for well-control matching.",
            "02 is the initial non-modifiable pool, including hard fault surfaces and patches not needing well correction.",
            "03 is the cumulative result after adjusting existing matched patches, before adding unmatched well-control patches.",
            "04 contains only existing patches adjusted to well controls.",
            "05 contains only newly added well-control patches.",
            "06 is the final corrected DFN with hard fault surfaces removed for display comparison.",
            "07 contains corrected predicted patches only; the formal well_corrected_dfn_raw_time.vtk also carries unchanged original-fault triangles.",
        ],
        "exported_vtks": exported,
        "counts": {
            "initial_all": int(len(initial_df)),
            "initial_modifiable_pool": int(modifiable_mask.sum()),
            "initial_nonmodifiable": int((~modifiable_mask).sum()),
            "after_adjust_before_add": int(len(no_added_df)),
            "adjusted_existing_only": int(len(adjusted_only_df)),
            "added_well_control_only": int(len(added_only_df)),
            "final_without_hard_fault_surface": int(len(no_hard_fault_df)),
            "final_all": int(len(corrected_df)),
        },
        "config_export_debug_step_vtks": bool(config.get("export_debug_step_vtks", False)),
    }
    (debug_dir / "debug_step_vtk_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return exported


def layer_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in df["LayerGroup"].value_counts(dropna=False).sort_index().items()}


def load_density_mass_from_initial_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = read_json(path)
    legacy_mass = {
        str(key): float(value)
        for key, value in payload.get("density_volume", {}).get("density_mass_by_layer", {}).items()
    }
    if legacy_mass:
        return legacy_mass
    return {
        str(layer): float(summary.get("density_mass", 0.0))
        for layer, summary in payload.get("macro_distribution_check", {}).items()
        if isinstance(summary, dict)
    }


def macro_distribution(density_mass: dict[str, float], initial_df: pd.DataFrame, corrected_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    total_density = float(sum(density_mass.values()))
    total_patches = int(len(corrected_df))
    total_initial = int(len(initial_df))
    patch_counts = corrected_df["LayerGroup"].value_counts().to_dict()
    initial_counts = initial_df["LayerGroup"].value_counts().to_dict()
    result: dict[str, dict[str, float]] = {}
    for layer in ALLOWED_LAYERS:
        target_share = (
            float(density_mass.get(layer, 0.0)) / total_density
            if total_density > 0
            else float(initial_counts.get(layer, 0)) / total_initial if total_initial > 0 else 0.0
        )
        patch_share = float(patch_counts.get(layer, 0)) / total_patches if total_patches > 0 else 0.0
        result[layer] = {
            "density_mass": float(density_mass.get(layer, 0.0)),
            "target_share": target_share,
            "target_share_basis": "density_mass" if total_density > 0 else "initial_patch_count",
            "initial_patch_count": int(initial_counts.get(layer, 0)),
            "initial_patch_share": float(initial_counts.get(layer, 0)) / total_initial if total_initial > 0 else 0.0,
            "patch_count": int(patch_counts.get(layer, 0)),
            "patch_count_share": patch_share,
            "absolute_share_difference": abs(target_share - patch_share),
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
    macro = macro_distribution(density_mass=density_mass, initial_df=initial_df, corrected_df=corrected_df)
    action_counts = {
        str(key): int(value)
        for key, value in audit_df["Action"].value_counts(dropna=False).to_dict().items()
    } if not audit_df.empty else {}
    changed_initial_count = int((corrected_df["CorrectionAction"].astype(str) == "adjust_to_well_control").sum())
    added_count = int((corrected_df["CorrectionAction"].astype(str) == "add_well_control_patch").sum())
    well_control_mask = safe_numeric(corrected_df.get("IsWellControlPatch", pd.Series(0, index=corrected_df.index))).fillna(0).astype(int).eq(1)
    well_control_df = corrected_df[well_control_mask].copy()
    before_mean = float(before_dist.mean()) if before_dist.notna().any() else None
    after_mean = float(after_dist.mean()) if after_dist.notna().any() else None
    before_median = float(before_dist.median()) if before_dist.notna().any() else None
    after_median = float(after_dist.median()) if after_dist.notna().any() else None
    changed_fraction = float(changed_initial_count + added_count) / max(float(len(initial_df)), 1.0)
    max_allowed_offset = float(config.get("well_control_center_offset_max_m", 45.0))
    applied_audit = audit_df[
        audit_df.get("Action", pd.Series("", index=audit_df.index)).astype(str).isin(
            ["adjust_to_well_control", "add_well_control_patch"]
        )
    ].copy()
    audit_after = safe_numeric(applied_audit["AfterMatchDistance"]) if "AfterMatchDistance" in applied_audit.columns else pd.Series(dtype=float)
    audit_offsets = safe_numeric(applied_audit["CenterOffsetM"]) if "CenterOffsetM" in applied_audit.columns else pd.Series(dtype=float)
    control_envelope = safe_numeric(applied_audit["ControlInsidePatchEnvelope"]).fillna(0) if "ControlInsidePatchEnvelope" in applied_audit.columns else pd.Series(dtype=float)
    layer_window_available = pd.Series(False, index=corrected_df.index)
    centers_within_layer_windows = True
    layer_window_missing_count = 0
    layer_window_checked_count = 0
    layer_window_surface_order_repaired_count = int(
        safe_numeric(
            corrected_df.get(
                "LayerWindowSurfaceOrderRepaired",
                pd.Series(0, index=corrected_df.index),
            )
        ).fillna(0).eq(1).sum()
    )
    if {"CenterTime", "TimeWindowMin", "TimeWindowMax"}.issubset(corrected_df.columns):
        modified_mask = corrected_df["CorrectionAction"].astype(str).isin(["adjust_to_well_control", "add_well_control_patch"])
        layer_window_available = corrected_df[["CenterTime", "TimeWindowMin", "TimeWindowMax"]].notna().all(axis=1) & modified_mask
        layer_window_missing_count = int((~layer_window_available).sum())
        layer_window_checked_count = int(layer_window_available.sum())
        if layer_window_checked_count > 0:
            centers_within_layer_windows = bool(
                corrected_df.loc[layer_window_available, "CenterTime"].between(
                    corrected_df.loc[layer_window_available, "TimeWindowMin"],
                    corrected_df.loc[layer_window_available, "TimeWindowMax"],
                ).all()
            )
    initial_dfn_vtk = (
        Path(str(config["initial_dfn_vtk"])).resolve()
        if config.get("initial_dfn_vtk")
        else None
    )
    original_fault_manifest = (
        Path(str(config["original_fault_manifest_csv"])).resolve()
        if config.get("original_fault_manifest_csv")
        else None
    )
    scale_geometry_preserved = {}
    for scale in ["medium", "large"]:
        before_scale = initial_df[initial_df["FractureScale"].astype(str).str.lower().eq(scale)]
        after_scale = corrected_df[corrected_df["FractureScale"].astype(str).str.lower().eq(scale)]
        scale_geometry_preserved[scale] = bool(
            len(before_scale) == len(after_scale)
            and patch_geometry_fingerprint(before_scale) == patch_geometry_fingerprint(after_scale)
        )
    checks = {
        "has_well_controls_in_target_block": int(len(control_df)) > 0,
        "linked_patch_centers_no_farther_than_before_mean": bool(
            not audit_after.empty and before_mean is not None and float(audit_after.mean()) <= before_mean
        ),
        "control_points_inside_patch_envelopes": bool(not control_envelope.empty and control_envelope.eq(1).all()),
        "well_control_center_offsets_within_limit": bool(not audit_offsets.empty and audit_offsets.fillna(np.inf).max() <= max_allowed_offset + 1.0e-6),
        "macro_distribution_preserved": bool(all(item["absolute_share_difference"] <= 0.05 for item in macro.values())),
        "far_field_preserved": bool(changed_fraction <= 0.15),
        "audit_rows_match_control_points": bool(len(audit_df) == len(control_df)),
        "md_join_audit_output_exists": bool(
            paths["md_join_audit_csv"].exists() and paths["md_join_audit_csv"].stat().st_size > 0
        ),
        "md_join_unmatched_output_exists": bool(paths["md_join_unmatched_csv"].exists()),
        "segment_coverage_output_exists": bool(
            paths["segment_coverage_csv"].exists() and paths["segment_coverage_csv"].stat().st_size > 0
        ),
        "md_join_unmatched_fraction_within_limit": bool(
            float(config.get("_md_join_qc", {}).get("unmatched_point_count", 0))
            <= float(
                dict(config.get("segment_join", {})).get(
                    "max_unmatched_fraction", config.get("md_join_max_unmatched_fraction", 0.02)
                )
            )
            * max(float(config.get("_md_join_qc", {}).get("step4_input_point_count", 0) + config.get("_md_join_qc", {}).get("step3_input_point_count", 0)), 1.0)
        ),
        # P0-4′/P0-4″：Step4 的井级输出自带 X/Y/TIME，Step8 不再做 MD 回接；
        # 这里改为检查"Step4 控制点坐标完整"与"Step3 成像组逐段回接无未命中"。
        "step4_control_geometry_complete": bool(
            int(config.get("_md_join_qc", {}).get("step4_input_point_count", 0)) > 0
            and int(config.get("_md_join_qc", {}).get("step4_matched_count", 0))
            == int(config.get("_md_join_qc", {}).get("step4_input_point_count", -1))
        ),
        "step3_imaging_join_has_no_unmatched": bool(
            int(config.get("_md_join_qc", {}).get("step3_input_point_count", 0)) > 0
            and int(config.get("_md_join_qc", {}).get("step3_matched_count", 0))
            == int(config.get("_md_join_qc", {}).get("step3_input_point_count", -1))
        ),
        "layers_limited_to_taigu_target_layers": bool(set(corrected_df["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS))),
        "modified_centers_within_layer_windows": centers_within_layer_windows,
        "medium_geometry_preserved": scale_geometry_preserved["medium"],
        "large_geometry_preserved": scale_geometry_preserved["large"],
        "orientation_fields_complete": bool(corrected_df[["AzimuthDeg", "DipDeg"]].notna().all().all()),
        "raw_vtk_output_exists": bool(paths["raw_vtk"].exists()),
        "initial_unified_dfn_vtk_exists": bool(
            initial_dfn_vtk is not None
            and initial_dfn_vtk.exists()
            and initial_dfn_vtk.stat().st_size > 0
        ),
        "original_fault_manifest_passthrough_exists": bool(
            original_fault_manifest is not None
            and original_fault_manifest.exists()
            and original_fault_manifest.stat().st_size > 0
        ),
        "unified_vtk_has_predicted_and_original_groups": bool(
            int(config.get("_unified_vtk_summary", {}).get("predicted_cell_count", -1)) == len(corrected_df)
            and int(config.get("_unified_vtk_summary", {}).get("original_fault_cell_count", 0)) > 0
        ),
        "original_fault_geometry_preserved": bool(
            config.get("_unified_vtk_summary", {}).get("original_fault_geometry_fingerprint")
            == config.get("_input_original_fault_fingerprint")
        ),
        "original_fault_render_area_uses_predicted_median": bool(
            config.get("_unified_vtk_write", {}).get("original_fault_render_area_policy")
            == "predicted_patch_area_median_not_physical_triangle_area"
        ),
        "imaging_well_region_audit_exists": bool(paths["region_audit_csv"].exists()),
        "imaging_well_region_correction_applied": bool(
            not config.get("enable_imaging_well_region_correction", False)
            or int(config.get("_imaging_well_region_summary", {}).get("corrected_patch_count", 0)) > 0
            or (
                int(config.get("_imaging_well_region_summary", {}).get("control_point_count", 0)) > 0
                and int(config.get("_imaging_well_region_summary", {}).get("within_radius_candidate_count", 0)) == 0
            )
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "initial_dfn_csv": str(Path(config["initial_dfn_csv"]).resolve()),
        "fracture_points_csv": str(Path(config["fracture_points_csv"]).resolve()),
        "real_well_samples_root": str(Path(config["real_well_samples_root"]).resolve()),
        "corrected_dfn_csv": str(paths["corrected_csv"]),
        "corrected_dfn_raw_vtk": str(paths["raw_vtk"]),
        "initial_dfn_vtk": str(initial_dfn_vtk) if initial_dfn_vtk is not None else "",
        "original_fault_manifest_csv": str(original_fault_manifest) if original_fault_manifest is not None else "",
        "well_control_correction_audit_csv": str(paths["audit_csv"]),
        "imaging_well_region_correction_audit_csv": str(paths["region_audit_csv"]),
        "well_control_input_qc_csv": str(paths["input_qc_csv"]),
        "well_control_outside_layer_audit_csv": str(paths["outside_layer_audit_csv"]),
        "summary_json": str(paths["summary_json"]),
        "target_block": config["target_block"],
        "correction_logic": {
            "step3_imaging_points_are_hard_controls": True,
            "step4_prediction_events_are_weak_controls": True,
            "adjust_existing_patch_when_local_candidate_available": True,
            "add_patch_when_unmatched_step3_or_high_confidence_step4": True,
            "far_field_density_volume_patches_unchanged": True,
            "original_fault_triangles_carried_unchanged_from_step7d_unified_vtk": True,
            "xy_search_radius_m": float(config.get("xy_search_radius_m", 80.0)),
            "time_search_radius_ms": float(config.get("time_search_radius_ms", 30.0)),
            "horizon_contract_table": str(Path(config["horizon_contract_table"]).resolve()),
            "surface_windows_recomputed_for_modified_small_patches": True,
            "modified_center_policy": "keep_assigned_layer_and_clamp_time_to_local_top_mid_base_window",
            "use_dip_geometry": bool(config.get("use_dip_geometry", False)),
            "geometry_time_scale_m_per_ms": float(config.get("geometry_time_scale_m_per_ms", 1.0)),
            "enable_well_control_center_offset": bool(config.get("enable_well_control_center_offset", True)),
            "well_control_center_offset_max_m": max_allowed_offset,
            "step3_imaging_gt_enabled": bool(config.get("step3_groups_root") or config.get("step3_imaging_group_csvs")),
            "replace_step4_inside_step3_imaging_windows": bool(config.get("replace_step4_inside_step3_imaging_windows", True)),
            "exclude_controls_outside_assigned_layer_window": True,
            "outside_layer_controls_are_audited_not_clamped": True,
            "outside_layer_control_count": int(config.get("_outside_layer_control_count", 0)),
            "eligible_control_count": int(config.get("_eligible_control_count", len(control_df))),
            "well_control_size": dict(config.get("well_control_size", {})),
            "force_well_control_small_scale": bool(config.get("force_well_control_small_scale", False)),
            "well_control_match_scales": list(config.get("well_control_match_scales", [])),
            "well_control_template_scales": list(config.get("well_control_template_scales", [])),
            "imaging_well_region_correction": dict(config.get("_imaging_well_region_summary", {})),
            "step4_small_orientation_policy": dict(config.get("step4_small_orientation_policy", {})),
            "step4_unmatched_addition": dict(config.get("step4_unmatched_addition", {})),
            "step4_event_aggregation": {
                "enabled": bool(config.get("enable_step4_event_aggregation", False)),
                "gap_ms": float(config.get("step4_event_gap_ms", 6.0)),
                "max_span_ms": float(config.get("step4_event_max_span_ms", 18.0)),
                "min_points": int(config.get("step4_event_min_points", 1)),
                "density_aggregation": str(config.get("step4_event_density_aggregation", "max")),
                "raw_step4_point_count": int(config.get("_raw_step4_control_count", 0)),
                "aggregated_step4_event_count": int(config.get("_aggregated_step4_control_count", 0)),
            },
        },
        "initial_dfn": {
            "patch_count": int(len(initial_df)),
            "layer_distribution": layer_counts(initial_df),
        },
        "well_controls": {
            "control_point_count": int(len(control_df)),
            "well_count": int(control_df["WellName"].nunique()) if not control_df.empty else 0,
            "layer_distribution": layer_counts(control_df) if not control_df.empty else {},
            "source_distribution": {
                str(key): int(value)
                for key, value in control_df["ControlSource"].value_counts(dropna=False).sort_index().items()
            } if "ControlSource" in control_df.columns else {},
            "step4_event_point_count_stats": finite_stats(control_df["EventPointCount"]) if "EventPointCount" in control_df.columns else finite_stats([]),
            "step4_event_time_span_ms_stats": finite_stats(control_df["EventTimeSpanMs"]) if "EventTimeSpanMs" in control_df.columns else finite_stats([]),
            "imaging_ground_truth_count": int(safe_numeric(control_df.get("IsImagingGroundTruth", pd.Series(dtype=float))).fillna(0).sum()),
            "temporary_neighbor_time_depth_count": int(safe_numeric(control_df.get("temporary_neighbor_time_depth", pd.Series(dtype=float))).fillna(0).sum()),
            "density_score_stats": finite_stats(control_df.get("DensityScore", pd.Series(dtype=float))),
            "input_horizon_qc": dict(config.get("_well_control_input_qc", {})),
            "md_join_qc": dict(config.get("_md_join_qc", {})),
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
            "well_control_scale_distribution": {
                str(key): int(value)
                for key, value in well_control_df.get("FractureScale", pd.Series(dtype=str)).astype(str).value_counts(dropna=False).sort_index().items()
            },
            "well_control_orientation_source_distribution": {
                str(key): int(value)
                for key, value in well_control_df.get("OrientationSource", pd.Series(dtype=str)).astype(str).value_counts(dropna=False).sort_index().items()
            },
            "well_control_dip_stats": finite_stats(well_control_df["DipDeg"]) if "DipDeg" in well_control_df else finite_stats([]),
            "center_x_stats": finite_stats(corrected_df["CenterX"]),
            "center_y_stats": finite_stats(corrected_df["CenterY"]),
            "center_time_stats": finite_stats(corrected_df["CenterTime"]),
        },
        "unified_vtk": {
            **dict(config.get("_unified_vtk_write", {})),
            **dict(config.get("_unified_vtk_summary", {})),
        },
        "match_quality": {
            "before_distance_stats": finite_stats(before_dist),
            "after_distance_stats": finite_stats(after_dist),
            "linked_patch_after_distance_stats": finite_stats(audit_after),
            "linked_patch_center_offset_m_stats": finite_stats(audit_offsets),
            "mean_distance_improvement": (before_mean - after_mean) if before_mean is not None and after_mean is not None else None,
            "median_distance_improvement": (before_median - after_median) if before_median is not None and after_median is not None else None,
        },
        "macro_distribution_check": macro,
        "layer_window_check": {
            "checked_patch_count": layer_window_checked_count,
            "missing_window_patch_count": layer_window_missing_count,
            "surface_order_repaired_patch_count": layer_window_surface_order_repaired_count,
            "note": "Only Step8-added or Step8-moved small patches are checked; unchanged Step7D geometry is passed through.",
        },
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    paths = output_paths(output_dir)
    ensure_dir(output_dir)
    flow_start = time.time()

    def log(message: str, reset: bool = True) -> None:
        elapsed = time.time() - flow_start
        print(f"[{elapsed:7.1f}s] {message}", flush=True)
        if reset:
            log.stage_start = time.time()

    log.stage_start = time.time()

    initial_csv = Path(config["initial_dfn_csv"]).resolve()
    initial_vtk = Path(config["initial_dfn_vtk"]).resolve()
    fracture_csv = Path(config["fracture_points_csv"]).resolve()
    samples_root = Path(config["real_well_samples_root"]).resolve()
    initial_summary_json = Path(config.get("initial_dfn_summary_json", "")).resolve()
    horizon_contract = Path(config["horizon_contract_table"]).resolve()
    for label, path in [("initial_dfn_csv", initial_csv), ("initial_dfn_vtk", initial_vtk), ("fracture_points_csv", fracture_csv), ("real_well_samples_root", samples_root), ("horizon_contract_table", horizon_contract)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    log("[Step8 1/7] 校验 Step7D 多尺度输入和原始断层几何")
    input_original_fault_mesh = extract_geometry_group(
        initial_vtk,
        ORIGINAL_FAULT_GEOMETRY_GROUP_CODE,
    ).triangulate()
    if input_original_fault_mesh.n_cells <= 0:
        raise RuntimeError("Step7D unified VTK contains no original-fault triangles")
    config["_input_original_fault_fingerprint"] = geometry_fingerprint(input_original_fault_mesh)

    initial_df = load_initial_dfn(initial_csv)
    log("[Step8 2/7] 读取 Step4 井级常规井预测点（P0-4′：输出自带 X/Y/TIME）")
    # 仅 Step3 成像组行需要逐段回接坐标；Step4 的井级输出直接带坐标。
    join_config = dict(config.get("segment_join", {}))
    tolerance_setting = join_config.get("tolerance_m", config.get("md_join_tolerance_m"))
    md_tolerance = (
        None
        if tolerance_setting is None or str(tolerance_setting).strip().lower() in {"", "auto"}
        else float(tolerance_setting)
    )
    tolerance_options = {
        "half_step_multiplier": float(join_config.get("half_step_multiplier", 0.5)),
        "tolerance_floor_m": float(join_config.get("tolerance_floor_m", 0.02)),
        "tolerance_cap_m": float(join_config.get("tolerance_cap_m", 0.5)),
    }
    merged_density_value = config.get("step4_merged_density_csv")
    merged_density_csv = (
        Path(str(merged_density_value)).resolve()
        if merged_density_value
        else fracture_csv.parent / "all_wells_merged_density_prediction.csv"
    )
    step4_control_df, step4_join_audit = load_control_points(
        fracture_csv,
        target_block=dict(config["target_block"]),
        merged_density_csv=merged_density_csv,
    )
    raw_step4_count = int(len(step4_control_df))
    step4_control_df = aggregate_step4_controls_to_events(step4_control_df, config=config)
    config["_raw_step4_control_count"] = raw_step4_count
    config["_aggregated_step4_control_count"] = int(len(step4_control_df))
    log("[Step8 3/7] 逐段回接 Step3 成像解释坐标")
    if config.get("step3_groups_root"):
        step3_group_paths = sorted(Path(config["step3_groups_root"]).resolve().glob("*.csv"))
    else:
        step3_group_paths = [Path(str(value)).resolve() for value in config.get("step3_imaging_group_csvs", [])]
    for path in step3_group_paths:
        if not path.exists():
            raise FileNotFoundError(f"step3_imaging_group_csvs entry does not exist: {path}")
    if step3_group_paths:
        step3_control_df, step3_join_audit = load_step3_imaging_controls(
            paths=step3_group_paths,
            samples_root=samples_root,
            target_block=dict(config["target_block"]),
            tolerance=md_tolerance,
            tolerance_options=tolerance_options,
        )
        step3_windows = load_step3_imaging_time_windows(step3_control_df)
        control_df = merge_step3_imaging_controls(
            step4_df=step4_control_df,
            step3_df=step3_control_df,
            windows=step3_windows,
            config=config,
        )
    else:
        control_df = step4_control_df
        step3_join_audit = pd.DataFrame()
    md_join_audit = pd.concat([step4_join_audit, step3_join_audit], ignore_index=True, sort=False)
    unmatched_audit = md_join_audit[
        md_join_audit["MDJoinStatus"].astype(str).ne("matched")
    ].copy() if not md_join_audit.empty else pd.DataFrame()
    if not md_join_audit.empty:
        md_join_audit.to_csv(paths["md_join_audit_csv"], index=False, encoding="utf-8-sig")
        unmatched_audit.to_csv(paths["md_join_unmatched_csv"], index=False, encoding="utf-8-sig")
    segment_coverage = build_segment_coverage_audit(
        samples_root,
        sorted(md_join_audit["WellName"].astype(str).unique()) if not md_join_audit.empty else [],
        {well: cached_well_segment_pool(samples_root, well) for well in sorted(
            md_join_audit["WellName"].astype(str).unique()
        )} if not md_join_audit.empty else {},
        md_join_audit,
    )
    if not segment_coverage.empty:
        segment_coverage.to_csv(paths["segment_coverage_csv"], index=False, encoding="utf-8-sig")
    config["_md_join_qc"] = {
        "tolerance_m": "auto_half_sampling_step" if md_tolerance is None else float(md_tolerance),
        "step4_geometry_policy": "step4_well_level_output_with_xy_time; no md re-attachment",
        "step4_geometry_source": (
            str(step4_join_audit["MDJoinReason"].iloc[0]) if not step4_join_audit.empty else ""
        ),
        "step4_input_point_count": int(len(step4_join_audit)),
        "step4_matched_count": int(step4_join_audit["MDJoinStatus"].astype(str).eq("matched").sum()) if not step4_join_audit.empty else 0,
        "step4_used_count": int(safe_numeric(step4_join_audit.get("UsedForStep8", pd.Series(dtype=float))).fillna(0).sum()) if not step4_join_audit.empty else 0,
        "step3_segment_selection_policy": "per_segment_exact_within_batch",
        "step3_input_point_count": int(len(step3_join_audit)),
        "step3_matched_count": int(step3_join_audit["MDJoinStatus"].astype(str).eq("matched").sum()) if not step3_join_audit.empty else 0,
        "step3_used_count": int(safe_numeric(step3_join_audit.get("UsedForStep8", pd.Series(dtype=float))).fillna(0).sum()) if not step3_join_audit.empty else 0,
        "unmatched_point_count": int(len(unmatched_audit)),
        "unmatched_reason_counts": unmatched_audit["MDJoinReason"].value_counts().to_dict() if not unmatched_audit.empty else {},
        "join_status_counts": md_join_audit["MDJoinStatus"].value_counts().to_dict() if not md_join_audit.empty else {},
        "selected_segment_counts": {
            str(key): int(value)
            for key, value in md_join_audit.loc[
                md_join_audit["SelectedSegmentID"].astype(str).ne(""), "SelectedSegmentID"
            ].value_counts().items()
        } if not md_join_audit.empty else {},
        "segment_count_total": int(len(segment_coverage)),
        "segments_with_gaps": int(safe_numeric(segment_coverage.get("GapCount", pd.Series(dtype=float))).fillna(0).gt(0).sum()) if not segment_coverage.empty else 0,
        "single_row_segment_count": int(safe_numeric(segment_coverage.get("Rows", pd.Series(dtype=float))).fillna(0).le(2).sum()) if not segment_coverage.empty else 0,
        "step4_join_summary": summarize_join(step4_join_audit),
        "step3_join_summary": summarize_join(step3_join_audit),
    }
    log(
        f"[Step8 3a/7] 测井段回接：Step4 {len(step4_join_audit)} 点（匹配 "
        f"{config['_md_join_qc']['step4_matched_count']}，采用 {config['_md_join_qc']['step4_used_count']}），"
        f"Step3 {len(step3_join_audit)} 点（采用 {config['_md_join_qc']['step3_used_count']}），"
        f"未回接 {len(unmatched_audit)}",
    )
    initial_df, control_df = add_density_scores(initial_df, control_df)
    tracks = load_real_well_tracks(samples_root, set(control_df["WellName"].astype(str).unique()))
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    surface_lookup = build_spatial_lookup(config)
    input_qc = control_df.copy()
    input_qc["NearestTrackDistance"] = nearest_track_distances(control_df, tracks, time_scale=time_scale)
    horizon_rows: list[dict[str, Any]] = []
    for _, control in input_qc.iterrows():
        local = surface_lookup.query(float(control["X"]), float(control["Y"]))
        layer = str(control["LayerGroup"])
        top = local["TopTimeMs"] if layer == "上部复合层" else local["MidTimeMs"]
        base = local["MidTimeMs"] if layer == "上部复合层" else local["BaseTimeMs"]
        horizon_rows.append({
            "NearestTraceIdx": local["TraceIdx"],
            "NearestTraceDistanceM": local["DistanceM"],
            "AssignedLayerTopTimeMs": top,
            "AssignedLayerBaseTimeMs": base,
            "ControlTimeInsideAssignedLayer": int(np.isfinite(top) and np.isfinite(base) and top <= float(control["TIME"]) <= base),
        })
    input_qc = pd.concat([input_qc.reset_index(drop=True), pd.DataFrame(horizon_rows)], axis=1)
    input_qc.to_csv(paths["input_qc_csv"], index=False, encoding="utf-8-sig")
    inside_mask = input_qc["ControlTimeInsideAssignedLayer"].astype(int).eq(1)
    outside_layer_audit = input_qc.loc[~inside_mask].copy()
    outside_layer_audit["ExclusionReason"] = "control_time_outside_assigned_local_layer_window"
    outside_layer_audit["ControlUsedForStep8Correction"] = 0
    outside_layer_audit.to_csv(paths["outside_layer_audit_csv"], index=False, encoding="utf-8-sig")
    config["_well_control_input_qc"] = {
        "control_count": int(len(input_qc)),
        "inside_assigned_layer_count": int(input_qc["ControlTimeInsideAssignedLayer"].sum()),
        "outside_assigned_layer_count": int((input_qc["ControlTimeInsideAssignedLayer"] == 0).sum()),
        "temporary_time_depth_count": int(safe_numeric(input_qc["temporary_neighbor_time_depth"]).fillna(0).sum()),
    }
    # Keep layer-outside observations for audit, but do not use them as hard
    # controls and never clamp them onto a horizon boundary.
    eligible_positions = np.flatnonzero(inside_mask.to_numpy())
    control_df = control_df.iloc[eligible_positions].reset_index(drop=True)
    config["_outside_layer_control_count"] = int((~inside_mask).sum())
    config["_eligible_control_count"] = int(len(control_df))
    log(
        f"[Step8 3b/7] 层位窗口筛选：原始控制点 {len(input_qc)}，"
        f"保留 {len(control_df)}，层外审计 {len(outside_layer_audit)}",
    )
    track_dist = nearest_track_distances(control_df, tracks, time_scale=time_scale)
    before_dist = nearest_patch_distances(initial_df, control_df, time_scale=time_scale)
    log(f"[Step8 4/7] 执行井控匹配：控制事件 {len(control_df)} 个")
    corrected_df, audit_df = apply_well_controls(
        initial_df=initial_df,
        control_df=control_df,
        track_distances=track_dist,
        config=config,
        surface_lookup=surface_lookup,
    )
    corrected_df, region_audit_df = apply_imaging_well_region_correction(
        patch_df=corrected_df,
        control_df=control_df,
        config=config,
    )
    config["_imaging_well_region_summary"] = {
        "enabled": bool(config.get("enable_imaging_well_region_correction", False)),
        "positive_evidence_only": True,
        "xy_radius_m": float(config.get("imaging_well_region_xy_radius_m", 150.0)),
        "time_radius_ms": float(config.get("imaging_well_region_time_radius_ms", 40.0)),
        "density_blend": float(config.get("imaging_well_region_density_blend", 0.65)),
        "corrected_patch_count": int(len(region_audit_df)),
        "control_point_count": int(config.get("_imaging_well_region_eligibility", {}).get("control_point_count", 0)),
        "candidate_patch_count": int(config.get("_imaging_well_region_eligibility", {}).get("candidate_patch_count", 0)),
        "within_radius_candidate_count": int(config.get("_imaging_well_region_eligibility", {}).get("within_radius_candidate_count", 0)),
        "mean_influence": float(region_audit_df["Influence"].mean()) if not region_audit_df.empty else 0.0,
        "mean_density_increase": float(
            (region_audit_df["CorrectedSourceDensity"] - region_audit_df["OriginalSourceDensity"]).mean()
        ) if not region_audit_df.empty else 0.0,
    }
    log("[Step8 5/7] 检查新增/移动小尺度片的太古界层位合同")
    corrected_df, final_horizon_qc = enforce_final_center_horizon_contract(corrected_df, surface_lookup)
    after_dist = nearest_patch_distances(corrected_df, control_df, time_scale=time_scale)

    display_z_scale = float(config.get("display_z_scale", 5.0))
    use_dip_geometry = bool(config.get("use_dip_geometry", False))
    geometry_time_scale = float(config.get("geometry_time_scale_m_per_ms", 1.0))
    corrected_df = refresh_well_control_vertices(
        corrected_df,
        use_dip_geometry=use_dip_geometry,
        geometry_time_scale_m_per_ms=geometry_time_scale,
    )
    corrected_df, vertex_horizon_qc = materialize_and_clip_vertices_to_horizon_contract(
        corrected_df,
        lookup=surface_lookup,
        use_dip_geometry=use_dip_geometry,
        geometry_time_scale_m_per_ms=geometry_time_scale,
    )
    log("[Step8 6/7] 写出预测裂缝和原始断层统一 VTK")
    final_horizon_qc["vertex_geometry"] = vertex_horizon_qc
    corrected_df.to_csv(paths["corrected_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    region_audit_df.to_csv(paths["region_audit_csv"], index=False, encoding="utf-8-sig")
    predicted_vtk = output_dir / ".well_corrected_predicted_only_raw_time.vtk"
    write_legacy_vtk(
        predicted_vtk,
        corrected_df,
        "well_corrected_predicted_only_raw_time",
        display=False,
        display_z_scale=display_z_scale,
        use_dip_geometry=use_dip_geometry,
        geometry_time_scale_m_per_ms=geometry_time_scale,
    )
    config["_unified_vtk_write"] = write_unified_dfn_vtk(
        paths["raw_vtk"],
        predicted_vtk,
        initial_vtk,
    )
    predicted_vtk.unlink(missing_ok=True)
    config["_unified_vtk_summary"] = unified_vtk_summary(paths["raw_vtk"])
    debug_vtks: dict[str, str] = {}
    if bool(config.get("export_debug_step_vtks", False)):
        debug_vtks = export_debug_step_vtks(
            output_dir=output_dir,
            initial_df=initial_df,
            corrected_df=corrected_df,
            config=config,
            display_z_scale=display_z_scale,
            use_dip_geometry=use_dip_geometry,
            geometry_time_scale_m_per_ms=geometry_time_scale,
        )

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
    if debug_vtks:
        summary["debug_step_vtks"] = debug_vtks
    summary["horizon_contract_qc"] = final_horizon_qc
    summary.setdefault("checks", {})["all_modified_patch_vertices_inside_local_target_layer"] = bool(
        vertex_horizon_qc["all_modified_vertices_inside_local_layer_window"]
    )
    summary["status"] = "pass" if all(bool(value) for value in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("[Step8 7/7] 完成输出和自动验收")
    print(f"Corrected DFN CSV: {paths['corrected_csv']}")
    print(f"Corrected DFN raw VTK: {paths['raw_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Control points: {len(control_df)} corrected patches: {len(corrected_df)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
