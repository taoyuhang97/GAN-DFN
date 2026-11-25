from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_well_control_correction.json"
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import load_surface_tables  # noqa: E402
from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    HorizonSpatialLookup,
    build_spatial_lookup,
)
from common.unified_dfn_vtk import (  # noqa: E402
    ORIGINAL_FAULT_GEOMETRY_GROUP_CODE,
    extract_geometry_group,
    geometry_fingerprint,
    unified_vtk_summary,
    write_unified_dfn_vtk,
)
from common.strata_framework import load_framework  # noqa: E402

_large_control_path = FORMAL_ROOT.parents[1] / "common" / "large_fault_conventional_controls.py"
_large_control_spec = importlib.util.spec_from_file_location(
    "sand_step8_large_conventional_controls", _large_control_path
)
if _large_control_spec is None or _large_control_spec.loader is None:
    raise ImportError(f"cannot load large conventional control helper: {_large_control_path}")
large_conventional_controls = importlib.util.module_from_spec(_large_control_spec)
_large_control_spec.loader.exec_module(large_conventional_controls)

_match_density_path = (
    FORMAL_ROOT.parents[1] / "common" / "step8_well_match_density.py"
)
_match_density_spec = importlib.util.spec_from_file_location(
    "sand_step8_well_match_density", _match_density_path
)
if _match_density_spec is None or _match_density_spec.loader is None:
    raise ImportError(f"cannot load Step8 match/density helper: {_match_density_path}")
step8_match_density = importlib.util.module_from_spec(_match_density_spec)
_match_density_spec.loader.exec_module(step8_match_density)

_FRAMEWORK = load_framework()
ALLOWED_LAYERS = list(_FRAMEWORK.layer_names())
LAYER_CODE = _FRAMEWORK.layer_code_map()
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct Step 8 initial DFN with hard real-well fracture controls.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent_name = payload.pop("_extends", None)
    if not parent_name:
        return payload
    parent_path = (path.parent / str(parent_name)).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Step8 config parent does not exist: {parent_path}")
    parent = read_json(parent_path)
    parent.update(payload)
    return parent


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
        "summary_json": output_dir / "well_corrected_dfn_summary.json",
        "audit_csv": output_dir / "well_control_correction_audit.csv",
        "region_audit_csv": output_dir / "imaging_well_region_correction_audit.csv",
        "imaging_match_json": output_dir / "imaging_match_summary.json",
        "imaging_match_selection_csv": output_dir / "imaging_match_selection_audit.csv",
        "density_compensation_requests_csv": output_dir / "density_compensation_requests.csv",
        "density_compensation_summary_csv": output_dir / "density_compensation_ring_summary.csv",
        "seismic_guided_audit_csv": output_dir / "seismic_guided_densification_audit.csv",
        "seismic_guided_summary_json": output_dir / "seismic_guided_densification_summary.json",
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
    out["IsImagingMatchPatch"] = 0
    out["IsDensityCompensationPatch"] = 0
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
    out["ControlSource"] = "step4_predicted"
    out["IsImagingGroundTruth"] = 0
    return out


def load_step3_imaging_controls(paths: list[Path], well_name: str, target_block: dict[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        df = read_csv_flexible(path, low_memory=False)
        required = {"X", "Y", "TIME", "GT_POINT_FLAG"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Step3 imaging group csv missing columns {missing}: {path}")
        work = df[pd.to_numeric(df["GT_POINT_FLAG"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
        work["WellName"] = well_name
        work["Step3GroupCSV"] = str(path)
        parts.append(work)
    if not parts:
        return pd.DataFrame()
    work = pd.concat(parts, ignore_index=True)
    for column in ["X", "Y", "TIME", "TVD", "DEPT", "Density", "Frac_Azimuth", "Frac_Dip"]:
        if column in work.columns:
            work[column] = safe_numeric(work[column])
    work = work.dropna(subset=["X", "Y", "TIME"]).copy()
    x_min = float(target_block["x_min"])
    x_max = float(target_block["x_max"])
    y_min = float(target_block["y_min"])
    y_max = float(target_block["y_max"])
    work = work[work["X"].between(x_min, x_max) & work["Y"].between(y_min, y_max)].copy()
    layer_col = "StrataName" if "StrataName" in work.columns else "LayerGroup" if "LayerGroup" in work.columns else None
    if layer_col is None:
        raise ValueError("Step3 imaging group csv missing StrataName/LayerGroup")
    work["LayerGroup"] = work[layer_col].astype(str)
    work = work[work["LayerGroup"].isin(ALLOWED_LAYERS)].copy()
    if work.empty:
        return pd.DataFrame()
    if "SampleID" in work.columns:
        sample_id = work["SampleID"].astype(str)
    else:
        sample_id = pd.Series(["step3_gt_%06d" % (idx + 1) for idx in range(len(work))], index=work.index)
    work["WellControlSampleID"] = "step3_gt_" + sample_id
    work["ControlSource"] = "step3_imaging_gt"
    work["IsImagingGroundTruth"] = 1
    keep = [
        "WellName",
        "X",
        "Y",
        "TIME",
        "TVD",
        "DEPT",
        "LayerGroup",
        "Density",
        "WellControlSampleID",
        "ControlSource",
        "IsImagingGroundTruth",
        "Frac_Azimuth",
        "Frac_Dip",
        "RawPointCount",
        "Step3GroupCSV",
    ]
    keep = [column for column in keep if column in work.columns]
    return work[keep].sort_values(["WellName", "LayerGroup", "TIME", "WellControlSampleID"]).reset_index(drop=True)


def load_step3_imaging_time_windows(paths: list[Path], well_name: str, target_block: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        df = read_csv_flexible(path, low_memory=False)
        required = {"X", "Y", "TIME"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Step3 imaging group csv missing columns {missing}: {path}")
        layer_col = "StrataName" if "StrataName" in df.columns else "LayerGroup" if "LayerGroup" in df.columns else None
        if layer_col is None:
            raise ValueError("Step3 imaging group csv missing StrataName/LayerGroup")
        work = df.copy()
        for column in ["X", "Y", "TIME"]:
            work[column] = safe_numeric(work[column])
        work = work.dropna(subset=["X", "Y", "TIME"]).copy()
        work = work[
            work["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
            & work["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
        ].copy()
        if work.empty:
            continue
        layer = str(work[layer_col].dropna().astype(str).mode().iloc[0]) if work[layer_col].notna().any() else ""
        if layer not in ALLOWED_LAYERS:
            continue
        rows.append(
            {
                "WellName": well_name,
                "LayerGroup": layer,
                "TimeMin": float(work["TIME"].min()),
                "TimeMax": float(work["TIME"].max()),
                "Step3GroupCSV": str(path),
            }
        )
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


def load_surface_time_lookup(surface_dir: Path | None) -> dict[str, Any] | None:
    if surface_dir is None or not surface_dir.exists():
        return None
    payload: dict[str, Any] = {}
    surfaces = load_surface_tables(surface_dir)
    for code in ["T4", "T6", "T7"]:
        table = surfaces[code]["table"]
        xy = table[["X", "Y"]].to_numpy(dtype=float)
        time = table["Z"].to_numpy(dtype=float)
        payload[code] = {"tree": cKDTree(xy), "time": time}
    return payload


def query_surface_time(surface_lookup: dict[str, Any], code: str, x: float, y: float) -> float:
    item = surface_lookup[code]
    _, idx = item["tree"].query(np.asarray([[float(x), float(y)]]), k=1, p=1)
    return float(item["time"][int(idx[0])])


def update_layer_window_from_surfaces(
    row: pd.Series | dict[str, Any],
    surface_lookup: HorizonSpatialLookup | None,
) -> dict[str, float | int]:
    if surface_lookup is None:
        return {}
    layer = str(row["LayerGroup"])
    x = float(row["CenterX"])
    y = float(row["CenterY"])
    local = surface_lookup.query(x, y)
    try:
        layer_item = _FRAMEWORK.layer(layer)
    except Exception:
        return {}
    top_column, base_column = _FRAMEWORK.surface_pair(layer_item)
    top = float(local[top_column])
    base = float(local[base_column])
    present = bool(local[layer_item.presence_column])
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
        "LocalT4Time": float(local["T4"]),
        "LocalT6Time": float(local["T6"]),
        "LocalT7Time": float(local["T7"]),
        "LocalShasanPresent": int(local["ShasanPresent"]),
        "LocalShasiPresent": int(local["ShasiPresent"]),
    }


def enforce_final_center_horizon_contract(
    frame: pd.DataFrame,
    lookup: HorizonSpatialLookup,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[pd.Series] = []
    rejected = 0
    reassigned = 0
    kept_center_outside_voxel_coverage = 0
    for _, source_row in frame.iterrows():
        row = source_row.copy()
        x = float(row["CenterX"])
        y = float(row["CenterY"])
        time = float(row["CenterTime"])
        interval = lookup.interval(x, y, time)
        if interval is None:
            if str(row.get("WindowValidationMode", "")).strip() == "voxel_coverage":
                # Panel body validated by evidence-voxel window coverage in Step7C;
                # keep it unchanged so mine-scale display preserves the fault.
                kept_center_outside_voxel_coverage += 1
                rows.append(row)
                continue
            rejected += 1
            continue
        expected_layer = _FRAMEWORK.layer_by_interval(interval).name
        if str(row["LayerGroup"]) != expected_layer:
            reassigned += 1
        row["LayerGroup"] = expected_layer
        row["LayerCode"] = LAYER_CODE[expected_layer]
        row.update(update_layer_window_from_surfaces(row, lookup))
        rows.append(row)
    output = pd.DataFrame(rows)
    return output.reset_index(drop=True), {
        "input_count": int(len(frame)),
        "kept_count": int(len(output)),
        "rejected_outside_local_t4_t7_count": int(rejected),
        "layer_reassigned_from_local_t6_count": int(reassigned),
        "kept_center_outside_voxel_coverage_count": int(kept_center_outside_voxel_coverage),
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


SEISMIC_ATTRIBUTE_COLUMNS = {
    "AntTrack": "SeismicAntTrack",
    "Coherence": "SeismicCoherence",
    "CurvatureMax": "SeismicCurvatureMax",
}


def _nearest_axis_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.searchsorted(axis, values, side="left")
    right = np.clip(right, 1, len(axis) - 1)
    left = right - 1
    choose_left = np.abs(values - axis[left]) <= np.abs(axis[right] - values)
    return np.where(choose_left, left, right).astype(np.int64)


class SandstoneSeismicAttributeSampler:
    """Sample the three sandstone attribute volumes on their native 12.5 m grid."""

    def __init__(self, mapping_path: Path, volume_paths: dict[str, Path], rules: dict[str, dict[str, float]]):
        with np.load(mapping_path, allow_pickle=False) as packed:
            ix = packed["ix"].astype(np.int64)
            iy = packed["iy"].astype(np.int64)
            x = packed["x"].astype(float)
            y = packed["y"].astype(float)
            source_trace_idx = packed["source_trace_idx"].astype(np.int64)
        self.nx = int(ix.max()) + 1
        self.ny = int(iy.max()) + 1
        self.x_axis = np.full(self.nx, np.nan, dtype=float)
        self.y_axis = np.full(self.ny, np.nan, dtype=float)
        self.x_axis[ix] = x
        self.y_axis[iy] = y
        if not np.isfinite(self.x_axis).all() or not np.isfinite(self.y_axis).all():
            raise ValueError("sand seismic trace mapping does not cover a complete X/Y lattice")
        if not np.all(np.diff(self.x_axis) > 0.0) or not np.all(np.diff(self.y_axis) > 0.0):
            raise ValueError("sand seismic trace mapping axes must be strictly increasing")
        self.trace_by_ij = np.full((self.ny, self.nx), -1, dtype=np.int64)
        self.trace_by_ij[iy, ix] = source_trace_idx
        if np.any(self.trace_by_ij < 0):
            raise ValueError("sand seismic trace mapping contains missing lattice nodes")
        self.handles: dict[str, Any] = {}
        self.samples: dict[str, np.ndarray] = {}
        self.rules = dict(rules)
        for name, path in volume_paths.items():
            handle = segyio.open(str(Path(path).resolve()), "r", ignore_geometry=True)
            handle.mmap()
            self.handles[name] = handle
            self.samples[name] = np.asarray(handle.samples, dtype=float)
        self.trace_cache: dict[tuple[str, int], np.ndarray] = {}

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.trace_cache.clear()

    def _trace(self, name: str, trace_index: int) -> np.ndarray:
        key = (name, int(trace_index))
        value = self.trace_cache.get(key)
        if value is None:
            value = np.asarray(self.handles[name].trace[int(trace_index)], dtype=float)
            self.trace_cache[key] = value
        return value

    def sample_points(self, points: pd.DataFrame) -> pd.DataFrame:
        x = safe_numeric(points["X"]).to_numpy(dtype=float)
        y = safe_numeric(points["Y"]).to_numpy(dtype=float)
        time_value = safe_numeric(points["TIME"]).to_numpy(dtype=float)
        valid_xy = np.isfinite(x) & np.isfinite(y)
        ix = np.zeros(len(points), dtype=np.int64)
        iy = np.zeros(len(points), dtype=np.int64)
        ix[valid_xy] = _nearest_axis_index(self.x_axis, x[valid_xy])
        iy[valid_xy] = _nearest_axis_index(self.y_axis, y[valid_xy])
        trace_ids = np.full(len(points), -1, dtype=np.int64)
        trace_ids[valid_xy] = self.trace_by_ij[iy[valid_xy], ix[valid_xy]]
        output: dict[str, np.ndarray] = {}
        for name, column in SEISMIC_ATTRIBUTE_COLUMNS.items():
            values = np.full(len(points), np.nan, dtype=float)
            samples = self.samples[name]
            time_index = np.searchsorted(samples, time_value, side="left")
            in_time = np.isfinite(time_value) & (time_value >= samples[0]) & (time_value <= samples[-1])
            time_index = np.clip(time_index, 0, len(samples) - 1)
            for row_index in np.flatnonzero(valid_xy & in_time & (trace_ids >= 0)):
                trace = self._trace(name, int(trace_ids[row_index]))
                candidate_index = int(time_index[row_index])
                if candidate_index + 1 < len(samples):
                    if samples[candidate_index + 1] - time_value[row_index] < time_value[row_index] - samples[candidate_index]:
                        candidate_index += 1
                values[row_index] = float(trace[candidate_index])
            values[~np.isfinite(values) | (values <= -1.0e5)] = np.nan
            rule = self.rules.get(name) or {}
            if rule:
                values[
                    (values < float(rule.get("min", -np.inf)))
                    | (values > float(rule.get("max", np.inf)))
                ] = np.nan
            output[column] = values
        return pd.DataFrame(output, index=points.index)


def _robust_normalization_scale(values: np.ndarray, floor: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return max(float(floor), 1.0e-9)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return max(1.4826 * mad, float(floor), 1.0e-9)


def build_seismic_guided_density_patches(
    patch_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    control_df: pd.DataFrame,
    config: dict[str, Any],
    surface_lookup: HorizonSpatialLookup | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add steep 3D patches selected by seismic-signature similarity.

    The candidate centers are distributed through three radial bands and are
    displaced in TWT along the local dip direction. They are never copies at
    the control point's original horizontal time slice.
    """

    spec = dict(config.get("seismic_guided_densification") or {})
    enabled = bool(spec.get("enabled", False))
    if not enabled:
        return pd.DataFrame(), pd.DataFrame(), {
            "enabled": False,
            "status": "disabled",
            "created_patch_count": 0,
        }
    well_names = {str(value) for value in spec.get("control_well_names", [])}
    controls = control_df[control_df["WellName"].astype(str).isin(well_names)].copy()
    if controls.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "enabled": True,
            "status": "no_matching_control_wells",
            "control_well_names": sorted(well_names),
            "created_patch_count": 0,
        }

    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    bands = [
        (float(row[0]), float(row[1]))
        for row in spec.get(
            "radial_bands_m",
            [[0.0, 50.0], [50.0, 100.0], [100.0, 200.0]],
        )
    ]
    scale_cycle = [
        str(value).strip().lower()
        for value in spec.get("scale_cycle", ["small", "medium", "small"])
    ]
    allowed_scales = {
        str(value).strip().lower()
        for value in spec.get("template_scales", ["small", "medium"])
    }
    time_offsets = [
        float(value)
        for value in spec.get("center_time_offsets_ms", [-8.0, -5.0, 5.0, 8.0])
    ]
    min_time_offset = abs(float(spec.get("minimum_abs_time_offset_ms", 3.0)))
    min_separation = float(spec.get("minimum_3d_separation_m", 12.0))
    min_dip = float(spec.get("minimum_template_dip_deg", 35.0))
    template_xy_radius = float(spec.get("template_xy_radius_m", 350.0))
    template_time_radius = float(spec.get("template_time_radius_ms", 40.0))
    template_limit = int(spec.get("max_local_templates_per_scale", 20))
    minimum_radius = float(spec.get("minimum_radius_m", 10.0))
    minimum_similarity = float(spec.get("minimum_similarity", 0.0))
    size_multiplier = float(spec.get("patch_size_multiplier", 0.9))
    height_multiplier = float(spec.get("patch_height_multiplier", 0.9))
    max_total = int(spec.get("max_total_patches", 300))
    attribute_weights = {
        str(key): float(value)
        for key, value in spec.get(
            "attribute_weights",
            {"AntTrack": 0.4, "Coherence": 0.4, "CurvatureMax": 0.2},
        ).items()
    }
    attribute_floors = {
        str(key): float(value)
        for key, value in spec.get(
            "attribute_normalization_floor",
            {"AntTrack": 0.2, "Coherence": 10.0, "CurvatureMax": 0.01},
        ).items()
    }
    volume_paths = {
        str(name): Path(str(path)).resolve()
        for name, path in dict(spec["volume_paths"]).items()
    }
    for name, path in volume_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"seismic_guided_densification.volume_paths[{name}] does not exist: {path}")
    mapping_path = Path(str(spec["trace_mapping_npz"])).resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(f"seismic-guided trace mapping does not exist: {mapping_path}")

    sampler = SandstoneSeismicAttributeSampler(
        mapping_path=mapping_path,
        volume_paths=volume_paths,
        rules=dict(spec.get("attribute_rules", {})),
    )
    patch_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    serial = 0
    try:
        for (well_name, layer), well_controls in controls.sort_values(["WellName", "LayerGroup", "TIME"]).groupby(
            ["WellName", "LayerGroup"],
            dropna=False,
        ):
            layer_pool = initial_df[
                initial_df["LayerGroup"].astype(str).eq(str(layer))
                & initial_df["FractureScale"].astype(str).str.lower().isin(allowed_scales)
                & safe_numeric(initial_df["DipDeg"]).ge(min_dip)
            ].copy()
            if layer_pool.empty:
                continue
            occupied = patch_df[patch_df["LayerGroup"].astype(str).eq(str(layer))].copy()
            occupied_coords = np.column_stack(
                [
                    safe_numeric(occupied["CenterX"]).to_numpy(dtype=float),
                    safe_numeric(occupied["CenterY"]).to_numpy(dtype=float),
                    safe_numeric(occupied["CenterTime"]).to_numpy(dtype=float) * time_scale,
                ]
            )
            occupied_coords = occupied_coords[np.isfinite(occupied_coords).all(axis=1)]
            occupied_tree = cKDTree(occupied_coords)
            for _, event in well_controls.sort_values("TIME").iterrows():
                event_attrs = sampler.sample_points(
                    pd.DataFrame(
                        [{"X": event["X"], "Y": event["Y"], "TIME": event["TIME"]}]
                    )
                ).iloc[0]
                for band_index, (ring_min, ring_max) in enumerate(bands):
                    if len(patch_rows) >= max_total:
                        break
                    scale = scale_cycle[band_index % len(scale_cycle)]
                    if scale not in allowed_scales:
                        continue
                    local = layer_pool[
                        layer_pool["FractureScale"].astype(str).str.lower().eq(scale)
                    ].copy()
                    if local.empty:
                        continue
                    local = local[
                        safe_numeric(local["CenterX"]).sub(float(event["X"])).abs().le(template_xy_radius)
                        & safe_numeric(local["CenterY"]).sub(float(event["Y"])).abs().le(template_xy_radius)
                        & safe_numeric(local["CenterTime"]).sub(float(event["TIME"])).abs().le(template_time_radius)
                    ].copy()
                    if local.empty:
                        local = layer_pool[
                            layer_pool["FractureScale"].astype(str).str.lower().eq(scale)
                        ].copy()
                        local["_distance"] = np.hypot(
                            safe_numeric(local["CenterX"]).sub(float(event["X"])),
                            safe_numeric(local["CenterY"]).sub(float(event["Y"])),
                        ) + np.abs(safe_numeric(local["CenterTime"]).sub(float(event["TIME"]))) * time_scale
                        local = local.nsmallest(min(template_limit, len(local)), "_distance")
                    else:
                        local["_distance"] = np.hypot(
                            safe_numeric(local["CenterX"]).sub(float(event["X"])),
                            safe_numeric(local["CenterY"]).sub(float(event["Y"])),
                        ) + np.abs(safe_numeric(local["CenterTime"]).sub(float(event["TIME"]))) * time_scale
                        local = local.nsmallest(min(template_limit, len(local)), "_distance")

                    local_dip = float(np.nanmedian(safe_numeric(local["DipDeg"])))
                    if not np.isfinite(local_dip) or local_dip < min_dip:
                        continue
                    azimuth_column = (
                        "DipAzimuthDeg"
                        if "DipAzimuthDeg" in local.columns and safe_numeric(local["DipAzimuthDeg"]).notna().any()
                        else "AzimuthDeg"
                    )
                    local_azimuth = float(np.nanmedian(safe_numeric(local[azimuth_column])))
                    if not np.isfinite(local_azimuth):
                        local_azimuth = 0.0
                    if azimuth_column == "AzimuthDeg":
                        local_azimuth += 90.0
                    dip_direction = np.asarray(
                        [
                            -np.sin(np.deg2rad(local_azimuth)),
                            np.cos(np.deg2rad(local_azimuth)),
                        ],
                        dtype=float,
                    )
                    strike_direction = np.asarray(
                        [-dip_direction[1], dip_direction[0]],
                        dtype=float,
                    )
                    effective_min = max(ring_min, minimum_radius if band_index == 0 else ring_min)
                    radius = 0.5 * (effective_min + ring_max)
                    candidate_records: list[dict[str, Any]] = []
                    for time_offset in time_offsets:
                        if abs(time_offset) < min_time_offset:
                            continue
                        dip_shift = time_offset * time_scale / max(np.tan(np.deg2rad(local_dip)), 1.0e-6)
                        if abs(dip_shift) >= radius:
                            continue
                        strike_shift = math.sqrt(max(radius * radius - dip_shift * dip_shift, 0.0))
                        for strike_sign in (-1.0, 1.0):
                            xy_offset = dip_shift * dip_direction + strike_sign * strike_shift * strike_direction
                            template = local.iloc[
                                min(
                                    int(
                                        2 * band_index
                                        + (0 if strike_sign < 0 else 1)
                                        + int(round(abs(time_offset)))
                                    ),
                                    len(local) - 1,
                                )
                            ]
                            candidate_records.append(
                                {
                                    "WellName": str(well_name),
                                    "LayerGroup": str(layer),
                                    "WellControlSampleID": str(event["WellControlSampleID"]),
                                    "RadialBandLabel": f"{ring_min:.0f}-{ring_max:.0f}m",
                                    "RadialBandMinM": float(ring_min),
                                    "RadialBandMaxM": float(ring_max),
                                    "RadialDistanceM": float(radius),
                                    "RequestedScale": scale,
                                    "X": float(event["X"]) + float(xy_offset[0]),
                                    "Y": float(event["Y"]) + float(xy_offset[1]),
                                    "TIME": float(event["TIME"]) + float(time_offset),
                                    "TimeOffsetFromControlMs": float(time_offset),
                                    "TemplatePatchID": str(template["PatchID"]),
                                    "_template": template,
                                }
                            )
                    if not candidate_records:
                        continue
                    candidate_df = pd.DataFrame(candidate_records)
                    candidate_attrs = sampler.sample_points(
                        candidate_df[["X", "Y", "TIME"]]
                    )
                    attribute_columns = list(SEISMIC_ATTRIBUTE_COLUMNS.values())
                    all_attribute_values = candidate_attrs.copy()
                    all_attribute_values.loc[len(all_attribute_values)] = [
                        event_attrs.get(column, np.nan)
                        for column in attribute_columns
                    ]
                    distance_sq = np.zeros(len(candidate_df), dtype=float)
                    total_weight = 0.0
                    for attribute_name, column in SEISMIC_ATTRIBUTE_COLUMNS.items():
                        values = safe_numeric(all_attribute_values[column]).to_numpy(dtype=float)
                        normalized_scale = _robust_normalization_scale(
                            values,
                            attribute_floors.get(attribute_name, 1.0),
                        )
                        reference_value = values[-1]
                        if not np.isfinite(reference_value):
                            reference_value = float(np.nanmedian(values[:-1])) if np.isfinite(values[:-1]).any() else 0.0
                        weight = attribute_weights.get(attribute_name, 1.0)
                        distance_sq += weight * np.square(
                            (values[:-1] - reference_value) / normalized_scale
                        )
                        total_weight += weight
                    candidate_df["SeismicAttributeDistance"] = np.sqrt(
                        distance_sq / max(total_weight, 1.0e-9)
                    )
                    candidate_df["SeismicSimilarity"] = np.exp(
                        -0.5 * candidate_df["SeismicAttributeDistance"].to_numpy(dtype=float)
                    )
                    for column in attribute_columns:
                        candidate_df[column] = candidate_attrs[column].to_numpy(dtype=float)
                    candidate_df["RejectionReason"] = ""
                    candidate_df.loc[
                        candidate_attrs.isna().any(axis=1),
                        "RejectionReason",
                    ] = "missing_seismic_attribute"
                    candidate_df.loc[
                        candidate_df["SeismicSimilarity"] < minimum_similarity,
                        "RejectionReason",
                    ] = "below_minimum_similarity"
                    candidate_coords = np.column_stack(
                        [
                            candidate_df["X"].to_numpy(dtype=float),
                            candidate_df["Y"].to_numpy(dtype=float),
                            candidate_df["TIME"].to_numpy(dtype=float) * time_scale,
                        ]
                    )
                    nearest_distances, _ = occupied_tree.query(
                        candidate_coords,
                        k=1,
                    )
                    candidate_df["NearestExisting3DDistanceM"] = np.asarray(
                        nearest_distances,
                        dtype=float,
                    )
                    candidate_df.loc[
                        candidate_df["NearestExisting3DDistanceM"] < min_separation,
                        "RejectionReason",
                    ] = "too_close_to_existing_patch"

                    eligible = candidate_df[candidate_df["RejectionReason"].eq("")].copy()
                    if eligible.empty:
                        for _, rejected in candidate_df.iterrows():
                            audit_rows.append(
                                {
                                    key: value
                                    for key, value in rejected.items()
                                    if key != "_template"
                                }
                                | {"Selected": 0}
                            )
                        continue
                    selected = eligible.sort_values(
                        ["SeismicSimilarity", "NearestExisting3DDistanceM"],
                        ascending=[False, False],
                    ).iloc[0]
                    template = selected["_template"]
                    serial += 1
                    row = dict(template)
                    row["PatchID"] = f"seismic_guided_dfn_{serial:06d}"
                    row["OriginalPatchID"] = str(template["PatchID"])
                    row["ParentPatchID"] = str(template["PatchID"])
                    row["OriginalCenterX"] = float(template["CenterX"])
                    row["OriginalCenterY"] = float(template["CenterY"])
                    row["OriginalCenterTime"] = float(template["CenterTime"])
                    row["CenterX"] = float(selected["X"])
                    row["CenterY"] = float(selected["Y"])
                    row["CenterTime"] = float(selected["TIME"])
                    row["LengthM"] = float(template["LengthM"]) * size_multiplier
                    row["HeightTimeMs"] = float(template["HeightTimeMs"]) * height_multiplier
                    row["SourceDensity"] = float(template.get("SourceDensity", 0.5)) * (
                        0.65 + 0.35 * float(selected["SeismicSimilarity"])
                    )
                    row["CorrectionAction"] = "add_density_compensation_patch"
                    row["CorrectionReason"] = "seismic_similarity_guided_3d_density"
                    row["SourceType"] = "seismic_guided_conventional_density"
                    row["GenerationStage"] = "step8_seismic_guided_density"
                    row["ControlSource"] = "seismic_guided_conventional_prediction"
                    row["ControlX"] = float(event["X"])
                    row["ControlY"] = float(event["Y"])
                    row["ControlTime"] = float(event["TIME"])
                    row["WellControlSampleID"] = f"seismic_{event['WellControlSampleID']}_{serial:06d}"
                    row["WellControlWellName"] = str(well_name)
                    row["IsWellControlPatch"] = 0
                    row["IsImagingGroundTruth"] = 0
                    row["IsImagingMatchPatch"] = 0
                    row["IsDensityCompensationPatch"] = 1
                    row["NeedsWellCorrection"] = 0
                    row["Confidence"] = min(float(template.get("Confidence", 0.5)), 0.65)
                    row["DensityCompensationWellName"] = str(well_name)
                    row["DensityCompensationRing"] = str(selected["RadialBandLabel"])
                    row["DensityCompensationRingMinM"] = float(selected["RadialBandMinM"])
                    row["DensityCompensationRingMaxM"] = float(selected["RadialBandMaxM"])
                    row["DensityCompensationRadialWeight"] = 1.0
                    row["DensityCompensationTargetDensity"] = np.nan
                    row["DensityCompensationCurrentDensity"] = np.nan
                    row["DensityCompensationDeficit"] = np.nan
                    row["DensityCompensationReferencePointWeight"] = np.nan
                    row["NearestTrajectoryDistance"] = float(selected["NearestExisting3DDistanceM"])
                    row["SeismicGuidedDensity"] = 1
                    row["SeismicSimilarity"] = float(selected["SeismicSimilarity"])
                    row["SeismicAttributeDistance"] = float(selected["SeismicAttributeDistance"])
                    row["SeismicAntTrack"] = float(selected["SeismicAntTrack"])
                    row["SeismicCoherence"] = float(selected["SeismicCoherence"])
                    row["SeismicCurvatureMax"] = float(selected["SeismicCurvatureMax"])
                    row["SeismicReferenceControlID"] = str(event["WellControlSampleID"])
                    row["SeismicCandidateBand"] = str(selected["RadialBandLabel"])
                    row["SeismicTemplatePatchID"] = str(selected["TemplatePatchID"])
                    row["TimeOffsetFromControlMs"] = float(selected["TimeOffsetFromControlMs"])
                    row["Minimum3DSeparationM"] = float(selected["NearestExisting3DDistanceM"])
                    row.update(update_layer_window_from_surfaces(row, surface_lookup))
                    if int(row.get("HorizonCenterInside", 0)) != 1:
                        selected["RejectionReason"] = "outside_local_layer_window"
                    else:
                        patch_rows.append(row)
                        occupied_tree = cKDTree(
                            np.vstack(
                                [
                                    occupied_coords,
                                    np.asarray(
                                        [[row["CenterX"], row["CenterY"], row["CenterTime"] * time_scale]],
                                        dtype=float,
                                    ),
                                ]
                            )
                        )
                        occupied_coords = np.vstack(
                            [
                                occupied_coords,
                                np.asarray(
                                    [[row["CenterX"], row["CenterY"], row["CenterTime"] * time_scale]],
                                    dtype=float,
                                ),
                            ]
                        )
                    for _, candidate in candidate_df.iterrows():
                        is_selected = str(candidate["TemplatePatchID"]) == str(selected["TemplatePatchID"]) and (
                            float(candidate["X"]) == float(selected["X"])
                            and float(candidate["Y"]) == float(selected["Y"])
                            and float(candidate["TIME"]) == float(selected["TIME"])
                        )
                        audit_rows.append(
                            {
                                key: value
                                for key, value in candidate.items()
                                if key != "_template"
                            }
                            | {
                                "Selected": int(is_selected),
                                "SelectionReason": (
                                    str(selected["RejectionReason"])
                                    if is_selected and selected["RejectionReason"]
                                    else "selected_highest_seismic_similarity"
                                    if is_selected
                                    else candidate["RejectionReason"]
                                    or "not_highest_similarity_in_band"
                                ),
                            }
                        )
    finally:
        sampler.close()

    patch_output = pd.DataFrame(patch_rows)
    audit_output = pd.DataFrame(audit_rows)
    selected_audit = audit_output[audit_output.get("Selected", pd.Series(dtype=int)).eq(1)]
    identical_center_count = int(
        patch_output.duplicated(["CenterX", "CenterY", "CenterTime"]).sum()
        if not patch_output.empty
        else 0
    )
    summary = {
        "enabled": True,
        "status": "applied" if not patch_output.empty else "no_selected_candidates",
        "control_well_names": sorted(well_names),
        "control_event_count": int(len(controls)),
        "candidate_count": int(len(audit_output)),
        "created_patch_count": int(len(patch_output)),
        "created_patch_count_by_well": {
            str(key): int(value)
            for key, value in patch_output.get("DensityCompensationWellName", pd.Series(dtype=str)).value_counts().items()
        },
        "created_patch_count_by_layer": {
            str(key): int(value)
            for key, value in patch_output.get("LayerGroup", pd.Series(dtype=str)).value_counts().items()
        },
        "created_patch_count_by_scale": {
            str(key): int(value)
            for key, value in patch_output.get("FractureScale", pd.Series(dtype=str)).value_counts().items()
        },
        "created_patch_count_by_band": {
            str(key): int(value)
            for key, value in patch_output.get("SeismicCandidateBand", pd.Series(dtype=str)).value_counts().sort_index().items()
        },
        "seismic_similarity_min": float(selected_audit["SeismicSimilarity"].min()) if not selected_audit.empty else None,
        "seismic_similarity_median": float(selected_audit["SeismicSimilarity"].median()) if not selected_audit.empty else None,
        "seismic_similarity_max": float(selected_audit["SeismicSimilarity"].max()) if not selected_audit.empty else None,
        "minimum_abs_center_time_offset_ms": float(selected_audit["TimeOffsetFromControlMs"].abs().min()) if not selected_audit.empty else None,
        "minimum_selected_3d_separation_m": float(selected_audit["NearestExisting3DDistanceM"].min()) if not selected_audit.empty else None,
        "identical_center_count": identical_center_count,
        "horizontal_like_patch_count": int(
            (
                safe_numeric(patch_output.get("DipDeg", pd.Series(dtype=float))).lt(min_dip)
                & safe_numeric(patch_output.get("TimeWindowMax", pd.Series(dtype=float))).notna()
            ).sum()
        ),
        "parameter_summary": {
            "radial_bands_m": [list(item) for item in bands],
            "scale_cycle": scale_cycle,
            "minimum_template_dip_deg": min_dip,
            "minimum_abs_time_offset_ms": min_time_offset,
            "minimum_3d_separation_m": min_separation,
            "time_scale_m_per_ms": time_scale,
            "attribute_weights": attribute_weights,
            "trace_mapping_npz": str(mapping_path),
            "volume_paths": {key: str(value) for key, value in volume_paths.items()},
        },
    }
    return patch_output, audit_output, summary


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
                safe_numeric(pd.Series([out.at[patch_idx, "SourceDensity"]])).fillna(0.0).iloc[0]
            )
            control_density = float(
                safe_numeric(pd.Series([control.get("Density", np.nan)])).fillna(original_density).iloc[0]
            )
            corrected_density = max(
                original_density,
                original_density + blend * influence * max(control_density - original_density, 0.0),
            )
            out.at[patch_idx, "SourceDensity"] = corrected_density
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
    if "Density" not in out.columns:
        out["Density"] = np.nan
    out["Density"] = safe_numeric(out["Density"])
    out["WellControlSizeFactor"] = 0.5
    for layer, group in out.groupby("LayerGroup", dropna=False):
        density = safe_numeric(group["Density"])
        if density.notna().any():
            lo = float(density.quantile(0.05))
            hi = float(density.quantile(0.95))
            if hi > lo:
                factor = ((density.fillna(density.median()) - lo) / (hi - lo)).clip(0.0, 1.0)
            else:
                factor = pd.Series(0.5, index=group.index)
        else:
            factor = pd.Series(0.5, index=group.index)
        out.loc[group.index, "WellControlSizeFactor"] = factor
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
        return float(azimuth) % 360.0, float(np.clip(dip, 1.0, 89.0)), "step3_imaging_gt_orientation"
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
    if int(pd.to_numeric(pd.Series([control.get("ForceWellControlCenter", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
        return {
            "CenterX": float(control["X"]),
            "CenterY": float(control["Y"]),
            "CenterTime": float(control["TIME"]),
            "CenterOffsetM": 0.0,
            "CenterOffsetTimeMs": 0.0,
            "ControlInsidePatchEnvelope": 1,
        }
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
    is_imaging_ground_truth = int(control.get("IsImagingGroundTruth", 0) or 0) == 1
    geometry_azimuth = (
        float((azimuth - 90.0) % 180.0)
        if is_imaging_ground_truth
        else float(azimuth)
    )
    length = float(control["WellControlLengthM"]) if "WellControlLengthM" in control and pd.notna(control["WellControlLengthM"]) else float(template["LengthM"])
    height = float(control["WellControlHeightTimeMs"]) if "WellControlHeightTimeMs" in control and pd.notna(control["WellControlHeightTimeMs"]) else float(template["HeightTimeMs"])
    center = offset_center_near_control(control=control, azimuth_deg=geometry_azimuth, dip_deg=dip, length_m=length, height_time_ms=height, config=config)
    out.update(center)
    out["LengthM"] = length
    out["HeightTimeMs"] = height
    if is_imaging_ground_truth:
        out["DipAzimuthDeg"] = azimuth % 360.0
        out["AzimuthDeg"] = geometry_azimuth
    else:
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
    out["IsImagingGroundTruth"] = int(is_imaging_ground_truth)
    out["IsImagingMatchPatch"] = int(
        is_imaging_ground_truth
        and int(control.get("ImagingMatchSelected", 0) or 0) == 1
    )
    out["IsDensityCompensationPatch"] = 0
    out["ImagingMatchTargetAzimuthDeg"] = (
        float(control.get("Frac_Azimuth", np.nan)) if is_imaging_ground_truth else np.nan
    )
    out["ImagingMatchTargetDipDeg"] = (
        float(control.get("Frac_Dip", np.nan)) if is_imaging_ground_truth else np.nan
    )
    out = force_well_control_small_scale(out, control=control, config=config)
    return out


def build_added_patch(
    template: pd.Series,
    control: pd.Series,
    add_index: int,
    before_distance: float,
    track_distance: float,
    config: dict[str, Any],
    surface_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = template.to_dict()
    row["PatchID"] = f"well_ctrl_dfn_{add_index:06d}"
    row["OriginalPatchID"] = ""
    row["OriginalCenterX"] = np.nan
    row["OriginalCenterY"] = np.nan
    row["OriginalCenterTime"] = np.nan
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
    surface_lookup: dict[str, Any] | None = None,
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


def build_radial_density_compensation_patches(
    initial_df: pd.DataFrame,
    requests: pd.DataFrame,
    config: dict[str, Any],
    surface_lookup: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tagged radial compensation patches without changing match accounting."""

    if requests.empty:
        return pd.DataFrame(), pd.DataFrame()
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    length_multiplier = float(config.get("density_compensation_size_multiplier", 0.65))
    height_multiplier = float(config.get("density_compensation_height_multiplier", 0.65))
    confidence = float(config.get("density_compensation_confidence", 0.45))
    target = dict(config.get("target_block") or {})
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for request_index, request in requests.reset_index(drop=True).iterrows():
        template = nearest_template_patch(
            initial_df=initial_df,
            layer=str(request["LayerGroup"]),
            x=float(request["X"]),
            y=float(request["Y"]),
            time_value=float(request["TIME"]),
            time_scale=time_scale,
            config=config,
        )
        if template is None:
            continue
        comp_id = f"density_comp_{request_index + 1:06d}"
        pseudo_control = pd.Series(
            {
                "WellName": str(request["WellName"]),
                "X": float(request["X"]),
                "Y": float(request["Y"]),
                "TIME": float(request["TIME"]),
                "LayerGroup": str(request["LayerGroup"]),
                "Density": float(template.get("SourceDensity", 0.5)),
                "Frac_Azimuth": float(request["Frac_Azimuth"]),
                "Frac_Dip": float(request["Frac_Dip"]),
                "WellControlSampleID": comp_id,
                "ControlSource": "density_compensation_radial",
                "IsImagingGroundTruth": 1,
                "ImagingMatchSelected": 0,
                "WellControlLengthM": float(
                    np.clip(float(template["LengthM"]) * length_multiplier, 20.0, 80.0)
                ),
                "WellControlHeightTimeMs": float(
                    np.clip(float(template["HeightTimeMs"]) * height_multiplier, 3.0, 12.0)
                ),
            }
        )
        row = build_added_patch(
            template=template,
            control=pseudo_control,
            add_index=request_index + 1,
            before_distance=np.nan,
            track_distance=float(request["RadialDistanceM"]),
            config=config,
            surface_lookup=surface_lookup,
        )
        row["PatchID"] = f"density_comp_dfn_{request_index + 1:06d}"
        row["SourceType"] = "density_compensation_well_ring"
        row["CorrectionAction"] = "add_density_compensation_patch"
        row["CorrectionReason"] = "radial_band_density_deficit"
        row["ControlSource"] = "density_compensation_radial"
        row["IsImagingGroundTruth"] = 0
        row["IsImagingMatchPatch"] = 0
        row["IsDensityCompensationPatch"] = 1
        row["IsWellControlPatch"] = 0
        row["NeedsWellCorrection"] = 0
        row["Confidence"] = confidence
        row["DensityCompensationWellName"] = str(request["WellName"])
        row["DensityCompensationRing"] = str(request["RadialBandLabel"])
        row["DensityCompensationRingMinM"] = float(request["RadialBandMinM"])
        row["DensityCompensationRingMaxM"] = float(request["RadialBandMaxM"])
        row["DensityCompensationRadialWeight"] = float(request["RadialWeight"])
        row["DensityCompensationTargetDensity"] = float(request["TargetAreaDensity"])
        row["DensityCompensationCurrentDensity"] = float(request["CurrentAreaDensity"])
        row["DensityCompensationDeficit"] = float(request["DeficitAreaDensity"])
        row["DensityCompensationReferencePointWeight"] = float(request["ReferencePointWeight"])
        row["DipAzimuthDeg"] = float(request["Frac_Azimuth"]) % 360.0
        row["AzimuthDeg"] = float((float(request["Frac_Azimuth"]) - 90.0) % 180.0)
        row["DipDeg"] = float(request["Frac_Dip"])
        row["OrientationSource"] = "radial_density_compensation_local_orientation"
        if target:
            row["CenterX"] = float(np.clip(row["CenterX"], float(target["x_min"]), float(target["x_max"])))
            row["CenterY"] = float(np.clip(row["CenterY"], float(target["y_min"]), float(target["y_max"])))
        row.update(update_layer_window_from_surfaces(row, surface_lookup))
        rows.append(row)
        audit_rows.append(
            {
                "PatchID": row["PatchID"],
                **request.to_dict(),
                "SourceType": row["SourceType"],
                "CorrectionAction": row["CorrectionAction"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


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
    if preserve_vertices and correction_action not in {
        "add_well_control_patch",
        "adjust_to_well_control",
        "add_density_compensation_patch",
    }:
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
    modified_mask = out["CorrectionAction"].astype(str).isin([
        "add_well_control_patch",
        "adjust_to_well_control",
        "add_density_compensation_patch",
    ])
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
    lookup: HorizonSpatialLookup,
    use_dip_geometry: bool,
    geometry_time_scale_m_per_ms: float,
    preserve_existing_vertices: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = patch_df.reset_index(drop=True).copy()
    vertex_cols = [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
    for column in vertex_cols:
        if column not in out.columns:
            out[column] = np.nan

    all_vertices = np.empty((len(out), 4, 3), dtype=np.float64)
    for row_idx, row in out.iterrows():
        correction_action = str(row.get("CorrectionAction", "unchanged_density_volume"))
        existing = np.asarray(
            [
                [
                    float(row[f"V{vertex_idx}X"]),
                    float(row[f"V{vertex_idx}Y"]),
                    float(row[f"V{vertex_idx}Z"]),
                ]
                for vertex_idx in range(1, 5)
            ],
            dtype=np.float64,
        )
        if (
            preserve_existing_vertices
            and np.isfinite(existing).all()
            and correction_action not in {
                "add_well_control_patch",
                "adjust_to_well_control",
                "add_density_compensation_patch",
            }
        ):
            all_vertices[row_idx] = existing
        else:
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
    trace_idx = lookup.trace_idx[np.asarray(positions, dtype=np.int64)]
    horizon_rows = lookup.table[trace_idx]
    vertex_layers = np.repeat(out["LayerGroup"].astype(str).to_numpy(), 4)
    top = np.full(len(vertex_layers), np.nan, dtype=np.float64)
    base = np.full(len(vertex_layers), np.nan, dtype=np.float64)
    present = np.zeros(len(vertex_layers), dtype=bool)
    for layer_item in _FRAMEWORK.layers:
        layer_mask = vertex_layers == layer_item.name
        if not layer_mask.any():
            continue
        top_column, base_column = _FRAMEWORK.surface_pair(layer_item)
        top[layer_mask] = horizon_rows[top_column][layer_mask].astype(np.float64)
        base[layer_mask] = horizon_rows[base_column][layer_mask].astype(np.float64)
        present[layer_mask] = horizon_rows[layer_item.presence_column][layer_mask].astype(bool)
    valid = present & np.isfinite(top) & np.isfinite(base) & (base >= top)
    voxel_coverage = out.get("WindowValidationMode", pd.Series("", index=out.index)).astype(str).eq("voxel_coverage").to_numpy()
    voxel_coverage_vertices = voxel_coverage.repeat(4)
    valid = valid | voxel_coverage_vertices

    # A hard fault center remains fixed. If a corner reaches a local trace without a
    # valid T4-T7 interval, shorten only that corner along the center-to-corner edge.
    for flat_idx in np.where(~valid)[0]:
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
            patch_item = _FRAMEWORK.layer(patch_layer)
            patch_top, patch_base = _FRAMEWORK.surface_pair(patch_item)
            local_present = bool(local[patch_item.presence_column])
            local_top = float(local[patch_top])
            local_base = float(local[patch_base])
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
        patch_item = _FRAMEWORK.layer(patch_layer)
        patch_top, patch_base = _FRAMEWORK.surface_pair(patch_item)
        top[flat_idx] = float(local[patch_top])
        base[flat_idx] = float(local[patch_base])
        local_present = bool(local[patch_item.presence_column])
        valid[flat_idx] = bool(
            local_present
            and np.isfinite(top[flat_idx])
            and np.isfinite(base[flat_idx])
            and base[flat_idx] >= top[flat_idx]
        )
    original_z = flat_vertices[:, 2].copy()
    clipped_z = np.where(valid, np.clip(original_z, top, base), original_z)
    flat_vertices[:, 2] = clipped_z
    all_vertices = flat_vertices.reshape(len(out), 4, 3)
    if preserve_existing_vertices and "ParentPatchID" in out.columns:
        parent_ids = out["ParentPatchID"].fillna("").astype(str)
        for parent_id, group in out.groupby(parent_ids, sort=False):
            if not parent_id or len(group) < 2 or "LargeFaultSegmentIndex" not in group.columns:
                continue
            positions = group.sort_values("LargeFaultSegmentIndex").index.to_numpy(dtype=int)
            for left_pos, right_pos in zip(positions[:-1], positions[1:]):
                shared = 0.5 * (
                    all_vertices[left_pos, [1, 2], :]
                    + all_vertices[right_pos, [0, 3], :]
                )
                all_vertices[left_pos, [1, 2], :] = shared
                all_vertices[right_pos, [0, 3], :] = shared
        flat_vertices = all_vertices.reshape(-1, 3)
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
    out["HorizonVerticesInsideT4T7"] = valid.reshape(len(out), 4).all(axis=1).astype(np.uint8)
    out["PatchAreaM2"] = [polygon_area([tuple(point) for point in vertices]) for vertices in all_vertices]
    return out, {
        "patch_count": int(len(out)),
        "vertex_count": int(len(flat_vertices)),
        "clipped_patch_count": int((out["HorizonClippedVertexCount"] > 0).sum()),
        "clipped_vertex_count": int((delta > 1.0e-6).sum()),
        "horizontally_clipped_patch_count": int((out["HorizonHorizontallyClippedVertexCount"] > 0).sum()),
        "horizontally_clipped_vertex_count": int((xy_delta > 1.0e-6).sum()),
        "invalid_local_window_vertex_count": int((~valid).sum()),
        "all_vertices_inside_local_layer_window": bool(valid.all()),
        "all_vertices_inside_local_t4_t7": bool(valid.all()),
        "voxel_coverage_exempted_patch_count": int(voxel_coverage.sum()),
        "voxel_coverage_exempted_vertex_count": int(voxel_coverage_vertices.sum()),
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
    audit_after = safe_numeric(audit_df["AfterMatchDistance"]) if "AfterMatchDistance" in audit_df.columns else pd.Series(dtype=float)
    audit_offsets = safe_numeric(audit_df["CenterOffsetM"]) if "CenterOffsetM" in audit_df.columns else pd.Series(dtype=float)
    control_envelope = safe_numeric(audit_df["ControlInsidePatchEnvelope"]).fillna(0) if "ControlInsidePatchEnvelope" in audit_df.columns else pd.Series(dtype=float)
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
        layer_window_available = corrected_df[["CenterTime", "TimeWindowMin", "TimeWindowMax"]].notna().all(axis=1)
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
        "layers_limited_to_sha3_sha4": bool(set(corrected_df["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS))),
        "centers_within_layer_windows": centers_within_layer_windows,
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
        "summary_json": str(paths["summary_json"]),
        "target_block": config["target_block"],
        "correction_logic": {
            "well_points_inside_target_are_hard_controls": True,
            "adjust_existing_patch_when_local_candidate_available": True,
            "add_patch_when_no_local_candidate_available": True,
            "far_field_density_volume_patches_unchanged": True,
            "original_fault_triangles_carried_unchanged_from_step7d_unified_vtk": True,
            "xy_search_radius_m": float(config.get("xy_search_radius_m", 80.0)),
            "time_search_radius_ms": float(config.get("time_search_radius_ms", 30.0)),
            "surface_dir": str(Path(config["surface_dir"]).resolve()) if config.get("surface_dir") else None,
            "surface_windows_recomputed_for_well_control_patches": bool(config.get("surface_dir")),
            "inverted_surface_window_policy": "sort_finite_boundaries_then_include_hard_control_center",
            "use_dip_geometry": bool(config.get("use_dip_geometry", False)),
            "geometry_time_scale_m_per_ms": float(config.get("geometry_time_scale_m_per_ms", 1.0)),
            "enable_well_control_center_offset": bool(config.get("enable_well_control_center_offset", True)),
            "well_control_center_offset_max_m": max_allowed_offset,
            "step3_imaging_gt_enabled": bool(config.get("step3_imaging_group_csvs")),
            "replace_step4_inside_step3_imaging_windows": bool(config.get("replace_step4_inside_step3_imaging_windows", True)),
            "well_control_size": dict(config.get("well_control_size", {})),
            "force_well_control_small_scale": bool(config.get("force_well_control_small_scale", False)),
            "well_control_match_scales": list(config.get("well_control_match_scales", [])),
            "well_control_template_scales": list(config.get("well_control_template_scales", [])),
            "imaging_well_region_correction": dict(config.get("_imaging_well_region_summary", {})),
            "step4_small_orientation_policy": dict(config.get("step4_small_orientation_policy", {})),
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
            "note": "Only patches with non-null CenterTime/TimeWindowMin/TimeWindowMax are checked.",
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

    initial_csv = Path(config["initial_dfn_csv"]).resolve()
    initial_vtk = Path(config["initial_dfn_vtk"]).resolve()
    fracture_csv = Path(config["fracture_points_csv"]).resolve()
    additional_fracture_csvs = [
        Path(str(value)).resolve()
        for value in config.get("additional_fracture_points_csvs", [])
    ]
    samples_root = Path(config["real_well_samples_root"]).resolve()
    additional_samples_roots = [
        Path(str(value)).resolve()
        for value in config.get("additional_real_well_samples_roots", [])
    ]
    initial_summary_json = Path(config.get("initial_dfn_summary_json", "")).resolve()
    surface_dir = Path(config["surface_dir"]).resolve() if config.get("surface_dir") else None
    required_paths = [
        ("initial_dfn_csv", initial_csv),
        ("initial_dfn_vtk", initial_vtk),
        ("fracture_points_csv", fracture_csv),
        ("real_well_samples_root", samples_root),
        *[(f"additional_fracture_points_csvs[{index}]", path) for index, path in enumerate(additional_fracture_csvs)],
        *[(f"additional_real_well_samples_roots[{index}]", path) for index, path in enumerate(additional_samples_roots)],
    ]
    for label, path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if surface_dir is not None and not surface_dir.exists():
        raise FileNotFoundError(f"surface_dir does not exist: {surface_dir}")
    input_original_fault_mesh = extract_geometry_group(
        initial_vtk,
        ORIGINAL_FAULT_GEOMETRY_GROUP_CODE,
    ).triangulate()
    if input_original_fault_mesh.n_cells <= 0:
        raise RuntimeError("Step7D unified VTK contains no original-fault triangles")
    config["_input_original_fault_fingerprint"] = geometry_fingerprint(input_original_fault_mesh)

    initial_df = load_initial_dfn(initial_csv)
    step4_control_df = load_control_points(fracture_csv, dict(config["target_block"]))
    raw_step4_count = int(len(step4_control_df))
    additional_control_count = 0
    for path in additional_fracture_csvs:
        additional = load_control_points(path, dict(config["target_block"]))
        additional["ControlSource"] = "new_conventional_prediction"
        additional["IsNewConventionalControl"] = 1
        additional["ForceWellControlCenter"] = 1
        additional_control_count += int(len(additional))
        step4_control_df = pd.concat([step4_control_df, additional], ignore_index=True, sort=False)
    if "IsNewConventionalControl" in step4_control_df.columns:
        step4_control_df["IsNewConventionalControl"] = (
            safe_numeric(step4_control_df["IsNewConventionalControl"]).fillna(0).astype(int)
        )
    if "ForceWellControlCenter" in step4_control_df.columns:
        step4_control_df["ForceWellControlCenter"] = (
            safe_numeric(step4_control_df["ForceWellControlCenter"]).fillna(0).astype(int)
        )
    step4_control_df = aggregate_step4_controls_to_events(step4_control_df, config=config)
    config["_raw_step4_control_count"] = raw_step4_count
    config["_additional_control_point_count"] = additional_control_count
    config["_aggregated_step4_control_count"] = int(len(step4_control_df))
    step3_group_paths = [Path(str(value)).resolve() for value in config.get("step3_imaging_group_csvs", [])]
    for path in step3_group_paths:
        if not path.exists():
            raise FileNotFoundError(f"step3_imaging_group_csvs entry does not exist: {path}")
    if step3_group_paths:
        step3_well_name = str(config.get("step3_imaging_well_name", "车页1导眼"))
        step3_control_df = load_step3_imaging_controls(
            paths=step3_group_paths,
            well_name=step3_well_name,
            target_block=dict(config["target_block"]),
        )
        step3_windows = load_step3_imaging_time_windows(
            paths=step3_group_paths,
            well_name=step3_well_name,
            target_block=dict(config["target_block"]),
        )
        control_df = merge_step3_imaging_controls(
            step4_df=step4_control_df,
            step3_df=step3_control_df,
            windows=step3_windows,
            config=config,
        )
    else:
        control_df = step4_control_df
    control_df, imaging_match_selection_df, imaging_match_calibration = (
        step8_match_density.calibrate_imaging_point_controls(
            control_df,
            config,
        )
    )
    config["_imaging_match_calibration"] = imaging_match_calibration
    track_roots = [samples_root, *additional_samples_roots]
    tracks: dict[str, pd.DataFrame] = {}
    for root in track_roots:
        tracks.update(
            load_real_well_tracks(
                root,
                set(control_df["WellName"].astype(str).unique()),
            )
        )
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    surface_lookup = build_spatial_lookup(config)
    track_dist = nearest_track_distances(control_df, tracks, time_scale=time_scale)
    before_dist = nearest_patch_distances(initial_df, control_df, time_scale=time_scale)
    corrected_df, audit_df = apply_well_controls(
        initial_df=initial_df,
        control_df=control_df,
        track_distances=track_dist,
        config=config,
        surface_lookup=surface_lookup,
    )
    corrected_df, large_control_audit_df, large_control_summary = (
        large_conventional_controls.apply_large_fault_conventional_support(
            corrected_df,
            control_df,
            config,
        )
    )
    config["_large_conventional_control_summary"] = large_control_summary
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
    seismic_patches_df, seismic_audit_df, seismic_guided_summary = (
        build_seismic_guided_density_patches(
            patch_df=corrected_df,
            initial_df=initial_df,
            control_df=control_df,
            config=config,
            surface_lookup=surface_lookup,
        )
    )
    if not seismic_patches_df.empty:
        corrected_df = pd.concat(
            [corrected_df, seismic_patches_df],
            ignore_index=True,
            sort=False,
        )
    config["_seismic_guided_densification_summary"] = seismic_guided_summary
    selected_imaging_controls = control_df[
        safe_numeric(
            control_df.get("IsImagingGroundTruth", pd.Series(0, index=control_df.index))
        ).fillna(0).astype(int).eq(1)
        & safe_numeric(
            control_df.get("ImagingMatchSelected", pd.Series(0, index=control_df.index))
        ).fillna(0).astype(int).eq(1)
    ].copy()
    density_requests_df, density_ring_summary_df, density_compensation_summary = (
        step8_match_density.build_radial_density_compensation_requests(
            patches=corrected_df,
            selected_imaging_controls=selected_imaging_controls,
            tracks=tracks,
            config=config,
        )
    )
    density_patches_df, density_patch_audit_df = build_radial_density_compensation_patches(
        initial_df=initial_df,
        requests=density_requests_df,
        config=config,
        surface_lookup=surface_lookup,
    )
    if not density_patches_df.empty:
        corrected_df = pd.concat(
            [corrected_df, density_patches_df],
            ignore_index=True,
            sort=False,
        )
    density_compensation_summary["created_patch_count"] = int(len(density_patches_df))
    config["_radial_density_compensation_summary"] = density_compensation_summary
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
        preserve_existing_vertices=bool(config.get("preserve_existing_vertices", False)),
    )
    final_horizon_qc["vertex_geometry"] = vertex_horizon_qc
    corrected_df.to_csv(paths["corrected_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    region_audit_df.to_csv(paths["region_audit_csv"], index=False, encoding="utf-8-sig")
    imaging_match_selection_df.to_csv(
        paths["imaging_match_selection_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    density_requests_df.to_csv(
        paths["density_compensation_requests_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    density_ring_summary_df.to_csv(
        paths["density_compensation_summary_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    seismic_audit_df.to_csv(
        paths["seismic_guided_audit_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["seismic_guided_summary_json"].write_text(
        json.dumps(seismic_guided_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    large_control_audit_df.to_csv(
        output_dir / "large_conventional_control_support.csv",
        index=False,
        encoding="utf-8-sig",
    )
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
    imaging_match_summary = step8_match_density.summarize_imaging_match_patches(
        corrected_df=corrected_df,
        active_controls=control_df,
        calibration_summary=imaging_match_calibration,
        config=config,
    )
    paths["imaging_match_json"].write_text(
        json.dumps(imaging_match_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["imaging_match"] = imaging_match_summary
    summary["radial_density_compensation"] = dict(
        config.get("_radial_density_compensation_summary", {})
    )
    summary["radial_density_compensation"]["patch_audit"] = {
        "created_patch_count": int(len(density_patch_audit_df)),
        "by_band": {
            str(key): int(value)
            for key, value in density_patch_audit_df.get(
                "RadialBandLabel", pd.Series(dtype=str)
            ).value_counts().sort_index().items()
        },
    }
    summary["seismic_guided_densification"] = dict(
        config.get("_seismic_guided_densification_summary", {})
    )
    summary.setdefault("checks", {})["seismic_guided_density_has_no_duplicate_centers"] = bool(
        summary["seismic_guided_densification"].get("identical_center_count", 0) == 0
    )
    summary.setdefault("checks", {})["seismic_guided_density_is_steep"] = bool(
        summary["seismic_guided_densification"].get("horizontal_like_patch_count", 0) == 0
    )
    summary.setdefault("checks", {})["imaging_match_ratio_within_85_90"] = bool(
        imaging_match_summary["geometry_match_ratio_within_required_interval"]
    )
    summary.setdefault("checks", {})["imaging_orientation_histogram_preserved"] = bool(
        imaging_match_summary["orientation_histogram_overlap"]
        >= imaging_match_summary["minimum_orientation_histogram_overlap"]
    )
    summary.setdefault("checks", {})["imaging_match_plane_tolerance"] = bool(
        imaging_match_summary["geometry_match_ratio_within_required_interval"]
    )
    summary["large_conventional_control_support"] = dict(
        config.get("_large_conventional_control_summary", {})
    )
    if debug_vtks:
        summary["debug_step_vtks"] = debug_vtks
    summary["horizon_contract_qc"] = final_horizon_qc
    summary.setdefault("checks", {})["all_patch_vertices_inside_local_t4_t7"] = bool(
        vertex_horizon_qc["all_vertices_inside_local_t4_t7"]
    )
    summary["status"] = "pass" if all(bool(value) for value in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Corrected DFN CSV: {paths['corrected_csv']}")
    print(f"Corrected DFN raw VTK: {paths['raw_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Control points: {len(control_df)} corrected patches: {len(corrected_df)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
