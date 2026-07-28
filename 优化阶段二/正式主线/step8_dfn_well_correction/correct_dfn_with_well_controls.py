from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_well_control_correction.json"
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import load_surface_tables  # noqa: E402

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


def update_layer_window_from_surfaces(row: pd.Series | dict[str, Any], surface_lookup: dict[str, Any] | None) -> dict[str, float]:
    if surface_lookup is None:
        return {}
    layer = str(row["LayerGroup"])
    x = float(row["CenterX"])
    y = float(row["CenterY"])
    if layer == "沙三段":
        top = query_surface_time(surface_lookup, "T4", x, y)
        base = query_surface_time(surface_lookup, "T6", x, y)
    elif layer == "沙四段":
        top = query_surface_time(surface_lookup, "T6", x, y)
        base = query_surface_time(surface_lookup, "T7", x, y)
    else:
        return {}
    if not np.isfinite(top) or not np.isfinite(base):
        return {}
    surface_order_repaired = int(base <= top)
    if surface_order_repaired:
        top, base = min(top, base), max(top, base)
    center_time = float(row["CenterTime"])
    if np.isfinite(center_time):
        top = min(top, center_time)
        base = max(base, center_time)
    if base <= top:
        return {}
    return {
        "TimeWindowMin": top,
        "TimeWindowMax": base,
        "LayerThickness": base - top,
        "LayerWindowSurfaceOrderRepaired": surface_order_repaired,
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
    ]


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
            "07 is the full final corrected DFN, equivalent in content to well_corrected_dfn_raw_time.vtk.",
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
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "initial_dfn_csv": str(Path(config["initial_dfn_csv"]).resolve()),
        "fracture_points_csv": str(Path(config["fracture_points_csv"]).resolve()),
        "real_well_samples_root": str(Path(config["real_well_samples_root"]).resolve()),
        "corrected_dfn_csv": str(paths["corrected_csv"]),
        "corrected_dfn_raw_vtk": str(paths["raw_vtk"]),
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
    fracture_csv = Path(config["fracture_points_csv"]).resolve()
    samples_root = Path(config["real_well_samples_root"]).resolve()
    initial_summary_json = Path(config.get("initial_dfn_summary_json", "")).resolve()
    surface_dir = Path(config["surface_dir"]).resolve() if config.get("surface_dir") else None
    for label, path in [("initial_dfn_csv", initial_csv), ("fracture_points_csv", fracture_csv), ("real_well_samples_root", samples_root)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if surface_dir is not None and not surface_dir.exists():
        raise FileNotFoundError(f"surface_dir does not exist: {surface_dir}")

    initial_df = load_initial_dfn(initial_csv)
    step4_control_df = load_control_points(fracture_csv, dict(config["target_block"]))
    raw_step4_count = int(len(step4_control_df))
    step4_control_df = aggregate_step4_controls_to_events(step4_control_df, config=config)
    config["_raw_step4_control_count"] = raw_step4_count
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
    tracks = load_real_well_tracks(samples_root, set(control_df["WellName"].astype(str).unique()))
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    surface_lookup = load_surface_time_lookup(surface_dir)
    track_dist = nearest_track_distances(control_df, tracks, time_scale=time_scale)
    before_dist = nearest_patch_distances(initial_df, control_df, time_scale=time_scale)
    corrected_df, audit_df = apply_well_controls(
        initial_df=initial_df,
        control_df=control_df,
        track_distances=track_dist,
        config=config,
        surface_lookup=surface_lookup,
    )
    after_dist = nearest_patch_distances(corrected_df, control_df, time_scale=time_scale)

    display_z_scale = float(config.get("display_z_scale", 5.0))
    use_dip_geometry = bool(config.get("use_dip_geometry", False))
    geometry_time_scale = float(config.get("geometry_time_scale_m_per_ms", 1.0))
    corrected_df = refresh_well_control_vertices(
        corrected_df,
        use_dip_geometry=use_dip_geometry,
        geometry_time_scale_m_per_ms=geometry_time_scale,
    )
    corrected_df.to_csv(paths["corrected_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    write_legacy_vtk(
        paths["raw_vtk"],
        corrected_df,
        "well_corrected_dfn_raw_time",
        display=False,
        display_z_scale=display_z_scale,
        use_dip_geometry=use_dip_geometry,
        geometry_time_scale_m_per_ms=geometry_time_scale,
    )
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
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Corrected DFN CSV: {paths['corrected_csv']}")
    print(f"Corrected DFN raw VTK: {paths['raw_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Control points: {len(control_df)} corrected patches: {len(corrected_df)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
