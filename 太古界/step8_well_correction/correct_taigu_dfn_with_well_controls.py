#!/usr/bin/env python3
"""Step8 (太古界 v2): correct small-scale patches with well controls.

Improvements vs v1:
  * per-row InputSegmentPath join -> full imaging point set (405: 316 points);
  * 3D KDTree matching per layer (x, y, time*time_scale), k<=30 candidates,
    xy 80 m / time 30 ms;
  * imaging well region correction: nearby small patches have their density
    blended toward imaging density (radius 150 m / 40 ms, blend 0.65, gaussian
    influence) without changing geometry;
  * matched controls replace orientation (imaging) and optionally shift the
    patch center along dip (20-70 m); unmatched imaging controls add patches;
  * weak Step4 events (406) adjust matched patches (center/size/density).

Outputs (config.output_dir):
  well_corrected_fracture_patches.csv / .vtk
  well_control_correction_audit.csv
  step8_summary.json   status=pass
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct Step7A patches with imaging/Step4 well controls (太古界 v2).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--max-patches", type=int, default=0, help="Smoke-test cap.")
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


def write_legacy_vtk(path: Path, points: np.ndarray, quads: np.ndarray, cell_data: dict[str, np.ndarray]) -> None:
    lines = [
        "# vtk DataFile Version 3.0",
        "taigu_well_corrected_fracture_patches_raw_time",
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


def patch_quads(patches: pd.DataFrame, z_scale: float) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    quads: list[np.ndarray] = []
    for _, patch in patches.iterrows():
        center = np.array([patch["X"], patch["Y"], patch["TIME"] * z_scale])
        strike_rad = np.radians(patch["AzimuthDeg"])
        strike = np.array([np.cos(strike_rad), np.sin(strike_rad), 0.0])
        dip_dir = np.array([-np.sin(strike_rad), np.cos(strike_rad), 0.0])
        dip_rad = np.radians(patch["DipDeg"])
        dip_vec = np.array([np.sin(dip_rad) * dip_dir[0], np.sin(dip_rad) * dip_dir[1], np.cos(dip_rad)])
        half_l = patch["PatchLengthM"] / 2.0
        half_h = patch["PatchHeightMs"] * z_scale / 2.0
        corners = [
            center + half_l * strike + half_h * dip_vec,
            center + half_l * strike - half_h * dip_vec,
            center - half_l * strike - half_h * dip_vec,
            center - half_l * strike + half_h * dip_vec,
        ]
        base = len(points)
        points.extend(corners)
        quads.append(np.array([base, base + 1, base + 2, base + 3], dtype=np.int64))
    return (np.stack(points) if points else np.empty((0, 3))), (np.stack(quads) if quads else np.empty((0, 4), dtype=np.int64))


def load_step3_controls(config: dict[str, Any], well: str) -> pd.DataFrame:
    """Per-row InputSegmentPath join (multi-segment groups fully joined)."""
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
                tolerance=float(config["md_merge_tolerance"]),
            )
            parts.append(merged)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    for column in ("Density", "HasFractureDensity", "GT_POINT_FLAG", "FracAzimuth", "FracDip"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["X", "Y", "TIME", "Density"]).copy()


def load_step4_points_with_xy(config: dict[str, Any], well: str) -> pd.DataFrame:
    points = pd.read_csv(config["step4_fracture_points_csv"], encoding="utf-8-sig")
    points = points[points["WellName"] == well].copy()
    if points.empty:
        return points
    for column in ("MD", "TVD"):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    segment_dir = Path(config["step2_segments_root"]) / well
    segment_files = sorted(segment_dir.glob("*.csv")) if segment_dir.exists() else []
    parts = []
    for segment_file in segment_files:
        segment = pd.read_csv(segment_file, encoding="utf-8-sig")
        for column in ("MD", "TVD", "X", "Y", "TIME"):
            segment[column] = pd.to_numeric(segment[column], errors="coerce")
        part = pd.merge_asof(
            points.sort_values("MD"),
            segment[["MD", "X", "Y", "TIME"]].sort_values("MD"),
            on="MD",
            direction="nearest",
            tolerance=float(config["md_merge_tolerance"]),
        )
        parts.append(part)
    if not parts:
        return points
    frame = pd.concat(parts, ignore_index=True)
    return frame.dropna(subset=["X", "Y", "TIME"]).drop_duplicates(subset=["WellName", "MD"]).copy()


def aggregate_step4_events(points: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if points.empty:
        return points
    gap_ms = float(config["step4_event_gap_ms"])
    max_span_ms = float(config["step4_event_max_span_ms"])
    work = points.sort_values("TIME").reset_index(drop=True)
    events: list[list[int]] = []
    current: list[int] = []
    event_start: float | None = None
    for idx, row in work.iterrows():
        if not current:
            current = [idx]
            event_start = float(row["TIME"])
            continue
        if float(row["TIME"]) - event_start > max_span_ms or float(row["TIME"]) - float(work.loc[current[-1], "TIME"]) > gap_ms:
            events.append(current)
            current = [idx]
            event_start = float(row["TIME"])
        else:
            current.append(idx)
    if current:
        events.append(current)
    rows = []
    for group in events:
        sub = work.loc[group]
        rows.append(
            {
                "WellName": sub["WellName"].iloc[0],
                "X": float(sub["X"].mean()),
                "Y": float(sub["Y"].mean()),
                "TIME": float(sub["TIME"].mean()),
                "EventPointCount": int(len(sub)),
                "EventTimeSpanMs": float(sub["TIME"].max() - sub["TIME"].min()),
                "Density": float(sub["PredDensity"].max()),
                "ControlSource": "step4_event",
            }
        )
    return pd.DataFrame(rows)


def build_layer_patch_tree(patch_df: pd.DataFrame, layer: str, time_scale: float) -> tuple[cKDTree | None, np.ndarray]:
    sub = patch_df[patch_df["LayerGroup"].astype(str).eq(layer)]
    if sub.empty:
        return None, np.empty(0, dtype=np.int64)
    coords = np.column_stack(
        [
            sub["X"].to_numpy(dtype=np.float64),
            sub["Y"].to_numpy(dtype=np.float64),
            sub["TIME"].to_numpy(dtype=np.float64) * time_scale,
        ]
    )
    return cKDTree(coords), sub.index.to_numpy(dtype=np.int64)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_csv = output_dir / "well_corrected_fracture_patches.csv"
    vtk_path = output_dir / "well_corrected_fracture_patches.vtk"
    audit_csv = output_dir / "well_control_correction_audit.csv"
    summary_path = output_dir / "step8_summary.json"
    for path in (corrected_csv, vtk_path, audit_csv, summary_path):
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing Step8 output: {path}")
    started = time.time()

    patches = pd.read_csv(config["step7a_patches_csv"], encoding="utf-8-sig")
    if args.max_patches > 0:
        patches = patches.head(int(args.max_patches)).copy()
    for column in ("X", "Y", "TIME", "Density", "DipDeg", "AzimuthDeg", "PatchLengthM", "PatchHeightMs", "PatchAreaM2"):
        patches[column] = pd.to_numeric(patches[column], errors="coerce")
    patches = patches.dropna(subset=["X", "Y", "TIME"]).reset_index(drop=True)
    if "LayerGroup" not in patches.columns:
        patches["LayerGroup"] = "上部复合层"
    patches["CorrectionReason"] = "far_from_or_not_selected_by_well_control"
    patches["ControlSource"] = "none"
    patches["temporary_neighbor_time_depth"] = 0
    patches["ImagingWellRegionCorrected"] = 0
    patches["ImagingWellRegionInfluence"] = 0.0
    patches["ImagingWellRegionOriginalDensity"] = np.nan
    patches["ImagingWellRegionCorrectedDensity"] = np.nan
    patches["WellControlCenterOffsetM"] = 0.0
    initial_patch_count = int(len(patches))
    z_scale = float(config["display_z_scale_m_per_ms"])
    time_scale = float(config["time_scale_m_per_ms"])
    audit_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)

    # ---- imaging well region correction (density blend, no geometry change) ----
    if bool(config.get("enable_imaging_well_region_correction", False)):
        xy_radius = float(config["imaging_well_region_xy_radius_m"])
        time_radius = float(config["imaging_well_region_time_radius_ms"])
        blend = float(np.clip(config["imaging_well_region_density_blend"], 0.0, 1.0))
        min_influence = float(config["imaging_well_region_min_influence"])
        for well in config["strong_control_wells"]:
            controls = load_step3_controls(config, well)
            positive = controls[controls["HasFractureDensity"].fillna(0).astype(int) == 1]
            if positive.empty:
                continue
            for layer, group in positive.groupby("StrataName"):
                candidates = patches[
                    (patches["LayerGroup"].astype(str).eq(str(layer)))
                    & (patches["FractureScale"].astype(str).eq("small"))
                    & (patches["ControlSource"].eq("none"))
                ].copy()
                if candidates.empty:
                    continue
                control_coords = np.column_stack(
                    [
                        group["X"].to_numpy(dtype=np.float64) / xy_radius,
                        group["Y"].to_numpy(dtype=np.float64) / xy_radius,
                        group["TIME"].to_numpy(dtype=np.float64) / time_radius,
                    ]
                )
                tree = cKDTree(control_coords)
                candidate_coords = np.column_stack(
                    [
                        candidates["X"].to_numpy(dtype=np.float64) / xy_radius,
                        candidates["Y"].to_numpy(dtype=np.float64) / xy_radius,
                        candidates["TIME"].to_numpy(dtype=np.float64) / time_radius,
                    ]
                )
                distance, positions = tree.query(candidate_coords, k=1)
                for patch_idx, dist, pos in zip(candidates.index, distance, positions):
                    if not np.isfinite(dist) or float(dist) > 1.0:
                        continue
                    influence = float(np.exp(-0.5 * float(dist) ** 2))
                    if influence < min_influence:
                        continue
                    control_density = float(group.iloc[int(pos)]["Density"])
                    original = float(patches.loc[patch_idx, "Density"])
                    corrected = original * (1.0 - blend * influence) + control_density * blend * influence
                    patches.loc[patch_idx, "ImagingWellRegionCorrected"] = 1
                    patches.loc[patch_idx, "ImagingWellRegionInfluence"] = influence
                    patches.loc[patch_idx, "ImagingWellRegionOriginalDensity"] = original
                    patches.loc[patch_idx, "ImagingWellRegionCorrectedDensity"] = corrected
                    patches.loc[patch_idx, "Density"] = corrected
                    audit_rows.append(
                        {
                            "PatchID": patches.loc[patch_idx, "PatchID"],
                            "ControlSource": "imaging_gt",
                            "CorrectionReason": "imaging_region_enhanced",
                            "ControlWell": well,
                            "ImagingWellRegionInfluence": influence,
                            "OriginalDensity": original,
                            "CorrectedDensity": corrected,
                        }
                    )

    # ---- strong controls: imaging points -> replace/add ----
    used_patch_indices: set[int] = set()
    trees: dict[str, tuple[cKDTree | None, np.ndarray]] = {}
    for layer in ("上部复合层", "太古界风化壳"):
        trees[layer] = build_layer_patch_tree(patches, layer, time_scale)
    added_records: list[dict[str, Any]] = []
    for well in config["strong_control_wells"]:
        strong = load_step3_controls(config, well)
        strong_positive = strong[strong["GT_POINT_FLAG"].fillna(0).astype(int) == 1].copy()
        for _, control in strong_positive.iterrows():
            layer = str(control["StrataName"])
            tree, layer_idx = trees[layer]
            if tree is None or layer_idx.size == 0:
                matched_idx = None
            else:
                k = min(int(config.get("max_query_candidates", 30)), int(layer_idx.size))
                query = np.asarray([[float(control["X"]), float(control["Y"]), float(control["TIME"]) * time_scale]])
                distances, positions = tree.query(query, k=k)
                distances = np.atleast_1d(distances[0] if np.asarray(distances).ndim > 1 else distances)
                positions = np.atleast_1d(positions[0] if np.asarray(positions).ndim > 1 else positions)
                matched_idx = None
                for dist, pos in zip(distances, positions):
                    patch_idx = int(layer_idx[int(pos)])
                    if patch_idx in used_patch_indices:
                        continue
                    patch = patches.loc[patch_idx]
                    xy_dist = float(np.hypot(patch["X"] - control["X"], patch["Y"] - control["Y"]))
                    time_dist = abs(float(patch["TIME"]) - float(control["TIME"]))
                    if xy_dist <= float(config["xy_search_radius_m"]) and time_dist <= float(config["time_search_radius_ms"]):
                        matched_idx = patch_idx
                        break
            if matched_idx is not None:
                patch_idx = matched_idx
                used_patch_indices.add(patch_idx)
                original_x, original_y, original_time = (
                    float(patches.loc[patch_idx, "X"]),
                    float(patches.loc[patch_idx, "Y"]),
                    float(patches.loc[patch_idx, "TIME"]),
                )
                offset_m = float(
                    rng.uniform(
                        config["well_control_center_offset_min_m"],
                        config["well_control_center_offset_max_m"],
                    )
                )
                strike_rad = np.radians(float(patches.loc[patch_idx, "AzimuthDeg"]))
                dip_dir = np.array([-np.sin(strike_rad), np.cos(strike_rad), 0.0])
                new_x = float(control["X"]) + dip_dir[0] * offset_m
                new_y = float(control["Y"]) + dip_dir[1] * offset_m
                patches.loc[patch_idx, "X"] = new_x
                patches.loc[patch_idx, "Y"] = new_y
                patches.loc[patch_idx, "TIME"] = float(control["TIME"])
                if pd.notna(control.get("FracAzimuth")) and pd.notna(control.get("FracDip")):
                    patches.loc[patch_idx, "AzimuthDeg"] = float(control["FracAzimuth"]) % 180.0
                    patches.loc[patch_idx, "DipDeg"] = float(np.clip(control["FracDip"], 0.0, 90.0))
                patches.loc[patch_idx, "CorrectionReason"] = "imaging_replaced"
                patches.loc[patch_idx, "ControlSource"] = "imaging_gt"
                patches.loc[patch_idx, "WellControlCenterOffsetM"] = offset_m
                patches.loc[patch_idx, "temporary_neighbor_time_depth"] = 1
                audit_rows.append(
                    {
                        "PatchID": patches.loc[patch_idx, "PatchID"],
                        "ControlSource": "imaging_gt",
                        "CorrectionReason": "imaging_replaced",
                        "ControlWell": well,
                        "OriginalX": original_x,
                        "OriginalY": original_y,
                        "OriginalTIME": original_time,
                        "WellControlCenterOffsetM": offset_m,
                    }
                )
            else:
                added_records.append(
                    {
                        "PatchID": f"taigu_small_added_{len(added_records):07d}",
                        "TraceIdx": -1,
                        "X": float(control["X"]),
                        "Y": float(control["Y"]),
                        "TIME": float(control["TIME"]),
                        "LayerGroup": layer,
                        "Density": float(control["Density"]),
                        "DipDeg": float(np.clip(control["FracDip"] if pd.notna(control.get("FracDip")) else 60.0, 0.0, 90.0)),
                        "AzimuthDeg": float(control["FracAzimuth"] % 180.0 if pd.notna(control.get("FracAzimuth")) else 0.0),
                        "LocalPcaDipDeg": np.nan,
                        "LocalPcaAzimuthDeg": np.nan,
                        "OrientationBaseSource": "imaging_ground_truth",
                        "OrientationFamily": "imaging_ground_truth",
                        "PatchLengthM": float(np.mean(config["added_patch_length_m"])),
                        "PatchHeightMs": float(np.mean(config["added_patch_height_ms"])),
                        "PatchAreaM2": float(np.mean(config["added_patch_length_m"]) * np.mean(config["added_patch_height_ms"]) * z_scale),
                        "FractureScale": "small",
                        "WindowCode": 1,
                        "CorrectionReason": "imaging_added",
                        "ControlSource": "imaging_gt",
                        "temporary_neighbor_time_depth": 1,
                        "ImagingWellRegionCorrected": 0,
                        "ImagingWellRegionInfluence": 0.0,
                        "ImagingWellRegionOriginalDensity": np.nan,
                        "ImagingWellRegionCorrectedDensity": np.nan,
                        "WellControlCenterOffsetM": 0.0,
                    }
                )
                audit_rows.append(
                    {
                        "PatchID": added_records[-1]["PatchID"],
                        "ControlSource": "imaging_gt",
                        "CorrectionReason": "imaging_added",
                        "ControlWell": well,
                    }
                )
    if added_records:
        patches = pd.concat([patches, pd.DataFrame(added_records)], ignore_index=True)

    # ---- weak controls: 406 Step4 events -> adjust matched patches ----
    weak_points = load_step4_points_with_xy(config, config["weak_control_well"])
    weak_events = aggregate_step4_events(weak_points, config)
    patch_xy = patches[["X", "Y"]].to_numpy(dtype=np.float64)
    patch_time = patches["TIME"].to_numpy(dtype=np.float64)
    if len(weak_events):
        event_xy = weak_events[["X", "Y"]].to_numpy(dtype=np.float64)
        event_time = weak_events["TIME"].to_numpy(dtype=np.float64)
        tree = cKDTree(patch_xy)
        for pos in range(len(weak_events)):
            dist, index = tree.query(event_xy[pos : pos + 1], k=1)
            patch_idx = int(index[0])
            time_dist = abs(patch_time[patch_idx] - event_time[pos]) * time_scale
            if dist[0] > float(config["xy_search_radius_m"]) or time_dist > float(config["time_search_radius_ms"]):
                continue
            if patches.loc[patch_idx, "ControlSource"] == "imaging_gt":
                continue
            event = weak_events.iloc[pos]
            patches.loc[patch_idx, "X"] = float(event["X"])
            patches.loc[patch_idx, "Y"] = float(event["Y"])
            patches.loc[patch_idx, "TIME"] = float(event["TIME"])
            patches.loc[patch_idx, "Density"] = max(float(patches.loc[patch_idx, "Density"]), float(event["Density"]))
            patches.loc[patch_idx, "PatchLengthM"] = float(rng.uniform(*config["adjusted_patch_length_m"]))
            patches.loc[patch_idx, "PatchHeightMs"] = float(rng.uniform(*config["adjusted_patch_height_ms"]))
            patches.loc[patch_idx, "PatchAreaM2"] = float(
                patches.loc[patch_idx, "PatchLengthM"] * patches.loc[patch_idx, "PatchHeightMs"] * z_scale
            )
            patches.loc[patch_idx, "CorrectionReason"] = "step4_event_adjusted"
            patches.loc[patch_idx, "ControlSource"] = "step4_event"
            audit_rows.append(
                {
                    "PatchID": patches.loc[patch_idx, "PatchID"],
                    "ControlSource": "step4_event",
                    "CorrectionReason": "step4_event_adjusted",
                    "ControlWell": config["weak_control_well"],
                }
            )

    patches.to_csv(corrected_csv, index=False, encoding="utf-8-sig")
    points_out, quads_out = patch_quads(patches, z_scale)
    cell_data = {
        "PatchID": np.arange(len(patches), dtype=np.int64),
        "Density": patches["Density"].to_numpy(dtype=np.float64),
        "DipDeg": patches["DipDeg"].to_numpy(dtype=np.float64),
        "AzimuthDeg": patches["AzimuthDeg"].to_numpy(dtype=np.float64),
        "PatchAreaM2": patches["PatchAreaM2"].to_numpy(dtype=np.float64),
        "FractureScale": np.full(len(patches), 1, dtype=np.int32),
        "WindowCode": patches["WindowCode"].to_numpy(dtype=np.int32),
    }
    write_legacy_vtk(vtk_path, points_out, quads_out, cell_data)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")

    reason_counts = patches["CorrectionReason"].value_counts().to_dict()
    region_corrected = int(patches["ImagingWellRegionCorrected"].sum())
    strong_points = 0
    for well in config["strong_control_wells"]:
        controls = load_step3_controls(config, well)
        strong_points += int((controls["GT_POINT_FLAG"].fillna(0).astype(int) == 1).sum())
    checks = {
        "all_patches_small_scale": bool((patches["FractureScale"] == "small").all()),
        "orientation_ranges_valid": bool(patches["DipDeg"].between(0, 90).all() and patches["AzimuthDeg"].between(0, 180).all()),
        "geometry_finite": bool(np.isfinite(patches[["X", "Y", "TIME", "PatchAreaM2"]]).all().all()),
        "correction_applied": any(
            reason in reason_counts for reason in ("imaging_replaced", "imaging_added", "step4_event_adjusted")
        ),
        "imaging_region_correction_applied": region_corrected > 0,
        "full_imaging_point_set_used": strong_points >= 300,
        "audit_consistent": int(len(audit)) >= int(
            reason_counts.get("imaging_replaced", 0)
            + reason_counts.get("imaging_added", 0)
            + reason_counts.get("step4_event_adjusted", 0)
        ),
        "vtk_exists": vtk_path.exists(),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "output_paths": {
            "corrected_patches_csv": str(corrected_csv),
            "corrected_patches_vtk": str(vtk_path),
            "correction_audit_csv": str(audit_csv),
            "summary_json": str(summary_path),
        },
        "initial_patch_count": initial_patch_count,
        "corrected_patch_count": int(len(patches)),
        "correction_reason_counts": reason_counts,
        "imaging_region_corrected_patch_count": region_corrected,
        "strong_control_point_count": strong_points,
        "weak_event_count": int(len(weak_events)),
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
