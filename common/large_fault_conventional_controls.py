"""Soft Step4 conventional-log support for inferred large-fault patches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def apply_large_fault_conventional_support(
    patch_df: pd.DataFrame,
    control_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Raise large-fault confidence near Step4 conventional-log prediction points.

    The function deliberately does not move or rotate a large fault. Conventional
    log points constrain confidence and audit provenance; the interpreted and
    seismically inferred fault geometry remains the spatial source of truth.
    """
    options = dict(config.get("large_conventional_control_policy", {}))
    audit_columns = [
        "PatchID",
        "LayerGroup",
        "WellName",
        "ControlID",
        "XYDistanceM",
        "TimeDistanceMs",
        "Influence",
        "OriginalConfidence",
        "CorrectedConfidence",
    ]
    if not bool(options.get("enabled", False)):
        return patch_df.copy(), pd.DataFrame(columns=audit_columns), {
            "enabled": False,
            "candidate_patch_count": 0,
            "supported_patch_count": 0,
        }

    out = patch_df.copy()
    defaults = {
        "LargeConventionalControlSupported": 0,
        "LargeConventionalControlInfluence": 0.0,
        "LargeConventionalControlID": "",
        "LargeConventionalControlWellName": "",
        "LargeConventionalControlDistanceM": np.nan,
        "LargeConventionalControlTimeDistanceMs": np.nan,
        "LargeConventionalControlOriginalConfidence": np.nan,
        "LargeConventionalControlCorrectedConfidence": np.nan,
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default

    controls = control_df.copy()
    control_source = controls.get("ControlSource", pd.Series("", index=controls.index)).astype(str).str.lower()
    imaging = _numeric(controls, "IsImagingGroundTruth", 0.0).fillna(0.0).astype(int).eq(1)
    step4 = control_source.str.contains("step4", na=False) & ~imaging
    probability = _numeric(controls, "PredFractureProb", np.nan)
    density_score = _numeric(controls, "DensityScore", np.nan)
    min_probability = float(options.get("min_probability", 0.5))
    keep_score = probability.fillna(1.0).ge(min_probability) | density_score.fillna(0.0).ge(
        float(options.get("min_density_score", 0.5))
    )
    controls = controls.loc[step4 & keep_score].copy()
    for column in ("X", "Y", "TIME", "LayerGroup"):
        if column not in controls.columns:
            controls = controls.iloc[0:0]
            break
    if controls.empty:
        return out, pd.DataFrame(columns=audit_columns), {
            "enabled": True,
            "candidate_patch_count": 0,
            "supported_patch_count": 0,
            "reason": "no_eligible_step4_controls",
        }

    large_mask = (
        out.get("FractureScale", pd.Series("", index=out.index)).astype(str).str.lower().eq("large")
        & out.get("SourceType", pd.Series("", index=out.index)).astype(str).str.contains("large_inferred", na=False)
    )
    candidate_count = int(large_mask.sum())
    xy_radius = max(float(options.get("xy_radius_m", 200.0)), 1.0)
    time_radius = max(float(options.get("time_radius_ms", 60.0)), 1.0)
    confidence_blend = float(np.clip(options.get("confidence_blend", 0.20), 0.0, 1.0))
    min_influence = float(np.clip(options.get("min_influence", 0.05), 0.0, 1.0))
    rows: list[dict[str, Any]] = []

    for layer, layer_controls in controls.groupby("LayerGroup", dropna=False):
        patches = out[large_mask & out["LayerGroup"].astype(str).eq(str(layer))]
        if patches.empty:
            continue
        control_coords = np.column_stack(
            [
                _numeric(layer_controls, "X").to_numpy(dtype=float) / xy_radius,
                _numeric(layer_controls, "Y").to_numpy(dtype=float) / xy_radius,
                _numeric(layer_controls, "TIME").to_numpy(dtype=float) / time_radius,
            ]
        )
        valid = np.isfinite(control_coords).all(axis=1)
        layer_controls = layer_controls.iloc[np.flatnonzero(valid)].reset_index(drop=True)
        control_coords = control_coords[valid]
        if not len(layer_controls):
            continue
        tree = cKDTree(control_coords)
        patch_coords = np.column_stack(
            [
                _numeric(patches, "CenterX").to_numpy(dtype=float) / xy_radius,
                _numeric(patches, "CenterY").to_numpy(dtype=float) / xy_radius,
                _numeric(patches, "CenterTime").to_numpy(dtype=float) / time_radius,
            ]
        )
        distance, positions = tree.query(patch_coords, k=1)
        for patch_index, scaled_distance, control_position in zip(patches.index, distance, positions):
            if not np.isfinite(scaled_distance) or float(scaled_distance) > 1.0:
                continue
            influence = float(np.exp(-0.5 * float(scaled_distance) ** 2))
            if influence < min_influence:
                continue
            control = layer_controls.iloc[int(control_position)]
            old_confidence = float(
                pd.to_numeric(pd.Series([out.at[patch_index, "Confidence"]]), errors="coerce").fillna(0.75).iloc[0]
            )
            new_confidence = old_confidence + confidence_blend * influence * (1.0 - old_confidence)
            new_confidence = float(np.clip(new_confidence, 0.0, 1.0))
            out.at[patch_index, "Confidence"] = new_confidence
            out.at[patch_index, "LargeConventionalControlSupported"] = 1
            out.at[patch_index, "LargeConventionalControlInfluence"] = influence
            out.at[patch_index, "LargeConventionalControlID"] = str(
                control.get("WellControlSampleID", control.get("SampleID", ""))
            )
            out.at[patch_index, "LargeConventionalControlWellName"] = str(control.get("WellName", ""))
            out.at[patch_index, "LargeConventionalControlDistanceM"] = float(
                np.hypot(
                    float(out.at[patch_index, "CenterX"]) - float(control["X"]),
                    float(out.at[patch_index, "CenterY"]) - float(control["Y"]),
                )
            )
            out.at[patch_index, "LargeConventionalControlTimeDistanceMs"] = abs(
                float(out.at[patch_index, "CenterTime"]) - float(control["TIME"])
            )
            out.at[patch_index, "LargeConventionalControlOriginalConfidence"] = old_confidence
            out.at[patch_index, "LargeConventionalControlCorrectedConfidence"] = new_confidence
            rows.append(
                {
                    "PatchID": str(out.at[patch_index, "PatchID"]),
                    "LayerGroup": str(layer),
                    "WellName": str(control.get("WellName", "")),
                    "ControlID": str(control.get("WellControlSampleID", control.get("SampleID", ""))),
                    "XYDistanceM": float(out.at[patch_index, "LargeConventionalControlDistanceM"]),
                    "TimeDistanceMs": float(out.at[patch_index, "LargeConventionalControlTimeDistanceMs"]),
                    "Influence": influence,
                    "OriginalConfidence": old_confidence,
                    "CorrectedConfidence": new_confidence,
                }
            )

    audit = pd.DataFrame(rows, columns=audit_columns)
    return out, audit, {
        "enabled": True,
        "candidate_patch_count": candidate_count,
        "eligible_control_count": int(len(controls)),
        "supported_patch_count": int(len(audit)),
        "xy_radius_m": xy_radius,
        "time_radius_ms": time_radius,
        "confidence_blend": confidence_blend,
        "mean_influence": float(audit["Influence"].mean()) if len(audit) else 0.0,
        "mean_confidence_increase": float(
            (audit["CorrectedConfidence"] - audit["OriginalConfidence"]).mean()
        ) if len(audit) else 0.0,
        "geometry_changed": False,
    }
