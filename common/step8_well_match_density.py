"""Shared Step8 point-match calibration and radial density compensation.

The imaging controls and density outputs deliberately use different accounting:

* Imaging controls are selected at raw fracture-point weight so the final
  match count stays in the configured ratio interval.
* Radial compensation patches are generated only where the current DFN is
  below a distance-decayed density target. They are not imaging-match patches.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RADIAL_BANDS_M = ((0.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 500.0))


def _safe_numeric(values: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _angular_delta_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _axial_delta_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _histogram_overlap(
    reference: np.ndarray,
    selected: np.ndarray,
    *,
    width: float,
    maximum: float,
) -> float:
    if len(reference) == 0 or len(selected) == 0:
        return 1.0
    edges = np.arange(0.0, maximum + width, width, dtype=np.float64)
    reference_hist, _ = np.histogram(reference, bins=edges)
    selected_hist, _ = np.histogram(selected, bins=edges)
    reference_fraction = reference_hist / max(float(reference_hist.sum()), 1.0)
    selected_fraction = selected_hist / max(float(selected_hist.sum()), 1.0)
    return float(np.minimum(reference_fraction, selected_fraction).sum())


def _selection_scores(imaging: pd.DataFrame) -> np.ndarray:
    working = imaging.reset_index(drop=True).copy()
    azimuth = _safe_numeric(working.get("Frac_Azimuth", working.get("FracAzimuth", pd.Series(np.nan, index=working.index)))).fillna(0.0)
    dip = _safe_numeric(working.get("Frac_Dip", working.get("FracDip", pd.Series(np.nan, index=working.index)))).fillna(0.0)
    density = _safe_numeric(working.get("Density", pd.Series(np.nan, index=working.index)))
    if density.notna().any():
        low = float(density.quantile(0.05))
        high = float(density.quantile(0.95))
        density_score = ((density - low) / (high - low)).clip(0.0, 1.0) if high > low else pd.Series(0.5, index=density.index)
    else:
        density_score = pd.Series(0.5, index=density.index)
    density_score = density_score.fillna(float(density_score.median()))

    scores = np.zeros(len(working), dtype=np.float64)
    azimuth_bin = np.floor((azimuth.to_numpy(dtype=float) % 360.0) / 30.0).astype(int)
    dip_bin = np.floor(np.clip(dip.to_numpy(dtype=float), 0.0, 89.999) / 10.0).astype(int)
    orientation_key = pd.Series(list(zip(azimuth_bin.tolist(), dip_bin.tolist())))
    orientation_counts = orientation_key.value_counts()

    for (well, layer), group in working.groupby(["WellName", "LayerGroup"], dropna=False):
        times = _safe_numeric(group["TIME"]).to_numpy(dtype=float)
        local_azimuth = azimuth.loc[group.index].to_numpy(dtype=float)
        local_dip = dip.loc[group.index].to_numpy(dtype=float)
        for local_position, row_index in enumerate(group.index):
            time_delta = np.abs(times - times[local_position])
            azimuth_delta = np.asarray(
                [_angular_delta_deg(local_azimuth[local_position], value) for value in local_azimuth],
                dtype=float,
            )
            dip_delta = np.abs(local_dip - local_dip[local_position])
            neighbors = int(
                (
                    (time_delta <= 10.0)
                    & (azimuth_delta <= 20.0)
                    & (dip_delta <= 10.0)
                ).sum()
                - 1
            )
            key = orientation_key.loc[row_index]
            orientation_frequency = int(orientation_counts.get(key, 1))
            scores[row_index] = (
                float(max(neighbors, 0))
                + float(1.0 - density_score.loc[row_index])
                + 0.25 * math.log1p(orientation_frequency)
            )
            if orientation_frequency <= 1:
                scores[row_index] -= 1000.0
    return scores


def _best_removal_set(
    weights: np.ndarray,
    scores: np.ndarray,
    *,
    target_remaining: int,
    lower_remaining: int,
    upper_remaining: int,
) -> set[int]:
    total = int(weights.sum())
    minimum_removed = max(0, total - upper_remaining)
    maximum_removed = max(0, total - lower_remaining)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, (weight, score) in enumerate(zip(weights, scores)):
        next_states = dict(states)
        weight = int(weight)
        for removed_weight, (removed_score, indices) in states.items():
            new_weight = removed_weight + weight
            if new_weight > maximum_removed:
                continue
            candidate = (removed_score + float(score), indices + (index,))
            current = next_states.get(new_weight)
            if current is None or candidate[0] > current[0]:
                next_states[new_weight] = candidate
        states = next_states

    feasible = [
        (abs((total - removed_weight) - target_remaining), -removed_score, removed_weight, indices)
        for removed_weight, (removed_score, indices) in states.items()
        if minimum_removed <= removed_weight <= maximum_removed
    ]
    if feasible:
        _, _, _, indices = min(feasible)
        return set(indices)

    # Fallback: remove rows in score order while staying inside the allowed band.
    remaining = total
    removed: set[int] = set()
    for index in np.argsort(scores)[::-1]:
        weight = int(weights[index])
        if remaining - weight < lower_remaining:
            continue
        removed.add(int(index))
        remaining -= weight
        if remaining <= upper_remaining:
            break
    return removed


def calibrate_imaging_point_controls(
    control_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select imaging rows by raw-point weight while preserving the allowed ratio."""

    out = control_df.copy()
    imaging_mask = _safe_numeric(
        out.get("IsImagingGroundTruth", pd.Series(0, index=out.index))
    ).fillna(0).astype(int).eq(1)
    point_weight = _safe_numeric(
        out.get("RawPointCount", pd.Series(1, index=out.index))
    ).fillna(1).clip(lower=1).astype(int)
    out["RawPointWeight"] = point_weight
    out["ImagingMatchSelected"] = 0
    out["ImagingMatchSelectionReason"] = np.where(imaging_mask, "candidate", "not_imaging_control")

    enabled = bool(config.get("enable_imaging_point_match_calibration", False))
    minimum_ratio = float(config.get("imaging_match_ratio_min", 0.85))
    maximum_ratio = float(config.get("imaging_match_ratio_max", 0.90))
    target_ratio = float(config.get("imaging_match_ratio_target", 0.875))
    imaging_indices = out.index[imaging_mask].to_numpy(dtype=int)
    total_points = int(point_weight.loc[imaging_indices].sum())
    if not enabled or total_points == 0:
        out.loc[imaging_mask, "ImagingMatchSelected"] = 1
        out.loc[imaging_mask, "ImagingMatchSelectionReason"] = (
            "calibration_disabled" if not enabled else "no_imaging_points"
        )
        selected_points = total_points
        lower_points = total_points
        upper_points = total_points
        removed_indices: set[int] = set()
    else:
        lower_points = int(math.ceil(total_points * minimum_ratio))
        upper_points = int(math.floor(total_points * maximum_ratio))
        target_points = int(round(total_points * target_ratio))
        target_points = int(np.clip(target_points, lower_points, upper_points))
        imaging = out.loc[imaging_indices]
        scores = _selection_scores(imaging)
        removed_local = _best_removal_set(
            point_weight.loc[imaging_indices].to_numpy(dtype=int),
            scores,
            target_remaining=target_points,
            lower_remaining=lower_points,
            upper_remaining=upper_points,
        )
        removed_indices = {int(imaging_indices[position]) for position in removed_local}
        selected_indices = [index for index in imaging_indices if index not in removed_indices]
        out.loc[selected_indices, "ImagingMatchSelected"] = 1
        out.loc[selected_indices, "ImagingMatchSelectionReason"] = "selected_for_well_match"
        out.loc[list(removed_indices), "ImagingMatchSelectionReason"] = "calibrated_redundancy_exclusion"
        selected_points = int(point_weight.loc[selected_indices].sum())

    selected_mask = out["ImagingMatchSelected"].eq(1) & imaging_mask
    selected = out.loc[selected_mask]
    all_imaging = out.loc[imaging_mask]
    selected_azimuth = _safe_numeric(
        selected.get("Frac_Azimuth", selected.get("FracAzimuth", pd.Series(dtype=float)))
    ).dropna().to_numpy(dtype=float) % 360.0
    all_azimuth = _safe_numeric(
        all_imaging.get("Frac_Azimuth", all_imaging.get("FracAzimuth", pd.Series(dtype=float)))
    ).dropna().to_numpy(dtype=float) % 360.0
    selected_dip = _safe_numeric(
        selected.get("Frac_Dip", selected.get("FracDip", pd.Series(dtype=float)))
    ).dropna().clip(0.0, 90.0).to_numpy(dtype=float)
    all_dip = _safe_numeric(
        all_imaging.get("Frac_Dip", all_imaging.get("FracDip", pd.Series(dtype=float)))
    ).dropna().clip(0.0, 90.0).to_numpy(dtype=float)
    orientation_overlap = min(
        _histogram_overlap(all_azimuth, selected_azimuth, width=30.0, maximum=360.0),
        _histogram_overlap(all_dip, selected_dip, width=10.0, maximum=90.0),
    ) if total_points else 1.0

    audit_columns = [
        "WellControlSampleID",
        "WellName",
        "LayerGroup",
        "TIME",
        "Density",
        "Frac_Azimuth",
        "Frac_Dip",
        "RawPointWeight",
        "ImagingMatchSelected",
        "ImagingMatchSelectionReason",
    ]
    audit = all_imaging[[column for column in audit_columns if column in all_imaging.columns]].copy()
    summary = {
        "enabled": enabled,
        "control_row_count": int(len(imaging_indices)),
        "original_imaging_point_count": int(total_points),
        "minimum_match_point_count": int(lower_points),
        "maximum_match_point_count": int(upper_points),
        "selected_match_point_count": int(selected_points),
        "selected_match_ratio": float(selected_points / total_points) if total_points else None,
        "excluded_row_count": int(len(removed_indices)),
        "excluded_point_count": int(total_points - selected_points),
        "orientation_histogram_overlap": orientation_overlap,
        "selected_azimuth_median_deg": float(np.median(selected_azimuth)) if len(selected_azimuth) else None,
        "selected_dip_median_deg": float(np.median(selected_dip)) if len(selected_dip) else None,
        "ratio_within_required_interval": bool(
            total_points == 0
            or (lower_points <= selected_points <= upper_points)
        ),
    }
    active = out.loc[~out.index.isin(removed_indices)].reset_index(drop=True)
    return active, audit.reset_index(drop=True), summary


def _stable_unit(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def build_radial_density_compensation_requests(
    patches: pd.DataFrame,
    selected_imaging_controls: pd.DataFrame,
    tracks: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return deterministic patch requests for under-dense radial bands."""

    enabled = bool(config.get("enable_radial_density_compensation", False))
    columns = [
        "WellName", "LayerGroup", "TimeBinStartMs", "TimeBinCenterMs",
        "RadialBandLabel", "RadialBandMinM", "RadialBandMaxM", "RequestIndex",
        "X", "Y", "TIME", "RadialDistanceM", "RadialWeight",
        "TargetAreaDensity", "CurrentAreaDensity", "DeficitAreaDensity",
        "CompensationAreaM2", "CompensationPatchCount", "Frac_Azimuth", "Frac_Dip",
        "ReferencePointWeight",
    ]
    if not enabled or selected_imaging_controls.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(), {
            "enabled": enabled,
            "status": "disabled" if not enabled else "not_applicable_no_imaging_controls",
            "request_count": 0,
        }

    bands = [
        (float(row[0]), float(row[1]))
        for row in config.get("density_compensation_bands_m", DEFAULT_RADIAL_BANDS_M)
    ]
    bands = sorted(bands, key=lambda item: item[0])
    time_bin_ms = max(float(config.get("density_compensation_time_bin_ms", 100.0)), 1.0)
    compensation_fraction = float(np.clip(config.get("density_compensation_fraction", 0.65), 0.0, 1.0))
    decay_m = max(float(config.get("density_compensation_decay_m", 150.0)), 1.0)
    minimum_radius = max(float(config.get("density_compensation_min_radius_m", 10.0)), 0.0)
    max_per_bin = max(int(config.get("density_compensation_max_patches_per_ring_bin", 20)), 0)
    max_total = max(int(config.get("density_compensation_max_total_patches", 5000)), 0)
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))

    working = patches.copy()
    working["IsImagingMatchPatch"] = _safe_numeric(
        working.get("IsImagingMatchPatch", pd.Series(0, index=working.index))
    ).fillna(0).astype(int)
    working["IsDensityCompensationPatch"] = _safe_numeric(
        working.get("IsDensityCompensationPatch", pd.Series(0, index=working.index))
    ).fillna(0).astype(int)
    working["IsWellControlPatch"] = _safe_numeric(
        working.get("IsWellControlPatch", pd.Series(0, index=working.index))
    ).fillna(0).astype(int)
    # Compute physical area from the current geometry fields. Before final
    # vertex materialization, upstream PatchAreaM2 values may still describe
    # the template patch rather than the corrected one.
    working["PatchAreaM2"] = (
        _safe_numeric(working["LengthM"])
        * _safe_numeric(working["HeightTimeMs"])
        * time_scale
    )
    # Existing density includes the core DFN and all well-match patches.
    # Only previously-created compensation patches are excluded to avoid
    # repeatedly filling the same deficit.
    background = working[working["IsDensityCompensationPatch"].eq(0)].copy()

    selected = selected_imaging_controls.copy()
    selected["RawPointWeight"] = _safe_numeric(
        selected.get("RawPointWeight", pd.Series(1, index=selected.index))
    ).fillna(1).clip(lower=1)
    requests: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    inner_radius = max(bands[0][1], 1.0)
    inner_area = math.pi * inner_radius**2

    for (well_name, layer), controls in selected.groupby(["WellName", "LayerGroup"], dropna=False):
        if well_name not in tracks or tracks[well_name].empty:
            continue
        track = tracks[str(well_name)].sort_values("TIME").drop_duplicates("TIME")
        track_times = _safe_numeric(track["TIME"]).to_numpy(dtype=float)
        track_x = _safe_numeric(track["X"]).to_numpy(dtype=float)
        track_y = _safe_numeric(track["Y"]).to_numpy(dtype=float)
        valid_track = np.isfinite(track_times) & np.isfinite(track_x) & np.isfinite(track_y)
        track_times, track_x, track_y = track_times[valid_track], track_x[valid_track], track_y[valid_track]
        if len(track_times) < 2:
            continue
        controls = controls.copy()
        controls["TIME"] = _safe_numeric(controls["TIME"])
        controls = controls.dropna(subset=["TIME"]).sort_values("TIME")
        if controls.empty:
            continue
        layer_core = background[background["LayerGroup"].astype(str).eq(str(layer))].copy()
        layer_match = working[
            working["LayerGroup"].astype(str).eq(str(layer))
            & working["IsImagingMatchPatch"].eq(1)
        ].copy()
        well_patch_template_area = float(layer_match["PatchAreaM2"].median()) if not layer_match.empty else float(
            layer_core["PatchAreaM2"].median()
        )
        if not np.isfinite(well_patch_template_area) or well_patch_template_area <= 0.0:
            well_patch_template_area = 2000.0
        small_background = layer_core[
            layer_core.get("FractureScale", pd.Series("", index=layer_core.index))
            .astype(str)
            .str.lower()
            .eq("small")
        ]
        area_source = small_background if not small_background.empty else layer_core
        compensation_template_area = float(area_source["PatchAreaM2"].median()) * float(
            config.get("density_compensation_size_multiplier", 0.65)
        ) * float(config.get("density_compensation_height_multiplier", 0.65))
        if not np.isfinite(compensation_template_area) or compensation_template_area <= 0.0:
            compensation_template_area = 800.0

        first_bin = math.floor(float(controls["TIME"].min()) / time_bin_ms) * time_bin_ms
        last_bin = math.floor(float(controls["TIME"].max()) / time_bin_ms) * time_bin_ms
        for bin_start in np.arange(first_bin, last_bin + time_bin_ms * 0.5, time_bin_ms):
            bin_stop = float(bin_start + time_bin_ms)
            bin_center = float(bin_start + 0.5 * time_bin_ms)
            bin_controls = controls[
                controls["TIME"].between(float(bin_start), bin_stop, inclusive="left")
            ].copy()
            if bin_controls.empty:
                continue
            reference_weight = float(bin_controls["RawPointWeight"].sum())
            well_x = float(np.interp(bin_center, track_times, track_x))
            well_y = float(np.interp(bin_center, track_times, track_y))
            well_density = reference_weight * well_patch_template_area / inner_area

            candidates = layer_core[
                layer_core["CenterTime"].between(float(bin_start), bin_stop, inclusive="left")
            ].copy()
            if candidates.empty:
                current_area = {index: 0.0 for index in range(len(bands))}
            else:
                candidate_times = _safe_numeric(candidates["CenterTime"]).to_numpy(dtype=float)
                candidate_well_x = np.interp(candidate_times, track_times, track_x)
                candidate_well_y = np.interp(candidate_times, track_times, track_y)
                radius = np.hypot(
                    _safe_numeric(candidates["CenterX"]).to_numpy(dtype=float) - candidate_well_x,
                    _safe_numeric(candidates["CenterY"]).to_numpy(dtype=float) - candidate_well_y,
                )
                candidate_area = _safe_numeric(candidates["PatchAreaM2"]).fillna(0.0).to_numpy(dtype=float)
                current_area = {}
                for band_index, (ring_min, ring_max) in enumerate(bands):
                    mask = (radius >= ring_min) & (radius < ring_max)
                    current_area[band_index] = float(candidate_area[mask].sum())

            orientation_rows = bin_controls[
                ["Frac_Azimuth", "Frac_Dip"]
            ].copy() if {"Frac_Azimuth", "Frac_Dip"}.issubset(bin_controls.columns) else pd.DataFrame()
            for band_index, (ring_min, ring_max) in enumerate(bands):
                effective_min = max(ring_min, minimum_radius if band_index == 0 else ring_min)
                ring_area = math.pi * max(ring_max**2 - effective_min**2, 1.0)
                current_density = current_area.get(band_index, 0.0) / ring_area
                midpoint = 0.5 * (effective_min + ring_max)
                radial_weight = math.exp(-max(midpoint - inner_radius, 0.0) / decay_m)
                target_density = well_density * radial_weight
                deficit = max(target_density - current_density, 0.0)
                required_area = deficit * ring_area * compensation_fraction
                requested = int(
                    math.ceil(required_area / compensation_template_area - 1.0e-12)
                ) if required_area > 0.0 else 0
                requested = min(requested, max_per_bin, max_total - len(requests))
                summary_rows.append(
                    {
                        "WellName": str(well_name),
                        "LayerGroup": str(layer),
                        "TimeBinStartMs": float(bin_start),
                        "RadialBandLabel": f"{ring_min:.0f}-{ring_max:.0f}m",
                        "RadialBandMinM": float(ring_min),
                        "RadialBandMaxM": float(ring_max),
                        "ReferencePointWeight": reference_weight,
                        "WellPatchTemplateAreaM2": well_patch_template_area,
                        "CompensationTemplateAreaM2": compensation_template_area,
                        "WellReferenceAreaDensity": well_density,
                        "RadialWeight": radial_weight,
                        "TargetAreaDensity": target_density,
                        "CurrentAreaDensity": current_density,
                        "DeficitAreaDensity": deficit,
                        "CompensationPatchCount": requested,
                    }
                )
                for request_index in range(max(requested, 0)):
                    key = (
                        f"{well_name}|{layer}|{bin_start:.3f}|{band_index}|{request_index}"
                    )
                    fraction = _stable_unit(key + "|radius")
                    radius = math.sqrt(
                        effective_min**2 + fraction * max(ring_max**2 - effective_min**2, 0.0)
                    )
                    angle = 2.0 * math.pi * _stable_unit(key + "|angle")
                    time_value = float(
                        bin_start
                        + time_bin_ms * (0.1 + 0.8 * _stable_unit(key + "|time"))
                    )
                    if orientation_rows.empty:
                        azimuth, dip = 0.0, 60.0
                    else:
                        orientation = orientation_rows.iloc[request_index % len(orientation_rows)]
                        azimuth = float(_safe_numeric(pd.Series([orientation.get("Frac_Azimuth", 0.0)])).fillna(0.0).iloc[0])
                        dip = float(_safe_numeric(pd.Series([orientation.get("Frac_Dip", 60.0)])).fillna(60.0).iloc[0])
                    requests.append(
                        {
                            "WellName": str(well_name),
                            "LayerGroup": str(layer),
                            "TimeBinStartMs": float(bin_start),
                            "TimeBinCenterMs": bin_center,
                            "RadialBandLabel": f"{ring_min:.0f}-{ring_max:.0f}m",
                            "RadialBandMinM": float(ring_min),
                            "RadialBandMaxM": float(ring_max),
                            "RequestIndex": int(request_index),
                            "X": well_x + radius * math.cos(angle),
                            "Y": well_y + radius * math.sin(angle),
                            "TIME": time_value,
                            "RadialDistanceM": float(radius),
                            "RadialWeight": radial_weight,
                            "TargetAreaDensity": target_density,
                            "CurrentAreaDensity": current_density,
                            "DeficitAreaDensity": deficit,
                            "CompensationAreaM2": required_area,
                            "CompensationPatchCount": requested,
                            "Frac_Azimuth": azimuth % 360.0,
                            "Frac_Dip": float(np.clip(dip, 1.0, 89.0)),
                            "ReferencePointWeight": reference_weight,
                        }
                    )
                if len(requests) >= max_total:
                    break
            if len(requests) >= max_total:
                break
        if len(requests) >= max_total:
            break

    request_df = pd.DataFrame(requests, columns=columns)
    summary_df = pd.DataFrame(summary_rows)
    summary = {
        "enabled": True,
        "status": "applied" if len(request_df) else "no_positive_density_deficit",
        "radial_bands_m": [list(band) for band in bands],
        "time_bin_ms": time_bin_ms,
        "compensation_fraction": compensation_fraction,
        "decay_m": decay_m,
        "minimum_radius_m": minimum_radius,
        "request_count": int(len(request_df)),
        "requests_by_band": {
            str(key): int(value)
            for key, value in request_df["RadialBandLabel"].value_counts().sort_index().items()
        } if not request_df.empty else {},
        "summary_row_count": int(len(summary_df)),
        "density_units": "fracture patch area / annulus area, using 0-50 m well reference",
    }
    return request_df, summary_df, summary


def summarize_imaging_match_patches(
    corrected_df: pd.DataFrame,
    active_controls: pd.DataFrame,
    calibration_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    controls = active_controls[
        _safe_numeric(active_controls.get("IsImagingGroundTruth", pd.Series(0, index=active_controls.index)))
        .fillna(0)
        .astype(int)
        .eq(1)
    ].copy()
    patches = corrected_df[
        _safe_numeric(corrected_df.get("IsImagingMatchPatch", pd.Series(0, index=corrected_df.index)))
        .fillna(0)
        .astype(int)
        .eq(1)
    ].copy()
    patch_by_control = {
        str(value): group
        for value, group in patches.groupby("WellControlSampleID", dropna=False)
    }
    matched_weight = 0
    orientation_errors: list[float] = []
    dip_errors: list[float] = []
    missing_ids: list[str] = []
    plane_distances: list[float] = []
    plane_tolerance_m = float(config.get("imaging_match_plane_distance_tolerance_m", 1.0))
    time_scale = float(config.get("time_scale_m_per_ms", 2.0))
    vertex_columns = [f"V{i}{axis}" for i in range(1, 5) for axis in "XYZ"]
    for row in controls.itertuples(index=False):
        control_id = str(getattr(row, "WellControlSampleID", ""))
        group = patch_by_control.get(control_id)
        if group is None or group.empty:
            missing_ids.append(control_id)
            continue
        patch = group.iloc[0]
        if int(_safe_numeric(pd.Series([patch.get("ControlInsidePatchEnvelope", 0)])).fillna(0).iloc[0]) != 1:
            missing_ids.append(control_id)
            continue
        plane_distance = np.nan
        if all(column in patch.index and pd.notna(patch.get(column)) for column in vertex_columns):
            vertices = np.asarray(
                [
                    [
                        float(patch[f"V{i}X"]),
                        float(patch[f"V{i}Y"]),
                        float(patch[f"V{i}Z"]) * time_scale,
                    ]
                    for i in range(1, 5)
                ]
            ).reshape(4, 3)
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            norm = float(np.linalg.norm(normal))
            if norm > 1.0e-12:
                control_point = np.asarray(
                    [
                        float(getattr(row, "X", np.nan)),
                        float(getattr(row, "Y", np.nan)),
                        float(getattr(row, "TIME", np.nan)) * time_scale,
                    ],
                    dtype=float,
                )
                if np.isfinite(control_point).all():
                    plane_distance = abs(float(np.dot(control_point - vertices[0], normal / norm)))
        if np.isfinite(plane_distance):
            plane_distances.append(plane_distance)
        if np.isfinite(plane_distance) and plane_distance > plane_tolerance_m:
            missing_ids.append(control_id)
            continue
        matched_weight += int(
            _safe_numeric(pd.Series([getattr(row, "RawPointWeight", 1)])).fillna(1).iloc[0]
        )
        target_azimuth = float(
            _safe_numeric(pd.Series([getattr(row, "Frac_Azimuth", np.nan)])).iloc[0]
        )
        target_dip = float(
            _safe_numeric(pd.Series([getattr(row, "Frac_Dip", np.nan)])).iloc[0]
        )
        patch_azimuth = float(
            _safe_numeric(pd.Series([patch.get("DipAzimuthDeg", patch.get("AzimuthDeg", np.nan))])).iloc[0]
        )
        patch_dip = float(_safe_numeric(pd.Series([patch.get("DipDeg", np.nan)])).iloc[0])
        if np.isfinite(target_azimuth) and np.isfinite(patch_azimuth):
            orientation_errors.append(_axial_delta_deg(target_azimuth, patch_azimuth))
        if np.isfinite(target_dip) and np.isfinite(patch_dip):
            dip_errors.append(abs(target_dip - patch_dip))

    original_points = int(calibration_summary.get("original_imaging_point_count", 0))
    minimum_points = int(calibration_summary.get("minimum_match_point_count", 0))
    maximum_points = int(calibration_summary.get("maximum_match_point_count", 0))
    ratio = float(matched_weight / original_points) if original_points else None
    return {
        **calibration_summary,
        "geometry_match_patch_count": int(len(patches)),
        "geometry_matched_point_count": int(matched_weight),
        "geometry_match_ratio": ratio,
        "geometry_match_ratio_within_required_interval": bool(
            original_points == 0
            or (minimum_points <= matched_weight <= maximum_points)
        ),
        "missing_control_count": int(len(missing_ids)),
        "missing_control_ids": missing_ids[:50],
        "plane_distance_tolerance_m": plane_tolerance_m,
        "plane_distance_median_m": float(np.median(plane_distances)) if plane_distances else None,
        "plane_distance_p95_m": float(np.quantile(plane_distances, 0.95)) if plane_distances else None,
        "plane_distance_max_m": float(np.max(plane_distances)) if plane_distances else None,
        "azimuth_error_median_deg": float(np.median(orientation_errors)) if orientation_errors else None,
        "azimuth_error_p90_deg": float(np.quantile(orientation_errors, 0.9)) if orientation_errors else None,
        "dip_error_median_deg": float(np.median(dip_errors)) if dip_errors else None,
        "dip_error_p90_deg": float(np.quantile(dip_errors, 0.9)) if dip_errors else None,
        "minimum_orientation_histogram_overlap": float(
            config.get("imaging_match_min_orientation_histogram_overlap", 0.75)
        ),
    }
