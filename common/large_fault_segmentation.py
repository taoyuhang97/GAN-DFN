"""Split inferred large-fault patches into connected, attitude-varying blocks."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from section_readability import (
    apparent_dip_deg,
    dip_azimuth_dip_from_normal_depth,
)


def _stable_phase(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / float(0xFFFFFFFF)) * 2.0 * np.pi


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 else np.asarray([1.0, 0.0, 0.0])


def _vertices(row: pd.Series) -> np.ndarray:
    columns = [f"V{i}{axis}" for i in range(1, 5) for axis in "XYZ"]
    return pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float).reshape(4, 3)


def _set_vertices(row: dict[str, Any], vertices: np.ndarray) -> None:
    for index, vertex in enumerate(vertices, start=1):
        row[f"V{index}X"] = float(vertex[0])
        row[f"V{index}Y"] = float(vertex[1])
        row[f"V{index}Z"] = float(vertex[2])


def _quad_area(vertices: np.ndarray) -> float:
    a = 0.5 * float(np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])))
    b = 0.5 * float(np.linalg.norm(np.cross(vertices[3] - vertices[0], vertices[2] - vertices[0])))
    return a + b


def _boundary_edges(
    parent_vertices: np.ndarray,
    count: int,
    time_scale: float,
    azimuth_amplitude: float,
    dip_amplitude: float,
    dip_floor: float,
    phase: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for boundary_index in range(count + 1):
        fraction = boundary_index / float(count)
        upper = parent_vertices[0] + fraction * (parent_vertices[1] - parent_vertices[0])
        lower = parent_vertices[3] + fraction * (parent_vertices[2] - parent_vertices[3])
        if 0 < boundary_index < count and (azimuth_amplitude > 0.0 or dip_amplitude > 0.0):
            midpoint = 0.5 * (upper + lower)
            edge = lower - upper
            horizontal_norm = float(np.linalg.norm(edge[:2]))
            if horizontal_norm > 1.0e-9:
                azimuth = float(
                    np.degrees(np.arctan2(edge[0], edge[1]))
                    + azimuth_amplitude * np.sin(2.0 * np.pi * fraction + phase)
                ) % 360.0
                original_dip = float(
                    np.degrees(np.arctan2(abs(edge[2]) * time_scale, horizontal_norm))
                )
                dip = float(
                    np.clip(
                        original_dip
                        + dip_amplitude
                        * (0.5 + 0.5 * np.sin(2.0 * np.pi * fraction + phase)),
                        dip_floor,
                        89.0,
                    )
                )
                radius = float(
                    np.hypot(horizontal_norm, abs(edge[2]) * time_scale)
                )
                half_horizontal = 0.5 * radius * np.cos(np.deg2rad(dip))
                half_vertical_ms = 0.5 * radius * np.sin(np.deg2rad(dip)) / time_scale
                direction = np.asarray(
                    [np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth))],
                    dtype=float,
                )
                upper = midpoint + np.asarray(
                    [-half_horizontal * direction[0], -half_horizontal * direction[1], -half_vertical_ms]
                )
                lower = midpoint + np.asarray(
                    [half_horizontal * direction[0], half_horizontal * direction[1], half_vertical_ms]
                )
        edges.append((upper, lower))
    return edges


def _segment_patch(
    row: pd.Series,
    count: int,
    az_amplitude: float,
    dip_amplitude: float,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    options = dict(config.get("large_fault_segmentation", {}))
    dip_floor = float(options.get("min_true_dip_deg", 65.0))
    apparent_floor = float(options.get("min_apparent_dip_deg", 30.0))
    max_adjacent_azimuth_change = float(options.get("max_adjacent_azimuth_change_deg", 20.0))
    max_adjacent_dip_change = float(options.get("max_adjacent_dip_change_deg", 10.0))
    time_scale = float(options.get("time_scale_m_per_ms", 2.0))
    vertices = _vertices(row)
    phase = _stable_phase(str(row["PatchID"]))

    for factor in (1.0, 0.5, 0.25, 0.0):
        edges = _boundary_edges(
            vertices,
            count,
            time_scale,
            az_amplitude * factor,
            dip_amplitude * factor,
            dip_floor,
            phase,
        )
        child_vertices: list[np.ndarray] = []
        child_orientations: list[tuple[float, float]] = []
        valid = True
        for panel_index in range(count):
            upper_start, lower_start = edges[panel_index]
            upper_end, lower_end = edges[panel_index + 1]
            quad = np.asarray([upper_start, upper_end, lower_end, lower_start], dtype=float)
            child_vertices.append(quad)
            child_normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
            child_azimuth, child_dip = dip_azimuth_dip_from_normal_depth(child_normal)
            child_orientations.append((child_azimuth, child_dip))
            if (
                child_dip + 1.0e-9 < dip_floor
                or apparent_dip_deg(child_azimuth, child_dip, "XZ") + 1.0e-9 < apparent_floor
                or apparent_dip_deg(child_azimuth, child_dip, "YZ") + 1.0e-9 < apparent_floor
            ):
                valid = False
                break
        if valid:
            for previous, current in zip(child_orientations[:-1], child_orientations[1:]):
                azimuth_change = abs((current[0] - previous[0] + 180.0) % 360.0 - 180.0)
                dip_change = abs(float(current[1]) - float(previous[1]))
                if (
                    azimuth_change > max_adjacent_azimuth_change + 1.0e-9
                    or dip_change > max_adjacent_dip_change + 1.0e-9
                ):
                    valid = False
                    break
        if valid:
            break
    else:
        raise RuntimeError(f"cannot create valid segmented chain for {row['PatchID']}")

    bounds = dict(config.get("target_block", {}))
    if all(key in bounds for key in ("x_min", "x_max", "y_min", "y_max")):
        clamped_vertices: list[np.ndarray] = []
        clamped_orientations: list[tuple[float, float]] = []
        clamp_valid = True
        for quad in child_vertices:
            clamped = quad.copy()
            clamped[:, 0] = np.clip(clamped[:, 0], float(bounds["x_min"]), float(bounds["x_max"]))
            clamped[:, 1] = np.clip(clamped[:, 1], float(bounds["y_min"]), float(bounds["y_max"]))
            child_normal = np.cross(clamped[1] - clamped[0], clamped[2] - clamped[0])
            child_azimuth, child_dip = dip_azimuth_dip_from_normal_depth(child_normal)
            if (
                child_dip + 1.0e-9 < dip_floor
                or apparent_dip_deg(child_azimuth, child_dip, "XZ") + 1.0e-9 < apparent_floor
                or apparent_dip_deg(child_azimuth, child_dip, "YZ") + 1.0e-9 < apparent_floor
            ):
                clamp_valid = False
                break
            clamped_vertices.append(clamped)
            clamped_orientations.append((child_azimuth, child_dip))
        if clamp_valid:
            for previous, current in zip(clamped_orientations[:-1], clamped_orientations[1:]):
                azimuth_change = abs((current[0] - previous[0] + 180.0) % 360.0 - 180.0)
                dip_change = abs(float(current[1]) - float(previous[1]))
                if (
                    azimuth_change > max_adjacent_azimuth_change + 1.0e-9
                    or dip_change > max_adjacent_dip_change + 1.0e-9
                ):
                    clamp_valid = False
                    break
        if not clamp_valid:
            fallback_edges = _boundary_edges(
                vertices,
                count,
                time_scale,
                0.0,
                0.0,
                dip_floor,
                phase,
            )
            clamped_vertices = []
            clamped_orientations = []
            for panel_index in range(count):
                upper_start, lower_start = fallback_edges[panel_index]
                upper_end, lower_end = fallback_edges[panel_index + 1]
                quad = np.asarray([upper_start, upper_end, lower_end, lower_start], dtype=float)
                child_normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
                child_azimuth, child_dip = dip_azimuth_dip_from_normal_depth(child_normal)
                if (
                    child_dip + 1.0e-9 < dip_floor
                    or apparent_dip_deg(child_azimuth, child_dip, "XZ") + 1.0e-9 < apparent_floor
                    or apparent_dip_deg(child_azimuth, child_dip, "YZ") + 1.0e-9 < apparent_floor
                ):
                    raise RuntimeError(f"fallback segmentation invalidated {row['PatchID']}")
                clamped_vertices.append(quad)
                clamped_orientations.append((child_azimuth, child_dip))
        child_vertices = clamped_vertices
        child_orientations = clamped_orientations

    rows: list[dict[str, Any]] = []
    for panel_index, (quad, orientation) in enumerate(zip(child_vertices, child_orientations), start=1):
        child = row.to_dict()
        child["ParentPatchID"] = str(row["PatchID"])
        child["PatchID"] = f"{row['PatchID']}__seg{panel_index:02d}of{count:02d}"
        child["LargeFaultChainID"] = str(row.get("FaultName", row["PatchID"]))
        child["LargeFaultSegmentIndex"] = panel_index
        child["LargeFaultSegmentCount"] = count
        child["LargeFaultSharedBoundaryStart"] = f"{child['ParentPatchID']}:b{panel_index - 1:02d}"
        child["LargeFaultSharedBoundaryEnd"] = f"{child['ParentPatchID']}:b{panel_index:02d}"
        child["LargeFaultBoundaryConnected"] = 1
        child["CenterX"] = float(quad[:, 0].mean())
        child["CenterY"] = float(quad[:, 1].mean())
        child["CenterTime"] = float(quad[:, 2].mean())
        child["LengthM"] = float(
            np.linalg.norm(
                (0.5 * (quad[1] + quad[2]) - 0.5 * (quad[0] + quad[3]))[:2]
            )
        )
        child["HeightTimeMs"] = float(np.nanmax(quad[:, 2]) - np.nanmin(quad[:, 2]))
        child["PatchAreaM2"] = _quad_area(quad)
        azimuth, dip_value = orientation
        child["DipAzimuthDeg"] = float(azimuth)
        child["AzimuthDeg"] = float((azimuth - 90.0) % 180.0)
        child["DipDeg"] = float(dip_value)
        child["ObjectBandDipAzimuthDeg"] = float(azimuth)
        child["ObjectBandAzimuthDeg"] = float((azimuth - 90.0) % 180.0)
        child["ObjectBandDipDeg"] = float(dip_value)
        child["BandPatchOrdinal"] = panel_index
        child["BandLengthM"] = float(child["LengthM"])
        child["BandTimeExtentMs"] = float(child["HeightTimeMs"])
        child["OrientationSource"] = "segmented_chain_local_attitude"
        _set_vertices(child, quad)
        rows.append(child)
    return rows, {
        "width": float(
            np.linalg.norm(
                (vertices[1] - vertices[0])[:2]
            )
        ),
        "azimuth_amplitude_used": float(az_amplitude * factor),
        "dip_amplitude_used": float(dip_amplitude * factor),
    }


def segment_large_fault_patches(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    options = dict(config.get("large_fault_segmentation", {}))
    if not bool(options.get("enabled", False)):
        return frame.copy(), {"enabled": False}
    target_length = max(float(options.get("target_panel_length_m", 200.0)), 20.0)
    min_panels = max(int(options.get("min_panels", 2)), 2)
    max_panels = max(int(options.get("max_panels", 4)), min_panels)
    az_amplitude = max(float(options.get("azimuth_variation_deg", 10.0)), 0.0)
    dip_amplitude = max(float(options.get("dip_variation_deg", 5.0)), 0.0)
    max_adjacent_azimuth_change = float(
        options.get("max_adjacent_azimuth_change_deg", 20.0)
    )
    max_adjacent_dip_change = float(options.get("max_adjacent_dip_change_deg", 10.0))
    source_prefix = str(options.get("apply_to_source_prefix", "large_inferred"))

    output_rows: list[dict[str, Any]] = []
    segmented_count = 0
    for _, row in frame.iterrows():
        if source_prefix not in str(row.get("SourceType", "")):
            output_rows.append(row.to_dict())
            continue
        count = int(np.ceil(float(row["LengthM"]) / target_length))
        count = int(np.clip(count, min_panels, max_panels))
        rows, _ = _segment_patch(row, count, az_amplitude, dip_amplitude, config)
        output_rows.extend(rows)
        segmented_count += 1
    output = pd.DataFrame(output_rows)
    return output.reset_index(drop=True), {
        "enabled": True,
        "input_patch_count": int(len(frame)),
        "segmented_parent_count": int(segmented_count),
        "output_patch_count": int(len(output)),
        "target_panel_length_m": target_length,
        "min_panels": min_panels,
        "max_panels": max_panels,
        "max_adjacent_azimuth_change_deg": max_adjacent_azimuth_change,
        "max_adjacent_dip_change_deg": max_adjacent_dip_change,
        "azimuth_variation_deg": az_amplitude,
        "dip_variation_deg": dip_amplitude,
        "boundary_connected": True,
    }
