"""Section-readability constraints for predicted DFN fracture patches.

Coordinates are ``x=east, y=north, z=TIME`` (positive down).  The two formal
sections are treated as the orthogonal vertical planes XZ and YZ.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


def dip_azimuth_dip_from_normal_depth(normal: np.ndarray) -> tuple[float, float]:
    """Return true dip azimuth and dip from a normal in the depth frame."""
    n = _normalize(np.asarray(normal, dtype=float))
    n_up = np.asarray([n[0], n[1], -n[2]], dtype=float)
    if n_up[2] < 0.0:
        n_up = -n_up
    dip = float(np.degrees(np.arccos(np.clip(abs(n_up[2]), 0.0, 1.0))))
    horizontal = n_up[:2]
    if float(np.linalg.norm(horizontal)) <= 1.0e-12:
        return 0.0, dip
    azimuth = float(np.degrees(np.arctan2(horizontal[0], horizontal[1])) % 360.0)
    return azimuth, dip


def axes_from_dip_azimuth_dip(
    dip_azimuth_deg: float, dip_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return long strike axis and down-dip axis for patch construction."""
    az = np.deg2rad(float(dip_azimuth_deg) % 360.0)
    dip = np.deg2rad(float(np.clip(dip_deg, 1.0, 89.0)))
    normal = np.asarray(
        [
            np.sin(az) * np.sin(dip),
            np.cos(az) * np.sin(dip),
            -np.cos(dip),
        ],
        dtype=float,
    )
    strike = _normalize(np.asarray([-normal[1], normal[0], 0.0], dtype=float))
    down_dip = _normalize(np.cross(normal, strike))
    if down_dip[2] < 0.0:
        down_dip = -down_dip
    return strike, down_dip


def apparent_dip_deg(dip_azimuth_deg: float, dip_deg: float, projection: str) -> float:
    az = np.deg2rad(float(dip_azimuth_deg) % 360.0)
    dip = np.deg2rad(float(dip_deg))
    if str(projection).upper() == "XZ":
        ratio = abs(np.sin(az)) * np.tan(dip)
    elif str(projection).upper() == "YZ":
        ratio = abs(np.cos(az)) * np.tan(dip)
    else:
        raise ValueError(f"unsupported projection: {projection}")
    return float(np.degrees(np.arctan(ratio)))


def segment_apparent_dip_deg(line: np.ndarray, time_scale_m_per_ms: float) -> float:
    arr = np.asarray(line, dtype=float)
    horizontal = abs(float(arr[1, 0] - arr[0, 0]))
    vertical_m = abs(float(arr[1, 1] - arr[0, 1])) * max(float(time_scale_m_per_ms), 1.0e-9)
    return float(np.degrees(np.arctan2(vertical_m, horizontal))) if horizontal > 1.0e-12 else 90.0


def _vertices_from_row(row: pd.Series) -> np.ndarray | None:
    columns = [f"V{i}{axis}" for i in range(1, 5) for axis in "XYZ"]
    if not all(column in row.index for column in columns):
        return None
    values = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return values.reshape(4, 3)


def patch_orientation(row: pd.Series) -> tuple[float, float]:
    vertices = _vertices_from_row(row)
    if vertices is not None:
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        if float(np.linalg.norm(normal)) > 1.0e-12:
            return dip_azimuth_dip_from_normal_depth(normal)
    azimuth = pd.to_numeric(pd.Series([row.get("DipAzimuthDeg")]), errors="coerce").iloc[0]
    dip = pd.to_numeric(pd.Series([row.get("DipDeg")]), errors="coerce").iloc[0]
    if np.isfinite(azimuth) and np.isfinite(dip):
        return float(azimuth) % 360.0, float(dip)
    raise ValueError(f"cannot determine patch orientation for {row.get('PatchID', '<unknown>')}")


class OrientationPriorIndex:
    def __init__(self, samples: pd.DataFrame, time_scale_m_per_ms: float):
        self.samples = samples
        self.time_scale_m_per_ms = float(time_scale_m_per_ms)
        self.tree: cKDTree | None = None
        if self.samples.empty:
            self.tree = None
            return
        xyz = self.samples[["CenterX", "CenterY", "CenterTime"]].to_numpy(dtype=float).copy()
        xyz[:, 2] *= self.time_scale_m_per_ms
        self.tree = cKDTree(xyz)

    def nearest(self, center: np.ndarray, radius_m: float) -> dict[str, Any] | None:
        if self.tree is None:
            return None
        query = np.asarray(
            [float(center[0]), float(center[1]), float(center[2]) * self.time_scale_m_per_ms],
            dtype=float,
        )
        distance, index = self.tree.query(query, k=1)
        if not np.isfinite(distance) or float(distance) > float(radius_m):
            return None
        row = self.samples.iloc[int(index)]
        return {
            "dip_azimuth_deg": float(row["DipAzimuthDeg"]),
            "dip_deg": float(row["DipDeg"]),
            "distance_m": float(distance),
            "fault_name": str(row.get("FaultName", "")),
        }


def build_orientation_prior_index(
    mesh: Any,
    *,
    time_scale_m_per_ms: float = 2.0,
    unit_scalar: str = "FaultUnitID",
    name_scalar: str = "FaultName",
) -> OrientationPriorIndex:
    """Aggregate interpreted fault triangles into area-weighted unit attitudes."""
    if mesh is None or int(getattr(mesh, "n_cells", 0)) <= 0:
        return OrientationPriorIndex(pd.DataFrame(), time_scale_m_per_ms)
    poly = mesh if hasattr(mesh, "faces") else mesh.extract_surface(algorithm="dataset_surface")
    if not hasattr(poly, "faces"):
        return OrientationPriorIndex(pd.DataFrame(), time_scale_m_per_ms)
    poly = poly.triangulate()
    points = np.asarray(poly.points, dtype=float).copy()
    points[:, 2] *= float(time_scale_m_per_ms)
    faces = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    unit_values = None
    name_values = None
    if unit_scalar in poly.cell_data:
        unit_values = np.asarray(poly.cell_data[unit_scalar]).reshape(-1)
    if name_scalar in poly.cell_data:
        name_values = np.asarray(poly.cell_data[name_scalar]).reshape(-1)

    records: list[dict[str, Any]] = []
    for cell_index, face in enumerate(faces):
        triangle = points[face]
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(cross))
        if norm <= 1.0e-12:
            continue
        unit = int(unit_values[cell_index]) if unit_values is not None else cell_index
        fault_name = str(name_values[cell_index]) if name_values is not None else f"unit_{unit}"
        records.append(
            {
                "unit": unit,
                "fault_name": fault_name,
                "center": triangle.mean(axis=0),
                "weighted_normal": cross,
                "area": 0.5 * norm,
            }
        )
    if not records:
        return OrientationPriorIndex(pd.DataFrame(), time_scale_m_per_ms)

    source = pd.DataFrame(records)
    rows: list[dict[str, Any]] = []
    for unit, group in source.groupby("unit", sort=False):
        weight = max(float(group["area"].sum()), 1.0e-12)
        center_scaled = np.average(np.vstack(group["center"].to_numpy()), axis=0, weights=group["area"])
        normal = np.sum(np.vstack(group["weighted_normal"].to_numpy()), axis=0)
        azimuth, dip = dip_azimuth_dip_from_normal_depth(normal)
        names = group["fault_name"].astype(str)
        name = str(names.mode().iloc[0]) if not names.mode().empty else str(names.iloc[0])
        rows.append(
            {
                "FaultUnitID": int(unit),
                "FaultName": name,
                "CenterX": float(center_scaled[0]),
                "CenterY": float(center_scaled[1]),
                "CenterTime": float(center_scaled[2] / max(float(time_scale_m_per_ms), 1.0e-9)),
                "DipAzimuthDeg": float(azimuth),
                "DipDeg": float(dip),
                "TriangleAreaWeight": weight,
            }
        )
    return OrientationPriorIndex(pd.DataFrame(rows), time_scale_m_per_ms)


def _angle_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _stable_unit(value: str) -> float:
    integer = int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)
    return 2.0 * (integer / float(0xFFFFFFFFFFFF)) - 1.0


def _nearest_feasible_orientation(
    current_azimuth: float,
    current_dip: float,
    prior: dict[str, Any] | None,
    *,
    min_xz: float,
    min_yz: float,
    min_true_dip: float,
    target_true_dip: float,
    target_true_dip_jitter: float,
    patch_id: str,
    prior_weight: float,
) -> tuple[float, float, float, float]:
    deltas = np.arange(0.0, 360.0, 1.0)
    candidate_dip = float(np.clip(current_dip, 1.0, 89.0))
    if candidate_dip < float(min_true_dip):
        if prior is not None and float(prior["dip_deg"]) >= float(min_true_dip):
            candidate_dip = float(prior["dip_deg"])
        else:
            candidate_dip = float(target_true_dip) + _stable_unit(patch_id) * float(
                target_true_dip_jitter
            )
    candidate_dip = float(np.clip(max(candidate_dip, float(min_true_dip)), 1.0, 89.0))
    required = np.degrees(np.arctan(np.sqrt(2.0) * np.tan(np.deg2rad(max(min_xz, min_yz)))))
    candidate_dip = float(np.clip(max(candidate_dip, required), 1.0, 89.0))

    xz = np.degrees(np.arctan(np.abs(np.sin(np.deg2rad(deltas))) * np.tan(np.deg2rad(candidate_dip))))
    yz = np.degrees(np.arctan(np.abs(np.cos(np.deg2rad(deltas))) * np.tan(np.deg2rad(candidate_dip))))
    feasible = (xz >= float(min_xz) - 1.0e-9) & (yz >= float(min_yz) - 1.0e-9)
    if not feasible.any():
        candidate_dip = min(89.0, required + 0.1)
        xz = np.degrees(np.arctan(np.abs(np.sin(np.deg2rad(deltas))) * np.tan(np.deg2rad(candidate_dip))))
        yz = np.degrees(np.arctan(np.abs(np.cos(np.deg2rad(deltas))) * np.tan(np.deg2rad(candidate_dip))))
        feasible = (xz >= float(min_xz) - 1.0e-9) & (yz >= float(min_yz) - 1.0e-9)
    if not feasible.any():
        raise RuntimeError("no feasible orientation exists for requested section thresholds")

    cost = np.asarray([_angle_delta(value, current_azimuth) for value in deltas], dtype=float)
    if prior is not None:
        prior_cost = np.asarray([_angle_delta(value, float(prior["dip_azimuth_deg"])) for value in deltas], dtype=float)
        cost = cost + float(prior_weight) * prior_cost
    cost = np.where(feasible, cost, np.inf)
    index = int(np.argmin(cost))
    return float(deltas[index]), candidate_dip, float(xz[index]), float(yz[index])


def apply_patch_readability(
    frame: pd.DataFrame,
    prior_index: OrientationPriorIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Constrain inferred large patches while preserving centre and dimensions."""
    options = dict(config.get("section_readability", config))
    if not bool(options.get("enabled", False)):
        return frame.copy(), {"enabled": False}
    min_xz = float(options.get("min_apparent_dip_xz_deg", 30.0))
    min_yz = float(options.get("min_apparent_dip_yz_deg", 30.0))
    min_true_dip = float(options.get("min_true_dip_deg", 0.0))
    target_true_dip = float(options.get("target_true_dip_deg", max(min_true_dip, 75.0)))
    target_true_dip_jitter = float(options.get("target_true_dip_jitter_deg", 3.0))
    radius = float(options.get("prior_search_radius_m", 750.0))
    prior_weight = float(options.get("prior_weight", 0.5))
    source_prefix = str(options.get("apply_to_source_prefix", "large_inferred"))

    out = frame.reset_index(drop=True).copy()
    audit: list[dict[str, Any]] = []
    for index, row in out.iterrows():
        if source_prefix not in str(row.get("SourceType", "")):
            continue
        center = row[["CenterX", "CenterY", "CenterTime"]].to_numpy(dtype=float)
        current_azimuth, current_dip = patch_orientation(row)
        before_xz = apparent_dip_deg(current_azimuth, current_dip, "XZ")
        before_yz = apparent_dip_deg(current_azimuth, current_dip, "YZ")
        prior = prior_index.nearest(center, radius)
        changed = before_xz < min_xz - 1.0e-9 or before_yz < min_yz - 1.0e-9
        corrected_azimuth, corrected_dip, after_xz, after_yz = _nearest_feasible_orientation(
            current_azimuth,
            current_dip,
            prior,
            min_xz=min_xz,
            min_yz=min_yz,
            min_true_dip=min_true_dip,
            target_true_dip=target_true_dip,
            target_true_dip_jitter=target_true_dip_jitter,
            patch_id=str(row.get("PatchID", "")),
            prior_weight=prior_weight,
        )
        strike, down_dip = axes_from_dip_azimuth_dip(corrected_azimuth, corrected_dip)
        patch_center = np.asarray(
            [
                float(row["CenterX"]),
                float(row["CenterY"]),
                float(row["CenterTime"]),
            ],
            dtype=float,
        )
        half_length = 0.5 * float(row["LengthM"])
        half_height = 0.5 * float(row["HeightTimeMs"])
        vertices = np.asarray(
            [
                patch_center - half_length * strike - half_height * down_dip,
                patch_center + half_length * strike - half_height * down_dip,
                patch_center + half_length * strike + half_height * down_dip,
                patch_center - half_length * strike + half_height * down_dip,
            ],
            dtype=float,
        )
        for vertex_index, vertex in enumerate(vertices, start=1):
            out.at[index, f"V{vertex_index}X"] = float(vertex[0])
            out.at[index, f"V{vertex_index}Y"] = float(vertex[1])
            out.at[index, f"V{vertex_index}Z"] = float(vertex[2])
        if changed:
            out.at[index, "OrientationSource"] = "step7c_section_readability_prior"
        out.at[index, "DipAzimuthDeg"] = corrected_azimuth
        out.at[index, "AzimuthDeg"] = float((corrected_azimuth - 90.0) % 180.0)
        out.at[index, "DipDeg"] = corrected_dip
        out.at[index, "ObjectBandDipAzimuthDeg"] = corrected_azimuth
        out.at[index, "ObjectBandAzimuthDeg"] = float((corrected_azimuth - 90.0) % 180.0)
        out.at[index, "ObjectBandDipDeg"] = corrected_dip
        out.at[index, "SectionReadabilityOriginalDipAzimuthDeg"] = current_azimuth
        out.at[index, "SectionReadabilityOriginalDipDeg"] = current_dip
        out.at[index, "SectionReadabilityCorrectedDipAzimuthDeg"] = corrected_azimuth
        out.at[index, "SectionReadabilityCorrectedDipDeg"] = corrected_dip
        out.at[index, "SectionReadabilityXZBeforeDeg"] = before_xz
        out.at[index, "SectionReadabilityXZAfterDeg"] = after_xz
        out.at[index, "SectionReadabilityYZBeforeDeg"] = before_yz
        out.at[index, "SectionReadabilityYZAfterDeg"] = after_yz
        out.at[index, "SectionReadabilityChanged"] = int(changed)
        out.at[index, "SectionReadabilityPriorSource"] = (
            f"original_fault:{prior['fault_name']}" if prior is not None else "section_constraint_only"
        )
        out.at[index, "SectionReadabilityPriorDistanceM"] = (
            float(prior["distance_m"]) if prior is not None else np.nan
        )
        audit.append(
            {
                "PatchID": str(row["PatchID"]),
                "Changed": int(changed),
                "OriginalDipAzimuthDeg": current_azimuth,
                "OriginalDipDeg": current_dip,
                "CorrectedDipAzimuthDeg": corrected_azimuth,
                "CorrectedDipDeg": corrected_dip,
                "XZBeforeDeg": before_xz,
                "XZAfterDeg": after_xz,
                "YZBeforeDeg": before_yz,
                "YZAfterDeg": after_yz,
                "PriorSource": (
                    f"original_fault:{prior['fault_name']}"
                    if prior is not None
                    else "section_constraint_only"
                ),
                "PriorDistanceM": float(prior["distance_m"]) if prior is not None else np.nan,
            }
        )
    summary = {
        "enabled": True,
        "min_apparent_dip_xz_deg": min_xz,
        "min_apparent_dip_yz_deg": min_yz,
        "min_true_dip_deg": min_true_dip,
        "target_true_dip_deg": target_true_dip,
        "target_true_dip_jitter_deg": target_true_dip_jitter,
        "prior_search_radius_m": radius,
        "input_count": int(len(out)),
        "applied_count": len(audit),
        "changed_count": int(sum(item["Changed"] for item in audit)),
        "visible_below_threshold_count": int(
            sum(
                item["XZAfterDeg"] < min_xz - 1.0e-9
                or item["YZAfterDeg"] < min_yz - 1.0e-9
                for item in audit
            )
        ),
        "prior_matched_count": int(sum(np.isfinite(item["PriorDistanceM"]) for item in audit)),
        "patch_count": int(len(out)),
    }
    return out, summary
