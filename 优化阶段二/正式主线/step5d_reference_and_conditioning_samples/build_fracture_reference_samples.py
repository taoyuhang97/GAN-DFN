#!/usr/bin/env python3
"""Build well-matched reference models and direct-conditioning fracture samples.

Two modes share the same Step8 patch geometry contract:

* ``well-match`` exactly intersects remaining well trajectories with Step8
  quadrilateral patches. One selected non-empty trajectory becomes one model.
* ``direct-cell`` assigns patches to no-well 300 m x 300 m XY cells. One
  selected cell becomes one direct-conditioning sample.

The canonical sample record is the Step8 CSV subset with all original columns
preserved. VTK files are visualization companions, not the sample contract.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


SCALES = ("small", "medium", "large")
SCALE_CODE = {"small": 1, "medium": 2, "large": 3}
GEOMETRY_COLUMNS = (
    "PatchID",
    "FractureScale",
    "CenterX",
    "CenterY",
    "CenterTime",
    *(f"V{vertex}{axis}" for vertex in range(1, 5) for axis in "XYZ"),
)


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists; pass --overwrite to replace it: {output_dir}")
        if output_dir == output_dir.anchor or output_dir == Path.home():
            raise ValueError(f"refusing broad output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def load_patch_inputs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = sorted(set(GEOMETRY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required patch columns: {missing}")
    if frame["PatchID"].isna().any() or frame["PatchID"].duplicated().any():
        raise ValueError(f"{path}: PatchID must be non-null and globally unique")
    scales = frame["FractureScale"].astype(str).str.strip().str.lower()
    unknown = sorted(set(scales.unique()) - set(SCALES))
    if unknown:
        raise ValueError(f"{path}: unexpected FractureScale values: {unknown}")
    centers = frame[["CenterX", "CenterY", "CenterTime"]].to_numpy(dtype=float)
    vertices = np.stack(
        [frame[[f"V{vertex}{axis}" for axis in "XYZ"]].to_numpy(dtype=float) for vertex in range(1, 5)],
        axis=1,
    )
    if not np.isfinite(centers).all() or not np.isfinite(vertices).all():
        raise ValueError(f"{path}: patch centers and vertices must be finite")
    geometry = pd.DataFrame(
        {
            "CenterX": centers[:, 0],
            "CenterY": centers[:, 1],
            "CenterTime": centers[:, 2],
            "FractureScale": scales.to_numpy(dtype=str),
            "_ScaleNormalized": scales.to_numpy(dtype=str),
        }
    )
    return frame, geometry, vertices


def discover_well_tracks(config: dict[str, Any], config_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    wells_root = resolve_path(config["wells_root"], config_dir)
    if not wells_root.is_dir():
        raise FileNotFoundError(wells_root)
    sample_config_path = resolve_path(config["prior_sample_config"], config_dir)
    prior = read_json(sample_config_path)
    excluded_sources = set(str(value) for value in prior["source_wells"])
    expected_virtual = int(config.get("virtual_per_source", prior.get("virtual_per_source", 25)))
    all_sources = sorted((path for path in wells_root.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name))
    remaining_sources = [path for path in all_sources if path.name not in excluded_sources]
    if not remaining_sources:
        raise RuntimeError("no source wells remain after excluding the prior sample construction wells")

    tracks: list[dict[str, Any]] = []
    for source_dir in remaining_sources:
        real_path = source_dir / str(config.get("real_relative_path", "real_well/real_well_samples.csv"))
        if not real_path.is_file():
            raise FileNotFoundError(real_path)
        tracks.append(
            {
                "SourceWellName": source_dir.name,
                "TrackKind": "real",
                "TrackName": source_dir.name,
                "TrackPath": real_path,
                "TrackOrderInSource": 0,
            }
        )
        virtual_root = source_dir / "virtual_wells"
        virtual_dirs = sorted((path for path in virtual_root.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name))
        if len(virtual_dirs) != expected_virtual:
            raise RuntimeError(f"{source_dir.name}: expected {expected_virtual} virtual wells, found {len(virtual_dirs)}")
        template = str(config.get("virtual_relative_template", "virtual_wells/{virtual_well_name}/virtual_well_training_samples.csv"))
        for order, virtual_dir in enumerate(virtual_dirs, start=1):
            track_path = source_dir / template.format(virtual_well_name=virtual_dir.name)
            if not track_path.is_file():
                raise FileNotFoundError(track_path)
            tracks.append(
                {
                    "SourceWellName": source_dir.name,
                    "TrackKind": "virtual",
                    "TrackName": virtual_dir.name,
                    "TrackPath": track_path,
                    "TrackOrderInSource": order,
                }
            )
    return tracks, sorted(excluded_sources)


def segment_triangle_intersection(start: np.ndarray, end: np.ndarray, triangle: np.ndarray, epsilon: float = 1.0e-9) -> np.ndarray:
    """Vectorized two-sided Moller-Trumbore segment/triangle intersection."""
    v0 = triangle[0]
    edge1 = triangle[1] - v0
    edge2 = triangle[2] - v0
    direction = end - start
    pvec = np.cross(edge1, edge2)
    determinant = direction @ pvec
    valid = np.abs(determinant) > epsilon
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    tvec = v0[None, :] - start
    bary_u = (tvec @ pvec) * inverse
    qvec = np.cross(tvec, direction)
    bary_v = (qvec @ edge2) * inverse
    segment_t = (qvec * direction).sum(axis=1) * inverse
    return (
        valid
        & (bary_u >= -1.0e-7)
        & (bary_u <= 1.0 + 1.0e-7)
        & (bary_v >= -1.0e-7)
        & (bary_u + bary_v <= 1.0 + 1.0e-7)
        & (segment_t >= 0.0)
        & (segment_t <= 1.0)
    )


def patch_intersects_track(patch_index: int, points: np.ndarray, vertices: np.ndarray) -> bool:
    if len(points) < 2:
        return False
    start = points[:-1]
    end = points[1:]
    vertices_here = vertices[patch_index]
    segment_min = np.minimum(start, end)
    segment_max = np.maximum(start, end)
    patch_min = vertices_here.min(axis=0)
    patch_max = vertices_here.max(axis=0)
    keep = np.all(segment_max >= patch_min - 1.0e-6, axis=1) & np.all(segment_min <= patch_max + 1.0e-6, axis=1)
    if not keep.any():
        return False
    tri_012 = vertices_here[[0, 1, 2]]
    tri_023 = vertices_here[[0, 2, 3]]
    return bool(
        (
            segment_triangle_intersection(start[keep], end[keep], tri_012)
            | segment_triangle_intersection(start[keep], end[keep], tri_023)
        ).any()
    )


def scan_well_matches(
    config: dict[str, Any],
    config_dir: Path,
    geometry: pd.DataFrame,
    vertices: np.ndarray,
) -> tuple[list[dict[str, Any]], list[str]]:
    tracks, excluded_sources = discover_well_tracks(config, config_dir)
    centers = geometry[["CenterX", "CenterY", "CenterTime"]].to_numpy(dtype=float)
    center_tree = cKDTree(centers)
    circumradius = np.linalg.norm(vertices - centers[:, None, :], axis=2).max(axis=1)
    query_radius = float(np.max(circumradius)) + 1.0
    results: list[dict[str, Any]] = []

    for position, track in enumerate(tracks, start=1):
        points = pd.read_csv(track["TrackPath"], encoding="utf-8-sig", usecols=["X", "Y", "TIME"])
        points = points.apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
        points = points.to_numpy(dtype=float)
        points = points[np.isfinite(points).all(axis=1)]
        points = points[np.argsort(points[:, 2], kind="stable")]
        candidate_indices: set[int] = set()
        if len(points):
            query_points = points if len(points) < 2 else np.vstack([points, (points[:-1] + points[1:]) / 2.0])
            for indices in center_tree.query_ball_point(query_points, r=query_radius):
                candidate_indices.update(indices)
        hit_indices = sorted(index for index in candidate_indices if patch_intersects_track(index, points, vertices))
        hit_scales = geometry.iloc[hit_indices]["_ScaleNormalized"].astype(str).to_numpy()
        counts = {scale: int(np.sum(hit_scales == scale)) for scale in SCALES}
        results.append(
            {
                **track,
                "CandidateOrder": position,
                "TrackPointCount": int(len(points)),
                "TrackTimeMinMs": float(points[:, 2].min()) if len(points) else np.nan,
                "TrackTimeMaxMs": float(points[:, 2].max()) if len(points) else np.nan,
                "HitPatchIndices": hit_indices,
                "HitPatchCount": int(len(hit_indices)),
                "SmallCount": counts["small"],
                "MediumCount": counts["medium"],
                "LargeCount": counts["large"],
                "ScaleCount": int(sum(value > 0 for value in counts.values())),
            }
        )
        if position % 50 == 0 or position == len(tracks):
            print(f"[well-match] scan {position}/{len(tracks)} nonempty={sum(bool(row['HitPatchCount']) for row in results)}", flush=True)
    return results, excluded_sources


def select_well_candidates(candidates: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    nonempty = [row for row in candidates if row["HitPatchCount"] > 0]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in nonempty:
        by_source.setdefault(str(row["SourceWellName"]), []).append(row)
    for rows in by_source.values():
        rows.sort(
            key=lambda row: (
                0 if row["TrackKind"] == "real" else 1,
                -int(row["ScaleCount"]),
                -int(row["HitPatchCount"]),
                natural_key(str(row["TrackName"])),
            )
        )
    source_names = sorted(by_source, key=natural_key)
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < target_count:
        added = False
        for source in source_names:
            rows = by_source[source]
            if round_index < len(rows):
                selected.append(rows[round_index])
                added = True
                if len(selected) == target_count:
                    break
        if not added:
            break
        round_index += 1
    if len(selected) < target_count:
        raise RuntimeError(f"only {len(selected)} non-empty well-matched candidates available; target={target_count}")
    return selected


def parse_well_trajectory_exclusion_points(config: dict[str, Any], config_dir: Path) -> tuple[np.ndarray, str]:
    vtk_path = resolve_path(config["well_exclusion_vtk"], config_dir)
    if not vtk_path.is_file():
        raise FileNotFoundError(vtk_path)
    mesh = pv.read(vtk_path)
    points = np.asarray(mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points[:, :2]).all():
        raise ValueError(f"invalid well trajectory points in {vtk_path}")
    return points[:, :2].copy(), str(vtk_path)


def build_direct_cell_candidates(
    config: dict[str, Any],
    config_dir: Path,
    geometry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[int, int], list[int]], dict[str, Any]]:
    cell_size = float(config.get("cell_size_m", 300.0))
    target_block = config["target_block"]
    origin_x = math.floor(float(target_block["x_min"]) / cell_size) * cell_size
    origin_y = math.floor(float(target_block["y_min"]) / cell_size) * cell_size
    x = geometry["CenterX"].to_numpy(dtype=float)
    y = geometry["CenterY"].to_numpy(dtype=float)
    ix = np.floor((x - origin_x) / cell_size).astype(np.int64)
    iy = np.floor((y - origin_y) / cell_size).astype(np.int64)
    center_x = origin_x + (ix + 0.5) * cell_size
    center_y = origin_y + (iy + 0.5) * cell_size
    inside_target = (
        (center_x >= float(target_block["x_min"]))
        & (center_x <= float(target_block["x_max"]))
        & (center_y >= float(target_block["y_min"]))
        & (center_y <= float(target_block["y_max"]))
    )
    assignment = pd.DataFrame(
        {
            "PatchPosition": np.arange(len(geometry), dtype=np.int64),
            "CellIX": ix,
            "CellIY": iy,
            "InsideTarget": inside_target,
            "Scale": geometry["_ScaleNormalized"].to_numpy(dtype=str),
        }
    )
    assignment = assignment.loc[assignment["InsideTarget"]].copy()
    patch_lists: dict[tuple[int, int], list[int]] = {}
    for (cell_ix, cell_iy), group in assignment.groupby(["CellIX", "CellIY"], sort=True):
        patch_lists[(int(cell_ix), int(cell_iy))] = group["PatchPosition"].astype(int).tolist()
    rows: list[dict[str, Any]] = []
    for (cell_ix, cell_iy), patch_positions in sorted(patch_lists.items()):
        scales = geometry.iloc[patch_positions]["_ScaleNormalized"].astype(str)
        rows.append(
            {
                "CellIX": cell_ix,
                "CellIY": cell_iy,
                "CellXMinM": origin_x + cell_ix * cell_size,
                "CellXMaxM": origin_x + (cell_ix + 1) * cell_size,
                "CellYMinM": origin_y + cell_iy * cell_size,
                "CellYMaxM": origin_y + (cell_iy + 1) * cell_size,
                "CellCenterX": center_x[patch_positions[0]],
                "CellCenterY": center_y[patch_positions[0]],
                "PatchCount": len(patch_positions),
                "SmallCount": int((scales == "small").sum()),
                "MediumCount": int((scales == "medium").sum()),
                "LargeCount": int((scales == "large").sum()),
                "ScaleCount": int(scales.nunique()),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("no occupied target cells were found")

    well_points, well_vtk = parse_well_trajectory_exclusion_points(config, config_dir)
    well_tree = cKDTree(well_points)
    candidate_centers = candidates[["CellCenterX", "CellCenterY"]].to_numpy(dtype=float)
    nearest_well_distance = well_tree.query(candidate_centers, workers=-1)[0]
    safety_distance = float(config.get("no_well_safety_m", 100.0))
    required_distance = math.sqrt(2.0) * cell_size / 2.0 + safety_distance
    candidates["NearestWellTraceDistanceM"] = nearest_well_distance
    candidates["NoWellRequiredDistanceM"] = required_distance
    candidates["IsNoWell"] = nearest_well_distance > required_distance
    min_patch_count = int(config.get("min_patch_count", 1))
    minimum_scale_count = int(config.get("minimum_scale_count", 1))
    candidates["Eligible"] = (
        candidates["IsNoWell"]
        & (candidates["PatchCount"] >= min_patch_count)
        & (candidates["ScaleCount"] >= minimum_scale_count)
    )
    metadata = {
        "cell_size_m": cell_size,
        "cell_origin_x_m": origin_x,
        "cell_origin_y_m": origin_y,
        "no_well_safety_m": safety_distance,
        "no_well_required_center_distance_m": required_distance,
        "well_exclusion_vtk": well_vtk,
        "well_exclusion_point_count": int(len(well_points)),
        "occupied_cell_count": int(len(candidates)),
        "eligible_cell_count": int(candidates["Eligible"].sum()),
        "min_patch_count": min_patch_count,
        "minimum_scale_count": minimum_scale_count,
    }
    return candidates, patch_lists, metadata


def select_direct_cells(candidates: pd.DataFrame, target_count: int) -> pd.DataFrame:
    eligible = candidates.loc[candidates["Eligible"]].copy().reset_index(drop=False).rename(columns={"index": "CandidateIndex"})
    if len(eligible) < target_count:
        raise RuntimeError(f"only {len(eligible)} eligible no-well cells available; target={target_count}")
    centers = eligible[["CellCenterX", "CellCenterY"]].to_numpy(dtype=float)
    region_center = np.array(
        [
            (float(candidates["CellXMinM"].min()) + float(candidates["CellXMaxM"].max())) / 2.0,
            (float(candidates["CellYMinM"].min()) + float(candidates["CellYMaxM"].max())) / 2.0,
        ]
    )
    patch_values = eligible["PatchCount"].to_numpy(dtype=float)
    quality_threshold = float(np.quantile(patch_values, 0.75))
    seed_pool = np.flatnonzero(patch_values >= quality_threshold)
    center_distance = np.linalg.norm(centers - region_center, axis=1)
    first_position = int(seed_pool[np.argmin(center_distance[seed_pool])])
    selected_positions = [first_position]
    selected_minimum_distances = [0.0]
    minimum_distance_squared = np.sum((centers - centers[first_position]) ** 2, axis=1)
    while len(selected_positions) < target_count:
        minimum_distance_squared[selected_positions] = -1.0
        maximum = float(np.max(minimum_distance_squared))
        tied = np.flatnonzero(np.isclose(minimum_distance_squared, maximum, rtol=1.0e-12, atol=1.0e-6))
        if len(tied) > 1:
            tied_frame = eligible.iloc[tied]
            tie_order = np.lexsort(
                (
                    tied_frame["CellIY"].to_numpy(dtype=int),
                    tied_frame["CellIX"].to_numpy(dtype=int),
                    -tied_frame["PatchCount"].to_numpy(dtype=int),
                )
            )
            next_position = int(tied[tie_order[0]])
        else:
            next_position = int(tied[0])
        selected_minimum_distances.append(float(np.sqrt(max(minimum_distance_squared[next_position], 0.0))))
        selected_positions.append(next_position)
        distance_to_new = np.sum((centers - centers[next_position]) ** 2, axis=1)
        minimum_distance_squared = np.minimum(minimum_distance_squared, distance_to_new)
    selected = eligible.iloc[selected_positions].copy().reset_index(drop=True)
    selected["SelectionOrder"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    selected["MinDistanceToSelectedCentersM"] = np.asarray(selected_minimum_distances, dtype=float)
    return selected


def write_patch_vtk(path: Path, patches: pd.DataFrame, title: str) -> None:
    vertices = np.stack(
        [patches[[f"V{vertex}{axis}" for axis in "XYZ"]].to_numpy(dtype=float) for vertex in range(1, 5)],
        axis=1,
    )
    points = vertices.reshape(-1, 3)
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} double",
    ]
    lines.extend(f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}" for point in points)
    lines.append(f"POLYGONS {len(patches)} {5 * len(patches)}")
    lines.extend(f"4 {4 * index} {4 * index + 1} {4 * index + 2} {4 * index + 3}" for index in range(len(patches)))
    lines.append(f"CELL_DATA {len(patches)}")
    numeric_arrays = {
        "PatchIndex": np.arange(1, len(patches) + 1, dtype=np.int64),
        "FractureScaleCode": patches["FractureScale"].astype(str).str.lower().map(SCALE_CODE).to_numpy(dtype=np.int64),
        "CenterX": patches["CenterX"].to_numpy(dtype=float),
        "CenterY": patches["CenterY"].to_numpy(dtype=float),
        "CenterTimeMs": patches["CenterTime"].to_numpy(dtype=float),
        "PatchAreaM2": pd.to_numeric(patches.get("PatchAreaM2", np.zeros(len(patches))), errors="coerce").fillna(0).to_numpy(dtype=float),
        "Confidence": pd.to_numeric(patches.get("Confidence", np.zeros(len(patches))), errors="coerce").fillna(0).to_numpy(dtype=float),
    }
    for name, values in numeric_arrays.items():
        if np.issubdtype(values.dtype, np.integer):
            lines.extend([f"SCALARS {name} int 1", "LOOKUP_TABLE default", *(str(int(value)) for value in values)])
        else:
            lines.extend([f"SCALARS {name} double 1", "LOOKUP_TABLE default", *(f"{float(value):.17g}" for value in values)])
    lines.extend(["FIELD FieldData 1", f"PatchID 1 {len(patches)} string"])
    lines.extend(str(value) for value in patches["PatchID"].astype(str))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(path)


def write_trajectory_vtk(path: Path, points: np.ndarray, title: str) -> None:
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} double",
    ]
    lines.extend(f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}" for point in points)
    lines.append(f"LINES 1 {len(points) + 1}")
    lines.append(f"{len(points)} {' '.join(str(index) for index in range(len(points)))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(path)


def run_well_match(
    config: dict[str, Any],
    config_dir: Path,
    config_path: Path,
    output_dir: Path,
    target_count: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    patch_csv = resolve_path(config["input_patch_csv"], config_dir)
    full_patches, geometry, vertices = load_patch_inputs(patch_csv)
    candidates, excluded_sources = scan_well_matches(config, config_dir, geometry, vertices)
    target = int(target_count or config["target_count"])
    selected = select_well_candidates(candidates, target)
    prefix = str(config["group_prefix"])
    models_root = output_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    selected_lookup = {id(row): position for position, row in enumerate(selected, start=1)}
    for row in candidates:
        selected_order = selected_lookup.get(id(row), 0)
        candidate_rows.append(
            {
                "Region": config["region"],
                "CandidateOrder": row["CandidateOrder"],
                "SourceWellName": row["SourceWellName"],
                "TrackKind": row["TrackKind"],
                "TrackName": row["TrackName"],
                "TrackPointCount": row["TrackPointCount"],
                "TrackTimeMinMs": row["TrackTimeMinMs"],
                "TrackTimeMaxMs": row["TrackTimeMaxMs"],
                "HitPatchCount": row["HitPatchCount"],
                "SmallCount": row["SmallCount"],
                "MediumCount": row["MediumCount"],
                "LargeCount": row["LargeCount"],
                "ScaleCount": row["ScaleCount"],
                "Selected": selected_order > 0,
                "SelectionOrder": selected_order,
            }
        )
    write_csv(pd.DataFrame(candidate_rows), output_dir / "candidate_index.csv")

    index_rows: list[dict[str, Any]] = []
    for order, row in enumerate(selected, start=1):
        group_id = f"{prefix}_{order:04d}"
        group_dir = models_root / group_id
        group_dir.mkdir(parents=True, exist_ok=False)
        patch_positions = list(row["HitPatchIndices"])
        patches = full_patches.iloc[patch_positions].copy()
        patch_path = group_dir / "patches.csv"
        write_csv(patches, patch_path)

        trajectory = pd.read_csv(row["TrackPath"], encoding="utf-8-sig", usecols=["X", "Y", "TIME"])
        trajectory = trajectory.apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
        trajectory = trajectory.to_numpy(dtype=float)
        trajectory = trajectory[np.argsort(trajectory[:, 2], kind="stable")]
        trajectory_frame = pd.DataFrame(trajectory, columns=["X", "Y", "TIME"])
        trajectory_frame.insert(0, "TrackOrder", np.arange(1, len(trajectory_frame) + 1, dtype=np.int64))
        trajectory_path = group_dir / "trajectory.csv"
        write_csv(trajectory_frame, trajectory_path)
        patch_vtk_path = group_dir / "fractures.vtk"
        trajectory_vtk_path = group_dir / "trajectory.vtk"
        write_patch_vtk(patch_vtk_path, patches, f"{group_id} well-matched fracture patches")
        write_trajectory_vtk(trajectory_vtk_path, trajectory, f"{group_id} matched well trajectory")
        metadata = {
            "status": "pass",
            "group_id": group_id,
            "region": config["region"],
            "source_well_name": row["SourceWellName"],
            "track_kind": row["TrackKind"],
            "track_name": row["TrackName"],
            "track_input_path": str(row["TrackPath"]),
            "patch_input_path": str(patch_csv),
            "patch_count": int(len(patches)),
            "small_count": int((patches["FractureScale"].astype(str).str.lower() == "small").sum()),
            "medium_count": int((patches["FractureScale"].astype(str).str.lower() == "medium").sum()),
            "large_count": int((patches["FractureScale"].astype(str).str.lower() == "large").sum()),
            "track_point_count": int(len(trajectory)),
            "time_range_ms": [float(trajectory[:, 2].min()), float(trajectory[:, 2].max())],
            "patch_ids": patches["PatchID"].astype(str).tolist(),
            "selection_order": order,
            "match_contract": "exact_segment_vs_quadrilateral_triangles",
            "patches_csv": str(patch_path),
            "trajectory_csv": str(trajectory_path),
            "fractures_vtk": str(patch_vtk_path),
            "trajectory_vtk": str(trajectory_vtk_path),
        }
        metadata_path = group_dir / "metadata.json"
        write_json(metadata_path, metadata)
        index_rows.append(
            {
                "GroupID": group_id,
                "Region": config["region"],
                "SourceWellName": row["SourceWellName"],
                "TrackKind": row["TrackKind"],
                "TrackName": row["TrackName"],
                "PatchCount": int(len(patches)),
                "SmallCount": metadata["small_count"],
                "MediumCount": metadata["medium_count"],
                "LargeCount": metadata["large_count"],
                "ScaleCount": int(row["ScaleCount"]),
                "PatchesCSV": str(patch_path.relative_to(output_dir)),
                "TrajectoryCSV": str(trajectory_path.relative_to(output_dir)),
                "FracturesVTK": str(patch_vtk_path.relative_to(output_dir)),
                "TrajectoryVTK": str(trajectory_vtk_path.relative_to(output_dir)),
                "MetadataJSON": str(metadata_path.relative_to(output_dir)),
            }
        )
        if order % 25 == 0 or order == len(selected):
            print(f"[well-match] write {order}/{len(selected)} groups", flush=True)
    index = pd.DataFrame(index_rows)
    write_csv(index, output_dir / "model_index.csv")
    summary = {
        "status": "pass",
        "mode": "well-match",
        "region": config["region"],
        "input_patch_csv": str(patch_csv),
        "config_path": str(config_path),
        "target_count": target,
        "model_count": int(len(index)),
        "source_well_count": int(index["SourceWellName"].nunique()),
        "real_model_count": int((index["TrackKind"] == "real").sum()),
        "virtual_model_count": int((index["TrackKind"] == "virtual").sum()),
        "excluded_prior_source_wells": excluded_sources,
        "candidate_track_count": int(len(candidates)),
        "nonempty_candidate_count": int(sum(row["HitPatchCount"] > 0 for row in candidates)),
        "total_patch_rows": int(index["PatchCount"].sum()),
        "scale_patch_totals": {
            "small": int(index["SmallCount"].sum()),
            "medium": int(index["MediumCount"].sum()),
            "large": int(index["LargeCount"].sum()),
        },
        "match_contract": "exact_segment_vs_quadrilateral_triangles",
        "canonical_record": "patches.csv (all original Step8 columns preserved)",
        "vtk_role": "visualization companion",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config_resolved.json", {**config, "config_path": str(config_path), "resolved_patch_csv": str(patch_csv), "resolved_output_dir": str(output_dir)})
    (output_dir / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[well-match] finished models={len(index)} output={output_dir}", flush=True)
    return summary


def run_direct_cells(
    config: dict[str, Any],
    config_dir: Path,
    config_path: Path,
    output_dir: Path,
    target_count: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    patch_csv = resolve_path(config["input_patch_csv"], config_dir)
    full_patches, geometry, _ = load_patch_inputs(patch_csv)
    candidates, patch_lists, cell_metadata = build_direct_cell_candidates(config, config_dir, geometry)
    target = int(target_count or config["target_count"])
    selected = select_direct_cells(candidates, target)
    prefix = str(config["sample_prefix"])
    samples_root = output_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    selected_keys = set(zip(selected["CellIX"].astype(int), selected["CellIY"].astype(int)))
    candidate_output = candidates.copy()
    candidate_output["Selected"] = [tuple(row) in selected_keys for row in candidate_output[["CellIX", "CellIY"]].astype(int).to_numpy()]
    selected_lookup = {
        (int(row.CellIX), int(row.CellIY)): int(row.SelectionOrder)
        for row in selected.itertuples(index=False)
    }
    candidate_output["SelectionOrder"] = [
        selected_lookup.get((int(row.CellIX), int(row.CellIY)), 0) for row in candidate_output.itertuples(index=False)
    ]
    write_csv(candidate_output, output_dir / "candidate_cells.csv")

    index_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        order = int(row.SelectionOrder)
        cell_ix = int(row.CellIX)
        cell_iy = int(row.CellIY)
        sample_id = f"{prefix}_{order:04d}"
        sample_dir = samples_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=False)
        patch_positions = patch_lists[(cell_ix, cell_iy)]
        patches = full_patches.iloc[patch_positions].copy()
        patch_path = sample_dir / "patches.csv"
        write_csv(patches, patch_path)
        patch_vtk_path = sample_dir / "fractures.vtk"
        write_patch_vtk(patch_vtk_path, patches, f"{sample_id} direct-conditioning fracture patches")
        metadata = {
            "status": "pass",
            "sample_id": sample_id,
            "region": config["region"],
            "cell_ix": cell_ix,
            "cell_iy": cell_iy,
            "cell_size_m": float(cell_metadata["cell_size_m"]),
            "cell_bounds_m": {
                "x_min": float(row.CellXMinM),
                "x_max": float(row.CellXMaxM),
                "y_min": float(row.CellYMinM),
                "y_max": float(row.CellYMaxM),
            },
            "cell_center_xy_m": [float(row.CellCenterX), float(row.CellCenterY)],
            "nearest_well_trace_distance_m": float(row.NearestWellTraceDistanceM),
            "no_well_required_center_distance_m": float(row.NoWellRequiredDistanceM),
            "patch_count": int(len(patches)),
            "small_count": int((patches["FractureScale"].astype(str).str.lower() == "small").sum()),
            "medium_count": int((patches["FractureScale"].astype(str).str.lower() == "medium").sum()),
            "large_count": int((patches["FractureScale"].astype(str).str.lower() == "large").sum()),
            "time_range_ms": [float(patches["CenterTime"].min()), float(patches["CenterTime"].max())],
            "patch_ids": patches["PatchID"].astype(str).tolist(),
            "selection_order": order,
            "assignment_contract": "patch center inside one non-overlapping 300 m XY cell",
            "no_well_contract": "cell and outward safety buffer contain no Step8 real-well trajectory point",
            "patch_input_path": str(patch_csv),
            "patches_csv": str(patch_path),
            "fractures_vtk": str(patch_vtk_path),
        }
        metadata_path = sample_dir / "metadata.json"
        write_json(metadata_path, metadata)
        index_rows.append(
            {
                "SampleID": sample_id,
                "Region": config["region"],
                "CellIX": cell_ix,
                "CellIY": cell_iy,
                "CellXMinM": float(row.CellXMinM),
                "CellXMaxM": float(row.CellXMaxM),
                "CellYMinM": float(row.CellYMinM),
                "CellYMaxM": float(row.CellYMaxM),
                "PatchCount": metadata["patch_count"],
                "SmallCount": metadata["small_count"],
                "MediumCount": metadata["medium_count"],
                "LargeCount": metadata["large_count"],
                "ScaleCount": int((np.asarray([metadata["small_count"], metadata["medium_count"], metadata["large_count"]]) > 0).sum()),
                "NearestWellTraceDistanceM": float(row.NearestWellTraceDistanceM),
                "PatchesCSV": str(patch_path.relative_to(output_dir)),
                "FracturesVTK": str(patch_vtk_path.relative_to(output_dir)),
                "MetadataJSON": str(metadata_path.relative_to(output_dir)),
            }
        )
        if order % 20 == 0 or order == len(selected):
            print(f"[direct-cell] write {order}/{len(selected)} samples", flush=True)
    index = pd.DataFrame(index_rows)
    index = index.sort_values("SampleID", kind="stable").reset_index(drop=True)
    write_csv(index, output_dir / "sample_index.csv")
    summary = {
        "status": "pass",
        "mode": "direct-cell",
        "region": config["region"],
        "input_patch_csv": str(patch_csv),
        "config_path": str(config_path),
        "target_count": target,
        "sample_count": int(len(index)),
        "total_patch_rows": int(index["PatchCount"].sum()),
        "scale_patch_totals": {
            "small": int(index["SmallCount"].sum()),
            "medium": int(index["MediumCount"].sum()),
            "large": int(index["LargeCount"].sum()),
        },
        "cell_contract": "300 m x 300 m XY cells; patch center assignment; no overlap between samples",
        "no_well_contract": "Step8 real-well trajectories excluded with configured safety buffer",
        "canonical_record": "patches.csv (all original Step8 columns preserved)",
        "vtk_role": "visualization companion",
        **cell_metadata,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config_resolved.json", {**config, "config_path": str(config_path), "resolved_patch_csv": str(patch_csv), "resolved_output_dir": str(output_dir)})
    (output_dir / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[direct-cell] finished samples={len(index)} output={output_dir}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override configured output directory")
    parser.add_argument("--target-count", type=int, default=None, help="Override configured group/sample count")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_dir = config_path.parent
    config = read_json(config_path)
    mode = str(config["mode"]).strip().lower()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else resolve_path(config["output_dir"], config_dir)
    prepare_output(output_dir, overwrite=args.overwrite)
    if mode == "well-match":
        run_well_match(config, config_dir, config_path, output_dir, args.target_count)
    elif mode == "direct-cell":
        run_direct_cells(config, config_dir, config_path, output_dir, args.target_count)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
