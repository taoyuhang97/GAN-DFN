# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from merge_unit_dfn_vtks import write_legacy_vtk_polygons

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class LocalPatch:
    local_id: int
    original_index: int
    center: np.ndarray
    vertices: np.ndarray
    area: float
    azimuth: float
    dip: float
    length: float
    height: float
    participates_in_fixed_y: bool
    participates_in_fixed_x: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local display-only DFN VTK by adding conservative bridge patches "
            "between nearby, similarly oriented fracture patches."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--output-vtk", type=Path, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--half-width", type=float, default=12.5)
    parser.add_argument("--along-half-width", type=float, default=2400.0)
    parser.add_argument("--max-connect-distance", type=float, default=150.0)
    parser.add_argument("--max-z-gap", type=float, default=35.0)
    parser.add_argument("--minor-offset-limit", type=float, default=35.0)
    parser.add_argument("--azimuth-tol", type=float, default=22.5)
    parser.add_argument("--dip-tol", type=float, default=20.0)
    parser.add_argument("--min-along-ratio", type=float, default=0.70)
    parser.add_argument("--max-connect-area", type=float, default=40000.0)
    parser.add_argument("--min-connect-area", type=float, default=1.0)
    parser.add_argument("--max-bridges", type=int, default=1500)
    parser.add_argument("--max-bridges-per-patch", type=int, default=2)
    parser.add_argument("--bridge-height-scale", type=float, default=0.75)
    parser.add_argument("--min-bridge-height", type=float, default=6.0)
    parser.add_argument("--max-bridge-height", type=float, default=45.0)
    parser.add_argument("--max-bridge-length", type=float, default=180.0)
    parser.add_argument("--progress-interval", type=int, default=100000)
    return parser


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def next_nonempty(handle) -> str:
    for line in handle:
        stripped = line.strip()
        if stripped:
            return stripped
    raise ValueError("unexpected end of VTK file")


def read_points(handle, point_count: int) -> np.ndarray:
    points = np.empty((point_count, 3), dtype=np.float64)
    cursor = 0
    buffer: list[float] = []
    while cursor < point_count:
        line = next_nonempty(handle)
        buffer.extend(float(value) for value in line.split())
        while len(buffer) >= 3 and cursor < point_count:
            points[cursor] = (buffer[0], buffer[1], buffer[2])
            del buffer[:3]
            cursor += 1
    return points


def polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    origin = vertices[0]
    area = 0.0
    for idx in range(1, len(vertices) - 1):
        area += 0.5 * float(np.linalg.norm(np.cross(vertices[idx] - origin, vertices[idx + 1] - origin)))
    return area


def axial_angle_diff(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def axial_angle_mean(a: float, b: float) -> float:
    angles = np.deg2rad(np.asarray([a, b], dtype=float) * 2.0)
    mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    return float((np.rad2deg(mean_angle) / 2.0) % 180.0)


def estimate_geometry(vertices: np.ndarray) -> tuple[float, float, float, float]:
    center = vertices.mean(axis=0)
    xy = vertices[:, :2]
    best_i, best_j = 0, 1 if len(vertices) > 1 else 0
    best_xy_dist = -1.0
    best_3d_dist = -1.0
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dxy = float(np.linalg.norm(xy[i] - xy[j]))
            d3d = float(np.linalg.norm(vertices[i] - vertices[j]))
            if dxy > best_xy_dist:
                best_xy_dist = dxy
                best_3d_dist = d3d
                best_i, best_j = i, j
    vec = vertices[best_j] - vertices[best_i]
    azimuth = float((np.rad2deg(math.atan2(vec[1], vec[0])) + 180.0) % 180.0)
    length = max(best_3d_dist, 1e-6)

    normal = np.zeros(3, dtype=float)
    if len(vertices) >= 3:
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 1e-8:
        normal = normal / normal_norm
        dip = float(np.rad2deg(math.acos(min(1.0, abs(float(normal[2]))))))
    else:
        dip = 60.0

    projected = vertices - center
    strike_vec = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth)), 0.0])
    along = projected @ strike_vec
    residual = projected - np.outer(along, strike_vec)
    height = max(float(np.ptp(np.linalg.norm(residual, axis=1))) * 2.0, 1.0)
    return azimuth, dip, length, height


def in_cross_region(mins: np.ndarray, maxs: np.ndarray, x0: float, y0: float, half_width: float, along_half_width: float) -> tuple[bool, bool]:
    fixed_y = mins[1] <= y0 + half_width and maxs[1] >= y0 - half_width and mins[0] <= x0 + along_half_width and maxs[0] >= x0 - along_half_width
    fixed_x = mins[0] <= x0 + half_width and maxs[0] >= x0 - half_width and mins[1] <= y0 + along_half_width and maxs[1] >= y0 - along_half_width
    return fixed_y, fixed_x


def load_local_patches(args: argparse.Namespace) -> list[LocalPatch]:
    patches: list[LocalPatch] = []
    with args.input_vtk.open("r", encoding="utf-8") as handle:
        header = [next_nonempty(handle) for _ in range(4)]
        expect(header[0].startswith("# vtk DataFile"), f"unsupported VTK header: {args.input_vtk}")
        expect(header[2] == "ASCII", f"only ASCII VTK is supported: {args.input_vtk}")
        expect(header[3] == "DATASET POLYDATA", f"only POLYDATA VTK is supported: {args.input_vtk}")
        point_header = next_nonempty(handle).split()
        expect(len(point_header) >= 3 and point_header[0] == "POINTS", f"POINTS block missing: {args.input_vtk}")
        point_count = int(point_header[1])
        print(f"[connect-display] reading points: {point_count}", flush=True)
        points = read_points(handle, point_count)

        polygon_header = next_nonempty(handle).split()
        expect(len(polygon_header) >= 3 and polygon_header[0] == "POLYGONS", f"POLYGONS block missing: {args.input_vtk}")
        polygon_count = int(polygon_header[1])
        print(f"[connect-display] scanning polygons: {polygon_count}", flush=True)
        iterator = range(polygon_count)
        if tqdm is not None:
            iterator = tqdm(iterator, total=polygon_count, desc="select local patches", unit="patch")
        for polygon_index in iterator:
            parts = next_nonempty(handle).split()
            vertex_count = int(parts[0])
            idx = [int(value) for value in parts[1:]]
            if len(idx) != vertex_count or vertex_count < 3:
                continue
            vertices = points[np.asarray(idx, dtype=np.int64)]
            mins = np.nanmin(vertices, axis=0)
            maxs = np.nanmax(vertices, axis=0)
            fixed_y, fixed_x = in_cross_region(mins, maxs, args.x, args.y, args.half_width, args.along_half_width)
            if not (fixed_y or fixed_x):
                continue
            area = polygon_area(vertices)
            azimuth, dip, length, height = estimate_geometry(vertices)
            patches.append(
                LocalPatch(
                    local_id=len(patches),
                    original_index=polygon_index,
                    center=vertices.mean(axis=0),
                    vertices=vertices.copy(),
                    area=area,
                    azimuth=azimuth,
                    dip=dip,
                    length=length,
                    height=height,
                    participates_in_fixed_y=bool(fixed_y),
                    participates_in_fixed_x=bool(fixed_x),
                )
            )
            if tqdm is None and args.progress_interval > 0 and polygon_index % args.progress_interval == 0:
                print(f"[connect-display] scanned {polygon_index}/{polygon_count}, selected={len(patches)}", flush=True)
    print(f"[connect-display] selected local patches: {len(patches)}", flush=True)
    return patches


def build_bridge_vertices(center: np.ndarray, azimuth: float, dip: float, length: float, height: float) -> np.ndarray:
    az_rad = math.radians(azimuth)
    dip_rad = math.radians(dip)
    u_hat = np.array([math.cos(az_rad), math.sin(az_rad), 0.0], dtype=float)
    px, py = -math.sin(az_rad), math.cos(az_rad)
    v_hat = np.array([px * math.cos(dip_rad), py * math.cos(dip_rad), -math.sin(dip_rad)], dtype=float)
    norm = float(np.linalg.norm(v_hat))
    if norm <= 1e-8:
        v_hat = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        v_hat /= norm
    half_u = length / 2.0
    half_v = height / 2.0
    return np.asarray(
        [
            center - half_u * u_hat - half_v * v_hat,
            center + half_u * u_hat - half_v * v_hat,
            center + half_u * u_hat + half_v * v_hat,
            center - half_u * u_hat + half_v * v_hat,
        ],
        dtype=float,
    )


def propose_bridges(patches: list[LocalPatch], args: argparse.Namespace) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    connectable = [
        p for p in patches
        if args.min_connect_area <= p.area <= args.max_connect_area and np.all(np.isfinite(p.center))
    ]
    print(f"[connect-display] connectable patches: {len(connectable)}", flush=True)
    centers = np.asarray([p.center for p in connectable], dtype=float)
    if len(connectable) < 2:
        return []

    for i in range(len(connectable)):
        p_i = connectable[i]
        delta = centers[i + 1:] - p_i.center
        if len(delta) == 0:
            continue
        dxy = np.linalg.norm(delta[:, :2], axis=1)
        dz = np.abs(delta[:, 2])
        nearby = np.where((dxy > 1e-6) & (dxy <= args.max_connect_distance) & (dz <= args.max_z_gap))[0]
        for rel_j in nearby.tolist():
            j = i + 1 + int(rel_j)
            p_j = connectable[j]
            if not (p_i.participates_in_fixed_y and p_j.participates_in_fixed_y) and not (p_i.participates_in_fixed_x and p_j.participates_in_fixed_x):
                continue
            az_diff = axial_angle_diff(p_i.azimuth, p_j.azimuth)
            if az_diff > args.azimuth_tol:
                continue
            dip_diff = abs(p_i.dip - p_j.dip)
            if dip_diff > args.dip_tol:
                continue
            mean_az = axial_angle_mean(p_i.azimuth, p_j.azimuth)
            strike_vec = np.array([math.cos(math.radians(mean_az)), math.sin(math.radians(mean_az))], dtype=float)
            delta_xy = p_j.center[:2] - p_i.center[:2]
            dist_xy = float(np.linalg.norm(delta_xy))
            if dist_xy <= 1e-6:
                continue
            along = abs(float(delta_xy @ strike_vec))
            along_ratio = along / dist_xy
            if along_ratio < args.min_along_ratio:
                continue
            minor = math.sqrt(max(dist_xy * dist_xy - along * along, 0.0))
            if minor > args.minor_offset_limit:
                continue
            score = (1.0 - az_diff / max(args.azimuth_tol, 1e-6)) * 0.35
            score += (1.0 - min(dip_diff / max(args.dip_tol, 1e-6), 1.0)) * 0.20
            score += (1.0 - min(dist_xy / max(args.max_connect_distance, 1e-6), 1.0)) * 0.25
            score += along_ratio * 0.20
            candidates.append({"i": p_i.local_id, "j": p_j.local_id, "score": score, "distance": dist_xy, "mean_az": mean_az})

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    used_per_patch = {p.local_id: 0 for p in patches}
    accepted: list[dict[str, object]] = []
    for cand in candidates:
        i = int(cand["i"])
        j = int(cand["j"])
        if used_per_patch[i] >= args.max_bridges_per_patch or used_per_patch[j] >= args.max_bridges_per_patch:
            continue
        accepted.append(cand)
        used_per_patch[i] += 1
        used_per_patch[j] += 1
        if len(accepted) >= args.max_bridges:
            break
    print(f"[connect-display] bridge candidates={len(candidates)}, accepted={len(accepted)}", flush=True)
    return accepted


def build_output_payload(patches: list[LocalPatch], bridges: list[dict[str, object]], args: argparse.Namespace) -> tuple[np.ndarray, list[list[int]], dict[str, np.ndarray], dict[str, str]]:
    points: list[list[float]] = []
    polygons: list[list[int]] = []
    connection_type: list[int] = []
    is_bridge: list[int] = []
    source_a: list[int] = []
    source_b: list[int] = []
    original_index: list[int] = []
    local_id_values: list[int] = []
    patch_area: list[float] = []
    bridge_score: list[float] = []

    for patch in patches:
        start = len(points)
        points.extend(patch.vertices.tolist())
        polygons.append([start + offset for offset in range(len(patch.vertices))])
        connection_type.append(0)
        is_bridge.append(0)
        source_a.append(patch.local_id)
        source_b.append(-1)
        original_index.append(patch.original_index)
        local_id_values.append(patch.local_id)
        patch_area.append(patch.area)
        bridge_score.append(0.0)

    patch_by_id = {patch.local_id: patch for patch in patches}
    for bridge_id, bridge in enumerate(bridges):
        p_i = patch_by_id[int(bridge["i"])]
        p_j = patch_by_id[int(bridge["j"])]
        center = (p_i.center + p_j.center) / 2.0
        mean_az = float(bridge["mean_az"])
        dip = float((p_i.dip + p_j.dip) / 2.0)
        length = min(max(float(bridge["distance"]), 5.0), float(args.max_bridge_length))
        height = float(np.clip((p_i.height + p_j.height) / 2.0 * args.bridge_height_scale, args.min_bridge_height, args.max_bridge_height))
        vertices = build_bridge_vertices(center, mean_az, dip, length, height)
        area = polygon_area(vertices)
        start = len(points)
        points.extend(vertices.tolist())
        polygons.append([start, start + 1, start + 2, start + 3])
        connection_type.append(1)
        is_bridge.append(1)
        source_a.append(p_i.local_id)
        source_b.append(p_j.local_id)
        original_index.append(-1)
        local_id_values.append(len(patches) + bridge_id)
        patch_area.append(area)
        bridge_score.append(float(bridge["score"]))

    cell_data = {
        "ConnectionType": np.asarray(connection_type, dtype=int),
        "IsBridge": np.asarray(is_bridge, dtype=int),
        "SourcePatchA": np.asarray(source_a, dtype=int),
        "SourcePatchB": np.asarray(source_b, dtype=int),
        "OriginalPatchIndex": np.asarray(original_index, dtype=int),
        "LocalPatchID": np.asarray(local_id_values, dtype=int),
        "PatchArea": np.asarray(patch_area, dtype=float),
        "BridgeScore": np.asarray(bridge_score, dtype=float),
    }
    scalar_types = {
        "ConnectionType": "int",
        "IsBridge": "int",
        "SourcePatchA": "int",
        "SourcePatchB": "int",
        "OriginalPatchIndex": "int",
        "LocalPatchID": "int",
        "PatchArea": "float",
        "BridgeScore": "float",
    }
    return np.asarray(points, dtype=float), polygons, cell_data, scalar_types


def main() -> None:
    args = build_parser().parse_args()
    patches = load_local_patches(args)
    bridges = propose_bridges(patches, args)
    points, polygons, cell_data, scalar_types = build_output_payload(patches, bridges, args)
    write_legacy_vtk_polygons(
        args.output_vtk,
        title="connected_display_local_dfn",
        points=points,
        polygons=polygons,
        cell_data=cell_data,
        scalar_types=scalar_types,
    )
    print(f"[connect-display] output_vtk={args.output_vtk}", flush=True)
    print(f"[connect-display] original_local_patches={len(patches)}", flush=True)
    print(f"[connect-display] bridge_patches={len(bridges)}", flush=True)
    print(f"[connect-display] total_patches={len(polygons)}", flush=True)


if __name__ == "__main__":
    main()
