"""Reusable data contract for layered Step9 profile construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


INTERSECTION_COLUMNS = [
    "region",
    "projection",
    "source",
    "layer",
    "scale",
    "patch_id",
    "h1",
    "z1",
    "h2",
    "z2",
    "display_mode",
    "surface_distance",
    "patch_area",
]
FAULT_COLUMNS = ["region", "projection", "h1", "z1", "h2", "z2"]


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
            count += 1
    return count


def write_base_layer(
    output_dir: Path,
    *,
    region: str,
    xz_section: Any,
    yz_section: Any,
    curves: dict[str, dict[str, np.ndarray]],
    track: Any,
    faults: dict[str, list[list[list[float]]]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write reusable seismic/well/horizon/fault data and a small manifest."""

    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "xz_h": np.asarray(xz_section.h, dtype=np.float64),
        "xz_time": np.asarray(xz_section.time, dtype=np.float64),
        "xz_values": np.asarray(xz_section.values, dtype=np.float64),
        "yz_h": np.asarray(yz_section.h, dtype=np.float64),
        "yz_time": np.asarray(yz_section.time, dtype=np.float64),
        "yz_values": np.asarray(yz_section.values, dtype=np.float64),
        "well_x": np.asarray(track["X"], dtype=np.float64),
        "well_y": np.asarray(track["Y"], dtype=np.float64),
        "well_time": np.asarray(track["TIME"], dtype=np.float64),
    }
    for projection, projection_curves in curves.items():
        for name, values in projection_curves.items():
            payload[f"{projection.lower()}_{name}"] = np.asarray(values, dtype=np.float64)
    data_path = base_dir / "base_section_data.npz"
    np.savez_compressed(data_path, **payload)

    fault_rows = []
    for projection, values in faults.items():
        for line in values:
            if hasattr(line, "h1") and hasattr(line, "z1"):
                array = np.asarray(
                    [[float(line.h1), float(line.z1)], [float(line.h2), float(line.z2)]],
                    dtype=float,
                )
            else:
                array = np.asarray(line, dtype=float)
            fault_rows.append(
                {
                    "region": region,
                    "projection": projection,
                    "h1": float(array[0, 0]),
                    "z1": float(array[0, 1]),
                    "h2": float(array[1, 0]),
                    "z2": float(array[1, 1]),
                }
            )
    fault_count = _write_csv(base_dir / "base_faults.csv", FAULT_COLUMNS, fault_rows)
    manifest = {
        "status": "pass",
        "region": region,
        "data": str(data_path),
        "faults": str(base_dir / "base_faults.csv"),
        "fault_count": fault_count,
        "curve_names": {
            projection: sorted(values)
            for projection, values in curves.items()
        },
        "metadata": metadata,
    }
    (base_dir / "base_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_intersection_layer(
    output_dir: Path,
    rows: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    layer_dir = output_dir / "intersections"
    count = _write_csv(
        layer_dir / "dfn_intersections.csv",
        INTERSECTION_COLUMNS,
        rows,
    )
    manifest = {
        "status": "pass",
        "rows": count,
        "intersections": str(layer_dir / "dfn_intersections.csv"),
        "metadata": metadata,
    }
    (layer_dir / "intersection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def projection_segment_rows(
    segments: Iterable[Any],
    *,
    region: str,
    layer_by_interval: dict[str, str],
    source: str = "dfn",
) -> list[dict[str, Any]]:
    rows = []
    for index, segment in enumerate(segments):
        interval = str(getattr(segment, "interval", "")).strip()
        rows.append(
            {
                "region": region,
                "projection": str(segment.projection),
                "source": source,
                "layer": layer_by_interval.get(interval, interval),
                "scale": str(getattr(segment, "scale", "")).strip().lower(),
                "patch_id": getattr(segment, "polygon_index", index),
                "h1": float(segment.h1),
                "z1": float(segment.z1),
                "h2": float(segment.h2),
                "z2": float(segment.z2),
                "display_mode": str(getattr(segment, "display_mode", "intersection")),
                "surface_distance": float(getattr(segment, "surface_distance", 0.0)),
                "patch_area": float(getattr(segment, "patch_area", 0.0)),
            }
        )
    return rows


def suppress_v_short_edges(
    segments: list[list[list[float]]],
    *,
    max_short_length_m: float = 25.0,
    min_turn_angle_deg: float = 20.0,
) -> tuple[list[list[list[float]]], dict[str, int]]:
    """Remove short fault edges that form sharp V-like turns at shared nodes."""

    import math
    from collections import defaultdict

    if not segments:
        return segments, {
            "input_count": 0,
            "output_count": 0,
            "removed_count": 0,
        }
    endpoint_nodes: dict[tuple[float, float], int] = {}
    node_sequences: list[tuple[int, int]] = []
    for line in segments:
        array = np.asarray(line, dtype=float)
        nodes = []
        for h_value, time_value in array:
            key = (round(float(h_value), 2), round(float(time_value), 2))
            if key not in endpoint_nodes:
                endpoint_nodes[key] = len(endpoint_nodes)
            nodes.append(endpoint_nodes[key])
        node_sequences.append((nodes[0], nodes[1]))

    incident: dict[int, list[int]] = defaultdict(list)
    for index, (first, second) in enumerate(node_sequences):
        incident[first].append(index)
        incident[second].append(index)

    display_line = []
    axial_angle = []
    for line in segments:
        array = np.asarray(line, dtype=float)
        first = np.asarray([array[0, 0], 2.0 * array[0, 1]])
        second = np.asarray([array[1, 0], 2.0 * array[1, 1]])
        display_line.append((first, second))
        axial_angle.append(
            math.degrees(
                math.atan2(
                    float(second[1] - first[1]),
                    float(second[0] - first[0]),
                )
            )
            % 180.0
        )
    lengths = [
        float(np.linalg.norm(second - first))
        for first, second in display_line
    ]

    remove: set[int] = set()
    for incident_indices in incident.values():
        if len(incident_indices) < 2:
            continue
        for short_index in incident_indices:
            if lengths[short_index] > float(max_short_length_m):
                continue
            for neighbor_index in incident_indices:
                if neighbor_index == short_index:
                    continue
                angle = abs(
                    (
                        axial_angle[short_index]
                        - axial_angle[neighbor_index]
                        + 90.0
                    )
                    % 180.0
                    - 90.0
                )
                if angle >= float(min_turn_angle_deg):
                    remove.add(short_index)
                    break

    kept = [
        line for index, line in enumerate(segments) if index not in remove
    ]
    return kept, {
        "input_count": int(len(segments)),
        "output_count": int(len(kept)),
        "removed_count": int(len(remove)),
        "removed_indices": sorted(int(value) for value in remove),
        "max_short_length_m": float(max_short_length_m),
        "min_turn_angle_deg": float(min_turn_angle_deg),
    }
