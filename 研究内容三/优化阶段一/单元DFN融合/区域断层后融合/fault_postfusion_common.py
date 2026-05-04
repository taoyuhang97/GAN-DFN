# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv


THIS_DIR = Path(__file__).resolve().parent
FUSION_DIR = THIS_DIR.parent
OPT_STAGE_DIR = FUSION_DIR.parent
BASELINE_DIR = OPT_STAGE_DIR / "G_DFN监督基线"

for candidate in (THIS_DIR, FUSION_DIR, BASELINE_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_csv_utf8, write_json
from merge_unit_dfn_vtks import read_legacy_vtk_polygons


UNIT_ID_PATTERN = re.compile(r"^BX(?P<block_x>\d+)_BY(?P<block_y>\d+)$", flags=re.IGNORECASE)
FAULT_PATCH_FILE_PATTERN = re.compile(
    r"^(?P<fault_name>.+?)__i(?P<cell_i>\d+)_j(?P<cell_j>\d+)$",
    flags=re.IGNORECASE,
)

PATCH_ORIGIN_CODE_TO_TEXT = {
    0: "original",
    1: "fault_parallel",
    2: "fault_perpendicular",
    3: "fault_surface",
}
PATCH_ORIGIN_TEXT_TO_CODE = {value: key for key, value in PATCH_ORIGIN_CODE_TO_TEXT.items()}

FAULT_ACTION_CODE_TO_TEXT = {
    0: "keep",
    1: "remove",
    2: "shrink",
    3: "induced",
    4: "surface",
}
FAULT_ACTION_TEXT_TO_CODE = {value: key for key, value in FAULT_ACTION_CODE_TO_TEXT.items()}

TEXT_EXPORT_COLUMNS = {
    "PatchOriginText",
    "FaultActionText",
    "NearestFaultName",
    "FaultName",
    "SourceUnitIDs",
}


def parse_unit_id(unit_id: str) -> tuple[int, int]:
    matched = UNIT_ID_PATTERN.match(str(unit_id).strip())
    if not matched:
        raise ValueError(f"invalid UnitID: {unit_id}")
    return int(matched.group("block_x")), int(matched.group("block_y"))


def build_target_unit_ids(
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
) -> list[str]:
    x0 = int(min(block_x_start, block_x_end))
    x1 = int(max(block_x_start, block_x_end))
    y0 = int(min(block_y_start, block_y_end))
    y1 = int(max(block_y_start, block_y_end))
    unit_ids = [
        f"BX{block_x}_BY{block_y}"
        for block_x in range(x0, x1 + 1)
        for block_y in range(y0, y1 + 1)
    ]
    return sorted(unit_ids, key=parse_unit_id)


def parse_fault_patch_file_info(path: Path) -> dict[str, Any]:
    file_path = Path(path)
    matched = FAULT_PATCH_FILE_PATTERN.match(file_path.stem)
    if not matched:
        raise ValueError(f"invalid fault patch filename: {file_path.name}")
    return {
        "fault_name": str(matched.group("fault_name")),
        "cell_i": int(matched.group("cell_i")),
        "cell_j": int(matched.group("cell_j")),
    }


def discover_fault_patch_files(
    fault_patches_root: Path,
    suffixes: tuple[str, ...] = (".vtp", ".vtk", ".vtu"),
) -> list[Path]:
    root = Path(fault_patches_root)
    if not root.exists():
        raise FileNotFoundError(f"fault_patches_root not found: {root}")
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        try:
            parse_fault_patch_file_info(path)
        except ValueError:
            continue
        candidates.append(path)
    return candidates


def is_fault_patch_in_region(
    path: Path,
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
) -> bool:
    info = parse_fault_patch_file_info(path)
    x0 = int(min(block_x_start, block_x_end))
    x1 = int(max(block_x_start, block_x_end))
    y0 = int(min(block_y_start, block_y_end))
    y1 = int(max(block_y_start, block_y_end))
    return x0 <= int(info["cell_i"]) <= x1 and y0 <= int(info["cell_j"]) <= y1


def normalize_vector(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        return arr / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return fallback / fallback_norm


def azimuth_diff_deg(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def best_fit_normal(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize_vector(vh[-1, :], fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    if normal[2] < 0.0:
        normal = -normal
    return normal


def strike_dip_from_normal(normal: np.ndarray) -> tuple[float, float]:
    nx, ny, nz = normalize_vector(np.asarray(normal, dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    horiz = math.sqrt(nx * nx + ny * ny) + 1e-12
    strike_rad = math.atan2(nx, ny)
    strike = (math.degrees(strike_rad) + 360.0) % 180.0
    dip = math.degrees(math.atan2(abs(nz), horiz))
    return float(strike), float(dip)


def build_plane_axes_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plane_normal = normalize_vector(np.asarray(normal, dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    strike_vec = np.array([-plane_normal[1], plane_normal[0], 0.0], dtype=float)
    strike_vec = normalize_vector(strike_vec, fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    dip_vec = normalize_vector(np.cross(plane_normal, strike_vec), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    if dip_vec[2] < 0.0:
        dip_vec = -dip_vec
    return plane_normal, strike_vec, dip_vec


def compute_quad_area(vertices: np.ndarray) -> float:
    pts = np.asarray(vertices, dtype=float)
    if pts.shape != (4, 3):
        raise ValueError(f"vertices must have shape (4, 3), got {pts.shape}")
    area1 = 0.5 * float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
    area2 = 0.5 * float(np.linalg.norm(np.cross(pts[3] - pts[0], pts[2] - pts[0])))
    return float(area1 + area2)


def vertices_from_panel_row(panel_row: pd.Series) -> np.ndarray:
    vertices = []
    for vertex_idx in range(1, 5):
        vertices.append(
            [
                float(panel_row[f"V{vertex_idx}X"]),
                float(panel_row[f"V{vertex_idx}Y"]),
                float(panel_row[f"V{vertex_idx}Z"]),
            ]
        )
    return np.asarray(vertices, dtype=float)


def make_patch_row_from_axes(
    center: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    length: float,
    height: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    patch_length = max(float(length), 1e-6)
    patch_height = max(float(height), 1e-6)
    center = np.asarray(center, dtype=float)
    u_axis = normalize_vector(np.asarray(u_vec, dtype=float), fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    v_axis = normalize_vector(np.asarray(v_vec, dtype=float), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    half_length = 0.5 * patch_length
    half_height = 0.5 * patch_height
    vertices = np.asarray(
        [
            center - half_length * u_axis - half_height * v_axis,
            center + half_length * u_axis - half_height * v_axis,
            center + half_length * u_axis + half_height * v_axis,
            center - half_length * u_axis + half_height * v_axis,
        ],
        dtype=float,
    )
    normal = normalize_vector(np.cross(u_axis, v_axis), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    azimuth_deg, dip_deg = strike_dip_from_normal(normal)
    row = {
        "CenterX": float(center[0]),
        "CenterY": float(center[1]),
        "CenterTIME": float(center[2]),
        "PatchLength": float(patch_length),
        "PatchHeight": float(patch_height),
        "PatchArea": float(compute_quad_area(vertices)),
        "Azimuth": float(azimuth_deg),
        "Dip": float(dip_deg),
        "NormalX": float(normal[0]),
        "NormalY": float(normal[1]),
        "NormalZ": float(normal[2]),
        "BBoxXMin": float(vertices[:, 0].min()),
        "BBoxXMax": float(vertices[:, 0].max()),
        "BBoxYMin": float(vertices[:, 1].min()),
        "BBoxYMax": float(vertices[:, 1].max()),
        "BBoxZMin": float(vertices[:, 2].min()),
        "BBoxZMax": float(vertices[:, 2].max()),
    }
    for vertex_idx in range(1, 5):
        row[f"V{vertex_idx}X"] = float(vertices[vertex_idx - 1, 0])
        row[f"V{vertex_idx}Y"] = float(vertices[vertex_idx - 1, 1])
        row[f"V{vertex_idx}Z"] = float(vertices[vertex_idx - 1, 2])
    if extra:
        row.update(extra)
    return row


def make_patch_row_from_panel_row(panel_row: pd.Series, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    vertices = vertices_from_panel_row(panel_row)
    center = vertices.mean(axis=0)
    row = {
        "CenterX": float(center[0]),
        "CenterY": float(center[1]),
        "CenterTIME": float(center[2]),
        "PatchLength": float(panel_row.get("PanelLength", panel_row.get("PatchLength", 1.0))),
        "PatchHeight": float(panel_row.get("PanelHeight", panel_row.get("PatchHeight", 1.0))),
        "PatchArea": float(panel_row.get("PanelArea", compute_quad_area(vertices))),
        "Azimuth": float(panel_row.get("StrikeDeg", 0.0)),
        "Dip": float(panel_row.get("DipDeg", 45.0)),
        "NormalX": float(panel_row.get("NormalX", 0.0)),
        "NormalY": float(panel_row.get("NormalY", 0.0)),
        "NormalZ": float(panel_row.get("NormalZ", 1.0)),
        "BBoxXMin": float(vertices[:, 0].min()),
        "BBoxXMax": float(vertices[:, 0].max()),
        "BBoxYMin": float(vertices[:, 1].min()),
        "BBoxYMax": float(vertices[:, 1].max()),
        "BBoxZMin": float(vertices[:, 2].min()),
        "BBoxZMax": float(vertices[:, 2].max()),
    }
    for vertex_idx in range(1, 5):
        row[f"V{vertex_idx}X"] = float(vertices[vertex_idx - 1, 0])
        row[f"V{vertex_idx}Y"] = float(vertices[vertex_idx - 1, 1])
        row[f"V{vertex_idx}Z"] = float(vertices[vertex_idx - 1, 2])
    if extra:
        row.update(extra)
    return row


def fit_plane_from_points(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0) if len(pts) else np.zeros(3, dtype=float)
    normal = best_fit_normal(pts)
    normal, strike_vec, dip_vec = build_plane_axes_from_normal(normal)
    rel = pts - center if len(pts) else np.zeros((0, 3), dtype=float)
    strike_proj = rel @ strike_vec if len(pts) else np.zeros(0, dtype=float)
    dip_proj = rel @ dip_vec if len(pts) else np.zeros(0, dtype=float)
    strike_min = float(strike_proj.min()) if len(strike_proj) else -0.5
    strike_max = float(strike_proj.max()) if len(strike_proj) else 0.5
    dip_min = float(dip_proj.min()) if len(dip_proj) else -0.5
    dip_max = float(dip_proj.max()) if len(dip_proj) else 0.5
    panel_vertices = np.asarray(
        [
            center + strike_min * strike_vec + dip_min * dip_vec,
            center + strike_max * strike_vec + dip_min * dip_vec,
            center + strike_max * strike_vec + dip_max * dip_vec,
            center + strike_min * strike_vec + dip_max * dip_vec,
        ],
        dtype=float,
    )
    strike_deg, dip_deg = strike_dip_from_normal(normal)
    return {
        "center": center,
        "normal": normal,
        "strike_vec": strike_vec,
        "dip_vec": dip_vec,
        "strike_deg": float(strike_deg),
        "dip_deg": float(dip_deg),
        "strike_min": float(strike_min),
        "strike_max": float(strike_max),
        "dip_min": float(dip_min),
        "dip_max": float(dip_max),
        "panel_vertices": panel_vertices,
        "panel_length": float(max(strike_max - strike_min, 1e-6)),
        "panel_height": float(max(dip_max - dip_min, 1e-6)),
    }


def load_fault_patch_polydata(vtp_path: Path) -> dict[str, Any]:
    poly = pv.read(str(vtp_path))
    if not isinstance(poly, pv.PolyData):
        poly = poly.extract_surface().clean().triangulate()
    points = np.asarray(poly.points, dtype=float)
    plane = fit_plane_from_points(points)
    return {
        "points": points,
        "faces": np.asarray(poly.faces, dtype=int) if getattr(poly, "faces", None) is not None else np.zeros(0, dtype=int),
        "center": plane["center"],
        "area": float(poly.area),
        "bounds": tuple(float(value) for value in poly.bounds),
        "normal": plane["normal"],
        "strike_vec": plane["strike_vec"],
        "dip_vec": plane["dip_vec"],
        "strike_deg": plane["strike_deg"],
        "dip_deg": plane["dip_deg"],
    }


def read_regional_vtk_to_df(input_vtk: Path) -> tuple[pd.DataFrame, dict[str, str], str]:
    payload = read_legacy_vtk_polygons(Path(input_vtk))
    points = np.asarray(payload["points"], dtype=float)
    polygons = payload["polygons"]
    n_cells = int(len(polygons))
    df = pd.DataFrame(index=np.arange(n_cells))
    for name, values in payload["cell_data"].items():
        df[name] = values

    decode_maps: dict[str, dict[int, str]] = {
        "ReliabilityLevel": {2: "high", 1: "medium", 0: "low"},
        "ConnectionType": {0: "original", 1: "boundary_matched", 2: "supplemented", 3: "boundary_connected"},
        "ScaleClass": {2: "macro_core", 1: "meso_link", 0: "micro_bg"},
        "AggregationMode": {0: "original", 1: "macro_agg", 2: "meso_agg", 3: "boundary_bridge", 4: "seam_fill"},
        "PatchOriginCode": PATCH_ORIGIN_CODE_TO_TEXT,
        "FaultActionCode": FAULT_ACTION_CODE_TO_TEXT,
    }
    for col, mapping in decode_maps.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            if col == "PatchOriginCode":
                df["PatchOriginText"] = numeric.astype(int).map(mapping).fillna("original")
            elif col == "FaultActionCode":
                df["FaultActionText"] = numeric.astype(int).map(mapping).fillna("keep")
            else:
                df[col] = numeric.astype(int).map(mapping).fillna(df[col])

    centers = np.zeros((n_cells, 3), dtype=float)
    for polygon_idx, polygon in enumerate(polygons):
        polygon_points = points[np.asarray(polygon, dtype=int)]
        centers[polygon_idx] = polygon_points.mean(axis=0)
    df["CenterX"] = centers[:, 0]
    df["CenterY"] = centers[:, 1]
    df["CenterTIME"] = centers[:, 2]

    for vertex_idx in range(1, 5):
        vx_list: list[float] = []
        vy_list: list[float] = []
        vz_list: list[float] = []
        for polygon_idx, polygon in enumerate(polygons):
            if vertex_idx - 1 < len(polygon):
                point = points[int(polygon[vertex_idx - 1])]
            else:
                point = centers[polygon_idx]
            vx_list.append(float(point[0]))
            vy_list.append(float(point[1]))
            vz_list.append(float(point[2]))
        df[f"V{vertex_idx}X"] = vx_list
        df[f"V{vertex_idx}Y"] = vy_list
        df[f"V{vertex_idx}Z"] = vz_list

    default_values = [
        ("Azimuth", 0.0),
        ("Dip", 45.0),
        ("Confidence", 0.5),
        ("PatchLength", 10.0),
        ("PatchHeight", 10.0),
        ("PatchOriginCode", 0),
        ("FaultActionCode", 0),
    ]
    for col, default in default_values:
        if col not in df.columns:
            df[col] = default
    if "PatchArea" not in df.columns:
        df["PatchArea"] = (
            pd.to_numeric(df["PatchLength"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["PatchHeight"], errors="coerce").fillna(0.0)
        )
    if "PatchOriginText" not in df.columns:
        df["PatchOriginText"] = pd.to_numeric(df["PatchOriginCode"], errors="coerce").fillna(0).astype(int).map(PATCH_ORIGIN_CODE_TO_TEXT)
    if "FaultActionText" not in df.columns:
        df["FaultActionText"] = pd.to_numeric(df["FaultActionCode"], errors="coerce").fillna(0).astype(int).map(FAULT_ACTION_CODE_TO_TEXT)
    if "AggregationMode" not in df.columns:
        df["AggregationMode"] = "original"
    if "ParentPatchCount" not in df.columns:
        df["ParentPatchCount"] = 1
    return df, dict(payload.get("scalar_types", {})), str(payload.get("title", "DFN"))


def write_df_to_regional_vtk(
    df: pd.DataFrame,
    title: str,
    output_vtk: Path,
    scalar_types: dict[str, str] | None = None,
) -> None:
    new_points: list[list[float]] = []
    new_polygons: list[list[int]] = []
    skip_cols = {"CenterX", "CenterY", "CenterTIME"}
    skip_cols |= {f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")}
    skip_cols |= TEXT_EXPORT_COLUMNS
    data_cols = [col for col in df.columns if col not in skip_cols]

    cell_data_lists: dict[str, list[Any]] = {col: [] for col in data_cols}
    for _, row in df.iterrows():
        start_idx = len(new_points)
        for vertex_idx in range(1, 5):
            new_points.append(
                [
                    float(row[f"V{vertex_idx}X"]),
                    float(row[f"V{vertex_idx}Y"]),
                    float(row[f"V{vertex_idx}Z"]),
                ]
            )
        new_polygons.append(list(range(start_idx, start_idx + 4)))
        for col in data_cols:
            cell_data_lists[col].append(row[col])

    out_points = np.asarray(new_points, dtype=float)
    out_cell_data: dict[str, np.ndarray] = {}
    out_scalar_types: dict[str, str] = dict(scalar_types) if scalar_types else {}
    int_cols = {
        "FractureSet",
        "CorridorID",
        "CorridorSupport",
        "NeighborCount",
        "IsSupplemented",
        "ConnectionID",
        "ParentPatchCount",
        "AlongNeighborCount",
        "FaultPanelID",
        "NearestFaultPanelID",
        "PatchOriginCode",
        "FaultActionCode",
        "SourcePatchCount",
        "SourceUnitCount",
    }
    str_maps: dict[str, dict[str, int]] = {
        "ReliabilityLevel": {"high": 2, "medium": 1, "low": 0},
        "ConnectionType": {"original": 0, "boundary_matched": 1, "supplemented": 2, "boundary_connected": 3},
        "ScaleClass": {"micro_bg": 0, "meso_link": 1, "macro_core": 2},
        "AggregationMode": {"original": 0, "macro_agg": 1, "meso_agg": 2, "boundary_bridge": 3, "seam_fill": 4},
    }

    for col, values in cell_data_lists.items():
        if col in str_maps:
            mapping = str_maps[col]
            out_cell_data[col] = np.asarray([mapping.get(str(value), -1) for value in values], dtype=int)
            out_scalar_types[col] = "int"
            continue
        try:
            arr = np.asarray(values)
            if col in int_cols:
                numeric = pd.to_numeric(pd.Series(arr), errors="coerce").fillna(-1).astype(int)
                out_cell_data[col] = numeric.to_numpy()
                out_scalar_types[col] = "int"
            else:
                numeric = pd.to_numeric(pd.Series(arr), errors="coerce").fillna(0.0).astype(float)
                out_cell_data[col] = numeric.to_numpy()
                out_scalar_types[col] = "float"
        except Exception:
            continue

    write_legacy_vtk_polygons_preserve_patch_area(
        path=Path(output_vtk),
        title=str(title),
        points=out_points,
        polygons=new_polygons,
        cell_data=out_cell_data,
        scalar_types=out_scalar_types,
        recompute_patch_area=True,
    )


def compute_polygon_areas_for_vtk(points: np.ndarray, polygons: list[list[int]]) -> np.ndarray:
    areas = np.zeros(len(polygons), dtype=float)
    for polygon_idx, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        polygon_points = np.asarray([points[int(point_idx)] for point_idx in polygon], dtype=float)
        origin = polygon_points[0]
        area = 0.0
        for vertex_idx in range(1, len(polygon_points) - 1):
            vec1 = polygon_points[vertex_idx] - origin
            vec2 = polygon_points[vertex_idx + 1] - origin
            area += 0.5 * float(np.linalg.norm(np.cross(vec1, vec2)))
        areas[polygon_idx] = area
    return areas


def write_vtk_scalar_block(lines: list[str], name: str, values: np.ndarray, scalar_type: str) -> None:
    dtype = str(scalar_type).lower()
    if dtype == "int":
        formatter = lambda value: str(int(value))
    else:
        dtype = "float"
        formatter = lambda value: f"{float(value):.6f}"
    lines.append(f"SCALARS {name} {dtype} 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(formatter(value) for value in np.asarray(values).tolist())


def write_legacy_vtk_polygons_preserve_patch_area(
    path: Path,
    title: str,
    points: np.ndarray,
    polygons: list[list[int]],
    cell_data: dict[str, np.ndarray],
    scalar_types: dict[str, str],
    recompute_patch_area: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local_points = np.asarray(points, dtype=float)
    local_cell_data = {name: np.asarray(values) for name, values in cell_data.items()}
    local_scalar_types = dict(scalar_types)
    if polygons and (bool(recompute_patch_area) or "PatchArea" not in local_cell_data):
        local_cell_data["PatchArea"] = compute_polygon_areas_for_vtk(local_points, polygons)
        local_scalar_types["PatchArea"] = "float"

    total_polygon_size = sum(len(polygon) + 1 for polygon in polygons)
    lines = [
        "# vtk DataFile Version 3.0",
        str(title),
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(local_points)} float",
    ]
    lines.extend(
        f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f}"
        for point in local_points
    )
    lines.append(f"POLYGONS {len(polygons)} {total_polygon_size}")
    lines.extend(
        f"{len(polygon)} {' '.join(str(int(idx)) for idx in polygon)}"
        for polygon in polygons
    )
    if local_cell_data:
        lines.append(f"CELL_DATA {len(polygons)}")
        for name, values in local_cell_data.items():
            write_vtk_scalar_block(lines, name, values, local_scalar_types.get(name, "float"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def predict_fault_time_at_xy(panel_row: pd.Series, x: float, y: float) -> float:
    cx = float(panel_row["CenterX"])
    cy = float(panel_row["CenterY"])
    cz = float(panel_row["CenterTIME"])
    nx = float(panel_row.get("NormalX", 0.0))
    ny = float(panel_row.get("NormalY", 0.0))
    nz = float(panel_row.get("NormalZ", 1.0))
    if abs(nz) <= 1e-8:
        return cz
    return float(cz - (nx * (float(x) - cx) + ny * (float(y) - cy)) / nz)


def panel_contains_xy(panel_row: pd.Series, x: float, y: float, buffer_m: float) -> bool:
    return (
        float(panel_row["BBoxXMin"]) - float(buffer_m) <= float(x) <= float(panel_row["BBoxXMax"]) + float(buffer_m)
        and float(panel_row["BBoxYMin"]) - float(buffer_m) <= float(y) <= float(panel_row["BBoxYMax"]) + float(buffer_m)
    )


def scale_patch_row_geometry(row: pd.Series, area_ratio: float) -> pd.Series:
    ratio = float(np.clip(area_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return row
    scale = math.sqrt(ratio)
    center = np.array([float(row["CenterX"]), float(row["CenterY"]), float(row["CenterTIME"])], dtype=float)
    updated = row.copy()
    vertices = []
    for vertex_idx in range(1, 5):
        point = np.array(
            [
                float(row[f"V{vertex_idx}X"]),
                float(row[f"V{vertex_idx}Y"]),
                float(row[f"V{vertex_idx}Z"]),
            ],
            dtype=float,
        )
        scaled_point = center + scale * (point - center)
        vertices.append(scaled_point)
        updated[f"V{vertex_idx}X"] = float(scaled_point[0])
        updated[f"V{vertex_idx}Y"] = float(scaled_point[1])
        updated[f"V{vertex_idx}Z"] = float(scaled_point[2])
    vertices_arr = np.asarray(vertices, dtype=float)
    updated["PatchLength"] = float(row.get("PatchLength", 0.0)) * scale
    updated["PatchHeight"] = float(row.get("PatchHeight", 0.0)) * scale
    updated["PatchArea"] = float(compute_quad_area(vertices_arr))
    updated["BBoxXMin"] = float(vertices_arr[:, 0].min())
    updated["BBoxXMax"] = float(vertices_arr[:, 0].max())
    updated["BBoxYMin"] = float(vertices_arr[:, 1].min())
    updated["BBoxYMax"] = float(vertices_arr[:, 1].max())
    updated["BBoxZMin"] = float(vertices_arr[:, 2].min())
    updated["BBoxZMax"] = float(vertices_arr[:, 2].max())
    return updated


def build_default_postprocess_kwargs(
    phase: int = 2,
    enable_boundary_connect: bool = True,
    enable_seam_fill: bool = True,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_fracture_sets": 6,
        "n_fracture_sets": None,
        "neighbor_radius_xy": 100.0,
        "neighbor_radius_z": 15.0,
        "isolation_min_neighbors": 2,
        "confidence_floor": 0.3,
        "boundary_tol_xy": 25.0,
    }
    if int(phase) == 2:
        kwargs.update(
            {
                "smooth_bandwidth_xy": 150.0,
                "smooth_bandwidth_z": 20.0,
                "smooth_blend_alpha": 0.4,
                "corridor_search_radius": 200.0,
                "corridor_min_patches": 5,
                "enable_supplement": True,
                "min_pair_confidence": 0.5,
                "max_supplement_length": 80.0,
                "enable_aggregation": True,
                "agg_cluster_radius": 20.0,
                "agg_min_patches": 3,
                "agg_azimuth_tol": 25.0,
                "agg_dip_tol": 15.0,
                "agg_max_length": 80.0,
                "agg_max_height": 40.0,
                "scale_major_radius": 160.0,
                "scale_minor_radius": 25.0,
                "scale_z_radius": 20.0,
                "macro_score_quantile": 0.85,
                "macro_max_fraction": 0.10,
                "meso_score_quantile": 0.45,
                "macro_min_span": 80.0,
                "meso_min_span": 30.0,
                "macro_min_neighbors": 6,
                "meso_min_neighbors": 3,
                "macro_agg_major": 140.0,
                "macro_agg_minor": 24.0,
                "macro_agg_max_length": 320.0,
                "macro_agg_max_height": 90.0,
                "enable_elongation": True,
                "elongation_max_stretch": 2.5,
                "elongation_gap_fill": 0.7,
                "elongation_max_neighbor_dist": 120.0,
                "macro_elongation_stretch": 4.0,
                "macro_elongation_range": 220.0,
                "jitter_sigma_xy": 8.0,
                "jitter_along_strike_factor": 1.5,
                "jitter_seed": 42,
            }
        )
    if enable_boundary_connect:
        kwargs.update(
            {
                "enable_boundary_connect": True,
                "boundary_strip_width": 40.0,
                "connect_max_gap": 60.0,
                "bc_macro_gap": 180.0,
                "bc_minor_limit": 25.0,
                "bc_z_gap": 20.0,
                "bc_azimuth_tol_deg": 30.0,
                "bc_dip_tol_deg": 20.0,
                "bc_min_score_threshold": 0.35,
                "bc_corridor_bonus": 0.15,
                "bc_require_corridor": False,
                "bc_perp_ratio": 0.8,
                "bc_enable_supplement": True,
                "bc_max_stretch_ratio": 2.5,
            }
        )
    if enable_seam_fill:
        kwargs.update(
            {
                "enable_seam_fill": True,
                "seam_half_width": 50.0,
                "seam_interior_depth": 80.0,
                "seam_fill_fraction": 1.0,
                "seam_confidence_decay": 0.65,
                "seam_seed": 42,
                "seam_blend_half_width": 60.0,
                "seam_blend_strength": 0.5,
                "seam_min_shared_ratio": 0.15,
                "seam_high_rel_boost": 1.5,
            }
        )
    return kwargs
