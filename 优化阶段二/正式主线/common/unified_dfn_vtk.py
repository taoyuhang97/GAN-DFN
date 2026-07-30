from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


PREDICTED_GEOMETRY_GROUP_CODE = 1
ORIGINAL_FAULT_GEOMETRY_GROUP_CODE = 2


def _as_polydata(path: Path) -> pv.PolyData:
    loaded = pv.read(path)
    mesh = loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface(algorithm="dataset_surface")
    return mesh.copy(deep=True)


def _original_fault_polydata(path: Path) -> pv.PolyData:
    mesh = _as_polydata(path)
    if "GeometryGroupCode" in mesh.cell_data:
        values = np.asarray(mesh.cell_data["GeometryGroupCode"], dtype=np.int32)
        indices = np.where(values == ORIGINAL_FAULT_GEOMETRY_GROUP_CODE)[0]
        if not len(indices):
            raise ValueError(f"unified DFN VTK contains no original-fault cells: {path}")
        mesh = mesh.extract_cells(indices).extract_surface(algorithm="dataset_surface")
    return mesh.triangulate()


def _numeric_array(values: Any, count: int, *, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != count or array.dtype.kind not in "biuf":
        raise ValueError("unified DFN VTK supports only one-dimensional numeric cell arrays")
    return array.astype(dtype or array.dtype, copy=False)


def _median_positive(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    return float(np.median(finite)) if finite.size else 1.0


def _write_ascii_legacy_polydata(path: Path, mesh: pv.PolyData) -> None:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    polygons: list[np.ndarray] = []
    cursor = 0
    while cursor < len(faces):
        size = int(faces[cursor])
        indices = faces[cursor + 1 : cursor + 1 + size]
        if size < 3 or len(indices) != size:
            raise ValueError("invalid polygon connectivity in unified DFN mesh")
        polygons.append(indices)
        cursor += size + 1
    if len(polygons) != int(mesh.n_cells):
        raise ValueError("unified DFN mesh contains non-polygon cells")

    lines = [
        "# vtk DataFile Version 3.0",
        "unified_dfn_raw_time",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {mesh.n_points} double",
    ]
    lines.extend(
        f"{float(point[0]):.17g} {float(point[1]):.17g} {float(point[2]):.17g}"
        for point in np.asarray(mesh.points, dtype=float)
    )
    lines.append(f"POLYGONS {len(polygons)} {sum(len(poly) + 1 for poly in polygons)}")
    lines.extend(f"{len(poly)} {' '.join(str(int(index)) for index in poly)}" for poly in polygons)
    lines.append(f"CELL_DATA {mesh.n_cells}")
    for name in sorted(mesh.cell_data.keys()):
        values = _numeric_array(mesh.cell_data[name], int(mesh.n_cells))
        if values.dtype.kind in "biu":
            lines.append(f"SCALARS {name} int 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(str(int(value)) for value in values)
        elif values.dtype.kind == "f":
            lines.append(f"SCALARS {name} double 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(f"{float(value):.17g}" for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _align_numeric_cell_arrays(predicted: pv.PolyData, original: pv.PolyData, render_area_m2: float) -> None:
    for mesh in (predicted, original):
        for name in list(mesh.cell_data.keys()):
            if str(name).lower().startswith("vtkoriginal"):
                mesh.cell_data.pop(name, None)
    predicted_count = int(predicted.n_cells)
    original_count = int(original.n_cells)
    if "PatchAreaM2" not in predicted.cell_data:
        predicted.cell_data["PatchAreaM2"] = np.full(predicted_count, render_area_m2, dtype=np.float64)
    if "PatchArea" not in predicted.cell_data:
        predicted.cell_data["PatchArea"] = np.asarray(predicted.cell_data["PatchAreaM2"], dtype=np.float64)
    names = sorted(set(predicted.cell_data.keys()) | set(original.cell_data.keys()))
    for name in names:
        predicted_values = predicted.cell_data.get(name)
        original_values = original.cell_data.get(name)
        source = predicted_values if predicted_values is not None else original_values
        if source is None or np.asarray(source).dtype.kind not in "biuf" or np.asarray(source).ndim != 1:
            predicted.cell_data.pop(name, None)
            original.cell_data.pop(name, None)
            continue
        dtype = np.asarray(source).dtype
        if predicted_values is None:
            fill = -1 if dtype.kind in "iu" else 0.0
            predicted.cell_data[name] = np.full(predicted_count, fill, dtype=dtype)
        else:
            values = _numeric_array(predicted_values, predicted_count, dtype=dtype)
            predicted.cell_data[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if original_values is None:
            fill = -1 if dtype.kind in "iu" else 0.0
            original.cell_data[name] = np.full(original_count, fill, dtype=dtype)
        else:
            values = _numeric_array(original_values, original_count, dtype=dtype)
            original.cell_data[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    predicted.cell_data["GeometryGroupCode"] = np.full(
        predicted_count, PREDICTED_GEOMETRY_GROUP_CODE, dtype=np.int32
    )
    original.cell_data["GeometryGroupCode"] = np.full(
        original_count, ORIGINAL_FAULT_GEOMETRY_GROUP_CODE, dtype=np.int32
    )
    predicted.cell_data["OriginalFaultFlag"] = np.zeros(predicted_count, dtype=np.int32)
    original.cell_data["OriginalFaultFlag"] = np.ones(original_count, dtype=np.int32)
    predicted.cell_data["PredictedPatchOrdinal"] = np.arange(predicted_count, dtype=np.int32)
    original.cell_data["PredictedPatchOrdinal"] = np.full(original_count, -1, dtype=np.int32)
    predicted.cell_data["PatchAreaM2"] = np.asarray(
        predicted.cell_data.get("PatchAreaM2", np.full(predicted_count, render_area_m2)), dtype=np.float64
    )
    original.cell_data["PatchAreaM2"] = np.full(original_count, render_area_m2, dtype=np.float64)
    predicted.cell_data["PatchArea"] = np.asarray(
        predicted.cell_data.get("PatchArea", predicted.cell_data["PatchAreaM2"]), dtype=np.float64
    )
    original.cell_data["PatchArea"] = np.full(original_count, render_area_m2, dtype=np.float64)


def write_unified_dfn_vtk(
    output_path: Path,
    predicted_vtk: Path,
    original_fault_vtk: Path,
) -> dict[str, Any]:
    predicted = _as_polydata(predicted_vtk)
    original = _original_fault_polydata(original_fault_vtk)
    render_area_m2 = _median_positive(
        np.asarray(predicted.cell_data.get("PatchAreaM2", np.ones(predicted.n_cells)), dtype=float)
    )
    predicted.clear_point_data()
    original.clear_point_data()
    _align_numeric_cell_arrays(predicted, original, render_area_m2)
    combined = predicted.append_polydata(original)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ascii_legacy_polydata(output_path, combined)
    return {
        "output_vtk": str(output_path),
        "predicted_cell_count": int(predicted.n_cells),
        "original_fault_cell_count": int(original.n_cells),
        "combined_cell_count": int(combined.n_cells),
        "predicted_point_count": int(predicted.n_points),
        "original_fault_point_count": int(original.n_points),
        "combined_point_count": int(combined.n_points),
        "original_fault_cell_start": int(predicted.n_cells),
        "original_fault_render_area_m2": float(render_area_m2),
        "original_fault_render_area_policy": "predicted_patch_area_median_not_physical_triangle_area",
        "geometry_group_codes": {
            "predicted_patches": PREDICTED_GEOMETRY_GROUP_CODE,
            "original_fault_triangles": ORIGINAL_FAULT_GEOMETRY_GROUP_CODE,
        },
    }


def extract_geometry_group(path: Path, group_code: int) -> pv.PolyData:
    mesh = _as_polydata(path)
    if "GeometryGroupCode" not in mesh.cell_data:
        raise ValueError(f"GeometryGroupCode missing from unified DFN VTK: {path}")
    values = np.asarray(mesh.cell_data["GeometryGroupCode"], dtype=np.int32)
    indices = np.where(values == int(group_code))[0]
    if not len(indices):
        return pv.PolyData()
    return mesh.extract_cells(indices).extract_surface(algorithm="dataset_surface")


def geometry_group_counts(path: Path) -> dict[str, int]:
    mesh = _as_polydata(path)
    if "GeometryGroupCode" not in mesh.cell_data:
        raise ValueError(f"GeometryGroupCode missing from unified DFN VTK: {path}")
    values = np.asarray(mesh.cell_data["GeometryGroupCode"], dtype=np.int32)
    predicted_count = int(np.sum(values == PREDICTED_GEOMETRY_GROUP_CODE))
    original_count = int(np.sum(values == ORIGINAL_FAULT_GEOMETRY_GROUP_CODE))
    if predicted_count + original_count != int(mesh.n_cells):
        raise ValueError(f"unexpected geometry group code in unified DFN VTK: {path}")
    if predicted_count and not np.all(values[:predicted_count] == PREDICTED_GEOMETRY_GROUP_CODE):
        raise ValueError(f"predicted cells are not the leading block in unified DFN VTK: {path}")
    if original_count and not np.all(values[predicted_count:] == ORIGINAL_FAULT_GEOMETRY_GROUP_CODE):
        raise ValueError(f"original-fault cells are not the trailing block in unified DFN VTK: {path}")
    return {
        "combined_cell_count": int(mesh.n_cells),
        "predicted_cell_count": predicted_count,
        "original_fault_cell_count": original_count,
        "original_fault_cell_start": predicted_count,
    }


def geometry_fingerprint(mesh: pv.PolyData) -> str:
    triangles = mesh.triangulate()
    faces = np.asarray(triangles.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    coordinates = np.asarray(triangles.points, dtype=np.float64)[faces]
    normalized_triangles = []
    for triangle in coordinates:
        order = np.lexsort((triangle[:, 2], triangle[:, 1], triangle[:, 0]))
        normalized_triangles.append(triangle[order])
    normalized = np.asarray(normalized_triangles, dtype=np.float64)
    if normalized.size:
        flattened = normalized.reshape(len(normalized), -1)
        order = np.lexsort(tuple(flattened[:, idx] for idx in range(flattened.shape[1] - 1, -1, -1)))
        normalized = normalized[order]
    digest = hashlib.sha256()
    digest.update(np.asarray(normalized.shape, dtype=np.int64).tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def unified_vtk_summary(path: Path) -> dict[str, Any]:
    mesh = _as_polydata(path)
    values = np.asarray(mesh.cell_data.get("GeometryGroupCode", []), dtype=np.int32)
    predicted = extract_geometry_group(path, PREDICTED_GEOMETRY_GROUP_CODE)
    original = extract_geometry_group(path, ORIGINAL_FAULT_GEOMETRY_GROUP_CODE)
    return {
        "path": str(path),
        "point_count": int(mesh.n_points),
        "cell_count": int(mesh.n_cells),
        "predicted_cell_count": int(np.sum(values == PREDICTED_GEOMETRY_GROUP_CODE)),
        "original_fault_cell_count": int(np.sum(values == ORIGINAL_FAULT_GEOMETRY_GROUP_CODE)),
        "original_fault_point_count": int(original.n_points),
        "original_fault_geometry_fingerprint": geometry_fingerprint(original),
        "predicted_geometry_nonempty": bool(predicted.n_cells > 0),
    }
