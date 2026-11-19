"""Export layer-filtered multiscale fracture patches as standalone VTK files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv


SCALE_CODES = {"small": 1, "medium": 2, "large": 3}
LAYER_CODES = {"沙三段": 1, "沙四段": 2, "上部复合层": 1, "太古界风化壳": 2}
VERTEX_COLUMNS = [f"V{i}{axis}" for i in range(1, 5) for axis in "XYZ"]


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def export_layer_scale_vtk(
    input_csv: Path,
    output_vtk: Path,
    *,
    layer_group: str,
    layer_code: int,
) -> dict[str, Any]:
    frame = pd.read_csv(input_csv, low_memory=False)
    scale = frame["FractureScale"].astype(str).str.strip().str.lower()
    layer = frame["LayerGroup"].astype(str).str.strip()
    selected = frame[
        layer.eq(layer_group) & scale.isin(tuple(SCALE_CODES))
    ].reset_index(drop=True)
    if selected.empty:
        raise RuntimeError(f"no multiscale patches for layer {layer_group!r}")
    missing = [column for column in VERTEX_COLUMNS if column not in selected.columns]
    if missing:
        raise ValueError(f"input CSV missing vertex columns: {missing}")

    vertices = selected[VERTEX_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64).reshape(len(selected), 4, 3)
    if not np.isfinite(vertices).all():
        raise ValueError(f"non-finite patch vertices in layer {layer_group!r}")

    points = vertices.reshape(-1, 3)
    faces = np.empty((len(selected), 5), dtype=np.int64)
    faces[:, 0] = 4
    faces[:, 1:] = np.arange(len(selected) * 4, dtype=np.int64).reshape(-1, 4)
    mesh = pv.PolyData(points, faces.reshape(-1))

    selected_scale = selected["FractureScale"].astype(str).str.strip().str.lower()
    scale_codes = selected_scale.map(SCALE_CODES).astype(np.int32).to_numpy()
    mesh.cell_data["PatchIndex"] = np.arange(1, len(selected) + 1, dtype=np.int32)
    mesh.cell_data["LayerCode"] = np.full(len(selected), int(layer_code), dtype=np.int32)
    mesh.cell_data["FractureScaleCode"] = scale_codes
    mesh.cell_data["GeometryGroupCode"] = np.ones(len(selected), dtype=np.int32)
    mesh.cell_data["IsOriginalFault"] = np.zeros(len(selected), dtype=np.int32)
    for source, target, default in (
        ("SourceDensity", "SourceDensity", np.nan),
        ("Confidence", "Confidence", np.nan),
        ("DipAzimuthDeg", "DipAzimuthDeg", np.nan),
        ("DipDeg", "DipDeg", np.nan),
        ("LengthM", "LengthM", np.nan),
        ("HeightTimeMs", "HeightTimeMs", np.nan),
    ):
        mesh.cell_data[target] = _numeric(selected, source, default)

    output_vtk.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(output_vtk, binary=True)
    counts = {
        str(key): int(value)
        for key, value in selected_scale.value_counts().sort_index().items()
    }
    return {
        "input_csv": str(input_csv),
        "output_vtk": str(output_vtk),
        "layer_group": layer_group,
        "cell_count": int(mesh.n_cells),
        "point_count": int(mesh.n_points),
        "scale_counts": counts,
    }


def print_export_summary(summary: dict[str, Any]) -> None:
    print(
        f"[layer-scale-vtk] {summary['layer_group']} "
        f"cells={summary['cell_count']} scales={summary['scale_counts']} "
        f"output={summary['output_vtk']}",
        flush=True,
    )
