# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from build_all_area_section_visualization import SurfaceSectionCurve


HORIZON_CODES = ("T4", "T5", "T6", "T7")
DEFAULT_HORIZON_TRACE_TABLE = (
    Path(__file__).resolve().parents[1]
    / "common/horizon_trace_table/output/formal_horizon_trace_table_v2/horizon_trace_table.npy"
)


def resolve_horizon_trace_table(config: dict[str, Any]) -> Path:
    value = config.get("horizon_trace_table_path", DEFAULT_HORIZON_TRACE_TABLE)
    path = Path(str(value)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"horizon trace table not found: {path}")
    return path


def load_horizon_trace_table(path: Path) -> np.ndarray:
    table = np.load(path, mmap_mode="r")
    required = {"TraceIdx", *HORIZON_CODES}
    fields = set(table.dtype.names or ())
    if not required.issubset(fields):
        raise ValueError(f"horizon trace table missing fields: {sorted(required - fields)}")
    if len(table) == 0 or int(table["TraceIdx"][0]) != 0 or int(table["TraceIdx"][-1]) != len(table) - 1:
        raise ValueError("horizon trace table does not satisfy row index equals TraceIdx")
    return table


def section_trace_ids(
    projection: str,
    h_values: np.ndarray,
    times: np.ndarray,
    well_time: np.ndarray,
    well_x: np.ndarray,
    well_y: np.ndarray,
    trace_tree,
    trace_ids: np.ndarray,
) -> np.ndarray:
    if projection == "XZ":
        curve_coordinate = np.interp(times, well_time, well_y)
        points = np.column_stack([h_values, curve_coordinate])
    elif projection == "YZ":
        curve_coordinate = np.interp(times, well_time, well_x)
        points = np.column_stack([curve_coordinate, h_values])
    else:
        raise ValueError(f"unsupported projection: {projection}")
    tree_positions = np.asarray(trace_tree.query(points, k=1)[1], dtype=np.int64).reshape(-1)
    return trace_ids[tree_positions].astype(np.int64, copy=False)


def solve_horizon_curve(
    projection: str,
    code: str,
    h_values: np.ndarray,
    well_df,
    trace_tree,
    trace_ids: np.ndarray,
    horizon_table: np.ndarray,
    iteration_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    well_time = well_df["TIME"].to_numpy(dtype=np.float64)
    well_x = well_df["X"].to_numpy(dtype=np.float64)
    well_y = well_df["Y"].to_numpy(dtype=np.float64)
    source_values = np.asarray(horizon_table[code][trace_ids], dtype=np.float64)
    initial_time = float(np.nanmedian(source_values))
    current = np.full(len(h_values), initial_time, dtype=np.float64)
    candidates = [current.copy()]
    for _ in range(max(1, iteration_count)):
        mapped_trace_ids = section_trace_ids(
            projection,
            h_values,
            current,
            well_time,
            well_x,
            well_y,
            trace_tree,
            trace_ids,
        )
        current = np.asarray(horizon_table[code][mapped_trace_ids], dtype=np.float64)
        candidates.append(current.copy())

    candidate_matrix = np.vstack(candidates)
    residual_matrix = np.empty_like(candidate_matrix)
    mapped_id_matrix = np.empty(candidate_matrix.shape, dtype=np.int64)
    for candidate_index, candidate_times in enumerate(candidate_matrix):
        mapped_trace_ids = section_trace_ids(
            projection,
            h_values,
            candidate_times,
            well_time,
            well_x,
            well_y,
            trace_tree,
            trace_ids,
        )
        mapped_horizon = np.asarray(horizon_table[code][mapped_trace_ids], dtype=np.float64)
        mapped_id_matrix[candidate_index] = mapped_trace_ids
        residual_matrix[candidate_index] = np.abs(candidate_times - mapped_horizon)

    best_candidate_index = np.argmin(residual_matrix, axis=0)
    columns = np.arange(len(h_values), dtype=np.int64)
    best_times = candidate_matrix[best_candidate_index, columns]
    best_trace_ids = mapped_id_matrix[best_candidate_index, columns]
    best_residual = residual_matrix[best_candidate_index, columns]
    return best_times, {
        "point_count": int(len(h_values)),
        "iteration_count": int(max(1, iteration_count)),
        "unique_source_trace_count": int(len(np.unique(best_trace_ids))),
        "residual_zero_count": int(np.sum(best_residual <= 1.0e-6)),
        "residual_max_ms": float(np.max(best_residual)),
        "residual_q99_ms": float(np.quantile(best_residual, 0.99)),
    }


def build_trace_horizon_section_curves(
    horizon_table_path: Path,
    well_df,
    trace_tree,
    trace_ids: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    iteration_count: int = 12,
) -> tuple[list[SurfaceSectionCurve], dict[str, Any]]:
    table = load_horizon_trace_table(horizon_table_path)
    if int(np.max(trace_ids)) >= len(table):
        raise ValueError("section trace ids exceed horizon trace table row count")
    curves: list[SurfaceSectionCurve] = []
    curve_stats: dict[str, Any] = {}
    values_by_projection: dict[str, dict[str, np.ndarray]] = {"XZ": {}, "YZ": {}}
    for projection, h_values in [("XZ", x_values), ("YZ", y_values)]:
        curve_stats[projection] = {}
        for code in HORIZON_CODES:
            times, stats = solve_horizon_curve(
                projection,
                code,
                np.asarray(h_values, dtype=np.float64),
                well_df,
                trace_tree,
                trace_ids,
                table,
                iteration_count,
            )
            curves.append(
                SurfaceSectionCurve(
                    projection=projection,
                    surface=code,
                    h=np.asarray(h_values, dtype=np.float64).copy(),
                    z=times,
                )
            )
            values_by_projection[projection][code] = times
            curve_stats[projection][code] = stats

        matrix = np.vstack([values_by_projection[projection][code] for code in HORIZON_CODES])
        pair_reversals = {
            f"{upper}>={lower}": int(np.sum(matrix[index] >= matrix[index + 1]))
            for index, (upper, lower) in enumerate(zip(HORIZON_CODES[:-1], HORIZON_CODES[1:]))
        }
        curve_stats[projection]["ordering"] = {
            "pair_reversal_counts": pair_reversals,
            "any_reversal_count": int(np.sum(np.any(matrix[:-1] >= matrix[1:], axis=0))),
            "fully_ordered_count": int(np.sum(np.all(matrix[:-1] < matrix[1:], axis=0))),
        }
    summary = {
        "horizon_trace_table_path": str(horizon_table_path),
        "method": "per-lateral-trace well-controlled curved-surface intersection",
        "xy_projection_logic": {
            "XZ": "solve T = HorizonTraceTable[nearest TraceIdx(x, Ywell(T))] for every X coordinate",
            "YZ": "solve T = HorizonTraceTable[nearest TraceIdx(Xwell(T), y)] for every Y coordinate",
        },
        "curve_stats": curve_stats,
    }
    return curves, summary


__all__ = [
    "HORIZON_CODES",
    "build_trace_horizon_section_curves",
    "load_horizon_trace_table",
    "resolve_horizon_trace_table",
]
