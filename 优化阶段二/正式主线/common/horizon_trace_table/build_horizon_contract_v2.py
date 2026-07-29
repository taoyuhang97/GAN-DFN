from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


TABLE_DTYPE = np.dtype(
    [
        ("TraceIdx", "<i4"),
        ("T4", "<f4"),
        ("T5", "<f4"),
        ("T6", "<f4"),
        ("T7", "<f4"),
        ("SurfaceOrderValid", "u1"),
        ("ShasanPresent", "u1"),
        ("ShasiPresent", "u1"),
        ("HorizonCorrectionCode", "u1"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the versioned T4-T7 horizon contract used by the 10 km v2 flow.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def required_path(config: dict[str, Any], key: str) -> Path:
    path = Path(str(config[key])).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def grid_layout(trace_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.sort(trace_df["X"].unique()).astype(np.float64)
    y_axis = np.sort(trace_df["Y"].unique()).astype(np.float64)
    ix = np.searchsorted(x_axis, trace_df["X"].to_numpy(dtype=np.float64))
    iy = np.searchsorted(y_axis, trace_df["Y"].to_numpy(dtype=np.float64))
    if len(x_axis) * len(y_axis) != len(trace_df):
        raise ValueError("trace headers do not form a complete Cartesian grid")
    trace_grid = np.full((len(y_axis), len(x_axis)), -1, dtype=np.int32)
    trace_grid[iy, ix] = trace_df["TraceIdx"].to_numpy(dtype=np.int32)
    if np.any(trace_grid < 0):
        raise ValueError("trace grid contains missing cells")
    return x_axis, y_axis, ix, iy


def axis_window_indices(samples: np.ndarray, top: np.ndarray, base: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.searchsorted(samples, top, side="left").astype(np.int32)
    stop = np.searchsorted(samples, base, side="right").astype(np.int32)
    start = np.clip(start, 0, len(samples))
    stop = np.clip(stop, 0, len(samples))
    invalid = (~present) | (~np.isfinite(top)) | (~np.isfinite(base)) | (base <= top) | (stop <= start)
    start[invalid] = -1
    stop[invalid] = -1
    return start, stop


def write_axis_contract(
    path: Path,
    trace_idx: np.ndarray,
    t4: np.ndarray,
    t6: np.ndarray,
    t7: np.ndarray,
    shasan_present: np.ndarray,
    shasi_present: np.ndarray,
    start_ms: float,
    stop_ms: float,
    interval_ms: float,
) -> dict[str, Any]:
    count = int(np.floor((stop_ms - start_ms) / interval_ms + 1.0e-9)) + 1
    samples = start_ms + np.arange(count, dtype=np.float64) * interval_ms
    shasan_start, shasan_stop = axis_window_indices(samples, t4, t6, shasan_present)
    shasi_start, shasi_stop = axis_window_indices(samples, t6, t7, shasi_present)
    np.savez_compressed(
        path,
        TraceIdx=trace_idx.astype(np.int32),
        samples=samples.astype(np.float32),
        T4=t4.astype(np.float32),
        T6=t6.astype(np.float32),
        T7=t7.astype(np.float32),
        ShasanPresent=shasan_present.astype(np.uint8),
        ShasiPresent=shasi_present.astype(np.uint8),
        ShasanStartIdx=shasan_start,
        ShasanStopIdxExclusive=shasan_stop,
        ShasiStartIdx=shasi_start,
        ShasiStopIdxExclusive=shasi_stop,
    )
    return {
        "path": str(path),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "sample_interval_ms": float(interval_ms),
        "sample_count": int(len(samples)),
        "shasan_trace_count": int(np.count_nonzero(shasan_start >= 0)),
        "shasi_trace_count": int(np.count_nonzero(shasi_start >= 0)),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    source_table_path = required_path(config, "source_horizon_table")
    trace_header_path = required_path(config, "trace_header_csv")
    output_dir = Path(str(config["output_dir"])).resolve()

    source = np.load(source_table_path, mmap_mode="r")
    required_fields = {"TraceIdx", "T4", "T5", "T6", "T7"}
    if source.dtype.names is None or not required_fields.issubset(source.dtype.names):
        raise ValueError(f"source horizon table must contain {sorted(required_fields)}")
    trace_df = pd.read_csv(trace_header_path, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    trace_df = trace_df.sort_values("TraceIdx").reset_index(drop=True)
    if len(trace_df) != len(source) or not np.array_equal(
        trace_df["TraceIdx"].to_numpy(dtype=np.int64), source["TraceIdx"].astype(np.int64)
    ):
        raise ValueError("trace header and source horizon table TraceIdx contracts differ")
    x_axis, y_axis, ix, iy = grid_layout(trace_df)

    block = dict(config["target_block"])
    x = trace_df["X"].to_numpy(dtype=np.float64)
    y = trace_df["Y"].to_numpy(dtype=np.float64)
    selected = (
        (x >= float(block["x_min"]))
        & (x <= float(block["x_max"]))
        & (y >= float(block["y_min"]))
        & (y <= float(block["y_max"]))
    )
    if not np.any(selected):
        raise ValueError("target block contains no seismic traces")
    if args.check_only:
        print(f"[horizon-v2] check-only=pass selected_traces={int(selected.sum())}")
        return 0

    t4 = source["T4"].astype(np.float32, copy=True)
    t5 = source["T5"].astype(np.float32, copy=True)
    t6 = source["T6"].astype(np.float32, copy=True)
    original_t7 = source["T7"].astype(np.float32, copy=True)
    t7 = original_t7.copy()
    finite = np.isfinite(t4) & np.isfinite(t5) & np.isfinite(t6) & np.isfinite(t7)

    # The coherent T6/T7 reversal is represented explicitly as Shasi absence. This preserves T6,
    # does not invent a positive thickness, and leaves every trace outside the v2 target block untouched.
    correction_mask = selected & finite & (t7 <= t6)
    t7[correction_mask] = t6[correction_mask]
    correction_code = np.zeros(len(source), dtype=np.uint8)
    correction_code[correction_mask] = 1

    shasan_present = finite & (t4 < t6)
    shasi_present = finite & (t7 > t6)
    surface_order_valid = finite & (t4 < t5) & (t5 < t6) & ((t6 < t7) | ((t7 == t6) & (~shasi_present)))

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / str(config.get("output_filename", "horizon_trace_table.npy"))
    temporary_path = table_path.with_suffix(table_path.suffix + ".tmp")
    table = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=TABLE_DTYPE, shape=(len(source),))
    table["TraceIdx"] = source["TraceIdx"]
    table["T4"] = t4
    table["T5"] = t5
    table["T6"] = t6
    table["T7"] = t7
    table["SurfaceOrderValid"] = surface_order_valid.astype(np.uint8)
    table["ShasanPresent"] = shasan_present.astype(np.uint8)
    table["ShasiPresent"] = shasi_present.astype(np.uint8)
    table["HorizonCorrectionCode"] = correction_code
    table.flush()
    del table
    os.replace(temporary_path, table_path)

    selected_grid = np.zeros((len(y_axis), len(x_axis)), dtype=bool)
    selected_grid[iy[selected], ix[selected]] = True
    reversal_grid = np.zeros_like(selected_grid)
    reversal_grid[iy[correction_mask], ix[correction_mask]] = True
    labels, component_count = ndimage.label(reversal_grid, structure=np.ones((3, 3), dtype=np.uint8))
    dx = float(np.median(np.diff(x_axis))) if len(x_axis) > 1 else 1.0
    dy = float(np.median(np.diff(y_axis))) if len(y_axis) > 1 else 1.0
    boundary_distance = ndimage.distance_transform_edt(reversal_grid, sampling=(dy, dx))

    corrected_idx = np.flatnonzero(correction_mask)
    component_ids = labels[iy[corrected_idx], ix[corrected_idx]].astype(np.int32)
    audit = pd.DataFrame(
        {
            "TraceIdx": source["TraceIdx"][corrected_idx].astype(np.int32),
            "X": x[corrected_idx],
            "Y": y[corrected_idx],
            "T6OriginalMs": t6[corrected_idx],
            "T7OriginalMs": original_t7[corrected_idx],
            "T7CorrectedMs": t7[corrected_idx],
            "OriginalThicknessMs": original_t7[corrected_idx] - t6[corrected_idx],
            "CorrectedThicknessMs": t7[corrected_idx] - t6[corrected_idx],
            "CorrectionMethod": "set_T7_equal_T6_mark_Shasi_absent",
            "CorrectionCode": correction_code[corrected_idx],
            "ComponentID": component_ids,
            "DistanceToComponentBoundaryM": boundary_distance[iy[corrected_idx], ix[corrected_idx]],
            "ShasiPresent": shasi_present[corrected_idx].astype(np.uint8),
        }
    )
    audit_path = output_dir / "horizon_correction_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    component_rows: list[dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        yy, xx = np.where(labels == component_id)
        if yy.size == 0:
            continue
        trace_indices = labels[iy, ix] == component_id
        thickness = original_t7[trace_indices] - t6[trace_indices]
        component_rows.append(
            {
                "ComponentID": component_id,
                "TraceCount": int(yy.size),
                "XMin": float(x_axis[xx].min()),
                "XMax": float(x_axis[xx].max()),
                "YMin": float(y_axis[yy].min()),
                "YMax": float(y_axis[yy].max()),
                "OriginalThicknessMinMs": float(np.min(thickness)),
                "OriginalThicknessMedianMs": float(np.median(thickness)),
                "OriginalThicknessMaxMs": float(np.max(thickness)),
            }
        )
    components_path = output_dir / "t6_t7_reversal_components.csv"
    pd.DataFrame(component_rows).to_csv(components_path, index=False, encoding="utf-8-sig")

    wells = list(config.get("qc_wells", []))
    well_rows: list[dict[str, Any]] = []
    for well in wells:
        distance = np.hypot(x - float(well["x"]), y - float(well["y"]))
        nearest = int(np.argmin(distance))
        radius = float(well.get("radius_m", 100.0))
        local = selected & (distance <= radius)
        well_rows.append(
            {
                "WellName": str(well["name"]),
                "WellX": float(well["x"]),
                "WellY": float(well["y"]),
                "NearestTraceIdx": int(source["TraceIdx"][nearest]),
                "NearestTraceDistanceM": float(distance[nearest]),
                "T6OriginalMs": float(t6[nearest]),
                "T7OriginalMs": float(original_t7[nearest]),
                "T7CorrectedMs": float(t7[nearest]),
                "ShasiPresent": int(shasi_present[nearest]),
                "LocalRadiusM": radius,
                "LocalTraceCount": int(local.sum()),
                "LocalOriginalReversalFraction": float(np.mean(original_t7[local] <= t6[local])) if local.any() else None,
                "LocalCorrectedOrderFraction": float(np.mean(t7[local] >= t6[local])) if local.any() else None,
            }
        )
    well_qc_path = output_dir / "che571_che572_horizon_qc.csv"
    pd.DataFrame(well_rows).to_csv(well_qc_path, index=False, encoding="utf-8-sig")

    region_indices = np.flatnonzero(selected)
    mask_contracts: dict[str, Any] = {}
    for interval in config.get("sample_axes", []):
        interval_ms = float(interval["interval_ms"])
        name = str(interval.get("name", f"{interval_ms:g}ms"))
        mask_contracts[name] = write_axis_contract(
            output_dir / f"horizon_windows_{name}.npz",
            source["TraceIdx"][region_indices],
            t4[region_indices],
            t6[region_indices],
            t7[region_indices],
            shasan_present[region_indices],
            shasi_present[region_indices],
            float(interval["start_ms"]),
            float(interval["stop_ms"]),
            interval_ms,
        )

    x_selected = x_axis[(x_axis >= float(block["x_min"])) & (x_axis <= float(block["x_max"]))]
    y_selected = y_axis[(y_axis >= float(block["y_min"])) & (y_axis <= float(block["y_max"]))]
    selected_trace_grid = np.full((len(y_selected), len(x_selected)), -1, dtype=np.int32)
    sx = np.searchsorted(x_selected, x[selected])
    sy = np.searchsorted(y_selected, y[selected])
    selected_trace_grid[sy, sx] = region_indices.astype(np.int32)
    original_thickness_grid = original_t7[selected_trace_grid] - t6[selected_trace_grid]
    corrected_thickness_grid = t7[selected_trace_grid] - t6[selected_trace_grid]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    extent = [x_selected[0], x_selected[-1], y_selected[0], y_selected[-1]]
    limit = float(np.nanpercentile(np.abs(original_thickness_grid), 99.0))
    image0 = axes[0].imshow(original_thickness_grid, origin="lower", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0].set_title("Original T7-T6 thickness (ms)")
    image1 = axes[1].imshow(corrected_thickness_grid, origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title("v2 T7-T6 thickness (ms)")
    for axis in axes:
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        for well in wells:
            axis.plot(float(well["x"]), float(well["y"]), marker="x", color="black", markersize=7)
            axis.text(float(well["x"]), float(well["y"]), str(well["name"]), fontsize=8)
    fig.colorbar(image0, ax=axes[0], shrink=0.85)
    fig.colorbar(image1, ax=axes[1], shrink=0.85)
    thickness_map_path = output_dir / "t7_minus_t6_thickness_qc.png"
    fig.savefig(thickness_map_path, dpi=180)
    plt.close(fig)

    selected_valid = surface_order_valid[selected]
    metadata = {
        "status": "pass",
        "version": str(config.get("version", "formal_horizon_trace_table_v2")),
        "config_path": str(config_path),
        "source_horizon_table": str(source_table_path),
        "table_path": str(table_path),
        "target_block": block,
        "row_count": int(len(source)),
        "target_trace_count": int(selected.sum()),
        "correction_policy": {
            "scope": "target block only",
            "condition": "finite T7 <= T6",
            "action": "preserve T6, set T7=T6, set ShasiPresent=0",
            "interpretation": "explicit Shasi absence/pinchout candidate; no positive thickness invented",
        },
        "correction_count": int(correction_mask.sum()),
        "reversal_component_count": int(component_count),
        "target_surface_order_valid_count": int(selected_valid.sum()),
        "target_surface_order_valid_fraction": float(selected_valid.mean()),
        "target_shasan_present_count": int(shasan_present[selected].sum()),
        "target_shasi_present_count": int(shasi_present[selected].sum()),
        "non_target_changed_count": int(np.count_nonzero((~selected) & (t7 != original_t7))),
        "fields": {name: str(TABLE_DTYPE.fields[name][0]) for name in TABLE_DTYPE.names or []},
        "mask_contracts": mask_contracts,
        "outputs": {
            "audit_csv": str(audit_path),
            "components_csv": str(components_path),
            "well_qc_csv": str(well_qc_path),
            "thickness_map_png": str(thickness_map_path),
        },
    }
    metadata_path = output_dir / "horizon_contract_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
