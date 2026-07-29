from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REQUIRED_FIELDS = {
    "TraceIdx",
    "T4",
    "T5",
    "T6",
    "T7",
    "SurfaceOrderValid",
    "ShasanPresent",
    "ShasiPresent",
}


@dataclass(frozen=True)
class HorizonTraceContract:
    table_path: Path
    trace_idx: np.ndarray
    t4: np.ndarray
    t5: np.ndarray
    t6: np.ndarray
    t7: np.ndarray
    surface_order_valid: np.ndarray
    shasan_present: np.ndarray
    shasi_present: np.ndarray
    correction_code: np.ndarray

    @property
    def trace_count(self) -> int:
        return int(len(self.trace_idx))


def resolve_horizon_contract_path(config: dict[str, Any]) -> Path:
    value = config.get("horizon_contract_table") or config.get("horizon_trace_table_path")
    if not value:
        raise KeyError("configuration must define horizon_contract_table")
    path = Path(str(value)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"horizon contract table not found: {path}")
    return path


def load_horizon_table(path: Path) -> np.ndarray:
    table = np.load(path, mmap_mode="r")
    fields = set(table.dtype.names or ())
    missing = sorted(REQUIRED_FIELDS - fields)
    if missing:
        raise ValueError(f"horizon contract table missing fields: {missing}")
    if len(table) == 0:
        raise ValueError("horizon contract table is empty")
    trace_idx = np.asarray(table["TraceIdx"], dtype=np.int64)
    if int(trace_idx[0]) != 0 or int(trace_idx[-1]) != len(table) - 1 or not np.array_equal(
        trace_idx, np.arange(len(table), dtype=np.int64)
    ):
        raise ValueError("horizon contract row index must equal TraceIdx")
    return table


def load_contract_for_trace_indices(path: Path, trace_idx: np.ndarray) -> HorizonTraceContract:
    table = load_horizon_table(path)
    indices = np.asarray(trace_idx, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("source TraceIdx must be a non-empty one-dimensional array")
    if int(indices.min()) < 0 or int(indices.max()) >= len(table):
        raise ValueError("source TraceIdx exceeds horizon contract table")
    if not np.array_equal(np.asarray(table["TraceIdx"][indices], dtype=np.int64), indices):
        raise ValueError("horizon contract TraceIdx lookup is inconsistent")
    correction = (
        np.asarray(table["HorizonCorrectionCode"][indices], dtype=np.uint8)
        if "HorizonCorrectionCode" in (table.dtype.names or ())
        else np.zeros(len(indices), dtype=np.uint8)
    )
    return HorizonTraceContract(
        table_path=path,
        trace_idx=indices.copy(),
        t4=np.asarray(table["T4"][indices], dtype=np.float32),
        t5=np.asarray(table["T5"][indices], dtype=np.float32),
        t6=np.asarray(table["T6"][indices], dtype=np.float32),
        t7=np.asarray(table["T7"][indices], dtype=np.float32),
        surface_order_valid=np.asarray(table["SurfaceOrderValid"][indices], dtype=bool),
        shasan_present=np.asarray(table["ShasanPresent"][indices], dtype=bool),
        shasi_present=np.asarray(table["ShasiPresent"][indices], dtype=bool),
        correction_code=correction,
    )


def load_contract_for_mapping(config: dict[str, Any], mapping: dict[str, np.ndarray]) -> HorizonTraceContract:
    if "source_trace_idx" not in mapping:
        raise ValueError("trace mapping missing source_trace_idx")
    return load_contract_for_trace_indices(
        resolve_horizon_contract_path(config),
        np.asarray(mapping["source_trace_idx"], dtype=np.int64),
    )


def regular_axis_interval_ms(samples: np.ndarray) -> float:
    axis = np.asarray(samples, dtype=np.float64)
    if axis.ndim != 1 or len(axis) == 0:
        raise ValueError("sample axis must be non-empty and one-dimensional")
    if len(axis) == 1:
        return 0.0
    delta = np.diff(axis)
    interval = float(np.median(delta))
    if interval <= 0.0 or not np.allclose(delta, interval, atol=1.0e-6, rtol=0.0):
        raise ValueError("sample axis must be positive and regular")
    return interval


def validate_window_contract(
    config: dict[str, Any],
    contract: HorizonTraceContract,
    samples: np.ndarray,
) -> dict[str, Any]:
    interval = regular_axis_interval_ms(samples)
    key = f"{interval:g}ms"
    configured = dict(config.get("horizon_window_contracts", {}))
    if key not in configured:
        raise KeyError(f"horizon_window_contracts does not define {key}")
    path = Path(str(configured[key])).resolve()
    if not path.exists():
        raise FileNotFoundError(f"horizon window contract not found: {path}")
    with np.load(path) as payload:
        required = {"TraceIdx", "samples", "T4", "T6", "T7", "ShasanPresent", "ShasiPresent"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"horizon window contract missing fields: {missing}")
        expected_samples = np.asarray(payload["samples"], dtype=np.float64)
        if not np.array_equal(np.asarray(payload["TraceIdx"], dtype=np.int64), contract.trace_idx):
            raise ValueError(f"horizon window TraceIdx mismatch: {path}")
        if len(expected_samples) != len(samples) or not np.allclose(
            expected_samples, samples, atol=1.0e-6, rtol=0.0
        ):
            raise ValueError(
                f"horizon window sample axis mismatch for {path}: "
                f"expected=({expected_samples[0]}, {expected_samples[-1]}, {len(expected_samples)}) "
                f"actual=({samples[0]}, {samples[-1]}, {len(samples)})"
            )
        for name, values in {"T4": contract.t4, "T6": contract.t6, "T7": contract.t7}.items():
            if not np.allclose(np.asarray(payload[name], dtype=np.float32), values, atol=1.0e-5, rtol=0.0):
                raise ValueError(f"horizon window {name} differs from table contract: {path}")
        if not np.array_equal(np.asarray(payload["ShasanPresent"], dtype=bool), contract.shasan_present):
            raise ValueError(f"horizon window ShasanPresent differs from table contract: {path}")
        if not np.array_equal(np.asarray(payload["ShasiPresent"], dtype=bool), contract.shasi_present):
            raise ValueError(f"horizon window ShasiPresent differs from table contract: {path}")
    return {
        "status": "pass",
        "path": str(path),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "sample_interval_ms": interval,
        "sample_count": int(len(samples)),
        "trace_count": contract.trace_count,
    }


def trace_window_mask(
    contract: HorizonTraceContract,
    samples: np.ndarray,
    start: int,
    stop: int,
    layer: str = "T4-T7",
) -> np.ndarray:
    axis = np.asarray(samples, dtype=np.float64)[None, :]
    sl = slice(start, stop)
    if layer == "Shasan":
        return (
            contract.shasan_present[sl, None]
            & (axis >= contract.t4[sl, None])
            & (axis < contract.t6[sl, None])
        )
    if layer == "Shasi":
        return (
            contract.shasi_present[sl, None]
            & (axis >= contract.t6[sl, None])
            & (axis <= contract.t7[sl, None])
        )
    if layer != "T4-T7":
        raise ValueError(f"unsupported horizon layer: {layer}")
    return trace_window_mask(contract, samples, start, stop, "Shasan") | trace_window_mask(
        contract, samples, start, stop, "Shasi"
    )


def apply_window_inplace(
    values: np.ndarray,
    contract: HorizonTraceContract,
    samples: np.ndarray,
    *,
    fill_value: float | int | bool = 0,
    layer: str = "T4-T7",
    chunk_traces: int = 4096,
) -> dict[str, Any]:
    if values.ndim != 2 or values.shape != (contract.trace_count, len(samples)):
        raise ValueError(
            f"horizon mask shape mismatch: values={values.shape}, expected={(contract.trace_count, len(samples))}"
        )
    kept = 0
    total = int(values.size)
    for start in range(0, contract.trace_count, max(1, int(chunk_traces))):
        stop = min(start + max(1, int(chunk_traces)), contract.trace_count)
        valid = trace_window_mask(contract, samples, start, stop, layer)
        block = values[start:stop]
        kept += int(valid.sum())
        block[~valid] = fill_value
    return {
        "layer": layer,
        "voxel_count": total,
        "inside_voxel_count": kept,
        "outside_voxel_count": total - kept,
        "inside_fraction": float(kept / total) if total else 0.0,
    }


def apply_validity_inplace(
    valid: np.ndarray,
    contract: HorizonTraceContract,
    samples: np.ndarray,
    *,
    layer: str = "T4-T7",
    chunk_traces: int = 4096,
) -> dict[str, Any]:
    if valid.dtype != np.bool_:
        raise ValueError("validity array must have boolean dtype")
    return apply_window_inplace(
        valid,
        contract,
        samples,
        fill_value=False,
        layer=layer,
        chunk_traces=chunk_traces,
    )


def surface_grids_from_contract(
    mapping: dict[str, np.ndarray],
    contract: HorizonTraceContract,
) -> dict[str, np.ndarray]:
    ix = np.asarray(mapping["ix"], dtype=np.int32)
    iy = np.asarray(mapping["iy"], dtype=np.int32)
    if len(ix) != contract.trace_count:
        raise ValueError("mapping and horizon contract trace counts differ")
    shape = (int(iy.max()) + 1, int(ix.max()) + 1)

    def grid(values: np.ndarray, dtype: np.dtype | type = np.float32) -> np.ndarray:
        output = np.zeros(shape, dtype=dtype)
        output[iy, ix] = values
        return output

    any_layer_present = contract.shasan_present | contract.shasi_present
    return {
        "T4_TIME": grid(contract.t4),
        "T5_TIME": grid(contract.t5),
        "T6_TIME": grid(contract.t6),
        "T7_TIME": grid(contract.t7),
        "SurfaceOrderValid": grid(contract.surface_order_valid.astype(np.uint8), np.uint8),
        "ShasanPresent": grid(contract.shasan_present.astype(np.uint8), np.uint8),
        "ShasiPresent": grid(contract.shasi_present.astype(np.uint8), np.uint8),
        "Check_All": grid(any_layer_present.astype(np.uint8), np.uint8).astype(bool),
        "SourceTraceIdx": grid(contract.trace_idx.astype(np.int32), np.int32),
    }


class HorizonSpatialLookup:
    def __init__(self, table_path: Path, trace_header_csv: Path, target_block: dict[str, Any] | None = None):
        table = load_horizon_table(table_path)
        frame = pd.read_csv(trace_header_csv, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
        for column in ["TraceIdx", "X", "Y"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame = frame.sort_values("TraceIdx").reset_index(drop=True)
        if len(frame) != len(table) or not np.array_equal(
            frame["TraceIdx"].to_numpy(dtype=np.int64), np.asarray(table["TraceIdx"], dtype=np.int64)
        ):
            raise ValueError("trace header and horizon contract table differ")
        if target_block:
            frame = frame[
                frame["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
                & frame["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
            ].copy()
        if frame.empty:
            raise ValueError("horizon spatial lookup contains no traces")
        self.table_path = table_path
        self.trace_idx = frame["TraceIdx"].to_numpy(dtype=np.int64)
        self.xy = frame[["X", "Y"]].to_numpy(dtype=np.float64)
        self.tree = cKDTree(self.xy)
        self.table = table

    def query(self, x: float, y: float) -> dict[str, Any]:
        distance, position = self.tree.query(np.asarray([[float(x), float(y)]], dtype=np.float64), k=1)
        trace_idx = int(self.trace_idx[int(np.asarray(position).reshape(-1)[0])])
        row = self.table[trace_idx]
        return {
            "TraceIdx": trace_idx,
            "DistanceM": float(np.asarray(distance).reshape(-1)[0]),
            "T4": float(row["T4"]),
            "T5": float(row["T5"]),
            "T6": float(row["T6"]),
            "T7": float(row["T7"]),
            "SurfaceOrderValid": bool(row["SurfaceOrderValid"]),
            "ShasanPresent": bool(row["ShasanPresent"]),
            "ShasiPresent": bool(row["ShasiPresent"]),
            "HorizonCorrectionCode": int(row["HorizonCorrectionCode"])
            if "HorizonCorrectionCode" in (self.table.dtype.names or ())
            else 0,
        }

    def interval(self, x: float, y: float, time_ms: float) -> str | None:
        row = self.query(x, y)
        time = float(time_ms)
        if row["ShasanPresent"] and row["T4"] <= time < row["T6"]:
            return "T4->T6"
        if row["ShasiPresent"] and row["T6"] <= time <= row["T7"]:
            return "T6->T7"
        return None

    def layer_group(self, x: float, y: float, time_ms: float) -> str | None:
        interval = self.interval(x, y, time_ms)
        return {"T4->T6": "沙三段", "T6->T7": "沙四段"}.get(interval)


def build_spatial_lookup(config: dict[str, Any]) -> HorizonSpatialLookup:
    trace_header = config.get("trace_header_csv")
    if not trace_header:
        raise KeyError("configuration must define trace_header_csv for spatial horizon lookup")
    path = Path(str(trace_header)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"trace header CSV not found: {path}")
    return HorizonSpatialLookup(
        resolve_horizon_contract_path(config),
        path,
        dict(config.get("target_block") or {}) or None,
    )


def contract_summary(contract: HorizonTraceContract) -> dict[str, Any]:
    return {
        "table_path": str(contract.table_path),
        "trace_count": contract.trace_count,
        "surface_order_valid_count": int(contract.surface_order_valid.sum()),
        "shasan_present_count": int(contract.shasan_present.sum()),
        "shasi_present_count": int(contract.shasi_present.sum()),
        "corrected_trace_count": int(np.count_nonzero(contract.correction_code)),
    }


__all__ = [
    "HorizonSpatialLookup",
    "HorizonTraceContract",
    "apply_validity_inplace",
    "apply_window_inplace",
    "build_spatial_lookup",
    "contract_summary",
    "load_contract_for_mapping",
    "load_contract_for_trace_indices",
    "load_horizon_table",
    "regular_axis_interval_ms",
    "resolve_horizon_contract_path",
    "surface_grids_from_contract",
    "trace_window_mask",
    "validate_window_contract",
]
