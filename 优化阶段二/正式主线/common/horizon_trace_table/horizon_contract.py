from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:  # 作为 common.horizon_trace_table 包导入时
    from ..strata_framework import Layer, StrataFramework, load_framework, resolve_framework
except ImportError:  # 作为顶层 horizon_trace_table 模块导入时（太古界走这条）
    from strata_framework import Layer, StrataFramework, load_framework, resolve_framework


DEFAULT_FRAMEWORK = load_framework()


def framework_required_fields(framework: StrataFramework | None = None) -> set[str]:
    """契约表必须具备的列：TraceIdx + 有效标志 + 各层位面 + 各层存在标志。"""
    fw = framework or DEFAULT_FRAMEWORK
    return {
        "TraceIdx",
        "SurfaceOrderValid",
        *fw.surface_columns(),
        *fw.legacy_presence_columns(),
    }


# 向后兼容：历史代码直接引用 REQUIRED_FIELDS
REQUIRED_FIELDS = framework_required_fields(DEFAULT_FRAMEWORK)


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
    framework_id: str = DEFAULT_FRAMEWORK.framework_id
    # 框架原生视图：Surface.id -> 逐道时间；地层 code -> 逐道存在标志
    surface_times: dict[str, np.ndarray] = field(default_factory=dict)
    layer_present: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def trace_count(self) -> int:
        return int(len(self.trace_idx))

    @property
    def framework(self) -> StrataFramework:
        return load_framework(self.framework_id)

    def surface_time(self, surface_id: str) -> np.ndarray:
        if surface_id in self.surface_times:
            return self.surface_times[surface_id]
        return np.asarray(getattr(self, surface_id.lower()), dtype=np.float32)


def resolve_horizon_contract_path(config: dict[str, Any]) -> Path:
    value = config.get("horizon_contract_table") or config.get("horizon_trace_table_path")
    if not value:
        raise KeyError("configuration must define horizon_contract_table")
    path = Path(str(value)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"horizon contract table not found: {path}")
    return path


def load_horizon_table(
    path: Path,
    framework: StrataFramework | None = None,
) -> np.ndarray:
    fw = framework or DEFAULT_FRAMEWORK
    table = np.load(path, mmap_mode="r")
    fields = set(table.dtype.names or ())
    missing = sorted(framework_required_fields(fw) - fields)
    if missing:
        raise ValueError(
            f"horizon contract table missing fields for {fw.framework_id}: {missing}"
        )
    if len(table) == 0:
        raise ValueError("horizon contract table is empty")
    trace_idx = np.asarray(table["TraceIdx"], dtype=np.int64)
    if int(trace_idx[0]) != 0 or int(trace_idx[-1]) != len(table) - 1 or not np.array_equal(
        trace_idx, np.arange(len(table), dtype=np.int64)
    ):
        raise ValueError("horizon contract row index must equal TraceIdx")
    return table


def load_contract_for_trace_indices(
    path: Path,
    trace_idx: np.ndarray,
    framework: StrataFramework | None = None,
) -> HorizonTraceContract:
    fw = framework or DEFAULT_FRAMEWORK
    table = load_horizon_table(path, fw)
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
    surface_times = {
        surface.id: np.asarray(table[surface.column][indices], dtype=np.float32)
        for surface in fw.surfaces
    }
    layer_present = {
        layer.code: np.asarray(table[layer.presence_column][indices], dtype=bool)
        for layer in fw.layers
        if layer.presence_column
    }
    # 兼容字段：只有当框架确实声明了 T4..T7 时才有意义
    def legacy(name: str) -> np.ndarray:
        if name in surface_times:
            return surface_times[name]
        return np.full(len(indices), np.nan, dtype=np.float32)

    shasan = layer_present.get(3)
    shasi = layer_present.get(4)
    if shasan is None and "Shasan" in fw.legacy_layer_aliases:
        shasan = layer_present.get(fw.layer_by_name(fw.legacy_layer_aliases["Shasan"]).code)
    if shasi is None and "Shasi" in fw.legacy_layer_aliases:
        shasi = layer_present.get(fw.layer_by_name(fw.legacy_layer_aliases["Shasi"]).code)
    if shasan is None:
        shasan = np.asarray(table["ShasanPresent"][indices], dtype=bool) if "ShasanPresent" in (table.dtype.names or ()) else np.zeros(len(indices), dtype=bool)
    if shasi is None:
        shasi = np.asarray(table["ShasiPresent"][indices], dtype=bool) if "ShasiPresent" in (table.dtype.names or ()) else np.zeros(len(indices), dtype=bool)

    return HorizonTraceContract(
        table_path=path,
        trace_idx=indices.copy(),
        t4=legacy("T4"),
        t5=legacy("T5"),
        t6=legacy("T6"),
        t7=legacy("T7"),
        surface_order_valid=np.asarray(table["SurfaceOrderValid"][indices], dtype=bool),
        shasan_present=np.asarray(shasan, dtype=bool),
        shasi_present=np.asarray(shasi, dtype=bool),
        correction_code=correction,
        framework_id=fw.framework_id,
        surface_times=surface_times,
        layer_present=layer_present,
    )


def load_contract_for_mapping(config: dict[str, Any], mapping: dict[str, np.ndarray]) -> HorizonTraceContract:
    if "source_trace_idx" not in mapping:
        raise ValueError("trace mapping missing source_trace_idx")
    return load_contract_for_trace_indices(
        resolve_horizon_contract_path(config),
        np.asarray(mapping["source_trace_idx"], dtype=np.int64),
        resolve_framework(config),
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
    fw = contract.framework
    with np.load(path) as payload:
        required = {
            "TraceIdx",
            "samples",
            *fw.layer_surface_columns(),
            *fw.legacy_presence_columns(),
        }
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
        referenced = set(fw.layer_surface_columns())
        for surface in fw.surfaces:
            if surface.column not in referenced:
                continue
            if not np.allclose(
                np.asarray(payload[surface.column], dtype=np.float32),
                contract.surface_time(surface.id),
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise ValueError(
                    f"horizon window {surface.column} differs from table contract: {path}"
                )
        for layer in fw.layers:
            column = layer.presence_column
            if not column:
                continue
            if not np.array_equal(
                np.asarray(payload[column], dtype=bool),
                np.asarray(contract.layer_present[layer.code], dtype=bool),
            ):
                raise ValueError(f"horizon window {column} differs from table contract: {path}")
    return {
        "status": "pass",
        "path": str(path),
        "sample_min_ms": float(samples[0]),
        "sample_max_ms": float(samples[-1]),
        "sample_interval_ms": interval,
        "sample_count": int(len(samples)),
        "trace_count": contract.trace_count,
    }


def resolve_layers(contract: HorizonTraceContract, layer: str | int | None) -> tuple[Layer, ...]:
    """把层引用解析成地层列表。

    支持：框架 LayerCode(int)、地层名、历史别名(Shasan/Shasi)、
    以及 "T4-T7"/"main"/None 表示"全部地层"。
    """
    framework = contract.framework
    if layer is None or str(layer) in {"T4-T7", "main", "all"}:
        return tuple(framework.layers)
    alias = framework.resolve_layer_alias(str(layer))
    if alias is not None:
        return (alias,)
    return (framework.layer(layer),)


def layer_window_mask(
    contract: HorizonTraceContract,
    samples: np.ndarray,
    start: int,
    stop: int,
    layer: Layer,
) -> np.ndarray:
    """单个地层的层窗掩膜（框架驱动，不再按地层名写 if/else）。"""
    axis = np.asarray(samples, dtype=np.float64)[None, :]
    sl = slice(start, stop)
    top = contract.surface_time(layer.top)[sl, None]
    base = contract.surface_time(layer.base)[sl, None]
    if layer.extension_ms > 0:
        base = base + np.float64(layer.extension_ms)
    present = contract.layer_present.get(layer.code)
    if present is None:
        present = contract.surface_order_valid
    inside = axis <= base if layer.inclusive_base else axis < base
    return np.asarray(present[sl, None], dtype=bool) & (axis >= top) & inside


def trace_window_mask(
    contract: HorizonTraceContract,
    samples: np.ndarray,
    start: int,
    stop: int,
    layer: str | int | None = "T4-T7",
) -> np.ndarray:
    layers = resolve_layers(contract, layer)
    mask = layer_window_mask(contract, samples, start, stop, layers[0])
    for item in layers[1:]:
        mask = mask | layer_window_mask(contract, samples, start, stop, item)
    return mask


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

    framework = contract.framework
    output: dict[str, np.ndarray] = {
        f"{surface.id}_TIME": grid(contract.surface_time(surface.id))
        for surface in framework.surfaces
    }
    output["SurfaceOrderValid"] = grid(
        contract.surface_order_valid.astype(np.uint8), np.uint8
    )
    any_layer_present = np.zeros(contract.trace_count, dtype=bool)
    for layer in framework.layers:
        present = np.asarray(
            contract.layer_present.get(layer.code, contract.surface_order_valid),
            dtype=bool,
        )
        any_layer_present |= present
        if layer.presence_column:
            output[layer.presence_column] = grid(present.astype(np.uint8), np.uint8)
    output["Check_All"] = grid(any_layer_present.astype(np.uint8), np.uint8).astype(bool)
    output["SourceTraceIdx"] = grid(contract.trace_idx.astype(np.int32), np.int32)
    return output


class HorizonSpatialLookup:
    def __init__(
        self,
        table_path: Path,
        trace_header_csv: Path,
        target_block: dict[str, Any] | None = None,
        framework: StrataFramework | None = None,
    ):
        self.framework = framework or DEFAULT_FRAMEWORK
        table = load_horizon_table(table_path, self.framework)
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
        surface_times = {
            surface.id: float(row[surface.column]) for surface in self.framework.surfaces
        }
        layer_present = {
            layer.code: bool(row[layer.presence_column])
            for layer in self.framework.layers
            if layer.presence_column
        }
        payload = {
            "TraceIdx": trace_idx,
            "DistanceM": float(np.asarray(distance).reshape(-1)[0]),
            "SurfaceOrderValid": bool(row["SurfaceOrderValid"]),
            "SurfaceTimes": surface_times,
            "LayerPresent": layer_present,
            "HorizonCorrectionCode": int(row["HorizonCorrectionCode"])
            if "HorizonCorrectionCode" in (self.table.dtype.names or ())
            else 0,
        }
        # 兼容字段：砂砾岩历史调用直接读 T4..T7 / ShasanPresent / ShasiPresent
        for surface in self.framework.surfaces:
            payload[surface.id] = surface_times[surface.id]
        for layer in self.framework.layers:
            if layer.presence_column:
                payload[layer.presence_column] = layer_present[layer.code]
        return payload

    def interval(self, x: float, y: float, time_ms: float) -> str | None:
        row = self.query(x, y)
        time = float(time_ms)
        for layer in self.framework.layers:
            if not row["LayerPresent"].get(layer.code, False):
                continue
            top = row["SurfaceTimes"][layer.top]
            base = row["SurfaceTimes"][layer.base]
            if layer.extension_ms > 0:
                base += layer.extension_ms
            inside = top <= time <= base if layer.inclusive_base else top <= time < base
            if inside:
                return layer.interval_label()
        return None

    def layer_group(self, x: float, y: float, time_ms: float) -> str | None:
        row = self.query(x, y)
        time = float(time_ms)
        for layer in self.framework.layers:
            if not row["LayerPresent"].get(layer.code, False):
                continue
            top = row["SurfaceTimes"][layer.top]
            base = row["SurfaceTimes"][layer.base]
            if layer.extension_ms > 0:
                base += layer.extension_ms
            inside = top <= time <= base if layer.inclusive_base else top <= time < base
            if inside:
                return layer.name
        return None


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
        resolve_framework(config),
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
