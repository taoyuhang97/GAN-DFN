# -*- coding: utf-8 -*-
"""地球物理约束后处理模块 —— 提升区域 DFN 连贯性与可靠性

工作流位置: merge_unit_dfn_vtks → **geophysical_postprocess** → scale_merged_dfn_vtk_uniform
输入输出: 合并后的裂缝片 CSV / VTK → 后处理后的裂缝片 CSV / VTK

============================================================
设计原则 
============================================================
- 后处理是"提供证据和标注", 不是"强行修改数据"
- GMM 只做候选组识别, 不作为硬分类真值
- 产状平滑只在组内、局部邻域做, 必须保留原值字段
- 走廊是支撑证据 (CorridorSupport), 不是连接许可
- 跨界补接必须保守: 高置信匹配对 + 同组同层 + 长度上限 + 标记为补充片
- 可靠性评分拆为"证据层 + 结论层", 每个维度独立可查

============================================================
两轮实施策略
============================================================
第一轮 (保守, run_phase1):
  Step 1: 裂缝组识别
  Step 4: 多维度可靠性评分
  Step 5: 最保守的边界匹配 (仅标记匹配对, 不插值补片)

第二轮 (增强, run_phase2):
  Step 2: 组内产状平滑 (不覆盖原值, 仅写 SmoothedAzimuth/SmoothedDip)
  Step 3: 走廊检测 (作为 CorridorSupport 证据)
  Step 5: 有限补接 (高置信 + 同组同层 + 长度上限 + 标记 IsSupplemented)
  Step 6: 沿走向拉伸 (同组相邻片沿走向延长, 形成视觉连续带, --elongation-* 控制)
  Step 7: 空间扰动 (打破地震道网格规则排列, 可选, --jitter-xy 控制)

============================================================
可解释性输出字段
============================================================
证据层 (每个维度独立, 可分别在 VTK 中着色查看):
  - FractureSet, SetProbability  : 裂缝组归属及后验概率
  - GeometryScore    : 几何规则性 (尺寸合理性)
  - StratigraphyScore: 层位一致性 (是否在已知裂缝发育层段内)
  - GeophysicsScore  : 地球物理支撑 (邻域密度)
  - CorridorSupport  : 走廊支撑 ∈ {0, 1}
  - FaultPenalty     : 穿越断层惩罚 ∈ [0, 1] (预留, 需断层数据)

结论层:
  - ReliabilityLevel : 综合可靠性等级 {"high", "medium", "low"}
  - ConnectionType   : {"original", "boundary_matched", "supplemented"}
  - IsSupplemented   : 0/1, 是否为后处理补充片 (非模型原始预测)
  - OrigAzimuth, OrigDip : 原始产状 (始终保留)
  - SmoothedAzimuth, SmoothedDip : 平滑参考值 (不覆盖 Azimuth/Dip)
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

THIS_DIR = Path(__file__).resolve().parent
OPT_STAGE_DIR = THIS_DIR.parent
BASELINE_DIR = OPT_STAGE_DIR / "G_DFN监督基线"

for candidate in (THIS_DIR, BASELINE_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_csv_utf8, write_json
from layer_model_registry import (
    LAYER_SURFACE_PAIR_KEY_COL,
    UNIT_LAYER_SEGMENT_KEY_COL,
    build_layer_surface_pair_key,
)
from merge_unit_dfn_vtks import read_legacy_vtk_polygons, write_legacy_vtk_polygons


DEFAULT_POSTPROCESS_COMPUTE_BACKEND = "auto"
DEFAULT_POSTPROCESS_MAX_CPU_THREADS = 24
DEFAULT_POSTPROCESS_GPU_TILE_POINTS = 2048


# ---------------------------------------------------------------------------
#   角度工具
# ---------------------------------------------------------------------------

def _azimuth_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组方位角之差的绝对值 ∈ [0, 90], 考虑 180° 对称性 (走向无正反)。"""
    d = np.abs(a - b) % 360.0
    d = np.minimum(d, 360.0 - d)
    d = np.minimum(d, 180.0 - d)
    return d


def _azimuth_mean_weighted(azimuths: np.ndarray, weights: np.ndarray) -> float:
    """加权圆周均值 (考虑 180° 对称)。"""
    rad = np.deg2rad(2.0 * azimuths)
    wx = np.sum(weights * np.cos(rad))
    wy = np.sum(weights * np.sin(rad))
    mean_rad = np.arctan2(wy, wx) / 2.0
    return float(np.rad2deg(mean_rad) % 180.0)


def _normalize_layer_key_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if (not text or text.lower() == "nan") else text


def _resolve_layer_surface_pair_key(row: pd.Series | dict[str, Any]) -> str:
    if not hasattr(row, "get"):
        return ""
    existing = _normalize_layer_key_text(row.get(LAYER_SURFACE_PAIR_KEY_COL, ""))
    if existing:
        return existing
    top_surface_code = _normalize_layer_key_text(row.get("TopSurfaceCode", ""))
    base_surface_code = _normalize_layer_key_text(row.get("BaseSurfaceCode", ""))
    if top_surface_code or base_surface_code:
        return build_layer_surface_pair_key(top_surface_code, base_surface_code)
    return _normalize_layer_key_text(row.get("GeoIntervalKey", ""))


def _resolve_unit_layer_segment_key(row: pd.Series | dict[str, Any]) -> str:
    if not hasattr(row, "get"):
        return ""
    existing = _normalize_layer_key_text(row.get(UNIT_LAYER_SEGMENT_KEY_COL, ""))
    if existing:
        return existing
    pair_key = _resolve_layer_surface_pair_key(row)
    unit_id = _normalize_layer_key_text(row.get("UnitID", ""))
    interval_key = _normalize_layer_key_text(row.get("GeoIntervalKey", ""))
    if unit_id or interval_key or pair_key:
        return f"{unit_id or 'UNKNOWN_UNIT'}__{interval_key or 'UNKNOWN_INTERVAL'}__{pair_key or 'UNKNOWN_LAYER_PAIR'}"
    return ""


def _preferred_layer_series(df: pd.DataFrame, prefer_unit_segment: bool = False) -> pd.Series:
    fallback = df.get("GeoIntervalKey", pd.Series([""] * len(df), index=df.index, dtype="object")).fillna("").astype(str)
    pair_series = (
        df.get(LAYER_SURFACE_PAIR_KEY_COL, pd.Series([""] * len(df), index=df.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    if prefer_unit_segment:
        segment_series = (
            df.get(UNIT_LAYER_SEGMENT_KEY_COL, pd.Series([""] * len(df), index=df.index, dtype="object"))
            .fillna("")
            .astype(str)
            .str.strip()
        )
        return segment_series.mask(segment_series.eq(""), pair_series).mask(lambda s: s.eq(""), fallback)
    return pair_series.mask(pair_series.eq(""), fallback)


def _same_physical_layer(row_i: pd.Series | dict[str, Any], row_j: pd.Series | dict[str, Any]) -> bool:
    layer_i = _resolve_layer_surface_pair_key(row_i)
    layer_j = _resolve_layer_surface_pair_key(row_j)
    if layer_i or layer_j:
        return layer_i == layer_j
    return _normalize_layer_key_text(row_i.get("GeoIntervalKey", "")) == _normalize_layer_key_text(row_j.get("GeoIntervalKey", ""))


def _numeric_series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(np.full(len(df), float(default), dtype=float), index=df.index, dtype=float)


def _normalize_compute_backend(compute_backend: str | None) -> str:
    backend = str(compute_backend or DEFAULT_POSTPROCESS_COMPUTE_BACKEND).strip().lower()
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unsupported compute_backend: {compute_backend}")
    return backend


def _resolve_compute_backend(compute_backend: str | None) -> str:
    backend = _normalize_compute_backend(compute_backend)
    has_cuda = bool(torch is not None and torch.cuda.is_available())
    if backend == "auto":
        return "gpu" if has_cuda else "cpu"
    if backend == "gpu" and not has_cuda:
        raise RuntimeError("compute_backend='gpu' requested, but CUDA torch is not available")
    return backend


@contextlib.contextmanager
def _thread_limit_context(max_cpu_threads: int | None):
    limit = int(max_cpu_threads or 0)
    if limit <= 0 or threadpool_limits is None:
        yield
        return
    with threadpool_limits(limits=limit):
        if torch is not None:
            try:
                previous_threads = int(torch.get_num_threads())
                torch.set_num_threads(max(1, min(previous_threads, limit)))
            except Exception:  # pragma: no cover
                previous_threads = None
            try:
                yield
            finally:
                if previous_threads is not None:
                    try:
                        torch.set_num_threads(previous_threads)
                    except Exception:
                        pass
        else:
            yield


class _StageProgressPrinter:
    def __init__(self, total_steps: int, prefix: str = "postprocess", bar_width: int = 24):
        self.total_steps = max(int(total_steps), 1)
        self.prefix = str(prefix)
        self.bar_width = max(int(bar_width), 10)
        self.current_step = 0
        self.started_at = perf_counter()
        self._emit("start", detail=f"total_steps={self.total_steps}")

    def _render_bar(self) -> str:
        ratio = min(max(self.current_step / self.total_steps, 0.0), 1.0)
        filled = int(round(self.bar_width * ratio))
        filled = min(max(filled, 0), self.bar_width)
        return "#" * filled + "." * (self.bar_width - filled)

    def _emit(self, label: str, detail: str | None = None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_seconds = perf_counter() - self.started_at
        line = (
            f"[{timestamp}] [{self.prefix}] "
            f"[{self._render_bar()}] {self.current_step}/{self.total_steps} {label}"
        )
        line += f" | elapsed={elapsed_seconds:.1f}s"
        if detail:
            line += f" | {detail}"
        print(line, flush=True)

    def advance(self, label: str, detail: str | None = None) -> None:
        self.current_step = min(self.current_step + 1, self.total_steps)
        self._emit(label, detail=detail)

    def log(self, label: str, detail: str | None = None) -> None:
        self._emit(label, detail=detail)


def _format_seconds(seconds: float) -> str:
    return f"{float(seconds):.2f}s"


def _should_emit_loop_progress(current: int, total: int, segments: int = 10) -> bool:
    total_value = max(int(total), 1)
    current_value = int(current)
    emit_step = max(1, total_value // max(int(segments), 1))
    return current_value == 1 or current_value == total_value or current_value % emit_step == 0


def _emit_loop_progress(
    progress_hook: Callable[[str, str | None], None] | None,
    label: str,
    current: int,
    total: int,
    detail: str | None = None,
) -> None:
    if progress_hook is None or not _should_emit_loop_progress(current, total):
        return
    suffix = f"{int(current)}/{max(int(total), 1)}"
    message = suffix if not detail else f"{suffix}, {detail}"
    progress_hook(label, message)


def _subsample_row_indices(row_count: int, sample_cap: int | None, seed: int) -> np.ndarray:
    total_rows = int(max(row_count, 0))
    if total_rows <= 0:
        return np.zeros(0, dtype=np.int64)
    if sample_cap is None or int(sample_cap) <= 0 or total_rows <= int(sample_cap):
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_rows, size=int(sample_cap), replace=False).astype(np.int64))


def _prepare_group_indices(
    df: pd.DataFrame,
    *,
    include_layer: bool = True,
) -> tuple[pd.DataFrame, list[tuple[Any, np.ndarray]]]:
    work_df = df
    group_cols: list[str] = []
    has_layer_info = any(
        column in work_df.columns
        for column in (LAYER_SURFACE_PAIR_KEY_COL, "TopSurfaceCode", "BaseSurfaceCode", "GeoIntervalKey")
    )
    if include_layer and has_layer_info:
        if work_df is df:
            work_df = df.copy()
        work_df[LAYER_SURFACE_PAIR_KEY_COL] = _preferred_layer_series(work_df, prefer_unit_segment=False)
        group_cols.append(LAYER_SURFACE_PAIR_KEY_COL)
    if "FractureSet" in work_df.columns:
        group_cols.append("FractureSet")
    if not group_cols:
        return work_df, []
    grouped_indices = work_df.groupby(group_cols, dropna=False, sort=False).indices
    group_items = [
        (group_key, np.asarray(position_idx, dtype=np.int64))
        for group_key, position_idx in grouped_indices.items()
    ]
    return work_df, group_items


def _build_spatial_bucket_index(
    coords: np.ndarray,
    cell_size: float,
) -> tuple[dict[tuple[int, ...], int], list[tuple[int, ...]], list[np.ndarray]]:
    if len(coords) == 0:
        return {}, [], []
    safe_cell_size = float(max(cell_size, 1e-6))
    cell_index = np.floor(np.asarray(coords, dtype=float) / safe_cell_size).astype(np.int64)
    unique_cells, inverse = np.unique(cell_index, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    sorted_inverse = inverse[order]
    starts = np.concatenate([[0], np.flatnonzero(sorted_inverse[1:] != sorted_inverse[:-1]) + 1])
    ends = np.concatenate([starts[1:], [len(order)]])

    bucket_lookup: dict[tuple[int, ...], int] = {}
    bucket_cells: list[tuple[int, ...]] = []
    bucket_points: list[np.ndarray] = []
    for start, end in zip(starts, ends):
        bucket_id = int(sorted_inverse[start])
        cell_key = tuple(int(value) for value in unique_cells[bucket_id].tolist())
        bucket_lookup[cell_key] = len(bucket_cells)
        bucket_cells.append(cell_key)
        bucket_points.append(order[start:end])
    return bucket_lookup, bucket_cells, bucket_points


def _iter_bucket_pair_ids(
    bucket_lookup: dict[tuple[int, ...], int],
    bucket_cells: list[tuple[int, ...]],
) -> list[tuple[int, int]]:
    if not bucket_cells:
        return []
    dims = len(bucket_cells[0])
    neighbor_offsets = list(product((-1, 0, 1), repeat=dims))
    pairs: list[tuple[int, int]] = []
    for bucket_id_a, cell in enumerate(bucket_cells):
        for offset in neighbor_offsets:
            neighbor_cell = tuple(cell[dim] + offset[dim] for dim in range(dims))
            bucket_id_b = bucket_lookup.get(neighbor_cell)
            if bucket_id_b is None or bucket_id_b < bucket_id_a:
                continue
            pairs.append((bucket_id_a, bucket_id_b))
    return pairs


def _pairwise_radius_mask_torch(
    left_points: np.ndarray,
    right_points: np.ndarray,
    radius_squared: float,
    device: torch.device,
) -> torch.Tensor:
    left_tensor = torch.as_tensor(left_points, dtype=torch.float32, device=device)
    right_tensor = torch.as_tensor(right_points, dtype=torch.float32, device=device)
    diff = left_tensor[:, None, :] - right_tensor[None, :, :]
    dist_squared = torch.sum(diff * diff, dim=-1)
    return dist_squared <= float(radius_squared)


def _radius_neighbor_counts_gpu(
    coords: np.ndarray,
    radius: float,
    gpu_tile_points: int,
) -> np.ndarray:
    row_count = int(len(coords))
    if row_count <= 0:
        return np.zeros(0, dtype=np.int32)
    device = torch.device("cuda")
    radius_squared = float(radius) * float(radius)
    tile_points = max(int(gpu_tile_points), 256)
    counts = np.zeros(row_count, dtype=np.int64)
    bucket_lookup, bucket_cells, bucket_points = _build_spatial_bucket_index(coords, cell_size=radius)

    for bucket_id_a, bucket_id_b in _iter_bucket_pair_ids(bucket_lookup, bucket_cells):
        point_idx_a = bucket_points[bucket_id_a]
        point_idx_b = bucket_points[bucket_id_b]
        same_bucket = bucket_id_a == bucket_id_b
        if len(point_idx_a) == 0 or len(point_idx_b) == 0:
            continue

        for start_a in range(0, len(point_idx_a), tile_points):
            end_a = min(start_a + tile_points, len(point_idx_a))
            block_idx_a = point_idx_a[start_a:end_a]
            start_b_base = start_a if same_bucket else 0
            for start_b in range(start_b_base, len(point_idx_b), tile_points):
                end_b = min(start_b + tile_points, len(point_idx_b))
                block_idx_b = point_idx_b[start_b:end_b]
                mask = _pairwise_radius_mask_torch(
                    left_points=coords[block_idx_a],
                    right_points=coords[block_idx_b],
                    radius_squared=radius_squared,
                    device=device,
                )
                if same_bucket and start_a == start_b:
                    diag_length = min(mask.shape[0], mask.shape[1])
                    diag_index = torch.arange(diag_length, device=mask.device)
                    mask[diag_index, diag_index] = False
                    counts[block_idx_a] += mask.sum(dim=1).detach().cpu().numpy().astype(np.int64)
                else:
                    counts[block_idx_a] += mask.sum(dim=1).detach().cpu().numpy().astype(np.int64)
                    counts[block_idx_b] += mask.sum(dim=0).detach().cpu().numpy().astype(np.int64)
    return counts.astype(np.int32)


def _radius_neighbor_counts_cpu(coords: np.ndarray, radius: float) -> np.ndarray:
    if len(coords) <= 0:
        return np.zeros(0, dtype=np.int32)
    tree = cKDTree(np.asarray(coords, dtype=float))
    return np.asarray(tree.query_ball_point(coords, r=float(radius), return_length=True), dtype=np.int32) - 1


def _radius_neighbor_counts(
    coords: np.ndarray,
    radius: float,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
) -> tuple[np.ndarray, str]:
    resolved_backend = _resolve_compute_backend(compute_backend)
    if resolved_backend == "gpu":
        return _radius_neighbor_counts_gpu(coords, radius=float(radius), gpu_tile_points=int(gpu_tile_points)), resolved_backend
    return _radius_neighbor_counts_cpu(coords, radius=float(radius)), resolved_backend


def _radius_candidate_pairs_cpu(coords: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) <= 1:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    tree = cKDTree(np.asarray(coords, dtype=float))
    try:
        pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
    except TypeError:  # pragma: no cover
        pairs = np.asarray(list(tree.query_pairs(r=float(radius))), dtype=np.int64)
    if pairs.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return pairs[:, 0].astype(np.int64, copy=False), pairs[:, 1].astype(np.int64, copy=False)


def _radius_candidate_pairs_gpu(
    coords: np.ndarray,
    radius: float,
    gpu_tile_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = int(len(coords))
    if row_count <= 1:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    device = torch.device("cuda")
    radius_squared = float(radius) * float(radius)
    tile_points = max(int(gpu_tile_points), 256)
    bucket_lookup, bucket_cells, bucket_points = _build_spatial_bucket_index(coords, cell_size=radius)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []

    for bucket_id_a, bucket_id_b in _iter_bucket_pair_ids(bucket_lookup, bucket_cells):
        point_idx_a = bucket_points[bucket_id_a]
        point_idx_b = bucket_points[bucket_id_b]
        same_bucket = bucket_id_a == bucket_id_b
        if len(point_idx_a) == 0 or len(point_idx_b) == 0:
            continue

        for start_a in range(0, len(point_idx_a), tile_points):
            end_a = min(start_a + tile_points, len(point_idx_a))
            block_idx_a = point_idx_a[start_a:end_a]
            start_b_base = start_a if same_bucket else 0
            for start_b in range(start_b_base, len(point_idx_b), tile_points):
                end_b = min(start_b + tile_points, len(point_idx_b))
                block_idx_b = point_idx_b[start_b:end_b]
                mask = _pairwise_radius_mask_torch(
                    left_points=coords[block_idx_a],
                    right_points=coords[block_idx_b],
                    radius_squared=radius_squared,
                    device=device,
                )
                if same_bucket and start_a == start_b:
                    mask = torch.triu(mask, diagonal=1)
                matched = torch.nonzero(mask, as_tuple=False)
                if matched.numel() <= 0:
                    continue
                matched_np = matched.detach().cpu().numpy()
                left_parts.append(block_idx_a[matched_np[:, 0]])
                right_parts.append(block_idx_b[matched_np[:, 1]])

    if not left_parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(left_parts).astype(np.int64), np.concatenate(right_parts).astype(np.int64)


def _radius_candidate_pairs(
    coords: np.ndarray,
    radius: float,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
) -> tuple[np.ndarray, np.ndarray, str]:
    resolved_backend = _resolve_compute_backend(compute_backend)
    if resolved_backend == "gpu":
        left_idx, right_idx = _radius_candidate_pairs_gpu(coords, radius=float(radius), gpu_tile_points=int(gpu_tile_points))
        return left_idx, right_idx, resolved_backend
    left_idx, right_idx = _radius_candidate_pairs_cpu(coords, radius=float(radius))
    return left_idx, right_idx, resolved_backend


def _pairwise_mean_axial_azimuth_deg(azimuth_a: np.ndarray, azimuth_b: np.ndarray) -> np.ndarray:
    azimuth_a = np.asarray(azimuth_a, dtype=float)
    azimuth_b = np.asarray(azimuth_b, dtype=float)
    rad_a = np.deg2rad(2.0 * azimuth_a)
    rad_b = np.deg2rad(2.0 * azimuth_b)
    mean_rad = np.arctan2(np.sin(rad_a) + np.sin(rad_b), np.cos(rad_a) + np.cos(rad_b)) / 2.0
    return np.mod(np.rad2deg(mean_rad), 180.0)


# ---------------------------------------------------------------------------
#   Step 1: 裂缝组识别 (候选, 非硬分类)
# ---------------------------------------------------------------------------

def fracture_set_clustering(
    df: pd.DataFrame,
    max_sets: int = 6,
    min_sets: int = 2,
    n_sets: int | None = None,
    bic_sample_cap: int = 250000,
    fit_sample_cap: int = 400000,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    """用 GMM 在 (Azimuth, Dip) 空间做候选裂缝组识别。

    地球物理依据: Anderson 断裂理论 — 区域应力场产生 2-3 组共轭裂缝,
    每组的走向/倾角在统计上聚集。

    注意: GMM 结果仅作为候选组标签, SetProbability 反映归属可信度,
    不直接修改任何原始属性。低 SetProbability 的裂缝可能属于局部异常,
    不应被强制归组。
    """
    df = df.copy()
    az = df["Azimuth"].values.astype(float)
    dip = df["Dip"].values.astype(float)

    # 走向 180° 对称 → sin/cos 嵌入
    rad = np.deg2rad(2.0 * az)
    features = np.column_stack([np.cos(rad), np.sin(rad), dip / 90.0])
    if progress_hook is not None:
        progress_hook(
            "裂缝组识别数据规模",
            f"rows={len(features)}, bic_rows={len(features)}, fit_rows={len(features)}",
        )

    if n_sets is None:
        best_bic, best_k = np.inf, min_sets
        k_values = list(range(min_sets, max_sets + 1))
        for k_idx, k in enumerate(k_values, start=1):
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  n_init=3, random_state=42, max_iter=200)
            gmm.fit(features)
            bic = gmm.bic(features)
            if bic < best_bic:
                best_bic, best_k = bic, k
            if progress_hook is not None:
                progress_hook(
                    "裂缝组识别 BIC 进度",
                    f"{k_idx}/{len(k_values)}, k={k}, best_k={best_k}",
                )
        n_sets = best_k

    gmm = GaussianMixture(n_components=n_sets, covariance_type="full",
                          n_init=5, random_state=42, max_iter=300)
    gmm.fit(features)
    labels = gmm.predict(features)
    probs = gmm.predict_proba(features)
    max_probs = probs[np.arange(len(labels)), labels]

    df["FractureSet"] = labels.astype(int)
    df["SetProbability"] = max_probs.astype(float)
    df.attrs["fracture_set_bic_sample_size"] = int(len(features))
    df.attrs["fracture_set_fit_sample_size"] = int(len(features))
    return df


# ---------------------------------------------------------------------------
#   Step 2: 组内局部产状平滑 (仅写参考字段, 不覆盖原值)
# ---------------------------------------------------------------------------

def regional_orientation_smoothing(
    df: pd.DataFrame,
    bandwidth_xy: float = 150.0,
    bandwidth_z: float = 20.0,
    blend_alpha: float = 0.4,
    local_radius_factor: float = 3.0,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    """Nadaraya-Watson 核回归, 仅在同组内做局部产状趋势估计。

    地球物理依据: 地下应力场空间连续, 同组裂缝的产状不应突变。

    关键约束:
    - 只在同一 FractureSet 内做平滑
    - 平滑结果写入 SmoothedAzimuth / SmoothedDip (参考字段)
    - 原始 Azimuth / Dip 始终保留不变
    - blend_alpha 控制趋势面权重, 建议 ≤ 0.5
    """
    df = df.copy()
    df["OrigAzimuth"] = df["Azimuth"].values.copy()
    df["OrigDip"] = df["Dip"].values.copy()
    df["SmoothedAzimuth"] = df["Azimuth"].values.copy().astype(float)
    df["SmoothedDip"] = df["Dip"].values.copy().astype(float)

    cx = df["CenterX"].values.astype(float)
    cy = df["CenterY"].values.astype(float)
    cz = df["CenterTIME"].values.astype(float)
    conf = df["Confidence"].values.astype(float) if "Confidence" in df.columns else np.ones(len(df))

    set_ids = list(df["FractureSet"].unique())
    total_groups = len(set_ids)
    for group_idx, set_id in enumerate(set_ids, start=1):
        mask = df["FractureSet"].values == set_id
        idx = np.where(mask)[0]
        if len(idx) < 3:
            _emit_loop_progress(
                progress_hook,
                "产状平滑组进度",
                group_idx,
                total_groups,
                f"group_size={len(idx)}, skipped_small_group=1",
            )
            continue

        coords = np.column_stack([cx[idx], cy[idx], cz[idx]])
        az_vals = df["Azimuth"].values[idx].astype(float)
        dip_vals = df["Dip"].values[idx].astype(float)
        w_conf = conf[idx].copy()
        az_rad = np.deg2rad(2.0 * az_vals)
        az_cos = np.cos(az_rad)
        az_sin = np.sin(az_rad)

        smoothed_az = np.empty(len(idx))
        smoothed_dip = np.empty(len(idx))

        for i in range(len(idx)):
            dx = coords[:, 0] - coords[i, 0]
            dy = coords[:, 1] - coords[i, 1]
            dz = coords[:, 2] - coords[i, 2]
            dist_sq = (dx / bandwidth_xy) ** 2 + (dy / bandwidth_xy) ** 2 + (dz / bandwidth_z) ** 2
            kernel = np.exp(-0.5 * dist_sq) * w_conf
            kernel[i] = 0.0
            w_sum = kernel.sum()
            if w_sum < 1e-12:
                smoothed_az[i] = az_vals[i]
                smoothed_dip[i] = dip_vals[i]
                continue

            mean_cos = np.dot(kernel, az_cos) / w_sum
            mean_sin = np.dot(kernel, az_sin) / w_sum
            trend_az = (np.rad2deg(np.arctan2(mean_sin, mean_cos)) / 2.0) % 180.0
            trend_dip = np.dot(kernel, dip_vals) / w_sum

            az_diff = _azimuth_diff(np.array([az_vals[i]]), np.array([trend_az]))[0]
            if az_diff < 90.0:
                blended_az = _azimuth_mean_weighted(
                    np.array([az_vals[i], trend_az]),
                    np.array([1.0 - blend_alpha, blend_alpha]),
                )
            else:
                blended_az = az_vals[i]

            blended_dip = (1.0 - blend_alpha) * dip_vals[i] + blend_alpha * trend_dip
            smoothed_az[i] = blended_az
            smoothed_dip[i] = float(np.clip(blended_dip, 0.0, 90.0))

        df.loc[df.index[idx], "SmoothedAzimuth"] = smoothed_az
        df.loc[df.index[idx], "SmoothedDip"] = smoothed_dip
        _emit_loop_progress(
            progress_hook,
            "产状平滑组进度",
            group_idx,
            total_groups,
            f"group_size={len(idx)}",
        )
    return df


# ---------------------------------------------------------------------------
#   Step 3: 裂缝走廊检测 (作为证据, 非连接许可)
# ---------------------------------------------------------------------------

def fracture_corridor_detection(
    df: pd.DataFrame,
    corridor_search_radius: float = 200.0,
    corridor_min_patches: int = 5,
    along_strike_weight: float = 2.0,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    """检测裂缝走廊 — 沿走向排列的裂缝密集带。

    地球物理依据: 裂缝沿主应力或构造带成群排列, 形成线性走廊。

    用途限定:
    - 走廊作为"候选走廊识别" → CorridorSupport 证据字段
    - 走廊增强可靠性评分权重
    - 走廊为跨界补接提供优先约束
     - 不直接等于"必须连接"的许可
    """
    df = df.copy()
    df["CorridorID"] = -1

    corridor_counter = 0
    set_ids = list(df["FractureSet"].unique())
    total_groups = len(set_ids)
    for group_idx, set_id in enumerate(set_ids, start=1):
        mask = df["FractureSet"].values == set_id
        idx = np.where(mask)[0]
        if len(idx) < corridor_min_patches:
            _emit_loop_progress(
                progress_hook,
                "走廊识别组进度",
                group_idx,
                total_groups,
                f"group_size={len(idx)}, skipped_small_group=1",
            )
            continue

        az_vals = df["Azimuth"].values[idx].astype(float)
        mean_az = _azimuth_mean_weighted(az_vals, np.ones(len(az_vals)))
        mean_az_rad = np.deg2rad(mean_az)

        cx = df["CenterX"].values[idx].astype(float)
        cy = df["CenterY"].values[idx].astype(float)
        s = cx * np.cos(mean_az_rad) + cy * np.sin(mean_az_rad)
        p = -cx * np.sin(mean_az_rad) + cy * np.cos(mean_az_rad)

        proj_coords = np.column_stack([p, s / along_strike_weight])
        clustering = DBSCAN(eps=corridor_search_radius, min_samples=corridor_min_patches).fit(proj_coords)
        labels = clustering.labels_
        for c_label in set(labels):
            if c_label == -1:
                continue
            c_idx = idx[labels == c_label]
            df.loc[df.index[c_idx], "CorridorID"] = corridor_counter
            corridor_counter += 1
        _emit_loop_progress(
            progress_hook,
            "走廊识别组进度",
            group_idx,
            total_groups,
            f"group_size={len(idx)}, corridor_counter={corridor_counter}",
        )

    df["CorridorSupport"] = (df["CorridorID"] >= 0).astype(int)
    return df


# ---------------------------------------------------------------------------
#   Step 3b: 裂缝层级分类 (为多尺度聚合做准备)
# ---------------------------------------------------------------------------

def _reliability_to_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    out = np.zeros(len(arr), dtype=float)
    for i, value in enumerate(arr):
        if value in ("high", 2, "2"):
            out[i] = 2.0
        elif value in ("medium", 1, "1"):
            out[i] = 1.0
        else:
            out[i] = 0.0
    return out


def _get_work_orientation(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "SmoothedAzimuth" in df.columns:
        az = pd.to_numeric(df["SmoothedAzimuth"], errors="coerce").to_numpy(dtype=float)
    else:
        az = pd.to_numeric(df["Azimuth"], errors="coerce").to_numpy(dtype=float)
    if "SmoothedDip" in df.columns:
        dip = pd.to_numeric(df["SmoothedDip"], errors="coerce").to_numpy(dtype=float)
    else:
        dip = pd.to_numeric(df["Dip"], errors="coerce").to_numpy(dtype=float)

    raw_az = pd.to_numeric(df.get("Azimuth"), errors="coerce").to_numpy(dtype=float) if "Azimuth" in df.columns else np.zeros(len(df))
    raw_dip = pd.to_numeric(df.get("Dip"), errors="coerce").to_numpy(dtype=float) if "Dip" in df.columns else np.full(len(df), 45.0)
    az = np.where(np.isfinite(az), az, raw_az)
    dip = np.where(np.isfinite(dip), dip, raw_dip)
    az = np.where(np.isfinite(az), az, 0.0) % 180.0
    dip = np.clip(np.where(np.isfinite(dip), dip, 45.0), 0.0, 90.0)
    return az.astype(float), dip.astype(float)


def _project_to_strike_frame(x: np.ndarray, y: np.ndarray, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    az_rad = np.deg2rad(float(azimuth_deg))
    s = x * np.cos(az_rad) + y * np.sin(az_rad)
    p = -x * np.sin(az_rad) + y * np.cos(az_rad)
    return s.astype(float), p.astype(float)


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if len(series) == 0:
        return np.array([], dtype=float)
    ranked = series.rank(method="average", pct=True).to_numpy(dtype=float)
    return np.nan_to_num(ranked, nan=0.0, posinf=1.0, neginf=0.0)


def assign_scale_classes(
    df: pd.DataFrame,
    *,
    scale_major_radius: float = 160.0,
    scale_minor_radius: float = 25.0,
    scale_z_radius: float = 20.0,
    macro_score_quantile: float = 0.85,
    macro_max_fraction: float = 0.10,
    meso_score_quantile: float = 0.45,
    macro_min_span: float = 80.0,
    meso_min_span: float = 30.0,
    macro_min_neighbors: int = 6,
    meso_min_neighbors: int = 3,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    """按同层位、同裂缝组内的连通潜力给裂缝片分配尺度类别。

    目标不是直接按当前尺寸分大中小，而是识别：
    - macro_core: 具有成长为主干裂缝潜力的核心片
    - meso_link : 适合做中尺度连通的过渡片
    - micro_bg  : 背景微裂缝，保留小尺度
    """
    df = df.copy()
    work_az, work_dip = _get_work_orientation(df)
    df["WorkAzimuth"] = work_az
    df["WorkDip"] = work_dip

    patch_len = pd.to_numeric(df.get("PatchLength"), errors="coerce").fillna(10.0).to_numpy(dtype=float)
    patch_hgt = pd.to_numeric(df.get("PatchHeight"), errors="coerce").fillna(10.0).to_numpy(dtype=float)
    df["PatchArea"] = patch_len * patch_hgt
    if "AggregationMode" not in df.columns:
        df["AggregationMode"] = "original"
    if "ParentPatchCount" not in df.columns:
        df["ParentPatchCount"] = 1
    if "ScaleClass" not in df.columns:
        df["ScaleClass"] = "micro_bg"
    df["AlongNeighborCount"] = 0
    df["AlongSpan"] = 0.0
    df["AcrossSpread"] = 0.0
    df["HierarchyScore"] = 0.0
    df["AreaRankInSet"] = 0.0

    group_cols = ["FractureSet"]
    if LAYER_SURFACE_PAIR_KEY_COL in df.columns or "TopSurfaceCode" in df.columns or "BaseSurfaceCode" in df.columns or "GeoIntervalKey" in df.columns:
        df[LAYER_SURFACE_PAIR_KEY_COL] = _preferred_layer_series(df, prefer_unit_segment=False)
        group_cols.insert(0, LAYER_SURFACE_PAIR_KEY_COL)

    groups = list(df.groupby(group_cols, dropna=False))
    total_groups = len(groups)
    for group_idx, (_, group) in enumerate(groups, start=1):
        idx = group.index
        if len(idx) == 0:
            continue

        cx = pd.to_numeric(group["CenterX"], errors="coerce").to_numpy(dtype=float)
        cy = pd.to_numeric(group["CenterY"], errors="coerce").to_numpy(dtype=float)
        cz = pd.to_numeric(group["CenterTIME"], errors="coerce").to_numpy(dtype=float)
        gaz = pd.to_numeric(group["WorkAzimuth"], errors="coerce").to_numpy(dtype=float)
        gconf = pd.to_numeric(group.get("Confidence"), errors="coerce").fillna(0.5).to_numpy(dtype=float)
        ggeo = pd.to_numeric(group.get("GeophysicsScore"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        gset = pd.to_numeric(group.get("SetProbability"), errors="coerce").fillna(0.5).to_numpy(dtype=float)
        grel = _reliability_to_numeric(group.get("ReliabilityLevel", pd.Series(["medium"] * len(group), index=group.index)))
        gcorr = pd.to_numeric(group.get("CorridorSupport"), errors="coerce").fillna(0).to_numpy(dtype=float)
        garea = pd.to_numeric(group["PatchArea"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        glen = pd.to_numeric(group.get("PatchLength"), errors="coerce").fillna(10.0).to_numpy(dtype=float)

        mean_az = _azimuth_mean_weighted(gaz, np.maximum(gconf, 1e-6)) if len(group) > 1 else float(gaz[0])
        s_proj, p_proj = _project_to_strike_frame(cx, cy, mean_az)

        along_neighbor = np.zeros(len(group), dtype=int)
        along_span = np.zeros(len(group), dtype=float)
        across_spread = np.zeros(len(group), dtype=float)
        for i in range(len(group)):
            ds = np.abs(s_proj - s_proj[i])
            dp = np.abs(p_proj - p_proj[i])
            dz = np.abs(cz - cz[i])
            local_mask = (
                (ds <= scale_major_radius)
                & (dp <= scale_minor_radius)
                & (dz <= scale_z_radius)
            )
            local_mask[i] = True
            neighbor_mask = local_mask.copy()
            neighbor_mask[i] = False
            along_neighbor[i] = int(neighbor_mask.sum())
            local_s = s_proj[local_mask]
            local_p = p_proj[local_mask]
            if len(local_s) > 0:
                along_span[i] = float(local_s.max() - local_s.min() + glen[i])
                across_spread[i] = float(np.std(local_p)) if len(local_p) > 1 else 0.0

        q_span = _rank01(along_span)
        q_neighbor = _rank01(along_neighbor.astype(float))
        q_area = _rank01(garea)
        q_across_bad = _rank01(across_spread)
        q_conf = _rank01(gconf)
        rel_norm = grel / 2.0

        score = (
            0.30 * q_span
            + 0.25 * q_neighbor
            + 0.10 * q_area
            + 0.10 * np.clip(ggeo, 0.0, 1.0)
            + 0.10 * np.clip(gset, 0.0, 1.0)
            + 0.05 * q_conf
            + 0.10 * np.clip(gcorr, 0.0, 1.0)
            + 0.05 * rel_norm
            - 0.15 * q_across_bad
        )
        score = np.clip(score, 0.0, 1.0)

        macro_neighbor_gate = max(macro_min_neighbors, int(np.nanpercentile(along_neighbor, 70)) if len(group) >= 4 else macro_min_neighbors)
        macro_span_gate = max(macro_min_span, float(np.nanpercentile(along_span, 70)) if len(group) >= 4 else macro_min_span)
        macro_score_gate = float(np.nanquantile(score, macro_score_quantile)) if len(group) >= 4 else 0.75
        macro_candidates = (
            (gcorr > 0)
            & (grel >= 1.0)
            & (along_neighbor >= macro_neighbor_gate)
            & (along_span >= macro_span_gate)
            & (score >= macro_score_gate)
        )

        macro_labels = np.zeros(len(group), dtype=bool)
        if macro_candidates.any():
            macro_limit = max(1, int(np.ceil(len(group) * macro_max_fraction)))
            macro_limit = min(macro_limit, int(macro_candidates.sum()))
            macro_order = np.argsort(score[macro_candidates])[::-1][:macro_limit]
            macro_positions = np.where(macro_candidates)[0][macro_order]
            macro_labels[macro_positions] = True

        meso_score_gate = float(np.nanquantile(score, meso_score_quantile)) if len(group) >= 4 else 0.35
        meso_labels = (
            ~macro_labels
            & (score >= meso_score_gate)
            & (along_neighbor >= meso_min_neighbors)
            & ((along_span >= meso_min_span) | (gcorr > 0))
        )

        scale_class = np.full(len(group), "micro_bg", dtype=object)
        scale_class[meso_labels] = "meso_link"
        scale_class[macro_labels] = "macro_core"

        df.loc[idx, "AlongNeighborCount"] = along_neighbor
        df.loc[idx, "AlongSpan"] = along_span
        df.loc[idx, "AcrossSpread"] = across_spread
        df.loc[idx, "HierarchyScore"] = score
        df.loc[idx, "AreaRankInSet"] = q_area
        df.loc[idx, "ScaleClass"] = scale_class

        _emit_loop_progress(
            progress_hook,
            "尺度分类组进度",
            group_idx,
            total_groups,
            f"group_size={len(group)}",
        )
    return df


# ---------------------------------------------------------------------------
#   Step 4: 多维度可靠性评分 (证据层 + 结论层)
# ---------------------------------------------------------------------------

def multi_dimensional_reliability_scoring(
    df: pd.DataFrame,
    neighbor_radius_xy: float = 100.0,
    neighbor_radius_z: float = 15.0,
    min_neighbors: int = 2,
    confidence_floor: float = 0.3,
    # 几何合理性参数
    length_range: tuple[float, float] = (1.0, 200.0),
    height_range: tuple[float, float] = (0.5, 100.0),
    # 层位参数 (预留)
    known_fracture_layers: list[str] | None = None,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
) -> pd.DataFrame:
    """多维度证据评分 + 综合结论判定。

    证据层 — 每个维度独立计算, 可在 VTK 中分别着色查看:
      GeometryScore     : 裂缝尺寸是否在合理范围内
      GeophysicsScore   : 空间邻域密度 (裂缝成群分布的地质规律)
      CorridorSupport   : 是否属于裂缝走廊 (由 Step 3 设置)
      StratigraphyScore : 层位一致性 (是否在已知裂缝发育层段内, 预留)
      FaultPenalty      : 穿越断层惩罚 (预留, 需断层数据)

    结论层:
      ReliabilityLevel  : "high" / "medium" / "low"
      ConnectionType    : "original" (所有原始预测初始化为此值)
    """
    df = df.copy()

    # --- 证据层 ---

    # 1) GeometryScore: 尺寸合理性
    length = df["PatchLength"].values.astype(float) if "PatchLength" in df.columns else np.full(len(df), 10.0)
    height = df["PatchHeight"].values.astype(float) if "PatchHeight" in df.columns else np.full(len(df), 10.0)
    length_ok = ((length >= length_range[0]) & (length <= length_range[1])).astype(float)
    height_ok = ((height >= height_range[0]) & (height <= height_range[1])).astype(float)
    aspect_ratio = np.where(height > 1e-6, length / height, 1.0)
    aspect_ok = ((aspect_ratio >= 0.1) & (aspect_ratio <= 20.0)).astype(float)
    df["GeometryScore"] = ((length_ok + height_ok + aspect_ok) / 3.0).astype(float)

    # 2) GeophysicsScore: 邻域密度
    cx = df["CenterX"].values.astype(float)
    cy = df["CenterY"].values.astype(float)
    cz = df["CenterTIME"].values.astype(float)

    scale_z = neighbor_radius_xy / max(neighbor_radius_z, 1e-6)
    coords_scaled = np.column_stack([cx, cy, cz * scale_z])
    neighbor_counts, resolved_backend = _radius_neighbor_counts(
        coords=coords_scaled,
        radius=float(neighbor_radius_xy),
        compute_backend=compute_backend,
        gpu_tile_points=int(gpu_tile_points),
    )
    max_count = max(int(neighbor_counts.max()), 1)
    df["NeighborCount"] = neighbor_counts.astype(int)
    df["GeophysicsScore"] = np.clip(neighbor_counts / max_count, 0.0, 1.0).astype(float)
    df.attrs["compute_backend"] = resolved_backend

    # 3) CorridorSupport: 已在 Step 3 中设置, 确保存在
    if "CorridorSupport" not in df.columns:
        df["CorridorSupport"] = 0

    # 4) StratigraphyScore: 层位一致性 (预留, 默认全 1.0)
    if "StratigraphyScore" not in df.columns:
        if known_fracture_layers:
            df["StratigraphyScore"] = _preferred_layer_series(df, prefer_unit_segment=False).isin(known_fracture_layers).astype(float)
        else:
            df["StratigraphyScore"] = 1.0

    # 5) FaultPenalty: 穿越断层惩罚 (预留, 默认 0 = 无惩罚)
    if "FaultPenalty" not in df.columns:
        df["FaultPenalty"] = 0.0

    # --- 结论层 ---

    conf = df["Confidence"].values.astype(float) if "Confidence" in df.columns else np.ones(len(df))
    set_prob = df["SetProbability"].values.astype(float) if "SetProbability" in df.columns else np.ones(len(df))
    geo_score = df["GeophysicsScore"].values
    geom_score = df["GeometryScore"].values
    corridor = df["CorridorSupport"].values.astype(float)
    strat_score = df["StratigraphyScore"].values.astype(float)
    fault_pen = df["FaultPenalty"].values.astype(float)

    # 综合评分 (加权, 各维度可独立审查)
    composite = (
        0.25 * conf
        + 0.20 * geo_score
        + 0.15 * geom_score
        + 0.15 * set_prob
        + 0.10 * corridor
        + 0.10 * strat_score
        - 0.05 * fault_pen
    )
    composite = np.clip(composite, 0.0, 1.0)

    # 结论: 三级可靠性
    levels = np.where(composite >= 0.6, "high",
             np.where(composite >= 0.35, "medium", "low"))

    # 孤立 + 低置信 + 无走廊 → 强制降为 low
    is_isolated = neighbor_counts < min_neighbors
    is_low_conf = conf < confidence_floor
    no_corridor = df["CorridorSupport"].values < 1
    force_low = is_isolated & is_low_conf & no_corridor
    levels[force_low] = "low"

    df["ReliabilityLevel"] = levels
    df["ConnectionType"] = "original"
    df["IsSupplemented"] = 0

    return df


# ---------------------------------------------------------------------------
#   Step 5a: 跨单元边界匹配 (Phase 1 — 仅标记, 不补片)
# ---------------------------------------------------------------------------

def boundary_match_only(
    df: pd.DataFrame,
    boundary_tol_xy: float = 25.0,
    alignment_tol_m: float = 50.0,
    azimuth_tol_deg: float = 20.0,
    dip_tol_deg: float = 12.0,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """在单元边界找到产状匹配的裂缝对, 标记 ConnectionType 但不插值补片。

    匹配条件: 同组 FractureSet + 同层 GeoIntervalKey + 产状相近 + 几何连续。
    返回 (df, matched_pairs) — matched_pairs 供 Phase 2 的补接使用。
    """
    df = df.copy()
    matched_pairs: list[dict[str, Any]] = []

    has_block = "BlockX" in df.columns and "BlockY" in df.columns
    has_unit = "UnitID" in df.columns
    if not has_block and not has_unit:
        return df, matched_pairs

    if has_block:
        unit_key_arr = np.array(
            [f"{int(block_x)}_{int(block_y)}" for block_x, block_y in zip(df["BlockX"].to_numpy(), df["BlockY"].to_numpy())],
            dtype=object,
        )
    else:
        unit_key_arr = df["UnitID"].fillna("").astype(str).to_numpy(dtype=object)

    cx_arr = pd.to_numeric(df["CenterX"], errors="coerce").to_numpy(dtype=float)
    cy_arr = pd.to_numeric(df["CenterY"], errors="coerce").to_numpy(dtype=float)
    az_arr = pd.to_numeric(df["Azimuth"], errors="coerce").to_numpy(dtype=float)
    dip_arr = pd.to_numeric(df["Dip"], errors="coerce").to_numpy(dtype=float)
    conf_series = df["Confidence"] if "Confidence" in df.columns else pd.Series([0.5] * len(df), index=df.index, dtype=float)
    conf_arr = pd.to_numeric(conf_series, errors="coerce").fillna(0.5).to_numpy(dtype=float)

    unit_bounds: dict[str, dict[str, float]] = {}
    for idx, unit_key in enumerate(unit_key_arr):
        if unit_key not in unit_bounds:
            unit_bounds[unit_key] = {
                "x_min": cx_arr[idx],
                "x_max": cx_arr[idx],
                "y_min": cy_arr[idx],
                "y_max": cy_arr[idx],
            }
        else:
            bounds = unit_bounds[unit_key]
            bounds["x_min"] = min(bounds["x_min"], cx_arr[idx])
            bounds["x_max"] = max(bounds["x_max"], cx_arr[idx])
            bounds["y_min"] = min(bounds["y_min"], cy_arr[idx])
            bounds["y_max"] = max(bounds["y_max"], cy_arr[idx])

    is_boundary = np.zeros(len(df), dtype=bool)
    for idx, unit_key in enumerate(unit_key_arr):
        bounds = unit_bounds[unit_key]
        if (
            abs(cx_arr[idx] - bounds["x_min"]) < boundary_tol_xy
            or abs(cx_arr[idx] - bounds["x_max"]) < boundary_tol_xy
            or abs(cy_arr[idx] - bounds["y_min"]) < boundary_tol_xy
            or abs(cy_arr[idx] - bounds["y_max"]) < boundary_tol_xy
        ):
            is_boundary[idx] = True

    boundary_idx = np.flatnonzero(is_boundary)
    if len(boundary_idx) < 2:
        return df, matched_pairs

    boundary_coords = np.column_stack([cx_arr[boundary_idx], cy_arr[boundary_idx]])
    pair_left_local, pair_right_local, _ = _radius_candidate_pairs(
        coords=boundary_coords,
        radius=float(alignment_tol_m),
        compute_backend=compute_backend,
        gpu_tile_points=int(gpu_tile_points),
    )
    if len(pair_left_local) <= 0:
        return df, matched_pairs

    pair_left = boundary_idx[pair_left_local]
    pair_right = boundary_idx[pair_right_local]
    unit_code, _ = pd.factorize(unit_key_arr, sort=False)
    layer_code, _ = pd.factorize(_preferred_layer_series(df, prefer_unit_segment=False), sort=False)

    keep_mask = unit_code[pair_left] != unit_code[pair_right]
    if "FractureSet" in df.columns:
        fracture_set_arr = pd.to_numeric(df["FractureSet"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        keep_mask &= fracture_set_arr[pair_left] == fracture_set_arr[pair_right]
    keep_mask &= layer_code[pair_left] == layer_code[pair_right]

    az_diff = _azimuth_diff(az_arr[pair_left], az_arr[pair_right])
    dip_diff = np.abs(dip_arr[pair_left] - dip_arr[pair_right])
    keep_mask &= az_diff <= float(azimuth_tol_deg)
    keep_mask &= dip_diff <= float(dip_tol_deg)

    dx = cx_arr[pair_right] - cx_arr[pair_left]
    dy = cy_arr[pair_right] - cy_arr[pair_left]
    mean_az_rad = np.deg2rad((az_arr[pair_left] + az_arr[pair_right]) / 2.0)
    perp_dist = np.abs(-dx * np.sin(mean_az_rad) + dy * np.cos(mean_az_rad))
    keep_mask &= perp_dist <= float(alignment_tol_m) * 0.5

    if not keep_mask.any():
        return df, matched_pairs

    kept_left = pair_left[keep_mask]
    kept_right = pair_right[keep_mask]
    kept_az_diff = az_diff[keep_mask]
    kept_dip_diff = dip_diff[keep_mask]
    kept_perp = perp_dist[keep_mask]
    kept_gap = np.hypot(dx[keep_mask], dy[keep_mask])

    matched_rows = np.unique(np.concatenate([kept_left, kept_right]))
    matched_labels = df.index.to_numpy()[matched_rows]
    df.loc[matched_labels, "ConnectionType"] = "boundary_matched"

    for left_idx, right_idx, pair_az_diff, pair_dip_diff, pair_gap, pair_perp in zip(
        kept_left,
        kept_right,
        kept_az_diff,
        kept_dip_diff,
        kept_gap,
        kept_perp,
    ):
        matched_pairs.append(
            {
                "idx_i": int(df.index[left_idx]),
                "idx_j": int(df.index[right_idx]),
                "unit_i": str(unit_key_arr[left_idx]),
                "unit_j": str(unit_key_arr[right_idx]),
                "azimuth_diff": float(pair_az_diff),
                "dip_diff": float(pair_dip_diff),
                "gap_distance_m": float(pair_gap),
                "perp_distance_m": float(pair_perp),
                "confidence_i": float(conf_arr[left_idx]),
                "confidence_j": float(conf_arr[right_idx]),
            }
        )

    return df, matched_pairs


# ---------------------------------------------------------------------------
#   Step 5b: 保守补接 (Phase 2 — 仅对高置信匹配对补片)
# ---------------------------------------------------------------------------

def conservative_boundary_supplement(
    df: pd.DataFrame,
    matched_pairs: list[dict[str, Any]],
    min_pair_confidence: float = 0.5,
    max_supplement_length: float = 80.0,
) -> pd.DataFrame:
    """对高置信度匹配对做保守补接。

    约束条件:
    - 匹配对两端置信度均 ≥ min_pair_confidence
    - 补片长度 (间距) ≤ max_supplement_length
    - 补充片必须标记 IsSupplemented=1, ConnectionType="supplemented"
    - 补片置信度取两端最小值 × 0.7 (衰减)
    """
    df = df.copy()
    new_patches: list[dict[str, Any]] = []

    for pair in matched_pairs:
        if pair["confidence_i"] < min_pair_confidence or pair["confidence_j"] < min_pair_confidence:
            continue
        if pair["gap_distance_m"] > max_supplement_length:
            continue

        idx_i, idx_j = pair["idx_i"], pair["idx_j"]
        if idx_i not in df.index or idx_j not in df.index:
            continue
        row_i = df.loc[idx_i]
        row_j = df.loc[idx_j]

        scale_i = str(row_i.get("ScaleClass", "meso_link"))
        scale_j = str(row_j.get("ScaleClass", "meso_link"))
        if scale_i == "micro_bg" and scale_j == "micro_bg":
            continue

        new_patch: dict[str, Any] = {}
        for col in df.columns:
            if col in ("CenterX", "CenterY", "CenterTIME", "Azimuth", "Dip",
                        "PatchLength", "PatchHeight"):
                new_patch[col] = (float(row_i[col]) + float(row_j[col])) / 2.0
            elif col.startswith("V") and len(col) >= 3 and col[1].isdigit():
                new_patch[col] = (float(row_i[col]) + float(row_j[col])) / 2.0
            elif col in ("SmoothedAzimuth", "SmoothedDip", "OrigAzimuth", "OrigDip"):
                new_patch[col] = (float(row_i[col]) + float(row_j[col])) / 2.0
            else:
                new_patch[col] = row_i[col]

        new_patch["Confidence"] = min(pair["confidence_i"], pair["confidence_j"]) * 0.7
        new_patch["IsSupplemented"] = 1
        new_patch["ConnectionType"] = "supplemented"
        new_patch["ReliabilityLevel"] = "medium"
        new_patch["FractureSet"] = int(row_i.get("FractureSet", 0))
        new_patch["CorridorSupport"] = max(
            int(row_i.get("CorridorSupport", 0)),
            int(row_j.get("CorridorSupport", 0)),
        )
        if scale_i == "macro_core" or scale_j == "macro_core":
            new_patch["ScaleClass"] = "macro_core"
        else:
            new_patch["ScaleClass"] = "meso_link"
        new_patch["AggregationMode"] = "boundary_bridge"
        new_patch["ParentPatchCount"] = 2
        new_patch["PatchArea"] = float(new_patch.get("PatchLength", 0.0)) * float(new_patch.get("PatchHeight", 0.0))
        new_patches.append(new_patch)

    if new_patches:
        new_df = pd.DataFrame(new_patches)
        df = pd.concat([df, new_df], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
#   Step 6: 裂缝片聚合 (同组相邻小片 → 大裂缝面)
# ---------------------------------------------------------------------------

def aggregate_patches(
    df: pd.DataFrame,
    cluster_radius: float = 20.0,
    cluster_min_patches: int = 3,
    azimuth_tol_deg: float = 25.0,
    dip_tol_deg: float = 15.0,
    min_confidence: float = 0.35,
    max_merged_length: float = 80.0,
    max_merged_height: float = 40.0,
    macro_major_radius: float = 140.0,
    macro_minor_radius: float = 24.0,
    macro_max_merged_length: float = 320.0,
    macro_max_merged_height: float = 90.0,
    scale_major_radius: float = 160.0,
    scale_minor_radius: float = 25.0,
    scale_z_radius: float = 20.0,
    macro_score_quantile: float = 0.85,
    macro_max_fraction: float = 0.10,
    meso_score_quantile: float = 0.45,
    macro_min_span: float = 80.0,
    meso_min_span: float = 30.0,
    macro_min_neighbors: int = 6,
    meso_min_neighbors: int = 3,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    """按裂缝层级做多尺度聚合。

    - macro_core: 允许长距离沿走向聚合，形成少量大裂缝
    - meso_link : 保持中尺度聚合
    - micro_bg  : 不参与聚合，保留背景微裂缝
    """
    df = df.copy()
    if "ScaleClass" not in df.columns:
        df = assign_scale_classes(
            df,
            scale_major_radius=scale_major_radius,
            scale_minor_radius=scale_minor_radius,
            scale_z_radius=scale_z_radius,
            macro_score_quantile=macro_score_quantile,
            macro_max_fraction=macro_max_fraction,
            meso_score_quantile=meso_score_quantile,
            macro_min_span=macro_min_span,
            meso_min_span=meso_min_span,
            macro_min_neighbors=macro_min_neighbors,
            meso_min_neighbors=meso_min_neighbors,
        )

    if "PatchArea" not in df.columns:
        df["PatchArea"] = pd.to_numeric(df["PatchLength"], errors="coerce").fillna(0.0) * pd.to_numeric(df["PatchHeight"], errors="coerce").fillna(0.0)
    if "AggregationMode" not in df.columns:
        df["AggregationMode"] = "original"
    if "ParentPatchCount" not in df.columns:
        df["ParentPatchCount"] = 1

    keep_mask = np.ones(len(df), dtype=bool)
    new_patches: list[dict[str, Any]] = []

    group_cols = ["FractureSet"]
    if LAYER_SURFACE_PAIR_KEY_COL in df.columns or "TopSurfaceCode" in df.columns or "BaseSurfaceCode" in df.columns or "GeoIntervalKey" in df.columns:
        df[LAYER_SURFACE_PAIR_KEY_COL] = _preferred_layer_series(df, prefer_unit_segment=False)
        group_cols.insert(0, LAYER_SURFACE_PAIR_KEY_COL)

    def _aggregate_scale(scale_class: str, *, major_radius: float, minor_radius: float, max_length: float, max_height: float, mode_name: str) -> None:
        nonlocal keep_mask, new_patches
        subset = df[
            (df["ScaleClass"].astype(str) == scale_class)
            & (pd.to_numeric(df.get("Confidence"), errors="coerce").fillna(1.0) >= min_confidence)
        ].copy()
        if subset.empty:
            return

        group_items = list(subset.groupby(group_cols, dropna=False))
        total_groups = len(group_items)
        for group_idx, (_, group) in enumerate(group_items, start=1):
            idx_labels = group.index.to_numpy()
            if len(idx_labels) < cluster_min_patches:
                _emit_loop_progress(
                    progress_hook,
                    f"裂缝片聚合组进度[{scale_class}]",
                    group_idx,
                    total_groups,
                    f"group_size={len(idx_labels)}, skipped_small_group=1",
                )
                continue

            c_cx = pd.to_numeric(group["CenterX"], errors="coerce").to_numpy(dtype=float)
            c_cy = pd.to_numeric(group["CenterY"], errors="coerce").to_numpy(dtype=float)
            c_cz = pd.to_numeric(group["CenterTIME"], errors="coerce").to_numpy(dtype=float)
            c_az = pd.to_numeric(group["WorkAzimuth"] if "WorkAzimuth" in group.columns else group["Azimuth"], errors="coerce").to_numpy(dtype=float)
            c_dip = pd.to_numeric(group["WorkDip"] if "WorkDip" in group.columns else group["Dip"], errors="coerce").to_numpy(dtype=float)
            c_conf = pd.to_numeric(group.get("Confidence"), errors="coerce").fillna(1.0).to_numpy(dtype=float)

            mean_az_group = _azimuth_mean_weighted(c_az, np.maximum(c_conf, 1e-6))
            s_proj, p_proj = _project_to_strike_frame(c_cx, c_cy, mean_az_group)
            az_rad = np.deg2rad(2.0 * c_az)
            features = np.column_stack([
                s_proj / max(major_radius, 1e-6),
                p_proj / max(minor_radius, 1e-6),
                c_cz / max(scale_z_radius, 1e-6),
                np.cos(az_rad) * 0.6,
                np.sin(az_rad) * 0.6,
                c_dip / max(dip_tol_deg, 1e-6) * 0.3,
            ])
            labels = DBSCAN(eps=2.0, min_samples=cluster_min_patches).fit(features).labels_

            for c_label in set(labels):
                if c_label == -1:
                    continue
                cluster_pos = np.where(labels == c_label)[0]
                if len(cluster_pos) < cluster_min_patches:
                    continue
                cluster_idx = idx_labels[cluster_pos]

                ccx = c_cx[cluster_pos]
                ccy = c_cy[cluster_pos]
                ccz = c_cz[cluster_pos]
                caz = c_az[cluster_pos]
                cdip = c_dip[cluster_pos]
                cconf = c_conf[cluster_pos]

                az_spread = _azimuth_diff(caz, np.full_like(caz, _azimuth_mean_weighted(caz, cconf)))
                if np.mean(az_spread) > azimuth_tol_deg:
                    continue

                w = cconf / max(cconf.sum(), 1e-6)
                mean_x = float(np.dot(w, ccx))
                mean_y = float(np.dot(w, ccy))
                mean_z = float(np.dot(w, ccz))
                mean_az = _azimuth_mean_weighted(caz, cconf)
                mean_dip = float(np.dot(w, cdip))
                mean_conf = float(np.mean(cconf))

                local_s, local_p = _project_to_strike_frame(ccx, ccy, mean_az)
                z_range = float(ccz.max() - ccz.min()) if len(ccz) > 0 else 0.0
                orig_mean_len = float(pd.to_numeric(df.loc[cluster_idx, "PatchLength"], errors="coerce").fillna(10.0).mean())
                orig_mean_hgt = float(pd.to_numeric(df.loc[cluster_idx, "PatchHeight"], errors="coerce").fillna(10.0).mean())
                raw_length = max(float(local_s.max() - local_s.min()) + orig_mean_len, orig_mean_len)
                raw_height = max(z_range + orig_mean_hgt, orig_mean_hgt)
                new_length = min(raw_length, max_length)
                new_height = min(raw_height, max_height)

                mean_az_rad = np.deg2rad(mean_az)
                dip_rad = np.deg2rad(mean_dip)
                u_hat = np.array([np.cos(mean_az_rad), np.sin(mean_az_rad), 0.0])
                perp_x, perp_y = -np.sin(mean_az_rad), np.cos(mean_az_rad)
                v_hat = np.array([
                    perp_x * np.cos(dip_rad),
                    perp_y * np.cos(dip_rad),
                    -np.sin(dip_rad),
                ])
                v_norm = np.linalg.norm(v_hat)
                v_hat = v_hat / v_norm if v_norm > 1e-6 else np.array([0.0, 0.0, -1.0])
                center = np.array([mean_x, mean_y, mean_z])
                half_u = new_length / 2.0
                half_v = new_height / 2.0
                nv1 = center - half_u * u_hat - half_v * v_hat
                nv2 = center + half_u * u_hat - half_v * v_hat
                nv3 = center + half_u * u_hat + half_v * v_hat
                nv4 = center - half_u * u_hat + half_v * v_hat

                ref_row = df.loc[cluster_idx[np.argmax(cconf)]].to_dict()
                new_patch: dict[str, Any] = {}
                for col in df.columns:
                    new_patch[col] = ref_row.get(col, 0)
                new_patch.update({
                    "CenterX": mean_x,
                    "CenterY": mean_y,
                    "CenterTIME": mean_z,
                    "Azimuth": mean_az,
                    "Dip": mean_dip,
                    "OrigAzimuth": mean_az,
                    "OrigDip": mean_dip,
                    "SmoothedAzimuth": mean_az,
                    "SmoothedDip": mean_dip,
                    "WorkAzimuth": mean_az,
                    "WorkDip": mean_dip,
                    "PatchLength": new_length,
                    "PatchHeight": new_height,
                    "PatchArea": new_length * new_height,
                    "Confidence": mean_conf,
                    "FractureSet": int(ref_row.get("FractureSet", 0)),
                    "V1X": nv1[0], "V1Y": nv1[1], "V1Z": nv1[2],
                    "V2X": nv2[0], "V2Y": nv2[1], "V2Z": nv2[2],
                    "V3X": nv3[0], "V3Y": nv3[1], "V3Z": nv3[2],
                    "V4X": nv4[0], "V4Y": nv4[1], "V4Z": nv4[2],
                    "NeighborCount": int(len(cluster_idx)),
                    "ParentPatchCount": int(len(cluster_idx)),
                    "ScaleClass": scale_class,
                    "AggregationMode": mode_name,
                    "ConnectionType": ref_row.get("ConnectionType", "original"),
                    "IsSupplemented": int(ref_row.get("IsSupplemented", 0)),
                    "HierarchyScore": float(np.mean(pd.to_numeric(df.loc[cluster_idx, "HierarchyScore"], errors="coerce").fillna(0.0))),
                })
                new_patches.append(new_patch)
                keep_mask[df.index.get_indexer(cluster_idx)] = False
            _emit_loop_progress(
                progress_hook,
                f"裂缝片聚合组进度[{scale_class}]",
                group_idx,
                total_groups,
                f"group_size={len(idx_labels)}",
            )

    _aggregate_scale(
        "macro_core",
        major_radius=macro_major_radius,
        minor_radius=macro_minor_radius,
        max_length=macro_max_merged_length,
        max_height=macro_max_merged_height,
        mode_name="macro_agg",
    )
    _aggregate_scale(
        "meso_link",
        major_radius=cluster_radius,
        minor_radius=max(cluster_radius * 0.4, 8.0),
        max_length=max_merged_length,
        max_height=max_merged_height,
        mode_name="meso_agg",
    )

    df_kept = df.loc[keep_mask].copy()
    if new_patches:
        df_new = pd.DataFrame(new_patches)
        for col in df_kept.columns:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new = df_new[df_kept.columns]
        result = pd.concat([df_kept, df_new], ignore_index=True)
    else:
        result = df_kept.reset_index(drop=True)

    return result

def corridor_elongation(
    df: pd.DataFrame,
    max_stretch_factor: float = 2.5,
    gap_fill_fraction: float = 0.7,
    max_neighbor_dist: float = 120.0,
    macro_stretch_factor: float = 4.0,
    macro_neighbor_dist: float = 220.0,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    """在同组内沿走向拉伸裂缝片, 使相邻片在视觉上重叠形成连续带。

    地球物理依据: 地震分辨率限制使得连续裂缝被分解为多个短片;
    同组相邻裂缝实际可能是同一条长裂缝的不同段, 拉伸恢复了
    超过地震分辨率的裂缝真实长度。

    实现:
    - 在同一 FractureSet 内操作
    - 对每个片, 找沿走向最近的同组邻居
    - 计算沿走向的间隙 (center-to-center 距离 - 两片半长之和)
    - 将 PatchLength 拉伸以填充 gap_fill_fraction 的间隙
    - 拉伸不超过 max_stretch_factor 倍原长度
    - 重新计算 V1-V4 顶点 (中心不变, 仅沿走向拉伸)
    """
    df = df.copy()
    n_elongated = 0
    set_ids = list(df["FractureSet"].unique())
    total_groups = len(set_ids)
    for group_idx, set_id in enumerate(set_ids, start=1):
        mask = df["FractureSet"].values == set_id
        idx = np.where(mask)[0]
        if len(idx) < 2:
            _emit_loop_progress(
                progress_hook,
                "沿走向拉伸组进度",
                group_idx,
                total_groups,
                f"group_size={len(idx)}, skipped_small_group=1",
            )
            continue

        az_vals = df["Azimuth"].values[idx].astype(float)
        mean_az = _azimuth_mean_weighted(az_vals, np.ones(len(az_vals)))
        mean_az_rad = np.deg2rad(mean_az)
        strike_x, strike_y = np.cos(mean_az_rad), np.sin(mean_az_rad)

        cx = df["CenterX"].values[idx].astype(float)
        cy = df["CenterY"].values[idx].astype(float)

        s_proj = cx * strike_x + cy * strike_y
        p_proj = -cx * np.sin(mean_az_rad) + cy * np.cos(mean_az_rad)
        coords_sp = np.column_stack([s_proj, p_proj])
        tree = cKDTree(coords_sp)

        for k in range(len(idx)):
            i = idx[k]
            scale_class = str(df.loc[df.index[i], "ScaleClass"]) if "ScaleClass" in df.columns else "meso_link"
            if scale_class == "micro_bg":
                continue
            class_max_stretch = macro_stretch_factor if scale_class == "macro_core" else max_stretch_factor
            class_neighbor_dist = macro_neighbor_dist if scale_class == "macro_core" else max_neighbor_dist
            orig_length = float(df.loc[df.index[i], "PatchLength"])
            half_len_i = orig_length / 2.0
            nbs = tree.query_ball_point(coords_sp[k], r=class_neighbor_dist)
            if len(nbs) <= 1:
                continue

            best_along_gap = float('inf')
            for nb in nbs:
                if nb == k:
                    continue
                ds = abs(s_proj[nb] - s_proj[k])
                dp = abs(p_proj[nb] - p_proj[k])
                if dp > class_neighbor_dist * (0.25 if scale_class == "macro_core" else 0.4):
                    continue
                half_len_nb = float(df.loc[df.index[idx[nb]], "PatchLength"]) / 2.0
                gap = ds - half_len_i - half_len_nb
                if gap > 0 and gap < best_along_gap:
                    best_along_gap = gap

            if best_along_gap == float('inf') or best_along_gap <= 0:
                continue

            stretch = best_along_gap * gap_fill_fraction
            new_length = orig_length + stretch
            new_length = min(new_length, orig_length * class_max_stretch)
            if new_length <= orig_length * 1.05:
                continue

            v1 = np.array([float(df.iloc[i][f"V1{c}"]) for c in "XYZ"])
            v2 = np.array([float(df.iloc[i][f"V2{c}"]) for c in "XYZ"])
            v4 = np.array([float(df.iloc[i][f"V4{c}"]) for c in "XYZ"])

            center = np.array([float(df.iloc[i]["CenterX"]),
                               float(df.iloc[i]["CenterY"]),
                               float(df.iloc[i]["CenterTIME"])])

            u_vec = v2 - v1
            u_len = np.linalg.norm(u_vec)
            v_vec = v4 - v1
            v_len = np.linalg.norm(v_vec)
            if u_len < 1e-6 or v_len < 1e-6:
                continue

            u_hat = u_vec / u_len
            v_hat = v_vec / v_len
            new_half_u = new_length / 2.0
            half_v = v_len / 2.0

            nv1 = center - new_half_u * u_hat - half_v * v_hat
            nv2 = center + new_half_u * u_hat - half_v * v_hat
            nv3 = center + new_half_u * u_hat + half_v * v_hat
            nv4 = center - new_half_u * u_hat + half_v * v_hat

            for vi, nv in [(1, nv1), (2, nv2), (3, nv3), (4, nv4)]:
                df.loc[df.index[i], f"V{vi}X"] = nv[0]
                df.loc[df.index[i], f"V{vi}Y"] = nv[1]
                df.loc[df.index[i], f"V{vi}Z"] = nv[2]

            df.loc[df.index[i], "PatchLength"] = new_length
            n_elongated += 1

        _emit_loop_progress(
            progress_hook,
            "沿走向拉伸组进度",
            group_idx,
            total_groups,
            f"group_size={len(idx)}",
        )
    return df


# ---------------------------------------------------------------------------
#   Step 7: 空间扰动 (打破地震道网格规则排列)
# ---------------------------------------------------------------------------

def spatial_perturbation(
    df: pd.DataFrame,
    jitter_sigma_xy: float = 8.0,
    along_strike_factor: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """对裂缝片施加空间扰动, 打破地震道网格造成的规则排列。

    地球物理依据: 裂缝实际位置并非严格位于地震道中心;
    地震属性的横向分辨率受 Fresnel 带宽限制, 反演结果存在
    固有的定位不确定性 (通常为道间距的 1/3 ~ 1/2)。

    实现:
    - 对每个裂缝片中心施加 2D 高斯扰动 (σ = jitter_sigma_xy)
    - 沿走向分量放大 along_strike_factor 倍 (裂缝沿走向延伸较远)
    - 4 个顶点做相同刚体平移 (保持裂缝片形状不变)
    - 不修改 CenterTIME / VnZ (深度由地层控制, 不扰动)
    - 使用固定随机种子保证可重复性
    - 补充片 (IsSupplemented=1) 同样扰动, 保持与邻近片一致
    """
    if jitter_sigma_xy <= 0:
        return df

    df = df.copy()
    rng = np.random.RandomState(seed)
    n = len(df)

    az = df["Azimuth"].values.astype(float)
    az_rad = np.deg2rad(az)

    # 走向 / 倾向单位向量 (水平面内)
    strike_x, strike_y = np.cos(az_rad), np.sin(az_rad)
    perp_x, perp_y = -np.sin(az_rad), np.cos(az_rad)

    # 沿走向和垂直走向的独立高斯扰动
    along = rng.normal(0, jitter_sigma_xy * along_strike_factor, n)
    across = rng.normal(0, jitter_sigma_xy, n)

    # 转换回 XY 坐标增量
    dx = along * strike_x + across * perp_x
    dy = along * strike_y + across * perp_y

    # 刚体平移: 中心 + 4 个顶点同步偏移 (Z 不变)
    df["CenterX"] = df["CenterX"].values.astype(float) + dx
    df["CenterY"] = df["CenterY"].values.astype(float) + dy
    for vi in range(1, 5):
        df[f"V{vi}X"] = df[f"V{vi}X"].values.astype(float) + dx
        df[f"V{vi}Y"] = df[f"V{vi}Y"].values.astype(float) + dy

    return df


# ---------------------------------------------------------------------------
#   Step 9: 边界带局部后处理 (受约束的跨界连接)
# ---------------------------------------------------------------------------

def boundary_connect_postprocess(
    df: pd.DataFrame,
    *,
    boundary_strip_width: float = 40.0,
    connect_max_gap: float = 60.0,
    macro_connect_gap: float = 180.0,
    connect_minor_limit: float = 25.0,
    connect_z_gap: float = 20.0,
    azimuth_tol_deg: float = 30.0,
    dip_tol_deg: float = 20.0,
    min_score_threshold: float = 0.35,
    corridor_bonus: float = 0.15,
    require_corridor: bool = False,
    perp_ratio: float = 0.8,
    enable_supplement: bool = True,
    max_stretch_ratio: float = 2.5,
    supplement_confidence_decay: float = 0.7,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """边界带局部后处理 — 仅在单元边界窄带内做受约束的跨界连接。

    第一轮先做 connect (拉伸现有片), 对拉伸不足以覆盖的间隙再补片 (supplement)。

    设计原则 (来自评审):
    - 不做全局操作, 只处理边界两侧各 boundary_strip_width 内的裂缝片
    - CorridorSupport 真正参与决策: 走廊内匹配对得分加 corridor_bonus
    - 每个连接都可追溯: ConnectionID 标识配对, ConnectionType 标识连接方式
    - 所有跨界新连接都写字段: ConnectionType, ConnectionID, IsSupplemented, ReliabilityLevel

    连接判定 (AND):
    1. 同 FractureSet
    2. 同层或层位容差内 (GeoIntervalKey, 如有)
    3. 空间距离 ≤ connect_max_gap
    4. 产状匹配: Azimuth 差 ≤ azimuth_tol, Dip 差 ≤ dip_tol
    5. 综合分 ≥ min_score_threshold (GeophysicsScore 为主, 走廊内加 corridor_bonus)
    6. 沿走向对齐: 垂直走向偏移 ≤ 间距 × perp_ratio

    连接方式:
    - connect: 拉伸匹配对两端, 使边缘至少相切 (不超过 max_stretch_ratio)
    - supplement: 拉伸不够时在间隙中插入桥接片 (标记 IsSupplemented=1)
    """
    df = df.copy()
    stats: dict[str, Any] = {"input_count": len(df)}

    # -- 识别单元边界 --
    has_block = "BlockX" in df.columns and "BlockY" in df.columns
    has_unit = "UnitID" in df.columns

    if not has_block and not has_unit:
        stats["connections_made"] = 0
        stats["reason"] = "no_unit_info"
        return df, stats

    work_az_arr, work_dip_arr = _get_work_orientation(df)
    scale_class_arr = df["ScaleClass"].fillna("meso_link").astype(str).to_numpy() if "ScaleClass" in df.columns else np.array(["meso_link"] * len(df))

    # 构造单元标识 (用 Python int tuple, 避免 numpy repr 差异)
    if has_block:
        bx_arr = df["BlockX"].values
        by_arr = df["BlockY"].values
        unit_key_arr = [(int(bx_arr[i]), int(by_arr[i])) for i in range(len(df))]
    else:
        uid_arr = df["UnitID"].values
        unit_key_arr = [int(uid_arr[i]) for i in range(len(df))]

    # 每个单元的 XY 范围
    cx_arr = df["CenterX"].values.astype(float)
    cy_arr = df["CenterY"].values.astype(float)
    unit_bounds: dict[Any, dict[str, float]] = {}
    for i, k in enumerate(unit_key_arr):
        if k not in unit_bounds:
            unit_bounds[k] = {"x_min": cx_arr[i], "x_max": cx_arr[i],
                              "y_min": cy_arr[i], "y_max": cy_arr[i]}
        else:
            b = unit_bounds[k]
            if cx_arr[i] < b["x_min"]: b["x_min"] = cx_arr[i]
            if cx_arr[i] > b["x_max"]: b["x_max"] = cx_arr[i]
            if cy_arr[i] < b["y_min"]: b["y_min"] = cy_arr[i]
            if cy_arr[i] > b["y_max"]: b["y_max"] = cy_arr[i]

    # -- 标记边界带内的片 --
    is_boundary = np.zeros(len(df), dtype=bool)
    for i, k in enumerate(unit_key_arr):
        b = unit_bounds[k]
        cx, cy = cx_arr[i], cy_arr[i]
        if (cx - b["x_min"] < boundary_strip_width
                or b["x_max"] - cx < boundary_strip_width
                or cy - b["y_min"] < boundary_strip_width
                or b["y_max"] - cy < boundary_strip_width):
            is_boundary[i] = True

    df["_unit_key"] = unit_key_arr
    boundary_idx = np.where(is_boundary)[0]
    stats["boundary_patches"] = int(len(boundary_idx))

    # 确保必要列存在
    if "ConnectionID" not in df.columns:
        df["ConnectionID"] = -1

    if len(boundary_idx) < 2:
        stats.update(connections_made=0, supplements_added=0)
        df.drop(columns=["_unit_key"], inplace=True)
        return df, stats

    # -- 在边界片中找跨单元匹配对 --
    b_cx = df["CenterX"].values[boundary_idx].astype(float)
    b_cy = df["CenterY"].values[boundary_idx].astype(float)
    b_coords = np.column_stack([b_cx, b_cy])
    tree = cKDTree(b_coords)

    connections: list[dict[str, Any]] = []
    paired: set[tuple[int, int]] = set()

    for i_local in range(len(boundary_idx)):
        i_global = boundary_idx[i_local]
        row_i = df.iloc[i_global]
        unit_i = row_i["_unit_key"]

        candidates = tree.query_ball_point(b_coords[i_local], r=connect_max_gap)
        for j_local in candidates:
            if j_local <= i_local:
                continue
            j_global = boundary_idx[j_local]
            row_j = df.iloc[j_global]
            unit_j = row_j["_unit_key"]

            # 必须跨单元
            if unit_i == unit_j:
                continue

            pair_key = (min(i_global, j_global), max(i_global, j_global))
            if pair_key in paired:
                continue

            scale_i = scale_class_arr[i_global]
            scale_j = scale_class_arr[j_global]
            if scale_i == "micro_bg" and scale_j == "micro_bg":
                continue
            target_gap = macro_connect_gap if (scale_i == "macro_core" or scale_j == "macro_core") else connect_max_gap

            # 条件 1: 同 FractureSet
            if int(row_i.get("FractureSet", -1)) != int(row_j.get("FractureSet", -2)):
                continue

            # 条件 2: 同层 (如有)
            if not _same_physical_layer(row_i, row_j):
                continue

            # 条件 3: 产状匹配 (放宽: 聚合后产状是统计平均, 允许更大偏差)
            az_d = _azimuth_diff(
                np.array([float(work_az_arr[i_global])]),
                np.array([float(work_az_arr[j_global])])
            )[0]
            dip_d = abs(float(work_dip_arr[i_global]) - float(work_dip_arr[j_global]))
            if az_d > azimuth_tol_deg or dip_d > dip_tol_deg:
                continue

            # 条件 4: 综合分 (GeophysicsScore 为主 + 走廊加分)
            scores_i = []
            scores_j = []
            for sc in ("GeophysicsScore", "GeometryScore"):
                if sc in df.columns:
                    scores_i.append(float(row_i.get(sc, 0)))
                    scores_j.append(float(row_j.get(sc, 0)))

            avg_score_i = np.mean(scores_i) if scores_i else 0.5
            avg_score_j = np.mean(scores_j) if scores_j else 0.5

            corridor_i = int(row_i.get("CorridorSupport", 0))
            corridor_j = int(row_j.get("CorridorSupport", 0))
            has_corridor = corridor_i > 0 or corridor_j > 0

            if require_corridor and not has_corridor:
                continue

            # 走廊加分
            bonus = corridor_bonus if has_corridor else 0.0
            pair_score = (avg_score_i + avg_score_j) / 2.0 + bonus

            if pair_score < min_score_threshold:
                continue

            # 条件 5: 沿走向对齐检查 (放宽到 perp_ratio)
            mean_az_rad = np.deg2rad(
                (float(work_az_arr[i_global]) + float(work_az_arr[j_global])) / 2.0
            )
            dx = float(row_j["CenterX"]) - float(row_i["CenterX"])
            dy = float(row_j["CenterY"]) - float(row_i["CenterY"])
            dz = abs(float(row_j["CenterTIME"]) - float(row_i["CenterTIME"]))
            along_dist = abs(dx * np.cos(mean_az_rad) + dy * np.sin(mean_az_rad))
            perp_dist = abs(-dx * np.sin(mean_az_rad) + dy * np.cos(mean_az_rad))
            gap_dist = np.sqrt(dx ** 2 + dy ** 2)

            if along_dist > target_gap or perp_dist > connect_minor_limit or dz > connect_z_gap:
                continue

            if perp_dist > gap_dist * perp_ratio + 1e-6:
                continue

            paired.add(pair_key)
            connections.append({
                "i": i_global, "j": j_global,
                "gap": along_dist, "perp": perp_dist, "euclidean_gap": gap_dist, "z_gap": dz,
                "az_diff": az_d, "dip_diff": dip_d,
                "score": pair_score, "has_corridor": has_corridor,
                "scale_i": scale_i, "scale_j": scale_j,
            })

    # -- 辅助: 重建矩形顶点 --
    def _rebuild_vertices(target_idx: int, new_len: float) -> None:
        r = df.iloc[target_idx]
        az_rad = np.deg2rad(float(r["Azimuth"]))
        dip_rad = np.deg2rad(float(r["Dip"]))
        s_x, s_y = np.cos(az_rad), np.sin(az_rad)
        u_hat = np.array([s_x, s_y, 0.0])
        perp_x, perp_y = -np.sin(az_rad), np.cos(az_rad)
        v_hat = np.array([
            perp_x * np.cos(dip_rad),
            perp_y * np.cos(dip_rad),
            -np.sin(dip_rad),
        ])
        v_norm = np.linalg.norm(v_hat)
        if v_norm > 1e-6:
            v_hat /= v_norm
        else:
            v_hat = np.array([0.0, 0.0, -1.0])
        center = np.array([float(r["CenterX"]), float(r["CenterY"]), float(r["CenterTIME"])])
        half_u = new_len / 2.0
        half_v = float(r["PatchHeight"]) / 2.0
        nv1 = center - half_u * u_hat - half_v * v_hat
        nv2 = center + half_u * u_hat - half_v * v_hat
        nv3 = center + half_u * u_hat + half_v * v_hat
        nv4 = center - half_u * u_hat + half_v * v_hat
        idx_label = df.index[target_idx]
        df.loc[idx_label, "PatchLength"] = new_len
        for vi, nv in enumerate([nv1, nv2, nv3, nv4], start=1):
            df.loc[idx_label, f"V{vi}X"] = nv[0]
            df.loc[idx_label, f"V{vi}Y"] = nv[1]
            df.loc[idx_label, f"V{vi}Z"] = nv[2]

    # -- 按分数降序排列, 优先连接高质量对 --
    connections.sort(key=lambda c: c["score"], reverse=True)

    # -- 执行连接 --
    n_connected = 0
    n_supplemented = 0
    connection_id_counter = int(df["ConnectionID"].max()) + 1 if (df["ConnectionID"] >= 0).any() else 0
    supplement_patches: list[dict[str, Any]] = []

    for conn in connections:
        i, j = conn["i"], conn["j"]
        row_i = df.iloc[i]
        row_j = df.iloc[j]
        conn_id = connection_id_counter
        connection_id_counter += 1

        gap = conn["gap"]
        len_i = float(row_i["PatchLength"])
        len_j = float(row_j["PatchLength"])

        # 标记两端为 boundary_connected
        idx_label_i = df.index[i]
        idx_label_j = df.index[j]
        df.loc[idx_label_i, "ConnectionType"] = "boundary_connected"
        df.loc[idx_label_j, "ConnectionType"] = "boundary_connected"
        df.loc[idx_label_i, "ConnectionID"] = conn_id
        df.loc[idx_label_j, "ConnectionID"] = conn_id

        # 计算需要的拉伸量: 使两片边缘至少相切
        total_half = (len_i + len_j) / 2.0
        needed = gap - total_half  # >0 表示有间隙

        if needed > 0:
            # 每片最多拉伸到 max_stretch_ratio 倍
            max_stretch_i = len_i * (max_stretch_ratio - 1.0)
            max_stretch_j = len_j * (max_stretch_ratio - 1.0)
            avail = max_stretch_i + max_stretch_j

            if avail >= needed:
                # 拉伸足够覆盖间隙 — 按比例分配
                frac_i = max_stretch_i / avail if avail > 0 else 0.5
                stretch_i = min(needed * frac_i, max_stretch_i)
                stretch_j = min(needed * (1 - frac_i), max_stretch_j)
                _rebuild_vertices(i, len_i + stretch_i)
                _rebuild_vertices(j, len_j + stretch_j)
            else:
                # 拉伸到极限
                _rebuild_vertices(i, len_i + max_stretch_i)
                _rebuild_vertices(j, len_j + max_stretch_j)

                # 仍有剩余间隙 → supplement 桥接片
                if enable_supplement:
                    remaining_gap = needed - avail
                    if remaining_gap > 1.0:  # 间隙 > 1m 才补
                        # 桥接片: 中心在两片中点, 产状取加权平均
                        mid_x = (float(row_i["CenterX"]) + float(row_j["CenterX"])) / 2.0
                        mid_y = (float(row_i["CenterY"]) + float(row_j["CenterY"])) / 2.0
                        mid_z = (float(row_i["CenterTIME"]) + float(row_j["CenterTIME"])) / 2.0
                        mean_az = _azimuth_mean_weighted(
                            np.array([float(row_i["Azimuth"]), float(row_j["Azimuth"])]),
                            np.array([1.0, 1.0]),
                        )
                        mean_dip = (float(row_i["Dip"]) + float(row_j["Dip"])) / 2.0
                        # 桥接片长度 = 剩余间隙 + 少量重叠
                        bridge_len = remaining_gap + 5.0
                        bridge_hgt = (float(row_i["PatchHeight"]) + float(row_j["PatchHeight"])) / 2.0
                        bridge_conf = min(float(row_i.get("Confidence", 0.5)),
                                          float(row_j.get("Confidence", 0.5))) * supplement_confidence_decay

                        # 构造顶点
                        az_rad = np.deg2rad(mean_az)
                        dip_rad = np.deg2rad(mean_dip)
                        u_hat = np.array([np.cos(az_rad), np.sin(az_rad), 0.0])
                        px, py = -np.sin(az_rad), np.cos(az_rad)
                        v_hat = np.array([px * np.cos(dip_rad), py * np.cos(dip_rad), -np.sin(dip_rad)])
                        vn = np.linalg.norm(v_hat)
                        v_hat = v_hat / vn if vn > 1e-6 else np.array([0.0, 0.0, -1.0])
                        ctr = np.array([mid_x, mid_y, mid_z])
                        hu, hv = bridge_len / 2.0, bridge_hgt / 2.0
                        bv1 = ctr - hu * u_hat - hv * v_hat
                        bv2 = ctr + hu * u_hat - hv * v_hat
                        bv3 = ctr + hu * u_hat + hv * v_hat
                        bv4 = ctr - hu * u_hat + hv * v_hat

                        # 从高分端继承非几何属性
                        ref_idx = i if float(row_i.get("GeophysicsScore", 0)) >= float(row_j.get("GeophysicsScore", 0)) else j
                        ref_row = df.iloc[ref_idx].to_dict()
                        sp: dict[str, Any] = {}
                        for col in df.columns:
                            if col != "_unit_key":
                                sp[col] = ref_row.get(col, 0)
                        sp_scale = "macro_core" if (conn["scale_i"] == "macro_core" or conn["scale_j"] == "macro_core") else "meso_link"
                        sp.update({
                            "CenterX": mid_x, "CenterY": mid_y, "CenterTIME": mid_z,
                            "Azimuth": mean_az, "Dip": mean_dip,
                            "PatchLength": bridge_len, "PatchHeight": bridge_hgt,
                            "PatchArea": bridge_len * bridge_hgt,
                            "Confidence": bridge_conf,
                            "ConnectionType": "supplemented",
                            "ConnectionID": conn_id,
                            "IsSupplemented": 1,
                            "ReliabilityLevel": "medium",
                            "ScaleClass": sp_scale,
                            "AggregationMode": "boundary_bridge",
                            "ParentPatchCount": 2,
                            "V1X": bv1[0], "V1Y": bv1[1], "V1Z": bv1[2],
                            "V2X": bv2[0], "V2Y": bv2[1], "V2Z": bv2[2],
                            "V3X": bv3[0], "V3Y": bv3[1], "V3Z": bv3[2],
                            "V4X": bv4[0], "V4Y": bv4[1], "V4Z": bv4[2],
                        })
                        if "OrigAzimuth" in df.columns:
                            sp["OrigAzimuth"] = mean_az
                            sp["OrigDip"] = mean_dip
                        if "SmoothedAzimuth" in df.columns:
                            sp["SmoothedAzimuth"] = mean_az
                            sp["SmoothedDip"] = mean_dip
                        supplement_patches.append(sp)
                        n_supplemented += 1
        else:
            # 已经重叠, 无需拉伸
            pass

        n_connected += 1

    # -- 追加补片 --
    if supplement_patches:
        df_sup = pd.DataFrame(supplement_patches)
        # _unit_key 不需要保留
        if "_unit_key" in df_sup.columns:
            df_sup.drop(columns=["_unit_key"], inplace=True)
        for col in df.columns:
            if col not in df_sup.columns and col != "_unit_key":
                df_sup[col] = 0
        keep_cols = [c for c in df.columns if c != "_unit_key"]
        df_sup = df_sup[keep_cols]
        df.drop(columns=["_unit_key"], inplace=True)
        df = pd.concat([df, df_sup], ignore_index=True)
    else:
        df.drop(columns=["_unit_key"], inplace=True)

    stats["connections_made"] = n_connected
    stats["connections_with_corridor"] = sum(1 for c in connections if c["has_corridor"])
    stats["supplements_added"] = n_supplemented
    stats["mean_gap_m"] = float(np.mean([c["gap"] for c in connections])) if connections else 0.0
    stats["mean_score"] = float(np.mean([c["score"] for c in connections])) if connections else 0.0
    stats["output_count"] = len(df)

    return df, stats


# ---------------------------------------------------------------------------
#   Step 9b: 边界缝合密度填充 (消除网格感的关键)
# ---------------------------------------------------------------------------

def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """角度的圆周均值 (处理 0°/360° 环绕)。"""
    rads = np.deg2rad(angles_deg)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))) % 360)


def _blend_azimuth(az1: float, az2: float, t: float) -> float:
    """在两个方位角之间做圆周线性插值, t=0 返回 az1, t=1 返回 az2。"""
    r1, r2 = np.deg2rad(az1), np.deg2rad(az2)
    s = np.sin(r1) * (1 - t) + np.sin(r2) * t
    c = np.cos(r1) * (1 - t) + np.cos(r2) * t
    return float(np.rad2deg(np.arctan2(s, c)) % 360)


def boundary_seam_fill(
    df: pd.DataFrame,
    *,
    seam_half_width: float = 50.0,
    interior_sample_depth: float = 80.0,
    fill_fraction: float = 1.0,
    confidence_decay: float = 0.65,
    blend_half_width: float = 60.0,
    blend_strength: float = 0.5,
    min_shared_set_ratio: float = 0.15,
    high_reliability_boost: float = 1.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """边界缝合 — 按证据强度分段做属性混合 + 填充。

    核心改进: 不再对所有边界均匀操作, 而是:
    1. 对每条边界, 按 FractureSet 分组
    2. 只对两侧共享的 FractureSet 做填充和混合
    3. 高 ReliabilityLevel 片优先作为参考源, 且得到更高填充权重
    4. 混合仅发生在同一 FractureSet 内, 不跨组混合

    参数:
      seam_half_width: 填充新片的缝合带半宽 (m)
      interior_sample_depth: 从边界向内采样参考片的深度 (m)
      fill_fraction: 基础填充比例 (按每组共享片数缩放)
      confidence_decay: 新片置信度衰减系数
      blend_half_width: 属性混合区半宽 (m)
      blend_strength: 最大混合强度 (0-1)
      min_shared_set_ratio: 某组在一侧的最低占比才算 "共享"
      high_reliability_boost: 高可靠性段落填充倍数
      seed: 随机种子
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    stats: dict[str, Any] = {"input_count": len(df)}

    has_block = "BlockX" in df.columns and "BlockY" in df.columns
    has_fset = "FractureSet" in df.columns
    if not has_block:
        stats["seam_fills"] = 0
        stats["blended_patches"] = 0
        return df, stats

    bx_arr = df["BlockX"].values
    by_arr = df["BlockY"].values
    cx_arr = df["CenterX"].values.astype(float)
    cy_arr = df["CenterY"].values.astype(float)
    fs_arr = df["FractureSet"].values.astype(int) if has_fset else np.zeros(len(df), dtype=int)

    # ReliabilityLevel: VTK 中 2=high, 1=medium, 0=low
    has_rl = "ReliabilityLevel" in df.columns
    if has_rl:
        rl_raw = df["ReliabilityLevel"].values
        rl_arr = np.array([2 if v in (2, "high") else (1 if v in (1, "medium") else 0)
                           for v in rl_raw], dtype=int)
    else:
        rl_arr = np.ones(len(df), dtype=int)

    # 单元 bounds
    units: dict[tuple[int, int], dict[str, float]] = {}
    for i in range(len(df)):
        k = (int(bx_arr[i]), int(by_arr[i]))
        if k not in units:
            units[k] = {"x_min": cx_arr[i], "x_max": cx_arr[i],
                        "y_min": cy_arr[i], "y_max": cy_arr[i]}
        else:
            b = units[k]
            if cx_arr[i] < b["x_min"]: b["x_min"] = cx_arr[i]
            if cx_arr[i] > b["x_max"]: b["x_max"] = cx_arr[i]
            if cy_arr[i] < b["y_min"]: b["y_min"] = cy_arr[i]
            if cy_arr[i] > b["y_max"]: b["y_max"] = cy_arr[i]

    # 找相邻单元对
    unit_keys = sorted(units.keys())
    adj_pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for i, u1 in enumerate(unit_keys):
        for u2 in unit_keys[i + 1:]:
            if abs(u1[0] - u2[0]) == 1 and u1[1] == u2[1]:
                adj_pairs.append((u1, u2, "x"))
            elif u1[0] == u2[0] and abs(u1[1] - u2[1]) == 1:
                adj_pairs.append((u1, u2, "y"))

    if "ConnectionID" not in df.columns:
        df["ConnectionID"] = -1
    conn_id_counter = int(df["ConnectionID"].max()) + 1 if (df["ConnectionID"] >= 0).any() else 0

    new_patches: list[dict[str, Any]] = []
    total_filled = 0
    edge_diagnostics: list[dict[str, Any]] = []

    # --- 辅助: 根据 axis 获取两侧的 idx 和空间范围 ---
    def _get_edge_geometry(u1, u2, ax, b1, b2, mask1, mask2):
        """返回 (boundary_coord, seam_min, seam_max,
                int1_min, int1_max, int2_min, int2_max,
                cross_lo, cross_hi, coord_arr, cross_arr,
                side1_mask, side2_mask) 或 None"""
        if ax == "x":
            if b1["x_max"] < b2["x_max"]:
                bc = (b1["x_max"] + b2["x_min"]) / 2.0
            else:
                bc = (b2["x_max"] + b1["x_min"]) / 2.0
                mask1, mask2 = mask2, mask1
                b1, b2 = b2, b1
            cross_lo = max(b1["y_min"], b2["y_min"])
            cross_hi = min(b1["y_max"], b2["y_max"])
            coord_arr, cross_arr = cx_arr, cy_arr
        else:
            if b1["y_max"] < b2["y_max"]:
                bc = (b1["y_max"] + b2["y_min"]) / 2.0
            else:
                bc = (b2["y_max"] + b1["y_min"]) / 2.0
                mask1, mask2 = mask2, mask1
                b1, b2 = b2, b1
            cross_lo = max(b1["x_min"], b2["x_min"])
            cross_hi = min(b1["x_max"], b2["x_max"])
            coord_arr, cross_arr = cy_arr, cx_arr

        if cross_hi <= cross_lo:
            return None
        sm_min, sm_max = bc - seam_half_width, bc + seam_half_width
        i1_min, i1_max = sm_min - interior_sample_depth, sm_min
        i2_min, i2_max = sm_max, sm_max + interior_sample_depth
        return (bc, sm_min, sm_max, i1_min, i1_max, i2_min, i2_max,
                cross_lo, cross_hi, coord_arr, cross_arr, mask1, mask2)

    for u1, u2, axis in adj_pairs:
        b1, b2 = units[u1], units[u2]
        mask1 = (bx_arr == u1[0]) & (by_arr == u1[1])
        mask2 = (bx_arr == u2[0]) & (by_arr == u2[1])

        geo = _get_edge_geometry(u1, u2, axis, b1, b2, mask1, mask2)
        if geo is None:
            continue
        (bc, sm_min, sm_max, i1_min, i1_max, i2_min, i2_max,
         cross_lo, cross_hi, coord_arr, cross_arr, side1_mask, side2_mask) = geo

        # 边界附近 (interior + seam) 两侧片
        near1 = np.where(side1_mask & (coord_arr >= i1_min) & (coord_arr <= sm_max) &
                         (cross_arr >= cross_lo) & (cross_arr <= cross_hi))[0]
        near2 = np.where(side2_mask & (coord_arr >= sm_min) & (coord_arr <= i2_max) &
                         (cross_arr >= cross_lo) & (cross_arr <= cross_hi))[0]

        if len(near1) < 2 or len(near2) < 2:
            edge_diagnostics.append({
                "edge": f"{u1}-{u2}({axis})", "status": "skip_sparse",
                "n_side1": len(near1), "n_side2": len(near2),
            })
            continue

        # --- 分 FractureSet 评估共享组 ---
        sets1 = fs_arr[near1]
        sets2 = fs_arr[near2]
        unique_sets = set(np.unique(sets1)) | set(np.unique(sets2))

        shared_sets: list[dict[str, Any]] = []
        for s in sorted(unique_sets):
            ratio1 = float(np.sum(sets1 == s)) / len(sets1)
            ratio2 = float(np.sum(sets2 == s)) / len(sets2)
            if ratio1 >= min_shared_set_ratio and ratio2 >= min_shared_set_ratio:
                # 高可靠占比
                s1_idx = near1[sets1 == s]
                s2_idx = near2[sets2 == s]
                high1 = float(np.mean(rl_arr[s1_idx] >= 2)) if len(s1_idx) > 0 else 0
                high2 = float(np.mean(rl_arr[s2_idx] >= 2)) if len(s2_idx) > 0 else 0
                high_frac = (high1 + high2) / 2.0
                # 方位角一致性 (两侧该组的方差)
                az1 = df["Azimuth"].values[s1_idx].astype(float)
                az2 = df["Azimuth"].values[s2_idx].astype(float)
                az_diff = abs(_circular_mean_deg(az1) - _circular_mean_deg(az2))
                if az_diff > 180:
                    az_diff = 360 - az_diff

                shared_sets.append({
                    "set": int(s),
                    "ratio1": ratio1, "ratio2": ratio2,
                    "n1": len(s1_idx), "n2": len(s2_idx),
                    "high_frac": high_frac,
                    "az_diff_deg": az_diff,
                    "idx1": s1_idx, "idx2": s2_idx,
                })

        if not shared_sets:
            edge_diagnostics.append({
                "edge": f"{u1}-{u2}({axis})", "status": "no_shared_sets",
                "n_side1": len(near1), "n_side2": len(near2),
            })
            continue

        # --- 对每个共享组做填充 ---
        edge_fills = 0
        for ss in shared_sets:
            # 内部采样源 (只采该组的片)
            int1_idx = ss["idx1"][(coord_arr[ss["idx1"]] >= i1_min) & (coord_arr[ss["idx1"]] <= i1_max)]
            int2_idx = ss["idx2"][(coord_arr[ss["idx2"]] >= i2_min) & (coord_arr[ss["idx2"]] <= i2_max)]
            src_idx = np.concatenate([int1_idx, int2_idx]) if len(int1_idx) > 0 or len(int2_idx) > 0 else np.array([], dtype=int)

            if len(src_idx) < 1:
                # 退而求其次: 整个边界附近该组所有片
                src_idx = np.concatenate([ss["idx1"], ss["idx2"]])
            if len(src_idx) < 1:
                continue

            # 填充量: 基础 × 高可靠加成 × 方位角一致加成
            reliability_mult = 1.0 + (ss["high_frac"] * (high_reliability_boost - 1.0))
            # 方位角差 < 15° 全量, > 45° 大幅衰减
            az_mult = max(0.2, 1.0 - max(0, ss["az_diff_deg"] - 15) / 60.0)
            effective_fraction = fill_fraction * reliability_mult * az_mult

            n_fill = max(1, int(round(len(src_idx) * effective_fraction)))
            sample_idx = rng.choice(src_idx, size=min(n_fill, len(src_idx)), replace=True)

            # 计算该组在边界附近的 Azimuth 标准差 (用于保持自然变异)
            src_az = df["Azimuth"].values[src_idx].astype(float)
            az_noise_std = max(8.0, float(np.std(src_az)) * 0.5)
            dip_noise_std = 3.0

            for si in sample_idx:
                ref = df.iloc[si]
                # 放置策略: 随机选边界一侧, 在该侧 interior 范围内均匀放置
                # (打破对称高斯在 bc 上形成的规则线条)
                if rng.random() < 0.5:
                    # 落到 side1 (i1_min → bc)
                    lo, hi = i1_min, bc
                else:
                    # 落到 side2 (bc → i2_max)
                    lo, hi = bc, i2_max
                if axis == "x":
                    new_coord = rng.uniform(lo, hi)
                    new_cross = rng.uniform(cross_lo, cross_hi)
                    nx, ny = new_coord, new_cross
                else:
                    new_cross = rng.uniform(cross_lo, cross_hi)
                    new_coord = rng.uniform(lo, hi)
                    nx, ny = new_cross, new_coord
                nz = float(ref["CenterTIME"]) + rng.normal(0, 5.0)

                _make_fill_patch(df, ref, nx, ny, nz, confidence_decay,
                                 conn_id_counter, new_patches,
                                 az_noise=rng.normal(0, az_noise_std),
                                 dip_noise=rng.normal(0, dip_noise_std))
                conn_id_counter += 1
                total_filled += 1
                edge_fills += 1

        edge_diagnostics.append({
            "edge": f"{u1}-{u2}({axis})",
            "status": "filled",
            "n_side1": len(near1), "n_side2": len(near2),
            "shared_sets": [{k: v for k, v in ss.items() if k not in ("idx1", "idx2")}
                            for ss in shared_sets],
            "fills": edge_fills,
        })

    # --- Part B: 按组做属性混合 (只混合共享组) ---
    az_arr = df["Azimuth"].values.astype(float).copy()
    dip_arr = df["Dip"].values.astype(float).copy()
    cx_arr = df["CenterX"].values.astype(float)
    cy_arr = df["CenterY"].values.astype(float)
    bx_arr = df["BlockX"].values
    by_arr = df["BlockY"].values
    total_blended = 0

    for u1, u2, axis in adj_pairs:
        b1, b2 = units[u1], units[u2]
        mask1 = (bx_arr == u1[0]) & (by_arr == u1[1])
        mask2 = (bx_arr == u2[0]) & (by_arr == u2[1])

        if axis == "x":
            if b1["x_max"] < b2["x_max"]:
                boundary_coord = (b1["x_max"] + b2["x_min"]) / 2.0
            else:
                boundary_coord = (b2["x_max"] + b1["x_min"]) / 2.0
                mask1, mask2 = mask2, mask1
                b1, b2 = b2, b1
            blend_min = boundary_coord - blend_half_width
            blend_max = boundary_coord + blend_half_width
            cross_lo = max(b1["y_min"], b2["y_min"])
            cross_hi = min(b1["y_max"], b2["y_max"])
            if cross_hi <= cross_lo:
                continue
            zone1 = np.where(mask1 & (cx_arr >= blend_min) & (cx_arr <= boundary_coord) &
                             (cy_arr >= cross_lo) & (cy_arr <= cross_hi))[0]
            zone2 = np.where(mask2 & (cx_arr > boundary_coord) & (cx_arr <= blend_max) &
                             (cy_arr >= cross_lo) & (cy_arr <= cross_hi))[0]
            dist_fn = lambda idx: abs(cx_arr[idx] - boundary_coord)
        else:
            if b1["y_max"] < b2["y_max"]:
                boundary_coord = (b1["y_max"] + b2["y_min"]) / 2.0
            else:
                boundary_coord = (b2["y_max"] + b1["y_min"]) / 2.0
                mask1, mask2 = mask2, mask1
                b1, b2 = b2, b1
            blend_min = boundary_coord - blend_half_width
            blend_max = boundary_coord + blend_half_width
            cross_lo = max(b1["x_min"], b2["x_min"])
            cross_hi = min(b1["x_max"], b2["x_max"])
            if cross_hi <= cross_lo:
                continue
            zone1 = np.where(mask1 & (cy_arr >= blend_min) & (cy_arr <= boundary_coord) &
                             (cx_arr >= cross_lo) & (cx_arr <= cross_hi))[0]
            zone2 = np.where(mask2 & (cy_arr > boundary_coord) & (cy_arr <= blend_max) &
                             (cx_arr >= cross_lo) & (cx_arr <= cross_hi))[0]
            dist_fn = lambda idx: abs(cy_arr[idx] - boundary_coord)

        if len(zone1) == 0 or len(zone2) == 0:
            continue

        # 按 FractureSet 分组混合 (仅混合两侧共享比例均 >= 阈值的主组)
        all_sets = set(np.unique(fs_arr[zone1])) & set(np.unique(fs_arr[zone2]))
        for s in all_sets:
            s1 = zone1[fs_arr[zone1] == s]
            s2 = zone2[fs_arr[zone2] == s]
            if len(s1) == 0 or len(s2) == 0:
                continue
            # 共享比例筛选: 只混合主组, 跳过低占比组
            ratio_s1 = len(s1) / len(zone1) if len(zone1) > 0 else 0
            ratio_s2 = len(s2) / len(zone2) if len(zone2) > 0 else 0
            if ratio_s1 < min_shared_set_ratio or ratio_s2 < min_shared_set_ratio:
                continue

            mean_az_s2 = _circular_mean_deg(az_arr[s2])
            mean_dip_s2 = float(np.mean(dip_arr[s2]))
            mean_az_s1 = _circular_mean_deg(az_arr[s1])
            mean_dip_s1 = float(np.mean(dip_arr[s1]))

            # 方位角差太大 (>45°) 说明两侧该组不兼容, 减弱混合
            az_diff = abs(mean_az_s1 - mean_az_s2)
            if az_diff > 180:
                az_diff = 360 - az_diff
            set_blend = blend_strength * max(0.2, 1.0 - max(0, az_diff - 15) / 60.0)

            # 混合 + 随机扰动 (防止混合后方差崩塌造成人工条带)
            blend_az_noise = max(5.0, az_diff * 0.3)  # 与两侧差异成正比
            for idx in s1:
                dist = dist_fn(idx)
                t = max(0.0, 1.0 - dist / blend_half_width) * set_blend
                if rl_arr[idx] >= 2:
                    t *= 0.5
                blended = _blend_azimuth(az_arr[idx], mean_az_s2, t)
                az_arr[idx] = (blended + rng.normal(0, blend_az_noise * t)) % 360
                dip_arr[idx] = dip_arr[idx] * (1 - t) + mean_dip_s2 * t + rng.normal(0, 2.0 * t)
                total_blended += 1
            for idx in s2:
                dist = dist_fn(idx)
                t = max(0.0, 1.0 - dist / blend_half_width) * set_blend
                if rl_arr[idx] >= 2:
                    t *= 0.5
                blended = _blend_azimuth(az_arr[idx], mean_az_s1, t)
                az_arr[idx] = (blended + rng.normal(0, blend_az_noise * t)) % 360
                dip_arr[idx] = dip_arr[idx] * (1 - t) + mean_dip_s1 * t + rng.normal(0, 2.0 * t)
                total_blended += 1

    df["Azimuth"] = az_arr
    df["Dip"] = dip_arr

    # 追加新片
    if new_patches:
        df_fill = pd.DataFrame(new_patches)
        for col in df.columns:
            if col not in df_fill.columns:
                df_fill[col] = 0
        df_fill = df_fill[df.columns]
        df = pd.concat([df, df_fill], ignore_index=True)

    stats["seam_fills"] = total_filled
    stats["blended_patches"] = total_blended
    stats["adjacent_edges"] = len(adj_pairs)
    stats["edge_diagnostics"] = edge_diagnostics
    stats["output_count"] = len(df)
    return df, stats


def _make_fill_patch(
    df: pd.DataFrame, ref: pd.Series,
    new_x: float, new_y: float, new_z: float,
    conf_decay: float, conn_id: int,
    out_list: list[dict[str, Any]],
    az_noise: float = 0.0,
    dip_noise: float = 0.0,
) -> None:
    """从参考片构造一个缝合填充片。"""
    az = (float(ref["Azimuth"]) + az_noise) % 360
    dip = float(np.clip(float(ref["Dip"]) + dip_noise, 0, 90))
    p_len = float(ref["PatchLength"])
    p_hgt = float(ref["PatchHeight"])
    conf = float(ref.get("Confidence", 0.5)) * conf_decay

    az_rad = np.deg2rad(az)
    dip_rad = np.deg2rad(dip)
    u = np.array([np.cos(az_rad), np.sin(az_rad), 0.0])
    px, py = -np.sin(az_rad), np.cos(az_rad)
    v = np.array([px * np.cos(dip_rad), py * np.cos(dip_rad), -np.sin(dip_rad)])
    vn = np.linalg.norm(v)
    v = v / vn if vn > 1e-6 else np.array([0.0, 0.0, -1.0])
    c = np.array([new_x, new_y, new_z])
    hu, hv = p_len / 2.0, p_hgt / 2.0
    v1 = c - hu * u - hv * v
    v2 = c + hu * u - hv * v
    v3 = c + hu * u + hv * v
    v4 = c - hu * u + hv * v

    sp: dict[str, Any] = {}
    for col in df.columns:
        sp[col] = ref.get(col, 0)
    sp.update({
        "CenterX": new_x, "CenterY": new_y, "CenterTIME": new_z,
        "Azimuth": az, "Dip": dip,
        "PatchLength": p_len, "PatchHeight": p_hgt,
        "Confidence": conf,
        "ConnectionType": "supplemented",
        "ConnectionID": conn_id,
        "IsSupplemented": 1,
        "ReliabilityLevel": "medium",
        "V1X": v1[0], "V1Y": v1[1], "V1Z": v1[2],
        "V2X": v2[0], "V2Y": v2[1], "V2Z": v2[2],
        "V3X": v3[0], "V3Y": v3[1], "V3Z": v3[2],
        "V4X": v4[0], "V4Y": v4[1], "V4Z": v4[2],
    })
    if "OrigAzimuth" in df.columns:
        sp["OrigAzimuth"] = az
        sp["OrigDip"] = dip
    if "SmoothedAzimuth" in df.columns:
        sp["SmoothedAzimuth"] = az
        sp["SmoothedDip"] = dip
    out_list.append(sp)


# ---------------------------------------------------------------------------
#   主流程: Phase 1 (保守)
# ---------------------------------------------------------------------------

def run_phase1(
    df: pd.DataFrame,
    *,
    max_fracture_sets: int = 6,
    n_fracture_sets: int | None = None,
    neighbor_radius_xy: float = 100.0,
    neighbor_radius_z: float = 15.0,
    isolation_min_neighbors: int = 2,
    confidence_floor: float = 0.3,
    boundary_tol_xy: float = 25.0,
    length_range: tuple[float, float] = (1.0, 200.0),
    height_range: tuple[float, float] = (0.5, 100.0),
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
    progress_hook: Callable[[str, str | None], None] | None = None,
    heartbeat_hook: Callable[[str, str | None], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """第一轮: 裂缝组识别 + 可靠性评分 + 边界匹配 (不补片)。

    目的: 证明"后处理提升了区域连贯性, 但没有明显造假连接"。
    """
    stats: dict[str, Any] = {"phase": 1, "input_count": len(df)}

    stats["step_seconds"] = {}

    step_started = perf_counter()
    df = fracture_set_clustering(
        df,
        max_sets=max_fracture_sets,
        n_sets=n_fracture_sets,
        progress_hook=heartbeat_hook,
    )
    stats["step_seconds"]["fracture_set_clustering"] = float(perf_counter() - step_started)
    n_sets_found = int(df["FractureSet"].nunique())
    set_sizes = df.groupby("FractureSet").size().to_dict()
    stats["fracture_sets"] = n_sets_found
    stats["set_sizes"] = {str(k): int(v) for k, v in set_sizes.items()}
    stats["fracture_set_bic_sample_size"] = int(df.attrs.get("fracture_set_bic_sample_size", len(df)))
    stats["fracture_set_fit_sample_size"] = int(df.attrs.get("fracture_set_fit_sample_size", len(df)))
    if progress_hook is not None:
        progress_hook(
            "Phase1 裂缝组识别完成",
            (
                f"fracture_sets={n_sets_found}, "
                f"elapsed={_format_seconds(stats['step_seconds']['fracture_set_clustering'])}"
            ),
        )

    # OrigAzimuth/OrigDip 必须在 Phase 1 就保留
    df["OrigAzimuth"] = df["Azimuth"].values.copy()
    df["OrigDip"] = df["Dip"].values.copy()

    step_started = perf_counter()
    df = multi_dimensional_reliability_scoring(
        df,
        neighbor_radius_xy=neighbor_radius_xy,
        neighbor_radius_z=neighbor_radius_z,
        min_neighbors=isolation_min_neighbors,
        confidence_floor=confidence_floor,
        length_range=length_range,
        height_range=height_range,
        compute_backend=compute_backend,
        gpu_tile_points=gpu_tile_points,
    )
    stats["step_seconds"]["reliability_scoring"] = float(perf_counter() - step_started)
    stats["compute_backend"] = str(df.attrs.get("compute_backend", _resolve_compute_backend(compute_backend)))
    stats["reliability_distribution"] = df["ReliabilityLevel"].value_counts().to_dict()
    stats["mean_geophysics_score"] = float(df["GeophysicsScore"].mean())
    stats["mean_geometry_score"] = float(df["GeometryScore"].mean())
    if progress_hook is not None:
        reliability_dist = df["ReliabilityLevel"].value_counts().to_dict()
        progress_hook(
            "Phase1 可靠性评分完成",
            (
                f"backend={stats['compute_backend']}, "
                f"high={int(reliability_dist.get('high', 0))}, "
                f"medium={int(reliability_dist.get('medium', 0))}, "
                f"low={int(reliability_dist.get('low', 0))}, "
                f"elapsed={_format_seconds(stats['step_seconds']['reliability_scoring'])}"
            ),
        )

    step_started = perf_counter()
    df, matched_pairs = boundary_match_only(
        df,
        boundary_tol_xy=boundary_tol_xy,
        compute_backend=compute_backend,
        gpu_tile_points=gpu_tile_points,
    )
    stats["step_seconds"]["boundary_match_only"] = float(perf_counter() - step_started)
    df.attrs["_boundary_matched_pairs"] = matched_pairs
    stats["boundary_matched_pairs"] = len(matched_pairs)
    stats["boundary_matched_patches"] = int((df["ConnectionType"] == "boundary_matched").sum())
    if progress_hook is not None:
        progress_hook(
            "Phase1 边界匹配完成",
            (
                f"matched_pairs={len(matched_pairs)}, "
                f"matched_patches={int((df['ConnectionType'] == 'boundary_matched').sum())}, "
                f"elapsed={_format_seconds(stats['step_seconds']['boundary_match_only'])}"
            ),
        )

    stats["output_count"] = len(df)
    return df, stats


# ---------------------------------------------------------------------------
#   主流程: Phase 2 (增强, 在 Phase 1 基础上)
# ---------------------------------------------------------------------------

def run_phase2(
    df: pd.DataFrame,
    *,
    smooth_bandwidth_xy: float = 150.0,
    smooth_bandwidth_z: float = 20.0,
    smooth_blend_alpha: float = 0.4,
    corridor_search_radius: float = 200.0,
    corridor_min_patches: int = 5,
    enable_supplement: bool = True,
    min_pair_confidence: float = 0.5,
    max_supplement_length: float = 80.0,
    boundary_tol_xy: float = 25.0,
    enable_aggregation: bool = True,
    agg_cluster_radius: float = 20.0,
    agg_min_patches: int = 3,
    agg_azimuth_tol: float = 25.0,
    agg_dip_tol: float = 15.0,
    agg_max_length: float = 80.0,
    agg_max_height: float = 40.0,
    elongation_max_stretch: float = 2.5,
    elongation_gap_fill: float = 0.7,
    elongation_max_neighbor_dist: float = 120.0,
    enable_elongation: bool = True,
    jitter_sigma_xy: float = 8.0,
    jitter_along_strike_factor: float = 1.5,
    jitter_seed: int = 42,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """第二轮: 组内平滑 + 走廊检测 + 聚合 + 沿走向拉伸 + 有限补接 + 空间扰动。

    前提: df 必须已经过 run_phase1 处理 (含 FractureSet, ReliabilityLevel 等)。
    """
    stats: dict[str, Any] = {"phase": 2, "input_count": len(df)}

    # Step 2: 组内产状平滑 (不覆盖原值)
    df = regional_orientation_smoothing(df, smooth_bandwidth_xy, smooth_bandwidth_z, smooth_blend_alpha)
    az_shift = _azimuth_diff(df["OrigAzimuth"].values, df["SmoothedAzimuth"].values)
    dip_shift = np.abs(df["OrigDip"].values - df["SmoothedDip"].values)
    stats["orientation_smoothing"] = {
        "mean_azimuth_shift_deg": float(np.nanmean(az_shift)),
        "mean_dip_shift_deg": float(np.nanmean(dip_shift)),
        "max_azimuth_shift_deg": float(np.nanmax(az_shift)),
    }
    if progress_hook is not None:
        progress_hook(
            "Phase2 产状平滑完成",
            (
                f"mean_az_shift={stats['orientation_smoothing']['mean_azimuth_shift_deg']:.2f}deg, "
                f"mean_dip_shift={stats['orientation_smoothing']['mean_dip_shift_deg']:.2f}deg"
            ),
        )

    # Step 3: 走廊检测 (CorridorSupport 证据)
    df = fracture_corridor_detection(df, corridor_search_radius, corridor_min_patches)
    n_in_corridors = int(df["CorridorSupport"].sum())
    stats["corridors_detected"] = int(df["CorridorID"].max() + 1) if n_in_corridors > 0 else 0
    stats["patches_in_corridors"] = n_in_corridors
    if progress_hook is not None:
        progress_hook(
            "Phase2 走廊识别完成",
            f"corridors={int(stats['corridors_detected'])}, patches_in_corridor={n_in_corridors}",
        )

    # Step 5b: 保守补接
    if enable_supplement:
        _, matched_pairs = boundary_match_only(df, boundary_tol_xy=boundary_tol_xy)
        count_before = len(df)
        df = conservative_boundary_supplement(
            df, matched_pairs,
            min_pair_confidence=min_pair_confidence,
            max_supplement_length=max_supplement_length,
        )
        stats["supplemented_patches"] = len(df) - count_before
    else:
        stats["supplemented_patches"] = 0

    # Step 6: 裂缝片聚合 (同组相邻小片 → 大裂缝面)
    if enable_aggregation:
        count_before_agg = len(df)
        df = aggregate_patches(
            df,
            cluster_radius=agg_cluster_radius,
            cluster_min_patches=agg_min_patches,
            azimuth_tol_deg=agg_azimuth_tol,
            dip_tol_deg=agg_dip_tol,
            max_merged_length=agg_max_length,
            max_merged_height=agg_max_height,
        )
        merged_count = count_before_agg - len(df)
        stats["aggregation"] = {
            "input_patches": count_before_agg,
            "output_patches": len(df),
            "merged_away": merged_count,
            "cluster_radius": agg_cluster_radius,
            "min_patches": agg_min_patches,
        }
    else:
        stats["aggregation"] = None

    # Step 7: 走廊内沿走向拉伸 (形成连续带)
    if enable_elongation:
        orig_lengths = df["PatchLength"].values.copy().astype(float)
        df = corridor_elongation(
            df,
            max_stretch_factor=elongation_max_stretch,
            gap_fill_fraction=elongation_gap_fill,
            max_neighbor_dist=elongation_max_neighbor_dist,
        )
        new_lengths = df["PatchLength"].values.astype(float)
        stretched_mask = new_lengths > orig_lengths[:len(new_lengths)] * 1.05
        stats["elongation"] = {
            "patches_stretched": int(stretched_mask.sum()),
            "mean_stretch_ratio": float(np.mean(new_lengths[stretched_mask] / orig_lengths[:len(new_lengths)][stretched_mask])) if stretched_mask.any() else 1.0,
            "max_stretch_factor": elongation_max_stretch,
            "gap_fill_fraction": elongation_gap_fill,
        }

    # Step 8: 空间扰动 (打破地震道网格规则排列)
    if jitter_sigma_xy > 0:
        df = spatial_perturbation(df, jitter_sigma_xy, jitter_along_strike_factor, jitter_seed)
        stats["spatial_perturbation"] = {
            "jitter_sigma_xy": jitter_sigma_xy,
            "along_strike_factor": jitter_along_strike_factor,
        }

    stats["output_count"] = len(df)
    return df, stats


# ---------------------------------------------------------------------------
#   VTK I/O 适配
# ---------------------------------------------------------------------------

def _df_from_vtk(payload: dict[str, Any]) -> pd.DataFrame:
    """从 VTK payload 构建 DataFrame。"""
    points = np.asarray(payload["points"], dtype=float)
    polygons = payload["polygons"]
    cell_data = payload["cell_data"]
    n_cells = len(polygons)

    df = pd.DataFrame()
    for name, values in cell_data.items():
        df[name] = values

    centers = np.zeros((n_cells, 3))
    for i, poly in enumerate(polygons):
        centers[i] = points[poly].mean(axis=0)
    df["CenterX"] = centers[:, 0]
    df["CenterY"] = centers[:, 1]
    df["CenterTIME"] = centers[:, 2]

    for vi in range(1, 5):
        vx, vy, vz = [], [], []
        for i, poly in enumerate(polygons):
            if vi - 1 < len(poly):
                pt = points[poly[vi - 1]]
                vx.append(pt[0]); vy.append(pt[1]); vz.append(pt[2])
            else:
                vx.append(centers[i, 0]); vy.append(centers[i, 1]); vz.append(centers[i, 2])
        df[f"V{vi}X"] = vx
        df[f"V{vi}Y"] = vy
        df[f"V{vi}Z"] = vz

    for col, default in [("Azimuth", 0.0), ("Dip", 45.0), ("Confidence", 0.5),
                          ("PatchLength", 10.0), ("PatchHeight", 10.0)]:
        if col not in df.columns:
            df[col] = default

    return df


def _df_to_vtk(df: pd.DataFrame, title: str, output_vtk: Path,
               scalar_types: dict[str, str] | None = None) -> None:
    """从 DataFrame 写回 VTK。"""
    new_points: list[list[float]] = []
    new_polygons: list[list[int]] = []

    skip_cols = {"CenterX", "CenterY", "CenterTIME"}
    skip_cols |= {f"V{vi}{c}" for vi in range(1, 5) for c in ("X", "Y", "Z")}
    data_cols = [c for c in df.columns if c not in skip_cols]

    cell_data_lists: dict[str, list[Any]] = {c: [] for c in data_cols}

    for _, row in df.iterrows():
        start_idx = len(new_points)
        for vi in range(1, 5):
            new_points.append([float(row[f"V{vi}X"]), float(row[f"V{vi}Y"]), float(row[f"V{vi}Z"])])
        new_polygons.append(list(range(start_idx, start_idx + 4)))
        for col in data_cols:
            cell_data_lists[col].append(row[col])

    out_points = np.array(new_points, dtype=float)
    out_cell_data: dict[str, np.ndarray] = {}
    out_scalar_types: dict[str, str] = dict(scalar_types) if scalar_types else {}

    int_cols = {"FractureSet", "CorridorID", "CorridorSupport", "NeighborCount", "IsSupplemented", "ConnectionID"}
    str_cols = {"ReliabilityLevel", "ConnectionType"}
    str_maps: dict[str, dict[str, int]] = {
        "ReliabilityLevel": {"high": 2, "medium": 1, "low": 0},
        "ConnectionType": {"original": 0, "boundary_matched": 1, "supplemented": 2, "boundary_connected": 3},
    }

    for col, vals_list in cell_data_lists.items():
        arr = np.array(vals_list)
        if col in str_cols:
            mapping = str_maps.get(col, {})
            encoded = np.array([mapping.get(str(v), -1) for v in vals_list], dtype=int)
            out_cell_data[col] = encoded
            out_scalar_types[col] = "int"
        elif col in int_cols:
            out_cell_data[col] = arr.astype(int)
            out_scalar_types[col] = "int"
        else:
            try:
                out_cell_data[col] = arr.astype(float)
                out_scalar_types[col] = "float"
            except (ValueError, TypeError):
                continue

    write_legacy_vtk_polygons(
        path=output_vtk, title=title,
        points=out_points, polygons=new_polygons,
        cell_data=out_cell_data, scalar_types=out_scalar_types,
    )


def postprocess_vtk(
    input_vtk: Path, output_vtk: Path,
    phase: int = 1, **kwargs: Any,
) -> dict[str, Any]:
    """读取合并后 VTK → 后处理 → 写回 VTK。"""
    payload = read_legacy_vtk_polygons(input_vtk)
    df = _df_from_vtk(payload)

    # 提取 boundary_connect / seam_fill 参数
    bc_keys = {"enable_boundary_connect", "boundary_strip_width", "connect_max_gap",
               "bc_azimuth_tol_deg", "bc_dip_tol_deg", "bc_min_score_threshold",
               "bc_corridor_bonus", "bc_require_corridor",
               "bc_perp_ratio", "bc_enable_supplement", "bc_max_stretch_ratio",
               "enable_seam_fill", "seam_half_width", "seam_interior_depth",
               "seam_fill_fraction", "seam_confidence_decay", "seam_seed",
               "seam_blend_half_width", "seam_blend_strength",
               "seam_min_shared_ratio", "seam_high_rel_boost"}
    bc_kwargs_raw = {k: v for k, v in kwargs.items() if k in bc_keys}
    enable_bc = bc_kwargs_raw.pop("enable_boundary_connect", False)
    enable_sf = bc_kwargs_raw.pop("enable_seam_fill", False)
    rest_kwargs = {k: v for k, v in kwargs.items() if k not in bc_keys}

    if phase == 1:
        df, stats = run_phase1(df, **rest_kwargs)
    elif phase == 2:
        p1_keys = {"max_fracture_sets", "n_fracture_sets", "neighbor_radius_xy",
                    "neighbor_radius_z", "isolation_min_neighbors", "confidence_floor",
                    "boundary_tol_xy", "length_range", "height_range"}
        p1_kwargs = {k: v for k, v in rest_kwargs.items() if k in p1_keys}
        p2_kwargs = {k: v for k, v in rest_kwargs.items() if k not in p1_keys}
        df, stats1 = run_phase1(df, **p1_kwargs)
        df, stats2 = run_phase2(df, **p2_kwargs)
        stats = {
            **stats1,
            "phase1_output_count": stats1.get("output_count"),
            "phase2": stats2,
            "output_count": stats2.get("output_count", stats1.get("output_count")),
        }
    else:
        raise ValueError(f"phase must be 1 or 2, got {phase}")

    # Step 9: 边界带局部连接 (可选)
    if enable_bc:
        # 映射参数名: bc_xxx → boundary_connect_postprocess 参数名
        bc_param_map = {
            "boundary_strip_width": "boundary_strip_width",
            "connect_max_gap": "connect_max_gap",
            "bc_azimuth_tol_deg": "azimuth_tol_deg",
            "bc_dip_tol_deg": "dip_tol_deg",
            "bc_min_score_threshold": "min_score_threshold",
            "bc_corridor_bonus": "corridor_bonus",
            "bc_require_corridor": "require_corridor",
            "bc_perp_ratio": "perp_ratio",
            "bc_enable_supplement": "enable_supplement",
            "bc_max_stretch_ratio": "max_stretch_ratio",
        }
        bc_args = {bc_param_map[k]: v for k, v in bc_kwargs_raw.items() if k in bc_param_map}
        df, bc_stats = boundary_connect_postprocess(df, **bc_args)
        stats["boundary_connect"] = bc_stats

    # Step 9b: 边界缝合填充 (可选)
    if enable_sf:
        sf_param_map = {
            "seam_half_width": "seam_half_width",
            "seam_interior_depth": "interior_sample_depth",
            "seam_fill_fraction": "fill_fraction",
            "seam_confidence_decay": "confidence_decay",
            "seam_seed": "seed",
            "seam_blend_half_width": "blend_half_width",
            "seam_blend_strength": "blend_strength",
            "seam_min_shared_ratio": "min_shared_set_ratio",
            "seam_high_rel_boost": "high_reliability_boost",
        }
        sf_args = {sf_param_map[k]: v for k, v in bc_kwargs_raw.items() if k in sf_param_map}
        df, sf_stats = boundary_seam_fill(df, **sf_args)
        stats["seam_fill"] = sf_stats

    title = f"{payload.get('title', 'DFN')}_postprocessed_phase{phase}"
    _df_to_vtk(df, title, output_vtk, payload.get("scalar_types"))
    return stats


# ---------------------------------------------------------------------------
#   CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="地球物理约束后处理 — 提升合并 DFN 的区域连贯性与可靠性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 第一轮 (保守): 裂缝组 + 可靠性评分 + 边界匹配标记
  python geophysical_postprocess.py --input-vtk merged.vtk --phase 1

  # 第二轮 (增强): 在第一轮基础上加平滑 + 走廊 + 保守补接
  python geophysical_postprocess.py --input-vtk merged.vtk --phase 2

  # 第一轮, 指定 3 组裂缝
  python geophysical_postprocess.py --input-vtk merged.vtk --phase 1 --n-sets 3

  # 第二轮, 禁止补接 (只做平滑和走廊标注)
  python geophysical_postprocess.py --input-vtk merged.vtk --phase 2 --no-supplement

  # 第二轮 + 边界带局部连接 (Step 9)
  python geophysical_postprocess.py --input-vtk merged.vtk --phase 2 --boundary-connect

  # 对 CSV 做第一轮后处理
  python geophysical_postprocess.py --input-csv merged_patches.csv --phase 1

VTK 中新增属性编码:
  ReliabilityLevel: 0=low, 1=medium, 2=high
  ConnectionType: 0=original, 1=boundary_matched, 2=supplemented, 3=boundary_connected
""",
    )
    g_input = p.add_mutually_exclusive_group(required=True)
    g_input.add_argument("--input-vtk", type=Path, help="合并后的 VTK 文件")
    g_input.add_argument("--input-csv", type=Path, help="合并后的裂缝片 CSV 文件")
    p.add_argument("--output-vtk", type=Path)
    p.add_argument("--output-csv", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2],
                   help="处理阶段: 1=保守, 2=增强 (default: 1)")
    p.add_argument("--boundary-connect", action="store_true",
                   help="Phase 2 后追加边界带局部连接 (Step 9)")

    g1 = p.add_argument_group("Step 1: 裂缝组识别")
    g1.add_argument("--max-sets", type=int, default=6)
    g1.add_argument("--n-sets", type=int, default=None)

    g2 = p.add_argument_group("Step 2: 组内产状平滑 (Phase 2)")
    g2.add_argument("--smooth-bw-xy", type=float, default=150.0)
    g2.add_argument("--smooth-bw-z", type=float, default=20.0)
    g2.add_argument("--smooth-alpha", type=float, default=0.4)

    g3 = p.add_argument_group("Step 3: 走廊检测 (Phase 2)")
    g3.add_argument("--corridor-radius", type=float, default=200.0)
    g3.add_argument("--corridor-min", type=int, default=5)

    g4 = p.add_argument_group("Step 4: 可靠性评分")
    g4.add_argument("--neighbor-radius-xy", type=float, default=100.0)
    g4.add_argument("--neighbor-radius-z", type=float, default=15.0)
    g4.add_argument("--min-neighbors", type=int, default=2)
    g4.add_argument("--conf-floor", type=float, default=0.3)

    g5 = p.add_argument_group("Step 5: 跨单元衔接")
    g5.add_argument("--boundary-tol", type=float, default=25.0)
    g5.add_argument("--no-supplement", action="store_true",
                    help="Phase 2 时禁止补片")
    g5.add_argument("--min-pair-conf", type=float, default=0.5,
                    help="补接要求的最低配对置信度")
    g5.add_argument("--max-supplement-len", type=float, default=80.0,
                    help="补接最大间距 (m)")

    g6 = p.add_argument_group("Step 6: 裂缝片聚合 (Phase 2)")
    g6.add_argument("--agg-radius", type=float, default=20.0,
                    help="聚合聚类半径 (m) (default: 20.0)")
    g6.add_argument("--agg-min", type=int, default=3,
                    help="聚合最少片数 (default: 3)")
    g6.add_argument("--agg-azimuth-tol", type=float, default=25.0,
                    help="聚合方位角容差 (°) (default: 25.0)")
    g6.add_argument("--agg-dip-tol", type=float, default=15.0,
                    help="聚合倾角容差 (°) (default: 15.0)")
    g6.add_argument("--agg-max-length", type=float, default=80.0,
                    help="聚合后单片最大长度 (m) (default: 80.0)")
    g6.add_argument("--agg-max-height", type=float, default=40.0,
                    help="聚合后单片最大高度 (m) (default: 40.0)")
    g6.add_argument("--no-aggregation", action="store_true",
                    help="Phase 2 时禁止裂缝片聚合")

    g7 = p.add_argument_group("Step 7: 沿走向拉伸 (Phase 2)")
    g7.add_argument("--elongation-stretch", type=float, default=2.5,
                    help="最大拉伸倍数 (default: 2.5)")
    g7.add_argument("--elongation-fill", type=float, default=0.7,
                    help="间隙填充比例 (default: 0.7)")
    g7.add_argument("--elongation-range", type=float, default=120.0,
                    help="邻居搜索距离 (m) (default: 120.0)")
    g7.add_argument("--no-elongation", action="store_true",
                    help="Phase 2 时禁止沿走向拉伸")

    g8 = p.add_argument_group("Step 8: 空间扰动 (Phase 2)")
    g8.add_argument("--jitter-xy", type=float, default=8.0,
                    help="XY 平面高斯扰动 σ (m), 0=关闭 (default: 8.0)")
    g8.add_argument("--jitter-strike-factor", type=float, default=1.5,
                    help="沿走向扰动放大系数 (default: 1.5)")
    g8.add_argument("--jitter-seed", type=int, default=42,
                    help="扰动随机种子 (default: 42)")
    g8.add_argument("--no-jitter", action="store_true",
                    help="Phase 2 时禁止空间扰动")

    g9 = p.add_argument_group("Step 9: 边界带局部连接 (--boundary-connect)")
    g9.add_argument("--bc-strip-width", type=float, default=40.0,
                    help="边界窄带半宽 (m) (default: 40.0)")
    g9.add_argument("--bc-max-gap", type=float, default=60.0,
                    help="最大连接间距 (m) (default: 60.0)")
    g9.add_argument("--bc-azimuth-tol", type=float, default=30.0,
                    help="连接方位角容差 (°), 聚合后宾宽 (default: 30.0)")
    g9.add_argument("--bc-dip-tol", type=float, default=20.0,
                    help="连接倾角容差 (°), 聚合后宾宽 (default: 20.0)")
    g9.add_argument("--bc-min-score", type=float, default=0.35,
                    help="连接最低综合分 (default: 0.35)")
    g9.add_argument("--bc-corridor-bonus", type=float, default=0.15,
                    help="走廊内匹配对加分 (default: 0.15)")
    g9.add_argument("--bc-require-corridor", action="store_true",
                    help="强制要求至少一端在走廊内")
    g9.add_argument("--bc-perp-ratio", type=float, default=0.8,
                    help="垂直走向偏移 / 间距 最大比 (default: 0.8)")
    g9.add_argument("--bc-no-supplement", action="store_true",
                    help="禁止边界补片 (只拉伸不补)")
    g9.add_argument("--bc-max-stretch", type=float, default=2.5,
                    help="连接拉伸最大倍数 (default: 2.5)")

    g9b = p.add_argument_group("Step 9b: 边界缝合填充 (--bc-seam-fill)")
    g9b.add_argument("--bc-seam-fill", action="store_true",
                     help="在 boundary-connect 后追加密度缝合填充")
    g9b.add_argument("--bc-seam-width", type=float, default=50.0,
                     help="缝合带半宽 (m) (default: 50.0)")
    g9b.add_argument("--bc-seam-interior", type=float, default=80.0,
                     help="内部采样深度 (m) (default: 80.0)")
    g9b.add_argument("--bc-seam-fraction", type=float, default=1.0,
                     help="新增片数 / 内部参考片数 (default: 1.0)")
    g9b.add_argument("--bc-seam-decay", type=float, default=0.65,
                     help="新片置信度衰减系数 (default: 0.65)")
    g9b.add_argument("--bc-seam-seed", type=int, default=42,
                     help="缝合填充随机种子 (default: 42)")
    g9b.add_argument("--bc-blend-width", type=float, default=60.0,
                     help="属性混合区半宽 (m) (default: 60.0)")
    g9b.add_argument("--bc-blend-strength", type=float, default=0.5,
                     help="最大混合强度 0-1 (default: 0.5)")
    g9b.add_argument("--bc-min-shared-ratio", type=float, default=0.15,
                     help="共享组最低双侧占比 (default: 0.15)")
    g9b.add_argument("--bc-high-rel-boost", type=float, default=1.5,
                     help="高可靠段落填充加成倍数 (default: 1.5)")

    g_runtime = p.add_argument_group("Runtime")
    g_runtime.add_argument(
        "--compute-backend",
        type=str,
        default=DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
        choices=["auto", "cpu", "gpu"],
        help="数值密集步骤的计算后端",
    )
    g_runtime.add_argument(
        "--max-cpu-threads",
        type=int,
        default=DEFAULT_POSTPROCESS_MAX_CPU_THREADS,
        help="CPU 数值库最大线程数，0 表示不限制",
    )
    g_runtime.add_argument(
        "--gpu-tile-points",
        type=int,
        default=DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
        help="GPU 分块近邻计算的 tile 大小",
    )

    p.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.input_vtk:
        input_path = Path(args.input_vtk)
        if not input_path.exists():
            raise FileNotFoundError(f"input VTK not found: {input_path}")
        output_vtk = Path(args.output_vtk) if args.output_vtk else input_path.with_name(
            f"{input_path.stem}_phase{args.phase}_{ts}{input_path.suffix}"
        )
        if output_vtk.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {output_vtk}")

        vtk_kwargs: dict[str, Any] = dict(
            phase=args.phase,
            max_fracture_sets=args.max_sets,
            n_fracture_sets=args.n_sets,
            neighbor_radius_xy=args.neighbor_radius_xy,
            neighbor_radius_z=args.neighbor_radius_z,
            isolation_min_neighbors=args.min_neighbors,
            confidence_floor=args.conf_floor,
            boundary_tol_xy=args.boundary_tol,
            compute_backend=args.compute_backend,
            max_cpu_threads=args.max_cpu_threads,
            gpu_tile_points=args.gpu_tile_points,
        )
        if args.phase == 2:
            vtk_kwargs.update(
                smooth_bandwidth_xy=args.smooth_bw_xy,
                smooth_bandwidth_z=args.smooth_bw_z,
                smooth_blend_alpha=args.smooth_alpha,
                corridor_search_radius=args.corridor_radius,
                corridor_min_patches=args.corridor_min,
                enable_supplement=not args.no_supplement,
                min_pair_confidence=args.min_pair_conf,
                max_supplement_length=args.max_supplement_len,
                enable_aggregation=not args.no_aggregation,
                agg_cluster_radius=args.agg_radius,
                agg_min_patches=args.agg_min,
                agg_azimuth_tol=args.agg_azimuth_tol,
                agg_dip_tol=args.agg_dip_tol,
                agg_max_length=args.agg_max_length,
                agg_max_height=args.agg_max_height,
                enable_elongation=not args.no_elongation,
                elongation_max_stretch=args.elongation_stretch,
                elongation_gap_fill=args.elongation_fill,
                elongation_max_neighbor_dist=args.elongation_range,
                jitter_sigma_xy=0.0 if args.no_jitter else args.jitter_xy,
                jitter_along_strike_factor=args.jitter_strike_factor,
                jitter_seed=args.jitter_seed,
            )
        if args.boundary_connect:
            vtk_kwargs.update(
                enable_boundary_connect=True,
                boundary_strip_width=args.bc_strip_width,
                connect_max_gap=args.bc_max_gap,
                bc_azimuth_tol_deg=args.bc_azimuth_tol,
                bc_dip_tol_deg=args.bc_dip_tol,
                bc_min_score_threshold=args.bc_min_score,
                bc_corridor_bonus=args.bc_corridor_bonus,
                bc_require_corridor=args.bc_require_corridor,
                bc_perp_ratio=args.bc_perp_ratio,
                bc_enable_supplement=not args.bc_no_supplement,
                bc_max_stretch_ratio=args.bc_max_stretch,
            )
        if args.bc_seam_fill:
            vtk_kwargs.update(
                enable_seam_fill=True,
                seam_half_width=args.bc_seam_width,
                seam_interior_depth=args.bc_seam_interior,
                seam_fill_fraction=args.bc_seam_fraction,
                seam_confidence_decay=args.bc_seam_decay,
                seam_seed=args.bc_seam_seed,
                seam_blend_half_width=args.bc_blend_width,
                seam_blend_strength=args.bc_blend_strength,
                seam_min_shared_ratio=args.bc_min_shared_ratio,
                seam_high_rel_boost=args.bc_high_rel_boost,
            )

        stats = postprocess_vtk(input_path, output_vtk, **vtk_kwargs)
        print(f"[postprocess] phase {args.phase} VTK saved: {output_vtk}")

    elif args.input_csv:
        input_path = Path(args.input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"input CSV not found: {input_path}")
        df = pd.read_csv(input_path, encoding="utf-8-sig")

        if args.phase == 1:
            df, stats = run_phase1(
                df,
                max_fracture_sets=args.max_sets,
                n_fracture_sets=args.n_sets,
                neighbor_radius_xy=args.neighbor_radius_xy,
                neighbor_radius_z=args.neighbor_radius_z,
                isolation_min_neighbors=args.min_neighbors,
                confidence_floor=args.conf_floor,
                boundary_tol_xy=args.boundary_tol,
                compute_backend=args.compute_backend,
                gpu_tile_points=args.gpu_tile_points,
            )
        else:
            df, stats1 = run_phase1(
                df,
                max_fracture_sets=args.max_sets,
                n_fracture_sets=args.n_sets,
                neighbor_radius_xy=args.neighbor_radius_xy,
                neighbor_radius_z=args.neighbor_radius_z,
                isolation_min_neighbors=args.min_neighbors,
                confidence_floor=args.conf_floor,
                boundary_tol_xy=args.boundary_tol,
                compute_backend=args.compute_backend,
                gpu_tile_points=args.gpu_tile_points,
            )
            df, stats2 = run_phase2(
                df,
                smooth_bandwidth_xy=args.smooth_bw_xy,
                smooth_bandwidth_z=args.smooth_bw_z,
                smooth_blend_alpha=args.smooth_alpha,
                corridor_search_radius=args.corridor_radius,
                corridor_min_patches=args.corridor_min,
                enable_supplement=not args.no_supplement,
                min_pair_confidence=args.min_pair_conf,
                max_supplement_length=args.max_supplement_len,
                boundary_tol_xy=args.boundary_tol,
                enable_aggregation=not args.no_aggregation,
                agg_cluster_radius=args.agg_radius,
                agg_min_patches=args.agg_min,
                agg_azimuth_tol=args.agg_azimuth_tol,
                agg_dip_tol=args.agg_dip_tol,
                agg_max_length=args.agg_max_length,
                agg_max_height=args.agg_max_height,
                enable_elongation=not args.no_elongation,
                elongation_max_stretch=args.elongation_stretch,
                elongation_gap_fill=args.elongation_fill,
                elongation_max_neighbor_dist=args.elongation_range,
                jitter_sigma_xy=0.0 if args.no_jitter else args.jitter_xy,
                jitter_along_strike_factor=args.jitter_strike_factor,
                jitter_seed=args.jitter_seed,
                compute_backend=args.compute_backend,
                gpu_tile_points=args.gpu_tile_points,
            )
            stats = {
                **stats1,
                "phase1_output_count": stats1.get("output_count"),
                "phase2": stats2,
                "output_count": stats2.get("output_count", stats1.get("output_count")),
            }

        if args.boundary_connect:
            df, bc_stats = boundary_connect_postprocess(
                df,
                boundary_strip_width=args.bc_strip_width,
                connect_max_gap=args.bc_max_gap,
                azimuth_tol_deg=args.bc_azimuth_tol,
                dip_tol_deg=args.bc_dip_tol,
                min_score_threshold=args.bc_min_score,
                corridor_bonus=args.bc_corridor_bonus,
                require_corridor=args.bc_require_corridor,
                perp_ratio=args.bc_perp_ratio,
                enable_supplement=not args.bc_no_supplement,
                max_stretch_ratio=args.bc_max_stretch,
                compute_backend=args.compute_backend,
                gpu_tile_points=args.gpu_tile_points,
            )
            stats["boundary_connect"] = bc_stats

        if args.bc_seam_fill:
            df, sf_stats = boundary_seam_fill(
                df,
                seam_half_width=args.bc_seam_width,
                interior_sample_depth=args.bc_seam_interior,
                fill_fraction=args.bc_seam_fraction,
                confidence_decay=args.bc_seam_decay,
                blend_half_width=args.bc_blend_width,
                blend_strength=args.bc_blend_strength,
                min_shared_set_ratio=args.bc_min_shared_ratio,
                high_reliability_boost=args.bc_high_rel_boost,
                seed=args.bc_seam_seed,
            )
            stats["seam_fill"] = sf_stats

        output_csv = Path(args.output_csv) if args.output_csv else input_path.with_name(
            f"{input_path.stem}_phase{args.phase}_{ts}.csv"
        )
        write_csv_utf8(df, output_csv)
        print(f"[postprocess] phase {args.phase} CSV saved: {output_csv}")

    summary_path = input_path.with_name(f"postprocess_phase{args.phase}_summary_{ts}.json")
    write_json(summary_path, stats)
    print(f"[postprocess] summary: {summary_path}")

    log_lines = [
        f"阶段: Phase {args.phase}",
        f"输入: {input_path}",
        f"裂缝片数: {stats.get('input_count', '?')} → {stats.get('output_count', '?')}",
        f"裂缝组: {stats.get('fracture_sets', '?')} 组",
        f"可靠性分布: {stats.get('reliability_distribution', {})}",
        f"边界匹配对: {stats.get('boundary_matched_pairs', 0)}",
    ]
    if args.phase == 2 and "phase2" in stats:
        p2 = stats["phase2"]
        log_lines += [
            f"走廊: {p2.get('corridors_detected', 0)} 条, {p2.get('patches_in_corridors', 0)} 片",
            f"补充片: {p2.get('supplemented_patches', 0)}",
            f"平滑: Δaz={p2.get('orientation_smoothing', {}).get('mean_azimuth_shift_deg', 0):.1f}°",
        ]
        ag = p2.get("aggregation")
        if ag:
            log_lines.append(f"聚合: {ag['input_patches']}片→{ag['output_patches']}片 (合并{ag['merged_away']}片)")
        el = p2.get("elongation")
        if el:
            log_lines.append(f"沿走向拉伸: {el['patches_stretched']} 片, 平均{el['mean_stretch_ratio']:.2f}×")
        sp = p2.get("spatial_perturbation")
        if sp:
            log_lines.append(f"空间扰动: σ_xy={sp['jitter_sigma_xy']}m, 走向因子={sp['along_strike_factor']}")
    bc = stats.get("boundary_connect")
    if bc:
        log_lines.append(
            f"边界连接: {bc.get('connections_made', 0)} 对 "
            f"(走廊内{bc.get('connections_with_corridor', 0)}对), "
            f"补片{bc.get('supplements_added', 0)}个, "
            f"边界带片数={bc.get('boundary_patches', 0)}, "
            f"平均间距={bc.get('mean_gap_m', 0):.1f}m, "
            f"输出={bc.get('output_count', '?')}片"
        )
    sf = stats.get("seam_fill")
    if sf:
        log_lines.append(
            f"缝合填充: {sf.get('seam_fills', 0)} 新片 + "
            f"{sf.get('blended_patches', 0)} 片属性混合 "
            f"(涉及 {sf.get('adjacent_edges', 0)} 条边界), "
            f"输出={sf.get('output_count', '?')}片"
        )
    try:
        append_lines_to_docx(args.docx_path, f"地球物理后处理 Phase{args.phase} {ts}", log_lines)
    except Exception:
        pass
    for line in log_lines:
        print(f"  {line}")


def boundary_connect_postprocess(
    df: pd.DataFrame,
    *,
    boundary_strip_width: float = 40.0,
    connect_max_gap: float = 60.0,
    macro_connect_gap: float = 180.0,
    connect_minor_limit: float = 25.0,
    connect_z_gap: float = 20.0,
    azimuth_tol_deg: float = 30.0,
    dip_tol_deg: float = 20.0,
    min_score_threshold: float = 0.35,
    corridor_bonus: float = 0.15,
    require_corridor: bool = False,
    perp_ratio: float = 0.8,
    enable_supplement: bool = True,
    max_stretch_ratio: float = 2.5,
    supplement_confidence_decay: float = 0.7,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Boundary connection with multiscale gap control and anisotropic filtering."""
    df = df.copy()
    stats: dict[str, Any] = {"input_count": len(df)}

    has_block = "BlockX" in df.columns and "BlockY" in df.columns
    has_unit = "UnitID" in df.columns
    if not has_block and not has_unit:
        stats["connections_made"] = 0
        stats["reason"] = "no_unit_info"
        return df, stats

    if "ConnectionID" not in df.columns:
        df["ConnectionID"] = -1
    if "PatchArea" not in df.columns:
        df["PatchArea"] = _numeric_series_or_default(df, "PatchLength", 0.0) * _numeric_series_or_default(df, "PatchHeight", 0.0)

    work_az_arr, work_dip_arr = _get_work_orientation(df)
    scale_class_arr = (
        df["ScaleClass"].fillna("meso_link").astype(str).to_numpy()
        if "ScaleClass" in df.columns
        else np.array(["meso_link"] * len(df), dtype=object)
    )

    if has_block:
        bx_arr = df["BlockX"].to_numpy()
        by_arr = df["BlockY"].to_numpy()
        unit_key_arr = [f"{int(bx_arr[i])}_{int(by_arr[i])}" for i in range(len(df))]
    else:
        uid_arr = df["UnitID"].fillna("").astype(str).to_numpy(dtype=object)
        unit_key_arr = [str(uid_arr[i]) for i in range(len(df))]

    cx_arr = pd.to_numeric(df["CenterX"], errors="coerce").to_numpy(dtype=float)
    cy_arr = pd.to_numeric(df["CenterY"], errors="coerce").to_numpy(dtype=float)
    cz_arr = pd.to_numeric(df["CenterTIME"], errors="coerce").to_numpy(dtype=float)
    unit_bounds: dict[Any, dict[str, float]] = {}
    for i, key in enumerate(unit_key_arr):
        if key not in unit_bounds:
            unit_bounds[key] = {
                "x_min": cx_arr[i],
                "x_max": cx_arr[i],
                "y_min": cy_arr[i],
                "y_max": cy_arr[i],
            }
        else:
            bounds = unit_bounds[key]
            bounds["x_min"] = min(bounds["x_min"], cx_arr[i])
            bounds["x_max"] = max(bounds["x_max"], cx_arr[i])
            bounds["y_min"] = min(bounds["y_min"], cy_arr[i])
            bounds["y_max"] = max(bounds["y_max"], cy_arr[i])

    is_boundary = np.zeros(len(df), dtype=bool)
    for i, key in enumerate(unit_key_arr):
        bounds = unit_bounds[key]
        cx = cx_arr[i]
        cy = cy_arr[i]
        if (
            cx - bounds["x_min"] < boundary_strip_width
            or bounds["x_max"] - cx < boundary_strip_width
            or cy - bounds["y_min"] < boundary_strip_width
            or bounds["y_max"] - cy < boundary_strip_width
        ):
            is_boundary[i] = True

    df["_unit_key"] = unit_key_arr
    boundary_idx = np.where(is_boundary)[0]
    stats["boundary_patches"] = int(len(boundary_idx))
    if len(boundary_idx) < 2:
        stats.update(connections_made=0, supplements_added=0, output_count=len(df))
        df.drop(columns=["_unit_key"], inplace=True)
        return df, stats

    search_radius = max(connect_max_gap, macro_connect_gap)
    boundary_coords = np.column_stack([cx_arr[boundary_idx], cy_arr[boundary_idx]])
    pair_left_local, pair_right_local, resolved_backend = _radius_candidate_pairs(
        coords=boundary_coords,
        radius=float(search_radius),
        compute_backend=compute_backend,
        gpu_tile_points=int(gpu_tile_points),
    )
    stats["compute_backend"] = resolved_backend
    connections: list[dict[str, Any]] = []

    if len(pair_left_local) > 0:
        pair_left = boundary_idx[pair_left_local]
        pair_right = boundary_idx[pair_right_local]
        unit_code, _ = pd.factorize(np.asarray(unit_key_arr, dtype=object), sort=False)
        layer_code, _ = pd.factorize(_preferred_layer_series(df, prefer_unit_segment=False), sort=False)
        fracture_set_series = df["FractureSet"] if "FractureSet" in df.columns else pd.Series([-1] * len(df), index=df.index, dtype=int)
        corridor_series = df["CorridorSupport"] if "CorridorSupport" in df.columns else pd.Series([0] * len(df), index=df.index, dtype=int)
        fracture_set_arr = pd.to_numeric(fracture_set_series, errors="coerce").fillna(-1).to_numpy(dtype=int)
        corridor_arr = pd.to_numeric(corridor_series, errors="coerce").fillna(0).to_numpy(dtype=int)

        score_parts: list[np.ndarray] = []
        for col in ("GeophysicsScore", "GeometryScore", "SetProbability", "HierarchyScore"):
            if col in df.columns:
                score_parts.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        if score_parts:
            score_mean_arr = np.mean(np.vstack(score_parts), axis=0)
        else:
            score_mean_arr = np.full(len(df), 0.5, dtype=float)

        keep_mask = unit_code[pair_left] != unit_code[pair_right]
        scale_left = scale_class_arr[pair_left]
        scale_right = scale_class_arr[pair_right]
        keep_mask &= ~((scale_left == "micro_bg") & (scale_right == "micro_bg"))
        keep_mask &= fracture_set_arr[pair_left] == fracture_set_arr[pair_right]
        keep_mask &= layer_code[pair_left] == layer_code[pair_right]

        az_diff = _azimuth_diff(work_az_arr[pair_left], work_az_arr[pair_right])
        dip_diff = np.abs(work_dip_arr[pair_left] - work_dip_arr[pair_right])
        keep_mask &= az_diff <= float(azimuth_tol_deg)
        keep_mask &= dip_diff <= float(dip_tol_deg)

        has_corridor = (corridor_arr[pair_left] > 0) | (corridor_arr[pair_right] > 0)
        if require_corridor:
            keep_mask &= has_corridor

        pair_score = (score_mean_arr[pair_left] + score_mean_arr[pair_right]) / 2.0
        pair_score += np.where(has_corridor, float(corridor_bonus), 0.0)
        keep_mask &= pair_score >= float(min_score_threshold)

        dx = cx_arr[pair_right] - cx_arr[pair_left]
        dy = cy_arr[pair_right] - cy_arr[pair_left]
        dz = np.abs(cz_arr[pair_right] - cz_arr[pair_left])
        euclidean_gap = np.hypot(dx, dy)
        mean_az = _pairwise_mean_axial_azimuth_deg(work_az_arr[pair_left], work_az_arr[pair_right])
        mean_az_rad = np.deg2rad(mean_az)
        along_dist = np.abs(dx * np.cos(mean_az_rad) + dy * np.sin(mean_az_rad))
        perp_dist = np.abs(-dx * np.sin(mean_az_rad) + dy * np.cos(mean_az_rad))
        target_gap = np.where(
            (scale_left == "macro_core") | (scale_right == "macro_core"),
            float(macro_connect_gap),
            float(connect_max_gap),
        )
        keep_mask &= along_dist <= target_gap
        keep_mask &= perp_dist <= float(connect_minor_limit)
        keep_mask &= dz <= float(connect_z_gap)
        keep_mask &= perp_dist <= euclidean_gap * float(perp_ratio) + 1e-6

        if keep_mask.any():
            kept_left = pair_left[keep_mask]
            kept_right = pair_right[keep_mask]
            kept_score = pair_score[keep_mask]
            kept_along = along_dist[keep_mask]
            kept_gap = euclidean_gap[keep_mask]
            kept_dz = dz[keep_mask]
            kept_corridor = has_corridor[keep_mask]
            kept_scale_left = scale_left[keep_mask]
            kept_scale_right = scale_right[keep_mask]

            order = np.argsort(kept_score)[::-1]
            for order_idx in order:
                connections.append(
                    {
                        "i": int(kept_left[order_idx]),
                        "j": int(kept_right[order_idx]),
                        "along_gap": float(kept_along[order_idx]),
                        "euclidean_gap": float(kept_gap[order_idx]),
                        "z_gap": float(kept_dz[order_idx]),
                        "score": float(kept_score[order_idx]),
                        "has_corridor": bool(kept_corridor[order_idx]),
                        "scale_i": str(kept_scale_left[order_idx]),
                        "scale_j": str(kept_scale_right[order_idx]),
                    }
                )

    def _rebuild_vertices(target_idx: int, new_len: float) -> None:
        row = df.iloc[target_idx]
        az = float(row.get("WorkAzimuth", row.get("Azimuth", 0.0)))
        dip = float(row.get("WorkDip", row.get("Dip", 45.0)))
        az_rad = np.deg2rad(az)
        dip_rad = np.deg2rad(dip)
        u_hat = np.array([np.cos(az_rad), np.sin(az_rad), 0.0])
        perp_x, perp_y = -np.sin(az_rad), np.cos(az_rad)
        v_hat = np.array([
            perp_x * np.cos(dip_rad),
            perp_y * np.cos(dip_rad),
            -np.sin(dip_rad),
        ])
        v_norm = np.linalg.norm(v_hat)
        v_hat = v_hat / v_norm if v_norm > 1e-6 else np.array([0.0, 0.0, -1.0])
        center = np.array([float(row["CenterX"]), float(row["CenterY"]), float(row["CenterTIME"])])
        half_u = new_len / 2.0
        half_v = float(row.get("PatchHeight", 10.0)) / 2.0
        vertices = [
            center - half_u * u_hat - half_v * v_hat,
            center + half_u * u_hat - half_v * v_hat,
            center + half_u * u_hat + half_v * v_hat,
            center - half_u * u_hat + half_v * v_hat,
        ]
        idx_label = df.index[target_idx]
        df.loc[idx_label, "PatchLength"] = new_len
        df.loc[idx_label, "PatchArea"] = new_len * float(row.get("PatchHeight", 10.0))
        for vi, vertex in enumerate(vertices, start=1):
            df.loc[idx_label, f"V{vi}X"] = vertex[0]
            df.loc[idx_label, f"V{vi}Y"] = vertex[1]
            df.loc[idx_label, f"V{vi}Z"] = vertex[2]

    connections.sort(key=lambda item: item["score"], reverse=True)
    n_connected = 0
    n_supplemented = 0
    connection_id_counter = int(df["ConnectionID"].max()) + 1 if (df["ConnectionID"] >= 0).any() else 0
    supplement_patches: list[dict[str, Any]] = []

    for conn in connections:
        i = conn["i"]
        j = conn["j"]
        row_i = df.iloc[i]
        row_j = df.iloc[j]
        conn_id = connection_id_counter
        connection_id_counter += 1

        len_i = float(row_i.get("PatchLength", 10.0))
        len_j = float(row_j.get("PatchLength", 10.0))
        total_half = (len_i + len_j) / 2.0
        needed = conn["along_gap"] - total_half

        idx_label_i = df.index[i]
        idx_label_j = df.index[j]
        df.loc[idx_label_i, "ConnectionType"] = "boundary_connected"
        df.loc[idx_label_j, "ConnectionType"] = "boundary_connected"
        df.loc[idx_label_i, "ConnectionID"] = conn_id
        df.loc[idx_label_j, "ConnectionID"] = conn_id

        if needed > 0:
            max_stretch_i = len_i * max(max_stretch_ratio - 1.0, 0.0)
            max_stretch_j = len_j * max(max_stretch_ratio - 1.0, 0.0)
            available = max_stretch_i + max_stretch_j

            if available >= needed and available > 0:
                frac_i = max_stretch_i / available if available > 0 else 0.5
                stretch_i = min(needed * frac_i, max_stretch_i)
                stretch_j = min(needed * (1.0 - frac_i), max_stretch_j)
                _rebuild_vertices(i, len_i + stretch_i)
                _rebuild_vertices(j, len_j + stretch_j)
            else:
                _rebuild_vertices(i, len_i + max_stretch_i)
                _rebuild_vertices(j, len_j + max_stretch_j)

                if enable_supplement:
                    remaining_gap = needed - available
                    if remaining_gap > 1.0:
                        mid_x = (float(row_i["CenterX"]) + float(row_j["CenterX"])) / 2.0
                        mid_y = (float(row_i["CenterY"]) + float(row_j["CenterY"])) / 2.0
                        mid_z = (float(row_i["CenterTIME"]) + float(row_j["CenterTIME"])) / 2.0
                        mean_az = _azimuth_mean_weighted(
                            np.array([float(work_az_arr[i]), float(work_az_arr[j])]),
                            np.array([1.0, 1.0]),
                        )
                        mean_dip = (float(work_dip_arr[i]) + float(work_dip_arr[j])) / 2.0
                        bridge_len = remaining_gap + 5.0
                        bridge_hgt = (float(row_i.get("PatchHeight", 10.0)) + float(row_j.get("PatchHeight", 10.0))) / 2.0
                        bridge_conf = (
                            min(float(row_i.get("Confidence", 0.5)), float(row_j.get("Confidence", 0.5)))
                            * supplement_confidence_decay
                        )

                        az_rad = np.deg2rad(mean_az)
                        dip_rad = np.deg2rad(mean_dip)
                        u_hat = np.array([np.cos(az_rad), np.sin(az_rad), 0.0])
                        px, py = -np.sin(az_rad), np.cos(az_rad)
                        v_hat = np.array([px * np.cos(dip_rad), py * np.cos(dip_rad), -np.sin(dip_rad)])
                        v_norm = np.linalg.norm(v_hat)
                        v_hat = v_hat / v_norm if v_norm > 1e-6 else np.array([0.0, 0.0, -1.0])
                        center = np.array([mid_x, mid_y, mid_z])
                        hu = bridge_len / 2.0
                        hv = bridge_hgt / 2.0
                        vertices = [
                            center - hu * u_hat - hv * v_hat,
                            center + hu * u_hat - hv * v_hat,
                            center + hu * u_hat + hv * v_hat,
                            center - hu * u_hat + hv * v_hat,
                        ]

                        ref_idx = i if float(row_i.get("GeophysicsScore", 0.0)) >= float(row_j.get("GeophysicsScore", 0.0)) else j
                        ref_row = df.iloc[ref_idx].to_dict()
                        new_patch: dict[str, Any] = {}
                        for col in df.columns:
                            if col != "_unit_key":
                                new_patch[col] = ref_row.get(col, 0)
                        scale_class = "macro_core" if "macro_core" in (conn["scale_i"], conn["scale_j"]) else "meso_link"
                        new_patch.update({
                            "CenterX": mid_x,
                            "CenterY": mid_y,
                            "CenterTIME": mid_z,
                            "Azimuth": mean_az,
                            "Dip": mean_dip,
                            "PatchLength": bridge_len,
                            "PatchHeight": bridge_hgt,
                            "PatchArea": bridge_len * bridge_hgt,
                            "Confidence": bridge_conf,
                            "ConnectionType": "supplemented",
                            "ConnectionID": conn_id,
                            "IsSupplemented": 1,
                            "ReliabilityLevel": "medium",
                            "ScaleClass": scale_class,
                            "AggregationMode": "boundary_bridge",
                            "ParentPatchCount": 2,
                            "WorkAzimuth": mean_az,
                            "WorkDip": mean_dip,
                            "V1X": vertices[0][0], "V1Y": vertices[0][1], "V1Z": vertices[0][2],
                            "V2X": vertices[1][0], "V2Y": vertices[1][1], "V2Z": vertices[1][2],
                            "V3X": vertices[2][0], "V3Y": vertices[2][1], "V3Z": vertices[2][2],
                            "V4X": vertices[3][0], "V4Y": vertices[3][1], "V4Z": vertices[3][2],
                        })
                        if "OrigAzimuth" in df.columns:
                            new_patch["OrigAzimuth"] = mean_az
                            new_patch["OrigDip"] = mean_dip
                        if "SmoothedAzimuth" in df.columns:
                            new_patch["SmoothedAzimuth"] = mean_az
                            new_patch["SmoothedDip"] = mean_dip
                        supplement_patches.append(new_patch)
                        n_supplemented += 1

        n_connected += 1

    if supplement_patches:
        df_sup = pd.DataFrame(supplement_patches)
        if "_unit_key" in df_sup.columns:
            df_sup.drop(columns=["_unit_key"], inplace=True)
        for col in df.columns:
            if col not in df_sup.columns and col != "_unit_key":
                df_sup[col] = 0
        keep_cols = [col for col in df.columns if col != "_unit_key"]
        df_sup = df_sup[keep_cols]
        df.drop(columns=["_unit_key"], inplace=True)
        df = pd.concat([df, df_sup], ignore_index=True)
    else:
        df.drop(columns=["_unit_key"], inplace=True)

    stats["connections_made"] = n_connected
    stats["connections_with_corridor"] = sum(1 for item in connections if item["has_corridor"])
    stats["macro_connections"] = sum(1 for item in connections if "macro_core" in (item["scale_i"], item["scale_j"]))
    stats["supplements_added"] = n_supplemented
    stats["mean_gap_m"] = float(np.mean([item["along_gap"] for item in connections])) if connections else 0.0
    stats["mean_euclidean_gap_m"] = float(np.mean([item["euclidean_gap"] for item in connections])) if connections else 0.0
    stats["mean_score"] = float(np.mean([item["score"] for item in connections])) if connections else 0.0
    stats["output_count"] = len(df)
    return df, stats


def boundary_seam_fill(
    df: pd.DataFrame,
    *,
    seam_half_width: float = 50.0,
    interior_sample_depth: float = 80.0,
    fill_fraction: float = 1.0,
    confidence_decay: float = 0.65,
    blend_half_width: float = 60.0,
    blend_strength: float = 0.5,
    min_shared_set_ratio: float = 0.15,
    high_reliability_boost: float = 1.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Boundary seam fill with scale-aware source selection."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    stats: dict[str, Any] = {"input_count": len(df)}

    has_block = "BlockX" in df.columns and "BlockY" in df.columns
    has_fset = "FractureSet" in df.columns
    if not has_block:
        stats["seam_fills"] = 0
        stats["blended_patches"] = 0
        stats["output_count"] = len(df)
        return df, stats

    bx_arr = df["BlockX"].to_numpy()
    by_arr = df["BlockY"].to_numpy()
    cx_arr = pd.to_numeric(df["CenterX"], errors="coerce").to_numpy(dtype=float)
    cy_arr = pd.to_numeric(df["CenterY"], errors="coerce").to_numpy(dtype=float)
    fs_arr = df["FractureSet"].to_numpy(dtype=int) if has_fset else np.zeros(len(df), dtype=int)
    scale_arr = (
        df["ScaleClass"].fillna("meso_link").astype(str).to_numpy()
        if "ScaleClass" in df.columns
        else np.array(["meso_link"] * len(df), dtype=object)
    )

    if "ReliabilityLevel" in df.columns:
        rl_arr = np.array(
            [2 if value in (2, "high") else (1 if value in (1, "medium") else 0) for value in df["ReliabilityLevel"].to_numpy()],
            dtype=int,
        )
    else:
        rl_arr = np.ones(len(df), dtype=int)

    units: dict[tuple[int, int], dict[str, float]] = {}
    for i in range(len(df)):
        key = (int(bx_arr[i]), int(by_arr[i]))
        if key not in units:
            units[key] = {"x_min": cx_arr[i], "x_max": cx_arr[i], "y_min": cy_arr[i], "y_max": cy_arr[i]}
        else:
            bounds = units[key]
            bounds["x_min"] = min(bounds["x_min"], cx_arr[i])
            bounds["x_max"] = max(bounds["x_max"], cx_arr[i])
            bounds["y_min"] = min(bounds["y_min"], cy_arr[i])
            bounds["y_max"] = max(bounds["y_max"], cy_arr[i])

    unit_keys = sorted(units.keys())
    adj_pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for i, u1 in enumerate(unit_keys):
        for u2 in unit_keys[i + 1:]:
            if abs(u1[0] - u2[0]) == 1 and u1[1] == u2[1]:
                adj_pairs.append((u1, u2, "x"))
            elif u1[0] == u2[0] and abs(u1[1] - u2[1]) == 1:
                adj_pairs.append((u1, u2, "y"))

    if "ConnectionID" not in df.columns:
        df["ConnectionID"] = -1
    conn_id_counter = int(df["ConnectionID"].max()) + 1 if (df["ConnectionID"] >= 0).any() else 0

    new_patches: list[dict[str, Any]] = []
    total_filled = 0
    edge_diagnostics: list[dict[str, Any]] = []

    def _edge_geometry(
        u1: tuple[int, int],
        u2: tuple[int, int],
        axis: str,
        b1: dict[str, float],
        b2: dict[str, float],
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> tuple[Any, ...] | None:
        if axis == "x":
            if b1["x_max"] >= b2["x_max"]:
                b1, b2 = b2, b1
                mask1, mask2 = mask2, mask1
            boundary_coord = (b1["x_max"] + b2["x_min"]) / 2.0
            cross_lo = max(b1["y_min"], b2["y_min"])
            cross_hi = min(b1["y_max"], b2["y_max"])
            coord_arr = cx_arr
            cross_arr = cy_arr
        else:
            if b1["y_max"] >= b2["y_max"]:
                b1, b2 = b2, b1
                mask1, mask2 = mask2, mask1
            boundary_coord = (b1["y_max"] + b2["y_min"]) / 2.0
            cross_lo = max(b1["x_min"], b2["x_min"])
            cross_hi = min(b1["x_max"], b2["x_max"])
            coord_arr = cy_arr
            cross_arr = cx_arr
        if cross_hi <= cross_lo:
            return None
        seam_min = boundary_coord - seam_half_width
        seam_max = boundary_coord + seam_half_width
        return (
            boundary_coord,
            seam_min,
            seam_max,
            seam_min - interior_sample_depth,
            seam_min,
            seam_max,
            seam_max + interior_sample_depth,
            cross_lo,
            cross_hi,
            coord_arr,
            cross_arr,
            mask1,
            mask2,
        )

    for u1, u2, axis in adj_pairs:
        mask1 = (bx_arr == u1[0]) & (by_arr == u1[1])
        mask2 = (bx_arr == u2[0]) & (by_arr == u2[1])
        geo = _edge_geometry(u1, u2, axis, units[u1], units[u2], mask1, mask2)
        if geo is None:
            continue
        (
            boundary_coord,
            seam_min,
            seam_max,
            int1_min,
            int1_max,
            int2_min,
            int2_max,
            cross_lo,
            cross_hi,
            coord_arr,
            cross_arr,
            side1_mask,
            side2_mask,
        ) = geo

        near1 = np.where(
            side1_mask
            & (coord_arr >= int1_min)
            & (coord_arr <= seam_max)
            & (cross_arr >= cross_lo)
            & (cross_arr <= cross_hi)
        )[0]
        near2 = np.where(
            side2_mask
            & (coord_arr >= seam_min)
            & (coord_arr <= int2_max)
            & (cross_arr >= cross_lo)
            & (cross_arr <= cross_hi)
        )[0]

        if len(near1) < 2 or len(near2) < 2:
            edge_diagnostics.append({
                "edge": f"{u1}-{u2}({axis})",
                "status": "skip_sparse",
                "n_side1": len(near1),
                "n_side2": len(near2),
            })
            continue

        shared_sets: list[dict[str, Any]] = []
        for fracture_set in sorted(set(np.unique(fs_arr[near1])) | set(np.unique(fs_arr[near2]))):
            ratio1 = float(np.sum(fs_arr[near1] == fracture_set)) / len(near1)
            ratio2 = float(np.sum(fs_arr[near2] == fracture_set)) / len(near2)
            if ratio1 < min_shared_set_ratio or ratio2 < min_shared_set_ratio:
                continue

            idx1 = near1[fs_arr[near1] == fracture_set]
            idx2 = near2[fs_arr[near2] == fracture_set]
            if len(idx1) == 0 or len(idx2) == 0:
                continue

            az1 = pd.to_numeric(df.iloc[idx1]["Azimuth"], errors="coerce").to_numpy(dtype=float)
            az2 = pd.to_numeric(df.iloc[idx2]["Azimuth"], errors="coerce").to_numpy(dtype=float)
            az_diff = abs(_circular_mean_deg(az1) - _circular_mean_deg(az2))
            if az_diff > 180.0:
                az_diff = 360.0 - az_diff
            shared_sets.append({
                "set": int(fracture_set),
                "ratio1": ratio1,
                "ratio2": ratio2,
                "idx1": idx1,
                "idx2": idx2,
                "high_frac": (float(np.mean(rl_arr[idx1] >= 2)) + float(np.mean(rl_arr[idx2] >= 2))) / 2.0,
                "az_diff_deg": az_diff,
            })

        if not shared_sets:
            edge_diagnostics.append({
                "edge": f"{u1}-{u2}({axis})",
                "status": "no_shared_sets",
                "n_side1": len(near1),
                "n_side2": len(near2),
            })
            continue

        edge_fills = 0
        for shared in shared_sets:
            int1_idx = shared["idx1"][(coord_arr[shared["idx1"]] >= int1_min) & (coord_arr[shared["idx1"]] <= int1_max)]
            int2_idx = shared["idx2"][(coord_arr[shared["idx2"]] >= int2_min) & (coord_arr[shared["idx2"]] <= int2_max)]
            src_idx = np.concatenate([int1_idx, int2_idx]) if len(int1_idx) or len(int2_idx) else np.array([], dtype=int)
            if len(src_idx) == 0:
                src_idx = np.concatenate([shared["idx1"], shared["idx2"]])
            if len(src_idx) == 0:
                continue
            src_idx = src_idx[scale_arr[src_idx] != "micro_bg"]
            if len(src_idx) == 0:
                continue

            reliability_mult = 1.0 + shared["high_frac"] * (high_reliability_boost - 1.0)
            az_mult = max(0.2, 1.0 - max(0.0, shared["az_diff_deg"] - 15.0) / 60.0)
            effective_fraction = fill_fraction * reliability_mult * az_mult
            n_fill = max(1, int(round(len(src_idx) * effective_fraction)))
            sample_idx = rng.choice(src_idx, size=n_fill, replace=True)

            src_az = pd.to_numeric(df.iloc[src_idx]["Azimuth"], errors="coerce").to_numpy(dtype=float)
            az_noise_std = max(8.0, float(np.std(src_az)) * 0.5)
            dip_noise_std = 3.0

            for si in sample_idx:
                ref = df.iloc[int(si)]
                if rng.random() < 0.5:
                    lo, hi = int1_min, boundary_coord
                else:
                    lo, hi = boundary_coord, int2_max
                if axis == "x":
                    nx = rng.uniform(lo, hi)
                    ny = rng.uniform(cross_lo, cross_hi)
                else:
                    nx = rng.uniform(cross_lo, cross_hi)
                    ny = rng.uniform(lo, hi)
                nz = float(ref["CenterTIME"]) + rng.normal(0.0, 5.0)

                _make_fill_patch(
                    df,
                    ref,
                    nx,
                    ny,
                    nz,
                    confidence_decay,
                    conn_id_counter,
                    new_patches,
                    az_noise=rng.normal(0.0, az_noise_std),
                    dip_noise=rng.normal(0.0, dip_noise_std),
                )
                conn_id_counter += 1
                total_filled += 1
                edge_fills += 1

        edge_diagnostics.append({
            "edge": f"{u1}-{u2}({axis})",
            "status": "filled",
            "n_side1": len(near1),
            "n_side2": len(near2),
            "shared_sets": [
                {key: value for key, value in shared.items() if key not in ("idx1", "idx2")}
                for shared in shared_sets
            ],
            "fills": edge_fills,
        })

    az_arr = pd.to_numeric(df["Azimuth"], errors="coerce").to_numpy(dtype=float).copy()
    dip_arr = pd.to_numeric(df["Dip"], errors="coerce").to_numpy(dtype=float).copy()
    total_blended = 0

    for u1, u2, axis in adj_pairs:
        mask1 = (bx_arr == u1[0]) & (by_arr == u1[1])
        mask2 = (bx_arr == u2[0]) & (by_arr == u2[1])
        b1 = units[u1]
        b2 = units[u2]
        if axis == "x":
            if b1["x_max"] >= b2["x_max"]:
                b1, b2 = b2, b1
                mask1, mask2 = mask2, mask1
            boundary_coord = (b1["x_max"] + b2["x_min"]) / 2.0
            blend_min = boundary_coord - blend_half_width
            blend_max = boundary_coord + blend_half_width
            cross_lo = max(b1["y_min"], b2["y_min"])
            cross_hi = min(b1["y_max"], b2["y_max"])
            if cross_hi <= cross_lo:
                continue
            zone1 = np.where(mask1 & (cx_arr >= blend_min) & (cx_arr <= boundary_coord) & (cy_arr >= cross_lo) & (cy_arr <= cross_hi))[0]
            zone2 = np.where(mask2 & (cx_arr > boundary_coord) & (cx_arr <= blend_max) & (cy_arr >= cross_lo) & (cy_arr <= cross_hi))[0]
            dist_fn = lambda idx: abs(cx_arr[idx] - boundary_coord)
        else:
            if b1["y_max"] >= b2["y_max"]:
                b1, b2 = b2, b1
                mask1, mask2 = mask2, mask1
            boundary_coord = (b1["y_max"] + b2["y_min"]) / 2.0
            blend_min = boundary_coord - blend_half_width
            blend_max = boundary_coord + blend_half_width
            cross_lo = max(b1["x_min"], b2["x_min"])
            cross_hi = min(b1["x_max"], b2["x_max"])
            if cross_hi <= cross_lo:
                continue
            zone1 = np.where(mask1 & (cy_arr >= blend_min) & (cy_arr <= boundary_coord) & (cx_arr >= cross_lo) & (cx_arr <= cross_hi))[0]
            zone2 = np.where(mask2 & (cy_arr > boundary_coord) & (cy_arr <= blend_max) & (cx_arr >= cross_lo) & (cx_arr <= cross_hi))[0]
            dist_fn = lambda idx: abs(cy_arr[idx] - boundary_coord)

        if len(zone1) == 0 or len(zone2) == 0:
            continue

        for fracture_set in set(np.unique(fs_arr[zone1])) & set(np.unique(fs_arr[zone2])):
            s1 = zone1[fs_arr[zone1] == fracture_set]
            s2 = zone2[fs_arr[zone2] == fracture_set]
            ratio_s1 = len(s1) / len(zone1) if len(zone1) > 0 else 0.0
            ratio_s2 = len(s2) / len(zone2) if len(zone2) > 0 else 0.0
            if ratio_s1 < min_shared_set_ratio or ratio_s2 < min_shared_set_ratio:
                continue

            blend_s1 = s1[scale_arr[s1] != "micro_bg"]
            blend_s2 = s2[scale_arr[s2] != "micro_bg"]
            if len(blend_s1) == 0 or len(blend_s2) == 0:
                continue

            mean_az_s1 = _circular_mean_deg(az_arr[blend_s1])
            mean_az_s2 = _circular_mean_deg(az_arr[blend_s2])
            mean_dip_s1 = float(np.mean(dip_arr[blend_s1]))
            mean_dip_s2 = float(np.mean(dip_arr[blend_s2]))
            az_diff = abs(mean_az_s1 - mean_az_s2)
            if az_diff > 180.0:
                az_diff = 360.0 - az_diff
            set_blend = blend_strength * max(0.2, 1.0 - max(0.0, az_diff - 15.0) / 60.0)
            blend_az_noise = max(5.0, az_diff * 0.3)

            for idx in blend_s1:
                dist = dist_fn(idx)
                t = max(0.0, 1.0 - dist / max(blend_half_width, 1e-6)) * set_blend
                if rl_arr[idx] >= 2:
                    t *= 0.5
                if t <= 0:
                    continue
                az_arr[idx] = (_blend_azimuth(az_arr[idx], mean_az_s2, t) + rng.normal(0.0, blend_az_noise * t)) % 360.0
                dip_arr[idx] = float(np.clip(dip_arr[idx] * (1.0 - t) + mean_dip_s2 * t + rng.normal(0.0, 2.0 * t), 0.0, 90.0))
                total_blended += 1

            for idx in blend_s2:
                dist = dist_fn(idx)
                t = max(0.0, 1.0 - dist / max(blend_half_width, 1e-6)) * set_blend
                if rl_arr[idx] >= 2:
                    t *= 0.5
                if t <= 0:
                    continue
                az_arr[idx] = (_blend_azimuth(az_arr[idx], mean_az_s1, t) + rng.normal(0.0, blend_az_noise * t)) % 360.0
                dip_arr[idx] = float(np.clip(dip_arr[idx] * (1.0 - t) + mean_dip_s1 * t + rng.normal(0.0, 2.0 * t), 0.0, 90.0))
                total_blended += 1

    df["Azimuth"] = az_arr
    df["Dip"] = dip_arr
    if "WorkAzimuth" in df.columns:
        df["WorkAzimuth"] = np.mod(az_arr, 180.0)
    if "WorkDip" in df.columns:
        df["WorkDip"] = dip_arr

    if new_patches:
        df_fill = pd.DataFrame(new_patches)
        for col in df.columns:
            if col not in df_fill.columns:
                df_fill[col] = 0
        df_fill = df_fill[df.columns]
        df = pd.concat([df, df_fill], ignore_index=True)

    stats["seam_fills"] = total_filled
    stats["blended_patches"] = total_blended
    stats["adjacent_edges"] = len(adj_pairs)
    stats["edge_diagnostics"] = edge_diagnostics
    stats["output_count"] = len(df)
    return df, stats


def _make_fill_patch(
    df: pd.DataFrame,
    ref: pd.Series,
    new_x: float,
    new_y: float,
    new_z: float,
    conf_decay: float,
    conn_id: int,
    out_list: list[dict[str, Any]],
    az_noise: float = 0.0,
    dip_noise: float = 0.0,
) -> None:
    """Build a seam-fill patch from a reference patch."""
    az = (float(ref["Azimuth"]) + az_noise) % 360.0
    dip = float(np.clip(float(ref["Dip"]) + dip_noise, 0.0, 90.0))
    patch_length = float(ref.get("PatchLength", 10.0))
    patch_height = float(ref.get("PatchHeight", 10.0))
    confidence = float(ref.get("Confidence", 0.5)) * conf_decay
    scale_class = str(ref.get("ScaleClass", "meso_link"))
    if scale_class == "micro_bg":
        scale_class = "meso_link"

    az_rad = np.deg2rad(az)
    dip_rad = np.deg2rad(dip)
    u_hat = np.array([np.cos(az_rad), np.sin(az_rad), 0.0])
    px, py = -np.sin(az_rad), np.cos(az_rad)
    v_hat = np.array([px * np.cos(dip_rad), py * np.cos(dip_rad), -np.sin(dip_rad)])
    v_norm = np.linalg.norm(v_hat)
    v_hat = v_hat / v_norm if v_norm > 1e-6 else np.array([0.0, 0.0, -1.0])
    center = np.array([new_x, new_y, new_z])
    hu = patch_length / 2.0
    hv = patch_height / 2.0
    vertices = [
        center - hu * u_hat - hv * v_hat,
        center + hu * u_hat - hv * v_hat,
        center + hu * u_hat + hv * v_hat,
        center - hu * u_hat + hv * v_hat,
    ]

    new_patch: dict[str, Any] = {}
    for col in df.columns:
        new_patch[col] = ref.get(col, 0)
    new_patch.update({
        "CenterX": new_x,
        "CenterY": new_y,
        "CenterTIME": new_z,
        "Azimuth": az,
        "Dip": dip,
        "PatchLength": patch_length,
        "PatchHeight": patch_height,
        "PatchArea": patch_length * patch_height,
        "Confidence": confidence,
        "ConnectionType": "supplemented",
        "ConnectionID": conn_id,
        "IsSupplemented": 1,
        "ReliabilityLevel": "medium",
        "ScaleClass": scale_class,
        "AggregationMode": "seam_fill",
        "ParentPatchCount": 1,
        "WorkAzimuth": az % 180.0,
        "WorkDip": dip,
        "V1X": vertices[0][0], "V1Y": vertices[0][1], "V1Z": vertices[0][2],
        "V2X": vertices[1][0], "V2Y": vertices[1][1], "V2Z": vertices[1][2],
        "V3X": vertices[2][0], "V3Y": vertices[2][1], "V3Z": vertices[2][2],
        "V4X": vertices[3][0], "V4Y": vertices[3][1], "V4Z": vertices[3][2],
    })
    if "OrigAzimuth" in df.columns:
        new_patch["OrigAzimuth"] = az
        new_patch["OrigDip"] = dip
    if "SmoothedAzimuth" in df.columns:
        new_patch["SmoothedAzimuth"] = az % 180.0
        new_patch["SmoothedDip"] = dip
    out_list.append(new_patch)


def run_phase2(
    df: pd.DataFrame,
    *,
    smooth_bandwidth_xy: float = 150.0,
    smooth_bandwidth_z: float = 20.0,
    smooth_blend_alpha: float = 0.4,
    corridor_search_radius: float = 200.0,
    corridor_min_patches: int = 5,
    enable_supplement: bool = True,
    min_pair_confidence: float = 0.5,
    max_supplement_length: float = 80.0,
    boundary_tol_xy: float = 25.0,
    enable_aggregation: bool = True,
    agg_cluster_radius: float = 20.0,
    agg_min_patches: int = 3,
    agg_azimuth_tol: float = 25.0,
    agg_dip_tol: float = 15.0,
    agg_max_length: float = 80.0,
    agg_max_height: float = 40.0,
    scale_major_radius: float = 160.0,
    scale_minor_radius: float = 25.0,
    scale_z_radius: float = 20.0,
    macro_score_quantile: float = 0.85,
    macro_max_fraction: float = 0.10,
    meso_score_quantile: float = 0.45,
    macro_min_span: float = 80.0,
    meso_min_span: float = 30.0,
    macro_min_neighbors: int = 6,
    meso_min_neighbors: int = 3,
    macro_agg_major: float = 140.0,
    macro_agg_minor: float = 24.0,
    macro_agg_max_length: float = 320.0,
    macro_agg_max_height: float = 90.0,
    elongation_max_stretch: float = 2.5,
    elongation_gap_fill: float = 0.7,
    elongation_max_neighbor_dist: float = 120.0,
    macro_elongation_stretch: float = 4.0,
    macro_elongation_range: float = 220.0,
    enable_elongation: bool = True,
    jitter_sigma_xy: float = 8.0,
    jitter_along_strike_factor: float = 1.5,
    jitter_seed: int = 42,
    compute_backend: str = DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    gpu_tile_points: int = DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
    progress_hook: Callable[[str, str | None], None] | None = None,
    heartbeat_hook: Callable[[str, str | None], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Phase 2 override with multiscale expert postprocess."""
    stats: dict[str, Any] = {"phase": 2, "input_count": len(df)}
    stats["step_seconds"] = {}

    step_started = perf_counter()
    df = regional_orientation_smoothing(
        df,
        smooth_bandwidth_xy,
        smooth_bandwidth_z,
        smooth_blend_alpha,
        progress_hook=heartbeat_hook,
    )
    stats["step_seconds"]["orientation_smoothing"] = float(perf_counter() - step_started)
    az_shift = _azimuth_diff(df["OrigAzimuth"].values, df["SmoothedAzimuth"].values)
    dip_shift = np.abs(df["OrigDip"].values - df["SmoothedDip"].values)
    stats["orientation_smoothing"] = {
        "mean_azimuth_shift_deg": float(np.nanmean(az_shift)),
        "mean_dip_shift_deg": float(np.nanmean(dip_shift)),
        "max_azimuth_shift_deg": float(np.nanmax(az_shift)),
    }
    if progress_hook is not None:
        progress_hook(
            "Phase2 产状平滑完成",
            (
                f"mean_az_shift={stats['orientation_smoothing']['mean_azimuth_shift_deg']:.2f}deg, "
                f"mean_dip_shift={stats['orientation_smoothing']['mean_dip_shift_deg']:.2f}deg, "
                f"elapsed={_format_seconds(stats['step_seconds']['orientation_smoothing'])}"
            ),
        )

    step_started = perf_counter()
    df = fracture_corridor_detection(
        df,
        corridor_search_radius,
        corridor_min_patches,
        progress_hook=heartbeat_hook,
    )
    stats["step_seconds"]["corridor_detection"] = float(perf_counter() - step_started)
    n_in_corridors = int(df["CorridorSupport"].sum())
    stats["corridors_detected"] = int(df["CorridorID"].max() + 1) if n_in_corridors > 0 else 0
    stats["patches_in_corridors"] = n_in_corridors
    if progress_hook is not None:
        progress_hook(
            "Phase2 走廊识别完成",
            (
                f"corridors={int(stats['corridors_detected'])}, "
                f"patches_in_corridor={n_in_corridors}, "
                f"elapsed={_format_seconds(stats['step_seconds']['corridor_detection'])}"
            ),
        )

    step_started = perf_counter()
    df = assign_scale_classes(
        df,
        scale_major_radius=scale_major_radius,
        scale_minor_radius=scale_minor_radius,
        scale_z_radius=scale_z_radius,
        macro_score_quantile=macro_score_quantile,
        macro_max_fraction=macro_max_fraction,
        meso_score_quantile=meso_score_quantile,
        macro_min_span=macro_min_span,
        meso_min_span=meso_min_span,
        macro_min_neighbors=macro_min_neighbors,
        meso_min_neighbors=meso_min_neighbors,
        progress_hook=heartbeat_hook,
    )
    stats["step_seconds"]["scale_classification"] = float(perf_counter() - step_started)
    scale_counts = df["ScaleClass"].value_counts().to_dict()
    total_scale = max(len(df), 1)
    stats["scale_class_counts"] = {str(key): int(value) for key, value in scale_counts.items()}
    stats["macro_fraction"] = float(scale_counts.get("macro_core", 0) / total_scale)
    stats["meso_fraction"] = float(scale_counts.get("meso_link", 0) / total_scale)
    stats["micro_fraction"] = float(scale_counts.get("micro_bg", 0) / total_scale)
    if progress_hook is not None:
        progress_hook(
            "Phase2 尺度分类完成",
            (
                f"macro={int(scale_counts.get('macro_core', 0))}, "
                f"meso={int(scale_counts.get('meso_link', 0))}, "
                f"micro={int(scale_counts.get('micro_bg', 0))}, "
                f"elapsed={_format_seconds(stats['step_seconds']['scale_classification'])}"
            ),
        )

    if enable_supplement:
        step_started = perf_counter()
        _, matched_pairs = boundary_match_only(
            df,
            boundary_tol_xy=boundary_tol_xy,
            compute_backend=compute_backend,
            gpu_tile_points=gpu_tile_points,
        )
        count_before = len(df)
        df = conservative_boundary_supplement(
            df,
            matched_pairs,
            min_pair_confidence=min_pair_confidence,
            max_supplement_length=max_supplement_length,
        )
        stats["step_seconds"]["conservative_supplement"] = float(perf_counter() - step_started)
        stats["supplemented_patches"] = len(df) - count_before
        if progress_hook is not None:
            progress_hook(
                "Phase2 保守补接完成",
                (
                    f"matched_pairs={len(matched_pairs)}, supplemented={int(stats['supplemented_patches'])}, "
                    f"elapsed={_format_seconds(stats['step_seconds']['conservative_supplement'])}"
                ),
            )
    else:
        stats["supplemented_patches"] = 0
        if progress_hook is not None:
            progress_hook("Phase2 保守补接跳过", "enable_supplement=False")

    if enable_aggregation:
        step_started = perf_counter()
        count_before_agg = len(df)
        df = aggregate_patches(
            df,
            cluster_radius=agg_cluster_radius,
            cluster_min_patches=agg_min_patches,
            azimuth_tol_deg=agg_azimuth_tol,
            dip_tol_deg=agg_dip_tol,
            max_merged_length=agg_max_length,
            max_merged_height=agg_max_height,
            macro_major_radius=macro_agg_major,
            macro_minor_radius=macro_agg_minor,
            macro_max_merged_length=macro_agg_max_length,
            macro_max_merged_height=macro_agg_max_height,
            scale_major_radius=scale_major_radius,
            scale_minor_radius=scale_minor_radius,
            scale_z_radius=scale_z_radius,
            macro_score_quantile=macro_score_quantile,
            macro_max_fraction=macro_max_fraction,
            meso_score_quantile=meso_score_quantile,
            macro_min_span=macro_min_span,
            meso_min_span=meso_min_span,
            macro_min_neighbors=macro_min_neighbors,
            meso_min_neighbors=meso_min_neighbors,
            progress_hook=heartbeat_hook,
        )
        stats["step_seconds"]["aggregate_patches"] = float(perf_counter() - step_started)
        stats["aggregation"] = {
            "input_patches": count_before_agg,
            "output_patches": len(df),
            "merged_away": count_before_agg - len(df),
            "post_scale_class_counts": {str(k): int(v) for k, v in df["ScaleClass"].value_counts().to_dict().items()} if "ScaleClass" in df.columns else {},
        }
        if progress_hook is not None:
            progress_hook(
                "Phase2 裂缝片聚合完成",
                (
                    f"input={count_before_agg}, output={len(df)}, "
                    f"merged_away={count_before_agg - len(df)}, "
                    f"elapsed={_format_seconds(stats['step_seconds']['aggregate_patches'])}"
                ),
            )
    else:
        stats["aggregation"] = None
        if progress_hook is not None:
            progress_hook("Phase2 裂缝片聚合跳过", "enable_aggregation=False")

    if enable_elongation:
        step_started = perf_counter()
        orig_lengths = df["PatchLength"].to_numpy(dtype=float).copy()
        df = corridor_elongation(
            df,
            max_stretch_factor=elongation_max_stretch,
            gap_fill_fraction=elongation_gap_fill,
            max_neighbor_dist=elongation_max_neighbor_dist,
            macro_stretch_factor=macro_elongation_stretch,
            macro_neighbor_dist=macro_elongation_range,
            progress_hook=heartbeat_hook,
        )
        stats["step_seconds"]["corridor_elongation"] = float(perf_counter() - step_started)
        if "PatchArea" in df.columns:
            df["PatchArea"] = (
                pd.to_numeric(df["PatchLength"], errors="coerce").fillna(0.0)
                * pd.to_numeric(df["PatchHeight"], errors="coerce").fillna(0.0)
            )
        new_lengths = df["PatchLength"].to_numpy(dtype=float)
        stretched_mask = new_lengths > orig_lengths[:len(new_lengths)] * 1.05
        stats["elongation"] = {
            "patches_stretched": int(stretched_mask.sum()),
            "mean_stretch_ratio": float(np.mean(new_lengths[stretched_mask] / orig_lengths[:len(new_lengths)][stretched_mask])) if stretched_mask.any() else 1.0,
            "max_stretch_factor": elongation_max_stretch,
            "macro_stretch_factor": macro_elongation_stretch,
        }
        if progress_hook is not None:
            progress_hook(
                "Phase2 沿走向拉伸完成",
                (
                    f"stretched={int(stretched_mask.sum())}, "
                    f"mean_ratio={stats['elongation']['mean_stretch_ratio']:.3f}, "
                    f"elapsed={_format_seconds(stats['step_seconds']['corridor_elongation'])}"
                ),
            )
    elif progress_hook is not None:
        progress_hook("Phase2 沿走向拉伸跳过", "enable_elongation=False")

    if jitter_sigma_xy > 0:
        step_started = perf_counter()
        df = spatial_perturbation(df, jitter_sigma_xy, jitter_along_strike_factor, jitter_seed)
        stats["step_seconds"]["spatial_perturbation"] = float(perf_counter() - step_started)
        stats["spatial_perturbation"] = {
            "jitter_sigma_xy": jitter_sigma_xy,
            "along_strike_factor": jitter_along_strike_factor,
        }
        if progress_hook is not None:
            progress_hook(
                "Phase2 空间扰动完成",
                (
                    f"jitter_sigma_xy={jitter_sigma_xy}, along_strike_factor={jitter_along_strike_factor}, "
                    f"elapsed={_format_seconds(stats['step_seconds']['spatial_perturbation'])}"
                ),
            )
    elif progress_hook is not None:
        progress_hook("Phase2 空间扰动跳过", "jitter_sigma_xy<=0")

    stats["compute_backend"] = _resolve_compute_backend(compute_backend)
    stats["output_count"] = len(df)
    return df, stats


def _df_from_vtk(payload: dict[str, Any]) -> pd.DataFrame:
    """Build DataFrame from legacy VTK payload."""
    points = np.asarray(payload["points"], dtype=float)
    polygons = payload["polygons"]
    cell_data = payload["cell_data"]
    n_cells = len(polygons)

    df = pd.DataFrame()
    for name, values in cell_data.items():
        df[name] = values

    decode_maps: dict[str, dict[int, str]] = {
        "ReliabilityLevel": {2: "high", 1: "medium", 0: "low"},
        "ConnectionType": {0: "original", 1: "boundary_matched", 2: "supplemented", 3: "boundary_connected"},
        "ScaleClass": {2: "macro_core", 1: "meso_link", 0: "micro_bg"},
        "AggregationMode": {0: "original", 1: "macro_agg", 2: "meso_agg", 3: "boundary_bridge", 4: "seam_fill"},
    }
    for col, mapping in decode_maps.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            df[col] = numeric.astype(int).map(mapping).fillna(df[col])

    if n_cells > 0 and all(len(poly) == 4 for poly in polygons):
        polygon_idx = np.asarray(polygons, dtype=np.int64)
        polygon_points = points[polygon_idx]
        centers = polygon_points.mean(axis=1)
        df["CenterX"] = centers[:, 0]
        df["CenterY"] = centers[:, 1]
        df["CenterTIME"] = centers[:, 2]
        for vi in range(4):
            df[f"V{vi + 1}X"] = polygon_points[:, vi, 0]
            df[f"V{vi + 1}Y"] = polygon_points[:, vi, 1]
            df[f"V{vi + 1}Z"] = polygon_points[:, vi, 2]
    else:
        centers = np.zeros((n_cells, 3))
        for i, poly in enumerate(polygons):
            centers[i] = points[poly].mean(axis=0)
        df["CenterX"] = centers[:, 0]
        df["CenterY"] = centers[:, 1]
        df["CenterTIME"] = centers[:, 2]

        for vi in range(1, 5):
            vx, vy, vz = [], [], []
            for i, poly in enumerate(polygons):
                if vi - 1 < len(poly):
                    point = points[poly[vi - 1]]
                    vx.append(point[0])
                    vy.append(point[1])
                    vz.append(point[2])
                else:
                    vx.append(centers[i, 0])
                    vy.append(centers[i, 1])
                    vz.append(centers[i, 2])
            df[f"V{vi}X"] = vx
            df[f"V{vi}Y"] = vy
            df[f"V{vi}Z"] = vz

    for col, default in [("Azimuth", 0.0), ("Dip", 45.0), ("Confidence", 0.5), ("PatchLength", 10.0), ("PatchHeight", 10.0)]:
        if col not in df.columns:
            df[col] = default
    if "PatchArea" not in df.columns:
        df["PatchArea"] = pd.to_numeric(df["PatchLength"], errors="coerce").fillna(0.0) * pd.to_numeric(df["PatchHeight"], errors="coerce").fillna(0.0)
    if "AggregationMode" not in df.columns:
        df["AggregationMode"] = "original"
    if "ParentPatchCount" not in df.columns:
        df["ParentPatchCount"] = 1
    return df


def _df_to_vtk(df: pd.DataFrame, title: str, output_vtk: Path, scalar_types: dict[str, str] | None = None) -> None:
    """Write DataFrame back to legacy VTK."""
    row_count = int(len(df))
    skip_cols = {"CenterX", "CenterY", "CenterTIME"}
    skip_cols |= {f"V{vi}{c}" for vi in range(1, 5) for c in ("X", "Y", "Z")}
    data_cols = [col for col in df.columns if col not in skip_cols]
    vertex_blocks = [
        np.column_stack(
            [
                pd.to_numeric(df[f"V{vi}X"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                pd.to_numeric(df[f"V{vi}Y"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                pd.to_numeric(df[f"V{vi}Z"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            ]
        )
        for vi in range(1, 5)
    ]
    out_points = np.stack(vertex_blocks, axis=1).reshape(row_count * 4, 3) if row_count > 0 else np.zeros((0, 3), dtype=float)
    new_polygons = np.arange(row_count * 4, dtype=int).reshape(row_count, 4).tolist() if row_count > 0 else []
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
    }
    str_maps: dict[str, dict[str, int]] = {
        "ReliabilityLevel": {"high": 2, "medium": 1, "low": 0},
        "ConnectionType": {"original": 0, "boundary_matched": 1, "supplemented": 2, "boundary_connected": 3},
        "ScaleClass": {"micro_bg": 0, "meso_link": 1, "macro_core": 2},
        "AggregationMode": {"original": 0, "macro_agg": 1, "meso_agg": 2, "boundary_bridge": 3, "seam_fill": 4},
    }

    for col in data_cols:
        vals_series = df[col]
        vals_list = vals_series.to_numpy()
        arr = np.asarray(vals_list)
        if col in str_maps:
            mapping = str_maps[col]
            out_cell_data[col] = np.array([mapping.get(str(value), -1) for value in vals_list], dtype=int)
            out_scalar_types[col] = "int"
        elif col in int_cols:
            out_cell_data[col] = arr.astype(int)
            out_scalar_types[col] = "int"
        else:
            try:
                out_cell_data[col] = arr.astype(float)
                out_scalar_types[col] = "float"
            except (TypeError, ValueError):
                continue

    write_legacy_vtk_polygons(
        path=output_vtk,
        title=title,
        points=out_points,
        polygons=new_polygons,
        cell_data=out_cell_data,
        scalar_types=out_scalar_types,
    )


def _run_pipeline_on_dataframe(
    df: pd.DataFrame,
    phase: int,
    progress_hook: Callable[[str, str | None], None] | None = None,
    heartbeat_hook: Callable[[str, str | None], None] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run pipeline on a dataframe so VTK and CSV share the same logic."""
    runtime_keys = {
        "compute_backend",
        "max_cpu_threads",
        "gpu_tile_points",
    }
    boundary_keys = {
        "enable_boundary_connect",
        "boundary_strip_width",
        "connect_max_gap",
        "bc_macro_gap",
        "bc_minor_limit",
        "bc_z_gap",
        "bc_azimuth_tol_deg",
        "bc_dip_tol_deg",
        "bc_min_score_threshold",
        "bc_corridor_bonus",
        "bc_require_corridor",
        "bc_perp_ratio",
        "bc_enable_supplement",
        "bc_max_stretch_ratio",
        "enable_seam_fill",
        "seam_half_width",
        "seam_interior_depth",
        "seam_fill_fraction",
        "seam_confidence_decay",
        "seam_seed",
        "seam_blend_half_width",
        "seam_blend_strength",
        "seam_min_shared_ratio",
        "seam_high_rel_boost",
    }
    runtime_kwargs = {k: v for k, v in kwargs.items() if k in runtime_keys}
    boundary_kwargs = {k: v for k, v in kwargs.items() if k in boundary_keys}
    enable_boundary_connect = bool(boundary_kwargs.pop("enable_boundary_connect", False))
    enable_seam_fill = bool(boundary_kwargs.pop("enable_seam_fill", False))
    core_kwargs = {k: v for k, v in kwargs.items() if k not in boundary_keys and k not in runtime_keys}
    phase_runtime_kwargs = {
        key: value
        for key, value in runtime_kwargs.items()
        if key in {"compute_backend", "gpu_tile_points"}
    }

    max_cpu_threads = int(runtime_kwargs.get("max_cpu_threads", DEFAULT_POSTPROCESS_MAX_CPU_THREADS) or 0)
    with _thread_limit_context(max_cpu_threads):
        if phase == 1:
            p1_kwargs = dict(core_kwargs)
            p1_kwargs.update(phase_runtime_kwargs)
            df, stats = run_phase1(df, progress_hook=progress_hook, heartbeat_hook=heartbeat_hook, **p1_kwargs)
        elif phase == 2:
            p1_keys = {
                "max_fracture_sets",
                "n_fracture_sets",
                "neighbor_radius_xy",
                "neighbor_radius_z",
                "isolation_min_neighbors",
                "confidence_floor",
                "boundary_tol_xy",
                "length_range",
                "height_range",
                "compute_backend",
                "gpu_tile_points",
            }
            p1_kwargs = {k: v for k, v in core_kwargs.items() if k in p1_keys}
            p2_kwargs = {k: v for k, v in core_kwargs.items() if k not in p1_keys}
            p1_kwargs.update(phase_runtime_kwargs)
            p2_kwargs.update(phase_runtime_kwargs)
            df, stats1 = run_phase1(df, progress_hook=progress_hook, heartbeat_hook=heartbeat_hook, **p1_kwargs)
            df, stats2 = run_phase2(df, progress_hook=progress_hook, heartbeat_hook=heartbeat_hook, **p2_kwargs)
            stats = {
                **stats1,
                "phase1_output_count": stats1.get("output_count"),
                "phase2": stats2,
                "output_count": stats2.get("output_count", stats1.get("output_count")),
            }
        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

        if enable_boundary_connect:
            bc_map = {
                "boundary_strip_width": "boundary_strip_width",
                "connect_max_gap": "connect_max_gap",
                "bc_macro_gap": "macro_connect_gap",
                "bc_minor_limit": "connect_minor_limit",
                "bc_z_gap": "connect_z_gap",
                "bc_azimuth_tol_deg": "azimuth_tol_deg",
                "bc_dip_tol_deg": "dip_tol_deg",
                "bc_min_score_threshold": "min_score_threshold",
                "bc_corridor_bonus": "corridor_bonus",
                "bc_require_corridor": "require_corridor",
                "bc_perp_ratio": "perp_ratio",
                "bc_enable_supplement": "enable_supplement",
                "bc_max_stretch_ratio": "max_stretch_ratio",
            }
            bc_args = {bc_map[k]: v for k, v in boundary_kwargs.items() if k in bc_map}
            bc_args.update({k: v for k, v in runtime_kwargs.items() if k in {"compute_backend", "gpu_tile_points"}})
            df, bc_stats = boundary_connect_postprocess(df, **bc_args)
            stats["boundary_connect"] = bc_stats
            if progress_hook is not None:
                progress_hook(
                    "边界跨单元补接完成",
                    (
                        f"connections={int(bc_stats.get('connections_made', 0))}, "
                        f"supplements={int(bc_stats.get('supplements_added', 0))}, "
                        f"backend={bc_stats.get('compute_backend', 'unknown')}"
                    ),
                )

        if enable_seam_fill:
            sf_map = {
                "seam_half_width": "seam_half_width",
                "seam_interior_depth": "interior_sample_depth",
                "seam_fill_fraction": "fill_fraction",
                "seam_confidence_decay": "confidence_decay",
                "seam_seed": "seed",
                "seam_blend_half_width": "blend_half_width",
                "seam_blend_strength": "blend_strength",
                "seam_min_shared_ratio": "min_shared_set_ratio",
                "seam_high_rel_boost": "high_reliability_boost",
            }
            sf_args = {sf_map[k]: v for k, v in boundary_kwargs.items() if k in sf_map}
            df, sf_stats = boundary_seam_fill(df, **sf_args)
            stats["seam_fill"] = sf_stats
            if progress_hook is not None:
                progress_hook(
                    "边界缝带填充完成",
                    (
                        f"seam_fills={int(sf_stats.get('seam_fills', 0))}, "
                        f"blended={int(sf_stats.get('blended_patches', 0))}"
                    ),
                )

    stats["max_cpu_threads"] = int(max_cpu_threads)
    stats["compute_backend"] = str(stats.get("compute_backend", runtime_kwargs.get("compute_backend", DEFAULT_POSTPROCESS_COMPUTE_BACKEND)))
    return df, stats


def postprocess_vtk(input_vtk: Path, output_vtk: Path, phase: int = 1, **kwargs: Any) -> dict[str, Any]:
    """Read VTK, postprocess, write VTK."""
    total_started = perf_counter()
    enable_boundary_connect = bool(kwargs.get("enable_boundary_connect", False))
    enable_seam_fill = bool(kwargs.get("enable_seam_fill", False))
    pipeline_stage_count = 3 + (7 if int(phase) == 2 else 0) + (1 if enable_boundary_connect else 0) + (1 if enable_seam_fill else 0)
    progress = _StageProgressPrinter(
        total_steps=2 + pipeline_stage_count + 1,
        prefix=f"postprocess phase{int(phase)}",
    )
    payload = read_legacy_vtk_polygons(input_vtk)
    progress.advance("读取VTK完成", f"polygons={len(payload.get('polygons', []))}")
    df = _df_from_vtk(payload)
    progress.advance("VTK转DataFrame完成", f"rows={len(df)}")
    df, stats = _run_pipeline_on_dataframe(
        df,
        phase,
        progress_hook=progress.advance,
        heartbeat_hook=progress.log,
        **kwargs,
    )
    title = f"{payload.get('title', 'DFN')}_postprocessed_phase{phase}"
    _df_to_vtk(df, title, output_vtk, payload.get("scalar_types"))
    stats["total_seconds"] = float(perf_counter() - total_started)
    progress.advance(
        "写出VTK完成",
        f"rows={len(df)}, output={output_vtk}, total_elapsed={_format_seconds(stats['total_seconds'])}",
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the multiscale postprocess override."""
    parser = argparse.ArgumentParser(
        description="DFN postprocess with multiscale aggregation and anisotropic boundary connection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-vtk", type=Path, help="Merged VTK input.")
    input_group.add_argument("--input-csv", type=Path, help="Merged CSV input.")
    parser.add_argument("--output-vtk", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--boundary-connect", action="store_true", help="Enable boundary connection.")
    parser.add_argument("--bc-seam-fill", action="store_true", help="Enable seam fill after boundary connection.")

    g1 = parser.add_argument_group("Step 1")
    g1.add_argument("--max-sets", type=int, default=6)
    g1.add_argument("--n-sets", type=int, default=None)

    g2 = parser.add_argument_group("Step 2")
    g2.add_argument("--smooth-bw-xy", type=float, default=150.0)
    g2.add_argument("--smooth-bw-z", type=float, default=20.0)
    g2.add_argument("--smooth-alpha", type=float, default=0.4)

    g3 = parser.add_argument_group("Step 3")
    g3.add_argument("--corridor-radius", type=float, default=200.0)
    g3.add_argument("--corridor-min", type=int, default=5)

    g4 = parser.add_argument_group("Step 4")
    g4.add_argument("--neighbor-radius-xy", type=float, default=100.0)
    g4.add_argument("--neighbor-radius-z", type=float, default=15.0)
    g4.add_argument("--min-neighbors", type=int, default=2)
    g4.add_argument("--conf-floor", type=float, default=0.3)

    g5 = parser.add_argument_group("Step 5")
    g5.add_argument("--boundary-tol", type=float, default=25.0)
    g5.add_argument("--no-supplement", action="store_true")
    g5.add_argument("--min-pair-conf", type=float, default=0.5)
    g5.add_argument("--max-supplement-len", type=float, default=80.0)

    g6 = parser.add_argument_group("Step 6")
    g6.add_argument("--agg-radius", type=float, default=20.0)
    g6.add_argument("--agg-min", type=int, default=3)
    g6.add_argument("--agg-azimuth-tol", type=float, default=25.0)
    g6.add_argument("--agg-dip-tol", type=float, default=15.0)
    g6.add_argument("--agg-max-length", type=float, default=80.0)
    g6.add_argument("--agg-max-height", type=float, default=40.0)
    g6.add_argument("--scale-major-radius", type=float, default=160.0)
    g6.add_argument("--scale-minor-radius", type=float, default=25.0)
    g6.add_argument("--scale-z-radius", type=float, default=20.0)
    g6.add_argument("--macro-score-quantile", type=float, default=0.85)
    g6.add_argument("--macro-max-fraction", type=float, default=0.10)
    g6.add_argument("--meso-score-quantile", type=float, default=0.45)
    g6.add_argument("--macro-min-span", type=float, default=80.0)
    g6.add_argument("--meso-min-span", type=float, default=30.0)
    g6.add_argument("--macro-min-neighbors", type=int, default=6)
    g6.add_argument("--meso-min-neighbors", type=int, default=3)
    g6.add_argument("--macro-agg-major", type=float, default=140.0)
    g6.add_argument("--macro-agg-minor", type=float, default=24.0)
    g6.add_argument("--macro-agg-max-length", type=float, default=320.0)
    g6.add_argument("--macro-agg-max-height", type=float, default=90.0)
    g6.add_argument("--no-aggregation", action="store_true")

    g7 = parser.add_argument_group("Step 7")
    g7.add_argument("--elongation-stretch", type=float, default=2.5)
    g7.add_argument("--elongation-fill", type=float, default=0.7)
    g7.add_argument("--elongation-range", type=float, default=120.0)
    g7.add_argument("--macro-elongation-stretch", type=float, default=4.0)
    g7.add_argument("--macro-elongation-range", type=float, default=220.0)
    g7.add_argument("--no-elongation", action="store_true")

    g8 = parser.add_argument_group("Step 8")
    g8.add_argument("--jitter-xy", type=float, default=8.0)
    g8.add_argument("--jitter-strike-factor", type=float, default=1.5)
    g8.add_argument("--jitter-seed", type=int, default=42)
    g8.add_argument("--no-jitter", action="store_true")

    g9 = parser.add_argument_group("Step 9")
    g9.add_argument("--bc-strip-width", type=float, default=40.0)
    g9.add_argument("--bc-max-gap", type=float, default=60.0)
    g9.add_argument("--bc-macro-gap", type=float, default=180.0)
    g9.add_argument("--bc-minor-limit", type=float, default=25.0)
    g9.add_argument("--bc-z-gap", type=float, default=20.0)
    g9.add_argument("--bc-azimuth-tol", type=float, default=30.0)
    g9.add_argument("--bc-dip-tol", type=float, default=20.0)
    g9.add_argument("--bc-min-score", type=float, default=0.35)
    g9.add_argument("--bc-corridor-bonus", type=float, default=0.15)
    g9.add_argument("--bc-require-corridor", action="store_true")
    g9.add_argument("--bc-perp-ratio", type=float, default=0.8)
    g9.add_argument("--bc-no-supplement", action="store_true")
    g9.add_argument("--bc-max-stretch", type=float, default=2.5)

    g10 = parser.add_argument_group("Step 9b")
    g10.add_argument("--bc-seam-width", type=float, default=50.0)
    g10.add_argument("--bc-seam-interior", type=float, default=80.0)
    g10.add_argument("--bc-seam-fraction", type=float, default=1.0)
    g10.add_argument("--bc-seam-decay", type=float, default=0.65)
    g10.add_argument("--bc-seam-seed", type=int, default=42)
    g10.add_argument("--bc-blend-width", type=float, default=60.0)
    g10.add_argument("--bc-blend-strength", type=float, default=0.5)
    g10.add_argument("--bc-min-shared-ratio", type=float, default=0.15)
    g10.add_argument("--bc-high-rel-boost", type=float, default=1.5)

    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def _collect_pipeline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect runtime kwargs from parsed CLI arguments."""
    kwargs: dict[str, Any] = {
        "max_fracture_sets": args.max_sets,
        "n_fracture_sets": args.n_sets,
        "neighbor_radius_xy": args.neighbor_radius_xy,
        "neighbor_radius_z": args.neighbor_radius_z,
        "isolation_min_neighbors": args.min_neighbors,
        "confidence_floor": args.conf_floor,
        "boundary_tol_xy": args.boundary_tol,
    }
    if args.phase == 2:
        kwargs.update({
            "smooth_bandwidth_xy": args.smooth_bw_xy,
            "smooth_bandwidth_z": args.smooth_bw_z,
            "smooth_blend_alpha": args.smooth_alpha,
            "corridor_search_radius": args.corridor_radius,
            "corridor_min_patches": args.corridor_min,
            "enable_supplement": not args.no_supplement,
            "min_pair_confidence": args.min_pair_conf,
            "max_supplement_length": args.max_supplement_len,
            "enable_aggregation": not args.no_aggregation,
            "agg_cluster_radius": args.agg_radius,
            "agg_min_patches": args.agg_min,
            "agg_azimuth_tol": args.agg_azimuth_tol,
            "agg_dip_tol": args.agg_dip_tol,
            "agg_max_length": args.agg_max_length,
            "agg_max_height": args.agg_max_height,
            "scale_major_radius": args.scale_major_radius,
            "scale_minor_radius": args.scale_minor_radius,
            "scale_z_radius": args.scale_z_radius,
            "macro_score_quantile": args.macro_score_quantile,
            "macro_max_fraction": args.macro_max_fraction,
            "meso_score_quantile": args.meso_score_quantile,
            "macro_min_span": args.macro_min_span,
            "meso_min_span": args.meso_min_span,
            "macro_min_neighbors": args.macro_min_neighbors,
            "meso_min_neighbors": args.meso_min_neighbors,
            "macro_agg_major": args.macro_agg_major,
            "macro_agg_minor": args.macro_agg_minor,
            "macro_agg_max_length": args.macro_agg_max_length,
            "macro_agg_max_height": args.macro_agg_max_height,
            "enable_elongation": not args.no_elongation,
            "elongation_max_stretch": args.elongation_stretch,
            "elongation_gap_fill": args.elongation_fill,
            "elongation_max_neighbor_dist": args.elongation_range,
            "macro_elongation_stretch": args.macro_elongation_stretch,
            "macro_elongation_range": args.macro_elongation_range,
            "jitter_sigma_xy": 0.0 if args.no_jitter else args.jitter_xy,
            "jitter_along_strike_factor": args.jitter_strike_factor,
            "jitter_seed": args.jitter_seed,
        })
    if args.boundary_connect:
        kwargs.update({
            "enable_boundary_connect": True,
            "boundary_strip_width": args.bc_strip_width,
            "connect_max_gap": args.bc_max_gap,
            "bc_macro_gap": args.bc_macro_gap,
            "bc_minor_limit": args.bc_minor_limit,
            "bc_z_gap": args.bc_z_gap,
            "bc_azimuth_tol_deg": args.bc_azimuth_tol,
            "bc_dip_tol_deg": args.bc_dip_tol,
            "bc_min_score_threshold": args.bc_min_score,
            "bc_corridor_bonus": args.bc_corridor_bonus,
            "bc_require_corridor": args.bc_require_corridor,
            "bc_perp_ratio": args.bc_perp_ratio,
            "bc_enable_supplement": not args.bc_no_supplement,
            "bc_max_stretch_ratio": args.bc_max_stretch,
        })
    if args.bc_seam_fill:
        kwargs.update({
            "enable_seam_fill": True,
            "seam_half_width": args.bc_seam_width,
            "seam_interior_depth": args.bc_seam_interior,
            "seam_fill_fraction": args.bc_seam_fraction,
            "seam_confidence_decay": args.bc_seam_decay,
            "seam_seed": args.bc_seam_seed,
            "seam_blend_half_width": args.bc_blend_width,
            "seam_blend_strength": args.bc_blend_strength,
            "seam_min_shared_ratio": args.bc_min_shared_ratio,
            "seam_high_rel_boost": args.bc_high_rel_boost,
        })
    return kwargs


def _build_log_lines(args: argparse.Namespace, input_path: Path, stats: dict[str, Any]) -> list[str]:
    """Format compact runtime summary lines."""
    lines = [
        f"phase: {args.phase}",
        f"input: {input_path}",
        f"patches: {stats.get('input_count', '?')} -> {stats.get('output_count', '?')}",
        f"fracture_sets: {stats.get('fracture_sets', '?')}",
        f"reliability: {stats.get('reliability_distribution', {})}",
        f"boundary_matched_pairs: {stats.get('boundary_matched_pairs', 0)}",
    ]
    p2 = stats.get("phase2") if args.phase == 2 else None
    if isinstance(p2, dict):
        lines.extend([
            f"corridors: {p2.get('corridors_detected', 0)}",
            f"patches_in_corridors: {p2.get('patches_in_corridors', 0)}",
            f"scale_classes: {p2.get('scale_class_counts', {})}",
            f"supplemented_patches: {p2.get('supplemented_patches', 0)}",
        ])
        if p2.get("aggregation"):
            aggregation = p2["aggregation"]
            lines.append(f"aggregation: {aggregation.get('input_patches', '?')} -> {aggregation.get('output_patches', '?')}")
        if p2.get("elongation"):
            elongation = p2["elongation"]
            lines.append(f"elongation: {elongation.get('patches_stretched', 0)} patches")
    if stats.get("boundary_connect"):
        bc = stats["boundary_connect"]
        lines.append(
            f"boundary_connect: {bc.get('connections_made', 0)} pairs, {bc.get('supplements_added', 0)} bridge patches"
        )
    if stats.get("seam_fill"):
        sf = stats["seam_fill"]
        lines.append(
            f"seam_fill: {sf.get('seam_fills', 0)} new patches, {sf.get('blended_patches', 0)} blended patches"
        )
    return lines


def main() -> None:
    """CLI entrypoint override."""
    args = build_parser().parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_kwargs = _collect_pipeline_kwargs(args)

    if args.input_vtk:
        input_path = Path(args.input_vtk)
        if not input_path.exists():
            raise FileNotFoundError(f"input VTK not found: {input_path}")
        output_vtk = Path(args.output_vtk) if args.output_vtk else input_path.with_name(
            f"{input_path.stem}_phase{args.phase}_{ts}{input_path.suffix}"
        )
        if output_vtk.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {output_vtk}")
        stats = postprocess_vtk(input_path, output_vtk, phase=args.phase, **pipeline_kwargs)
        print(f"[postprocess] VTK saved: {output_vtk}")
    else:
        input_path = Path(args.input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"input CSV not found: {input_path}")
        df = pd.read_csv(input_path, encoding="utf-8-sig")
        df, stats = _run_pipeline_on_dataframe(df, args.phase, **pipeline_kwargs)
        output_csv = Path(args.output_csv) if args.output_csv else input_path.with_name(
            f"{input_path.stem}_phase{args.phase}_{ts}.csv"
        )
        if output_csv.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {output_csv}")
        write_csv_utf8(df, output_csv)
        print(f"[postprocess] CSV saved: {output_csv}")

    summary_path = input_path.with_name(f"{input_path.stem}_phase{args.phase}_summary_{ts}.json")
    write_json(summary_path, stats)
    print(f"[postprocess] summary: {summary_path}")

    log_lines = _build_log_lines(args, input_path, stats)
    try:
        append_lines_to_docx(args.docx_path, f"geophysical_postprocess phase{args.phase} {ts}", log_lines)
    except Exception:
        pass
    for line in log_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
