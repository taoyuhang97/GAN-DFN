#!/usr/bin/env python3
"""Step7A (太古界): small-scale fracture patches with layered weighted
probability sampling, density-priority 3D spacing, and geologic orientations.

Changes vs v1:
  * per-layer candidate reference quantile (no global threshold that removed
    the upper layer -> no more 'one surface' concentration);
  * occurrence probability proportional to density/reference (glutenite
    v2_weighted_probability style, no greedy global suppression);
  * orientation = multi-well, per-layer orientation families combined with
    local density-gradient/ridge evidence; manual templates are last-resort
    fallbacks only;
  * imaging match QC: 405 well-corridor spatial bins + per-well distributional
    comparison for the other imaging wells.

Outputs (config.output_dir):
  fracture_patches.csv / fracture_patches.vtk
  orientation_families.json / orientation_families.csv（现为“按井经验产状库”审计表）
  step7a_imaging_match_qc.csv
  step7a_summary.json    status=pass

Changes vs v4（2026-09-17，对齐砂砾岩"背景 + 受限构造派生"口径）:
  * 分域采样：背景域（Step6D background_small_density）为主体；断层派生域
    （Step6D large_damage_small_density）单独抽样，总量按
    `damage_domain.max_selected_fraction_of_background` 限制（砂砾岩同名参数 0.75）；
  * 每片带 SmallDomain / SmallDomainCode / StructuralRelation，便于对外解释
    "哪些小尺度片来自背景、哪些来自断层旁派生"。
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

# P0-4：测井段坐标回接统一走公共模块（太古界/common/well_segment_join）
TAIGU_ROOT = Path(__file__).resolve().parents[1]
if str(TAIGU_ROOT) not in sys.path:
    sys.path.insert(0, str(TAIGU_ROOT))
from common.well_segment_join import (  # noqa: E402
    attach_geometry_per_segment,
    summarize_join,
)
from common.orientation_frame import convention as orientation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small-scale fracture patches (太古界 Step7A v2).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=0, help="Smoke-test cap.")
    parser.add_argument(
        "--sampling-qc-only",
        action="store_true",
        help="Stop after candidate sampling and spatial thinning; skip orientation/VTK.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def layer_parameter(value: Any, layer: str) -> float:
    """Read either a scalar or a per-layer numeric configuration value."""
    if isinstance(value, dict):
        if layer not in value:
            raise KeyError(f"missing per-layer sampling parameter for {layer}")
        return float(value[layer])
    return float(value)


def thin_by_3d_spacing(
    rows: np.ndarray,
    samples: np.ndarray,
    priority: np.ndarray,
    row_xy: np.ndarray,
    sample_axis: np.ndarray,
    min_distance_m: float,
    time_scale_m_per_ms: float,
) -> np.ndarray:
    """Greedily retain stronger candidates separated in scaled XYZ space."""
    if min_distance_m <= 0.0 or len(rows) <= 1:
        return np.arange(len(rows), dtype=np.int64)
    points = np.column_stack(
        [row_xy[rows, 0], row_xy[rows, 1], sample_axis[samples] * time_scale_m_per_ms]
    )
    order = np.argsort(-priority, kind="stable")
    buckets: dict[tuple[int, int, int], list[int]] = {}
    retained: list[int] = []
    threshold_sq = min_distance_m * min_distance_m
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for candidate_index in tqdm(order, desc="Step7A 3D spacing", unit="candidate"):
        point = points[candidate_index]
        key = tuple(np.floor(point / min_distance_m).astype(np.int64))
        too_close = False
        for di, dj, dk in offsets:
            for accepted_index in buckets.get((key[0] + di, key[1] + dj, key[2] + dk), ()):
                delta = point - points[accepted_index]
                if float(delta @ delta) < threshold_sq:
                    too_close = True
                    break
            if too_close:
                break
        if not too_close:
            retained.append(int(candidate_index))
            buckets.setdefault(key, []).append(int(candidate_index))
    return np.asarray(retained, dtype=np.int64)


def normal_to_dip_azimuth(normal: np.ndarray) -> tuple[float, float]:
    """**向下三维帧**面法向 → ``(真倾角, 真倾向方位[0,360))``。

    2026-09-22 统一口径：本函数只保留"罗盘倾向方位"一种语义，
    历史 ``math_angle`` / ``compass_strike`` 两个开关已删除
    （口径混乱正是本轮剖面异常的根因）。换算统一走
    :mod:`common.orientation_frame.convention`。
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1.0e-12)
    dip_azimuth, dip = orientation.dip_azimuth_dip_from_normal_depth(normal)
    return dip, dip_azimuth


def write_legacy_vtk(path: Path, points: np.ndarray, quads: np.ndarray, cell_data: dict[str, np.ndarray]) -> None:
    lines = [
        "# vtk DataFile Version 3.0",
        "taigu_small_scale_fracture_patches_raw_time",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} double",
    ]
    lines.extend(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}" for p in points)
    lines.append(f"POLYGONS {len(quads)} {sum(4 + 1 for _ in quads)}")
    lines.extend(f"4 {' '.join(str(int(i)) for i in quad)}" for quad in quads)
    lines.append(f"CELL_DATA {len(quads)}")
    for name, values in cell_data.items():
        values = np.asarray(values)
        if values.dtype.kind in "biu":
            lines.append(f"SCALARS {name} int 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(str(int(v)) for v in values)
        else:
            lines.append(f"SCALARS {name} double 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(f"{float(v):.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def dip_azimuth_mean_std_deg(degrees: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """**全圆**平均/离散：真倾向方位 ``0–360``（不是 axial 0–180）。

    2026-09-22 统一口径：倾向方位有方向性（"往哪边倒"），不能按
    ``theta`` 与 ``theta+180`` 等价来统计。旧实现 ``axial_mean_std_deg``
    把倾向方位按半圆统计，是口径混乱的一部分，已删除。
    """
    angle = np.asarray(degrees, dtype=np.float64)
    valid = np.isfinite(angle)
    angle = angle[valid]
    if not len(angle):
        return np.nan, np.nan
    weight = np.ones(len(angle), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)[valid]
    weight = weight / max(float(weight.sum()), 1.0e-12)
    vector = np.sum(weight * np.exp(1j * np.deg2rad(angle)))
    mean = float(np.rad2deg(np.angle(vector)) % 360.0)
    resultant = float(np.clip(abs(vector), 1.0e-12, 1.0))
    std = float(np.rad2deg(np.sqrt(max(-2.0 * np.log(resultant), 0.0))))
    return mean, std


def load_multiwell_orientation_points(
    groups_root: Path,
    allowed_wells: dict[str, list[str]],
    samples_root: Path,
    tolerance: float | None = None,
) -> pd.DataFrame:
    """Load valid image-log orientations and attach their interpolated well XY.

    P0-4：回接改走公共 well_segment_join 模块，容差按各井 MD 半采样步长自动推导
    （旧实现写死 0.05 m，且未命中的点会被静默丢弃）。
    """
    parts: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    allowed = {layer: set(wells) for layer, wells in allowed_wells.items()}
    for group_file in sorted(Path(groups_root).glob("*.csv")):
        group = pd.read_csv(group_file, encoding="utf-8-sig")
        required = {"WellName", "StrataName", "GT_POINT_FLAG", "FracAzimuth", "FracDip", "MD", "InputSegmentPath"}
        if group.empty or not required.issubset(group.columns):
            continue
        points = group[group["GT_POINT_FLAG"].fillna(0).astype(int).eq(1)].copy()
        # 2026-09-22 统一口径：甲方 FracAzimuth 是**真倾向方位**（0–360，自北顺时针），
        # 不能再折叠到 0–180（折叠会把"往哪边倒"抹掉，与下游倾向方位口径冲突）。
        points["FracAzimuth"] = pd.to_numeric(points["FracAzimuth"], errors="coerce") % 360.0
        points["FracDip"] = pd.to_numeric(points["FracDip"], errors="coerce")
        points["MD"] = pd.to_numeric(points["MD"], errors="coerce")
        points = points.dropna(subset=["FracAzimuth", "FracDip", "MD"])
        keep = np.asarray(
            [str(well) in allowed.get(str(layer), set()) for well, layer in zip(points["WellName"], points["StrataName"])],
            dtype=bool,
        )
        points = points.loc[keep].copy()
        if points.empty:
            continue
        # 逐段回接：每行自带 InputSegmentPath，只在该段内取坐标
        matched, audit = attach_geometry_per_segment(
            points,
            preferred_column="InputSegmentPath",
            tolerance_m=tolerance,
            geometry_source="step7a_orientation",
            samples_root=samples_root,
        )
        # 保持固定的行序（段 × MD）：抽井用的平滑随机场与片序无关，但固定顺序便于复现。
        if {"InputSegmentPath", "MD"}.issubset(matched.columns):
            matched = matched.sort_values(["InputSegmentPath", "MD"], kind="mergesort").reset_index(drop=True)
        parts.append(matched)
        audits.append(audit)
    if not parts:
        return pd.DataFrame(columns=["WellName", "StrataName", "FracAzimuth", "FracDip", "X", "Y"]), summarize_join(pd.DataFrame())
    merged_all = pd.concat(parts, ignore_index=True)
    join_summary = summarize_join(pd.concat(audits, ignore_index=True) if audits else pd.DataFrame())
    matched_mask = merged_all["MDJoinStatus"].astype(str).eq("matched")
    keep_columns = ["WellName", "StrataName", "FracAzimuth", "FracDip", "X", "Y", "InHorizonLayer"]
    keep_columns = [column for column in keep_columns if column in merged_all.columns]
    essential = [column for column in ("WellName", "StrataName", "FracAzimuth", "FracDip", "X", "Y") if column in keep_columns]
    return (
        merged_all.loc[matched_mask, keep_columns].dropna(subset=essential).reset_index(drop=True),
        join_summary,
    )


def blend_dip_azimuth(base_azimuth: float, other_azimuth: float, weight: float) -> float:
    """真倾向方位（0–360）的**全圆**加权平均。"""
    weight = float(np.clip(weight, 0.0, 1.0))
    base_vector = np.exp(1j * np.deg2rad(base_azimuth))
    other_vector = np.exp(1j * np.deg2rad(other_azimuth))
    return float(np.rad2deg(np.angle((1.0 - weight) * base_vector + weight * other_vector)) % 360.0)


def build_well_orientation_library(
    points: pd.DataFrame,
    wells_by_layer: dict[str, list[str]],
    config: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """按 ``(层, 井)`` 建立**经验产状库**：原样保留该井的 (倾向方位, 倾角) 点集。

    **不做聚类、不做平均**——聚类/取均值会把多峰与陡倾尾巴抹掉，正是上一轮
    "DFN 与成像分布对不上"的根因（见 `太古界DFN产状赋值逻辑修正方案_20260923.md` §1）。

    决策 D1：使用**全部**成像点（含落在目标层外的点），并额外记录层内占比供审计。
    """
    library: dict[str, dict[str, dict[str, Any]]] = {}
    min_points = int(config.get("well_library_min_points", 5))
    full_weight_points = int(config.get("well_library_min_points_for_full_weight", 50))
    small_sample_discount = float(config.get("well_library_small_sample_discount", 0.5))
    for layer, wells in wells_by_layer.items():
        entries: dict[str, dict[str, Any]] = {}
        for well in wells:
            sub = points[
                points["StrataName"].astype(str).eq(str(layer))
                & points["WellName"].astype(str).eq(str(well))
            ]
            if sub.empty:
                continue
            dip_azimuth = pd.to_numeric(sub["FracAzimuth"], errors="coerce").to_numpy(dtype=np.float64) % 360.0
            dip = pd.to_numeric(sub["FracDip"], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(dip_azimuth) & np.isfinite(dip)
            dip_azimuth, dip = dip_azimuth[valid], dip[valid]
            if len(dip_azimuth) < min_points:
                continue
            if "InHorizonLayer" in sub.columns:
                in_horizon = pd.to_numeric(sub["InHorizonLayer"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)[valid]
                in_horizon_fraction = float(np.mean(in_horizon > 0.5)) if len(in_horizon) else 0.0
            else:
                in_horizon_fraction = float("nan")
            entries[str(well)] = {
                "dip_azimuth_deg": dip_azimuth,
                "dip_deg": dip,
                "x": float(pd.to_numeric(sub["X"], errors="coerce").median()),
                "y": float(pd.to_numeric(sub["Y"], errors="coerce").median()),
                "point_count": int(len(dip_azimuth)),
                # R2：点数过少的井（如埕北310 仅 34 点）经验分布不稳，降权
                "sample_weight_scale": 1.0 if len(dip_azimuth) >= full_weight_points else small_sample_discount,
                "in_horizon_fraction": in_horizon_fraction,
                "dip_azimuth_circular_mean_deg": orientation.circular_mean_dip_azimuth(dip_azimuth),
                "dip_median_deg": float(np.median(dip)),
            }
        library[str(layer)] = entries
    return library


def well_distance_weights(
    x: float,
    y: float,
    entries: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    """按**高斯距离核**给出该点各井的权重（决策 D2）。

    用高斯核而不是硬性"最近 K 口井"：硬截断会在 K 边界产生权重跳变，
    高斯核处处光滑，实际效果等价于"只受最近 2–4 口井支配"。
    """
    scale_m = float(config.get("well_weight_scale_m", 6000.0))
    min_share = float(config.get("well_weight_min_share", 0.02))
    names: list[str] = []
    raw: list[float] = []
    for well, entry in entries.items():
        distance = float(np.hypot(x - float(entry["x"]), y - float(entry["y"])))
        kernel = float(np.exp(-((distance / max(scale_m, 1.0)) ** 2)))
        names.append(well)
        raw.append(kernel * float(entry.get("sample_weight_scale", 1.0)))
    weights = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(names), dtype=np.float64)
    weights = weights / weights.sum()
    keep = weights >= min_share
    if not keep.any():
        keep = np.zeros(len(names), dtype=bool)
        keep[int(np.argmax(weights))] = True
    names = [name for name, flag in zip(names, keep) if flag]
    weights = weights[keep]
    weights = weights / weights.sum()
    return names, weights


class CoherentUniformField:
    """空间相关的均匀随机场：粗格随机数 → 高斯平滑 → 双线性插值（决策 D2 的落地关键）。

    逐片独立抽井会让相邻片随机来自不同井，看起来仍然"乱"；用一张平滑随机场做分位点，
    相邻片就会倾向抽到同一口井，形成**空间连贯的产状域**。
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, spacing_m: float, seed: int, sigma_cells: float = 1.0) -> None:
        self.spacing = max(float(spacing_m), 1.0)
        x0 = float(np.nanmin(xs)) - self.spacing
        x1 = float(np.nanmax(xs)) + self.spacing
        y0 = float(np.nanmin(ys)) - self.spacing
        y1 = float(np.nanmax(ys)) + self.spacing
        self.x0, self.y0 = x0, y0
        self.nx = max(int(np.ceil((x1 - x0) / self.spacing)) + 1, 3)
        self.ny = max(int(np.ceil((y1 - y0) / self.spacing)) + 1, 3)
        rng = np.random.default_rng(int(seed))
        coarse = rng.random((self.ny, self.nx))
        self.field = gaussian_filter(coarse, sigma=float(max(sigma_cells, 0.0)), mode="nearest")
        lo, hi = float(self.field.min()), float(self.field.max())
        self.field = (self.field - lo) / max(hi - lo, 1e-12)

    def value(self, x: float, y: float) -> float:
        fx = (float(x) - self.x0) / self.spacing
        fy = (float(y) - self.y0) / self.spacing
        ix = int(np.clip(np.floor(fx), 0, self.nx - 2))
        iy = int(np.clip(np.floor(fy), 0, self.ny - 2))
        tx = float(np.clip(fx - ix, 0.0, 1.0))
        ty = float(np.clip(fy - iy, 0.0, 1.0))
        v00 = self.field[iy, ix]
        v10 = self.field[iy, ix + 1]
        v01 = self.field[iy + 1, ix]
        v11 = self.field[iy + 1, ix + 1]
        top = v00 * (1.0 - tx) + v10 * tx
        bottom = v01 * (1.0 - tx) + v11 * tx
        return float(np.clip(top * (1.0 - ty) + bottom * ty, 0.0, 1.0 - 1e-9))


def sample_statistical_orientation(
    names: list[str],
    weights: np.ndarray,
    entries: dict[str, dict[str, Any]],
    quantile: float,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[float, float, str]:
    """按井权重在累积权重上反查井（用空间相关分位点），再从该井经验库采样 + 抖动。"""
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, float(quantile), side="right"))
    index = int(np.clip(index, 0, len(names) - 1))
    well = names[index]
    entry = entries[well]
    dip_azimuth_pool = np.asarray(entry["dip_azimuth_deg"], dtype=np.float64)
    dip_pool = np.asarray(entry["dip_deg"], dtype=np.float64)
    pick = int(rng.integers(0, len(dip_pool)))
    azimuth = (float(dip_azimuth_pool[pick]) + float(rng.normal(0.0, float(config.get("azimuth_jitter_std_deg", 8.0))))) % 360.0
    dip = float(dip_pool[pick]) + float(rng.normal(0.0, float(config.get("dip_jitter_std_deg", 5.0))))
    dip = float(np.clip(dip, float(config.get("dip_min_deg", 5.0)), float(config.get("dip_max_deg", 89.0))))
    return azimuth, dip, well


def fuse_local_geometry_beta(
    dip_azimuth: float,
    dip: float,
    local: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, float, int, float, float]:
    """决策 D3（β 方案）：**只在走向一致时小幅融合倾角**。

    比较的是**走向**（轴向 0–180）而不是倾向方位——平面法向的正负本身是任意的，
    用走向比较才稳健。不满足条件时完全忽略局部几何。
    """
    local_dip = float(local.get("dip", np.nan))
    local_azimuth = float(local.get("dip_azimuth", np.nan))
    quality = float(local.get("quality", 0.0))
    if not (np.isfinite(local_dip) and np.isfinite(local_azimuth)):
        return dip_azimuth, dip, 0, 0.0, float("nan")
    strike_delta = orientation.axial_distance_deg(
        orientation.strike_from_dip_azimuth(dip_azimuth),
        orientation.strike_from_dip_azimuth(local_azimuth),
    )
    tolerance = float(config.get("local_geometry_strike_tolerance_deg", 30.0))
    min_quality = float(config.get("local_geometry_min_quality", 0.5))
    max_blend = float(config.get("local_geometry_dip_blend_max", 0.30))
    if strike_delta <= tolerance and quality >= min_quality:
        beta = max_blend * float(np.clip(quality, 0.0, 1.0))
        blended_dip = float(
            np.clip(
                (1.0 - beta) * dip + beta * local_dip,
                float(config.get("dip_min_deg", 5.0)),
                float(config.get("dip_max_deg", 89.0)),
            )
        )
        blended_azimuth = blend_dip_azimuth(dip_azimuth, local_azimuth, beta)
        return blended_azimuth, blended_dip, 1, beta, strike_delta
    return dip_azimuth, dip, 0, 0.0, strike_delta


def local_gradient_ridge_orientation(
    row: int,
    sample: int,
    center_density: float,
    cell_to_row: dict[tuple[int, int], int],
    row_to_ix: np.ndarray,
    row_to_iy: np.ndarray,
    dens_by_row: dict[int, np.ndarray],
    sample_axis: np.ndarray,
    xy_radius: int,
    time_radius: int,
    dx_m: float,
    dy_m: float,
    time_scale_m_per_ms: float,
    density_fraction: float,
    gradient_quantile: float,
    smoothing_sigma: float,
) -> dict[str, float | int]:
    """由局部密度梯度/脊线估计裂缝面法向，返回**真倾向方位**（0–360）。

    2026-09-22 统一口径：``normal`` 在 ``(x=东, y=北, z=TIME向下)`` 帧里，
    方位换算走 :func:`common.orientation_frame.convention.dip_azimuth_dip_from_normal_depth`；
    旧开关 ``ridge_azimuth_semantics`` 已删除。
    """
    shape = (2 * xy_radius + 1, 2 * xy_radius + 1, 2 * time_radius + 1)
    cube = np.full(shape, np.nan, dtype=np.float64)
    center_ix, center_iy = int(row_to_ix[row]), int(row_to_iy[row])
    for xi, di in enumerate(range(-xy_radius, xy_radius + 1)):
        for yi, dj in enumerate(range(-xy_radius, xy_radius + 1)):
            neighbor = cell_to_row.get((center_ix + di, center_iy + dj))
            if neighbor is None or neighbor not in dens_by_row:
                continue
            trace = dens_by_row[neighbor]
            for zi, ds in enumerate(range(-time_radius, time_radius + 1)):
                source_sample = sample + ds
                if 0 <= source_sample < len(trace):
                    cube[xi, yi, zi] = float(trace[source_sample])
    valid = np.isfinite(cube)
    if int(valid.sum()) < 27:
        return {"dip": np.nan, "dip_azimuth": np.nan, "anisotropy": 0.0, "gradient_strength": 0.0,
                "ridge_strength": 0.0, "quality": 0.0, "support_count": int(valid.sum())}
    fill_value = float(np.nanmedian(cube))
    filled = np.where(valid, cube, fill_value)
    smooth = gaussian_filter(filled, sigma=smoothing_sigma, mode="nearest")
    dz_m = max(float(np.median(np.diff(sample_axis))) * time_scale_m_per_ms, 1.0e-6)
    gx, gy, gz = np.gradient(smooth, dx_m, dy_m, dz_m, edge_order=1)
    gradient_magnitude = np.sqrt(gx * gx + gy * gy + gz * gz)
    finite_gradient = gradient_magnitude[valid]
    if not len(finite_gradient) or float(np.nanmax(finite_gradient)) <= 0.0:
        return {"dip": np.nan, "dip_azimuth": np.nan, "anisotropy": 0.0, "gradient_strength": 0.0,
                "ridge_strength": 0.0, "quality": 0.0, "support_count": int(valid.sum())}
    gradient_threshold = float(np.quantile(finite_gradient, gradient_quantile))
    gxx = np.gradient(gx, dx_m, axis=0, edge_order=1)
    gyy = np.gradient(gy, dy_m, axis=1, edge_order=1)
    gzz = np.gradient(gz, dz_m, axis=2, edge_order=1)
    ridge = np.maximum(-(gxx + gyy + gzz), 0.0)
    evidence = valid & (smooth >= density_fraction * center_density) & (
        (gradient_magnitude >= gradient_threshold) | (ridge >= np.quantile(ridge[valid], gradient_quantile))
    )
    support_count = int(evidence.sum())
    if support_count < 8:
        return {"dip": np.nan, "dip_azimuth": np.nan, "anisotropy": 0.0, "gradient_strength": 0.0,
                "ridge_strength": 0.0, "quality": 0.0, "support_count": support_count}
    vectors = np.column_stack([gx[evidence], gy[evidence], gz[evidence]])
    local_gradient = gradient_magnitude[evidence]
    local_ridge = ridge[evidence]
    ridge_norm = local_ridge / max(float(np.quantile(ridge[valid], 0.95)), 1.0e-12)
    weights = local_gradient * (1.0 + np.clip(ridge_norm, 0.0, 2.0))
    weights /= max(float(weights.sum()), 1.0e-12)
    tensor = (vectors * weights[:, None]).T @ vectors
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalues)[::-1]
    leading, second = float(eigenvalues[order[0]]), float(eigenvalues[order[1]])
    anisotropy = max((leading - second) / max(leading, 1.0e-12), 0.0)
    normal = eigenvectors[:, order[0]]
    dip, dip_azimuth = normal_to_dip_azimuth(normal)
    p50 = float(np.quantile(finite_gradient, 0.50))
    p90 = float(np.quantile(finite_gradient, 0.90))
    gradient_strength = float(np.clip((p90 - p50) / max(p90, 1.0e-12), 0.0, 1.0))
    ridge_strength = float(np.clip(np.median(ridge_norm), 0.0, 1.0))
    quality = float(np.clip(0.65 * anisotropy + 0.25 * gradient_strength + 0.10 * ridge_strength, 0.0, 1.0))
    return {
        "dip": dip, "dip_azimuth": dip_azimuth, "anisotropy": anisotropy,
        "gradient_strength": gradient_strength, "ridge_strength": ridge_strength,
        "quality": quality, "support_count": support_count,
    }


def load_imaging_density_profile(config: dict[str, Any], well: str) -> pd.DataFrame:
    """Imaging density vs TIME (borrowed) for QC, joining every segment path.

    P0-4：回接改走公共 well_segment_join 模块；未命中的点不再静默丢弃。
    """
    group_files = sorted(Path(config["step3_groups_root"]).glob(f"{well}_*.csv"))
    parts = []
    audits = []
    samples_root = Path(config["step2_segments_root"])
    join_config = dict(config.get("segment_join", {}))
    tolerance_setting = join_config.get("tolerance_m", config.get("md_merge_tolerance"))
    tolerance = (
        None
        if tolerance_setting is None or str(tolerance_setting).strip().lower() in {"", "auto"}
        else float(tolerance_setting)
    )
    for group_file in group_files:
        group = pd.read_csv(group_file, encoding="utf-8-sig")
        if group.empty or not {"WellName", "MD"}.issubset(group.columns):
            continue
        group = group.copy()
        group["MD"] = pd.to_numeric(group["MD"], errors="coerce")
        matched, audit = attach_geometry_per_segment(
            group,
            preferred_column="InputSegmentPath",
            tolerance_m=tolerance,
            geometry_source="step7a_density_profile",
            samples_root=samples_root,
        )
        parts.append(matched)
        audits.append(audit)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    config["_density_profile_join_qc"] = summarize_join(
        pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    )
    for column in ("X", "Y", "TIME", "Density"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["X", "Y", "TIME", "Density"]).copy()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_csv = output_dir / "fracture_patches.csv"
    vtk_path = output_dir / "fracture_patches.vtk"
    qc_csv = output_dir / "step7a_imaging_match_qc.csv"
    families_json_path = output_dir / "orientation_well_library.json"
    families_csv_path = output_dir / "orientation_well_library.csv"
    sampling_qc_path = output_dir / "step7a_sampling_qc.json"
    summary_path = output_dir / "step7a_summary.json"
    output_paths_to_check = (sampling_qc_path,) if args.sampling_qc_only else (
        patches_csv, vtk_path, qc_csv, families_json_path, families_csv_path,
        sampling_qc_path, summary_path
    )
    for path in output_paths_to_check:
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing Step7A output: {path}")

    started = time.time()
    grid = pd.read_csv(config["demo_grid_csv"], encoding="utf-8-sig")
    # SGY/window_code 的物理存储顺序是 output_trace_index；demo_grid 的
    # TraceIdx 是合同道序，二者不能直接互换。按坐标回接正式道序。
    mapping_path = config.get("trace_mapping_npz")
    if mapping_path:
        with np.load(mapping_path) as mp:
            mapping_df = pd.DataFrame({
                "X": mp["x"].astype(float), "Y": mp["y"].astype(float),
                "OutputTraceIndex": mp["output_trace_index"].astype(np.int64),
            })
        grid = grid.merge(mapping_df, on=["X", "Y"], how="left", validate="one_to_one")
        if grid["OutputTraceIndex"].isna().any():
            raise RuntimeError("demo grid contains coordinates absent from trace mapping")
        grid["OutputTraceIndex"] = grid["OutputTraceIndex"].astype(np.int64)
    else:
        # 兼容旧配置，但明确提示该模式只适用于两种道序恰好一致的输入。
        grid["OutputTraceIndex"] = np.arange(len(grid), dtype=np.int64)
    horizon = pd.read_csv(config["horizon_contract_csv"], encoding="utf-8-sig")
    for df in (grid, horizon):
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grid = grid.merge(
        horizon[["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"]],
        on="TraceIdx",
        how="left",
    ).sort_values("TraceIdx").reset_index(drop=True)
    grid["Row"] = np.arange(len(grid))
    if grid["TraceIdx"].duplicated().any():
        raise RuntimeError("demo grid has duplicate TraceIdx")
    cell_to_row = {(int(ix), int(iy)): int(row) for ix, iy, row in zip(grid["IX"], grid["IY"], grid["Row"])}
    row_to_ix = grid["IX"].to_numpy(dtype=np.int32)
    row_to_iy = grid["IY"].to_numpy(dtype=np.int32)
    row_xy = grid[["X", "Y"]].to_numpy(dtype=np.float64)

    windows = np.load(config["window_code_npz"])
    window_codes = windows["window_code"]
    sample_axis = windows["sample_axis"].astype(np.float64)
    density_sgy = Path(config["density_sgy"]).resolve()
    prediction_summary_path = Path(config["step6a_prediction_summary_json"]).resolve()
    prediction_summary = read_json(prediction_summary_path)
    if prediction_summary.get("status") != "pass":
        raise RuntimeError("Step6A prediction summary is not pass")
    # 背景域密度体：v5 起指向 Step6D 的 background_small_density.sgy
    # （= 0.06 + 0.85×Step6A 小尺度背景分数），不再直接用最大值合成体。
    declared_outputs = dict(prediction_summary.get("output_paths", {}))
    density_contract_key = str(config.get("density_contract_output_key", "density_sgy"))
    if Path(declared_outputs.get(density_contract_key, "")).resolve() != density_sgy:
        raise RuntimeError(
            f"Step7A density_sgy does not match the declared Step6A/6D contract key '{density_contract_key}'"
        )
    damage_cfg = dict(config.get("damage_domain", {}))
    damage_enabled = bool(damage_cfg.get("enabled", False))
    damage_domain_name = str(damage_cfg.get("name", "large_fault_damage"))
    damage_domain_code = int(damage_cfg.get("domain_code", 2))
    damage_structural_relation = str(
        damage_cfg.get("structural_relation", "small_derivative_near_large_fault")
    )
    damage_density_sgy: Path | None = None
    if damage_enabled:
        if not config.get("damage_density_sgy"):
            raise RuntimeError("damage_domain.enabled requires config['damage_density_sgy']")
        damage_density_sgy = Path(config["damage_density_sgy"]).resolve()
        damage_contract_key = str(config.get("damage_contract_output_key", "damage_density_sgy"))
        if Path(declared_outputs.get(damage_contract_key, "")).resolve() != damage_density_sgy:
            raise RuntimeError(
                f"Step7A damage_density_sgy does not match the declared contract key '{damage_contract_key}'"
            )
    expected_step6a_contract = config.get("expected_step6a_model_contract_version")
    if expected_step6a_contract and prediction_summary.get("model_contract_version") != expected_step6a_contract:
        raise RuntimeError(
            f"Step6A model contract mismatch: {prediction_summary.get('model_contract_version')} "
            f"!= {expected_step6a_contract}"
        )
    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as density_handle:
        if density_handle.tracecount != len(grid):
            raise RuntimeError("Step6A density trace count does not match Step7A demo grid")
        if len(density_handle.samples) != len(sample_axis) or not np.allclose(density_handle.samples, sample_axis):
            raise RuntimeError("Step6A density sample axis does not match window_code sample axis")
    if damage_density_sgy is not None:
        with segyio.open(str(damage_density_sgy), "r", ignore_geometry=True) as damage_handle:
            if damage_handle.tracecount != len(grid):
                raise RuntimeError("Step7A damage density trace count does not match demo grid")
            if len(damage_handle.samples) != len(sample_axis) or not np.allclose(
                damage_handle.samples, sample_axis
            ):
                raise RuntimeError("Step7A damage density sample axis does not match window_code sample axis")
    if window_codes.shape != (len(grid), len(sample_axis)):
        raise RuntimeError("window_code shape does not match Step7A grid/sample axis")
    block_x_lines = max(int(config.get("block_x_line_count", 20)), 1)
    x_line_count = int(grid["IX"].max()) + 1
    sampling_cfg = config["sampling"]
    # 不按 demo 区域固定总片数；数量应随候选体素数量和 occurrence_rate
    # 自然增长。若工程上需要防止配置错误导致内存爆炸，只接受显式的
    # safety_max_patch_count 保护阈值，不参与正常采样和缩放。
    safety_max_patches = sampling_cfg.get("safety_max_patch_count")
    safety_max_patches = int(safety_max_patches) if safety_max_patches is not None else None
    reference_quantile = float(sampling_cfg["reference_quantile"])
    # 与砂砾岩小尺度流程一致：先按层内高分位筛掉背景体素，再以较低的
    # 出现概率抽样。这样不会把每个正密度体素都画成一块裂缝片。
    candidate_quantile_cfg = sampling_cfg.get("candidate_quantile", reference_quantile)
    occurrence_rate_cfg = sampling_cfg.get("occurrence_rate", 0.20)
    density_power = float(sampling_cfg.get("density_power", 1.0))
    min_center_distance_m = float(sampling_cfg.get("min_center_distance_m", 0.0))
    spacing_time_scale = float(sampling_cfg.get("spacing_time_scale_m_per_ms", 2.0))
    length_range = [float(v) for v in config["patch_length_m"]]
    height_range = [float(v) for v in config["patch_height_ms"]]
    z_scale = float(config["display_z_scale_m_per_ms"])
    # VTK 使用原始 TIME(ms) 作为 Z；保留 z_scale 仅用于既有面积/展示口径。
    vtk_z_scale = float(config.get("vtk_z_scale_m_per_ms", 1.0))
    orient_cfg = config["orientation"]
    orient_xy_radius = int(orient_cfg["window_xy_radius"])
    orient_time_half = float(orient_cfg["time_half_span_ms"])
    orient_time_span = int(round(orient_time_half / (sample_axis[1] - sample_axis[0])))
    orient_density_fraction = float(orient_cfg["density_fraction"])
    # 2026-09-23：产状改由"井经验分布重采样"给出，倾角范围取自采样器的裁剪区间
    # （旧键 min_dip_deg/max_dip_deg 只留作参考，不再参与门控——按成像实测分布，
    #  真实的浅倾角（<25°）应当允许出现）。
    dip_floor = float(orient_cfg.get("dip_min_deg", 5.0))
    dip_ceil = float(orient_cfg.get("dip_max_deg", 89.0))
    sampling_rng = np.random.default_rng(int(sampling_cfg["random_seed"]))
    orientation_rng = np.random.default_rng(int(orient_cfg["random_seed"]))

    # per-layer positive-density samples for reference quantiles
    layer_refs: dict[str, float] = {}
    layer_positives: dict[str, list[np.ndarray]] = {"上部复合层": [], "太古界风化壳": []}
    candidate_parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "上部复合层": [],
        "太古界风化壳": [],
    }
    damage_parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "上部复合层": [],
        "太古界风化壳": [],
    }
    valid_mask = grid["SurfaceValid"].fillna(0).astype(bool).to_numpy()
    top = grid["TopTimeMs"].to_numpy(dtype=np.float64)
    mid = grid["MidTimeMs"].to_numpy(dtype=np.float64)
    base = grid["BaseTimeMs"].to_numpy(dtype=np.float64)

    with contextlib.ExitStack() as stack:
        handle = stack.enter_context(segyio.open(str(density_sgy), "r", ignore_geometry=True))
        handle.mmap()
        damage_handle = None
        if damage_density_sgy is not None:
            damage_handle = stack.enter_context(segyio.open(str(damage_density_sgy), "r", ignore_geometry=True))
            damage_handle.mmap()
        for ix_start in tqdm(range(0, x_line_count, block_x_lines), desc="Step7A density blocks", unit="block"):
            ix_stop = min(ix_start + block_x_lines, x_line_count)
            block = grid[grid["IX"].between(ix_start, ix_stop - 1)].sort_values("OutputTraceIndex")
            output_indices = block["OutputTraceIndex"].to_numpy(dtype=np.int64)
            matrix = np.stack([np.asarray(handle.trace[int(r)], dtype=np.float32) for r in output_indices])
            damage_matrix = (
                np.stack([np.asarray(damage_handle.trace[int(r)], dtype=np.float32) for r in output_indices])
                if damage_handle is not None
                else None
            )
            codes = window_codes[output_indices]
            valid_block = block["SurfaceValid"].fillna(0).astype(bool).to_numpy()
            times_2d = sample_axis[None, :]
            upper_mask = (codes == 1) & (times_2d >= top[block["Row"].to_numpy(dtype=np.int64), None]) & (times_2d <= mid[block["Row"].to_numpy(dtype=np.int64), None])
            crust_mask = (codes == 1) & (times_2d > mid[block["Row"].to_numpy(dtype=np.int64), None]) & (times_2d <= base[block["Row"].to_numpy(dtype=np.int64), None])
            upper_mask &= valid_block[:, None] & np.isfinite(matrix) & (matrix > 0.0)
            crust_mask &= valid_block[:, None] & np.isfinite(matrix) & (matrix > 0.0)
            for name, mask in (("上部复合层", upper_mask), ("太古界风化壳", crust_mask)):
                if not mask.any():
                    continue
                layer_positives[name].append(matrix[mask].astype(np.float32))
                rows_m, samples_m = np.where(mask)
                candidate_parts[name].append(
                    (
                        block["Row"].to_numpy(dtype=np.int64)[rows_m],
                        samples_m.astype(np.int32),
                        matrix[mask].astype(np.float32),
                    )
                )
            if damage_matrix is not None:
                block_rows = block["Row"].to_numpy(dtype=np.int64)
                damage_upper = upper_mask & np.isfinite(damage_matrix) & (damage_matrix > 0.0)
                damage_crust = crust_mask & np.isfinite(damage_matrix) & (damage_matrix > 0.0)
                for name, mask in (("上部复合层", damage_upper), ("太古界风化壳", damage_crust)):
                    if not mask.any():
                        continue
                    rows_m, samples_m = np.where(mask)
                    damage_parts[name].append(
                        (
                            block_rows[rows_m],
                            samples_m.astype(np.int32),
                            damage_matrix[mask].astype(np.float32),
                        )
                    )
            print(f"[step7a] block ix={ix_start}:{ix_stop} rss_mb={rss_mb():.0f}", flush=True)
            if args.max_blocks > 0 and ix_start // block_x_lines + 1 >= args.max_blocks:
                break

    for name in ("上部复合层", "太古界风化壳"):
        positives = np.concatenate(layer_positives[name]) if layer_positives[name] else np.empty(0, dtype=np.float32)
        layer_refs[name] = float(np.quantile(positives, reference_quantile)) if positives.size else 0.0

    sampled_rows: list[np.ndarray] = []
    sampled_samples: list[np.ndarray] = []
    sampled_density: list[np.ndarray] = []
    sampled_priority: list[np.ndarray] = []
    sampled_layer: list[str] = []
    sampled_domain: list[str] = []
    layer_sampling_audit: dict[str, dict[str, Any]] = {}
    for name in ("上部复合层", "太古界风化壳"):
        if not candidate_parts[name]:
            continue
        rows = np.concatenate([p[0] for p in candidate_parts[name]])
        samples = np.concatenate([p[1] for p in candidate_parts[name]])
        density = np.concatenate([p[2] for p in candidate_parts[name]])
        # 候选阈值和参考值均在本层正值样本上计算，避免不同层厚/振幅
        # 分布差异造成某一层被异常过采样。
        positive_count = int(len(density))
        candidate_quantile = layer_parameter(candidate_quantile_cfg, name)
        occurrence_rate = layer_parameter(occurrence_rate_cfg, name)
        candidate_ref = max(float(np.quantile(density, candidate_quantile)), 1.0e-9)
        candidate_mask = density >= candidate_ref
        rows, samples, density = rows[candidate_mask], samples[candidate_mask], density[candidate_mask]
        reference = max(float(layer_refs[name]), 1.0e-9)
        probability = occurrence_rate * np.power(np.clip(density / reference, 0.0, 2.0), density_power)
        probability = np.clip(probability, 0.0, 1.0)
        expected = float(probability.sum())
        keep = sampling_rng.random(len(rows)) < probability
        keep_pos = np.where(keep)[0]
        keep = np.zeros(len(rows), dtype=bool)
        keep[keep_pos] = True
        sampled_rows.append(rows[keep])
        sampled_samples.append(samples[keep])
        sampled_density.append(density[keep])
        # 优先级用"相对本域候选阈值的倍数"，保证背景域与构造派生域在
        # 同一个三维最小间距去重里可比（否则阈值口径不同会系统性剔除某一域）。
        sampled_priority.append((density[keep] / candidate_ref).astype(np.float32))
        if not (len(rows[keep]) == len(samples[keep]) == len(density[keep])):
            raise RuntimeError(f"sampled candidate length mismatch for layer {name}")
        sampled_layer.extend([name] * int(keep.sum()))
        sampled_domain.extend(["background"] * int(keep.sum()))
        layer_sampling_audit[name] = {
            "positive_voxel_count": positive_count,
            "candidate_quantile": candidate_quantile,
            "candidate_threshold": candidate_ref,
            "candidate_count": int(len(rows)),
            "reference_quantile": reference_quantile,
            "reference_density": reference,
            "occurrence_rate": occurrence_rate,
            "density_power": density_power,
            "expected_before_spacing": expected,
            "sampled_before_spacing": int(keep.sum()),
            "domain": "background",
            "structural_relation": "small_background",
        }
        print(f"[step7a] layer={name} candidate_q={candidate_quantile:.2f} threshold={candidate_ref:.3f} ref={reference:.3f} candidates={len(rows)} sampled={int(keep.sum())}", flush=True)

    # --- 断层派生小尺度域（second domain，砂砾岩口径的 damage domain） ---
    damage_sampling_audit: dict[str, dict[str, Any]] = {}
    damage_selected_count = 0
    if damage_enabled:
        damage_rng = np.random.default_rng(
            int(damage_cfg.get("random_seed", int(sampling_cfg["random_seed"]) + 101))
        )
        damage_quantile_cfg = damage_cfg.get("candidate_quantile", 0.5)
        damage_rate_cfg = damage_cfg.get("occurrence_rate", 0.005)
        damage_ref_quantile = float(damage_cfg.get("reference_quantile", 0.95))
        damage_power = float(damage_cfg.get("density_power", 1.0))
        for name in ("上部复合层", "太古界风化壳"):
            if not damage_parts[name]:
                continue
            rows = np.concatenate([p[0] for p in damage_parts[name]])
            samples = np.concatenate([p[1] for p in damage_parts[name]])
            density = np.concatenate([p[2] for p in damage_parts[name]])
            positive_count = int(len(density))
            damage_quantile = layer_parameter(damage_quantile_cfg, name)
            damage_rate = layer_parameter(damage_rate_cfg, name)
            damage_ref = max(float(np.quantile(density, damage_ref_quantile)), 1.0e-9)
            candidate_thr = max(float(np.quantile(density, damage_quantile)), 1.0e-9)
            candidate_mask = density >= candidate_thr
            rows, samples, density = rows[candidate_mask], samples[candidate_mask], density[candidate_mask]
            probability = np.clip(
                damage_rate * np.power(np.clip(density / damage_ref, 0.0, 2.0), damage_power), 0.0, 1.0
            )
            expected = float(probability.sum())
            keep = damage_rng.random(len(rows)) < probability
            keep = keep.astype(bool)
            sampled_rows.append(rows[keep])
            sampled_samples.append(samples[keep])
            sampled_density.append(density[keep])
            sampled_priority.append((density[keep] / candidate_thr).astype(np.float32))
            sampled_layer.extend([name] * int(keep.sum()))
            sampled_domain.extend([damage_domain_name] * int(keep.sum()))
            damage_selected_count += int(keep.sum())
            damage_sampling_audit[name] = {
                "positive_voxel_count": positive_count,
                "candidate_quantile": damage_quantile,
                "candidate_threshold": candidate_thr,
                "candidate_count": int(len(rows)),
                "reference_quantile": damage_ref_quantile,
                "reference_density": damage_ref,
                "occurrence_rate": damage_rate,
                "density_power": damage_power,
                "expected_before_spacing": expected,
                "sampled_before_spacing": int(keep.sum()),
                "domain": damage_domain_name,
                "structural_relation": damage_structural_relation,
            }
            print(
                f"[step7a] damage layer={name} q={damage_quantile:.2f} threshold={candidate_thr:.3f} "
                f"ref={damage_ref:.3f} candidates={len(rows)} sampled={int(keep.sum())}",
                flush=True,
            )

    accepted_rows = np.concatenate(sampled_rows) if sampled_rows else np.empty(0, dtype=np.int64)
    accepted_samples = np.concatenate(sampled_samples) if sampled_samples else np.empty(0, dtype=np.int64)
    accepted_density = np.concatenate(sampled_density) if sampled_density else np.empty(0, dtype=np.float32)
    accepted_priority = np.concatenate(sampled_priority) if sampled_priority else np.empty(0, dtype=np.float32)
    accepted_layer = np.asarray(sampled_layer)
    accepted_domain = np.asarray(sampled_domain)
    if len(accepted_rows) == 0:
        raise RuntimeError("layered weighted probability sampling produced zero patches")
    pre_spacing_count = int(len(accepted_rows))
    retained = thin_by_3d_spacing(
        accepted_rows,
        accepted_samples,
        accepted_priority,
        row_xy,
        sample_axis,
        min_center_distance_m,
        spacing_time_scale,
    )
    accepted_rows = accepted_rows[retained]
    accepted_samples = accepted_samples[retained]
    accepted_density = accepted_density[retained]
    accepted_priority = accepted_priority[retained]
    accepted_layer = accepted_layer[retained]
    accepted_domain = accepted_domain[retained]
    # --- 构造派生域总量上限（对齐砂砾岩 max_damage_selected_fraction_of_background）---
    damage_cap_audit: dict[str, Any] = {"enabled": bool(damage_enabled)}
    if damage_enabled:
        background_index = np.where(accepted_domain == "background")[0]
        damage_index = np.where(accepted_domain == damage_domain_name)[0]
        fraction_limit = float(damage_cfg.get("max_selected_fraction_of_background", 0.75))
        fraction_cap = int(np.floor(len(background_index) * fraction_limit))
        absolute_cap = damage_cfg.get("max_selected_count")
        cap = min(fraction_cap, int(absolute_cap)) if absolute_cap is not None else fraction_cap
        keep_index = np.arange(len(accepted_rows), dtype=np.int64)
        if len(damage_index) > cap:
            cap_rng = np.random.default_rng(
                int(damage_cfg.get("cap_random_seed", int(damage_cfg.get("random_seed", 20260823)) + 7))
            )
            dropped = cap_rng.choice(damage_index, size=len(damage_index) - cap, replace=False)
            keep_index = np.setdiff1d(keep_index, dropped, assume_unique=False)
        accepted_rows = accepted_rows[keep_index]
        accepted_samples = accepted_samples[keep_index]
        accepted_density = accepted_density[keep_index]
        accepted_priority = accepted_priority[keep_index]
        accepted_layer = accepted_layer[keep_index]
        accepted_domain = accepted_domain[keep_index]
        damage_cap_audit.update(
            {
                "max_selected_fraction_of_background": fraction_limit,
                "max_selected_count": None if absolute_cap is None else int(absolute_cap),
                "cap_applied": cap,
                "background_after_spacing": int(len(background_index)),
                "damage_before_cap": int(len(damage_index)),
                "damage_after_cap": int((accepted_domain == damage_domain_name).sum()),
            }
        )
    post_counts = pd.Series(accepted_layer).value_counts().to_dict()
    for layer, audit in layer_sampling_audit.items():
        audit["retained_after_spacing"] = int(post_counts.get(layer, 0))
    damage_layer_counts = pd.Series(accepted_layer[accepted_domain == damage_domain_name]).value_counts().to_dict()
    for layer, audit in damage_sampling_audit.items():
        audit["retained_after_spacing"] = int(damage_layer_counts.get(layer, 0))
    domain_counts = {
        str(name): int(count)
        for name, count in zip(*np.unique(accepted_domain, return_counts=True))
    } if len(accepted_domain) else {}
    sampling_qc = {
        "status": "pass" if len(accepted_rows) > 0 else "fail",
        "config_path": str(args.config.resolve()),
        "mode": sampling_cfg["mode"],
        "layer_audit": layer_sampling_audit,
        "damage_domain": {
            "enabled": bool(damage_enabled),
            "name": damage_domain_name,
            "domain_code": damage_domain_code,
            "structural_relation": damage_structural_relation,
            "density_sgy": None if damage_density_sgy is None else str(damage_density_sgy),
            "layer_audit": damage_sampling_audit,
            "cap_audit": damage_cap_audit,
            "patch_count_before_cap": int(damage_selected_count),
        },
        "domain_patch_counts": domain_counts,
        "pre_spacing_patch_count": pre_spacing_count,
        "post_spacing_patch_count": int(len(accepted_rows)),
        "spacing_removed_count": int(pre_spacing_count - len(accepted_rows)),
        "min_center_distance_m": min_center_distance_m,
        "spacing_time_scale_m_per_ms": spacing_time_scale,
        "fixed_total_patch_cap_used": False,
    }
    write_json(sampling_qc_path, json_ready(sampling_qc))
    print(
        f"[step7a] sampled_before_spacing={pre_spacing_count} "
        f"retained_after_spacing={len(accepted_rows)} rss_mb={rss_mb():.0f}",
        flush=True,
    )
    if args.sampling_qc_only:
        print(json.dumps(json_ready(sampling_qc), ensure_ascii=False, indent=2))
        return 0 if sampling_qc["status"] == "pass" else 1

    # --- orientation: 按井经验分布重采样 + 局部几何 β 门控 ------------------------
    # 方案：太古界/太古界DFN产状赋值逻辑修正方案_20260923.md（决策 D1–D4）
    #   D1 全部成像点进库；D2 最近几口井高斯加权 + 空间相关抽井；
    #   D3 局部几何只在走向一致时小幅融合倾角；D4 中尺度不动（本步不涉及）。
    imaging_points, orientation_join_qc = load_multiwell_orientation_points(
        Path(config["step3_groups_root"]),
        orient_cfg["imaging_wells_by_layer"],
        Path(config["step2_segments_root"]),
    )
    config["_orientation_join_qc"] = dict(orientation_join_qc)
    well_library = build_well_orientation_library(
        imaging_points, orient_cfg["imaging_wells_by_layer"], orient_cfg
    )
    library_audit = {
        layer: {
            well: {
                "point_count": int(entry["point_count"]),
                "x": float(entry["x"]),
                "y": float(entry["y"]),
                "sample_weight_scale": float(entry["sample_weight_scale"]),
                "in_horizon_fraction": (
                    None if not np.isfinite(entry["in_horizon_fraction"]) else float(entry["in_horizon_fraction"])
                ),
                "dip_azimuth_circular_mean_deg": float(entry["dip_azimuth_circular_mean_deg"]),
                "dip_median_deg": float(entry["dip_median_deg"]),
            }
            for well, entry in entries.items()
        }
        for layer, entries in well_library.items()
    }
    write_json(families_json_path, json_ready({
        "method": "per_well_empirical_dip_azimuth_library",
        "note": "按井保留原始 (倾向方位, 倾角) 点集，不做聚类/平均（方案 D1）",
        "imaging_wells_by_layer": orient_cfg["imaging_wells_by_layer"],
        "point_count": int(len(imaging_points)),
        "well_library": library_audit,
        "well_weight": {
            "kernel": str(orient_cfg.get("well_weight_kernel", "gaussian")),
            "scale_m": float(orient_cfg.get("well_weight_scale_m", 6000.0)),
            "min_share": float(orient_cfg.get("well_weight_min_share", 0.02)),
        },
        "coherent_assignment": {
            "coherence_m": float(orient_cfg.get("well_assignment_coherence_m", 800.0)),
            "seed": int(orient_cfg.get("well_assignment_seed", orient_cfg.get("random_seed", 20260822))),
        },
        "local_geometry_gate": {
            "strike_tolerance_deg": float(orient_cfg.get("local_geometry_strike_tolerance_deg", 30.0)),
            "dip_blend_max": float(orient_cfg.get("local_geometry_dip_blend_max", 0.30)),
            "min_quality": float(orient_cfg.get("local_geometry_min_quality", 0.5)),
        },
    }))
    library_rows: list[dict[str, Any]] = []
    for layer, entries in library_audit.items():
        for well, entry in entries.items():
            library_rows.append({"LayerGroup": layer, "WellName": well, **entry})
    pd.DataFrame(library_rows).to_csv(families_csv_path, index=False, encoding="utf-8-sig")
    missing_library_layers = [
        layer for layer in ("上部复合层", "太古界风化壳") if not well_library.get(layer)
    ]
    if missing_library_layers:
        raise RuntimeError(f"no imaging well orientation library for {missing_library_layers}")

    gradient_quantile = float(orient_cfg["gradient_quantile"])
    gradient_smoothing_sigma = float(orient_cfg["gradient_smoothing_sigma"])
    gradient_time_scale = float(orient_cfg["gradient_time_scale_m_per_ms"])
    coherent_field = CoherentUniformField(
        np.asarray(grid.loc[accepted_rows, "X"], dtype=np.float64),
        np.asarray(grid.loc[accepted_rows, "Y"], dtype=np.float64),
        float(orient_cfg.get("well_assignment_coherence_m", 800.0)),
        int(orient_cfg.get("well_assignment_seed", orient_cfg.get("random_seed", 20260822))),
        float(orient_cfg.get("well_assignment_sigma_cells", 1.0)),
    )
    dx_candidates = grid.groupby("IX")["X"].median().sort_index().diff().abs().dropna()
    dy_candidates = grid.groupby("IY")["Y"].median().sort_index().diff().abs().dropna()
    dx_m = float(dx_candidates[dx_candidates > 0].median())
    dy_m = float(dy_candidates[dy_candidates > 0].median())
    if not np.isfinite(dx_m) or not np.isfinite(dy_m):
        raise RuntimeError("cannot derive physical attribute-grid spacing for local orientation")

    dips: list[float] = []
    dip_azimuths: list[float] = []
    local_dips: list[float] = []
    local_dip_azimuths: list[float] = []
    local_gradient_strengths: list[float] = []
    local_anisotropies: list[float] = []
    local_ridge_strengths: list[float] = []
    local_support_counts: list[int] = []
    orientation_sources: list[str] = []
    selected_family_ids: list[str] = []
    orientation_confidences: list[float] = []
    family_probabilities: list[float] = []
    well_draws: list[str] = []
    well_weight_jsons: list[str] = []
    effective_well_counts: list[float] = []
    local_geometry_used: list[int] = []
    local_geometry_blend_weights: list[float] = []
    local_geometry_strike_deltas: list[float] = []
    orientation_chunk = 5000
    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as handle:
        for chunk_start in range(0, len(accepted_rows), orientation_chunk):
            chunk_end = min(chunk_start + orientation_chunk, len(accepted_rows))
            chunk_rows = accepted_rows[chunk_start:chunk_end].tolist()
            chunk_samples = accepted_samples[chunk_start:chunk_end].tolist()
            chunk_layers = accepted_layer[chunk_start:chunk_end].tolist()
            needed_rows: set[int] = set()
            for row in chunk_rows:
                ix = int(row_to_ix[row])
                iy = int(row_to_iy[row])
                for di in range(-orient_xy_radius, orient_xy_radius + 1):
                    for dj in range(-orient_xy_radius, orient_xy_radius + 1):
                        nrow = cell_to_row.get((ix + di, iy + dj))
                        if nrow is not None:
                            needed_rows.add(nrow)
            dens_by_row: dict[int, np.ndarray] = {
                row: np.asarray(handle.trace[int(grid.loc[row, "OutputTraceIndex"])], dtype=np.float32) for row in needed_rows
            }
            for local_idx, (row, sample, layer) in enumerate(zip(chunk_rows, chunk_samples, chunk_layers)):
                center_density = float(accepted_density[chunk_start + local_idx])
                local = local_gradient_ridge_orientation(
                    row=row, sample=sample, center_density=center_density,
                    cell_to_row=cell_to_row, row_to_ix=row_to_ix, row_to_iy=row_to_iy,
                    dens_by_row=dens_by_row, sample_axis=sample_axis,
                    xy_radius=orient_xy_radius, time_radius=orient_time_span,
                    dx_m=dx_m, dy_m=dy_m, time_scale_m_per_ms=gradient_time_scale,
                    density_fraction=orient_density_fraction, gradient_quantile=gradient_quantile,
                    smoothing_sigma=gradient_smoothing_sigma,
                )
                local_dip = float(local["dip"])
                local_dip_azimuth = float(local["dip_azimuth"])
                # D2：按最近的几口井高斯加权（权重随位置连续变化）
                entry_pool = well_library.get(layer, {})
                if not entry_pool:
                    raise RuntimeError(f"no well orientation library for layer {layer}")
                point_x = float(grid.loc[row, "X"])
                point_y = float(grid.loc[row, "Y"])
                well_names, well_weights = well_distance_weights(point_x, point_y, entry_pool, orient_cfg)
                # D2：用空间相关随机场做分位点抽井 → 相邻片倾向同一口井，保持连续
                quantile = coherent_field.value(point_x, point_y)
                base_dip_azimuth, base_dip, well_draw = sample_statistical_orientation(
                    well_names, well_weights, entry_pool, quantile, orient_cfg, orientation_rng
                )
                # D3：局部几何只在走向一致时小幅融合倾角
                dip_azimuth, dip, used_local, beta, strike_delta = fuse_local_geometry_beta(
                    base_dip_azimuth, base_dip, local, orient_cfg
                )
                if used_local:
                    source = "statistical_well_resample_local_fused"
                else:
                    source = "statistical_well_resample"
                confidence = float(np.clip(np.max(well_weights), 0.0, 1.0))
                well_draws.append(well_draw)
                well_weight_jsons.append(
                    json.dumps({name: round(float(weight), 6) for name, weight in zip(well_names, well_weights)}, ensure_ascii=False)
                )
                effective_well_counts.append(1.0 / max(float(np.sum(np.square(well_weights))), 1e-12))
                local_geometry_used.append(int(used_local))
                local_geometry_blend_weights.append(float(beta))
                local_geometry_strike_deltas.append(
                    float(strike_delta) if np.isfinite(strike_delta) else float("nan")
                )
                selected_family_ids.append(well_draw)
                family_probabilities.append(float(confidence))
                dip_azimuths.append(dip_azimuth)
                dips.append(dip)
                local_dips.append(local_dip)
                local_dip_azimuths.append(local_dip_azimuth)
                local_gradient_strengths.append(float(local["gradient_strength"]))
                local_anisotropies.append(float(local["anisotropy"]))
                local_ridge_strengths.append(float(local["ridge_strength"]))
                local_support_counts.append(int(local["support_count"]))
                orientation_sources.append(source)
                orientation_confidences.append(float(confidence))
            del dens_by_row
            gc.collect()
            print(f"[step7a] orientation chunk {chunk_start}:{chunk_end} of {len(accepted_rows)}", flush=True)

    lengths = orientation_rng.uniform(length_range[0], length_range[1], size=len(accepted_rows))
    heights = orientation_rng.uniform(height_range[0], height_range[1], size=len(accepted_rows))
    areas = lengths * heights * z_scale
    patches = pd.DataFrame(
        {
            "PatchID": [f"taigu_small_{i:07d}" for i in range(len(accepted_rows))],
            "TraceIdx": [int(grid.loc[r, "TraceIdx"]) for r in accepted_rows],
            "X": [float(grid.loc[r, "X"]) for r in accepted_rows],
            "Y": [float(grid.loc[r, "Y"]) for r in accepted_rows],
            "TIME": [float(sample_axis[s]) for s in accepted_samples],
            "LayerGroup": accepted_layer,
            "Density": accepted_density,
            "DipDeg": dips,
            # 2026-09-22 统一口径：唯一真值字段是倾向方位（0–360）。
            "DipAzimuthDeg": dip_azimuths,
            "AzimuthDeg": [orientation.strike_from_dip_azimuth(value) for value in dip_azimuths],
            "LocalGradientDipDeg": local_dips,
            "LocalGradientDipAzimuthDeg": local_dip_azimuths,
            "LocalGradientAzimuthDeg": [
                orientation.strike_from_dip_azimuth(value) for value in local_dip_azimuths
            ],
            # Compatibility aliases for consumers of the previous Step7A
            # schema. They now expose gradient/ridge estimates, not PCA fits.
            "LocalPcaDipDeg": local_dips,
            "LocalPcaDipAzimuthDeg": local_dip_azimuths,
            "LocalPcaAzimuthDeg": [
                orientation.strike_from_dip_azimuth(value) for value in local_dip_azimuths
            ],
            "LocalGradientStrength": local_gradient_strengths,
            "LocalAnisotropy": local_anisotropies,
            "LocalRidgeStrength": local_ridge_strengths,
            "LocalSupportCount": local_support_counts,
            "OrientationSource": orientation_sources,
            "OrientationBaseSource": orientation_sources,
            "OrientationFamily": selected_family_ids,
            "OrientationConfidence": orientation_confidences,
            "OrientationFamilyProbability": family_probabilities,
            # 方案 20260923 的审计列（D2/D3 的可追溯性）
            "OrientationWellDraw": well_draws,
            "OrientationWellWeights": well_weight_jsons,
            "OrientationEffectiveWellCount": effective_well_counts,
            "LocalGeometryUsed": local_geometry_used,
            "LocalGeometryBlendWeight": local_geometry_blend_weights,
            "LocalGeometryStrikeDeltaDeg": local_geometry_strike_deltas,
            "PatchLengthM": lengths,
            "PatchHeightMs": heights,
            "PatchAreaM2": areas,
            "FractureScale": "small",
            "SmallDomain": accepted_domain,
            "SmallDomainCode": np.where(
                accepted_domain == "background", 1, damage_domain_code
            ).astype(np.int32),
            "StructuralRelation": np.where(
                accepted_domain == "background", "small_background", damage_structural_relation
            ),
            "SamplingPriority": accepted_priority,
            "WindowCode": 1,
        }
    )
    patches.to_csv(patches_csv, index=False, encoding="utf-8-sig")

    points: list[np.ndarray] = []
    quads: list[np.ndarray] = []
    for i, patch in patches.iterrows():
        center = np.array([patch["X"], patch["Y"], patch["TIME"] * vtk_z_scale])
        # 2026-09-22 统一口径：由**真倾向方位**构造，长边 = 走向线，短边沿下倾方向。
        # 帧内 z = TIME * vtk_z_scale（向下为正），故下倾方向 z 分量为 +sin(dip)。
        dip_azimuth_rad = np.radians(float(patch["DipAzimuthDeg"]))
        cos_az, sin_az = np.cos(dip_azimuth_rad), np.sin(dip_azimuth_rad)
        strike = np.array([cos_az, -sin_az, 0.0])
        dip_dir = np.array([sin_az, cos_az, 0.0])
        dip_rad = np.radians(patch["DipDeg"])
        dip_vec = np.array([np.sin(dip_rad) * dip_dir[0], np.sin(dip_rad) * dip_dir[1], np.cos(dip_rad)])
        half_l = patch["PatchLengthM"] / 2.0
        half_h = patch["PatchHeightMs"] * vtk_z_scale / 2.0
        corners = [
            center + half_l * strike + half_h * dip_vec,
            center + half_l * strike - half_h * dip_vec,
            center - half_l * strike - half_h * dip_vec,
            center - half_l * strike + half_h * dip_vec,
        ]
        base = len(points)
        points.extend(corners)
        quads.append(np.array([base, base + 1, base + 2, base + 3], dtype=np.int64))
    cell_data = {
        "PatchID": np.arange(len(patches), dtype=np.int64),
        "Density": patches["Density"].to_numpy(dtype=np.float64),
        "DipDeg": patches["DipDeg"].to_numpy(dtype=np.float64),
        "DipAzimuthDeg": patches["DipAzimuthDeg"].to_numpy(dtype=np.float64),
        "AzimuthDeg": patches["AzimuthDeg"].to_numpy(dtype=np.float64),
        "PatchAreaM2": patches["PatchAreaM2"].to_numpy(dtype=np.float64),
        "PatchLengthM": patches["PatchLengthM"].to_numpy(dtype=np.float64),
        "PatchHeightMs": patches["PatchHeightMs"].to_numpy(dtype=np.float64),
        "CenterTimeMs": patches["TIME"].to_numpy(dtype=np.float64),
        "LayerCode": patches["LayerGroup"].map({"上部复合层": 1, "太古界风化壳": 2}).to_numpy(dtype=np.int32),
        "FractureScale": np.full(len(patches), 1, dtype=np.int32),
        "WindowCode": np.full(len(patches), 1, dtype=np.int32),
        "OrientationSourceCode": patches["OrientationSource"].map({
            "statistical_well_resample": 1,
            "statistical_well_resample_local_fused": 2,
        }).fillna(0).to_numpy(dtype=np.int32),
        "OrientationFamilyCode": pd.factorize(patches["OrientationFamily"], sort=True)[0].astype(np.int32) + 1,
        "OrientationConfidence": patches["OrientationConfidence"].to_numpy(dtype=np.float64),
        "LocalGradientStrength": patches["LocalGradientStrength"].to_numpy(dtype=np.float64),
        "LocalAnisotropy": patches["LocalAnisotropy"].to_numpy(dtype=np.float64),
        "LocalRidgeStrength": patches["LocalRidgeStrength"].to_numpy(dtype=np.float64),
        "SmallDomainCode": patches["SmallDomainCode"].to_numpy(dtype=np.int32),
    }
    write_legacy_vtk(vtk_path, np.stack(points) if points else np.empty((0, 3)), np.stack(quads) if quads else np.empty((0, 4), dtype=np.int64), cell_data)

    # --- imaging match QC ---
    qc_rows: list[dict[str, Any]] = []
    profile_well = str(config["profile_well"])
    profile = load_imaging_density_profile(config, profile_well)
    if len(profile):
        corridor = patches[
            (patches["X"] >= profile["X"].min() - 500)
            & (patches["X"] <= profile["X"].max() + 500)
            & (patches["Y"] >= profile["Y"].min() - 500)
            & (patches["Y"] <= profile["Y"].max() + 500)
        ]
        bins = np.arange(profile["TIME"].min(), profile["TIME"].max() + 20, 20)
        imaging_means = []
        patch_counts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            imaging_means.append(float(profile[(profile["TIME"] >= lo) & (profile["TIME"] < hi)]["Density"].mean()))
            patch_counts.append(int(((corridor["TIME"] >= lo) & (corridor["TIME"] < hi)).sum()))
        from scipy.stats import spearmanr
        corr = spearmanr(imaging_means, patch_counts) if len(imaging_means) >= 3 else None
        qc_rows.append(
            {
                "WellName": profile_well,
                "QCKind": "well_corridor_spatial",
                "CorridorHalfWidthM": 500.0,
                "TimeBinMs": 20.0,
                "ImagingTimeRangeMs": [float(profile["TIME"].min()), float(profile["TIME"].max())],
                "PatchCountInCorridor": int(len(corridor)),
                "SpearmanDensityVsPatches": float(corr.statistic) if corr is not None else None,
                "Note": "co-located amplitude available",
            }
        )
    strong_wells = ["埕北310", "埕北313", "埕北816", "桩斜169", "桩海102"]
    for well in strong_wells:
        img = load_imaging_density_profile(config, well)
        if img.empty:
            continue
        for strata, sub in img.groupby("StrataName"):
            dens = sub["Density"].to_numpy(dtype=np.float64)
            dfn_dens = patches.loc[patches["LayerGroup"] == strata, "Density"].to_numpy(dtype=np.float64)
            qc_rows.append(
                {
                    "WellName": well,
                    "QCKind": "distributional_reference",
                    "CorridorHalfWidthM": None,
                    "TimeBinMs": None,
                    "StrataName": str(strata),
                    "ImagingDensityP50": float(np.median(dens)),
                    "ImagingDensityP90": float(np.quantile(dens, 0.9)),
                    "DfnDensityP50": float(np.median(dfn_dens)) if len(dfn_dens) else None,
                    "DfnDensityP90": float(np.quantile(dfn_dens, 0.9)) if len(dfn_dens) else None,
                    "Note": "non co-located, distributional reference only",
                }
            )
    pd.DataFrame(qc_rows).to_csv(qc_csv, index=False, encoding="utf-8-sig")

    layer_counts = patches["LayerGroup"].value_counts().to_dict()
    upper_fraction = float(layer_counts.get("上部复合层", 0) / max(len(patches), 1))
    spacing_points = np.column_stack(
        [
            patches["X"].to_numpy(dtype=np.float64),
            patches["Y"].to_numpy(dtype=np.float64),
            patches["TIME"].to_numpy(dtype=np.float64) * spacing_time_scale,
        ]
    )
    if len(spacing_points) > 1:
        nearest_distance = cKDTree(spacing_points).query(spacing_points, k=2, workers=-1)[0][:, 1]
    else:
        nearest_distance = np.asarray([np.nan])
    cell_counts = (
        patches.assign(CellI=np.floor(patches["X"] / 300.0), CellJ=np.floor(patches["Y"] / 300.0))
        .groupby(["CellI", "CellJ"])
        .size()
        .to_numpy(dtype=np.int64)
    )
    checks = {
        "patch_count_positive": len(patches) > 0,
        "patch_count_within_cap": safety_max_patches is None or len(patches) <= safety_max_patches,
        "both_layers_represented": upper_fraction > 0.05 and float(layer_counts.get("太古界风化壳", 0)) > 0,
        "damage_domain_within_configured_cap": bool(
            (not damage_enabled)
            or int((patches["SmallDomain"] == damage_domain_name).sum())
            <= int(np.floor(int((patches["SmallDomain"] == "background").sum())
                            * float(damage_cfg.get("max_selected_fraction_of_background", 0.75))))
        ),
        "background_domain_is_majority": bool(
            (not damage_enabled)
            or int((patches["SmallDomain"] == "background").sum())
            > int((patches["SmallDomain"] == damage_domain_name).sum())
        ),
        "all_patches_main_window": bool((patches["WindowCode"] == 1).all()),
        "orientation_ranges_valid": bool(
            patches["DipDeg"].between(0, 90).all()
            and patches["DipAzimuthDeg"].between(0, 360).all()
            and patches["AzimuthDeg"].between(0, 180).all()
        ),
        "azimuth_is_derived_strike": bool(
            np.allclose(
                patches["AzimuthDeg"].to_numpy(dtype=np.float64),
                (patches["DipAzimuthDeg"].to_numpy(dtype=np.float64) - 90.0) % 180.0,
                atol=1e-9,
            )
        ),
        "dip_min_constraint": bool((patches["DipDeg"] >= dip_floor - 1.0e-6).all() and (patches["DipDeg"] <= dip_ceil + 1.0e-6).all()),
        "geometry_finite": bool(np.isfinite(patches[["X", "Y", "TIME", "PatchAreaM2"]]).all().all()),
        "imaging_match_qc_exists": qc_csv.exists(),
        "orientation_point_join_has_no_unmatched": int(
            dict(config.get("_orientation_join_qc", {})).get("unmatched_count", 0)
        )
        == 0,
        "vtk_exists": vtk_path.exists(),
        "minimum_3d_spacing_respected": bool(
            min_center_distance_m <= 0.0
            or len(patches) <= 1
            or float(np.nanmin(nearest_distance)) >= min_center_distance_m - 1.0e-6
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "segment_join_qc": {
            "orientation_points": dict(config.get("_orientation_join_qc", {})),
            "imaging_density_profile": dict(config.get("_density_profile_join_qc", {})),
        },
        "output_paths": {
            "patches_csv": str(patches_csv),
            "patches_vtk": str(vtk_path),
            "imaging_match_qc_csv": str(qc_csv),
            "sampling_qc_json": str(sampling_qc_path),
            "orientation_well_library_json": str(families_json_path),
            "orientation_well_library_csv": str(families_csv_path),
            "summary_json": str(summary_path),
        },
        "step6a_input": {
            "prediction_summary_json": str(prediction_summary_path),
            "background_density_sgy": str(density_sgy),
            "damage_density_sgy": None if damage_density_sgy is None else str(damage_density_sgy),
            "model_contract_version": prediction_summary.get("model_contract_version"),
            "attribute_normalization_contract_version": prediction_summary.get("attribute_normalization_contract_version"),
        },
        "sampling": {
            "mode": sampling_cfg["mode"],
            "reference_quantiles": layer_refs,
            "layer_audit": layer_sampling_audit,
            "damage_domain": {
                "enabled": bool(damage_enabled),
                "name": damage_domain_name,
                "domain_code": damage_domain_code,
                "structural_relation": damage_structural_relation,
                "layer_audit": damage_sampling_audit,
                "cap_audit": damage_cap_audit,
            },
            "domain_patch_counts": {
                str(name): int(count) for name, count in patches["SmallDomain"].value_counts().items()
            },
            "layer_patch_counts": layer_counts,
            "upper_layer_fraction": upper_fraction,
            "pre_spacing_patch_count": pre_spacing_count,
            "post_spacing_patch_count": int(len(patches)),
            "min_center_distance_m": min_center_distance_m,
            "spacing_time_scale_m_per_ms": spacing_time_scale,
            "nearest_neighbor_distance_m": {
                "min": float(np.nanmin(nearest_distance)),
                "median": float(np.nanmedian(nearest_distance)),
                "p90": float(np.nanquantile(nearest_distance, 0.90)),
            },
            "patches_per_300m_cell": {
                "cell_count": int(len(cell_counts)),
                "mean": float(np.mean(cell_counts)),
                "median": float(np.median(cell_counts)),
                "p90": float(np.quantile(cell_counts, 0.90)),
                "max": int(np.max(cell_counts)),
            },
        },
        "orientation": {
            "method": "per_well_empirical_resample_plus_beta_gated_local_geometry",
            "imaging_point_count": int(len(imaging_points)),
            "well_library": library_audit,
            "well_weights_config": {
                "scale_m": float(orient_cfg.get("well_weight_scale_m", 6000.0)),
                "min_share": float(orient_cfg.get("well_weight_min_share", 0.02)),
                "coherence_m": float(orient_cfg.get("well_assignment_coherence_m", 800.0)),
                "strike_tolerance_deg": float(orient_cfg.get("local_geometry_strike_tolerance_deg", 30.0)),
                "dip_blend_max": float(orient_cfg.get("local_geometry_dip_blend_max", 0.30)),
            },
            "well_draw_counts": patches["OrientationWellDraw"].value_counts().to_dict(),
            "local_geometry_used_fraction": float(np.mean(patches["LocalGeometryUsed"])),
            "effective_well_count_stats": {
                "min": float(np.nanmin(patches["OrientationEffectiveWellCount"])),
                "median": float(np.nanmedian(patches["OrientationEffectiveWellCount"])),
                "max": float(np.nanmax(patches["OrientationEffectiveWellCount"])),
            },
            "source_counts": patches["OrientationSource"].value_counts().to_dict(),
            "family_counts": patches["OrientationFamily"].value_counts().to_dict(),
            "dip_stats": {"min": float(patches["DipDeg"].min()), "max": float(patches["DipDeg"].max()), "mean": float(patches["DipDeg"].mean())},
            "dip_azimuth_stats": {
                "min": float(patches["DipAzimuthDeg"].min()),
                "max": float(patches["DipAzimuthDeg"].max()),
                "circular_mean": orientation.circular_mean_dip_azimuth(patches["DipAzimuthDeg"]),
            },
            "azimuth_semantics": "AzimuthDeg = (DipAzimuthDeg - 90) % 180 （派生走向，兼容字段）",
            "azimuth_stats": {"min": float(patches["AzimuthDeg"].min()), "max": float(patches["AzimuthDeg"].max()), "mean": float(patches["AzimuthDeg"].mean())},
        },
        "accepted_patch_count": int(len(patches)),
        "imaging_match_qc_rows": int(len(qc_rows)),
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
