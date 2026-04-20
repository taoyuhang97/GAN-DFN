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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

THIS_DIR = Path(__file__).resolve().parent
OPT_STAGE_DIR = THIS_DIR.parent
BASELINE_DIR = OPT_STAGE_DIR / "G_DFN监督基线"

for candidate in (THIS_DIR, BASELINE_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_csv_utf8, write_json
from merge_unit_dfn_vtks import read_legacy_vtk_polygons, write_legacy_vtk_polygons


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


# ---------------------------------------------------------------------------
#   Step 1: 裂缝组识别 (候选, 非硬分类)
# ---------------------------------------------------------------------------

def fracture_set_clustering(
    df: pd.DataFrame,
    max_sets: int = 6,
    min_sets: int = 2,
    n_sets: int | None = None,
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

    if n_sets is None:
        best_bic, best_k = np.inf, min_sets
        for k in range(min_sets, max_sets + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  n_init=3, random_state=42, max_iter=200)
            gmm.fit(features)
            bic = gmm.bic(features)
            if bic < best_bic:
                best_bic, best_k = bic, k
        n_sets = best_k

    gmm = GaussianMixture(n_components=n_sets, covariance_type="full",
                          n_init=5, random_state=42, max_iter=300)
    gmm.fit(features)
    labels = gmm.predict(features)
    probs = gmm.predict_proba(features)
    max_probs = probs[np.arange(len(labels)), labels]

    df["FractureSet"] = labels.astype(int)
    df["SetProbability"] = max_probs.astype(float)
    return df


# ---------------------------------------------------------------------------
#   Step 2: 组内局部产状平滑 (仅写参考字段, 不覆盖原值)
# ---------------------------------------------------------------------------

def regional_orientation_smoothing(
    df: pd.DataFrame,
    bandwidth_xy: float = 150.0,
    bandwidth_z: float = 20.0,
    blend_alpha: float = 0.4,
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

    for set_id in df["FractureSet"].unique():
        mask = df["FractureSet"].values == set_id
        idx = np.where(mask)[0]
        if len(idx) < 3:
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
            kernel[i] = 0.0  # leave-one-out
            w_sum = kernel.sum()
            if w_sum < 1e-12:
                smoothed_az[i] = az_vals[i]
                smoothed_dip[i] = dip_vals[i]
                continue

            mean_cos = np.dot(kernel, az_cos) / w_sum
            mean_sin = np.dot(kernel, az_sin) / w_sum
            trend_az = (np.rad2deg(np.arctan2(mean_sin, mean_cos)) / 2.0) % 180.0
            trend_dip = np.dot(kernel, dip_vals) / w_sum

            # 混合: 差异过大则不平滑 (可能是局部真实异常)
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
        # 注意: 不覆盖 Azimuth / Dip

    return df


# ---------------------------------------------------------------------------
#   Step 3: 裂缝走廊检测 (作为证据, 非连接许可)
# ---------------------------------------------------------------------------

def fracture_corridor_detection(
    df: pd.DataFrame,
    corridor_search_radius: float = 200.0,
    corridor_min_patches: int = 5,
    along_strike_weight: float = 2.0,
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
    for set_id in df["FractureSet"].unique():
        mask = df["FractureSet"].values == set_id
        idx = np.where(mask)[0]
        if len(idx) < corridor_min_patches:
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

    # CorridorSupport 作为证据字段 (bool → int)
    df["CorridorSupport"] = (df["CorridorID"] >= 0).astype(int)
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
    tree = cKDTree(coords_scaled)
    neighbor_counts = np.array(tree.query_ball_point(coords_scaled, r=neighbor_radius_xy, return_length=True)) - 1
    max_count = max(int(neighbor_counts.max()), 1)
    df["NeighborCount"] = neighbor_counts.astype(int)
    df["GeophysicsScore"] = np.clip(neighbor_counts / max_count, 0.0, 1.0).astype(float)

    # 3) CorridorSupport: 已在 Step 3 中设置, 确保存在
    if "CorridorSupport" not in df.columns:
        df["CorridorSupport"] = 0

    # 4) StratigraphyScore: 层位一致性 (预留, 默认全 1.0)
    if "StratigraphyScore" not in df.columns:
        if known_fracture_layers and "GeoIntervalKey" in df.columns:
            df["StratigraphyScore"] = df["GeoIntervalKey"].isin(known_fracture_layers).astype(float)
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
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """在单元边界找到产状匹配的裂缝对, 标记 ConnectionType 但不插值补片。

    匹配条件: 同组 FractureSet + 同层 GeoIntervalKey + 产状相近 + 几何连续。
    返回 (df, matched_pairs) — matched_pairs 供 Phase 2 的补接使用。
    """
    df = df.copy()
    matched_pairs: list[dict[str, Any]] = []

    if "UnitID" not in df.columns and "BlockX" not in df.columns:
        return df, matched_pairs

    if "BlockX" in df.columns and "BlockY" in df.columns:
        unit_groups = df.groupby(["BlockX", "BlockY"])
    elif "UnitID" in df.columns:
        unit_groups = df.groupby("UnitID")
    else:
        return df, matched_pairs

    # 收集边界裂缝
    boundary_info: list[tuple[str, int, pd.Series]] = []
    for unit_key, group in unit_groups:
        x_min, x_max = group["CenterX"].min(), group["CenterX"].max()
        y_min, y_max = group["CenterY"].min(), group["CenterY"].max()
        for df_idx, row in group.iterrows():
            is_boundary = (
                abs(row["CenterX"] - x_min) < boundary_tol_xy
                or abs(row["CenterX"] - x_max) < boundary_tol_xy
                or abs(row["CenterY"] - y_min) < boundary_tol_xy
                or abs(row["CenterY"] - y_max) < boundary_tol_xy
            )
            if is_boundary:
                boundary_info.append((str(unit_key), df_idx, row))

    if len(boundary_info) < 2:
        return df, matched_pairs

    coords = np.array([[r["CenterX"], r["CenterY"]] for _, _, r in boundary_info])
    tree = cKDTree(coords)
    paired_set: set[tuple[int, int]] = set()

    for i, (uid_i, idx_i, row_i) in enumerate(boundary_info):
        candidates = tree.query_ball_point([row_i["CenterX"], row_i["CenterY"]], r=alignment_tol_m)
        for j in candidates:
            if j <= i:
                continue
            uid_j, idx_j, row_j = boundary_info[j]
            if uid_i == uid_j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in paired_set:
                continue

            # 必须同组
            if "FractureSet" in df.columns:
                if row_i.get("FractureSet", -1) != row_j.get("FractureSet", -2):
                    continue

            # 层位一致
            if "GeoIntervalKey" in df.columns:
                if row_i.get("GeoIntervalKey", "") != row_j.get("GeoIntervalKey", ""):
                    continue

            # 产状匹配
            az_d = _azimuth_diff(np.array([row_i["Azimuth"]]), np.array([row_j["Azimuth"]]))[0]
            dip_d = abs(float(row_i["Dip"]) - float(row_j["Dip"]))
            if az_d > azimuth_tol_deg or dip_d > dip_tol_deg:
                continue

            # 几何连续: 沿走向对齐
            mean_az_rad = np.deg2rad((float(row_i["Azimuth"]) + float(row_j["Azimuth"])) / 2.0)
            dx = float(row_j["CenterX"]) - float(row_i["CenterX"])
            dy = float(row_j["CenterY"]) - float(row_i["CenterY"])
            perp_dist = abs(-dx * np.sin(mean_az_rad) + dy * np.cos(mean_az_rad))
            if perp_dist > alignment_tol_m * 0.5:
                continue

            paired_set.add(pair_key)

            # 标记为边界匹配
            df.loc[idx_i, "ConnectionType"] = "boundary_matched"
            df.loc[idx_j, "ConnectionType"] = "boundary_matched"

            gap_dist = np.sqrt(dx ** 2 + dy ** 2)
            matched_pairs.append({
                "idx_i": idx_i, "idx_j": idx_j,
                "unit_i": uid_i, "unit_j": uid_j,
                "azimuth_diff": float(az_d), "dip_diff": float(dip_d),
                "gap_distance_m": float(gap_dist),
                "perp_distance_m": float(perp_dist),
                "confidence_i": float(row_i.get("Confidence", 0.5)),
                "confidence_j": float(row_j.get("Confidence", 0.5)),
            })

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
) -> pd.DataFrame:
    """将同组、空间相邻、产状一致的小片聚合为更大的裂缝面。

    地球物理依据: 模型在 12.5m 体素级别独立预测裂缝, 导致一条连续
    裂缝被分解为多个 ~8m 的碎片。聚合恢复了裂缝的地质尺度。

    方法:
    1. 在同一 FractureSet 内, 用 DBSCAN 按空间邻近+产状一致性聚类
    2. 每个聚类替换为一个大矩形片:
       - 中心 = 聚类质心 (confidence 加权)
       - 产状 = 聚类内加权平均 Azimuth/Dip
       - 尺寸 = 沿走向/垂直走向方向的聚类范围
    3. 孤立片 / 低置信片保留原样
    """
    df = df.copy()

    # 先标记: 所有片默认保留
    keep_mask = np.ones(len(df), dtype=bool)
    new_patches: list[dict[str, Any]] = []

    for set_id in df["FractureSet"].unique():
        set_mask = df["FractureSet"].values == set_id
        idx = np.where(set_mask)[0]
        if len(idx) < cluster_min_patches:
            continue

        cx = df["CenterX"].values[idx].astype(float)
        cy = df["CenterY"].values[idx].astype(float)
        cz = df["CenterTIME"].values[idx].astype(float)
        az = df["Azimuth"].values[idx].astype(float)
        dip = df["Dip"].values[idx].astype(float)
        conf = df["Confidence"].values[idx].astype(float) if "Confidence" in df.columns else np.ones(len(idx))

        # 用产状+空间做聚类特征
        # 归一化: XY 按 cluster_radius, Azimuth 按 azimuth_tol, Dip 按 dip_tol
        az_rad = np.deg2rad(2.0 * az)
        features = np.column_stack([
            cx / cluster_radius,
            cy / cluster_radius,
            cz / cluster_radius,  # 深度也参与 (尺度与XY一致)
            np.cos(az_rad) * 2.0,  # 方位角嵌入, 权重放大
            np.sin(az_rad) * 2.0,
            dip / dip_tol_deg * 0.5,
        ])

        clustering = DBSCAN(
            eps=2.0,  # 归一化后的邻接距离阈值
            min_samples=cluster_min_patches,
        ).fit(features)
        labels = clustering.labels_

        for c_label in set(labels):
            if c_label == -1:
                continue  # 噪声点保留原样

            c_idx = idx[labels == c_label]
            n_in_cluster = len(c_idx)
            if n_in_cluster < cluster_min_patches:
                continue

            # 聚类内属性
            c_cx = df["CenterX"].values[c_idx].astype(float)
            c_cy = df["CenterY"].values[c_idx].astype(float)
            c_cz = df["CenterTIME"].values[c_idx].astype(float)
            c_az = df["Azimuth"].values[c_idx].astype(float)
            c_dip = df["Dip"].values[c_idx].astype(float)
            c_conf = conf[labels == c_label]

            # 产状一致性检查: 如果聚类内方差太大, 跳过
            az_spread = _azimuth_diff(c_az, np.full_like(c_az,
                _azimuth_mean_weighted(c_az, c_conf)))
            if np.mean(az_spread) > azimuth_tol_deg:
                continue

            # 加权质心
            w = c_conf / c_conf.sum()
            mean_x = np.dot(w, c_cx)
            mean_y = np.dot(w, c_cy)
            mean_z = np.dot(w, c_cz)
            mean_az = _azimuth_mean_weighted(c_az, c_conf)
            mean_dip = float(np.dot(w, c_dip))
            mean_conf = float(c_conf.mean())

            # 沿走向/垂直走向的投影范围 → 新片尺寸
            mean_az_rad = np.deg2rad(mean_az)
            s_x, s_y = np.cos(mean_az_rad), np.sin(mean_az_rad)
            s_proj = c_cx * s_x + c_cy * s_y
            p_proj = -c_cx * np.sin(mean_az_rad) + c_cy * np.cos(mean_az_rad)
            z_range = c_cz.max() - c_cz.min()

            # 新长度 = 沿走向范围 + 两端各加半个原始片长, 受上限约束
            orig_mean_len = float(df["PatchLength"].values[c_idx].astype(float).mean())
            orig_mean_hgt = float(df["PatchHeight"].values[c_idx].astype(float).mean())
            raw_length = max(s_proj.max() - s_proj.min() + orig_mean_len, orig_mean_len)
            raw_height = max(z_range + orig_mean_hgt, orig_mean_hgt)
            new_length = min(raw_length, max_merged_length)
            new_height = min(raw_height, max_merged_height)

            # 构造法向量 → u_vec (沿走向), v_vec (沿倾向/深度方向)
            dip_rad = np.deg2rad(mean_dip)
            # 水平面沿走向
            u_hat = np.array([s_x, s_y, 0.0])
            # 倾向方向 (在走向法平面内, 考虑倾角)
            perp_x, perp_y = -np.sin(mean_az_rad), np.cos(mean_az_rad)
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

            center = np.array([mean_x, mean_y, mean_z])
            half_u = new_length / 2.0
            half_v = new_height / 2.0

            nv1 = center - half_u * u_hat - half_v * v_hat
            nv2 = center + half_u * u_hat - half_v * v_hat
            nv3 = center + half_u * u_hat + half_v * v_hat
            nv4 = center - half_u * u_hat + half_v * v_hat

            # 从聚类中最常见的行继承非几何属性
            ref_row = df.iloc[c_idx[np.argmax(c_conf)]].to_dict()

            new_patch: dict[str, Any] = {}
            for col in df.columns:
                new_patch[col] = ref_row.get(col, 0)

            new_patch.update({
                "CenterX": mean_x, "CenterY": mean_y, "CenterTIME": mean_z,
                "Azimuth": mean_az, "Dip": mean_dip,
                "OrigAzimuth": mean_az, "OrigDip": mean_dip,
                "SmoothedAzimuth": mean_az, "SmoothedDip": mean_dip,
                "PatchLength": new_length, "PatchHeight": new_height,
                "Confidence": mean_conf,
                "FractureSet": int(set_id),
                "V1X": nv1[0], "V1Y": nv1[1], "V1Z": nv1[2],
                "V2X": nv2[0], "V2Y": nv2[1], "V2Z": nv2[2],
                "V3X": nv3[0], "V3Y": nv3[1], "V3Z": nv3[2],
                "V4X": nv4[0], "V4Y": nv4[1], "V4Z": nv4[2],
                "NeighborCount": n_in_cluster,
                "IsSupplemented": 0,
                "ConnectionType": "original",
            })
            new_patches.append(new_patch)

            # 标记原始片为删除
            keep_mask[c_idx] = False

    # 保留未聚合的片 + 新聚合片
    df_kept = df.loc[keep_mask].copy()
    if new_patches:
        df_new = pd.DataFrame(new_patches)
        # 确保列顺序一致
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
    for set_id in df["FractureSet"].unique():
        mask = df["FractureSet"].values == set_id
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue

        az_vals = df["Azimuth"].values[idx].astype(float)
        mean_az = _azimuth_mean_weighted(az_vals, np.ones(len(az_vals)))
        mean_az_rad = np.deg2rad(mean_az)
        strike_x, strike_y = np.cos(mean_az_rad), np.sin(mean_az_rad)

        cx = df["CenterX"].values[idx].astype(float)
        cy = df["CenterY"].values[idx].astype(float)

        # 沿走向 / 垂直走向投影
        s_proj = cx * strike_x + cy * strike_y
        p_proj = -cx * np.sin(mean_az_rad) + cy * np.cos(mean_az_rad)

        # 用 KDTree 在组内找邻居
        coords_sp = np.column_stack([s_proj, p_proj])
        tree = cKDTree(coords_sp)

        for k in range(len(idx)):
            i = idx[k]
            orig_length = float(df.loc[df.index[i], "PatchLength"])
            half_len_i = orig_length / 2.0

            # 搜索同组邻居 (沿走向+垂直走向)
            nbs = tree.query_ball_point(coords_sp[k], r=max_neighbor_dist)
            if len(nbs) <= 1:
                continue

            # 找沿走向最近的邻居 (排除自身, 垂直走向距离需小于合理阈值)
            best_along_gap = float('inf')
            for nb in nbs:
                if nb == k:
                    continue
                ds = abs(s_proj[nb] - s_proj[k])
                dp = abs(p_proj[nb] - p_proj[k])
                # 垂直走向距离不能太大 (否则不是沿走向邻居)
                if dp > max_neighbor_dist * 0.4:
                    continue
                # 计算真实间隙
                half_len_nb = float(df.loc[df.index[idx[nb]], "PatchLength"]) / 2.0
                gap = ds - half_len_i - half_len_nb
                if gap > 0 and gap < best_along_gap:
                    best_along_gap = gap

            if best_along_gap == float('inf') or best_along_gap <= 0:
                continue

            # 拉伸量
            stretch = best_along_gap * gap_fill_fraction
            new_length = orig_length + stretch
            new_length = min(new_length, orig_length * max_stretch_factor)

            if new_length <= orig_length * 1.05:
                continue

            # 从顶点恢复 u/v 方向
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
    azimuth_tol_deg: float = 30.0,
    dip_tol_deg: float = 20.0,
    min_score_threshold: float = 0.35,
    corridor_bonus: float = 0.15,
    require_corridor: bool = False,
    perp_ratio: float = 0.8,
    enable_supplement: bool = True,
    max_stretch_ratio: float = 2.5,
    supplement_confidence_decay: float = 0.7,
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

            # 条件 1: 同 FractureSet
            if int(row_i.get("FractureSet", -1)) != int(row_j.get("FractureSet", -2)):
                continue

            # 条件 2: 同层 (如有)
            if "GeoIntervalKey" in df.columns:
                if str(row_i.get("GeoIntervalKey", "")) != str(row_j.get("GeoIntervalKey", "")):
                    continue

            # 条件 3: 产状匹配 (放宽: 聚合后产状是统计平均, 允许更大偏差)
            az_d = _azimuth_diff(
                np.array([float(row_i["Azimuth"])]),
                np.array([float(row_j["Azimuth"])])
            )[0]
            dip_d = abs(float(row_i["Dip"]) - float(row_j["Dip"]))
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
                (float(row_i["Azimuth"]) + float(row_j["Azimuth"])) / 2.0
            )
            dx = float(row_j["CenterX"]) - float(row_i["CenterX"])
            dy = float(row_j["CenterY"]) - float(row_i["CenterY"])
            perp_dist = abs(-dx * np.sin(mean_az_rad) + dy * np.cos(mean_az_rad))
            gap_dist = np.sqrt(dx ** 2 + dy ** 2)

            if perp_dist > gap_dist * perp_ratio + 1e-6:
                continue

            paired.add(pair_key)
            connections.append({
                "i": i_global, "j": j_global,
                "gap": gap_dist, "perp": perp_dist,
                "az_diff": az_d, "dip_diff": dip_d,
                "score": pair_score, "has_corridor": has_corridor,
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
                        sp.update({
                            "CenterX": mid_x, "CenterY": mid_y, "CenterTIME": mid_z,
                            "Azimuth": mean_az, "Dip": mean_dip,
                            "PatchLength": bridge_len, "PatchHeight": bridge_hgt,
                            "Confidence": bridge_conf,
                            "ConnectionType": "supplemented",
                            "ConnectionID": conn_id,
                            "IsSupplemented": 1,
                            "ReliabilityLevel": "medium",
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """第一轮: 裂缝组识别 + 可靠性评分 + 边界匹配 (不补片)。

    目的: 证明"后处理提升了区域连贯性, 但没有明显造假连接"。
    """
    stats: dict[str, Any] = {"phase": 1, "input_count": len(df)}

    # Step 1: 裂缝组识别
    df = fracture_set_clustering(df, max_sets=max_fracture_sets, n_sets=n_fracture_sets)
    n_sets_found = int(df["FractureSet"].nunique())
    set_sizes = df.groupby("FractureSet").size().to_dict()
    stats["fracture_sets"] = n_sets_found
    stats["set_sizes"] = {str(k): int(v) for k, v in set_sizes.items()}

    # OrigAzimuth/OrigDip 必须在 Phase 1 就保留
    df["OrigAzimuth"] = df["Azimuth"].values.copy()
    df["OrigDip"] = df["Dip"].values.copy()

    # Step 4: 多维度可靠性评分
    df = multi_dimensional_reliability_scoring(
        df,
        neighbor_radius_xy=neighbor_radius_xy,
        neighbor_radius_z=neighbor_radius_z,
        min_neighbors=isolation_min_neighbors,
        confidence_floor=confidence_floor,
        length_range=length_range,
        height_range=height_range,
    )
    stats["reliability_distribution"] = df["ReliabilityLevel"].value_counts().to_dict()
    stats["mean_geophysics_score"] = float(df["GeophysicsScore"].mean())
    stats["mean_geometry_score"] = float(df["GeometryScore"].mean())

    # Step 5a: 边界匹配 (仅标记)
    df, matched_pairs = boundary_match_only(df, boundary_tol_xy=boundary_tol_xy)
    stats["boundary_matched_pairs"] = len(matched_pairs)
    stats["boundary_matched_patches"] = int((df["ConnectionType"] == "boundary_matched").sum())

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

    # Step 3: 走廊检测 (CorridorSupport 证据)
    df = fracture_corridor_detection(df, corridor_search_radius, corridor_min_patches)
    n_in_corridors = int(df["CorridorSupport"].sum())
    stats["corridors_detected"] = int(df["CorridorID"].max() + 1) if n_in_corridors > 0 else 0
    stats["patches_in_corridors"] = n_in_corridors

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


if __name__ == "__main__":
    main()
