from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_AROUND_DATA_CSV = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/井斜/测井-地震时窗/车页1导眼_around_data.csv"
)
DEFAULT_REAL_FRACTURE_CSV = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝提取/车页1导眼_fractures.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze unit DFN reasonableness under a real imaging well constraint.")
    parser.add_argument("--unit-dir", type=Path, required=True)
    parser.add_argument("--around-data-csv", type=Path, default=DEFAULT_AROUND_DATA_CSV)
    parser.add_argument("--real-fracture-csv", type=Path, default=DEFAULT_REAL_FRACTURE_CSV)
    parser.add_argument("--well-name", type=str, default="车页1导眼")
    parser.add_argument("--corridor-radii-m", type=float, nargs="+", default=[12.5, 25.0, 50.0])
    parser.add_argument("--real-band-padding-ms", type=float, default=10.0)
    parser.add_argument("--match-time-tol-ms", type=float, default=20.0)
    parser.add_argument("--match-radius-thresholds-m", type=float, nargs="+", default=[12.5, 25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def angle_diff_deg(a_deg: np.ndarray, b_deg: float) -> np.ndarray:
    diff = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def circular_mean_deg(values_deg: np.ndarray) -> float | None:
    if values_deg.size == 0:
        return None
    rad = np.deg2rad(values_deg)
    sin_sum = float(np.sin(rad).sum())
    cos_sum = float(np.cos(rad).sum())
    if math.isclose(sin_sum, 0.0, abs_tol=1e-12) and math.isclose(cos_sum, 0.0, abs_tol=1e-12):
        return None
    angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
    return angle


def circular_resultant_length(values_deg: np.ndarray) -> float | None:
    if values_deg.size == 0:
        return None
    rad = np.deg2rad(values_deg)
    sin_mean = float(np.sin(rad).mean())
    cos_mean = float(np.cos(rad).mean())
    return float(np.hypot(sin_mean, cos_mean))


def circular_median_deg(values_deg: np.ndarray) -> float | None:
    if values_deg.size == 0:
        return None
    values = np.mod(values_deg.astype(float), 360.0)
    best = None
    best_score = None
    for candidate in values:
        score = float(angle_diff_deg(values, candidate).sum())
        if best_score is None or score < best_score:
            best = float(candidate)
            best_score = score
    return best


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    return f"{float(value):.{digits}f}"


def assign_layer_by_time(times: np.ndarray, layer_df: pd.DataFrame) -> list[str]:
    layer_rows = layer_df.sort_values("TopTime").reset_index(drop=True)
    labels: list[str] = []
    for time_value in times:
        matched_label = "OUTSIDE"
        for _, row in layer_rows.iterrows():
            top_time = float(row["TopTime"])
            base_time = float(row["BaseTime"])
            if top_time <= float(time_value) <= base_time:
                matched_label = str(row["LayerSurfacePairKey"])
                break
        labels.append(matched_label)
    return labels


def build_real_fracture_mapping(
    around_df: pd.DataFrame,
    real_df: pd.DataFrame,
    layer_df: pd.DataFrame,
) -> pd.DataFrame:
    around_sorted = around_df.sort_values("DEPT").reset_index(drop=True)
    dept = around_sorted["DEPT"].to_numpy()
    md = real_df["MD"].to_numpy()
    idx = np.searchsorted(dept, md)
    idx = np.clip(idx, 0, len(dept) - 1)
    left = np.clip(idx - 1, 0, len(dept) - 1)
    choose_left = np.abs(dept[left] - md) <= np.abs(dept[idx] - md)
    nearest = np.where(choose_left, left, idx)
    matched = around_sorted.iloc[nearest].reset_index(drop=True)

    result = real_df.copy()
    result["MappedDEPT"] = matched["DEPT"]
    result["MappedTIME"] = matched["TIME"]
    result["MappedX"] = matched["X"]
    result["MappedY"] = matched["Y"]
    if "UnitID" in matched.columns:
        result["UnitID"] = matched["UnitID"]
    result["LayerSurfacePairKey"] = assign_layer_by_time(result["MappedTIME"].to_numpy(), layer_df)
    return result


def add_well_distance_metrics(
    predicted_df: pd.DataFrame,
    around_df: pd.DataFrame,
) -> pd.DataFrame:
    work = predicted_df.copy()
    well_xy = around_df[["X", "Y"]].to_numpy(dtype=float)
    well_time = around_df["TIME"].to_numpy(dtype=float)
    tree = cKDTree(well_xy)
    distance_xy, nearest_idx = tree.query(work[["CenterX", "CenterY"]].to_numpy(dtype=float), k=1)
    nearest_time = well_time[nearest_idx]
    work["NearestWellDistanceXYM"] = distance_xy
    work["NearestWellTIME"] = nearest_time
    work["NearestWellTimeDeltaMS"] = np.abs(work["CenterTIME"].to_numpy(dtype=float) - nearest_time)
    return work


def summarize_orientation(values_deg: np.ndarray) -> dict[str, float | None]:
    if values_deg.size == 0:
        return {
            "均值方向(圆统计)": None,
            "中位方向(圆统计)": None,
            "最小值": None,
            "最大值": None,
            "集中度R(越接近1越集中)": None,
        }
    return {
        "均值方向(圆统计)": circular_mean_deg(values_deg),
        "中位方向(圆统计)": circular_median_deg(values_deg),
        "最小值": float(np.min(values_deg)),
        "最大值": float(np.max(values_deg)),
        "集中度R(越接近1越集中)": circular_resultant_length(values_deg),
    }


def summarize_dip(values_deg: np.ndarray) -> dict[str, float | None]:
    if values_deg.size == 0:
        return {
            "中位倾角": None,
            "平均倾角": None,
            "最小倾角": None,
            "最大倾角": None,
        }
    return {
        "中位倾角": float(np.median(values_deg)),
        "平均倾角": float(np.mean(values_deg)),
        "最小倾角": float(np.min(values_deg)),
        "最大倾角": float(np.max(values_deg)),
    }


def build_predicted_corridor_summary(
    predicted_df: pd.DataFrame,
    real_time_min: float,
    real_time_max: float,
    layer_df: pd.DataFrame,
    radii_m: list[float],
    padding_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    for radius_m in radii_m:
        corridor_mask = predicted_df["NearestWellDistanceXYM"] <= float(radius_m)
        band_mask = (
            predicted_df["CenterTIME"].between(real_time_min - padding_ms, real_time_max + padding_ms)
        )
        subset = predicted_df[corridor_mask & band_mask].copy()
        az = subset["Azimuth"].to_numpy(dtype=float) if len(subset) else np.array([], dtype=float)
        dip = subset["Dip"].to_numpy(dtype=float) if len(subset) else np.array([], dtype=float)
        rows.append(
            {
                "走廊半径m": float(radius_m),
                "预测裂缝片数": int(len(subset)),
                "时间范围最小ms": float(subset["CenterTIME"].min()) if len(subset) else np.nan,
                "时间范围最大ms": float(subset["CenterTIME"].max()) if len(subset) else np.nan,
                "平均井距m": float(subset["NearestWellDistanceXYM"].mean()) if len(subset) else np.nan,
                "方向均值(圆统计)": circular_mean_deg(az),
                "方向中位(圆统计)": circular_median_deg(az),
                "方向集中度R": circular_resultant_length(az),
                "倾角中位数": float(np.median(dip)) if len(dip) else np.nan,
                "倾角均值": float(np.mean(dip)) if len(dip) else np.nan,
                "长度中位数": float(np.median(subset["PatchLength"])) if len(subset) else np.nan,
                "高度中位数": float(np.median(subset["PatchHeight"])) if len(subset) else np.nan,
            }
        )
        counts = subset["LayerSurfacePairKey"].value_counts().to_dict()
        for _, layer_row in layer_df.iterrows():
            layer_key = str(layer_row["LayerSurfacePairKey"])
            layer_rows.append(
                {
                    "走廊半径m": float(radius_m),
                    "LayerSurfacePairKey": layer_key,
                    "预测裂缝片数": int(counts.get(layer_key, 0)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(layer_rows)


def build_real_layer_summary(real_df: pd.DataFrame, layer_df: pd.DataFrame) -> pd.DataFrame:
    counts = real_df["LayerSurfacePairKey"].value_counts().to_dict()
    rows: list[dict[str, object]] = []
    for _, row in layer_df.iterrows():
        layer_key = str(row["LayerSurfacePairKey"])
        rows.append(
            {
                "LayerSurfacePairKey": layer_key,
                "真实裂缝条数": int(counts.get(layer_key, 0)),
                "TopTime": float(row["TopTime"]),
                "BaseTime": float(row["BaseTime"]),
                "LayerThicknessMs": float(row["BaseTime"] - row["TopTime"]),
            }
        )
    return pd.DataFrame(rows)


def build_real_predicted_match_tables(
    real_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    *,
    time_tol_ms: float,
    radius_thresholds_m: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    radius_thresholds = sorted(float(value) for value in radius_thresholds_m)
    rows: list[dict[str, object]] = []
    for _, real_row in real_df.iterrows():
        real_time = float(real_row["MappedTIME"])
        real_x = float(real_row["MappedX"])
        real_y = float(real_row["MappedY"])
        real_layer = str(real_row["LayerSurfacePairKey"])
        cand = predicted_df[np.abs(predicted_df["CenterTIME"].to_numpy(dtype=float) - real_time) <= float(time_tol_ms)].copy()
        cand_same_layer = cand[cand["LayerSurfacePairKey"].astype(str) == real_layer].copy()
        record: dict[str, object] = {
            "RealRowIndex": int(real_row.name),
            "RealMD": float(real_row["MD"]),
            "RealMappedTIME": real_time,
            "RealMappedX": real_x,
            "RealMappedY": real_y,
            "RealLayerSurfacePairKey": real_layer,
            "RealAzimuthDeg": float(real_row["Azimuth(0~360)"]),
            "RealDipDeg": float(real_row["Angle(0~90)"]),
            "AnyLayerCandidateCountWithinTimeTol": int(len(cand)),
            "SameLayerCandidateCountWithinTimeTol": int(len(cand_same_layer)),
            "MatchedSameLayerWithinTimeTol": False,
        }
        if cand_same_layer.empty:
            for threshold in radius_thresholds:
                record[f"MatchedWithin_{str(threshold).replace('.', 'p')}m"] = False
            rows.append(record)
            continue

        dx = cand_same_layer["CenterX"].to_numpy(dtype=float) - real_x
        dy = cand_same_layer["CenterY"].to_numpy(dtype=float) - real_y
        distances = np.hypot(dx, dy)
        nearest_idx = int(np.argmin(distances))
        pred_row = cand_same_layer.iloc[nearest_idx]
        nearest_distance = float(distances[nearest_idx])
        az_diff = float(angle_diff_deg(np.array([float(pred_row["Azimuth"])]), float(real_row["Azimuth(0~360)"]))[0])
        dip_diff = float(abs(float(pred_row["Dip"]) - float(real_row["Angle(0~90)"])))

        record.update(
            {
                "MatchedSameLayerWithinTimeTol": True,
                "NearestPredPatchID": str(pred_row.get("PatchID", "")),
                "NearestPredCenterTIME": float(pred_row["CenterTIME"]),
                "NearestPredDistanceXYM": nearest_distance,
                "NearestPredTimeDeltaMS": float(abs(float(pred_row["CenterTIME"]) - real_time)),
                "NearestPredAzimuthDeg": float(pred_row["Azimuth"]),
                "NearestPredDipDeg": float(pred_row["Dip"]),
                "NearestPredAzimuthDiffDeg": az_diff,
                "NearestPredDipDiffDeg": dip_diff,
                "NearestPredPatchLength": float(pred_row["PatchLength"]),
                "NearestPredPatchHeight": float(pred_row["PatchHeight"]),
                "NearestPredPatchArea": float(pred_row["PatchArea"]) if "PatchArea" in pred_row else float(pred_row["PatchLength"] * pred_row["PatchHeight"]),
            }
        )
        for threshold in radius_thresholds:
            record[f"MatchedWithin_{str(threshold).replace('.', 'p')}m"] = bool(nearest_distance <= float(threshold))
        rows.append(record)

    match_df = pd.DataFrame(rows)
    matched_df = match_df[match_df["MatchedSameLayerWithinTimeTol"]].copy()
    summary: dict[str, object] = {
        "TimeToleranceMS": float(time_tol_ms),
        "RealFractureCount": int(len(match_df)),
        "MatchedSameLayerCount": int(len(matched_df)),
        "MatchedSameLayerRecall": float(len(matched_df) / len(match_df)) if len(match_df) else 0.0,
        "MedianNearestDistanceXYM": float(matched_df["NearestPredDistanceXYM"].median()) if len(matched_df) else np.nan,
        "P75NearestDistanceXYM": float(matched_df["NearestPredDistanceXYM"].quantile(0.75)) if len(matched_df) else np.nan,
        "MedianNearestAzimuthDiffDeg": float(matched_df["NearestPredAzimuthDiffDeg"].median()) if len(matched_df) else np.nan,
        "MedianNearestDipDiffDeg": float(matched_df["NearestPredDipDiffDeg"].median()) if len(matched_df) else np.nan,
    }
    for threshold in radius_thresholds:
        col = f"MatchedWithin_{str(threshold).replace('.', 'p')}m"
        summary[f"MatchedCountWithin_{str(threshold).replace('.', 'p')}m"] = int(match_df[col].fillna(False).sum())
        summary[f"MatchedRecallWithin_{str(threshold).replace('.', 'p')}m"] = float(match_df[col].fillna(False).mean()) if len(match_df) else 0.0
    summary_df = pd.DataFrame([summary])

    layer_rows: list[dict[str, object]] = []
    for layer_key, layer_group in match_df.groupby("RealLayerSurfacePairKey"):
        layer_matched = layer_group[layer_group["MatchedSameLayerWithinTimeTol"]].copy()
        layer_record: dict[str, object] = {
            "LayerSurfacePairKey": str(layer_key),
            "RealFractureCount": int(len(layer_group)),
            "MatchedSameLayerCount": int(len(layer_matched)),
            "MatchedSameLayerRecall": float(len(layer_matched) / len(layer_group)) if len(layer_group) else 0.0,
            "MedianNearestDistanceXYM": float(layer_matched["NearestPredDistanceXYM"].median()) if len(layer_matched) else np.nan,
            "MedianNearestAzimuthDiffDeg": float(layer_matched["NearestPredAzimuthDiffDeg"].median()) if len(layer_matched) else np.nan,
            "MedianNearestDipDiffDeg": float(layer_matched["NearestPredDipDiffDeg"].median()) if len(layer_matched) else np.nan,
        }
        for threshold in radius_thresholds:
            col = f"MatchedWithin_{str(threshold).replace('.', 'p')}m"
            layer_record[f"MatchedCountWithin_{str(threshold).replace('.', 'p')}m"] = int(layer_group[col].fillna(False).sum())
        layer_rows.append(layer_record)
    layer_summary_df = pd.DataFrame(layer_rows)
    return match_df, summary_df, layer_summary_df


def build_reasonableness_report(
    unit_id: str,
    well_name: str,
    real_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    real_layer_summary: pd.DataFrame,
    corridor_summary: pd.DataFrame,
    corridor_layer_summary: pd.DataFrame,
    match_summary_df: pd.DataFrame,
    match_layer_summary_df: pd.DataFrame,
) -> str:
    real_az = real_df["Azimuth(0~360)"].to_numpy(dtype=float)
    real_dip = real_df["Angle(0~90)"].to_numpy(dtype=float)
    real_orientation = summarize_orientation(real_az)
    real_dip_summary = summarize_dip(real_dip)
    focus_radius = 25.0 if 25.0 in corridor_summary["走廊半径m"].values else float(corridor_summary.iloc[0]["走廊半径m"])
    focus_row = corridor_summary[corridor_summary["走廊半径m"] == focus_radius].iloc[0]
    focus_layer_rows = corridor_layer_summary[corridor_layer_summary["走廊半径m"] == focus_radius].copy()
    focus_layer_rows = focus_layer_rows.sort_values("预测裂缝片数", ascending=False)
    real_layer_rows = real_layer_summary[real_layer_summary["真实裂缝条数"] > 0].copy()
    real_layer_rows = real_layer_rows.sort_values("真实裂缝条数", ascending=False)
    match_row = match_summary_df.iloc[0]
    layer_match_rows = match_layer_summary_df.sort_values("RealFractureCount", ascending=False)
    real_main_layers = "、".join(
        f"{row['LayerSurfacePairKey']}({int(row['真实裂缝条数'])}条)" for _, row in real_layer_rows.iterrows()
    ) or "无"
    pred_main_layers = "、".join(
        f"{row['LayerSurfacePairKey']}({int(row['预测裂缝片数'])}片)" for _, row in focus_layer_rows.head(3).iterrows() if int(row["预测裂缝片数"]) > 0
    ) or "无"
    layer_match_text = "、".join(
        f"{row['LayerSurfacePairKey']}({int(row['MatchedSameLayerCount'])}/{int(row['RealFractureCount'])})"
        for _, row in layer_match_rows.iterrows()
        if int(row["RealFractureCount"]) > 0
    ) or "无"

    lines = [
        f"# {well_name} - {unit_id} 井约束合理性评估",
        "",
        "## 1. 基本事实",
        f"- 真实成像裂缝总数：{len(real_df)} 条",
        f"- 真实裂缝时间范围：{format_float(float(real_df['MappedTIME'].min()), 1)} ~ {format_float(float(real_df['MappedTIME'].max()), 1)} ms",
        f"- 预测单元裂缝总数：{len(pred_df)} 片",
        f"- 重点井附近走廊：半径 {format_float(focus_radius, 1)} m，且限制在真实裂缝时间带 ±10 ms 内",
        f"- 该走廊内预测裂缝片数：{int(focus_row['预测裂缝片数'])} 片",
        "",
        "## 2. 真实井告诉我们的主信息",
        f"- 真实裂缝主要层段：{real_main_layers}",
        f"- 真实方向中位数：{format_float(real_orientation['中位方向(圆统计)'], 1)}°",
        f"- 真实方向集中度 R：{format_float(real_orientation['集中度R(越接近1越集中)'], 3)}",
        f"- 真实倾角中位数：{format_float(real_dip_summary['中位倾角'], 1)}°",
        "",
        "## 3. 预测单元在井附近的响应",
        f"- 井附近主预测层段：{pred_main_layers}",
        f"- 井附近方向中位数：{format_float(focus_row['方向中位(圆统计)'], 1)}°",
        f"- 井附近方向集中度 R：{format_float(focus_row['方向集中度R'], 3)}",
        f"- 井附近倾角中位数：{format_float(focus_row['倾角中位数'], 1)}°",
        f"- 井附近长度中位数：{format_float(focus_row['长度中位数'], 2)}",
        f"- 井附近高度中位数：{format_float(focus_row['高度中位数'], 2)}",
        "",
        "## 4. 逐条配对检查（同层且时差容差内）",
        f"- 配对时间容差：{format_float(match_row['TimeToleranceMS'], 1)} ms",
        f"- 可配对真实裂缝：{int(match_row['MatchedSameLayerCount'])}/{int(match_row['RealFractureCount'])} 条",
        f"- 配对召回率：{format_float(match_row['MatchedSameLayerRecall'] * 100.0, 1)}%",
        f"- 最近预测裂缝距离中位数：{format_float(match_row['MedianNearestDistanceXYM'], 2)} m",
        f"- 配对方向差中位数：{format_float(match_row['MedianNearestAzimuthDiffDeg'], 1)}°",
        f"- 配对倾角差中位数：{format_float(match_row['MedianNearestDipDiffDeg'], 1)}°",
        f"- 25 m 内命中：{int(match_row.get('MatchedCountWithin_25p0m', 0))}/{int(match_row['RealFractureCount'])} 条",
        f"- 50 m 内命中：{int(match_row.get('MatchedCountWithin_50p0m', 0))}/{int(match_row['RealFractureCount'])} 条",
        f"- 分层配对情况：{layer_match_text}",
        "",
        "## 5. 如何解读是否合理",
        "- 如果井附近主预测层段和真实裂缝主层段一致，说明模型至少把“裂缝主要发育在哪一层”判断对了。",
        "- 如果井附近方向中位数与真实方向中位数接近，而且方向集中度都不低，说明裂缝产状主方向是对得上的。",
        "- 如果井附近倾角中位数与真实倾角中位数接近，说明裂缝姿态整体没有跑偏。",
        "- 如果真实井明显发育的层段在预测里完全没有裂缝，或者预测主裂缝都落在真实井没有裂缝的层段，这就是明显不合理。",
        "- 裂缝尺寸目前只能做“间接判断”。因为当前真实成像裂缝 CSV 里没有直接和单元 DFN 一一对应的裂缝片长度/面积真值，所以尺寸更适合后续再结合裂缝长度 LAS 成果做补充验证。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    unit_dir = args.unit_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (unit_dir / "well_constrained_reasonableness")
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_path = unit_dir / "predicted_unit_patches.csv"
    layer_path = unit_dir / "unit_layers_input.csv"
    if not predicted_path.exists():
        raise FileNotFoundError(f"predicted_unit_patches.csv not found: {predicted_path}")
    if not layer_path.exists():
        raise FileNotFoundError(f"unit_layers_input.csv not found: {layer_path}")

    predicted_df = pd.read_csv(predicted_path, encoding="utf-8-sig")
    layer_df = pd.read_csv(layer_path, encoding="utf-8-sig")
    around_df = pd.read_csv(args.around_data_csv, encoding="utf-8-sig")
    real_df = pd.read_csv(args.real_fracture_csv, encoding="utf-8-sig")

    unit_id = str(layer_df.iloc[0]["UnitID"])
    if {"BlockX", "BlockY"}.issubset(around_df.columns):
        around_unit_df = around_df[(around_df["BlockX"] == int(layer_df.iloc[0]["BlockX"])) & (around_df["BlockY"] == int(layer_df.iloc[0]["BlockY"]))].copy()
    else:
        around_unit_df = around_df.copy()
    if len(around_unit_df) == 0:
        around_unit_df = around_df.copy()

    real_mapped_df = build_real_fracture_mapping(around_unit_df, real_df, layer_df)
    predicted_with_well_df = add_well_distance_metrics(predicted_df, around_unit_df)

    real_time_min = float(real_mapped_df["MappedTIME"].min())
    real_time_max = float(real_mapped_df["MappedTIME"].max())

    real_layer_summary = build_real_layer_summary(real_mapped_df, layer_df)
    corridor_summary, corridor_layer_summary = build_predicted_corridor_summary(
        predicted_with_well_df,
        real_time_min=real_time_min,
        real_time_max=real_time_max,
        layer_df=layer_df,
        radii_m=[float(x) for x in args.corridor_radii_m],
        padding_ms=float(args.real_band_padding_ms),
    )
    match_df, match_summary_df, match_layer_summary_df = build_real_predicted_match_tables(
        real_mapped_df,
        predicted_with_well_df,
        time_tol_ms=float(args.match_time_tol_ms),
        radius_thresholds_m=[float(x) for x in args.match_radius_thresholds_m],
    )

    report_md = build_reasonableness_report(
        unit_id=unit_id,
        well_name=args.well_name,
        real_df=real_mapped_df,
        pred_df=predicted_df,
        real_layer_summary=real_layer_summary,
        corridor_summary=corridor_summary,
        corridor_layer_summary=corridor_layer_summary,
        match_summary_df=match_summary_df,
        match_layer_summary_df=match_layer_summary_df,
    )

    real_mapped_df.to_csv(output_dir / "real_imaging_fractures_mapped.csv", index=False, encoding="utf-8-sig")
    predicted_with_well_df.to_csv(output_dir / "predicted_patches_with_well_distance.csv", index=False, encoding="utf-8-sig")
    real_layer_summary.to_csv(output_dir / "real_layer_summary.csv", index=False, encoding="utf-8-sig")
    corridor_summary.to_csv(output_dir / "predicted_corridor_summary.csv", index=False, encoding="utf-8-sig")
    corridor_layer_summary.to_csv(output_dir / "predicted_corridor_layer_summary.csv", index=False, encoding="utf-8-sig")
    match_df.to_csv(output_dir / "real_to_predicted_nearest_match.csv", index=False, encoding="utf-8-sig")
    match_summary_df.to_csv(output_dir / "real_to_predicted_match_summary.csv", index=False, encoding="utf-8-sig")
    match_layer_summary_df.to_csv(output_dir / "real_to_predicted_match_summary_by_layer.csv", index=False, encoding="utf-8-sig")
    (output_dir / "reasonableness_report.md").write_text(report_md, encoding="utf-8")

    summary_payload = {
        "well_name": args.well_name,
        "unit_id": unit_id,
        "real_fracture_count": int(len(real_mapped_df)),
        "predicted_patch_count": int(len(predicted_df)),
        "real_time_min_ms": real_time_min,
        "real_time_max_ms": real_time_max,
        "corridor_radii_m": [float(x) for x in args.corridor_radii_m],
        "match_time_tolerance_ms": float(args.match_time_tol_ms),
        "match_radius_thresholds_m": [float(x) for x in args.match_radius_thresholds_m],
        "matched_same_layer_count": int(match_summary_df.iloc[0]["MatchedSameLayerCount"]),
        "matched_same_layer_recall": float(match_summary_df.iloc[0]["MatchedSameLayerRecall"]),
        "real_layer_counts": {
            str(row["LayerSurfacePairKey"]): int(row["真实裂缝条数"]) for _, row in real_layer_summary.iterrows()
        },
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"unit_id={unit_id}")
    print(f"real_fracture_count={len(real_mapped_df)}")
    print(f"predicted_patch_count={len(predicted_df)}")
    print(f"real_time_range_ms=({real_time_min:.3f}, {real_time_max:.3f})")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
