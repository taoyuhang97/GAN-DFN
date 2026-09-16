#!/usr/bin/env python3
"""Attach TaiGuJie imaging density and fracture points to Step2 v3 segments.

v3 changes (2026-08-12):
- Depth semantics are TVD: density and point depths from the imaging
  interpretation are TVD, and Step2 segments carry both MD and TVD, so labels
  are interpolated/attached by TVD;
- Reads the Step2 v3 segment manifest and aggregates all segments of a well;
- Point mapping is performed once at well level (`DroppedPointRows` counts
  unique unmapped points);
- Upstream-rejected wells are reported in `wells_without_groups` with the Step2
  rejection reason.

v4 changes (2026-09-16):
- 多密度源合并改为"按源声明深度范围裁剪 + 同深度取最大值"。
  旧实现 `concat → sort_values → drop_duplicates(keep="last")` 在深度区间完全重叠的
  多个源之间会随机丢失有效值：埕北313 的两个密度文件都从 0.5 m 写起、各自覆盖整口井，
  该写法把主段 1747 个正值行砍到 874 行，积分 1245.9 → 620.3（且换排序方式会变成 0）。
- 密度按井标定到"每条缝 = 1 条"：`Density` 输出标定后的线密度（条/米），原值保留在
  `DensityRaw`，系数记录在 `DensityScaleFactor` 与 `density_calibration_audit.csv`。
  标定点数取该井落在常规测井覆盖内的产状点数，除以同一覆盖内的密度积分。
- 落在常规测井覆盖之外的成像数据不再静默丢弃：密度段外部分记入
  `imaging_out_of_log_coverage.csv`，产状点明细记入 `unmapped_imaging_points.csv`
  （供 Step8 井控使用）。
- 产状点吸附放宽：只要落在常规测井覆盖内就吸附到最近采样点，不再因"离网格超过容差"丢弃；
  吸附残差（中位/P95/最大）与超过半个采样步长的行数记入 `point_mapping_qc.csv`。
- 监督层级 `SupervisionTier`（strong / presence_only / audit_only）写入组 CSV，
  供 Step4 做监督门控；层级由 `supervision_status` 映射，可按井覆盖。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INVALID = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)
LABEL_COLUMNS = [
    "GroupID", "WellName", "InputSegmentPath", "MD", "TVD", "StrataName",
    "Density", "DensityRaw", "DensityScaleFactor", "ImagingWindowID",
    "HasFractureDensity", "GT_POINT_FLAG", "RawPointCount", "FracAzimuth", "FracDip",
    "DensityKind", "FractureScope", "DensitySourcePaths", "PointSourcePaths",
    "DensitySupportStatus", "SupervisionStatus", "SupervisionTier",
]

DEFAULT_SUPERVISION_TIERS = {
    "supervision_ready": "strong",
    "candidate_scope_pending": "presence_only",
    # 无产状点、刻度未锚定的井（桩海102）同样参与"存在性"监督（正/负样本都要），
    # 只是它的密度数值不参与绝对水平标定——见 Step4 的 supervision_gate。
    "density_only_scope_pending": "presence_only",
}


def lookup_window_id(tvd_values: np.ndarray, windows: list[tuple[str, float, float]]) -> np.ndarray:
    """按深度给每个采样点标出所属的成像解释窗口（不属于任何窗口留空）。"""
    out = np.full(len(tvd_values), "", dtype=object)
    for window_id, lo, hi in windows:
        inside = (tvd_values >= lo) & (tvd_values <= hi)
        out[inside & (out == "")] = window_id
    return out


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """合并重叠/相邻的深度区间（用于常规测井覆盖范围）。"""
    out: list[tuple[float, float]] = []
    for lo, hi in sorted((float(a), float(b)) for a, b in ranges if b >= a):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def gap_break_for(depth: np.ndarray, floor_m: float = 0.5) -> float:
    """缺口阈值：不小于 3 倍采样步长（保护 1 m 采样的曲线不被误判成全是缺口）。"""
    if len(depth) < 2:
        return floor_m
    step = float(np.median(np.diff(np.sort(depth))))
    return max(3.0 * step, floor_m)


def integrate_runs(depth: np.ndarray, values: np.ndarray, gap_break_m: float | None = None) -> float:
    """按不连续段分别做梯形积分，避免跨缺口连乘出虚假面积。"""
    if len(depth) < 2:
        return 0.0
    if gap_break_m is None:
        gap_break_m = gap_break_for(depth)
    breaks = np.flatnonzero(np.diff(depth) > gap_break_m)
    total = 0.0
    for run in np.split(np.arange(len(depth)), breaks + 1):
        if len(run) >= 2:
            total += float(np.trapezoid(values[run], depth[run]))
    return total


def runs_of(depth: np.ndarray, gap_break_m: float | None = None) -> list[tuple[float, float]]:
    """把已排序的深度数组切成连续段（缺口大于阈值处断开）。"""
    if len(depth) == 0:
        return []
    if gap_break_m is None:
        gap_break_m = gap_break_for(depth)
    breaks = np.flatnonzero(np.diff(depth) > gap_break_m)
    return [
        (float(depth[run[0]]), float(depth[run[-1]]))
        for run in np.split(np.arange(len(depth)), breaks + 1)
        if len(run) >= 2
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TaiGuJie Step3 imaging supervision groups (v3)")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "taigu_step3_imaging_groups.json")
    p.add_argument("--replace-output", action="store_true")
    return p.parse_args()


def clean(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    for value in INVALID:
        out = out.mask(np.isclose(out, value, equal_nan=False))
    return out


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot read csv: {path}")


def numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if values:
            rows.append(values)
    return rows


def read_density(item: dict[str, Any], window_id: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """读取一个密度源，并按它自己的解释窗口裁剪。

    窗口 = 配置里声明的 `depth_min_m`/`depth_max_m`；没声明时用文件自身的深度轴范围。
    裁剪的目的是把"文件为了写满深度轴而填的 0"与"解释区间内真实的 0"区分开：
    填充区不进入合并结果，于是下游得到的是**无资料（NaN）**而不是"密度为 0 的强负样本"。
    """
    path = Path(item["path"])
    fmt = item["format"]
    if fmt == "whitespace":
        rows = numeric_rows(path)
        depth_index, value_index = int(item["depth_index"]), int(item["value_index"])
        data = [[row[depth_index], row[value_index]] for row in rows if len(row) > max(depth_index, value_index)]
        df = pd.DataFrame(data, columns=["TVD", "Density"])
    elif fmt == "csv":
        raw = read_csv(path)
        df = raw[[item["depth_column"], item["value_column"]]].rename(
            columns={item["depth_column"]: "TVD", item["value_column"]: "Density"}
        )
    else:
        raise ValueError(f"unsupported density format: {fmt}")
    df["TVD"], df["Density"] = clean(df["TVD"]), clean(df["Density"])
    df = df.dropna(subset=["TVD", "Density"]).sort_values("TVD").groupby("TVD", as_index=False)["Density"].mean()
    rows_in_file = int(len(df))
    declared_min = item.get("depth_min_m")
    declared_max = item.get("depth_max_m")
    window_min = float(declared_min) if declared_min is not None else float(df["TVD"].min())
    window_max = float(declared_max) if declared_max is not None else float(df["TVD"].max())
    keep = (df["TVD"].to_numpy(dtype=float) >= window_min) & (df["TVD"].to_numpy(dtype=float) <= window_max)
    df = df[keep].reset_index(drop=True)
    positive = df.loc[df.Density.gt(0.0), "TVD"]
    df["DensitySourcePath"] = str(path)
    df["WindowID"] = window_id
    stat: dict[str, object] = {
        "WindowID": window_id,
        "DensitySourcePath": str(path),
        "DeclaredDepthMinM": declared_min,
        "DeclaredDepthMaxM": declared_max,
        "WindowStartM": round(window_min, 2),
        "WindowEndM": round(window_max, 2),
        "WindowSource": "declared" if (declared_min is not None or declared_max is not None) else "file_axis",
        "RowsInFile": rows_in_file,
        "RowsAfterRangeClip": int(len(df)),
        "PaddingRowsDropped": rows_in_file - int(len(df)),
        "PositiveRows": int(df.Density.gt(0.0).sum()),
        "PositiveTVDMin": float(positive.min()) if len(positive) else None,
        "PositiveTVDMax": float(positive.max()) if len(positive) else None,
    }
    return df, stat


def merge_density_sources(parts: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """把同一口井的多个解释窗口按 TVD 拼成一条曲线。

    每个窗口已经按自己的解释区间裁剪过，所以正常情况下窗口之间没有重叠；仍然保留
    "同深度取最大值"作为兜底规则，并返回"同一深度有多个窗口同时给出正值且数值不同"
    的深度个数，用于审计是否真的存在口径冲突。
    """
    if not parts:
        return pd.DataFrame(columns=["TVD", "Density", "WindowID", "SourceCount"]), 0
    stacked_parts = []
    for frame in parts:
        part = frame[["TVD", "Density", "WindowID"]].copy()
        part["_SourceIndex"] = part["WindowID"].astype(str)
        stacked_parts.append(part)
    stacked = pd.concat(stacked_parts, ignore_index=True)
    grouped = stacked.groupby("TVD", as_index=False).agg(
        Density=("Density", "max"),
        SourceCount=("_SourceIndex", "nunique"),
        PositiveSourceCount=("Density", lambda values: int((values > 0.0).sum())),
        PositiveMin=("Density", lambda values: float(values[values > 0.0].min()) if (values > 0.0).any() else 0.0),
        PositiveMax=("Density", "max"),
    )
    conflicts = int(
        ((grouped.PositiveSourceCount > 1) & ((grouped.PositiveMax - grouped.PositiveMin) > 1.0e-9)).sum()
    )
    owner = (
        stacked.sort_values(["TVD", "Density"], ascending=[True, False], kind="stable")
        .drop_duplicates("TVD", keep="first")[["TVD", "WindowID"]]
    )
    merged = (
        grouped.merge(owner, on="TVD", how="left")
        .sort_values("TVD")
        .reset_index(drop=True)[["TVD", "Density", "WindowID", "SourceCount"]]
    )
    return merged, conflicts


def read_points(item: dict[str, Any]) -> pd.DataFrame:
    path = Path(item["path"])
    fmt = item["format"]
    if fmt == "whitespace":
        rows = numeric_rows(path)
        indices = [int(item[key]) for key in ("depth_index", "azimuth_index", "dip_index")]
        data = [[row[i] for i in indices] for row in rows if len(row) > max(indices)]
        df = pd.DataFrame(data, columns=["TVD", "FracAzimuth", "FracDip"])
    elif fmt == "csv":
        raw = read_csv(path)
        df = raw[[item["depth_column"], item["azimuth_column"], item["dip_column"]]].rename(
            columns={item["depth_column"]: "TVD", item["azimuth_column"]: "FracAzimuth", item["dip_column"]: "FracDip"}
        )
    else:
        raise ValueError(f"unsupported point format: {fmt}")
    for column in ("TVD", "FracAzimuth", "FracDip"):
        df[column] = clean(df[column])
    return df.dropna(subset=["TVD"]).assign(PointSourcePath=str(path))


def interpolate_density(density: pd.DataFrame, target_tvd: pd.Series, max_gap_m: float) -> np.ndarray:
    x, y = density["TVD"].to_numpy(float), density["Density"].to_numpy(float)
    target = target_tvd.to_numpy(float)
    if len(x) < 2:
        return np.full(len(target), np.nan)
    result = np.interp(target, x, y, left=np.nan, right=np.nan)
    right = np.searchsorted(x, target, side="left").clip(0, len(x) - 1)
    left = (right - 1).clip(0, len(x) - 1)
    exact = np.isclose(x[right], target, atol=1e-8) | np.isclose(x[left], target, atol=1e-8)
    supported = (target >= x[0]) & (target <= x[-1]) & (exact | ((x[right] - x[left]) <= max_gap_m))
    result[~supported] = np.nan
    return result


def attach_points(
    labels: pd.DataFrame,
    points: pd.DataFrame,
    coverage: list[tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Attach points once at well level by TVD.

    Returns `(labels, unmapped, residual_stats)`。**覆盖判定取代容差判定**：只要点落在常规
    测井覆盖内，就吸附到最近的采样点（不再因为离网格超过容差而丢弃），残差只记录不拦截；
    只有落在覆盖之外的点才进 `unmapped`（成像测到了、常规测井没有覆盖），供审计与 Step8 井控使用。
    """
    out = labels.copy()
    out["GT_POINT_FLAG"] = 0
    out["RawPointCount"] = 0
    out["FracAzimuth"] = np.nan
    out["FracDip"] = np.nan
    unmapped_columns = [
        "PointRowIndex", "TVD", "FracAzimuth", "FracDip", "PointSourcePath", "Reason",
        "NearestGridTVD", "DistanceToGridM",
    ]
    empty_stats = {
        "GridStepM": np.nan,
        "AttachResidualMedianM": np.nan,
        "AttachResidualP95M": np.nan,
        "AttachResidualMaxM": np.nan,
        "AttachedRows": 0.0,
        "AttachedBeyondHalfStepRows": 0.0,
    }
    if out.empty or points.empty:
        return out, pd.DataFrame(columns=unmapped_columns), empty_stats
    tvd = out["TVD"].to_numpy(float)
    step = float(np.median(np.diff(tvd))) if len(tvd) > 1 else 0.1
    ranges = merge_ranges(coverage) if coverage else [(float(tvd.min()), float(tvd.max()))]

    def inside_coverage(value: float) -> bool:
        return any(lo - 1.0e-6 <= value <= hi + 1.0e-6 for lo, hi in ranges)

    unmapped_rows: list[dict[str, object]] = []
    residuals: list[float] = []
    for point in points.itertuples(index=False):
        pos = int(np.searchsorted(tvd, point.TVD))
        candidates = [idx for idx in (pos - 1, pos) if 0 <= idx < len(tvd)]
        idx = min(candidates, key=lambda value: abs(tvd[value] - point.TVD)) if candidates else None
        distance = abs(tvd[idx] - point.TVD) if idx is not None else np.inf
        if idx is None or not inside_coverage(float(point.TVD)):
            unmapped_rows.append({
                "PointRowIndex": int(getattr(point, "PointRowIndex", -1)),
                "TVD": float(point.TVD),
                "FracAzimuth": float(point.FracAzimuth) if pd.notna(point.FracAzimuth) else np.nan,
                "FracDip": float(point.FracDip) if pd.notna(point.FracDip) else np.nan,
                "PointSourcePath": str(getattr(point, "PointSourcePath", "")),
                "Reason": "outside_log_coverage",
                "NearestGridTVD": float(tvd[idx]) if idx is not None else np.nan,
                "DistanceToGridM": float(distance) if np.isfinite(distance) else np.nan,
            })
            continue
        residuals.append(float(distance))
        out.loc[idx, "GT_POINT_FLAG"] = 1
        out.loc[idx, "RawPointCount"] += 1
        if pd.isna(out.loc[idx, "FracAzimuth"]):
            out.loc[idx, "FracAzimuth"] = point.FracAzimuth
            out.loc[idx, "FracDip"] = point.FracDip
    residual_array = np.asarray(residuals, dtype=float)
    stats = {
        "GridStepM": float(step),
        "AttachResidualMedianM": float(np.median(residual_array)) if residual_array.size else np.nan,
        "AttachResidualP95M": float(np.quantile(residual_array, 0.95)) if residual_array.size else np.nan,
        "AttachResidualMaxM": float(residual_array.max()) if residual_array.size else np.nan,
        "AttachedRows": float(residual_array.size),
        "AttachedBeyondHalfStepRows": float((residual_array > step * 0.5).sum()) if residual_array.size else 0.0,
    }
    return out, pd.DataFrame(unmapped_rows, columns=unmapped_columns), stats


def split_groups(labels: pd.DataFrame) -> list[pd.DataFrame]:
    usable = labels[labels["Density"].notna()].copy().sort_values("TVD").reset_index(drop=True)
    if usable.empty:
        return []
    step = float(np.median(np.diff(usable["TVD"]))) if len(usable) > 1 else 0.1
    new_group = (
        usable["StrataName"].ne(usable["StrataName"].shift())
        | usable["ImagingWindowID"].astype(str).ne(usable["ImagingWindowID"].astype(str).shift())
        | usable["TVD"].diff().gt(max(step * 3.0, 0.5))
    )
    usable["DensityCoverageSegmentID"] = new_group.cumsum().astype(int)
    return [frame.reset_index(drop=True) for _, frame in usable.groupby("DensityCoverageSegmentID", sort=True)]


def main() -> int:
    ns = parse_args()
    cfg = json.loads(ns.config.read_text(encoding="utf-8"))
    step2_root = Path(cfg["step2_output_dir"])
    output = Path(cfg["output_dir"])
    if not step2_root.is_absolute():
        step2_root = Path(__file__).resolve().parents[2] / step2_root
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    if output.exists() and any(output.iterdir()):
        if not ns.replace_output:
            raise RuntimeError(f"output exists; pass --replace-output: {output}")
        shutil.rmtree(output)
    groups_dir = output / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(step2_root / "taigu_step2_segment_manifest.csv", encoding="utf-8-sig")
    rejected = pd.read_csv(step2_root / "taigu_step2_rejected_wells.csv", encoding="utf-8-sig")
    rejected_reason = {str(r.WellName): str(r.Reason) for _, r in rejected.iterrows()} if not rejected.empty else {}

    group_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    point_qc_rows: list[dict[str, object]] = []
    density_source_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    out_of_coverage_rows: list[dict[str, object]] = []
    unmapped_rows: list[dict[str, object]] = []
    wells_without_groups: list[dict[str, object]] = []

    for well_cfg in cfg["wells"]:
        well = well_cfg["well_name"]
        density_parts = []
        windows: list[tuple[str, float, float]] = []
        window_source_path: dict[str, str] = {}
        for index, item in enumerate(well_cfg["density_sources"], start=1):
            window_id = str(item.get("window_id") or f"{well}_W{index:02d}")
            frame, stat = read_density(item, window_id)
            stat["WellName"] = well
            density_source_rows.append(stat)
            density_parts.append(frame)
            windows.append((window_id, float(stat["WindowStartM"]), float(stat["WindowEndM"])))
            window_source_path[window_id] = str(item["path"])
        density, density_conflicts = merge_density_sources(density_parts)
        points = (
            pd.concat([read_points(item) for item in well_cfg.get("point_sources", [])], ignore_index=True)
            if well_cfg.get("point_sources")
            else pd.DataFrame(columns=["TVD", "FracAzimuth", "FracDip", "PointSourcePath"])
        )
        points["PointRowIndex"] = np.arange(len(points), dtype=int)
        density_paths = "|".join(str(item["path"]) for item in well_cfg["density_sources"])
        point_paths = "|".join(str(item["path"]) for item in well_cfg.get("point_sources", []))
        source_rows.append({
            "WellName": well, "DensityKind": well_cfg["density_kind"], "FractureScope": well_cfg["fracture_scope"],
            "SupervisionStatus": well_cfg["supervision_status"], "DensitySourcePaths": density_paths,
            "PointSourcePaths": point_paths, "DensityRows": int(len(density)), "PointRows": int(len(points)),
            "Windows": "|".join(window_id for window_id, _, _ in windows),
            "DensitySourceConflicts": int(density_conflicts),
        })

        segments = manifest[manifest["WellName"].eq(well)]
        if segments.empty:
            wells_without_groups.append({
                "WellName": well, "GroupCount": 0,
                "Step2Status": "rejected",
                "Step2RejectReason": rejected_reason.get(well, "no_step2_segments"),
            })
            continue

        frames = []
        for seg in segments.itertuples(index=False):
            seg_df = pd.read_csv(seg.OutputFilePath, encoding="utf-8-sig")
            seg_df["InputSegmentPath"] = str(seg.OutputFilePath)
            frames.append(seg_df)
        base = pd.concat(frames, ignore_index=True).sort_values("TVD").reset_index(drop=True)
        coverage = merge_ranges([(float(frame.TVD.min()), float(frame.TVD.max())) for frame in frames])
        coverage_m = float(sum(hi - lo for lo, hi in coverage))
        density_depth = density.TVD.to_numpy(dtype=float)
        density_gap_m = gap_break_for(density_depth)
        inside_mask = np.zeros(len(density), dtype=bool)
        for lo, hi in coverage:
            inside_mask |= (density_depth >= lo) & (density_depth <= hi)
        labels = base[["MD", "TVD", "StrataName", "InputSegmentPath"]].copy()
        labels["WellName"] = well
        labels["ImagingWindowID"] = lookup_window_id(labels.TVD.to_numpy(dtype=float), windows)
        labels["Density"] = interpolate_density(density, labels["TVD"], float(well_cfg["max_density_interpolation_gap_m"]))
        in_window = labels["ImagingWindowID"].astype(str).ne("").to_numpy()
        labels["DensitySupportStatus"] = np.where(
            ~in_window,
            "outside_imaging_window",
            np.where(labels["Density"].notna(), "supported", "missing_density_within_window"),
        )
        labels, unmapped, attach_stats = attach_points(labels, points, coverage)
        supervision_tier = str(
            well_cfg.get("supervision_tier")
            or DEFAULT_SUPERVISION_TIERS.get(str(well_cfg["supervision_status"]), "audit_only")
        )

        # ---- 成像解释窗口登记 + 按窗口标定（"每条缝 = 1 条"）----
        # 每个文件是一次独立解释作业，刻度按文件（窗口）算；同时保留按井汇总结算做对照。
        mapped = points.loc[~points.PointRowIndex.isin(unmapped.PointRowIndex)] if len(points) else points
        mapped_points = int(len(mapped))

        def window_mass(window_id: str, lo: float, hi: float) -> float:
            total = 0.0
            for a, b in coverage:
                low, high = max(a, lo), min(b, hi)
                if high <= low:
                    continue
                subset = density[(density.TVD >= low) & (density.TVD <= high)]
                if len(subset) >= 2:
                    total += integrate_runs(
                        subset.TVD.to_numpy(dtype=float), subset.Density.to_numpy(dtype=float), density_gap_m
                    )
            return total

        window_scale: dict[str, float] = {}
        well_mass_total = 0.0
        well_points_total = 0
        for window_id, lo, hi in windows:
            mass = window_mass(window_id, lo, hi)
            points_in_window = int(((mapped.TVD >= lo) & (mapped.TVD <= hi)).sum()) if len(mapped) else 0
            well_mass_total += mass
            well_points_total += points_in_window
            scale = float(points_in_window / mass) if (mass > 1.0e-9 and points_in_window > 0) else 1.0
            window_scale[window_id] = scale
            window_rows.append({
                "WellName": well, "WindowID": window_id,
                "WindowStartM": round(lo, 2), "WindowEndM": round(hi, 2),
                "WindowThicknessM": round(hi - lo, 2),
                "DensitySourcePath": window_source_path.get(window_id, ""),
                "DensityIntegralInLogCoverage": round(mass, 3),
                "PointRowsMappedInWindow": points_in_window,
                "DensityScaleFactor": round(scale, 4),
                "DensityIntegralCalibrated": round(mass * scale, 3),
                "Note": "ok" if points_in_window > 0 else "no_points_in_window",
            })
        density_scale = (
            float(well_points_total / well_mass_total) if (well_mass_total > 1.0e-9 and well_points_total > 0) else 1.0
        )
        labels["DensityRaw"] = labels["Density"]
        labels["DensityScaleFactor"] = [
            window_scale.get(str(value), 1.0) for value in labels["ImagingWindowID"]
        ]
        labels["Density"] = labels["DensityRaw"] * labels["DensityScaleFactor"]

        # ---- 段外成像数据审计：密度段外部分 + 未吸附的产状点 ----
        outside = density[~inside_mask]
        for lo, hi in runs_of(outside.TVD.to_numpy(dtype=float), density_gap_m):
            if hi - lo <= 0.05:
                continue
            subset = outside[(outside.TVD >= lo) & (outside.TVD <= hi)]
            out_of_coverage_rows.append({
                "WellName": well, "DataType": "density", "TVDStart": lo, "TVDEnd": hi,
                "ThicknessM": round(hi - lo, 2),
                "DensityMass": round(integrate_runs(subset.TVD.to_numpy(dtype=float),
                                                    subset.Density.to_numpy(dtype=float), density_gap_m), 2),
                "PointCount": 0,
                "Reason": "outside_log_coverage",
            })
        for reason, subset in unmapped.groupby("Reason", sort=False):
            if subset.empty:
                continue
            unmapped_rows.extend(subset.assign(WellName=well).to_dict("records"))
            if reason == "outside_log_coverage":
                depths = np.sort(subset.TVD.to_numpy(dtype=float))
                start = depths[0]
                for index in range(1, len(depths) + 1):
                    if index == len(depths) or depths[index] - depths[index - 1] > 5.0:
                        end = depths[index - 1]
                        count = int(((depths >= start) & (depths <= end)).sum())
                        out_of_coverage_rows.append({
                            "WellName": well, "DataType": "point",
                            "TVDStart": float(start), "TVDEnd": float(end),
                            "ThicknessM": round(float(end - start), 2),
                            "DensityMass": 0.0, "PointCount": count,
                            "Reason": "outside_log_coverage",
                        })
                        if index < len(depths):
                            start = depths[index]
        calibration_rows.append({
            "WellName": well, "DensityKind": well_cfg["density_kind"],
            "SupervisionStatus": well_cfg["supervision_status"],
            "WindowCount": len(windows),
            "WindowIDs": "|".join(window_id for window_id, _, _ in windows),
            "WindowScaleFactors": "|".join(f"{w}:{window_scale[w]:.4f}" for w, _, _ in windows),
            "LogCoverageM": round(coverage_m, 1),
            "DensityRowsInCoverage": int(inside_mask.sum()),
            "DensityIntegralRaw": round(well_mass_total, 3),
            "PointRowsTotal": int(len(points)),
            "PointRowsMapped": mapped_points,
            "DensityScaleFactor": round(density_scale, 4),
            "DensityIntegralCalibrated": round(well_mass_total * density_scale, 3),
            "PointsPerMeter": round(mapped_points / coverage_m, 4) if coverage_m > 0 else None,
            "DensitySourceConflicts": int(density_conflicts),
            "Note": "ok" if mapped_points > 0 else "no_points_density_only",
        })

        for coverage_index, group in enumerate(split_groups(labels), start=1):
            group_id = f"{well}_g{coverage_index:03d}_{group.StrataName.iloc[0]}"
            group["GroupID"] = group_id
            group["HasFractureDensity"] = group["Density"].gt(0.0).astype(int)
            group["DensityKind"] = well_cfg["density_kind"]
            group["FractureScope"] = well_cfg["fracture_scope"]
            group["DensitySourcePaths"] = density_paths
            group["PointSourcePaths"] = point_paths
            group["SupervisionStatus"] = well_cfg["supervision_status"]
            group["SupervisionTier"] = supervision_tier
            group_window_id = str(group.ImagingWindowID.iloc[0])
            group_scale = float(window_scale.get(group_window_id, density_scale))
            group = group[LABEL_COLUMNS]
            group_path = groups_dir / f"{group_id}.csv"
            group.to_csv(group_path, index=False, encoding="utf-8-sig")
            group_rows.append({
                "GroupID": group_id, "WellName": well, "DensityCoverageSegmentID": coverage_index,
                "StrataName": group.StrataName.iloc[0], "MDMin": float(group.MD.min()), "MDMax": float(group.MD.max()),
                "TVDMin": float(group.TVD.min()), "TVDMax": float(group.TVD.max()),
                "Rows": int(len(group)),
                "DensityPositiveRows": int(group.HasFractureDensity.sum()),
                "DensityZeroRows": int(group.Density.eq(0.0).sum()),
                "DensityMin": float(group.Density.min()),
                "DensityMedian": float(group.Density.median()),
                "DensityMax": float(group.Density.max()),
                "DensityPositiveRatio": float(group.HasFractureDensity.mean()),
                "GTPointRows": int(group.GT_POINT_FLAG.sum()), "InputSegmentPaths": "|".join(sorted(group.InputSegmentPath.dropna().unique().astype(str))),
                "CoLocatedPointCount": int(max(0, int(group.RawPointCount.sum()) - int(group.GT_POINT_FLAG.sum()))),
                "GroupPath": str(group_path), "DensityKind": well_cfg["density_kind"], "FractureScope": well_cfg["fracture_scope"],
                "SupervisionStatus": well_cfg["supervision_status"],
                "SupervisionTier": supervision_tier,
                "ImagingWindowID": group_window_id,
                "DensityScaleFactor": group_scale,
            })
        mapped_inside = int(labels.loc[labels.GT_POINT_FLAG.eq(1) & labels.Density.notna(), "RawPointCount"].sum())
        mapped_outside = int(labels.loc[labels.GT_POINT_FLAG.eq(1) & labels.Density.isna(), "RawPointCount"].sum())
        point_qc_rows.append({
            "WellName": well, "InputRows": int(len(base)),
            "DensitySupportedRows": int(labels.Density.notna().sum()),
            "TotalPointRows": int(len(points)),
            "MappedPointRows": int(labels.GT_POINT_FLAG.sum()),
            "MappedUniquePoints": mapped_points,
            "MappedPointsInsideDensitySupport": mapped_inside,
            "MappedPointsOutsideDensitySupport": mapped_outside,
            "DroppedPointRows": int(len(unmapped)),
            "DroppedOutsideLogCoverage": int((unmapped.Reason == "outside_log_coverage").sum()) if len(unmapped) else 0,
            "DroppedBeyondGridTolerance": 0,
            "GridStepM": round(float(attach_stats["GridStepM"]), 5) if attach_stats["GridStepM"] == attach_stats["GridStepM"] else None,
            "AttachResidualMedianM": round(float(attach_stats["AttachResidualMedianM"]), 4) if attach_stats["AttachResidualMedianM"] == attach_stats["AttachResidualMedianM"] else None,
            "AttachResidualP95M": round(float(attach_stats["AttachResidualP95M"]), 4) if attach_stats["AttachResidualP95M"] == attach_stats["AttachResidualP95M"] else None,
            "AttachResidualMaxM": round(float(attach_stats["AttachResidualMaxM"]), 4) if attach_stats["AttachResidualMaxM"] == attach_stats["AttachResidualMaxM"] else None,
            "AttachedBeyondHalfStepRows": int(attach_stats["AttachedBeyondHalfStepRows"]),
            "SupervisionTier": supervision_tier,
        })

    group_df = pd.DataFrame(group_rows)
    source_df = pd.DataFrame(source_rows)
    point_qc = pd.DataFrame(point_qc_rows)
    density_source_df = pd.DataFrame(density_source_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    window_df = pd.DataFrame(window_rows)
    out_of_coverage_df = pd.DataFrame(out_of_coverage_rows)
    unmapped_df = pd.DataFrame(
        unmapped_rows,
        columns=["WellName", "PointRowIndex", "TVD", "FracAzimuth", "FracDip", "PointSourcePath",
                 "Reason", "NearestGridTVD", "DistanceToGridM"],
    )
    group_df.to_csv(output / "sample_group_manifest.csv", index=False, encoding="utf-8-sig")
    source_df.to_csv(output / "source_manifest.csv", index=False, encoding="utf-8-sig")
    point_qc.to_csv(output / "point_mapping_qc.csv", index=False, encoding="utf-8-sig")
    density_source_df.to_csv(output / "density_source_merge_audit.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(output / "density_calibration_audit.csv", index=False, encoding="utf-8-sig")
    window_df.to_csv(output / "imaging_window_registry.csv", index=False, encoding="utf-8-sig")
    out_of_coverage_df.to_csv(output / "imaging_out_of_log_coverage.csv", index=False, encoding="utf-8-sig")
    unmapped_df.to_csv(output / "unmapped_imaging_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(wells_without_groups).to_csv(output / "wells_without_groups.csv", index=False, encoding="utf-8-sig")

    # 解释段 TVD 覆盖完整度（成像井，well 级）
    well_meta = pd.read_csv(step2_root / "taigu_step2_well_metadata.csv", encoding="utf-8-sig")
    interpreted: dict[str, list[list[float]]] = {}
    for _, r in well_meta.iterrows():
        iv = r.InterpretedTVDIntervals
        if isinstance(iv, str) and iv.strip():
            try:
                interpreted[str(r.WellName)] = [[float(a), float(b)] for a, b in json.loads(iv)]
            except Exception:
                continue

    coverage_gaps: list[dict[str, object]] = []
    for well, intervals in interpreted.items():
        groups = group_df[group_df.WellName.eq(well)] if not group_df.empty else pd.DataFrame()
        if groups.empty:
            continue
        merged = merge_ranges([(float(g.TVDMin), float(g.TVDMax)) for _, g in groups.iterrows()])
        for a, b in intervals:
            covered = sum(max(0.0, min(hi, b) - max(lo, a)) for lo, hi in merged if max(lo, a) < min(hi, b))
            uncovered = (b - a) - covered
            if uncovered > 1.0:
                coverage_gaps.append({
                    "WellName": well, "TVDInterval": [a, b],
                    "CoveredM": round(covered, 2), "UncoveredM": round(uncovered, 2),
                    "UncoveredRatio": round(uncovered / max(b - a, 1e-9), 3),
                })

    configured_wells_status: list[dict[str, object]] = []
    for well_cfg in cfg["wells"]:
        well = well_cfg["well_name"]
        well_groups = group_df[group_df.WellName.eq(well)] if not group_df.empty else pd.DataFrame()
        if well_groups.empty:
            configured_wells_status.append({
                "WellName": well, "Status": "upstream_rejected",
                "Step2RejectReason": rejected_reason.get(well, "no_step2_segments"),
            })
        else:
            configured_wells_status.append({
                "WellName": well, "Status": str(well_groups.iloc[0].SupervisionStatus),
                "GroupCount": int(len(well_groups)),
            })

    summary = {
        "version": "taigu_step3_imaging_v4",
        "group_count": int(len(group_df)),
        "well_count": int(group_df.WellName.nunique()) if not group_df.empty else 0,
        "group_rows": int(group_df.Rows.sum()) if not group_df.empty else 0,
        "candidate_scope_pending_groups": int(group_df.SupervisionStatus.eq("candidate_scope_pending").sum()) if not group_df.empty else 0,
        "wells_configured": len(cfg["wells"]),
        "wells_without_groups": wells_without_groups,
        "supervision_status_counts": {str(k): int(v) for k, v in group_df.SupervisionStatus.value_counts().items()} if not group_df.empty else {},
        "configured_wells_status": configured_wells_status,
        "coverage_gaps": coverage_gaps,
        "density_merge_rule": "clip_each_source_to_its_imaging_window_then_max_per_tvd",
        "density_calibration_rule": "per_window: scale = points_in_window_in_log_coverage / density_integral_in_window_in_log_coverage; well-level pooled value reported for comparison",
        "point_attach_rule": "accept every point inside log coverage, snap to nearest sample, record residual (no tolerance rejection)",
        "supervision_tiers": {well: str(well_cfg.get("supervision_tier") or DEFAULT_SUPERVISION_TIERS.get(str(well_cfg["supervision_status"]), "audit_only")) for well_cfg in cfg["wells"]},
        "density_calibration": calibration_rows,
        "imaging_windows": window_rows,
        "out_of_log_coverage": out_of_coverage_rows,
        "unmapped_point_rows": int(len(unmapped_df)),
        "point_mapping_scope": "well_level_unique",
        "depth_semantics": "labels_and_segments_in_tvd",
    }
    (output / "step3_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
