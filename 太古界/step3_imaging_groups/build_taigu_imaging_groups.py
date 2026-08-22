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
    "GroupID", "WellName", "InputSegmentPath", "MD", "TVD", "StrataName", "Density",
    "HasFractureDensity", "GT_POINT_FLAG", "RawPointCount", "FracAzimuth", "FracDip",
    "DensityKind", "FractureScope", "DensitySourcePaths", "PointSourcePaths",
    "DensitySupportStatus", "SupervisionStatus",
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


def read_density(item: dict[str, Any]) -> pd.DataFrame:
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
    df["DensitySourcePath"] = str(path)
    return df


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


def attach_points(labels: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach points once at well level by TVD; returns (labels, unique unmapped count)."""
    out = labels.copy()
    out["GT_POINT_FLAG"] = 0
    out["RawPointCount"] = 0
    out["FracAzimuth"] = np.nan
    out["FracDip"] = np.nan
    if out.empty or points.empty:
        return out, 0
    tvd = out["TVD"].to_numpy(float)
    step = float(np.median(np.diff(tvd))) if len(tvd) > 1 else 0.1
    tolerance = min(max(step * 0.51, 0.05), 0.25)
    dropped = 0
    for point in points.itertuples(index=False):
        pos = int(np.searchsorted(tvd, point.TVD))
        candidates = [idx for idx in (pos - 1, pos) if 0 <= idx < len(tvd)]
        if not candidates:
            dropped += 1
            continue
        idx = min(candidates, key=lambda value: abs(tvd[value] - point.TVD))
        if abs(tvd[idx] - point.TVD) > tolerance:
            dropped += 1
            continue
        out.loc[idx, "GT_POINT_FLAG"] = 1
        out.loc[idx, "RawPointCount"] += 1
        if pd.isna(out.loc[idx, "FracAzimuth"]):
            out.loc[idx, "FracAzimuth"] = point.FracAzimuth
            out.loc[idx, "FracDip"] = point.FracDip
    return out, dropped


def split_groups(labels: pd.DataFrame) -> list[pd.DataFrame]:
    usable = labels[labels["Density"].notna()].copy().sort_values("TVD").reset_index(drop=True)
    if usable.empty:
        return []
    step = float(np.median(np.diff(usable["TVD"]))) if len(usable) > 1 else 0.1
    new_group = usable["StrataName"].ne(usable["StrataName"].shift()) | usable["TVD"].diff().gt(max(step * 3.0, 0.5))
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
    wells_without_groups: list[dict[str, object]] = []

    for well_cfg in cfg["wells"]:
        well = well_cfg["well_name"]
        density_parts = [read_density(item) for item in well_cfg["density_sources"]]
        density = pd.concat(density_parts, ignore_index=True).sort_values("TVD").drop_duplicates("TVD", keep="last").reset_index(drop=True)
        points = (
            pd.concat([read_points(item) for item in well_cfg.get("point_sources", [])], ignore_index=True)
            if well_cfg.get("point_sources")
            else pd.DataFrame(columns=["TVD", "FracAzimuth", "FracDip", "PointSourcePath"])
        )
        density_paths = "|".join(str(item["path"]) for item in well_cfg["density_sources"])
        point_paths = "|".join(str(item["path"]) for item in well_cfg.get("point_sources", []))
        source_rows.append({
            "WellName": well, "DensityKind": well_cfg["density_kind"], "FractureScope": well_cfg["fracture_scope"],
            "SupervisionStatus": well_cfg["supervision_status"], "DensitySourcePaths": density_paths,
            "PointSourcePaths": point_paths, "DensityRows": int(len(density)), "PointRows": int(len(points)),
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
        labels = base[["MD", "TVD", "StrataName", "InputSegmentPath"]].copy()
        labels["WellName"] = well
        labels["Density"] = interpolate_density(density, labels["TVD"], float(well_cfg["max_density_interpolation_gap_m"]))
        labels["DensitySupportStatus"] = np.where(labels["Density"].notna(), "supported", "outside_density_coverage_or_gap")
        labels, dropped = attach_points(labels, points)

        for coverage_index, group in enumerate(split_groups(labels), start=1):
            group_id = f"{well}_g{coverage_index:03d}_{group.StrataName.iloc[0]}"
            group["GroupID"] = group_id
            group["HasFractureDensity"] = group["Density"].gt(0.0).astype(int)
            group["DensityKind"] = well_cfg["density_kind"]
            group["FractureScope"] = well_cfg["fracture_scope"]
            group["DensitySourcePaths"] = density_paths
            group["PointSourcePaths"] = point_paths
            group["SupervisionStatus"] = well_cfg["supervision_status"]
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
            })
        mapped_inside = int(labels.loc[labels.GT_POINT_FLAG.eq(1) & labels.Density.notna(), "RawPointCount"].sum())
        mapped_outside = int(labels.loc[labels.GT_POINT_FLAG.eq(1) & labels.Density.isna(), "RawPointCount"].sum())
        point_qc_rows.append({
            "WellName": well, "InputRows": int(len(base)),
            "DensitySupportedRows": int(labels.Density.notna().sum()),
            "TotalPointRows": int(len(points)),
            "MappedPointRows": int(labels.GT_POINT_FLAG.sum()),
            "MappedUniquePoints": int(len(points)) - int(dropped),
            "MappedPointsInsideDensitySupport": mapped_inside,
            "MappedPointsOutsideDensitySupport": mapped_outside,
            "DroppedPointRows": int(dropped),
        })

    group_df = pd.DataFrame(group_rows)
    source_df = pd.DataFrame(source_rows)
    point_qc = pd.DataFrame(point_qc_rows)
    group_df.to_csv(output / "sample_group_manifest.csv", index=False, encoding="utf-8-sig")
    source_df.to_csv(output / "source_manifest.csv", index=False, encoding="utf-8-sig")
    point_qc.to_csv(output / "point_mapping_qc.csv", index=False, encoding="utf-8-sig")
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

    def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for lo, hi in sorted((float(a), float(b)) for a, b in ranges if b >= a):
            if out and lo <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], hi))
            else:
                out.append((lo, hi))
        return out

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
        "version": "taigu_step3_imaging_v3",
        "group_count": int(len(group_df)),
        "well_count": int(group_df.WellName.nunique()) if not group_df.empty else 0,
        "group_rows": int(group_df.Rows.sum()) if not group_df.empty else 0,
        "candidate_scope_pending_groups": int(group_df.SupervisionStatus.eq("candidate_scope_pending").sum()) if not group_df.empty else 0,
        "wells_configured": len(cfg["wells"]),
        "wells_without_groups": wells_without_groups,
        "supervision_status_counts": {str(k): int(v) for k, v in group_df.SupervisionStatus.value_counts().items()} if not group_df.empty else {},
        "configured_wells_status": configured_wells_status,
        "coverage_gaps": coverage_gaps,
        "point_mapping_scope": "well_level_unique",
        "depth_semantics": "labels_and_segments_in_tvd",
    }
    (output / "step3_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
