"""按 MD 把 Step2 测井段坐标回接到任意点表，并逐点留审计。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .segment_pool import (
    WellSegmentPool,
    inferred_md_tolerance,
    list_segment_files,
)

MD_JOIN_AUDIT_COLUMNS = [
    "MDJoinStatus",
    "MDJoinReason",
    "MDJoinToleranceM",
    "Step2SegmentCount",
    "MDCandidateCount",
    "MDCandidateSegmentIDs",
    "MDCandidateDistancesM",
    "PreferredSegmentIDs",
    "PreferredSegmentMatched",
    "SelectedSegmentID",
    "SelectedSegmentPath",
    "SelectedSegmentMD",
    "SelectedSegmentTimeMs",
    "SelectedSegmentTVD",
    "MDNearestDifferenceM",
    "MDNearestDifferenceAnySegmentM",
    "NearestSegmentIDAnyDistance",
    "InsideSelectedSegmentMDRange",
    "SegmentSelectionPolicy",
    "MDJoinDuplicateRow",
]

PoolResolver = Callable[[str], "WellSegmentPool | None"]


def _split_preferred(raw: Any) -> list[str]:
    """行内显式来源段（可以是路径或段名，分号分隔多条）。"""
    if not isinstance(raw, str) or not raw.strip():
        return []
    return sorted({Path(part).stem for part in raw.split(";") if part.strip()})


def attach_geometry_by_md(
    points: pd.DataFrame,
    pool_resolver: PoolResolver,
    tolerance_m: float | None = None,
    policy: str = "prefer_source_segment",
    preferred_column: str | None = None,
    preferred_ids_by_row: pd.Series | None = None,
    geometry_source: str = "",
    overwrite: bool = False,
    existing_columns: tuple[str, ...] = ("X", "Y", "TIME"),
    half_step_multiplier: float = 0.5,
    tolerance_floor_m: float = 0.02,
    tolerance_cap_m: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 Step2 段的 X/Y/TIME 回接到 `points` 的每一行。

    约定（P0-4）：

    * **一个输入点产出一行**，行序与输入完全一致，不重排、不重新编号；
    * 选取规则：行内显式来源段 > `preferred_ids_by_row` 来源段 > |ΔMD| 最小 > 段序；
    * 容差默认按该井 MD 半采样步长自动推导（`tolerance_m=None` 时）；
    * 未命中的点**不删除**，照原样返回并在审计表里写明原因；
    * `overwrite=False` 时，已有完整 X/Y/TIME 的行原样保留（审计状态 `existing_geometry`）。

    返回 `(matched, audit)`：`matched` 为回接后的点表，`audit` 每个输入点一行。
    """
    if "WellName" not in points.columns or "MD" not in points.columns:
        raise ValueError("attach_geometry_by_md requires WellName and MD columns")

    records = points.to_dict("records")
    row_count = len(records)
    well_values = points["WellName"].astype(str).to_numpy()
    md_values = pd.to_numeric(points["MD"], errors="coerce").to_numpy(dtype=float)
    tvd_values = (
        pd.to_numeric(points["TVD"], errors="coerce").to_numpy(dtype=float)
        if "TVD" in points.columns
        else np.full(row_count, np.nan)
    )
    strata_values = points["StrataName"].to_numpy() if "StrataName" in points.columns else np.full(row_count, np.nan)
    support_values = (
        points["SupportSegmentID"].to_numpy() if "SupportSegmentID" in points.columns else np.full(row_count, np.nan)
    )
    if preferred_ids_by_row is not None:
        preferred_values = preferred_ids_by_row.reindex(points.index).fillna("").astype(str).to_numpy()
    elif preferred_column and preferred_column in points.columns:
        preferred_values = points[preferred_column].fillna("").astype(str).to_numpy()
    else:
        preferred_values = np.array([""] * row_count, dtype=object)
    existing = pd.DataFrame(
        {
            column: (pd.to_numeric(points[column], errors="coerce").to_numpy(dtype=float) if column in points.columns else np.full(row_count, np.nan))
            for column in existing_columns
        }
    )

    matched_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, float]] = set()
    pool_cache: dict[str, WellSegmentPool | None] = {}

    for position in range(row_count):
        point = records[position]
        well = str(well_values[position])
        md = float(md_values[position])
        preferred_ids = _split_preferred(preferred_values[position])
        if well not in pool_cache:
            pool_cache[well] = pool_resolver(well)
        pool = pool_cache[well]
        tolerance = (
            float(tolerance_m)
            if tolerance_m is not None
            else inferred_md_tolerance(
                pool,
                half_step_multiplier=half_step_multiplier,
                floor_m=tolerance_floor_m,
                cap_m=tolerance_cap_m,
            )
        )
        audit: dict[str, Any] = {
            "WellName": well,
            "MD": md,
            "TVD": float(tvd_values[position]),
            "StrataName": strata_values[position],
            "X": np.nan,
            "Y": np.nan,
            "GeometrySource": geometry_source,
            "MDJoinToleranceM": float(tolerance),
            "SegmentSelectionPolicy": policy,
            "PreferredSegmentIDs": ";".join(preferred_ids),
            "PreferredSegmentMatched": 0,
            "Step4SourceSegmentMatched": 0,
            "SelectedSegmentID": "",
            "SelectedSegmentPath": "",
            "SelectedSegmentMD": np.nan,
            "SelectedSegmentTimeMs": np.nan,
            "SelectedSegmentTVD": np.nan,
            "MDNearestDifferenceM": np.nan,
            "MDNearestDifferenceAnySegmentM": np.nan,
            "NearestSegmentIDAnyDistance": "",
            "InsideSelectedSegmentMDRange": 0,
            "MDCandidateCount": 0,
            "MDCandidateSegmentIDs": "",
            "MDCandidateDistancesM": "",
            "Step2SegmentCount": 0,
            "InputRowIndex": int(position),
            "MDJoinDuplicateRow": 0,
            "Step4SupportSegmentID": support_values[position],
        }
        key = (well, md)
        if key in seen_keys:
            audit["MDJoinDuplicateRow"] = 1
        else:
            seen_keys.add(key)

        row = dict(point)
        already_complete = bool(np.isfinite(existing.to_numpy(dtype=float)[position]).all())
        if already_complete and not overwrite:
            audit["MDJoinStatus"] = "existing_geometry"
            audit["MDJoinReason"] = "input_already_has_xy_time"
            for column in existing_columns:
                row[column] = float(existing[column].to_numpy(dtype=float)[position])
                audit[column] = float(existing[column].to_numpy(dtype=float)[position])
            matched_rows.append(row)
            audit_rows.append(audit)
            continue

        if pool is None:
            audit["MDJoinStatus"] = "unmatched"
            audit["MDJoinReason"] = "step2_well_directory_has_no_usable_segment_csv"
            audit["Step2SegmentCount"] = 0
            matched_rows.append(row)
            audit_rows.append(audit)
            continue

        audit["Step2SegmentCount"] = int(len(pool.segment_ids_unique))
        audit.update(pool.nearest_any(md))
        candidates = pool.candidates(md, tolerance)
        if candidates.empty:
            audit["MDJoinStatus"] = "unmatched"
            audit["MDJoinReason"] = (
                "md_not_finite" if not np.isfinite(md) else "no_segment_row_within_md_tolerance"
            )
            matched_rows.append(row)
            audit_rows.append(audit)
            continue

        ordered = candidates.sort_values(["MDMatchDistanceM", "SegmentOrder"], kind="mergesort")
        audit["MDCandidateCount"] = int(len(ordered))
        audit["MDCandidateSegmentIDs"] = "|".join(ordered["SegmentID"].astype(str).tolist()[:8])
        audit["MDCandidateDistancesM"] = "|".join(f"{value:.6f}" for value in ordered["MDMatchDistanceM"].tolist()[:8])
        if policy == "prefer_source_segment" and preferred_ids:
            mask = ordered["SegmentID"].astype(str).isin(set(preferred_ids))
            if mask.any():
                ordered = ordered[mask]
                audit["PreferredSegmentMatched"] = 1
                audit["Step4SourceSegmentMatched"] = 1

        chosen = ordered.iloc[0]
        chosen_id = str(chosen["SegmentID"])
        md_min, md_max = pool.md_ranges.get(chosen_id, (np.nan, np.nan))
        audit.update(
            {
                "MDJoinStatus": "matched",
                "MDJoinReason": "nearest_segment_row_within_md_tolerance",
                "X": float(chosen["X"]) if pd.notna(chosen.get("X")) else np.nan,
                "Y": float(chosen["Y"]) if pd.notna(chosen.get("Y")) else np.nan,
                "SelectedSegmentID": chosen_id,
                "SelectedSegmentPath": str(chosen.get("SegmentPath", "")),
                "SelectedSegmentMD": float(chosen["MD"]),
                "SelectedSegmentTimeMs": float(chosen["TIME"]) if pd.notna(chosen.get("TIME")) else np.nan,
                "SelectedSegmentTVD": float(chosen["TVD"]) if pd.notna(chosen.get("TVD")) else np.nan,
                "MDNearestDifferenceM": float(chosen["MDMatchDistanceM"]),
                "InsideSelectedSegmentMDRange": int(
                    np.isfinite(md_min) and np.isfinite(md_max) and md_min <= md <= md_max
                ),
            }
        )
        row["X"] = audit["X"]
        row["Y"] = audit["Y"]
        row["TIME"] = float(chosen["TIME"]) if pd.notna(chosen.get("TIME")) else np.nan
        if "TVD" in chosen.index and pd.notna(chosen["TVD"]):
            row["TVD_Step2"] = float(chosen["TVD"])
        if "StrataName" in chosen.index and isinstance(chosen["StrataName"], str):
            row["StrataName_Step2"] = chosen["StrataName"]
        row["InputSegmentPath"] = str(chosen.get("SegmentPath", ""))
        row["SelectedSegmentID"] = chosen_id
        row["SelectedSegmentPath"] = str(chosen.get("SegmentPath", ""))
        row["MDNearestDifferenceM"] = float(chosen["MDMatchDistanceM"])
        row["MDCandidateCount"] = int(len(candidates))
        row["MDJoinToleranceM"] = float(tolerance)
        row["MDJoinStatus"] = "matched"
        row["SegmentSelectionPolicy"] = policy
        matched_rows.append(row)
        audit_rows.append(audit)

    matched = pd.DataFrame(matched_rows)
    audit_df = pd.DataFrame(audit_rows)
    return matched, audit_df


def summarize_join(audit: pd.DataFrame) -> dict[str, Any]:
    """把审计表压缩成可直接放进各步 summary 的统计块。"""
    if audit is None or audit.empty:
        return {"point_count": 0, "matched_count": 0, "unmatched_count": 0}
    status = audit["MDJoinStatus"].astype(str)
    matched_mask = status.isin(["matched", "existing_geometry"])
    distances = pd.to_numeric(audit.get("MDNearestDifferenceM"), errors="coerce") if "MDNearestDifferenceM" in audit.columns else pd.Series(dtype=float)
    distances = distances[distances.notna()]
    summary: dict[str, Any] = {
        "point_count": int(len(audit)),
        "matched_count": int(matched_mask.sum()),
        "existing_geometry_count": int(status.eq("existing_geometry").sum()),
        "unmatched_count": int((~matched_mask).sum()),
        "join_status_counts": status.value_counts().to_dict(),
        "unmatched_reason_counts": audit.loc[~matched_mask, "MDJoinReason"].astype(str).value_counts().to_dict()
        if "MDJoinReason" in audit.columns
        else {},
        "duplicate_md_count": int(pd.to_numeric(audit.get("MDJoinDuplicateRow"), errors="coerce").fillna(0).sum())
        if "MDJoinDuplicateRow" in audit.columns
        else 0,
        "md_match_distance_stats": {
            "count": int(distances.size),
            "min": float(distances.min()) if distances.size else None,
            "median": float(distances.median()) if distances.size else None,
            "p95": float(distances.quantile(0.95)) if distances.size else None,
            "max": float(distances.max()) if distances.size else None,
        },
        "selected_segment_counts": audit.loc[matched_mask, "SelectedSegmentID"].astype(str).value_counts().to_dict()
        if "SelectedSegmentID" in audit.columns
        else {},
        "tolerance_stats": {
            "min": float(pd.to_numeric(audit["MDJoinToleranceM"], errors="coerce").min())
            if "MDJoinToleranceM" in audit.columns
            else None,
            "max": float(pd.to_numeric(audit["MDJoinToleranceM"], errors="coerce").max())
            if "MDJoinToleranceM" in audit.columns
            else None,
        },
    }
    if "PreferredSegmentMatched" in audit.columns:
        summary["preferred_segment_matched_count"] = int(
            pd.to_numeric(audit["PreferredSegmentMatched"], errors="coerce").fillna(0).sum()
        )
    return summary


def segment_file_inventory(segment_dir: Path) -> dict[str, Any]:
    """轻量目录清单，供各步在启动时确认段文件可读。"""
    files = list_segment_files(segment_dir)
    return {
        "segment_dir": str(segment_dir),
        "segment_file_count": int(len(files)),
        "segment_files": [path.name for path in files[:20]],
    }
