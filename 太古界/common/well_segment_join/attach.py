"""逐段回接：按每行自带的测井段，在该段内取 X/Y/TIME，并逐点留审计。

设计口径（2026-09-16 确认）：

* 建模单元是**测井段**（同一批次观测）；Step4 的输出以**井**为单位，
  因此 Step4 的密度曲线与裂缝点都直接携带井级 `X/Y/TIME`，下游不需要回接；
* 仍需回接的只有 **Step3 成像监督组行**：它们按设计就是逐段行（带
  `InputSegmentPath`），但只有 MD/TVD，需要在各自所属段内取坐标；
* 回接必须**逐段进行**，不跨段匹配、不做段间平均——保证同一批次内部一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .segment_pool import inferred_md_tolerance_from_md, read_csv_flexible

MD_JOIN_AUDIT_COLUMNS = [
    "MDJoinStatus",
    "MDJoinReason",
    "MDJoinToleranceM",
    "SelectedSegmentID",
    "SelectedSegmentPath",
    "SelectedSegmentMD",
    "SelectedSegmentTimeMs",
    "SelectedSegmentTVD",
    "MDNearestDifferenceM",
    "InsideSelectedSegmentMDRange",
    "SegmentSelectionPolicy",
    "MDJoinDuplicateRow",
]

_SEGMENT_TABLE_CACHE: dict[str, pd.DataFrame] = {}


def _segment_key(raw: Any) -> str:
    """行内来源段标识统一成段名（文件名去扩展名）。"""
    text = str(raw).strip()
    return Path(text).stem if text else ""


def _resolve_segment_path(key: str, raw: str, samples_root: Path | None) -> Path | None:
    direct = Path(raw)
    if direct.exists():
        return direct
    if samples_root is not None and Path(samples_root).exists():
        matches = sorted(Path(samples_root).rglob(f"{key}.csv"))
        if matches:
            return matches[0]
    return None


def _load_segment_table(path: Path) -> pd.DataFrame:
    key = str(path)
    if key not in _SEGMENT_TABLE_CACHE:
        table = read_csv_flexible(path, low_memory=False)
        required = {"MD", "X", "Y", "TIME"}
        if not required.issubset(table.columns):
            raise ValueError(f"segment file missing columns {sorted(required - set(table.columns))}: {path}")
        for column in ("MD", "X", "Y", "TIME", "TVD"):
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce")
        table = table.dropna(subset=["MD"]).sort_values("MD", kind="mergesort").reset_index(drop=True)
        _SEGMENT_TABLE_CACHE[key] = table
    return _SEGMENT_TABLE_CACHE[key]


def _nearest_in_segment(table: pd.DataFrame, md: float, tolerance: float) -> tuple[pd.Series | None, float]:
    grid = table["MD"].to_numpy(dtype=float)
    if grid.size == 0 or not np.isfinite(md):
        return None, float("nan")
    position = int(np.searchsorted(grid, md))
    best_index, best_distance = None, float("inf")
    for candidate in (position - 1, position):
        if 0 <= candidate < grid.size:
            distance = abs(float(grid[candidate]) - md)
            if distance <= tolerance and distance < best_distance:
                best_index, best_distance = candidate, distance
    if best_index is None:
        return None, best_distance if np.isfinite(best_distance) else float("nan")
    return table.iloc[best_index], best_distance


def attach_geometry_per_segment(
    points: pd.DataFrame,
    preferred_column: str = "InputSegmentPath",
    tolerance_m: float | None = None,
    geometry_source: str = "",
    samples_root: Path | None = None,
    half_step_multiplier: float = 0.5,
    tolerance_floor_m: float = 0.02,
    tolerance_cap_m: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 `preferred_column` 指定的测井段，逐段把 X/Y/TIME 回接到每一行。

    约定：

    * **一个输入点产出一行，行序与输入一致**；
    * 只在"该行自己所属的段"内匹配，不跨段、不平均；
    * 容差默认按该段 MD 半采样步长自动推导；
    * 未命中（段文件缺失、MD 超容差）不删除该行，只写审计。
    """
    if "WellName" not in points.columns or "MD" not in points.columns:
        raise ValueError("attach_geometry_per_segment requires WellName and MD columns")
    if preferred_column not in points.columns:
        raise ValueError(f"attach_geometry_per_segment requires column {preferred_column}")

    records = points.to_dict("records")
    row_count = len(records)
    wells = points["WellName"].astype(str).to_numpy()
    mds = pd.to_numeric(points["MD"], errors="coerce").to_numpy(dtype=float)
    tvds = (
        pd.to_numeric(points["TVD"], errors="coerce").to_numpy(dtype=float)
        if "TVD" in points.columns
        else np.full(row_count, np.nan)
    )
    strata = points["StrataName"].to_numpy() if "StrataName" in points.columns else np.full(row_count, np.nan)
    raw_sources = points[preferred_column].fillna("").astype(str).to_numpy()

    # 按段分组，逐段加载一次；返回时恢复原始行序。
    groups: dict[str, list[int]] = {}
    for position, raw in enumerate(raw_sources):
        groups.setdefault(_segment_key(raw), []).append(position)

    matched_rows: list[dict[str, Any] | None] = [None] * row_count
    audit_rows: list[dict[str, Any] | None] = [None] * row_count
    seen: set[tuple[str, float]] = set()

    for key, positions in groups.items():
        raw = raw_sources[positions[0]]
        path = _resolve_segment_path(key, raw, samples_root) if key else None
        table = _load_segment_table(path) if path is not None else None
        segment_md = table["MD"].to_numpy(dtype=float) if table is not None else np.array([])
        tolerance = (
            float(tolerance_m)
            if tolerance_m is not None
            else inferred_md_tolerance_from_md(
                segment_md,
                half_step_multiplier=half_step_multiplier,
                floor_m=tolerance_floor_m,
                cap_m=tolerance_cap_m,
            )
        )
        md_min = float(segment_md.min()) if segment_md.size else np.nan
        md_max = float(segment_md.max()) if segment_md.size else np.nan
        for position in positions:
            well, md = wells[position], float(mds[position])
            audit: dict[str, Any] = {
                "WellName": well,
                "MD": md,
                "TVD": float(tvds[position]),
                "StrataName": strata[position],
                "GeometrySource": geometry_source,
                "SourceSegmentID": key,
                "SourceSegmentPath": str(path) if path is not None else str(raw),
                "MDJoinToleranceM": float(tolerance),
                "SegmentSelectionPolicy": "per_segment_exact_within_batch",
                "SelectedSegmentID": key,
                "SelectedSegmentPath": str(path) if path is not None else "",
                "SelectedSegmentMD": np.nan,
                "SelectedSegmentTimeMs": np.nan,
                "SelectedSegmentTVD": np.nan,
                "MDNearestDifferenceM": np.nan,
                "InsideSelectedSegmentMDRange": 0,
                "MDJoinDuplicateRow": 0,
                "X": np.nan,
                "Y": np.nan,
            }
            dedupe_key = (well, md)
            if dedupe_key in seen:
                audit["MDJoinDuplicateRow"] = 1
            else:
                seen.add(dedupe_key)
            row = dict(records[position])

            if path is None or table is None:
                audit["MDJoinStatus"] = "unmatched"
                audit["MDJoinReason"] = "segment_file_missing"
            else:
                audit["InsideSelectedSegmentMDRange"] = int(
                    np.isfinite(md_min) and np.isfinite(md_max) and md_min <= md <= md_max
                )
                chosen, distance = _nearest_in_segment(table, md, tolerance)
                if chosen is None:
                    audit["MDJoinStatus"] = "unmatched"
                    audit["MDJoinReason"] = (
                        "md_not_finite" if not np.isfinite(md) else "no_segment_row_within_md_tolerance"
                    )
                    audit["MDNearestDifferenceM"] = distance
                else:
                    audit.update(
                        {
                            "MDJoinStatus": "matched",
                            "MDJoinReason": "within_segment_nearest_row_within_tolerance",
                            "SelectedSegmentMD": float(chosen["MD"]),
                            "SelectedSegmentTimeMs": float(chosen["TIME"]),
                            "SelectedSegmentTVD": float(chosen["TVD"])
                            if "TVD" in chosen.index and pd.notna(chosen["TVD"])
                            else np.nan,
                            "MDNearestDifferenceM": float(distance),
                            "X": float(chosen["X"]),
                            "Y": float(chosen["Y"]),
                        }
                    )
                    row["X"] = float(chosen["X"])
                    row["Y"] = float(chosen["Y"])
                    row["TIME"] = float(chosen["TIME"])
                    if "TVD" in chosen.index and pd.notna(chosen["TVD"]):
                        row["TVD_Step2"] = float(chosen["TVD"])
                    row["InputSegmentPath"] = str(path)
                    row["MDJoinStatus"] = "matched"
                    row["MDJoinToleranceM"] = float(tolerance)
                    row["MDNearestDifferenceM"] = float(distance)
                    row["SegmentSelectionPolicy"] = "per_segment_exact_within_batch"
            matched_rows[position] = row
            audit_rows[position] = audit

    return pd.DataFrame(matched_rows), pd.DataFrame(audit_rows)


def summarize_join(audit: pd.DataFrame) -> dict[str, Any]:
    """把审计表压缩成可放进各步 summary 的统计块。"""
    if audit is None or audit.empty:
        return {"point_count": 0, "matched_count": 0, "unmatched_count": 0}
    status = audit["MDJoinStatus"].astype(str)
    matched_mask = status.eq("matched")
    distances = (
        pd.to_numeric(audit["MDNearestDifferenceM"], errors="coerce")
        if "MDNearestDifferenceM" in audit.columns
        else pd.Series(dtype=float)
    )
    distances = distances[distances.notna()]
    summary: dict[str, Any] = {
        "point_count": int(len(audit)),
        "matched_count": int(matched_mask.sum()),
        "unmatched_count": int((~matched_mask).sum()),
        "join_status_counts": status.value_counts().to_dict(),
        "unmatched_reason_counts": audit.loc[~matched_mask, "MDJoinReason"].astype(str).value_counts().to_dict()
        if "MDJoinReason" in audit.columns
        else {},
        "duplicate_md_count": int(
            pd.to_numeric(audit.get("MDJoinDuplicateRow"), errors="coerce").fillna(0).sum()
        )
        if "MDJoinDuplicateRow" in audit.columns
        else 0,
        "md_match_distance_stats": {
            "count": int(distances.size),
            "min": float(distances.min()) if distances.size else None,
            "median": float(distances.median()) if distances.size else None,
            "p95": float(distances.quantile(0.95)) if distances.size else None,
            "max": float(distances.max()) if distances.size else None,
        },
        "segment_count": int(audit["SourceSegmentID"].astype(str).nunique())
        if "SourceSegmentID" in audit.columns
        else 0,
        "tolerance_stats": {
            "min": float(pd.to_numeric(audit["MDJoinToleranceM"], errors="coerce").min())
            if "MDJoinToleranceM" in audit.columns
            else None,
            "max": float(pd.to_numeric(audit["MDJoinToleranceM"], errors="coerce").max())
            if "MDJoinToleranceM" in audit.columns
            else None,
        },
    }
    return summary
