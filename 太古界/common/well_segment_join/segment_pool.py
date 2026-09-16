"""Step2 测井段池：一口井的全部段栈在同一条 MD 轴上。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 段文件里需要带回来的几何/地层列；StrataName 只作参考，不覆盖调用方自己的地层。
SEGMENT_GEOMETRY_COLUMNS = ("MD", "X", "Y", "TIME", "TVD", "StrataName")
REQUIRED_SEGMENT_COLUMNS = ("MD", "X", "Y", "TIME")
SEGMENT_FILE_GLOBS = ("*_seg_*.csv", "*.csv")

# 容差按"半采样步长"推导：步长 0.125 m 的井 => 0.0625 m。
# 下限用于步长为 0 或极小的异常段，上限防止个别稀疏段把容差放得过大。
DEFAULT_HALF_STEP_MULTIPLIER = 0.5
DEFAULT_TOLERANCE_FLOOR_M = 0.02
DEFAULT_TOLERANCE_CAP_M = 0.5


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as error:  # 换下一个编码
            last_error = error
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def _numeric(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def list_segment_files(segment_dir: Path) -> list[Path]:
    """一口井的段文件；优先 `*_seg_*.csv`，并为旧目录保留 `*.csv` 回退。"""
    segment_dir = Path(segment_dir)
    if not segment_dir.exists():
        return []
    for pattern in SEGMENT_FILE_GLOBS:
        files = sorted(segment_dir.glob(pattern))
        if files:
            return files
    return []


class WellSegmentPool:
    """一口井全部测井段，按 MD 升序排在同一条轴上。

    `SegmentOrder` 为段文件在排序中的位置，只用于让"同距离候选"的取舍稳定可复现。
    """

    def __init__(self, well: str, frames: list[pd.DataFrame]) -> None:
        if not frames:
            raise ValueError(f"empty segment frames for well {well}")
        self.well = str(well)
        stacked = pd.concat(frames, ignore_index=True)
        stacked = stacked.dropna(subset=["MD"]).sort_values(["MD", "SegmentOrder"], kind="mergesort")
        self.table = stacked.reset_index(drop=True)
        self.md = self.table["MD"].to_numpy(dtype=float)
        self.segment_ids = self.table["SegmentID"].astype(str).to_numpy()
        self.segment_order = self.table["SegmentOrder"].to_numpy(dtype=int)
        self.segment_ids_unique = sorted(pd.unique(self.segment_ids).tolist())
        self.md_ranges = {
            str(segment_id): (float(group["MD"].min()), float(group["MD"].max()))
            for segment_id, group in self.table.groupby("SegmentID", dropna=False)
        }

    def __len__(self) -> int:
        return int(self.md.size)

    def md_step_median(self) -> float:
        """段内 MD 中位步长（所有段合并后去重的 MD 轴差分中位数）。"""
        if self.md.size < 2:
            return float("nan")
        steps = np.diff(np.unique(self.md))
        steps = steps[np.isfinite(steps) & (steps > 0)]
        if steps.size == 0:
            return float("nan")
        return float(np.median(steps))

    def candidates(self, md: float, tolerance: float) -> pd.DataFrame:
        """MD 容差内的全部候选段行（含 `MDMatchDistanceM`）。"""
        if self.md.size == 0 or not np.isfinite(md):
            return self.table.iloc[0:0]
        lo = int(np.searchsorted(self.md, md - tolerance, side="left"))
        hi = int(np.searchsorted(self.md, md + tolerance, side="right"))
        if hi <= lo:
            return self.table.iloc[0:0]
        window = self.table.iloc[lo:hi].copy()
        window["MDMatchDistanceM"] = np.abs(window["MD"].to_numpy(dtype=float) - md)
        return window

    def nearest_any(self, md: float) -> dict[str, Any]:
        """不限容差的最邻近行，仅用于审计"为什么没匹配上"。"""
        if self.md.size == 0 or not np.isfinite(md):
            return {}
        position = int(np.searchsorted(self.md, md))
        candidates = [idx for idx in (position - 1, position) if 0 <= idx < self.md.size]
        if not candidates:
            return {}
        best = min(candidates, key=lambda idx: abs(self.md[idx] - md))
        return {
            "NearestSegmentIDAnyDistance": str(self.segment_ids[best]),
            "NearestMDDifferenceAnyDistanceM": float(abs(self.md[best] - md)),
        }


def load_well_segment_pool(well: str, segment_dir: Path) -> WellSegmentPool | None:
    frames: list[pd.DataFrame] = []
    for order, segment_path in enumerate(list_segment_files(segment_dir)):
        segment = read_csv_flexible(segment_path, low_memory=False)
        if not set(REQUIRED_SEGMENT_COLUMNS).issubset(segment.columns):
            continue
        part = segment.copy()
        for column in SEGMENT_GEOMETRY_COLUMNS:
            if column in part.columns and column != "StrataName":
                part[column] = _numeric(part[column])
        keep = [column for column in SEGMENT_GEOMETRY_COLUMNS if column in part.columns]
        part = part[keep].copy()
        part["SegmentID"] = segment_path.stem
        part["SegmentPath"] = str(segment_path)
        part["SegmentOrder"] = int(order)
        frames.append(part)
    if not frames:
        return None
    return WellSegmentPool(well=well, frames=frames)


_POOL_CACHE: dict[str, WellSegmentPool | None] = {}


def cached_well_segment_pool(samples_root: Path, well: str) -> WellSegmentPool | None:
    """进程内缓存：一次运行里同一口井的段文件只读一遍。"""
    key = f"{well}::{Path(samples_root) / str(well)}"
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = load_well_segment_pool(well, Path(samples_root) / str(well))
    return _POOL_CACHE[key]


def inferred_md_tolerance(
    pool: WellSegmentPool | None,
    half_step_multiplier: float = DEFAULT_HALF_STEP_MULTIPLIER,
    floor_m: float = DEFAULT_TOLERANCE_FLOOR_M,
    cap_m: float = DEFAULT_TOLERANCE_CAP_M,
) -> float:
    """按半采样步长推导 MD 回接容差。

    固定 0.011 m 的旧口径比 Step4 合并多段造成的网格偏移（实测 0.031 m）还小，
    会造成静默丢点；这里改成随该井 MD 步长自适应的半采样口径。
    """
    if pool is None:
        return float(floor_m)
    return inferred_md_tolerance_from_md(
        pool.md,
        half_step_multiplier=half_step_multiplier,
        floor_m=floor_m,
        cap_m=cap_m,
    )


def inferred_md_tolerance_from_md(
    md_values: np.ndarray,
    half_step_multiplier: float = DEFAULT_HALF_STEP_MULTIPLIER,
    floor_m: float = DEFAULT_TOLERANCE_FLOOR_M,
    cap_m: float = DEFAULT_TOLERANCE_CAP_M,
) -> float:
    """由一段（或一口井）的 MD 采样轴推导回接容差。"""
    values = np.asarray(md_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float(floor_m)
    steps = np.diff(np.unique(values))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return float(floor_m)
    step = float(np.median(steps))
    return float(min(max(half_step_multiplier * step, floor_m), cap_m))
