"""用 Step4 来源明细把合并预测点回溯到真正产出它的 Step2 段。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .segment_pool import read_csv_flexible


class Step4SourceSegmentLookup:
    """Step4 合并点 -> 来源段集合。

    Step4 按 TVD 邻域合并重叠测次并取均值，因此合并后的 MD 不再落在任何单一段的
    网格上；`all_wells_source_detail_predictions.csv` 保留了逐段明细，用它可以在多个
    候选段之间优先选"真正产出该预测"的那一段，而不是"文件名排序第一段"。
    """

    def __init__(self, detail_csv: Path | None, tvd_tolerance_m: float = 0.25) -> None:
        self.tolerance = float(tvd_tolerance_m)
        self.available = False
        self.source_path = str(detail_csv) if detail_csv is not None else ""
        self._tvd: dict[str, np.ndarray] = {}
        self._segment_ids: dict[str, np.ndarray] = {}
        if detail_csv is None:
            return
        detail_path = Path(detail_csv)
        if not detail_path.exists():
            return
        detail = read_csv_flexible(detail_path, low_memory=False)
        required = {"WellName", "TVD", "SegmentID"}
        if not required.issubset(detail.columns):
            return
        detail = detail[["WellName", "TVD", "SegmentID"]].copy()
        detail["TVD"] = pd.to_numeric(detail["TVD"], errors="coerce")
        detail = detail.dropna(subset=["TVD"])
        for well, group in detail.groupby("WellName", dropna=False):
            group = group.sort_values("TVD", kind="mergesort")
            self._tvd[str(well)] = group["TVD"].to_numpy(dtype=float)
            self._segment_ids[str(well)] = group["SegmentID"].astype(str).to_numpy()
        self.available = bool(self._tvd)

    def preferred_segment_ids(self, well: str, tvd: float) -> list[str]:
        tvd_values = self._tvd.get(str(well))
        segment_ids = self._segment_ids.get(str(well))
        if tvd_values is None or tvd_values.size == 0 or segment_ids is None:
            return []
        if not np.isfinite(tvd):
            return []
        lo = int(np.searchsorted(tvd_values, tvd - self.tolerance, side="left"))
        hi = int(np.searchsorted(tvd_values, tvd + self.tolerance, side="right"))
        if hi <= lo:
            return []
        return sorted(set(segment_ids[lo:hi].tolist()))
