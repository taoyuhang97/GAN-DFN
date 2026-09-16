"""太古界公共测井段回接模块（P0-4）。

太古界一口井有多个测井段（可能重叠、相邻或存在空档），而 Step3 成像组行与
Step4 合并预测点都不带 X/Y/TIME，必须按 MD 回接到 Step2 的段文件取坐标。
本模块把这段逻辑收成唯一实现，供 Step5 / Step7A / Step8 / Step9 共用。

用法：

    from common.well_segment_join import (
        attach_geometry_by_md, cached_well_segment_pool, summarize_join,
    )

    matched, audit = attach_geometry_by_md(
        points,                                  # 需要含 WellName / MD
        lambda well: cached_well_segment_pool(samples_root, well),
        preferred_column="InputSegmentPath",
        geometry_source="step3_imaging_gt",
    )
"""
from __future__ import annotations

from .attach import (
    MD_JOIN_AUDIT_COLUMNS,
    attach_geometry_by_md,
    summarize_join,
)
from .segment_pool import (
    DEFAULT_HALF_STEP_MULTIPLIER,
    DEFAULT_TOLERANCE_FLOOR_M,
    SEGMENT_GEOMETRY_COLUMNS,
    WellSegmentPool,
    cached_well_segment_pool,
    inferred_md_tolerance,
    list_segment_files,
    load_well_segment_pool,
    read_csv_flexible,
)
from .source_segment import Step4SourceSegmentLookup

__all__ = [
    "MD_JOIN_AUDIT_COLUMNS",
    "DEFAULT_HALF_STEP_MULTIPLIER",
    "DEFAULT_TOLERANCE_FLOOR_M",
    "SEGMENT_GEOMETRY_COLUMNS",
    "Step4SourceSegmentLookup",
    "WellSegmentPool",
    "attach_geometry_by_md",
    "cached_well_segment_pool",
    "inferred_md_tolerance",
    "list_segment_files",
    "load_well_segment_pool",
    "read_csv_flexible",
    "summarize_join",
]
