"""太古界公共测井段回接模块（逐段口径）。

设计口径（2026-09-16 确认）：

* 建模单元是**测井段**（同一批次观测），Step4 的输出以**井**为单位，
  因此 Step4 的密度曲线与裂缝点直接携带井级 `X/Y/TIME`，下游不需要回接；
* 仍需回接的是 **Step3 成像监督组行**（逐段行，带 `InputSegmentPath`，
  但只有 MD/TVD），须在各自所属段内取坐标；
* 回接一律**逐段进行**：不跨段匹配、不做段间平均，未命中只写审计、不静默丢弃。

用法：

    from common.well_segment_join import (
        attach_geometry_per_segment, summarize_join,
    )

    matched, audit = attach_geometry_per_segment(
        points,                                  # 需含 WellName / MD / InputSegmentPath
        preferred_column="InputSegmentPath",
        geometry_source="step3_imaging_gt",
    )
    summary_block = summarize_join(audit)
"""
from __future__ import annotations

from .attach import (
    MD_JOIN_AUDIT_COLUMNS,
    attach_geometry_per_segment,
    summarize_join,
)
from .segment_pool import (
    DEFAULT_HALF_STEP_MULTIPLIER,
    DEFAULT_TOLERANCE_CAP_M,
    DEFAULT_TOLERANCE_FLOOR_M,
    SEGMENT_GEOMETRY_COLUMNS,
    WellSegmentPool,
    cached_well_segment_pool,
    inferred_md_tolerance,
    inferred_md_tolerance_from_md,
    list_segment_files,
    load_well_segment_pool,
    read_csv_flexible,
)

__all__ = [
    "MD_JOIN_AUDIT_COLUMNS",
    "DEFAULT_HALF_STEP_MULTIPLIER",
    "DEFAULT_TOLERANCE_CAP_M",
    "DEFAULT_TOLERANCE_FLOOR_M",
    "SEGMENT_GEOMETRY_COLUMNS",
    "WellSegmentPool",
    "attach_geometry_per_segment",
    "cached_well_segment_pool",
    "inferred_md_tolerance",
    "inferred_md_tolerance_from_md",
    "list_segment_files",
    "load_well_segment_pool",
    "read_csv_flexible",
    "summarize_join",
]
