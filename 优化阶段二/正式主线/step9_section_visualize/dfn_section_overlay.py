# -*- coding: utf-8 -*-
"""Shared DFN and Step3 imaging-log overlay artists for Step9 section renderers.

The geometry/projection rules remain owned by the retained Cheye1 DFN/coherence
workflow.  This module only packages those proven artists as an optional layer
for the geological and seismic section renderers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from build_all_area_section_visualization import INTERVAL_LABELS, ProjectionSegment
from build_cheye1_dfn_coherence_sections import (
    DFN_INTERVAL_COLORS,
    add_dfn_segments,
    add_fault_trace_segments,
    add_imaging_fracture_patch_segments,
    draw_fracture_labels,
    draw_imaging_segment_trajectory,
)


@dataclass(frozen=True)
class SectionOverlayContext:
    """Already-projected overlay inputs for one overview or local section scope."""

    name: str
    dfn_segments: list[ProjectionSegment]
    fault_segments: list[ProjectionSegment]
    imaging_patch_segments: list[ProjectionSegment]
    fracture_df: pd.DataFrame
    imaging_df: pd.DataFrame
    display_summary: dict[str, float]


def draw_section_overlays(ax: Any, context: SectionOverlayContext, projection: str) -> dict[str, int]:
    """Draw DFN and real imaging-log observations above an existing background."""
    dfn_count = add_dfn_segments(ax, context.dfn_segments, projection, overlay=True)
    fault_count = add_fault_trace_segments(
        ax,
        context.fault_segments,
        projection,
        overview=context.name == "overview",
    )
    draw_imaging_segment_trajectory(ax, context.imaging_df, projection)
    patch_count = add_imaging_fracture_patch_segments(ax, context.imaging_patch_segments, projection)
    point_count = draw_fracture_labels(ax, context.fracture_df, projection, context.display_summary)
    for interval, color in DFN_INTERVAL_COLORS.items():
        ax.plot([], [], color=color, linewidth=2.4, label=INTERVAL_LABELS.get(interval, interval))
    return {
        "dfn_segment_count": int(dfn_count),
        "fault_segment_count": int(fault_count),
        "imaging_patch_count": int(patch_count),
        "imaging_point_count": int(point_count),
    }


def outside_legend(fig: Any, ax: Any, *, include_colorbar: bool) -> None:
    """Put overlay legend outside the data area without reducing source resolution."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    x = 0.855 if include_colorbar else 0.83
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, 0.89), fontsize=8.4, framealpha=0.92)
