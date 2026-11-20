"""Shared contract for staged 200 m imaging-versus-DFN comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PLANES = ("XZ", "YZ")
TIME_SCALE_M_PER_MS = 2.0
DFN_SCALE_WIDTHS = {"small": 1.10, "medium": 2.20, "large": 3.20}
DFN_LARGE_HALO_COLOR = "#111827"
DFN_LARGE_HALO_WIDTH = 5.20
IMAGING_COLOR = "#ff2bd6"
IMAGING_HALO_COLOR = "#3b0764"
IMAGING_LINE_WIDTH = 2.20
IMAGING_HALO_WIDTH = 5.00
WELL_COLOR = "#ffea00"
WELL_HALO_COLOR = "#5b4b00"
WELL_LINE_WIDTH = 2.20
WELL_HALO_WIDTH = 3.60
INTERSECTION_COLUMNS = [
    "region",
    "projection",
    "source",
    "category",
    "layer",
    "scale",
    "patch_id",
    "h1",
    "z1",
    "h2",
    "z2",
    "line_length_m",
    "apparent_dip_deg",
    "line_angle_deg",
]


def tight_imaging_window(
    times: Iterable[float],
    *,
    padding_ms: float = 20.0,
    sample_interval_ms: float = 2.0,
) -> tuple[float, float]:
    finite = np.asarray(list(times), dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("imaging time list is empty")
    interval = max(float(sample_interval_ms), 1.0e-9)
    lower = math.floor((float(finite.min()) - float(padding_ms)) / interval) * interval
    upper = math.ceil((float(finite.max()) + float(padding_ms)) / interval) * interval
    return float(lower), float(upper)


def line_metrics(
    h1: float,
    z1: float,
    h2: float,
    z2: float,
    *,
    time_scale_m_per_ms: float = TIME_SCALE_M_PER_MS,
) -> tuple[float, float, float]:
    horizontal = float(h2) - float(h1)
    vertical_m = (float(z2) - float(z1)) * float(time_scale_m_per_ms)
    length = float(math.hypot(horizontal, vertical_m))
    apparent_dip = (
        float(math.degrees(math.atan2(abs(vertical_m), abs(horizontal))))
        if length > 1.0e-12
        else 90.0
    )
    line_angle = (
        float(math.degrees(math.atan2(float(z2) - float(z1), horizontal)) % 180.0)
        if length > 1.0e-12
        else 0.0
    )
    return length, apparent_dip, line_angle


def segment_rows(
    segments: Iterable[Any],
    *,
    region: str,
    source: str,
    category: str,
    layer_by_interval: dict[str, str] | None = None,
    time_scale_m_per_ms: float = TIME_SCALE_M_PER_MS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        length, apparent_dip, line_angle = line_metrics(
            float(segment.h1),
            float(segment.z1),
            float(segment.h2),
            float(segment.z2),
            time_scale_m_per_ms=time_scale_m_per_ms,
        )
        interval = str(getattr(segment, "interval", "")).strip()
        rows.append(
            {
                "region": region,
                "projection": str(segment.projection),
                "source": source,
                "category": str(getattr(segment, "source_category", category)),
                "layer": (layer_by_interval or {}).get(interval, interval),
                "scale": str(getattr(segment, "scale", "")).strip().lower(),
                "patch_id": int(getattr(segment, "polygon_index", index)),
                "h1": float(segment.h1),
                "z1": float(segment.z1),
                "h2": float(segment.h2),
                "z2": float(segment.z2),
                "line_length_m": length,
                "apparent_dip_deg": apparent_dip,
                "line_angle_deg": line_angle,
            }
        )
    return rows


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"count": 0}
    quantiles = np.percentile(finite, [10, 25, 50, 75, 90])
    return {
        "count": int(len(finite)),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "mean": float(finite.mean()),
    }


def _histogram_overlap(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) == 0 or len(right) == 0:
        return None
    bins = np.arange(0.0, 181.0, 10.0)
    left_hist, _ = np.histogram(left, bins=bins)
    right_hist, _ = np.histogram(right, bins=bins)
    left_fraction = left_hist / max(float(left_hist.sum()), 1.0)
    right_fraction = right_hist / max(float(right_hist.sum()), 1.0)
    return float(np.minimum(left_fraction, right_fraction).sum())


def comparison_stats(
    imaging_rows: pd.DataFrame,
    dfn_rows: pd.DataFrame,
    *,
    time_min: float,
    time_max: float,
    bin_ms: float = 20.0,
) -> dict[str, Any]:
    edges = np.arange(
        math.floor(time_min / bin_ms) * bin_ms,
        math.ceil(time_max / bin_ms) * bin_ms + bin_ms * 0.5,
        bin_ms,
    )
    by_projection: dict[str, Any] = {}
    for projection in PLANES:
        imaging = imaging_rows[imaging_rows["projection"].eq(projection)]
        dfn = dfn_rows[dfn_rows["projection"].eq(projection)]
        imaging_dip = imaging["apparent_dip_deg"].to_numpy(dtype=float)
        dfn_dip = dfn["apparent_dip_deg"].to_numpy(dtype=float)
        imaging_angle = imaging["line_angle_deg"].to_numpy(dtype=float)
        dfn_angle = dfn["line_angle_deg"].to_numpy(dtype=float)
        by_projection[projection] = {
            "imaging": {
                "apparent_dip": _distribution(imaging_dip),
                "line_angle": _distribution(imaging_angle),
            },
            "dfn": {
                "apparent_dip": _distribution(dfn_dip),
                "line_angle": _distribution(dfn_angle),
            },
            "overlap": {
                "apparent_dip": _histogram_overlap(imaging_dip, dfn_dip),
                "line_angle": _histogram_overlap(imaging_angle, dfn_angle),
            },
            "scale_counts": {
                str(key): int(value)
                for key, value in dfn["scale"].value_counts().sort_index().items()
            },
        }
    imaging_bins = pd.cut(
        imaging_rows["z1"] * 0.5 + imaging_rows["z2"] * 0.5,
        bins=edges,
        right=False,
    ).value_counts()
    dfn_bins = pd.cut(
        dfn_rows["z1"] * 0.5 + dfn_rows["z2"] * 0.5,
        bins=edges,
        right=False,
    ).value_counts()
    depth_bins = [
        {
            "time_start_ms": float(interval.left),
            "time_stop_ms": float(interval.right),
            "imaging_line_count": int(imaging_bins.get(interval, 0)),
            "dfn_line_count": int(dfn_bins.get(interval, 0)),
        }
        for interval in pd.IntervalIndex.from_breaks(edges, closed="left")
    ]
    return {
        "statistics_scope": "final_Step8_DFN_vs_Step3_imaging",
        "angle_units": "degrees",
        "time_scale_m_per_ms": TIME_SCALE_M_PER_MS,
        "depth_bin_ms": float(bin_ms),
        "by_projection": by_projection,
        "depth_bins": depth_bins,
        "dfn_category_counts": {
            str(key): int(value)
            for key, value in dfn_rows["category"].value_counts().sort_index().items()
        },
    }


def write_intersection_layer(
    output_dir: Path,
    *,
    region: str,
    imaging_rows: list[dict[str, Any]],
    dfn_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    layer_dir = output_dir / "intersections"
    layer_dir.mkdir(parents=True, exist_ok=True)
    imaging_path = layer_dir / "imaging_intersections.csv"
    dfn_path = layer_dir / "dfn_intersections.csv"
    pd.DataFrame(imaging_rows, columns=INTERSECTION_COLUMNS).to_csv(
        imaging_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(dfn_rows, columns=INTERSECTION_COLUMNS).to_csv(
        dfn_path,
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "status": "pass",
        "region": region,
        "imaging_rows": int(len(imaging_rows)),
        "dfn_rows": int(len(dfn_rows)),
        "imaging_intersections": str(imaging_path),
        "dfn_intersections": str(dfn_path),
        "metadata": metadata,
    }
    (layer_dir / "intersection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
