"""Shared imaging-log interval overlay for Step9 section figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd


IMAGING_LOG_STYLE = {
    "label": "成像测井段",
    "color": "#7b2cbf",
    "halo_color": "#3b0764",
    "line_width": 3.4,
    "halo_width": 5.4,
    "alpha": 0.98,
    "zorder": 15.5,
}


def load_imaging_log_track(
    paths: Iterable[str | Path],
    *,
    well_name: str | None = None,
    require_in_imaging_interval: bool = False,
    target_block: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load imaging-log coverage samples with X/Y/TIME and source group IDs."""

    parts: list[pd.DataFrame] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {"X", "Y", "TIME"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"imaging-log CSV missing columns {missing}: {path}")

        if well_name and "WellName" in frame.columns:
            frame = frame[
                frame["WellName"].astype(str).str.strip().eq(str(well_name).strip())
            ].copy()
        if require_in_imaging_interval:
            if "InImagingInterval" not in frame.columns:
                raise ValueError(
                    f"imaging-log CSV missing InImagingInterval column: {path}"
                )
            frame = frame[
                pd.to_numeric(
                    frame["InImagingInterval"], errors="coerce"
                ).fillna(0).astype(int).eq(1)
            ].copy()
        if frame.empty:
            continue

        for column in ("X", "Y", "TIME", "MD", "DEPT", "TVD"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["X", "Y", "TIME"]).copy()
        if target_block:
            frame = frame[
                frame["X"].between(
                    float(target_block["x_min"]),
                    float(target_block["x_max"]),
                )
                & frame["Y"].between(
                    float(target_block["y_min"]),
                    float(target_block["y_max"]),
                )
            ].copy()
        if frame.empty:
            continue

        frame["ImagingLogGroup"] = path.stem
        frame["ImagingLogSource"] = str(path)
        parts.append(frame)

    if not parts:
        raise RuntimeError(
            "no imaging-log samples remain after well/interval/block filtering"
        )
    track = pd.concat(parts, ignore_index=True)
    return track.sort_values(
        ["ImagingLogGroup", "TIME"]
    ).reset_index(drop=True)


def add_imaging_log_track(
    ax,
    track: pd.DataFrame,
    projection: str,
    *,
    display_time_min: float | None = None,
    display_time_max: float | None = None,
) -> int:
    """Draw each imaging-log source as a purple line following the well track."""

    if track.empty:
        return 0
    h_column = "X" if projection == "XZ" else "Y"
    work = track.dropna(subset=[h_column, "TIME"])
    if display_time_min is not None and display_time_max is not None:
        low = min(float(display_time_min), float(display_time_max))
        high = max(float(display_time_min), float(display_time_max))
        work = work[work["TIME"].between(low, high)]

    count = 0
    labelled = False
    for _, group in work.groupby("ImagingLogGroup", sort=False):
        group = group.sort_values("TIME")
        if len(group) < 2:
            continue
        line = ax.plot(
            np.asarray(group[h_column], dtype=float),
            np.asarray(group["TIME"], dtype=float),
            color=IMAGING_LOG_STYLE["color"],
            linewidth=IMAGING_LOG_STYLE["line_width"],
            alpha=IMAGING_LOG_STYLE["alpha"],
            solid_capstyle="round",
            zorder=IMAGING_LOG_STYLE["zorder"],
            label=IMAGING_LOG_STYLE["label"] if not labelled else "_nolegend_",
        )[0]
        line.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=IMAGING_LOG_STYLE["halo_width"],
                    foreground=IMAGING_LOG_STYLE["halo_color"],
                ),
                path_effects.Normal(),
            ]
        )
        labelled = True
        count += 1
    return count


def imaging_log_track_summary(track: pd.DataFrame) -> dict[str, Any]:
    """Return source, group and coordinate extents for audit summaries."""

    if track.empty:
        return {"sample_count": 0, "group_count": 0}
    md_column = "MD" if "MD" in track.columns else "DEPT" if "DEPT" in track.columns else None
    summary: dict[str, Any] = {
        "sample_count": int(len(track)),
        "group_count": int(track["ImagingLogGroup"].nunique()),
        "groups": [
            str(value) for value in track["ImagingLogGroup"].drop_duplicates()
        ],
        "sources": [
            str(value)
            for value in track["ImagingLogSource"].drop_duplicates()
        ],
        "time_ms": [
            float(track["TIME"].min()),
            float(track["TIME"].max()),
        ],
        "x_m": [float(track["X"].min()), float(track["X"].max())],
        "y_m": [float(track["Y"].min()), float(track["Y"].max())],
        "style": dict(IMAGING_LOG_STYLE),
    }
    if md_column is not None:
        values = pd.to_numeric(track[md_column], errors="coerce").dropna()
        if not values.empty:
            summary["md_column"] = md_column
            summary["md_m"] = [float(values.min()), float(values.max())]
    return summary
