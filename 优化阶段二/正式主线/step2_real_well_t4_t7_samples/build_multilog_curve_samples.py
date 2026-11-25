#!/usr/bin/env python3
"""Build DEPT-aligned GR/AC/CNL/DEN plus resistivity-pair tables for sand wells."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import build_gr_resistivity_samples as legacy


DEFAULT_OUTPUT_ROOTS = (
    legacy.CURRENT_DIR / "output/formal_all_wells",
    legacy.CURRENT_DIR / "output/formal_imaging_outer_wells",
)
CURVE_ALIASES = {
    "GR": ("GR", "GR1", "GRSL"),
    "AC": ("AC", "AC1", "DT", "DTC", "DT24", "HAC"),
    "CNL": ("CNL", "NPHI", "CNCF"),
    "DEN": ("DEN", "RHOB", "ZDEN"),
}
OUTPUT_COLUMNS = [
    "DEPT",
    "GR",
    "AC",
    "CNL",
    "DEN",
    "LLD",
    "LLS",
    "RD",
    "RS",
    "RILD",
    "RILM",
]


@dataclass(frozen=True)
class CurvePass:
    pass_id: str
    path: Path
    frame: pd.DataFrame
    median_step_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, action="append", required=True)
    parser.add_argument("--las-roots", nargs="*", type=Path, default=list(legacy.DEFAULT_LAS_ROOTS))
    parser.add_argument("--wells", nargs="*", default=[])
    parser.add_argument("--max-interpolation-gap-m", type=float, default=0.5)
    return parser.parse_args()


def pass_id_for(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def select_curve(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    available = {str(column).upper().strip() for column in columns}
    return next((name for name in aliases if name in available), None)


def curve_pass_from_path(path: Path) -> CurvePass | None:
    raw = legacy.read_las_frame(path)
    selected = {
        canonical: select_curve(raw.columns, aliases)
        for canonical, aliases in CURVE_ALIASES.items()
    }
    if any(value is None for value in selected.values()):
        return None
    frame = pd.DataFrame(
        {
            "MD": legacy.clean_numeric(raw["MD"]),
            **{
                canonical: legacy.clean_numeric(raw[source])
                for canonical, source in selected.items()
                if source is not None
            },
        }
    )
    complete = frame[list(CURVE_ALIASES)].notna().all(axis=1)
    if not complete.any():
        return None
    frame = frame.loc[complete].sort_values("MD").drop_duplicates("MD", keep="last").reset_index(drop=True)
    spacing = frame["MD"].diff().dropna().abs()
    return CurvePass(
        pass_id=pass_id_for(path),
        path=path,
        frame=frame,
        median_step_m=float(spacing.median()) if not spacing.empty else np.inf,
    )


def discover_curve_passes(
    las_roots: Iterable[Path],
    known_wells: set[str],
) -> tuple[dict[str, list[CurvePass]], list[dict[str, str]]]:
    result: dict[str, list[CurvePass]] = {well: [] for well in known_wells}
    errors: list[dict[str, str]] = []
    seen_content: set[tuple[str, str]] = set()
    for root in las_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".las":
                continue
            source_well = legacy.source_well_name(path)
            target_wells = legacy.target_wells_for_source(source_well, known_wells)
            if not target_wells:
                continue
            try:
                item = curve_pass_from_path(path)
            except Exception as exc:
                errors.append({"WellName": ";".join(target_wells), "SourcePath": str(path), "Error": str(exc)})
                continue
            if item is None:
                continue
            for well in target_wells:
                key = (well, item.pass_id)
                if key in seen_content:
                    continue
                seen_content.add(key)
                result[well].append(item)
    return result, errors


def combine_curve_passes(
    passes: list[CurvePass],
    target_md: np.ndarray,
    max_gap_m: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, int], dict[str, np.ndarray]]:
    output_columns = ["GR", "AC", "CNL", "DEN"]
    combined = pd.DataFrame(
        {column: np.full(len(target_md), np.nan) for column in output_columns}
    )
    source_ids = {column: np.full(len(target_md), "", dtype=object) for column in output_columns}
    aligned: list[tuple[CurvePass, pd.DataFrame]] = []
    for item in passes:
        frame = pd.DataFrame(index=np.arange(len(target_md)))
        for column in output_columns:
            frame[column] = legacy.interpolate_with_gap(
                item.frame, column, target_md, max_gap_m
            )
        frame["_ValidCount"] = frame[output_columns].notna().sum(axis=1)
        aligned.append((item, frame))
    aligned.sort(
        key=lambda pair: (
            -int((pair[1]["_ValidCount"] > 0).sum()),
            pair[0].median_step_m,
            str(pair[0].path),
        )
    )
    pass_counts = np.zeros(len(target_md), dtype=np.int32)
    for item, frame in aligned:
        for column in output_columns:
            fill = combined[column].isna() & frame[column].notna()
            combined.loc[fill, column] = frame.loc[fill, column].to_numpy()
            source_ids[column][fill.to_numpy()] = item.pass_id
        pass_counts += frame[output_columns].notna().any(axis=1).to_numpy(dtype=np.int32)
    coverage = {
        column: {
            "valid": int(combined[column].notna().sum()),
            "missing": int(combined[column].isna().sum()),
        }
        for column in output_columns
    }
    valid_counts = {
        "all_four_valid": int(combined[output_columns].notna().all(axis=1).sum()),
        "at_least_one_valid": int(combined[output_columns].notna().any(axis=1).sum()),
    }
    return combined, coverage, valid_counts, source_ids


def source_intervals(
    well: str,
    curve: str,
    target_md: np.ndarray,
    source_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = np.asarray(source_ids, dtype=object)
    start: int | None = None
    active = ""
    for index in range(len(values) + 1):
        current = str(values[index]) if index < len(values) and values[index] not in ("", None) else ""
        if index == len(values) or current != active:
            if start is not None and active:
                rows.append(
                    {
                        "WellName": well,
                        "CurveName": curve,
                        "PassID": active,
                        "StartDEPT": float(target_md[start]),
                        "EndDEPT": float(target_md[index - 1]),
                        "RowCount": int(index - start),
                    }
                )
            start = index if current else None
            active = current
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_output_root(
    output_root: Path,
    las_roots: list[Path],
    selected_wells: set[str],
    max_gap_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = output_root.resolve()
    selected = {legacy.normalize_well_token(well) for well in selected_wells}
    well_dirs = legacy.formal_well_directories(output_root, selected)
    if not well_dirs:
        raise RuntimeError(f"no formal Step2 wells found under {output_root}")
    known = {well for well, _ in well_dirs}

    # Build the existing pair table first; it remains the authoritative same-pass
    # resistivity-pair contract.
    if not all(legacy.output_path_for_well(well_dir, well).exists() for well, well_dir in well_dirs):
        legacy.run_enrichment(
            output_root=output_root,
            las_roots=las_roots,
            selected_wells=known,
            max_interpolation_gap_m=max_gap_m,
        )

    passes, errors = discover_curve_passes(las_roots, known)
    source_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for well, well_dir in well_dirs:
        pair_path = legacy.output_path_for_well(well_dir, well)
        pairs = pd.read_csv(pair_path, encoding="utf-8-sig")
        target_md = pd.to_numeric(pairs["DEPT"], errors="coerce").to_numpy(dtype=float)
        petro, coverage, valid_counts, curve_sources = combine_curve_passes(
            passes.get(well, []), target_md, max_gap_m
        )
        output = pd.DataFrame(
            {
                "DEPT": target_md,
                "GR": petro["GR"],
                "AC": petro["AC"],
                "CNL": petro["CNL"],
                "DEN": petro["DEN"],
                **{
                    curve: pairs[curve]
                    for curve in ("LLD", "LLS", "RD", "RS", "RILD", "RILM")
                },
            }
        )[OUTPUT_COLUMNS]
        output_path = well_dir / f"{well}_t4_t7_real_well_conventional_curves.csv"
        legacy.atomic_write_csv(output, output_path)
        for item in passes.get(well, []):
            source_rows.append(
                {
                    "WellName": well,
                    "PassID": item.pass_id,
                    "SourcePath": str(item.path),
                    "Rows": int(len(item.frame)),
                    "MDMin": float(item.frame["MD"].min()),
                    "MDMax": float(item.frame["MD"].max()),
                    "MedianStepM": item.median_step_m,
                    "Curves": "GR;AC;CNL;DEN",
                }
            )
        for curve, ids in curve_sources.items():
            interval_rows.extend(source_intervals(well, curve, target_md, ids))
        petro_ok = valid_counts["all_four_valid"]
        any_pair = pd.Series(False, index=output.index)
        for spec in legacy.PAIR_SPECS:
            any_pair |= output[["GR", spec.deep_curve, spec.reference_curve]].notna().all(axis=1)
        full6 = any_pair & output[["GR", "AC", "CNL", "DEN"]].notna().all(axis=1)
        summary_rows.append(
            {
                "WellName": well,
                "Rows": int(len(output)),
                "PetrophysicalAllFourValidRows": int(petro_ok),
                "AnyResistivityPairValidRows": int(any_pair.sum()),
                "AllSixCurveValidRows": int(full6.sum()),
                "OutputPath": str(output_path),
                **{
                    f"{column}_{key}": value
                    for column, values in coverage.items()
                    for key, value in values.items()
                },
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(source_rows),
        pd.DataFrame(interval_rows),
        pd.DataFrame(errors, columns=["WellName", "SourcePath", "Error"]),
    )


def main() -> int:
    args = parse_args()
    all_summary = []
    all_sources = []
    all_intervals = []
    all_errors = []
    for output_root in args.output_root:
        summary, sources, intervals, errors = process_output_root(
            output_root,
            [path.resolve() for path in args.las_roots],
            set(args.wells),
            float(args.max_interpolation_gap_m),
        )
        all_summary.append(summary)
        all_sources.append(sources)
        all_intervals.append(intervals)
        all_errors.append(errors)
    summary = pd.concat(all_summary, ignore_index=True)
    sources = pd.concat(all_sources, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    errors = pd.concat(all_errors, ignore_index=True)
    primary_root = args.output_root[0].resolve()
    legacy.atomic_write_csv(summary, primary_root / "multilog_curve_build_summary.csv")
    legacy.atomic_write_csv(sources, primary_root / "conventional_curve_source_manifest.csv")
    legacy.atomic_write_csv(intervals, primary_root / "conventional_curve_source_intervals.csv")
    legacy.atomic_write_csv(errors, primary_root / "multilog_curve_read_errors.csv")
    write_json(
        primary_root / "multilog_curve_build_summary.json",
        {
            "status": "pass" if errors.empty else "pass_with_read_errors",
            "output_roots": [str(path.resolve()) for path in args.output_root],
            "rows_total": int(summary["Rows"].sum()),
            "all_six_curve_valid_rows": int(summary["AllSixCurveValidRows"].sum()),
            "wells": int(len(summary)),
            "read_errors": int(len(errors)),
        },
    )
    print(summary.to_string(index=False))
    return 0 if len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
