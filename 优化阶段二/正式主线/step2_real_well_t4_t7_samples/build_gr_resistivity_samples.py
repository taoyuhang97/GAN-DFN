from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lasio
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = CURRENT_DIR / "output/formal_all_wells"
DEFAULT_LAS_ROOTS = (
    Path("/data/shared/project-oil/wx数据/砂砾岩/测井"),
    Path("/data/shared/project-oil/wx数据/砂砾岩/成像测井-测井曲线"),
    Path("/data/shared/project-oil/wx数据/砂砾岩/测井数据补充-20260723"),
)
# These are conventional negative missing markers seen in the source LAS files.
# Positive high values are retained: Step2 is an alignment layer, not an outlier
# or instrument-saturation filter.
DEFAULT_MISSING_SENTINELS = (-999.25, -9999.0, -99999.0)
GR_ALIASES = ("GR", "GR1", "GRSL")
OUTPUT_COLUMNS = [
    "DEPT",
    "GR_LLD_LLS",
    "LLD",
    "LLS",
    "GR_RD_RS",
    "RD",
    "RS",
    "GR_RILD_RILM",
    "RILD",
    "RILM",
]


@dataclass(frozen=True)
class PairSpec:
    key: str
    output_gr: str
    deep_curve: str
    reference_curve: str


@dataclass
class PairPass:
    path: Path
    gr_curve: str
    frame: pd.DataFrame
    median_step_m: float
    content_sha256: str


PAIR_SPECS = (
    PairSpec("LLD_LLS", "GR_LLD_LLS", "LLD", "LLS"),
    PairSpec("RD_RS", "GR_RD_RS", "RD", "RS"),
    PairSpec("RILD_RILM", "GR_RILD_RILM", "RILD", "RILM"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align same-pass GR and reviewed resistivity pairs to existing formal Step2 MD rows."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--las-roots", nargs="*", type=Path, default=list(DEFAULT_LAS_ROOTS))
    parser.add_argument("--wells", nargs="*", default=[], help="Optional formal well-directory names for a focused run.")
    parser.add_argument("--max-interpolation-gap-m", type=float, default=0.5)
    parser.add_argument("--conflict-log-tolerance", type=float, default=0.1)
    parser.add_argument("--conflict-gr-tolerance", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without publishing per-well files.")
    return parser.parse_args()


def normalize_well_token(value: str) -> str:
    text = str(value).strip().replace("_", "").replace(" ", "")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("(导眼井)", "").replace("(导眼)", "")
    if "车页1" in text or text == "导眼井":
        return "车页1导眼"
    if text.startswith("车151HF"):
        return "车151HF"
    return text


def source_well_name(path: Path) -> str:
    return normalize_well_token(path.stem.split("@")[0])


def target_wells_for_source(source_well: str, known_wells: set[str]) -> list[str]:
    if source_well in known_wells:
        return [source_well]
    segment_aliases = {
        "车660": ["车660-1", "车660-2"],
    }
    return [well for well in segment_aliases.get(source_well, []) if well in known_wells]


def clean_numeric(values: pd.Series, *, null_values: Iterable[float] = ()) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    sentinels = tuple(DEFAULT_MISSING_SENTINELS) + tuple(float(value) for value in null_values)
    for sentinel in sentinels:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    return out


def read_las_frame(path: Path) -> pd.DataFrame:
    las = lasio.read(str(path), ignore_header_errors=True)
    null_values: list[float] = []
    try:
        null_value = las.well.NULL.value
        if null_value is not None:
            null_values.append(float(null_value))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    raw = las.df().reset_index()
    raw = raw.rename(columns={raw.columns[0]: "MD"})
    raw.columns = [str(column).upper().strip() for column in raw.columns]
    if raw.columns.duplicated().any():
        raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    # Apply only explicit missing-value handling. Do not infer saturation from
    # magnitude; the source files are assumed to be quality-controlled.
    for column in raw.columns:
        raw[column] = clean_numeric(raw[column], null_values=null_values)
    return raw.dropna(subset=["MD"]).sort_values("MD").drop_duplicates("MD", keep="last").reset_index(drop=True)


def select_gr_curve(columns: Iterable[str]) -> str | None:
    available = {str(column).upper().strip() for column in columns}
    return next((curve for curve in GR_ALIASES if curve in available), None)


def pair_pass_from_frame(path: Path, raw: pd.DataFrame, spec: PairSpec) -> PairPass | None:
    gr_curve = select_gr_curve(raw.columns)
    required = {spec.deep_curve, spec.reference_curve}
    if gr_curve is None or not required.issubset(raw.columns):
        return None
    frame = pd.DataFrame(
        {
            "MD": clean_numeric(raw["MD"]),
            "GR": clean_numeric(raw[gr_curve]),
            "Deep": clean_numeric(raw[spec.deep_curve]),
            "Reference": clean_numeric(raw[spec.reference_curve]),
        }
    ).dropna(subset=["MD"])
    complete = frame[["GR", "Deep", "Reference"]].notna().all(axis=1)
    if not complete.any():
        return None
    frame = frame.loc[complete].sort_values("MD").drop_duplicates("MD", keep="last").reset_index(drop=True)
    spacing = frame["MD"].diff().dropna().abs()
    median_step = float(spacing.median()) if not spacing.empty else np.inf
    return PairPass(
        path=path,
        gr_curve=gr_curve,
        frame=frame,
        median_step_m=median_step,
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def discover_pair_passes(las_roots: Iterable[Path], known_wells: set[str]) -> tuple[dict[str, dict[str, list[PairPass]]], list[dict[str, str]]]:
    passes: dict[str, dict[str, list[PairPass]]] = {well: {spec.key: [] for spec in PAIR_SPECS} for well in known_wells}
    errors: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    seen_content: set[tuple[str, str, str]] = set()
    for root in las_roots:
        if not root.exists():
            errors.append({"WellName": "", "SourcePath": str(root), "Error": "missing_root"})
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".las", ".lis"} or path in seen_paths:
                continue
            seen_paths.add(path)
            source_well = source_well_name(path)
            target_wells = target_wells_for_source(source_well, known_wells)
            if not target_wells:
                continue
            try:
                raw = read_las_frame(path)
                for spec in PAIR_SPECS:
                    item = pair_pass_from_frame(path, raw, spec)
                    if item is None:
                        continue
                    for well_name in target_wells:
                        duplicate_key = (well_name, spec.key, item.content_sha256)
                        if duplicate_key in seen_content:
                            continue
                        seen_content.add(duplicate_key)
                        passes[well_name][spec.key].append(item)
            except Exception as exc:
                errors.append({"WellName": ";".join(target_wells), "SourcePath": str(path), "Error": str(exc)})
    return passes, errors


def interpolate_with_gap(source: pd.DataFrame, value_column: str, target_md: np.ndarray, max_gap_m: float) -> np.ndarray:
    x = pd.to_numeric(source["MD"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(source[value_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.full(len(target_md), np.nan)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.concatenate(([True], np.diff(x) > 1.0e-9))
    x = x[keep]
    y = y[keep]
    if len(x) < 2:
        return np.full(len(target_md), np.nan)
    out = np.interp(target_md, x, y, left=np.nan, right=np.nan)
    raw_right = np.searchsorted(x, target_md, side="left")
    right = np.clip(raw_right, 0, len(x) - 1)
    left = np.clip(raw_right - 1, 0, len(x) - 1)
    exact = np.isclose(x[right], target_md, atol=1.0e-8, rtol=0.0) | np.isclose(
        x[left], target_md, atol=1.0e-8, rtol=0.0
    )
    bracket_gap = x[right] - x[left]
    supported = (target_md >= x[0]) & (target_md <= x[-1]) & (exact | (bracket_gap <= max_gap_m))
    out[~supported] = np.nan
    return out


def aligned_pass(item: PairPass, target_md: np.ndarray, max_gap_m: float) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(target_md)))
    for column in ("GR", "Deep", "Reference"):
        out[column] = interpolate_with_gap(item.frame, column, target_md, max_gap_m)
    complete = out.notna().all(axis=1)
    out.loc[~complete, ["GR", "Deep", "Reference"]] = np.nan
    return out


def pass_priority(item: PairPass, aligned: pd.DataFrame) -> tuple[float, float, str]:
    valid_rows = int(aligned.notna().all(axis=1).sum())
    step = item.median_step_m if np.isfinite(item.median_step_m) else 1.0e12
    return (-valid_rows, step, str(item.path))


def overlap_and_conflict_counts(
    aligned_frames: list[pd.DataFrame],
    log_tolerance: float,
    gr_tolerance: float,
) -> tuple[int, int]:
    if len(aligned_frames) < 2:
        return 0, 0
    complete = np.column_stack([frame.notna().all(axis=1).to_numpy() for frame in aligned_frames])
    overlap_mask = complete.sum(axis=1) >= 2
    conflict = np.zeros(len(complete), dtype=bool)
    for row_index in np.flatnonzero(overlap_mask):
        valid_frames = [frame.iloc[row_index] for frame in aligned_frames if frame.iloc[row_index].notna().all()]
        gr_values = np.asarray([row["GR"] for row in valid_frames], dtype=float)
        deep_raw = np.asarray([row["Deep"] for row in valid_frames], dtype=float)
        reference_raw = np.asarray([row["Reference"] for row in valid_frames], dtype=float)

        def resistivity_conflict(values: np.ndarray) -> bool:
            if np.all(values > 0.0):
                return bool(np.ptp(np.log10(values)) > log_tolerance)
            # Step2 preserves nonmissing raw values, including zero or negative
            # values. For QC only, identical nonpositive overlap is accepted;
            # any disagreement is reported as a conflict without taking log10.
            return not bool(np.allclose(values, values[0], atol=1.0e-12, rtol=0.0))

        conflict[row_index] = (
            np.ptp(gr_values) > gr_tolerance
            or resistivity_conflict(deep_raw)
            or resistivity_conflict(reference_raw)
        )
    return int(overlap_mask.sum()), int(conflict.sum())


def fill_short_combined_gaps(frame: pd.DataFrame, target_md: np.ndarray, max_gap_m: float) -> pd.DataFrame:
    out = frame.copy()
    complete = out.notna().all(axis=1).to_numpy()
    valid_indices = np.flatnonzero(complete)
    if len(valid_indices) < 2:
        return out
    for left_index, right_index in zip(valid_indices[:-1], valid_indices[1:]):
        if right_index <= left_index + 1:
            continue
        depth_span = target_md[right_index] - target_md[left_index]
        if depth_span <= 0.0 or depth_span > max_gap_m:
            continue
        gap_indices = np.arange(left_index + 1, right_index)
        for column in ("GR", "Deep", "Reference"):
            out.loc[gap_indices, column] = np.interp(
                target_md[gap_indices],
                [target_md[left_index], target_md[right_index]],
                [out.at[left_index, column], out.at[right_index, column]],
            )
    return out


def combine_pair_passes(
    items: list[PairPass],
    target_md: np.ndarray,
    max_gap_m: float,
    log_tolerance: float,
    gr_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    empty = pd.DataFrame({"GR": np.nan, "Deep": np.nan, "Reference": np.nan}, index=np.arange(len(target_md)))
    if not items:
        return empty, {
            "SourceFiles": "",
            "SourceFileCount": 0,
            "ContributingFileCount": 0,
            "ValidRows": 0,
            "MissingRows": int(len(target_md)),
            "OverlapRows": 0,
            "ConflictRows": 0,
        }
    aligned_items = [(item, aligned_pass(item, target_md, max_gap_m)) for item in items]
    aligned_items.sort(key=lambda pair: pass_priority(pair[0], pair[1]))
    overlap_rows, conflict_rows = overlap_and_conflict_counts(
        [frame for _, frame in aligned_items], log_tolerance, gr_tolerance
    )
    combined = empty.copy()
    contributors: list[str] = []
    for item, frame in aligned_items:
        available = combined.isna().all(axis=1) & frame.notna().all(axis=1)
        if not available.any():
            continue
        combined.loc[available, ["GR", "Deep", "Reference"]] = frame.loc[
            available, ["GR", "Deep", "Reference"]
        ].to_numpy()
        contributors.append(item.path.name)
    combined = fill_short_combined_gaps(combined, target_md, max_gap_m)
    valid_rows = int(combined.notna().all(axis=1).sum())
    return combined, {
        "SourceFiles": ";".join(item.path.name for item, _ in aligned_items),
        "SourceFileCount": len(aligned_items),
        "ContributingFileCount": len(contributors),
        "ValidRows": valid_rows,
        "MissingRows": int(len(target_md) - valid_rows),
        "OverlapRows": overlap_rows,
        "ConflictRows": conflict_rows,
    }


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def output_path_for_well(well_dir: Path, well_name: str) -> Path:
    return well_dir / f"{well_name}_t4_t7_real_well_gr_resistivity.csv"


def process_well(
    well_name: str,
    well_dir: Path,
    pair_passes: dict[str, list[PairPass]],
    max_gap_m: float,
    log_tolerance: float,
    gr_tolerance: float,
    dry_run: bool,
) -> dict[str, object]:
    main_path = well_dir / f"{well_name}_t4_t7_real_well_main.csv"
    main = pd.read_csv(main_path, usecols=["DEPT"])
    target_md = pd.to_numeric(main["DEPT"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target_md).all() or np.any(np.diff(target_md) < 0.0):
        raise RuntimeError("main DEPT must be finite and nondecreasing")
    result = pd.DataFrame({"DEPT": target_md})
    summary: dict[str, object] = {
        "WellName": well_name,
        "Status": "ok",
        "MainRowCount": int(len(main)),
        "OutputPath": str(output_path_for_well(well_dir, well_name)),
    }
    for spec in PAIR_SPECS:
        combined, stats = combine_pair_passes(
            pair_passes.get(spec.key, []), target_md, max_gap_m, log_tolerance, gr_tolerance
        )
        result[spec.output_gr] = combined["GR"].to_numpy()
        result[spec.deep_curve] = combined["Deep"].to_numpy()
        result[spec.reference_curve] = combined["Reference"].to_numpy()
        for key, value in stats.items():
            summary[f"{spec.key}_{key}"] = value
    result = result[OUTPUT_COLUMNS]
    if len(result) != len(main) or not np.allclose(result["DEPT"], target_md, atol=1.0e-9, rtol=0.0):
        raise RuntimeError("output row/depth contract mismatch")
    for spec in PAIR_SPECS:
        complete_count = result[[spec.output_gr, spec.deep_curve, spec.reference_curve]].notna().sum(axis=1)
        if not complete_count.isin([0, 3]).all():
            raise RuntimeError(f"partial same-pass triple detected for {spec.key}")
    if not dry_run:
        atomic_write_csv(result, output_path_for_well(well_dir, well_name))
    if all(int(summary[f"{spec.key}_ValidRows"]) == 0 for spec in PAIR_SPECS):
        summary["Status"] = "no_supported_pair_coverage"
    return summary


def formal_well_directories(output_root: Path, selected_wells: set[str]) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for well_dir in sorted(path for path in output_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        well_name = normalize_well_token(well_dir.name)
        if selected_wells and well_name not in selected_wells:
            continue
        main_path = well_dir / f"{well_name}_t4_t7_real_well_main.csv"
        if main_path.exists():
            rows.append((well_name, well_dir))
    return rows


def run_enrichment(
    output_root: Path,
    las_roots: Iterable[Path] = DEFAULT_LAS_ROOTS,
    selected_wells: set[str] | None = None,
    max_interpolation_gap_m: float = 0.5,
    conflict_log_tolerance: float = 0.1,
    conflict_gr_tolerance: float = 10.0,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root = output_root.resolve()
    selected_wells = {normalize_well_token(value) for value in (selected_wells or set())}
    well_dirs = formal_well_directories(output_root, selected_wells)
    if not well_dirs:
        raise RuntimeError(f"no formal Step2 main tables found under {output_root}")
    known_wells = {well_name for well_name, _ in well_dirs}
    print(f"[GR-RES] scanning LAS sources for {len(known_wells)} formal wells", flush=True)
    passes, read_errors = discover_pair_passes(las_roots, known_wells)
    summary_rows: list[dict[str, object]] = []
    for well_name, well_dir in well_dirs:
        try:
            row = process_well(
                well_name,
                well_dir,
                passes.get(well_name, {}),
                float(max_interpolation_gap_m),
                float(conflict_log_tolerance),
                float(conflict_gr_tolerance),
                bool(dry_run),
            )
        except Exception as exc:
            row = {
                "WellName": well_name,
                "Status": "failed",
                "MainRowCount": 0,
                "OutputPath": str(output_path_for_well(well_dir, well_name)),
                "Reason": str(exc),
            }
        summary_rows.append(row)
        print(
            f"[GR-RES] {well_name}: {row['Status']} | "
            f"LLD={row.get('LLD_LLS_ValidRows', 0)} "
            f"RD={row.get('RD_RS_ValidRows', 0)} "
            f"RILD={row.get('RILD_RILM_ValidRows', 0)}",
            flush=True,
        )
    summary = pd.DataFrame(summary_rows).sort_values("WellName").reset_index(drop=True)
    error_frame = pd.DataFrame(read_errors, columns=["WellName", "SourcePath", "Error"])
    if not dry_run:
        atomic_write_csv(summary, output_root / "gr_resistivity_build_summary.csv")
        atomic_write_csv(error_frame, output_root / "gr_resistivity_read_errors.csv")
    failed = int(summary["Status"].eq("failed").sum())
    print(
        f"[GR-RES] completed wells={len(summary)} failed={failed} read_errors={len(error_frame)} dry_run={dry_run}",
        flush=True,
    )
    return summary, error_frame


def main() -> int:
    args = parse_args()
    summary, _ = run_enrichment(
        output_root=args.output_root,
        las_roots=args.las_roots,
        selected_wells=set(args.wells),
        max_interpolation_gap_m=float(args.max_interpolation_gap_m),
        conflict_log_tolerance=float(args.conflict_log_tolerance),
        conflict_gr_tolerance=float(args.conflict_gr_tolerance),
        dry_run=bool(args.dry_run),
    )
    failed = int(summary["Status"].eq("failed").sum())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
