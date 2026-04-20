from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\两阶段裂缝预测流程结果\现有常规测井裂缝预测"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\两阶段裂缝预测流程结果\层位裂缝发育情况统计"
)

SEGMENTATION_FILENAME = "final_strata_segmentation.csv"
POINT_FILENAME_CANDIDATES = [
    "final_fracture_points.csv",
    "whole_well_pred_fracture_points.csv",
]
SINGLE_WELL_FILE_SUFFIX = "_fracture_intensity.csv"
DETAIL_FILENAME = "regional_strata_fracture_intensity_detail.csv"
SUMMARY_FILENAME = "regional_strata_fracture_intensity_summary.csv"

MISSING_TEXT_SET = {"", "nan", "none", "null", "na", "n/a"}


@dataclass
class InferResult:
    value: str
    source: str
    confidence: float
    support: int


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in MISSING_TEXT_SET else text


def sanitize_filename(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "unknown_well"
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", normalized)
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip("._ ") or "unknown_well"


def safe_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def read_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(f"Missing required columns, candidates={candidates}")
    return ""


def discover_result_dirs(input_root: Path) -> list[Path]:
    if input_root.is_file():
        raise ValueError(f"Input path must be a directory, got file: {input_root}")
    if (input_root / SEGMENTATION_FILENAME).exists():
        return [input_root]
    return sorted({path.parent for path in input_root.rglob(SEGMENTATION_FILENAME)})


def choose_surface_label(code_value: object, name_value: object) -> str:
    code_text = normalize_text(code_value)
    if code_text:
        return code_text
    return normalize_text(name_value)


def choose_strata_name(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "StrataAssignmentResolvedStrataName",
        "AssignedStrataName",
        "StrataName",
        "ProvidedStrataName",
    ]
    values = pd.Series([""] * len(df), dtype=object)
    for column in candidates:
        if column not in df.columns:
            continue
        current = df[column].map(normalize_text)
        mask = values.eq("") & current.ne("")
        if mask.any():
            values.loc[mask] = current.loc[mask]
    return values


def build_depth_fallback(depth_value: float) -> str:
    return f"{float(depth_value):.3f}m" if np.isfinite(depth_value) else ""


def join_unique_text(values: pd.Series) -> str:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        items.append(text)
        seen.add(text)
    return ",".join(items)


def keep_complete_surface_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stat_top_surface"] = out["raw_top_surface"].map(normalize_text)
    out["stat_base_surface"] = out["raw_base_surface"].map(normalize_text)
    out = out[
        out["stat_top_surface"].ne("")
        & out["stat_base_surface"].ne("")
    ].copy()
    out["group_top_surface"] = out["stat_top_surface"]
    out["group_base_surface"] = out["stat_base_surface"]
    out["pair_group_source"] = "raw_complete_only"
    out["top_fill_source"] = "raw"
    out["base_fill_source"] = "raw"
    out["top_fill_confidence"] = np.nan
    out["base_fill_confidence"] = np.nan
    out["top_fill_support"] = np.nan
    out["base_fill_support"] = np.nan
    return out.reset_index(drop=True)


def load_segmentation_frame(result_dir: Path) -> pd.DataFrame:
    csv_path = result_dir / SEGMENTATION_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {csv_path}")

    raw_df = read_csv(csv_path)
    if raw_df.empty:
        raise ValueError(f"Segmentation file is empty: {csv_path}")

    well_col = pick_first_existing_column(raw_df, ["WellName"], required=False)
    segment_col = pick_first_existing_column(raw_df, ["GeoSegmentID", "SegmentID"], required=False)
    depth_min_col = pick_first_existing_column(raw_df, ["GeoDepthMin", "DepthMin"])
    depth_max_col = pick_first_existing_column(raw_df, ["GeoDepthMax", "DepthMax"])
    thickness_col = pick_first_existing_column(raw_df, ["SegmentLength"], required=False)

    out = pd.DataFrame(index=raw_df.index.copy())
    out["result_dir"] = pd.Series(str(result_dir), index=raw_df.index, dtype=object)
    out["well_name"] = (
        raw_df[well_col].fillna("").astype(str).str.strip()
        if well_col
        else pd.Series(str(result_dir.name), index=raw_df.index, dtype=object)
    )
    out["segment_key"] = (
        raw_df[segment_col].fillna("").astype(str).str.strip()
        if segment_col
        else pd.Series([f"segment_{idx + 1:03d}" for idx in range(len(raw_df))], dtype=object)
    )
    out["depth_min"] = pd.to_numeric(raw_df[depth_min_col], errors="coerce")
    out["depth_max"] = pd.to_numeric(raw_df[depth_max_col], errors="coerce")
    if thickness_col:
        out["thickness_m"] = pd.to_numeric(raw_df[thickness_col], errors="coerce")
    else:
        out["thickness_m"] = out["depth_max"] - out["depth_min"]
    out["strata_name"] = choose_strata_name(raw_df)
    out["raw_top_surface"] = [
        choose_surface_label(
            raw_df["TopSurfaceCode"].iloc[idx] if "TopSurfaceCode" in raw_df.columns else "",
            raw_df["TopSurfaceName"].iloc[idx] if "TopSurfaceName" in raw_df.columns else "",
        )
        for idx in range(len(raw_df))
    ]
    out["raw_base_surface"] = [
        choose_surface_label(
            raw_df["BaseSurfaceCode"].iloc[idx] if "BaseSurfaceCode" in raw_df.columns else "",
            raw_df["BaseSurfaceName"].iloc[idx] if "BaseSurfaceName" in raw_df.columns else "",
        )
        for idx in range(len(raw_df))
    ]

    out = out[
        out["depth_min"].notna()
        & out["depth_max"].notna()
        & (out["depth_max"] >= out["depth_min"])
    ].copy()
    if out.empty:
        raise ValueError(f"No valid segmentation intervals remain after cleaning: {csv_path}")

    out["thickness_m"] = np.where(
        pd.to_numeric(out["thickness_m"], errors="coerce").notna(),
        pd.to_numeric(out["thickness_m"], errors="coerce"),
        out["depth_max"] - out["depth_min"],
    )
    out = out[out["thickness_m"].notna() & (out["thickness_m"] >= 0.0)].copy()
    out = out.sort_values(["depth_min", "depth_max", "segment_key"]).reset_index(drop=True)
    return out


def resolve_point_csv_path(result_dir: Path) -> Path:
    for file_name in POINT_FILENAME_CANDIDATES:
        candidate = result_dir / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing fracture point file under {result_dir}, candidates={POINT_FILENAME_CANDIDATES}"
    )


def load_point_frame(result_dir: Path) -> pd.DataFrame:
    csv_path = resolve_point_csv_path(result_dir)
    raw_df = read_csv(csv_path)
    if raw_df.empty:
        raw_df.attrs["depth_col"] = ""
        return raw_df

    depth_col = pick_first_existing_column(raw_df, ["TVD", "DEPT", "MD", "NearestSampleDepth"])
    out = raw_df.copy()
    out[depth_col] = pd.to_numeric(out[depth_col], errors="coerce")
    out = out[out[depth_col].notna()].copy()
    if out.empty:
        out.attrs["depth_col"] = depth_col
        return out

    dedupe_cols = [
        col
        for col in ["WellName", "Segment_ID", "Point_ID_In_Segment", depth_col, "TIME", "X", "Y"]
        if col in out.columns
    ]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="first").copy()
    out = out.sort_values(depth_col).reset_index(drop=True)
    out.attrs["depth_col"] = depth_col
    out.attrs["point_csv_path"] = str(csv_path)
    return out


def assign_points_to_segments(seg_df: pd.DataFrame, point_df: pd.DataFrame) -> np.ndarray:
    if point_df.empty:
        return np.zeros(len(seg_df), dtype=np.int64)

    depth_col = str(point_df.attrs.get("depth_col") or "")
    if not depth_col or depth_col not in point_df.columns:
        raise ValueError("Point dataframe depth column is missing")

    starts = seg_df["depth_min"].to_numpy(dtype=np.float64)
    ends = seg_df["depth_max"].to_numpy(dtype=np.float64)
    depths = pd.to_numeric(point_df[depth_col], errors="coerce").to_numpy(dtype=np.float64)

    interval_idx = np.searchsorted(starts, depths, side="right") - 1
    base_valid_mask = np.isfinite(depths) & (interval_idx >= 0) & (interval_idx < len(seg_df))
    valid_mask = base_valid_mask.copy()
    if np.any(base_valid_mask):
        end_lookup = np.full(len(depths), np.nan, dtype=np.float64)
        end_lookup[base_valid_mask] = ends[interval_idx[base_valid_mask]]
        valid_mask = base_valid_mask & (depths <= (end_lookup + 1e-9))

    if not np.any(valid_mask):
        return np.zeros(len(seg_df), dtype=np.int64)

    return np.bincount(interval_idx[valid_mask], minlength=len(seg_df)).astype(np.int64)


def build_pair_libraries(seg_frames: list[pd.DataFrame]) -> dict[str, dict]:
    by_strata_top_from_base: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_strata_base_from_top: dict[tuple[str, str], Counter] = defaultdict(Counter)
    global_top_from_base: dict[str, Counter] = defaultdict(Counter)
    global_base_from_top: dict[str, Counter] = defaultdict(Counter)

    for frame in seg_frames:
        for row in frame.itertuples(index=False):
            top_surface = normalize_text(getattr(row, "raw_top_surface", ""))
            base_surface = normalize_text(getattr(row, "raw_base_surface", ""))
            strata_name = normalize_text(getattr(row, "strata_name", ""))
            if not top_surface or not base_surface:
                continue
            global_top_from_base[base_surface][top_surface] += 1
            global_base_from_top[top_surface][base_surface] += 1
            if strata_name:
                by_strata_top_from_base[(strata_name, base_surface)][top_surface] += 1
                by_strata_base_from_top[(strata_name, top_surface)][base_surface] += 1

    return {
        "by_strata_top_from_base": by_strata_top_from_base,
        "by_strata_base_from_top": by_strata_base_from_top,
        "global_top_from_base": global_top_from_base,
        "global_base_from_top": global_base_from_top,
    }


def choose_dominant_candidate(counter: Counter, min_ratio: float) -> tuple[str, int, float]:
    if not counter:
        return "", 0, np.nan
    best_label, best_count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    total_count = int(sum(counter.values()))
    confidence = float(best_count / total_count) if total_count > 0 else np.nan
    if total_count <= 0 or not np.isfinite(confidence) or confidence < min_ratio:
        return "", best_count, confidence
    return best_label, best_count, confidence


def infer_missing_surface(
    libraries: dict[str, dict],
    strata_name: str,
    known_surface: str,
    *,
    infer_top: bool,
    min_ratio: float,
) -> InferResult | None:
    if not known_surface:
        return None

    if infer_top:
        same_key = (strata_name, known_surface)
        same_counter = libraries["by_strata_top_from_base"].get(same_key, Counter())
        global_counter = libraries["global_top_from_base"].get(known_surface, Counter())
        same_source = "pair_same_strata_from_base"
        global_source = "pair_global_from_base"
    else:
        same_key = (strata_name, known_surface)
        same_counter = libraries["by_strata_base_from_top"].get(same_key, Counter())
        global_counter = libraries["global_base_from_top"].get(known_surface, Counter())
        same_source = "pair_same_strata_from_top"
        global_source = "pair_global_from_top"

    if strata_name:
        value, support, confidence = choose_dominant_candidate(same_counter, min_ratio=min_ratio)
        if value:
            return InferResult(value=value, source=same_source, confidence=confidence, support=support)

    value, support, confidence = choose_dominant_candidate(global_counter, min_ratio=min_ratio)
    if value:
        return InferResult(value=value, source=global_source, confidence=confidence, support=support)
    return None


def apply_adjacency_fill(frame: pd.DataFrame) -> None:
    for idx in range(len(frame)):
        if normalize_text(frame.at[idx, "stat_top_surface"]):
            continue
        if idx <= 0:
            continue
        prev_base = normalize_text(frame.at[idx - 1, "stat_base_surface"])
        if not prev_base:
            continue
        frame.at[idx, "stat_top_surface"] = prev_base
        frame.at[idx, "top_fill_source"] = "adjacent_prev_base"

    for idx in range(len(frame) - 1, -1, -1):
        if normalize_text(frame.at[idx, "stat_base_surface"]):
            continue
        if idx >= len(frame) - 1:
            continue
        next_top = normalize_text(frame.at[idx + 1, "stat_top_surface"])
        if not next_top:
            continue
        frame.at[idx, "stat_base_surface"] = next_top
        frame.at[idx, "base_fill_source"] = "adjacent_next_top"


def fill_surface_labels(frame: pd.DataFrame, libraries: dict[str, dict], min_ratio: float) -> pd.DataFrame:
    out = frame.copy()
    out["stat_top_surface"] = out["raw_top_surface"].map(normalize_text)
    out["stat_base_surface"] = out["raw_base_surface"].map(normalize_text)
    out["top_fill_source"] = np.where(out["stat_top_surface"].ne(""), "raw", "missing")
    out["base_fill_source"] = np.where(out["stat_base_surface"].ne(""), "raw", "missing")
    out["top_fill_confidence"] = np.nan
    out["base_fill_confidence"] = np.nan
    out["top_fill_support"] = np.nan
    out["base_fill_support"] = np.nan

    for _ in range(2):
        apply_adjacency_fill(out)

    for idx, row in out.iterrows():
        strata_name = normalize_text(row["strata_name"])
        top_surface = normalize_text(row["stat_top_surface"])
        base_surface = normalize_text(row["stat_base_surface"])

        if not top_surface:
            inferred = infer_missing_surface(
                libraries=libraries,
                strata_name=strata_name,
                known_surface=base_surface,
                infer_top=True,
                min_ratio=min_ratio,
            )
            if inferred is not None:
                out.at[idx, "stat_top_surface"] = inferred.value
                out.at[idx, "top_fill_source"] = inferred.source
                out.at[idx, "top_fill_confidence"] = inferred.confidence
                out.at[idx, "top_fill_support"] = inferred.support

        if not base_surface:
            inferred = infer_missing_surface(
                libraries=libraries,
                strata_name=strata_name,
                known_surface=normalize_text(out.at[idx, "stat_top_surface"]),
                infer_top=False,
                min_ratio=min_ratio,
            )
            if inferred is not None:
                out.at[idx, "stat_base_surface"] = inferred.value
                out.at[idx, "base_fill_source"] = inferred.source
                out.at[idx, "base_fill_confidence"] = inferred.confidence
                out.at[idx, "base_fill_support"] = inferred.support

    for _ in range(2):
        apply_adjacency_fill(out)

    for idx, row in out.iterrows():
        if not normalize_text(row["stat_top_surface"]):
            out.at[idx, "stat_top_surface"] = "TOP_OF_DATA"
            out.at[idx, "top_fill_source"] = "depth_fallback"
        if not normalize_text(row["stat_base_surface"]):
            out.at[idx, "stat_base_surface"] = "BOTTOM_OF_DATA"
            out.at[idx, "base_fill_source"] = "depth_fallback"

    return out


def build_single_well_output(seg_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    top_col = "group_top_surface" if "group_top_surface" in seg_df.columns else "stat_top_surface"
    base_col = "group_base_surface" if "group_base_surface" in seg_df.columns else "stat_base_surface"
    out["top_surface"] = seg_df[top_col].astype(str)
    out["base_surface"] = seg_df[base_col].astype(str)
    out["fracture_count"] = seg_df["fracture_count"].astype(np.int64)
    out["thickness_m"] = seg_df["thickness_m"].astype(float).round(6)
    out["fracture_density_per_m"] = seg_df["fracture_density_per_m"].astype(float).round(6)
    out["fracture_density_per_100m"] = seg_df["fracture_density_per_100m"].astype(float).round(6)
    return out


def build_detail_frame(seg_frames: list[pd.DataFrame]) -> pd.DataFrame:
    detail_df = pd.concat(seg_frames, ignore_index=True) if seg_frames else pd.DataFrame()
    if detail_df.empty:
        return detail_df
    keep_cols = [
        "well_name",
        "segment_key",
        "strata_name",
        "raw_top_surface",
        "raw_base_surface",
        "stat_top_surface",
        "stat_base_surface",
        "group_top_surface",
        "group_base_surface",
        "pair_group_source",
        "top_fill_source",
        "base_fill_source",
        "top_fill_confidence",
        "base_fill_confidence",
        "top_fill_support",
        "base_fill_support",
        "depth_min",
        "depth_max",
        "thickness_m",
        "fracture_count",
        "fracture_density_per_m",
        "fracture_density_per_100m",
        "result_dir",
    ]
    out = detail_df[keep_cols].copy()
    out = out.rename(
        columns={
            "well_name": "well_name",
            "segment_key": "segment_key",
            "strata_name": "strata_name",
            "raw_top_surface": "raw_top_surface",
            "raw_base_surface": "raw_base_surface",
            "stat_top_surface": "resolved_top_surface",
            "stat_base_surface": "resolved_base_surface",
            "group_top_surface": "top_surface",
            "group_base_surface": "base_surface",
            "pair_group_source": "pair_group_source",
            "top_fill_source": "top_fill_source",
            "base_fill_source": "base_fill_source",
            "top_fill_confidence": "top_fill_confidence",
            "base_fill_confidence": "base_fill_confidence",
            "top_fill_support": "top_fill_support",
            "base_fill_support": "base_fill_support",
            "depth_min": "depth_min",
            "depth_max": "depth_max",
            "thickness_m": "thickness_m",
            "fracture_count": "fracture_count",
            "fracture_density_per_m": "fracture_density_per_m",
            "fracture_density_per_100m": "fracture_density_per_100m",
            "result_dir": "result_dir",
        }
    )
    numeric_cols = [
        "top_fill_confidence",
        "base_fill_confidence",
        "top_fill_support",
        "base_fill_support",
        "depth_min",
        "depth_max",
        "thickness_m",
        "fracture_count",
        "fracture_density_per_m",
        "fracture_density_per_100m",
    ]
    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values(["well_name", "depth_min", "depth_max", "segment_key"]).reset_index(drop=True)
    return out


def build_summary_frame(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    summary_df = (
        detail_df.groupby(["top_surface", "base_surface"], dropna=False)
        .agg(
            strata_names=("strata_name", join_unique_text),
            well_count=("well_name", "nunique"),
            interval_count=("segment_key", "size"),
            total_fracture_count=("fracture_count", "sum"),
            total_thickness_m=("thickness_m", "sum"),
            mean_depth_min=("depth_min", "mean"),
        )
        .reset_index()
    )
    summary_df["fracture_density_per_m"] = np.where(
        summary_df["total_thickness_m"].to_numpy(dtype=np.float64) > 1e-12,
        summary_df["total_fracture_count"].to_numpy(dtype=np.float64)
        / summary_df["total_thickness_m"].to_numpy(dtype=np.float64),
        0.0,
    )
    summary_df["fracture_density_per_100m"] = (
        summary_df["fracture_density_per_m"].to_numpy(dtype=np.float64) * 100.0
    )
    summary_df["total_thickness_m"] = summary_df["total_thickness_m"].round(6)
    summary_df["fracture_density_per_m"] = summary_df["fracture_density_per_m"].round(6)
    summary_df["fracture_density_per_100m"] = summary_df["fracture_density_per_100m"].round(6)
    summary_df = summary_df.sort_values(
        ["mean_depth_min", "top_surface", "base_surface"],
        na_position="last",
    ).reset_index(drop=True)
    summary_df = summary_df.drop(columns=["mean_depth_min"])
    return summary_df[
        [
            "top_surface",
            "base_surface",
            "strata_names",
            "well_count",
            "interval_count",
            "total_fracture_count",
            "total_thickness_m",
            "fracture_density_per_m",
            "fracture_density_per_100m",
        ]
    ]


def unique_output_path(output_root: Path, base_name: str, used_names: set[str]) -> Path:
    stem = base_name
    suffix = ""
    counter = 2
    while f"{stem}{suffix}" in used_names:
        suffix = f"__{counter}"
        counter += 1
    used_names.add(f"{stem}{suffix}")
    return output_root / f"{stem}{suffix}{SINGLE_WELL_FILE_SUFFIX}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pair-min-ratio", type=float, default=0.6)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    pair_min_ratio = float(np.clip(float(args.pair_min_ratio), 0.0, 1.0))

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    result_dirs = discover_result_dirs(input_root)
    if not result_dirs:
        raise FileNotFoundError(
            f"No result directory containing {SEGMENTATION_FILENAME} was found under: {input_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    raw_seg_frames: list[pd.DataFrame] = []
    point_frames: dict[str, pd.DataFrame] = {}
    for result_dir in result_dirs:
        seg_df = load_segmentation_frame(result_dir)
        point_df = load_point_frame(result_dir)
        raw_seg_frames.append(seg_df)
        point_frames[str(result_dir)] = point_df

    detail_parts: list[pd.DataFrame] = []
    used_single_names: set[str] = set()

    for seg_df in raw_seg_frames:
        result_dir = Path(str(seg_df["result_dir"].iloc[0]))
        filled_df = keep_complete_surface_intervals(seg_df)
        point_df = point_frames[str(result_dir)]
        filled_df["fracture_count"] = assign_points_to_segments(filled_df, point_df)
        filled_df["fracture_density_per_m"] = np.where(
            filled_df["thickness_m"].to_numpy(dtype=np.float64) > 1e-12,
            filled_df["fracture_count"].to_numpy(dtype=np.float64)
            / filled_df["thickness_m"].to_numpy(dtype=np.float64),
            0.0,
        )
        filled_df["fracture_density_per_100m"] = (
            filled_df["fracture_density_per_m"].to_numpy(dtype=np.float64) * 100.0
        )
        detail_parts.append(filled_df)

        if filled_df.empty:
            well_name = normalize_text(seg_df["well_name"].iloc[0]) or sanitize_filename(result_dir.name)
        else:
            well_name = normalize_text(filled_df["well_name"].iloc[0]) or sanitize_filename(result_dir.name)
        output_path = unique_output_path(
            output_root=output_root,
            base_name=sanitize_filename(well_name),
            used_names=used_single_names,
        )
        build_single_well_output(filled_df).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[ok] {well_name}: {output_path}")

    detail_df = build_detail_frame(detail_parts)
    summary_df = build_summary_frame(detail_df)

    detail_path = output_root / DETAIL_FILENAME
    summary_path = output_root / SUMMARY_FILENAME
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"detail saved: {detail_path}")
    print(f"summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
