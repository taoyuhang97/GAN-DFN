from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from strata_expert_deploy.pipeline import main as deploy_main
from strata_expert_deploy.runtime import DEFAULT_LOG_FEATURES, parse_json_list, resolve_depth_column, sanitize


DEFAULT_TARGET_SAMPLE_CSV = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\井斜\测井-地震时窗\车斜255_around_data.csv"
)
DEFAULT_LAYER_DIR = Path(r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\层位")
DEFAULT_STAGE1_LIBRARY_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\两阶段裂缝预测流程结果\outer_holdout\outer_holdout_cheye1_v3\inner_stage1_library"
)
DEFAULT_STAGE2_LIBRARY_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\两阶段裂缝预测流程结果\outer_holdout\outer_holdout_cheye1_v3\inner_stage2_library"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\两阶段裂缝预测流程结果\常规测井验证"
)
DEFAULT_DOCX_PATH = Path(r"D:\项目\石油开采\断缝储实验\实验记录20260323.docx")

SURFACE_CODE_PATTERN = re.compile(r"^(T\d+)")
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
UPDATED_SURFACE_HINTS = ("20240715", "DM_Sm", "地震", "Ato", "gljm", "顶面", "底面")
FINAL_LOG_FILENAME = "final_log_with_fractures.csv"
FINAL_POINTS_FILENAME = "final_fracture_points.csv"
FINAL_SEGMENTS_FILENAME = "final_fracture_segments.csv"
FINAL_STRATA_SEGMENTATION_FILENAME = "final_strata_segmentation.csv"
FINAL_META_FILENAME = "prediction_meta.json"

FINAL_STRATA_BASE_COLUMNS = [
    "WellName",
    "GeoSegmentID",
    "GeoIntervalKey",
    "GeoRangeOrder",
    "GeoDepthMin",
    "GeoDepthMax",
    "StrataName",
    "TopSurfaceCode",
    "TopSurfaceName",
    "BaseSurfaceCode",
    "BaseSurfaceName",
    "StrataIntervalSource",
    "ProvidedStrataName",
    "StrataAssignmentSource",
    "StrataAssignmentTopKMeanSimilarity",
    "StrataAssignmentBestSimilarityScore",
    "StrataAssignmentScoreMargin",
    "StrataAssignmentBestExpertWell",
    "StrataAssignmentResolvedStrataName",
]
FINAL_STRATA_EXPERT_COLUMNS = [
    "PredSelectedExpertWell",
    "PredSelectedSimilarityScore",
    "PredSelectedSimilarityScoreSeismic",
    "PredSelectedSimilarityScoreAC",
    "PredSelectedSimilarityScoreGR",
    "PredSelectionJointScore",
    "PredSelectionQualityQualified",
    "PredStage1ModelDir",
    "PredStage2ModelDir",
    "PredStage1InnerF1",
    "PredStage2InnerCountErrorPct",
]
FINAL_STRATA_COMPACT_COLUMNS = [
    "GeoSegmentID",
    "GeoIntervalKey",
    "GeoRangeOrder",
    "GeoDepthMin",
    "GeoDepthMax",
    "TopSurfaceCode",
    "TopSurfaceName",
    "BaseSurfaceCode",
    "BaseSurfaceName",
    "StrataIntervalSource",
    "StrataAssignmentSource",
    "StrataAssignmentTopKMeanSimilarity",
    "StrataAssignmentBestSimilarityScore",
    "StrataAssignmentScoreMargin",
    "StrataAssignmentBestExpertWell",
    "PredSelectedExpertWell",
    "PredSelectedSimilarityScore",
    "PredSelectedSimilarityScoreSeismic",
    "PredSelectedSimilarityScoreAC",
    "PredSelectedSimilarityScoreGR",
    "PredSelectionJointScore",
    "PredSelectionQualityQualified",
]
FINAL_STRATA_NUMERIC_COLUMNS = {
    "GeoRangeOrder",
    "GeoDepthMin",
    "GeoDepthMax",
    "StrataAssignmentTopKMeanSimilarity",
    "StrataAssignmentBestSimilarityScore",
    "StrataAssignmentScoreMargin",
    "PredSelectedSimilarityScore",
    "PredSelectedSimilarityScoreSeismic",
    "PredSelectedSimilarityScoreAC",
    "PredSelectedSimilarityScoreGR",
    "PredSelectionJointScore",
    "PredStage1InnerF1",
    "PredStage2InnerCountErrorPct",
}


def read_csv_flexible(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_error


def format_float(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def open_or_create_doc(docx_path: Path) -> Document:
    if docx_path.exists():
        try:
            return Document(str(docx_path))
        except PackageNotFoundError:
            pass
    doc = Document()
    doc.add_heading("实验记录 20260323", level=1)
    return doc


def append_validation_to_docx(
    docx_path: Path,
    title: str,
    config: dict,
    crossing_df: pd.DataFrame,
    target_range_df: pd.DataFrame,
    selected_expert_df: pd.DataFrame,
    stage1_summary_df: pd.DataFrame,
    stage2_summary_df: pd.DataFrame,
    result_root: Path,
    final_meta: dict,
) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    doc.add_paragraph(
        "\n".join(
            [
                "task: 常规测井整井地层约束裂缝预测验证",
                f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"well_name: {config.get('target_well_name', '')}",
                f"target_sample_csv: {config.get('target_sample_csv', '')}",
                f"layer_dir: {config.get('layer_dir', '')}",
                f"stage1_library_dir: {config.get('stage1_library_dir', '')}",
                f"stage2_library_dir: {config.get('stage2_library_dir', '')}",
                f"result_root: {result_root}",
                f"save_debug: {int(bool(config.get('save_debug', False)))}",
                f"crossed_surfaces: {int(crossing_df['Status'].astype(str).eq('crossed').sum()) if not crossing_df.empty else 0}",
                f"generated_intervals: {int(len(target_range_df))}",
                f"final_strata_count: {int(len(selected_expert_df))}",
            ]
        )
    )
    if not crossing_df.empty:
        crossed = crossing_df[crossing_df["Status"].astype(str).eq("crossed")].copy()
        if not crossed.empty:
            crossed = crossed.sort_values("CrossingDepth").reset_index(drop=True)
            doc.add_paragraph(
                "crossed_surface_sequence: "
                + " -> ".join(
                    [
                        f"{row['SurfaceCode']}({format_float(row['CrossingDepth'], 2)})"
                        for row in crossed.to_dict(orient="records")
                    ]
                )
            )
    if not selected_expert_df.empty:
        table = doc.add_table(rows=1, cols=8)
        headers = [
            "Strata",
            "Expert",
            "SimScore",
            "Stage1Thr",
            "Stage1PosCenters",
            "PredSegments",
            "PredPoints",
            "OrientMode",
        ]
        for idx, header in enumerate(headers):
            table.rows[0].cells[idx].text = header

        expert_lookup = {
            str(row.get("strata_name", "")).strip(): row
            for row in selected_expert_df.to_dict(orient="records")
        }
        stage1_lookup = {
            str(row.get("StrataName", "")).strip(): row
            for row in stage1_summary_df.to_dict(orient="records")
        }
        stage2_lookup = {
            str(row.get("StrataName", "")).strip(): row
            for row in stage2_summary_df.to_dict(orient="records")
        }
        strata_names = list(dict.fromkeys(
            list(expert_lookup.keys()) + list(stage1_lookup.keys()) + list(stage2_lookup.keys())
        ))
        for strata_name in strata_names:
            expert_row = expert_lookup.get(strata_name, {})
            stage1_row = stage1_lookup.get(strata_name, {})
            stage2_row = stage2_lookup.get(strata_name, {})
            cells = table.add_row().cells
            cells[0].text = strata_name
            cells[1].text = str(expert_row.get("selected_expert_well", ""))
            cells[2].text = format_float(expert_row.get("similarity_score", np.nan))
            cells[3].text = format_float(stage1_row.get("Stage1Threshold", np.nan))
            cells[4].text = str(stage1_row.get("Stage1NumPredictedPositiveCenters", ""))
            cells[5].text = str(stage2_row.get("NumPredSegments", ""))
            cells[6].text = str(stage2_row.get("NumPredPoints", ""))
            cells[7].text = str(stage2_row.get("OrientationMode", ""))

    doc.add_paragraph(
        "\n".join(
            [
                f"final_geo_segments: {int(final_meta.get('num_final_geo_segments', 0))}",
                f"final_pred_segments: {int(final_meta.get('num_final_fracture_segments', 0))}",
                f"final_pred_points: {int(final_meta.get('num_final_fracture_points', 0))}",
                f"final_strata_segmentation_csv: {final_meta.get('final_strata_segmentation_csv', '')}",
                f"final_log_csv: {final_meta.get('final_log_csv', '')}",
                f"final_segments_csv: {final_meta.get('final_segments_csv', '')}",
                f"final_points_csv: {final_meta.get('final_points_csv', '')}",
                f"prediction_meta_json: {final_meta.get('prediction_meta_json', '')}",
            ]
        )
    )
    doc.add_paragraph("")
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))


def infer_well_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_around_data", "_sample", "_full_log", "_data"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.strip() or path.stem


def surface_code_key(code: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(code))
    number = int(match.group(1)) if match else 9999
    return number, str(code)


def surface_preference_key(name: str) -> tuple[int, int, int, int, int, int, int, str]:
    text = str(name)
    hint_count = sum(int(hint in text) for hint in UPDATED_SURFACE_HINTS)
    return (
        int("20240715" in text),
        int("DM_Sm" in text),
        int("地震" in text),
        int("Ato" in text),
        int("gljm" in text),
        int("顶面" in text or "底面" in text),
        hint_count,
        len(text),
        text,
    )


def choose_surface_files(layer_dir: Path) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[Path]] = {}
    all_rows: list[dict] = []
    for path in sorted(layer_dir.glob("*.dat")):
        match = SURFACE_CODE_PATTERN.match(path.name)
        if not match:
            continue
        code = match.group(1)
        grouped.setdefault(code, []).append(path)
        all_rows.append(
            {
                "SurfaceCode": code,
                "SurfaceName": path.stem,
                "SurfaceFile": str(path),
                "IsSelected": 0,
                "PreferenceKey": json.dumps(surface_preference_key(path.stem), ensure_ascii=False),
            }
        )

    selected_rows: list[dict] = []
    for code in sorted(grouped.keys(), key=surface_code_key):
        candidates = sorted(grouped[code], key=lambda item: surface_preference_key(item.stem), reverse=True)
        selected = candidates[0]
        selected_rows.append(
            {
                "SurfaceCode": code,
                "SurfaceName": selected.stem,
                "SurfaceFile": str(selected),
            }
        )
        for row in all_rows:
            if row["SurfaceCode"] == code and row["SurfaceFile"] == str(selected):
                row["IsSelected"] = 1
                break
    return selected_rows, all_rows


def read_dat_surface(surface_path: Path) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    with surface_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if len(rows) < 3:
        raise ValueError(f"Surface file has too few valid XYZ rows: {surface_path}")
    return np.asarray(rows, dtype=np.float64)


def load_target_well_df(target_sample_csv: Path, target_well_name: str) -> tuple[pd.DataFrame, str, str]:
    df, used_encoding = read_csv_flexible(target_sample_csv)
    if df.empty:
        raise ValueError(f"Target sample CSV is empty: {target_sample_csv}")

    required_cols = {"TIME", "X", "Y"}
    missing = [col for col in sorted(required_cols) if col not in df.columns]
    if missing:
        raise ValueError(f"Target sample CSV missing required columns: {missing}")

    depth_col = resolve_depth_column(df)
    work_df = df.copy()
    numeric_cols = [depth_col, "TIME", "X", "Y"]
    for col in numeric_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    if "TVD" in work_df.columns:
        work_df["TVD"] = pd.to_numeric(work_df["TVD"], errors="coerce")
    else:
        work_df["TVD"] = work_df[depth_col]
    if "DEPT" in work_df.columns:
        work_df["DEPT"] = pd.to_numeric(work_df["DEPT"], errors="coerce")
    else:
        work_df["DEPT"] = work_df[depth_col]

    work_df = work_df.dropna(subset=[depth_col, "TIME", "X", "Y"]).copy()
    if work_df.empty:
        raise ValueError(f"No valid rows remain after cleaning target sample CSV: {target_sample_csv}")
    work_df = work_df.sort_values(depth_col).reset_index(drop=True)
    work_df["WellName"] = str(target_well_name).strip()
    work_df["ROW_IN_WELL"] = np.arange(len(work_df), dtype=np.int64)
    return work_df, depth_col, used_encoding


def locate_surface_crossing(
    well_df: pd.DataFrame,
    depth_col: str,
    surface_code: str,
    surface_name: str,
    surface_path: Path,
    crossing_tolerance_ms: float,
) -> dict:
    surface_xyz = read_dat_surface(surface_path)
    tree = cKDTree(surface_xyz[:, :2])
    query_xy = well_df[["X", "Y"]].to_numpy(dtype=np.float64)
    nearest_dist, nearest_idx = tree.query(query_xy, k=1)
    surface_time = surface_xyz[nearest_idx, 2]

    well_time = well_df["TIME"].to_numpy(dtype=np.float64)
    depth = well_df[depth_col].to_numpy(dtype=np.float64)
    tvd = well_df["TVD"].to_numpy(dtype=np.float64)
    dept = well_df["DEPT"].to_numpy(dtype=np.float64)
    diff = well_time - surface_time
    finite_mask = np.isfinite(diff) & np.isfinite(depth)

    if not np.any(finite_mask):
        return {
            "SurfaceCode": surface_code,
            "SurfaceName": surface_name,
            "SurfaceFile": str(surface_path),
            "Status": "no_valid_overlap",
            "CrossingMethod": "",
            "CrossingRowFloat": np.nan,
            "CrossingDepth": np.nan,
            "CrossingTVD": np.nan,
            "CrossingDEPT": np.nan,
            "CrossingTime": np.nan,
            "NearestAbsTimeDiffMs": np.nan,
            "NearestXYDistance": np.nan,
            "MeanXYDistance": np.nan,
            "SurfaceTimeMinAlongWell": np.nan,
            "SurfaceTimeMaxAlongWell": np.nan,
        }

    nearest_abs_idx = int(np.nanargmin(np.abs(diff)))
    valid_indices = np.where(finite_mask)[0]
    crossing_pairs: list[int] = []
    for idx in valid_indices[:-1]:
        next_idx = idx + 1
        if not finite_mask[next_idx]:
            continue
        if diff[idx] == 0.0 or diff[next_idx] == 0.0 or (diff[idx] < 0.0 and diff[next_idx] > 0.0) or (diff[idx] > 0.0 and diff[next_idx] < 0.0):
            crossing_pairs.append(idx)

    status = "no_crossing"
    method = ""
    crossing_row_float = np.nan
    crossing_depth = np.nan
    crossing_tvd = np.nan
    crossing_dept = np.nan
    crossing_time = np.nan
    crossing_xy_distance = np.nan

    if crossing_pairs:
        best_idx = min(
            crossing_pairs,
            key=lambda idx: max(abs(float(diff[idx])), abs(float(diff[idx + 1]))),
        )
        d0 = float(diff[best_idx])
        d1 = float(diff[best_idx + 1])
        denom = abs(d0) + abs(d1)
        frac = abs(d0) / denom if denom > 1e-12 else 0.5
        crossing_row_float = float(best_idx) + frac
        crossing_depth = float(depth[best_idx] + frac * (depth[best_idx + 1] - depth[best_idx]))
        crossing_tvd = float(tvd[best_idx] + frac * (tvd[best_idx + 1] - tvd[best_idx]))
        crossing_dept = float(dept[best_idx] + frac * (dept[best_idx + 1] - dept[best_idx]))
        crossing_time = float(well_time[best_idx] + frac * (well_time[best_idx + 1] - well_time[best_idx]))
        crossing_xy_distance = float(nearest_dist[best_idx] + frac * (nearest_dist[best_idx + 1] - nearest_dist[best_idx]))
        status = "crossed"
        method = "sign_change"
    elif abs(float(diff[nearest_abs_idx])) <= float(crossing_tolerance_ms):
        crossing_row_float = float(nearest_abs_idx)
        crossing_depth = float(depth[nearest_abs_idx])
        crossing_tvd = float(tvd[nearest_abs_idx])
        crossing_dept = float(dept[nearest_abs_idx])
        crossing_time = float(well_time[nearest_abs_idx])
        crossing_xy_distance = float(nearest_dist[nearest_abs_idx])
        status = "crossed"
        method = "nearest_within_tolerance"

    return {
        "SurfaceCode": surface_code,
        "SurfaceName": surface_name,
        "SurfaceFile": str(surface_path),
        "Status": status,
        "CrossingMethod": method,
        "CrossingRowFloat": crossing_row_float,
        "CrossingDepth": crossing_depth,
        "CrossingTVD": crossing_tvd,
        "CrossingDEPT": crossing_dept,
        "CrossingTime": crossing_time,
        "NearestAbsTimeDiffMs": float(abs(diff[nearest_abs_idx])),
        "NearestXYDistance": crossing_xy_distance if np.isfinite(crossing_xy_distance) else float(nearest_dist[nearest_abs_idx]),
        "MeanXYDistance": float(np.nanmean(nearest_dist)),
        "SurfaceTimeMinAlongWell": float(np.nanmin(surface_time)),
        "SurfaceTimeMaxAlongWell": float(np.nanmax(surface_time)),
    }


def build_target_range_df(
    well_df: pd.DataFrame,
    depth_col: str,
    target_well_name: str,
    crossing_df: pd.DataFrame,
    min_interval_depth: float,
) -> pd.DataFrame:
    depth_series = pd.to_numeric(well_df[depth_col], errors="coerce")
    depth_min = float(depth_series.min())
    depth_max = float(depth_series.max())

    valid_crossings = crossing_df[crossing_df["Status"].astype(str).eq("crossed")].copy()
    if not valid_crossings.empty:
        valid_crossings["CrossingDepth"] = pd.to_numeric(valid_crossings["CrossingDepth"], errors="coerce")
        valid_crossings = valid_crossings[valid_crossings["CrossingDepth"].notna()].copy()
        valid_crossings = valid_crossings.sort_values("CrossingDepth").reset_index(drop=True)

    boundary_rows: list[dict] = []
    for row in valid_crossings.to_dict(orient="records"):
        depth_value = float(row["CrossingDepth"])
        if boundary_rows and abs(depth_value - float(boundary_rows[-1]["CrossingDepth"])) < float(min_interval_depth):
            continue
        boundary_rows.append(row)

    interval_rows: list[dict] = []
    start_depth = depth_min
    top_code = ""
    top_name = ""

    if not boundary_rows:
        interval_rows.append(
            {
                "WellName": target_well_name,
                "RangeId": "whole_well_001",
                "DepthMin": depth_min,
                "DepthMax": depth_max,
                "StrataTop": np.nan,
                "StrataBase": np.nan,
                "StrataName": "",
                "TopSurfaceCode": "",
                "TopSurfaceName": "",
                "BaseSurfaceCode": "",
                "BaseSurfaceName": "",
                "IntervalSource": "whole_well_fallback",
            }
        )
        return pd.DataFrame(interval_rows)

    for boundary in boundary_rows:
        end_depth = float(boundary["CrossingDepth"])
        if end_depth - start_depth >= float(min_interval_depth):
            if not top_code:
                range_id = f"above_{boundary['SurfaceCode']}"
            else:
                range_id = f"between_{top_code}_{boundary['SurfaceCode']}"
            interval_rows.append(
                {
                    "WellName": target_well_name,
                    "RangeId": range_id,
                    "DepthMin": start_depth,
                    "DepthMax": end_depth,
                    "StrataTop": start_depth if top_code else np.nan,
                    "StrataBase": end_depth,
                    "StrataName": "",
                    "TopSurfaceCode": top_code,
                    "TopSurfaceName": top_name,
                    "BaseSurfaceCode": boundary["SurfaceCode"],
                    "BaseSurfaceName": boundary["SurfaceName"],
                    "IntervalSource": "surface_crossing",
                }
            )
        start_depth = end_depth
        top_code = str(boundary["SurfaceCode"])
        top_name = str(boundary["SurfaceName"])

    if depth_max - start_depth >= float(min_interval_depth):
        interval_rows.append(
            {
                "WellName": target_well_name,
                "RangeId": f"below_{top_code}" if top_code else "tail_interval_001",
                "DepthMin": start_depth,
                "DepthMax": depth_max,
                "StrataTop": start_depth if top_code else np.nan,
                "StrataBase": np.nan,
                "StrataName": "",
                "TopSurfaceCode": top_code,
                "TopSurfaceName": top_name,
                "BaseSurfaceCode": "",
                "BaseSurfaceName": "",
                "IntervalSource": "surface_crossing",
            }
        )

    out_df = pd.DataFrame(interval_rows)
    if out_df.empty:
        raise ValueError("No valid target ranges were generated from the selected surface crossings")
    out_df = out_df.sort_values(["DepthMin", "DepthMax"]).reset_index(drop=True)
    return out_df


def run_deploy_with_args(arg_list: list[str]) -> int:
    old_argv = sys.argv[:]
    sys.argv = ["run_deploy_strata_expert_prediction.py", *arg_list]
    try:
        return int(deploy_main())
    finally:
        sys.argv = old_argv


def pick_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def standardize_strata_context_df(
    resolved_target_range_df: pd.DataFrame,
    selected_expert_df: pd.DataFrame,
) -> pd.DataFrame:
    if resolved_target_range_df.empty:
        return pd.DataFrame(columns=FINAL_STRATA_BASE_COLUMNS + FINAL_STRATA_EXPERT_COLUMNS)

    strata_df = resolved_target_range_df.copy().rename(
        columns={
            "RangeId": "GeoSegmentID",
            "IntervalKey": "GeoIntervalKey",
            "RangeOrder": "GeoRangeOrder",
            "DepthMin": "GeoDepthMin",
            "DepthMax": "GeoDepthMax",
            "IntervalSource": "StrataIntervalSource",
            "AssignmentTopKMeanSimilarity": "StrataAssignmentTopKMeanSimilarity",
            "AssignmentBestSimilarityScore": "StrataAssignmentBestSimilarityScore",
            "AssignmentScoreMargin": "StrataAssignmentScoreMargin",
            "AssignedBestExpertWell": "StrataAssignmentBestExpertWell",
            "AssignedStrataName": "StrataAssignmentResolvedStrataName",
        }
    )
    if "StrataName" in strata_df.columns:
        strata_df["StrataName"] = strata_df["StrataName"].fillna("").astype(str).str.strip()
    if "ProvidedStrataName" in strata_df.columns:
        strata_df["ProvidedStrataName"] = strata_df["ProvidedStrataName"].fillna("").astype(str).str.strip()

    if not selected_expert_df.empty:
        expert_df = selected_expert_df.copy().rename(
            columns={
                "strata_name": "StrataName",
                "selected_expert_well": "PredSelectedExpertWell",
                "similarity_score": "PredSelectedSimilarityScore",
                "similarity_score_seismic": "PredSelectedSimilarityScoreSeismic",
                "similarity_score_AC": "PredSelectedSimilarityScoreAC",
                "similarity_score_GR": "PredSelectedSimilarityScoreGR",
                "selection_joint_score": "PredSelectionJointScore",
                "quality_qualified": "PredSelectionQualityQualified",
                "stage1_model_dir": "PredStage1ModelDir",
                "stage2_model_dir": "PredStage2ModelDir",
                "inner_stage1_f1": "PredStage1InnerF1",
                "inner_stage2_count_error_pct": "PredStage2InnerCountErrorPct",
            }
        )
        if "StrataName" in expert_df.columns:
            expert_df["StrataName"] = expert_df["StrataName"].fillna("").astype(str).str.strip()
            expert_df = expert_df.drop_duplicates(subset=["StrataName"], keep="first")
            strata_df = strata_df.merge(
                expert_df[
                    pick_existing_columns(
                        expert_df,
                        ["StrataName", *FINAL_STRATA_EXPERT_COLUMNS],
                    )
                ],
                on="StrataName",
                how="left",
            )

    for column in FINAL_STRATA_NUMERIC_COLUMNS:
        if column in strata_df.columns:
            strata_df[column] = pd.to_numeric(strata_df[column], errors="coerce")
    sort_cols = pick_existing_columns(strata_df, ["GeoRangeOrder", "GeoDepthMin", "GeoDepthMax", "GeoSegmentID"])
    if sort_cols:
        strata_df = strata_df.sort_values(sort_cols).reset_index(drop=True)
    return strata_df


def build_final_strata_segmentation_df(
    resolved_target_range_df: pd.DataFrame,
    selected_expert_df: pd.DataFrame,
) -> pd.DataFrame:
    strata_df = standardize_strata_context_df(
        resolved_target_range_df=resolved_target_range_df,
        selected_expert_df=selected_expert_df,
    )
    ordered_cols = pick_existing_columns(
        strata_df,
        FINAL_STRATA_BASE_COLUMNS + FINAL_STRATA_EXPERT_COLUMNS,
    )
    if not ordered_cols:
        return strata_df.reset_index(drop=True)
    return strata_df[ordered_cols].reset_index(drop=True)


def attach_strata_context_by_depth(
    df: pd.DataFrame,
    depth_col: str,
    strata_context_df: pd.DataFrame,
    context_cols: list[str],
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    out_df = df.copy()
    if out_df.empty:
        return out_df

    rename_map = rename_map or {}
    requested_pairs = [(source_col, rename_map.get(source_col, source_col)) for source_col in context_cols]
    missing_target_cols = [target_col for _, target_col in requested_pairs if target_col not in out_df.columns]
    for target_col in missing_target_cols:
        out_df[target_col] = np.nan

    if depth_col not in out_df.columns or strata_context_df.empty:
        return out_df

    work_context_df = strata_context_df.copy()
    if "GeoDepthMin" not in work_context_df.columns or "GeoDepthMax" not in work_context_df.columns:
        return out_df

    work_context_df["GeoDepthMin"] = pd.to_numeric(work_context_df["GeoDepthMin"], errors="coerce")
    work_context_df["GeoDepthMax"] = pd.to_numeric(work_context_df["GeoDepthMax"], errors="coerce")
    work_context_df = work_context_df[
        work_context_df["GeoDepthMin"].notna() & work_context_df["GeoDepthMax"].notna()
    ].copy()
    if work_context_df.empty:
        return out_df

    sort_cols = pick_existing_columns(work_context_df, ["GeoDepthMin", "GeoDepthMax", "GeoRangeOrder", "GeoSegmentID"])
    if sort_cols:
        work_context_df = work_context_df.sort_values(sort_cols).reset_index(drop=True)

    starts = work_context_df["GeoDepthMin"].to_numpy(dtype=np.float64)
    ends = work_context_df["GeoDepthMax"].to_numpy(dtype=np.float64)
    depths = pd.to_numeric(out_df[depth_col], errors="coerce").to_numpy(dtype=np.float64)
    interval_idx = np.searchsorted(starts, depths, side="right") - 1
    base_valid_mask = (
        np.isfinite(depths)
        & (interval_idx >= 0)
        & (interval_idx < len(work_context_df))
    )
    valid_mask = base_valid_mask.copy()
    if np.any(base_valid_mask):
        end_lookup = np.full(len(out_df), np.nan, dtype=np.float64)
        end_lookup[base_valid_mask] = ends[interval_idx[base_valid_mask]]
        valid_mask = base_valid_mask & (depths <= (end_lookup + 1e-9))
    interval_idx = np.where(valid_mask, interval_idx, -1)

    numeric_context_cols = set(FINAL_STRATA_NUMERIC_COLUMNS)
    for source_col, target_col in requested_pairs:
        if source_col not in work_context_df.columns:
            continue
        if source_col in numeric_context_cols:
            source_values = pd.to_numeric(work_context_df[source_col], errors="coerce").to_numpy(dtype=np.float64)
            attached = np.full(len(out_df), np.nan, dtype=np.float64)
            mask = interval_idx >= 0
            if np.any(mask):
                attached[mask] = source_values[interval_idx[mask]]
            out_df[target_col] = attached
        else:
            source_values = work_context_df[source_col].fillna("").astype(str).to_numpy(dtype=object)
            attached = np.full(len(out_df), "", dtype=object)
            mask = interval_idx >= 0
            if np.any(mask):
                attached[mask] = source_values[interval_idx[mask]]
            out_df[target_col] = attached
    return out_df


def join_unique_text(values: pd.Series) -> str:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values.fillna("").astype(str):
        text = value.strip()
        if not text or text in seen:
            continue
        unique_values.append(text)
        seen.add(text)
    return ",".join(unique_values)


def build_merge_key(
    df: pd.DataFrame,
    depth_col: str,
    depth_source_col: str,
    digits: int = 6,
) -> pd.Series:
    work_df = df.copy()
    key_cols = []
    for source_col, key_name in [
        (depth_source_col, "__k_depth"),
        ("TIME", "__k_time"),
        ("X", "__k_x"),
        ("Y", "__k_y"),
    ]:
        if source_col not in work_df.columns:
            work_df[key_name] = ""
            key_cols.append(key_name)
            continue
        numeric = pd.to_numeric(work_df[source_col], errors="coerce").round(digits)
        work_df[key_name] = numeric.map(lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}")
        key_cols.append(key_name)
    if "WellName" in work_df.columns:
        work_df["__k_well"] = work_df["WellName"].fillna("").astype(str).str.strip()
    else:
        work_df["__k_well"] = ""
    key_cols.insert(0, "__k_well")
    return work_df[key_cols].agg("|".join, axis=1)


def build_final_segment_df(
    whole_segment_df: pd.DataFrame,
    strata_context_df: pd.DataFrame,
) -> pd.DataFrame:
    if whole_segment_df.empty:
        return pd.DataFrame(
            columns=[
                "WellName",
                "StrataName",
                "GeoSegmentID",
                "GeoIntervalKey",
                "GeoRangeOrder",
                "GeoDepthMin",
                "GeoDepthMax",
                "TopSurfaceCode",
                "TopSurfaceName",
                "BaseSurfaceCode",
                "BaseSurfaceName",
                "StrataIntervalSource",
                "StrataAssignmentSource",
                "StrataAssignmentTopKMeanSimilarity",
                "StrataAssignmentBestSimilarityScore",
                "StrataAssignmentScoreMargin",
                "StrataAssignmentBestExpertWell",
                "PredSelectedExpertWell",
                "PredSelectedSimilarityScore",
                "PredSelectedSimilarityScoreSeismic",
                "PredSelectedSimilarityScoreAC",
                "PredSelectedSimilarityScoreGR",
                "PredSelectionJointScore",
                "PredSelectionQualityQualified",
                "StrataDepthMin",
                "StrataDepthMax",
                "Segment_ID",
                "SegStartDepth",
                "SegEndDepth",
                "SegLength",
                "PredPointCount",
                "PredDensityStrength",
                "PredP10MassPerLength",
                "PredP10Mass",
                "PredDensityStrengthLevel",
                "PredOrientationFamily",
                "PredOrientationConfidence",
                "PredAzimuth",
                "PredDip",
            ]
        )
    work_df = whole_segment_df.copy()
    work_df["PredPointCount"] = pd.to_numeric(work_df.get("PredPointCount"), errors="coerce").fillna(0).astype(int)
    work_df = work_df[work_df["PredPointCount"] > 0].copy()
    work_df["__SegmentMidDepth"] = np.nan
    if "SegStartDepth" in work_df.columns and "SegEndDepth" in work_df.columns:
        seg_start = pd.to_numeric(work_df["SegStartDepth"], errors="coerce")
        seg_end = pd.to_numeric(work_df["SegEndDepth"], errors="coerce")
        work_df["__SegmentMidDepth"] = (seg_start + seg_end) / 2.0
    elif "SegStartDepth" in work_df.columns:
        work_df["__SegmentMidDepth"] = pd.to_numeric(work_df["SegStartDepth"], errors="coerce")
    elif "SegEndDepth" in work_df.columns:
        work_df["__SegmentMidDepth"] = pd.to_numeric(work_df["SegEndDepth"], errors="coerce")
    work_df = attach_strata_context_by_depth(
        df=work_df,
        depth_col="__SegmentMidDepth",
        strata_context_df=strata_context_df,
        context_cols=FINAL_STRATA_COMPACT_COLUMNS,
    )
    keep_cols = pick_existing_columns(
        work_df,
        [
            "WellName",
            "StrataName",
            *FINAL_STRATA_COMPACT_COLUMNS,
            "StrataDepthMin",
            "StrataDepthMax",
            "Segment_ID",
            "SegStartDepth",
            "SegEndDepth",
            "SegLength",
            "PredPointCount",
            "PredDensityStrength",
            "PredP10MassPerLength",
            "PredP10Mass",
            "PredDensityStrengthLevel",
            "PredOrientationFamily",
            "PredOrientationConfidence",
            "PredAzimuth",
            "PredDip",
        ],
    )
    sort_cols = pick_existing_columns(work_df, ["WellName", "SegStartDepth", "SegEndDepth", "Segment_ID"])
    out_df = work_df[keep_cols].copy()
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)
    out_df = out_df.drop(columns=pick_existing_columns(out_df, ["__SegmentMidDepth"]))
    return out_df.reset_index(drop=True)


def build_final_point_df(
    whole_points_df: pd.DataFrame,
    depth_col: str,
    strata_context_df: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "WellName",
        "StrataName",
        *FINAL_STRATA_COMPACT_COLUMNS,
        "StrataDepthMin",
        "StrataDepthMax",
        "Segment_ID",
        "Point_ID_In_Segment",
    ]
    ordered_depth_cols = []
    for column in [depth_col, "TVD", "DEPT", "TIME"]:
        if column and column not in ordered_depth_cols:
            ordered_depth_cols.append(column)
    keep_cols = base_cols + ordered_depth_cols + [
        "X",
        "Y",
        "SegStartDepth",
        "SegEndDepth",
        "SegLength",
        "PredPointCount",
        "PointDensityMassPerLengthAllocated",
        "PointDensityMassAllocated",
        "PointAzimuth",
        "PointDip",
        "PredOrientationFamily",
        "PredOrientationConfidence",
    ]
    if whole_points_df.empty:
        return pd.DataFrame(columns=keep_cols)
    work_df = whole_points_df.copy()
    depth_source_col = "NearestSampleDepth" if "NearestSampleDepth" in work_df.columns else depth_col
    if depth_source_col in work_df.columns:
        work_df = attach_strata_context_by_depth(
            df=work_df,
            depth_col=depth_source_col,
            strata_context_df=strata_context_df,
            context_cols=FINAL_STRATA_COMPACT_COLUMNS,
        )
    existing_cols = pick_existing_columns(work_df, keep_cols)
    sort_cols = pick_existing_columns(work_df, ["WellName", depth_col, "TVD", "TIME", "Segment_ID", "Point_ID_In_Segment"])
    out_df = work_df[existing_cols].copy()
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)
    return out_df.reset_index(drop=True)


def build_final_log_df(
    cleaned_well_df: pd.DataFrame,
    final_point_df: pd.DataFrame,
    depth_col: str,
    strata_context_df: pd.DataFrame,
) -> pd.DataFrame:
    log_df = attach_strata_context_by_depth(
        df=cleaned_well_df,
        depth_col=depth_col,
        strata_context_df=strata_context_df,
        context_cols=[
            "StrataName",
            *FINAL_STRATA_COMPACT_COLUMNS,
        ],
        rename_map={
            "StrataName": "InterpStrataName",
            "GeoSegmentID": "InterpGeoSegmentID",
            "GeoIntervalKey": "InterpGeoIntervalKey",
            "GeoRangeOrder": "InterpGeoRangeOrder",
            "GeoDepthMin": "InterpGeoDepthMin",
            "GeoDepthMax": "InterpGeoDepthMax",
            "TopSurfaceCode": "InterpTopSurfaceCode",
            "TopSurfaceName": "InterpTopSurfaceName",
            "BaseSurfaceCode": "InterpBaseSurfaceCode",
            "BaseSurfaceName": "InterpBaseSurfaceName",
            "StrataIntervalSource": "InterpStrataIntervalSource",
            "StrataAssignmentSource": "InterpStrataAssignmentSource",
            "StrataAssignmentTopKMeanSimilarity": "InterpStrataAssignmentTopKMeanSimilarity",
            "StrataAssignmentBestSimilarityScore": "InterpStrataAssignmentBestSimilarityScore",
            "StrataAssignmentScoreMargin": "InterpStrataAssignmentScoreMargin",
            "StrataAssignmentBestExpertWell": "InterpStrataAssignmentBestExpertWell",
            "PredSelectedExpertWell": "InterpSelectedExpertWell",
            "PredSelectedSimilarityScore": "InterpSelectedSimilarityScore",
            "PredSelectedSimilarityScoreSeismic": "InterpSelectedSimilarityScoreSeismic",
            "PredSelectedSimilarityScoreAC": "InterpSelectedSimilarityScoreAC",
            "PredSelectedSimilarityScoreGR": "InterpSelectedSimilarityScoreGR",
            "PredSelectionJointScore": "InterpSelectionJointScore",
            "PredSelectionQualityQualified": "InterpSelectionQualityQualified",
        },
    )
    log_df["PredFractureFlag"] = 0
    log_df["PredDensityMassPerLength"] = np.nan
    log_df["PredAzimuth"] = np.nan
    log_df["PredDip"] = np.nan
    log_df["PredStrataName"] = ""
    log_df["PredSegmentID"] = ""
    if final_point_df.empty:
        return log_df

    point_df = final_point_df.copy()
    depth_source_col = "NearestSampleDepth" if "NearestSampleDepth" in point_df.columns else depth_col
    if depth_source_col not in point_df.columns:
        depth_source_col = depth_col
    if depth_source_col not in point_df.columns:
        return log_df

    log_df["_PredMergeKey"] = build_merge_key(log_df, depth_col=depth_col, depth_source_col=depth_col)
    point_df["_PredMergeKey"] = build_merge_key(point_df, depth_col=depth_col, depth_source_col=depth_source_col)
    agg_df = (
        point_df.groupby("_PredMergeKey", dropna=False)
        .agg(
            PredFractureFlag=("Segment_ID", lambda values: 1 if len(values) > 0 else 0),
            PredDensityMassPerLength=("PointDensityMassPerLengthAllocated", "sum"),
            PredAzimuth=("PointAzimuth", "mean"),
            PredDip=("PointDip", "mean"),
            PredStrataName=("StrataName", join_unique_text),
            PredSegmentID=("Segment_ID", lambda values: join_unique_text(pd.Series(values))),
        )
        .reset_index()
    )
    out_df = log_df.merge(agg_df, on="_PredMergeKey", how="left", suffixes=("", "_agg"))
    for column in [
        "PredFractureFlag",
        "PredDensityMassPerLength",
        "PredAzimuth",
        "PredDip",
        "PredStrataName",
        "PredSegmentID",
    ]:
        agg_col = f"{column}_agg"
        if agg_col in out_df.columns:
            out_df[column] = out_df[agg_col].where(out_df[agg_col].notna(), out_df[column])
            out_df = out_df.drop(columns=[agg_col])
    out_df["PredFractureFlag"] = pd.to_numeric(out_df["PredFractureFlag"], errors="coerce").fillna(0).astype(int)
    out_df = out_df.drop(columns=["_PredMergeKey"])
    if "ROW_IN_WELL" in out_df.columns:
        out_df = out_df.drop(columns=["ROW_IN_WELL"])
    return out_df


def build_prediction_meta(
    preprocess_summary: dict,
    selected_expert_df: pd.DataFrame,
    final_log_csv: Path,
    final_segments_csv: Path,
    final_points_csv: Path,
    final_strata_segmentation_csv: Path,
    final_log_df: pd.DataFrame,
    final_segment_df: pd.DataFrame,
    final_point_df: pd.DataFrame,
    final_strata_segmentation_df: pd.DataFrame,
    save_debug: bool,
) -> dict:
    return {
        "exp_id": str(preprocess_summary.get("exp_id", "")),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_well_name": str(preprocess_summary.get("target_well_name", "")),
        "target_sample_csv": str(preprocess_summary.get("target_sample_csv", "")),
        "stage1_library_dir": str(preprocess_summary.get("stage1_library_dir", "")),
        "stage2_library_dir": str(preprocess_summary.get("stage2_library_dir", "")),
        "num_surface_candidates": int(preprocess_summary.get("num_surface_candidates", 0)),
        "num_selected_surfaces": int(preprocess_summary.get("num_selected_surfaces", 0)),
        "num_crossed_surfaces": int(preprocess_summary.get("num_crossed_surfaces", 0)),
        "num_generated_intervals": int(preprocess_summary.get("num_generated_intervals", 0)),
        "save_debug": bool(save_debug),
        "selected_experts_by_strata": selected_expert_df[
            pick_existing_columns(
                selected_expert_df,
                ["strata_name", "selected_expert_well", "similarity_score"],
            )
        ].to_dict(orient="records") if not selected_expert_df.empty else [],
        "num_cleaned_log_rows": int(len(final_log_df)),
        "num_final_geo_segments": int(len(final_strata_segmentation_df)),
        "num_final_fracture_segments": int(len(final_segment_df)),
        "num_final_fracture_points": int(len(final_point_df)),
        "num_flagged_log_rows": int(final_log_df["PredFractureFlag"].sum()) if "PredFractureFlag" in final_log_df.columns else 0,
        "final_log_csv": str(final_log_csv),
        "final_segments_csv": str(final_segments_csv),
        "final_points_csv": str(final_points_csv),
        "final_strata_segmentation_csv": str(final_strata_segmentation_csv),
        "prediction_meta_json": "",
    }


def cleanup_result_root(result_root: Path, keep_names: set[str]) -> None:
    if not result_root.exists():
        return
    for child in result_root.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=False)
        else:
            child.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="convlog_validate_车斜255_v1")
    parser.add_argument("--target-sample-csv", default=str(DEFAULT_TARGET_SAMPLE_CSV))
    parser.add_argument("--target-well-name", default="")
    parser.add_argument("--layer-dir", default=str(DEFAULT_LAYER_DIR))
    parser.add_argument("--stage1-library-dir", default=str(DEFAULT_STAGE1_LIBRARY_DIR))
    parser.add_argument("--stage2-library-dir", default=str(DEFAULT_STAGE2_LIBRARY_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--docx-path", default=str(DEFAULT_DOCX_PATH))
    parser.add_argument(
        "--requested-log-features-json",
        default=json.dumps(DEFAULT_LOG_FEATURES, ensure_ascii=False),
    )
    parser.add_argument(
        "--strata-determination-mode",
        default="always_auto",
        choices=["use_csv", "auto_if_missing", "always_auto"],
    )
    parser.add_argument("--strata-determination-topk", type=int, default=2)
    parser.add_argument("--unknown-strata-policy", default="skip", choices=["skip", "error"])
    parser.add_argument("--first-stage-threshold", type=float, default=np.nan)
    parser.add_argument("--crossing-tolerance-ms", type=float, default=8.0)
    parser.add_argument("--min-interval-depth", type=float, default=0.5)
    parser.add_argument("--save-debug", type=int, default=0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    target_sample_csv = Path(args.target_sample_csv)
    layer_dir = Path(args.layer_dir)
    stage1_library_dir = Path(args.stage1_library_dir)
    stage2_library_dir = Path(args.stage2_library_dir)
    output_root = Path(args.output_root)
    docx_path = Path(args.docx_path)

    if not target_sample_csv.exists():
        raise FileNotFoundError(f"Target sample CSV not found: {target_sample_csv}")
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer directory not found: {layer_dir}")
    if not stage1_library_dir.exists():
        raise FileNotFoundError(f"Stage1 library dir not found: {stage1_library_dir}")
    if not stage2_library_dir.exists():
        raise FileNotFoundError(f"Stage2 library dir not found: {stage2_library_dir}")

    target_well_name = str(args.target_well_name).strip() or infer_well_name(target_sample_csv)
    requested_log_features = parse_json_list(args.requested_log_features_json)
    if not requested_log_features:
        raise ValueError("requested_log_features cannot be empty")

    result_root = output_root / sanitize(args.exp_id)
    preprocess_dir = result_root / "pre_segmentation"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    selected_surface_rows, all_surface_rows = choose_surface_files(layer_dir)
    if not selected_surface_rows:
        raise ValueError(f"No valid T-code surface files were found in: {layer_dir}")
    pd.DataFrame(all_surface_rows).to_csv(
        preprocess_dir / "surface_file_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selected_surface_rows).to_csv(
        preprocess_dir / "selected_surface_files.csv",
        index=False,
        encoding="utf-8-sig",
    )

    well_df, depth_col, sample_encoding = load_target_well_df(
        target_sample_csv=target_sample_csv,
        target_well_name=target_well_name,
    )
    missing_requested_log_features = [feature for feature in requested_log_features if feature not in well_df.columns]
    if missing_requested_log_features:
        raise ValueError(
            f"Target sample CSV missing requested log features: {missing_requested_log_features}; "
            f"prediction aborted for reliability: {target_sample_csv}"
        )
    well_df.to_csv(preprocess_dir / f"{target_well_name}_cleaned_input.csv", index=False, encoding="utf-8-sig")

    crossing_rows = []
    for row in selected_surface_rows:
        crossing_rows.append(
            locate_surface_crossing(
                well_df=well_df,
                depth_col=depth_col,
                surface_code=str(row["SurfaceCode"]),
                surface_name=str(row["SurfaceName"]),
                surface_path=Path(str(row["SurfaceFile"])),
                crossing_tolerance_ms=float(args.crossing_tolerance_ms),
            )
        )
    crossing_df = pd.DataFrame(crossing_rows)
    crossing_df.to_csv(preprocess_dir / "surface_crossings.csv", index=False, encoding="utf-8-sig")

    target_range_df = build_target_range_df(
        well_df=well_df,
        depth_col=depth_col,
        target_well_name=target_well_name,
        crossing_df=crossing_df,
        min_interval_depth=float(args.min_interval_depth),
    )
    target_range_csv = preprocess_dir / "target_strata_range_from_surfaces.csv"
    target_range_df.to_csv(target_range_csv, index=False, encoding="utf-8-sig")

    preprocess_summary = {
        "exp_id": str(args.exp_id),
        "target_well_name": target_well_name,
        "target_sample_csv": str(target_sample_csv),
        "target_sample_encoding": sample_encoding,
        "depth_col": depth_col,
        "layer_dir": str(layer_dir),
        "num_surface_candidates": int(len(all_surface_rows)),
        "num_selected_surfaces": int(len(selected_surface_rows)),
        "num_crossed_surfaces": int(
            crossing_df["Status"].astype(str).eq("crossed").sum()
        ) if not crossing_df.empty else 0,
        "num_generated_intervals": int(len(target_range_df)),
        "target_range_csv": str(target_range_csv),
        "result_root": str(result_root),
        "stage1_library_dir": str(stage1_library_dir),
        "stage2_library_dir": str(stage2_library_dir),
        "save_debug": bool(int(args.save_debug)),
    }
    with (preprocess_dir / "pre_segmentation_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(preprocess_summary, file_obj, ensure_ascii=False, indent=2)

    deploy_args = [
        "--exp-id",
        str(args.exp_id),
        "--target-sample-csv",
        str(target_sample_csv),
        "--target-well-name",
        target_well_name,
        "--target-strata-range-csv",
        str(target_range_csv),
        "--stage1-library-dir",
        str(stage1_library_dir),
        "--stage2-library-dir",
        str(stage2_library_dir),
        "--requested-log-features-json",
        json.dumps(requested_log_features, ensure_ascii=False),
        "--strata-determination-mode",
        str(args.strata_determination_mode),
        "--strata-determination-topk",
        str(int(args.strata_determination_topk)),
        "--unknown-strata-policy",
        str(args.unknown_strata_policy),
        "--result-dir",
        str(result_root),
    ]
    if math.isfinite(float(args.first_stage_threshold)):
        deploy_args.extend(["--first-stage-threshold", str(float(args.first_stage_threshold))])

    exit_code = run_deploy_with_args(deploy_args)
    if exit_code != 0:
        raise SystemExit(exit_code)

    selected_expert_df, _ = read_csv_flexible(result_root / "selected_expert_by_strata.csv")
    resolved_target_range_df, _ = read_csv_flexible(
        result_root / "target_range_resolution" / "resolved_target_strata_range.csv"
    )
    stage1_summary_df, _ = read_csv_flexible(result_root / "stage1_deploy_summary.csv")
    stage2_summary_df, _ = read_csv_flexible(result_root / "stage2_deploy_summary.csv")
    whole_segment_df, _ = read_csv_flexible(result_root / "whole_well_pred_segment_summary.csv")
    whole_points_df, _ = read_csv_flexible(result_root / "whole_well_pred_fracture_points.csv")
    final_strata_segmentation_df = build_final_strata_segmentation_df(
        resolved_target_range_df=resolved_target_range_df,
        selected_expert_df=selected_expert_df,
    )

    final_log_df = build_final_log_df(
        cleaned_well_df=well_df,
        final_point_df=whole_points_df,
        depth_col=depth_col,
        strata_context_df=final_strata_segmentation_df,
    )
    final_segment_df = build_final_segment_df(
        whole_segment_df=whole_segment_df,
        strata_context_df=final_strata_segmentation_df,
    )
    final_point_df = build_final_point_df(
        whole_points_df=whole_points_df,
        depth_col=depth_col,
        strata_context_df=final_strata_segmentation_df,
    )

    final_log_csv = result_root / FINAL_LOG_FILENAME
    final_segments_csv = result_root / FINAL_SEGMENTS_FILENAME
    final_points_csv = result_root / FINAL_POINTS_FILENAME
    final_strata_segmentation_csv = result_root / FINAL_STRATA_SEGMENTATION_FILENAME
    final_log_df.to_csv(final_log_csv, index=False, encoding="utf-8-sig")
    final_segment_df.to_csv(final_segments_csv, index=False, encoding="utf-8-sig")
    final_point_df.to_csv(final_points_csv, index=False, encoding="utf-8-sig")
    final_strata_segmentation_df.to_csv(final_strata_segmentation_csv, index=False, encoding="utf-8-sig")

    prediction_meta = build_prediction_meta(
        preprocess_summary=preprocess_summary,
        selected_expert_df=selected_expert_df,
        final_log_csv=final_log_csv,
        final_segments_csv=final_segments_csv,
        final_points_csv=final_points_csv,
        final_strata_segmentation_csv=final_strata_segmentation_csv,
        final_log_df=final_log_df,
        final_segment_df=final_segment_df,
        final_point_df=final_point_df,
        final_strata_segmentation_df=final_strata_segmentation_df,
        save_debug=bool(int(args.save_debug)),
    )
    prediction_meta_json = result_root / FINAL_META_FILENAME
    prediction_meta["prediction_meta_json"] = str(prediction_meta_json)
    with prediction_meta_json.open("w", encoding="utf-8") as file_obj:
        json.dump(prediction_meta, file_obj, ensure_ascii=False, indent=2)

    append_validation_to_docx(
        docx_path=docx_path,
        title=f"实验 {args.exp_id}",
        config={
            "target_well_name": target_well_name,
            "target_sample_csv": str(target_sample_csv),
            "layer_dir": str(layer_dir),
            "stage1_library_dir": str(stage1_library_dir),
            "stage2_library_dir": str(stage2_library_dir),
            "save_debug": bool(int(args.save_debug)),
        },
        crossing_df=crossing_df,
        target_range_df=target_range_df,
        selected_expert_df=selected_expert_df,
        stage1_summary_df=stage1_summary_df,
        stage2_summary_df=stage2_summary_df,
        result_root=result_root,
        final_meta=prediction_meta,
    )

    if not bool(int(args.save_debug)):
        cleanup_result_root(
            result_root=result_root,
            keep_names={
                FINAL_LOG_FILENAME,
                FINAL_SEGMENTS_FILENAME,
                FINAL_POINTS_FILENAME,
                FINAL_STRATA_SEGMENTATION_FILENAME,
                FINAL_META_FILENAME,
            },
        )

    print(
        json.dumps(
            {
                **preprocess_summary,
                "docx_path": str(docx_path),
                "final_log_csv": str(final_log_csv),
                "final_segments_csv": str(final_segments_csv),
                "final_points_csv": str(final_points_csv),
                "final_strata_segmentation_csv": str(final_strata_segmentation_csv),
                "prediction_meta_json": str(prediction_meta_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
