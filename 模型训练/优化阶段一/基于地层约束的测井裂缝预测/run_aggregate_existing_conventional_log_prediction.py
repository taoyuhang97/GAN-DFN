from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_conventional_log_strata_validation import (
    FINAL_META_FILENAME,
    FINAL_POINTS_FILENAME,
    FINAL_SEGMENTS_FILENAME,
    FINAL_STRATA_SEGMENTATION_FILENAME,
)


DEFAULT_BATCH_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\两阶段裂缝预测流程结果\现有常规测井裂缝预测"
)
DEFAULT_SUMMARY_CSV = "batch_prediction_summary.csv"
DEFAULT_AGGREGATE_DIRNAME = "汇总结果"
AGG_POINTS_FILENAME = "all_final_fracture_points.csv"
AGG_SEGMENTS_FILENAME = "all_final_fracture_segments.csv"
AGG_STRATA_FILENAME = "all_final_strata_segmentation.csv"
AGG_WELL_SUMMARY_FILENAME = "all_predicted_well_summary.csv"
AGG_META_FILENAME = "aggregation_meta.json"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"Failed to read CSV: {path}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--batch-summary-csv-name", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--aggregate-dirname", default=DEFAULT_AGGREGATE_DIRNAME)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    batch_root = Path(args.batch_root)
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")

    batch_summary_csv = batch_root / str(args.batch_summary_csv_name)
    if not batch_summary_csv.exists():
        raise FileNotFoundError(f"Batch summary CSV not found: {batch_summary_csv}")

    aggregate_root = batch_root / str(args.aggregate_dirname)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    batch_summary_df = read_csv_flexible(batch_summary_csv)
    batch_lookup = {
        str(row.get("WellName", "")).strip(): row
        for row in batch_summary_df.to_dict(orient="records")
        if str(row.get("WellName", "")).strip()
    }

    points_frames: list[pd.DataFrame] = []
    segments_frames: list[pd.DataFrame] = []
    strata_frames: list[pd.DataFrame] = []
    well_summary_rows: list[dict] = []

    predicted_dirs = sorted(
        [
            path
            for path in batch_root.iterdir()
            if path.is_dir()
            and (path / FINAL_POINTS_FILENAME).exists()
            and (path / FINAL_SEGMENTS_FILENAME).exists()
            and (path / FINAL_STRATA_SEGMENTATION_FILENAME).exists()
            and (path / FINAL_META_FILENAME).exists()
        ],
        key=lambda item: item.name,
    )

    for well_dir in predicted_dirs:
        well_name = str(well_dir.name).strip()
        batch_row = batch_lookup.get(well_name, {})
        meta = load_json(well_dir / FINAL_META_FILENAME)

        points_df = read_csv_flexible(well_dir / FINAL_POINTS_FILENAME)
        points_df.insert(0, "ResultDir", str(well_dir))
        points_df.insert(0, "SourceGroup", str(batch_row.get("SourceGroup", "")))
        points_df.insert(0, "FeatureHandling", str(batch_row.get("FeatureHandling", "")))
        points_df.insert(0, "BatchRunStatus", str(batch_row.get("RunStatus", "")))
        points_frames.append(points_df)

        segments_df = read_csv_flexible(well_dir / FINAL_SEGMENTS_FILENAME)
        segments_df.insert(0, "ResultDir", str(well_dir))
        segments_df.insert(0, "SourceGroup", str(batch_row.get("SourceGroup", "")))
        segments_df.insert(0, "FeatureHandling", str(batch_row.get("FeatureHandling", "")))
        segments_df.insert(0, "BatchRunStatus", str(batch_row.get("RunStatus", "")))
        segments_frames.append(segments_df)

        strata_df = read_csv_flexible(well_dir / FINAL_STRATA_SEGMENTATION_FILENAME)
        strata_df.insert(0, "ResultDir", str(well_dir))
        strata_df.insert(0, "SourceGroup", str(batch_row.get("SourceGroup", "")))
        strata_df.insert(0, "FeatureHandling", str(batch_row.get("FeatureHandling", "")))
        strata_df.insert(0, "BatchRunStatus", str(batch_row.get("RunStatus", "")))
        strata_frames.append(strata_df)

        well_summary_rows.append(
            {
                "WellName": well_name,
                "SourceGroup": str(batch_row.get("SourceGroup", "")),
                "FeatureHandling": str(batch_row.get("FeatureHandling", "")),
                "BatchRunStatus": str(batch_row.get("RunStatus", "")),
                "CsvPath": str(batch_row.get("CsvPath", "")),
                "ResultDir": str(well_dir),
                "TargetSampleCsv": str(meta.get("target_sample_csv", "")),
                "NumSurfaceCandidates": meta.get("num_surface_candidates", 0),
                "NumSelectedSurfaces": meta.get("num_selected_surfaces", 0),
                "NumCrossedSurfaces": meta.get("num_crossed_surfaces", 0),
                "NumGeneratedIntervals": meta.get("num_generated_intervals", 0),
                "NumCleanedLogRows": meta.get("num_cleaned_log_rows", 0),
                "NumFinalGeoSegments": meta.get("num_final_geo_segments", 0),
                "NumFinalFractureSegments": meta.get("num_final_fracture_segments", 0),
                "NumFinalFracturePoints": meta.get("num_final_fracture_points", 0),
                "NumFlaggedLogRows": meta.get("num_flagged_log_rows", 0),
                "SelectedExpertsByStrataJSON": json.dumps(
                    meta.get("selected_experts_by_strata", []),
                    ensure_ascii=False,
                ),
                "PredictionMetaJson": str(well_dir / FINAL_META_FILENAME),
            }
        )

    all_points_df = pd.concat(points_frames, ignore_index=True) if points_frames else pd.DataFrame()
    all_segments_df = pd.concat(segments_frames, ignore_index=True) if segments_frames else pd.DataFrame()
    all_strata_df = pd.concat(strata_frames, ignore_index=True) if strata_frames else pd.DataFrame()
    all_well_summary_df = pd.DataFrame(well_summary_rows)
    if not all_well_summary_df.empty:
        all_well_summary_df = all_well_summary_df.sort_values(["SourceGroup", "WellName"]).reset_index(drop=True)

    points_csv = aggregate_root / AGG_POINTS_FILENAME
    segments_csv = aggregate_root / AGG_SEGMENTS_FILENAME
    strata_csv = aggregate_root / AGG_STRATA_FILENAME
    well_summary_csv = aggregate_root / AGG_WELL_SUMMARY_FILENAME
    all_points_df.to_csv(points_csv, index=False, encoding="utf-8-sig")
    all_segments_df.to_csv(segments_csv, index=False, encoding="utf-8-sig")
    all_strata_df.to_csv(strata_csv, index=False, encoding="utf-8-sig")
    all_well_summary_df.to_csv(well_summary_csv, index=False, encoding="utf-8-sig")

    agg_meta = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_root": str(batch_root),
        "aggregate_root": str(aggregate_root),
        "num_predicted_wells": int(len(predicted_dirs)),
        "num_all_fracture_points": int(len(all_points_df)),
        "num_all_fracture_segments": int(len(all_segments_df)),
        "num_all_strata_segments": int(len(all_strata_df)),
        "points_csv": str(points_csv),
        "segments_csv": str(segments_csv),
        "strata_csv": str(strata_csv),
        "well_summary_csv": str(well_summary_csv),
    }
    agg_meta_path = aggregate_root / AGG_META_FILENAME
    with agg_meta_path.open("w", encoding="utf-8") as file_obj:
        json.dump(agg_meta, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(agg_meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
