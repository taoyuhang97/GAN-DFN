from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_conventional_log_strata_validation import (
    DEFAULT_LAYER_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGE1_LIBRARY_DIR,
    DEFAULT_STAGE2_LIBRARY_DIR,
    DEFAULT_TARGET_SAMPLE_CSV,
    build_target_range_df,
    choose_surface_files,
    infer_well_name,
    load_target_well_df,
    locate_surface_crossing,
)
from strata_expert_deploy.runtime import DEFAULT_LOG_FEATURES, OUTER_FLOW_SCRIPT, load_module, parse_json_list, sanitize
from strata_expert_deploy.strata_resolution import (
    discover_available_library_strata,
    normalize_target_strata_range_df,
    resolve_target_strata_range_df,
)


FINAL_SEGMENTATION_FILENAME = "final_strata_segmentation.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="convlog_strata_segment_车斜255_v1")
    parser.add_argument("--target-sample-csv", default=str(DEFAULT_TARGET_SAMPLE_CSV))
    parser.add_argument("--target-well-name", default="")
    parser.add_argument("--layer-dir", default=str(DEFAULT_LAYER_DIR))
    parser.add_argument("--stage1-library-dir", default=str(DEFAULT_STAGE1_LIBRARY_DIR))
    parser.add_argument("--stage2-library-dir", default=str(DEFAULT_STAGE2_LIBRARY_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
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
    parser.add_argument("--crossing-tolerance-ms", type=float, default=8.0)
    parser.add_argument("--min-interval-depth", type=float, default=0.5)
    return parser


def build_final_segmentation_df(resolved_df: pd.DataFrame) -> pd.DataFrame:
    if resolved_df.empty:
        return pd.DataFrame(
            columns=[
                "WellName",
                "SegmentID",
                "RangeId",
                "DepthMin",
                "DepthMax",
                "SegmentLength",
                "AssignedStrataName",
                "TopSurfaceCode",
                "TopSurfaceName",
                "BaseSurfaceCode",
                "BaseSurfaceName",
                "StrataAssignmentSource",
                "AssignedBestExpertWell",
                "AssignmentBestSimilarityScore",
                "AssignmentScoreMargin",
            ]
        )

    work_df = resolved_df.copy()
    work_df["DepthMin"] = pd.to_numeric(work_df["DepthMin"], errors="coerce")
    work_df["DepthMax"] = pd.to_numeric(work_df["DepthMax"], errors="coerce")
    work_df["SegmentLength"] = work_df["DepthMax"] - work_df["DepthMin"]
    if "RangeOrder" not in work_df.columns:
        work_df["RangeOrder"] = np.arange(len(work_df), dtype=np.int64)
    if "IntervalKey" not in work_df.columns:
        work_df["IntervalKey"] = [
            f"{int(order) + 1:03d}"
            for order in pd.to_numeric(work_df["RangeOrder"], errors="coerce").fillna(0).astype(int).tolist()
        ]
    work_df["SegmentID"] = [
        f"segment_{int(order) + 1:03d}"
        for order in pd.to_numeric(work_df["RangeOrder"], errors="coerce").fillna(0).astype(int).tolist()
    ]
    work_df["AssignedStrataName"] = work_df["StrataName"].fillna("").astype(str).str.strip()

    out_df = work_df[
        [
            "WellName",
            "SegmentID",
            "RangeId",
            "DepthMin",
            "DepthMax",
            "SegmentLength",
            "AssignedStrataName",
            "TopSurfaceCode",
            "TopSurfaceName",
            "BaseSurfaceCode",
            "BaseSurfaceName",
            "StrataAssignmentSource",
            "AssignedBestExpertWell",
            "AssignmentBestSimilarityScore",
            "AssignmentScoreMargin",
        ]
    ].copy()
    return out_df.sort_values(["DepthMin", "DepthMax"]).reset_index(drop=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    target_sample_csv = Path(args.target_sample_csv)
    layer_dir = Path(args.layer_dir)
    stage1_library_dir = Path(args.stage1_library_dir)
    stage2_library_dir = Path(args.stage2_library_dir)
    output_root = Path(args.output_root)

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
    resolution_dir = result_root / "target_range_resolution"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    resolution_dir.mkdir(parents=True, exist_ok=True)

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

    available_library_strata = discover_available_library_strata(
        stage1_library_dir=stage1_library_dir,
        stage2_library_dir=stage2_library_dir,
    )
    if not available_library_strata:
        raise ValueError("No overlapping strata directories were found between stage1_library_dir and stage2_library_dir")

    outer_module = load_module(OUTER_FLOW_SCRIPT, "segmentation_only_outer_flow")
    registry_df = outer_module.build_expert_registry(
        stage1_library_dir=stage1_library_dir,
        stage2_library_dir=stage2_library_dir,
        target_strata=available_library_strata,
    )
    registry_df.to_csv(result_root / "expert_model_registry.csv", index=False, encoding="utf-8-sig")

    normalized_range_df = normalize_target_strata_range_df(
        strata_range_csv=target_range_csv,
        target_well_name=target_well_name,
    )
    resolved_range_df, resolution_info = resolve_target_strata_range_df(
        outer_module=outer_module,
        target_sample_csv=target_sample_csv,
        target_well_name=target_well_name,
        target_range_df=normalized_range_df,
        registry_df=registry_df,
        requested_log_features=requested_log_features,
        output_dir=resolution_dir,
        determination_mode=str(args.strata_determination_mode),
        determination_top_k=int(args.strata_determination_topk),
    )

    final_segmentation_df = build_final_segmentation_df(resolved_range_df)
    final_segmentation_csv = result_root / FINAL_SEGMENTATION_FILENAME
    final_segmentation_df.to_csv(final_segmentation_csv, index=False, encoding="utf-8-sig")

    summary = {
        "exp_id": str(args.exp_id),
        "target_well_name": target_well_name,
        "target_sample_csv": str(target_sample_csv),
        "target_sample_encoding": sample_encoding,
        "depth_col": depth_col,
        "layer_dir": str(layer_dir),
        "stage1_library_dir": str(stage1_library_dir),
        "stage2_library_dir": str(stage2_library_dir),
        "num_selected_surfaces": int(len(selected_surface_rows)),
        "num_crossed_surfaces": int(crossing_df["Status"].astype(str).eq("crossed").sum()) if not crossing_df.empty else 0,
        "num_generated_intervals": int(len(target_range_df)),
        "num_resolved_intervals": int(len(final_segmentation_df)),
        "pre_segmentation_csv": str(target_range_csv),
        "resolved_range_csv": str(resolution_dir / "resolved_target_strata_range.csv"),
        "final_segmentation_csv": str(final_segmentation_csv),
        "resolution_info": resolution_info,
    }
    with (result_root / "segmentation_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
