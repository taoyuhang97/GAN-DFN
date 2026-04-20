from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import (
    DEFAULT_LOG_FEATURES,
    DEFAULT_SAVE_ROOT,
    NEAREST_SCRIPT_PATH,
    OUTER_FLOW_SCRIPT,
    RAW_POINT_SCRIPT_PATH,
    load_module,
    parse_json_list,
    sanitize,
    sort_by_depth_if_possible,
)
from .stage_prediction import run_second_stage_prediction
from .strata_resolution import (
    build_target_segment_files,
    discover_available_library_strata,
    normalize_target_strata_range_df,
    resolve_target_strata_range_df,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--target-sample-csv", required=True)
    parser.add_argument("--target-well-name", required=True)
    parser.add_argument("--target-strata-range-csv", required=True)
    parser.add_argument("--stage1-library-dir", required=True)
    parser.add_argument("--stage2-library-dir", required=True)
    parser.add_argument("--requested-log-features-json", default=json.dumps(DEFAULT_LOG_FEATURES, ensure_ascii=False))
    parser.add_argument(
        "--strata-determination-mode",
        default="auto_if_missing",
        choices=["use_csv", "auto_if_missing", "always_auto"],
    )
    parser.add_argument("--strata-determination-topk", type=int, default=2)
    parser.add_argument("--unknown-strata-policy", default="skip", choices=["skip", "error"])
    parser.add_argument("--first-stage-threshold", type=float, default=np.nan)
    parser.add_argument("--result-dir", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    target_sample_csv = Path(args.target_sample_csv)
    target_range_csv = Path(args.target_strata_range_csv)
    stage1_library_dir = Path(args.stage1_library_dir)
    stage2_library_dir = Path(args.stage2_library_dir)
    if not target_sample_csv.exists():
        raise FileNotFoundError(f"Target sample CSV not found: {target_sample_csv}")
    if not target_range_csv.exists():
        raise FileNotFoundError(f"Target strata range CSV not found: {target_range_csv}")
    if not stage1_library_dir.exists():
        raise FileNotFoundError(f"Stage1 library dir not found: {stage1_library_dir}")
    if not stage2_library_dir.exists():
        raise FileNotFoundError(f"Stage2 library dir not found: {stage2_library_dir}")

    requested_log_features = parse_json_list(args.requested_log_features_json)
    if not requested_log_features:
        raise ValueError("requested_log_features cannot be empty")

    result_root = Path(args.result_dir) if str(args.result_dir).strip() else (DEFAULT_SAVE_ROOT / sanitize(args.exp_id))
    result_root.mkdir(parents=True, exist_ok=True)

    outer_module = load_module(OUTER_FLOW_SCRIPT, "deploy_outer_flow_module")
    nearest_module = load_module(NEAREST_SCRIPT_PATH, "deploy_nearest_module")
    raw_module = load_module(RAW_POINT_SCRIPT_PATH, "deploy_raw_point_module")

    available_library_strata = discover_available_library_strata(
        stage1_library_dir=stage1_library_dir,
        stage2_library_dir=stage2_library_dir,
    )
    if not available_library_strata:
        raise ValueError("No overlapping strata directories were found between stage1_library_dir and stage2_library_dir")
    registry_df = outer_module.build_expert_registry(
        stage1_library_dir=stage1_library_dir,
        stage2_library_dir=stage2_library_dir,
        target_strata=available_library_strata,
    )
    registry_df.to_csv(result_root / "expert_model_registry.csv", index=False, encoding="utf-8-sig")

    target_range_input_df = normalize_target_strata_range_df(
        strata_range_csv=target_range_csv,
        target_well_name=str(args.target_well_name),
    )
    target_range_df, strata_resolution_info = resolve_target_strata_range_df(
        outer_module=outer_module,
        target_sample_csv=target_sample_csv,
        target_well_name=str(args.target_well_name),
        target_range_df=target_range_input_df,
        registry_df=registry_df,
        requested_log_features=requested_log_features,
        output_dir=result_root / "target_range_resolution",
        determination_mode=str(args.strata_determination_mode),
        determination_top_k=int(args.strata_determination_topk),
    )
    segment_paths, merged_range_df, _ = build_target_segment_files(
        target_sample_csv=target_sample_csv,
        target_well_name=str(args.target_well_name),
        target_range_df=target_range_df,
        output_dir=result_root / "target_segments",
    )
    if not segment_paths:
        raise ValueError("No target strata segment files were exported")

    target_strata = list(segment_paths.keys())

    available_strata = set(registry_df["strata_name"].astype(str).tolist()) if not registry_df.empty else set()
    unknown_rows = []
    known_strata = []
    for strata_name in target_strata:
        if strata_name in available_strata:
            known_strata.append(strata_name)
        else:
            unknown_rows.append(
                {
                    "WellName": args.target_well_name,
                    "StrataName": strata_name,
                    "Status": "missing_library_strata",
                }
            )
    if unknown_rows and args.unknown_strata_policy == "error":
        raise ValueError(f"Missing expert library strata: {[row['StrataName'] for row in unknown_rows]}")

    known_segment_paths = {name: path for name, path in segment_paths.items() if name in known_strata}
    if not known_segment_paths:
        raise ValueError("No known strata remain after applying unknown_strata_policy")

    known_registry_df = registry_df[registry_df["strata_name"].astype(str).isin(known_strata)].copy()
    score_df = outer_module.compute_similarity_scores(
        segment_paths=known_segment_paths,
        registry_df=known_registry_df,
        requested_log_features=requested_log_features,
    )
    if score_df.empty:
        raise ValueError("No expert similarity rows were produced for known strata")
    score_df.to_csv(result_root / "expert_selection_scores.csv", index=False, encoding="utf-8-sig")

    selected_expert_df = outer_module.select_expert_for_strata(score_df)
    selected_expert_df.to_csv(result_root / "selected_expert_by_strata.csv", index=False, encoding="utf-8-sig")
    selected_map = {
        str(row["strata_name"]): row
        for row in selected_expert_df.to_dict(orient="records")
    }

    strata_status_rows = []
    stage1_frames = []
    segment_frames = []
    point_frames = []
    stage1_summary_rows = []
    stage2_summary_rows = []

    threshold_override = None if not np.isfinite(args.first_stage_threshold) else float(args.first_stage_threshold)
    for strata_name in target_strata:
        strata_output_dir = result_root / "prediction_by_strata" / sanitize(strata_name)
        strata_output_dir.mkdir(parents=True, exist_ok=True)

        if strata_name not in selected_map:
            strata_status_rows.append(
                {
                    "WellName": args.target_well_name,
                    "StrataName": strata_name,
                    "Status": "skipped_missing_selected_expert",
                    "SelectedExpertWell": "",
                    "TargetSampleCsv": str(segment_paths[strata_name]),
                }
            )
            continue

        selected_row = selected_map[strata_name]
        expert_well = str(selected_row["selected_expert_well"])
        stage1_info = nearest_module.run_first_stage_prediction(
            input_csv=segment_paths[strata_name],
            well_name=str(args.target_well_name),
            model_well=expert_well,
            model_dir=Path(str(selected_row["stage1_model_dir"])),
            output_dir=strata_output_dir,
            threshold_override=threshold_override,
        )
        stage1_df = pd.read_csv(stage1_info["output_csv"], encoding="utf-8-sig")
        stage1_df["StrataName"] = strata_name
        stage1_frames.append(stage1_df)
        stage1_summary_rows.append(
            {
                "WellName": args.target_well_name,
                "StrataName": strata_name,
                "SelectedExpertWell": expert_well,
                "Stage1ModelDir": str(selected_row["stage1_model_dir"]),
                "Stage1Threshold": stage1_info["threshold"],
                "Stage1NumInputRows": stage1_info["num_input_rows"],
                "Stage1NumRowsAfterClean": stage1_info["num_rows_after_clean"],
                "Stage1NumRemovedRows": stage1_info["num_removed_rows"],
                "Stage1NumSequences": stage1_info["num_sequences"],
                "Stage1NumPredictedPositiveCenters": stage1_info["num_predicted_positive_centers"],
                "Stage1OutputCsv": str(stage1_info["output_csv"]),
            }
        )

        pred_segment_df, pred_points_df, stage2_info = run_second_stage_prediction(
            raw_module=raw_module,
            target_well_name=str(args.target_well_name),
            strata_name=strata_name,
            stage1_pred_csv=Path(stage1_info["output_csv"]),
            selected_expert_row=selected_row,
            strata_range_df=merged_range_df,
            output_dir=strata_output_dir,
        )
        if not pred_segment_df.empty:
            pred_segment_df["StrataName"] = strata_name
            segment_frames.append(pred_segment_df)
        if not pred_points_df.empty:
            pred_points_df["StrataName"] = strata_name
            point_frames.append(pred_points_df)
        stage2_summary_rows.append(
            {
                "WellName": args.target_well_name,
                "StrataName": strata_name,
                "SelectedExpertWell": expert_well,
                "Stage2ModelDir": stage2_info["stage2_model_dir"],
                "ArtifactTarget": stage2_info["artifact_target"],
                "ArtifactType": stage2_info["artifact_type"],
                "OrientationMode": stage2_info["orientation_mode"],
                "OrientationNumFamilies": stage2_info["orientation_num_families"],
                "NumPredSegments": stage2_info["num_segments"],
                "NumLowconfSegments": stage2_info["num_lowconf_segments"],
                "NumPredPoints": stage2_info["num_pred_points"],
            }
        )
        strata_status_rows.append(
            {
                "WellName": args.target_well_name,
                "StrataName": strata_name,
                "Status": "ok",
                "SelectedExpertWell": expert_well,
                "TargetSampleCsv": str(segment_paths[strata_name]),
            }
        )

    strata_status_df = pd.DataFrame(strata_status_rows + unknown_rows)
    strata_status_df.to_csv(result_root / "deploy_strata_status.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stage1_summary_rows).to_csv(result_root / "stage1_deploy_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stage2_summary_rows).to_csv(result_root / "stage2_deploy_summary.csv", index=False, encoding="utf-8-sig")

    whole_stage1_df = sort_by_depth_if_possible(pd.concat(stage1_frames, ignore_index=True)) if stage1_frames else pd.DataFrame()
    whole_segment_df = sort_by_depth_if_possible(pd.concat(segment_frames, ignore_index=True)) if segment_frames else pd.DataFrame()
    whole_points_df = sort_by_depth_if_possible(pd.concat(point_frames, ignore_index=True)) if point_frames else pd.DataFrame()
    whole_stage1_df.to_csv(result_root / "whole_well_stage1_pred_full_log.csv", index=False, encoding="utf-8-sig")
    whole_segment_df.to_csv(result_root / "whole_well_pred_segment_summary.csv", index=False, encoding="utf-8-sig")
    whole_points_df.to_csv(result_root / "whole_well_pred_fracture_points.csv", index=False, encoding="utf-8-sig")

    summary = {
        "exp_id": args.exp_id,
        "target_well_name": str(args.target_well_name),
        "target_sample_csv": str(target_sample_csv),
        "target_strata_range_csv": str(target_range_csv),
        "strata_determination_mode": str(args.strata_determination_mode),
        "strata_determination_topk": int(args.strata_determination_topk),
        "strata_resolution": strata_resolution_info,
        "stage1_library_dir": str(stage1_library_dir),
        "stage2_library_dir": str(stage2_library_dir),
        "requested_log_features": requested_log_features,
        "unknown_strata_policy": str(args.unknown_strata_policy),
        "num_target_strata": int(len(target_strata)),
        "num_known_strata": int(len(known_strata)),
        "num_skipped_strata": int(len(strata_status_df[strata_status_df["Status"].astype(str).str.startswith("skipped") | strata_status_df["Status"].astype(str).eq("missing_library_strata")])),
        "whole_well_stage1_pred_csv": str(result_root / "whole_well_stage1_pred_full_log.csv"),
        "whole_well_segment_csv": str(result_root / "whole_well_pred_segment_summary.csv"),
        "whole_well_points_csv": str(result_root / "whole_well_pred_fracture_points.csv"),
        "status_csv": str(result_root / "deploy_strata_status.csv"),
        "selected_expert_csv": str(result_root / "selected_expert_by_strata.csv"),
        "score_csv": str(result_root / "expert_selection_scores.csv"),
        "result_root": str(result_root),
    }
    with (result_root / "deployment_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))
    return 0
