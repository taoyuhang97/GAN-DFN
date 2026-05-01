import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from manual_strata_workflow.common import (
    DEFAULT_TARGET_STRATA,
    DEFAULT_WELL_FILES,
    DEFAULT_WELL_NAMES,
    canonicalize_well_name,
    parse_json_list,
    parse_json_object,
    sanitize,
)
from manual_strata_workflow.dataset import (
    build_labeled_dataset,
    export_filtered_strata_dataset as export_filtered_strata_dataset_common,
)
from workflow_paths import RAW_POINT_SCRIPT_PATH, WORKFLOW_ROOT


ROOT = WORKFLOW_ROOT
RUN_REFINE_SCRIPT = RAW_POINT_SCRIPT_PATH
DEFAULT_SAMPLE_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝样本"
)
DEFAULT_RAW_LABEL_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/成像测井/裂缝标注"
)
DEFAULT_SAVE_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝位置预测/基于密度的裂缝点位分析/裂缝点位精细化/地层划分验证"
)
DEFAULT_DOCX_PATH = Path(r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/实验记录/实验记录20260323.docx")
FLOW_RESULT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果"
)
DEFAULT_SAVE_ROOT = FLOW_RESULT_ROOT / "stage2_refine"
DEFAULT_EXIST_DIR_MAP = {
    "沙三段": str(
        Path(
            r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM/地层划分验证/exp_strata_s3_balanced_logscale_v1/沙三段_lstm"
        )
    ),
    "沙四段": str(
        Path(
            r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM/地层划分验证/exp_strata_ac_gr_v1/沙四段_lstm"
        )
    ),
}
DEFAULT_CONFIG_PROFILE_MAP = {
    "沙三段": "sand3_probmass_rule_v1",
    "沙四段": "sand4_balanced_v2",
}
def export_filtered_strata_dataset(
    labeled_df: pd.DataFrame,
    target_strata: str,
    selected_wells: list[str],
    source_columns: dict[str, list[str]],
    output_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    def _summary_builder(well_name: str, target_strata: str, sub: pd.DataFrame) -> dict:
        depth_col = ""
        depth_min = np.nan
        depth_max = np.nan
        strata_top = np.nan
        strata_base = np.nan
        if not sub.empty:
            depth_col = next((col for col in ["TVD", "DEPT", "MD"] if col in sub.columns), "")
            if depth_col:
                depth_min = float(pd.to_numeric(sub[depth_col], errors="coerce").min())
                depth_max = float(pd.to_numeric(sub[depth_col], errors="coerce").max())
            if "StrataTop" in sub.columns and sub["StrataTop"].notna().any():
                strata_top = float(pd.to_numeric(sub["StrataTop"], errors="coerce").dropna().iloc[0])
            if "StrataBase" in sub.columns and sub["StrataBase"].notna().any():
                strata_base = float(pd.to_numeric(sub["StrataBase"], errors="coerce").dropna().iloc[0])
        return {
            "WellName": well_name,
            "StrataName": target_strata,
            "SampleCount": int(len(sub)),
            "DepthColumn": depth_col,
            "DepthMin": depth_min,
            "DepthMax": depth_max,
            "StrataTop": strata_top,
            "StrataBase": strata_base,
        }

    summary_df, valid_wells = export_filtered_strata_dataset_common(
        labeled_df=labeled_df,
        target_strata=target_strata,
        selected_wells=selected_wells,
        source_columns=source_columns,
        output_dir=output_dir,
        summary_filename="strata_range.csv",
        summary_builder=_summary_builder,
    )
    summary_df.to_csv(output_dir / "formation_well_depth_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df, valid_wells


def list_verify_logs(exist_exp_dir: Path) -> dict[str, Path]:
    logs = {}
    if not exist_exp_dir.exists():
        raise FileNotFoundError(f"exist_exp_dir not found: {exist_exp_dir}")
    for verify_dir in sorted(exist_exp_dir.glob("verify_*")):
        csv_path = verify_dir / "val_pred_full_log.csv"
        if csv_path.exists():
            logs[verify_dir.name.replace("verify_", "", 1)] = csv_path
    if not logs:
        raise FileNotFoundError(f"No verify_* val_pred_full_log.csv found under: {exist_exp_dir}")
    return logs


def run_strata_refine_experiment(
    args,
    target_strata: str,
    strata_sample_dir: Path,
    strata_range_csv: Path,
    strata_exist_dir: Path,
    strata_save_dir: Path,
    valid_wells: list[str],
) -> pd.DataFrame:
    exp_id = f"{args.exp_id}_{target_strata}"
    explicit_profile = str(getattr(args, "config_profile", "") or "").strip()
    config_profile_map = parse_json_object(getattr(args, "config_profile_map_json", ""))
    if explicit_profile:
        profile_for_strata = explicit_profile
    else:
        profile_for_strata = str(config_profile_map.get(str(target_strata), "")).strip()
    cmd = [
        sys.executable,
        str(RUN_REFINE_SCRIPT),
        "--exp-id",
        exp_id,
        "--config-profile",
        profile_for_strata,
        "--exist-exp-dir",
        str(strata_exist_dir),
        "--sample-dir",
        str(strata_sample_dir),
        "--raw-label-dir",
        str(args.raw_label_dir),
        "--strata-name",
        str(target_strata),
        "--strata-range-csv",
        str(strata_range_csv),
        "--docx-path",
        str(args.docx_path),
        "--result-dir",
        str(strata_save_dir),
        "--well-names",
        ",".join(valid_wells),
        "--gt-dev-rule",
        str(args.gt_dev_rule),
        "--prob-weight-gamma",
        str(args.prob_weight_gamma),
        "--pred-min-points-per-segment",
        str(args.pred_min_points_per_segment),
        "--rounding-mode",
        str(args.rounding_mode),
        "--fallback-count-mode",
        str(args.fallback_count_mode),
        "--count-train-mode",
        str(args.count_train_mode),
        "--positive-floor-prob-min",
        str(args.positive_floor_prob_min),
        "--positive-floor-allowed-labels",
        str(args.positive_floor_allowed_labels),
        "--positive-floor-target-wells",
        str(args.positive_floor_target_wells),
        "--type-calibration-mode",
        str(args.type_calibration_mode),
        "--type-calibration-scale-min",
        str(args.type_calibration_scale_min),
        "--type-calibration-scale-max",
        str(args.type_calibration_scale_max),
        "--count-fusion-mode",
        str(args.count_fusion_mode),
        "--fusion-learned-weight",
        str(args.fusion_learned_weight),
        "--global-post-scale-mode",
        str(args.global_post_scale_mode),
        "--global-post-scale-min",
        str(args.global_post_scale_min),
        "--global-post-scale-max",
        str(args.global_post_scale_max),
        "--selective-post-mode",
        str(args.selective_post_mode),
        "--selective-post-target-wells",
        str(args.selective_post_target_wells),
        "--selective-post-feature",
        str(args.selective_post_feature),
        "--selective-post-feature-2",
        str(args.selective_post_feature_2),
        "--selective-post-quantile",
        str(args.selective_post_quantile),
        "--selective-post-quantile-2",
        str(args.selective_post_quantile_2),
        "--selective-post-max-pred-float",
        str(args.selective_post_max_pred_float),
        "--soft-negative-weight",
        str(args.soft_negative_weight),
        "--train-distance-weight-mode",
        str(args.train_distance_weight_mode),
        "--train-distance-csv",
        str(args.train_distance_csv),
        "--train-distance-weight-power",
        str(args.train_distance_weight_power),
        "--train-distance-weight-min",
        str(args.train_distance_weight_min),
        "--train-distance-weight-max",
        str(args.train_distance_weight_max),
        "--boundary-expand-mode",
        str(args.boundary_expand_mode),
        "--boundary-expand-prob-min",
        str(args.boundary_expand_prob_min),
        "--boundary-expand-max-steps",
        str(args.boundary_expand_max_steps),
        "--boundary-expand-max-depth",
        str(args.boundary_expand_max_depth),
        "--gate-mode",
        str(args.gate_mode),
        "--gate-prob-threshold",
        str(args.gate_prob_threshold),
        "--gate-target-wells",
        str(args.gate_target_wells),
        "--count-bin-mode",
        str(args.count_bin_mode),
        "--count-bin-blend-weight",
        str(args.count_bin_blend_weight),
        "--density-strength-mode",
        str(args.density_strength_mode),
        "--density-strength-basis-col",
        str(args.density_strength_basis_col),
        "--density-strength-scale-min",
        str(args.density_strength_scale_min),
        "--density-strength-scale-max",
        str(args.density_strength_scale_max),
        "--orientation-mode",
        str(args.orientation_mode),
        "--orientation-num-families",
        str(args.orientation_num_families),
        "--orientation-min-points-per-segment",
        str(args.orientation_min_points_per_segment),
        "--model-name",
        str(args.model_name),
        "--model-alpha",
        str(args.model_alpha),
        "--model-max-iter",
        str(args.model_max_iter),
        "--tweedie-power",
        str(args.tweedie_power),
        "--feature-preset",
        str(args.feature_preset),
        "--feature-cols",
        str(args.feature_cols),
        "--xgb-n-estimators",
        str(args.xgb_n_estimators),
        "--xgb-max-depth",
        str(args.xgb_max_depth),
        "--xgb-learning-rate",
        str(args.xgb_learning_rate),
        "--xgb-subsample",
        str(args.xgb_subsample),
        "--xgb-colsample-bytree",
        str(args.xgb_colsample_bytree),
        "--hgb-learning-rate",
        str(args.hgb_learning_rate),
        "--hgb-max-depth",
        str(args.hgb_max_depth),
        "--save-model-artifacts",
        str(args.save_model_artifacts),
    ]
    if str(args.saved_model_root or "").strip():
        cmd.extend(["--saved-model-root", str(args.saved_model_root)])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    (strata_save_dir / "run_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (strata_save_dir / "run_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{target_strata} refine run failed with returncode={completed.returncode}\n"
            f"stdout_tail={completed.stdout[-2000:]}\n"
            f"stderr_tail={completed.stderr[-2000:]}"
        )

    result_csv = strata_save_dir / "raw_point_guided_refine_summary.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"Missing refine summary csv: {result_csv}")

    result_df = pd.read_csv(result_csv, encoding="utf-8-sig")
    result_df["StrataName"] = target_strata
    result_df["FirstStageExistDir"] = str(strata_exist_dir)
    result_df["ResultDir"] = str(strata_save_dir)
    return result_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--config-profile", default="")
    parser.add_argument(
        "--config-profile-map-json",
        default=json.dumps(DEFAULT_CONFIG_PROFILE_MAP, ensure_ascii=False),
    )
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--raw-label-dir", default=str(DEFAULT_RAW_LABEL_DIR))
    parser.add_argument("--docx-path", default=str(DEFAULT_DOCX_PATH))
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--well-names", default=",".join(DEFAULT_WELL_NAMES))
    parser.add_argument("--target-strata", default=",".join(DEFAULT_TARGET_STRATA))
    parser.add_argument(
        "--exist-exp-map-json",
        default=json.dumps(DEFAULT_EXIST_DIR_MAP, ensure_ascii=False),
    )
    parser.add_argument("--boundary-tolerance", type=float, default=1.0)
    parser.add_argument("--skip-run", action="store_true")

    parser.add_argument("--gt-dev-rule", default="any_density")
    parser.add_argument("--prob-weight-gamma", type=float, default=2.0)
    parser.add_argument("--pred-min-points-per-segment", type=int, default=0)
    parser.add_argument("--rounding-mode", default="round")
    parser.add_argument("--fallback-count-mode", default="")
    parser.add_argument("--count-train-mode", default="all_segments")
    parser.add_argument("--positive-floor-prob-min", type=float, default=0.7)
    parser.add_argument("--positive-floor-allowed-labels", default="other,short_strong")
    parser.add_argument("--positive-floor-target-wells", default="")
    parser.add_argument("--type-calibration-mode", default="none")
    parser.add_argument("--type-calibration-scale-min", type=float, default=0.4)
    parser.add_argument("--type-calibration-scale-max", type=float, default=1.3)
    parser.add_argument("--count-fusion-mode", default="learned_only")
    parser.add_argument("--fusion-learned-weight", type=float, default=0.7)
    parser.add_argument("--global-post-scale-mode", default="none")
    parser.add_argument("--global-post-scale-min", type=float, default=0.5)
    parser.add_argument("--global-post-scale-max", type=float, default=1.5)
    parser.add_argument("--selective-post-mode", default="none")
    parser.add_argument("--selective-post-target-wells", default="")
    parser.add_argument("--selective-post-feature", default="ProbMassPerLength")
    parser.add_argument("--selective-post-feature-2", default="ProbMean")
    parser.add_argument("--selective-post-quantile", type=float, default=0.5)
    parser.add_argument("--selective-post-quantile-2", type=float, default=0.5)
    parser.add_argument("--selective-post-max-pred-float", type=float, default=-1.0)
    parser.add_argument("--soft-negative-weight", type=float, default=0.5)
    parser.add_argument("--train-distance-weight-mode", default="none")
    parser.add_argument("--train-distance-csv", default="")
    parser.add_argument("--train-distance-weight-power", type=float, default=1.0)
    parser.add_argument("--train-distance-weight-min", type=float, default=0.7)
    parser.add_argument("--train-distance-weight-max", type=float, default=1.3)
    parser.add_argument("--boundary-expand-mode", default="none")
    parser.add_argument("--boundary-expand-prob-min", type=float, default=0.35)
    parser.add_argument("--boundary-expand-max-steps", type=int, default=2)
    parser.add_argument("--boundary-expand-max-depth", type=float, default=0.0)
    parser.add_argument("--gate-mode", default="none")
    parser.add_argument("--gate-prob-threshold", type=float, default=0.5)
    parser.add_argument("--gate-target-wells", default="")
    parser.add_argument("--count-bin-mode", default="none")
    parser.add_argument("--count-bin-blend-weight", type=float, default=0.7)
    parser.add_argument("--density-strength-mode", default="none")
    parser.add_argument("--density-strength-basis-col", default="ProbMassPerLength")
    parser.add_argument("--density-strength-scale-min", type=float, default=0.5)
    parser.add_argument("--density-strength-scale-max", type=float, default=3.0)
    parser.add_argument("--orientation-mode", default="none")
    parser.add_argument("--orientation-num-families", type=int, default=3)
    parser.add_argument("--orientation-min-points-per-segment", type=int, default=1)
    parser.add_argument("--model-name", default="tweedie")
    parser.add_argument("--model-alpha", type=float, default=0.03)
    parser.add_argument("--model-max-iter", type=int, default=3000)
    parser.add_argument("--tweedie-power", type=float, default=1.5)
    parser.add_argument("--feature-preset", default="high_density_compact_v1")
    parser.add_argument("--feature-cols", default="")
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-depth", type=int, default=3)
    parser.add_argument("--save-model-artifacts", type=int, default=1)
    parser.add_argument("--saved-model-root", default="")
    args = parser.parse_args()

    selected_wells = []
    for well_name in parse_json_list(args.well_names):
        canonical_name = canonicalize_well_name(well_name)
        if canonical_name and canonical_name not in selected_wells:
            selected_wells.append(canonical_name)
    target_strata = parse_json_list(args.target_strata)
    exist_exp_map = {str(key).strip(): Path(str(value)) for key, value in parse_json_object(args.exist_exp_map_json).items()}
    if not selected_wells:
        raise ValueError("No well names provided")
    if not target_strata:
        raise ValueError("No target strata provided")

    result_root = Path(args.result_dir) if args.result_dir else (DEFAULT_SAVE_ROOT / sanitize(args.exp_id))
    result_root.mkdir(parents=True, exist_ok=True)

    labeled_df, source_columns = build_labeled_dataset(
        data_dir=Path(args.sample_dir),
        selected_wells=selected_wells,
        boundary_tolerance=float(args.boundary_tolerance),
    )

    labeled_data_dir = result_root / "labeled_datasets"
    labeled_data_dir.mkdir(parents=True, exist_ok=True)
    for well_name in selected_wells:
        labeled_df[labeled_df["WellName"] == well_name].to_csv(
            labeled_data_dir / f"{well_name}_sample_labeled.csv",
            index=False,
            encoding="utf-8-sig",
        )

    label_summary = (
        labeled_df.groupby(["WellName", "StrataName"], dropna=False)
        .agg(
            SampleCount=("WellName", "size"),
            DepthMin=("DepthForStrata", "min"),
            DepthMax=("DepthForStrata", "max"),
            FractureRatio=("Frac_Azimuth", lambda s: float(s.notna().mean()) if len(s) else np.nan),
        )
        .reset_index()
    )
    label_summary.to_csv(result_root / "manual_strata_label_summary.csv", index=False, encoding="utf-8-sig")

    run_plan_rows = []
    metric_parts = []
    for strata_name in target_strata:
        if strata_name not in exist_exp_map:
            raise ValueError(f"Missing exist_exp_dir mapping for strata: {strata_name}")
        strata_exist_dir = Path(exist_exp_map[strata_name])
        verify_logs = list_verify_logs(strata_exist_dir)

        strata_tag = sanitize(strata_name)
        strata_data_dir = result_root / "filtered_datasets" / strata_tag
        strata_summary_df, valid_wells = export_filtered_strata_dataset(
            labeled_df=labeled_df,
            target_strata=strata_name,
            selected_wells=selected_wells,
            source_columns=source_columns,
            output_dir=strata_data_dir,
        )
        strata_range_csv = strata_data_dir / "strata_range.csv"
        valid_val_wells = [well for well in valid_wells if well in verify_logs]

        run_row = {
            "StrataName": strata_name,
            "FirstStageExistDir": str(strata_exist_dir),
            "StrataDataDir": str(strata_data_dir),
            "StrataRangeCsv": str(strata_range_csv),
            "ValidWells": valid_val_wells,
        }
        if len(valid_val_wells) < 2:
            run_row["RunStatus"] = "skip_not_enough_valid_wells"
            run_plan_rows.append(run_row)
            continue

        strata_save_dir = result_root / f"{strata_tag}_refine"
        strata_save_dir.mkdir(parents=True, exist_ok=True)
        run_row["ResultDir"] = str(strata_save_dir)

        if args.skip_run:
            run_row["RunStatus"] = "skip_run"
            run_plan_rows.append(run_row)
            continue

        result_df = run_strata_refine_experiment(
            args=args,
            target_strata=strata_name,
            strata_sample_dir=strata_data_dir,
            strata_range_csv=strata_range_csv,
            strata_exist_dir=strata_exist_dir,
            strata_save_dir=strata_save_dir,
            valid_wells=valid_val_wells,
        )
        metric_parts.append(result_df)
        strata_summary_df.to_csv(
            strata_save_dir / "formation_well_depth_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        run_row["RunStatus"] = "ok"
        run_plan_rows.append(run_row)

    pd.DataFrame(run_plan_rows).to_csv(result_root / "formation_run_plan.csv", index=False, encoding="utf-8-sig")

    if metric_parts:
        summary_df = pd.concat(metric_parts, ignore_index=True)
        summary_df.to_csv(result_root / "manual_strata_refine_summary.csv", index=False, encoding="utf-8-sig")
    else:
        summary_df = pd.DataFrame()

    summary = {
        "exp_id": args.exp_id,
        "config_profile": str(args.config_profile),
        "config_profile_map": parse_json_object(args.config_profile_map_json),
        "result_root": str(result_root),
        "sample_dir": str(args.sample_dir),
        "raw_label_dir": str(args.raw_label_dir),
        "well_names": selected_wells,
        "target_strata": target_strata,
        "exist_exp_map": {key: str(value) for key, value in exist_exp_map.items()},
        "label_summary_csv": str(result_root / "manual_strata_label_summary.csv"),
        "run_plan_csv": str(result_root / "formation_run_plan.csv"),
        "summary_csv": str(result_root / "manual_strata_refine_summary.csv") if not summary_df.empty else "",
    }
    with (result_root / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
