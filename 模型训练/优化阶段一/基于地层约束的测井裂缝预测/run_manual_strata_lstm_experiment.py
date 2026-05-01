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
    sanitize,
)
from manual_strata_workflow.dataset import (
    build_labeled_dataset,
    export_filtered_strata_dataset as export_filtered_strata_dataset_common,
)


ROOT = Path(__file__).resolve().parent
RUN_LSTM_SCRIPT = ROOT / "run_lstm_experiment.py"
DEFAULT_DATA_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝样本"
)
DEFAULT_SAVE_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM/地层划分验证"
)
DEFAULT_DOCX_PATH = Path(r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/实验记录/实验记录20260323.docx")
FLOW_RESULT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果"
)
DEFAULT_SAVE_ROOT = FLOW_RESULT_ROOT / "stage1_lstm"
DEFAULT_LSTM_FEATURES = ["AC", "GR"]
DISTANCE_ANALYSIS_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/成像测井数据分布分析/well_distance_analysis"
)
DIST_MATRIX_FILES = {
    "3x3x7": "seismic_3x3x7_plus_imaging_distance_matrix.csv",
    "3x3": "seismic_3x3_plane_plus_imaging_distance_matrix.csv",
    "1": "seismic_single_amplitude_plus_imaging_distance_matrix.csv",
}
def compute_valid_sequence_count(row_in_well: np.ndarray, seq_len: int) -> int:
    if len(row_in_well) < seq_len:
        return 0
    half = seq_len // 2
    count = 0
    for idx in range(half, len(row_in_well) - half):
        window = row_in_well[idx - half: idx + half + 1]
        if np.all(np.diff(window) == 1):
            count += 1
    return count


def load_distance_matrix(seis_mode: str) -> pd.DataFrame:
    matrix_name = DIST_MATRIX_FILES.get(str(seis_mode))
    if matrix_name is None:
        raise ValueError(f"Unsupported seis_mode: {seis_mode}")
    matrix_path = DISTANCE_ANALYSIS_DIR / matrix_name
    if not matrix_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(matrix_path, encoding="utf-8-sig", index_col=0)
    df.index = [str(idx).strip() for idx in df.index]
    df.columns = [str(col).strip() for col in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def select_train_wells_by_distance(
    val_well: str,
    valid_wells: list[str],
    distance_df: pd.DataFrame,
    threshold: float,
    min_wells: int,
) -> list[str]:
    candidate_wells = [well for well in valid_wells if well != val_well]
    if not candidate_wells:
        return []
    if distance_df.empty or val_well not in distance_df.index:
        return candidate_wells
    available = [
        well for well in candidate_wells
        if well in distance_df.columns and pd.notna(pd.to_numeric(distance_df.loc[val_well, well], errors="coerce"))
    ]
    if not available:
        return candidate_wells
    dists = pd.to_numeric(distance_df.loc[val_well, available], errors="coerce").dropna()
    close_wells = dists[dists < threshold].index.tolist()
    if len(close_wells) >= min_wells:
        return close_wells
    return dists.sort_values().index[: min(min_wells, len(dists))].tolist()
def export_filtered_strata_dataset(
    labeled_df: pd.DataFrame,
    target_strata: str,
    selected_wells: list[str],
    source_columns: dict[str, list[str]],
    output_dir: Path,
    seq_len: int,
    min_seq_per_well: int,
) -> tuple[pd.DataFrame, list[str]]:
    valid_wells_set: set[str] = set()

    def _summary_builder(well_name: str, target_strata: str, sub: pd.DataFrame) -> dict:
        depth_col = ""
        depth_min = np.nan
        depth_max = np.nan
        sequence_count = 0
        if not sub.empty:
            row_in_well = pd.to_numeric(sub["ROW_IN_WELL"], errors="coerce").dropna().to_numpy(dtype=np.int64)
            sequence_count = compute_valid_sequence_count(row_in_well, seq_len)
            if sequence_count >= min_seq_per_well:
                valid_wells_set.add(well_name)
            depth_col = next((col for col in ["TVD", "DEPT", "MD"] if col in sub.columns), "")
            if depth_col:
                depth_min = float(pd.to_numeric(sub[depth_col], errors="coerce").min())
                depth_max = float(pd.to_numeric(sub[depth_col], errors="coerce").max())
        return {
            "WellName": well_name,
            "StrataName": target_strata,
            "SampleCount": int(len(sub)),
            "ValidSequenceCount": int(sequence_count),
            "DepthColumn": depth_col,
            "DepthMin": depth_min,
            "DepthMax": depth_max,
        }

    summary_df, _ = export_filtered_strata_dataset_common(
        labeled_df=labeled_df,
        target_strata=target_strata,
        selected_wells=selected_wells,
        source_columns=source_columns,
        output_dir=output_dir,
        summary_filename="formation_well_sequence_counts.csv",
        summary_builder=_summary_builder,
    )
    valid_wells = [canonicalize_well_name(well_name) for well_name in selected_wells if canonicalize_well_name(well_name) in valid_wells_set]
    return summary_df, valid_wells


def build_custom_train_wells_map(
    valid_wells: list[str],
    train_selection_mode: str,
    distance_df: pd.DataFrame,
    dist_threshold: float,
    min_train_wells: int,
) -> dict[str, list[str]]:
    custom_map = {}
    max_available_train_wells = max(0, len(valid_wells) - 1)
    required_train_wells = max(1, min(int(min_train_wells), max_available_train_wells)) if max_available_train_wells > 0 else 1
    for val_well in valid_wells:
        if train_selection_mode == "distance":
            train_wells = select_train_wells_by_distance(
                val_well=val_well,
                valid_wells=valid_wells,
                distance_df=distance_df,
                threshold=dist_threshold,
                min_wells=required_train_wells,
            )
        else:
            train_wells = [well for well in valid_wells if well != val_well]

        if len(train_wells) >= required_train_wells:
            custom_map[val_well] = train_wells
    return custom_map


def run_stage1_experiment(
    args,
    target_strata: str,
    strata_data_dir: Path,
    strata_save_dir: Path,
    valid_val_wells: list[str],
    custom_train_wells_map: dict[str, list[str]],
) -> pd.DataFrame:
    exp_id = f"{args.exp_id}_{target_strata}"
    cmd = [
        sys.executable,
        str(RUN_LSTM_SCRIPT),
        "--exp-id",
        exp_id,
        "--seq-len",
        str(args.seq_len),
        "--seis-mode",
        str(args.seis_mode),
        "--features-json",
        json.dumps(args.lstm_features, ensure_ascii=False),
        "--docx-path",
        str(args.docx_path),
        "--data-dir",
        str(strata_data_dir),
        "--result-dir",
        str(strata_save_dir),
        "--selection-metric",
        str(args.selection_metric),
        "--use-dynamic-threshold",
        str(args.use_dynamic_threshold).lower(),
        "--use-density-regression",
        str(args.use_density_regression).lower(),
        "--use-domain-adversarial",
        str(args.use_domain_adversarial).lower(),
        "--use-all-other-wells",
        "false",
        "--use-wellwise-log-scaling",
        str(args.use_wellwise_log_scaling).lower(),
        "--use-well-balanced-sampling",
        str(args.use_well_balanced_sampling).lower(),
        "--use-oversample",
        str(args.use_oversample).lower(),
        "--manual-pos-weight",
        str(args.manual_pos_weight),
        "--target-ratio",
        str(args.target_ratio),
        "--edge-exclude",
        str(args.edge_exclude),
        "--max-pos-weight",
        str(args.max_pos_weight),
        "--use-dice-loss",
        str(args.use_dice_loss).lower(),
        "--dice-loss-weight",
        str(args.dice_loss_weight),
        "--fixed-threshold",
        str(args.fixed_threshold),
        "--thresh-search-min",
        str(args.thresh_search_min),
        "--thresh-search-max",
        str(args.thresh_search_max),
        "--thresh-search-step",
        str(args.thresh_search_step),
        "--use-selection-accuracy-floor",
        str(args.use_selection_accuracy_floor).lower(),
        "--min-selection-accuracy",
        str(args.min_selection_accuracy),
        "--use-threshold-constraints",
        str(args.use_threshold_constraints).lower(),
        "--min-selection-recall",
        str(args.min_selection_recall),
        "--min-selection-pos-ratio",
        str(args.min_selection_pos_ratio),
        "--min-selection-pos-ratio-scale",
        str(args.min_selection_pos_ratio_scale),
        "--use-selection-accuracy-floor-by-well-json",
        str(args.use_selection_accuracy_floor_by_well_json),
        "--min-selection-accuracy-by-well-json",
        str(args.min_selection_accuracy_by_well_json),
        "--use-threshold-constraints-by-well-json",
        str(args.use_threshold_constraints_by_well_json),
        "--min-selection-recall-by-well-json",
        str(args.min_selection_recall_by_well_json),
        "--min-selection-pos-ratio-by-well-json",
        str(args.min_selection_pos_ratio_by_well_json),
        "--min-selection-pos-ratio-scale-by-well-json",
        str(args.min_selection_pos_ratio_scale_by_well_json),
        "--min-train-wells",
        str(args.min_train_wells),
        "--dist-threshold",
        str(args.dist_threshold),
        "--only-val-wells",
        ",".join(valid_val_wells),
        "--custom-train-wells-json",
        json.dumps(custom_train_wells_map, ensure_ascii=False),
    ]
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
            f"{target_strata} run failed with returncode={completed.returncode}\n"
            f"stdout_tail={completed.stdout[-2000:]}\n"
            f"stderr_tail={completed.stderr[-2000:]}"
        )

    result_csv = strata_save_dir / "loo_results.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"Missing loo result csv: {result_csv}")

    result_df = pd.read_csv(result_csv, encoding="utf-8-sig")
    result_df["StrataName"] = target_strata
    result_df["ResultDir"] = str(strata_save_dir)
    return result_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--docx-path", default=str(DEFAULT_DOCX_PATH))
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--well-names", default=",".join(DEFAULT_WELL_NAMES))
    parser.add_argument("--target-strata", default=",".join(DEFAULT_TARGET_STRATA))
    parser.add_argument("--lstm-features-json", default=json.dumps(DEFAULT_LSTM_FEATURES, ensure_ascii=False))
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--seis-mode", default="3x3")
    parser.add_argument("--selection-metric", default="iou")
    parser.add_argument("--use-dynamic-threshold", default=True, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-density-regression", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-domain-adversarial", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-wellwise-log-scaling", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-well-balanced-sampling", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-oversample", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--manual-pos-weight", type=float, default=2.0)
    parser.add_argument("--target-ratio", type=float, default=0.35)
    parser.add_argument("--edge-exclude", type=int, default=1)
    parser.add_argument("--max-pos-weight", type=float, default=3.0)
    parser.add_argument("--use-dice-loss", default=True, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--thresh-search-min", type=float, default=0.30)
    parser.add_argument("--thresh-search-max", type=float, default=0.90)
    parser.add_argument("--thresh-search-step", type=float, default=0.02)
    parser.add_argument("--use-selection-accuracy-floor", default=True, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--min-selection-accuracy", type=float, default=0.8)
    parser.add_argument("--use-threshold-constraints", default=True, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--min-selection-recall", type=float, default=0.20)
    parser.add_argument("--min-selection-pos-ratio", type=float, default=0.03)
    parser.add_argument("--min-selection-pos-ratio-scale", type=float, default=0.35)
    parser.add_argument("--use-selection-accuracy-floor-by-well-json", default="")
    parser.add_argument("--min-selection-accuracy-by-well-json", default="")
    parser.add_argument("--use-threshold-constraints-by-well-json", default="")
    parser.add_argument("--min-selection-recall-by-well-json", default="")
    parser.add_argument("--min-selection-pos-ratio-by-well-json", default="")
    parser.add_argument("--min-selection-pos-ratio-scale-by-well-json", default="")
    parser.add_argument("--dist-threshold", type=float, default=0.35)
    parser.add_argument("--min-train-wells", type=int, default=1)
    parser.add_argument("--min-seq-per-well", type=int, default=30)
    parser.add_argument("--train-selection-mode", choices=["all_other", "distance"], default="all_other")
    parser.add_argument("--boundary-tolerance", type=float, default=1.0)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    args.lstm_features = parse_json_list(args.lstm_features_json)
    selected_wells = [canonicalize_well_name(item) for item in parse_json_list(args.well_names)]
    target_strata = parse_json_list(args.target_strata)
    if not selected_wells:
        raise ValueError("No well names provided")
    if not target_strata:
        raise ValueError("No target strata provided")

    result_root = Path(args.result_dir) if args.result_dir else (DEFAULT_SAVE_ROOT / sanitize(args.exp_id))
    result_root.mkdir(parents=True, exist_ok=True)

    labeled_df, source_columns = build_labeled_dataset(
        data_dir=Path(args.data_dir),
        selected_wells=selected_wells,
        boundary_tolerance=float(args.boundary_tolerance),
    )

    labeled_data_dir = result_root / "labeled_datasets"
    labeled_data_dir.mkdir(parents=True, exist_ok=True)
    for well_name in selected_wells:
        canonical_name = canonicalize_well_name(well_name)
        labeled_df[labeled_df["WellName"] == canonical_name].to_csv(
            labeled_data_dir / f"{canonical_name}_sample_labeled.csv",
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

    distance_df = load_distance_matrix(args.seis_mode) if args.train_selection_mode == "distance" else pd.DataFrame()
    run_plan_rows = []
    metric_parts = []

    for strata_name in target_strata:
        strata_tag = sanitize(strata_name)
        strata_data_dir = result_root / "filtered_datasets" / strata_tag
        strata_summary_df, valid_wells = export_filtered_strata_dataset(
            labeled_df=labeled_df,
            target_strata=strata_name,
            selected_wells=selected_wells,
            source_columns=source_columns,
            output_dir=strata_data_dir,
            seq_len=int(args.seq_len),
            min_seq_per_well=int(args.min_seq_per_well),
        )

        custom_train_wells_map = build_custom_train_wells_map(
            valid_wells=valid_wells,
            train_selection_mode=args.train_selection_mode,
            distance_df=distance_df,
            dist_threshold=float(args.dist_threshold),
            min_train_wells=int(args.min_train_wells),
        )
        valid_val_wells = [well for well in valid_wells if well in custom_train_wells_map]

        run_row = {
            "StrataName": strata_name,
            "StrataDataDir": str(strata_data_dir),
            "ValidWells": valid_val_wells,
            "CustomTrainWellsMap": custom_train_wells_map,
            "MinSeqPerWell": int(args.min_seq_per_well),
            "TrainSelectionMode": args.train_selection_mode,
        }

        if len(valid_val_wells) < 2:
            run_row["RunStatus"] = "skip_not_enough_valid_wells"
            run_plan_rows.append(run_row)
            continue

        strata_save_dir = result_root / f"{strata_tag}_lstm"
        strata_save_dir.mkdir(parents=True, exist_ok=True)
        run_row["ResultDir"] = str(strata_save_dir)

        if args.skip_run:
            run_row["RunStatus"] = "skip_run"
            run_plan_rows.append(run_row)
            continue

        result_df = run_stage1_experiment(
            args=args,
            target_strata=strata_name,
            strata_data_dir=strata_data_dir,
            strata_save_dir=strata_save_dir,
            valid_val_wells=valid_val_wells,
            custom_train_wells_map=custom_train_wells_map,
        )
        metric_parts.append(result_df)
        run_row["RunStatus"] = "ok"
        run_plan_rows.append(run_row)

        strata_summary_df.to_csv(
            strata_save_dir / "formation_well_sequence_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(run_plan_rows).to_csv(result_root / "formation_run_plan.csv", index=False, encoding="utf-8-sig")

    if metric_parts:
        summary_df = pd.concat(metric_parts, ignore_index=True)
        summary_df.to_csv(result_root / "manual_strata_lstm_summary.csv", index=False, encoding="utf-8-sig")
    else:
        summary_df = pd.DataFrame()

    summary = {
        "exp_id": args.exp_id,
        "result_root": str(result_root),
        "data_dir": str(args.data_dir),
        "well_names": selected_wells,
        "target_strata": target_strata,
        "lstm_features": args.lstm_features,
        "seq_len": int(args.seq_len),
        "seis_mode": args.seis_mode,
        "train_selection_mode": args.train_selection_mode,
        "use_wellwise_log_scaling": bool(args.use_wellwise_log_scaling),
        "use_well_balanced_sampling": bool(args.use_well_balanced_sampling),
        "use_oversample": bool(args.use_oversample),
        "manual_pos_weight": float(args.manual_pos_weight),
        "target_ratio": float(args.target_ratio),
        "edge_exclude": int(args.edge_exclude),
        "max_pos_weight": float(args.max_pos_weight),
        "use_dice_loss": bool(args.use_dice_loss),
        "dice_loss_weight": float(args.dice_loss_weight),
        "fixed_threshold": float(args.fixed_threshold),
        "thresh_search_min": float(args.thresh_search_min),
        "thresh_search_max": float(args.thresh_search_max),
        "thresh_search_step": float(args.thresh_search_step),
        "use_selection_accuracy_floor": bool(args.use_selection_accuracy_floor),
        "min_selection_accuracy": float(args.min_selection_accuracy),
        "use_threshold_constraints": bool(args.use_threshold_constraints),
        "min_selection_recall": float(args.min_selection_recall),
        "min_selection_pos_ratio": float(args.min_selection_pos_ratio),
        "min_selection_pos_ratio_scale": float(args.min_selection_pos_ratio_scale),
        "use_selection_accuracy_floor_by_well_json": str(args.use_selection_accuracy_floor_by_well_json),
        "min_selection_accuracy_by_well_json": str(args.min_selection_accuracy_by_well_json),
        "use_threshold_constraints_by_well_json": str(args.use_threshold_constraints_by_well_json),
        "min_selection_recall_by_well_json": str(args.min_selection_recall_by_well_json),
        "min_selection_pos_ratio_by_well_json": str(args.min_selection_pos_ratio_by_well_json),
        "min_selection_pos_ratio_scale_by_well_json": str(args.min_selection_pos_ratio_scale_by_well_json),
        "min_train_wells": int(args.min_train_wells),
        "min_seq_per_well": int(args.min_seq_per_well),
        "boundary_tolerance": float(args.boundary_tolerance),
        "label_summary_csv": str(result_root / "manual_strata_label_summary.csv"),
        "run_plan_csv": str(result_root / "formation_run_plan.csv"),
        "summary_csv": str(result_root / "manual_strata_lstm_summary.csv") if not summary_df.empty else "",
    }
    with (result_root / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
