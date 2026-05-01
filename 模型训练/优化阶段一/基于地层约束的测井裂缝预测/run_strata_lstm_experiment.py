import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RUN_LSTM_SCRIPT = ROOT / "run_lstm_experiment.py"
DEFAULT_DATA_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝样本"
)
DEFAULT_SAVE_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM/地层划分验证"
)
DEFAULT_DOCX_PATH = Path(r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/实验记录/实验记录20260323.docx")
DISTANCE_ANALYSIS_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/成像测井数据分布分析/well_distance_analysis"
)
DIST_MATRIX_FILES = {
    "3x3x7": "seismic_3x3x7_plus_imaging_distance_matrix.csv",
    "3x3": "seismic_3x3_plane_plus_imaging_distance_matrix.csv",
    "1": "seismic_single_amplitude_plus_imaging_distance_matrix.csv",
}
DEFAULT_WELL_FILES = {
    "车660-1": "车660-1_sample.csv",
    "车660-2": "车660-2_sample.csv",
    "车662": "车662_sample.csv",
    "车663": "车663_sample.csv",
}
DEFAULT_STRATA_FEATURES = ["AC", "CAL", "CNL", "CON1", "DEN", "GR", "GRSL", "K", "KTH", "PE", "TH", "U"]
DEFAULT_LSTM_FEATURES = ["AC", "GR"]
LOG_MISSING_PLACEHOLDERS = {-999.25, -9999.0}
ZERO_AS_MISSING_LOG_FEATURES = {"GR"}


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text)).strip("_")


def parse_json_list(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError
        return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        return [item.strip() for item in raw.split(",") if item.strip()]


def load_distance_matrix(seis_mode: str) -> pd.DataFrame:
    matrix_name = DIST_MATRIX_FILES.get(str(seis_mode))
    if matrix_name is None:
        raise ValueError(f"Unsupported seis_mode: {seis_mode}")
    matrix_path = DISTANCE_ANALYSIS_DIR / matrix_name
    if not matrix_path.exists():
        raise FileNotFoundError(f"Distance matrix not found: {matrix_path}")
    df = pd.read_csv(matrix_path, encoding="utf-8-sig", index_col=0)
    df.index = [str(idx).strip() for idx in df.index]
    df.columns = [str(col).strip() for col in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def clean_invalid_log_values(df: pd.DataFrame, log_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    summary_rows = []
    for col in log_features:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        invalid_mask = series.isna() | series.isin(LOG_MISSING_PLACEHOLDERS) | (series <= -999)
        if col in ZERO_AS_MISSING_LOG_FEATURES:
            invalid_mask |= series == 0
        out[col] = series.mask(invalid_mask)
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())
            summary_rows.append({"Feature": col, "InvalidCount": invalid_count})
    return out, pd.DataFrame(summary_rows)


def rolling_mean_by_well(df: pd.DataFrame, feature_cols: list[str], window: int) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        smooth_col = f"{col}__smooth"
        out[smooth_col] = (
            out.groupby("WellName")[col]
            .transform(lambda s: s.rolling(window=window, center=True, min_periods=1).mean())
        )
    return out


def majority_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(labels) == 0:
        return labels.copy()
    half = window // 2
    out = labels.copy()
    for idx in range(len(labels)):
        left = max(0, idx - half)
        right = min(len(labels), idx + half + 1)
        values, counts = np.unique(labels[left:right], return_counts=True)
        out[idx] = values[np.argmax(counts)]
    return out


def merge_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    if len(labels) == 0 or min_len <= 1:
        return labels.copy()
    out = labels.copy()
    changed = True
    while changed:
        changed = False
        start = 0
        while start < len(out):
            end = start + 1
            while end < len(out) and out[end] == out[start]:
                end += 1
            seg_len = end - start
            if seg_len < min_len:
                left_label = out[start - 1] if start > 0 else None
                right_label = out[end] if end < len(out) else None
                if left_label is None and right_label is None:
                    start = end
                    continue
                if left_label is None:
                    replace_label = right_label
                elif right_label is None:
                    replace_label = left_label
                else:
                    left_len = 0
                    cursor = start - 1
                    while cursor >= 0 and out[cursor] == left_label:
                        left_len += 1
                        cursor -= 1
                    right_len = 0
                    cursor = end
                    while cursor < len(out) and out[cursor] == right_label:
                        right_len += 1
                        cursor += 1
                    replace_label = left_label if left_len >= right_len else right_label
                out[start:end] = replace_label
                changed = True
                break
            start = end
    return out


def build_segment_rows(well_df: pd.DataFrame) -> list[dict]:
    rows = []
    if well_df.empty:
        return rows
    start = 0
    labels = well_df["StrataCluster"].to_numpy()
    tvd = pd.to_numeric(well_df["TVD"], errors="coerce").to_numpy() if "TVD" in well_df.columns else np.full(len(well_df), np.nan)
    while start < len(well_df):
        end = start + 1
        while end < len(well_df) and labels[end] == labels[start]:
            end += 1
        seg = well_df.iloc[start:end]
        rows.append(
            {
                "WellName": str(seg["WellName"].iloc[0]),
                "StrataCluster": int(seg["StrataCluster"].iloc[0]),
                "StrataName": str(seg["StrataName"].iloc[0]),
                "SampleCount": int(len(seg)),
                "StartTVD": float(tvd[start]) if np.isfinite(tvd[start]) else np.nan,
                "EndTVD": float(tvd[end - 1]) if np.isfinite(tvd[end - 1]) else np.nan,
            }
        )
        start = end
    return rows


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


def select_train_wells(
    val_well: str,
    candidate_wells: list[str],
    distance_df: pd.DataFrame,
    threshold: float,
    min_wells: int,
) -> list[str]:
    if not candidate_wells:
        return []
    dists = distance_df.loc[val_well, candidate_wells].dropna()
    if dists.empty:
        return []
    close_wells = dists[dists < threshold].index.tolist()
    if len(close_wells) >= min_wells:
        return close_wells
    return dists.sort_values().index[: min(min_wells, len(dists))].tolist()


def infer_cluster_names(df: pd.DataFrame) -> dict[int, str]:
    cluster_stats = []
    for cluster_id, sub in df.groupby("StrataCluster"):
        cluster_stats.append(
            {
                "cluster_id": int(cluster_id),
                "mean_gr": float(pd.to_numeric(sub.get("GR"), errors="coerce").mean()) if "GR" in sub.columns else np.nan,
                "mean_ac": float(pd.to_numeric(sub.get("AC"), errors="coerce").mean()) if "AC" in sub.columns else np.nan,
                "mean_den": float(pd.to_numeric(sub.get("DEN"), errors="coerce").mean()) if "DEN" in sub.columns else np.nan,
                "mean_sh": float(pd.to_numeric(sub.get("SH"), errors="coerce").mean()) if "SH" in sub.columns else np.nan,
                "mean_sand": float(pd.to_numeric(sub.get("SAND"), errors="coerce").mean()) if "SAND" in sub.columns else np.nan,
            }
        )
    stats_df = pd.DataFrame(cluster_stats).sort_values("cluster_id").reset_index(drop=True)

    semantic_names = []
    for _, row in stats_df.iterrows():
        if np.isfinite(row["mean_sh"]) and np.isfinite(row["mean_sand"]):
            if row["mean_sh"] >= row["mean_sand"] + 10:
                semantic_names.append("shale_rich")
            elif row["mean_sand"] >= row["mean_sh"] + 10:
                semantic_names.append("sand_rich")
            else:
                semantic_names.append("mixed")
        else:
            semantic_names.append("cluster")

    if len(set(semantic_names)) == len(semantic_names):
        return {int(stats_df.iloc[idx]["cluster_id"]): semantic_names[idx] for idx in range(len(stats_df))}

    stats_df["gr_rank"] = stats_df["mean_gr"].rank(method="dense", ascending=True).astype(int)
    stats_df["ac_rank"] = stats_df["mean_ac"].rank(method="dense", ascending=True).astype(int)
    stats_df["den_rank"] = stats_df["mean_den"].rank(method="dense", ascending=False).astype(int)

    def rank_name(value: int, total: int) -> str:
        if total <= 1:
            return "mid"
        if value == 1:
            return "low"
        if value == total:
            return "high"
        return "mid"

    total_gr = int(stats_df["gr_rank"].max())
    total_ac = int(stats_df["ac_rank"].max())
    total_den = int(stats_df["den_rank"].max())
    mapping = {}
    for _, row in stats_df.iterrows():
        cluster_id = int(row["cluster_id"])
        name = (
            f"gr_{rank_name(int(row['gr_rank']), total_gr)}_"
            f"ac_{rank_name(int(row['ac_rank']), total_ac)}_"
            f"den_{rank_name(int(row['den_rank']), total_den)}"
        )
        mapping[cluster_id] = name
    return mapping


def run_stage1_experiment(
    args,
    cluster_id: int,
    cluster_name: str,
    cluster_data_dir: Path,
    cluster_save_dir: Path,
    valid_wells: list[str],
    custom_train_wells_map: dict[str, list[str]],
) -> dict:
    exp_id = f"{args.exp_id}_cluster{cluster_id}_{cluster_name}"
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
        str(cluster_data_dir),
        "--result-dir",
        str(cluster_save_dir),
        "--selection-metric",
        str(args.selection_metric),
        "--use-dynamic-threshold",
        str(args.use_dynamic_threshold).lower(),
        "--use-density-regression",
        str(args.use_density_regression).lower(),
        "--use-domain-adversarial",
        str(args.use_domain_adversarial).lower(),
        "--use-all-other-wells",
        str(args.use_all_other_wells).lower(),
        "--use-selection-accuracy-floor",
        str(args.use_selection_accuracy_floor).lower(),
        "--min-selection-accuracy",
        str(args.min_selection_accuracy),
        "--min-train-wells",
        str(args.min_train_wells),
        "--dist-threshold",
        str(args.dist_threshold),
        "--only-val-wells",
        ",".join(valid_wells),
        "--custom-train-wells-json",
        json.dumps(custom_train_wells_map, ensure_ascii=False),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    (cluster_save_dir / "strata_run_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cluster_save_dir / "strata_run_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cluster {cluster_id} run failed with returncode={completed.returncode}\n"
            f"stdout_tail={completed.stdout[-2000:]}\n"
            f"stderr_tail={completed.stderr[-2000:]}"
        )
    result_csv = cluster_save_dir / "loo_results.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"Missing loo result csv: {result_csv}")
    result_df = pd.read_csv(result_csv, encoding="utf-8-sig")
    result_df["StrataCluster"] = int(cluster_id)
    result_df["StrataName"] = str(cluster_name)
    result_df["ClusterResultDir"] = str(cluster_save_dir)
    return {
        "cluster_result_df": result_df,
        "save_dir": str(cluster_save_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--docx-path", default=str(DEFAULT_DOCX_PATH))
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--well-names", default=",".join(DEFAULT_WELL_FILES.keys()))
    parser.add_argument("--strata-features-json", default=json.dumps(DEFAULT_STRATA_FEATURES, ensure_ascii=False))
    parser.add_argument("--lstm-features-json", default=json.dumps(DEFAULT_LSTM_FEATURES, ensure_ascii=False))
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--label-smooth-window", type=int, default=9)
    parser.add_argument("--min-segment-len", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--seis-mode", default="3x3")
    parser.add_argument("--selection-metric", default="iou")
    parser.add_argument("--use-dynamic-threshold", default=True, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-density-regression", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-domain-adversarial", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-all-other-wells", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--use-selection-accuracy-floor", default=True, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--min-selection-accuracy", type=float, default=0.8)
    parser.add_argument("--dist-threshold", type=float, default=0.35)
    parser.add_argument("--min-train-wells", type=int, default=2)
    parser.add_argument("--min-seq-per-well", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    args.strata_features = parse_json_list(args.strata_features_json)
    args.lstm_features = parse_json_list(args.lstm_features_json)

    selected_wells = parse_json_list(args.well_names)
    if not selected_wells:
        raise ValueError("No well names provided")

    save_dir = Path(args.result_dir) if args.result_dir else (DEFAULT_SAVE_DIR / sanitize(args.exp_id))
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    invalid_summary_rows = []
    source_columns = {}
    for well_name in selected_wells:
        file_name = DEFAULT_WELL_FILES.get(well_name)
        if not file_name:
            raise ValueError(f"Unsupported well name: {well_name}")
        csv_path = Path(args.data_dir) / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing sample csv: {csv_path}")
        df = pd.read_csv(csv_path).copy()
        df["WellName"] = well_name
        if "ROW_IN_WELL" not in df.columns:
            df["ROW_IN_WELL"] = np.arange(len(df))
        df["FRACTURE_FLAG"] = df["Frac_Azimuth"].notna().astype(int) if "Frac_Azimuth" in df.columns else 0
        source_columns[well_name] = list(df.columns)
        df, invalid_df = clean_invalid_log_values(df, args.strata_features)
        if not invalid_df.empty:
            invalid_df["WellName"] = well_name
            invalid_summary_rows.append(invalid_df)
        raw_rows.append(df)

    all_df = pd.concat(raw_rows, ignore_index=True)
    missing_features = [col for col in args.strata_features if col not in all_df.columns]
    if missing_features:
        raise ValueError(f"Missing strata features in dataset: {missing_features}")

    clean_df = all_df.dropna(subset=args.strata_features).copy()
    if clean_df.empty:
        raise ValueError("No rows left after dropping NaN strata features")

    clean_df = clean_df.sort_values(["WellName", "ROW_IN_WELL"]).reset_index(drop=True)
    clean_df = rolling_mean_by_well(clean_df, args.strata_features, max(1, int(args.smooth_window)))
    smooth_feature_cols = [f"{col}__smooth" for col in args.strata_features]

    scaler = StandardScaler()
    x = scaler.fit_transform(clean_df[smooth_feature_cols].fillna(0.0))
    model = KMeans(n_clusters=int(args.n_clusters), random_state=int(args.random_state), n_init=20)
    clean_df["StrataClusterRaw"] = model.fit_predict(x)

    labeled_parts = []
    for well_name, sub in clean_df.groupby("WellName", sort=False):
        sub = sub.sort_values("ROW_IN_WELL").copy()
        raw_labels = sub["StrataClusterRaw"].to_numpy(dtype=np.int64)
        smoothed_labels = majority_smooth(raw_labels, max(1, int(args.label_smooth_window)))
        merged_labels = merge_short_segments(smoothed_labels, max(1, int(args.min_segment_len)))
        sub["StrataCluster"] = merged_labels
        labeled_parts.append(sub)
    labeled_df = pd.concat(labeled_parts, ignore_index=True)

    cluster_name_map = infer_cluster_names(labeled_df)
    labeled_df["StrataName"] = labeled_df["StrataCluster"].map(cluster_name_map)
    labeled_df["StrataLabel"] = labeled_df["StrataCluster"].map(
        lambda cluster_id: f"cluster{int(cluster_id)}_{cluster_name_map[int(cluster_id)]}"
    )

    invalid_summary_df = pd.concat(invalid_summary_rows, ignore_index=True) if invalid_summary_rows else pd.DataFrame()
    if not invalid_summary_df.empty:
        invalid_summary_df.to_csv(save_dir / "strata_invalid_log_summary.csv", index=False, encoding="utf-8-sig")

    cluster_summary = (
        labeled_df.groupby(["StrataCluster", "StrataName"])
        .agg(
            SampleCount=("WellName", "size"),
            WellCount=("WellName", "nunique"),
            FractureRatio=("FRACTURE_FLAG", "mean"),
            MeanGR=("GR", "mean"),
            MeanAC=("AC", "mean"),
            MeanDEN=("DEN", "mean"),
            MeanSH=("SH", "mean"),
            MeanSAND=("SAND", "mean"),
        )
        .reset_index()
        .sort_values("StrataCluster")
    )
    cluster_summary.to_csv(save_dir / "strata_cluster_summary.csv", index=False, encoding="utf-8-sig")

    segment_rows = []
    labeled_data_dir = save_dir / "labeled_datasets"
    labeled_data_dir.mkdir(parents=True, exist_ok=True)
    for well_name in selected_wells:
        well_df = labeled_df[labeled_df["WellName"] == well_name].copy()
        segment_rows.extend(build_segment_rows(well_df))
        well_df.to_csv(labeled_data_dir / f"{well_name}_sample_labeled.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(save_dir / "strata_segment_summary.csv", index=False, encoding="utf-8-sig")

    distance_df = load_distance_matrix(args.seis_mode)
    filtered_root = save_dir / "filtered_cluster_datasets"
    cluster_result_rows = []
    all_cluster_metric_parts = []

    for cluster_id, cluster_name in sorted(cluster_name_map.items()):
        cluster_dir = filtered_root / f"cluster{cluster_id}_{cluster_name}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        valid_wells = []
        custom_train_wells_map = {}
        well_rows = []
        for well_name in selected_wells:
            well_source_cols = source_columns[well_name]
            well_cluster_df = labeled_df[
                (labeled_df["WellName"] == well_name) & (labeled_df["StrataCluster"] == cluster_id)
            ].copy()
            if "ROW_IN_WELL" not in well_cluster_df.columns:
                well_cluster_df["ROW_IN_WELL"] = np.arange(len(well_cluster_df))
            sequence_count = compute_valid_sequence_count(
                pd.to_numeric(well_cluster_df["ROW_IN_WELL"], errors="coerce").dropna().to_numpy(dtype=np.int64),
                int(args.seq_len),
            )
            well_rows.append(
                {
                    "WellName": well_name,
                    "StrataCluster": int(cluster_id),
                    "StrataName": cluster_name,
                    "SampleCount": int(len(well_cluster_df)),
                    "ValidSequenceCount": int(sequence_count),
                }
            )
            if sequence_count >= int(args.min_seq_per_well):
                valid_wells.append(well_name)
            output_cols = [col for col in well_source_cols if col in well_cluster_df.columns]
            if "ROW_IN_WELL" not in output_cols:
                output_cols.append("ROW_IN_WELL")
            if well_cluster_df.empty:
                pd.DataFrame(columns=output_cols).to_csv(
                    cluster_dir / DEFAULT_WELL_FILES[well_name],
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                well_cluster_df.sort_values("ROW_IN_WELL")[output_cols].to_csv(
                    cluster_dir / DEFAULT_WELL_FILES[well_name],
                    index=False,
                    encoding="utf-8-sig",
                )

        well_count_df = pd.DataFrame(well_rows)
        well_count_df.to_csv(cluster_dir / "cluster_well_sequence_counts.csv", index=False, encoding="utf-8-sig")

        for val_well in list(valid_wells):
            candidate_train_wells = [well for well in valid_wells if well != val_well]
            train_wells = (
                candidate_train_wells
                if args.use_all_other_wells
                else select_train_wells(
                    val_well=val_well,
                    candidate_wells=candidate_train_wells,
                    distance_df=distance_df,
                    threshold=float(args.dist_threshold),
                    min_wells=int(args.min_train_wells),
                )
            )
            if train_wells:
                custom_train_wells_map[val_well] = train_wells
        valid_val_wells = [well for well in valid_wells if well in custom_train_wells_map]

        cluster_row = {
            "StrataCluster": int(cluster_id),
            "StrataName": cluster_name,
            "ClusterDataDir": str(cluster_dir),
            "ValidWells": valid_val_wells,
            "CustomTrainWellsMap": custom_train_wells_map,
        }

        if len(valid_val_wells) < 2:
            cluster_row["RunStatus"] = "skip_not_enough_valid_wells"
            cluster_result_rows.append(cluster_row)
            continue

        cluster_save_dir = save_dir / f"cluster{cluster_id}_{cluster_name}_lstm"
        cluster_save_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_run:
            cluster_row["RunStatus"] = "skip_run"
            cluster_row["ClusterResultDir"] = str(cluster_save_dir)
            cluster_result_rows.append(cluster_row)
            continue

        run_info = run_stage1_experiment(
            args=args,
            cluster_id=int(cluster_id),
            cluster_name=cluster_name,
            cluster_data_dir=cluster_dir,
            cluster_save_dir=cluster_save_dir,
            valid_wells=valid_val_wells,
            custom_train_wells_map=custom_train_wells_map,
        )
        all_cluster_metric_parts.append(run_info["cluster_result_df"])
        cluster_row["RunStatus"] = "ok"
        cluster_row["ClusterResultDir"] = run_info["save_dir"]
        cluster_result_rows.append(cluster_row)

    with (save_dir / "cluster_name_map.json").open("w", encoding="utf-8") as file_obj:
        json.dump({str(k): v for k, v in cluster_name_map.items()}, file_obj, ensure_ascii=False, indent=2)

    pd.DataFrame(cluster_result_rows).to_csv(save_dir / "cluster_run_plan.csv", index=False, encoding="utf-8-sig")

    if all_cluster_metric_parts:
        strata_metric_df = pd.concat(all_cluster_metric_parts, ignore_index=True)
        strata_metric_df.to_csv(save_dir / "strata_lstm_summary.csv", index=False, encoding="utf-8-sig")
    else:
        strata_metric_df = pd.DataFrame()

    summary = {
        "exp_id": args.exp_id,
        "save_dir": str(save_dir),
        "data_dir": str(args.data_dir),
        "well_names": selected_wells,
        "strata_features": args.strata_features,
        "lstm_features": args.lstm_features,
        "n_clusters": int(args.n_clusters),
        "smooth_window": int(args.smooth_window),
        "label_smooth_window": int(args.label_smooth_window),
        "min_segment_len": int(args.min_segment_len),
        "seq_len": int(args.seq_len),
        "min_seq_per_well": int(args.min_seq_per_well),
        "skip_run": bool(args.skip_run),
        "num_labeled_rows": int(len(labeled_df)),
        "num_cluster_runs": int(len([row for row in cluster_result_rows if row.get("RunStatus") == "ok"])),
        "cluster_summary_csv": str(save_dir / "strata_cluster_summary.csv"),
        "cluster_run_plan_csv": str(save_dir / "cluster_run_plan.csv"),
        "strata_lstm_summary_csv": str(save_dir / "strata_lstm_summary.csv") if not strata_metric_df.empty else "",
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
