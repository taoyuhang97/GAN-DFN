from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
FIRST_STAGE_DEFAULT_EXP_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\LSTM\单井验证\成像测井裂缝预测\cnn+lstm\exp40_missing_drop_ac_gr_3x3_seq5_AC_GR"
)
SECOND_STAGE_DEFAULT_EXP_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝位置预测\基于密度的裂缝点位分析\裂缝点位精细化\raw_point_tweedie_probmass_max_v1"
)
OUTPUT_BASE_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝位置预测\基于密度的裂缝点位分析\常规测井最近成像井预测"
)
DEFAULT_CANDIDATE_WELLS = ["车660-1", "车660-2", "车662", "车663"]
RAW_POINT_GUIDED_SCRIPT = ROOT / "raw_point_guided_segment_refine.py"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_MISSING_PLACEHOLDERS = {-999.25, -9999.0}
ZERO_AS_MISSING_LOG_FEATURES = {"GR"}
DROPOUT = 0.3


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "中", "文"} else "_" for ch in str(text)).strip("_")


def parse_list_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def format_verify_well_dir_name(well_name: str) -> str:
    well_name = str(well_name).strip()
    return well_name if well_name.startswith("verify_") else f"verify_{well_name}"


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def infer_well_name(input_csv: Path, explicit_well_name: str) -> str:
    if explicit_well_name:
        return str(explicit_well_name).strip()
    stem = input_csv.stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def parse_distance_json(raw: str, candidate_wells: list[str]) -> dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}

    maybe_path = Path(text)
    if maybe_path.exists():
        text = maybe_path.read_text(encoding="utf-8")

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--distance-map-json must be a JSON object or a JSON file path")

    result = {}
    for well_name in candidate_wells:
        if well_name not in payload:
            continue
        value = pd.to_numeric(pd.Series([payload[well_name]]), errors="coerce").iloc[0]
        if np.isfinite(value):
            result[well_name] = float(value)
    return result


def load_distance_map_from_csv(distance_csv: Path, target_well: str, candidate_wells: list[str]) -> dict[str, float]:
    df = pd.read_csv(distance_csv, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Distance CSV is empty: {distance_csv}")

    candidate_set = set(candidate_wells)
    columns = set(df.columns)

    if candidate_set.issubset(columns):
        if "WellName" in df.columns:
            subset = df[df["WellName"].astype(str) == str(target_well)]
            if not subset.empty:
                row = subset.iloc[0]
            elif len(df) == 1:
                row = df.iloc[0]
            else:
                raise ValueError(f"Target well {target_well} not found in distance CSV: {distance_csv}")
        elif "井名" in df.columns:
            subset = df[df["井名"].astype(str) == str(target_well)]
            if not subset.empty:
                row = subset.iloc[0]
            elif len(df) == 1:
                row = df.iloc[0]
            else:
                raise ValueError(f"Target well {target_well} not found in distance CSV: {distance_csv}")
        elif len(df) == 1:
            row = df.iloc[0]
        else:
            matrix_df = pd.read_csv(distance_csv, encoding="utf-8-sig", index_col=0)
            if target_well not in matrix_df.index:
                raise ValueError(f"Target well {target_well} not found in distance matrix: {distance_csv}")
            row = matrix_df.loc[target_well]

        result = {}
        for well_name in candidate_wells:
            value = pd.to_numeric(pd.Series([row[well_name]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                result[well_name] = float(value)
        return result

    long_cols = {col.lower(): col for col in df.columns}
    candidate_col = None
    distance_col = None
    target_col = None
    for key in ("imagingwell", "candidatewell", "well", "wellname", "成像井", "候选井"):
        if key in long_cols:
            candidate_col = long_cols[key]
            break
    for key in ("distance", "dist", "距离"):
        if key in long_cols:
            distance_col = long_cols[key]
            break
    for key in ("targetwell", "target", "井名", "wellname_target", "常规井"):
        if key in long_cols:
            target_col = long_cols[key]
            break

    if candidate_col and distance_col:
        work_df = df.copy()
        if target_col and target_well:
            work_df = work_df[work_df[target_col].astype(str) == str(target_well)].copy()
        result = {}
        for _, row in work_df.iterrows():
            candidate_name = str(row[candidate_col]).strip()
            if candidate_name not in candidate_set:
                continue
            value = pd.to_numeric(pd.Series([row[distance_col]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                result[candidate_name] = float(value)
        if result:
            return result

    matrix_df = pd.read_csv(distance_csv, encoding="utf-8-sig", index_col=0)
    if target_well not in matrix_df.index:
        raise ValueError(f"Unsupported distance CSV format or target well missing: {distance_csv}")

    result = {}
    for well_name in candidate_wells:
        if well_name not in matrix_df.columns:
            continue
        value = pd.to_numeric(pd.Series([matrix_df.loc[target_well, well_name]]), errors="coerce").iloc[0]
        if np.isfinite(value):
            result[well_name] = float(value)
    return result


def resolve_distance_map(
    target_well: str,
    candidate_wells: list[str],
    distance_csv: str,
    distance_map_json: str,
) -> dict[str, float]:
    distance_map = {}
    if distance_map_json:
        distance_map.update(parse_distance_json(distance_map_json, candidate_wells))
    if distance_csv:
        distance_map.update(load_distance_map_from_csv(Path(distance_csv), target_well, candidate_wells))
    if not distance_map:
        raise ValueError("No distance information provided. Use --selected-imaging-well or provide --distance-csv / --distance-map-json")
    return distance_map


def select_nearest_imaging_well(distance_map: dict[str, float]) -> tuple[str, float]:
    valid_items = [(well_name, float(distance)) for well_name, distance in distance_map.items() if np.isfinite(distance)]
    if not valid_items:
        raise ValueError("No valid finite distances found for candidate imaging wells")
    valid_items.sort(key=lambda item: (item[1], item[0]))
    return valid_items[0]


def clean_invalid_log_values(df: pd.DataFrame, log_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    summary_rows = []
    for col in log_features:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = series.isna() | series.isin(LOG_MISSING_PLACEHOLDERS) | (series <= -999)
        if col in ZERO_AS_MISSING_LOG_FEATURES:
            invalid_mask |= series == 0
        df[col] = series.mask(invalid_mask)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            summary_rows.append({"Feature": col, "InvalidCount": invalid_count})
    return df, pd.DataFrame(summary_rows)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class FractureCNNLSTM(nn.Module):
    def __init__(
        self,
        seis_mode: str,
        log_dim: int,
        n_domains: int,
        hidden_dim: int = 64,
        use_density_regression: bool = True,
        use_domain_adversarial: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_density_regression = use_density_regression
        self.use_domain_adversarial = use_domain_adversarial

        if self.use_domain_adversarial:
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_domains),
            )
        else:
            self.domain_classifier = None

        if seis_mode == "3x3x7":
            in_channels = 7
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.cnn_dropout = nn.Dropout(DROPOUT)
            cnn_out = 16
        elif seis_mode == "3x3":
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.cnn_dropout = nn.Dropout(DROPOUT)
            cnn_out = 8
        else:
            self.cnn = None
            self.cnn_dropout = None
            cnn_out = 1

        self.lstm = nn.LSTM(
            input_size=cnn_out + log_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc_cls = nn.Linear(hidden_dim, 1)
        self.fc_den = nn.Linear(hidden_dim, 3) if self.use_density_regression else None

    def forward(self, seis, log, alpha: float = 1.0):
        batch_size, time_steps, channels, height, width = seis.shape
        if self.cnn is not None:
            seis = seis.view(batch_size * time_steps, channels, height, width)
            feat = self.cnn(seis)
            feat = feat.view(batch_size, time_steps, -1)
            feat = self.cnn_dropout(feat)
        else:
            feat = seis.view(batch_size, time_steps, 1)

        x = torch.cat([feat, log], dim=2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)

        cls_out = self.fc_cls(out).squeeze(1)
        den_out = self.fc_den(out) if self.fc_den is not None else None

        if self.domain_classifier is not None:
            rev_feat = GRL.apply(out, alpha)
            domain_out = self.domain_classifier(rev_feat)
        else:
            domain_out = None
        return cls_out, den_out, domain_out, out


def reshape_seis(x_seis: np.ndarray, mode: str) -> np.ndarray:
    if mode == "3x3x7":
        x_seis = x_seis.reshape(-1, 3, 3, 7)
        x_seis = np.transpose(x_seis, (0, 3, 1, 2))
    elif mode == "3x3":
        x_seis = x_seis.reshape(-1, 3, 3, 7)
        x_seis = np.transpose(x_seis, (0, 3, 1, 2))
        center = 3
        x_seis = x_seis[:, center]
        x_seis = x_seis[:, None, :, :]
    elif mode == "1":
        center_index = 7 * 4 + 3
        x_seis = x_seis[:, center_index]
        x_seis = x_seis.reshape(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported SEIS_MODE: {mode}")
    return x_seis


def infer_n_domains_from_state_dict(state_dict: dict, config: dict) -> int:
    domain_weight_key = "domain_classifier.2.weight"
    if domain_weight_key in state_dict:
        return int(state_dict[domain_weight_key].shape[0])
    train_wells = config.get("train_wells", [])
    val_well = config.get("val_well")
    domain_names = list(train_wells)
    if val_well and val_well not in domain_names:
        domain_names.append(val_well)
    return max(len(domain_names), 1)


def build_inference_sequences(
    df_model: pd.DataFrame,
    x_seis: np.ndarray,
    x_log: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = seq_len // 2
    row_in_well = df_model["ROW_IN_WELL"].to_numpy(dtype=np.int64)
    raw_row_idx = df_model["RAW_ROW_IDX"].to_numpy(dtype=np.int64)

    seq_seis = []
    seq_log = []
    center_indices = []

    for i in range(half, len(df_model) - half):
        window_row_idx = row_in_well[i - half:i + half + 1]
        if not np.all(np.diff(window_row_idx) == 1):
            continue
        seq_seis.append(x_seis[i - half:i + half + 1])
        seq_log.append(x_log[i - half:i + half + 1])
        center_indices.append(raw_row_idx[i])

    if not seq_seis:
        raise ValueError("No valid continuous sequences can be built from input well after feature cleaning")

    return (
        np.asarray(seq_seis, dtype=np.float32),
        np.asarray(seq_log, dtype=np.float32),
        np.asarray(center_indices, dtype=np.int64),
    )


def run_first_stage_prediction(
    input_csv: Path,
    well_name: str,
    model_well: str,
    model_dir: Path,
    output_dir: Path,
    threshold_override: float | None,
) -> dict:
    config_path = model_dir / "config.json"
    scaler_path = model_dir / "scaler.pkl"
    model_path = model_dir / "model.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"First-stage config not found: {config_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"First-stage scaler not found: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"First-stage model not found: {model_path}")

    with config_path.open("r", encoding="utf-8-sig") as file_obj:
        config = json.load(file_obj)
    scaler = joblib.load(scaler_path)
    state_dict = torch.load(model_path, map_location=DEVICE)

    features = list(config["features"])
    seis_mode = str(config["SEIS_MODE"])
    seq_len = int(config["SEQ_LEN"])
    hidden_dim = int(config.get("hidden_dim", 32))
    use_density_regression = bool(config.get("use_density_regression", False))
    use_domain_adversarial = bool(config.get("use_domain_adversarial", False))
    log_features = features[63:]
    best_threshold = float(
        threshold_override
        if threshold_override is not None
        else config.get("best_thrESHOLD", config.get("best_threshold", 0.5))
    )

    model = FractureCNNLSTM(
        seis_mode=seis_mode,
        log_dim=len(log_features),
        n_domains=infer_n_domains_from_state_dict(state_dict, config),
        hidden_dim=hidden_dim,
        use_density_regression=use_density_regression,
        use_domain_adversarial=use_domain_adversarial,
    ).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    df_raw = pd.read_csv(input_csv, encoding="utf-8-sig")
    if df_raw.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    missing_features = [feature for feature in features if feature not in df_raw.columns]
    if missing_features:
        raise ValueError(f"Input CSV missing required first-stage features: {missing_features}")

    df_out = df_raw.copy()
    df_model = df_raw[features].copy()
    df_model["WellName"] = well_name
    df_model["ROW_IN_WELL"] = np.arange(len(df_model), dtype=np.int64)
    df_model["RAW_ROW_IDX"] = np.arange(len(df_model), dtype=np.int64)

    df_model, invalid_summary = clean_invalid_log_values(df_model, log_features)
    before_count = int(len(df_model))
    df_model = df_model[features + ["WellName", "ROW_IN_WELL", "RAW_ROW_IDX"]].dropna().copy()
    after_count = int(len(df_model))
    removed_count = before_count - after_count
    if df_model.empty:
        raise ValueError("All rows were removed after first-stage feature cleaning")

    scaled_features = scaler.transform(df_model[features].astype(np.float64))
    x_seis = reshape_seis(scaled_features[:, :63], seis_mode)
    x_log = scaled_features[:, 63:]
    x_seis_seq, x_log_seq, center_row_idx = build_inference_sequences(df_model, x_seis, x_log, seq_len)

    x_seis_t = torch.tensor(x_seis_seq, dtype=torch.float32, device=DEVICE)
    x_log_t = torch.tensor(x_log_seq, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        cls_out, den_out, _, _ = model(x_seis_t, x_log_t)
        prob = torch.sigmoid(cls_out).detach().cpu().numpy()
        pred = (prob >= best_threshold).astype(int)
        den_pred_real = None
        if use_density_regression and den_out is not None:
            den_pred_real = np.expm1(den_out.detach().cpu().numpy())

    df_out["PRED_PROB"] = np.nan
    df_out["PRED_LABEL"] = np.nan
    df_out["MODEL_WELL"] = model_well
    df_out["FIRST_STAGE_THRESHOLD"] = best_threshold

    df_out.loc[center_row_idx, "PRED_PROB"] = prob
    df_out.loc[center_row_idx, "PRED_LABEL"] = pred
    if den_pred_real is not None:
        df_out["P10_PRED"] = np.nan
        df_out["P21_PRED"] = np.nan
        df_out["P33_PRED"] = np.nan
        df_out.loc[center_row_idx, "P10_PRED"] = den_pred_real[:, 0]
        df_out.loc[center_row_idx, "P21_PRED"] = den_pred_real[:, 1]
        df_out.loc[center_row_idx, "P33_PRED"] = den_pred_real[:, 2]

    stage1_output_csv = output_dir / "stage1_pred_full_log.csv"
    df_out.to_csv(stage1_output_csv, index=False, encoding="utf-8-sig")

    invalid_summary_csv = output_dir / "stage1_invalid_log_summary.csv"
    if not invalid_summary.empty:
        invalid_summary.to_csv(invalid_summary_csv, index=False, encoding="utf-8-sig")

    return {
        "model_well": model_well,
        "model_dir": model_dir,
        "threshold": best_threshold,
        "output_csv": stage1_output_csv,
        "invalid_summary_csv": invalid_summary_csv if not invalid_summary.empty else None,
        "num_input_rows": int(len(df_raw)),
        "num_rows_after_clean": after_count,
        "num_removed_rows": removed_count,
        "num_sequences": int(len(center_row_idx)),
        "num_predicted_positive_centers": int(np.sum(pred)),
        "config_path": config_path,
    }


def run_second_stage_refine(
    predict_csv: Path,
    well_name: str,
    model_well: str,
    second_stage_exp_dir: Path,
    output_dir: Path,
) -> dict:
    raw_point_module = load_module(RAW_POINT_GUIDED_SCRIPT, "raw_point_guided_segment_refine_runtime")
    model_dir = second_stage_exp_dir / format_verify_well_dir_name(model_well)
    if not model_dir.exists():
        raise FileNotFoundError(f"Second-stage model directory not found: {model_dir}")

    artifact = raw_point_module.load_model_artifact(model_dir)
    config = raw_point_module.dataset_builder.make_config(
        exist_exp_dir=str(artifact.get("exist_exp_dir", "")),
        prob_weight_gamma=float(artifact.get("prob_weight_gamma", 2.0)),
        gt_min_points_per_segment=1,
        gt_rounding_mode="round",
        density_gt_col="P10",
    )

    well_data = raw_point_module.build_predict_segment_dataset(
        exist_csv=str(predict_csv),
        well_name=well_name,
        config=config,
    )
    if well_data["segment_df"].empty:
        raise ValueError("Second-stage refinement found no predicted development segments in first-stage output")

    pred_segment_df, lowconf_mask = raw_point_module.predict_segments_with_artifact(
        segment_df=well_data["segment_df"].copy(),
        artifact=artifact,
        include_gt_columns=False,
    )
    pred_points = raw_point_module.build_pred_points_from_segment_df(
        df_sorted=well_data["df_sorted"],
        payloads=well_data["payloads"],
        pred_segment_df=pred_segment_df,
        config=config,
    )

    segment_summary_csv = output_dir / "pred_segment_summary.csv"
    pred_points_csv = output_dir / "pred_fracture_points.csv"
    pred_segment_df.to_csv(segment_summary_csv, index=False, encoding="utf-8-sig")
    pred_points.to_csv(pred_points_csv, index=False, encoding="utf-8-sig")

    second_stage_info = {
        "model_well": model_well,
        "model_dir": model_dir,
        "artifact_type": artifact.get("artifact_type", ""),
        "artifact_target": artifact.get("artifact_target", ""),
        "segment_summary_csv": segment_summary_csv,
        "pred_points_csv": pred_points_csv,
        "num_segments": int(len(pred_segment_df)),
        "num_lowconf_segments": int(np.sum(lowconf_mask.astype(np.int64))),
        "num_pred_points": int(len(pred_points)),
        "feature_cols": artifact.get("feature_cols", []),
    }
    return second_stage_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--well-name", default="")
    parser.add_argument("--candidate-wells", default=",".join(DEFAULT_CANDIDATE_WELLS))
    parser.add_argument("--selected-imaging-well", default="")
    parser.add_argument("--distance-csv", default="")
    parser.add_argument("--distance-map-json", default="")
    parser.add_argument("--first-stage-exp-dir", default=str(FIRST_STAGE_DEFAULT_EXP_DIR))
    parser.add_argument("--second-stage-exp-dir", default=str(SECOND_STAGE_DEFAULT_EXP_DIR))
    parser.add_argument("--first-stage-threshold", type=float, default=np.nan)
    parser.add_argument("--output-dir", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    candidate_wells = parse_list_arg(args.candidate_wells)
    if not candidate_wells:
        raise ValueError("candidate_wells cannot be empty")

    well_name = infer_well_name(input_csv, args.well_name)
    if args.selected_imaging_well:
        model_well = str(args.selected_imaging_well).strip()
        selected_distance = np.nan
        distance_map = {}
    else:
        distance_map = resolve_distance_map(
            target_well=well_name,
            candidate_wells=candidate_wells,
            distance_csv=str(args.distance_csv),
            distance_map_json=str(args.distance_map_json),
        )
        model_well, selected_distance = select_nearest_imaging_well(distance_map)

    run_name = sanitize(f"{well_name}__nearest_{model_well}")
    run_dir = Path(args.output_dir) if args.output_dir else (OUTPUT_BASE_DIR / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    first_stage_exp_dir = Path(args.first_stage_exp_dir)
    first_stage_model_dir = first_stage_exp_dir / format_verify_well_dir_name(model_well)
    threshold_override = None if not np.isfinite(args.first_stage_threshold) else float(args.first_stage_threshold)

    stage1_info = run_first_stage_prediction(
        input_csv=input_csv,
        well_name=well_name,
        model_well=model_well,
        model_dir=first_stage_model_dir,
        output_dir=run_dir,
        threshold_override=threshold_override,
    )

    second_stage_info = run_second_stage_refine(
        predict_csv=Path(stage1_info["output_csv"]),
        well_name=well_name,
        model_well=model_well,
        second_stage_exp_dir=Path(args.second_stage_exp_dir),
        output_dir=run_dir,
    )

    distance_map_sorted = {
        well: float(distance_map[well])
        for well in sorted(distance_map, key=lambda key: distance_map[key])
    }
    run_summary = {
        "input_csv": input_csv,
        "well_name": well_name,
        "candidate_wells": candidate_wells,
        "selected_imaging_well": model_well,
        "selected_distance": selected_distance,
        "distance_map": distance_map_sorted,
        "first_stage_exp_dir": first_stage_exp_dir,
        "first_stage_model_dir": first_stage_model_dir,
        "second_stage_exp_dir": Path(args.second_stage_exp_dir),
        "second_stage_model_dir": second_stage_info["model_dir"],
        "output_dir": run_dir,
        "stage1": stage1_info,
        "stage2": second_stage_info,
    }
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(to_jsonable(run_summary), file_obj, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            to_jsonable(
                {
                    "well_name": well_name,
                    "selected_imaging_well": model_well,
                    "selected_distance": selected_distance,
                    "output_dir": run_dir,
                    "stage1_output_csv": stage1_info["output_csv"],
                    "stage2_points_csv": second_stage_info["pred_points_csv"],
                    "summary_json": summary_path,
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
