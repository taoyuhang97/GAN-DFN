import os
import copy
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def get_env_json_list(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON list")
    return parsed


def get_env_json_dict(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


LOG_MISSING_PLACEHOLDERS = {-999.25, -9999.0}
ZERO_AS_MISSING_LOG_FEATURES = {"GR"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 5
HALF = SEQ_LEN // 2
BATCH_SIZE = 32
LR = 1e-4
DROPOUT = 0.3
HIDDEN_DIM = 32
EPOCHS = 60
PATIENCE = 10
MIN_DELTA = 1e-4
DIST_THRESHOLD = 0.35
MIN_TRAIN_WELLS = 2
USE_AUGMENTATION = False
LOSS_NAME = "smooth_l1"
TARGET_COLS = ["P10"]
TARGET_WEIGHTS = [1.0]
TARGET_TRANSFORM_SCALES = {"P10": 1.0, "P21": 1.0, "P33": 10000.0}
SEIS_MODE = "3x3"
SAVE_PRED_FOR_ALL_CENTERS = False  # Default: only save predictions where GT density exists

DISTANCE_ANALYSIS_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\成像测井数据分布分析\well_distance_analysis"
DATA_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
SAVE_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝位置预测\基于密度的裂缝点位分析\裂缝密度预测\cnn+lstm"

imaging_well_features = ["AC", "GR"]
only_val_wells = []
custom_train_wells_map = {}
use_all_other_wells = False

SEQ_LEN = get_env_int("EXP_SEQ_LEN", SEQ_LEN)
HALF = SEQ_LEN // 2
BATCH_SIZE = get_env_int("EXP_BATCH_SIZE", BATCH_SIZE)
LR = get_env_float("EXP_LR", LR)
HIDDEN_DIM = get_env_int("EXP_HIDDEN_DIM", HIDDEN_DIM)
EPOCHS = get_env_int("EXP_EPOCHS", EPOCHS)
PATIENCE = get_env_int("EXP_PATIENCE", PATIENCE)
DIST_THRESHOLD = get_env_float("EXP_DIST_THRESHOLD", DIST_THRESHOLD)
MIN_TRAIN_WELLS = get_env_int("EXP_MIN_TRAIN_WELLS", MIN_TRAIN_WELLS)
USE_AUGMENTATION = get_env_bool("EXP_USE_AUGMENTATION", USE_AUGMENTATION)
LOSS_NAME = os.getenv("EXP_DENSITY_LOSS", LOSS_NAME)
TARGET_COLS = get_env_json_list("EXP_DENSITY_TARGET_COLS", TARGET_COLS)
TARGET_WEIGHTS = get_env_json_list("EXP_DENSITY_TARGET_WEIGHTS", TARGET_WEIGHTS)
TARGET_TRANSFORM_SCALES = get_env_json_dict("EXP_DENSITY_TARGET_TRANSFORM_SCALES", TARGET_TRANSFORM_SCALES)
SEIS_MODE = os.getenv("EXP_SEIS_MODE", SEIS_MODE)
imaging_well_features = get_env_json_list("EXP_IMAGING_FEATURES", imaging_well_features)
only_val_wells = get_env_json_list("EXP_ONLY_VAL_WELLS", only_val_wells)
custom_train_wells_map = get_env_json_dict("EXP_CUSTOM_TRAIN_WELLS_MAP", custom_train_wells_map)
use_all_other_wells = get_env_bool("EXP_USE_ALL_OTHER_WELLS", use_all_other_wells)
SAVE_PRED_FOR_ALL_CENTERS = get_env_bool(
    "EXP_SAVE_PRED_FOR_ALL_CENTERS",
    SAVE_PRED_FOR_ALL_CENTERS,
)
save_dir_override = os.getenv("EXP_SAVE_DIR")
if save_dir_override:
    SAVE_DIR = save_dir_override

os.makedirs(SAVE_DIR, exist_ok=True)

seis_features = [f"SEIS_{i}" for i in range(63)]
all_density_cols = ["P10", "P21", "P33"]
features = seis_features + imaging_well_features

DIST_MATRIX_FILES = {
    "3x3x7": "seismic_3x3x7_plus_imaging_distance_matrix.csv",
    "3x3": "seismic_3x3_plane_plus_imaging_distance_matrix.csv",
    "1": "seismic_single_amplitude_plus_imaging_distance_matrix.csv",
}


def clean_invalid_log_values(df, log_features):
    df = df.copy()
    summary_rows = []

    for col in log_features:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = series.isna() | series.isin(LOG_MISSING_PLACEHOLDERS) | (series <= -999)
        if col in ZERO_AS_MISSING_LOG_FEATURES:
            invalid_mask |= (series == 0)

        df[col] = series.mask(invalid_mask)

        if invalid_mask.any() and "WellName" in df.columns:
            invalid_counts = df.loc[invalid_mask].groupby("WellName").size()
            for well_name, invalid_count in invalid_counts.items():
                summary_rows.append(
                    {
                        "WellName": well_name,
                        "Feature": col,
                        "InvalidCount": int(invalid_count),
                    }
                )

    return df, pd.DataFrame(summary_rows)


def get_distance_matrix_path(seis_mode):
    matrix_name = DIST_MATRIX_FILES.get(seis_mode)
    if matrix_name is None:
        raise ValueError(f"Unsupported SEIS_MODE for distance matrix: {seis_mode}")

    matrix_path = os.path.join(DISTANCE_ANALYSIS_DIR, matrix_name)
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Distance matrix not found: {matrix_path}")

    return matrix_path


def select_train_wells(val_well, distance_df, threshold=0.35, min_wells=2):
    dists = distance_df.loc[val_well].drop(val_well)
    close_wells = dists[dists < threshold].index.tolist()

    if len(close_wells) >= min_wells:
        train_wells = close_wells
        mode = "threshold"
    else:
        train_wells = dists.sort_values().index[:min_wells].tolist()
        mode = "nearest"

    print(f"\n验证井: {val_well}")
    print("训练井选择模式:", mode)
    print("训练井:", train_wells)
    print("对应距离:", {w: round(float(dists[w]), 4) for w in train_wells})
    return train_wells


def validate_custom_train_wells(val_well, train_wells, all_wells):
    invalid = [w for w in train_wells if w not in all_wells]
    if invalid:
        raise ValueError(f"Custom train wells for {val_well} contain unknown wells: {invalid}")
    if val_well in train_wells:
        raise ValueError(f"Custom train wells for {val_well} cannot include itself")
    if not train_wells:
        raise ValueError(f"Custom train wells for {val_well} cannot be empty")
    return train_wells


def normalize_target_weights(target_cols, target_weights):
    if not target_weights:
        return [1.0] * len(target_cols)
    if len(target_weights) == 1 and len(target_cols) > 1:
        return [float(target_weights[0])] * len(target_cols)
    if len(target_weights) != len(target_cols):
        raise ValueError(
            f"TARGET_WEIGHTS length mismatch: len(target_cols)={len(target_cols)}, "
            f"len(target_weights)={len(target_weights)}"
        )
    return [float(v) for v in target_weights]


def validate_target_cols(target_cols):
    invalid = [col for col in target_cols if col not in all_density_cols]
    if invalid:
        raise ValueError(f"Unsupported target cols: {invalid}")
    if not target_cols:
        raise ValueError("TARGET_COLS cannot be empty")
    return target_cols


def transform_target_array(raw_array, target_cols, target_transform_scales):
    transformed = np.zeros_like(raw_array, dtype=np.float32)
    for idx, col in enumerate(target_cols):
        scale = float(target_transform_scales.get(col, 1.0))
        transformed[:, idx] = np.log1p(np.clip(raw_array[:, idx], a_min=0.0, a_max=None) * scale)
    return transformed


def inverse_transform_target_array(transformed_array, target_cols, target_transform_scales):
    restored = np.zeros_like(transformed_array, dtype=np.float32)
    for idx, col in enumerate(target_cols):
        scale = float(target_transform_scales.get(col, 1.0))
        restored[:, idx] = np.expm1(np.clip(transformed_array[:, idx], a_min=0.0, a_max=None)) / scale
    return np.clip(restored, a_min=0.0, a_max=None)


def reshape_seis(X_seis, mode):
    if mode == "3x3x7":
        X_seis = X_seis.reshape(-1, 3, 3, 7)
        X_seis = np.transpose(X_seis, (0, 3, 1, 2))
    elif mode == "3x3":
        X_seis = X_seis.reshape(-1, 3, 3, 7)
        X_seis = np.transpose(X_seis, (0, 3, 1, 2))
        X_seis = X_seis[:, 3]
        X_seis = X_seis[:, None, :, :]
    elif mode == "1":
        center_index = 7 * 4 + 3
        X_seis = X_seis[:, center_index]
        X_seis = X_seis.reshape(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported SEIS_MODE: {mode}")
    return X_seis


class DensityCNNLSTM(nn.Module):
    def __init__(self, seis_mode, log_dim, hidden_dim=32, out_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seis_mode = seis_mode

        if seis_mode == "3x3x7":
            self.cnn = nn.Sequential(
                nn.Conv2d(7, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.cnn_out = 16
        elif seis_mode == "3x3":
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.cnn_out = 8
        elif seis_mode == "1":
            self.cnn = None
            self.cnn_out = 1
        else:
            raise ValueError(f"Unsupported SEIS_MODE: {seis_mode}")

        self.cnn_dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(
            input_size=self.cnn_out + log_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, seis, log):
        batch_size, seq_len, channels, height, width = seis.shape

        if self.cnn is not None:
            seis = seis.view(batch_size * seq_len, channels, height, width)
            seis_feat = self.cnn(seis).view(batch_size, seq_len, -1)
            seis_feat = self.cnn_dropout(seis_feat)
        else:
            seis_feat = seis.view(batch_size, seq_len, 1)

        x = torch.cat([seis_feat, log], dim=2)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def build_sequences_by_well(df, X_seis, X_log, y_den, has_density, seq_len):
    x_seis_seq = []
    x_log_seq = []
    y_den_seq = []
    has_density_seq = []
    center_row_idx_seq = []
    half = seq_len // 2
    start = 0
    skipped_windows = 0

    for well in df["WellName"].unique():
        well_df = df[df["WellName"] == well]
        n_rows = len(well_df)
        row_in_well = well_df["ROW_IN_WELL"].values
        raw_row_idx = well_df["RAW_ROW_IDX"].values

        seis_well = X_seis[start:start + n_rows]
        log_well = X_log[start:start + n_rows]
        y_den_well = y_den[start:start + n_rows]
        has_density_well = has_density[start:start + n_rows]

        for i in range(half, n_rows - half):
            window_row_idx = row_in_well[i - half:i + half + 1]
            if not np.all(np.diff(window_row_idx) == 1):
                skipped_windows += 1
                continue

            x_seis_seq.append(seis_well[i - half:i + half + 1])
            x_log_seq.append(log_well[i - half:i + half + 1])
            y_den_seq.append(y_den_well[i])
            has_density_seq.append(has_density_well[i])
            center_row_idx_seq.append(raw_row_idx[i])

        start += n_rows

    if skipped_windows > 0:
        print(f"因删除缺失值导致窗口不连续，跳过序列数: {skipped_windows}")

    return (
        np.array(x_seis_seq),
        np.array(x_log_seq),
        np.array(y_den_seq),
        np.array(has_density_seq).astype(bool),
        np.array(center_row_idx_seq),
    )


def augment_training_data(x_seis, x_log, y_den):
    seis_noise = np.random.normal(0, 0.03, x_seis.shape)
    seis_amp = np.random.uniform(0.9, 1.1)
    x_seis_aug = x_seis * seis_amp + seis_noise
    log_noise = np.random.normal(0, 0.01, x_log.shape)
    x_log_aug = x_log + log_noise

    x_seis_new = np.concatenate([x_seis, x_seis_aug], axis=0)
    x_log_new = np.concatenate([x_log, x_log_aug], axis=0)
    y_den_new = np.concatenate([y_den, y_den], axis=0)
    return x_seis_new, x_log_new, y_den_new


def density_loss(pred, target, target_weights, loss_name):
    if loss_name == "mse":
        per_item = (pred - target) ** 2
    elif loss_name == "l1":
        per_item = torch.abs(pred - target)
    else:
        per_item = nn.functional.smooth_l1_loss(pred, target, reduction="none")

    weighted = per_item * target_weights.view(1, -1)
    return weighted.mean()


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]):
        return np.nan
    return r2_score(y_true, y_pred)


def compute_density_metrics(y_true, y_pred, target_cols):
    metrics = {}
    r2_values = []
    mae_values = []
    rmse_values = []

    for idx, col in enumerate(target_cols):
        y_true_col = y_true[:, idx]
        y_pred_col = y_pred[:, idx]
        mae = mean_absolute_error(y_true_col, y_pred_col)
        rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
        r2 = safe_r2(y_true_col, y_pred_col)
        bias = float(np.mean(y_pred_col - y_true_col))

        metrics[f"{col}_MAE"] = float(mae)
        metrics[f"{col}_RMSE"] = float(rmse)
        metrics[f"{col}_R2"] = float(r2) if not np.isnan(r2) else np.nan
        metrics[f"{col}_BIAS"] = bias

        mae_values.append(mae)
        rmse_values.append(rmse)
        if not np.isnan(r2):
            r2_values.append(r2)

    metrics["AVG_MAE"] = float(np.mean(mae_values))
    metrics["AVG_RMSE"] = float(np.mean(rmse_values))
    metrics["AVG_R2"] = float(np.mean(r2_values)) if r2_values else np.nan
    return metrics


def print_density_metrics(val_well, metrics, n_samples, target_cols):
    print(f"\nval_well={val_well}")
    print(f"有效密度样本数={n_samples}")
    for col in target_cols:
        print(
            f"{col}: "
            f"MAE={metrics[f'{col}_MAE']:.4f}, "
            f"RMSE={metrics[f'{col}_RMSE']:.4f}, "
            f"R2={metrics[f'{col}_R2']:.4f}, "
            f"BIAS={metrics[f'{col}_BIAS']:.4f}"
        )
    print(
        f"AVG: MAE={metrics['AVG_MAE']:.4f}, "
        f"RMSE={metrics['AVG_RMSE']:.4f}, "
        f"R2={metrics['AVG_R2']:.4f}"
    )


DIST_MATRIX_PATH = get_distance_matrix_path(SEIS_MODE)
TARGET_COLS = validate_target_cols(TARGET_COLS)
TARGET_WEIGHTS = normalize_target_weights(TARGET_COLS, TARGET_WEIGHTS)

csv_files = [
    os.path.join(DATA_DIR, "车660-1_sample.csv"),
    os.path.join(DATA_DIR, "车660-2_sample.csv"),
    os.path.join(DATA_DIR, "车662_sample.csv"),
    os.path.join(DATA_DIR, "车663_sample.csv"),
]

df_raw_list = []
for file_path in csv_files:
    well_df = pd.read_csv(file_path)
    well_df["WellName"] = os.path.basename(file_path).split("_")[0]
    well_df["ROW_IN_WELL"] = np.arange(len(well_df))
    df_raw_list.append(well_df.copy())

df_raw = pd.concat(df_raw_list, ignore_index=True)
df_raw["RAW_ROW_IDX"] = np.arange(len(df_raw))
df = df_raw.copy()

df, invalid_log_summary = clean_invalid_log_values(df, imaging_well_features)
if not invalid_log_summary.empty:
    print("检测到测井缺失占位值，已按缺失处理：")
    print(invalid_log_summary.sort_values(["WellName", "Feature"]).to_string(index=False))

for col in all_density_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["TARGET_AVAILABLE"] = df[TARGET_COLS].notna().all(axis=1)
df["HAS_DENSITY"] = df["TARGET_AVAILABLE"] & (df[TARGET_COLS].max(axis=1) > 0)
df[all_density_cols] = df[all_density_cols].fillna(0.0).clip(lower=0.0)

required_columns = features + all_density_cols + ["WellName", "ROW_IN_WELL", "RAW_ROW_IDX", "HAS_DENSITY"]
before_drop_counts = df.groupby("WellName").size().to_dict()
df = df[required_columns].dropna().copy()
after_drop_counts = df.groupby("WellName").size().to_dict()
if before_drop_counts != after_drop_counts:
    print("按当前建模特征删除缺失样本后，各井剩余样本：")
    for well_name in before_drop_counts:
        print(
            f"{well_name}: {after_drop_counts.get(well_name, 0)}/"
            f"{before_drop_counts.get(well_name, 0)} "
            f"(removed={before_drop_counts[well_name] - after_drop_counts.get(well_name, 0)})"
        )

positive_counts = df.groupby("WellName")["HAS_DENSITY"].sum().to_dict()
print("各井可用于密度训练的中心样本数：")
for well_name in sorted(positive_counts):
    print(f"{well_name}: {int(positive_counts[well_name])}")

well_names = df["WellName"].unique()
dist_df = pd.read_csv(DIST_MATRIX_PATH, index_col=0)

all_results = []

for val_well in well_names:
    if only_val_wells and val_well not in only_val_wells:
        continue

    print(f"\n================ 验证 {val_well} ================")

    if use_all_other_wells:
        train_wells = [well for well in well_names if well != val_well]
    elif val_well in custom_train_wells_map:
        train_wells = validate_custom_train_wells(
            val_well,
            custom_train_wells_map[val_well],
            well_names.tolist(),
        )
        print(f"\n验证井: {val_well}")
        print("训练井选择模式: custom")
        print("训练井:", train_wells)
    else:
        train_wells = select_train_wells(
            val_well,
            dist_df,
            threshold=DIST_THRESHOLD,
            min_wells=MIN_TRAIN_WELLS,
        )

    train_df = df[df["WellName"].isin(train_wells)].copy()
    val_df = df[df["WellName"] == val_well].copy()

    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    train_df_scaled[features] = scaler.fit_transform(train_df[features])
    val_df_scaled[features] = scaler.transform(val_df[features])

    x_train_all = train_df_scaled[features].values
    x_val_all = val_df_scaled[features].values
    x_train_seis = reshape_seis(x_train_all[:, :63], SEIS_MODE)
    x_val_seis = reshape_seis(x_val_all[:, :63], SEIS_MODE)
    x_train_log = x_train_all[:, 63:]
    x_val_log = x_val_all[:, 63:]

    y_train_den_raw = train_df[TARGET_COLS].values.astype(np.float32)
    y_val_den_raw = val_df[TARGET_COLS].values.astype(np.float32)
    train_has_density = train_df["HAS_DENSITY"].astype(np.int8).values
    val_has_density = val_df["HAS_DENSITY"].astype(np.int8).values

    (
        x_train_seis_seq_all,
        x_train_log_seq_all,
        y_train_den_seq_all,
        train_has_density_seq,
        _,
    ) = build_sequences_by_well(
        train_df,
        x_train_seis,
        x_train_log,
        y_train_den_raw,
        train_has_density,
        SEQ_LEN,
    )

    (
        x_val_seis_seq_all,
        x_val_log_seq_all,
        y_val_den_seq_all,
        val_has_density_seq,
        val_center_row_idx,
    ) = build_sequences_by_well(
        val_df,
        x_val_seis,
        x_val_log,
        y_val_den_raw,
        val_has_density,
        SEQ_LEN,
    )

    train_mask = train_has_density_seq.astype(bool)
    val_mask = val_has_density_seq.astype(bool)

    if train_mask.sum() == 0:
        print(f"{val_well} 跳过：训练集中无可用密度样本")
        continue
    if val_mask.sum() == 0:
        print(f"{val_well} 跳过：验证集中无可用密度样本")
        continue

    x_train_seis_seq = x_train_seis_seq_all[train_mask]
    x_train_log_seq = x_train_log_seq_all[train_mask]
    y_train_den_seq_raw = y_train_den_seq_all[train_mask]

    x_val_seis_seq = x_val_seis_seq_all[val_mask]
    x_val_log_seq = x_val_log_seq_all[val_mask]
    y_val_den_seq_raw = y_val_den_seq_all[val_mask]

    y_train_den_seq_trans = transform_target_array(
        y_train_den_seq_raw,
        TARGET_COLS,
        TARGET_TRANSFORM_SCALES,
    )
    y_val_den_seq_trans = transform_target_array(
        y_val_den_seq_raw,
        TARGET_COLS,
        TARGET_TRANSFORM_SCALES,
    )

    target_scaler = StandardScaler()
    y_train_den_seq = target_scaler.fit_transform(y_train_den_seq_trans).astype(np.float32)
    y_val_den_seq = target_scaler.transform(y_val_den_seq_trans).astype(np.float32)

    if USE_AUGMENTATION:
        x_train_seis_seq, x_train_log_seq, y_train_den_seq = augment_training_data(
            x_train_seis_seq,
            x_train_log_seq,
            y_train_den_seq,
        )

    print("训练密度样本形状:")
    print("seis:", x_train_seis_seq.shape)
    print("log:", x_train_log_seq.shape)
    print("target:", y_train_den_seq.shape)

    x_train_seis_t = torch.tensor(x_train_seis_seq, dtype=torch.float32).to(DEVICE)
    x_train_log_t = torch.tensor(x_train_log_seq, dtype=torch.float32).to(DEVICE)
    y_train_den_t = torch.tensor(y_train_den_seq, dtype=torch.float32).to(DEVICE)
    x_val_seis_t = torch.tensor(x_val_seis_seq, dtype=torch.float32).to(DEVICE)
    x_val_log_t = torch.tensor(x_val_log_seq, dtype=torch.float32).to(DEVICE)
    y_val_den_t = torch.tensor(y_val_den_seq, dtype=torch.float32).to(DEVICE)
    x_val_seis_all_t = torch.tensor(x_val_seis_seq_all, dtype=torch.float32).to(DEVICE)
    x_val_log_all_t = torch.tensor(x_val_log_seq_all, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(x_train_seis_t, x_train_log_t, y_train_den_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = DensityCNNLSTM(
        seis_mode=SEIS_MODE,
        log_dim=len(imaging_well_features),
        hidden_dim=HIDDEN_DIM,
        out_dim=len(TARGET_COLS),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    target_weights_t = torch.tensor(TARGET_WEIGHTS, dtype=torch.float32).to(DEVICE)

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = -1
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for xb_seis, xb_log, yb_den in train_loader:
            optimizer.zero_grad()
            pred_den = model(xb_seis, xb_log)
            loss = density_loss(pred_den, yb_den, target_weights_t, LOSS_NAME)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            val_pred_den = model(x_val_seis_t, x_val_log_t)
            val_loss = density_loss(val_pred_den, y_val_den_t, target_weights_t, LOSS_NAME).item()
            val_pred_den_scaled = val_pred_den.detach().cpu().numpy()
            val_pred_den_trans = target_scaler.inverse_transform(val_pred_den_scaled)
            val_pred_den_raw = inverse_transform_target_array(
                val_pred_den_trans,
                TARGET_COLS,
                TARGET_TRANSFORM_SCALES,
            )
            val_metrics = compute_density_metrics(y_val_den_seq_raw, val_pred_den_raw, TARGET_COLS)

        metric_text = " | ".join(
            [f"{col}_R2={val_metrics[f'{col}_R2']:.4f}" for col in TARGET_COLS]
        )
        print(
            f"Epoch {epoch + 1} | "
            f"TrainLoss={avg_train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"{metric_text}"
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(
                f"EarlyStopping counter: {early_stop_counter}/{PATIENCE}, "
                f"best_val_loss: {best_val_loss:.4f}"
            )
            if early_stop_counter >= PATIENCE:
                print(
                    f"Early stopping triggered, best_val_loss: {best_val_loss:.4f}, "
                    f"best_epoch: {best_epoch + 1}"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        val_pred_den_all_scaled = model(x_val_seis_all_t, x_val_log_all_t).detach().cpu().numpy()

    val_pred_den_all_trans = target_scaler.inverse_transform(val_pred_den_all_scaled)
    val_pred_den_all_raw = inverse_transform_target_array(
        val_pred_den_all_trans,
        TARGET_COLS,
        TARGET_TRANSFORM_SCALES,
    )
    val_pred_den_raw = val_pred_den_all_raw[val_mask]
    metrics = compute_density_metrics(y_val_den_seq_raw, val_pred_den_raw, TARGET_COLS)
    print_density_metrics(val_well, metrics, int(val_mask.sum()), TARGET_COLS)

    result_row = {
        "val_well": val_well,
        "train_wells": ",".join(train_wells),
        "NumTrainDensitySeq": int(train_mask.sum()),
        "NumValDensitySeq": int(val_mask.sum()),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch + 1),
    }
    result_row.update(metrics)
    all_results.append(result_row)

    well_save_dir = os.path.join(SAVE_DIR, val_well)
    os.makedirs(well_save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(well_save_dir, "model.pth"))
    joblib.dump(scaler, os.path.join(well_save_dir, "scaler.pkl"))

    config = {
        "SEQ_LEN": SEQ_LEN,
        "SEIS_MODE": SEIS_MODE,
        "features": features,
        "imaging_well_features": imaging_well_features,
        "target_cols": TARGET_COLS,
        "target_transform_scales": TARGET_TRANSFORM_SCALES,
        "HIDDEN_DIM": HIDDEN_DIM,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "DIST_MATRIX_PATH": DIST_MATRIX_PATH,
        "DIST_THRESHOLD": DIST_THRESHOLD,
        "MIN_TRAIN_WELLS": MIN_TRAIN_WELLS,
        "USE_AUGMENTATION": USE_AUGMENTATION,
        "LOSS_NAME": LOSS_NAME,
        "TARGET_WEIGHTS": TARGET_WEIGHTS,
        "train_wells": train_wells,
        "val_well": val_well,
    }
    with open(os.path.join(well_save_dir, "config.json"), "w", encoding="utf-8-sig") as file_obj:
        json.dump(config, file_obj, indent=2, ensure_ascii=False)
    joblib.dump(target_scaler, os.path.join(well_save_dir, "target_scaler.pkl"))

    val_raw_df = df_raw[df_raw["WellName"] == val_well].copy()
    val_raw_df["CENTER_USED_FOR_SEQ"] = 0
    val_raw_df["CENTER_HAS_DENSITY"] = 0
    for col in TARGET_COLS:
        # Always keep RAW predictions for downstream point-refinement (it can be gated by PRED_LABEL later).
        val_raw_df[f"{col}_PRED_RAW"] = np.nan
        # Human-readable/analysis column: by default only filled where GT density exists.
        val_raw_df[f"{col}_PRED"] = np.nan

    pred_index = val_center_row_idx
    val_raw_df.loc[pred_index, "CENTER_USED_FOR_SEQ"] = 1
    val_raw_df.loc[pred_index[val_mask], "CENTER_HAS_DENSITY"] = 1

    for idx, col in enumerate(TARGET_COLS):
        # RAW predictions for every valid sequence center (includes non-fracture zones).
        val_raw_df.loc[pred_index, f"{col}_PRED_RAW"] = val_pred_den_all_raw[:, idx]
        if SAVE_PRED_FOR_ALL_CENTERS:
            # Debug/analysis mode: also fill {col}_PRED for every center.
            val_raw_df.loc[pred_index, f"{col}_PRED"] = val_pred_den_all_raw[:, idx]
        else:
            # Default: {col}_PRED is only defined where GT density exists (fracture zones).
            val_raw_df.loc[pred_index[val_mask], f"{col}_PRED"] = val_pred_den_all_raw[val_mask, idx]

    val_raw_df.to_csv(
        os.path.join(well_save_dir, "val_density_pred_full_log.csv"),
        index=False,
        encoding="utf-8-sig",
    )

result_df = pd.DataFrame(all_results)
result_path = os.path.join(SAVE_DIR, "loo_density_results.csv")
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

print("\n====== Density Leave-One-Well-Out 总结 ======")
print(result_df)
if not result_df.empty:
    print("\n平均性能：")
    print(result_df.mean(numeric_only=True))
print(f"\n结果已保存到: {result_path}")
