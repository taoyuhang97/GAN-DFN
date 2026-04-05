import os
import glob
import joblib
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


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
                summary_rows.append({
                    "WellName": well_name,
                    "Feature": col,
                    "InvalidCount": int(invalid_count),
                })

    summary_df = pd.DataFrame(summary_rows)
    return df, summary_df


# ===================== 基本设置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 3
HALF = SEQ_LEN // 2
BATCH_SIZE = 32
LR = 1e-4
Dropout = 0.3
# 测井聚类阈值
DIST_THRESHOLD = 0.4
MIN_TRAIN_WELLS = 2
# ---------- Early Stopping ----------
EPOCHS = 50
patience = 10
min_delta = 1e-4

# 动态阈值
use_dynamic_threshold = True
# use_dynamic_threshold = False
best_thr = 0.5
configured_fixed_threshold = best_thr
selection_metric = "iou"
THRESH_SEARCH_MIN = 0.3
THRESH_SEARCH_MAX = 0.9
THRESH_SEARCH_STEP = 0.02
use_selection_accuracy_floor = True
MIN_SELECTION_ACCURACY = 0.80
use_threshold_constraints = True
MIN_SELECTION_RECALL = 0.20
MIN_SELECTION_POS_RATIO = 0.03
MIN_SELECTION_POS_RATIO_SCALE = 0.35
# selection_metric = "f1"
# selection_metric = "iou"
# 概率平滑
# use_smooth = True
use_smooth = False
# 小段处理
# use_short_segment_processing = True
use_short_segment_processing = False
# 过采样操作
# use_over_smaple = True
use_over_smaple = False
MAX_N_POS = 1
manual_pos_weight = 2.0
MAX_POS_WEIGHT = 3.0
target_ratio = 0.35
edge_exclude = 1
use_dice_loss = True
DICE_LOSS_WEIGHT = 0.5
# 特征重要性检查
show_features_importance = False
show_one_well_features_importance = False
all_importances = []
# 密度考量
use_density_regression = False
# 域偏差考量
use_domain_adversarial = False
use_all_other_wells = False
use_wellwise_log_scaling = False
use_well_balanced_sampling = False
only_val_wells = []
custom_train_wells_map = {}
min_selection_accuracy_by_well = {}
min_selection_recall_by_well = {}
min_selection_pos_ratio_by_well = {}
min_selection_pos_ratio_scale_by_well = {}
use_selection_accuracy_floor_by_well = {}
use_threshold_constraints_by_well = {}

# SEIS_MODE = "3x3x7"
SEIS_MODE = "3x3"
# SEIS_MODE = "1"

DISTANCE_ANALYSIS_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\成像测井数据分布分析\well_distance_analysis"
DATA_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
SAVE_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\LSTM\单井验证\成像测井裂缝预测\cnn+lstm"

# ===================== 特征定义 =====================
# imaging_well_features = ['AC', 'GR', 'CAL', 'CNL', 'PE', 'DEN', 'CON1', 'GRSL', 'K', 'KTH', 'TH', 'U']
# imaging_well_features = ['DEN', 'CON1', 'GRSL', 'AC', 'GR']
imaging_well_features = ['DEN', 'GRSL', 'AC', 'GR']
# imaging_well_features = ['CON1', 'GRSL', 'AC', 'GR']
# imaging_well_features = ['GRSL', 'AC', 'GR']
# imaging_well_features = ['CON1', 'AC', 'GR']
# imaging_well_features = ['CON1', 'GR']
# imaging_well_features = ['AC', 'GR']

SEQ_LEN = get_env_int("EXP_SEQ_LEN", SEQ_LEN)
HALF = SEQ_LEN // 2
SEIS_MODE = os.getenv("EXP_SEIS_MODE", SEIS_MODE)
imaging_well_features = get_env_json_list("EXP_IMAGING_FEATURES", imaging_well_features)
selection_metric = os.getenv("EXP_SELECTION_METRIC", selection_metric)
THRESH_SEARCH_MIN = get_env_float("EXP_THRESH_SEARCH_MIN", THRESH_SEARCH_MIN)
THRESH_SEARCH_MAX = get_env_float("EXP_THRESH_SEARCH_MAX", THRESH_SEARCH_MAX)
THRESH_SEARCH_STEP = get_env_float("EXP_THRESH_SEARCH_STEP", THRESH_SEARCH_STEP)
use_selection_accuracy_floor = get_env_bool(
    "EXP_USE_SELECTION_ACCURACY_FLOOR",
    use_selection_accuracy_floor,
)
MIN_SELECTION_ACCURACY = get_env_float(
    "EXP_MIN_SELECTION_ACCURACY",
    MIN_SELECTION_ACCURACY,
)
use_threshold_constraints = get_env_bool("EXP_USE_THRESHOLD_CONSTRAINTS", use_threshold_constraints)
MIN_SELECTION_RECALL = get_env_float("EXP_MIN_SELECTION_RECALL", MIN_SELECTION_RECALL)
MIN_SELECTION_POS_RATIO = get_env_float("EXP_MIN_SELECTION_POS_RATIO", MIN_SELECTION_POS_RATIO)
MIN_SELECTION_POS_RATIO_SCALE = get_env_float("EXP_MIN_SELECTION_POS_RATIO_SCALE", MIN_SELECTION_POS_RATIO_SCALE)
use_dynamic_threshold = get_env_bool("EXP_USE_DYNAMIC_THRESHOLD", use_dynamic_threshold)
configured_fixed_threshold = get_env_float("EXP_FIXED_THRESHOLD", configured_fixed_threshold)
best_thr = configured_fixed_threshold
use_over_smaple = get_env_bool("EXP_USE_OVERSAMPLE", use_over_smaple)
manual_pos_weight = get_env_float("EXP_MANUAL_POS_WEIGHT", manual_pos_weight)
target_ratio = get_env_float("EXP_TARGET_RATIO", target_ratio)
edge_exclude = get_env_int("EXP_EDGE_EXCLUDE", edge_exclude)
use_dice_loss = get_env_bool("EXP_USE_DICE_LOSS", use_dice_loss)
DICE_LOSS_WEIGHT = get_env_float("EXP_DICE_LOSS_WEIGHT", DICE_LOSS_WEIGHT)
use_density_regression = get_env_bool("EXP_USE_DENSITY_REGRESSION", use_density_regression)
use_domain_adversarial = get_env_bool("EXP_USE_DOMAIN_ADVERSARIAL", use_domain_adversarial)
use_all_other_wells = get_env_bool("EXP_USE_ALL_OTHER_WELLS", use_all_other_wells)
use_wellwise_log_scaling = get_env_bool("EXP_USE_WELLWISE_LOG_SCALING", use_wellwise_log_scaling)
use_well_balanced_sampling = get_env_bool("EXP_USE_WELL_BALANCED_SAMPLING", use_well_balanced_sampling)
only_val_wells = get_env_json_list("EXP_ONLY_VAL_WELLS", only_val_wells)
custom_train_wells_map = get_env_json_dict("EXP_CUSTOM_TRAIN_WELLS_MAP", custom_train_wells_map)
min_selection_accuracy_by_well = get_env_json_dict("EXP_MIN_SELECTION_ACCURACY_BY_WELL", min_selection_accuracy_by_well)
min_selection_recall_by_well = get_env_json_dict("EXP_MIN_SELECTION_RECALL_BY_WELL", min_selection_recall_by_well)
min_selection_pos_ratio_by_well = get_env_json_dict("EXP_MIN_SELECTION_POS_RATIO_BY_WELL", min_selection_pos_ratio_by_well)
min_selection_pos_ratio_scale_by_well = get_env_json_dict("EXP_MIN_SELECTION_POS_RATIO_SCALE_BY_WELL", min_selection_pos_ratio_scale_by_well)
use_selection_accuracy_floor_by_well = get_env_json_dict("EXP_USE_SELECTION_ACCURACY_FLOOR_BY_WELL", use_selection_accuracy_floor_by_well)
use_threshold_constraints_by_well = get_env_json_dict("EXP_USE_THRESHOLD_CONSTRAINTS_BY_WELL", use_threshold_constraints_by_well)
DIST_THRESHOLD = get_env_float("EXP_DIST_THRESHOLD", DIST_THRESHOLD)
MIN_TRAIN_WELLS = get_env_int("EXP_MIN_TRAIN_WELLS", MIN_TRAIN_WELLS)
MAX_POS_WEIGHT = get_env_float("EXP_MAX_POS_WEIGHT", MAX_POS_WEIGHT)
save_dir_override = os.getenv("EXP_SAVE_DIR")
data_dir_override = os.getenv("EXP_DATA_DIR")
well_files_override = get_env_json_list("EXP_WELL_FILES", [])
if save_dir_override:
    SAVE_DIR = save_dir_override
if data_dir_override:
    DATA_DIR = data_dir_override

os.makedirs(SAVE_DIR, exist_ok=True)

seis_features = [f"SEIS_{i}" for i in range(63)]
features = seis_features + imaging_well_features
N_LOG = len(imaging_well_features)

density_cols = ["P10", "P21", "P33"]

DIST_MATRIX_FILES = {
    "3x3x7": "seismic_3x3x7_plus_imaging_distance_matrix.csv",
    "3x3": "seismic_3x3_plane_plus_imaging_distance_matrix.csv",
    "1": "seismic_single_amplitude_plus_imaging_distance_matrix.csv",
}


def get_distance_matrix_path(seis_mode):
    matrix_name = DIST_MATRIX_FILES.get(seis_mode)
    if matrix_name is None:
        raise ValueError(f"Unsupported SEIS_MODE for distance matrix: {seis_mode}")

    matrix_path = os.path.join(DISTANCE_ANALYSIS_DIR, matrix_name)
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Distance matrix not found: {matrix_path}")

    return matrix_path


DIST_MATRIX_PATH = get_distance_matrix_path(SEIS_MODE)


DEFAULT_CSV_FILE_BASENAMES = [
    "车660-1_sample.csv",
    "车660-2_sample.csv",
    "车662_sample.csv",
    "车663_sample.csv",
]


def resolve_input_csv_files(data_dir, data_dir_override=None, well_files_override=None):
    if well_files_override:
        csv_files = []
        for item in well_files_override:
            if os.path.isabs(item):
                csv_files.append(item)
            else:
                csv_files.append(os.path.join(data_dir, item))
        return csv_files

    if data_dir_override:
        discovered = sorted(glob.glob(os.path.join(data_dir, "*_sample.csv")))
        if discovered:
            return discovered

    return [os.path.join(data_dir, name) for name in DEFAULT_CSV_FILE_BASENAMES]


def format_distance_map(distance_df, val_well, train_wells):
    if distance_df is None or distance_df.empty or val_well not in distance_df.index:
        return {w: "NA" for w in train_wells}

    distance_map = {}
    for well_name in train_wells:
        if well_name not in distance_df.columns:
            distance_map[well_name] = "NA"
            continue
        value = pd.to_numeric(pd.Series([distance_df.loc[val_well, well_name]]), errors="coerce").iloc[0]
        distance_map[well_name] = "NA" if pd.isna(value) else round(float(value), 4)
    return distance_map

# ===================== 1. 数据读取 =====================
csv_files = resolve_input_csv_files(
    DATA_DIR,
    data_dir_override=data_dir_override,
    well_files_override=well_files_override,
)

df_list = []
df_raw_list = []
for f in csv_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Sample csv not found: {f}")
    df = pd.read_csv(f)
    df["WellName"] = os.path.basename(f).split("_")[0]
    if "ROW_IN_WELL" not in df.columns:
        df["ROW_IN_WELL"] = np.arange(len(df))
    df_list.append(df)
    df_raw_list.append(df.copy())

df_raw = pd.concat(df_raw_list, ignore_index=True)
df_raw["RAW_ROW_IDX"] = np.arange(len(df_raw))
df = df_raw.copy()

df, invalid_log_summary = clean_invalid_log_values(df, imaging_well_features)
if not invalid_log_summary.empty:
    print("检测到测井缺失占位值，已按缺失处理并准备删除对应样本：")
    print(
        invalid_log_summary.sort_values(["WellName", "Feature"]).to_string(index=False)
    )

df["FRACTURE_FLAG"] = df["Frac_Azimuth"].notna().astype(int)

# 只在有裂缝位置保留密度
for col in density_cols:
    if col not in df.columns:
        df[col] = 0.0

required_columns = features + density_cols + ["FRACTURE_FLAG", "WellName", "ROW_IN_WELL", "RAW_ROW_IDX"]
before_drop_counts = df.groupby("WellName").size().to_dict()
df = df[required_columns].dropna().copy()
after_drop_counts = df.groupby("WellName").size().to_dict()
removed_counts = {
    well_name: before_drop_counts.get(well_name, 0) - after_drop_counts.get(well_name, 0)
    for well_name in before_drop_counts
}
if any(count > 0 for count in removed_counts.values()):
    print("按当前建模特征删除缺失样本后，各井剩余样本：")
    for well_name in before_drop_counts:
        print(
            f"{well_name}: {after_drop_counts.get(well_name, 0)}/"
            f"{before_drop_counts.get(well_name, 0)} "
            f"(removed={removed_counts[well_name]})"
        )
# log 变换密度（推荐）
df[density_cols] = df[density_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
df[density_cols] = np.log1p(df[density_cols])

# 深度归一化
# df["DEPTH_NORM"] = df.groupby("WellName")["DEPTH"].transform(
#     lambda x: (x-x.min())/(x.max()-x.min())
# )

# df_raw["DEPTH_IDX"] = df_raw.groupby("WellName").cumcount()
# df["DEPTH_IDX"] = df.groupby("WellName").cumcount()

well_names = df["WellName"].unique()
print("参与建模井：", well_names)


# ===================== 井级标准化 =====================
def wellwise_standardize(df, features):
    df_scaled = []
    scalers = {}
    for well in df["WellName"].unique():
        well_df = df[df["WellName"] == well].copy()
        scaler = StandardScaler()
        well_df[features] = scaler.fit_transform(well_df[features])
        scalers[well] = scaler
        df_scaled.append(well_df)
    df_scaled = pd.concat(df_scaled)

    return df_scaled, scalers


def apply_mixed_feature_scaling(
    train_df,
    val_df,
    seis_features,
    log_features,
    use_wellwise_log_scaling=False,
):
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()

    if not use_wellwise_log_scaling:
        global_scaler = StandardScaler()
        full_features = list(seis_features) + list(log_features)
        train_df_scaled[full_features] = global_scaler.fit_transform(train_df[full_features])
        val_df_scaled[full_features] = global_scaler.transform(val_df[full_features])
        scaler_artifact = {
            "scaling_mode": "global_all_features",
            "feature_scaler": global_scaler,
            "seis_features": list(seis_features),
            "log_features": list(log_features),
        }
        return train_df_scaled, val_df_scaled, scaler_artifact

    seis_scaler = StandardScaler()
    train_df_scaled[seis_features] = seis_scaler.fit_transform(train_df[seis_features])
    val_df_scaled[seis_features] = seis_scaler.transform(val_df[seis_features])

    train_log_scaler_summary = {}
    for well_name in train_df["WellName"].unique():
        mask = train_df["WellName"] == well_name
        well_scaler = StandardScaler()
        train_df_scaled.loc[mask, log_features] = well_scaler.fit_transform(train_df.loc[mask, log_features])
        train_log_scaler_summary[str(well_name)] = {
            "mean": well_scaler.mean_.tolist(),
            "scale": well_scaler.scale_.tolist(),
        }

    val_log_scaler = StandardScaler()
    val_df_scaled[log_features] = val_log_scaler.fit_transform(val_df[log_features])

    scaler_artifact = {
        "scaling_mode": "global_seis_wellwise_log",
        "seis_scaler": seis_scaler,
        "seis_features": list(seis_features),
        "log_features": list(log_features),
        "train_log_scaler_summary": train_log_scaler_summary,
        "val_log_scaler_summary": {
            "well_name": str(val_df["WellName"].iloc[0]) if len(val_df) else "",
            "mean": val_log_scaler.mean_.tolist(),
            "scale": val_log_scaler.scale_.tolist(),
        },
    }
    return train_df_scaled, val_df_scaled, scaler_artifact


def build_well_balanced_sampler(domain_seq):
    domain_seq = np.asarray(domain_seq)
    if len(domain_seq) == 0:
        return None, {}

    unique_domains, counts = np.unique(domain_seq, return_counts=True)
    domain_count_map = {int(domain): int(count) for domain, count in zip(unique_domains, counts)}
    sample_weights = np.array(
        [1.0 / max(domain_count_map[int(domain_id)], 1) for domain_id in domain_seq],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, domain_count_map


# ===================== 自动井聚类 =====================
def cluster_wells(distance_matrix, well_names, n_clusters=2):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )

    labels = model.fit_predict(distance_matrix)

    cluster_map = dict(zip(well_names, labels))

    print("\nWell Clusters:")
    for w, c in cluster_map.items():
        print(w, "-> cluster", c)

    return cluster_map


# ===================== 基于距离矩阵自动选择训练井 =====================
def select_train_wells(val_well, distance_df, threshold=0.55, min_wells=2, candidate_wells=None):
    candidate_wells = list(candidate_wells or [])
    candidate_wells = [w for w in candidate_wells if w != val_well]
    if not candidate_wells:
        print(f"\n验证井: {val_well}")
        print("训练井选择模式: no_candidate_wells")
        print("训练井: []")
        return []

    if distance_df is None or distance_df.empty or val_well not in distance_df.index:
        train_wells = candidate_wells[:min(min_wells, len(candidate_wells))]
        print(f"\n验证井: {val_well}")
        print("训练井选择模式: fallback_no_distance_row")
        print("训练井:", train_wells)
        print("对应距离:", {w: "NA" for w in train_wells})
        return train_wells

    available_wells = [
        w for w in candidate_wells
        if w in distance_df.columns and pd.notna(pd.to_numeric(distance_df.loc[val_well, w], errors="coerce"))
    ]
    if not available_wells:
        train_wells = candidate_wells[:min(min_wells, len(candidate_wells))]
        print(f"\n验证井: {val_well}")
        print("训练井选择模式: fallback_no_candidate_distance")
        print("训练井:", train_wells)
        print("对应距离:", {w: "NA" for w in train_wells})
        return train_wells

    dists = pd.to_numeric(distance_df.loc[val_well, available_wells], errors="coerce").dropna()

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

    if len(train_wells) == 0:
        raise ValueError(f"Custom train wells for {val_well} cannot be empty")

    return train_wells


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


# ===================== 2. LSTM 定义 =====================
class FractureCNNLSTM(nn.Module):
    def __init__(self, seis_mode, log_dim, n_domains, hidden_dim=64,
                 use_density_regression=True, use_domain_adversarial=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_density_regression = use_density_regression
        self.use_domain_adversarial = use_domain_adversarial
        if self.use_domain_adversarial:
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_domains)
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
            self.cnn_dropout = nn.Dropout(Dropout)
            cnn_out = 16

        elif seis_mode == "3x3":
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.cnn_dropout = nn.Dropout(Dropout)
            cnn_out = 8

        else:  # 1×1
            self.cnn = None
            cnn_out = 1

        self.lstm = nn.LSTM(
            input_size=cnn_out + log_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.dropout = nn.Dropout(Dropout)

        self.fc_cls = nn.Linear(hidden_dim, 1)
        self.fc_den = nn.Linear(hidden_dim, 3) if self.use_density_regression else None

    def forward(self, seis, log, alpha=1.0):

        B, T, C, H, W = seis.shape

        if self.cnn is not None:
            seis = seis.view(B * T, C, H, W)
            feat = self.cnn(seis)
            feat = feat.view(B, T, -1)
            feat = self.cnn_dropout(feat)

        else:
            feat = seis.view(B, T, 1)

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


# ===================== 数据增强 =====================
def augment_seismic(seis):
    noise = np.random.normal(0, 0.03, seis.shape)
    amp = np.random.uniform(0.9, 1.1)
    seis_aug = seis * amp + noise
    return seis_aug


# ===================== 3. 按井构造序列 =====================
def reshape_seis(X_seis, mode):
    if mode == "3x3x7":
        X_seis = X_seis.reshape(-1, 3, 3, 7)
        X_seis = np.transpose(X_seis, (0, 3, 1, 2))
    elif mode == "3x3":
        X_seis = X_seis.reshape(-1, 3, 3, 7)
        X_seis = np.transpose(X_seis, (0, 3, 1, 2))
        center = 3
        X_seis = X_seis[:, center]  # (N,3,3)
        X_seis = X_seis[:, None, :, :]  # (N,1,3,3)
    elif mode == "1":
        center_index = 7 * 4 + 3
        X_seis = X_seis[:, center_index]
        X_seis = X_seis.reshape(-1, 1, 1, 1)
    return X_seis


def build_sequences_by_well(df, X_seis, X_log, y_cls, y_den, seq_len):
    X_seis_seq, X_log_seq = [], []
    y_cls_seq, y_den_seq = [], []
    domain_seq = []
    center_row_idx_seq = []
    half = seq_len // 2

    start = 0
    skipped_windows = 0

    for well in df["WellName"].unique():

        well_df = df[df["WellName"] == well]
        n = len(well_df)
        row_in_well = well_df["ROW_IN_WELL"].values
        raw_row_idx = well_df["RAW_ROW_IDX"].values

        seis_well = X_seis[start:start + n]
        log_well = X_log[start:start + n]

        y_cls_well = y_cls[start:start + n]
        y_den_well = y_den[start:start + n]

        domain_id = domain_map[well]

        for i in range(half, n - half):
            window_row_idx = row_in_well[i - half:i + half + 1]
            if not np.all(np.diff(window_row_idx) == 1):
                skipped_windows += 1
                continue

            X_seis_seq.append(seis_well[i - half:i + half + 1])
            X_log_seq.append(log_well[i - half:i + half + 1])

            y_cls_seq.append(y_cls_well[i])
            y_den_seq.append(y_den_well[i])

            domain_seq.append(domain_id)
            center_row_idx_seq.append(raw_row_idx[i])

        start += n

    if skipped_windows > 0:
        print(f"因删除缺失值导致窗口不连续，跳过序列数: {skipped_windows}")

    return (
        np.array(X_seis_seq),
        np.array(X_log_seq),
        np.array(y_cls_seq),
        np.array(y_den_seq),
        np.array(domain_seq),
        np.array(center_row_idx_seq)
    )


# ================= 中心段过采样 =================
def oversample_center_segments(seis, log, y_cls, y_den, train_domain,
                               target_ratio=0.35,
                               edge_exclude=1,
                               random_state=42):
    """
    只对连续裂缝段中心序列过采样
    同时扩展 seis / log / y_cls / y_den
    """

    rng = np.random.RandomState(random_state)

    seis = np.array(seis)
    log = np.array(log)
    y_cls = np.array(y_cls)
    y_den = np.array(y_den)
    train_domain = np.array(train_domain)

    # ---------- 找连续正样本段 ----------
    segments = []
    start = None

    for i in range(len(y_cls)):
        if y_cls[i] == 1 and start is None:
            start = i
        elif y_cls[i] == 0 and start is not None:
            segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(y_cls)))

    # ---------- 提取中心位置 ----------
    center_indices = []

    for s, e in segments:
        length = e - s

        if length <= 2 * edge_exclude:
            continue

        center_indices.extend(range(s + edge_exclude, e - edge_exclude))

    if len(center_indices) == 0:
        return seis, log, y_cls, y_den, train_domain

    center_indices = np.array(center_indices)

    # ---------- 当前比例 ----------
    n_pos = (y_cls == 1).sum()
    n_total = len(y_cls)
    current_ratio = n_pos / n_total

    if current_ratio >= target_ratio:
        return seis, log, y_cls, y_den, train_domain

    # ---------- 需要增加多少 ----------
    desired_pos = int(target_ratio * n_total / (1 - target_ratio))
    n_to_add = desired_pos - n_pos

    if n_to_add <= 0:
        return seis, log, y_cls, y_den, train_domain

    # ---------- 只从中心点采样 ----------
    add_idx = rng.choice(center_indices, size=n_to_add, replace=True)

    seis_new = np.concatenate([seis, seis[add_idx]], axis=0)
    log_new = np.concatenate([log, log[add_idx]], axis=0)

    y_cls_new = np.concatenate([y_cls, y_cls[add_idx]], axis=0)
    y_den_new = np.concatenate([y_den, y_den[add_idx]], axis=0)
    train_domain_new = np.concatenate([train_domain, train_domain[add_idx]], axis=0)

    return seis_new, log_new, y_cls_new, y_den_new, train_domain_new


def cnn_lstm_permutation_importance(model, X_seis, X_log, y, loss_fn, device="cpu"):
    model.eval()

    X_seis = X_seis.clone().to(device)
    X_log = X_log.clone().to(device)
    y = y.clone().to(device)

    with torch.no_grad():
        cls_out, _, _, _ = model(X_seis, X_log)
        baseline_loss = loss_fn(cls_out, y).item()

    n_features = X_log.shape[2]

    importances = []

    for f in range(n_features):

        scores = []

        for _ in range(5):
            X_log_perm = X_log.clone()

            idx = torch.randperm(X_log.shape[0], device=device)

            X_log_perm[:, :, f] = X_log[idx, :, f]

            with torch.no_grad():
                cls_out, _, _, _ = model(X_seis, X_log_perm)

            loss = loss_fn(cls_out, y).item()

            scores.append(loss)

        importance = np.mean(scores) - baseline_loss
        importances.append(importance)

    return np.array(importances)


# ---------- 段级IoU计算 ----------
def extract_segments(labels):
    segments = []
    start = None
    for i, v in enumerate(labels):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def compute_iou(gt, pred):
    gt_seg = extract_segments(gt)
    pred_seg = extract_segments(pred)

    if not gt_seg and not pred_seg:
        return 1.0
    if not gt_seg:
        return 0.0

    iou_list = []

    for g in gt_seg:
        best_iou = 0
        for p in pred_seg:
            inter = max(0, min(g[1], p[1]) - max(g[0], p[0]) + 1)
            union = (g[1] - g[0] + 1) + (p[1] - p[0] + 1) - inter
            if union > 0:
                best_iou = max(best_iou, inter / union)
        iou_list.append(best_iou)

    return np.mean(iou_list)


def soft_dice_loss_from_logits(logits, targets, smooth=1.0):
    prob = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (prob * targets).sum()
    denom = prob.sum() + targets.sum()

    dice_score = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice_score


def compute_classification_loss(logits, targets, bce_loss_fn):
    loss_bce = bce_loss_fn(logits, targets)

    if not use_dice_loss:
        return loss_bce, loss_bce, torch.tensor(0.0, device=logits.device)

    loss_dice = soft_dice_loss_from_logits(logits, targets)
    loss_total = loss_bce + DICE_LOSS_WEIGHT * loss_dice
    return loss_total, loss_bce, loss_dice


# ================= 概率平滑 =================
def smooth_prob(prob, window=3):
    smoothed = np.convolve(prob,
                           np.ones(window) / window,
                           mode='same')
    return smoothed


# ================= 小段过滤 =================
def remove_short_segments(labels, min_len=3):
    labels = labels.copy()
    start = None

    for i in range(len(labels)):
        if labels[i] == 1 and start is None:
            start = i
        elif labels[i] == 0 and start is not None:
            if i - start < min_len:
                labels[start:i] = 0
            start = None

    if start is not None:
        if len(labels) - start < min_len:
            labels[start:] = 0

    return labels


# ================= gap 填补 =================
def fill_small_gaps(labels, max_gap=2):
    labels = labels.copy()
    i = 0
    while i < len(labels):
        if labels[i] == 1:
            j = i
            while j < len(labels) and labels[j] == 1:
                j += 1
            k = j
            while k < len(labels) and labels[k] == 0:
                k += 1
            if k < len(labels) and (k - j) <= max_gap:
                labels[j:k] = 1
            i = k
        else:
            i += 1
    return labels


def postprocess_predictions(pred, metric):
    pred = pred.copy()

    if metric == "iou" or use_short_segment_processing:
        pred = remove_short_segments(pred, min_len=3)
        pred = fill_small_gaps(pred, max_gap=2)

    return pred


def evaluate_selection_metric_from_pred(y_true, pred, metric, val_df=None):
    if metric == "accuracy":
        return accuracy_score(y_true, pred)

    if metric == "f1":
        return f1_score(y_true, pred, zero_division=0)

    if metric == "iou":
        if val_df is None:
            raise ValueError("val_df is required when metric='iou'")

        iou_list = []
        start = 0
        for well in val_df["WellName"].unique():
            well_df = val_df[val_df["WellName"] == well]
            n = len(well_df)
            n_seq = max(0, n - 2 * HALF)

            if n_seq <= 0:
                continue

            gt_well = y_true[start:start + n_seq]
            pred_well = pred[start:start + n_seq]

            iou_list.append(compute_iou(gt_well, pred_well))
            start += n_seq

        return np.mean(iou_list) if iou_list else 0.0

    raise ValueError(f"Unsupported selection metric: {metric}")


def build_eval_pred(prob, threshold, metric):
    pred = (prob >= threshold).astype(int)
    return postprocess_predictions(pred, metric)


def threshold_candidate_is_valid(y_true, pred):
    stats = get_threshold_candidate_stats(y_true, pred)

    return threshold_candidate_meets_accuracy_floor(stats) and threshold_candidate_meets_soft_constraints(stats)


def parse_well_override_float(override_map, well_name, default_value):
    if not isinstance(override_map, dict):
        return float(default_value)
    raw = override_map.get(well_name)
    if raw is None:
        return float(default_value)
    try:
        return float(raw)
    except Exception:
        return float(default_value)


def parse_well_override_bool(override_map, well_name, default_value):
    if not isinstance(override_map, dict):
        return bool(default_value)
    raw = override_map.get(well_name)
    if raw is None:
        return bool(default_value)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_effective_threshold_policy(val_well=None):
    return {
        "use_selection_accuracy_floor": parse_well_override_bool(
            use_selection_accuracy_floor_by_well,
            val_well,
            use_selection_accuracy_floor,
        ),
        "min_selection_accuracy": parse_well_override_float(
            min_selection_accuracy_by_well,
            val_well,
            MIN_SELECTION_ACCURACY,
        ),
        "use_threshold_constraints": parse_well_override_bool(
            use_threshold_constraints_by_well,
            val_well,
            use_threshold_constraints,
        ),
        "min_selection_recall": parse_well_override_float(
            min_selection_recall_by_well,
            val_well,
            MIN_SELECTION_RECALL,
        ),
        "min_selection_pos_ratio": parse_well_override_float(
            min_selection_pos_ratio_by_well,
            val_well,
            MIN_SELECTION_POS_RATIO,
        ),
        "min_selection_pos_ratio_scale": parse_well_override_float(
            min_selection_pos_ratio_scale_by_well,
            val_well,
            MIN_SELECTION_POS_RATIO_SCALE,
        ),
    }


def get_threshold_candidate_stats(y_true, pred, threshold_policy=None):
    if threshold_policy is None:
        threshold_policy = build_effective_threshold_policy()
    true_pos_ratio = float(np.mean(y_true))
    pred_pos_ratio = float(np.mean(pred))
    recall = recall_score(y_true, pred, zero_division=0)
    accuracy = accuracy_score(y_true, pred)

    min_pos_ratio_required = max(
        threshold_policy["min_selection_pos_ratio"],
        true_pos_ratio * threshold_policy["min_selection_pos_ratio_scale"],
    )

    return {
        "accuracy": accuracy,
        "recall": recall,
        "pred_pos_ratio": pred_pos_ratio,
        "true_pos_ratio": true_pos_ratio,
        "min_pos_ratio_required": min_pos_ratio_required,
        "threshold_policy": threshold_policy,
    }


def threshold_candidate_meets_accuracy_floor(stats):
    threshold_policy = stats.get("threshold_policy", build_effective_threshold_policy())
    return (not threshold_policy["use_selection_accuracy_floor"]) or (
        stats["accuracy"] >= threshold_policy["min_selection_accuracy"]
    )


def threshold_candidate_meets_pos_ratio(stats):
    return stats["pred_pos_ratio"] >= stats["min_pos_ratio_required"]


def threshold_candidate_meets_soft_constraints(stats):
    threshold_policy = stats.get("threshold_policy", build_effective_threshold_policy())
    return (
        stats["recall"] >= threshold_policy["min_selection_recall"] and
        threshold_candidate_meets_pos_ratio(stats)
    )


def summarize_binary_prediction(y_true, pred):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pred_pos_ratio": float(np.mean(pred)),
        "true_pos_ratio": float(np.mean(y_true)),
    }


# ================= 动态阈值搜索 =================
def search_best_threshold(prob, y_true, val_df):
    best_thr = 0.5
    best_iou = -1

    for thr in np.arange(0.2, 0.71, 0.02):

        temp_pred = (prob >= thr).astype(int)
        temp_pred = remove_short_segments(temp_pred, min_len=3)
        temp_pred = fill_small_gaps(temp_pred, max_gap=2)

        # 井级 IoU
        iou_list = []
        start = 0
        for well in val_df["WellName"].unique():
            well_df = val_df[val_df["WellName"] == well]
            n = len(well_df)
            n_seq = max(0, n - 2 * HALF)

            if n_seq <= 0:
                continue

            gt_well = y_true[start:start + n_seq]
            pred_well = temp_pred[start:start + n_seq]

            iou_list.append(compute_iou(gt_well, pred_well))
            start += n_seq

        mean_iou = np.mean(iou_list) if iou_list else 0

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_thr = thr

    return best_thr, best_iou


def evaluate_selection_metric(prob, y_true, metric, val_df=None, threshold=0.5):
    pred = build_eval_pred(prob, threshold, metric)
    return evaluate_selection_metric_from_pred(y_true, pred, metric, val_df=val_df)


def search_best_threshold_by_metric(prob, y_true, metric, val_df=None, val_well=None):
    best_thr = None
    best_rank = None
    fallback_acc_pos_thr = None
    fallback_acc_pos_rank = None
    fallback_acc_thr = None
    fallback_acc_rank = None
    fallback_soft_thr = None
    fallback_soft_rank = None
    fallback_pos_thr = None
    fallback_pos_rank = None
    fallback_any_thr = 0.5
    fallback_any_rank = None
    threshold_policy = build_effective_threshold_policy(val_well)

    for thr in np.arange(
        THRESH_SEARCH_MIN,
        THRESH_SEARCH_MAX + THRESH_SEARCH_STEP * 0.5,
        THRESH_SEARCH_STEP,
    ):
        pred = build_eval_pred(prob, thr, metric)
        score = evaluate_selection_metric_from_pred(y_true, pred, metric, val_df=val_df)
        stats = get_threshold_candidate_stats(y_true, pred, threshold_policy=threshold_policy)

        meets_acc = threshold_candidate_meets_accuracy_floor(stats)
        meets_pos = threshold_candidate_meets_pos_ratio(stats)
        meets_soft = threshold_candidate_meets_soft_constraints(stats)

        primary_rank = (score, stats["accuracy"], stats["recall"])
        acc_pos_rank = (score, stats["recall"], stats["accuracy"])
        acc_rank = (score, stats["recall"], stats["pred_pos_ratio"])
        soft_rank = (score, stats["accuracy"], stats["pred_pos_ratio"])
        pos_rank = (score, stats["accuracy"], stats["recall"])
        any_rank = (stats["recall"], score, stats["accuracy"])

        if fallback_any_rank is None or any_rank > fallback_any_rank:
            fallback_any_rank = any_rank
            fallback_any_thr = thr

        if meets_pos and (fallback_pos_rank is None or pos_rank > fallback_pos_rank):
            fallback_pos_rank = pos_rank
            fallback_pos_thr = thr

        if meets_soft and (fallback_soft_rank is None or soft_rank > fallback_soft_rank):
            fallback_soft_rank = soft_rank
            fallback_soft_thr = thr

        if meets_acc and (fallback_acc_rank is None or acc_rank > fallback_acc_rank):
            fallback_acc_rank = acc_rank
            fallback_acc_thr = thr

        if meets_acc and meets_pos and (fallback_acc_pos_rank is None or acc_pos_rank > fallback_acc_pos_rank):
            fallback_acc_pos_rank = acc_pos_rank
            fallback_acc_pos_thr = thr

        if threshold_policy["use_threshold_constraints"] and not (meets_acc and meets_soft):
            continue

        if best_rank is None or primary_rank > best_rank:
            best_rank = primary_rank
            best_thr = thr

    if best_thr is not None:
        return best_thr, best_rank[0]

    if fallback_acc_pos_thr is not None:
        return fallback_acc_pos_thr, fallback_acc_pos_rank[0]

    if fallback_acc_thr is not None:
        return fallback_acc_thr, fallback_acc_rank[0]

    if fallback_soft_thr is not None:
        return fallback_soft_thr, fallback_soft_rank[0]

    if fallback_pos_thr is not None:
        return fallback_pos_thr, fallback_pos_rank[0]

    return fallback_any_thr, fallback_any_rank[1]


# ===================== 自动跨井CV策略 =====================
def generate_cv_splits(cluster_map):
    splits = []

    for cluster in set(cluster_map.values()):
        val_wells = [w for w, c in cluster_map.items() if c == cluster]

        train_wells = [w for w, c in cluster_map.items() if c != cluster]

        splits.append((train_wells, val_wells))

    return splits


def cross_well_loss(features, well_ids, well_sim_matrix, well_names):
    loss = 0
    count = 0

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            wi = well_names[well_ids[i]]
            wj = well_names[well_ids[j]]

            sim = well_sim_matrix.loc[wi, wj]

            diff = features[i] - features[j]

            loss += sim * torch.mean(diff ** 2)

            count += 1

    if count > 0:
        loss = loss / count

    return loss


# ===================== 4. Leave-One-Well-Out 循环 =====================
dist_df = pd.read_csv(DIST_MATRIX_PATH, index_col=0)
dist_df.index = [str(idx).strip() for idx in dist_df.index]
dist_df.columns = [str(col).strip() for col in dist_df.columns]
dist_df = dist_df.apply(pd.to_numeric, errors="coerce")

well_names = df["WellName"].drop_duplicates().tolist()
available_distance_wells = [w for w in well_names if w in dist_df.index and w in dist_df.columns]
well_sim = pd.DataFrame(
    np.eye(len(well_names), dtype=np.float32),
    index=well_names,
    columns=well_names,
)
if available_distance_wells:
    dist_subset = dist_df.loc[available_distance_wells, available_distance_wells]
    sim_subset = (1.0 / (dist_subset + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_sim = float(sim_subset.max().max()) if not sim_subset.empty else 0.0
    if max_sim > 0:
        sim_subset = sim_subset / max_sim
    well_sim.loc[available_distance_wells, available_distance_wells] = sim_subset.to_numpy(dtype=np.float32)
domain_map = {well: idx for idx, well in enumerate(well_names)}

all_results = []
loop_wells = [w for w in well_names if (not only_val_wells) or (w in only_val_wells)]
if not loop_wells:
    raise ValueError(f"No validation wells selected. only_val_wells={only_val_wells}")

for val_well in loop_wells:
    print(f"\n================ 验证井：{val_well} ================")
    well_save_dir = os.path.join(SAVE_DIR, f"verify_{val_well}")
    os.makedirs(well_save_dir, exist_ok=True)
    if val_well in custom_train_wells_map:
        train_wells = validate_custom_train_wells(
            val_well,
            custom_train_wells_map[val_well],
            well_names,
        )
        print(f"\n验证井: {val_well}")
        print("训练井选择模式: custom_train_wells")
        print("训练井:", train_wells)
        print("对应距离:", format_distance_map(dist_df, val_well, train_wells))
    elif use_all_other_wells:
        train_wells = [w for w in well_names if w != val_well]
        print(f"\n验证井: {val_well}")
        print("训练井选择模式: all_other_wells")
        print("训练井:", train_wells)
        print("对应距离:", format_distance_map(dist_df, val_well, train_wells))
    else:
        train_wells = select_train_wells(
            val_well,
            dist_df,
            candidate_wells=[w for w in well_names if w != val_well],
            threshold=DIST_THRESHOLD,
            min_wells=MIN_TRAIN_WELLS
        )
    print("测试井:", [val_well])
    print("训练井:", train_wells)

    if not train_wells:
        print(f"验证井 {val_well} 无可用训练井，跳过。")
        continue

    train_df = df[df["WellName"].isin(train_wells)]
    val_df = df[df["WellName"] == val_well]

    # ---------- 标准化 ----------
    train_df_scaled, val_df_scaled, scaler_artifact = apply_mixed_feature_scaling(
        train_df=train_df,
        val_df=val_df,
        seis_features=seis_features,
        log_features=imaging_well_features,
        use_wellwise_log_scaling=use_wellwise_log_scaling,
    )
    X_train_all = train_df_scaled[features].values
    X_val_all = val_df_scaled[features].values
    X_train_seis = reshape_seis(X_train_all[:, :63], SEIS_MODE)
    X_val_seis = reshape_seis(X_val_all[:, :63], SEIS_MODE)
    X_train_log = X_train_all[:, 63:]
    X_val_log = X_val_all[:, 63:]

    y_train = train_df["FRACTURE_FLAG"].values
    y_val = val_df["FRACTURE_FLAG"].values

    y_train_den = train_df[density_cols].values
    y_val_den = val_df[density_cols].values

    # ---------- 构造序列 ----------
    X_train_seis_seq, X_train_log_seq, y_train_seq, y_train_den_seq, train_domain_seq, _ = build_sequences_by_well(
        train_df,
        X_train_seis,
        X_train_log,
        y_train,
        y_train_den,
        SEQ_LEN
    )
    # 数据增强
    aug_seis = augment_seismic(X_train_seis_seq)
    X_train_seis_seq = np.concatenate([X_train_seis_seq, aug_seis])
    noise = np.random.normal(0, 0.01, X_train_log_seq.shape)
    X_train_log_seq = np.concatenate([X_train_log_seq, X_train_log_seq + noise])
    y_train_seq = np.concatenate([y_train_seq, y_train_seq])
    y_train_den_seq = np.concatenate([y_train_den_seq, y_train_den_seq])
    train_domain_seq = np.concatenate([train_domain_seq, train_domain_seq])

    if use_over_smaple:
        print("Before oversample:",
              (y_train_seq == 1).sum() / len(y_train_seq))
        X_train_seis_seq, X_train_log_seq, y_train_seq, y_train_den_seq, train_domain_seq = oversample_center_segments(
            X_train_seis_seq,
            X_train_log_seq,
            y_train_seq,
            y_train_den_seq,
            train_domain_seq,
            target_ratio=target_ratio,  # 建议 0.30~0.40
            edge_exclude=edge_exclude  # 去掉每段两端1个
        )
        print("After oversample:",
              (y_train_seq == 1).sum() / len(y_train_seq))

    X_val_seis_seq, X_val_log_seq, y_val_seq, y_val_den_seq, val_domain_seq, val_center_row_idx = build_sequences_by_well(
        val_df,
        X_val_seis,
        X_val_log,
        y_val,
        y_val_den,
        SEQ_LEN
    )

    # ---------- Tensor ----------
    X_train_seis_t = torch.tensor(X_train_seis_seq, dtype=torch.float32).to(DEVICE)
    X_train_log_t = torch.tensor(X_train_log_seq, dtype=torch.float32).to(DEVICE)
    X_val_seis_t = torch.tensor(X_val_seis_seq, dtype=torch.float32).to(DEVICE)
    X_val_log_t = torch.tensor(X_val_log_seq, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).to(DEVICE)
    y_train_den_t = torch.tensor(y_train_den_seq, dtype=torch.float32).to(DEVICE)
    y_val_den_t = torch.tensor(y_val_den_seq, dtype=torch.float32).to(DEVICE)
    train_domain_t = torch.tensor(train_domain_seq).long().to(DEVICE)

    print("Shapes:")
    print("seis:", X_train_seis_seq.shape)
    print("log:", X_train_log_seq.shape)
    print("y_cls:", y_train_seq.shape)
    print("y_den:", y_train_den_seq.shape)
    print("domain:", train_domain_seq.shape)

    train_sampler = None
    domain_count_map = {}
    if use_well_balanced_sampling:
        train_sampler, domain_count_map = build_well_balanced_sampler(train_domain_seq)
        readable_domain_count_map = {
            well_names[domain_id]: count
            for domain_id, count in sorted(domain_count_map.items(), key=lambda item: item[0])
            if domain_id < len(well_names)
        }
        print("启用井均衡采样，序列数分布:", readable_domain_count_map)

    train_loader = DataLoader(
        TensorDataset(
            X_train_seis_t,
            X_train_log_t,
            y_train_t,
            y_train_den_t,
            train_domain_t
        ),
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )

    # ---------- 模型 ----------
    model = FractureCNNLSTM(
        seis_mode=SEIS_MODE,
        log_dim=len(imaging_well_features),
        n_domains=len(domain_map),
        hidden_dim=32,
        use_density_regression=use_density_regression,
        use_domain_adversarial=use_domain_adversarial
    ).to(DEVICE)

    if use_over_smaple:
        pos_weight = manual_pos_weight
    else:
        n_pos = (y_train_seq == 1).sum()
        n_neg = (y_train_seq == 0).sum()
        pos_weight = n_neg / max(n_pos, MAX_N_POS)
        pos_weight = min(pos_weight, MAX_POS_WEIGHT)
    print(f"pos_weight = {pos_weight:.3f}")
    print(f"train pos ratio = {(y_train_seq == 1).mean():.3f}")

    criterion_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight).to(DEVICE)
    )
    criterion_domain = nn.CrossEntropyLoss() if use_domain_adversarial else None
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Early Stopping
    best_val_loss = float("inf")
    best_val_score = -1.0
    early_stop_counter = 0
    best_model_state = None
    best_epoch = -1
    best_epoch_thr = best_thr

    # ---------- 训练 ----------
    for epoch in range(EPOCHS):
        model.train()
        p = epoch / EPOCHS
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{val_well}] Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for xb_seis, xb_log, yb_cls, yb_den, yb_domain in pbar:
            optimizer.zero_grad()
            cls_out, den_out, domain_out, emb = model(xb_seis, xb_log, alpha)
            loss_cls, loss_bce, loss_dice = compute_classification_loss(cls_out, yb_cls, criterion_bce)
            if use_density_regression:
                mask = (yb_cls == 1).float().unsqueeze(1)
                if mask.sum().item() > 0:
                    loss_den = ((den_out - yb_den) ** 2 * mask).sum() / mask.sum()
                else:
                    loss_den = torch.tensor(0.0).to(DEVICE)
            else:
                loss_den = torch.tensor(0.0).to(DEVICE)
            if use_domain_adversarial:
                loss_domain = criterion_domain(domain_out, yb_domain)
            else:
                loss_domain = torch.tensor(0.0).to(DEVICE)
            # loss_cross = cross_well_loss(emb, yb_domain.cpu().numpy(), well_sim, well_names)
            # loss = loss_cls + 0.5 * loss_den + 0.3 * loss_domain + 0.2 * loss_cross
            loss = loss_cls + 0.5 * loss_den + 0.3 * loss_domain

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=loss.detach().item(),
                cls=loss_cls.detach().item(),
                dice=loss_dice.detach().item(),
            )

        avg_train_loss = epoch_loss / len(train_loader)

        # ===== validation loss =====
        model.eval()
        with torch.no_grad():
            cls_out_val, den_out_val, domain_out_val, emb_val = model(X_val_seis_t, X_val_log_t)
            val_loss_cls, val_loss_bce, val_loss_dice = compute_classification_loss(cls_out_val, y_val_t, criterion_bce)
            val_prob = torch.sigmoid(cls_out_val).cpu().numpy()
            if use_dynamic_threshold:
                epoch_thr, val_score = search_best_threshold_by_metric(
                    val_prob,
                    y_val_seq,
                    selection_metric,
                    val_df=val_df,
                    val_well=val_well,
                )
            else:
                epoch_thr = best_thr
                val_score = evaluate_selection_metric(
                    val_prob,
                    y_val_seq,
                    selection_metric,
                    val_df=val_df,
                    threshold=epoch_thr
                )
            if use_density_regression:
                mask = (y_val_t == 1).float().unsqueeze(1)
                if mask.sum().item() > 0:
                    val_loss_den = ((den_out_val - y_val_den_t) ** 2 * mask).sum() / mask.sum()
                else:
                    val_loss_den = torch.tensor(0.0).to(DEVICE)
            else:
                val_loss_den = torch.tensor(0.0).to(DEVICE)

            # val_loss = val_loss_cls + 0.5 * val_loss_den
            val_loss = val_loss_cls
            val_loss = val_loss.item()

        print(
            f"Epoch {epoch + 1} | TrainLoss={avg_train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | Val{selection_metric.upper()}={val_score:.4f} | "
            f"ValBCE={val_loss_bce.detach().item():.4f} | ValDice={val_loss_dice.detach().item():.4f} | Thr={epoch_thr:.2f}"
        )

        # ===== Early Stopping =====
        if (val_score > best_val_score + min_delta) or (
                abs(val_score - best_val_score) <= min_delta and val_loss < best_val_loss - min_delta
        ):
            best_val_score = val_score
            best_val_loss = val_loss
            best_epoch_thr = epoch_thr
            early_stop_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            early_stop_counter += 1
            print(
                f"EarlyStopping counter: {early_stop_counter}/{patience}, "
                f"best_val_{selection_metric}: {best_val_score:.4f}, best_val_loss: {best_val_loss:.4f}"
            )

            if early_stop_counter >= patience:
                print(
                    f"Early stopping triggered, best_val_{selection_metric}: {best_val_score:.4f}, "
                    f"best_val_loss: {best_val_loss:.4f}, best_epoch: {best_epoch + 1}"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        best_thr = best_epoch_thr

    # ===============================
    # LSTM 特征重要性分析
    # ===============================
    if show_features_importance:
        importances = cnn_lstm_permutation_importance(model, X_val_seis_t, X_val_log_t, y_val_t, loss_fn=criterion_bce,
                                                      device=DEVICE)
        all_importances.append(importances)

        print("\nFeature Importance:")
        for i, imp in enumerate(importances):
            print(f"{imaging_well_features[i]} : {imp:.6f}")

        # 按重要性排序
        if show_one_well_features_importance:
            idx = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 5))
            plt.bar(np.array(imaging_well_features)[idx], importances[idx])
            plt.xticks(rotation=60)
            plt.ylabel("Importance")
            plt.title(f"Feature Importance ({val_well})")
            plt.tight_layout()
            plt.show()

    # ---------- 验证 ----------
    model.eval()
    with torch.no_grad():
        cls_out, den_out, domain_out, emb = model(X_val_seis_t, X_val_log_t, alpha)
        prob = torch.sigmoid(cls_out).cpu().numpy()

        # 概率平滑
        if use_smooth:
            prob = smooth_prob(prob, window=3)
        # 动态阈值
        if use_dynamic_threshold:
            best_thr, best_metric_score = search_best_threshold_by_metric(
                prob,
                y_val_seq,
                selection_metric,
                val_df=val_df,
                val_well=val_well,
            )
        pred = build_eval_pred(prob, best_thr, selection_metric)
        den_pred = den_out.cpu().numpy() if den_out is not None else None

    pred_summary = summarize_binary_prediction(y_val_seq, pred)
    acc = accuracy_score(y_val_seq, pred)
    auc = roc_auc_score(y_val_seq, prob) if len(np.unique(y_val_seq)) > 1 else np.nan
    report = classification_report(y_val_seq, pred, output_dict=True, zero_division=0)

    if use_dynamic_threshold and selection_metric == "iou":
        iou = best_metric_score
    else:
        iou_list = []
        start = 0
        for well in val_df["WellName"].unique():
            well_df = val_df[val_df["WellName"] == well]
            n = len(well_df)
            n_seq = max(0, n - 2 * HALF)

            if n_seq <= 0:
                continue

            gt_well = y_val_seq[start:start + n_seq]
            pred_well = pred[start:start + n_seq]

            iou_list.append(compute_iou(gt_well, pred_well))
            start += n_seq

        iou = np.mean(iou_list) if iou_list else 0

    # 密度误差计算
    mask = (y_val_seq == 1) & (pred == 1)
    if use_density_regression and mask.sum() > 0:
        gt_den = np.expm1(y_val_den_seq[mask])
        pred_den = np.expm1(den_pred[mask])

        # percent_error = np.abs(pred_den - gt_den) / (gt_den + 1e-6) * 100
        percent_error = np.abs(pred_den - gt_den) / np.maximum(gt_den, 1.0) * 100
        mean_percent_error = percent_error.mean(axis=0)
    else:
        mean_percent_error = [0, 0, 0]

    print(f"""
    val_well={val_well}

    AUC={auc:.3f}
    Accuracy={acc:.3f}
    Precision={report['1']['precision']:.3f}
    Recall={report['1']['recall']:.3f}
    F1={report['1']['f1-score']:.3f}
    PredPositiveRatio={pred_summary['pred_pos_ratio']:.3f}
    TruePositiveRatio={pred_summary['true_pos_ratio']:.3f}
    Threshold={best_thr:.3f}
    IoU={iou:.3f}

    Density % Error:
    P10={mean_percent_error[0]:.2f}%
    P21={mean_percent_error[1]:.2f}%
    P33={mean_percent_error[2]:.2f}%
    """)

    all_results.append({
        "val_well": val_well,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"],
        "PredPositiveRatio": pred_summary["pred_pos_ratio"],
        "TruePositiveRatio": pred_summary["true_pos_ratio"],
        "Threshold": best_thr,
        "IoU": iou,
        "P10": mean_percent_error[0],
        "P21": mean_percent_error[1],
        "P33": mean_percent_error[2],
        "best_loss": best_val_loss,
        f"best_val_{selection_metric}": best_val_score
    })

    # ---------- 保存 ----------
    torch.save(model.state_dict(), os.path.join(well_save_dir, "model.pth"))
    joblib.dump(scaler_artifact,
                os.path.join(well_save_dir, "scaler.pkl"))

    config = {
        "SEQ_LEN": SEQ_LEN,
        "features": features,
        "model": "FractureLSTM",
        "hidden_dim": model.hidden_dim,
        "best_thrESHOLD": float(best_thr),
        "selection_metric": selection_metric,
        "configured_fixed_threshold": configured_fixed_threshold,
        "THRESH_SEARCH_MIN": THRESH_SEARCH_MIN,
        "THRESH_SEARCH_MAX": THRESH_SEARCH_MAX,
        "THRESH_SEARCH_STEP": THRESH_SEARCH_STEP,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "SEIS_MODE": SEIS_MODE,
        "DIST_MATRIX_PATH": DIST_MATRIX_PATH,
        "DIST_THRESHOLD": DIST_THRESHOLD,
        "MIN_TRAIN_WELLS": MIN_TRAIN_WELLS,
        "MAX_POS_WEIGHT": MAX_POS_WEIGHT,
        "use_dice_loss": use_dice_loss,
        "DICE_LOSS_WEIGHT": DICE_LOSS_WEIGHT,
        "use_selection_accuracy_floor": use_selection_accuracy_floor,
        "MIN_SELECTION_ACCURACY": MIN_SELECTION_ACCURACY,
        "use_threshold_constraints": use_threshold_constraints,
        "MIN_SELECTION_RECALL": MIN_SELECTION_RECALL,
        "MIN_SELECTION_POS_RATIO": MIN_SELECTION_POS_RATIO,
        "MIN_SELECTION_POS_RATIO_SCALE": MIN_SELECTION_POS_RATIO_SCALE,
        "use_selection_accuracy_floor_by_well": use_selection_accuracy_floor_by_well,
        "min_selection_accuracy_by_well": min_selection_accuracy_by_well,
        "use_threshold_constraints_by_well": use_threshold_constraints_by_well,
        "min_selection_recall_by_well": min_selection_recall_by_well,
        "min_selection_pos_ratio_by_well": min_selection_pos_ratio_by_well,
        "min_selection_pos_ratio_scale_by_well": min_selection_pos_ratio_scale_by_well,
        "only_val_wells": only_val_wells,
        "custom_train_wells_map": custom_train_wells_map,
        "use_over_smaple": use_over_smaple,
        "manual_pos_weight": manual_pos_weight,
        "target_ratio": target_ratio,
        "edge_exclude": edge_exclude,
        "use_density_regression": use_density_regression,
        "use_domain_adversarial": use_domain_adversarial,
        "use_all_other_wells": use_all_other_wells,
        "use_wellwise_log_scaling": use_wellwise_log_scaling,
        "use_well_balanced_sampling": use_well_balanced_sampling,
        "train_wells": train_df["WellName"].unique().tolist(),
        "val_well": val_well,
    }

    with open(os.path.join(well_save_dir, "config.json"), "w", encoding="utf-8-sig") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ===== 预测结果保存 =====
    val_raw_df = df_raw[df_raw["WellName"] == val_well].copy()

    # 初始化预测列
    val_raw_df["PRED_PROB"] = np.nan
    val_raw_df["PRED_LABEL"] = np.nan
    if use_density_regression:
        val_raw_df["P10_PRED"] = np.nan
        val_raw_df["P21_PRED"] = np.nan
        val_raw_df["P33_PRED"] = np.nan
    # 真实标签
    val_raw_df["GT_LABEL"] = np.nan

    # LSTM可预测中心点
    pred_index = val_center_row_idx
    if len(pred_index) != len(prob):
        raise ValueError(
            f"Prediction index length mismatch for {val_well}: "
            f"indices={len(pred_index)}, prob={len(prob)}"
        )

    # ---------- 分类预测 ----------
    val_raw_df.loc[pred_index, "PRED_PROB"] = prob
    val_raw_df.loc[pred_index, "PRED_LABEL"] = pred

    # ---------- 真实标签 ----------
    val_raw_df.loc[pred_index, "GT_LABEL"] = y_val_seq

    # ---------- 密度预测 ----------
    if use_density_regression and den_pred is not None:
        den_pred_real = np.expm1(den_pred)
        val_raw_df.loc[pred_index, "P10_PRED"] = den_pred_real[:, 0]
        val_raw_df.loc[pred_index, "P21_PRED"] = den_pred_real[:, 1]
        val_raw_df.loc[pred_index, "P33_PRED"] = den_pred_real[:, 2]

    # 删除全空列
    val_raw_df = val_raw_df.dropna(axis=1, how="all")

    # 保存
    val_raw_df.to_csv(
        os.path.join(well_save_dir, "val_pred_full_log.csv"),
        index=False,
        encoding="utf-8-sig"
    )

# ===================== 5. 汇总结果 =====================
result_df = pd.DataFrame(all_results)
result_df.to_csv(os.path.join(SAVE_DIR, "loo_results.csv"), index=False, encoding="utf-8-sig")

print("\n====== Leave-One-Well-Out 总结 ======")
print(result_df)
print("\n平均性能：")
print(result_df.mean(numeric_only=True))

if show_features_importance:
    all_importances = np.array(all_importances)

    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)
    print("\n==== Global Feature Importance ====")
    print("\nFeature\tMean\tStd")
    for f, mimp, simp in sorted(zip(imaging_well_features, mean_importance, std_importance), key=lambda x: x[1],
                                reverse=True):
        print(f"{f}:\t{mimp:.4f},\t{simp:.4f}")

    importance_df = pd.DataFrame({
        "Feature": imaging_well_features,
        "Importance": mean_importance
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        os.path.join(SAVE_DIR, "feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 画图
    idx = np.argsort(mean_importance)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(np.array(imaging_well_features)[idx], mean_importance[idx])
    plt.xticks(rotation=60)
    plt.ylabel("Importance")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.show()
