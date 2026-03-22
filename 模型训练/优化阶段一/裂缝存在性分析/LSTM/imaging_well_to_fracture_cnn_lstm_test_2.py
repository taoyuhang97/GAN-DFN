import os
import glob
import joblib
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ===================== 基本设置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 7
HALF = SEQ_LEN // 2
BATCH_SIZE = 32
LR = 1e-4
Dropout = 0.3
# ---------- Early Stopping ----------
EPOCHS = 50
patience = 8
min_delta = 1e-4

# 动态阈值
# use_dynamic_threshold = True
use_dynamic_threshold = False
best_thr = 0.5
# 概率平滑
# use_smooth = True
# use_smooth = False
# 小段处理
# use_short_segment_processing = True
# use_short_segment_processing = False
# 过采样操作
# use_over_smaple = True
use_over_smaple = False
manual_pos_weight = 2.0
target_ratio = 0.35
edge_exclude = 1
# 特征重要性检查
show_features_importance = False
show_one_well_features_importance = False
all_importances = []

DATA_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
SAVE_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\LSTM\单井验证\成像测井裂缝预测\cnn+lstm"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 特征定义 =====================
# imaging_well_features = ['AC', 'GR', 'CAL', 'CNL', 'PE', 'DEN', 'CON1', 'GRSL', 'K', 'KTH', 'TH', 'U']
imaging_well_features = ['GR', 'CON1', 'DEN', 'AC', 'GRSL']

# SEIS_MODE = "3x3x7"
# SEIS_MODE = "3x3"
SEIS_MODE = "1"
seis_features = [f"SEIS_{i}" for i in range(63)]
features = seis_features + imaging_well_features
N_LOG = len(imaging_well_features)

# ===================== 1. 数据读取 =====================
# csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
csv_files = [
    os.path.join(DATA_DIR, "车660-1_sample.csv"),
    os.path.join(DATA_DIR, "车660-2_sample.csv"),
    os.path.join(DATA_DIR, "车662_sample.csv"),
    os.path.join(DATA_DIR, "车663_sample.csv"),
]

df_list = []
df_raw_list = []
for f in csv_files:
    df = pd.read_csv(f)
    df["WellName"] = os.path.basename(f).split("_")[0]
    df_list.append(df)
    df_raw_list.append(df.copy())

df_raw = pd.concat(df_raw_list, ignore_index=True)
df = pd.concat(df_list, ignore_index=True).copy()

df["FRACTURE_FLAG"] = df["Frac_Azimuth"].notna().astype(int)
df = df[features + ["FRACTURE_FLAG", "WellName"]].dropna()

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


# ===================== 2. LSTM 定义 =====================
class FractureCNNLSTM(nn.Module):
    def __init__(self, seis_mode, log_dim, hidden_dim=64):
        super().__init__()
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

    def forward(self, seis, log):

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
        return cls_out


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


def build_sequences_by_well(df, X_seis, X_log, y_cls, seq_len):
    X_seis_seq, X_log_seq = [], []
    y_cls_seq = []

    start = 0

    for well in df["WellName"].unique():

        well_df = df[df["WellName"] == well]
        n = len(well_df)

        seis_well = X_seis[start:start + n]
        log_well = X_log[start:start + n]

        y_cls_well = y_cls[start:start + n]

        for i in range(HALF, n - HALF):
            X_seis_seq.append(seis_well[i - HALF:i + HALF + 1])
            X_log_seq.append(log_well[i - HALF:i + HALF + 1])

            y_cls_seq.append(y_cls_well[i])

        start += n

    return (
        np.array(X_seis_seq),
        np.array(X_log_seq),
        np.array(y_cls_seq)
    )


# ================= 中心段过采样 =================
def oversample_center_segments(seis, log, y_cls,
                               target_ratio=0.35,
                               edge_exclude=1,
                               random_state=42):
    """
    只对连续裂缝段中心序列过采样
    同时扩展 seis / log / y_cls
    """

    rng = np.random.RandomState(random_state)

    seis = np.array(seis)
    log = np.array(log)
    y_cls = np.array(y_cls)

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
        return seis, log, y_cls

    center_indices = np.array(center_indices)

    # ---------- 当前比例 ----------
    n_pos = (y_cls == 1).sum()
    n_total = len(y_cls)
    current_ratio = n_pos / n_total

    if current_ratio >= target_ratio:
        return seis, log, y_cls

    # ---------- 需要增加多少 ----------
    desired_pos = int(target_ratio * n_total / (1 - target_ratio))
    n_to_add = desired_pos - n_pos

    if n_to_add <= 0:
        return seis, log, y_cls

    # ---------- 只从中心点采样 ----------
    add_idx = rng.choice(center_indices,
                         size=n_to_add,
                         replace=True)

    seis_new = np.concatenate([seis, seis[add_idx]], axis=0)
    log_new = np.concatenate([log, log[add_idx]], axis=0)

    y_cls_new = np.concatenate([y_cls, y_cls[add_idx]], axis=0)
    return seis_new, log_new, y_cls_new


def cnn_lstm_permutation_importance(model, X_seis, X_log, y, loss_fn, device="cpu"):
    model.eval()

    X_seis = X_seis.clone().to(device)
    X_log = X_log.clone().to(device)
    y = y.clone().to(device)

    with torch.no_grad():
        cls_out = model(X_seis, X_log)
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
                cls_out = model(X_seis, X_log_perm)

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


# ================= 动态阈值搜索 =================
def search_best_threshold_by_accuracy(prob, y_true):
    best_thr = 0.5
    best_acc = -1

    for thr in np.arange(0.2, 0.71, 0.02):
        temp_pred = (prob >= thr).astype(int)
        acc = accuracy_score(y_true, temp_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_thr, best_acc


# ===================== 4. Leave-One-Well-Out 循环 =====================
all_results = []
for val_well in well_names:
    print(f"\n================ 验证井：{val_well} ================")
    well_save_dir = os.path.join(SAVE_DIR, f"verify_{val_well}")
    os.makedirs(well_save_dir, exist_ok=True)

    train_wells = [w for w in well_names if w != val_well]
    train_df = df[df["WellName"].isin(train_wells)].copy()
    val_df = df[df["WellName"] == val_well].copy()

    print("测试井:", [val_well])
    print("训练井:", train_wells)

    scaler = StandardScaler()
    train_df.loc[:, features] = scaler.fit_transform(train_df[features])
    val_df.loc[:, features] = scaler.transform(val_df[features])

    X_train_all = train_df[features].values
    X_val_all = val_df[features].values
    X_train_seis = reshape_seis(X_train_all[:, :63], SEIS_MODE)
    X_val_seis = reshape_seis(X_val_all[:, :63], SEIS_MODE)
    X_train_log = X_train_all[:, 63:]
    X_val_log = X_val_all[:, 63:]

    y_train = train_df["FRACTURE_FLAG"].values
    y_val = val_df["FRACTURE_FLAG"].values

    X_train_seis_seq, X_train_log_seq, y_train_seq = build_sequences_by_well(
        train_df,
        X_train_seis,
        X_train_log,
        y_train,
        SEQ_LEN
    )
    aug_seis = augment_seismic(X_train_seis_seq)
    X_train_seis_seq = np.concatenate([X_train_seis_seq, aug_seis])
    X_train_log_seq = np.concatenate([X_train_log_seq, X_train_log_seq])
    y_train_seq = np.concatenate([y_train_seq, y_train_seq])

    if use_over_smaple:
        print("Before oversample:", (y_train_seq == 1).sum() / len(y_train_seq))
        X_train_seis_seq, X_train_log_seq, y_train_seq = oversample_center_segments(
            X_train_seis_seq,
            X_train_log_seq,
            y_train_seq,
            target_ratio=target_ratio,
            edge_exclude=edge_exclude
        )
        print("After oversample:", (y_train_seq == 1).sum() / len(y_train_seq))

    X_val_seis_seq, X_val_log_seq, y_val_seq = build_sequences_by_well(
        val_df,
        X_val_seis,
        X_val_log,
        y_val,
        SEQ_LEN
    )

    X_train_seis_t = torch.tensor(X_train_seis_seq, dtype=torch.float32).to(DEVICE)
    X_train_log_t = torch.tensor(X_train_log_seq, dtype=torch.float32).to(DEVICE)
    X_val_seis_t = torch.tensor(X_val_seis_seq, dtype=torch.float32).to(DEVICE)
    X_val_log_t = torch.tensor(X_val_log_seq, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).to(DEVICE)

    print("Shapes:")
    print("seis:", X_train_seis_seq.shape)
    print("log:", X_train_log_seq.shape)
    print("y_cls:", y_train_seq.shape)

    train_loader = DataLoader(
        TensorDataset(X_train_seis_t, X_train_log_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = FractureCNNLSTM(
        seis_mode=SEIS_MODE,
        log_dim=len(imaging_well_features)
    ).to(DEVICE)

    if use_over_smaple:
        pos_weight = manual_pos_weight
    else:
        n_pos = (y_train_seq == 1).sum()
        n_neg = (y_train_seq == 0).sum()
        pos_weight = n_neg / max(n_pos, 1)
    print(f"pos_weight = {pos_weight:.3f}")
    print(f"train pos ratio = {(y_train_seq == 1).mean():.3f}")

    criterion_cls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None
    best_epoch = -1

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{val_well}] Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for xb_seis, xb_log, yb_cls in pbar:
            optimizer.zero_grad()
            cls_out = model(xb_seis, xb_log)
            loss = criterion_cls(cls_out, yb_cls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=float(loss))

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            cls_out_val = model(X_val_seis_t, X_val_log_t)
            val_loss = criterion_cls(cls_out_val, y_val_t).item()
            val_prob = torch.sigmoid(cls_out_val).cpu().numpy()

        epoch_thr, epoch_acc = search_best_threshold_by_accuracy(val_prob, y_val_seq)
        print(
            f"Epoch {epoch + 1} | TrainLoss={avg_train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | ValAcc={epoch_acc:.4f} | Thr={epoch_thr:.2f}"
        )

        is_better_acc = epoch_acc > best_val_acc + min_delta
        is_same_acc_lower_loss = abs(epoch_acc - best_val_acc) <= min_delta and val_loss < best_val_loss - min_delta
        if is_better_acc or is_same_acc_lower_loss:
            best_val_acc = epoch_acc
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter}/{patience}, best_val_acc: {best_val_acc:.4f}")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered, best_val_acc: {best_val_acc:.4f}, best_epoch: {best_epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if show_features_importance:
        importances = cnn_lstm_permutation_importance(
            model,
            X_val_seis_t,
            X_val_log_t,
            y_val_t,
            loss_fn=criterion_cls,
            device=DEVICE
        )
        all_importances.append(importances)

        print("\nFeature Importance:")
        for i, imp in enumerate(importances):
            print(f"{imaging_well_features[i]} : {imp:.6f}")

        if show_one_well_features_importance:
            idx = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 5))
            plt.bar(np.array(imaging_well_features)[idx], importances[idx])
            plt.xticks(rotation=60)
            plt.ylabel("Importance")
            plt.title(f"Feature Importance ({val_well})")
            plt.tight_layout()
            plt.show()

    model.eval()
    with torch.no_grad():
        cls_out = model(X_val_seis_t, X_val_log_t)
        prob = torch.sigmoid(cls_out).cpu().numpy()

    if use_smooth:
        prob = smooth_prob(prob, window=3)
    if use_dynamic_threshold:
        best_thr, _ = search_best_threshold_by_accuracy(prob, y_val_seq)
    pred = (prob >= best_thr).astype(int)

    if use_short_segment_processing:
        pred = remove_short_segments(pred, min_len=3)
        pred = fill_small_gaps(pred, max_gap=2)

    acc = accuracy_score(y_val_seq, pred)
    auc = roc_auc_score(y_val_seq, prob) if len(np.unique(y_val_seq)) > 1 else np.nan
    report = classification_report(y_val_seq, pred, output_dict=True, zero_division=0)

    print(f"""
    val_well={val_well}

    AUC={auc:.3f}
    Accuracy={acc:.3f}
    Precision={report['1']['precision']:.3f}
    Recall={report['1']['recall']:.3f}
    F1={report['1']['f1-score']:.3f}
    Threshold={best_thr:.3f}
    """)

    all_results.append({
        "val_well": val_well,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"],
        "Threshold": best_thr,
        "best_val_acc": best_val_acc,
        "best_loss": best_val_loss
    })

    torch.save(model.state_dict(), os.path.join(well_save_dir, "model.pth"))
    joblib.dump(scaler, os.path.join(well_save_dir, "scaler.pkl"))

    config = {
        "SEQ_LEN": SEQ_LEN,
        "features": features,
        "model": "FractureCNNLSTM_AccuracyFocused",
        "hidden_dim": 64,
        "BEST_THRESHOLD": float(best_thr),
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "SEIS_MODE": SEIS_MODE,
        "train_wells": train_df["WellName"].unique().tolist(),
        "val_well": val_well,
    }

    with open(os.path.join(well_save_dir, "config.json"), "w", encoding="utf-8-sig") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    val_raw_df = df_raw[df_raw["WellName"] == val_well].copy()
    val_raw_df["PRED_PROB"] = np.nan
    val_raw_df["PRED_LABEL"] = np.nan
    val_raw_df["GT_LABEL"] = np.nan

    pred_index = val_raw_df.index[HALF: len(val_raw_df) - HALF]
    val_raw_df.loc[pred_index, "PRED_PROB"] = prob
    val_raw_df.loc[pred_index, "PRED_LABEL"] = pred
    val_raw_df.loc[pred_index, "GT_LABEL"] = y_val_seq
    val_raw_df = val_raw_df.dropna(axis=1, how="all")

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
    for f, mimp, simp in sorted(zip(imaging_well_features, mean_importance, std_importance), key=lambda x: x[1], reverse=True):
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

    idx = np.argsort(mean_importance)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(np.array(imaging_well_features)[idx], mean_importance[idx])
    plt.xticks(rotation=60)
    plt.ylabel("Importance")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.show()
