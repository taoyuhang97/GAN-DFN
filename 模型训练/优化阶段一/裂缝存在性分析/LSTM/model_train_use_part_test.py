import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from tqdm import tqdm

# ===================== 基本设置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 11
HALF = SEQ_LEN // 2
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
N_FOLDS = 5
# 动态阈值
ues_dynamic_threshold = False
best_thr = 0.5
# 概率平滑
use_smooth = True
# 小段处理
use_short_segment_processing = True
# 过采样操作
use_over_smaple = True
pos_weight = 2.0
target_ratio = 0.35
edge_exclude = 1

DATA_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
SAVE_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\LSTM\井段拆分验证"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 特征定义 =====================
# features = ["SEIS_TRUE"] + [f"SEIS_{i}" for i in range(63)] + ["AC", "GR"]
# features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(3, 63, 7)] + ['AC', 'GR']
# features = [f"SEIS_{i}" for i in range(63)] + ["AC", "GR"]
# features = [f'SEIS_{i}' for i in range(3, 63, 7)] + ['AC', 'GR']
features = ['SEIS_TRUE'] + ['AC', 'GR']

density_cols = ["P10", "P21", "P33"]

# ===================== 1. 数据读取 =====================
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

df_list = []
df_raw_list = []

for f in csv_files:
    df = pd.read_csv(f)
    df["WellName"] = os.path.basename(f).split("_")[0]
    df_list.append(df)
    df_raw_list.append(df.copy())

df_raw = pd.concat(df_raw_list, ignore_index=True)
df = pd.concat(df_list, ignore_index=True)

df["FRACTURE_FLAG"] = df["Frac_Azimuth"].notna().astype(int)
# 只在有裂缝位置保留密度
for col in density_cols:
    if col not in df.columns:
        df[col] = 0.0
df = df[features + density_cols + ["FRACTURE_FLAG", "WellName"]].dropna()
# log 变换密度（推荐）
df[density_cols] = np.log1p(df[density_cols])

df_raw["DEPTH_IDX"] = df_raw.groupby("WellName").cumcount()
df["DEPTH_IDX"] = df.groupby("WellName").cumcount()

well_names = df["WellName"].unique()
print("参与建模井：", well_names)


# ===================== 2. LSTM 模型 =====================
class FractureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc_cls = nn.Linear(hidden_dim, 1)
        self.fc_den = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        cls_out = self.fc_cls(out).squeeze(1)
        den_out = self.fc_den(out)

        return cls_out, den_out


# ===================== 3. 按井构造序列 =====================
def build_sequences_by_well(df, X, y_cls, y_den, seq_len):
    X_seq, y_cls_seq, y_den_seq = [], [], []
    start = 0

    for well in df["WellName"].unique():
        well_df = df[df["WellName"] == well]
        n = len(well_df)

        X_well = X[start:start + n]
        y_cls_well = y_cls[start:start + n]
        y_den_well = y_den[start:start + n]

        for i in range(HALF, n - HALF):
            X_seq.append(X_well[i - HALF: i + HALF + 1])
            y_cls_seq.append(y_cls_well[i])
            y_den_seq.append(y_den_well[i])

        start += n

    return np.array(X_seq), np.array(y_cls_seq), np.array(y_den_seq)

# ================= 中心段过采样 =================
def oversample_center_segments(X, y_cls, y_den,
                               target_ratio=0.35,
                               edge_exclude=1,
                               random_state=42):
    """
    只对连续裂缝段中心序列过采样
    """

    rng = np.random.RandomState(random_state)

    X = np.array(X)
    y_cls = np.array(y_cls)
    y_den = np.array(y_den)

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
        return X, y_cls, y_den

    center_indices = np.array(center_indices)

    # ---------- 计算当前比例 ----------
    n_pos = (y_cls == 1).sum()
    n_total = len(y_cls)
    current_ratio = n_pos / n_total

    if current_ratio >= target_ratio:
        return X, y_cls, y_den

    # ⚠ 正确计算需要新增多少正样本
    desired_pos = int(target_ratio * n_total / (1 - target_ratio))
    n_to_add = desired_pos - n_pos

    if n_to_add <= 0:
        return X, y_cls, y_den

    # ---------- 只从中心点抽样 ----------
    add_idx = rng.choice(center_indices,
                         size=n_to_add,
                         replace=True)

    X_new = np.concatenate([X, X[add_idx]], axis=0)
    y_cls_new = np.concatenate([y_cls, y_cls[add_idx]], axis=0)
    y_den_new = np.concatenate([y_den, y_den[add_idx]], axis=0)

    return X_new, y_cls_new, y_den_new

# ===================== 4. 按深度比例切分（核心） =====================
def split_by_depth_ratio(df, n_folds, fold_id):
    train_parts, val_parts = [], []

    for well in df["WellName"].unique():
        well_df = df[df["WellName"] == well].reset_index(drop=True)
        n = len(well_df)

        start = int(n * fold_id / n_folds)
        end = int(n * (fold_id + 1) / n_folds)

        val_parts.append(well_df.iloc[start:end])
        train_parts.append(
            pd.concat([well_df.iloc[:start], well_df.iloc[end:]], axis=0)
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


# ===================== 5. 5-Fold 训练 =====================
all_results = []

for fold_id in range(N_FOLDS):
    print(f"\n================ Fold {fold_id + 1}/{N_FOLDS} =================")

    fold_dir = os.path.join(SAVE_DIR, f"fold_{fold_id + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df, val_df = split_by_depth_ratio(df, N_FOLDS, fold_id)

    # ---------- 标准化 ----------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].values)
    X_val = scaler.transform(val_df[features].values)

    y_train = train_df["FRACTURE_FLAG"].values
    y_val = val_df["FRACTURE_FLAG"].values

    y_train_den = train_df[density_cols].values
    y_val_den = val_df[density_cols].values

    # ---------- 构造序列 ----------
    X_train_seq, y_train_seq, y_train_den_seq = build_sequences_by_well(
        train_df, X_train, y_train, y_train_den, SEQ_LEN
    )
    if use_over_smaple:
        X_train_seq, y_train_seq, y_train_den_seq = oversample_center_segments(
            X_train_seq,
            y_train_seq,
            y_train_den_seq,
            target_ratio,  # 建议 0.30~0.40
            edge_exclude  # 去掉每段两端1个
        )

    X_val_seq, y_val_seq, y_val_den_seq = build_sequences_by_well(
        val_df, X_val, y_val, y_val_den, SEQ_LEN
    )

    # ---------- Tensor ----------
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE),
            torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE),
            torch.tensor(y_train_den_seq, dtype=torch.float32).to(DEVICE)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---------- 模型 ----------
    model = FractureLSTM(input_dim=X_train_seq.shape[2]).to(DEVICE)

    if not use_over_smaple:
        pos_weight = (y_train_seq == 0).sum() / (y_train_seq == 1).sum()
    criterion_cls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------- 训练 ----------
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fold {fold_id + 1} Epoch {epoch + 1}", leave=False)

        for xb, yb_cls, yb_den in pbar:

            cls_out, den_out = model(xb)

            loss_cls = criterion_cls(cls_out, yb_cls)

            mask = (yb_cls == 1).float().unsqueeze(1)

            if mask.sum().item() > 0:
                loss_den = ((den_out - yb_den) ** 2 * mask).sum() / mask.sum()
            else:
                loss_den = torch.tensor(0.0).to(DEVICE)

            loss = loss_cls + 0.5 * loss_den

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))


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
    def search_best_threshold(prob, y_true):
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

    # ---------- 验证 ----------
    model.eval()
    with torch.no_grad():
        cls_out, den_out = model(torch.tensor(X_val_seq, dtype=torch.float32).to(DEVICE))
        prob = torch.sigmoid(cls_out).cpu().numpy()

        # 概率平滑
        if use_smooth:
            prob = smooth_prob(prob, window=3)
        # 动态阈值
        if ues_dynamic_threshold:
            best_thr, best_iou = search_best_threshold(prob, y_val_seq)
        pred = (prob >= best_thr).astype(int)
        # 小段处理
        if use_short_segment_processing:
            pred = remove_short_segments(pred, min_len=3)
            pred = fill_small_gaps(pred, max_gap=2)
        den_pred = den_out.cpu().numpy()

    acc = accuracy_score(y_val_seq, pred)
    auc = roc_auc_score(y_val_seq, prob) if len(np.unique(y_val_seq)) > 1 else np.nan
    report = classification_report(y_val_seq, pred, output_dict=True)

    if ues_dynamic_threshold:
        iou = best_iou
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
    if mask.sum() > 0:
        gt_den = np.expm1(y_val_den_seq[mask])
        pred_den = np.expm1(den_pred[mask])

        # percent_error = np.abs(pred_den - gt_den) / (gt_den + 1e-6) * 100
        percent_error = np.abs(pred_den - gt_den) / np.maximum(gt_den, 1.0) * 100
        mean_percent_error = percent_error.mean(axis=0)
    else:
        mean_percent_error = [0, 0, 0]

    print(f"""
    Fold {fold_id + 1}

    AUC={auc:.3f}
    Accuracy={acc:.3f}
    Precision={report['1']['precision']:.3f}
    Recall={report['1']['recall']:.3f}
    F1={report['1']['f1-score']:.3f}
    Threshold={best_thr:.3f}
    IoU={iou:.3f}

    Density % Error:
    P10={mean_percent_error[0]:.2f}%
    P21={mean_percent_error[1]:.2f}%
    P33={mean_percent_error[2]:.2f}%
    """)

    all_results.append({
        "Accuracy": acc,
        "AUC": auc,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"],
        "Threshold": best_thr,
        "IoU": iou,
        "P10": mean_percent_error[0],
        "P21": mean_percent_error[1],
        "P33": mean_percent_error[2],
    })

    # ---------- 保存 ----------
    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pth"))
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    config = {
        "CV_TYPE": "DepthBlockedKFold",
        "N_FOLDS": N_FOLDS,
        "FOLD_ID": fold_id + 1,
        "SEQ_LEN": SEQ_LEN,
        "features": features,
        "model": "FractureLSTM",
        "hidden_dim": 64,
        "BEST_THRESHOLD": float(best_thr)
    }

    with open(os.path.join(fold_dir, "config.json"),
              "w", encoding="utf-8-sig") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ================= 保存“仅验证段”的预测结果 =================
    val_pred_rows = []

    seq_ptr = 0  # 用于在 y_val_seq / prob 中定位

    for well in val_df["WellName"].unique():
        well_raw = df_raw[df_raw["WellName"] == well].reset_index(drop=True)

        # 当前 fold 下该井的验证段
        well_val = val_df[val_df["WellName"] == well].reset_index(drop=True)
        n = len(well_val)

        # LSTM 在该井内可预测的中心点数量
        n_pred = max(0, n - 2 * HALF)

        if n_pred <= 0:
            continue

        # 验证段内中心点索引（相对 well_val）
        center_idx = np.arange(HALF, n - HALF)

        # 对应到原始井数据的索引
        raw_idx = well_val.loc[center_idx, "DEPTH_IDX"].values
        df_tmp = well_raw.loc[raw_idx].copy()
        df_tmp["PRED_PROB"] = prob[seq_ptr: seq_ptr + n_pred]
        df_tmp["PRED_LABEL"] = pred[seq_ptr: seq_ptr + n_pred]
        df_tmp["FOLD_ID"] = fold_id + 1
        df_tmp["GT_LABEL"] = well_val.loc[center_idx, "FRACTURE_FLAG"].values

        # -------- 裂缝密度预测 --------
        den_tmp = den_pred[seq_ptr: seq_ptr + n_pred]

        # 反log
        den_tmp = np.expm1(den_tmp)

        df_tmp["P10_PRED"] = den_tmp[:, 0]
        df_tmp["P21_PRED"] = den_tmp[:, 1]
        df_tmp["P33_PRED"] = den_tmp[:, 2]

        val_pred_rows.append(df_tmp)

        seq_ptr += n_pred

    # 合并并仅保留验证预测点
    if val_pred_rows:
        val_pred_df = pd.concat(val_pred_rows, ignore_index=True)
        val_pred_df = val_pred_df.dropna(axis=1, how="all")

        val_pred_df.to_csv(
            os.path.join(fold_dir, "val_predictions_only.csv"),
            index=False,
            encoding="utf-8-sig"
        )

# ===================== 6. 汇总 =====================
result_df = pd.DataFrame(all_results)
result_df.to_csv(
    os.path.join(SAVE_DIR, "depth5fold_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n====== Depth-Blocked 5-Fold 总结 ======")
print(result_df)
print("\n平均性能：")
print(result_df.mean(numeric_only=True))
