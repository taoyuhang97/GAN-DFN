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
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

# ===================== 基本设置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 21
HALF = SEQ_LEN // 2
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
N_FOLDS = 5

DATA_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
SAVE_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\LSTM\井段拆分验证"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 特征定义 =====================
features = (["SEIS_TRUE"] + [f"SEIS_{i}" for i in range(63)] + ["AC", "GR"])
# features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(3, 63, 7)] + ['AC', 'GR']
#
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
df = df[features + ["FRACTURE_FLAG", "WellName"]].dropna()

df_raw["DEPTH_IDX"] = df_raw.groupby("WellName").cumcount()
df["DEPTH_IDX"] = df.groupby("WellName").cumcount()

well_names = df["WellName"].unique()
print("参与建模井：", well_names)


# ===================== 2. LSTM 模型 =====================
class FractureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)


# ===================== 3. 按井构造序列 =====================
def build_sequences_by_well(df, X, y, seq_len):
    X_seq, y_seq = [], []
    start = 0

    for well in df["WellName"].unique():
        well_df = df[df["WellName"] == well]
        n = len(well_df)

        X_well = X[start:start + n]
        y_well = y[start:start + n]

        for i in range(HALF, n - HALF):
            X_seq.append(X_well[i - HALF: i + HALF + 1])
            y_seq.append(y_well[i])

        start += n

    return np.array(X_seq), np.array(y_seq)


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

    # ---------- 构造序列 ----------
    X_train_seq, y_train_seq = build_sequences_by_well(
        train_df, X_train, y_train, SEQ_LEN
    )
    X_val_seq, y_val_seq = build_sequences_by_well(
        val_df, X_val, y_val, SEQ_LEN
    )

    # ---------- Tensor ----------
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---------- 模型 ----------
    model = FractureLSTM(input_dim=X_train_seq.shape[2]).to(DEVICE)

    pos_weight = (y_train_t == 0).sum() / (y_train_t == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------- 训练 ----------
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fold {fold_id + 1} Epoch {epoch + 1}", leave=False)

        for xb, yb in pbar:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss))

    # ---------- 验证 ----------
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(X_val_t)).cpu().numpy()
        pred = (prob >= 0.5).astype(int)

    if len(np.unique(y_val_seq)) > 1:
        auc = roc_auc_score(y_val_seq, prob)
    else:
        auc = np.nan

    report = classification_report(y_val_seq, pred, output_dict=True)

    print(f"Fold {fold_id + 1} AUC={auc:.3f}, Recall={report['1']['recall']:.3f}")

    all_results.append({
        "Fold": fold_id + 1,
        "AUC": auc,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"]
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
        "hidden_dim": 64
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
