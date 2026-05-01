import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import json

# ===================== 基本设置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 影响待预测点位的上下数据区间
SEQ_LEN = 21
HALF = SEQ_LEN // 2
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

DATA_DIR = r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝样本"
SAVE_DIR = r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM"
os.makedirs(SAVE_DIR, exist_ok=True)

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
df = pd.concat(df_list, ignore_index=True).copy()

features = (
        ["SEIS_TRUE"] +
        [f"SEIS_{i}" for i in range(63)] +
        ["AC", "GR"]
)

df["FRACTURE_FLAG"] = df["Frac_Azimuth"].notna().astype(int)
df = df[features + ["FRACTURE_FLAG", "WellName"]].dropna()

well_names = df["WellName"].unique()
print("参与建模井：", well_names)


# ===================== 2. LSTM 定义 =====================
class FractureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)     # [B, T, H]
        out = out[:, -1, :]      # [B, H]
        out = self.fc(out)       # [B, 1]
        return out.squeeze(1)    # [B]



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
            y_seq.append(y_well[i])  # 中心点标签

        start += n

    return np.array(X_seq), np.array(y_seq)


# ===================== 4. Leave-One-Well-Out 循环 =====================
all_results = []

for val_well in well_names:
    print(f"\n================ 验证井：{val_well} ================")
    well_save_dir = os.path.join(SAVE_DIR, f"verify_{val_well}")
    os.makedirs(well_save_dir, exist_ok=True)

    train_df = df[df["WellName"] != val_well]
    val_df = df[df["WellName"] == val_well]

    # ---------- 标准化 ----------
    scaler = StandardScaler()
    X_train_raw = train_df[features].values
    X_val_raw = val_df[features].values

    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

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
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[{val_well}] Epoch {epoch + 1}/{EPOCHS}",
            leave=False
        )

        for xb, yb in pbar:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 每个 epoch 结束打印一次平均 loss（可选）
        avg_loss = epoch_loss / len(train_loader)

    # ---------- 验证 ----------
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        prob = torch.sigmoid(logits).cpu().numpy()
        pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val_seq, prob)
    report = classification_report(y_val_seq, pred, output_dict=True)

    all_results.append({
        "val_well": val_well,
        "AUC": auc,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"]
    })

    print(f"AUC={auc:.3f}, Recall={report['1']['recall']:.3f}")

    # 模型保存
    torch.save(
        model.state_dict(),
        os.path.join(well_save_dir, "model.pth")
    )

    joblib.dump(
        scaler,
        os.path.join(well_save_dir, "scaler.pkl")
    )

    # 验证指标记录
    metrics = {
        "val_well": val_well,
        "AUC": float(auc),
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"]
    }
    with open(os.path.join(well_save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 训练配置
    config = {
        "SEQ_LEN": SEQ_LEN,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "features": features,
        "train_wells": train_df["WellName"].unique().tolist(),
        "val_well": val_well,
        "model": "FractureLSTM",
        "hidden_dim": 64
    }
    with open(os.path.join(well_save_dir, "config.json"), "w", encoding="utf-8-sig") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ===== 预测结果保存=====
    val_raw_df = df_raw[df_raw["WellName"] == val_well].copy()
    val_raw_df["PRED_PROB"] = np.nan
    val_raw_df["PRED_LABEL"] = np.nan

    pred_index = val_raw_df.index[HALF: len(val_raw_df) - HALF]

    val_raw_df.loc[pred_index, "PRED_PROB"] = prob
    val_raw_df.loc[pred_index, "PRED_LABEL"] = pred

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
