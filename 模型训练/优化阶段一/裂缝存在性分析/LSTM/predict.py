import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ===================== 基本设置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\LSTM\verify_车151HF"
INPUT_CSV = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\测井\测井-地震时窗\车22_around_data.csv"
OUTPUT_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\LSTM\常规测井预测"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 1. 读取配置 =====================
with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8-sig") as f:
    config = json.load(f)

SEQ_LEN = config["SEQ_LEN"]
HALF = SEQ_LEN // 2
features = config["features"]

# ===================== 2. LSTM 定义（必须与训练一致） =====================
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
        out = out[:, -1, :]      # 中心点
        out = self.fc(out)       # [B, 1]
        return out.squeeze(1)    # [B]

# ===================== 3. 加载模型与 scaler =====================
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

model = FractureLSTM(input_dim=len(features)).to(DEVICE)
model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "model.pth"), map_location=DEVICE)
)
model.eval()

# ===================== 4. 读取待预测井数据 =====================
df_raw = pd.read_csv(INPUT_CSV)

# 保留完整原始数据用于最终输出
df_out = df_raw.copy()

# 只取模型需要的特征
df_feat = df_raw[features].copy()

# 删除特征缺失的行（否则 scaler / LSTM 会报错）
valid_mask = df_feat.notna().all(axis=1)
df_feat = df_feat.loc[valid_mask]

# ===================== 5. 标准化 =====================
X = scaler.transform(df_feat.values)

# ===================== 6. 构造中心点序列 =====================
X_seq = []
center_index = []

n = len(X)
for i in range(HALF, n - HALF):
    X_seq.append(X[i - HALF: i + HALF + 1])
    center_index.append(df_feat.index[i])

X_seq = np.array(X_seq)

# ===================== 7. LSTM 预测 =====================
X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    logits = model(X_t)
    prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(int)

# ===================== 8. 写回预测结果（对齐中心点） =====================
df_out["PRED_PROB"] = np.nan
df_out["PRED_LABEL"] = np.nan

df_out.loc[center_index, "PRED_PROB"] = prob
df_out.loc[center_index, "PRED_LABEL"] = pred

# 删除全空列（你之前提的需求）
df_out = df_out.dropna(axis=1, how="all")

# ===================== 9. 保存结果 =====================
well_name = os.path.basename(INPUT_CSV).split("_")[0]
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{well_name}.csv")
df_out.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("预测完成，结果已保存至：")
print(OUTPUT_CSV)
