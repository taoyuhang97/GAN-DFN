import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===== 配置 =====
logging_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝角度预测\模型\XGBoost"
os.makedirs(logging_dir, exist_ok=True)

well_log_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\测井-地震时窗"
fracture_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\裂缝标注"
model_save_path = os.path.join(logging_dir, "fracture_angle_model.pkl")

features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(63)]
# features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(3, 63, 7)]

# ===== 数据整合 =====
X_all, y_all = [], []

csv_files = glob.glob(os.path.join(well_log_dir, "*.csv"))

for csv_path in csv_files:
    well_name = os.path.basename(csv_path).replace("_around_data.csv", "")
    excel_path = os.path.join(fracture_dir, f"{well_name}.xlsx")

    if not os.path.exists(excel_path):
        print(f"[跳过] 未找到裂缝文件: {excel_path}")
        continue

    try:
        # 读取测井数据
        log_df = pd.read_csv(csv_path)
        if not all(f in log_df.columns for f in features):
            print(f"[跳过] {well_name} 缺少必要特征列")
            continue

        # 确保深度列为数值
        if 'DEPT' in log_df.columns:
            depth_col = 'DEPT'
        elif 'TVD' in log_df.columns:
            depth_col = 'TVD'
        else:
            print(f"[跳过] {well_name} 缺少深度列")
            continue
        log_df[depth_col] = pd.to_numeric(log_df[depth_col], errors='coerce')
        log_df = log_df.dropna(subset=[depth_col])

        # 读取裂缝角度数据（自定义列名）
        frac_df = pd.read_excel(excel_path, header=None, names=['Depth', 'Dip_Angle', 'Dip_Azimuth'])
        frac_df['Depth'] = pd.to_numeric(frac_df['Depth'], errors='coerce')
        frac_df['Dip_Azimuth'] = pd.to_numeric(frac_df['Dip_Azimuth'], errors='coerce')
        frac_df['Dip_Angle'] = pd.to_numeric(frac_df['Dip_Angle'], errors='coerce')
        frac_df = frac_df.dropna()

        for _, frac_row in frac_df.iterrows():
            target_depth = frac_row['Depth']
            nearest_idx = (log_df[depth_col] - target_depth).abs().idxmin()
            feat_values = log_df.loc[nearest_idx, features].values

            # 倾向转 sin/cos
            az_rad = np.deg2rad(frac_row['Dip_Azimuth'] % 360)
            az_sin = np.sin(az_rad)
            az_cos = np.cos(az_rad)
            dip_angle = frac_row['Dip_Angle']

            X_all.append(feat_values)
            y_all.append([az_sin, az_cos, dip_angle])

    except Exception as e:
        print(f"[错误] {well_name} 处理失败: {e}")

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"总样本数: {len(X_all)}, 特征维度: {X_all.shape[1]}")

# ===== 数据集划分 =====
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# ===== 模型训练 =====
base_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model = MultiOutputRegressor(base_model)

print("开始训练模型...")
model.fit(X_train, y_train)

# ===== 验证集评估 =====
y_pred = model.predict(X_val)

# 还原倾向
azimuth_pred = (np.degrees(np.arctan2(y_pred[:, 0], y_pred[:, 1])) + 360) % 360
azimuth_true = (np.degrees(np.arctan2(y_val[:, 0], y_val[:, 1])) + 360) % 360

# 倾向误差（考虑 0-360 环绕）
az_diff = np.abs(azimuth_pred - azimuth_true)
az_diff = np.minimum(az_diff, 360 - az_diff)
mae_azimuth = np.mean(az_diff)

# 倾角误差
mae_angle = mean_absolute_error(y_val[:, 2], y_pred[:, 2])

print(f"验证集 MAE - 倾向(°): {mae_azimuth:.2f}, 倾角(°): {mae_angle:.2f}")

# ========== 倾向-倾角极坐标图 ==========

# 将角度转成弧度
azimuth_true_rad = np.deg2rad(azimuth_true)
azimuth_pred_rad = np.deg2rad(azimuth_pred)

# 倾角作为半径（0°到90°）
radius_true = y_val[:, 2]
radius_pred = y_pred[:, 2]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# 真实值（蓝色）
ax.scatter(azimuth_true_rad, radius_true, c='blue', label='真实', alpha=0.6)

# 预测值（红色）
ax.scatter(azimuth_pred_rad, radius_pred, c='red', marker='x', label='预测', alpha=0.6)

# 设置标签
ax.set_theta_zero_location('N')  # 0°在北方
ax.set_theta_direction(-1)       # 顺时针方向
ax.set_rmax(90)                   # 最大半径 90°
ax.set_rticks([30, 60, 90])       # 半径刻度
ax.set_rlabel_position(135)       # 半径标签位置

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title('裂缝倾向-倾角预测对比', fontsize=14)
plt.tight_layout()
plt.savefig('裂缝倾向-倾角预测对比.png', dpi=300)
plt.show()

# ===== 保存模型 =====
joblib.dump(model, model_save_path)
print(f"模型已保存到 {model_save_path}")
