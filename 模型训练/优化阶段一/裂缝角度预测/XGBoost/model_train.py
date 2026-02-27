"""
Stage-2 模型：裂缝产状（倾向 + 倾角）回归
前提：裂缝存在性已确认
"""
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
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ===== 配置 =====
logging_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝角度预测\XGBoost\模型"
os.makedirs(logging_dir, exist_ok=True)

# well_log_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\测井-地震时窗"
# fracture_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\裂缝标注"
model_save_path = os.path.join(logging_dir, "fracture_angle_model.pkl")

# 特征选择
features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(63)] + ['AC', 'GR']
# features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(3, 63, 7)] + ['AC', 'GR']

# ===== 数据整合 =====
X_all, y_all = [], []

# csv_files = glob.glob(os.path.join(well_log_dir, "*.csv"))
csv_files = [
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本\车151HF_sample.csv"]

for csv_path in csv_files:
    well_name = os.path.basename(csv_path).replace("_around_data.csv", "")
    # excel_path = os.path.join(fracture_dir, f"{well_name}.xlsx")

    # if not os.path.exists(excel_path):
    #     print(f"[跳过] 未找到裂缝文件: {excel_path}")
    #     continue

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
        # frac_df = pd.read_excel(excel_path, header=None, names=['Depth', 'Dip_Angle', 'Dip_Azimuth'])
        frac_df = pd.read_csv(csv_path)
        frac_df['Depth'] = pd.to_numeric(frac_df[depth_col], errors='coerce')
        # frac_df['Dip_Azimuth'] = pd.to_numeric(frac_df['Dip_Azimuth'], errors='coerce')
        # frac_df['Dip_Angle'] = pd.to_numeric(frac_df['Dip_Angle'], errors='coerce')
        frac_df['Dip_Azimuth'] = pd.to_numeric(frac_df['Frac_Azimuth'], errors='coerce')
        frac_df['Dip_Angle'] = pd.to_numeric(frac_df['Frac_Dip'], errors='coerce')

        # 裂缝密度相关信息
        frac_df['P10'] = pd.to_numeric(frac_df['P10'], errors='coerce')
        frac_df['P21'] = pd.to_numeric(frac_df['P21'], errors='coerce')
        frac_df['P33'] = pd.to_numeric(frac_df['P33'], errors='coerce')

        # Stage-2 仅使用存在裂缝且有明确产状标注的点
        frac_df = frac_df[
            frac_df['Dip_Azimuth'].notna() &
            frac_df['Dip_Angle'].notna() &
            frac_df['P10'].notna() &
            frac_df['P21'].notna() &
            frac_df['P33'].notna()
            ]

        for _, frac_row in frac_df.iterrows():
            # target_depth = frac_row['Depth']
            # nearest_idx = (log_df[depth_col] - target_depth).abs().idxmin()
            # feat_values = log_df.loc[nearest_idx, features].values
            feat_values = frac_row[features].values

            # 倾向转 sin/cos
            az_rad = np.deg2rad(frac_row['Dip_Azimuth'] % 360)
            az_sin = np.sin(az_rad)
            az_cos = np.cos(az_rad)
            dip_angle_norm = frac_row['Dip_Angle'] / 90.0

            X_all.append(feat_values)
            y_all.append([
                az_sin,
                az_cos,
                dip_angle_norm,
                frac_row['P10'],
                frac_row['P21'],
                frac_row['P33']
            ])

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

# ---------- 倾向还原 ----------
azimuth_pred = (np.degrees(np.arctan2(y_pred[:, 0], y_pred[:, 1])) + 360) % 360
azimuth_true = (np.degrees(np.arctan2(y_val[:, 0], y_val[:, 1])) + 360) % 360

az_diff = np.abs(azimuth_pred - azimuth_true)
az_diff = np.minimum(az_diff, 360 - az_diff)
mae_azimuth = np.mean(az_diff)

# ---------- 倾角还原（关键修改） ----------
dip_pred = y_pred[:, 2] * 90.0
dip_true = y_val[:, 2] * 90.0
mae_angle = mean_absolute_error(dip_true, dip_pred)

mae_p10 = mean_absolute_error(y_val[:, 3], y_pred[:, 3])
mae_p21 = mean_absolute_error(y_val[:, 4], y_pred[:, 4])
mae_p33 = mean_absolute_error(y_val[:, 5], y_pred[:, 5])

print(
    f"验证集 MAE | "
    f"倾向(°): {mae_azimuth:.2f}, "
    f"倾角(°): {mae_angle:.2f}, "
    f"P10: {mae_p10:.4f}, "
    f"P21: {mae_p21:.4f}, "
    f"P33: {mae_p33:.4f}"
)


# ========== 倾向-倾角极坐标图 ==========

# 将角度转成弧度
azimuth_true_rad = np.deg2rad(azimuth_true)
azimuth_pred_rad = np.deg2rad(azimuth_pred)

# 倾角作为半径（0°到90°）
radius_true = y_val[:, 2] * 90.0
radius_pred = y_pred[:, 2] * 90.0

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# 真实值（蓝色）
ax.scatter(azimuth_true_rad, radius_true, c='blue', label='真实', alpha=0.6)

# 预测值（红色）
ax.scatter(azimuth_pred_rad, radius_pred, c='red', marker='x', label='预测', alpha=0.6)

# 设置标签
ax.set_theta_zero_location('N')  # 0°在北方
ax.set_theta_direction(-1)  # 顺时针方向
ax.set_rmax(90)  # 最大半径 90°
ax.set_rticks([30, 60, 90])  # 半径刻度
ax.set_rlabel_position(135)  # 半径标签位置

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title('裂缝倾向-倾角预测对比', fontsize=14)
plt.tight_layout()
plt.savefig('裂缝倾向-倾角预测对比.png', dpi=300)
plt.show()

# ===== 保存模型 =====
joblib.dump({
    "model": model,
    "features": features,
    "dip_angle_norm": True,
    "outputs": [
        "sin_azimuth",
        "cos_azimuth",
        "dip_angle_norm",
        "P10",
        "P21",
        "P33"
    ]
}, model_save_path)

print(f"模型已保存到 {model_save_path}")
