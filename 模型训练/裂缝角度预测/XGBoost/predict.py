import os
import glob
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# ===== 配置 =====
model_path = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝角度预测/模型/XGBoost/fracture_angle_model.pkl"
# well_log_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/测井/测井-地震时窗"
well_log_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/井斜/测井-地震时窗"
# fracture_depth_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝位置预测/峰值检测方法/测井"
fracture_depth_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝位置预测/峰值检测方法/井斜"
# output_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝角度预测/测井"
output_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝角度预测/井斜"
os.makedirs(output_dir, exist_ok=True)

# 特征列
features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(63)]

# ===== 加载模型 =====
model = joblib.load(model_path)

# ===== 遍历井 =====
csv_files = glob.glob(os.path.join(well_log_dir, "*.csv"))

for csv_path in csv_files:
    well_name = os.path.basename(csv_path).replace("_around_data.csv", "")
    depth_path = os.path.join(fracture_depth_dir, f"{well_name}_fracture_position.csv")

    if not os.path.exists(depth_path):
        print(f"[跳过] 未找到裂缝深度文件: {depth_path}")
        continue

    try:
        # 读取测井数据
        log_df = pd.read_csv(csv_path)

        # 检查特征列
        if not all(f in log_df.columns for f in features):
            print(f"[跳过] {well_name} 缺少必要特征列")
            continue

        # 确定深度列
        if 'DEPT' in log_df.columns:
            depth_col = 'DEPT'
        elif 'TVD' in log_df.columns:
            depth_col = 'TVD'
        else:
            print(f"[跳过] {well_name} 缺少深度列")
            continue

        # 转换深度列为数值型
        log_df[depth_col] = pd.to_numeric(log_df[depth_col], errors='coerce')
        log_df = log_df.dropna(subset=[depth_col])

        # 检查是否存在 X/Y/TIME 列
        required_columns = ['X', 'Y', 'TIME']
        if not all(col in log_df.columns for col in required_columns):
            print(f"[跳过] {well_name} 缺少 X/Y/TIME 列")
            continue

        # 读取裂缝深度文件
        depth_df = pd.read_csv(depth_path)
        if 'TVD' not in depth_df.columns:
            print(f"[跳过] {well_name} 裂缝文件缺少 TVD 列")
            continue

        depth_df['TVD'] = pd.to_numeric(depth_df['TVD'], errors='coerce')
        depth_df = depth_df.dropna()

        # 对每个裂缝深度进行预测
        pred_records = []
        for depth_val in depth_df['TVD']:
            # 找到最接近的深度索引
            nearest_idx = (log_df[depth_col] - depth_val).abs().idxmin()
            nearest_row = log_df.loc[nearest_idx]

            # 提取特征值
            feat_values = nearest_row[features].values.reshape(1, -1)

            # 模型预测
            pred = model.predict(feat_values)[0]

            # 将 sin/cos 转换为方位角 (Dip_Azimuth)
            azimuth = (np.degrees(np.arctan2(pred[0], pred[1])) + 360) % 360
            dip_angle = pred[2]  # 倾角

            # 记录结果，增加 X/Y/TIME
            pred_records.append([
                depth_val,
                nearest_row['X'],
                nearest_row['Y'],
                nearest_row['TIME'],
                azimuth,
                dip_angle
            ])

        # 转为 DataFrame 并保存
        pred_df = pd.DataFrame(pred_records, columns=['TVD', 'X', 'Y', 'TIME', 'Dip_Azimuth', 'Dip_Angle'])
        pred_df = pred_df.round({'TVD': 2, 'X': 2, 'Y': 2, 'TIME': 3, 'Dip_Azimuth': 2, 'Dip_Angle': 2})

        output_file = os.path.join(output_dir, f"{well_name}_predicted_angles.csv")
        pred_df.to_csv(output_file, index=False)
        print(f"{well_name} 预测完成，裂缝点数: {len(pred_records)}")

    except Exception as e:
        print(f"[错误] {well_name} 处理失败: {e}")

print("所有井裂缝角度预测完成 ✅")
