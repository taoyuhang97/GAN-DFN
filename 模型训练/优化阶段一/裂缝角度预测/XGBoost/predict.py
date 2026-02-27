import os
import glob
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# ===== 配置 =====
model_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝角度预测\XGBoost\模型\fracture_angle_model.pkl"
# well_log_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\测井\测井-地震时窗"
# well_log_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\井斜\测井-地震时窗"
# fracture_depth_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝位置预测\峰值检测方法\测井"
# fracture_depth_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝位置预测\峰值检测方法\井斜"
fracture_depth_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝位置预测\峰值检测方法\成像测井预测"
# output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝角度预测\测井"
output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝角度预测\成像测井预测"
os.makedirs(output_dir, exist_ok=True)

# ===== 加载模型 =====
model_dict = joblib.load(model_path)
if not isinstance(model_dict, dict) or 'model' not in model_dict:
    raise ValueError("模型文件格式错误，未找到 'model' 键")
model = model_dict['model']
features = model_dict['features']

# ===== 遍历井 =====
# csv_files = glob.glob(os.path.join(well_log_dir, "*.csv"))
csv_files = [
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本\车页1导眼_sample.csv"]

for csv_path in csv_files:
    well_name = os.path.basename(csv_path).replace("_sample.csv", "")
    depth_path = os.path.join(fracture_depth_dir, f"{well_name}_fracture_position.csv")

    if not os.path.exists(depth_path):
        print(f"[跳过] 未找到裂缝深度文件: {depth_path}")
        continue

    try:
        # 读取测井数据
        log_df = pd.read_csv(csv_path)

        # 检查特征列
        if not all(f in log_df.columns for f in features):
            missing = [f for f in features if f not in log_df.columns]
            print(f"[跳过] {well_name} 缺少特征列: {missing}")
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

            # --- 裂缝几何参数 ---
            azimuth = (np.degrees(np.arctan2(pred[0], pred[1])) + 360) % 360
            dip_angle = pred[2] * 90.0  # 倾角 (0–90)

            # --- 裂缝强度参数 ---
            P10 = np.expm1(pred[3])
            P21 = np.expm1(pred[4])
            P33 = np.expm1(pred[5])

            # 记录结果，增加 X/Y/TIME
            pred_records.append([
                depth_val,
                nearest_row['X'],
                nearest_row['Y'],
                nearest_row['TIME'],
                azimuth,
                dip_angle,
                P10,
                P21,
                P33
            ])

        # 转为 DataFrame 并保存
        pred_df = pd.DataFrame(
            pred_records,
            columns=[
                'TVD', 'X', 'Y', 'TIME',
                'Dip_Azimuth', 'Dip_Angle',
                'P10', 'P21', 'P33'
            ]
        )
        pred_df = pred_df.round({
            'TVD': 2,
            'X': 2,
            'Y': 2,
            'TIME': 3,
            'Dip_Azimuth': 2,
            'Dip_Angle': 2,
            'P10': 4,
            'P21': 4,
            'P33': 4
        })

        output_file = os.path.join(output_dir, f"{well_name}_predicted_angles.csv")
        pred_df.to_csv(output_file, index=False)
        print(f"{well_name} 预测完成，裂缝点数: {len(pred_records)}")

    except Exception as e:
        print(f"[错误] {well_name} 处理失败: {e}")

print("所有井裂缝角度预测完成 ✅")
