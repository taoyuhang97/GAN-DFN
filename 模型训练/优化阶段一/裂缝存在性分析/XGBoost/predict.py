import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging

# ========== 配置路径 ==========
base_model_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\XGboost\模型"
model_path = os.path.join(base_model_path, r"fracture_xgb_model.pkl")
scaler_path = os.path.join(base_model_path, r"fracture_scaler.pkl")
feature_path = os.path.join(base_model_path, r"fracture_features.pkl")

# predict_input_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\测井\测井-地震时窗"   # 需要预测的文件夹
# predict_output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\XGboost\裂缝存在预测\测井"    # 输出结果的目录
# predict_input_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\井斜\测井-地震时窗"   # 需要预测的文件夹
# predict_output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\XGboost\裂缝存在预测\井斜"    # 输出结果的目录
predict_output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\成像测井预测"

log_file = os.path.join(predict_output_dir, 'prediction_log.txt')
os.makedirs(predict_output_dir, exist_ok=True)
# ========== 设置日志 ==========
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========== 加载模型和标准化器 ==========
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ========== 定义特征 ==========
features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(63)] + ['AC', 'GR']
# features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(3, 63, 7)] + ['AC', 'GR']
# features = ['SEIS_TRUE'] + ['AC', 'GR']

# ========== 遍历所有文件 ==========
# csv_files = glob.glob(os.path.join(predict_input_dir, "*.csv"))
csv_files = [r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本\车页1导眼_sample.csv"]

for file in csv_files:
    try:
        print(f"正在处理 {os.path.basename(file).split('_')[0]} 井")
        df = pd.read_csv(file)

        # 检查是否缺少必要特征
        missing = [col for col in features if col not in df.columns]
        if missing:
            logging.warning(f"{os.path.basename(file)} 缺少特征列：{missing}，已跳过。")
            continue

        # 丢弃含缺失值的行
        clean_df = df[features].dropna()
        if clean_df.empty:
            logging.warning(f"{os.path.basename(file)} 数据行全为空，已跳过。")
            continue

        # 标准化
        X_scaled = scaler.transform(clean_df)

        # 预测
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]  # 可选：概率

        # 添加结果回原始行（需对齐索引）
        result_df = df.loc[clean_df.index].copy()
        result_df['FRACTURE_FLAG_PRED'] = y_pred
        result_df['FRACTURE_PROB'] = y_prob

        # 保存结果
        output_file = os.path.join(predict_output_dir, os.path.basename(file).split('_')[0] + '_exist_predict.csv')
        result_df.to_csv(output_file, index=False)

        logging.info(f"{os.path.basename(file)} 预测完成，结果保存至：{output_file}")

    except Exception as e:
        logging.error(f"处理文件 {os.path.basename(file)} 时出错：{str(e)}")
