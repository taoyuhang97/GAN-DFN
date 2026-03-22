import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# ===== 配置 =====
# predict_output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\XGboost\裂缝存在预测\测井"
# predict_output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\XGboost\裂缝存在预测\井斜"
predict_output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\成像测井预测"
method_path_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一"
# cluster_output_dir = os.path.join(method_path_dir, "裂缝位置预测", "聚类方法", "测井")
# cluster_output_dir = os.path.join(method_path_dir, "裂缝位置预测", "聚类方法", "井斜")
cluster_output_dir = os.path.join(method_path_dir, "裂缝位置预测", "聚类方法", "成像测井预测")
os.makedirs(cluster_output_dir, exist_ok=True)

prob_threshold = 0.5  # 裂缝概率阈值
eps_val = 0.5         # DBSCAN 聚类半径（单位与 TVD 一致，比如米）
min_samples_val = 4   # 一个簇的最小样本点数

# ===== 批量处理 =====
csv_files = glob.glob(os.path.join(predict_output_dir, "*_exist_predict.csv"))

def get_center_depth(depth_series):
    """
    返回中位数深度：
    - 奇数个值 -> 取真正的中位数
    - 偶数个值 -> 取靠上的那个（较小的那个）
    """
    sorted_depths = sorted(depth_series)
    n = len(sorted_depths)
    mid = n // 2
    if n % 2 == 1:
        return sorted_depths[mid]  # 奇数
    else:
        return sorted_depths[mid - 1]  # 偶数取靠上的

for file in csv_files:
    df = pd.read_csv(file)

    if 'TVD' not in df.columns:
        tqdm.write(f"{file} 缺少 TVD 列，跳过")
        continue

    # 1. 过滤高概率点
    high_prob_df = df[df['FRACTURE_PROB'] >= prob_threshold].copy()
    if high_prob_df.empty:
        tqdm.write(f"{file} 没有高概率裂缝点，跳过")
        continue

    # 2. 准备深度数据（reshape 成二维数组以便 DBSCAN）
    depth_values = high_prob_df['TVD'].values.reshape(-1, 1)

    # 3. DBSCAN 聚类
    db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    cluster_labels = db.fit_predict(depth_values)

    # -1 表示噪声点
    high_prob_df['CLUSTER_ID'] = cluster_labels

    # 4. 合并回原始数据
    df['CLUSTER_ID'] = np.nan
    df.loc[high_prob_df.index, 'CLUSTER_ID'] = high_prob_df['CLUSTER_ID']

    # 5. 统计每个簇信息（加中心点位）
    cluster_summary = []
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue
        cluster_points = high_prob_df[high_prob_df['CLUSTER_ID'] == cluster_id]
        start_depth = cluster_points['TVD'].min()
        end_depth = cluster_points['TVD'].max()
        center_depth = get_center_depth(cluster_points['TVD'])

        # 找到离中心深度最近的行
        nearest_idx = (df['TVD'] - center_depth).abs().idxmin()
        center_row = df.loc[nearest_idx]

        cluster_summary.append({
            'Cluster_ID': cluster_id,
            'Start_Depth': round(start_depth, 3),
            'End_Depth': round(end_depth, 3),
            'TVD': round(center_depth, 3),
            'X': round(center_row['X'], 2),
            'Y': round(center_row['Y'], 2),
            'TIME': round(center_row['TIME'], 3),
            'Num_Points': len(cluster_points)
        })

    summary_df = pd.DataFrame(cluster_summary).round(3)

    # 6. 保存结果
    well_name = os.path.basename(file).split('_')[0]
    # df.to_csv(os.path.join(cluster_output_dir, f"{well_name}_clustered.csv"), index=False)
    summary_df.to_csv(os.path.join(cluster_output_dir, f"{well_name}_fracture_position.csv"), index=False)

    tqdm.write(f"{well_name} 聚类完成，簇数: {len(summary_df)}")

print("所有文件聚类完成 ✅")
