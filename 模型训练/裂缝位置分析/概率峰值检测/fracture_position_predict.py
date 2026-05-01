import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 配置 =====
# predict_output_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/XGboost/裂缝存在预测/井斜"
predict_output_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/XGboost/裂缝存在预测/测井"
method_path_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝位置预测/峰值检测方法"
# output_dir = os.path.join(method_path_dir, "井斜")
output_dir = os.path.join(method_path_dir, "测井")
os.makedirs(output_dir, exist_ok=True)
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# ===== 参数 =====
smooth_sigma = 3           # 高斯平滑参数
min_peak_distance = 1.0    # 峰值间最小距离（米）
prominence_factor = 0.5    # 峰值显著性系数（相对于最大值）
height_factor = 0.4        # 峰值高度系数（相对于最大值）
peak_min_threshold = 0.6   # 峰值概率下限（归一化后），低于此值不记录

# ===== 指定需要额外保留的属性列 =====
extra_columns = ['TIME', 'X', 'Y']  # 你希望输出的其他属性列

# ===== 处理所有文件 =====
csv_files = glob.glob(os.path.join(predict_output_dir, "*_exist_predict.csv"))

for file in csv_files:
    df = pd.read_csv(file)

    # 检查必要列
    required_cols = ['TVD', 'FRACTURE_PROB'] + extra_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        tqdm.write(f"{file} 缺少必要列: {missing_cols}，跳过")
        continue

    # 按深度排序
    df = df.sort_values(by='TVD').reset_index(drop=True)

    depth = df['TVD'].values
    prob = df['FRACTURE_PROB'].values

    # 高斯平滑
    prob_smooth = gaussian_filter1d(prob, sigma=smooth_sigma)

    # 动态阈值
    max_prob = prob_smooth.max()
    height_threshold = max_prob * height_factor
    prominence_threshold = max_prob * prominence_factor

    # 采样间距
    depth_step = float(np.median(np.diff(depth)))
    min_distance_points = max(1, int(min_peak_distance / depth_step))

    # 峰值检测
    peak_indices, _ = find_peaks(prob_smooth,
                                 height=height_threshold,
                                 prominence=prominence_threshold,
                                 distance=min_distance_points)

    peak_depths = depth[peak_indices]
    peak_probs = prob_smooth[peak_indices] / max_prob  # 归一化到 0~1

    # 阈值过滤
    keep_mask = peak_probs >= peak_min_threshold
    kept_indices = peak_indices[keep_mask]  # 真实 DataFrame 索引

    # ===== 保存保留的峰值点及其属性 =====
    kept_df = df.loc[kept_indices, ['TVD', 'FRACTURE_PROB'] + extra_columns].copy()
    kept_df['Peak_Probability'] = np.round(prob_smooth[kept_indices] / max_prob, 3)
    kept_df.insert(0, 'Peak_ID', range(1, len(kept_df) + 1))  # 添加峰值编号

    # 井名
    well_name = os.path.basename(file).split('_')[0]
    output_path = os.path.join(output_dir, f"{well_name}_fracture_position.csv")

    # 保存
    kept_df.to_csv(output_path, index=False)

    # ===== 可视化 =====
    removed_indices = peak_indices[~keep_mask]

    plt.figure(figsize=(8, 5))
    plt.plot(depth, prob, color='gray', alpha=0.4, label='原始概率')
    plt.plot(depth, prob_smooth, color='blue', label='平滑概率')
    plt.scatter(df.loc[kept_indices, 'TVD'], df.loc[kept_indices, 'FRACTURE_PROB'],
                color='red', label='保留的峰值')
    plt.scatter(df.loc[removed_indices, 'TVD'], df.loc[removed_indices, 'FRACTURE_PROB'],
                color='orange', marker='x', label='被过滤峰值')
    plt.axhline(peak_min_threshold * max_prob, color='purple', linestyle='--', label='阈值')
    plt.xlabel('深度 (TVD, m)')
    plt.ylabel('裂缝概率')
    plt.title(f"{well_name} - 裂缝位置预测 (保留峰数={len(kept_df)})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{well_name}_peaks.png"), dpi=300)
    plt.close()

    tqdm.write(f"{well_name} 峰值检测完成，保留: {len(kept_df)} / 总检测: {len(peak_indices)}")

print("所有井的峰值检测完成 ✅")
