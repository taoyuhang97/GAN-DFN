import os
import glob
import numpy as np
import pandas as pd
import pywt  # 小波变换库
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback  # 用于异常堆栈信息

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 配置 =====
# data_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/井斜/测井-地震时窗"
# data_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/测井/测井-地震时窗"
data_dir = r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝样本"
method_path_dir = r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝位置预测/小波变换方法"
# output_dir = os.path.join(method_path_dir, "井斜")
# output_dir = os.path.join(method_path_dir, "测井")
output_dir = os.path.join(method_path_dir, "成像测井预测")
os.makedirs(output_dir, exist_ok=True)
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

log_file_path = "error_log.txt"

# 小波参数
wavelet_name = 'sym4'   # 小波类型
decomp_level = 2        # 分解层数
threshold_factor = 2    # 阈值因子

selected_logs = ['AC', 'DEN', 'GR', 'SP']

def wavelet_anomaly_detection(signal, wavelet, level, thresh_factor):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    detail_coeff = coeffs[1]
    thresh = np.mean(np.abs(detail_coeff)) + thresh_factor * np.std(detail_coeff)
    anomaly_idx = np.where(np.abs(detail_coeff) > thresh)[0]
    factor = len(signal) / len(detail_coeff)
    mapped_idx = (anomaly_idx * factor).astype(int)
    mapped_idx = mapped_idx[mapped_idx < len(signal)]
    return mapped_idx

with open(log_file_path, 'w', encoding='utf-8') as log_f:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    for file in csv_files:
        well_name = os.path.basename(file).split('_')[0]
        try:
            df = pd.read_csv(file)

            if not all(col in df.columns for col in selected_logs + ['TVD']):
                msg = f"{well_name} 缺少必要曲线，跳过\n"
                tqdm.write(msg.strip())
                log_f.write(msg)
                continue

            df = df.sort_values(by='TVD').reset_index(drop=True)
            depth = df['TVD'].values

            anomaly_points_all = []
            for log in selected_logs:
                signal = df[log].values
                anomalies = wavelet_anomaly_detection(signal, wavelet_name, decomp_level, threshold_factor)
                anomaly_points_all.append(set(anomalies))

            # 多曲线异常点合并
            common_anomalies = set.union(*anomaly_points_all)
            if len(common_anomalies) == 0:
                fracture_indices = np.array([], dtype=int)
            else:
                fracture_indices = np.array(sorted(common_anomalies), dtype=int)

            fracture_depths = depth[fracture_indices]

            result_df = pd.DataFrame({
                'X': np.round(df.loc[fracture_indices, 'X'].values),
                'Y': np.round(df.loc[fracture_indices, 'Y'].values),
                'TIME': np.round(df.loc[fracture_indices, 'TIME'].values),
                'TVD': np.round(fracture_depths, 3)
            })
            result_df.to_csv(os.path.join(output_dir, f"{well_name}_fracture_position.csv"), index=False)

            plt.figure(figsize=(10, 6))
            for log in selected_logs:
                plt.plot(depth, df[log], label=log)
            plt.scatter(fracture_depths, df.loc[list(common_anomalies), selected_logs[0]],
                        color='red', label='检测裂缝点', zorder=5)
            plt.xlabel('深度 (TVD)')
            plt.ylabel('测井值')
            plt.title(f"{well_name} - 小波裂缝检测")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{well_name}_wavelet_detection.png"), dpi=300)
            plt.close()

            tqdm.write(f"{well_name} 裂缝检测完成，检测点数: {len(fracture_depths)}")

        except Exception as e:
            error_msg = f"{well_name} 处理失败，错误信息:\n{traceback.format_exc()}\n"
            tqdm.write(error_msg)
            log_f.write(error_msg)

print("所有井小波裂缝检测完成 ✅")
