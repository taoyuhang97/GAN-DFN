import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===== 1. 加载地震道坐标数据 =====
trace_header_csv = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"

df = pd.read_csv(trace_header_csv, usecols=['TraceIdx', 'X', 'Y'], dtype={'TraceIdx': np.int32, 'X': np.int32, 'Y': np.int32})

df.rename(columns={'TraceIdx': 'id', 'X': 'x', 'Y': 'y'}, inplace=True)

# ===== 2. 统计每个 x 坐标下的点位数量 =====
x_count = df.groupby('x').size().reset_index(name='count')
print("每个 X 坐标下的点位数量：")
print(x_count)

# ===== 3. 统计每个 y 坐标下的点位数量 =====
y_count = df.groupby('y').size().reset_index(name='count')
print("\n每个 Y 坐标下的点位数量：")
print(y_count)

# ===== 4. 保存统计结果到 CSV =====
x_count.to_csv("x_count.csv", index=False)
y_count.to_csv("y_count.csv", index=False)
print("\n统计结果已保存为 x_count.csv 和 y_count.csv")

# ===== 5. 可视化统计结果 =====
# ---- X 统计结果 ----
plt.figure(figsize=(10,4))
plt.bar(x_count['x'], x_count['count'], color='blue', alpha=0.7)
plt.title('每个 X 坐标下的点位数量', fontsize=14)
plt.xlabel('X 坐标')
plt.ylabel('点位数量')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---- Y 统计结果 ----
plt.figure(figsize=(10,4))
plt.bar(y_count['y'], y_count['count'], color='green', alpha=0.7)
plt.title('每个 Y 坐标下的点位数量', fontsize=14)
plt.xlabel('Y 坐标')
plt.ylabel('点位数量')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
