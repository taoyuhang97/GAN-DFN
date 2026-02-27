import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv("trace_header_xy.csv")

# 统计X坐标值与数量
x_counts = df['X'].value_counts().reset_index()
x_counts.columns = ['X坐标值', '出现次数']
x_counts = x_counts.sort_values('X坐标值')  # 按坐标值排序

# 统计Y坐标值与数量
y_counts = df['Y'].value_counts().reset_index()
y_counts.columns = ['Y坐标值', '出现次数']
y_counts = y_counts.sort_values('Y坐标值')  # 按坐标值排序

# 保存为Excel文件
outdir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一"
x_output_file = os.path.join(outdir, "X坐标统计.xlsx")
y_output_file = os.path.join(outdir, "Y坐标统计.xlsx")

x_counts.to_excel(x_output_file, index=False)
y_counts.to_excel(y_output_file, index=False)

print("坐标统计结果已保存:")
print(f"X坐标统计: {x_output_file}")
print(f"  - 包含 {len(x_counts)} 个不同的X坐标值")
print(f"Y坐标统计: {y_output_file}")
print(f"  - 包含 {len(y_counts)} 个不同的Y坐标值")

# 显示统计摘要
print(f"\n统计摘要:")
print(f"X坐标范围: {x_counts['X坐标值'].min()} ~ {x_counts['X坐标值'].max()}")
print(f"Y坐标范围: {y_counts['Y坐标值'].min()} ~ {y_counts['Y坐标值'].max()}")
print(f"平均每个X坐标的道数: {df['X'].value_counts().mean():.2f}")
print(f"平均每个Y坐标的道数: {df['Y'].value_counts().mean():.2f}")