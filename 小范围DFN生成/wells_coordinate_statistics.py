import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ===== 1. 指定多个井轨迹目录 =====
base_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一"
well_dirs = [
    os.path.join(base_dir, "测井", "测井-地震时窗"),
    os.path.join(base_dir, "成像测井", "测井-地震时窗"),
    os.path.join(base_dir, "井斜", "测井-地震时窗"),
]

# ===== 2. 搜索所有CSV轨迹文件 =====
file_list = []
for directory in well_dirs:
    files = glob.glob(os.path.join(directory, '*.csv'))
    file_list.extend(files)

print(f"共找到 {len(file_list)} 个轨迹文件")
if len(file_list) == 0:
    raise FileNotFoundError("未找到任何轨迹文件，请检查目录设置。")

# ===== 3. 读取并合并所有井轨迹文件 =====
all_data = []

for file in file_list:
    try:
        well_name = os.path.splitext(os.path.basename(file))[0]  # 文件名作为井名
        df = pd.read_csv(file, dtype={'X': np.float32, 'Y': np.float32})

        # 检查必须字段
        if not {'X', 'Y'}.issubset(df.columns):
            raise ValueError(f"文件 {file} 缺少必须列 X 或 Y")

        df['WellName'] = well_name
        all_data.append(df)

    except Exception as e:
        print(f"⚠️ 文件读取失败: {file}, 错误信息: {e}")

# 合并所有井数据
combined_df = pd.concat(all_data, ignore_index=True)
print(f"总井数: {combined_df['WellName'].nunique()}, 总点数: {len(combined_df)}")

# ===== 4. 用户自定义网格参数 =====
# 起始坐标
x_start = 556150  # X坐标起始值
y_start = 4193975  # Y坐标起始值

# 单个网格大小
grid_size_x = 300  # 每个网格在X方向的长度
grid_size_y = 300  # 每个网格在Y方向的长度

# ===== 5. 计算整个范围 =====
x_min, x_max = combined_df['X'].min(), combined_df['X'].max()
y_min, y_max = combined_df['Y'].min(), combined_df['Y'].max()

# 向上取整，保证边界覆盖所有点
x_bins = np.arange(x_start, x_max + grid_size_x, grid_size_x)
y_bins = np.arange(y_start, y_max + grid_size_y, grid_size_y)

print(f"X 范围: {x_min} - {x_max}, 网格数量: {len(x_bins) - 1}")
print(f"Y 范围: {y_min} - {y_max}, 网格数量: {len(y_bins) - 1}")

# ===== 6. 可视化井轨迹与网格 =====
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制网格
for x in x_bins:
    ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=0.7)
for y in y_bins:
    ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.7)

# 绘制井轨迹
well_names = combined_df['WellName'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(well_names)))

for well_name, color in zip(well_names, colors):
    well_df = combined_df[combined_df['WellName'] == well_name]

    ax.plot(
        well_df['X'],
        well_df['Y'],
        marker='o',
        markersize=2,
        linestyle='-',
        color=color,
        label=well_name
    )

# ===== 7. 图形设置 =====
ax.set_xlim(x_start, x_max + grid_size_x)
ax.set_ylim(y_start, y_max + grid_size_y)

# ax.set_title("井轨迹分布与网格划分", fontsize=16)
ax.set_xlabel("X 坐标", fontsize=12)
ax.set_ylabel("Y 坐标", fontsize=12)
ax.grid(False)
ax.axis('equal')

# 图例设置（井多时只显示部分）
# if len(well_names) <= 20:
#     ax.legend(loc='best', fontsize=8)
ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig("井轨迹网格化分布图.png", dpi=300, bbox_inches='tight')
plt.show()

# ===== 8. 生成唯一XY点位集合 =====
unique_points = combined_df[['X', 'Y']].drop_duplicates()
output_path = "唯一井点位集合.csv"
unique_points.to_csv(output_path, index=False)
print(f"唯一井点位集合已保存至: {output_path}")
