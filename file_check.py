import os
import pandas as pd
import matplotlib.pyplot as plt

# 文件过滤函数（去掉 .cfg）
def file_filter(f):
    return not f.lower().endswith('.cfg')

# 提取文件主名（去掉扩展名）
def get_basename(filename):
    return os.path.splitext(filename)[0]  # "车15.las" → "车15"

# 主目录路径
data_path = r"/data/shared/project-oil/wx数据/砂砾岩"

# 要处理的子目录
subdirs = ['测井', '井斜', '时深']

# 存储所有主名，以及每个文件夹下的主名集合
all_names = set()
folder_names = {}

for folder in subdirs:
    folder_path = os.path.join(data_path, folder)
    try:
        # 获取主名
        files = list(filter(file_filter, os.listdir(folder_path)))
        base_names = [get_basename(f) for f in files]
        folder_names[folder] = base_names
        all_names.update(base_names)
    except FileNotFoundError:
        folder_names[folder] = []
        print(f"⚠️ 路径不存在: {folder_path}")

# 统一排序后的主名列表（列）
all_names = sorted(all_names)

# 构建 presence 表格（行=文件夹，列=主文件名）
df = pd.DataFrame(index=subdirs, columns=all_names)

for folder in subdirs:
    for name in all_names:
        df.at[folder, name] = 1 if name in folder_names[folder] else 0

# 检查哪些列全为1（即所有目录中该井文件都存在）
all_exist_columns = df.columns[(df == 1).all(axis=0)]

# 打印结果
if len(all_exist_columns) > 0:
    print("✅ 以下列在所有目录中均存在：")
    for col in all_exist_columns:
        print(f"  - {col}")
else:
    print("⚠️ 没有任何列在所有目录中都存在。")

# 显示或保存
print(df)
df.to_excel("不同目录中测井存在情况.xlsx")

# 设置中文字体（如果有中文井名，推荐使用 SimHei 或任意你本机可用的字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图像并绘制表格
fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.6), len(df) * 0.6))  # 自适应大小
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values,
                 rowLabels=df.index,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center')

# 可选：自动调整字体大小（避免过大或重叠）
table.auto_set_font_size(False)
table.set_fontsize(10)

# 输出路径
output_menu = os.path.join(data_path, "研究内容一")
# 自动创建父级目录（如果不存在）
os.makedirs(output_menu, exist_ok=True)
output_path = os.path.join(output_menu, "不同目录中测井存在情况.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 表格已保存为图片：{output_path}")
