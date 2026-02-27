import os
import pandas as pd

def extract_curve_names_from_las(filepath):
    """从 LAS 文件中提取 ~Curve 区块的属性名（如 AC、GR）"""
    curve_names = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    in_curve_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('~Curve') or line.startswith('~C'):
            in_curve_section = True
            continue
        if in_curve_section:
            if line.startswith('~') or line.lower().startswith('~Parameter'):
                break
            if '.' in line:
                name = line.split('.')[0].strip()
                if name != '':
                    curve_names.append(name)
    return curve_names

# 路径设置
las_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\测井"
all_files = [f for f in os.listdir(las_dir) if f.lower().endswith('.las')]

# 提取每个文件的属性集合
all_curve_sets = {}
all_curve_names = set()

for file in all_files:
    full_path = os.path.join(las_dir, file)
    curve_names = extract_curve_names_from_las(full_path)
    all_curve_sets[file] = set(curve_names)
    all_curve_names.update(curve_names)

# 创建属性存在矩阵
sorted_curves = sorted(all_curve_names)
df = pd.DataFrame(index=all_files, columns=sorted_curves)

for file, curves in all_curve_sets.items():
    for col in sorted_curves:
        df.at[file, col] = 1 if col in curves else 0

# 将缺失值填充为 0，并转换为整数（确保都是 0 或 1）
df = df.fillna(0).infer_objects(copy=False).astype(int)

# 获取除第一列外的所有列名
columns_to_check = df.columns

print("属性名\t总行数\t1的个数")
for col in columns_to_check:
    total_rows = len(df)
    ones_count = df[col].sum()
    print(f"{col}\t{total_rows}\t{ones_count}")

# 显示并保存
print(df)
df.to_excel("测井属性存在情况.xlsx")
