import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===== 配置 =====
# fracture_csv = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝角度预测/井斜/车页1导眼_predicted_angles.csv"
# fracture_csv = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝角度预测/测井/车22_predicted_angles.csv"
fracture_csv = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝角度预测/测井/车15_predicted_angles.csv"
output_dir = str(Path(__file__).resolve().parents[1] / "_outputs" / "裂缝片模型")
os.makedirs(output_dir, exist_ok=True)

# 裂缝片基础边长 (m)
L0 = 5.0
alpha = 2.0
density = 0.5  # 裂缝密度参数，可调节 0~1

# ===== 函数：生成正方形裂缝片四个顶点 =====
def generate_fracture_square(center, dip_azimuth, dip_angle, L0=5.0, alpha=2.0, density=0.5):
    az = np.radians(dip_azimuth)
    da = np.radians(dip_angle)

    # 法向量 n
    n = np.array([
        np.sin(da) * np.sin(az),
        np.sin(da) * np.cos(az),
        np.cos(da)
    ])
    n /= np.linalg.norm(n)

    # 找一个不与 n 平行的向量
    up = np.array([0, 0, 1])
    if np.allclose(n, up):
        up = np.array([1, 0, 0])

    # 计算裂缝平面两个正交向量 u 和 v
    u = np.cross(n, up)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # 边长计算
    L = L0 * (1 + alpha * density)
    half_L = L / 2

    # 四个顶点
    vertices = []
    for sign_u in [-1, 1]:
        for sign_v in [-1, 1]:
            point = center + sign_u * half_L * u + sign_v * half_L * v
            vertices.append(point)
    return np.array(vertices)

# ===== 读取裂缝预测数据 =====
df = pd.read_csv(fracture_csv)

# ===== 生成裂缝片 =====
fractures = []
for _, row in df.iterrows():
    # 中心点使用 X, Y, TIME
    center = np.array([row['X'], row['Y'], row['TIME']])
    verts = generate_fracture_square(center, row['Dip_Azimuth'], row['Dip_Angle'], L0, alpha, density)
    fractures.append(verts)
# verts = generate_fracture_square([0, 0, 0], 300, 60, L0, alpha, density)
# fractures.append(verts)

# ===== 三维可视化 =====
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 两个三角形顶点相交
# for verts in fractures[:100]:  # 仅显示前100个，避免图像过密
#     poly = Poly3DCollection([verts], alpha=0.5, facecolor='cyan', edgecolor='k')
#     ax.add_collection3d(poly)

# 两个三角形边相交
for verts in fractures[:100]:  # 仅显示前100个，避免图像过密
    # 将正方形分为两个三角形，以一条公共边结合
    # 创建两个三角形，共享边1-2
    triangle1 = [verts[0], verts[1], verts[2]]  # 第一个三角形
    triangle2 = [verts[3], verts[1], verts[2]]  # 第二个三角形，共享边1-2

    # 分别绘制两个三角形
    poly1 = Poly3DCollection([triangle1], alpha=0.5, facecolor='cyan', edgecolor='blue', linewidth=2)
    poly2 = Poly3DCollection([triangle2], alpha=0.5, facecolor='lightblue', edgecolor='red', linewidth=2)

    ax.add_collection3d(poly1)
    ax.add_collection3d(poly2)

    # 标记顶点和边，便于理解
    for i, vertex in enumerate(verts):
        ax.text(vertex[0], vertex[1], vertex[2], f'V{i}', color='black', fontsize=8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('TIME')
ax.set_title('三维裂缝片模型（Z=TIME）')

plt.show()

# ===== 保存裂缝片数据 =====
np.save(os.path.join(output_dir, "fracture_patches_time.npy"), np.array(fractures, dtype=object))
print(f"裂缝片生成完成，共生成 {len(fractures)} 个裂缝片 ✅")
