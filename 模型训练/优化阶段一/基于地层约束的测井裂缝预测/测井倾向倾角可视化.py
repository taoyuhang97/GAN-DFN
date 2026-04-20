import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取CSV文件
well_name = "车页1导眼"
# 优化前预测
# version = "预测结果"
# file_path = fr'E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝角度预测\井斜\{well_name}_predicted_angles.csv'
# data = pd.read_csv(file_path)
# azimuth_true = data['Dip_Azimuth']
# radius_true = data['Dip_Angle']
# 原始裂缝
version = "原始裂缝"
file_path = fr'E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝提取\{well_name}_fractures.csv'
data = pd.read_csv(file_path)
azimuth_true = data['Azimuth(0~360)']
radius_true = data['Angle(0~90)']
# 第一轮优化
# version = "第一轮优化"
# file_path = fr'E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\两阶段裂缝预测流程结果\现有常规测井裂缝预测\{well_name}\final_fracture_points.csv'
# data = pd.read_csv(file_path)
# azimuth_true = data['PointAzimuth']
# radius_true = data['PointDip']

# 如果存在预测值列，则提取预测值
# azimuth_pred = data['预测倾向']
# radius_pred = data['预测倾角']

# 将角度转成弧度
azimuth_true_rad = np.deg2rad(azimuth_true)
# azimuth_pred_rad = np.deg2rad(azimuth_pred)

# 创建极坐标图
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# 绘制真实值（蓝色散点）
ax.scatter(azimuth_true_rad, radius_true, c='blue', label='裂缝', alpha=0.6)

# 如果存在预测值，则绘制预测值（红色叉号）
# ax.scatter(azimuth_pred_rad, radius_pred, c='red', marker='x', label='预测', alpha=0.6)

# 设置图表参数
ax.set_theta_zero_location('N')  # 0°在北方
ax.set_theta_direction(-1)  # 顺时针方向
ax.set_rmax(90)  # 最大半径90°
ax.set_rticks([30, 60, 90])  # 半径刻度
ax.set_rlabel_position(135)  # 半径标签位置

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
# plt.title('裂缝倾向-倾角分布图', fontsize=14)
plt.tight_layout()
plt.savefig(f'{version}-{well_name}裂缝倾向-倾角分布图.png', dpi=300)
plt.show()