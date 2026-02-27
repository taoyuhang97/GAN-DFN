import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import DBSCAN

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ===== 函数：读取CSV格式地震数据 =====
def load_seismic_data_from_csv(seismic_csv):
    """
    从CSV文件读取地震数据
    CSV格式: 每行代表一个地震道，前三列为traceId, x, y，后面2701列为振幅数据
    """
    print(f"正在从CSV文件 {seismic_csv} 读取地震数据...")
    try:
        # 读取CSV文件
        df = pd.read_csv(seismic_csv)

        # 提取坐标信息
        trace_ids = df.iloc[:, 0].values  # 第一列: traceId
        x_coords = df.iloc[:, 1].values  # 第二列: x坐标
        y_coords = df.iloc[:, 2].values  # 第三列: y坐标

        # 提取振幅数据 (第4列到最后一列)
        amplitude_data = df.iloc[:, 3:].values

        print(f"地震数据加载成功: {amplitude_data.shape}")
        print(f"地震道数量: {len(trace_ids)}")
        print(f"每道采样点数: {amplitude_data.shape[1]}")
        print(f"X坐标范围: {x_coords.min():.2f} - {x_coords.max():.2f}")
        print(f"Y坐标范围: {y_coords.min():.2f} - {y_coords.max():.2f}")

        # 创建时间轴
        z_coords = np.arange(start_time, start_time + num_samples * sample_interval, sample_interval)

        # 将地震数据重构为三维体
        # 由于CSV数据是离散的，我们需要创建规则网格并插值
        seismic_3d, x_grid, y_grid, z_grid = create_3d_seismic_from_points(
            x_coords, y_coords, z_coords, amplitude_data
        )

        return seismic_3d, x_grid, y_grid, z_grid

    except Exception as e:
        print(f"地震数据加载失败: {e}")
        # 创建模拟地震数据用于演示
        return create_synthetic_seismic_data()


def create_3d_seismic_from_points(x_coords, y_coords, z_coords, amplitude_data):
    """
    将离散的地震道数据插值为规则的三维网格
    """
    print("将离散地震道数据插值为规则三维网格...")

    # 创建规则网格
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    z_unique = z_coords  # 时间轴已经是规则的

    # 如果坐标点太少，使用更粗的网格
    if len(x_unique) < 10:
        x_grid = np.linspace(x_coords.min(), x_coords.max(), 50)
    else:
        x_grid = x_unique

    if len(y_unique) < 10:
        y_grid = np.linspace(y_coords.min(), y_coords.max(), 50)
    else:
        y_grid = y_unique

    z_grid = z_unique

    # 创建三维网格
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # 初始化三维数据体
    seismic_3d = np.zeros((len(x_grid), len(y_grid), len(z_grid)))

    # 使用最近邻插值将离散点映射到网格
    from scipy.spatial import cKDTree

    # 创建坐标点数组
    points = np.column_stack([x_coords, y_coords])

    # 创建网格点数组
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # 使用KD树进行最近邻查找
    tree = cKDTree(points)
    distances, indices = tree.query(grid_points, k=1)

    # 将振幅数据映射到网格
    for i, idx in enumerate(indices):
        if distances[i] < max(x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]) * 2:  # 只在合理距离内插值
            x_idx = i // (len(y_grid) * len(z_grid))
            temp = i % (len(y_grid) * len(z_grid))
            y_idx = temp // len(z_grid)
            z_idx = temp % len(z_grid)

            # 复制整个时间序列
            seismic_3d[x_idx, y_idx, :] = amplitude_data[idx, :]

    print(f"三维地震数据体形状: {seismic_3d.shape}")
    return seismic_3d, x_grid, y_grid, z_grid


def create_synthetic_seismic_data():
    """
    创建合成地震数据用于演示
    """
    print("创建合成地震数据用于演示...")
    x = np.linspace(0, 1000, 100)
    y = np.linspace(0, 1000, 100)
    z = np.arange(start_time, start_time + num_samples * sample_interval, sample_interval)

    # 创建网格
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 创建合成地震数据（包含一些随机异常，模拟裂缝特征）
    seismic_data = np.random.normal(0, 0.1, (100, 100, len(z)))

    # 添加一些线性特征模拟裂缝
    for i in range(10):
        center_x, center_y = np.random.uniform(200, 800, 2)
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(100, 300)

        for t in np.linspace(0, length, 50):
            px = center_x + t * np.cos(angle)
            py = center_y + t * np.sin(angle)
            if 0 <= int(px) < 100 and 0 <= int(py) < 100:
                # 在裂缝位置添加异常值
                z_start = np.random.randint(500, 1000)  # 在1100-3800ms范围内
                z_end = min(z_start + np.random.randint(20, 50), len(z) - 1)
                seismic_data[int(px), int(py), z_start:z_end] += 0.5

    return seismic_data, x, y, z


# ===== 函数：基于地震属性识别潜在裂缝区域 =====
def detect_fracture_zones(seismic_data, threshold=0.3):
    """
    使用地震属性识别潜在裂缝区域
    """
    # 计算地震属性（这里使用简单的梯度作为示例）
    gradient_x = np.gradient(seismic_data, axis=0)
    gradient_y = np.gradient(seismic_data, axis=1)
    gradient_z = np.gradient(seismic_data, axis=2)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)

    # 标准化
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / \
                         (gradient_magnitude.max() - gradient_magnitude.min())

    # 识别高梯度区域（可能指示裂缝）
    fracture_mask = gradient_magnitude > threshold

    # 获取潜在裂缝点的坐标
    fracture_points = np.argwhere(fracture_mask)

    return fracture_points, gradient_magnitude


# ===== 函数：生成正方形裂缝片 =====
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


# ===== 函数：基于井点裂缝和地震数据拓展裂缝分布 =====
def expand_fractures_from_well(well_fractures, seismic_data, x_coords, y_coords, z_coords,
                               expansion_radius=100, seismic_threshold=0.3):
    """
    基于井点裂缝和地震数据拓展裂缝分布
    """
    all_fractures = []

    # 检测地震数据中的潜在裂缝区域
    fracture_points, gradient_magnitude = detect_fracture_zones(seismic_data, seismic_threshold)

    # 创建地震数据插值器
    interpolator = RegularGridInterpolator(
        (x_coords, y_coords, z_coords),
        gradient_magnitude,
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    for well_fracture in well_fractures:
        # 获取井点裂缝的中心和属性
        center = well_fracture['center']
        dip_azimuth = well_fracture['dip_azimuth']
        dip_angle = well_fracture['dip_angle']
        density = well_fracture['density']

        # 1. 首先添加井点裂缝本身
        well_fracture_patch = generate_fracture_square(
            center, dip_azimuth, dip_angle, L0, alpha, density
        )
        all_fractures.append({
            'vertices': well_fracture_patch,
            'type': 'well',
            'center': center
        })

        # 2. 在扩展半径内搜索潜在裂缝点
        x_min, x_max = center[0] - expansion_radius, center[0] + expansion_radius
        y_min, y_max = center[1] - expansion_radius, center[1] + expansion_radius
        z_min, z_max = center[2] - expansion_radius, center[2] + expansion_radius

        # 筛选在扩展半径内的潜在裂缝点
        nearby_points = []
        for point in fracture_points:
            x, y, z = x_coords[point[0]], y_coords[point[1]], z_coords[point[2]]
            if (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                if distance <= expansion_radius:
                    # 计算该点的地震属性强度
                    intensity = interpolator([x, y, z])[0]
                    nearby_points.append({
                        'point': np.array([x, y, z]),
                        'intensity': intensity,
                        'distance': distance
                    })

        # 3. 根据地震属性强度对潜在裂缝点进行聚类和筛选
        if nearby_points:
            points_array = np.array([p['point'] for p in nearby_points])
            intensities = np.array([p['intensity'] for p in nearby_points])

            # 使用DBSCAN聚类识别裂缝带
            clustering = DBSCAN(eps=50, min_samples=3).fit(points_array)
            labels = clustering.labels_

            # 为每个聚类生成裂缝片
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # 噪声点，跳过
                    continue

                cluster_points = points_array[labels == label]
                cluster_intensities = intensities[labels == label]

                # 计算聚类中心
                cluster_center = np.mean(cluster_points, axis=0)

                # 根据地震属性强度调整裂缝密度
                avg_intensity = np.mean(cluster_intensities)
                adjusted_density = density * (0.5 + 0.5 * avg_intensity)

                # 生成扩展裂缝片（使用与井点裂缝相似的方位角和倾角）
                expanded_fracture = generate_fracture_square(
                    cluster_center, dip_azimuth, dip_angle, L0, alpha, adjusted_density
                )

                all_fractures.append({
                    'vertices': expanded_fracture,
                    'type': 'expanded',
                    'center': cluster_center,
                    'intensity': avg_intensity
                })

    return all_fractures


def save_all_fractures_single_file(all_fractures, out_dir):
    """
    将一个单元内的所有裂缝片保存到一个文件
    """
    fracture_pyvista_list = []

    for fracture in all_fractures:
        vertices = fracture['vertices'].astype(np.float32)

        # 生成三角面索引
        if vertices.shape[0] == 4:
            faces = np.array([[0, 1, 2],
                              [0, 2, 3]], dtype=np.int32)
        else:
            faces = np.array([[0, 1, 2]], dtype=np.int32)

        fracture_pyvista_list.append({
            'points': vertices,
            'faces': faces,
            'type': fracture['type'],
            'center': fracture['center'],
            'intensity': fracture.get('intensity', 0)
        })

    # 保存所有裂缝片到单个 .npy 文件
    output_file = os.path.join(out_dir, "block_X32_Y24_expanded_fracture_patches.npy")
    np.save(output_file, np.array(fracture_pyvista_list, dtype=object))
    print(f"[INFO] 保存完成: {output_file}")

    # 同时保存属性信息到 CSV
    fracture_info = []
    for i, f in enumerate(all_fractures):
        fracture_info.append({
            'id': i,
            'type': f['type'],
            'center_x': f['center'][0],
            'center_y': f['center'][1],
            'center_z': f['center'][2],
            'intensity': f.get('intensity', 0)
        })
    df_info = pd.DataFrame(fracture_info)
    df_info.to_csv(os.path.join(out_dir, "block_X32_Y24_expanded_fracture_patches.csv"), index=False)
    print(f"[INFO] 属性信息 CSV 保存完成")


def main_single_unit():
    # 1. 读取井点裂缝数据
    print("读取井点裂缝数据...")
    df = pd.read_csv(fracture_csv)

    # 准备井点裂缝数据
    well_fractures = []
    for _, row in df.iterrows():
        well_fractures.append({
            'center': np.array([row['X'], row['Y'], row['TIME']]),
            'dip_azimuth': row['Dip_Azimuth'],
            'dip_angle': row['Dip_Angle'],
            'density': density
        })

    # 2. 读取地震数据
    print("读取地震数据...")
    seismic_data, x_coords, y_coords, z_coords = load_seismic_data_from_csv(seismic_csv)

    # 3. 基于地震数据拓展裂缝分布
    print("基于地震数据拓展裂缝分布...")
    all_fractures = expand_fractures_from_well(
        well_fractures, seismic_data, x_coords, y_coords, z_coords,
        expansion_radius, seismic_threshold
    )

    # 4. 可视化结果
    print("生成可视化...")
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    for fracture in all_fractures:
        if fracture['type'] == 'well':
            poly = Poly3DCollection([fracture['vertices']], alpha=0.7,
                                    facecolor='red', edgecolor='darkred', linewidth=2)
        else:
            intensity = fracture.get('intensity', 0.5)
            color = plt.cm.viridis(intensity)
            poly = Poly3DCollection([fracture['vertices']], alpha=0.3,
                                    facecolor=color, edgecolor='blue', linewidth=1)
        ax.add_collection3d(poly)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('TIME (ms)')
    ax.set_title(f'基于地震数据的裂缝分布模型\n井点裂缝: {len(well_fractures)}, 总裂缝片: {len(all_fractures)}')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('地震属性强度')
    plt.tight_layout()
    plt.show()

    # 5. 保存所有裂缝片到一个文件
    save_all_fractures_single_file(all_fractures, output_dir)

    print(f"处理完成! 共生成 {len(all_fractures)} 个裂缝片")
    print(f"其中井点裂缝: {len(well_fractures)}")
    print(f"扩展裂缝: {len(all_fractures) - len(well_fractures)}")


if __name__ == "__main__":
    # ===== 配置 =====
    fracture_csv = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝角度预测\测井\车22_predicted_angles.csv"
    seismic_csv = r"单元地震信息抽取\seismic_data_block_X32_Y24.csv"  # 替换为实际CSV地震数据路径
    output_dir = r"E:\项目\石油项目\断缝储\输出\裂缝片模型"
    os.makedirs(output_dir, exist_ok=True)

    # 裂缝片基础边长 (m)
    L0 = 5.0
    alpha = 2.0
    density = 0.5

    # 地震数据参数
    seismic_threshold = 0.3  # 地震属性阈值，用于识别潜在裂缝区域
    expansion_radius = 100  # 从井点扩展的半径 (m)

    # 地震数据采样参数
    start_time = 1100  # 起始时间 (ms)
    sample_interval = 1  # 采样间隔 (ms)
    num_samples = 2701  # 采样点数

    main_single_unit()
