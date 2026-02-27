# ------------------------------------------------------------
# 作用：有测井区域的DFN生成
# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
训练用单元裂缝数据生成(裂缝片表示)
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import segyio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import DBSCAN
import pyvista as pv

# Matplotlib 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========================= 前面保持不变的函数 =========================
def parse_unit_xy_from_filename(filename: str):
    match = re.search(r'_X(\d+)_Y(\d+)', filename)
    if not match:
        raise ValueError(f"文件名中未找到形如 _X23_Y32 的坐标信息：{filename}")
    return int(match.group(1)), int(match.group(2))


def extract_block(trace_header_file: str, block_x: int, block_y: int, block_size=25):
    print(f"=== 提取区块 (X={block_x}, Y={block_y}) ===")
    df = pd.read_csv(trace_header_file)

    unique_x = np.sort(df['X'].unique())
    unique_y = np.sort(df['Y'].unique())

    start_x = block_x * (block_size - 1)
    end_x = start_x + block_size
    start_y = block_y * (block_size - 1)
    end_y = start_y + block_size

    target_x = unique_x[start_x:end_x]
    target_y = unique_y[start_y:end_y]

    grid_df = df[df['X'].isin(target_x) & df['Y'].isin(target_y)]
    grid_df = grid_df.sort_values(by=['X', 'Y']).reset_index(drop=True)
    print(f"  → 共提取 {len(grid_df)} 条地震道。")
    return grid_df


def extract_sgy_data(sgy_file, trace_df):
    trace_indices = trace_df['TraceIdx'].tolist()
    x_coords, y_coords = trace_df['X'].to_numpy(), trace_df['Y'].to_numpy()

    with segyio.open(sgy_file, 'r', ignore_geometry=True) as sgy:
        num_samples = len(sgy.samples)
        data = np.zeros((len(trace_indices), num_samples), dtype=np.float32)
        for i, idx in enumerate(trace_indices):
            data[i, :] = sgy.trace[idx]
            if (i + 1) % 100 == 0 or i == len(trace_indices) - 1:
                print(f"  提取进度: {i + 1}/{len(trace_indices)} 道")

    print(f"✓ 地震数据提取完成：{len(trace_indices)} 道，每道 {num_samples} 采样点")
    return x_coords, y_coords, data


def create_3d_seismic(x_coords, y_coords, amp_data, start_time, num_samples, sample_interval):
    print("=== 构建三维地震体 ===")
    z_coords = np.arange(start_time, start_time + num_samples * sample_interval, sample_interval)
    x_unique, y_unique = np.unique(x_coords), np.unique(y_coords)
    seismic_3d = np.zeros((len(x_unique), len(y_unique), len(z_coords)))

    tree = cKDTree(np.column_stack([x_coords, y_coords]))
    for i, xi in enumerate(x_unique):
        for j, yj in enumerate(y_unique):
            _, idx = tree.query([xi, yj], k=1)
            seismic_3d[i, j, :] = amp_data[idx, :]

    print(f"✓ 三维地震体尺寸: {seismic_3d.shape}")
    return seismic_3d, x_unique, y_unique, z_coords


def detect_fracture_zones(seismic, threshold=0.3):
    gx, gy, gz = np.gradient(seismic, axis=(0, 1, 2))
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    mask = grad > threshold
    return np.argwhere(mask), grad


def generate_fracture_square(center, dip_azimuth, dip_angle, L0, alpha, density):
    az = np.radians(dip_azimuth)
    da = np.radians(dip_angle)
    n = np.array([np.sin(da) * np.sin(az), np.sin(da) * np.cos(az), np.cos(da)])
    n /= np.linalg.norm(n)
    up = np.array([0, 0, 1]) if not np.allclose(n, [0, 0, 1]) else np.array([1, 0, 0])
    u = np.cross(n, up); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    L = L0 * (1 + alpha * density)
    half_L = L / 2
    return np.array([ center + su * half_L * u + sv * half_L * v
                      for su in [-1, 1] for sv in [-1, 1] ])


def expand_fractures(well_fractures, seismic, xg, yg, zg, radius, threshold, L0, alpha):
    all_fractures = []
    frac_points, grad = detect_fracture_zones(seismic, threshold)
    interp = RegularGridInterpolator((xg, yg, zg), grad, bounds_error=False, fill_value=0)

    for wf in well_fractures:
        c, az, ang, den = wf['center'], wf['dip_azimuth'], wf['dip_angle'], wf['density']
        all_fractures.append({'vertices': generate_fracture_square(c, az, ang, L0, alpha, den),
                              'type': 'well', 'center': c})
        nearby_pts = []
        for p in frac_points:
            x, y, z = xg[p[0]], yg[p[1]], zg[p[2]]
            d = np.linalg.norm([x - c[0], y - c[1], z - c[2]])
            if d <= radius:
                nearby_pts.append((x, y, z, interp([x, y, z])[0]))
        if not nearby_pts:
            continue

        pts = np.array(nearby_pts)
        coords, intens = pts[:, :3], pts[:, 3]
        cl = DBSCAN(eps=50, min_samples=3).fit(coords)

        for lbl in set(cl.labels_):
            if lbl == -1:
                continue
            cluster_pts = coords[cl.labels_ == lbl]
            avg_int = np.mean(intens[cl.labels_ == lbl])
            center = np.mean(cluster_pts, axis=0)
            den_adj = den * (0.5 + 0.5 * avg_int)
            all_fractures.append({
                'vertices': generate_fracture_square(center, az, ang, L0, alpha, den_adj),
                'type': 'expanded',
                'center': center,
                'intensity': avg_int
            })

    return all_fractures


def visualize_fractures(fractures, title="DFN裂缝单元可视化"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14)

    for f in fractures:
        verts = f['vertices']
        face = [[verts[0], verts[1], verts[3], verts[2]]]
        color = 'red' if f['type'] == 'well' else 'blue'
        ax.add_collection3d(Poly3DCollection(face, alpha=0.5, facecolor=color, edgecolor='k'))
        ax.scatter(*f['center'], color=color, s=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time/Depth")
    plt.tight_layout()
    plt.show()


# ========================= 新增：查找断层文件（多级目录） =========================
def find_fault_files_for_unit(unitX, unitY, fault_root_dirs):
    """
    在多个根目录（含子目录）中查找与单元 (unitX, unitY) 对应的断层文件。
    支持的文件名模式（可根据你的文件命名再扩展）：
      - 包含 __i{unitX}_j{unitY}
      - 包含 _X{unitX}_Y{unitY}
    返回：找到的文件路径列表（可能为空）
    """
    patterns = [f"__i{unitX}_j{unitY}"]
    found = []
    for root in fault_root_dirs:
        if not os.path.isdir(root):
            continue
        # 递归搜索二级及更深目录
        for path, _, files in os.walk(root):
            for f in files:
                for pat in patterns:
                    if pat in f:
                        found.append(os.path.join(path, f))
                        break
    # 去重并排序（可选）
    found = sorted(list(dict.fromkeys(found)))
    return found


# ========================= 新增：合并多个断层和裂缝 =========================
def merge_faults_and_fractures(fault_files, fracture_file, output_file):
    """
    将多个断层面和裂缝片合并到一个npy文件中。

    参数:
        fault_files: 断层.npy 文件路径列表（每个为字典形式）
        fracture_file: 裂缝片.npy 文件路径（列表或字典形式）
        output_file: 输出文件路径
    """
    # ---- 检查输入 ----
    if isinstance(fault_files, str):
        fault_files = [fault_files]  # 兼容单文件输入

    for f in fault_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"断层文件不存在: {f}")

    if not os.path.exists(fracture_file):
        raise FileNotFoundError(f"裂缝文件不存在: {fracture_file}")

    # ---- 加载多个断层 ----
    fault_list = []
    for f in fault_files:
        fault_data = np.load(f, allow_pickle=True).item()
        fault_name = fault_data.get("fault_name", os.path.basename(f))
        print(f"✅ 已加载断层: {fault_name} ({os.path.basename(f)})")
        fault_list.append(fault_data)

    # ---- 加载裂缝 ----
    fracture_data = np.load(fracture_file, allow_pickle=True)
    if isinstance(fracture_data, np.ndarray) and fracture_data.dtype == object:
        fracture_data = fracture_data.tolist()  # 转换为列表

    # ---- 构建合并字典 ----
    merged_data = {
        "faults": fault_list,       # 多个断层
        "fractures": fracture_data  # 裂缝列表或字典
    }

    # ---- 保存 ----
    np.save(output_file, merged_data)
    print(f"✅ 已成功合并 {len(fault_list)} 个断层与裂缝片到: {output_file}")

def visualize_npy_fractures(file_path):
    """可视化裂缝片（支持单个或多个）"""
    data = np.load(file_path, allow_pickle=True)

    # 支持列表或字典
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.tolist()

    pl = pv.Plotter()

    if isinstance(data, list):
        for i, frag in enumerate(data):
            points = frag['points']
            faces = frag.get('faces', None)
            if faces is not None:
                faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
                mesh = pv.PolyData(points, faces_pv)
            else:
                mesh = pv.PolyData(points)
            pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)
    elif isinstance(data, dict):
        points = data['points']
        faces = data.get('faces', None)
        if faces is not None:
            faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
            mesh = pv.PolyData(points, faces_pv)
        else:
            mesh = pv.PolyData(points)
        pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)

    pl.add_text(f"裂缝片可视化: {os.path.basename(file_path)}", position='upper_left')
    pl.show()


def visualize_merged_fault_fractures(merged_file):
    """可视化合并后的多个断层与裂缝"""
    if not os.path.exists(merged_file):
        print(f"❌ 文件不存在: {merged_file}")
        return

    data = np.load(merged_file, allow_pickle=True).item()
    pl = pv.Plotter()

    # ---- 多个断层 ----
    faults = data.get("faults", [])
    if isinstance(faults, dict):  # 兼容旧格式
        faults = [faults]

    for i, fault_data in enumerate(faults):
        points = fault_data["points"]
        faces = fault_data["faces"]
        faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
        fault_mesh = pv.PolyData(points, faces_pv)
        pl.add_mesh(fault_mesh, color='lightblue', show_edges=True, opacity=0.8, label=f"Fault_{i+1}")

    # ---- 裂缝 ----
    fractures = data["fractures"]
    if isinstance(fractures, list):
        for frag in fractures:
            points = frag["points"]
            faces = frag.get("faces", None)
            if faces is not None:
                faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
                mesh = pv.PolyData(points, faces_pv)
            else:
                mesh = pv.PolyData(points)
            pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)
    elif isinstance(fractures, dict):
        points = fractures["points"]
        faces = fractures.get("faces", None)
        if faces is not None:
            faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
            mesh = pv.PolyData(points, faces_pv)
        else:
            mesh = pv.PolyData(points)
        pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)

    pl.add_legend(labels=[["Faults", "lightblue"], ["Fractures", "red"]])
    pl.show()



# ========================= 主流程（在生成裂缝后追加断层匹配合并） =========================
def main(well_block_csv, trace_header_xy_file, sgy_file, output_dir,
         start_time, sample_interval, num_samples,
         L0, alpha, density, seismic_threshold, expansion_radius,
         fault_root_dirs=None,
         show_plot=True):

    # --- 解析区块坐标 ---
    unitX, unitY = parse_unit_xy_from_filename(os.path.basename(well_block_csv))
    print(f"\n=== 处理单元: X{unitX} Y{unitY} ===")

    # --- 提取地震道与数据 ---
    trace_df = extract_block(trace_header_xy_file, unitX, unitY)
    x_coords, y_coords, amp_data = extract_sgy_data(sgy_file, trace_df)
    seismic, xg, yg, zg = create_3d_seismic(x_coords, y_coords, amp_data,
                                            start_time, num_samples, sample_interval)

    # --- 裂缝扩展 ---
    df = pd.read_csv(well_block_csv)
    well_fractures = [{
        'center': np.array([r['X'], r['Y'], r['TIME']]),
        'dip_azimuth': r['Dip_Azimuth'],
        'dip_angle': r['Dip_Angle'],
        'density': density
    } for _, r in df.iterrows()]

    fractures = expand_fractures(well_fractures, seismic, xg, yg, zg,
                                 expansion_radius, seismic_threshold, L0, alpha)

    # --- 保存结果（PyVista兼容结构） ---
    npy_file = os.path.join(output_dir, f"block_X{unitX}_Y{unitY}_fractures.npy")

    fracture_pyvista_list = []
    for f in fractures:
        vertices = f['vertices'].astype(np.float32)

        # 生成三角面索引（每个裂缝为两个三角面）
        if vertices.shape[0] == 4:
            faces = np.array([[0, 1, 2],
                              [1, 2, 3]], dtype=np.int32)
        else:
            faces = np.array([[0, 1, 2]], dtype=np.int32)

        fracture_pyvista_list.append({
            'points': vertices,  # 顶点坐标 (4×3)
            'faces': faces,  # 面片索引
            'type': f['type'],  # 'well' 或 'expanded'
            'center': f['center'].astype(np.float32),  # 裂缝中心
            'intensity': np.float32(f.get('intensity', 0.0))
        })

    np.save(npy_file, np.array(fracture_pyvista_list, dtype=object))
    print(f"✓ 裂缝结果保存完成 (PyVista格式)：\n  → {npy_file}")

    # --- 追加：自动查找断层并合并 ---
    if fault_root_dirs is None:
        fault_root_dirs = []

    # 查找该单元对应断层文件（支持多个匹配）
    found_faults = find_fault_files_for_unit(unitX, unitY, fault_root_dirs)
    if not found_faults:
        # 若没有找到断层，则保留裂缝文件并记录日志（按需可复制到一个备份目录）
        print(f"⚠ 未找到单元 (X{unitX}, Y{unitY}) 对应的断层文件。保留裂缝文件：{npy_file}")
        if show_plot:
            visualize_npy_fractures(npy_file)
    else:
        # 合并（若找到多个断层文件，则全部合并到同一个输出）
        merged_out = os.path.join(output_dir, f"block_X{unitX}_Y{unitY}_fault_fractures.npy")
        try:
            merge_faults_and_fractures(found_faults, npy_file, merged_out)
            if show_plot:
                visualize_merged_fault_fractures(merged_out)
        except Exception as e:
            print(f"❌ 合并失败: {e}")
            print("已保留原始裂缝文件。")

    # --- 可视化（保持你的原有可视化） ---
    # if show_plot:
    #     visualize_fractures(fractures, f"单元 X{unitX} Y{unitY} 裂缝网络")


# ========================= 主入口 =========================
if __name__ == "__main__":
    trace_header_xy_file = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"
    SGY_FILE = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\psdm_final_time.sgy"
    TRACE_LIST_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\测井区块生成\裂缝单元归属"
    OUTPUT_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\训练用数据\裂缝片表示"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 断层根目录列表（脚本会递归搜索这些目录下的子目录来查找断层文件）
    FAULT_ROOT_DIRS = [
        r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\单元实验\fault_patches_out_npy\npy",
        # 若有其他目录，可继续添加
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    START_TIME = 1100
    SAMPLE_INTERVAL = 1
    NUM_SAMPLES = 2701
    SEISMIC_THRESHOLD = 0.3
    EXPANSION_RADIUS = 100
    L0 = 5.0
    alpha = 2.0
    density = 0.5

    # 单井示例
    well_csv = os.path.join(TRACE_LIST_DIR, "车32_X14_Y21.csv")
    main(well_csv, trace_header_xy_file, SGY_FILE, OUTPUT_DIR,
         START_TIME, SAMPLE_INTERVAL, NUM_SAMPLES,
         L0, alpha, density, SEISMIC_THRESHOLD, EXPANSION_RADIUS,
         fault_root_dirs=FAULT_ROOT_DIRS,
         show_plot=True)

    # 多井循环处理（批量运行时启用）
    # for fname in os.listdir(TRACE_LIST_DIR):
    #     if fname.endswith(".csv"):
    #         fpath = os.path.join(TRACE_LIST_DIR, fname)
    #         main(fpath, trace_header_xy_file, SGY_FILE, OUTPUT_DIR,
    #              START_TIME, SAMPLE_INTERVAL, NUM_SAMPLES,
    #              L0, alpha, density, SEISMIC_THRESHOLD, EXPANSION_RADIUS,
    #              show_plot=False)