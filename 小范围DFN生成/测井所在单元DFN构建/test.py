# -*- coding: utf-8 -*-
"""
DFN单元自动裂缝生成脚本（带三维可视化）
------------------------------------------------------------
功能说明：
    输入 : wellname_X23_Y32.csv（井点裂缝所在区块）
    输出 : block_X23_Y32_fractures.npy / .csv + 可视化图形
    追加 : 自动查找并合并该单元的断层文件（支持多级目录搜索）
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
import chardet
import unicodedata
from scipy.interpolate import interp1d


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
    patterns = [f"__i{unitX}_j{unitY}.npy"]
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
# ========================= 断层时深转换：读取时深文件并返回 深度->时间 的插值函数 =========================
def read_timedepth_file(file_path):
    """读取时深文件并返回 深度→时间 插值函数"""
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(2048))['encoding']

    with open(file_path, "r", encoding=encoding, errors='ignore') as f:
        lines = f.readlines()

    # 自动定位表头
    header_line_index = -1
    for i, line in enumerate(lines):
        if not line.strip().startswith("#") and len(re.split(r'\s+', line.strip())) >= 2:
            header_line_index = i - 1
            break

    header_line = lines[header_line_index].lstrip("#").strip()
    col_names = re.split(r'\s+', header_line)
    data_lines = lines[header_line_index + 1:]
    data = [re.split(r'\s+', line.strip()) for line in data_lines if line.strip()]
    df = pd.DataFrame(data, columns=col_names)

    # 标准化列名
    df.columns = [unicodedata.normalize("NFKD", c).upper().strip() for c in df.columns]
    df['TVD'] = pd.to_numeric(df['TVD'], errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df = df.dropna(subset=['TVD', 'TIME'])

    # 构建 深度→时间 插值函数
    depth_to_time_func = interp1d(
        df['TVD'], df['TIME'], bounds_error=False, fill_value='extrapolate'
    )
    print(f"✅ 已加载时深文件: {file_path} (行数={len(df)})，并构建深度→时间插值函数")
    return depth_to_time_func



# ========================= 查找最近时深文件（改进） =========================
def find_nearest_td_file(well_name, td_dir, all_well_xy=None, current_xy=None):
    """
    优先按文件名匹配（包含井名字符串），若找不到且提供了坐标字典(all_well_xy)与当前井坐标(current_xy)，
    则根据坐标最近性选择一个时深文件（假设 td_dir 中每个时深文件名包含其井名）。
    返回时深文件路径或 None。
    """
    if not os.path.isdir(td_dir):
        return None

    # 1) 直接按文件名包含匹配
    for f in os.listdir(td_dir):
        if well_name in f:
            return os.path.join(td_dir, f)

    # 2) 如果提供了井坐标集合，按空间最近选择
    if all_well_xy and current_xy:
        dists = {wn: np.linalg.norm(np.array(xy) - np.array(current_xy)) for wn, xy in all_well_xy.items()}
        nearest = min(dists, key=dists.get)
        for f in os.listdir(td_dir):
            if nearest in f:
                print(f"⚠ 未找到 {well_name} 的时深文件，使用最近井 {nearest} 的时深数据。")
                return os.path.join(td_dir, f)

    # 3) 退化策略：如果目录非空，选最近修改的文件（备选）
    files = [os.path.join(td_dir, x) for x in os.listdir(td_dir) if os.path.isfile(os.path.join(td_dir, x))]
    if files:
        files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
        print(f"⚠ 未按井名匹配到时深文件，使用目录中最新文件: {os.path.basename(files[0])}")
        return files[0]

    return None


# ========================= convert_fault_depth_to_time（小改进，兼容向量/标量） =========================
def convert_fault_depth_to_time(fault_file, depth_to_time_func, output_dir=None, save=False):
    """
    将断层文件的 Z 坐标从深度域转换为时间域。
    参数:
        fault_file : str | dict   - 断层文件路径或已加载的fault字典
        depth_to_time_func : callable - 深度→时间 插值函数 (通常由 interp1d 生成)
        output_dir : str | None  - 若 save=True，则保存到此目录
        save : bool              - 是否保存转换后的文件（默认 False）
    返回:
        fault_data : dict        - 转换后的断层数据字典
    """
    # ---- 加载断层 ----
    if isinstance(fault_file, str):
        fault_data = np.load(fault_file, allow_pickle=True)
        if isinstance(fault_data, np.ndarray) and fault_data.dtype == object:
            fault_data = fault_data.item()
    elif isinstance(fault_file, dict):
        fault_data = fault_file.copy()
    else:
        raise ValueError(f"断层输入类型不支持: {type(fault_file)}")

    if 'points' not in fault_data:
        raise ValueError(f"断层数据缺少 'points' 字段: {fault_file}")

    pts = np.asarray(fault_data['points'], dtype=float)
    if pts.shape[1] < 3:
        raise ValueError(f"断层点坐标维度不足: {fault_file}")

    # ---- 深度 -> 时间 转换 ----
    depths = pts[:, 2]
    times = depth_to_time_func(depths)  # 插值函数：Depth → Time
    pts_time = pts.copy().astype(np.float32)
    pts_time[:, 2] = times

    fault_data['points'] = pts_time

    # ---- 打印对比信息 ----
    print(f"\n🔄 转换完成: {os.path.basename(fault_file) if isinstance(fault_file, str) else '[dict]'}")
    print(f"   原Z范围(Depth): {depths.min():.2f} ~ {depths.max():.2f}")
    print(f"   新Z范围(Time):  {times.min():.2f} ~ {times.max():.2f}")

    # ---- 可选保存 ----
    if save and output_dir:
        out_file = os.path.join(output_dir, os.path.basename(fault_file).replace('.npy', '_time.npy'))
        np.save(out_file, fault_data)
        print(f"✅ 已保存转换后的断层文件: {out_file}")
        return out_file

    return fault_data



# ========================= 新增：合并多个断层和裂缝 =========================
def merge_faults_and_fractures(fault_files, fracture_file, output_file):
    """
    将多个断层面和裂缝片合并到一个npy文件中。
    支持 fault_files 为 文件路径列表 或 fault_data 字典列表。
    """
    # ---- 标准化输入 ----
    if isinstance(fault_files, (str, dict)):
        fault_files = [fault_files]

    fault_list = []
    for f in fault_files:
        if isinstance(f, str):
            if not os.path.exists(f):
                raise FileNotFoundError(f"断层文件不存在: {f}")
            fault_data = np.load(f, allow_pickle=True).item()
        elif isinstance(f, dict):
            fault_data = f
        else:
            raise TypeError(f"不支持的断层输入类型: {type(f)}")

        fault_name = fault_data.get("fault_name", os.path.basename(f) if isinstance(f, str) else "fault_dict")
        print(f"✅ 已加载断层: {fault_name}")
        fault_list.append(fault_data)

    # ---- 加载裂缝 ----
    if not os.path.exists(fracture_file):
        raise FileNotFoundError(f"裂缝文件不存在: {fracture_file}")

    fracture_data = np.load(fracture_file, allow_pickle=True)
    if isinstance(fracture_data, np.ndarray) and fracture_data.dtype == object:
        fracture_data = fracture_data.tolist()

    # ---- 合并并保存 ----
    merged_data = {"faults": fault_list, "fractures": fracture_data}
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
def main(well_block_csv, trace_header_xy_file, td_dir, sgy_file, output_dir,
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
                              [0, 2, 3]], dtype=np.int32)
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
    # --- 时深转换部分 ---
    well_name = os.path.basename(well_block_csv).split("_")[0]  # 车32_X14_Y21 → 车32

    td_file = find_nearest_td_file(well_name, td_dir)
    print(f"时深文件:{td_file}")
    if td_file:
        td_func = read_timedepth_file(td_file)
        converted_faults = []
        for fault_file in found_faults:
            try:
                out_fault = convert_fault_depth_to_time(fault_file, td_func, output_dir)
                converted_faults.append(out_fault)
            except Exception as e:
                print(f"⚠ 断层转换失败: {fault_file} → {e}")
    else:
        print(f"⚠ 未找到井 {well_name} 的时深文件，跳过断层深度转换。")
        converted_faults = found_faults  # 直接使用原始断层

    # --- 合并断层与裂缝 ---
    if converted_faults:
        merged_file = os.path.join(output_dir, f"block_X{unitX}_Y{unitY}_merged.npy")
        merge_faults_and_fractures(converted_faults, npy_file, merged_file)
        print(f"✓ 已生成合并文件: {merged_file}")
    else:
        merged_file = None
        print(f"⚠ 未找到匹配的断层文件，跳过合并。")

    # --- 可视化部分 ---
    if show_plot:
        try:
            if merged_file and os.path.exists(merged_file):
                visualize_merged_fault_fractures(merged_file)
            else:
                visualize_npy_fractures(npy_file)
        except Exception as e:
            print(f"⚠ 可视化失败: {e}")

    print("\n✅ 单元处理完成。")

    # --- 可视化（保持你的原有可视化） ---
    # if show_plot:
    #     visualize_fractures(fractures, f"单元 X{unitX} Y{unitY} 裂缝网络")


# ========================= 主入口 =========================
if __name__ == "__main__":
    trace_header_xy_file = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv"
    SGY_FILE = r"/data/shared/project-oil/wx数据/砂砾岩/psdm_final_time.sgy"
    TRACE_LIST_DIR = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/测井区块生成/裂缝单元归属"
    OUTPUT_DIR = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/测井区块生成/单元裂缝网络-时深转换"
    TD_DIR = r"/data/shared/project-oil/wx数据/砂砾岩/时深"
    # 断层根目录列表（脚本会递归搜索这些目录下的子目录来查找断层文件）
    FAULT_ROOT_DIRS = [
        r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out_npy/npy",
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
    # well_csv = os.path.join(TRACE_LIST_DIR, "车32_X14_Y21.csv")
    # main(well_csv, trace_header_xy_file, TD_DIR, SGY_FILE, OUTPUT_DIR,
    #      START_TIME, SAMPLE_INTERVAL, NUM_SAMPLES,
    #      L0, alpha, density, SEISMIC_THRESHOLD, EXPANSION_RADIUS,
    #      fault_root_dirs=FAULT_ROOT_DIRS,
    #      show_plot=True)

    # 多井循环处理（批量运行时启用）
    for fname in os.listdir(TRACE_LIST_DIR):
        if fname.endswith(".csv"):
            fpath = os.path.join(TRACE_LIST_DIR, fname)
            main(fpath, trace_header_xy_file, TD_DIR, SGY_FILE, OUTPUT_DIR,
                 START_TIME, SAMPLE_INTERVAL, NUM_SAMPLES,
                 L0, alpha, density, SEISMIC_THRESHOLD, EXPANSION_RADIUS,
                 fault_root_dirs=FAULT_ROOT_DIRS,
                 show_plot=False)