import numpy as np
import os
import pyvista as pv
import re
import chardet
import unicodedata
import pandas as pd
from scipy.interpolate import interp1d


# ================== 基础工具 ==================
def inspect_npy_file(file_path):
    """详细检查.npy文件内容"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)

    print("=" * 50)
    print(f"文件: {os.path.basename(file_path)}")
    print("=" * 50)

    if isinstance(data, dict):
        print("数据结构: 字典")
        print("\n包含的键:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: ndarray, 形状: {value.shape}, 数据类型: {value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
    else:
        print(f"数据类型: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"形状: {data.shape}")
            print(f"数据类型: {data.dtype}")


# ================== 可视化部分 ==================
def visualize_npy_fault(file_path):
    """从.npy文件加载并可视化断层片"""
    data = np.load(file_path, allow_pickle=True).item()

    points = data['points']
    faces = data['faces']

    faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
    mesh = pv.PolyData(points, faces_pv)

    pl = pv.Plotter()
    pl.add_mesh(mesh, color='lightblue', show_edges=True, opacity=0.8)
    pl.add_text(f"断层: {data.get('fault_name', 'Unknown')}\n"
                f"网格: i={data.get('cell_i', '?')}, j={data.get('cell_j', '?')}\n"
                f"面积: {data.get('stats', {}).get('area_3d', 0):.2f}",
                position='upper_left')
    pl.show()


def visualize_npy_fractures(file_path):
    """可视化裂缝片（支持单个或多个）"""
    data = np.load(file_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.tolist()

    pl = pv.Plotter()
    if isinstance(data, list):
        for frag in data:
            points = frag['points']
            faces = frag.get('faces')
            if faces is not None:
                faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
                mesh = pv.PolyData(points, faces_pv)
            else:
                mesh = pv.PolyData(points)
            pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)
    elif isinstance(data, dict):
        points = data['points']
        faces = data.get('faces')
        if faces is not None:
            faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
            mesh = pv.PolyData(points, faces_pv)
        else:
            mesh = pv.PolyData(points)
        pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)

    pl.add_text(f"裂缝片可视化: {os.path.basename(file_path)}", position='upper_left')
    pl.show()


# ================== 时深转换逻辑 ==================
def read_timedepth_file(file_path):
    """智能读取 .dat 时深文件，返回 DataFrame"""
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(2048))['encoding']

    with open(file_path, "r", encoding=encoding, errors='ignore') as f:
        lines = f.readlines()

    # 找到数据头所在行
    header_line_index = -1
    for i, line in enumerate(lines):
        if not line.strip().startswith("#") and len(re.split(r'\s+', line.strip())) >= 5:
            header_line_index = i - 1
            break
    header_line = lines[header_line_index].lstrip("#").strip()
    column_names = re.split(r'\s+', header_line)
    data_lines = lines[header_line_index + 1:]
    data = [re.split(r'\s+', line.strip()) for line in data_lines if line.strip()]
    df = pd.DataFrame(data, columns=column_names)

    # 标准化列名
    new_cols = []
    seen = set()
    for col in df.columns:
        norm = unicodedata.normalize("NFKD", col)
        norm = norm.upper().replace("'", "").replace(".", "").strip()
        if norm not in seen:
            new_cols.append(norm)
            seen.add(norm)
        else:
            new_cols.append(col)
    df.columns = new_cols

    # 转为数值
    df['TVD'] = pd.to_numeric(df['TVD'], errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df = df[['TVD', 'TIME']].dropna()
    print(f"✅ 已加载时深文件: {file_path}  (共 {len(df)} 行)")
    return df


def load_time_depth_curve(file_path, convert_mode="time2depth"):
    """
    根据转换模式构建插值函数
    convert_mode = "time2depth"（默认）或 "depth2time"
    """
    df = read_timedepth_file(file_path)
    time = df['TIME'].values
    depth = df['TVD'].values

    if convert_mode == "time2depth":
        f = interp1d(time, depth, bounds_error=False, fill_value='extrapolate')
        print("[Info] 构建插值函数: 时间 → 深度")
    elif convert_mode == "depth2time":
        f = interp1d(depth, time, bounds_error=False, fill_value='extrapolate')
        print("[Info] 构建插值函数: 深度 → 时间")
    else:
        raise ValueError("convert_mode 参数错误，应为 'time2depth' 或 'depth2time'")
    return f


def convert_fault_to_depth(fault_file, td_func, output_dir):
    """将断层Z坐标从时间转换为深度，并打印转换前后对比"""
    data = np.load(fault_file, allow_pickle=True).item()
    points = data['points'].copy()

    z_time = points[:, 2]
    z_depth = td_func(z_time)

    # === 打印转换前后统计信息 ===
    print("\n" + "=" * 80)
    print(f"🔹 正在转换断层文件: {os.path.basename(fault_file)}")
    print(f"  点数: {len(points)}")
    print(f"  时间域 Z 范围: {z_time.min():.2f} ~ {z_time.max():.2f}, 平均: {z_time.mean():.2f}")
    print(f"  深度域 Z 范围: {z_depth.min():.2f} ~ {z_depth.max():.2f}, 平均: {z_depth.mean():.2f}")

    # 随机抽取部分点做对比（固定5个点）
    sample_idx = np.linspace(0, len(z_time) - 1, min(5, len(z_time)), dtype=int)
    print("\n  前后对比样例（单位：时间→深度）:")
    print("  索引 |   Z_time  |  Z_depth ")
    print("  -----------------------------")
    for idx in sample_idx:
        print(f"  {idx:>5d} | {z_time[idx]:>8.2f} | {z_depth[idx]:>8.2f}")

    # === 替换并保存 ===
    new_data = data.copy()
    new_data['points'][:, 2] = z_depth

    out_path = os.path.join(output_dir, os.path.basename(fault_file).replace('.npy', '_depth.npy'))
    np.save(out_path, new_data)
    print(f"✅ 断层已完成时深转换并保存: {out_path}")
    print("=" * 80 + "\n")

    return out_path


# ================== 合并与总可视化 ==================
def merge_fault_and_fractures(fault_files, fracture_file, output_file):
    """将多个断层面和裂缝片合并到一个npy文件中"""
    if isinstance(fault_files, str):
        fault_files = [fault_files]

    for f in fault_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"断层文件不存在: {f}")
    if not os.path.exists(fracture_file):
        raise FileNotFoundError(f"裂缝文件不存在: {fracture_file}")

    fault_list = []
    for f in fault_files:
        fault_data = np.load(f, allow_pickle=True).item()
        fault_list.append(fault_data)
        print(f"✅ 已加载断层: {fault_data.get('fault_name', os.path.basename(f))}")

    fracture_data = np.load(fracture_file, allow_pickle=True)
    if isinstance(fracture_data, np.ndarray) and fracture_data.dtype == object:
        fracture_data = fracture_data.tolist()

    merged_data = {"faults": fault_list, "fractures": fracture_data}
    np.save(output_file, merged_data)
    print(f"✅ 已成功合并 {len(fault_list)} 个断层与裂缝片到: {output_file}")


def visualize_merged_fault_fractures(merged_file):
    """可视化合并后的多个断层与裂缝"""
    data = np.load(merged_file, allow_pickle=True).item()
    pl = pv.Plotter()

    for i, fault_data in enumerate(data.get("faults", [])):
        points = fault_data["points"]
        faces = fault_data["faces"]
        faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
        fault_mesh = pv.PolyData(points, faces_pv)
        pl.add_mesh(fault_mesh, color='lightblue', show_edges=True, opacity=0.8, label=f"Fault_{i+1}")

    fractures = data["fractures"]
    if isinstance(fractures, list):
        for frag in fractures:
            points = frag["points"]
            faces = frag.get("faces")
            if faces is not None:
                faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
                mesh = pv.PolyData(points, faces_pv)
            else:
                mesh = pv.PolyData(points)
            pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)
    elif isinstance(fractures, dict):
        points = fractures["points"]
        faces = fractures.get("faces")
        if faces is not None:
            faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
            mesh = pv.PolyData(points, faces_pv)
        else:
            mesh = pv.PolyData(points)
        pl.add_mesh(mesh, color='red', show_edges=True, opacity=0.6)

    pl.add_legend(labels=[["Faults", "lightblue"], ["Fractures", "red"]])
    pl.show()


# ================== 使用示例 ==================
if __name__ == "__main__":
    fault_files = [
        r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\单元实验\fault_patches_out_npy\npy\F02\F02__i14_j21.npy",
        r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\单元实验\fault_patches_out_npy\npy\fp_F_che32_bei\fp_F_che32_bei__i14_j21.npy"
    ]
    fracture_file = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\测井区块生成\单元裂缝网络\block_X14_Y21_fractures.npy"
    output_file = r"E:\项目\石油项目\断缝储\输出\block_X14_Y21_faults_fractures.npy"
    td_file = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\时深\车32.dat"
    convert_mode = "depth2time"
    # 断层根目录列表（脚本会递归搜索这些目录下的子目录来查找断层文件）"
    output_dir = os.path.dirname(output_file)

    # === 时深转换 ===
    print("加载时深曲线中...")
    td_func = load_time_depth_curve(td_file, convert_mode)
    depth_faults = [convert_fault_to_depth(f, td_func, output_dir) for f in fault_files]

    # === 合并与可视化 ===
    merge_fault_and_fractures(depth_faults, fracture_file, output_file)
    visualize_merged_fault_fractures(output_file)
