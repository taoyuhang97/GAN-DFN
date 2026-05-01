import numpy as np
import os
import pyvista as pv

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

        # 详细查看主要数据
        print("\n详细内容:")
        for key, value in data.items():
            if key == 'points':
                print(f"\n{key} (顶点坐标):")
                print(f"  形状: {value.shape}")
                print(f"  前5个点:\n{value[:5]}")

            elif key == 'faces':
                print(f"\n{key} (三角面片):")
                print(f"  形状: {value.shape}")
                print(f"  面片数量: {len(value)}")
                print(f"  前5个面片:\n{value[:5]}")

            elif key == 'stats':
                print(f"\n{key} (统计信息):")
                for stat_key, stat_value in value.items():
                    print(f"  {stat_key}: {stat_value}")

    else:
        print(f"数据类型: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"前几个元素:\n{data}")

def visualize_npy_fault(file_path):
    """从.npy文件加载并可视化断层片"""
    data = np.load(file_path, allow_pickle=True).item()

    points = data['points']
    faces = data['faces']

    # 将面片转换为PyVista格式
    faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])

    # 创建网格
    mesh = pv.PolyData(points, faces_pv)

    # 可视化
    pl = pv.Plotter()
    pl.add_mesh(mesh, color='lightblue', show_edges=True, opacity=0.8)
    pl.add_text(f"断层: {data['fault_name']}\n"
                f"网格: i={data['cell_i']}, j={data['cell_j']}\n"
                f"面积: {data['stats']['area_3d']:.2f}",
                position='upper_left')
    pl.show()


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

def merge_fault_and_fractures(fault_files, fracture_file, output_file):
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
def visualize_merged_fault_fractures(merged_file):
    """可视化合并后的多个断层与裂缝"""
    if not os.path.exists(merged_file):
        print(f"❌ 文件不存在: {merged_file}")
        return

    data = np.load(merged_file, allow_pickle=True).item()
    pl = pv.Plotter()

    print("\n================ 文件内容开始 ================\n")
    print(data)  # 美观打印所有键值对
    print("\n================ 文件内容结束 ================\n")

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

# ---------------- 使用示例 ----------------
# 断层面展示
# fault_path = "/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out_npy/npy/F01/F01__i47_j4.npy"
# fault_path = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/测井区块生成/block_X73_Y47_fractures.csv"
# fault_path = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out_npy/npy/fp_F_che32_bei/fp_F_che32_bei__i14_j21.npy"
# inspect_npy_file(fault_path)
# visualize_npy_fault(fault_path)
# fault_path = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out_npy/npy/F02/F02__i14_j21.npy"
# inspect_npy_file(fault_path)
# visualize_npy_fault(fault_path)

# 裂缝片展示
fracture_path = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/测井区块生成/单元裂缝网络/block_X14_Y21_fractures.npy"
inspect_npy_file(fracture_path)
visualize_npy_fractures(fracture_path)

# 单元DFN展示
# fault_files = [
#     r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out_npy/npy/F02/F02__i14_j21.npy",
#     r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out_npy/npy/fp_F_che32_bei/fp_F_che32_bei__i14_j21.npy"
# ]
# fracture_file = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/测井区块生成/单元裂缝网络/block_X14_Y21_fractures.npy"
# output_file = r"E:\项目\石油项目\断缝储\输出\block_X14_Y21_faults_fractures.npy"
#
# merge_fault_and_fractures(fault_files, fracture_file, output_file)
# visualize_merged_fault_fractures(output_file)
