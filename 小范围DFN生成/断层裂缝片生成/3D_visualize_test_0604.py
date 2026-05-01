import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from collections import defaultdict
import re
import matplotlib.pyplot as plt

# ========= 1) 读取“FaultStick_0604.dat”文件 =========
def load_fault_sticks(txt_path):
    records = []
    pat_space = re.compile(r"\s+")  # 匹配多个空格
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue  # 跳过注释和空行

            # 使用正则按多个空格分隔
            parts = pat_space.split(s)

            if len(parts) < 7:  # 检查每行数据列数
                continue

            try:
                # 正确地将 X, Y, T 解析为浮点数
                x = float(parts[0])  # X 坐标
                y = float(parts[1])  # Y 坐标
                t = float(parts[2])  # T 深度

                # 解析其它字段
                sign = int(parts[3])  # 标识符
                fault = parts[4]  # 断层名称，保留为字符串
                inline = int(parts[5])  # Inline
                xline = int(parts[6])  # Xline

                # 保存记录
                records.append((x, y, t, sign, fault, inline, xline))
            except ValueError as e:
                print(f"解析出错: {ln}, 错误信息: {e}")
                continue  # 出错的行跳过

    if not records:
        raise ValueError("未解析到有效数据，请检查文件内容与分隔。")

    # 返回解析后的数据
    return np.array(records, dtype=object)




# ========= 2) 按 FaultName 分组 =========
def group_by_fault(arr):
    """
    将数据按 FaultName 分组
    """
    groups = defaultdict(lambda: {"points": [], "sign": [], "inline": [], "xline": []})
    for (x, y, t, sign, fault, inline, xline) in arr:
        groups[fault]["points"].append([float(x), float(y), float(t)])
        groups[fault]["sign"].append(int(sign))
        groups[fault]["inline"].append(int(inline))
        groups[fault]["xline"].append(int(xline))

    for f in groups:
        groups[f]["points"] = np.asarray(groups[f]["points"], dtype=float)
        groups[f]["sign"]   = np.asarray(groups[f]["sign"], dtype=int)
        groups[f]["inline"] = np.asarray(groups[f]["inline"], dtype=int)
        groups[f]["xline"]  = np.asarray(groups[f]["xline"], dtype=int)
    return groups

# ========= 3) 使用 Delaunay 三角剖分生成多边形 =========
def triangulate_by_xy(points_xyz):
    """
    使用 Delaunay 三角剖分法根据 X, Y 坐标生成三角面
    """
    xy = points_xyz[:, :2]
    if len(points_xyz) < 3:
        return None
    tri = Delaunay(xy)
    simplices = tri.simplices
    faces = np.hstack([np.column_stack([np.full(len(simplices), 3), simplices]).ravel()])
    mesh = pv.PolyData(points_xyz, faces)
    mesh = mesh.clean().triangulate()
    return mesh

# ========= 4) 可视化 断层面 =========
def generate_random_colors(n):
    """生成n个随机颜色"""
    return [plt.cm.jet(i / n) for i in range(n)]

def visualize_faults(meshes_dict):
    """
    可视化多个断层生成的网格
    """
    pl = pv.Plotter()

    # 使用 matplotlib 生成随机颜色
    colors = generate_random_colors(len(meshes_dict))

    for i, (name, mesh) in enumerate(meshes_dict.items()):
        color = colors[i]  # 为每个断层分配不同的颜色
        pl.add_mesh(mesh, color=color, opacity=0.65, show_edges=False, label=name)

    pl.add_legend()
    pl.show()

# ========= 5) 主程序 =========
def build_fault_surfaces(txt_path):
    data = load_fault_sticks(txt_path)

    # 解包数据
    X = data[:, 0].astype(float)
    Y = data[:, 1].astype(float)
    T = data[:, 2].astype(float)
    Sign = data[:, 3].astype(int)
    Fault = data[:, 4].astype(str)
    Inline = data[:, 5].astype(int)
    Xline = data[:, 6].astype(int)

    # 分组数据
    groups = group_by_fault(data)

    meshes = {}
    for f, g in groups.items():
        pts = np.asarray(g["points"], dtype=float)
        mesh = triangulate_by_xy(pts)
        # 检查网格是否包含至少一个面
        if mesh is not None and mesh.n_cells > 0:
            meshes[f] = mesh

    return meshes

if __name__ == "__main__":
    # 设置文件路径
    txt_file = r"断层/FaultStick_0604.dat"

    # 构建断层面
    meshes = build_fault_surfaces(txt_file)

    # 可视化结果
    visualize_faults(meshes)
