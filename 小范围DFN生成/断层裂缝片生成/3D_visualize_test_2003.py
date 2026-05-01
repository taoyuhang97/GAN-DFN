import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from collections import defaultdict
import re
import matplotlib.pyplot as plt

# ========= 1) 读取 FaultStick-Landmark*.dat 文件 =========
def load_fault_sticks(txt_path):
    """
    兼容两种格式：
    A) x y t sign inline xline fault ...
    B) x y t sign inline (xline+fault) ...   # 如: 1F01、2F01、12F07 等
    行尾可能还有: LM TIME futai meters ms 等描述字段，均忽略。
    """
    records = []
    pat_space = re.compile(r"\s+")
    # 匹配「若干数字」+「以字母开头的剩余部分」, e.g. "1F01" -> ("1", "F01")
    pat_xline_fault = re.compile(r"^(\d+)([A-Za-z].*)$")

    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue

            parts = pat_space.split(s)

            # 至少要有 x y t sign inline 这5列
            if len(parts) < 5:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                t = float(parts[2])
                sign = parts[3]                 # 保留为字符串（可能是数字串）

                inline = int(parts[4])          # 第5列：Inline

                fault = None
                xline = None

                # 优先尝试：第6列为 “xline+fault” 粘连形式，如 1F01 / 12F07
                if len(parts) >= 6:
                    m = pat_xline_fault.match(parts[5])
                    if m:
                        xline = int(m.group(1))
                        fault = m.group(2)
                    else:
                        # 回退：尝试老格式——第6列是 xline，第7列是 fault
                        # 例如：... inline  5   F01 ...
                        # 注意：这里第7列可能不存在或不是 fault 形式
                        try:
                            xline = int(parts[5])
                            # 尝试从第7列开始找第一个以字母开头的 token 当作 fault（如 F01/F1/FX）
                            fault_idx = None
                            for j in range(6, min(len(parts), 12)):  # 向后扫一小段
                                if re.match(r"^[A-Za-z]", parts[j]):
                                    fault_idx = j
                                    break
                            if fault_idx is not None:
                                fault = parts[fault_idx]
                            else:
                                # 实在找不到就标记为未知
                                fault = "UNKNOWN"
                        except ValueError:
                            # 第6列既不是“xline+fault”，也不是纯数字；再尝试把它当作 fault，
                            # 再从后面寻找可能的 xline（较少见，作为兜底）
                            if re.match(r"^[A-Za-z]", parts[5]):
                                fault = parts[5]
                                # 从第6列之后找第一个纯数字 token 当作 xline
                                xline_idx = None
                                for j in range(6, min(len(parts), 12)):
                                    if re.match(r"^\d+$", parts[j]):
                                        xline_idx = j
                                        break
                                xline = int(parts[xline_idx]) if xline_idx is not None else -1
                            else:
                                fault, xline = "UNKNOWN", -1
                else:
                    fault, xline = "UNKNOWN", -1

                records.append((x, y, t, sign, fault, inline, xline))

            except Exception as e:
                print(f"解析出错: {ln}\n错误: {e}")
                continue

    if not records:
        raise ValueError("未解析到有效数据，请检查文件内容与分隔。")

    return np.array(records, dtype=object)


# ========= 2) 按 FaultName 分组 =========
def group_by_fault(arr):
    groups = defaultdict(lambda: {"points": [], "sign": [], "inline": [], "xline": []})
    for (x, y, t, sign, fault, inline, xline) in arr:
        groups[fault]["points"].append([float(x), float(y), float(t)])
        groups[fault]["sign"].append(str(sign))
        groups[fault]["inline"].append(int(inline))
        groups[fault]["xline"].append(int(xline))

    for f in groups:
        groups[f]["points"] = np.asarray(groups[f]["points"], dtype=float)
        groups[f]["sign"]   = np.asarray(groups[f]["sign"], dtype=object)
        groups[f]["inline"] = np.asarray(groups[f]["inline"], dtype=int)
        groups[f]["xline"]  = np.asarray(groups[f]["xline"], dtype=int)
    return groups


# ========= 3) 使用 Delaunay 三角剖分生成多边形 =========
def triangulate_by_xy(points_xyz):
    if points_xyz is None or len(points_xyz) < 3:
        return None
    xy = points_xyz[:, :2]
    tri = Delaunay(xy)
    simplices = tri.simplices
    # PyVista faces: [n, i0, i1, i2,  n, j0, j1, j2, ...]
    faces = np.hstack([np.column_stack([np.full(len(simplices), 3), simplices]).ravel()])
    mesh = pv.PolyData(points_xyz, faces)
    mesh = mesh.clean().triangulate()
    return mesh


# ========= 4) 可视化 断层面 =========
def generate_random_colors(n):
    return [plt.cm.tab20(i % 20 / 20.0) for i in range(n)]  # 更舒服的分类色表

def visualize_faults(meshes_dict):
    pl = pv.Plotter()
    colors = generate_random_colors(len(meshes_dict))

    for i, (name, mesh) in enumerate(meshes_dict.items()):
        if mesh is None or mesh.n_cells == 0:
            continue
        pl.add_mesh(mesh, color=colors[i], opacity=0.65, show_edges=False, label=name)

    if len(meshes_dict) > 0:
        pl.add_legend()
    pl.show()


# ========= 5) 主程序 =========
def build_fault_surfaces(txt_path):
    data = load_fault_sticks(txt_path)

    # 若后续有用，可单独取列（当前流程只需要分组后的 points）
    # X = data[:, 0].astype(float)
    # Y = data[:, 1].astype(float)
    # T = data[:, 2].astype(float)
    # Sign = data[:, 3].astype(str)
    # Fault = data[:, 4].astype(str)
    # Inline = data[:, 5].astype(int)
    # Xline = data[:, 6].astype(int)

    groups = group_by_fault(data)

    meshes = {}
    for f, g in groups.items():
        pts = np.asarray(g["points"], dtype=float)
        mesh = triangulate_by_xy(pts)
        if mesh is not None and mesh.n_cells > 0:
            meshes[f] = mesh

    return meshes


if __name__ == "__main__":
    # 设置文件路径（注意：这里用 Landmark2003.dat）
    txt_file = r"/data/shared/project-oil/wx数据/砂砾岩/断层/FaultStick-Landmark2003.dat"

    meshes = build_fault_surfaces(txt_file)
    visualize_faults(meshes)
