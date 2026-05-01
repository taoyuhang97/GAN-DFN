import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from collections import defaultdict
import re
import matplotlib.pyplot as plt

# ========= 1) 读取 FaultStick-Landmark*.dat 文件 =========
def load_fault_sticks(txt_path):
    """
    统一解析多种常见格式，返回 (x, y, t, sign, fault, inline, xline)
    支持：
      A) x y t sign inline (xline+fault) ...
      B) x y t sign inline xline fault ...
      C) #Line Trace X Y Time Flag FaultName   ← 新增适配
    行尾多余描述字段将被忽略；表头或注释行(以#开头)会跳过。
    """
    import re
    import numpy as np

    records = []
    sp = re.compile(r"\s+")
    pat_xline_fault = re.compile(r"^(\d+)([A-Za-z].*)$")

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#"):
                # 跳过表头/注释
                continue

            parts = sp.split(s)
            # 忽略全是空白或异常短行
            if len(parts) < 3:
                continue

            parsed = False

            # ---------- 方案 C：有 Line/Trace/X/Y/Time/Flag/FaultName 的列式 ----------
            # 判断标准：前两列应为整数；接着至少三个浮点( X,Y,Time )；然后一个整数(Flag)，最后一个以字母开头(FaultName)
            if len(parts) >= 7:
                try:
                    line_c   = int(parts[0])         # Line → inline
                    trace_c  = int(parts[1])         # Trace → xline
                    x_c      = float(parts[2])
                    y_c      = float(parts[3])
                    t_c      = float(parts[4])       # Time
                    flag_c   = str(int(parts[5]))    # Flag → sign(字符串)
                    fault_c  = parts[6]              # FaultName（如 F01）
                    # 容错：有些行 FaultName 后可能还有空列/描述，不影响
                    if re.match(r"^[A-Za-z]", fault_c):
                        records.append((x_c, y_c, t_c, flag_c, fault_c, line_c, trace_c))
                        parsed = True
                except Exception:
                    parsed = False  # 回退到下方通用解析

            if parsed:
                continue

            # ---------- 方案 A：第6列为“xline+fault”，如 1F01 ----------
            try:
                if len(parts) >= 6:
                    x = float(parts[0])
                    y = float(parts[1])
                    t = float(parts[2])
                    sign = parts[3]          # 保留为字符串
                    inline = int(parts[4])

                    m = pat_xline_fault.match(parts[5])
                    if m:
                        xline = int(m.group(1))
                        fault = m.group(2)
                        records.append((x, y, t, sign, fault, inline, xline))
                        continue
            except Exception:
                pass  # 回退

            # ---------- 方案 B：第6列是 xline，第7列(或其后)是 fault ----------
            try:
                if len(parts) >= 6:
                    x = float(parts[0])
                    y = float(parts[1])
                    t = float(parts[2])
                    sign = parts[3]
                    inline = int(parts[4])

                    # 尝试把第6列当作 xline
                    xline = int(parts[5])

                    # 从后续列找第一个以字母开头的 token 作为 fault
                    fault = None
                    for j in range(6, min(len(parts), 12)):
                        if re.match(r"^[A-Za-z]", parts[j]):
                            fault = parts[j]
                            break
                    if fault is None:
                        fault = "UNKNOWN"

                    records.append((x, y, t, sign, fault, inline, xline))
                    continue
            except Exception:
                # 兜底：若第6列不是纯数字，且像 fault，则再找 xline
                try:
                    if len(parts) >= 6:
                        x = float(parts[0])
                        y = float(parts[1])
                        t = float(parts[2])
                        sign = parts[3]
                        inline = int(parts[4])

                        if re.match(r"^[A-Za-z]", parts[5]):
                            fault = parts[5]
                            xline = -1
                            for j in range(6, min(len(parts), 12)):
                                if re.match(r"^\d+$", parts[j]):
                                    xline = int(parts[j])
                                    break
                            records.append((x, y, t, sign, fault, inline, xline))
                            continue
                except Exception:
                    pass

            # 如果三种都未匹配，跳过该行但打印一次提示
            # （可按需注释掉以减少控制台输出）
            # print(f"未能识别的行：{s}")

    if not records:
        raise ValueError("未解析到有效数据，请检查文件内容/分隔/表头。")

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
    txt_file = r"/data/shared/project-oil/wx数据/砂砾岩/断层/FaultStick-GeoEast.dat"

    meshes = build_fault_surfaces(txt_file)
    visualize_faults(meshes)
