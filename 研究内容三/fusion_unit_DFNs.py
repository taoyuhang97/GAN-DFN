# ------------------------------------------------------------
# 作用：不同单元DFN合并
# ------------------------------------------------------------
import os
import re
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d



# --------------------------------------------------
# 加载 DFN
# --------------------------------------------------
def load_dfn(file_path):
    return np.load(file_path, allow_pickle=True).item()


# --------------------------------------------------
# 解析文件名 rebuild_dfn_block_X0_Y0.npy
# --------------------------------------------------
def parse_cell_index(file_name):
    m = re.search(r"X(\d+)_Y(\d+)", file_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"文件名格式错误：{file_name}")

# --------------------------------------------------
# 自动计算 cell_size
# --------------------------------------------------
def auto_compute_cell_size(first_file_path):
    dfn = load_dfn(first_file_path)
    verts = dfn["points"]

    min_x, max_x = verts[:, 0].min(), verts[:, 0].max()
    min_y, max_y = verts[:, 1].min(), verts[:, 1].max()

    cell_x = max_x - min_x
    cell_y = max_y - min_y

    print(f"📐 自动计算 cell_size = ({cell_x:.3f}, {cell_y:.3f}, 0)")

    return (cell_x, cell_y, 0.0)

# --------------------------------------------------
# 提取边界点索引
# --------------------------------------------------
def get_boundary_indices(verts, axis, side, tol=1e-4):
    """
    axis: 'x' 或 'y'
    side: 'min' 或 'max'
    """
    if axis == 'x':
        v = verts[:, 0]
    else:
        v = verts[:, 1]

    if side == 'max':
        boundary_val = v.max()
    else:
        boundary_val = v.min()

    idx = np.where(np.abs(v - boundary_val) < tol)[0]
    return idx


# --------------------------------------------------
# 将 A 单元边界点移动到 B 边界最近点（一定范围内）
# --------------------------------------------------
def snap_A_to_B_on_boundary(verts_A, verts_B, axis, cell_size, dist_threshold=1.0):
    """
    axis: 'x' 或 'y'
    dist_threshold: 查找范围（越小越严格）
    """
    verts_A = verts_A.copy()

    # 边界点：A 的右（max）或上（max）
    # B 的左（min）或下（min）
    if axis == "x":
        idx_A = get_boundary_indices(verts_A, 'x', 'max')
        idx_B = get_boundary_indices(verts_B, 'x', 'min')
    else:
        idx_A = get_boundary_indices(verts_A, 'y', 'max')
        idx_B = get_boundary_indices(verts_B, 'y', 'min')

    if len(idx_A) == 0 or len(idx_B) == 0:
        return verts_A  # 无接触点

    tree = cKDTree(verts_B[idx_B])

    for i in idx_A:
        p = verts_A[i]
        dist, j = tree.query(p)

        # 如果距离太远，不吸附
        if dist < dist_threshold:
            verts_A[i] = verts_B[idx_B][j]  # 直接吸附到最近点

    return verts_A


# --------------------------------------------------
# 主函数：合并 DFN
# --------------------------------------------------
def merge_dfns_in_folder(folder_path, cell_size=(100, 100, 0), out_file="dfn_merged.npy"):
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

    if not all_files:
        print("❌ 文件夹中没有 DFN 文件")
        return

    # 若 cell_size 未传入 → 自动计算
    if cell_size is None:
        first_file = os.path.join(folder_path, all_files[0])
        cell_size = auto_compute_cell_size(first_file)

    print(f"📦 使用 cell_size = {cell_size}")

    # 以 (x, y) 索引存储 DFN
    dfn_dict = {}
    for f in all_files:
        xy = parse_cell_index(f)
        dfn_dict[xy] = load_dfn(os.path.join(folder_path, f))

    # -----------------------------
    # 1) 边界吸附处理（不平滑、不糊化）
    # -----------------------------
    for (x, y), dfn_A in dfn_dict.items():
        verts_A = dfn_A["points"]

        # 右侧邻居
        if (x + 1, y) in dfn_dict:
            verts_B = dfn_dict[(x + 1, y)]["points"]
            verts_A = snap_A_to_B_on_boundary(verts_A, verts_B, axis="x", cell_size=cell_size)

        # 上侧邻居
        if (x, y + 1) in dfn_dict:
            verts_B = dfn_dict[(x, y + 1)]["points"]
            verts_A = snap_A_to_B_on_boundary(verts_A, verts_B, axis="y", cell_size=cell_size)

        # 更新
        dfn_A["points"] = verts_A


    # -----------------------------
    # 2) 对齐完成后再统一平移并合并
    # -----------------------------
    merged_verts = []
    merged_faces = []
    vert_offset = 0
    fault_names = []
    merged_stats = {}

    for (x, y), dfn in sorted(dfn_dict.items()):
        offset = np.array([x * cell_size[0], y * cell_size[1], 0.0])

        verts = dfn["points"] + offset
        faces = dfn["faces"] + vert_offset

        merged_verts.append(verts)
        merged_faces.append(faces)
        vert_offset += len(verts)

        for k, v in dfn.get("stats", {}).items():
            merged_stats[k] = merged_stats.get(k, 0) + v

        fault_names.append(dfn.get("fault_name", f"X{x}_Y{y}"))

    merged_dfn = {
        "points": np.vstack(merged_verts),
        "faces": np.vstack(merged_faces),
        "stats": merged_stats,
        "fault_name": "+".join(fault_names),
    }

    np.save(out_file, merged_dfn)
    print(f"✨ DFN 合并完成：{out_file}")

    return merged_dfn

def visualize_dfn(dfn):
    points = dfn['points']
    faces = dfn['faces']
    faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
    mesh = pv.PolyData(points, faces_pv)

    pl = pv.Plotter()
    pl.add_mesh(mesh, color='lightblue', show_edges=True, opacity=0.8)
    pl.add_text(f"合并 DFN: {dfn['fault_name']}", position='upper_left')
    pl.show()

if __name__ == "__main__":
    folder_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\3D_GAN输出\裂缝片转化"
    out_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\DFN合并"
    os.makedirs(out_dir, exist_ok=True)

    merged_dfn = merge_dfns_in_folder(folder_path, cell_size=(300,300,0), out_file=os.path.join(out_dir, "merged_dfn.npy"))
    visualize_dfn(merged_dfn)