# ------------------------------------------------------------
# 作用：裂缝片转体素表示
# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
mesh_to_volume.py
功能：
1. 读取 DFN NPY 文件（三角片网格）
2. 构建 25x25x2701 地震点阵 → 24x24x2700 voxel 网格
3. 计算体素 SDF、Soft Occupancy、Normals
4. 保存 5 通道体素数组
5. PyVista 可视化体素立方体 + 三角网格裂缝片
"""
import os
import re
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


# ------------------------------
# 工具函数
# ------------------------------
def parse_unit_xy_from_filename(filename: str):
    match = re.search(r'_X(\d+)_Y(\d+)', filename)
    if not match:
        raise ValueError(f"文件名中未找到坐标信息：{filename}")
    return int(match.group(1)), int(match.group(2))

def point_to_triangle_distance(points, tri):
    """
    points: (N,3)
    tri: (3,3)
    返回 voxel 点到三角形距离
    """
    A, B, C = tri
    AB, AC = B-A, C-A
    AP = points - A
    N = np.cross(AB, AC)
    N_unit = N / (np.linalg.norm(N) + 1e-12)
    signed_d = np.einsum('ij,j->i', AP, N_unit)
    proj = points - np.outer(signed_d, N_unit)

    # Barycentric 坐标
    v0, v1, v2 = AB, AC, proj-A
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    dot02 = np.einsum('ij,j->i', v2, v0)
    dot12 = np.einsum('ij,j->i', v2, v1)
    invDenom = 1.0/(dot00*dot11 - dot01*dot01 + 1e-12)
    u = (dot11*dot02 - dot01*dot12)*invDenom
    v = (dot00*dot12 - dot01*dot02)*invDenom
    inside = (u>=0) & (v>=0) & (u+v<=1)
    closest = np.copy(proj)

    def edge_closest(P,A,B):
        t = np.clip(np.einsum('ij,j->i', P-A, B-A)/(np.dot(B-A,B-A)+1e-12),0,1)
        return A + np.outer(t, B-A)

    closest_ab = edge_closest(points,A,B)
    closest_ac = edge_closest(points,A,C)
    closest_bc = edge_closest(points,B,C)

    d_ab = np.linalg.norm(points-closest_ab,axis=1)
    d_ac = np.linalg.norm(points-closest_ac,axis=1)
    d_bc = np.linalg.norm(points-closest_bc,axis=1)

    edge_pts = closest_ab.copy()
    edge_pts[d_ac<d_ab] = closest_ac[d_ac<d_ab]
    edge_pts[d_bc<np.minimum(d_ab,d_ac)] = closest_bc[d_bc<np.minimum(d_ab,d_ac)]
    closest[~inside] = edge_pts[~inside]

    dist = np.linalg.norm(points-closest,axis=1)
    signed_dist = np.abs(dist*np.sign(signed_d))
    return signed_dist, closest

def triangle_bbox_voxel_indices(tri, x_centers, y_centers, z_centers, trunc):
    tri_min, tri_max = tri.min(axis=0)-trunc, tri.max(axis=0)+trunc
    ix = np.where((x_centers>=tri_min[0]) & (x_centers<=tri_max[0]))[0]
    iy = np.where((y_centers>=tri_min[1]) & (y_centers<=tri_max[1]))[0]
    iz = np.where((z_centers>=tri_min[2]) & (z_centers<=tri_max[2]))[0]
    return ix, iy, iz

# ------------------------------
# DFN → Voxel SDF/SoftOcc/Normals
# ------------------------------
def mesh_to_voxel_fields(mesh_list, x_centers, y_centers, z_centers,
                         truncation=5.0, sigma=2.5, normal_distance_threshold=2.0):
    """
    将 DFN 三角片网格转换为体素 SDF、Soft Occupancy 和法向量。
    支持重叠裂缝片：
    - sdf: voxel 到最近裂缝片的最小距离
    - soft_occ: 基于 sdf 的软占据
    - normals: 重叠区域法向量累加归一化
    """
    nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
    Xc, Yc, Zc = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    sdf = np.ones((nx, ny, nz), dtype=np.float32) * truncation
    normals = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    normals_count = np.zeros((nx, ny, nz), dtype=np.int32)  # 记录累加次数

    for mesh in mesh_list:
        pts, faces = mesh["points"], mesh["faces"]
        for f in faces:
            tri = pts[f]
            tri_normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            tri_normal /= np.linalg.norm(tri_normal) + 1e-12

            ix, iy, iz = triangle_bbox_voxel_indices(tri, x_centers, y_centers, z_centers, truncation)
            if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
                continue

            XX, YY, ZZ = Xc[np.ix_(ix, iy, iz)].ravel(), Yc[np.ix_(ix, iy, iz)].ravel(), Zc[np.ix_(ix, iy, iz)].ravel()
            voxel_pts = np.stack([XX, YY, ZZ], axis=-1)
            dist, _ = point_to_triangle_distance(voxel_pts, tri)

            Dgrid = dist.reshape((len(ix), len(iy), len(iz)))
            Dold = sdf[np.ix_(ix, iy, iz)]
            mask = Dgrid < Dold
            Dold[mask] = Dgrid[mask]
            sdf[np.ix_(ix, iy, iz)] = Dold

            # -----------------------
            # 累加 normals
            # -----------------------
            mask_close = Dgrid < normal_distance_threshold
            if np.any(mask_close):
                normals_block = normals[np.ix_(ix, iy, iz)]
                count_block = normals_count[np.ix_(ix, iy, iz)]

                closest_normals = np.tile(tri_normal, (Dgrid.size, 1)).reshape((len(ix), len(iy), len(iz), 3))
                # 仅对 mask_close 累加
                normals_block[mask_close] += closest_normals[mask_close]
                count_block[mask_close] += 1

                normals[np.ix_(ix, iy, iz)] = normals_block
                normals_count[np.ix_(ix, iy, iz)] = count_block

    # 归一化 normals
    nonzero_mask = normals_count > 0
    normals[nonzero_mask] /= normals_count[nonzero_mask][..., None]
    # 避免 NaN
    normals[~nonzero_mask] = 0.0

    soft_occ = np.exp(-(sdf**2) / (2 * sigma**2))
    return sdf, soft_occ, normals

# ------------------------------
# DFN 文件读取
# ------------------------------
def load_dfn_mesh(file_path):
    """兼容你提供的 DFN NPY 格式"""
    data = np.load(file_path, allow_pickle=True)
    mesh_list = []
    for elem in data:
        pts = elem['points'].astype(np.float32)
        faces = elem['faces'].astype(np.int32)
        # 如果 faces 是四边形，拆分为两个三角形
        tri_faces = []
        for f in faces:
            if len(f) == 3:
                tri_faces.append(f)
            elif len(f) == 4:
                tri_faces.append(f[:3])
                tri_faces.append(f[[0,2,3]])
            else:
                raise ValueError("faces 中存在非三角/四边形")
        tri_faces = np.array(tri_faces, dtype=np.int32)
        mesh_list.append({'points': pts, 'faces': tri_faces})
    print(f"已读取 DFN，共 {len(mesh_list)} 个三角片网格")
    return mesh_list


# ------------------------------
# PyVista 可视化
# ------------------------------
def visualize_voxels_with_dfn(voxel_5ch, mesh_list, channel=1, opacity=0.3, threshold=None, show_all=False, near_dist=None):
    """
    修正版可视化：
    - 每个体素独立立方体（边长 = grid spacing），相邻立方体紧贴不重叠
    - show_all=False 时显示“裂缝附近体素”
    - near_dist: 如果为 None，自动使用体素半对角线作为“相交”阈值（单位与坐标一致）
    """
    nx, ny, nz, nch = voxel_5ch.shape
    voxel_values = voxel_5ch[..., channel]

    # threshold 初筛（基于选定通道）
    if threshold is not None:
        mask = voxel_values > threshold
    else:
        mask = np.ones_like(voxel_values, dtype=bool)

    # 构造索引网格 & 体素中心索引（被 threshold 保留的）
    xv, yv, zv = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    idxs = np.stack([xv, yv, zv], axis=-1).reshape(-1,3)
    mask_flat = mask.ravel()
    selected_idx = idxs[mask_flat]  # (M,3)

    if selected_idx.size == 0:
        print("Warning: 没有体素通过 threshold 筛选，退出显示。")
        return

    # 真实坐标（使用外部全局变量 x_centers,y_centers,z_centers）
    # 注意：函数依赖脚本中定义的 x_centers/y_centers/z_centers
    xc = x_centers[selected_idx[:,0]]
    yc = y_centers[selected_idx[:,1]]
    zc = z_centers[selected_idx[:,2]]
    voxel_centers = np.stack([xc, yc, zc], axis=-1).astype(np.float32)  # (M,3)

    # 计算 grid spacing -> 用于立方体大小（确保相邻体素贴合）
    # 若只有一个中心，退化为 1.0
    if len(x_centers) > 1:
        dx = float(np.mean(np.diff(x_centers)))
    else:
        dx = 1.0
    if len(y_centers) > 1:
        dy = float(np.mean(np.diff(y_centers)))
    else:
        dy = 1.0
    if len(z_centers) > 1:
        dz = float(np.mean(np.diff(z_centers)))
    else:
        dz = 1.0

    # 立方体边长取每方向 spacing（使相邻体素刚好贴合）
    cube_x, cube_y, cube_z = dx, dy, dz

    # 半对角线（用于判定“体素与三角片接触”的保守阈值）
    half_diag = 0.5 * np.sqrt(cube_x**2 + cube_y**2 + cube_z**2)

    # --------------------
    # 如果只显示靠近裂缝的体素，做距离过滤
    # --------------------
    if not show_all:
        # collect all fracture vertices (三角形顶点)
        all_pts = []
        for mesh in mesh_list:
            all_pts.append(mesh["points"].astype(np.float32))
        all_pts = np.vstack(all_pts)  # (K,3)

        # KDTree 最近点距离作为粗筛（比对到三角面真距离前的快速筛）
        tree = cKDTree(all_pts)
        # query 返回距离（单位与坐标一致）
        dists, _ = tree.query(voxel_centers, k=1)

        # 建议的 effective threshold：
        # 如果用户传入 near_dist（以坐标单位），使用它；否则使用 half_diag
        if near_dist is None:
            eff_thresh = half_diag
        else:
            # 把用户 near_dist 与半对角线结合：允许用户阈值比半对角线更宽松
            eff_thresh = max(near_dist, half_diag)

        near_mask = dists <= eff_thresh
        if near_mask.sum() == 0:
            print(f"WARNING: 没有体素位于 near_dist={near_dist} 或 half_diag={half_diag:.3f} 内，显示全部体素作为回退。")
            # keep voxel_centers unchanged and use full list
            # voxel_values 保持原 selected order
        else:
            # 只保留 near 的体素
            voxel_centers = voxel_centers[near_mask]
            voxel_values = voxel_values.ravel()[mask_flat][near_mask]

    else:
        # 展示全部：按 mask_flat 取值
        voxel_values = voxel_values.ravel()[mask_flat]

    # 归一化颜色（避免常数分母）
    vmin = voxel_values.min()
    vmax = voxel_values.max()
    if vmax - vmin < 1e-12:
        colors = np.ones_like(voxel_values, dtype=np.float32) * 0.5
    else:
        colors = (voxel_values - vmin) / (vmax - vmin)

    # --------------------
    # 构建每个体素的 8 个顶点 & hexa cells（不会用 glyph）
    # --------------------
    pts_list = []
    cells = []
    cell_types = []
    for i, c in enumerate(voxel_centers):
        x0, y0, z0 = c
        # 将 cube 的起点定义为左下近角（确保相邻贴合）
        # 为了贴合，用 x0 - cube_x/2 ... 使 center 参数为中心点
        x_min = x0 - cube_x/2.0
        y_min = y0 - cube_y/2.0
        z_min = z0 - cube_z/2.0

        pts = np.array([
            [x_min,          y_min,          z_min],
            [x_min + cube_x, y_min,          z_min],
            [x_min + cube_x, y_min + cube_y, z_min],
            [x_min,          y_min + cube_y, z_min],
            [x_min,          y_min,          z_min + cube_z],
            [x_min + cube_x, y_min,          z_min + cube_z],
            [x_min + cube_x, y_min + cube_y, z_min + cube_z],
            [x_min,          y_min + cube_y, z_min + cube_z],
        ], dtype=np.float32)

        pts_list.append(pts)
        start_idx = i * 8
        # VTK cell format: [8, i0,i1,...,i7]
        cells.append(np.hstack([[8], np.arange(start_idx, start_idx + 8)]))
        cell_types.append(pv.CellType.HEXAHEDRON)

    pts_array = np.vstack(pts_list)
    cells_array = np.hstack(cells).astype(np.int64)
    cell_types = np.array(cell_types)

    grid = pv.UnstructuredGrid(cells_array, cell_types, pts_array)
    # cell_data 长度应等于体素数
    grid.cell_data["values"] = colors

    # --------------------
    # 绘图
    # --------------------
    pl = pv.Plotter()
    pl.add_mesh(grid, scalars="values", opacity=opacity, show_edges=False, cmap='viridis')

    # 添加 DFN 三角片（红色半透明）
    for mesh in mesh_list:
        pts = mesh["points"].astype(np.float32)
        faces = mesh["faces"]
        if faces is not None and len(faces) > 0:
            faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
            tri_mesh = pv.PolyData(pts, faces_pv)
        else:
            tri_mesh = pv.PolyData(pts)
        pl.add_mesh(tri_mesh, color='red', opacity=0.6, show_edges=True)

    pl.add_text(f"show_all={show_all}, near_dist={near_dist}, eff_thresh={locals().get('eff_thresh', None)}", position='upper_left')
    pl.show()

# ------------------------------
# 主程序
# ------------------------------
def find_dfn_files(dfn_dir):
    """扫描目录，寻找 block_X?_Y?_fractures.npy 文件"""
    pattern = re.compile(r"block_X(\d+)_Y(\d+)_fractures\.npy$")
    matched_files = []
    for f in os.listdir(dfn_dir):
        m = pattern.match(f)
        if m:
            matched_files.append(os.path.join(dfn_dir, f))
    return matched_files


if __name__=="__main__":
    dfn_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容三/训练用数据/裂缝片表示"
    trace_header_file = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv"
    out_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容三/训练用数据/体素表示"
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. 找到所有 block_X?_Y?_fractures.npy 文件
    # --------------------------------------------------------
    dfn_files = find_dfn_files(dfn_dir)

    print(f"找到 {len(dfn_files)} 个 DFN mesh 文件")

    # 加载 XY 网格
    df = pd.read_csv(trace_header_file)
    unique_x = np.sort(df['X'].unique())
    unique_y = np.sort(df['Y'].unique())

    block_size = 25

    # --------------------------------------------------------
    # 2. 遍历每个 DFN mesh 文件并生成 voxel 数据
    # --------------------------------------------------------
    for dfn_path in dfn_files:
        fname = os.path.basename(dfn_path)

        # 解析 block_X?_Y?
        m = re.match(r"block_X(\d+)_Y(\d+)_fractures\.npy$", fname)
        block_x = int(m.group(1))
        block_y = int(m.group(2))

        print(f"\n>>> 处理 {fname} (X={block_x}, Y={block_y})")

        # ------------------- 加载 DFN mesh -------------------
        mesh_list = load_dfn_mesh(dfn_path)

        # ------------------- 构建 voxel grid -------------------
        start_x = block_x * (block_size - 1)
        end_x   = start_x + block_size
        start_y = block_y * (block_size - 1)
        end_y   = start_y + block_size

        # 顶点
        x_vertices = np.linspace(unique_x[start_x], unique_x[end_x - 1], block_size, dtype=np.float32)
        y_vertices = np.linspace(unique_y[start_y], unique_y[end_y - 1], block_size, dtype=np.float32)
        z_vertices = np.linspace(1100, 3800, 2701, dtype=np.float32)

        # 中心点
        x_centers = 0.5 * (x_vertices[:-1] + x_vertices[1:])
        y_centers = 0.5 * (y_vertices[:-1] + y_vertices[1:])
        z_centers = 0.5 * (z_vertices[:-1] + z_vertices[1:])

        # ------------------- DFN → voxel -------------------
        sdf, soft_occ, normals = mesh_to_voxel_fields(
            mesh_list, x_centers, y_centers, z_centers
        )

        # ------------------- 保存 5 通道 voxel -------------------
        voxel_5ch = np.zeros((len(x_centers), len(y_centers), len(z_centers), 5), dtype=np.float32)
        voxel_5ch[..., 0] = sdf
        voxel_5ch[..., 1] = soft_occ
        voxel_5ch[..., 2:5] = normals

        out_file = os.path.join(
            out_dir, f"voxel_block_X{block_x}_Y{block_y}.npy"
        )
        np.save(out_file, voxel_5ch)

        print(f"✓ 已保存: {out_file}")

    print("\n全部 DFN → voxel 转换完成！")
