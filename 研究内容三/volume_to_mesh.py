# ------------------------------------------------------------
# 作用：体素转裂缝
# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
voxel_to_dfn_fixed.py
功能：
1. 将 5 通道体素数据 (SDF, SoftOcc, Normals) 转为裂缝片 DFN
2. 分块处理（带重叠），避免块边界丢失薄裂缝
3. 法向插值平滑
4. 顶点合并（基于并查集）
5. 可视化（PyVista）
"""
import os
import re
import numpy as np
import pyvista as pv
from skimage import measure
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from collections import deque
from numba import njit, prange



# ------------------------------
# 功能函数模块
# ------------------------------
def load_voxel(voxel_file):
    """加载 5 通道体素数据"""
    return np.load(voxel_file)

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

import numpy as np
from scipy.ndimage import gaussian_filter
from math import ceil
from skimage import measure

# -----------------------
# 辅助函数
# -----------------------
def normalize_vec(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

# -----------------------
# 1) 将 mask + normals 嵌入为带方向的 soft occupancy（或 SDF）
# -----------------------
def embed_oriented_fracture_voxel(
    mask,                    # (nx,ny,nz) bool or {0,1}
    normals_voxel,           # (nx,ny,nz,3) float, 每 voxel 的 unit normal (或未归一化向量)
    x_centers, y_centers, z_centers,
    half_width_vox=1.0,      # 每侧影响半宽（以 voxel 单位计），可非整数
    sigma_in_vox=0.7,        # 高斯软占据 sigma（以 voxel 单位）
    mode="soft",             # "soft" (gaussian soft occ) or "binary"
    smoothing_sigma=None,    # 可选：对 normals 做预平滑（float, in vox）
    max_samples=None         # 若体素很多，可以限制采样数量（None=全部）
):
    """
    输出 soft_occ (nx,ny,nz) in [0,1]，以及 optional sdf (距离场)。
    主要思想：对每个 mask 为 True 的 voxel p0，按其法向 n 在邻域内画一个 thin plate：
        distance = abs((p - p0) . n)  （距离到平面）
        soft_occ += exp(-(distance/ (sigma*voxel_size) )**2)  (用 max 而不是累加)
    参数说明：
      - half_width_vox: 薄片厚度的一半（以 voxel 数量计），影响邻域范围 (邻域 radius = ceil(half_width_vox + 3*sigma))
      - sigma_in_vox: 高斯 sigma（以 voxel 单位），若 mode="binary" 则忽略 sigma，采用二值内涵
      - smoothing_sigma: 若 normals 噪声大，先做 gaussian_filter(normals_channel, sigma=smoothing_sigma) 再归一化
    返回:
      soft_occ: float32 array same shape as mask, 范围 0..1
      sdf (optional): 如果需要也可返回距离场（这里返回 None，若需要我可以改）
    注意：mask, normals_voxel 的索引顺序为 (nx,ny,nz) 对应 x_centers,y_centers,z_centers
    """

    nx, ny, nz = mask.shape
    # voxel spacing in physical units (not used in distance calc here — we work in vox units)
    dx = (x_centers[1] - x_centers[0]) if len(x_centers) > 1 else 1.0
    dy = (y_centers[1] - y_centers[0]) if len(y_centers) > 1 else 1.0
    dz = (z_centers[1] - z_centers[0]) if len(z_centers) > 1 else 1.0

    # optionally smooth normals voxel to remove noisy flips
    if smoothing_sigma is not None and smoothing_sigma > 0 and normals_voxel is not None:
        from scipy.ndimage import gaussian_filter
        nv = normals_voxel.copy().astype(np.float32)
        for c in range(3):
            gaussian_filter(nv[..., c], sigma=smoothing_sigma, output=nv[..., c])
        # re-normalize
        norms = np.linalg.norm(nv, axis=3, keepdims=True)
        norms[norms == 0] = 1.0
        normals_voxel = (nv / norms).astype(np.float32)

    # prepare output
    soft_occ = np.zeros((nx, ny, nz), dtype=np.float32)

    # build list of fracture voxels
    inds = np.array(np.nonzero(mask)).T  # shape (M,3), order (ix,iy,iz)
    M = len(inds)
    if M == 0:
        return soft_occ, None

    # optional sample if too many
    if max_samples is not None and M > max_samples:
        sel = np.linspace(0, M-1, max_samples).astype(int)
        inds = inds[sel]
        M = len(inds)

    # compute radius in voxels to consider for each seed
    # use half_width + 3*sigma for safety
    r = int(ceil(half_width_vox + 3.0 * sigma_in_vox))

    # Precompute coordinate arrays for speed when handling a single block
    # We'll loop per seed and update small block window.
    Xcoords = x_centers
    Ycoords = y_centers
    Zcoords = z_centers

    # Convert half_width & sigma to physical units? We keep calculations in voxel units (index-space),
    # so we use neighbor ranges in index coords, and distance to plane computed using physical coords.
    # For more accuracy we compute neighbor coordinates in physical space.

    # Precompute voxel centers grid along each axis for block sampling when needed
    # But to keep memory small we compute on the fly per-block.

    for idx, (ix, iy, iz) in enumerate(inds):
        # get normal at this voxel
        if normals_voxel is None:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            n = normals_voxel[ix, iy, iz].astype(np.float32)
            n = normalize_vec(n)
            if np.linalg.norm(n) < 1e-8:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # block index ranges (clamp to grid)
        ix0 = max(0, ix - r); ix1 = min(nx - 1, ix + r)
        iy0 = max(0, iy - r); iy1 = min(ny - 1, iy + r)
        iz0 = max(0, iz - r); iz1 = min(nz - 1, iz + r)

        # coords of seed center in physical space
        x0 = Xcoords[ix]
        y0 = Ycoords[iy]
        z0 = Zcoords[iz]
        p0 = np.array([x0, y0, z0], dtype=np.float32)

        # sample grid for the block
        xs = Xcoords[ix0:ix1+1]
        ys = Ycoords[iy0:iy1+1]
        zs = Zcoords[iz0:iz1+1]

        # create meshgrid in physical coords (small block)
        # Note: memory for small blocks is limited (r typically small like 1-4)
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')  # shape (nx_block, ny_block, nz_block)
        pts = np.stack([XX, YY, ZZ], axis=-1)  # (...,3)

        # compute point-to-plane distance: abs((p - p0) . n)
        diff = pts - p0
        dist_signed = np.tensordot(diff, n, axes=([3], [0]))  # shape (nx_block,ny_block,nz_block)
        dist = np.abs(dist_signed)

        # apply thin-plate mask or soft gaussian
        if mode == "binary":
            mask_local = dist <= (half_width_vox * max(dx, dy, dz))
            # set occupancy to 1.0
            soft_occ[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1][mask_local] = 1.0
        else:  # soft
            # convert sigma_in_vox to physical units: sigma_phys = sigma_in_vox * max(dx,dy,dz) (approx)
            voxel_scale = max(dx, dy, dz)
            sigma_phys = sigma_in_vox * voxel_scale
            # compute gaussian soft value
            vals = np.exp(-(dist**2) / (2.0 * (sigma_phys**2 + 1e-12))).astype(np.float32)
            # only keep values that are within half_width distance window (optional)
            vals[dist > (half_width_vox * voxel_scale)] = 0.0
            # merge by max to avoid over-saturation
            # note: need to map vals indices to global indices
            target = soft_occ[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1]
            np.maximum(target, vals, out=target)
            soft_occ[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1] = target

    # optional: global smooth of result to remove tiny holes
    # soft_occ = gaussian_filter(soft_occ, sigma=0.2)

    return soft_occ, None    # returning sdf=None for now (can be added)


def rotation_matrix_from_vectors(a, b):
    """
    返回 3x3 旋转矩阵 R，使 R @ a = b  (a,b 都应为单位向量)
    如果 a ≈ b，返回 identity。
    如果 a ≈ -b，选择任意垂直轴做 180° 旋转。
    """
    a = normalize(a)
    b = normalize(b)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return np.eye(3, dtype=np.float32)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot > 0.999999:
        return np.eye(3, dtype=np.float32)
    if dot < -0.999999:
        # 180 degree rotation: find an orthogonal vector
        orth = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(a[0]) > 0.9:
            orth = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = normalize(np.cross(a, orth))
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + 2.0 * (K @ K)
        return R
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + K + K @ K * ((1 - dot) / (s * s + 1e-16))
    return R

def connected_components_from_faces(n_vertices, faces):
    """
    返回连通分量（顶点索引列表的列表）——基于顶点共享面。
    快速实现：构建顶点邻接列表，然后 BFS。
    """
    adj = [[] for _ in range(n_vertices)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)

    visited = np.zeros(n_vertices, dtype=bool)
    components = []
    for i in range(n_vertices):
        if visited[i]:
            continue
        # BFS
        q = deque([i])
        comp = []
        visited[i] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        components.append(comp)
    return components

def marching_cubes_blocks(mask, normals_voxel=None,
                          x_centers=None, y_centers=None, z_centers=None,
                          block_size=50, threshold=0.05):
    """
    mask: 体素 mask (True=裂缝)
    normals_voxel: 对应体素法向
    x_centers, y_centers, z_centers: 全局坐标
    """
    nx, ny, nz = mask.shape
    verts_all_list = []
    faces_all_list = []
    face_offset = 0

    dx = np.mean(np.diff(x_centers))
    dy = np.mean(np.diff(y_centers))
    dz = np.mean(np.diff(z_centers))

    for ix0 in range(0, nx, block_size):
        ix1 = min(ix0 + block_size, nx)
        for iy0 in range(0, ny, block_size):
            iy1 = min(iy0 + block_size, ny)
            for iz0 in range(0, nz, block_size):
                iz1 = min(iz0 + block_size, nz)

                sub_mask = mask[ix0:ix1, iy0:iy1, iz0:iz1]

                if np.count_nonzero(sub_mask) == 0:
                    continue

                try:
                    verts, faces, _, _ = measure.marching_cubes(
                        sub_mask.astype(np.float32),
                        level=threshold,
                        spacing=(dx, dy, dz)
                    )
                except:
                    continue

                # 转换到全局坐标
                verts[:, 0] += x_centers[ix0]
                verts[:, 1] += y_centers[iy0]
                verts[:, 2] += z_centers[iz0]

                # ---- 可选：应用方向嵌入微调顶点位置 ----
                if normals_voxel is not None:
                    for i, v in enumerate(verts):
                        ix = min(int(round((v[0] - x_centers[0]) / dx)), nx - 1)
                        iy = min(int(round((v[1] - y_centers[0]) / dy)), ny - 1)
                        iz = min(int(round((v[2] - z_centers[0]) / dz)), nz - 1)

                        n = normals_voxel[ix, iy, iz]
                        if np.linalg.norm(n) > 1e-6:
                            n = n / np.linalg.norm(n)
                            # 沿法向微调顶点（幅度可调，如 0.5 个 voxel 尺寸）
                            verts[i] += n * 0.5 * max(dx, dy, dz)

                verts_all_list.append(verts)
                faces_all_list.append(faces + face_offset)
                face_offset += verts.shape[0]

    if len(verts_all_list) == 0:
        print("WARNING: marching cubes 未生成 mesh")
        return np.empty((0, 3)), np.empty((0, 3))

    verts_all = np.vstack(verts_all_list)
    faces_all = np.vstack(faces_all_list)
    return verts_all, faces_all

@njit(parallel=True)
def embed_oriented_soft_volume_numba(soft_occ, normals_voxel, half_width_vox=1.0, sigma_vox=0.7):
    """
    带方向的软体素场（裂缝薄片）——Numba 并行优化版本

    soft_occ: (nx, ny, nz) 原始 soft occupancy
    normals_voxel: (nx, ny, nz, 3) 裂缝法向
    half_width_vox: 薄片半宽（体素单位）
    sigma_vox: 高斯扩散参数
    """
    nx, ny, nz = soft_occ.shape
    oriented = np.zeros_like(soft_occ, dtype=np.float32)

    R = int(np.ceil(half_width_vox + 3 * sigma_vox))

    # 找出所有裂缝 voxel
    coords_list = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if soft_occ[ix, iy, iz] <= 0.3:
                    coords_list.append((ix, iy, iz))

    # 转为 numba 支持的数组
    coords = np.array(coords_list, dtype=np.int32)
    n_voxels = coords.shape[0]

    for idx in prange(n_voxels):
        ix, iy, iz = coords[idx]
        n = normals_voxel[ix, iy, iz]
        norm = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if norm < 1e-6:
            continue
        n0 = n[0] / norm
        n1 = n[1] / norm
        n2 = n[2] / norm

        # 遍历邻域
        for dx in range(-R, R + 1):
            x = ix + dx
            if x < 0 or x >= nx:
                continue
            for dy in range(-R, R + 1):
                y = iy + dy
                if y < 0 or y >= ny:
                    continue
                for dz in range(-R, R + 1):
                    z = iz + dz
                    if z < 0 or z >= nz:
                        continue
                    dist = abs(dx * n0 + dy * n1 + dz * n2)
                    val = np.exp(-(dist / sigma_vox) ** 2)
                    if val > oriented[x, y, z]:
                        oriented[x, y, z] = val
    return oriented

def merge_vertices(verts, merge_radius=1.5):
    """
    基于并查集合并顶点簇（距离 <= merge_radius 的顶点合并为一个点）。
    返回 new_verts (K,3) 和 mapping (N,) 使得 new_verts[mapping[i]] 是 verts[i] 的合并后顶点。
    """
    N = verts.shape[0]
    if N == 0:
        return verts, np.arange(N, dtype=np.int64)

    tree = cKDTree(verts)
    groups = tree.query_ball_tree(tree, r=merge_radius)

    # 并查集
    parent = np.arange(N, dtype=np.int64)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra = find(a); rb = find(b)
        if ra == rb: return
        parent[rb] = ra

    for i, neigh in enumerate(groups):
        for j in neigh:
            if j <= i:  # 避免重复
                continue
            union(i, j)

    # 生成映射到根
    roots = np.array([find(i) for i in range(N)], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    # 为每个组计算平均坐标作为新顶点
    new_verts = np.zeros((len(unique_roots), 3), dtype=np.float32)
    counts = np.zeros(len(unique_roots), dtype=np.int64)
    for i, root_idx in enumerate(inverse):
        new_verts[root_idx] += verts[i]
        counts[root_idx] += 1
    new_verts /= counts.reshape(-1,1)

    mapping = inverse.astype(np.int64)  # length N -> index in new_verts
    return new_verts, mapping

def interpolate_normals(verts, normals_voxel, x_centers, y_centers, z_centers):
    """利用 voxel 法向进行插值平滑"""
    nx, ny, nz, _ = normals_voxel.shape

    dx = x_centers[1] - x_centers[0]
    dy = y_centers[1] - y_centers[0]
    dz = z_centers[1] - z_centers[0]

    grid_x = np.arange(nx)
    grid_y = np.arange(ny)
    grid_z = np.arange(nz)

    interp_x = RegularGridInterpolator((grid_x, grid_y, grid_z), normals_voxel[...,0], bounds_error=False, fill_value=0)
    interp_y = RegularGridInterpolator((grid_x, grid_y, grid_z), normals_voxel[...,1], bounds_error=False, fill_value=0)
    interp_z = RegularGridInterpolator((grid_x, grid_y, grid_z), normals_voxel[...,2], bounds_error=False, fill_value=0)

    # 将 verts 转换为体素索引坐标（浮点）
    ix = (verts[:,0] - x_centers[0]) / dx
    iy = (verts[:,1] - y_centers[0]) / dy
    iz = (verts[:,2] - z_centers[0]) / dz
    pts_idx = np.stack([ix, iy, iz], axis=-1)

    normals_interp = np.stack([
        interp_x(pts_idx),
        interp_y(pts_idx),
        interp_z(pts_idx)
    ], axis=-1)

    norms = np.linalg.norm(normals_interp, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals_interp /= norms
    return normals_interp

def save_dfn(verts, faces, normals, out_file, mesh_type="expanded"):
    """
    保存 DFN 文件（numpy），参考原始 DFN 文件格式。
    每个元素是一个 dict，包括:
        'points': 顶点 (N,3)
        'faces': 三角面 (M,3)
        'type': 'well' 或 'expanded' 等
        'center': 顶点中心点 (3,)
        'intensity': 平均 soft_occ 或 0.0
    """
    center = verts.mean(axis=0).astype(np.float32)
    dfn_entry = {
        'points': verts.astype(np.float32),
        'faces': faces.astype(np.int64),
        'type': mesh_type,
        'center': center,
        'intensity': 0.0
    }
    # 可以考虑把 normals 也存进去，如果后续需要
    dfn_entry['normals'] = normals.astype(np.float32)

    # 保存成 numpy object 数组
    np.save(out_file, np.array([dfn_entry], dtype=object))
    print(f"已保存 DFN 文件: {out_file}，顶点数量: {verts.shape[0]} 面数量: {faces.shape[0]}")


def visualize_dfn(verts, faces, normals=None, show_edges=True, smooth_shading=True):
    """
    使用 PyVista 可视化 DFN mesh。
    verts: (N,3) 顶点坐标
    faces: (M,3) 三角面索引
    normals: (N,3) 法向，可选
    """
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        print("警告：没有顶点或面，无法可视化。")
        return

    # PyVista faces 格式：每个三角形前加一个 3
    faces_pv = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])

    mesh = pv.PolyData(verts.astype(np.float32), faces_pv)

    # 设置法向（如果提供）
    if normals is not None and normals.shape[0] == verts.shape[0]:
        # 使用 set_array 安全设置
        mesh.point_data.set_array(normals.astype(np.float32), "Normals")
        # 激活 normals
        try:
            mesh.point_data.active_normals_name = "Normals"
        except AttributeError:
            # 旧版本 PyVista 可能没有 active_normals_name
            mesh.compute_normals(inplace=True)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=None, show_edges=show_edges, smooth_shading=smooth_shading)
    plotter.show()

# ------------------------------
# main 入口
# ------------------------------
def find_voxel_files(voxel_dir):
    """
    支持两种格式：
    voxel_block_X69_Y29.npy
    voxel_pred_X14_Y21.npy
    """
    pattern = re.compile(r"(voxel_block|voxel_pred)_X(\d+)_Y(\d+)\.npy$")
    files = []
    for f in os.listdir(voxel_dir):
        if pattern.match(f):
            files.append(os.path.join(voxel_dir, f))
    return files


if __name__ == "__main__":
    # voxel_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\训练用数据\体素表示"
    # out_dfn_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\训练用数据\转化后DFN"
    voxel_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\3D_GAN输出\体素文件"
    out_dfn_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\3D_GAN输出\裂缝片转化"
    os.makedirs(out_dfn_dir, exist_ok=True)

    # marching cubes 参数
    # X范围，Y范围，Z范围
    unit_size = [0, 300, 0, 300, 1100, 3800]
    sdf_threshold = 0.5
    block_size = 50
    merge_radius = 1.5
    visualize = False
    # visualize = True

    # ---------------------------
    # 1. 搜索 voxel 文件
    # ---------------------------
    voxel_files = find_voxel_files(voxel_dir)
    print(f"共找到 {len(voxel_files)} 个 voxel 文件")
    # 单文件转化
    # voxel_files = [r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\3D_GAN输出\体素文件\voxel_pred_X0_Y0.npy"]
    # voxel_files = [r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\训练用数据\体素表示\voxel_block_X34_Y25.npy"]
    visualize = True

    # ---------------------------
    # 遍历所有 voxel_block_XY.npy
    # ---------------------------
    for voxel_file in voxel_files:

        fname = os.path.basename(voxel_file)
        print(f"\n>>> 正在处理 voxel 文件: {fname}")

        # 解析 X Y
        m = re.match(r"(voxel_block|voxel_pred)_X(\d+)_Y(\d+)\.npy$", fname)
        block_x = int(m.group(2))
        block_y = int(m.group(3))

        # ---------------------------
        # 加载 voxel 数据
        # ---------------------------
        voxel_5ch = load_voxel(voxel_file)

        sdf = voxel_5ch[..., 0]
        soft_occ = voxel_5ch[..., 1]
        normals_voxel = voxel_5ch[..., 2:5]

        # 体素尺寸 (24 × 24 × 2700)
        nx, ny, nz = sdf.shape

        # 重新计算中心点坐标
        x_centers = np.linspace(unit_size[0], unit_size[1], nx, dtype=np.float32)
        y_centers = np.linspace(unit_size[2], unit_size[3], ny, dtype=np.float32)
        z_centers = np.linspace(unit_size[4], unit_size[5], nz, dtype=np.float32)

        # ---------------------------
        # marching cubes
        # ---------------------------
        # === 必须加入：方向嵌入 ===
        # soft_oriented = embed_oriented_soft_volume_numba(soft_occ, normals_voxel, half_width_vox=1.0, sigma_vox=0.7)

        # 用定向 soft-occ 生成 mask
        # mask = soft_oriented < 0.3
        mask = soft_occ < 0.3

        print("mask voxel count:", np.count_nonzero(mask))

        verts, faces = marching_cubes_blocks(
            mask.astype(np.float32),
            normals_voxel=normals_voxel,
            x_centers=x_centers, y_centers=y_centers, z_centers=z_centers,
            block_size=block_size,
            threshold=sdf_threshold
        )

        if verts.shape[0] == 0:
            print(f"⚠ 未生成 mesh，跳过 block X{block_x} Y{block_y}")
            continue

        # ---------------------------
        # 顶点合并
        # ---------------------------
        new_verts, mapping = merge_vertices(verts, merge_radius=merge_radius)
        faces_merged = np.array([[mapping[i] for i in f] for f in faces], dtype=np.int64)

        # ---------------------------
        # 法向插值
        # ---------------------------
        normals_interp = interpolate_normals(
            new_verts, normals_voxel, x_centers, y_centers, z_centers
        )

        # ---------------------------
        # 保存 DFN 文件
        # ---------------------------
        out_name = f"rebuild_dfn_block_X{block_x}_Y{block_y}.npy"
        out_file = os.path.join(out_dfn_dir, out_name)

        save_dfn(new_verts, faces_merged, normals_interp, out_file, mesh_type="expanded")
        print(f"✓ 已保存 DFN: {out_file}")

        # ---------------------------
        # 可视化（可选）
        # ---------------------------
        if visualize:
            visualize_dfn(new_verts, faces_merged, normals_interp)

    print("\n全部 voxel → DFN 转换完成！")