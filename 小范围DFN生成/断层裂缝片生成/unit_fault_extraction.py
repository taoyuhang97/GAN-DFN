# -*- coding: utf-8 -*-
"""
fault_cut_by_grid_save_npy.py
基于原脚本的修改版本：按网格切割断层面并**将每个单元内的断层片保存为 .npy 文件**（便于后续 Python 处理）。
同时生成汇总 CSV。保留 preview（渲染）功能，但不再保存 .vtp/.obj。
"""
import os
import re
import csv
import math
import numpy as np
import pyvista as pv
from collections import defaultdict
from scipy.spatial import Delaunay

# ===== tqdm 兼容导入 =====
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x

# =========================
# 用户需要修改的输入参数
# =========================
sticks_path = r"/data/shared/project-oil/wx数据/砂砾岩/断层/FaultStick_0604.dat"  # 输入文件
origin_x = 556150.0  # 网格原点X（与你们300m网格对齐）
origin_y = 4193975.0  # 网格原点Y
outdir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out_npy"  # 输出目录
os.makedirs(outdir, exist_ok=True)

cell_size = 300.0  # 单元边长
zpad = 5000.0  # Z向加厚
min_area = 10.0  # 剔除小碎片面积阈值
save_obj = False  # 不再保存 OBJ（该脚本以 .npy 为主）
preview = True  # 是否最后打开可视化窗口（.npy 会被加载为网格用于预览）

# =========================
# 读取 sticks & 生成断层面（原样）
# =========================
def load_fault_sticks(txt_path):
    records = []
    pat_space = re.compile(r"\s+")
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = pat_space.split(s)
            if len(parts) < 7:
                continue
            try:
                x = float(parts[0]); y = float(parts[1]); t = float(parts[2])
                sign = int(parts[3]); fault = parts[4]
                inline = int(parts[5]); xline = int(parts[6])
                records.append((x, y, t, sign, fault, inline, xline))
            except Exception:
                continue
    return np.array(records, dtype=object)


def group_by_fault(arr):
    groups = defaultdict(lambda: {"points": []})
    for (x, y, t, sign, fault, inline, xline) in arr:
        groups[fault]["points"].append([float(x), float(y), float(t)])
    for f in groups:
        groups[f]["points"] = np.asarray(groups[f]["points"], dtype=float)
    return groups


def triangulate_by_xy(points_xyz):
    if points_xyz is None or len(points_xyz) < 3:
        return None
    xy = points_xyz[:, :2]
    tri = Delaunay(xy)
    simplices = tri.simplices
    faces = np.hstack([np.column_stack([np.full(len(simplices), 3), simplices]).ravel()])
    mesh = pv.PolyData(points_xyz, faces)
    return mesh.clean().triangulate()


def build_fault_surfaces(txt_path):
    data = load_fault_sticks(txt_path)
    groups = group_by_fault(data)
    meshes = {}
    for f, g in groups.items():
        mesh = triangulate_by_xy(g["points"])
        if mesh is not None and mesh.n_cells > 0:
            meshes[f] = mesh
    return meshes

# =========================
# 网格索引、裁剪与属性计算辅助
# =========================
def make_cell_index(x, y, origin_x, origin_y, cell):
    return int(math.floor((x - origin_x) / cell)), int(math.floor((y - origin_y) / cell))


def poly_to_tri_array(poly: pv.PolyData):
    """
    将 pyvista PolyData 的 faces 转换为 (n_faces, 3) 索引数组（三角面）。
    兼容多种 pyvista/VTK 版本的 faces 表示方式，稳健性更好。
    返回 dtype=int32 的形状 (n,3) 数组；若无面则返回 shape (0,3) 的数组。
    """
    # 快速排除：若没有任何单元（面）直接返回空
    if getattr(poly, "n_cells", 0) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # pyvista stores faces as a flat array: [n0, i0, i1, i2, n1, j0, j1, j2, ...]
    faces_flat = getattr(poly, "faces", None)
    if faces_flat is None:
        # 极端退化情况：尝试用 cells API
        try:
            cells = poly.cells
            # cells is also a flat array similar to faces; attempt reshape
            arr = np.asarray(cells, dtype=np.int64)
            if arr.size % 4 == 0:
                try:
                    tri = arr.reshape(-1, 4)[:, 1:4]
                    return tri.astype(np.int32)
                except Exception:
                    return np.zeros((0, 3), dtype=np.int32)
            else:
                return np.zeros((0, 3), dtype=np.int32)
        except Exception:
            return np.zeros((0, 3), dtype=np.int32)

    # 确保为 numpy 数组
    faces_flat = np.asarray(faces_flat, dtype=np.int64)

    # 常见情况： len(flat) == 4 * n_faces
    if faces_flat.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    if faces_flat.size % 4 == 0:
        try:
            tri = faces_flat.reshape(-1, 4)[:, 1:4]
            return tri.astype(np.int32)
        except Exception:
            # 如果 reshape 失败，退回空
            return np.zeros((0, 3), dtype=np.int32)

    # 备用：有时 faces_flat 不是 4*n 格式（极少见），尝试从 cells/Polys 获取三角索引
    try:
        # poly.faces_as_array() 在某些版本可用 —— 否则继续使用 poly.extract_cells
        arr = poly.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
        return arr
    except Exception:
        return np.zeros((0, 3), dtype=np.int32)



def best_fit_normal(points: np.ndarray):
    c = points.mean(axis=0)
    M = points - c
    _, _, vh = np.linalg.svd(M, full_matrices=False)
    n = vh[-1, :]
    return n / (np.linalg.norm(n) + 1e-12)


def strike_dip_from_normal(n: np.ndarray):
    nx, ny, nz = n
    h = np.sqrt(nx * nx + ny * ny) + 1e-12
    strike_rad = np.arctan2(nx, ny)
    strike = (np.degrees(strike_rad) + 360) % 180
    dip = np.degrees(np.arctan2(abs(nz), h))
    return strike, dip


def compute_patch_stats(poly: pv.PolyData):
    area = float(poly.area)
    pts = poly.points
    cx, cy, cz = pts.mean(axis=0).tolist()
    xmin, xmax, ymin, ymax, zmin, zmax = poly.bounds
    n = best_fit_normal(pts)
    strike, dip = strike_dip_from_normal(n)
    return dict(area_3d=area, cx=cx, cy=cy, cz=cz,
                bbox_xmin=xmin, bbox_xmax=xmax,
                bbox_ymin=ymin, bbox_ymax=ymax,
                bbox_zmin=zmin, bbox_zmax=zmax,
                strike_deg=strike, dip_deg=dip)

# =========================
# 主：按 cell 裁剪并保存为 npy（不再保存 vtp）
# =========================
def split_mesh_by_grid_and_save_npy(mesh: pv.PolyData, fault_name: str, origin_xy, outdir,
                                    cell=300.0, zpad=5000.0, min_area=10.0, show_progress=False):
    """
    对单个断层 mesh 按网格裁剪并把每个有效 patch 保存为 .npy（返回 rows 列表）
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    ox, oy = origin_xy
    i0, j0 = make_cell_index(xmin, ymin, ox, oy, cell)
    i1, j1 = make_cell_index(xmax, ymax, ox, oy, cell)

    total = (i1 - i0 + 1) * (j1 - j0 + 1)
    iterator = ((i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))
    if show_progress:
        iterator = tqdm(iterator, total=total, desc=f"裁剪 {fault_name}")

    saved_rows = []
    npy_dir = os.path.join(outdir, "npy", fault_name)
    os.makedirs(npy_dir, exist_ok=True)

    for (i, j) in iterator:
        bx0 = ox + i * cell; bx1 = bx0 + cell
        by0 = oy + j * cell; by1 = by0 + cell
        box = pv.Box(bounds=(bx0, bx1, by0, by1, zmin - zpad, zmax + zpad))

        sub = mesh.clip_box(box, invert=False)
        if sub.n_cells == 0:
            continue

        sub = sub.extract_surface().clean().triangulate()
        if not isinstance(sub, pv.PolyData) or sub.n_cells == 0:
            continue

        if sub.area < float(min_area):
            continue

        # 计算 stats
        stats = compute_patch_stats(sub)

        # 整理 points/faces 保存为 npy
        points = sub.points.astype(np.float32)
        faces = poly_to_tri_array(sub)  # (n_faces,3) int32

        npy_obj = {
            "points": points,
            "faces": faces,
            "fault_name": fault_name,
            "cell_i": int(i),
            "cell_j": int(j),
            "stats": stats
        }

        filename = f"{fault_name}__i{i}_j{j}.npy"
        npy_path = os.path.join(npy_dir, filename)
        # 使用 numpy 保存字典（将使用 pickle 内部存储）
        np.save(npy_path, npy_obj)

        saved_rows.append(dict(fault_name=fault_name, cell_i=int(i), cell_j=int(j),
                               npy_path=os.path.relpath(npy_path, outdir), **stats))

    return saved_rows


def cut_and_export_all_to_npy(meshes: dict, origin_xy, outdir,
                              cell=300.0, zpad=5000.0, min_area=10.0, show_progress=True):
    """
    遍历所有断层，裁剪并保存每个 patch 为 .npy
    返回：all_rows（统计信息字典列表）
    """
    all_rows = []
    items = list(meshes.items())
    if show_progress:
        items = tqdm(items, desc="处理断层")

    for fault_name, mesh in items:
        rows = split_mesh_by_grid_and_save_npy(mesh, fault_name, origin_xy, outdir,
                                               cell=cell, zpad=zpad, min_area=min_area, show_progress=show_progress)
        all_rows.extend(rows)
    return all_rows

def write_csv(rows, out_csv):
    if not rows:
        print("[INFO] 没有生成任何 patch，跳过 CSV 写入。")
        return
    # 固定列顺序：你可以自定义想要的顺序
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] CSV 已保存: {out_csv}")

# =========================
# Preview: 从 npy 载入并可视化（可选）
# =========================
def preview_from_npy(rows, outdir):
    if not rows:
        print("[INFO] 无 patch 可预览。")
        return
    pl = pv.Plotter()
    colors = np.random.rand(len(rows), 3)
    for idx, r in enumerate(tqdm(rows, desc="加载并渲染 patch")):
        npy_path = os.path.join(outdir, r["npy_path"])
        try:
            d = np.load(npy_path, allow_pickle=True).item()
            pts = d["points"]
            faces = d["faces"]
            # pyvista faces 一维格式： [3, i0, i1, i2, ...], 我们需要扁平化
            flat_faces = np.hstack([np.column_stack([np.full(len(faces), 3), faces]).ravel()])
            mesh = pv.PolyData(pts, flat_faces)
            pl.add_mesh(mesh, color=colors[idx], opacity=0.6)
        except Exception as e:
            print(f"[WARN] 无法加载 {npy_path}: {e}")
    pl.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(outdir, exist_ok=True)
    print("[INFO] 构建断层三角网 …")
    meshes = build_fault_surfaces(sticks_path)
    print(f"[INFO] 共读取断层 {len(meshes)} 条")
    print("[INFO] 开始裁剪并保存为 .npy …")

    rows = cut_and_export_all_to_npy(meshes, (origin_x, origin_y), outdir,
                                     cell=cell_size, zpad=zpad, min_area=min_area, show_progress=True)

    csv_path = os.path.join(outdir, "fault_patches_summary_npy.csv")
    write_csv(rows, csv_path)
    print("[OK] 导出完成，汇总 CSV:", csv_path)

    if preview and rows:
        preview_from_npy(rows, outdir)
    elif not rows:
        print("[INFO] 没有生成任何 patch，跳过预览。")
