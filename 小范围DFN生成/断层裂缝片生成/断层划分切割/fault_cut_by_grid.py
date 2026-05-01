# -*- coding: utf-8 -*-
"""
fault_cut_by_grid_nocli.py
按300m网格切割断层面并导出patch几何与属性
直接运行，不再需要命令行参数。
"""

import os
import re
import csv
import numpy as np
import pyvista as pv
from collections import defaultdict
from scipy.spatial import Delaunay

# =========================
# 用户需要修改的输入参数
# =========================
sticks_path = r"/data/shared/project-oil/wx数据/砂砾岩/断层/FaultStick_0604.dat"  # 输入文件
origin_x = 556150.0  # 网格原点X（与你们300m网格对齐）
origin_y = 4193975.0  # 网格原点Y
outdir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out"  # 输出目录

cell_size = 300.0  # 单元边长
zpad = 5000.0  # Z向加厚
min_area = 10.0  # 剔除小碎片面积阈值
save_obj = False  # 是否同时导出OBJ
preview = True  # 是否最后打开可视化窗口


# =========================
# 1) 读取 sticks
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
                x = float(parts[0]);
                y = float(parts[1]);
                t = float(parts[2])
                sign = int(parts[3]);
                fault = parts[4]
                inline = int(parts[5]);
                xline = int(parts[6])
                records.append((x, y, t, sign, fault, inline, xline))
            except:
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
# 2) 切割
# =========================
def make_cell_index(x, y, origin_x, origin_y, cell):
    i = int((x - origin_x) // cell)
    j = int((y - origin_y) // cell)
    return i, j


def split_mesh_by_grid(mesh: pv.PolyData, origin_xy, cell=300.0, zpad=5000.0, min_area=10.0):
    """
    用与DFN网格对齐的竖直盒，在3D中裁出每个cell内的断层子面。
    返回: dict[(i,j) -> pv.PolyData] （保证是 PolyData）
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    ox, oy = origin_xy

    i0, j0 = make_cell_index(xmin, ymin, ox, oy, cell)
    i1, j1 = make_cell_index(xmax, ymax, ox, oy, cell)

    patches = {}
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            bx0 = ox + i * cell
            bx1 = bx0 + cell
            by0 = oy + j * cell
            by1 = by0 + cell
            box = pv.Box(bounds=(bx0, bx1, by0, by1, zmin - zpad, zmax + zpad))

            # 裁剪得到 UnstructuredGrid
            sub = mesh.clip_box(box, invert=False)
            if sub.n_cells > 0:
                # 转回 PolyData 表面
                sub = sub.extract_surface().clean().triangulate()
                if isinstance(sub, pv.PolyData) and sub.n_cells > 0 and sub.area >= float(min_area):
                    patches[(i, j)] = sub
    return patches


def save_patch_mesh(ds, out_stem: str, save_obj=False):
    """
    ds: 可能是 PolyData 或 UnstructuredGrid
    out_stem: 不含扩展名的输出路径
    """
    if isinstance(ds, pv.PolyData):
        path = out_stem + ".vtp"
        ds.save(path)
        if save_obj:
            ds.save(out_stem + ".obj")
        return path
    else:
        # 兜底：非 PolyData 时保存为 .vtu
        path = out_stem + ".vtu"
        ds.save(path)
        return path


# =========================
# 3) 属性统计
# =========================
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
# 4) 导出
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p


def cut_and_export_all(meshes: dict, origin_xy, outdir,
                       cell=300.0, zpad=5000.0, min_area=10.0, save_obj=False):
    patches_root = ensure_dir(os.path.join(outdir, "patches"))
    all_rows = []

    for fault_name, mesh in meshes.items():
        subdir = ensure_dir(os.path.join(patches_root, fault_name))
        patches = split_mesh_by_grid(mesh, origin_xy, cell=cell, zpad=zpad, min_area=min_area)

        for (i, j), sub in patches.items():
            stats = compute_patch_stats(sub)
            row = dict(fault_name=fault_name, cell_i=i, cell_j=j, **stats)
            all_rows.append(row)

            # 用不带扩展名的 stem，save_patch_mesh 会自动加 .vtp 或 .vtu
            stem = os.path.join(subdir, f"{fault_name}__i{i}_j{j}")
            save_patch_mesh(sub, stem, save_obj=save_obj)

    return all_rows


def write_csv(rows, out_csv):
    if not rows: return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys);
        w.writeheader();
        w.writerows(rows)


# =========================
# 5) 渲染时为每个面片分配不同颜色
# =========================
def generate_random_colors(num):
    """
    随机生成指定数量的颜色
    """
    return np.random.rand(num, 3)


# =========================
# 6) 主程序
# =========================
if __name__ == "__main__":
    ensure_dir(outdir)
    print("[INFO] 构建断层三角网 …")
    meshes = build_fault_surfaces(sticks_path)
    print(f"[INFO] 共读取断层 {len(meshes)} 条")
    print("[INFO] 切割并导出 …")
    rows = cut_and_export_all(meshes, (origin_x, origin_y), outdir,
                              cell=cell_size, zpad=zpad, min_area=min_area, save_obj=save_obj)
    write_csv(rows, os.path.join(outdir, "fault_patches_summary.csv"))
    print("[OK] 导出完成")

    if preview:
        pl = pv.Plotter()
        patch_colors = generate_random_colors(len(rows))  # 为每个面片分配颜色
        for idx, row in enumerate(rows):
            fn = os.path.join(outdir, "patches", row["fault_name"],
                              f"{row['fault_name']}__i{row['cell_i']}_j{row['cell_j']}.vtp")
            poly = pv.read(fn)
            # 使用随机颜色渲染
            pl.add_mesh(poly, color=patch_colors[idx], opacity=0.6)
        pl.show()
