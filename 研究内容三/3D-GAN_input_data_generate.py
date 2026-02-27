# ------------------------------------------------------------
# 作用：单元地震数据分割
# prepare_3dgan_data.py
# SGY → 三维地震体 → 切块 → 3D-GAN训练样本
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree


# ------------------------------------------------------------
# 1. 提取 32×32 XY 区块
# ------------------------------------------------------------
def extract_block_xy(df, block_x, block_y, block_size=32):
    print(f"=== 提取 XY 区块 bx={block_x}, by={block_y} ===")

    unique_x = np.sort(df["X"].unique())
    unique_y = np.sort(df["Y"].unique())

    sx = block_x * (block_size - 1)
    ex = sx + block_size
    sy = block_y * (block_size - 1)
    ey = sy + block_size

    if ex > len(unique_x) or ey > len(unique_y):
        print("⚠ 区块越界，跳过")
        return None

    target_x = unique_x[sx:ex]
    target_y = unique_y[sy:ey]

    grid_df = df[df["X"].isin(target_x) & df["Y"].isin(target_y)]
    grid_df = grid_df.sort_values(["X", "Y"]).reset_index(drop=True)

    print(f"  → {len(grid_df)} traces extracted")
    return grid_df


# ------------------------------------------------------------
# 2. 提取 SGY 波形
# ------------------------------------------------------------
def extract_sgy_data(sgy_file, trace_df):
    trace_indices = trace_df["TraceIdx"].tolist()
    x_coords = trace_df["X"].to_numpy()
    y_coords = trace_df["Y"].to_numpy()

    with segyio.open(sgy_file, "r", ignore_geometry=True) as sgy:
        num_samples = len(sgy.samples)
        amp_data = np.zeros((len(trace_indices), num_samples), dtype=np.float32)

        for i, idx in enumerate(trace_indices):
            amp_data[i] = sgy.trace[idx]
            if (i + 1) % 200 == 0:
                print(f"  进度：{i + 1}/{len(trace_indices)}")

    print(f"✓ 提取完成：{amp_data.shape[0]} 道 × {amp_data.shape[1]} 采样点")
    return x_coords, y_coords, amp_data


# ------------------------------------------------------------
# 3. 构建三维地震体 (X, Y, Z)
# ------------------------------------------------------------
def create_3d_seismic(x_coords, y_coords, amp_data,
                      start_time, sample_interval):
    print("=== 构建三维地震体 ===")

    z_coords = start_time + np.arange(amp_data.shape[1]) * sample_interval
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)

    seismic = np.zeros((len(x_unique), len(y_unique), len(z_coords)), dtype=np.float32)

    tree = cKDTree(np.column_stack([x_coords, y_coords]))

    for i, xi in enumerate(x_unique):
        for j, yj in enumerate(y_unique):
            _, idx = tree.query([xi, yj], k=1)
            seismic[i, j] = amp_data[idx]

    print(f"✓ seismic cube 维度：{seismic.shape}")
    return seismic, x_unique, y_unique, z_coords


# ------------------------------------------------------------
# 4. 切分 3D block (32×32×64)
# ------------------------------------------------------------
def crop_block_for_gan(seismic_3d,
                       block_x,
                       block_y,
                       block_z,
                       bx=32,
                       by=32,
                       bz=64):
    x_start = block_x * (bx - 1)
    y_start = block_y * (by - 1)
    z_start = block_z * (bz // 2)

    x_end = x_start + bx
    y_end = y_start + by
    z_end = z_start + bz

    if x_end > seismic_3d.shape[0] or \
            y_end > seismic_3d.shape[1] or \
            z_end > seismic_3d.shape[2]:
        return None

    block = seismic_3d[x_start:x_end, y_start:y_end, z_start:z_end]
    return block


# ------------------------------------------------------------
# 5. 主流程：SGY → 立方体 → block → 保存 .npy
# ------------------------------------------------------------
def process_sgy_for_gan(
        trace_header_csv,
        sgy_file,
        out_dir="3dgan_data",
        block_xy_size=25,   # DFN 使用 25 个点 → 24 个 voxel
        block_z=2701,
        start_time=0,
        sample_interval=2):

    os.makedirs(f"{out_dir}/seismic", exist_ok=True)

    print("=== 加载道头文件 ===")
    df = pd.read_csv(trace_header_csv)
    print(f"✓ 道头数量：{len(df)}")

    # 保证 XY 是唯一且排序的
    x_unique = np.sort(df["X"].unique())
    y_unique = np.sort(df["Y"].unique())

    max_bx = len(x_unique) // (block_xy_size - 1)
    max_by = len(y_unique) // (block_xy_size - 1)
    print(f"max_bx: {max_bx}, max_by: {max_by}")

    for bx in range(max_bx):
        for by in range(max_by):
            bx, by = 73, 28

            block_df = extract_block_xy(df, bx, by, block_xy_size)
            if block_df is None:
                # continue
                print("block_df is None")

            x_coords, y_coords, amp = extract_sgy_data(sgy_file, block_df)

            seismic_3d, xu, yu, zu = create_3d_seismic(
                x_coords, y_coords, amp,
                start_time=start_time,
                sample_interval=sample_interval
            )

            # seismic_3d shape: [25, 25, Z]
            # DFN voxel expects: 24 × 24 × Z
            seismic_3d = seismic_3d[:25, :25, :]

            max_bz = seismic_3d.shape[2] // (block_z // 2)

            for bz in range(max_bz):
                raw_block = seismic_3d[:, :, bz:bz + block_z]

                if raw_block.shape[2] != block_z:
                    continue

                raw_block = raw_block.astype(np.float32)

                # 扩展 channel 维度 → 24×24×64×1
                block = raw_block[..., None]

                out_file = f"{out_dir}/seismic/sample_X{bx}_Y{by}.npy"
                np.save(out_file, block)

                print(f"✓ 保存样本：{out_file}  shape={block.shape}")

                # 打印 block 内容（你要求的）
                print("block 内容示例（前5个值）:", block.flatten()[:5])

            break
        break

    print("=== 3DGAN 地震样本全部生成完毕 ===")


# ------------------------------------------------------------
# main 函数
# ------------------------------------------------------------
if __name__ == "__main__":
    trace_header_csv = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"
    sgy_file = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\psdm_final_time.sgy"
    out_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容三\3D_GAN输入"

    process_sgy_for_gan(
        trace_header_csv,
        sgy_file,
        out_dir=out_dir,
        block_xy_size=25,  # GAN sample 32×32
        block_z=2701,  # Z 方向深度
        start_time=0,
        sample_interval=2  # ms
    )
