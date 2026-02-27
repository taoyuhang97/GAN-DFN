# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob

# ===========================================================
# 区块计算逻辑（find_blocks_by_xy_set）
# ===========================================================
def find_blocks_by_xy_set(trace_header_file, xy_points, block_size=25,
                          x_origin=None, y_origin=None):
    """根据井点坐标计算其所在的区块编号"""
    df = pd.read_csv(trace_header_file)
    unique_x = np.sort(df['X'].unique())
    unique_y = np.sort(df['Y'].unique())

    dx = np.mean(np.diff(unique_x))
    dy = np.mean(np.diff(unique_y))
    if np.isnan(dx) or np.isnan(dy):
        raise ValueError("无法计算道间距，请检查trace_header_xy.csv文件中X/Y分布是否正确。")

    if x_origin is None:
        x_origin = unique_x[0]
    if y_origin is None:
        y_origin = unique_y[0]

    results = []
    for (x, y) in xy_points:
        x_idx = int(round((x - x_origin) / dx))
        y_idx = int(round((y - y_origin) / dy))
        bx = x_idx // (block_size - 1)
        by = y_idx // (block_size - 1)
        results.append({'X': x, 'Y': y, 'block_x': bx, 'block_y': by})

    return pd.DataFrame(results)


# ===========================================================
# 主逻辑整合（测井轨迹→裂缝单元归属→地震道提取）
# ===========================================================
def process_all_wells(trace_header_file, xy_input_dir, output_dir,
                      x_origin=None, y_origin=None, block_size=25):
    os.makedirs(output_dir, exist_ok=True)
    xy_files = glob.glob(os.path.join(xy_input_dir, "*.csv"))

    if not xy_files:
        print(f"❌ 未在 {xy_input_dir} 找到任何测井 CSV 文件")
        return

    print(f"共检测到 {len(xy_files)} 个井轨迹文件。")

    for xy_file in xy_files:
        well_name = os.path.basename(xy_file).split('_')[0]
        print(f"\n{'='*70}\n处理测井文件: {well_name}")

        df_xy = pd.read_csv(xy_file)
        if not {'X', 'Y'}.issubset(df_xy.columns):
            print(f"⚠ {well_name} 文件缺少 X/Y 列，跳过。")
            continue

        xy_points = list(zip(df_xy['X'], df_xy['Y']))
        block_df = find_blocks_by_xy_set(
            trace_header_file, xy_points,
            block_size=block_size,
            x_origin=x_origin, y_origin=y_origin
        )

        # 与原始文件对齐（按行顺序合并）
        merged_df = pd.concat([df_xy.reset_index(drop=True), block_df[['block_x', 'block_y']]], axis=1)

        # 计算该井涉及的所有唯一区块
        unique_blocks = merged_df[['block_x', 'block_y']].drop_duplicates()
        print(f"✓ {well_name} 涉及 {len(unique_blocks)} 个区块: {list(map(tuple, unique_blocks.values))}")

        for _, blk in unique_blocks.iterrows():
            bx, by = int(blk['block_x']), int(blk['block_y'])

            # 选出属于该区块的所有点（保留原始所有列）
            frac_df = merged_df[(merged_df['block_x'] == bx) & (merged_df['block_y'] == by)].copy()
            frac_out = os.path.join(output_dir, f"{well_name}_X{bx}_Y{by}.csv")
            frac_df.to_csv(frac_out, index=False)
            print(f"  - 已保存裂缝坐标: {frac_out}")


# ===========================================================
# 参数配置与运行
# ===========================================================
if __name__ == "__main__":
    TRACE_HEADER_FILE = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"
    # XY_INPUT_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝角度预测\测井"
    XY_INPUT_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\裂缝角度预测\井斜"
    OUTPUT_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\测井区块生成\裂缝单元归属"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_ORIGIN = 556150.0
    Y_ORIGIN = 4193975.0
    BLOCK_SIZE = 25

    process_all_wells(
        trace_header_file=TRACE_HEADER_FILE,
        xy_input_dir=XY_INPUT_DIR,
        output_dir=OUTPUT_DIR,
        x_origin=X_ORIGIN,
        y_origin=Y_ORIGIN,
        block_size=BLOCK_SIZE
    )
