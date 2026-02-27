import pandas as pd
import numpy as np
import os


def find_block_by_xy(trace_header_file, well_x, well_y):
    """
    根据给定的井点XY坐标，找到距离最近的地震道，并计算出该道所在的25x25标准网格区块编号。

    Args:
        trace_header_file (str): 包含所有地震道头坐标的CSV文件 ('trace_header_xy.csv')。
        well_x (float): 井点的X坐标。
        well_y (float): 井点的Y坐标。
    """
    # --- 1. 加载数据并建立工区坐标系 ---
    print(f"正在从 {trace_header_file} 加载工区网格信息...")
    try:
        df = pd.read_csv(trace_header_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {trace_header_file}。")
        return
    except Exception as e:
        print(f"错误：读取文件时发生错误: {e}")
        return

    # --- 2. 查找距离井点坐标最近的地震道 ---
    print(f"正在查找距离坐标 (X={well_x}, Y={well_y}) 最近的地震道...")

    # 计算所有道到井点的距离
    distances = np.sqrt((df['X'] - well_x) ** 2 + (df['Y'] - well_y) ** 2)

    # 找到最小距离的索引
    min_idx = distances.idxmin()
    closest_trace = df.iloc[min_idx]

    closest_x = closest_trace['X']
    closest_y = closest_trace['Y']
    closest_trace_idx = closest_trace['TraceIdx']
    min_distance = distances[min_idx]

    print(f"✓ 找到最近的地震道:")
    print(f"  - TraceIdx: {closest_trace_idx}")
    print(f"  - 坐标: (X={closest_x}, Y={closest_y})")
    print(f"  - 距离井点的距离: {min_distance:.2f} 单位")

    # --- 3. 重建全局网格并计算区块编号 ---
    # 获取所有唯一的X和Y坐标，并进行排序
    unique_x = np.sort(df['X'].unique())
    unique_y = np.sort(df['Y'].unique())

    block_size = 25  # 标准区块大小为25道

    # 找到离最近道X坐标最近的实际道线的索引
    x_idx = np.argmin(np.abs(unique_x - closest_x))
    # 找到离最近道Y坐标最近的实际道线的索引
    y_idx = np.argmin(np.abs(unique_y - closest_y))

    # 利用整除运算，直接计算出该索引属于哪个区块
    block_index_x = x_idx // block_size
    block_index_y = y_idx // block_size

    # 计算区块的边界坐标
    block_start_x = unique_x[block_index_x * block_size]
    block_end_x = unique_x[min((block_index_x + 1) * block_size - 1, len(unique_x) - 1)]
    block_start_y = unique_y[block_index_y * block_size]
    block_end_y = unique_y[min((block_index_y + 1) * block_size - 1, len(unique_y) - 1)]

    # --- 4. 输出结果 ---
    print("-" * 50)
    print(f"查询井点坐标: (X={well_x}, Y={well_y})")
    print(f"✓ 最近地震道的标准网格区块编号为: (X区块 = {block_index_x}, Y区块 = {block_index_y})")
    print(f"✓ 区块坐标范围:")
    print(f"  - X方向: {block_start_x} 到 {block_end_x}")
    print(f"  - Y方向: {block_start_y} 到 {block_end_y}")
    print(f"✓ 对应的TraceIdx: {closest_trace_idx}")

    return {
        'block_x': block_index_x,
        'block_y': block_index_y,
        'closest_trace_idx': closest_trace_idx,
        'closest_x': closest_x,
        'closest_y': closest_y,
        'distance_to_well': min_distance,
        'block_bounds': {
            'x_start': block_start_x,
            'x_end': block_end_x,
            'y_start': block_start_y,
            'y_end': block_end_y
        }
    }


def find_multiple_wells_blocks(trace_header_file, wells_coordinates):
    """
    批量处理多个井点的区块查找

    Args:
        trace_header_file (str): 包含所有地震道头坐标的CSV文件
        wells_coordinates (list): 井点坐标列表，每个元素为 (well_name, x, y) 元组
    """
    results = {}

    for well_name, x, y in wells_coordinates:
        print(f"\n正在处理井点: {well_name}")
        result = find_block_by_xy(trace_header_file, x, y)
        results[well_name] = result

    return results


# --- 参数配置区 ---
# 输入文件名
TRACE_HEADER_FILE = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"

# --- 使用示例 ---
if __name__ == "__main__":
    # 示例1: 单个井点查询
    well = ("车22", 566456, 4201615)

    result = find_block_by_xy(
        trace_header_file=TRACE_HEADER_FILE,
        well_x = well[1],
        well_y = well[2]
    )

    # 示例2: 批量处理多个井点（取消注释使用）
    """
    wells = [
        ("井1", 500000.0, 4000000.0),
        ("井2", 501000.0, 4001000.0),
        ("井3", 502000.0, 4002000.0)
    ]

    results = find_multiple_wells_blocks(TRACE_HEADER_FILE, wells)

    # 打印批量处理结果摘要
    print("\n" + "="*60)
    print("批量处理结果摘要:")
    print("="*60)
    for well_name, result in results.items():
        print(f"{well_name}: 区块({result['block_x']}, {result['block_y']}), "
              f"最近道Idx: {result['closest_trace_idx']}, "
              f"距离: {result['distance_to_well']:.2f}")
    """