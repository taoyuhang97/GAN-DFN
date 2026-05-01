# 导入所需的库
import pandas as pd
import numpy as np
import os


def extract_block(trace_header_file, output_csv_file, block_index_x, block_index_y):
    """
    从整个工区的绝对坐标系出发，创建一个25x25道网格系统，并根据指定的网格编号提取一个区块。

    Args:
        trace_header_file (str): 包含所有地震道头坐标的CSV文件 ('trace_header_xy.csv')。
        output_csv_file (str): 输出的CSV文件名。
        block_index_x (int): 您想要提取的区块在X方向的编号（从0开始）。
        block_index_y (int): 您想要提取的区块在Y方向的编号（从0开始）。
    """
    # --- 1. 加载数据并建立工区坐标系 ---
    print(f"正在从 {trace_header_file} 加载道头数据...")
    try:
        df = pd.read_csv(trace_header_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {trace_header_file}。")
        return

    # 获取所有唯一的X和Y坐标，并进行排序
    unique_x = np.sort(df['X'].unique())
    unique_y = np.sort(df['Y'].unique())

    # --- 2. 计算全局网格信息 ---
    block_size = 25  # 定义每个标准单元的大小为25道

    # 计算在X和Y方向上，总共可以划分出多少个完整的区块
    num_blocks_x = len(unique_x) // (block_size - 1)
    num_blocks_y = len(unique_y) // (block_size - 1)

    print("-" * 50)
    print("全局网格系统信息:")
    print(f"  - 每个标准区块大小: {block_size} x {block_size} 道")
    print(f"  - X方向(Inline)总道数: {len(unique_x)}, 可划分为 {num_blocks_x} 个区块 (编号 0 至 {num_blocks_x - 1})")
    print(f"  - Y方向(Crossline)总道数: {len(unique_y)}, 可划分为 {num_blocks_y} 个区块 (编号 0 至 {num_blocks_y - 1})")
    print("-" * 50)

    # --- 3. 根据用户指定的区块编号，计算坐标范围 ---
    # 检查输入的编号是否有效
    if not (0 <= block_index_x < num_blocks_x and 0 <= block_index_y < num_blocks_y):
        print(f"错误：您输入的区块编号 ({block_index_x}, {block_index_y}) 超出有效范围！")
        return

    print(f"正在提取您指定的区块，编号为: (X={block_index_x}, Y={block_index_y})")

    # 计算该区块在坐标列表中的起始和结束索引
    start_x_idx = block_index_x * (block_size - 1)
    end_x_idx = start_x_idx + block_size
    print(f"X坐标起始索引为{start_x_idx}, 结束索引为{end_x_idx}")

    start_y_idx = block_index_y * (block_size - 1)
    end_y_idx = start_y_idx + block_size
    print(f"Y坐标起始索引为{start_y_idx}, 结束索引为{end_y_idx}")

    # 根据索引，从唯一的坐标列表中选出该区块包含的所有X和Y坐标
    target_x_coords = unique_x[start_x_idx:end_x_idx]
    target_y_coords = unique_y[start_y_idx:end_y_idx]

    # --- 4. 筛选并保存数据 ---
    print("正在从总道头文件中筛选目标区块的数据...")
    # 使用 isin 方法筛选出X和Y坐标同时在我们定义的网格内的所有道
    grid_df = df[df['X'].isin(target_x_coords) & df['Y'].isin(target_y_coords)]

    # 验证结果
    expected_traces = block_size * block_size
    if len(grid_df) != expected_traces:
        print(f"警告：最终提取到的道数为 {len(grid_df)}，而不是预期的{expected_traces}。")
        print("这可能表示在该区块的物理位置上，存在数据采集缺失。")
    else:
        print(f"✓ 验证成功！已精确找到 {len(grid_df)} ({block_size}x{block_size}) 条地震道。")

    # 排序并保存
    grid_df_sorted = grid_df.sort_values(by=['X', 'Y']).reset_index(drop=True)
    grid_df_sorted.to_csv(output_csv_file, index=False)
    print(f"✓ 成功！区块 (X={block_index_x}, Y={block_index_y}) 的道头信息已保存至: {output_csv_file}")


# --- 参数配置区 ---
# 输入文件名
trace_header_xy_file = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv"


# 请在这里指定您想要提取的区块的X和Y方向的“编号”
# 编号从0开始。建议先从工区中间的区块开始，例如选择总区块数的一半左右
BLOCK_INDEX_X = 32  # 示例：提取X方向的第41个区块
BLOCK_INDEX_Y = 24  # 示例：提取Y方向的第26个区块

# --- 主程序执行区 ---
if __name__ == "__main__":
    # 根据区块编号动态生成输出文件名
    output_filename = f'block_X{BLOCK_INDEX_X}_Y{BLOCK_INDEX_Y}_traces.csv'

    extract_block(
        trace_header_file=trace_header_xy_file,
        output_csv_file=output_filename,
        block_index_x=BLOCK_INDEX_X,
        block_index_y=BLOCK_INDEX_Y
    )
