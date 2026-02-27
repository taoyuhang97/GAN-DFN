# 导入所需的库
import pandas as pd
import segyio
import os
import numpy as np


def extract_sgy_data(sgy_file_path, trace_list_csv, output_data_csv):
    """
    根据给定的标准区块道头清单，从SGY文件中提取实际的地震道数据，
    并将数据转置，使每行代表一个地震道，每列代表一个时间点。

    Args:
        sgy_file_path (str): 原始SGY文件的路径。
        trace_list_csv (str): 包含地震道信息('TraceIdx', 'X', 'Y')的CSV文件。
        output_data_csv (str): 保存最终提取出的地震数据的CSV文件名。
    """
    # --- 1. 读取道头清单 (包含TraceIdx, X, Y) ---
    print(f"正在从清单文件 {trace_list_csv} 读取道头信息...")
    try:
        block_traces_df = pd.read_csv(trace_list_csv)
    except FileNotFoundError:
        print(f"错误：找不到清单文件 {trace_list_csv}。")
        print("请确保该文件存在，并且您在下面的参数配置区输入了正确的文件名。")
        return

    # 从清单中分别获取 TraceIdx, X坐标, 和 Y坐标
    trace_indices = block_traces_df['TraceIdx'].tolist()
    x_coords = block_traces_df['X'].tolist()
    y_coords = block_traces_df['Y'].tolist()

    print(f"准备从SGY文件中精确提取 {len(trace_indices)} 道数据。")

    # --- 2. 打开SGY文件并提取数据 ---
    print(f"正在打开SGY文件: {sgy_file_path}...")
    try:
        with segyio.open(sgy_file_path, 'r', ignore_geometry=True) as sgyfile:
            time_samples = sgyfile.samples
            num_samples = len(time_samples)
            print(f"每道的时间采样点数量为: {num_samples}")

            all_traces_data = np.zeros((len(trace_indices), num_samples), dtype=np.float32)

            for i, trace_idx in enumerate(trace_indices):
                all_traces_data[i, :] = sgyfile.trace[trace_idx]
                if (i + 1) % 100 == 0:
                    print(f"已提取 {i + 1} / {len(trace_indices)} 道...")

            print("所有目标地震道的数据提取完成。")

            # --- 3. 整理数据并创建转置的CSV文件 ---
            print("正在将数据整理并保存到CSV文件...")

            # 创建时间列名
            time_columns = [f"Time_{i}" for i in range(num_samples)]

            # 创建DataFrame，每行代表一个地震道
            output_df = pd.DataFrame(
                data=all_traces_data,
                columns=time_columns
            )

            # 在最左侧插入TraceId, X, Y列
            output_df.insert(0, "y", y_coords)
            output_df.insert(0, "x", x_coords)
            output_df.insert(0, "traceId", trace_indices)

            # 将最终整理好的DataFrame保存为CSV文件
            output_df.to_csv(output_data_csv, index=False)
            print(f"✓ 成功！转置后的地震数据块已保存至: {output_data_csv}")
            print(f"数据形状: {output_df.shape} (行数: {output_df.shape[0]}, 列数: {output_df.shape[1]})")
            print(f"前三列名称: {list(output_df.columns[:3])}")
            print(f"时间列数量: {len(time_columns)}")

    except FileNotFoundError:
        print(f"错误：找不到SGY文件 '{sgy_file_path}'。")
        print("请检查文件名是否正确，并确保SGY文件与脚本在同一个目录下。")
    except Exception as e:
        print(f"处理过程中发生了一个意料之外的错误: {e}")


# --- 参数配置区 ---
# SGY文件名
SGY_FILE = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\psdm_final_time.sgy"

# 当前处理的网格XY
unitX, unitY = 32, 24

# 输入的清单文件名 (这是第一步脚本的输出，请确保文件名完全匹配)
BLOCK_TRACES_LIST_FILE = f'block_X{unitX}_Y{unitY}_traces.csv'

# 最终输出的数据文件
OUTPUT_SEISMIC_DATA_FILE = f'seismic_data_block_X{unitX}_Y{unitY}.csv'

if __name__ == "__main__":
    # 使用转置格式（每行一个地震道）
    extract_sgy_data(
        sgy_file_path=SGY_FILE,
        trace_list_csv=BLOCK_TRACES_LIST_FILE,
        output_data_csv=OUTPUT_SEISMIC_DATA_FILE
    )
