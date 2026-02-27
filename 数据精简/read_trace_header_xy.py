import os
import segyio
import numpy as np
import pandas as pd
from tqdm import tqdm

# 路径设置
# 输入
data_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩"
# sgy_file = os.path.join(data_dir, "psdm_final_time.sgy")
sgy_file = os.path.join(data_dir, "che66_2019_psdm_time.sgy")
# 输出
output_dir = os.path.join(data_dir, "研究内容一")
os.makedirs(output_dir, exist_ok=True)
# output_trace_header_csv = os.path.join(output_dir, "trace_header_xy.csv")
output_trace_header_csv = os.path.join(output_dir, "che66_2019_trace_header_xy.csv")

# 打开 SEG-Y 文件，读取 X/Y 和 trace 索引
with segyio.open(sgy_file, "r", ignore_geometry=True) as f:
    f.mmap()  # 加速访问

    # 读取采样间隔（单位：微秒）和每道的样本数
    sample_interval_us = f.bin[segyio.BinField.Interval]  # 单位：μs
    num_samples_per_trace = f.bin[segyio.BinField.Samples]

    # 打印信息
    print(f"采样间隔：{sample_interval_us} μs ({sample_interval_us / 1000} 毫秒)")
    print(f"每道采样点数：{num_samples_per_trace}")

    # ===== 打印振幅序列 =====
    # trace_id = 100
    # trace_data = f.trace[trace_id]
    #
    # print("\n===== Trace Data =====")
    # print(f"振幅数组类型: {type(trace_data)}")
    # print(f"振幅数组长度: {len(trace_data)}")
    #
    # print("前 10 个采样点振幅:")
    # print(trace_data[:100])
    #
    # print("后 10 个采样点振幅:")
    # print(trace_data[-100:])

    n_traces = f.tracecount

    trace_idx = np.arange(n_traces)
    xs = np.empty(n_traces, dtype=np.int32)
    ys = np.empty(n_traces, dtype=np.int32)

    for i in tqdm(range(n_traces), desc="提取 Trace Header (X,Y)"):
        # 目标矿区
        # xs[i] = f.header[i][segyio.TraceField.SourceX]
        # ys[i] = f.header[i][segyio.TraceField.SourceY]
        # 车66区域
        xs[i] = f.header[i][segyio.TraceField.CDP_X]
        ys[i] = f.header[i][segyio.TraceField.CDP_Y]

# 构建 DataFrame
df = pd.DataFrame({
    "TraceIdx": trace_idx,
    "X": xs,
    "Y": ys
})

# 保存为 CSV 格式
df.to_csv(output_trace_header_csv, index=False)

print("Trace X/Y 索引表已保存：")
print("CSV:", output_trace_header_csv)