import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import chardet
import re
from scipy.interpolate import interp1d

# 路径设置
# 输入
data_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩"
sgy_file = os.path.join(data_dir, "psdm_final_time.sgy")
# 输出
output_dir = os.path.join(data_dir, "研究内容一")
os.makedirs(output_dir, exist_ok=True)

# 需要考虑空间变化的斜井名单
inclined_wells_path = os.path.join(data_dir, '井斜')
inclined_wells = list(well.split('.')[0] for well in os.listdir(inclined_wells_path))
output_inclined_wells_to_trace_id_file = os.path.join(output_dir, "井斜", "测井-地震道id")
os.makedirs(output_inclined_wells_to_trace_id_file, exist_ok=True)
# 不需要考虑空间变化的测井名单
logging_wells_path = os.path.join(data_dir, '测井')
logging_wells = list(well.split('.')[0] for well in os.listdir(logging_wells_path) if well.endswith('.Las'))
logging_wells = list(well for well in logging_wells if well not in inclined_wells)
output_logging_wells_to_trace_id_file = os.path.join(output_dir, "测井", "测井-地震道id")
os.makedirs(output_logging_wells_to_trace_id_file, exist_ok=True)
# 成像测井名单
imaging_wells = {
    "车660": [592331, 4212564],
    "车662": [592350, 4210905],
    "车663": [590100, 4212075]
}
output_imaging_wells_file = os.path.join(output_dir, "成像测井", "相关地震道信息", "测井-地震道id")
os.makedirs(output_imaging_wells_file, exist_ok=True)

# 读取已保存的 trace_header_xy.csv
df_tr = pd.read_csv("trace_header_xy.csv")

coords = df_tr[['X', 'Y']].values.astype(np.float32)  # (Ntr, 2)
tree = KDTree(coords, leaf_size=64)  # 建 KDTree

# 方便后续速查：TraceIdx → 行号（DataFrame index）
traceidx2row = pd.Series(df_tr.index.values, index=df_tr['TraceIdx']).to_dict()
row2traceidx = df_tr['TraceIdx'].to_numpy()  # 行号 → TraceIdx

def nearest_traces(x0, y0, k=9):
    """
    查找井轨迹点 (x0, y0) 附近最近的 3x3 地震道
    返回地震道 TraceIdx 列表
    """
    dist, idx = tree.query([[x0, y0]], k=k)
    return row2traceidx[idx.ravel()].tolist()

def process_inclined_well(file_path, sample_interval=5):
    """
    处理一个井斜文件，提取整条井轨迹的插值采样点，并找到其对应的地震道集合
    返回：集合 set(trace_idx)
    """
    try:
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read(2048))['encoding']

        with open(file_path, "r", encoding=encoding, errors='ignore') as f:
            lines = f.readlines()

        header_line = lines[1].lstrip("#").strip()
        column_names = re.split(r'\s+', header_line)
        data_lines = lines[2:]
        data = [re.split(r'\s+', line.strip()) for line in data_lines if line.strip()]
        df = pd.DataFrame(data, columns=column_names)

        # 确保 MD/X/Y 列为 float
        df = df.astype({'MD': float, 'X': float, 'Y': float})
        df = df.sort_values(by='MD')

        # 插值函数构建
        md_vals = df['MD'].values
        x_vals = df['X'].values
        y_vals = df['Y'].values

        f_x = interp1d(md_vals, x_vals, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_y = interp1d(md_vals, y_vals, kind='linear', bounds_error=False, fill_value="extrapolate")

        md_dense = np.arange(md_vals.min(), md_vals.max(), sample_interval)
        x_dense = f_x(md_dense)
        y_dense = f_y(md_dense)

        trace_set = set()
        for x, y in zip(x_dense, y_dense):
            traces = nearest_traces(x, y)
            trace_set.update(traces)

        return sorted(trace_set)

    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}，错误：{e}")
        return []


def process_logging_well(file_path):
    """
    处理一个测井文件，提取其 XCRD / YCRD 坐标，并返回最近的 3x3 地震道索引
    返回：列表 trace_idx
    """
    try:
        # 自动识别编码
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read(2048))['encoding']

        with open(file_path, "r", encoding=encoding, errors='ignore') as f:
            lines = f.readlines()

        in_parameter_block = False
        x = y = None

        for line in lines:
            line = line.strip()

            # 判断进入 ~Parameter 块
            if line.startswith("~Parameter"):
                in_parameter_block = True
                continue
            elif line.startswith("~") and in_parameter_block:
                break  # 离开 Parameter 段

            if in_parameter_block:
                if "XCRD" in line:
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if nums:
                        x = float(nums[0])
                elif "YCRD" in line:
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if nums:
                        y = float(nums[0])

        if x is not None and y is not None:
            return nearest_traces(x, y)
        else:
            print(f"⚠️ 坐标提取失败：{file_path}")
            return []

    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}，错误：{e}")
        return []


# 斜井处理
for inclined_well in inclined_wells:
    print(f"正在处理斜井-{inclined_well}")
    file_path = os.path.join(inclined_wells_path, f'{inclined_well}.dat')
    traces = process_inclined_well(file_path)
    pd.DataFrame(traces, columns=['TraceIdx']).to_csv(
        os.path.join(output_inclined_wells_to_trace_id_file, f"{inclined_well}_traces.csv"),
        index=False
    )

# 常规测井处理
for logging_well in logging_wells:
    print(f"正在处理测井-{logging_well}")
    file_path = os.path.join(logging_wells_path, f'{logging_well}.Las')
    traces = process_logging_well(file_path)
    pd.DataFrame(traces, columns=['TraceIdx']).to_csv(
        os.path.join(output_logging_wells_to_trace_id_file, f"{logging_well}_traces.csv"),
        index=False
    )

# 成像测井处理
for imaging_well, xy in imaging_wells.items():
    print(f"正在处理测井-{imaging_well}")
    traces = nearest_traces(xy[0], xy[1])
    pd.DataFrame(traces, columns=['TraceIdx']).to_csv(
        os.path.join(output_imaging_wells_file, f"{imaging_well}_traces.csv"),
        index=False
    )