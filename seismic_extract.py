import os
import segyio
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import chardet
import re

# 路径设置
# 输入
data_dir = r"/data/shared/project-oil/wx数据/砂砾岩"
sgy_file = os.path.join(data_dir, "psdm_final_time.sgy")
# 输出
output_dir = os.path.join(data_dir, "研究内容一")
os.makedirs(output_dir, exist_ok=True)

# 需要考虑空间变化的斜井名单
inclined_wells_path = os.path.join(data_dir, '井斜')
inclined_wells = list(well.split('.')[0] for well in os.listdir(inclined_wells_path))
output_inclined_wells_file = os.path.join(output_dir, "井斜", "相关地震道信息")
os.makedirs(output_inclined_wells_file, exist_ok=True)
# 不需要考虑空间变化的测井名单
logging_wells_path = os.path.join(data_dir, '测井')
logging_wells = list(well.split('.')[0] for well in os.listdir(logging_wells_path) if well.endswith('.Las'))
logging_wells = list(well for well in logging_wells if well not in inclined_wells)
output_logging_wells_file = os.path.join(output_dir, "测井", "相关地震道信息")
os.makedirs(output_logging_wells_file, exist_ok=True)
# 成像测井名单
imaging_wells = {
    "车660": [592331, 4212564],
    "车662": [592350, 4210905],
    "车663": [590100, 4212075]
}
output_imaging_wells_file = os.path.join(output_dir, "成像测井", "相关地震道信息")
os.makedirs(output_imaging_wells_file, exist_ok=True)

# 读取已保存的 trace_header_xy.csv
df_tr = pd.read_csv("数据精简/trace_header_xy.csv")

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


def process_inclined_well(file_path):
    """
    处理一个井斜文件，提取所有采样点的XY并找到其对应的地震道集合
    返回：集合 set(trace_idx)
    """
    try:
        # 尝试自动识别编码
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read(2048))['encoding']

        # 读取文件所有内容
        with open(file_path, "r", encoding=encoding, errors='ignore') as f:
            lines = f.readlines()

        # 提取第二行作为列名（去掉#开头）
        header_line = lines[1].lstrip("#").strip()
        column_names = re.split(r'\s+', header_line)
        # 读取数据内容，从第三行开始
        data_lines = lines[2:]
        data = [re.split(r'\s+', line.strip()) for line in data_lines if line.strip()]
        # 构造 DataFrame
        df = pd.DataFrame(data, columns=column_names)

        trace_set = set()
        for _, row in df.iterrows():
            try:
                x, y = float(row['X']), float(row['Y'])
                traces = nearest_traces(x, y)
                trace_set.update(traces)
            except:
                continue
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
    file_path = os.path.join(inclined_wells_path, f'{inclined_well}.dat')
    traces = process_inclined_well(file_path)
    pd.DataFrame(traces, columns=['TraceIdx']).to_csv(
        os.path.join(output_inclined_wells_file, f"{inclined_well}_traces.csv"),
        index=False
    )

# 常规测井处理
for logging_well in logging_wells:
    file_path = os.path.join(logging_wells_path, f'{logging_well}.Las')
    traces = process_logging_well(file_path)
    pd.DataFrame(traces, columns=['TraceIdx']).to_csv(
        os.path.join(output_logging_wells_file, f"{logging_well}_traces.csv"),
        index=False
    )

# 成像测井处理
for imaging_well, xy in imaging_wells.items():
    traces = nearest_traces(xy[0], xy[1])
    pd.DataFrame(traces, columns=['TraceIdx']).to_csv(
        os.path.join(output_imaging_wells_file, f"{imaging_well}_traces.csv"),
        index=False
    )

def save_traces_to_csv(trace_idx_list, segy_handle, out_csv):
    """
    trace_idx_list : List[int]
    segy_handle    : segyio file handle
    out_csv        : 输出 csv 路径
    """
    nsamp = segy_handle.samples.size
    records = []
    for tid in trace_idx_list:
        trace = segy_handle.trace[tid]  # numpy array (nsamp,)
        row = [tid] + trace.tolist()  # 第 1 列 TraceIdx，其余为样点
        records.append(row)

    # 构造列名：TraceIdx, S0, S1, ...
    colnames = ["TraceIdx"] + [f"S{i}" for i in range(nsamp)]
    pd.DataFrame(records, columns=colnames).to_csv(out_csv, index=False)


# -----------------------------------------
# 一次打开 SEG-Y，随后逐井提取
# -----------------------------------------
with segyio.open(sgy_file, "r", ignore_geometry=True) as f:
    f.mmap()
    print("SEG‑Y opened, start exporting per‑well traces …")

    # -------- 斜井 --------
    for csv_path in os.listdir(output_inclined_wells_file):
        if not csv_path.endswith("_traces.csv"):
            continue
        well_name = csv_path.replace("_traces.csv", "")
        trace_idx = pd.read_csv(os.path.join(output_inclined_wells_file, csv_path))["TraceIdx"].tolist()
        out_csv = os.path.join(output_inclined_wells_file, f"{well_name}_tracecube.csv")
        save_traces_to_csv(trace_idx, f, out_csv)
        print(f"✓ 斜井 {well_name}: {len(trace_idx)} 道 → {out_csv}")

    # -------- 直井(常规测井) --------
    for csv_path in os.listdir(output_logging_wells_file):
        if not csv_path.endswith("_traces.csv"):
            continue
        well_name = csv_path.replace("_traces.csv", "")
        trace_idx = pd.read_csv(os.path.join(output_logging_wells_file, csv_path))["TraceIdx"].tolist()
        out_csv = os.path.join(output_logging_wells_file, f"{well_name}_tracecube.csv")
        save_traces_to_csv(trace_idx, f, out_csv)
        print(f"✓ 直井 {well_name}: {len(trace_idx)} 道 → {out_csv}")

    # -------- 成像测井 --------
    for csv_path in os.listdir(output_imaging_wells_file):
        if not csv_path.endswith("_traces.csv"):
            continue
        well_name = csv_path.replace("_traces.csv", "")
        trace_idx = pd.read_csv(os.path.join(output_imaging_wells_file, csv_path))["TraceIdx"].tolist()
        out_csv = os.path.join(output_logging_wells_file, f"{well_name}_tracecube.csv")
        save_traces_to_csv(trace_idx, f, out_csv)
        print(f"✓ 成像测井 {well_name}: {len(trace_idx)} 道 → {out_csv}")