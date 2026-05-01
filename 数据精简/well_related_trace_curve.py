import os
import segyio
import pandas as pd

# 路径设置
# 输入
data_dir = r"/data/shared/project-oil/wx数据/砂砾岩"
sgy_file = os.path.join(data_dir, "psdm_final_time.sgy")
# 输出
output_dir = os.path.join(data_dir, "研究内容一")
os.makedirs(output_dir, exist_ok=True)

# 斜井
output_inclined_wells_to_trace_id_file = os.path.join(output_dir, "井斜", "测井-地震道id")
output_inclined_wells_to_trace_curve_file = os.path.join(output_dir, "井斜", "测井-地震曲线")
os.makedirs(output_inclined_wells_to_trace_curve_file, exist_ok=True)
# 测井
output_logging_wells_to_trace_id_file = os.path.join(output_dir, "测井", "测井-地震道id")
output_logging_wells_to_trace_curve_file = os.path.join(output_dir, "测井", "测井-地震曲线")
os.makedirs(output_logging_wells_to_trace_curve_file, exist_ok=True)
# 成像测井
output_imaging_wells_to_trace_id_file = os.path.join(output_dir, "成像测井", "相关地震道信息", "测井-地震道id")
output_imaging_wells_to_trace_curve_file = os.path.join(output_dir, "成像测井", "相关地震道信息", "测井-地震曲线")
os.makedirs(output_imaging_wells_to_trace_curve_file, exist_ok=True)

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
    dt_microseconds = segyio.dt(f)  # 采样间隔，单位：微秒
    dt_milliseconds = dt_microseconds / 1000.0
    print(f"采样时间间隔: {dt_milliseconds} ms")

    # -------- 斜井 --------
    # for csv_path in os.listdir(output_inclined_wells_to_trace_id_file):
    #     if not csv_path.endswith("_traces.csv"):
    #         continue
    #     well_name = csv_path.replace("_traces.csv", "")
    #     trace_idx = pd.read_csv(os.path.join(output_inclined_wells_to_trace_id_file, csv_path))["TraceIdx"].tolist()
    #     out_csv = os.path.join(output_inclined_wells_to_trace_curve_file, f"{well_name}_tracecube.csv")
    #     save_traces_to_csv(trace_idx, f, out_csv)
    #     print(f"✓ 斜井 {well_name}: {len(trace_idx)} 道 → {out_csv}")

    # -------- 直井(常规测井) --------
    # for csv_path in os.listdir(output_logging_wells_to_trace_id_file):
    #     if not csv_path.endswith("_traces.csv"):
    #         continue
    #     well_name = csv_path.replace("_traces.csv", "")
    #     trace_idx = pd.read_csv(os.path.join(output_logging_wells_to_trace_id_file, csv_path))["TraceIdx"].tolist()
    #     out_csv = os.path.join(output_logging_wells_to_trace_curve_file, f"{well_name}_tracecube.csv")
    #     save_traces_to_csv(trace_idx, f, out_csv)
    #     print(f"✓ 直井 {well_name}: {len(trace_idx)} 道 → {out_csv}")


    # -------- 成像测井 --------
    for csv_path in os.listdir(output_imaging_wells_to_trace_id_file):
        if not csv_path.endswith("_traces.csv"):
            continue
        well_name = csv_path.replace("_traces.csv", "")
        trace_idx = pd.read_csv(os.path.join(output_imaging_wells_to_trace_id_file, csv_path))["TraceIdx"].tolist()
        out_csv = os.path.join(output_imaging_wells_to_trace_curve_file, f"{well_name}_tracecube.csv")
        save_traces_to_csv(trace_idx, f, out_csv)
        print(f"✓ 直井 {well_name}: {len(trace_idx)} 道 → {out_csv}")