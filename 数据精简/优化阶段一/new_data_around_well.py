import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import chardet
import re
import unicodedata
from tqdm import tqdm
from sklearn.neighbors import KDTree
from datetime import datetime
import segyio

sampling_start_ms = 1100  # 地震道采样起始时间，单位：毫秒
sampling_interval_ms = 1.0  # 地震道采样时间间隔，单位：毫秒


# ==== 读取 LAS ====
def read_las_ascii_block(file_path):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(2048))['encoding']
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()
    columns = []
    in_columns_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('~Curve') or line.startswith('~C'):
            in_columns_section = True
            continue
        if in_columns_section:
            if line.startswith('~') or line.lower().startswith('~Parameter'):
                break
            if '.' in line:
                name = line.split('.')[0].strip()
                if name != '':
                    columns.append(name)

    start_idx = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("~ascii"))
    data = []
    for line in lines[start_idx + 1:]:
        if line.strip():
            data.append([float(v) for v in line.strip().split()])
    return pd.DataFrame(data, columns=columns)


# ==== 读取井轨迹 ====
def read_well_track(file_path):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(2048))['encoding']
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()
    header = re.split(r'\s+', lines[1].strip('#').strip())
    data = [re.split(r'\s+', line.strip()) for line in lines[2:] if line.strip()]
    df = pd.DataFrame(data, columns=header)
    df = df.astype({'TVD': float, 'X': float, 'Y': float})
    return df


# ==== 读取时深关系 ====
def read_timedepth_file(file_path):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(2048))['encoding']
    with open(file_path, "r", encoding=encoding, errors='ignore') as f:
        lines = f.readlines()

    header_line_index = -1
    for i, line in enumerate(lines):
        if not line.strip().startswith("#") and len(re.split(r'\s+', line.strip())) >= 5:
            header_line_index = i - 1
            break
    header_line = lines[header_line_index].lstrip("#").strip()
    column_names = re.split(r'\s+', header_line)
    data_lines = lines[header_line_index + 1:]
    data = [re.split(r'\s+', line.strip()) for line in data_lines if line.strip()]
    df = pd.DataFrame(data, columns=column_names)

    # 标准化列名
    new_cols = []
    seen = set()
    for col in df.columns:
        norm = unicodedata.normalize("NFKD", col)
        norm = norm.upper().replace("'", "").replace(".", "").strip()
        if norm not in seen:
            new_cols.append(norm)
            seen.add(norm)
        else:
            new_cols.append(col)
    df.columns = new_cols
    df['TVD'] = pd.to_numeric(df['TVD'], errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    return df[['TVD', 'TIME']].dropna()

# ==== 主过程 ====
# 全局变量：trace顺序map（初始化一次）
trace_sort_order = None


def build_trace_sort_order(df_traceheader):
    """
    构建 traceidx 到 (X,Y) 排序顺序的映射
    """
    df_sorted = df_traceheader.sort_values(by=['X', 'Y']).reset_index(drop=True)
    return {tid: i for i, tid in enumerate(df_sorted['TraceIdx'])}


def sort_traces_by_order(trace_idx_list):
    """
    按照构建好的 trace_sort_order 排序 trace_idx_list
    """
    return sorted(trace_idx_list, key=lambda tid: trace_sort_order.get(tid, float('inf')))


def sort_traces_by_xy(trace_indices, df_traceheader):
    sub_df = df_traceheader[df_traceheader['TraceIdx'].isin(trace_indices)].copy()
    sub_df = sub_df.sort_values(by=['X', 'Y'])  # 先按 X，再按 Y 升序
    return sub_df['TraceIdx'].tolist()


# 点位对应实际振幅计算
def get_trace_interp(
        tid,
        f_segy,
        trace_cache,
        interp_cache,
        sampling_start_ms,
        sampling_interval_ms,
        nsamp
):
    tid = int(tid)

    # 1. 缓存 trace
    if tid not in trace_cache:
        trace_cache[tid] = f_segy.trace[tid]

    # 2. 缓存时间插值器
    if tid not in interp_cache:
        trace_data = trace_cache[tid]
        trace_time = sampling_start_ms + np.arange(nsamp) * sampling_interval_ms

        interp_cache[tid] = interp1d(
            trace_time,
            trace_data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )

    return interp_cache[tid]


def interpolate_seismic_amplitude_from_sgy(x, y, t, tree, traceidx_array, f_segy, sampling_start_ms,
                                           sampling_interval_ms, nsamp, trace_cache,
                                           interp_cache=None,  # 保留接口，但不再使用
                                           grid_spacing=12.5):
    """
    规则网格条件下：
    1) 时间方向：直接数组线性插值（无 interp1d）
    2) XY 平面：双线性插值
    """

    # ========= Step 1: 定位所在网格 =========
    x0 = np.floor(x / grid_spacing) * grid_spacing
    y0 = np.floor(y / grid_spacing) * grid_spacing
    x1 = x0 + grid_spacing
    y1 = y0 + grid_spacing

    corners = [
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1)
    ]

    corner_amps = []

    # ========= Step 2: 时间索引一次算好 =========
    it = (t - sampling_start_ms) / sampling_interval_ms
    i0 = int(np.floor(it))
    w = it - i0

    # 边界保护（你说不在边界，但工程上建议留）
    if i0 < 0 or i0 + 1 >= nsamp:
        return np.nan

    # ========= Step 3: 四角点 =========
    for cx, cy in corners:
        _, idx = tree.query([[cx, cy]], k=1)
        tid = int(traceidx_array[idx[0][0]])

        # ---- 道缓存 ----
        if tid not in trace_cache:
            trace_cache[tid] = f_segy.trace[tid]

        trace_data = trace_cache[tid]

        # ---- 直接数组线性插值 ----
        amp = (1.0 - w) * trace_data[i0] + w * trace_data[i0 + 1]
        corner_amps.append(float(amp))

    a00, a10, a01, a11 = corner_amps

    # ========= Step 4: XY 双线性插值 =========
    alpha = (x - x0) / grid_spacing
    beta = (y - y0) / grid_spacing

    amp = (
            (1 - alpha) * (1 - beta) * a00 +
            alpha * (1 - beta) * a10 +
            (1 - alpha) * beta * a01 +
            alpha * beta * a11
    )

    return amp


def construct_inclined_samples(data_dir, output_dir, well_name):
    try:
        xy_offsets = [-12.5, 0, 12.5]
        t_offsets = [-3, -2, -1, 0, 1, 2, 3]

        sampling_start_ms = 1100  # 地震起始时间
        sampling_interval_ms = 1  # 1ms采采样间隔

        print(f"\n🚧 正在处理井：{well_name}")
        trace_header_csv = os.path.join(data_dir, "研究内容一", "trace_header_xy.csv")
        las_file = os.path.join(data_dir, "测井", f"{well_name}.Las")
        track_file = os.path.join(data_dir, "井斜", f"{well_name}.dat")
        timedepth_file = os.path.join(data_dir, "时深", f"{well_name}.dat")
        sgy_file = os.path.join(data_dir, "psdm_final_time.sgy")
        output_csv = os.path.join(output_dir, f"{well_name}_around_data.csv")

        # 检查文件存在性
        required_files = {
            "LAS": las_file,
            "井斜": track_file,
            "时深": timedepth_file,
            "TraceHeader": trace_header_csv
        }
        for k, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ 缺少文件：[{k}] {path}")

        print("读取测井、井轨迹、时深 ...")
        df_las = read_las_ascii_block(las_file)
        df_track = read_well_track(track_file)
        df_td = read_timedepth_file(timedepth_file)
        df_traceheader = pd.read_csv(trace_header_csv)
        global trace_sort_order
        trace_sort_order = build_trace_sort_order(df_traceheader)

        # 插值函数
        f_x = interp1d(df_track['TVD'], df_track['X'], bounds_error=False, fill_value="extrapolate")
        f_y = interp1d(df_track['TVD'], df_track['Y'], bounds_error=False, fill_value="extrapolate")
        f_t = interp1d(df_td['TVD'], df_td['TIME'], bounds_error=False, fill_value="extrapolate")

        trace_xy = df_traceheader[['X', 'Y']].values.astype(np.float32)
        traceidx_array = df_traceheader['TraceIdx'].values
        tree = KDTree(trace_xy, leaf_size=64)

        results = []

        with segyio.open(sgy_file, "r", ignore_geometry=True) as f_segy:
            f_segy.mmap()

            nsamp = f_segy.samples.size

            # 后面所有插值都用 f_segy

            trace_cache = {}
            interp_cache = {}

            for _, row in tqdm(df_las.iterrows(), total=len(df_las), desc=f"{well_name} 样本构建中"):
                tvd = row["DEPT"]
                x, y = f_x(tvd), f_y(tvd)
                t = f_t(tvd)

                if not np.isfinite(t):
                    continue

                # 根据采样间隔计算采样点索引
                seis_true = interpolate_seismic_amplitude_from_sgy(
                    x, y, t,
                    tree,
                    traceidx_array,
                    f_segy,
                    sampling_start_ms,
                    sampling_interval_ms,
                    nsamp,
                    trace_cache,
                    interp_cache
                )

                seis_window = []

                for dx in xy_offsets:
                    for dy in xy_offsets:
                        for dt in t_offsets:
                            amp = interpolate_seismic_amplitude_from_sgy(x + dx, y + dy, t + dt * sampling_interval_ms,
                                                                         tree, traceidx_array,
                                                                         f_segy, sampling_start_ms,
                                                                         sampling_interval_ms, nsamp, trace_cache,
                                                                         interp_cache)
                            if np.isnan(amp):
                                break

                            seis_window.append(amp)

                if len(seis_window) < 63:
                    continue
                well_attrs = {col: row[col] for col in df_las.columns if col not in ["TVD", "TIME"]}

                results.append({
                    "TVD": tvd, "TIME": t, "X": x, "Y": y,
                    "SEIS_TRUE": seis_true,
                    **{f"SEIS_{i}": v for i, v in enumerate(seis_window)},
                    **well_attrs
                })

        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        print(f"✅ 样本已保存：{output_csv}，共 {len(df_out)} 条")
        print(
            f"起始时间: {min(df_out['TIME'].iloc[0], df_out['TIME'].iloc[-1])}, 结束时间: {max(df_out['TIME'].iloc[0], df_out['TIME'].iloc[-1])}")

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("../construct_fail_log.txt", "a", encoding="utf-8") as f_log:
            f_log.write(f"[{timestamp}] {well_name} 构建失败：{e}\n")
        print(f"⚠️ 井 {well_name} 构建失败：{e}")


def get_wellhead_xy_from_las(file_path):
    import chardet
    import re

    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(2048))['encoding']

    with open(file_path, "r", encoding=encoding, errors='ignore') as f:
        lines = f.readlines()

    in_parameter_block = False
    x = y = None

    for line in lines:
        line = line.strip()

        if line.startswith("~Parameter") or line.lower().startswith("~p"):
            in_parameter_block = True
            continue
        elif line.startswith("~") and in_parameter_block:
            break

        if in_parameter_block:
            if "XCRD" in line.upper():
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if nums:
                    x = float(nums[0])
            elif "YCRD" in line.upper():
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if nums:
                    y = float(nums[0])

    return x, y


def construct_logging_samples(data_dir, output_dir, well_name):
    try:
        print(f"\n🚧 正在处理井：{well_name}")

        xy_offsets = [-12.5, 0, 12.5]
        t_offsets = [-3, -2, -1, 0, 1, 2, 3]

        sampling_start_ms = 1100
        sampling_interval_ms = 1

        trace_header_csv = os.path.join(data_dir, "研究内容一", "trace_header_xy.csv")
        las_file = os.path.join(data_dir, "测井", f"{well_name}.Las")
        timedepth_file = os.path.join(data_dir, "时深", f"{well_name}.dat")
        sgy_file = os.path.join(data_dir, "psdm_final_time.sgy")
        output_csv = os.path.join(output_dir, f"{well_name}_around_data.csv")

        required_files = {
            "LAS": las_file,
            "时深": timedepth_file,
            "TraceHeader": trace_header_csv,
            "SGY": sgy_file
        }

        for k, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ 缺少文件：[{k}] {path}")

        print("读取测井、时深、TraceHeader ...")
        df_las = read_las_ascii_block(las_file)
        df_td = read_timedepth_file(timedepth_file)
        df_traceheader = pd.read_csv(trace_header_csv)

        global trace_sort_order
        trace_sort_order = build_trace_sort_order(df_traceheader)

        # 井口坐标（直井）
        x, y = get_wellhead_xy_from_las(las_file)
        if x is None or y is None:
            raise ValueError(f"井口坐标提取失败：{las_file}")

        f_t = interp1d(
            df_td["TVD"], df_td["TIME"],
            bounds_error=False,
            fill_value="extrapolate"
        )

        trace_xy = df_traceheader[["X", "Y"]].values.astype(np.float32)
        traceidx_array = df_traceheader["TraceIdx"].values
        tree = KDTree(trace_xy, leaf_size=64)

        results = []

        with segyio.open(sgy_file, "r", ignore_geometry=True) as f_segy:
            f_segy.mmap()
            nsamp = f_segy.samples.size

            trace_cache = {}
            interp_cache = {}

            for _, row in tqdm(df_las.iterrows(), total=len(df_las), desc=f"{well_name} 样本构建中"):
                tvd = row["DEPT"]
                t = f_t(tvd)

                if not np.isfinite(t):
                    continue

                # ==== 实际点位振幅 ====
                seis_true = interpolate_seismic_amplitude_from_sgy(
                    x, y, t,
                    tree,
                    traceidx_array,
                    f_segy,
                    sampling_start_ms,
                    sampling_interval_ms,
                    nsamp,
                    trace_cache,
                    interp_cache
                )

                if np.isnan(seis_true):
                    continue

                seis_window = []
                valid = True

                for dx in xy_offsets:
                    for dy in xy_offsets:
                        for dt in t_offsets:
                            amp = interpolate_seismic_amplitude_from_sgy(
                                x + dx,
                                y + dy,
                                t + dt * sampling_interval_ms,
                                tree,
                                traceidx_array,
                                f_segy,
                                sampling_start_ms,
                                sampling_interval_ms,
                                nsamp,
                                trace_cache,
                                interp_cache
                            )

                            if np.isnan(amp):
                                valid = False
                                break

                            seis_window.append(amp)
                        if not valid:
                            break
                    if not valid:
                        break

                # ==== 时窗完整性判断 ====
                if not valid or len(seis_window) != 63:
                    continue

                well_attrs = {
                    col: row[col]
                    for col in df_las.columns
                    if col != "DEPT"
                }

                results.append({
                    "TVD": tvd,
                    "TIME": t,
                    "X": x,
                    "Y": y,
                    "SEIS_TRUE": seis_true,
                    **{f"SEIS_{i}": v for i, v in enumerate(seis_window)},
                    **well_attrs
                })

        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)

        print(f"✅ 样本已保存：{output_csv}，共 {len(df_out)} 条")

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("../construct_fail_log.txt", "a", encoding="utf-8") as f_log:
            f_log.write(f"[{timestamp}] {well_name} 构建失败：{e}\n")
        print(f"⚠️ 井 {well_name} 构建失败：{e}")


def construct_imaging_samples(xy, use_timedeep, output_dir, well_name):
    try:
        print(f"\n🚧 正在处理井：{well_name}")
        output_dir = os.path.join(output_dir, "FMI提取数据", well_name)
        x, y = xy  # ✅ 坐标直接由参数传入

        # 所需路径
        trace_header_csv = os.path.join(data_dir, "研究内容一", "trace_header_xy.csv")
        timedepth_file = os.path.join(data_dir, "时深", f"{use_timedeep}.dat")
        tracecube_file = os.path.join(data_dir, "研究内容一", "成像测井", "相关地震道信息",
                                      "测井-地震曲线", f"{well_name.split('_')[0]}_tracecube.csv")
        input_file = os.path.join(data_dir, "研究内容一", "成像测井", "FMI提取数据", well_name,
                                  f"{well_name}_processed.csv")
        output_csv = os.path.join(output_dir, f"{well_name}_around_data.csv")

        required_files = {
            "时深": timedepth_file,
            "TraceCube": tracecube_file,
            "TraceHeader": trace_header_csv,
            "成像测井": input_file
        }

        for k, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ 缺少文件：[{k}] {path}")

        # 加载数据
        df_csv = pd.read_csv(input_file)
        df_td = read_timedepth_file(timedepth_file)
        df_tracecube = pd.read_csv(tracecube_file)
        df_traceheader = pd.read_csv(trace_header_csv)

        f_t = interp1d(df_td['TVD'], df_td['TIME'], bounds_error=False, fill_value="extrapolate")

        # 构建 trace 索引查找结构
        global trace_sort_order
        trace_sort_order = build_trace_sort_order(df_traceheader)
        trace_xy = df_traceheader[['X', 'Y']].values.astype(np.float32)
        traceidx_array = df_traceheader['TraceIdx'].values
        tree = KDTree(trace_xy, leaf_size=64)

        # 地震数据字典
        trace_dict = {row["TraceIdx"]: row[1:].to_numpy() for _, row in df_tracecube.iterrows()}
        nsamp = len(next(iter(trace_dict.values())))

        window_half = 3
        window_size = 2 * window_half + 1
        results = []

        for _, row in tqdm(df_csv.iterrows(), total=len(df_csv), desc=f"{well_name} 样本构建中"):
            tvd = row["TVD"]
            t = f_t(tvd)

            if not np.isfinite(t):
                continue

            samp_idx = int(round((float(t) - sampling_start_ms) / sampling_interval_ms))

            # 取周围最近的9条道
            dist, idx = tree.query([[x, y]], k=9)
            nearest_traceidx = sort_traces_by_order(traceidx_array[idx[0]])

            waveform_matrix = []
            samp_start = max(0, samp_idx - window_half)
            samp_end = min(nsamp, samp_idx + window_half + 1)

            if samp_end - samp_start != window_size:
                continue

            for tid in nearest_traceidx:
                if tid in trace_dict:
                    data = trace_dict[tid]
                    waveform_matrix.append(data[samp_start:samp_end])

            if len(waveform_matrix) != 9:
                continue

            waveform_flat = np.array(waveform_matrix).flatten().tolist()

            # ==== 插值得到该坐标的实际振幅 ====
            dist_4, idx_4 = tree.query([[x, y]], k=4)
            traceids_4 = traceidx_array[idx_4[0]]
            distances = dist_4[0]

            amplitudes = []
            valid_weights = []

            for tid, d in zip(traceids_4, distances):
                if tid not in trace_dict:
                    continue
                if d == 0:
                    d = 1e-6

                trace_data = trace_dict[tid]
                trace_time = sampling_start_ms + np.arange(nsamp) * sampling_interval_ms
                f_interp = interp1d(trace_time, trace_data, kind='linear', bounds_error=False, fill_value="extrapolate")
                amp = f_interp(t)
                amplitudes.append(amp)
                valid_weights.append(1 / d)

            if len(amplitudes) == 0:
                continue

            weights = np.array(valid_weights)
            weights /= weights.sum()
            weighted_amp = np.dot(amplitudes, weights)

            well_attrs = {col: row[col] for col in df_csv.columns if col not in ["TVD", "TIME"]}

            results.append({
                "TVD": tvd, "TIME": t, "X": x, "Y": y,
                "SEIS_TRUE": weighted_amp,
                **{f"SEIS_{i}": v for i, v in enumerate(waveform_flat)},
                **well_attrs
            })

        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        print(f"✅ 样本已保存：{output_csv}，共 {len(df_out)} 条")
        print(
            f"起始时间: {min(df_out['TIME'].iloc[0], df_out['TIME'].iloc[-1])}, 结束时间: {max(df_out['TIME'].iloc[0], df_out['TIME'].iloc[-1])}")

    except Exception as e:
        print(f"❌ 处理井 {well_name} 失败：{e}")


# ==== 执行 ====
if __name__ == "__main__":
    data_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩"
    # # 斜井
    inclined_wells_path = os.path.join(data_dir, '井斜')
    inclined_wells = list(well.split('.')[0] for well in os.listdir(inclined_wells_path))
    output_inclined_wells_to_time_window_file = os.path.join(data_dir, "优化阶段一", "研究内容一", "井斜", "测井-地震时窗")
    os.makedirs(output_inclined_wells_to_time_window_file, exist_ok=True)
    processed_inclined_wells = list(well.split('_')[0] for well in os.listdir(output_inclined_wells_to_time_window_file))
    for well_name in inclined_wells:
        # if well_name in processed_inclined_wells:
        #     print(f"{well_name} 井已处理：{well_name}")
        #     continue
        construct_inclined_samples(data_dir, output_inclined_wells_to_time_window_file, well_name)

    # 普通测井
    # logging_wells_path = os.path.join(data_dir, '测井')
    # logging_wells = list(well.split('.')[0] for well in os.listdir(logging_wells_path) if well.endswith('.Las'))
    # logging_wells = list(well for well in logging_wells if well not in inclined_wells)
    # output_logging_wells_to_time_window_file = os.path.join(data_dir, "优化阶段一", "研究内容一", "测井", "测井-地震时窗")
    # os.makedirs(output_logging_wells_to_time_window_file, exist_ok=True)
    # processed_logging_wells = list(well.split('_')[0] for well in os.listdir(output_inclined_wells_to_time_window_file))
    # for well_name in logging_wells:
    #     if well_name in processed_logging_wells:
    #         print(f"{well_name} 井已处理：{well_name}")
    #         continue
    #     construct_logging_samples(data_dir, output_logging_wells_to_time_window_file, well_name)

    # 成像测井
    # pretreat_data_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\FMI提取数据"
    # imaging_wells = {
    #     "车660_1": {
    #         "file_path": r"车660_1\boreid-image\B72405.csv",
    #         "xy": [592331, 4212564],
    #         "use_timedeep": "车斜84"
    #     },
    #     "车660_2": {
    #         "file_path": r"车660_2\che660-down-boreid-image\B79911.csv",
    #         "xy": [592331, 4212564],
    #         "use_timedeep": "车古25"
    #     },
    #     "车662": {
    #         "file_path": r"车662\che662-fmi-boreid-image\B239983.csv",
    #         "xy": [592350, 4210905],
    #         "use_timedeep": "车74"
    #     },
    #     "车663": {
    #         "file_path": r"车663\che663-fmi-boreid-image\B227207.csv",
    #         "xy": [590100, 4212075],
    #         "use_timedeep": "车斜84"
    #     },
    # }
    # output_imaging_wells_to_time_window_file = os.path.join(data_dir, "优化阶段一", "研究内容一", "成像测井")
    # os.makedirs(output_imaging_wells_to_time_window_file, exist_ok=True)
    # for well_name in imaging_wells:
    #     construct_imaging_samples(imaging_wells[well_name]["xy"],
    #                               imaging_wells[well_name]["use_timedeep"],
    #                               output_imaging_wells_to_time_window_file,
    #                               well_name)
