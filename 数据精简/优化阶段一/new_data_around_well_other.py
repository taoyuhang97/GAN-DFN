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


# ==== 读取 LAS ====
def read_imaging_well_las(file_path):
    # 自动识别编码
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(2048))['encoding']

    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()

    columns = None
    data = []
    in_ascii = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 进入 Ascii 段
        if line.lower().startswith('~ascii'):
            in_ascii = True
            continue

        if in_ascii:
            # 列名行（以 #DEPTH 开头）
            if line.startswith('#DEPTH'):
                columns = line.lstrip('#').split()
                continue

            # 实际数据行
            if not line.startswith('#'):
                data.append([float(v) for v in line.split()])

    return pd.DataFrame(data, columns=columns)


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


def construct_imaging_samples(xy, use_timedeep, output_dir, well_name, las_file, depth_range):
    try:
        print(f"\n🚧 正在处理井：{well_name}")

        xy_offsets = [-1250, 0, 1250]
        t_offsets = [-3, -2, -1, 0, 1, 2, 3]

        sampling_start_ms = 0
        sampling_interval_ms = 1

        trace_header_csv = os.path.join(data_dir, "研究内容一", "che66_2019_trace_header_xy.csv")
        las_file = las_file
        timedepth_file = os.path.join(data_dir, "时深", f"{use_timedeep}.dat")
        sgy_file = os.path.join(data_dir, "che66_2019_psdm_time.sgy")
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
        df_las = read_imaging_well_las(las_file)
        print(df_las.columns)
        print(df_las.head())
        df_td = read_timedepth_file(timedepth_file)
        df_traceheader = pd.read_csv(trace_header_csv)

        global trace_sort_order
        trace_sort_order = build_trace_sort_order(df_traceheader)

        # 井口坐标（输入是 m）
        x_m, y_m = xy
        # # === 统一转为 cm ===
        x = x_m * 100.0
        y = y_m * 100.0
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
                tvd = row["DEPTH"]
                if tvd < depth_range[0] - 1 or tvd > depth_range[1] + 1:
                    continue
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
                    interp_cache,
                    grid_spacing=1250
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
                                interp_cache,
                                grid_spacing=1250
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
                    if col != "DEPTH"
                }

                results.append({
                    "TVD": tvd,
                    "TIME": t,
                    "X": x / 100,
                    "Y": y / 100,
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


# ==== 执行 ====
if __name__ == "__main__":
    data_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩"

    # 成像测井
    pretreat_data_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\FMI提取数据"
    imaging_wells = {
        "车660_1": {
            "xy": [592331, 4212564],
            "use_timedeep": "车斜84",
            "ues_las": "车660@常规测井评价(2006-01-15)@3.las",
            "depth_range": [3927.1905, 4303.8471]
        },
        "车660_2": {
            "xy": [592331, 4212564],
            "use_timedeep": "车古25",
            "ues_las": "车660@常规测井评价(2006-03-15)@4.las",
            "depth_range": [4300.1336, 4750.9328]
        },
        "车662": {
            "xy": [592350, 4210905],
            "use_timedeep": "车74",
            "ues_las": "车662@常规测井评价(2006-08-19)@2.las",
            "depth_range": [3597.4223, 3965.4683]
        },
        "车663": {
            "xy": [590100, 4212075],
            "use_timedeep": "车斜84",
            "ues_las": "车663@常规测井评价(2006-08-07)@1.las",
            "depth_range": [3874.2645, 4270.9617]
        },
    }
    imaging_well_las_dir = os.path.join(data_dir, "成像测井-测井曲线")
    output_imaging_wells_to_time_window_file = os.path.join(data_dir, "优化阶段一", "研究内容一", "成像测井",
                                                            "测井-地震时窗")
    os.makedirs(output_imaging_wells_to_time_window_file, exist_ok=True)
    for well_name in imaging_wells:
        construct_imaging_samples(imaging_wells[well_name]["xy"],
                                  imaging_wells[well_name]["use_timedeep"],
                                  output_imaging_wells_to_time_window_file,
                                  well_name, os.path.join(imaging_well_las_dir, imaging_wells[well_name]["ues_las"]),
                                  imaging_wells[well_name]["depth_range"])
