import os
import pandas as pd
import chardet
import re
import unicodedata
from scipy.interpolate import interp1d
from tqdm import tqdm

# 数据目录和文件路径
pretreat_data_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\FMI提取数据"
base_time_deep_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\时深"
useful_datas = {
    "车660_1": {
        "file_path": r"车660_1\boreid-image\B72405.csv",
        "xy": [592331, 4212564],
        "use_timedeep": "车斜84"
    },
    "车660_2": {
        "file_path": r"车660_2\che660-down-boreid-image\B79911.csv",
        "xy": [592331, 4212564],
        "use_timedeep": "车古25"
    },
    "车662": {
        "file_path": r"车662\che662-fmi-boreid-image\B239983.csv",
        "xy": [592350, 4210905],
        "use_timedeep": "车74"
    },
    "车663": {
        "file_path": r"车663\che663-fmi-boreid-image\B227207.csv",
        "xy": [590100, 4212075],
        "use_timedeep": "车斜84"
    },
}

# 要修改的列单位
unit_map = {
    "TDEP": "TVD"
}

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

for well, well_info in useful_datas.items():
    input_file = os.path.join(pretreat_data_path, well_info["file_path"])
    print(f"处理文件: {well}")

    # 读取 CSV
    df = pd.read_csv(input_file)

    # 检查是否包含深度列
    if "TDEP" not in df.columns:
        raise ValueError(f"{input_file} 不包含 TDEP 列")

    # 深度单位转换：英寸转米（保留3位小数）
    df["TDEP"] = (df["TDEP"] * 0.1 * 0.0254)

    # 重命名列（添加单位）
    df.rename(columns={col: unit_map[col] for col in df.columns if col in unit_map}, inplace=True)

    df_td = read_timedepth_file(os.path.join(base_time_deep_dir, f"{well_info["use_timedeep"]}.dat"))
    f_t = interp1d(df_td['TVD'], df_td['TIME'], bounds_error=False, fill_value="extrapolate")
    # 获取 TVD 列的索引位置
    tvd_index = df.columns.get_loc("TVD")
    # 先生成 TIME，再插入到 TVD 后面
    df.insert(loc=tvd_index + 1, column="TIME", value=f_t(df["TVD"]))

    # 保存文件，添加"_processed"后缀
    output_file = os.path.join(pretreat_data_path, well, f"{well}_processed.csv")
    df = df.iloc[::-1].reset_index(drop=True)
    # print(df["TVD"].iloc[0], df["TVD"].iloc[-1])
    df.to_csv(output_file, index=False)
    print(f"保存处理后的文件: {output_file}")
