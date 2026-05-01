import os
import pandas as pd
import numpy as np
import re
import chardet

# === 输入路径 ===
other_well_position_file = r"/data/shared/project-oil/wx数据/砂砾岩/ExportWellHead.dat"
deep_time_dir = r"/data/shared/project-oil/wx数据/砂砾岩/时深"

# === 已知井坐标 ===
well_positions = {
    "车660_1": {
        "xy": [592331, 4212564],
        "depth": [3925.9764, 4309.4148]
    },
    "车660_2": {
        "xy": [592331, 4212564],
        "depth": [4297.679999999999, 4753.0512]
    },
    "车662": {
        "xy": [592350, 4210905],
        "depth": [3474.2628, 3990.1367999999998]
    },
    "车663": {
        "xy": [590100, 4212075],
        "depth": [3872.9411999999998, 4284.726]
    }
}


# === 1. 读取其他井头信息 ===
def read_wellhead_positions(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        header_line = lines[1].lstrip("#").strip()
        column_names = re.split(r'\s+', header_line)
        data_lines = lines[2:]
        data = [re.split(r'\s+', line.strip()) for line in data_lines if line.strip()]
        df = pd.DataFrame(data, columns=column_names)
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df = df.dropna(subset=["X", "Y"])
    return df


# === 2. 获取时深目录下的井名集合 ===
def get_deep_time_wellnames(directory):
    files = os.listdir(directory)
    wellnames = set()
    for file in files:
        if not file.endswith(".dat"):
            continue
        basename = os.path.splitext(file)[0]
        wellnames.add(basename)
    return wellnames


# === 3. 读取时深文件的深度范围 ===
def get_depth_range_from_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            encoding = chardet.detect(f.read(2048))['encoding']
        with open(filepath, "r", encoding=encoding, errors='ignore') as f:
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
        return df['TVD'].iloc[0], df['TVD'].iloc[-1]
    except Exception as e:
        print(f"读取时深文件失败: {filepath}, 错误: {e}")
        return None, None


# === 4. 查找包含深度范围的最近井 ===
def find_compatible_well(target_x, target_y, depth_range, candidates_df, time_dir):
    coords = candidates_df[["X", "Y"]].values
    dists = np.sqrt((coords[:, 0] - target_x) ** 2 + (coords[:, 1] - target_y) ** 2)
    sorted_indices = np.argsort(dists)

    for idx in sorted_indices:
        name = candidates_df.iloc[idx]["Name"]
        dist = dists[idx]
        time_file = os.path.join(time_dir, name + ".dat")
        min_d, max_d = get_depth_range_from_file(time_file)
        if min_d is None:
            continue
        if float(min_d) <= depth_range[0] and float(max_d) >= depth_range[1]:
            return name, dist, (min_d, max_d)

    return None, None, None  # 没有找到合适的井


# === 主程序 ===
if __name__ == "__main__":
    wellhead_df = read_wellhead_positions(other_well_position_file)
    deep_time_wells = get_deep_time_wellnames(deep_time_dir)

    candidates_df = wellhead_df[wellhead_df["Name"].isin(deep_time_wells)].reset_index(drop=True)

    compatible_wells = {}

    for well_name, info in well_positions.items():
        x, y = info["xy"]
        depth_range = info["depth"]
        nearest_name, dist, used_range = find_compatible_well(x, y, depth_range, candidates_df, deep_time_dir)
        if nearest_name:
            compatible_wells[well_name] = {
                "匹配井": nearest_name,
                "距离": round(dist, 1),
                "使用深度范围": used_range
            }
        else:
            compatible_wells[well_name] = {
                "匹配井": None,
                "信息": "未找到覆盖目标深度区间的井"
            }

    # 输出结果
    for well, info in compatible_wells.items():
        if info["匹配井"]:
            print(
                f"{well} --> {info['匹配井']}（距离: {info['距离']} 米，深度范围: {info['使用深度范围'][0]} ~ {info['使用深度范围'][1]}）")
        else:
            print(f"{well} --> ❌ {info['信息']}")
