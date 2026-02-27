from dlisio import dlis
import os
import csv
import numpy as np
from collections import defaultdict

# === 全局变量 ===
channel_set_by_well = defaultdict(set)  # 每口井的通道集合
channel_longname_map = dict()           # === 新增：通道名 → long name
channel_uom_map = dict()  # 通道名 → 单位
all_channels = set()

def open_dlis(dlis_file_path, well_name):
    print("file name: ", os.path.basename(dlis_file_path), '\n')

    with dlis.load(dlis_file_path) as files:
        for f in files:
            for frame in f.frames:
                curves = frame.curves()
                channel_names = [ch.name for ch in frame.channels]
                n_channels = len(channel_names)

                for i in range(n_channels):
                    sample_value = curves[0][i + 1]
                    if not isinstance(sample_value, np.ndarray):
                        ch_name = channel_names[i]
                        channel_set_by_well[f"{well_name}_{frame.name}"].add(ch_name)
                        all_channels.add(ch_name)

                        # === 新增：记录 long_name，如果不存在则为 None
                        ch_long = frame.channels[i].long_name
                        if ch_name not in channel_longname_map:
                            channel_longname_map[ch_name] = ch_long

                        # === 新增：记录单位
                        ch_unit = frame.channels[i].units
                        if ch_name not in channel_uom_map:
                            channel_uom_map[ch_name] = ch_unit

                    else:
                        print(f"跳过含数组的列：{channel_names[i]}")

# === 设置路径 ===
imaging_logging_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井"
dlis_files = {
    "车660_1": r"che660_FMI\che660-client-result\che660-fmi-result\dlis",
    "车660_2": r"che660_FMI_1\che660-run2-result\che660-run2-fmi-result\dlis",
    "车662": r"che662_686_FMI\che662-fmi-client-disk\dlis",
    "车663": r"che663_687_FMI\che663-fmi-client-disk\dlis"
}

# === 遍历井并提取属性 ===
for well, subpath in dlis_files.items():
    dlis_file = os.path.join(imaging_logging_path, subpath)
    for file in os.listdir(dlis_file):
        if not file.lower().endswith('.dlis'):
            continue
        open_dlis(os.path.join(dlis_file, file), well)

# === 新增：写出通道名称与 long_name + uom 映射 ===
output_map_csv = "通道名称映射.csv"
with open(output_map_csv, mode='w', newline='', encoding='utf-8-sig') as f_map:
    writer = csv.writer(f_map)
    writer.writerow(["Channel Name", "Long Name", "Unit"])
    for ch in sorted(all_channels):
        long_name = channel_longname_map.get(ch, "")
        uom = channel_uom_map.get(ch, "")
        writer.writerow([
            ch,
            long_name if long_name is not None else "",
            uom if uom is not None else ""
        ])

# === 写出统计矩阵 CSV（每个 frame 为一行）===
output_csv = "每井每帧通道矩阵统计.csv"
all_channels_sorted = sorted(all_channels)

with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Well_Frame Name"] + all_channels_sorted)

    for well_frame in sorted(channel_set_by_well.keys()):
        row = [well_frame]
        for ch in all_channels_sorted:
            row.append(1 if ch in channel_set_by_well[well_frame] else 0)
        writer.writerow(row)

print(f"\n每井每帧通道矩阵统计表已保存到: {output_csv}")
