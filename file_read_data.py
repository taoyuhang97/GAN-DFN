from dlisio import dlis
import os
import csv
import numpy as np

all_channels = set()

def open_dlis(dlis_file_path, output_file_path):
    print("file name: ", os.path.basename(dlis_file_path), '\n')
    output_file = os.path.join(output_file_path, os.path.basename(dlis_file_path).split('.')[0])
    os.makedirs(output_file, exist_ok=True)
    # === 加载DLIS文件 ===
    with dlis.load(dlis_file_path) as files:
        for f in files:
            # === 遍历所有 Frame 和 Channel ===
            for frame in f.frames:
                frame_file = os.path.join(output_file, f"{frame.name}.csv")
                curves = frame.curves()

                # 获取通道名和列数
                channel_names = [ch.name for ch in frame.channels]
                n_channels = len(channel_names)

                # 判断哪些列是有效的（非 ndarray）
                valid_indices = []
                valid_channel_names = []
                for i in range(n_channels):
                    sample_value = curves[0][i + 1]  # 第0列通常是深度或序号
                    if not isinstance(sample_value, np.ndarray):
                        valid_indices.append(i + 1)
                        valid_channel_names.append(channel_names[i])
                    else:
                        print(f"跳过含数组的列：{channel_names[i]}")

                # 写入 CSV
                with open(frame_file, mode='w', newline='') as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(['SERIAL'] + valid_channel_names)

                    for row in curves:
                        row_out = [row[0]]  # SERIAL
                        row_out += [row[i] for i in valid_indices]
                        writer.writerow(row_out)

                print(f"保存完成: {frame_file}\n")

                # print(f"Frame: {frame.name}")
                # curves = frame.curves()
                # channel_names = [ch.name for ch in frame.channels]
                # # all_channels.update(channel_names)
                # for i, channel in enumerate(frame.channels):
                #     print(f"Channel: \t{channel.name}")
                #     print(f"值的格式: \t{curves[0][i + 1]}")
                #     print(f"值的类型: \t{type(curves[0][i + 1])}")
                #     if not channel.long_name is None:
                #         print(f"Long_name: \t{channel.long_name}")
                #     else:
                #         print(f"no long name")
                #     print()
                # print(channel_names, '\n')

imaging_logging_path = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井"
dlis_files = {
    "车660_1": r"che660_FMI\che660-client-result\che660-fmi-result\dlis",
    "车660_2": r"che660_FMI_1\che660-run2-result\che660-run2-fmi-result\dlis",
    "车662": r"che662_686_FMI\che662-fmi-client-disk\dlis",
    "车663": r"che663_687_FMI\che663-fmi-client-disk\dlis"
}
output_base_file = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一"

for well in dlis_files:
    dlis_file = os.path.join(imaging_logging_path, dlis_files[well])
    output_file_path = os.path.join(output_base_file, "成像测井", "FMI提取数据", well)
    os.makedirs(output_file_path, exist_ok=True)
    for file in os.listdir(dlis_file):
        if not file.lower().endswith('.dlis'):
            continue
        open_dlis(os.path.join(dlis_file, file), output_file_path)

# print("\n所有出现过的 Channel 名称（共 {} 个）:".format(len(all_channels)))
# for ch in sorted(all_channels):
#     print(ch)
