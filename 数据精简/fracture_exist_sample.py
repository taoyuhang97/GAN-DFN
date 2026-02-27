import os
import pandas as pd
import numpy as np

# 成像测井
pretreat_data_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\FMI提取数据"
imaging_wells = {
    "车660_1": {
        "data_file": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\che660_FMI\che660-client-result\che660-fmi-result\txt\fracture-porosity.txt",
        "columns": ["DEPTH", "PHIT", "VISO", "FVDC", "FVTL", "FVA", "FVAH", "FVPA"]
    },
    "车660_2": {
        "data_file": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\che660_FMI_1\che660-run2-result\che660-run2-fmi-result\txt\che660-down-fracture-porosity.txt",
        "columns": ["DEPTH", "PHIT", "VISO", "FVPA", "FVAH", "FVA", "FVTL", "FVDC"]
    },
    "车662": {
        "data_file": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\che662_686_FMI\che662-fmi-client-disk\txt\che662-fracture.txt",
        "columns": ["DEPTH", "FVDC", "FVTL", "FVA", "FVAH", "FVPA"]
    },
    "车663": {
        "data_file": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\che663_687_FMI\che663-fmi-client-disk\txt\che663-fracture-porosity.txt",
        "columns": ["DEPTH", "PHIT", "VISO", "FVDC", "FVTL", "FVA", "FVAH", "FVPA"]
    }
}
output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\裂缝存在样本"
os.makedirs(output_dir, exist_ok=True)


def is_in_segment(depth):
    for (start, end) in depth_segments:
        # 假设范围定义为：start <= depth <= end
        if start <= depth <= end:
            return 1
    return 0


for well in imaging_wells:
    # ==== 1. 读取 .txt 文件分析得到的 FVTL 有效区段 ====
    df = pd.read_csv(imaging_wells[well]["data_file"], sep=r"\s+", engine="python",
                     skiprows=4, na_values=["-999.25"], names=imaging_wells[well]["columns"])

    judge_col = 'FVTL'
    value_col = 'FVTL'

    # 存储结果
    segments = []

    start_idx = None
    values = []

    for idx, val in df[judge_col].items():
        if pd.notna(val):
            if start_idx is None:
                start_idx = idx
            values.append(df[value_col][idx])
        else:
            if start_idx is not None:
                # 处理一段非空区间
                weighted_avg = np.average(values)  # 可在此设置权重
                end_idx = idx
                segments.append((start_idx, end_idx, weighted_avg))
                # 重置
                start_idx = None
                values = []

    # 处理最后一段非空区间（如果文件结尾是非空）
    if start_idx is not None and values:
        weighted_avg = np.average(values)
        end_idx = df.index[-1]
        segments.append((start_idx, end_idx, weighted_avg))

    # 输出结果
    for s in segments:
        print(f"起始深度: {df['DEPTH'][s[1]]}, 结束深度: {df['DEPTH'][s[0]]}, 加权平均值: {s[2]:.4f},\
         裂缝数量: {(df['DEPTH'][s[0]] - df['DEPTH'][s[1]]) * s[2]:.4f}")

    # 构造有效区间段的实际深度值
    depth_segments = []

    for s in segments:
        start_depth = df['DEPTH'][s[1]]
        end_depth = df['DEPTH'][s[0]]
        depth_segments.append((start_depth, end_depth))  # 左闭右开或闭区间都可，根据你定义选择

    # ==== 2. 读取待标记的 CSV 文件 ====
    well_around_file = os.path.join(pretreat_data_dir, well, f"{well}_around_data.csv")
    df_csv = pd.read_csv(well_around_file)

    # 假设 CSV 中深度列为 'DEPTH'，新增一列为 'FRACTURE_FLAG'
    df_csv['FRACTURE_FLAG'] = 0  # 初始默认标记为 0

    # ==== 3. 对 CSV 每一行深度进行判断并标记 ====
    df_csv['FRACTURE_FLAG'] = df_csv['TVD'].apply(is_in_segment)

    # ==== 4. 可选：保存结果 ====
    out_csv = os.path.join(output_dir, f"{well}_fracture.csv")
    df_csv.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"标记完成，结果保存至：{out_csv}")
