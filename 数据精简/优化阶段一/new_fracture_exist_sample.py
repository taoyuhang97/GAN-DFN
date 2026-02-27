import os
import pandas as pd
import numpy as np
from bisect import bisect_left


def assign_fracture_orientation(df_samples, df_frac, density_cols=("P10", "P21", "P33")):
    """
    根据裂缝密度与裂缝解释结果为测井点赋予裂缝产状信息

    判定逻辑：
    1. 若裂缝密度参数存在且非 0 → 认为该点存在裂缝
    2. 存在裂缝时，取最近裂缝解释点的产状
    """

    df_samples = df_samples.copy()
    df_samples["Frac_Azimuth"] = np.nan
    df_samples["Frac_Dip"] = np.nan

    # 构造裂缝列表
    fractures = []
    for _, row in df_frac.iterrows():
        fractures.append({
            "z": row["MD"],
            "azi": row["Azimuth(0~360)"],
            "dip": row["Angle(0~90)"]
        })

    # 逐测井点处理
    for i, row in df_samples.iterrows():
        z = row["TVD"]

        # ---------- 1. 判断是否存在裂缝 ----------
        has_fracture = False
        for col in density_cols:
            if col in df_samples.columns:
                if pd.notna(row[col]) and row[col] > 0:
                    has_fracture = True
                    break

        if not has_fracture:
            continue

        # ---------- 2. 选择最近裂缝 ----------
        chosen = min(
            fractures,
            key=lambda f: abs(z - f["z"])
        )

        df_samples.at[i, "Frac_Azimuth"] = chosen["azi"]
        df_samples.at[i, "Frac_Dip"] = chosen["dip"]

    return df_samples


def read_fracture_density_las(las_path):
    """
    读取 LAS 格式裂缝密度文件
    数据区位于 ~Ascii 之后
    数据列：Depth, P10, P21, P33
    """

    ascii_start = None
    with open(las_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("~ascii"):
                ascii_start = i + 1
                break

    if ascii_start is None:
        raise ValueError(f"未在 LAS 文件中找到 ~Ascii 段：{las_path}")

    df = pd.read_csv(las_path, sep=r"\s+", engine="python", skiprows=ascii_start, names=["DEPTH", "P10", "P21", "P33"],
                     na_values=["-999.25", "-9999", "nan"])

    return df


def assign_fracture_density(df_samples, df_density):
    """
    根据 LAS 裂缝密度文件（P10/P21/P33）
    对测井点位进行线性插值赋值
    """

    df_samples = df_samples.copy()
    for col in ["P10", "P21", "P33"]:
        df_samples[col] = np.nan

    z_d = df_density["DEPTH"].values
    p10 = df_density["P10"].values
    p21 = df_density["P21"].values
    p33 = df_density["P33"].values

    for i, z in df_samples["TVD"].items():
        idx = bisect_left(z_d, z)

        if idx == 0 or idx >= len(z_d):
            continue

        z0, z1 = z_d[idx - 1], z_d[idx]
        if z1 == z0:
            continue

        w = (z - z0) / (z1 - z0)

        for name, arr in zip(
                ["P10", "P21", "P33"],
                [p10, p21, p33]
        ):
            v0, v1 = arr[idx - 1], arr[idx]
            if np.isnan(v0) or np.isnan(v1):
                continue

            df_samples.at[i, name] = v0 + (v1 - v0) * w

    return df_samples

def build_fracture_seismic_samples(around_csv, fracture_csv, density_las, output_csv, key_col="P10"):
    # 读取测井-地震样本
    df_samples = pd.read_csv(around_csv)

    # 读取 LAS 裂缝密度
    df_density = read_fracture_density_las(density_las)

    # 密度赋值（P10 / P21 / P33）
    df_samples = assign_fracture_density(df_samples, df_density)

    # 读取裂缝产状
    df_frac = pd.read_csv(fracture_csv)

    # 产状赋值（倾向、倾角）
    df_samples = assign_fracture_orientation(df_samples, df_frac)

    # ===== 关键修改：只保留指定列非空的样本 =====
    if key_col not in df_samples.columns:
        raise ValueError(f"指定的筛选列不存在：{key_col}")

    before = len(df_samples)
    df_samples = df_samples[df_samples[key_col].notna()]
    after = len(df_samples)

    print(f"📌 样本筛选：{key_col} 非空 {after}/{before}")

    # 保存
    df_samples.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ 裂缝-地震联合样本构建完成：{output_csv}")


if __name__ == "__main__":
    well_info = {
        "车151HF": {
            "around_csv": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\井斜\测井-地震时窗\车151HF_around_data.csv",
            "fracture_csv": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝提取\车151HF_fractures.csv",
            "density_las": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\che151HF_20240415XRMI\车151HF-成果数据\车151HF电成像  裂缝LAS文件\车151HF_裂缝参数_P10、P21、P33.las",
        },
        "车页1导眼": {
            "around_csv": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\井斜\测井-地震时窗\车页1导眼_around_data.csv",
            "fracture_csv": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝提取\车页1导眼_fractures.csv",
            "density_las": r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\车页1HF（导眼）井-FMI\DLIS&LAS成果数据\车页1HF井裂缝密度、裂缝长度、裂缝孔隙度成果数据_3496.5-3755m.las",
        }
    }
    output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
    os.makedirs(output_dir, exist_ok=True)
    for well in well_info:
        build_fracture_seismic_samples(
            around_csv=well_info[well]["around_csv"],
            fracture_csv=well_info[well]["fracture_csv"],
            density_las=well_info[well]["density_las"],
            output_csv=os.path.join(output_dir, f"{well}_sample.csv"),
            key_col = "P10"
        )
