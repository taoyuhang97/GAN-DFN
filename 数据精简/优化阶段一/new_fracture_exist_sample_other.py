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
    p10 = df_density["P10"].fillna(0).values
    p21 = df_density["P21"].fillna(0).values
    p33 = df_density["P33"].fillna(0).values

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


def build_fracture_seismic_samples(around_csv, fracture_file, density_file, columns, output_csv, key_col="P10"):
    # ===== 1. 读取测井-地震样本 =====
    df_samples = pd.read_csv(around_csv)

    # ===== 2. 读取 LAS 裂缝密度 =====
    df_density = pd.read_csv(density_file, sep=r"\s+", engine="python", skiprows=4, na_values=["-999.25"],
                             names=columns)
    # 裂缝密度参数名称映射
    df_density = df_density.rename(columns=column_change)
    # 原始裂缝密度文件为深度从深到浅排列，转换为从浅到深排列
    df_density = df_density.iloc[::-1].reset_index(drop=True)

    # ===== 3. 裂缝密度赋值（P10 / P21 / P33，按实际存在列） =====
    df_samples = assign_fracture_density(df_samples, df_density)

    # ===== 4. 裂缝产状赋值（FMI，仅做几何属性） =====
    if fracture_file is not None and os.path.exists(fracture_file):
        # 原始文件列名不一致
        df_frac = pd.read_excel(fracture_file, names=["MD", "Angle(0~90)", "Azimuth(0~360)"])
        df_samples = assign_fracture_orientation(df_samples, df_frac)
    else:
        print("⚠️ 未提供裂缝产状文件，仅使用密度信息")

    # ===== 5. 样本筛选：只保留 key_col 非空 =====
    if key_col not in df_samples.columns:
        raise ValueError(f"指定的筛选列不存在：{key_col}")

    before = len(df_samples)
    df_samples = df_samples[df_samples[key_col].notna()]
    after = len(df_samples)

    print(f"📌 样本筛选：{key_col} 非空 {after}/{before}")

    # ===== 6. 保存 =====
    df_samples.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ 裂缝-地震联合样本构建完成：{output_csv}")


if __name__ == "__main__":
    project_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩"
    around_data_dir = os.path.join(project_dir, "优化阶段一", "研究内容一", "成像测井", "测井-地震时窗")
    fracture_angle_dir = os.path.join(project_dir, "研究内容一", "成像测井", "裂缝标注")
    fracture_density_dir = os.path.join(project_dir, "车镇成像测井")
    column_change = {
        "FVDC": "P10",
        "FVTL": "P21",
        "FVPA": "P33"
    }
    well_info = {
        "车660-1": {
            "around_csv": os.path.join(around_data_dir, "车660_1_around_data_rebuild.csv"),
            "fracture_file": os.path.join(fracture_angle_dir, "车660_1.xlsx"),
            "density_file": os.path.join(fracture_density_dir,
                                         r"che660_FMI\che660-client-result\che660-fmi-result\txt\fracture-porosity.txt"),
            "columns": ["DEPTH", "PHIT", "VISO", "FVDC", "FVTL", "FVA", "FVAH", "FVPA"]
        },
        "车660-2": {
            "around_csv": os.path.join(around_data_dir, "车660_2_around_data.csv"),
            "fracture_file": os.path.join(fracture_angle_dir, "车660_2.xlsx"),
            "density_file": os.path.join(fracture_density_dir,
                                         r"che660_FMI_1\che660-run2-result\che660-run2-fmi-result\txt\che660-down-fracture-porosity.txt"),
            "columns": ["DEPTH", "PHIT", "VISO", "FVPA", "FVAH", "FVA", "FVTL", "FVDC"]
        },
        "车662": {
            "around_csv": os.path.join(around_data_dir, "车662_around_data.csv"),
            "fracture_file": os.path.join(fracture_angle_dir, "车662.xlsx"),
            "density_file": os.path.join(fracture_density_dir,
                                         r"che662_686_FMI\che662-fmi-client-disk\txt\che662-fracture.txt"),
            "columns": ["DEPTH", "FVDC", "FVTL", "FVA", "FVAH", "FVPA"]
        },
        "车663": {
            "around_csv": os.path.join(around_data_dir, "车663_around_data.csv"),
            "fracture_file": os.path.join(fracture_angle_dir, "车663.xlsx"),
            "density_file": os.path.join(fracture_density_dir,
                                         r"che663_687_FMI\che663-fmi-client-disk\txt\che663-fracture-porosity.txt"),
            "columns": ["DEPTH", "PHIT", "VISO", "FVDC", "FVTL", "FVA", "FVAH", "FVPA"]
        },
    }
    output_dir = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
    os.makedirs(output_dir, exist_ok=True)
    for well in well_info:
        build_fracture_seismic_samples(
            around_csv=well_info[well]["around_csv"],
            fracture_file=well_info[well]["fracture_file"],
            density_file=well_info[well]["density_file"],
            columns=well_info[well]["columns"],
            output_csv=os.path.join(output_dir, f"{well}_sample.csv"),
            key_col="P10"
        )
