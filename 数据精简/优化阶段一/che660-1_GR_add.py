import os
import numpy as np
import pandas as pd
import lasio
from scipy.interpolate import interp1d


def append_additional_las_to_samples(base_csv, add_las_file, output_csv, depth_col_base="TVD", depth_col_las="DEPT",
                                     null_value=-999.25, prefix="ADD_"):
    """
    将采样间隔更粗的 LAS 测井，通过插值方式，
    对齐到已有样本 CSV 的深度轴，并按列追加。

    Parameters
    ----------
    base_csv : str
        construct_imaging_samples 输出的 CSV 文件
    add_las_file : str
        需要追加的 LAS 文件
    output_csv : str
        输出 CSV 路径
    depth_col_base : str
        基准 CSV 中的深度列名（默认 TVD）
    depth_col_las : str
        LAS 中的深度列名（默认 DEPT）
    null_value : float
        LAS 中的无效值
    prefix : str
        追加曲线的前缀，防止字段冲突
    """

    print("📥 读取基础样本 CSV ...")
    df_base = pd.read_csv(base_csv)

    if depth_col_base not in df_base.columns:
        raise ValueError(f"基础 CSV 中未找到深度列：{depth_col_base}")

    base_depth = df_base[depth_col_base].values

    print("📥 读取追加 LAS 文件 ...")
    las = lasio.read(add_las_file)
    df_add = las.df().reset_index()

    if depth_col_las not in df_add.columns:
        raise ValueError(f"LAS 中未找到深度列：{depth_col_las}")

    # 统一列名
    df_add = df_add.rename(columns={depth_col_las: "DEPTH"})

    # 无效值处理
    df_add = df_add.replace(null_value, np.nan)

    interp_cols = [c for c in df_add.columns if c != "DEPTH"]

    print(f"🔧 开始插值，共 {len(interp_cols)} 条曲线 ...")

    interp_data = {depth_col_base: base_depth}

    for col in interp_cols:
        valid = df_add[["DEPTH", col]].dropna()

        if len(valid) < 2:
            # 插值点不足，直接填 NaN
            interp_data[col] = np.full_like(base_depth, np.nan, dtype=float)
            continue

        f_interp = interp1d(
            valid["DEPTH"].values,
            valid[col].values,
            bounds_error=False,
            fill_value=np.nan
        )

        interp_data[col] = f_interp(base_depth)

    df_add_interp = pd.DataFrame(interp_data)

    # 删除重复深度列
    df_add_interp = df_add_interp.drop(columns=[depth_col_base])

    # 加前缀，防止字段冲突
    # df_add_interp = df_add_interp.add_prefix(prefix)

    print("🔗 合并基础样本与追加测井 ...")
    df_final = pd.concat([df_base, df_add_interp], axis=1)

    print(f"💾 保存结果：{output_csv}")
    df_final.to_csv(output_csv, index=False)

    print(f"✅ 追加完成，共 {len(df_final)} 条样本，新增 {len(df_add_interp.columns)} 个字段")


# =========================
# 示例调用（按需修改路径）
# =========================
if __name__ == "__main__":
    project_dir = r"/data/shared/project-oil/wx数据/砂砾岩"
    base_csv = os.path.join(project_dir, "优化阶段一", "研究内容一", "成像测井", "测井-地震时窗",
                            "车660_1_around_data.csv")
    add_las_file = os.path.join(project_dir, "成像测井-测井曲线", "车660@常规测井评价(2006-01-15)@1.las")
    output_csv = os.path.join(project_dir, "优化阶段一", "研究内容一", "成像测井", "测井-地震时窗",
                              "车660_1_around_data_rebuild.csv")

    append_additional_las_to_samples(
        base_csv=base_csv,
        add_las_file=add_las_file,
        output_csv=output_csv
    )
