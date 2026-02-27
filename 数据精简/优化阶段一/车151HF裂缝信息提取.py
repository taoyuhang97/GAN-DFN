import pandas as pd
import os

# ============================================================
# 参数区（根据需要修改）
# ============================================================

# 成像测井裂缝解释 LAS 文件路径
LAS_FILE = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\che151HF_20240415XRMI\车151HF-成果数据\车151HF电成像  裂缝LAS文件\车151HF_层理_裂缝产状_类型_裂缝宽度Dips_Final.las"
well_name = "车151HF"

# LAS 文件头段行数（不同软件可能不同）
SKIP_ROWS = 68

# 输出目录
OUTPUT_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝提取"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 原始文件属性
COLUMNS = [
    "MD",
    "Azimuth",
    "Dip_TRU",
    "FVA_APPARENT",
    "FVAH_APPARENT",
    "Type"
]

# 代表裂缝的类型（按实际情况增减）
KEEP_TYPES = [
    "Conductive Fracture"
]

# 最终保留字段及字段映射
COLUMN_REMAP = {
    "MD": "MD",
    "Azimuth": "Azimuth(0~360)",
    "Dip_TRU": "Angle(0~90)"
}


# ============================================================
# 主流程
# ============================================================

def extract_fractures_from_las(las_path, skip_rows, keep_types, output_dir):
    """
    从成像测井裂缝解释 LAS 中提取目标裂缝
    """
    with open(las_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[skip_rows:]

    records = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split(None, 5)  # ⭐ 只分裂前 5 个空白
        if len(parts) < 6:
            continue
        records.append(parts)

    df = pd.DataFrame(
        records,
        columns=COLUMNS
    )

    # 类型转换
    df[["MD", "Azimuth", "Dip_TRU", "FVA_APPARENT", "FVAH_APPARENT"]] = \
        df[["MD", "Azimuth", "Dip_TRU", "FVA_APPARENT", "FVAH_APPARENT"]].astype(float)

    # 清洗 Type
    df["Type"] = (
        df["Type"]
        .str.replace('"', '', regex=False)
        .str.strip()
    )

    fracture_df = df[df["Type"].isin(keep_types)].copy()

    # 只保留需要的原始字段
    fracture_df = fracture_df[list(COLUMN_REMAP.keys())]

    # 字段重映射
    fracture_df = fracture_df.rename(columns=COLUMN_REMAP)

    # 保证输出列顺序一致
    fracture_df = fracture_df[list(COLUMN_REMAP.values())]

    # 输出结果
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{well_name}_fractures.csv"
    )

    fracture_df.to_csv(output_path, index=False, encoding="utf-8")

    # -------- 7. 打印统计信息 --------
    print("===================================")
    print("成像测井裂缝解释成果提取完成")
    print(f"输入文件 : {las_path}")
    print(f"输出文件 : {output_path}")
    print(f"裂缝条数 : {len(fracture_df)}")
    print("===================================")


# ============================================================
# 脚本入口
# ============================================================

if __name__ == "__main__":
    extract_fractures_from_las(las_path=LAS_FILE, skip_rows=SKIP_ROWS, keep_types=KEEP_TYPES,
                               output_dir=OUTPUT_DIR)
