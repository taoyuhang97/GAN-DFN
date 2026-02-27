import pandas as pd
import os

# ============================================================
# 参数区
# ============================================================

LAS_FILE = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\车页1HF（导眼）井-FMI\DLIS&LAS成果数据\车页1HF井成像测井蝌蚪成果数据_3496.5-3755m.las"
well_name = "车页1导眼"
SKIP_ROWS = 435

OUTPUT_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝提取"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 需要保留的裂缝类型（白名单）
KEEP_TYPES = ["高导缝"]

# 输出字段映射
COLUMN_REMAP = {
    "CenteredDepth": "MD",
    "Azimuth": "Azimuth(0~360)",
    "Dip_TRU": "Angle(0~90)"
}


# ============================================================
# 主函数
# ============================================================

def extract_fractures_from_las(las_path, skip_rows, keep_types, output_dir):
    records = []

    with open(las_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines()[skip_rows:]:

            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 10:
                continue

            try:
                centered_depth = float(parts[7])
                azimuth = float(parts[2])
                dip_tru = float(parts[9])
            except ValueError:
                continue

            # ⭐ Type = 行尾所有非数值内容
            type_str = " ".join(parts[31:]).replace('"', '').strip()

            records.append({
                "CenteredDepth": centered_depth,
                "Azimuth": azimuth,
                "Dip_TRU": dip_tru,
                "Type": type_str
            })

    df = pd.DataFrame(records)

    # 裂缝类型过滤
    df = df[df["Type"].isin(keep_types)].copy()

    # 字段重映射
    df = df.rename(columns=COLUMN_REMAP)
    df = df[list(COLUMN_REMAP.values())]

    # 输出
    output_path = os.path.join(output_dir, f"{well_name}_fractures.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("===================================")
    print("成像测井裂缝解释成果提取完成")
    print(f"井名       : {well_name}")
    print(f"裂缝类型   : {keep_types}")
    print(f"裂缝条数   : {len(df)}")
    print(f"输出文件   : {output_path}")
    print("===================================")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    extract_fractures_from_las(
        las_path=LAS_FILE,
        skip_rows=SKIP_ROWS,
        keep_types=KEEP_TYPES,
        output_dir=OUTPUT_DIR
    )
