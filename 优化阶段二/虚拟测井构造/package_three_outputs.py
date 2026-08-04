from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
PACKAGE_ROOT = ROOT / "交付物_三样结果"
ZIP_PATH = ROOT / "交付物_三样结果.zip"

REAL_DELIVERY_ROOT = (
    ROOT.parents[1] / "模型训练" / "优化阶段二" / "delivery_packages" / "连续密度曲线和裂缝点位"
)

INVALID_SENTINELS = {-999.25, -9999.0, -99999.0, 9999.0, 99999.0, -9999999.0}


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        s = s.mask((s - sentinel).abs() < 1e-6)
    s = s.mask(s.abs() >= 1e6)
    return s


def clean_attribute_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        name = str(col)
        if any(
            key in name
            for key in [
                "SEIS",
                "COHERENCE",
                "ANT",
                "CURVATURE",
                "FRACTURE_INV",
                "相干体",
                "蚂蚁体",
                "曲率",
                "裂缝反演",
            ]
        ):
            out[col] = clean_numeric_series(out[col])
    return out


def split_csv_by_column(input_csv: Path, output_dir: Path, key_col: str, filename_col: str | None = None) -> int:
    if not input_csv.exists():
        return 0
    df = pd.read_csv(input_csv, low_memory=False, encoding="utf-8-sig")
    if key_col not in df.columns:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for key, group in df.groupby(key_col, sort=True):
        safe_name = str(key).strip().replace("/", "_").replace("\\", "_")
        file_name = f"{safe_name}.csv" if filename_col is None else f"{safe_name}_{filename_col}.csv"
        group.to_csv(output_dir / file_name, index=False, encoding="utf-8-sig")
        count += 1
    return count


def split_virtual_by_source(input_csv: Path, output_dir: Path) -> int:
    if not input_csv.exists():
        return 0
    df = pd.read_csv(input_csv, low_memory=False, encoding="utf-8-sig")
    if "SourceWellName" not in df.columns:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_well, source_df in df.groupby("SourceWellName", sort=True):
        safe_well = str(source_well).strip().replace("/", "_").replace("\\", "_")
        well_dir = output_dir / safe_well
        well_dir.mkdir(parents=True, exist_ok=True)
        for virtual_name, group in source_df.groupby("VirtualWellName", sort=True):
            safe_virtual = str(virtual_name).strip().replace("/", "_").replace("\\", "_")
            group.to_csv(well_dir / f"{safe_virtual}.csv", index=False, encoding="utf-8-sig")
            count += 1
    return count


def split_source_volume_by_well(
    input_csv: Path,
    output_dir: Path,
    points_only: bool = False,
) -> int:
    if not input_csv.exists():
        return 0
    df = pd.read_csv(input_csv, low_memory=False, encoding="utf-8-sig")
    if "SourceWellName" not in df.columns:
        return 0
    if points_only:
        df = df[df.get("HasFracture", 0).fillna(0).astype(float) > 0].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_well, group in df.groupby("SourceWellName", sort=True):
        safe_well = str(source_well).strip().replace("/", "_").replace("\\", "_")
        suffix = "井周外推裂缝点位" if points_only else "井周外推裂缝密度体"
        group.to_csv(output_dir / f"{safe_well}_{suffix}.csv", index=False, encoding="utf-8-sig")
        count += 1
    return count


def main() -> int:
    ensure_clean_dir(PACKAGE_ROOT)

    part1_dir = PACKAGE_ROOT / "01_整合后的真实测井连续裂缝密度曲线和实际裂缝点位"
    part2_dir = PACKAGE_ROOT / "02_整合后的虚拟测井连续裂缝密度曲线和实际裂缝点位"
    part3_dir = PACKAGE_ROOT / "03_基于井点的DFN井轨迹校正控制包"
    part1_dir.mkdir()
    part2_dir.mkdir()
    part3_dir.mkdir()

    part1_curve_src = REAL_DELIVERY_ROOT / "分井连续密度曲线_含井周属性"
    part1_point_src = REAL_DELIVERY_ROOT / "分井裂缝点位_含井周属性"
    part1_curve_dst = part1_dir / "分井连续裂缝密度曲线"
    part1_point_dst = part1_dir / "分井实际裂缝点位"
    part1_curve_dst.mkdir()
    part1_point_dst.mkdir()

    real_curve_count = 0
    for src in sorted(part1_curve_src.glob("*.csv")):
        df = clean_attribute_frame(pd.read_csv(src, low_memory=False, encoding="utf-8-sig"))
        df.to_csv(part1_curve_dst / src.name, index=False, encoding="utf-8-sig")
        real_curve_count += 1

    real_point_count = 0
    for src in sorted(part1_point_src.glob("*.csv")):
        df = clean_attribute_frame(pd.read_csv(src, low_memory=False, encoding="utf-8-sig"))
        df.to_csv(part1_point_dst / src.name, index=False, encoding="utf-8-sig")
        real_point_count += 1

    part2_sample_src = OUTPUT_ROOT / "步骤5_训练样本汇总" / "virtual_well_training_samples.csv"
    part2_curve_dst = part2_dir / "分源井井周外推裂缝密度体"
    part2_point_dst = part2_dir / "分源井井周外推裂缝点位"
    virtual_curve_count = split_source_volume_by_well(part2_sample_src, part2_curve_dst, points_only=False)
    virtual_point_count = split_source_volume_by_well(part2_sample_src, part2_point_dst, points_only=True)

    part2_virtual_raw_dst = part2_dir / "分源井_分虚拟井原始外推样本"
    virtual_raw_count = split_virtual_by_source(part2_sample_src, part2_virtual_raw_dst)

    part3_src = OUTPUT_ROOT / "步骤6_DFN井轨迹校正包" / "well_track_dfn_control_package.csv"
    part3_dst = part3_dir / "分井DFN井轨迹校正控制包"
    part3_dst.mkdir()
    dfn_control_count = 0
    if part3_src.exists():
        dfn_df = clean_attribute_frame(pd.read_csv(part3_src, low_memory=False, encoding="utf-8-sig"))
        dfn_df.to_csv(part3_dir / "DFN井轨迹校正控制包_总表.csv", index=False, encoding="utf-8-sig")
        temp_dfn_csv = part3_dir / "_temp_dfn.csv"
        dfn_df.to_csv(temp_dfn_csv, index=False, encoding="utf-8-sig")
        dfn_control_count = split_csv_by_column(temp_dfn_csv, part3_dst, "WellName", "DFN井轨迹校正控制包")
        temp_dfn_csv.unlink(missing_ok=True)

    summary = pd.DataFrame(
        [
            {
                "Part": "01",
                "Meaning": "整合后的真实测井连续裂缝密度曲线和实际裂缝点位",
                "CurveFiles": real_curve_count,
                "PointFiles": real_point_count,
                "Granularity": "分井",
            },
            {
                "Part": "02",
                "Meaning": "第1步结果向井周5x5地震道外推后的近井裂缝密度体与裂缝点位",
                "CurveFiles": virtual_curve_count,
                "PointFiles": virtual_point_count,
                "Granularity": "主交付按源井成体输出, 附带保留分虚拟井原始样本",
            },
            {
                "Part": "03",
                "Meaning": "对基于裂缝密度体细化后的DFN进行基于井点的裂缝矫正",
                "CurveFiles": dfn_control_count,
                "PointFiles": 0,
                "Granularity": "分井",
            },
        ]
    )
    summary.to_csv(PACKAGE_ROOT / "三样结果说明.csv", index=False, encoding="utf-8-sig")

    (PACKAGE_ROOT / "README.txt").write_text(
        "本交付物严格整理为三样东西, 不再以单个汇总 CSV 作为主要交付形态。\n"
        "01 是整合后的真实测井连续裂缝密度曲线和实际裂缝点位, 按井拆分。\n"
        "02 是第1步结果向井周 5x5 地震道外推后的近井裂缝密度体与裂缝点位, 主交付按源井输出一个体表达 CSV。\n"
        "03 是对基于裂缝密度体细化后的 DFN 进行基于井点的裂缝矫正控制包, 按井拆分。\n"
        "其中第 01 和第 02 都保留 X, Y, TIME, Density, HasFracture 以及已提取的井周属性体列。\n"
        "第 02 的主文件不是单个虚拟井列表, 而是围绕每口真实井聚合后的近井外推体表达; 同时附带保留分虚拟井原始样本, 便于回查。\n"
        "第 03 用于后续 DFN 细化结果与井轨迹预测结果对齐校正。\n",
        encoding="utf-8",
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in PACKAGE_ROOT.rglob("*"):
            zf.write(path, path.relative_to(ROOT))

    print(PACKAGE_ROOT)
    print(ZIP_PATH)
    print(f"real_curve_files={real_curve_count}")
    print(f"real_point_files={real_point_count}")
    print(f"virtual_curve_files={virtual_curve_count}")
    print(f"virtual_point_files={virtual_point_count}")
    print(f"virtual_raw_files={virtual_raw_count}")
    print(f"dfn_control_files={dfn_control_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
