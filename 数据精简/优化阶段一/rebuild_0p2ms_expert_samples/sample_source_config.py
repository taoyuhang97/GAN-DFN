from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGING_AROUND_UNIFORM_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\测井-地震时窗_uniform_0p2ms"
)
DEVIATED_AROUND_UNIFORM_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\井斜\测井-地震时窗_uniform_0p2ms"
)
LEGACY_SAMPLE_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"
)
RAW_LABEL_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\成像测井\裂缝标注"
)
FRACTURE_CSV_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝提取"
)
OUTPUT_SAMPLE_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本_uniform_0p2ms"
)

W151_DENSITY_LAS = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\che151HF_20240415XRMI\车151HF-成果数据\车151HF电成像  裂缝LAS文件\车151HF_裂缝参数_P10、P21、P33.las"
)
CHEYE1_DENSITY_LAS = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\车镇成像测井\车页1HF（导眼）井-FMI\DLIS&LAS成果数据\车页1HF井裂缝密度、裂缝长度、裂缝孔隙度成果数据_3496.5-3755m.las"
)


@dataclass(frozen=True)
class WellSampleBuildConfig:
    well_name: str
    around_csv: Path
    density_source_kind: str
    density_source_path: Path
    raw_point_path: Path
    output_csv: Path


DEFAULT_WELL_CONFIGS = [
    WellSampleBuildConfig(
        well_name="车151HF",
        around_csv=DEVIATED_AROUND_UNIFORM_DIR / "车151HF_around_data.csv",
        density_source_kind="las",
        density_source_path=W151_DENSITY_LAS,
        raw_point_path=FRACTURE_CSV_DIR / "车151HF_fractures.csv",
        output_csv=OUTPUT_SAMPLE_DIR / "车151HF_sample.csv",
    ),
    WellSampleBuildConfig(
        well_name="车660-1",
        around_csv=IMAGING_AROUND_UNIFORM_DIR / "车660_1_around_data.csv",
        density_source_kind="sample_csv",
        density_source_path=LEGACY_SAMPLE_DIR / "车660-1_sample.csv",
        raw_point_path=RAW_LABEL_DIR / "车660_1.xlsx",
        output_csv=OUTPUT_SAMPLE_DIR / "车660-1_sample.csv",
    ),
    WellSampleBuildConfig(
        well_name="车660-2",
        around_csv=IMAGING_AROUND_UNIFORM_DIR / "车660_2_around_data.csv",
        density_source_kind="sample_csv",
        density_source_path=LEGACY_SAMPLE_DIR / "车660-2_sample.csv",
        raw_point_path=RAW_LABEL_DIR / "车660_2.xlsx",
        output_csv=OUTPUT_SAMPLE_DIR / "车660-2_sample.csv",
    ),
    WellSampleBuildConfig(
        well_name="车662",
        around_csv=IMAGING_AROUND_UNIFORM_DIR / "车662_around_data.csv",
        density_source_kind="sample_csv",
        density_source_path=LEGACY_SAMPLE_DIR / "车662_sample.csv",
        raw_point_path=RAW_LABEL_DIR / "车662.xlsx",
        output_csv=OUTPUT_SAMPLE_DIR / "车662_sample.csv",
    ),
    WellSampleBuildConfig(
        well_name="车663",
        around_csv=IMAGING_AROUND_UNIFORM_DIR / "车663_around_data.csv",
        density_source_kind="sample_csv",
        density_source_path=LEGACY_SAMPLE_DIR / "车663_sample.csv",
        raw_point_path=RAW_LABEL_DIR / "车663.xlsx",
        output_csv=OUTPUT_SAMPLE_DIR / "车663_sample.csv",
    ),
    WellSampleBuildConfig(
        well_name="车页1导眼",
        around_csv=DEVIATED_AROUND_UNIFORM_DIR / "车页1导眼_around_data.csv",
        density_source_kind="las",
        density_source_path=CHEYE1_DENSITY_LAS,
        raw_point_path=FRACTURE_CSV_DIR / "车页1导眼_fractures.csv",
        output_csv=OUTPUT_SAMPLE_DIR / "车页1导眼_sample.csv",
    ),
]
