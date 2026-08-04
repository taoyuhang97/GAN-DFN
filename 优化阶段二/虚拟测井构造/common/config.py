from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
PROJECT_STAGE_ROOT = WORKFLOW_ROOT.parents[1]
PROJECT_ROOT = PROJECT_STAGE_ROOT.parent

TRACE_HEADER_CSV = (
    PROJECT_ROOT / "原始数据" / "wx数据" / "砂砾岩" / "研究内容一" / "trace_header_xy.csv"
)
LAYER_DIR = PROJECT_ROOT / "原始数据" / "wx数据" / "砂砾岩" / "层位"
FAULT_PATCH_SUMMARY_CSV = (
    PROJECT_STAGE_ROOT
    / "小范围DFN生成"
    / "断层裂缝片生成"
    / "断层划分切割"
    / "fault_patches_out"
    / "fault_patches_summary.csv"
)

PREDICTION_ROOT = (
    PROJECT_STAGE_ROOT
    / "模型训练"
    / "优化阶段二"
    / "training_runs_stage2_imaging_only_labeled_t6_boundary"
    / "conventional_prediction"
)
CONVENTIONAL_SAMPLE_ROOT = (
    PROJECT_STAGE_ROOT
    / "模型训练"
    / "优化阶段二"
    / "near_well_attribute_samples_logs_stage1_compatible"
)

OUTPUT_ROOT = WORKFLOW_ROOT / "outputs"
STEP1_DIR = OUTPUT_ROOT / "步骤1_候选虚拟井索引"
STEP2_DIR = OUTPUT_ROOT / "步骤2_井周属性提取"
STEP3_DIR = OUTPUT_ROOT / "步骤3_弱标签生成"
STEP4_DIR = OUTPUT_ROOT / "步骤4_置信度计算"


@dataclass(frozen=True)
class NeighborConfig:
    xy_window_size: int = 5
    max_virtual_per_well: int = 25


@dataclass(frozen=True)
class WeakLabelConfig:
    distance_decay_length: float = 150.0
    min_confidence: float = 0.05


@dataclass(frozen=True)
class AttributeConfig:
    primary_attribute_cols: tuple[str, ...] = (
        "SEIS_TRUE",
        "COHERENCE",
        "ANT_TRACK",
        "CURVATURE_MAX",
        "CURVATURE_POS",
        "FRACTURE_INV",
    )


NEIGHBOR_CONFIG = NeighborConfig()
WEAK_LABEL_CONFIG = WeakLabelConfig()
ATTRIBUTE_CONFIG = AttributeConfig()
