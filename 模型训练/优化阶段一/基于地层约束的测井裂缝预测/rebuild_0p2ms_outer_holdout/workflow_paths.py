from __future__ import annotations

from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parent


def find_project_root(start: Path | None = None) -> Path:
    current = (start or WORKFLOW_ROOT).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() and (candidate / "模型训练").exists():
            return candidate
        if (candidate / "README.md").exists() and (candidate / "模型训练").exists():
            return candidate
    raise FileNotFoundError(f"Failed to locate project root from: {current}")


PROJECT_ROOT = find_project_root(WORKFLOW_ROOT)
STAGE1_ROOT = PROJECT_ROOT / "模型训练" / "优化阶段一"
FRACTURE_EXISTENCE_LSTM_SCRIPT_PATH = (
    STAGE1_ROOT / "裂缝存在性分析" / "LSTM" / "imaging_well_to_fracture_cnn_lstm_test_1.py"
)
FRACTURE_POSITION_ANALYSIS_DIR = STAGE1_ROOT / "裂缝位置分析" / "基于密度的裂缝点位分析"
NEAREST_SCRIPT_PATH = FRACTURE_POSITION_ANALYSIS_DIR / "nearest_imaging_well_predict_then_refine.py"
RAW_POINT_SCRIPT_PATH = FRACTURE_POSITION_ANALYSIS_DIR / "raw_point_guided_segment_refine.py"
FRACTURE_DENSITY_SCRIPT_PATH = FRACTURE_POSITION_ANALYSIS_DIR / "fracture_density_predict.py"
FRACTURE_POINT_REFINE_SCRIPT_PATH = FRACTURE_POSITION_ANALYSIS_DIR / "fracture_point_refine_by_density.py"
