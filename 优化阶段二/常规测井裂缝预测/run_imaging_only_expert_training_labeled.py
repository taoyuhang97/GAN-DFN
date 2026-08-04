from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
TRAIN_SCRIPT = (
    ROOT.parent
    / "优化阶段一"
    / "基于地层约束的测井裂缝预测"
    / "rebuild_0p2ms_outer_holdout"
    / "run_outer_holdout_expert_validation.py"
)

IMAGING_SAMPLE_DIR = ROOT / "near_well_attribute_samples_imaging" / "aligned_sample_csv_labeled_stage1_compatible"
OUTPUT_ROOT = ROOT / "training_runs_stage2_imaging_only_labeled"
DOCX_PATH = OUTPUT_ROOT / "imaging_only_expert_training_report.docx"

IMAGING_WELLS = ["车151HF", "车660-1", "车660-2", "车662", "车663"]
HOLDOUT_WELL = "车页1导眼"
TARGET_STRATA = "沙三段,沙四段"
FEATURES_JSON = '["AC","CAL","CNL","DEN","GR","RFOC","RILD","RILM","SP"]'
STAGE2_PROFILE_MAP_JSON = '{"沙三段":"sand3_probmass_rule_v1","沙四段":"sand4_balanced_v2"}'


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not TRAIN_SCRIPT.exists():
        print(f"找不到训练脚本: {TRAIN_SCRIPT}")
        return 1
    if not IMAGING_SAMPLE_DIR.exists():
        print(f"找不到样本目录: {IMAGING_SAMPLE_DIR}")
        return 1

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        PYTHON,
        "-X",
        "utf8",
        str(TRAIN_SCRIPT),
        "--exp-id",
        "imaging_only_stage1_rebuild_0p2ms_labeled",
        "--python-exe",
        str(PYTHON),
        "--sample-dir",
        str(IMAGING_SAMPLE_DIR),
        "--result-dir",
        str(OUTPUT_ROOT / "expert_library"),
        "--docx-path",
        str(DOCX_PATH),
        "--well-names",
        ",".join(IMAGING_WELLS),
        "--holdout-well",
        HOLDOUT_WELL,
        "--target-strata",
        TARGET_STRATA,
        "--stage1-lstm-features-json",
        FEATURES_JSON,
        "--stage1-seis-mode",
        "3x3",
        "--stage1-selection-metric",
        "iou",
        "--stage1-train-selection-mode",
        "distance",
        "--stage1-min-train-wells",
        "3",
        "--stage1-min-seq-per-well",
        "30",
        "--stage2-config-profile-map-json",
        STAGE2_PROFILE_MAP_JSON,
    ]
    print("启动命令:")
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
