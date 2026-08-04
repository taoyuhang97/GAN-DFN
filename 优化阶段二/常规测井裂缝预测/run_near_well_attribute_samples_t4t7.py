from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\项目\石油项目\断缝储")
PYTHON = sys.executable
SCRIPT = ROOT / "GAN-DFN-upload-first-round-improvement-20260420" / "模型训练" / "优化阶段二" / "build_near_well_attribute_samples.py"
TRACE_HEADER = ROOT / "原始数据" / "wx数据" / "砂砾岩" / "研究内容一" / "trace_header_xy.csv"
SEISMIC_ROOT = ROOT / "原始数据" / "wx数据" / "砂砾岩" / "补充材料-20260623"
TIMEDEPTH_ROOT = ROOT / "原始数据" / "wx数据" / "砂砾岩" / "层位"
LAYER_ROOT = ROOT / "原始数据" / "wx数据" / "砂砾岩" / "层位"
LOG_DIR = ROOT / "原始数据" / "wx数据" / "砂砾岩" / "测井"
OUTPUT_ROOT = ROOT / "GAN-DFN-upload-first-round-improvement-20260420" / "模型训练" / "优化阶段二" / "near_well_attribute_samples_logs_t4t7"

VOLUME_KEYWORDS_JSON = (
    '{"coherence":["T4-T7","相干","coherence"],'
    '"ant":["T4-T7","蚂蚁","ant"],'
    '"curvature":["T4-T7","曲率","curvature"]}'
)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(SCRIPT),
        "--log-dir",
        str(LOG_DIR),
        "--timedepth-root",
        str(TIMEDEPTH_ROOT),
        "--trace-header-csv",
        str(TRACE_HEADER),
        "--seismic-root",
        str(SEISMIC_ROOT),
        "--volume-keywords-json",
        VOLUME_KEYWORDS_JSON,
        "--output-root",
        str(OUTPUT_ROOT),
        "--layer-root",
        str(LAYER_ROOT),
        "--layer-top-name",
        "T4",
        "--layer-bottom-name",
        "T7",
        "--restrict-to-layer-interval",
        "--enable-volumes",
        "--overwrite",
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
