from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

SCRIPTS = [
    ROOT / "步骤1_候选虚拟井索引" / "build_virtual_well_index.py",
    ROOT / "步骤2_井周属性提取" / "build_virtual_well_attribute_table.py",
    ROOT / "步骤3_弱标签生成" / "build_virtual_well_density_weaklabel.py",
    ROOT / "步骤4_置信度计算" / "build_virtual_well_confidence.py",
    ROOT / "步骤5_训练样本汇总" / "build_virtual_well_training_package.py",
    ROOT / "步骤6_DFN井轨迹校正包" / "build_well_track_dfn_control_package.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"[RUN] {script}")
        rc = subprocess.run([PYTHON, str(script)], cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"[FAIL] {script}")
            return rc
    print("[DONE] stage3 virtual well workflow finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
