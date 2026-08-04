from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
TARGET_SCRIPT = ROOT / "run_batch_new_t4t7_conventional_prediction.py"


def main() -> int:
    if not TARGET_SCRIPT.exists():
        print(f"找不到入口脚本: {TARGET_SCRIPT}")
        return 1

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [PYTHON, "-X", "utf8", str(TARGET_SCRIPT)]
    print("启动命令:")
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
