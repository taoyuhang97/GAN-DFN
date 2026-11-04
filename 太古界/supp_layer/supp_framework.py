"""Runtime registration of the taigu_supp_v1 strata framework.

The original registry.py hard-codes its REGISTRY dict. This module only appends
one entry pointing at the config in this directory; no original code is edited.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO / "优化阶段二" / "正式主线"
for path in (FORMAL_ROOT, FORMAL_ROOT / "common"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.strata_framework import registry as framework_registry  # noqa: E402
from common.strata_framework import load_framework  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "taigu_supp_v1.json"
FRAMEWORK_ID = "taigu_supp_v1"


def register_supp_framework() -> None:
    """Register taigu_supp_v1 (idempotent)."""
    framework_registry.REGISTRY[FRAMEWORK_ID] = str(CONFIG_PATH.resolve())


def load_supp_framework():
    register_supp_framework()
    return load_framework(FRAMEWORK_ID)
