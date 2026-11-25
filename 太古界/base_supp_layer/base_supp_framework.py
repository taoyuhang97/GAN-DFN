"""Runtime registration for the Taigu base-below supplementary layer."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO / "优化阶段二" / "正式主线"
for path in (FORMAL_ROOT, FORMAL_ROOT / "common"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.strata_framework import load_framework  # noqa: E402
from common.strata_framework import registry as framework_registry  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "taigu_base_supp_v1.json"
FRAMEWORK_ID = "taigu_base_supp_v1"


def register_base_supp_framework() -> None:
    framework_registry.REGISTRY[FRAMEWORK_ID] = str(CONFIG_PATH.resolve())


def load_base_supp_framework():
    register_base_supp_framework()
    return load_framework(FRAMEWORK_ID)
