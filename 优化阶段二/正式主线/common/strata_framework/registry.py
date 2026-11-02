"""地层框架注册表：framework_id 到配置文件的映射。

两个工区共用同一套框架代码，差异全部落在 configs 目录：
  shaligyan_v1  - 砂砾岩（T4/T5/T6/T7 + 沙三段/沙四段）
  taigu_v1      - 太古界（Top/Mid/Base + 上部复合层/太古界风化壳）
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .loader import load_framework_file
from .schema import StrataFramework, StrataFrameworkError


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REGISTRY: dict[str, str] = {
    "shaligyan_v1": "shaligyan_v1.json",
    "taigu_v1": "taigu_v1.json",
}

# 砂砾岩正式主线历史上默认就是这套，保持向后兼容
DEFAULT_FRAMEWORK_ID = "shaligyan_v1"


def available_frameworks() -> tuple[str, ...]:
    return tuple(REGISTRY)


@lru_cache(maxsize=None)
def load_framework(framework_id: str = DEFAULT_FRAMEWORK_ID) -> StrataFramework:
    key = str(framework_id or DEFAULT_FRAMEWORK_ID)
    filename = REGISTRY.get(key)
    if filename is None:
        raise StrataFrameworkError(
            f"unknown strata framework {key!r}; available={available_frameworks()}"
        )
    return load_framework_file(CONFIG_DIR / filename)


def resolve_framework(config: dict | None = None) -> StrataFramework:
    """从步骤配置里取 strata_framework / strata_framework_id，缺省用砂砾岩。"""
    payload = dict(config or {})
    framework_id = (
        payload.get("strata_framework")
        or payload.get("strata_framework_id")
        or DEFAULT_FRAMEWORK_ID
    )
    if isinstance(framework_id, dict):
        framework_id = framework_id.get("id", DEFAULT_FRAMEWORK_ID)
    path = payload.get("strata_framework_config")
    if path:
        return load_framework_file(Path(str(path)))
    return load_framework(str(framework_id))
