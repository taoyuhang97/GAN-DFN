"""地层框架：把层位面与地层单元的定义从代码中解耦。"""

from .loader import framework_from_dict, load_framework_file
from .registry import (
    DEFAULT_FRAMEWORK_ID,
    available_frameworks,
    load_framework,
    resolve_framework,
)
from .schema import (
    Layer,
    StrataFramework,
    StrataFrameworkError,
    Surface,
)


__all__ = [
    "DEFAULT_FRAMEWORK_ID",
    "Layer",
    "StrataFramework",
    "StrataFrameworkError",
    "Surface",
    "available_frameworks",
    "framework_from_dict",
    "load_framework",
    "load_framework_file",
    "resolve_framework",
]
