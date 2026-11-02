"""地层框架（Strata Framework）的领域模型。

设计目标：把"有哪些层位面、哪些地层单元、每层用什么编码/颜色"从代码里挪到配置里，
让算法只遍历 `framework.layers`，不再出现 `if layer == "沙三段"` 这类判断。

三个概念：
  * Surface —— 一个层位界面（砂砾岩 T4/T5/T6/T7；太古界 Top/Mid/Base）；
  * Layer   —— 由上下两个界面圈出的地层单元（沙三段、太古界风化壳…）；
  * StrataFramework —— 某工区的一整套层位面 + 地层单元定义。

`LayerCode` 统一为 int，并在配置里显式声明；同一工区内唯一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class StrataFrameworkError(ValueError):
    """地层框架配置不合法。"""


@dataclass(frozen=True)
class Surface:
    """一个层位界面。"""

    id: str
    label: str
    order: int
    source_file: str | None = None
    # 历史数据表里的列名（砂砾岩 T4/T5/T6/T7；太古界 Top/Mid/Base 对应 T4/T6/T7）
    legacy_column: str | None = None

    @property
    def column(self) -> str:
        """读取历史契约时使用的列名。"""
        return self.legacy_column or self.id


@dataclass(frozen=True)
class Layer:
    """由上下两个层位界面圈出的地层单元。"""

    code: int
    name: str
    top: str          # Surface.id
    base: str         # Surface.id
    color: str = "#999999"
    display_order: int = 0
    # 层底是否包含在层内：砂砾岩沙四段 [T6, T7] 含底，沙三段 [T4, T6) 不含底
    inclusive_base: bool = False
    # 层底以下额外延伸的毫秒数（太古界风化壳 +50 ms）
    extension_ms: float = 0.0
    # 该层是否参与叠加（逐道）的列名；砂砾岩沿用 ShasanPresent/ShasiPresent
    presence_column: str | None = None

    def interval_label(self) -> str:
        return f"{self.top}->{self.base}"


@dataclass(frozen=True)
class StrataFramework:
    """一个工区的完整地层框架。"""

    framework_id: str
    area: str
    surfaces: tuple[Surface, ...]
    layers: tuple[Layer, ...]
    extension_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # 历史层名别名：如 {"Shasan": "沙三段", "Shasi": "沙四段"}
    legacy_layer_aliases: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------ surfaces
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.surfaces)

    def surface(self, surface_id: str) -> Surface:
        for item in self.surfaces:
            if item.id == surface_id:
                return item
        raise StrataFrameworkError(f"{self.framework_id}: unknown surface {surface_id!r}")

    def surface_columns(self) -> tuple[str, ...]:
        return tuple(item.column for item in self.surfaces)

    def layer_surface_columns(self) -> tuple[str, ...]:
        """被地层单元真正引用到的层位面列（未参与分层的面如 T5 不计入）。"""
        used: list[str] = []
        for layer in self.layers:
            for surface_id in (layer.top, layer.base):
                column = self.surface(surface_id).column
                if column not in used:
                    used.append(column)
        return tuple(used)

    # -------------------------------------------------------------- layers
    def layer_codes(self) -> tuple[int, ...]:
        return tuple(item.code for item in self.layers)

    def layer_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.layers)

    def layer_by_code(self, code: int | str) -> Layer:
        try:
            value = int(code)
        except (TypeError, ValueError) as exc:
            raise StrataFrameworkError(f"{self.framework_id}: layer code must be int, got {code!r}") from exc
        for item in self.layers:
            if item.code == value:
                return item
        raise StrataFrameworkError(f"{self.framework_id}: unknown layer code {value}")

    def layer_by_name(self, name: str) -> Layer:
        for item in self.layers:
            if item.name == str(name):
                return item
        raise StrataFrameworkError(f"{self.framework_id}: unknown layer name {name!r}")

    def layer(self, layer: int | str) -> Layer:
        """按 int code 或 name 取地层，便于逐步替换旧调用。"""
        if isinstance(layer, bool):
            raise StrataFrameworkError(f"{self.framework_id}: bool is not a layer reference")
        if isinstance(layer, (int,)):
            return self.layer_by_code(layer)
        text = str(layer)
        if text.isdigit():
            return self.layer_by_code(int(text))
        alias = self.resolve_layer_alias(text)
        if alias is not None:
            return alias
        return self.layer_by_name(text)

    def code_of(self, name: str) -> int:
        return self.layer_by_name(name).code

    def name_of(self, code: int) -> str:
        return self.layer_by_code(code).name

    def layer_code_map(self) -> dict[str, int]:
        """层名 → int 编码（替换散落在各步骤里的 LAYER_CODE 常量）。"""
        return {item.name: item.code for item in self.layers}

    def legacy_presence_columns(self) -> tuple[str, ...]:
        return tuple(item.presence_column for item in self.layers if item.presence_column)

    def resolve_layer_alias(self, alias: str) -> Layer | None:
        """把历史调用里的 Shasan/Shasi 等别名解析成地层。"""
        target = self.legacy_layer_aliases.get(str(alias))
        if target is None:
            return None
        return self.layer_by_name(target)

    # ----------------------------------------------------------- extension
    def extended_layer(self) -> Layer | None:
        for item in self.layers:
            if item.extension_ms > 0:
                return item
        return None

    # ----------------------------------------------------------- validation
    def validate(self) -> None:
        if not self.framework_id:
            raise StrataFrameworkError("framework_id is required")
        if not self.surfaces:
            raise StrataFrameworkError(f"{self.framework_id}: no surfaces declared")
        if not self.layers:
            raise StrataFrameworkError(f"{self.framework_id}: no layers declared")

        surface_ids = [item.id for item in self.surfaces]
        if len(surface_ids) != len(set(surface_ids)):
            raise StrataFrameworkError(f"{self.framework_id}: duplicate surface id")
        columns = [item.column for item in self.surfaces]
        if len(columns) != len(set(columns)):
            raise StrataFrameworkError(f"{self.framework_id}: duplicate surface column")

        ordered = [item.order for item in self.surfaces]
        if ordered != sorted(ordered):
            raise StrataFrameworkError(f"{self.framework_id}: surfaces must be declared in order")

        names = [item.name for item in self.layers]
        if len(names) != len(set(names)):
            raise StrataFrameworkError(f"{self.framework_id}: duplicate layer name")
        codes = [item.code for item in self.layers]
        if len(codes) != len(set(codes)):
            raise StrataFrameworkError(f"{self.framework_id}: duplicate layer code")
        for code in codes:
            if not isinstance(code, int) or isinstance(code, bool):
                raise StrataFrameworkError(
                    f"{self.framework_id}: LayerCode must be int, got {type(code).__name__} {code!r}"
                )

        known = set(surface_ids)
        presence_columns = [item.presence_column for item in self.layers if item.presence_column]
        if len(presence_columns) != len(set(presence_columns)):
            raise StrataFrameworkError(f"{self.framework_id}: duplicate presence column")
        for item in self.layers:
            if item.top not in known or item.base not in known:
                raise StrataFrameworkError(
                    f"{self.framework_id}: layer {item.name!r} references unknown surface "
                    f"({item.top!r} -> {item.base!r})"
                )
            if self.surface(item.top).order >= self.surface(item.base).order:
                raise StrataFrameworkError(
                    f"{self.framework_id}: layer {item.name!r} top must precede base"
                )
            if item.extension_ms < 0:
                raise StrataFrameworkError(f"{self.framework_id}: negative extension for {item.name!r}")
        for alias, target in self.legacy_layer_aliases.items():
            if target not in names:
                raise StrataFrameworkError(
                    f"{self.framework_id}: legacy alias {alias!r} points to unknown layer {target!r}"
                )

    # --------------------------------------------------------------- helper
    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "code": item.code,
                "name": item.name,
                "top": item.top,
                "base": item.base,
                "interval": item.interval_label(),
                "color": item.color,
                "extension_ms": item.extension_ms,
            }
            for item in self.layers
        ]


def layers_from_names(framework: StrataFramework, names: Iterable[str]) -> tuple[Layer, ...]:
    return tuple(framework.layer_by_name(name) for name in names)
