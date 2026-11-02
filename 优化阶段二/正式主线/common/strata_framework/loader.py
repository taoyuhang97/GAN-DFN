"""从 JSON 配置构造 `StrataFramework`。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Layer, StrataFramework, StrataFrameworkError, Surface


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise StrataFrameworkError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _as_int(value: Any, where: str) -> int:
    """严格接受 int；拒绝 float（含 3.0），避免 LayerCode 再次退化成浮点。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrataFrameworkError(
            f"{where}: LayerCode must be an integer literal, got {type(value).__name__} {value!r}"
        )
    return value


def framework_from_dict(payload: dict[str, Any], source: str = "<dict>") -> StrataFramework:
    framework_id = str(_require(payload, "framework_id", source))
    area = str(payload.get("area", framework_id))

    raw_surfaces = _require(payload, "surfaces", source)
    if not isinstance(raw_surfaces, list) or not raw_surfaces:
        raise StrataFrameworkError(f"{source}: surfaces must be a non-empty list")
    surfaces = []
    for index, item in enumerate(raw_surfaces):
        where = f"{source}:surfaces[{index}]"
        surfaces.append(
            Surface(
                id=str(_require(item, "id", where)),
                label=str(item.get("label", item["id"])),
                order=int(item.get("order", index)),
                source_file=item.get("source_file"),
                legacy_column=item.get("legacy_column"),
            )
        )

    raw_layers = _require(payload, "layers", source)
    if not isinstance(raw_layers, list) or not raw_layers:
        raise StrataFrameworkError(f"{source}: layers must be a non-empty list")
    layers = []
    for index, item in enumerate(raw_layers):
        where = f"{source}:layers[{index}]"
        layers.append(
            Layer(
                code=_as_int(_require(item, "code", where), where),
                name=str(_require(item, "name", where)),
                top=str(_require(item, "top", where)),
                base=str(_require(item, "base", where)),
                color=str(item.get("color", "#999999")),
                display_order=int(item.get("display_order", index)),
                inclusive_base=bool(item.get("inclusive_base", False)),
                extension_ms=float(item.get("extension_ms", 0.0)),
                presence_column=item.get("presence_column"),
                time_class=item.get("time_class"),
            )
        )

    framework = StrataFramework(
        framework_id=framework_id,
        area=area,
        surfaces=tuple(surfaces),
        layers=tuple(layers),
        extension_ms=float(payload.get("extension_ms", 0.0)),
        metadata=dict(payload.get("metadata") or {}),
        legacy_layer_aliases=dict(payload.get("legacy_layer_aliases") or {}),
    )
    framework.validate()
    return framework


def load_framework_file(path: Path) -> StrataFramework:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"strata framework config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return framework_from_dict(payload, source=str(path))
