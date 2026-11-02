#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证第 3 阶段（Step6–9b）地层框架改造。

检查项：
  1. 框架新增 API：layer_slug_map / layer_by_interval / surface_pair；
  2. Step9b：LAYERS 与输出 slug 保持 sha3/sha4；
  3. Step9：INTERVAL_LABELS / INTERVAL_COLORS 键与框架一致；
  4. Step7C：LayerCode 统一为框架 int（3/4），不再是历史的 1/2；
  5. Step7B：surface_pair 映射与旧 if/else 一致；
  6. Step6C：local_layer_relative_position 与旧硬编码逐元素一致（含 T6/T7 边界）。

用法::

    python3 scripts/verify_strata_step6_9b.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "优化阶段二" / "正式主线"
for path in (FORMAL_ROOT, FORMAL_ROOT / "common"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.strata_framework import load_framework  # noqa: E402


FRAMEWORK = load_framework("shaligyan_v1")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  [PASS] {message}")


def load_module(path: Path):
    path = path.resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    name = f"_verify3_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_framework_api() -> None:
    print("\n[1] 框架新增 API")
    check(FRAMEWORK.layer_slug_map() == {"沙三段": "sha3", "沙四段": "sha4"},
          "layer_slug_map 保持 sha3/sha4")
    check(FRAMEWORK.layer_by_interval("T4->T6").code == 3, "T4->T6 -> code 3")
    check(FRAMEWORK.layer_by_interval("T6->T7").code == 4, "T6->T7 -> code 4")
    check(FRAMEWORK.surface_pair("沙三段") == ("T4", "T6"), "沙三段 surface_pair = T4/T6")
    check(FRAMEWORK.surface_pair("沙四段") == ("T6", "T7"), "沙四段 surface_pair = T6/T7")


def verify_step9b() -> None:
    print("\n[2] Step9b")
    module = load_module(FORMAL_ROOT / "step9b_dfn_density_metrics/dfn_density_metrics.py")
    check(tuple(module.LAYERS) == FRAMEWORK.layer_names(), "LAYERS 来自框架")
    check(module.LAYER_SLUG == FRAMEWORK.layer_slug_map(), "LAYER_SLUG 来自框架")

    plot = load_module(FORMAL_ROOT / "step9b_dfn_density_metrics/plot_wells_on_p33.py")
    check(tuple(plot.LAYERS) == FRAMEWORK.layer_names(), "plot_wells LAYERS 来自框架")
    check(plot.LAYER_KEYS[:2] == (("沙三段", "sha3", "沙三段"), ("沙四段", "sha4", "沙四段")),
          "plot_wells LAYER_KEYS 保持历史文件名")


def verify_step9() -> None:
    print("\n[3] Step9 图例映射")
    module = load_module(
        FORMAL_ROOT / "step9_section_visualize/build_all_area_section_visualization.py"
    )
    expected = {"T4->T6": "沙三段(T4-T6)", "T6->T7": "沙四段(T6-T7)"}
    check(module.INTERVAL_LABELS == expected, f"INTERVAL_LABELS 保持旧标签 {expected}")
    check(set(module.INTERVAL_COLORS) == set(module.INTERVAL_LABELS),
          "INTERVAL_COLORS 覆盖全部框架地层")

    demo = load_module(
        FORMAL_ROOT / "step9_section_visualize/build_demo_well_section_visualization.py"
    )
    check(demo.INTERVAL_LABELS == expected, "demo 剖面 INTERVAL_LABELS 一致")


def verify_step7c() -> None:
    print("\n[4] Step7C LayerCode")
    module = load_module(FORMAL_ROOT / "step7c_large_fault_dfn/build_large_fault_dfn.py")
    mapping = {
        interval: module.FRAMEWORK.layer_by_interval(interval).code
        for interval in ("T4->T6", "T6->T7")
    }
    check(mapping == {"T4->T6": 3, "T6->T7": 4},
          "Step7C 区间 -> LayerCode 为 3/4（修复历史 1/2）")


def verify_step7b() -> None:
    print("\n[5] Step7B surface_pair")
    module = load_module(
        FORMAL_ROOT / "step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py"
    )
    for layer in FRAMEWORK.layer_names():
        pair = module.FRAMEWORK.surface_pair(layer)
        expected = ("T4", "T6") if layer == "沙三段" else ("T6", "T7")
        check(pair == expected, f"{layer}: surface_pair={pair}")


def verify_step6c() -> None:
    print("\n[6] Step6C 相对位置计算")
    module = load_module(
        FORMAL_ROOT / "step6b_demo_density_volume_3d/build_step6c_large_fault_prior.py"
    )
    yy = np.zeros(5, dtype=int)
    xx = np.zeros(5, dtype=int)
    surfaces = {
        "T4_TIME": np.array([[10.0]]),
        "T6_TIME": np.array([[20.0]]),
        "T7_TIME": np.array([[30.0]]),
        "ShasanPresent": np.array([[1]], dtype=np.uint8),
        "ShasiPresent": np.array([[1]], dtype=np.uint8),
    }
    time_ms = np.array([5.0, 10.0, 20.0, 30.0, 35.0])
    code, relative = module.local_layer_relative_position(yy, xx, time_ms, surfaces)

    # 旧硬编码：shasan 用 <=T6，shasi 用 >=T6 且 <=T7，后者覆盖前者的 T6 边界
    t4, t6, t7 = 10.0, 20.0, 30.0
    legacy = np.full(len(time_ms), np.nan)
    legacy_code = np.zeros(len(time_ms), dtype=np.uint8)
    shasan = (time_ms >= t4) & (time_ms <= t6)
    shasi = (time_ms >= t6) & (time_ms <= t7)
    legacy_code[shasan] = 1
    legacy_code[shasi] = 2
    legacy[shasan] = (time_ms[shasan] - t4) / max(t6 - t4, 1e-6)
    legacy[shasi] = (time_ms[shasi] - t6) / max(t7 - t6, 1e-6)

    check(np.array_equal(code, legacy_code), f"layer_code 与旧逻辑一致 {code.tolist()}")
    check(np.allclose(relative, legacy, equal_nan=True), f"relative 与旧逻辑一致 {relative.tolist()}")
    check(code[2] == 2, "T6 边界归沙四段（与旧逻辑覆盖顺序一致）")
    check(code[3] == 2, "T7 边界归沙四段")

    # 回归：T6 处若沙四缺失，旧逻辑仍把该点归到沙三段（闭区间）
    surfaces_no_crust = dict(surfaces)
    surfaces_no_crust["ShasiPresent"] = np.array([[0]], dtype=np.uint8)
    code2, relative2 = module.local_layer_relative_position(
        yy[:1], xx[:1], np.array([20.0]), surfaces_no_crust
    )
    check(code2.tolist() == [1], f"T6 且沙四缺失 -> 沙三段（旧语义）{code2.tolist()}")
    check(np.allclose(relative2, [1.0]), "T6 且沙四缺失 -> relative=1.0")


def main() -> int:
    print("=" * 70)
    print("第 3 阶段（Step6–9b）地层框架验证")
    print("=" * 70)
    verify_framework_api()
    verify_step9b()
    verify_step9()
    verify_step7c()
    verify_step7b()
    verify_step6c()
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
