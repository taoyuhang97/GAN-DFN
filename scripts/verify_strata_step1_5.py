#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证第 2 阶段（Step1–5）地层框架改造与旧硬编码等价。

检查项：
  1. Step1 surface_tools：SURFACE_CODES 来自框架；classify_time_domain 与旧公式逐样本一致；
  2. Step1 attach_sample_horizons：TIME_OUTPUT_MAP / CLASS_TO_LAYER_GROUP 与旧映射一致；
  3. Step2：TIME_CLASS_TO_LAYER 与旧映射一致，T4Time..T7Time 列齐全；
  4. Step4：TARGET_STRATA 等于框架层名；
  5. Step5A：layer_bounds 与旧 np.where 行为一致（含未知地层的兜底）；
  6. Step3：MANUAL_STRATA 中的地层名都在框架内。

用法::

    python3 scripts/verify_strata_step1_5.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "优化阶段二" / "正式主线"
for path in (FORMAL_ROOT, FORMAL_ROOT / "common", FORMAL_ROOT / "step1_surface_framework"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.strata_framework import load_framework  # noqa: E402


FRAMEWORK = load_framework("shaligyan_v1")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  [PASS] {message}")


def load_module(path: Path):
    """按文件路径导入模块（注册到 sys.modules，兼容 dataclass）。"""
    path = path.resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    name = f"_verify_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_step1_surface_tools() -> None:
    print("\n[1] Step1 surface_tools")
    module = load_module(FORMAL_ROOT / "step1_surface_framework/surface_tools.py")
    check(tuple(module.SURFACE_CODES) == FRAMEWORK.surface_ids(),
          f"SURFACE_CODES 来自框架 {module.SURFACE_CODES}")

    # 边界样本：正好落在 T4 / T6 / T7 以及层外
    frame = pd.DataFrame({
        "TIME": [99.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, np.nan],
        "T4_TIME": 100.0,
        "T6_TIME": 110.0,
        "T7_TIME": 120.0,
    })
    actual = module.classify_time_domain(frame)["TimeDomainClass"].tolist()

    time = frame["TIME"].to_numpy(dtype=float)
    t4 = frame["T4_TIME"].to_numpy(dtype=float)
    t6 = frame["T6_TIME"].to_numpy(dtype=float)
    t7 = frame["T7_TIME"].to_numpy(dtype=float)
    legacy = np.full(len(frame), "out_of_target", dtype=object)
    m3 = np.isfinite(time) & (time >= t4) & (time < t6)
    m4 = np.isfinite(time) & (time >= t6) & (time < t7)
    legacy[m3] = "sha3_t4_t6"
    legacy[m4] = "sha4_t6_t7"
    check(actual == legacy.tolist(), "classify_time_domain 与旧公式逐样本一致（含 T4/T6/T7 边界）")
    check(actual[0] == "out_of_target" and actual[3] == "sha4_t6_t7",
          "边界语义正确：99ms 层外、110ms 归沙四段")


def verify_step1_attach() -> None:
    print("\n[2] Step1 attach_sample_horizons")
    module = load_module(FORMAL_ROOT / "step1_surface_framework/attach_sample_horizons.py")
    expected_map = {"T4_TIME": "T4Time", "T5_TIME": "T5Time", "T6_TIME": "T6Time", "T7_TIME": "T7Time"}
    check(module.TIME_OUTPUT_MAP == expected_map, f"TIME_OUTPUT_MAP 与旧映射一致 {expected_map}")
    expected_class = {"sha3_t4_t6": "沙三段", "sha4_t6_t7": "沙四段", "out_of_target": "OUT_OF_TARGET"}
    check(module.CLASS_TO_LAYER_GROUP == expected_class, "CLASS_TO_LAYER_GROUP 与旧映射一致")


def verify_step2() -> None:
    print("\n[3] Step2 real well samples")
    module = load_module(FORMAL_ROOT / "step2_real_well_t4_t7_samples/build_real_well_t4_t7_samples.py")
    expected = {"sha3_t4_t6": "沙三段", "sha4_t6_t7": "沙四段"}
    check(module.TIME_CLASS_TO_LAYER == expected, "TIME_CLASS_TO_LAYER 与旧映射一致")
    check(module.SURFACE_TIME_COLUMNS == ("T4Time", "T5Time", "T6Time", "T7Time"),
          "SURFACE_TIME_COLUMNS 为 T4Time..T7Time")


def verify_step4() -> None:
    print("\n[4] Step4 target strata")
    expected = list(FRAMEWORK.layer_names())
    modules = [
        "step4_expert_real_well_prediction/train_and_predict_expert_library.py",
        "step4_expert_real_well_prediction/train_and_predict_gr_resistivity.py",
        "step4_expert_real_well_prediction/train_and_predict_gr_resistivity_strata_library.py",
        "step4_expert_real_well_prediction/train_and_predict_six_expert_library.py",
    ]
    for rel in modules:
        module = load_module(FORMAL_ROOT / rel)
        check(list(module.TARGET_STRATA) == expected, f"{Path(rel).name}: TARGET_STRATA 来自框架")


def verify_step5a() -> None:
    print("\n[5] Step5A layer_bounds")
    module = load_module(FORMAL_ROOT / "step5a_single_source_virtual_wells/build_single_source_virtual_wells.py")
    strata = np.array(["沙三段", "沙四段", "UNKNOWN"], dtype=object)
    surface_times = {
        "T4": np.array([10.0, 10.0, 10.0]),
        "T6": np.array([20.0, 20.0, 20.0]),
        "T7": np.array([30.0, 30.0, 30.0]),
    }
    top, base = module.layer_bounds(strata, surface_times)
    legacy_top = np.where(strata == "沙三段", surface_times["T4"], surface_times["T6"])
    legacy_base = np.where(strata == "沙三段", surface_times["T6"], surface_times["T7"])
    check(np.array_equal(top, legacy_top), "layer_bounds.top 与旧 np.where 一致")
    check(np.array_equal(base, legacy_base), "layer_bounds.base 与旧 np.where 一致")
    check(np.allclose([top[0], base[0]], [10.0, 20.0]), "沙三段 -> T4/T6")
    check(np.allclose([top[1], base[1]], [20.0, 30.0]), "沙四段 -> T6/T7")
    check(np.allclose([top[2], base[2]], [20.0, 30.0]), "未知地层保持旧的沙四段兜底")


def verify_step3() -> None:
    print("\n[6] Step3 manual strata names")
    module = load_module(FORMAL_ROOT / "step3_imaging_supervision_samples/build_formal_rebuild_groups.py")
    names = {strata for _, strata, _, _ in module.MANUAL_STRATA}
    check(names <= set(FRAMEWORK.layer_names()), f"MANUAL_STRATA 地层名均在框架内 {sorted(names)}")


def main() -> int:
    print("=" * 70)
    print("第 2 阶段（Step1–5）地层框架等价性验证")
    print(f"框架：{FRAMEWORK.framework_id}  layers={FRAMEWORK.layer_names()}")
    print("=" * 70)
    verify_step1_surface_tools()
    verify_step1_attach()
    verify_step2()
    verify_step4()
    verify_step5a()
    verify_step3()
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
