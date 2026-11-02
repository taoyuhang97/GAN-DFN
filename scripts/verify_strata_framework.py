#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证地层框架（StrataFramework）重构与旧硬编码逻辑等价。

检查项：
  1. 两个框架配置能加载，LayerCode 全部是 int 且唯一；
  2. 砂砾岩：层名 / 历史别名(Shasan/Shasi) / int code 三种引用解析到同一层；
  3. 砂砾岩：框架层窗掩膜与旧 hardcode 公式逐元素相等；
  4. 砂砾岩：interval 标签仍是 T4->T6 / T6->T7，layer_group 仍是沙三段/沙四段；
  5. 砂砾岩：surface_grids_from_contract 的键集合与旧版一致；
  6. 太古界：Top/Mid/Base 通过 legacy_column 映射到 T4/T6/T7，风化壳延伸 50 ms。

用法::

    python3 scripts/verify_strata_framework.py
    python3 scripts/verify_strata_framework.py --traces 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "优化阶段二" / "正式主线"
for path in (FORMAL_ROOT, FORMAL_ROOT / "common"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    DEFAULT_FRAMEWORK,
    HorizonSpatialLookup,
    load_contract_for_trace_indices,
    surface_grids_from_contract,
    trace_window_mask,
)
from strata_framework import available_frameworks, load_framework  # noqa: E402


DEFAULT_CONTRACT = (
    FORMAL_ROOT / "common/horizon_trace_table/output/formal_horizon_trace_table_v2_mine/horizon_trace_table.npy"
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  [PASS] {message}")


def verify_int_codes() -> None:
    print("\n[1] 框架配置与 LayerCode 类型")
    check(set(available_frameworks()) >= {"shaligyan_v1", "taigu_v1"}, "两个框架均已注册")
    for framework_id in ("shaligyan_v1", "taigu_v1"):
        framework = load_framework(framework_id)
        codes = framework.layer_codes()
        check(all(isinstance(code, int) and not isinstance(code, bool) for code in codes),
              f"{framework_id}: LayerCode 全为 int {codes}")
        check(len(codes) == len(set(codes)), f"{framework_id}: LayerCode 唯一")


def verify_layer_resolution() -> None:
    print("\n[2] 层引用解析（名称 / 历史别名 / int code）")
    framework = load_framework("shaligyan_v1")
    by_name = framework.layer("沙三段")
    by_alias = framework.layer("Shasan")
    by_code = framework.layer(3)
    check(by_name == by_alias == by_code, "沙三段: 名称 / Shasan / code=3 解析一致")
    check(framework.layer_code_map() == {"沙三段": 3, "沙四段": 4},
          "layer_code_map 为 {沙三段: 3, 沙四段: 4}（int）")
    check(framework.layer("沙三段").interval_label() == "T4->T6", "沙三段区间标签保持 T4->T6")
    check(framework.layer("沙四段").interval_label() == "T6->T7", "沙四段区间标签保持 T6->T7")
    check(framework.layer("沙三段").inclusive_base is False, "沙三段层底不含（<）")
    check(framework.layer("沙四段").inclusive_base is True, "沙四段层底含（<=）")


def verify_masks(contract_path: Path, n_traces: int) -> None:
    print("\n[3] 层窗掩膜：框架 vs 旧硬编码")
    raw = np.load(contract_path, mmap_mode="r")
    count = min(int(n_traces), len(raw))
    idx = np.arange(count, dtype=np.int64)
    contract = load_contract_for_trace_indices(contract_path, idx)
    samples = np.arange(
        float(np.nanmin(raw["T4"])),
        float(np.nanmax(raw["T7"])) + 2.0,
        2.0,
    )

    axis = samples[None, :]
    legacy_sha3 = (
        np.asarray(raw["ShasanPresent"][:count], dtype=bool)[:, None]
        & (axis >= np.asarray(raw["T4"][:count])[:, None])
        & (axis < np.asarray(raw["T6"][:count])[:, None])
    )
    legacy_sha4 = (
        np.asarray(raw["ShasiPresent"][:count], dtype=bool)[:, None]
        & (axis >= np.asarray(raw["T6"][:count])[:, None])
        & (axis <= np.asarray(raw["T7"][:count])[:, None])
    )

    checks = {
        "沙三段(名称)": trace_window_mask(contract, samples, 0, count, "沙三段"),
        "Shasan(别名)": trace_window_mask(contract, samples, 0, count, "Shasan"),
        "code=3": trace_window_mask(contract, samples, 0, count, 3),
    }
    for label, mask in checks.items():
        check(np.array_equal(mask, legacy_sha3), f"{label} 掩膜与旧公式逐元素相等")

    checks4 = {
        "沙四段(名称)": trace_window_mask(contract, samples, 0, count, "沙四段"),
        "Shasi(别名)": trace_window_mask(contract, samples, 0, count, "Shasi"),
        "code=4": trace_window_mask(contract, samples, 0, count, 4),
    }
    for label, mask in checks4.items():
        check(np.array_equal(mask, legacy_sha4), f"{label} 掩膜与旧公式逐元素相等")

    both = trace_window_mask(contract, samples, 0, count, "T4-T7")
    check(np.array_equal(both, legacy_sha3 | legacy_sha4), "T4-T7 合并掩膜与旧公式相等")

    # 抽样检查空间查询的区间 / 层名映射
    header = REPO / "数据精简/trace_header_xy.csv"
    valid = (
        np.asarray(raw["SurfaceOrderValid"][:count], dtype=bool)
        & np.asarray(raw["ShasanPresent"][:count], dtype=bool)
        & np.asarray(raw["ShasiPresent"][:count], dtype=bool)
        & np.isfinite(raw["T4"][:count])
        & np.isfinite(raw["T6"][:count])
        & np.isfinite(raw["T7"][:count])
    )
    if valid.any():
        index = int(np.flatnonzero(valid)[0])
        import pandas as pd

        frame = pd.read_csv(header, usecols=["X", "Y"], encoding="utf-8-sig")
        x = float(frame["X"].iloc[index])
        y = float(frame["Y"].iloc[index])
        lookup = HorizonSpatialLookup(contract_path, header)
        row = lookup.query(x, y)
        t4, t6, t7 = row["T4"], row["T6"], row["T7"]
        check(lookup.interval(x, y, (t4 + t6) / 2) == "T4->T6",
              "沙三段中部时间 -> interval T4->T6")
        check(lookup.layer_group(x, y, (t6 + t7) / 2) == "沙四段",
              "沙四段中部时间 -> layer_group 沙四段")


def verify_grid_keys(contract_path: Path) -> None:
    print("\n[4] surface_grids_from_contract 键集合")
    raw = np.load(contract_path, mmap_mode="r")
    count = min(1000, len(raw))
    contract = load_contract_for_trace_indices(contract_path, np.arange(count, dtype=np.int64))
    ix = np.zeros(count, dtype=np.int32)
    iy = np.arange(count, dtype=np.int32)
    grids = surface_grids_from_contract({"ix": ix, "iy": iy}, contract)
    expected = {
        "T4_TIME", "T5_TIME", "T6_TIME", "T7_TIME",
        "SurfaceOrderValid", "ShasanPresent", "ShasiPresent",
        "Check_All", "SourceTraceIdx",
    }
    check(set(grids) == expected, f"grid 键集合与旧版一致：{sorted(grids)}")
    check(np.allclose(grids["T4_TIME"][iy, ix], raw["T4"][:count], atol=1e-5),
          "grid 中 T4 数值与契约表一致")


def verify_taigu() -> None:
    print("\n[5] 太古界框架：Surface 命名与 legacy 列映射")
    framework = load_framework("taigu_v1")
    check(framework.surface_ids() == ("Top", "Mid", "Base"), "太古界层位面为 Top/Mid/Base")
    columns = {surface.id: surface.column for surface in framework.surfaces}
    check(columns == {"Top": "T4", "Mid": "T6", "Base": "T7"},
          "legacy_column 映射 Top->T4 / Mid->T6 / Base->T7")
    crust = framework.layer("太古界风化壳")
    check(crust.extension_ms == 50.0, "太古界风化壳带 50 ms 延伸窗")
    check(framework.layer_codes() == (1, 2), "太古界 LayerCode 为 int (1, 2)")
    check(framework.legacy_layer_aliases == {"Shasan": "上部复合层", "Shasi": "太古界风化壳"},
          "历史别名 Shasan/Shasi 指向太古界两个地层")


def verify_no_float_layer_code() -> None:
    print("\n[6] 代码中不再有浮点 LayerCode / 硬编码地层表")
    mainline = REPO / "优化阶段二" / "正式主线"
    offenders: list[str] = []
    for path in mainline.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("LAYER_CODE") and "3.0" in stripped:
                offenders.append(f"{path.relative_to(REPO)}:{number}: {stripped}")
    check(not offenders, "没有残留的 LAYER_CODE 浮点定义")
    for item in offenders:
        print(f"    {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="验证地层框架重构等价性")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--traces", type=int, default=2000)
    args = parser.parse_args()

    contract_path = Path(args.contract).resolve()
    if not contract_path.exists():
        raise FileNotFoundError(f"horizon contract not found: {contract_path}")
    print("=" * 70)
    print("地层框架等价性验证")
    print(f"契约：{contract_path}")
    print(f"默认框架：{DEFAULT_FRAMEWORK.framework_id}")
    print("=" * 70)

    verify_int_codes()
    verify_layer_resolution()
    verify_masks(contract_path, args.traces)
    verify_grid_keys(contract_path)
    verify_taigu()
    verify_no_float_layer_code()

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
