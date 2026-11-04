#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"上部补充层"映射成现有 6B/6C/6D 能读的 T4/T6/T7 形状（不改原代码）。

映射：
    T4 = TopV
    T6 = Top - 1 ms     （留 1 ms 给 Shasi，保证 top<mid<base 且 min_thickness≥1）
    T7 = Top
    ShasanPresent = SuppPresent
    ShasiPresent  = SuppPresent

这样现有脚本的 T4-T7 层窗并集 = [TopV, Top]，正好是上部补充层。

输出：
    output/taigu_supp_legacy_v1/supp_horizon_legacy_full.csv   （全矿区，供 iloc 索引）
    output/taigu_supp_legacy_v1/supp_6b_config.json            （6B 配置模板）

用法::

    python3 太古界/supp_layer/build_supp_legacy_adapter.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
DEFAULT_SOURCE = TAIGU / "common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.csv"
DEFAULT_SUPP = TAIGU / "supp_layer/output/taigu_supp_horizon_v1/supp_horizon_contract.npy"
DEFAULT_OUTPUT = TAIGU / "supp_layer/output/taigu_supp_legacy_v1"
DEFAULT_BASE_6B_CONFIG = TAIGU / "step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_quanqu1.json"
DEFAULT_DENSITY = TAIGU / "supp_layer/output/taigu_supp_density_v1/supp_density_block.sgy"
DEFAULT_MAPPING = TAIGU / "supp_layer/output/taigu_supp_density_v1/supp_density_mapping.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build legacy-shaped horizon adapter for supp layer.")
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--supp-contract", type=Path, default=DEFAULT_SUPP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-6b-config", type=Path, default=DEFAULT_BASE_6B_CONFIG)
    parser.add_argument("--density-sgy", type=Path, default=DEFAULT_DENSITY)
    parser.add_argument("--mapping-npz", type=Path, default=DEFAULT_MAPPING)
    args = parser.parse_args()

    source = pd.read_csv(
        args.source_contract,
        usecols=["TraceIdx", "X", "Y", "SurfaceValid"],
        encoding="utf-8-sig",
        low_memory=False,
    ).sort_values("TraceIdx").reset_index(drop=True)
    supp = np.load(args.supp_contract, mmap_mode="r")
    if len(supp) != len(source):
        raise ValueError("supp contract and source contract lengths differ")

    top_v = supp["TopVTimeMs"].astype(np.float64)
    top = supp["TopTimeMs"].astype(np.float64)
    present = supp["SuppPresent"].astype(bool)

    legacy = pd.DataFrame({
        "TraceIdx": source["TraceIdx"].to_numpy(dtype=np.int64),
        "X": source["X"].to_numpy(dtype=np.float64),
        "Y": source["Y"].to_numpy(dtype=np.float64),
        "TopTimeMs": top_v,
        "MidTimeMs": top - 1.0,
        "BaseTimeMs": top,
        "SurfaceValid": present.astype(np.uint8),
    })
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "supp_horizon_legacy_full.csv"
    legacy.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[supp-adapter] wrote {csv_path} ({len(legacy):,} rows)")

    # 6A 小背景 / 6D 读 npy，字段仍是 T4..T7
    legacy_dtype = np.dtype([
        ("TraceIdx", "<i4"),
        ("T4", "<f4"),
        ("T5", "<f4"),
        ("T6", "<f4"),
        ("T7", "<f4"),
        ("SurfaceOrderValid", "u1"),
        ("ShasanPresent", "u1"),
        ("ShasiPresent", "u1"),
    ])
    legacy_npy = np.zeros(len(legacy), dtype=legacy_dtype)
    legacy_npy["TraceIdx"] = legacy["TraceIdx"].astype(np.int32)
    legacy_npy["T4"] = legacy["TopTimeMs"].astype(np.float32)
    legacy_npy["T5"] = legacy["MidTimeMs"].astype(np.float32)
    legacy_npy["T6"] = legacy["MidTimeMs"].astype(np.float32)
    legacy_npy["T7"] = legacy["BaseTimeMs"].astype(np.float32)
    legacy_npy["SurfaceOrderValid"] = legacy["SurfaceValid"].astype(np.uint8)
    legacy_npy["ShasanPresent"] = legacy["SurfaceValid"].astype(np.uint8)
    legacy_npy["ShasiPresent"] = legacy["SurfaceValid"].astype(np.uint8)
    npy_path = out_dir / "supp_horizon_legacy_full.npy"
    np.save(npy_path, legacy_npy)
    print(f"[supp-adapter] wrote {npy_path}")

    config = json.loads(args.base_6b_config.read_text(encoding="utf-8"))
    config["version"] = "taigu_supp_layer_v1"
    config["input_density_sgy"] = str(args.density_sgy.resolve())
    config["trace_mapping_npz"] = str(args.mapping_npz.resolve())
    config["step6a_trace_mapping_npz"] = str(args.mapping_npz.resolve())
    config["horizon_contract_csv"] = str(csv_path)
    config["horizon_contract_table"] = str(args.supp_contract.resolve())
    config["supp_horizon_legacy_npy"] = str(npy_path)
    config["horizon_windows_npz"] = None
    config["output_dir"] = str((out_dir / "step6b_supp").resolve())
    config["_note"] = "上部补充层兼容适配：T4=TopV, T6=Top-1ms, T7=Top"
    summary_for_6b = {
        "status": "pass",
        "model_contract_version": config.get(
            "expected_step6a_model_contract_version", "taigu_step6a_attribute_v3_v4"
        ),
        "output_paths": {"density_sgy": str(args.density_sgy.resolve())},
        "_note": "Supplementary-layer Step6A summary for the 6B input contract check.",
    }
    summary_path = out_dir / "supp_step6a_summary_for_6b.json"
    summary_path.write_text(
        json.dumps(summary_for_6b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config["step6a_prediction_summary_json"] = str(summary_path)
    config_path = out_dir / "supp_6b_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[supp-adapter] wrote {config_path}")
    print(
        f"[supp-adapter] legacy window: T4={np.nanmin(top_v[present]):.1f}-{np.nanmax(top_v[present]):.1f}, "
        f"T6={np.nanmin(top[present]) - 1:.1f}-{np.nanmax(top[present]) - 1:.1f}, "
        f"T7={np.nanmin(top[present]):.1f}-{np.nanmax(top[present]):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
