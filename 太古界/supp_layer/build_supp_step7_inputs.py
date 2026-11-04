#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备 Step7A-D（上部补充层，区块）所需的输入与配置。

输出：
  supp_demo_grid_block.csv   区块网格（TraceIdx/X/Y/IX/IY，局部索引）
  supp_window_code.npz       window_code (道×样点) + sample_axis
  supp_7a_config.json ... supp_7d_config.json

用法::

    python3 太古界/supp_layer/build_supp_step7_inputs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import segyio


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
SUPP = TAIGU / "supp_layer/output"

DEFAULT_BLOCK = SUPP / "taigu_supp_horizon_v1/supp_horizon_contract_block.csv"
DEFAULT_LEGACY_CSV = SUPP / "taigu_supp_legacy_v1/supp_horizon_legacy_full.csv"
DEFAULT_MAPPING = SUPP / "taigu_supp_density_v1/supp_density_mapping.npz"
DEFAULT_BACKGROUND = SUPP / "taigu_supp_legacy_v1/step6d_supp/background_small_density.sgy"
DEFAULT_DAMAGE = SUPP / "taigu_supp_legacy_v1/step6d_supp/large_damage_small_density.sgy"
DEFAULT_6D_SUMMARY = SUPP / "taigu_supp_legacy_v1/step6d_supp/multiscale_bundle_summary.json"
DEFAULT_OUTPUT = SUPP / "taigu_supp_step7_v1"

BASE_7A = TAIGU / "step7a_small_scale_dfn/configs/taigu_step7a_quanqu1.json"
BASE_7B = TAIGU / "step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_quanqu1.json"
BASE_7C = TAIGU / "step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_quanqu1.json"
BASE_7D = TAIGU / "step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_quanqu1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Step7 inputs/configs for the supplementary layer.")
    parser.add_argument("--block-csv", type=Path, default=DEFAULT_BLOCK)
    parser.add_argument("--legacy-csv", type=Path, default=DEFAULT_LEGACY_CSV)
    parser.add_argument("--mapping-npz", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--background-sgy", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--damage-sgy", type=Path, default=DEFAULT_DAMAGE)
    parser.add_argument("--step6d-summary", type=Path, default=DEFAULT_6D_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    block = pd.read_csv(args.block_csv, encoding="utf-8-sig")
    grid = pd.DataFrame({
        "TraceIdx": block["TraceIdx"].to_numpy(dtype=np.int64),
        "X": block["X"].to_numpy(dtype=np.float64),
        "Y": block["Y"].to_numpy(dtype=np.float64),
        "IX": block["IXLocal"].to_numpy(dtype=np.int32),
        "IY": block["IYLocal"].to_numpy(dtype=np.int32),
    })
    grid_path = out_dir / "supp_demo_grid_block.csv"
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(f"[supp-step7] wrote {grid_path} ({len(grid):,} traces)")

    with segyio.open(str(args.background_sgy), "r", ignore_geometry=True) as handle:
        sample_axis = np.asarray(handle.samples, dtype=np.float64)
    top_v = pd.to_numeric(block["TopVTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    top = pd.to_numeric(block["TopTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    present = block["SuppPresent"].astype(bool).to_numpy()
    window_code = np.zeros((len(block), len(sample_axis)), dtype=np.uint8)
    chunk = 8192
    for start in range(0, len(block), chunk):
        stop = min(start + chunk, len(block))
        times = sample_axis[None, :]
        code = (
            present[start:stop, None]
            & (times >= top_v[start:stop, None])
            & (times < top[start:stop, None])
        )
        window_code[start:stop] = code.astype(np.uint8)
    windows_path = out_dir / "supp_window_code.npz"
    np.savez_compressed(
        windows_path,
        window_code=window_code,
        sample_axis=sample_axis,
        trace_idx=block["TraceIdx"].to_numpy(dtype=np.int64),
    )
    print(f"[supp-step7] wrote {windows_path} shape={window_code.shape} inside={int(window_code.sum()):,}")

    real_root = TAIGU / "step2_well_log_segments/output/taigu_step2_regular_md1"
    # 7A 的产状库层名写死在"上部复合层/太古界风化壳"；用原有成像组提供产状库，
    # 新层几何仍由本区块的 window_code 决定（输出 LayerGroup 事后统一改为上部补充层）。
    supp_groups = TAIGU / "step3_imaging_groups/output/taigu_step3_imaging_md1/groups"
    step6b = SUPP / "taigu_supp_legacy_v1/step6b_supp"
    step6c = SUPP / "taigu_supp_legacy_v1/step6c_supp"

    configs = {}
    cfg = json.loads(BASE_7A.read_text(encoding="utf-8"))
    cfg.update({
        "version": "taigu_supp_layer_v1_7a",
        "output_dir": str(out_dir / "step7a_supp"),
        "density_sgy": str(args.background_sgy.resolve()),
        "damage_density_sgy": str(args.damage_sgy.resolve()),
        "window_code_npz": str(windows_path),
        "step6a_prediction_summary_json": str(args.step6d_summary.resolve()),
        "trace_mapping_npz": str(args.mapping_npz.resolve()),
        "demo_grid_csv": str(grid_path),
        "horizon_contract_csv": str(args.legacy_csv.resolve()),
        "step3_groups_root": str(supp_groups),
        "step2_segments_root": str(real_root),
        "_note": "上部补充层 Step7A（区块）；T4=TopV, T6=Top-1ms, T7=Top",
    })
    configs["supp_7a_config.json"] = cfg

    cfg = json.loads(BASE_7B.read_text(encoding="utf-8"))
    cfg.update({
        "version": "taigu_supp_layer_v1_7b",
        "output_dir": str(out_dir / "step7b_supp"),
        "medium_prior_sgy": str(step6b / "medium_corridor_prior.sgy"),
        "medium_components_npz": str(step6b / "medium_corridor_components.npz"),
        "medium_component_summary_csv": str(step6b / "medium_corridor_component_summary.csv"),
        "step6b_qc_json": str(step6b / "medium_corridor_qc.json"),
        "trace_mapping_npz": str(args.mapping_npz.resolve()),
        "_note": "上部补充层 Step7B（区块）",
    })
    configs["supp_7b_config.json"] = cfg

    cfg = json.loads(BASE_7C.read_text(encoding="utf-8"))
    cfg.update({
        "version": "taigu_supp_layer_v1_7c",
        "output_dir": str(out_dir / "step7c_supp"),
        "large_prior_sgy": str(step6c / "large_fault_prior.sgy"),
        "large_prior_components_npz": str(step6c / "large_fault_prior_components.npz"),
        "large_component_summary_csv": str(step6c / "large_fault_component_summary.csv"),
        "original_fault_surface_vtk": str(step6c / "original_fault_units_demo_raw_time.vtk"),
        "original_fault_manifest_csv": str(step6c / "original_fault_unit_manifest.csv"),
        "fault_patch_overlap_csv": str(step6c / "original_fault_rasterization_audit.csv"),
        "trace_mapping_npz": str(args.mapping_npz.resolve()),
        "horizon_contract_table": str(args.legacy_csv.resolve()),
        "time_min_ms": 2630.0,
        "time_max_ms": 3700.0,
        "_note": "上部补充层 Step7C（区块）",
    })
    configs["supp_7c_config.json"] = cfg

    cfg = json.loads(BASE_7D.read_text(encoding="utf-8"))
    cfg.update({
        "output_dir": str(out_dir / "step7d_supp"),
        "small_dfn_csv": str(out_dir / "step7a_supp/fracture_patches.csv"),
        "medium_dfn_csv": str(out_dir / "step7b_supp/medium_dfn_patches.csv"),
        "large_dfn_csv": str(out_dir / "step7c_supp/large_fault_dfn_patches.csv"),
        "large_dfn_vtk": str(out_dir / "step7c_supp/large_fault_dfn_raw_time.vtk"),
        "original_fault_manifest_csv": str(step6c / "original_fault_unit_manifest.csv"),
        "horizon_contract_table": str(args.legacy_csv.resolve()),
        "_note": "上部补充层 Step7D（区块）",
    })
    configs["supp_7d_config.json"] = cfg

    for name, payload in configs.items():
        path = out_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[supp-step7] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
