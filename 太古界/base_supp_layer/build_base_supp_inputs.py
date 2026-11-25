#!/usr/bin/env python3
"""Build the 5x5 km base-below 300 ms pilot contract, grid and configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
LAYER_ROOT = TAIGU / "base_supp_layer"
OUTPUT_ROOT = LAYER_ROOT / "output/taigu_base_supp_5km_v1"

DEFAULT_SOURCE_CONTRACT = (
    TAIGU
    / "common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.csv"
)
DEFAULT_SUPP_CONFIG = TAIGU / "supp_layer/configs/taigu_supp_v1.json"
DEFAULT_ATTRIBUTE_CONFIG = (
    TAIGU
    / "step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_quanqu1.json"
)
DEFAULT_6B_CONFIG = (
    TAIGU
    / "step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_quanqu1.json"
)
DEFAULT_6C_CONFIG = (
    TAIGU
    / "step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_quanqu1.json"
)
DEFAULT_6D_CONFIG = (
    TAIGU
    / "step6d_multiscale_bundle/configs/taigu_step6d_multiscale_quanqu1.json"
)
DEFAULT_7A_CONFIG = TAIGU / "step7a_small_scale_dfn/configs/taigu_step7a_quanqu1.json"
DEFAULT_7B_CONFIG = (
    TAIGU
    / "step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_quanqu1.json"
)
DEFAULT_7C_CONFIG = (
    TAIGU
    / "step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_quanqu1_seg1.json"
)
DEFAULT_7D_CONFIG = (
    TAIGU
    / "step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_quanqu1_seg1.json"
)

LAYER_NAME = "基底以下补充层"
LAYER_CODE = 4
MODEL_LAYER = "太古界风化壳"
EXTENSION_MS = 300.0
ATTRIBUTE_BOTTOM_MS = 4800.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--supp-config", type=Path, default=DEFAULT_SUPP_CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--density-sgy",
        type=Path,
        default=None,
        help="Step6A density SGY. When present, also writes the Step7A window code.",
    )
    return parser.parse_args()


def build_contracts(
    source_path: Path,
    supp_config: dict[str, Any],
    output_root: Path,
    *,
    extension_ms: float = EXTENSION_MS,
) -> dict[str, Any]:
    metadata = dict(supp_config.get("metadata", {}))
    block_cfg = dict(metadata.get("pilot_block") or metadata.get("block") or {})
    required_block = {"x_min", "x_max", "y_min", "y_max"}
    if not required_block.issubset(block_cfg):
        raise ValueError(f"pilot block config is incomplete: {sorted(required_block - set(block_cfg))}")

    usecols = ["TraceIdx", "X", "Y", "IX", "IY", "BaseTimeMs", "SurfaceValid"]
    source = pd.read_csv(source_path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    for column in usecols:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.sort_values("TraceIdx").reset_index(drop=True)
    if len(source) != 2_156_550:
        raise ValueError(f"unexpected Taigu attribute grid size: {len(source)}")
    if not np.array_equal(
        source["TraceIdx"].to_numpy(dtype=np.int64),
        np.arange(len(source), dtype=np.int64),
    ):
        raise ValueError("source horizon contract row index must equal TraceIdx")

    x_min = float(block_cfg["x_min"])
    x_max = float(block_cfg["x_max"])
    y_min = float(block_cfg["y_min"])
    y_max = float(block_cfg["y_max"])
    center_x = float(block_cfg.get("center_x", 0.5 * (x_min + x_max)))
    center_y = float(block_cfg.get("center_y", 0.5 * (y_min + y_max)))
    half_cells = int(block_cfg.get("half_cells", 200))
    center_distance = np.square(source["X"].to_numpy() - center_x) + np.square(
        source["Y"].to_numpy() - center_y
    )
    center_row = int(np.argmin(center_distance))
    center_ix = int(source.at[center_row, "IX"])
    center_iy = int(source.at[center_row, "IY"])
    ix_min = center_ix - half_cells
    ix_max = center_ix + half_cells
    iy_min = center_iy - half_cells
    iy_max = center_iy + half_cells
    in_block = source["IX"].between(ix_min, ix_max) & source["IY"].between(iy_min, iy_max)
    block = source.loc[in_block].copy().sort_values("TraceIdx").reset_index(drop=True)
    if block.empty:
        raise ValueError("pilot block selected no traces")
    expected = (int(block["IX"].nunique()) * int(block["IY"].nunique()))
    if len(block) != expected:
        raise ValueError(f"pilot block is not rectangular: rows={len(block)} expected={expected}")

    ix0 = int(block["IX"].min())
    iy0 = int(block["IY"].min())
    block["IXLocal"] = block["IX"].astype(int) - ix0
    block["IYLocal"] = block["IY"].astype(int) - iy0
    base = block["BaseTimeMs"].astype(float)
    surface_valid = block["SurfaceValid"].astype(bool) & np.isfinite(base)
    base_v = np.minimum(base + float(extension_ms), ATTRIBUTE_BOTTOM_MS)
    extension = base_v - base
    present = surface_valid & np.isfinite(extension) & (extension > 0.0)

    block["BaseTimeMs"] = base
    block["BaseVTimeMs"] = base_v
    block["ExtensionMs"] = extension
    block["BaseSuppPresent"] = present.astype(np.uint8)
    # Compatibility names consumed by the existing supplementary-density predictor.
    block["TopVTimeMs"] = base
    block["TopTimeMs"] = base_v
    block["SuppPresent"] = present.astype(np.uint8)
    block_path = output_root / "base_supp_horizon_contract_block.csv"
    block.to_csv(block_path, index=False, encoding="utf-8-sig")

    full_valid = np.zeros(len(source), dtype=bool)
    full_base_v = np.full(len(source), np.nan, dtype=np.float32)
    full_base = np.full(len(source), np.nan, dtype=np.float32)
    positions = block["TraceIdx"].to_numpy(dtype=np.int64)
    full_valid[positions] = present.to_numpy(dtype=bool)
    full_base[positions] = base.to_numpy(dtype=np.float32)
    full_base_v[positions] = base_v.to_numpy(dtype=np.float32)

    dtype = np.dtype(
        [
            ("TraceIdx", "<i4"),
            ("T4", "<f4"),
            ("T5", "<f4"),
            ("T6", "<f4"),
            ("T7", "<f4"),
            ("SurfaceOrderValid", "u1"),
            ("ShasanPresent", "u1"),
            ("ShasiPresent", "u1"),
        ]
    )
    contract_npy = np.zeros(len(source), dtype=dtype)
    contract_npy["TraceIdx"] = source["TraceIdx"].to_numpy(dtype=np.int32)
    contract_npy["T4"] = full_base
    contract_npy["T5"] = full_base + 1.0
    contract_npy["T6"] = full_base + 1.0
    contract_npy["T7"] = full_base_v
    contract_npy["SurfaceOrderValid"] = full_valid.astype(np.uint8)
    contract_npy["ShasanPresent"] = full_valid.astype(np.uint8)
    contract_npy["ShasiPresent"] = full_valid.astype(np.uint8)
    npy_path = output_root / "base_supp_horizon_contract.npy"
    np.save(npy_path, contract_npy)

    full_csv = pd.DataFrame(
        {
            "TraceIdx": source["TraceIdx"].astype(np.int64),
            "X": source["X"],
            "Y": source["Y"],
            "IX": source["IX"].astype(np.int32),
            "IY": source["IY"].astype(np.int32),
            "TopTimeMs": full_base,
            "MidTimeMs": full_base + 1.0,
            "BaseTimeMs": full_base_v,
            "SurfaceValid": full_valid.astype(np.uint8),
        }
    )
    csv_path = output_root / "base_supp_horizon_contract.csv"
    full_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    step7_csv = full_csv.copy()
    step7_mid = np.where(
        full_valid,
        full_base + 0.5 * (full_base_v - full_base),
        np.nan,
    )
    step7_csv["MidTimeMs"] = step7_mid
    step7_csv_path = output_root / "base_supp_step7_horizon_contract.csv"
    step7_csv.to_csv(step7_csv_path, index=False, encoding="utf-8-sig")

    valid_values = extension[present]
    summary = {
        "status": "pass",
        "framework_id": "taigu_base_supp_v1",
        "layer_group": LAYER_NAME,
        "layer_code": LAYER_CODE,
        "model_proxy_layer": MODEL_LAYER,
        "pilot_block": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "width_m": x_max - x_min,
            "height_m": y_max - y_min,
            "area_km2": (x_max - x_min) * (y_max - y_min) / 1.0e6,
            "trace_count": int(len(block)),
            "ix_count": int(block["IX"].nunique()),
            "iy_count": int(block["IY"].nunique()),
            "ix_range": [int(block["IX"].min()), int(block["IX"].max())],
            "iy_range": [int(block["IY"].min()), int(block["IY"].max())],
            "center_ix": center_ix,
            "center_iy": center_iy,
            "half_cells": half_cells,
        },
        "window": {
        "definition": f"Base -> min(Base+{float(extension_ms):g} ms, 4800 ms)",
        "requested_extension_ms": float(extension_ms),
            "attribute_bottom_ms": ATTRIBUTE_BOTTOM_MS,
            "valid_trace_count": int(present.sum()),
            "valid_fraction": float(present.mean()),
            "extension_ms": {
                "min": float(valid_values.min()),
                "median": float(np.median(valid_values)),
                "max": float(valid_values.max()),
            },
            "base_ms": {
                "min": float(base[present].min()),
                "median": float(base[present].median()),
                "max": float(base[present].max()),
            },
            "base_v_ms": {
                "min": float(base_v[present].min()),
                "median": float(base_v[present].median()),
                "max": float(base_v[present].max()),
            },
            "full_requested_extension_trace_count": int(
                (extension[present] >= float(extension_ms) - 1.0e-6).sum()
            ),
        },
        "outputs": {
            "horizon_contract_npy": str(npy_path),
            "horizon_contract_csv": str(csv_path),
            "horizon_contract_block_csv": str(block_path),
            "step7_horizon_contract_csv": str(step7_csv_path),
        },
        "checks": {
            "source_grid_complete": len(source) == 2_156_550,
            "pilot_block_rectangular": len(block) == expected,
            "all_valid_extension_positive": bool((extension[present] > 0.0).all()),
            "all_valid_bottom_at_or_above_attribute_bottom": bool(
                (base_v[present] <= ATTRIBUTE_BOTTOM_MS + 1.0e-6).all()
            ),
        },
    }
    write_json(output_root / "base_supp_horizon_summary.json", summary)
    return {
        "block": block,
        "block_path": block_path,
        "npy_path": npy_path,
        "csv_path": csv_path,
        "step7_csv_path": step7_csv_path,
        "summary": summary,
    }


def build_configs(
    contracts: dict[str, Any],
    output_root: Path,
    source_contract: Path,
    density_sgy: Path | None,
    *,
    token: str,
) -> dict[str, Path]:
    block = contracts["block"]
    block_cfg = {
        "name": "taigu_supp_5km",
        "x_min": float(block["X"].min()),
        "x_max": float(block["X"].max()),
        "y_min": float(block["Y"].min()),
        "y_max": float(block["Y"].max()),
    }
    configs_dir = output_root / "configs"
    density_dir = output_root / "step6a_density"
    density_sgy = (density_sgy or (density_dir / "supp_density_block.sgy")).resolve()
    mapping_path = (density_dir / "supp_density_mapping.npz").resolve()
    legacy_npy = Path(contracts["npy_path"]).resolve()
    legacy_csv = Path(contracts["csv_path"]).resolve()
    step7_csv = Path(contracts["step7_csv_path"]).resolve()

    attr_cfg = read_json(DEFAULT_ATTRIBUTE_CONFIG)
    volume_paths = dict(attr_cfg["attribute_volume_paths"])
    time_origins = dict(attr_cfg["attribute_time_origins_ms"])

    sm_cfg = {
        "version": f"{token}_6a_small",
        "model_contract_version": "taigu_step6a_attribute_v3_v4",
        "density_sgy": str(density_sgy),
        "trace_mapping_npz": str(mapping_path),
        "horizon_contract_table": str(legacy_npy),
        "curvature_volume_path": volume_paths["CurvatureMax"],
        "curvature_time_origin_ms": float(time_origins["CurvatureMax"]),
        "output_dir": str((output_root / "step6a_small").resolve()),
        "small_evidence": {
            "density_weight": 0.4,
            "curvature_weight": 0.6,
            "curvature_transform": "absolute",
            "density_clip_low_q": 0.02,
            "density_clip_high_q": 0.98,
            "curvature_clip_low_q": 0.50,
            "curvature_clip_high_q": 0.995,
            "candidate_quantile": 0.88,
            "core_quantile": 0.95,
        },
    }
    sixa_summary = output_root / "step6a_density/supp_density_summary.json"
    summary_for_6b = output_root / "configs/step6a_summary_for_6b.json"
    write_json(
        summary_for_6b,
        {
            "status": "pass",
            "model_contract_version": "taigu_step6a_attribute_v3_v4",
            "output_paths": {"density_sgy": str(density_sgy)},
            "source_summary": str(sixa_summary),
            "layer_group": LAYER_NAME,
            "layer_code": LAYER_CODE,
        },
    )

    cfg6b = read_json(DEFAULT_6B_CONFIG)
    cfg6b.update(
        {
            "version": f"{token}_6b_medium",
            "input_density_sgy": str(density_sgy),
            "trace_mapping_npz": str(mapping_path),
            "step6a_trace_mapping_npz": str(mapping_path),
            "horizon_contract_table": str(legacy_npy),
            "horizon_contract_csv": str(legacy_csv),
            "output_dir": str((output_root / "step6b_medium").resolve()),
            "step6a_prediction_summary_json": str(summary_for_6b),
            "horizon_windows_npz": None,
            "write_intermediate_vtk": False,
            "_note": "基底以下补充层；T4=Base, T6=Base+1ms, T7=BaseV",
        }
    )

    cfg6c = read_json(DEFAULT_6C_CONFIG)
    cfg6c.update(
        {
            "version": f"{token}_6c_large",
            "input_density_sgy": str(density_sgy),
            "trace_mapping_npz": str(mapping_path),
            "horizon_contract_table": str(legacy_csv),
            "output_dir": str((output_root / "step6c_large").resolve()),
            "large_evidence": {
                "sample_start_ms": float(np.floor(float(block.loc[block.BaseSuppPresent.eq(1), "BaseTimeMs"].min()) / 10.0) * 10.0),
                "sample_stop_ms": float(np.ceil(float(block.loc[block.BaseSuppPresent.eq(1), "BaseVTimeMs"].max()) / 10.0) * 10.0),
                "sample_interval_ms": 10.0,
            },
            "_note": "基底以下补充层 6C；属性不越过 4800 ms",
        }
    )

    cfg6d = read_json(DEFAULT_6D_CONFIG)
    cfg6d.update(
        {
            "version": f"{token}_6d",
            "step6a_dir": str((output_root / "step6a_small").resolve()),
            "step6b_dir": str((output_root / "step6b_medium").resolve()),
            "step6c_dir": str((output_root / "step6c_large").resolve()),
            "input_density_sgy": str(density_sgy),
            "trace_mapping_npz": str(mapping_path),
            "horizon_contract_table": str(legacy_npy),
            "output_dir": str((output_root / "step6d_bundle").resolve()),
            "compact_multiscale_flow": False,
            "written_volumes": ["background_small_density", "large_damage_small_density"],
            "write_prior_bundle_npz": False,
            "_note": "基底以下补充层 Step6D；仅保留下游所需 SGY",
        }
    )
    cfg6d.pop("small_score_sgy", None)
    cfg6d.pop("small_qc", None)

    step7_root = output_root / "step7"
    grid_path = output_root / "base_supp_demo_grid_block.csv"
    grid = pd.DataFrame(
        {
            "TraceIdx": block["TraceIdx"].astype(np.int64),
            "X": block["X"].astype(float),
            "Y": block["Y"].astype(float),
            "IX": block["IXLocal"].astype(np.int32),
            "IY": block["IYLocal"].astype(np.int32),
        }
    )
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")

    cfg7a = read_json(DEFAULT_7A_CONFIG)
    cfg7a.update(
        {
            "version": f"{token}_7a",
            "output_dir": str((step7_root / "step7a").resolve()),
            "density_sgy": str((output_root / "step6d_bundle/background_small_density.sgy").resolve()),
            "damage_density_sgy": str((output_root / "step6d_bundle/large_damage_small_density.sgy").resolve()),
            "window_code_npz": str((step7_root / "base_supp_window_code.npz").resolve()),
            "step6a_prediction_summary_json": str((output_root / "step6d_bundle/multiscale_bundle_summary.json").resolve()),
            "trace_mapping_npz": str(mapping_path),
            "demo_grid_csv": str(grid_path),
            "horizon_contract_csv": str(step7_csv),
            "_note": "基底以下补充层 Step7A；临时槽位按 Base、Base+150ms、BaseV 映射",
        }
    )

    cfg7b = read_json(DEFAULT_7B_CONFIG)
    cfg7b.update(
        {
            "version": f"{token}_7b",
            "output_dir": str((step7_root / "step7b").resolve()),
            "medium_prior_sgy": str((output_root / "step6b_medium/medium_corridor_prior.sgy").resolve()),
            "medium_components_npz": str((output_root / "step6b_medium/medium_corridor_components.npz").resolve()),
            "medium_component_summary_csv": str((output_root / "step6b_medium/medium_corridor_component_summary.csv").resolve()),
            "step6b_qc_json": str((output_root / "step6b_medium/medium_corridor_qc.json").resolve()),
            "trace_mapping_npz": str(mapping_path),
            "target_block": block_cfg,
        }
    )

    cfg7c = read_json(DEFAULT_7C_CONFIG)
    cfg7c.update(
        {
            "version": f"{token}_7c",
            "output_dir": str((step7_root / "step7c").resolve()),
            "large_prior_sgy": str((output_root / "step6c_large/large_fault_prior.sgy").resolve()),
            "large_prior_components_npz": str((output_root / "step6c_large/large_fault_prior_components.npz").resolve()),
            "large_component_summary_csv": str((output_root / "step6c_large/large_fault_component_summary.csv").resolve()),
            "original_fault_surface_vtk": str((output_root / "step6c_large/original_fault_units_demo_raw_time.vtk").resolve()),
            "original_fault_manifest_csv": str((output_root / "step6c_large/original_fault_unit_manifest.csv").resolve()),
            "fault_patch_overlap_csv": str((output_root / "step6c_large/original_fault_rasterization_audit.csv").resolve()),
            "trace_mapping_npz": str(mapping_path),
            "horizon_contract_table": str(step7_csv),
            "target_block": block_cfg,
            "time_min_ms": float(np.floor(float(block.loc[block.BaseSuppPresent.eq(1), "BaseTimeMs"].min()) / 10.0) * 10.0),
            "time_max_ms": float(np.ceil(float(block.loc[block.BaseSuppPresent.eq(1), "BaseVTimeMs"].max()) / 10.0) * 10.0),
            "_note": "基底以下补充层 Step7C；沿用 seg1 近直立、连续分块和剖面可读性规则",
        }
    )

    cfg7d = read_json(DEFAULT_7D_CONFIG)
    cfg7d.update(
        {
            "version": f"{token}_7d",
            "output_dir": str((step7_root / "step7d").resolve()),
            "small_dfn_csv": str((step7_root / "step7a/fracture_patches.csv").resolve()),
            "medium_dfn_csv": str((step7_root / "step7b/medium_dfn_patches.csv").resolve()),
            "large_dfn_csv": str((step7_root / "step7c/large_fault_dfn_patches.csv").resolve()),
            "large_dfn_vtk": str((step7_root / "step7c/large_fault_dfn_raw_time.vtk").resolve()),
            "original_fault_manifest_csv": str((output_root / "step6c_large/original_fault_unit_manifest.csv").resolve()),
            "horizon_contract_table": str(step7_csv),
            "target_block": block_cfg,
            "write_vtk": True,
            "_note": "基底以下补充层 Step7D",
        }
    )

    outputs = {
        "6a": configs_dir / "step6a_small.json",
        "6b": configs_dir / "step6b_medium.json",
        "6c": configs_dir / "step6c_large.json",
        "6d": configs_dir / "step6d_bundle.json",
        "7a": configs_dir / "step7a.json",
        "7b": configs_dir / "step7b.json",
        "7c": configs_dir / "step7c.json",
        "7d": configs_dir / "step7d.json",
    }
    for key, payload in (
        ("6a", sm_cfg),
        ("6b", cfg6b),
        ("6c", cfg6c),
        ("6d", cfg6d),
        ("7a", cfg7a),
        ("7b", cfg7b),
        ("7c", cfg7c),
        ("7d", cfg7d),
    ):
        write_json(outputs[key], payload)
    write_json(
        output_root / "configs/paths.json",
        {
            "source_contract": str(source_contract.resolve()),
            "block_csv": str(contracts["block_path"].resolve()),
            "legacy_npy": str(legacy_npy),
            "legacy_csv": str(legacy_csv),
            "step7_legacy_csv": str(step7_csv),
            "density_sgy": str(density_sgy),
            "mapping_npz": str(mapping_path),
            "configs": {key: str(path.resolve()) for key, path in outputs.items()},
        },
    )
    return outputs


def build_step7_window(
    contracts: dict[str, Any],
    output_root: Path,
    density_sgy: Path,
) -> dict[str, Any]:
    block = contracts["block"]
    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
    top = block["TopVTimeMs"].to_numpy(dtype=np.float64)
    bottom = block["TopTimeMs"].to_numpy(dtype=np.float64)
    present = block["SuppPresent"].astype(bool).to_numpy()
    window_code = np.zeros((len(block), len(samples)), dtype=np.uint8)
    chunk = 8192
    for start in range(0, len(block), chunk):
        stop = min(start + chunk, len(block))
        axis = samples[None, :]
        inside = (
            present[start:stop, None]
            & (axis >= top[start:stop, None])
            & (axis < bottom[start:stop, None])
        )
        window_code[start:stop] = inside.astype(np.uint8)
    path = output_root / "step7/base_supp_window_code.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_code=window_code,
        sample_axis=samples.astype(np.float32),
        trace_idx=block["TraceIdx"].to_numpy(dtype=np.int64),
    )
    summary = {
        "status": "pass",
        "path": str(path),
        "shape": list(window_code.shape),
        "inside_voxel_count": int(window_code.sum()),
        "sample_axis_ms": [float(samples[0]), float(samples[-1]), int(len(samples))],
    }
    write_json(output_root / "step7/window_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = args.source_contract.resolve()
    supp_config = read_json(args.supp_config.resolve())
    layer_config = dict((supp_config.get("layers") or [{}])[0])
    extension_ms = float(layer_config.get("extension_ms", EXTENSION_MS))
    contracts = build_contracts(
        source_path,
        supp_config,
        output_root,
        extension_ms=extension_ms,
    )
    build_configs(
        contracts,
        output_root,
        source_path,
        args.density_sgy,
        token=output_root.name,
    )
    if args.density_sgy is not None:
        if not args.density_sgy.exists():
            raise FileNotFoundError(args.density_sgy)
        build_step7_window(contracts, output_root, args.density_sgy.resolve())
    print(json.dumps(contracts["summary"], ensure_ascii=False, indent=2))
    print(f"[base-supp] outputs: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
