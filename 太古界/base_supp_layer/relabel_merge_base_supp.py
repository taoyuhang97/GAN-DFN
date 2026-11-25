#!/usr/bin/env python3
"""Relabel, merge and QC the 5x5 km base-below supplementary DFN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
OUTPUT_ROOT = TAIGU / "base_supp_layer/output/taigu_base_supp_5km_v1"

DEFAULT_BASE_SUPP_CSV = OUTPUT_ROOT / "step7/step7d/fused_multiscale_dfn_patches.csv"
DEFAULT_BASE_SUPP_VTK = OUTPUT_ROOT / "step7/step7d/fused_multiscale_dfn_raw_time.vtk"
DEFAULT_HORIZON_CONTRACT = OUTPUT_ROOT / "base_supp_horizon_contract.csv"
DEFAULT_MAIN_CSV = (
    TAIGU
    / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1_seg1/well_corrected_dfn_fracture_patches.csv"
)
DEFAULT_MAIN_VTK = (
    TAIGU
    / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1_seg1/well_corrected_dfn_raw_time.vtk"
)
DEFAULT_MAIN_HORIZON_CONTRACT = (
    TAIGU
    / "common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.csv"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "merged"

LAYER_NAME = "基底以下补充层"
LAYER_CODE = 4
ATTRIBUTE_BOTTOM_MS = 4800.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-supp-csv", type=Path, default=DEFAULT_BASE_SUPP_CSV)
    parser.add_argument("--base-supp-vtk", type=Path, default=DEFAULT_BASE_SUPP_VTK)
    parser.add_argument("--horizon-contract", type=Path, default=DEFAULT_HORIZON_CONTRACT)
    parser.add_argument("--main-csv", type=Path, default=DEFAULT_MAIN_CSV)
    parser.add_argument("--main-vtk", type=Path, default=DEFAULT_MAIN_VTK)
    parser.add_argument(
        "--main-horizon-contract",
        type=Path,
        default=DEFAULT_MAIN_HORIZON_CONTRACT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_polydata(path: Path) -> pv.PolyData:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface(algorithm="dataset_surface")
    return mesh


def group_mesh(mesh: pv.PolyData, code: int) -> pv.PolyData:
    values = np.asarray(mesh.cell_data.get("GeometryGroupCode"), dtype=np.int32)
    if values.size != mesh.n_cells:
        raise ValueError("VTK is missing cell GeometryGroupCode")
    return mesh.extract_cells(np.flatnonzero(values == code))


def merge_meshes(meshes: list[pv.PolyData]) -> pv.PolyData:
    if not meshes:
        return pv.PolyData()
    merged = pv.merge(meshes, merge_points=False, progress_bar=False)
    if not isinstance(merged, pv.PolyData):
        merged = merged.extract_surface(algorithm="dataset_surface")
    return merged


def add_scalar_array(mesh: pv.PolyData, name: str, value: float | int) -> None:
    mesh.cell_data[name] = np.full(mesh.n_cells, value)


def attach_base_window(frame: pd.DataFrame, contract_path: Path) -> pd.DataFrame:
    contract = pd.read_csv(
        contract_path,
        usecols=["TraceIdx", "X", "Y", "TopTimeMs", "BaseTimeMs", "SurfaceValid"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    valid = (
        contract["SurfaceValid"].astype(bool)
        & np.isfinite(contract["TopTimeMs"])
        & np.isfinite(contract["BaseTimeMs"])
        & contract["BaseTimeMs"].gt(contract["TopTimeMs"])
    )
    contract = contract.loc[valid].reset_index(drop=True)
    if contract.empty:
        raise ValueError("base-supp horizon contract contains no valid traces")
    contract_by_trace = contract.set_index("TraceIdx")
    source_trace_idx = pd.to_numeric(
        frame.get("SourceTraceIdx", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    direct = source_trace_idx.map(contract_by_trace["TopTimeMs"])
    direct_distance = np.where(np.isfinite(direct.to_numpy(dtype=float)), 0.0, np.nan)
    missing = ~np.isfinite(direct.to_numpy(dtype=float))
    if missing.any():
        tree = cKDTree(contract[["X", "Y"]].to_numpy(dtype=np.float64))
        distance, position = tree.query(
            frame.loc[missing, ["CenterX", "CenterY"]].to_numpy(dtype=np.float64),
            k=1,
            workers=-1,
        )
        nearest = contract.iloc[np.asarray(position, dtype=np.int64)]
        direct.loc[missing] = nearest["TopTimeMs"].to_numpy(dtype=float)
        direct_distance[missing] = np.asarray(distance, dtype=float)
        fallback_trace = nearest["TraceIdx"].to_numpy(dtype=np.int64)
        fallback_basev = nearest["BaseTimeMs"].to_numpy(dtype=float)
    else:
        fallback_trace = np.asarray([], dtype=np.int64)
        fallback_basev = np.asarray([], dtype=float)
    out = frame.copy()
    out["BaseSuppHorizonTraceIdx"] = source_trace_idx.where(
        source_trace_idx.isin(contract_by_trace.index)
    ).to_numpy(dtype=float)
    out.loc[missing, "BaseSuppHorizonTraceIdx"] = fallback_trace
    out["BaseSuppHorizonDistanceM"] = direct_distance
    out["BaseAbsTimeMs"] = direct.to_numpy(dtype=np.float64)
    base_v = source_trace_idx.map(contract_by_trace["BaseTimeMs"])
    base_v.loc[missing] = fallback_basev
    out["BaseVTimeMs"] = base_v.to_numpy(dtype=np.float64)
    return out


def relabel_base_supp(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "PatchID",
        "CenterTime",
        "BaseAbsTimeMs",
        "BaseVTimeMs",
        "FractureScale",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"base-supp CSV missing columns: {missing}")
    out = frame.copy()
    out["OriginalPatchID"] = out["PatchID"].astype(str)
    out["PatchID"] = out["OriginalPatchID"].map(lambda value: f"base_supp_{value}")
    out["LayerGroup"] = LAYER_NAME
    out["LayerCode"] = LAYER_CODE
    out["IsOriginalFault"] = 0
    out["IsBaseSupplement"] = 1
    out["BaseSuppTopTimeMs"] = pd.to_numeric(out["BaseAbsTimeMs"], errors="coerce")
    out["BaseSuppBottomTimeMs"] = pd.to_numeric(out["BaseVTimeMs"], errors="coerce")
    out["RelativeToBaseMs"] = out["CenterTime"] - out["BaseSuppTopTimeMs"]
    extension = (out["BaseSuppBottomTimeMs"] - out["BaseSuppTopTimeMs"]).clip(lower=1.0)
    taper = np.clip(1.0 - 0.3 * out["RelativeToBaseMs"] / extension, 0.7, 1.0)
    out["ExtensionConfidence"] = taper
    if "Confidence" in out.columns:
        out["Confidence"] = pd.to_numeric(out["Confidence"], errors="coerce") * taper
    else:
        out["Confidence"] = taper
    out["BaseSuppGenerationMode"] = "step6_step7_model_proxy_without_below_base_labels"
    return out


def attach_main_base(frame: pd.DataFrame, contract_path: Path) -> pd.DataFrame:
    contract = pd.read_csv(
        contract_path,
        usecols=["TraceIdx", "X", "Y", "BaseTimeMs", "SurfaceValid"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    valid = (
        contract["SurfaceValid"].astype(bool)
        & np.isfinite(contract["BaseTimeMs"])
    )
    contract = contract.loc[valid].reset_index(drop=True)
    by_trace = contract.set_index("TraceIdx")
    source_trace_idx = pd.to_numeric(
        frame.get("SourceTraceIdx", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    main_base = source_trace_idx.map(by_trace["BaseTimeMs"])
    missing = ~np.isfinite(main_base.to_numpy(dtype=float))
    if missing.any():
        tree = cKDTree(contract[["X", "Y"]].to_numpy(dtype=np.float64))
        _, position = tree.query(
            frame.loc[missing, ["CenterX", "CenterY"]].to_numpy(dtype=np.float64),
            k=1,
            workers=-1,
        )
        main_base.loc[missing] = contract.iloc[np.asarray(position, dtype=np.int64)][
            "BaseTimeMs"
        ].to_numpy(dtype=float)
    out = frame.copy()
    out["MainBaseTimeMs"] = main_base.to_numpy(dtype=np.float64)
    return out


def prepare_main(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["IsOriginalFault"] = 0
    out["IsBaseSupplement"] = 0
    out["BaseSuppTopTimeMs"] = pd.to_numeric(out["MainBaseTimeMs"], errors="coerce")
    out["BaseSuppBottomTimeMs"] = np.nan
    out["RelativeToBaseMs"] = out["CenterTime"] - out["BaseSuppTopTimeMs"]
    out["ExtensionConfidence"] = 1.0
    return out


def continuity_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    relative = pd.to_numeric(frame["RelativeToBaseMs"], errors="coerce")
    near = frame.loc[relative.between(-100.0, 100.0, inclusive="both")].copy()
    near["RelativeToBaseMs"] = pd.to_numeric(near["RelativeToBaseMs"], errors="coerce")
    bins = [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0]
    labels = ["-100~-50", "-50~-25", "-25~0", "0~25", "25~50", "50~100"]
    near["BoundaryBin"] = pd.cut(
        near["RelativeToBaseMs"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    table = (
        near.groupby(["BoundaryBin", "FractureScale"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(labels, fill_value=0)
    )
    for scale in ("small", "medium", "large"):
        if scale not in table.columns:
            table[scale] = 0
    table["all"] = table[["small", "medium", "large"]].sum(axis=1)
    table = table.reset_index()

    core = np.sort(
        near.loc[
            near["RelativeToBaseMs"].between(-50.0, 50.0, inclusive="both"),
            "RelativeToBaseMs",
        ].to_numpy(dtype=float)
    )
    unique = np.unique(core)
    max_gap = float(np.max(np.diff(unique))) if len(unique) > 1 else float("inf")
    scale_stats: dict[str, Any] = {}
    for scale, group in frame.groupby("FractureScale"):
        values = pd.to_numeric(group["RelativeToBaseMs"], errors="coerce")
        scale_stats[str(scale)] = {
            "count": int(len(group)),
            "relative_ms": {
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            },
        }
    summary = {
        "bin_labels": labels,
        "bin_counts": table.to_dict(orient="records"),
        "scale_stats": scale_stats,
        "core_unique_center_count": int(len(unique)),
        "maximum_core_center_gap_ms": max_gap,
        "no_dead_band_over_20ms": bool(max_gap <= 20.0),
    }
    return table, summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lower_raw = pd.read_csv(args.base_supp_csv.resolve(), low_memory=False)
    main_raw = pd.read_csv(args.main_csv.resolve(), low_memory=False)
    main_raw = attach_main_base(main_raw, args.main_horizon_contract.resolve())
    lower_raw = attach_base_window(lower_raw, args.horizon_contract.resolve())
    lower = relabel_base_supp(lower_raw)
    main_pred = prepare_main(main_raw)

    lower_top = pd.to_numeric(lower["BaseSuppTopTimeMs"], errors="coerce")
    lower_bottom = pd.to_numeric(lower["BaseSuppBottomTimeMs"], errors="coerce")
    lower_time = pd.to_numeric(lower["CenterTime"], errors="coerce")
    inside_window = (
        lower_time.ge(lower_top)
        & lower_time.lt(lower_bottom)
        & lower_bottom.le(ATTRIBUTE_BOTTOM_MS + 1.0e-6)
    )
    if not bool(inside_window.all()):
        raise ValueError(
            f"{int((~inside_window).sum())} base-supp patches fall outside Base-BaseV"
        )
    if bool((lower["RelativeToBaseMs"] < -1.0e-6).any()):
        raise ValueError("base-supp DFN contains centers above the actual Base")

    main_relative = pd.to_numeric(main_pred["RelativeToBaseMs"], errors="coerce")
    pilot_mask = (
        pd.to_numeric(main_pred["CenterX"], errors="coerce").between(663450.0, 668450.0)
        & pd.to_numeric(main_pred["CenterY"], errors="coerce").between(4239262.0, 4244262.0)
    )
    main_pilot = main_pred.loc[pilot_mask].copy()
    main_pilot_relative = pd.to_numeric(main_pilot["RelativeToBaseMs"], errors="coerce")
    if bool((main_pilot_relative > 0.0).any()):
        raise ValueError("main DFN inside the pilot block contains centers below Base")

    lower_out = output_dir / "base_supp_dfn_fracture_patches.csv"
    lower.to_csv(lower_out, index=False, encoding="utf-8-sig")

    all_columns = list(dict.fromkeys([*main_pred.columns.tolist(), *lower.columns.tolist()]))
    merged = pd.concat(
        [
            main_pred.reindex(columns=all_columns),
            lower.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    if merged["PatchID"].duplicated().any():
        duplicates = merged.loc[merged["PatchID"].duplicated(), "PatchID"].head(10).tolist()
        raise ValueError(f"merged PatchID is not unique: {duplicates}")
    merged_out = output_dir / "merged_main_plus_base_supp_patches.csv"
    merged.to_csv(merged_out, index=False, encoding="utf-8-sig")

    lower_mesh = read_polydata(args.base_supp_vtk.resolve())
    main_mesh = read_polydata(args.main_vtk.resolve())
    lower_pred = group_mesh(lower_mesh, 1)
    main_pred_mesh = group_mesh(main_mesh, 1)
    main_original = group_mesh(main_mesh, 2)

    add_scalar_array(lower_pred, "LayerCode", LAYER_CODE)
    add_scalar_array(lower_pred, "IsOriginalFault", 0)
    add_scalar_array(lower_pred, "IsBaseSupplement", 1)
    add_scalar_array(main_pred_mesh, "IsOriginalFault", 0)
    add_scalar_array(main_pred_mesh, "IsBaseSupplement", 0)
    main_original["IsOriginalFault"] = np.ones(main_original.n_cells, dtype=np.int32)
    main_original["IsBaseSupplement"] = np.zeros(main_original.n_cells, dtype=np.int32)
    main_mesh["IsOriginalFault"] = (
        np.asarray(main_mesh.cell_data["GeometryGroupCode"], dtype=np.int32) == 2
    ).astype(np.int32)
    main_mesh["IsBaseSupplement"] = np.zeros(main_mesh.n_cells, dtype=np.int32)

    lower_pred_path = output_dir / "base_supp_dfn_predicted_raw_time.vtk"
    lower_pred.save(lower_pred_path)
    merged_pred_path = output_dir / "merged_predicted_dfn_raw_time.vtk"
    merged_pred = merge_meshes([main_pred_mesh, lower_pred])
    main_pred_layers = np.asarray(main_pred_mesh.cell_data.get("LayerCode"), dtype=np.int32)
    merged_pred_layers = np.concatenate(
        [main_pred_layers, np.full(lower_pred.n_cells, LAYER_CODE, dtype=np.int32)]
    )
    merged_pred.cell_data["LayerCode"] = merged_pred_layers
    merged_pred.cell_data["IsOriginalFault"] = np.zeros(merged_pred.n_cells, dtype=np.int32)
    merged_pred.cell_data["IsBaseSupplement"] = np.concatenate(
        [
            np.zeros(main_pred_mesh.n_cells, dtype=np.int32),
            np.ones(lower_pred.n_cells, dtype=np.int32),
        ]
    )
    merged_pred.save(merged_pred_path)
    unified_path = output_dir / "merged_unified_dfn_raw_time.vtk"
    unified = merge_meshes([main_mesh, lower_pred])
    main_layers = np.asarray(main_mesh.cell_data.get("LayerCode"), dtype=np.int32)
    unified.cell_data["LayerCode"] = np.concatenate(
        [main_layers, np.full(lower_pred.n_cells, LAYER_CODE, dtype=np.int32)]
    )
    unified_group = np.asarray(unified.cell_data["GeometryGroupCode"], dtype=np.int32)
    unified.cell_data["IsOriginalFault"] = (unified_group == 2).astype(np.int32)
    unified.cell_data["IsBaseSupplement"] = np.concatenate(
        [
            np.zeros(main_mesh.n_cells, dtype=np.int32),
            np.ones(lower_pred.n_cells, dtype=np.int32),
        ]
    )
    unified.save(unified_path)

    continuity, continuity_summary = continuity_table(
        pd.concat([main_pilot, lower], ignore_index=True)
    )
    continuity_path = output_dir / "base_supp_boundary_continuity.csv"
    continuity.to_csv(continuity_path, index=False, encoding="utf-8-sig")

    lower_scale_counts = {
        str(key): int(value)
        for key, value in lower["FractureScale"].value_counts().to_dict().items()
    }
    main_block_mask = pilot_mask
    lower_min = float(lower["RelativeToBaseMs"].min())
    lower_max = float(lower["RelativeToBaseMs"].max())
    checks = {
        "all_base_supp_centers_inside_base_to_basev": bool(inside_window.all()),
        "base_supp_centers_at_or_below_actual_base": bool(
            (lower["RelativeToBaseMs"] >= -1.0e-6).all()
        ),
        "main_in_pilot_at_or_above_actual_base": bool((main_pilot_relative <= 0.0).all()),
        "merged_patch_ids_unique": bool(not merged["PatchID"].duplicated().any()),
        "base_supp_has_three_scales": set(lower_scale_counts) == {"small", "medium", "large"},
        "base_supp_starts_within_30ms_of_base": bool(lower_min <= 30.0),
        "base_supp_reaches_at_least_250ms_below_base": bool(lower_max >= 250.0),
        "boundary_has_no_dead_band_over_20ms": bool(continuity_summary["no_dead_band_over_20ms"]),
        "merged_predicted_vtk_exists": bool(merged_pred_path.exists()),
        "merged_unified_vtk_exists": bool(unified_path.exists()),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "layer_group": LAYER_NAME,
        "layer_code": LAYER_CODE,
        "confidence_note": "基底以下没有目标层真实裂缝标签；本结果复用太古界风化壳模型，属于低置信度属性外推。",
        "inputs": {
            "base_supp_csv": str(args.base_supp_csv.resolve()),
            "base_supp_vtk": str(args.base_supp_vtk.resolve()),
            "main_csv": str(args.main_csv.resolve()),
            "main_vtk": str(args.main_vtk.resolve()),
            "horizon_contract": str(args.horizon_contract.resolve()),
            "main_horizon_contract": str(args.main_horizon_contract.resolve()),
            "main_csv_sha256": file_sha256(args.main_csv.resolve()),
            "main_vtk_sha256": file_sha256(args.main_vtk.resolve()),
        },
        "outputs": {
            "base_supp_patches_csv": str(lower_out),
            "merged_patches_csv": str(merged_out),
            "base_supp_predicted_vtk": str(lower_pred_path),
            "merged_predicted_vtk": str(merged_pred_path),
            "merged_unified_vtk": str(unified_path),
            "boundary_continuity_csv": str(continuity_path),
        },
        "counts": {
            "main_total": int(len(main_pred)),
            "main_in_pilot_block": int(main_block_mask.sum()),
            "preexisting_main_below_base_total": int((main_relative > 0.0).sum()),
            "preexisting_main_below_base_in_pilot": int((main_pilot_relative > 0.0).sum()),
            "base_supp": int(len(lower)),
            "merged": int(len(merged)),
            "base_supp_scales": lower_scale_counts,
            "main_block_scales": {
                str(key): int(value)
                for key, value in main_pred.loc[
                    main_block_mask, "FractureScale"
                ].value_counts().to_dict().items()
            },
        },
        "time_ranges_ms": {
            "base_supp_center": [
                float(lower["CenterTime"].min()),
                float(lower["CenterTime"].max()),
            ],
            "base_supp_relative_to_base": [lower_min, lower_max],
            "base_supp_horizon_distance_m_max": float(
                lower["BaseSuppHorizonDistanceM"].max()
            ),
            "preexisting_main_max_below_base_ms": float(
                main_relative.loc[main_relative > 0.0].max()
            )
            if bool((main_relative > 0.0).any())
            else 0.0,
            "merged_center": [
                float(merged["CenterTime"].min()),
                float(merged["CenterTime"].max()),
            ],
        },
        "vtk": {
            "base_supp_predicted_cells": int(lower_pred.n_cells),
            "main_predicted_cells": int(main_pred_mesh.n_cells),
            "main_original_fault_cells": int(main_original.n_cells),
            "merged_predicted_cells": int(read_polydata(merged_pred_path).n_cells),
            "merged_unified_cells": int(read_polydata(unified_path).n_cells),
            "original_fault_preserved_once": bool(main_original.n_cells > 0),
        },
        "boundary_continuity": continuity_summary,
        "checks": checks,
        "reflection": "下部补充层只追加预测片；主 VTK 的原始断层组保留一次，没有复制或重新旋转。",
    }
    write_json(output_dir / "base_supp_merge_qc_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
