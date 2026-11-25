#!/usr/bin/env python3
"""Merge current main DFN, upper supplement and base-below supplement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LAYER_ROOT = Path(__file__).resolve().parent
REPO = LAYER_ROOT.parents[1]
TAIGU = REPO / "太古界"
if str(LAYER_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_ROOT))

from relabel_merge_base_supp import (  # noqa: E402
    file_sha256,
    group_mesh,
    merge_meshes,
    read_polydata,
    write_json,
)


DEFAULT_MAIN_CSV = (
    TAIGU
    / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1_seg1/well_corrected_dfn_fracture_patches.csv"
)
DEFAULT_MAIN_VTK = (
    TAIGU
    / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1_seg1/well_corrected_dfn_raw_time.vtk"
)
DEFAULT_UPPER_CSV = (
    TAIGU
    / "supp_layer/output/taigu_supp_step8_v1/step8_supp/well_corrected_dfn_supp_patches.csv"
)
DEFAULT_UPPER_VTK = (
    TAIGU
    / "supp_layer/output/taigu_supp_step8_v1/step8_supp/well_corrected_dfn_raw_time.vtk"
)
DEFAULT_LOWER_CSV = (
    TAIGU
    / "base_supp_layer/output/taigu_base_supp_5km_v1/merged/base_supp_dfn_fracture_patches.csv"
)
DEFAULT_LOWER_VTK = (
    TAIGU
    / "base_supp_layer/output/taigu_base_supp_5km_v1/merged/base_supp_dfn_predicted_raw_time.vtk"
)
DEFAULT_OUTPUT = TAIGU / "base_supp_layer/output/taigu_three_way_merge_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path, default=DEFAULT_MAIN_CSV)
    parser.add_argument("--main-vtk", type=Path, default=DEFAULT_MAIN_VTK)
    parser.add_argument("--upper-csv", type=Path, default=DEFAULT_UPPER_CSV)
    parser.add_argument("--upper-vtk", type=Path, default=DEFAULT_UPPER_VTK)
    parser.add_argument("--lower-csv", type=Path, default=DEFAULT_LOWER_CSV)
    parser.add_argument("--lower-vtk", type=Path, default=DEFAULT_LOWER_VTK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def prepare_csv_source(
    frame: pd.DataFrame,
    *,
    source: str,
    prefix: str,
) -> pd.DataFrame:
    out = frame.copy()
    original_id = out["PatchID"].astype(str)
    out["OriginalPatchID"] = original_id
    needs_prefix = ~original_id.str.startswith(prefix)
    out.loc[needs_prefix, "PatchID"] = prefix + original_id.loc[needs_prefix]
    out["MergeSource"] = source
    out["IsOriginalFault"] = 0
    return out


def set_mesh_metadata(
    mesh,
    *,
    layer_code: int,
    source_code: int,
    is_upper: bool,
    is_lower: bool,
) -> None:
    mesh.cell_data["LayerCode"] = np.full(mesh.n_cells, layer_code, dtype=np.int32)
    mesh.cell_data["IsOriginalFault"] = np.zeros(mesh.n_cells, dtype=np.int32)
    mesh.cell_data["MergeSourceCode"] = np.full(mesh.n_cells, source_code, dtype=np.int32)
    mesh.cell_data["IsUpperSupplement"] = np.full(
        mesh.n_cells, int(is_upper), dtype=np.int32
    )
    mesh.cell_data["IsBaseSupplement"] = np.full(
        mesh.n_cells, int(is_lower), dtype=np.int32
    )


def restore_merged_metadata(
    merged,
    main_mesh,
    upper_mesh,
    lower_mesh,
) -> None:
    lengths = [main_mesh.n_cells, upper_mesh.n_cells, lower_mesh.n_cells]
    layers = np.concatenate(
        [
            np.asarray(main_mesh.cell_data.get("LayerCode"), dtype=np.int32),
            np.full(upper_mesh.n_cells, 3, dtype=np.int32),
            np.full(lower_mesh.n_cells, 4, dtype=np.int32),
        ]
    )
    source = np.concatenate(
        [
            np.zeros(main_mesh.n_cells, dtype=np.int32),
            np.full(upper_mesh.n_cells, 3, dtype=np.int32),
            np.full(lower_mesh.n_cells, 4, dtype=np.int32),
        ]
    )
    upper = np.concatenate(
        [
            np.zeros(main_mesh.n_cells, dtype=np.int32),
            np.ones(upper_mesh.n_cells, dtype=np.int32),
            np.zeros(lower_mesh.n_cells, dtype=np.int32),
        ]
    )
    lower = np.concatenate(
        [
            np.zeros(main_mesh.n_cells, dtype=np.int32),
            np.zeros(upper_mesh.n_cells, dtype=np.int32),
            np.ones(lower_mesh.n_cells, dtype=np.int32),
        ]
    )
    group = np.asarray(merged.cell_data.get("GeometryGroupCode"), dtype=np.int32)
    if group.size != merged.n_cells:
        raise ValueError("merged VTK is missing GeometryGroupCode")
    if sum(lengths) != merged.n_cells:
        raise ValueError(
            f"merged cell order mismatch: inputs={sum(lengths)} output={merged.n_cells}"
        )
    merged.cell_data["LayerCode"] = layers
    merged.cell_data["MergeSourceCode"] = source
    merged.cell_data["IsUpperSupplement"] = upper
    merged.cell_data["IsBaseSupplement"] = lower
    merged.cell_data["IsOriginalFault"] = (group == 2).astype(np.int32)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_csv = pd.read_csv(args.main_csv.resolve(), low_memory=False)
    upper_csv = pd.read_csv(args.upper_csv.resolve(), low_memory=False)
    lower_csv = pd.read_csv(args.lower_csv.resolve(), low_memory=False)
    main_csv = prepare_csv_source(main_csv, source="original_dfn", prefix="original_")
    upper_csv = prepare_csv_source(upper_csv, source="upper_supp", prefix="upper_supp_")
    lower_csv = prepare_csv_source(lower_csv, source="base_supp", prefix="base_supp_")

    all_columns = list(
        dict.fromkeys(
            [
                *main_csv.columns.tolist(),
                *upper_csv.columns.tolist(),
                *lower_csv.columns.tolist(),
            ]
        )
    )
    merged_csv = pd.concat(
        [
            main_csv.reindex(columns=all_columns),
            upper_csv.reindex(columns=all_columns),
            lower_csv.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    if merged_csv["PatchID"].duplicated().any():
        duplicate = merged_csv.loc[
            merged_csv["PatchID"].duplicated(), "PatchID"
        ].head(10).tolist()
        raise ValueError(f"three-way merged PatchID is not unique: {duplicate}")
    merged_csv_path = output_dir / "three_way_merged_dfn_patches.csv"
    merged_csv.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")

    main_mesh = read_polydata(args.main_vtk.resolve())
    upper_mesh = group_mesh(read_polydata(args.upper_vtk.resolve()), 1)
    lower_mesh = group_mesh(read_polydata(args.lower_vtk.resolve()), 1)
    main_pred = group_mesh(main_mesh, 1)
    main_original = group_mesh(main_mesh, 2)
    set_mesh_metadata(
        main_pred,
        layer_code=-1,
        source_code=0,
        is_upper=False,
        is_lower=False,
    )
    set_mesh_metadata(
        upper_mesh,
        layer_code=3,
        source_code=3,
        is_upper=True,
        is_lower=False,
    )
    set_mesh_metadata(
        lower_mesh,
        layer_code=4,
        source_code=4,
        is_upper=False,
        is_lower=True,
    )

    main_pred_layers = np.asarray(main_pred.cell_data.get("LayerCode"), dtype=np.int32)
    main_original.cell_data["IsOriginalFault"] = np.ones(
        main_original.n_cells, dtype=np.int32
    )
    main_original.cell_data["MergeSourceCode"] = np.zeros(
        main_original.n_cells, dtype=np.int32
    )
    main_original.cell_data["IsUpperSupplement"] = np.zeros(
        main_original.n_cells, dtype=np.int32
    )
    main_original.cell_data["IsBaseSupplement"] = np.zeros(
        main_original.n_cells, dtype=np.int32
    )

    predicted = merge_meshes([main_pred, upper_mesh, lower_mesh])
    predicted.cell_data["LayerCode"] = np.concatenate(
        [
            main_pred_layers,
            np.full(upper_mesh.n_cells, 3, dtype=np.int32),
            np.full(lower_mesh.n_cells, 4, dtype=np.int32),
        ]
    )
    predicted.cell_data["IsOriginalFault"] = np.zeros(predicted.n_cells, dtype=np.int32)
    predicted.cell_data["MergeSourceCode"] = np.concatenate(
        [
            np.zeros(main_pred.n_cells, dtype=np.int32),
            np.full(upper_mesh.n_cells, 3, dtype=np.int32),
            np.full(lower_mesh.n_cells, 4, dtype=np.int32),
        ]
    )
    predicted.cell_data["IsUpperSupplement"] = np.concatenate(
        [
            np.zeros(main_pred.n_cells, dtype=np.int32),
            np.ones(upper_mesh.n_cells, dtype=np.int32),
            np.zeros(lower_mesh.n_cells, dtype=np.int32),
        ]
    )
    predicted.cell_data["IsBaseSupplement"] = np.concatenate(
        [
            np.zeros(main_pred.n_cells, dtype=np.int32),
            np.zeros(upper_mesh.n_cells, dtype=np.int32),
            np.ones(lower_mesh.n_cells, dtype=np.int32),
        ]
    )
    predicted_path = output_dir / "three_way_predicted_dfn_raw_time.vtk"
    predicted.save(predicted_path)

    unified = merge_meshes([main_mesh, upper_mesh, lower_mesh])
    restore_merged_metadata(unified, main_mesh, upper_mesh, lower_mesh)
    unified_path = output_dir / "three_way_unified_dfn_raw_time.vtk"
    unified.save(unified_path)

    layer_counts = {
        str(key): int(value)
        for key, value in merged_csv["LayerGroup"].value_counts(dropna=False).items()
    }
    scale_counts = {
        str(key): int(value)
        for key, value in merged_csv["FractureScale"].value_counts(dropna=False).items()
    }
    source_counts = {
        "original_dfn": int(len(main_csv)),
        "upper_supp": int(len(upper_csv)),
        "base_supp": int(len(lower_csv)),
        "merged_predicted": int(len(merged_csv)),
    }
    checks = {
        "patch_ids_unique": bool(not merged_csv["PatchID"].duplicated().any()),
        "has_three_sources": set(merged_csv["MergeSource"].astype(str))
        == {"original_dfn", "upper_supp", "base_supp"},
        "has_three_layers": set(merged_csv["LayerGroup"].astype(str))
        == {"上部复合层", "太古界风化壳", "上部补充层", "基底以下补充层"},
        "has_three_scales": set(merged_csv["FractureScale"].astype(str))
        == {"small", "medium", "large"},
        "predicted_cell_count_matches_csv": int(predicted.n_cells) == int(len(merged_csv)),
        "original_fault_preserved_once": bool(main_original.n_cells > 0),
        "predicted_vtk_exists": predicted_path.exists(),
        "unified_vtk_exists": unified_path.exists(),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "generation_logic": "original_dfn_plus_upper_supp_plus_base_supp_geometric_union",
        "inputs": {
            "main_csv": str(args.main_csv.resolve()),
            "main_vtk": str(args.main_vtk.resolve()),
            "upper_csv": str(args.upper_csv.resolve()),
            "upper_vtk": str(args.upper_vtk.resolve()),
            "lower_csv": str(args.lower_csv.resolve()),
            "lower_vtk": str(args.lower_vtk.resolve()),
            "main_csv_sha256": file_sha256(args.main_csv.resolve()),
            "main_vtk_sha256": file_sha256(args.main_vtk.resolve()),
            "upper_csv_sha256": file_sha256(args.upper_csv.resolve()),
            "upper_vtk_sha256": file_sha256(args.upper_vtk.resolve()),
            "lower_csv_sha256": file_sha256(args.lower_csv.resolve()),
            "lower_vtk_sha256": file_sha256(args.lower_vtk.resolve()),
        },
        "outputs": {
            "merged_patches_csv": str(merged_csv_path),
            "merged_predicted_vtk": str(predicted_path),
            "merged_unified_vtk": str(unified_path),
        },
        "source_counts": source_counts,
        "layer_group_counts": layer_counts,
        "fracture_scale_counts": scale_counts,
        "time_range_ms": [
            float(merged_csv["CenterTime"].min()),
            float(merged_csv["CenterTime"].max()),
        ],
        "vtk": {
            "predicted_cells": int(predicted.n_cells),
            "unified_cells": int(unified.n_cells),
            "original_fault_cells": int(main_original.n_cells),
            "geometry_group_codes": {
                "predicted_patches": 1,
                "original_fault_triangles": 2,
            },
        },
        "checks": checks,
        "reflection": "上部补充层和下部补充层只追加预测片；当前主 DFN 的原始断层三角面保留一次。",
    }
    write_json(output_dir / "three_way_merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
