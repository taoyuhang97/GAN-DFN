# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from roundtrip_common import (
    DEFAULT_OUTPUT_ROOT,
    build_voxel_overlap_metrics,
    export_voxel_preview_vtk,
    fit_voxel_components_to_patches,
    load_layer_table,
    read_csv_utf8,
    load_voxel_package,
    rasterize_patches_to_voxel,
    write_csv_utf8,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将体素裂缝标签拟合回裂缝片参数表。")
    parser.add_argument("--voxel-npz", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="voxel_to_dfn")
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--min-component-voxels", type=int, default=6)
    parser.add_argument("--connectivity", type=int, default=1)
    parser.add_argument("--layers-csv", type=Path)
    parser.add_argument("--skip-vtk-preview", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    occupancy, normals, grid, metadata = load_voxel_package(args.voxel_npz)
    unit_id = str(metadata.get("unit_id", grid.unit_id or "UNKNOWN_UNIT"))
    layers_df = load_layer_table(args.voxel_npz.parent)
    if layers_df.empty and metadata.get("patch_csv"):
        layers_df = load_layer_table(Path(str(metadata["patch_csv"])).parent)
    if args.layers_csv is not None and args.layers_csv.exists():
        layers_df = read_csv_utf8(args.layers_csv)

    patch_df, fit_summary = fit_voxel_components_to_patches(
        occupancy=occupancy,
        grid=grid,
        threshold=args.occupancy_threshold,
        min_component_voxels=args.min_component_voxels,
        connectivity=args.connectivity,
        layers_df=layers_df,
        normals=normals,
        reference_patch_stats=metadata.get("reference_patch_stats"),
    )

    output_dir = args.output_root / args.run_name / unit_id
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_csv = output_dir / "roundtrip_patches.csv"
    write_csv_utf8(patch_df, patch_csv)

    reconstructed_occupancy = None
    overlap_metrics = {}
    preview_path = None
    if not patch_df.empty:
        reconstructed_occupancy, _, _ = rasterize_patches_to_voxel(
            patch_df=patch_df,
            grid=grid,
            thickness_vox=float(metadata.get("thickness_vox", 1.0)),
            include_normals=False,
        )
        overlap_metrics = build_voxel_overlap_metrics(
            input_occupancy=occupancy,
            output_occupancy=reconstructed_occupancy,
            threshold=args.occupancy_threshold,
        )
        if not args.skip_vtk_preview:
            preview_path = export_voxel_preview_vtk(output_dir / "roundtrip_voxel_preview.vtk", reconstructed_occupancy, grid)

    summary = {
        "unit_id": unit_id,
        "voxel_npz": str(args.voxel_npz),
        "output_dir": str(output_dir),
        "roundtrip_patch_csv": str(patch_csv),
        "roundtrip_preview_vtk": str(preview_path) if preview_path else "",
        **fit_summary,
        **overlap_metrics,
    }
    write_json(output_dir / "voxel_to_dfn_summary.json", summary)
    print(f"单位: {unit_id}")
    print(f"输出目录: {output_dir}")
    print(f"裂缝片文件: {patch_csv}")


if __name__ == "__main__":
    main()
