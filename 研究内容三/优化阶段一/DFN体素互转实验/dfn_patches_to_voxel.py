# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from roundtrip_common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRACE_HEADER_CSV,
    DEFAULT_UNIT_DFN_ROOT,
    build_grid_spec,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    compute_unit_bounds_from_trace_header,
    export_voxel_preview_vtk,
    load_layer_table,
    load_patch_table,
    rasterize_patches_to_voxel,
    resolve_patch_csv,
    save_voxel_package,
    summarize_patch_statistics,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将单元 DFN 裂缝片参数表转为体素标签。")
    parser.add_argument("--unit-id", type=str, default="")
    parser.add_argument("--patch-csv", type=Path)
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="dfn_to_voxel")
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--z-padding-ms", type=float, default=0.0)
    parser.add_argument("--thickness-vox", type=float, default=1.0)
    parser.add_argument("--channels", choices=["occupancy", "occupancy_normals"], default="occupancy")
    parser.add_argument("--skip-vtk-preview", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    patch_csv = resolve_patch_csv(args.unit_id or None, args.patch_csv, args.unit_dfn_root)
    patch_df = load_patch_table(patch_csv)
    if patch_df.empty:
        raise ValueError(f"裂缝片文件为空: {patch_csv}")
    unit_id = str(patch_df["UnitID"].iloc[0] or patch_csv.parent.name)
    block_x = int(patch_df["BlockX"].iloc[0])
    block_y = int(patch_df["BlockY"].iloc[0])
    layers_df = load_layer_table(patch_csv.parent)

    try:
        x_min, x_max, y_min, y_max = compute_unit_bounds_from_trace_header(
            args.trace_header_csv,
            block_x=block_x,
            block_y=block_y,
        )
    except Exception:
        x_min, x_max, y_min, y_max = compute_unit_bounds_from_patch_table(patch_df)

    z_min, z_max = compute_time_bounds(patch_df, layers_df, z_padding_ms=args.z_padding_ms)
    grid = build_grid_spec(
        unit_id=unit_id,
        block_x=block_x,
        block_y=block_y,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
        z_bounds=(z_min, z_max),
        xy_resolution=args.xy_resolution,
        z_step_ms=args.z_step_ms,
    )

    include_normals = args.channels == "occupancy_normals"
    occupancy, normals, raster_summary = rasterize_patches_to_voxel(
        patch_df=patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox,
        include_normals=include_normals,
    )

    output_dir = args.output_root / args.run_name / unit_id
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel_npz_path = output_dir / "orig_voxel_volume.npz"
    save_voxel_package(
        output_path=voxel_npz_path,
        occupancy=occupancy,
        normals=normals,
        grid=grid,
        metadata={
            "unit_id": unit_id,
            "block_x": block_x,
            "block_y": block_y,
            "patch_csv": str(patch_csv),
            "channels": args.channels,
            "thickness_vox": float(args.thickness_vox),
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
            "z_padding_ms": float(args.z_padding_ms),
            "reference_patch_stats": summarize_patch_statistics(patch_df),
        },
    )

    preview_path = None
    if not args.skip_vtk_preview:
        preview_path = export_voxel_preview_vtk(output_dir / "orig_voxel_preview.vtk", occupancy, grid)

    summary = {
        "unit_id": unit_id,
        "block_x": block_x,
        "block_y": block_y,
        "patch_csv": str(patch_csv),
        "output_dir": str(output_dir),
        "voxel_npz": str(voxel_npz_path),
        "preview_vtk": str(preview_path) if preview_path else "",
        **raster_summary,
    }
    write_json(output_dir / "dfn_to_voxel_summary.json", summary)
    print(f"单位: {unit_id}")
    print(f"输出目录: {output_dir}")
    print(f"体素文件: {voxel_npz_path}")


if __name__ == "__main__":
    main()
