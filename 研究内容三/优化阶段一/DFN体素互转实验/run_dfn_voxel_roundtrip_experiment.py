# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from roundtrip_common import (
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRACE_HEADER_CSV,
    DEFAULT_UNIT_DFN_ROOT,
    VtkPatchExportConfig,
    append_roundtrip_summary_to_docx,
    build_grid_spec,
    build_patch_match_metrics,
    build_voxel_overlap_metrics,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    compute_unit_bounds_from_trace_header,
    export_voxel_preview_vtk,
    export_patch_comparison_vtk,
    export_patch_vtk_files,
    fit_voxel_components_to_patches,
    load_layer_table,
    load_patch_table,
    load_unit_summary,
    rasterize_patches_to_voxel,
    resolve_patch_csv,
    save_voxel_package,
    summarize_patch_statistics,
    write_csv_utf8,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行单元 DFN 与体素 round-trip 验证。")
    parser.add_argument("--unit-id", type=str, default="")
    parser.add_argument("--patch-csv", type=Path)
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="roundtrip")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--z-padding-ms", type=float, default=0.0)
    parser.add_argument("--thickness-vox", type=float, default=1.0)
    parser.add_argument("--channels", choices=["occupancy", "occupancy_normals"], default="occupancy")
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--min-component-voxels", type=int, default=6)
    parser.add_argument("--connectivity", type=int, default=1)
    parser.add_argument("--vtk-display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-invert-time", action="store_true")
    parser.add_argument("--skip-vtk-preview", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    patch_csv = resolve_patch_csv(args.unit_id or None, args.patch_csv, args.unit_dfn_root)
    input_patch_df = load_patch_table(patch_csv)
    if input_patch_df.empty:
        raise ValueError(f"裂缝片文件为空: {patch_csv}")

    unit_id = str(input_patch_df["UnitID"].iloc[0] or patch_csv.parent.name)
    block_x = int(input_patch_df["BlockX"].iloc[0])
    block_y = int(input_patch_df["BlockY"].iloc[0])
    layers_df = load_layer_table(patch_csv.parent)
    unit_summary = load_unit_summary(patch_csv.parent)

    try:
        x_min, x_max, y_min, y_max = compute_unit_bounds_from_trace_header(
            args.trace_header_csv,
            block_x=block_x,
            block_y=block_y,
        )
    except Exception:
        x_min, x_max, y_min, y_max = compute_unit_bounds_from_patch_table(input_patch_df)

    z_min, z_max = compute_time_bounds(input_patch_df, layers_df, z_padding_ms=args.z_padding_ms)
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
    output_dir = args.output_root / args.run_name / unit_id
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_utf8(input_patch_df, output_dir / "input_patches.csv")
    vtk_config = VtkPatchExportConfig(
        display_z_scale=float(args.vtk_display_z_scale),
        invert_time=bool(args.vtk_invert_time),
    )
    reference_patch_stats = summarize_patch_statistics(input_patch_df)

    input_occupancy, input_normals, raster_summary = rasterize_patches_to_voxel(
        patch_df=input_patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox,
        include_normals=include_normals,
    )
    voxel_npz_path = output_dir / "orig_voxel_volume.npz"
    save_voxel_package(
        output_path=voxel_npz_path,
        occupancy=input_occupancy,
        normals=input_normals,
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
            "reference_patch_stats": reference_patch_stats,
        },
    )

    orig_preview_vtk = None
    if not args.skip_vtk_preview:
        orig_preview_vtk = export_voxel_preview_vtk(output_dir / "orig_voxel_preview.vtk", input_occupancy, grid)
    input_patch_vtk = export_patch_vtk_files(
        patch_df=input_patch_df,
        output_dir=output_dir,
        base_name="input_patches",
        title_prefix="input_patches",
        config=vtk_config,
    )

    roundtrip_patch_df, fit_summary = fit_voxel_components_to_patches(
        occupancy=input_occupancy,
        grid=grid,
        threshold=args.occupancy_threshold,
        min_component_voxels=args.min_component_voxels,
        connectivity=args.connectivity,
        layers_df=layers_df,
        normals=input_normals,
        reference_patch_stats=reference_patch_stats,
    )
    roundtrip_patch_csv = output_dir / "roundtrip_patches.csv"
    write_csv_utf8(roundtrip_patch_df, roundtrip_patch_csv)

    roundtrip_occupancy, _, roundtrip_raster_summary = rasterize_patches_to_voxel(
        patch_df=roundtrip_patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox,
        include_normals=False,
    )
    roundtrip_voxel_npz = output_dir / "roundtrip_voxel_volume.npz"
    save_voxel_package(
        output_path=roundtrip_voxel_npz,
        occupancy=roundtrip_occupancy,
        normals=None,
        grid=grid,
        metadata={
            "unit_id": unit_id,
            "block_x": block_x,
            "block_y": block_y,
            "source": "roundtrip_fit",
            "channels": "occupancy",
            "thickness_vox": float(args.thickness_vox),
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
        },
    )

    roundtrip_preview_vtk = None
    if not args.skip_vtk_preview:
        roundtrip_preview_vtk = export_voxel_preview_vtk(output_dir / "roundtrip_voxel_preview.vtk", roundtrip_occupancy, grid)
    roundtrip_patch_vtk = export_patch_vtk_files(
        patch_df=roundtrip_patch_df,
        output_dir=output_dir,
        base_name="roundtrip_patches",
        title_prefix="roundtrip_patches",
        config=vtk_config,
    )
    compare_patch_vtk = export_patch_comparison_vtk(
        input_patch_df=input_patch_df,
        output_patch_df=roundtrip_patch_df,
        output_dir=output_dir,
        config=vtk_config,
    )
    patch_vtk_mappings = {
        "input": input_patch_vtk.get("mappings", {}),
        "roundtrip": roundtrip_patch_vtk.get("mappings", {}),
        "compare": compare_patch_vtk.get("mappings", {}),
    }
    write_json(output_dir / "patch_vtk_mappings.json", patch_vtk_mappings)

    voxel_metrics = build_voxel_overlap_metrics(
        input_occupancy=input_occupancy,
        output_occupancy=roundtrip_occupancy,
        threshold=args.occupancy_threshold,
    )
    patch_metrics = build_patch_match_metrics(
        input_patches=input_patch_df,
        output_patches=roundtrip_patch_df,
        grid=grid,
    )

    summary = {
        "unit_id": unit_id,
        "block_x": block_x,
        "block_y": block_y,
        "patch_csv": str(patch_csv),
        "output_dir": str(output_dir),
        "orig_voxel_npz": str(voxel_npz_path),
        "roundtrip_voxel_npz": str(roundtrip_voxel_npz),
        "orig_preview_vtk": str(orig_preview_vtk) if orig_preview_vtk else "",
        "roundtrip_preview_vtk": str(roundtrip_preview_vtk) if roundtrip_preview_vtk else "",
        "roundtrip_patch_csv": str(roundtrip_patch_csv),
        "input_patches_raw_vtk": str(input_patch_vtk.get("raw_vtk", "")),
        "input_patches_display_vtk": str(input_patch_vtk.get("display_vtk", "")),
        "roundtrip_patches_raw_vtk": str(roundtrip_patch_vtk.get("raw_vtk", "")),
        "roundtrip_patches_display_vtk": str(roundtrip_patch_vtk.get("display_vtk", "")),
        "patch_compare_raw_vtk": str(compare_patch_vtk.get("raw_vtk", "")),
        "patch_compare_display_vtk": str(compare_patch_vtk.get("display_vtk", "")),
        "patch_vtk_mappings_json": str(output_dir / "patch_vtk_mappings.json"),
        "input_patch_count": int(len(input_patch_df)),
        "roundtrip_patch_count": int(len(roundtrip_patch_df)),
        "unit_data_mode": unit_summary.get("DataMode", ""),
        "unit_reliability_class": unit_summary.get("ReliabilityClass", ""),
        **raster_summary,
        **fit_summary,
        **roundtrip_raster_summary,
        **voxel_metrics,
        **patch_metrics,
    }
    summary_path = output_dir / "roundtrip_summary.json"
    summary["summary_json"] = str(summary_path)
    write_json(summary_path, summary)

    config = {
        "unit_id": unit_id,
        "patch_csv": str(patch_csv),
        "output_dir": str(output_dir),
        "xy_resolution": int(args.xy_resolution),
        "z_step_ms": float(args.z_step_ms),
        "thickness_vox": float(args.thickness_vox),
        "channels": args.channels,
        "occupancy_threshold": float(args.occupancy_threshold),
        "min_component_voxels": int(args.min_component_voxels),
        "vtk_display_z_scale": float(args.vtk_display_z_scale),
        "vtk_invert_time": bool(args.vtk_invert_time),
    }
    write_json(output_dir / "roundtrip_config.json", config)
    append_roundtrip_summary_to_docx(
        docx_path=args.docx_path,
        title=f"DFN体素互转验证 {unit_id}",
        config=config,
        summary=summary,
    )

    print(f"单位: {unit_id}")
    print(f"输出目录: {output_dir}")
    print(f"总结文件: {summary_path}")


if __name__ == "__main__":
    main()
