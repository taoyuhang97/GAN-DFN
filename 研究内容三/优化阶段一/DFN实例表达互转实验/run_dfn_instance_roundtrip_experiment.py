# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from instance_roundtrip_common import (
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SLOTS_PER_VOXEL,
    DEFAULT_TRACE_HEADER_CSV,
    DEFAULT_UNIT_DFN_ROOT,
    VtkPatchExportConfig,
    append_instance_roundtrip_summary_to_docx,
    build_grid_spec,
    build_patch_match_metrics,
    build_voxel_overlap_metrics,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    compute_unit_bounds_from_trace_header,
    decode_instance_label_to_patches,
    encode_patches_to_instance_label,
    export_patch_comparison_vtk,
    export_patch_vtk_files,
    load_layer_table,
    load_patch_table,
    load_unit_summary,
    rasterize_patches_to_voxel,
    resolve_patch_csv,
    save_instance_label_package,
    summarize_patch_statistics,
    write_csv_utf8,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DFN <-> instance-label roundtrip validation.")
    parser.add_argument("--unit-id", type=str)
    parser.add_argument("--patch-csv", type=Path)
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="instance_roundtrip")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--z-padding-ms", type=float, default=0.0)
    parser.add_argument("--slots-per-voxel", type=int, default=DEFAULT_SLOTS_PER_VOXEL)
    parser.add_argument("--center-threshold", type=float, default=0.5)
    parser.add_argument("--thickness-vox-for-compare", type=float, default=1.0)
    parser.add_argument("--vtk-display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-invert-time", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    patch_csv = resolve_patch_csv(args.unit_id or None, args.patch_csv, args.unit_dfn_root)
    input_patch_df = load_patch_table(patch_csv)
    if input_patch_df.empty:
        raise ValueError(f"Empty patch csv: {patch_csv}")

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

    output_dir = args.output_root / args.run_name / unit_id
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_config = VtkPatchExportConfig(
        display_z_scale=float(args.vtk_display_z_scale),
        invert_time=bool(args.vtk_invert_time),
    )

    write_csv_utf8(input_patch_df, output_dir / "input_patches.csv")

    label_payload, encoded_df, overflow_df, encode_summary = encode_patches_to_instance_label(
        patch_df=input_patch_df,
        grid=grid,
        slots_per_voxel=args.slots_per_voxel,
    )
    instance_npz = save_instance_label_package(
        output_path=output_dir / "instance_label_volume.npz",
        label_payload=label_payload,
        grid=grid,
        metadata={
            "unit_id": unit_id,
            "block_x": block_x,
            "block_y": block_y,
            "patch_csv": str(patch_csv),
            "output_dir": str(output_dir),
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
            "z_padding_ms": float(args.z_padding_ms),
            "slots_per_voxel": int(args.slots_per_voxel),
            "center_threshold": float(args.center_threshold),
            "thickness_vox_for_compare": float(args.thickness_vox_for_compare),
            "reference_patch_stats": summarize_patch_statistics(input_patch_df),
            "source_kind_mapping": encode_summary.get("source_kind_mapping", {}),
            "layer_surface_pair_mapping": encode_summary.get("layer_surface_pair_mapping", {}),
            "unit_layer_segment_mapping": encode_summary.get("unit_layer_segment_mapping", {}),
        },
    )
    write_csv_utf8(encoded_df, output_dir / "encoded_instances.csv")
    write_csv_utf8(overflow_df, output_dir / "overflow_instances.csv")

    roundtrip_patch_df, decoded_df, decode_summary = decode_instance_label_to_patches(
        label_payload=label_payload,
        grid=grid,
        layers_df=layers_df,
        threshold=args.center_threshold,
        label_metadata={
            "source_kind_mapping": encode_summary.get("source_kind_mapping", {}),
            "layer_surface_pair_mapping": encode_summary.get("layer_surface_pair_mapping", {}),
            "unit_layer_segment_mapping": encode_summary.get("unit_layer_segment_mapping", {}),
        },
    )
    write_csv_utf8(roundtrip_patch_df, output_dir / "roundtrip_patches.csv")
    write_csv_utf8(decoded_df, output_dir / "decoded_instances.csv")

    input_occ, _, _ = rasterize_patches_to_voxel(
        patch_df=input_patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox_for_compare,
        include_normals=False,
    )
    output_occ, _, _ = rasterize_patches_to_voxel(
        patch_df=roundtrip_patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox_for_compare,
        include_normals=False,
    )
    overlap_metrics = build_voxel_overlap_metrics(
        input_occupancy=input_occ,
        output_occupancy=output_occ,
        threshold=0.5,
    )
    patch_metrics = build_patch_match_metrics(
        input_patches=input_patch_df,
        output_patches=roundtrip_patch_df,
        grid=grid,
    )

    input_vtk = export_patch_vtk_files(
        patch_df=input_patch_df,
        output_dir=output_dir,
        base_name="input_patches",
        title_prefix="input_patches",
        config=vtk_config,
    )
    output_vtk = export_patch_vtk_files(
        patch_df=roundtrip_patch_df,
        output_dir=output_dir,
        base_name="roundtrip_patches",
        title_prefix="roundtrip_patches",
        config=vtk_config,
    )
    compare_vtk = export_patch_comparison_vtk(
        input_patch_df=input_patch_df,
        output_patch_df=roundtrip_patch_df,
        output_dir=output_dir,
        config=vtk_config,
    )
    write_json(
        output_dir / "patch_vtk_mappings.json",
        {
            "input": input_vtk.get("mappings", {}),
            "roundtrip": output_vtk.get("mappings", {}),
            "compare": compare_vtk.get("mappings", {}),
        },
    )

    summary = {
        "unit_id": unit_id,
        "block_x": block_x,
        "block_y": block_y,
        "patch_csv": str(patch_csv),
        "output_dir": str(output_dir),
        "instance_npz": str(instance_npz),
        "roundtrip_patch_csv": str(output_dir / "roundtrip_patches.csv"),
        "encoded_instances_csv": str(output_dir / "encoded_instances.csv"),
        "decoded_instances_csv": str(output_dir / "decoded_instances.csv"),
        "overflow_instances_csv": str(output_dir / "overflow_instances.csv"),
        "input_patches_raw_vtk": input_vtk.get("raw_vtk", ""),
        "input_patches_display_vtk": input_vtk.get("display_vtk", ""),
        "roundtrip_patches_raw_vtk": output_vtk.get("raw_vtk", ""),
        "roundtrip_patches_display_vtk": output_vtk.get("display_vtk", ""),
        "patch_compare_raw_vtk": compare_vtk.get("raw_vtk", ""),
        "patch_compare_display_vtk": compare_vtk.get("display_vtk", ""),
        "patch_vtk_mappings_json": str(output_dir / "patch_vtk_mappings.json"),
        "unit_data_mode": unit_summary.get("DataMode", ""),
        "unit_reliability_class": unit_summary.get("ReliabilityClass", ""),
        **encode_summary,
        **decode_summary,
        **overlap_metrics,
        **patch_metrics,
    }
    summary_path = output_dir / "instance_roundtrip_summary.json"
    summary["summary_json"] = str(summary_path)
    write_json(summary_path, summary)

    append_instance_roundtrip_summary_to_docx(
        docx_path=args.docx_path,
        title=f"DFN实例表达互转验证 - {unit_id}",
        config={
            "unit_id": unit_id,
            "patch_csv": str(patch_csv),
            "output_dir": str(output_dir),
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
            "slots_per_voxel": int(args.slots_per_voxel),
            "center_threshold": float(args.center_threshold),
            "thickness_vox_for_compare": float(args.thickness_vox_for_compare),
        },
        summary=summary,
    )

    print(f"unit_id: {unit_id}")
    print(f"output_dir: {output_dir}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
