# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from instance_roundtrip_common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SLOTS_PER_VOXEL,
    DEFAULT_TRACE_HEADER_CSV,
    DEFAULT_UNIT_DFN_ROOT,
    build_grid_spec,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    compute_unit_bounds_from_trace_header,
    encode_patches_to_instance_label,
    load_layer_table,
    load_patch_table,
    resolve_patch_csv,
    save_instance_label_package,
    summarize_patch_statistics,
    write_csv_utf8,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode DFN patches to center-slot instance labels.")
    parser.add_argument("--unit-id", type=str)
    parser.add_argument("--patch-csv", type=Path)
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="dfn_to_instance")
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--z-padding-ms", type=float, default=0.0)
    parser.add_argument("--slots-per-voxel", type=int, default=DEFAULT_SLOTS_PER_VOXEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    patch_csv = resolve_patch_csv(args.unit_id or None, args.patch_csv, args.unit_dfn_root)
    patch_df = load_patch_table(patch_csv)
    if patch_df.empty:
        raise ValueError(f"Empty patch csv: {patch_csv}")

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

    output_dir = args.output_root / args.run_name / unit_id
    output_dir.mkdir(parents=True, exist_ok=True)

    label_payload, encoded_df, overflow_df, encode_summary = encode_patches_to_instance_label(
        patch_df=patch_df,
        grid=grid,
        slots_per_voxel=args.slots_per_voxel,
    )
    package_path = save_instance_label_package(
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
            "reference_patch_stats": summarize_patch_statistics(patch_df),
        },
    )

    write_csv_utf8(patch_df, output_dir / "input_patches.csv")
    write_csv_utf8(encoded_df, output_dir / "encoded_instances.csv")
    write_csv_utf8(overflow_df, output_dir / "overflow_instances.csv")

    summary = {
        "unit_id": unit_id,
        "patch_csv": str(patch_csv),
        "instance_npz": str(package_path),
        "output_dir": str(output_dir),
        **encode_summary,
    }
    write_json(output_dir / "dfn_to_instance_summary.json", summary)
    print(f"unit_id: {unit_id}")
    print(f"output_dir: {output_dir}")
    print(f"instance_npz: {package_path}")


if __name__ == "__main__":
    main()
