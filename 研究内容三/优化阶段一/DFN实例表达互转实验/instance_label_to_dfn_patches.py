# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from instance_roundtrip_common import (
    DEFAULT_OUTPUT_ROOT,
    INSTANCE_PATCH_OUTPUT_COLUMNS,
    build_voxel_overlap_metrics,
    decode_instance_label_to_patches,
    load_instance_label_package,
    load_layer_table,
    load_patch_table,
    rasterize_patches_to_voxel,
    write_csv_utf8,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode center-slot instance labels to DFN patches.")
    parser.add_argument("--instance-npz", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="instance_to_dfn")
    parser.add_argument("--center-threshold", type=float, default=0.5)
    parser.add_argument("--thickness-vox-for-compare", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_payload, grid, metadata = load_instance_label_package(args.instance_npz)
    layers_df = load_layer_table(args.instance_npz.parent)
    if layers_df.empty and metadata.get("patch_csv"):
        layers_df = load_layer_table(Path(str(metadata["patch_csv"])).parent)

    patch_df, decoded_df, decode_summary = decode_instance_label_to_patches(
        label_payload=label_payload,
        grid=grid,
        layers_df=layers_df,
        threshold=args.center_threshold,
        label_metadata=metadata,
    )
    output_dir = args.output_root / args.run_name / str(metadata.get("unit_id", grid.unit_id or "UNKNOWN_UNIT"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if patch_df.empty:
        patch_df = patch_df.reindex(columns=INSTANCE_PATCH_OUTPUT_COLUMNS)
    write_csv_utf8(patch_df, output_dir / "roundtrip_patches.csv")
    write_csv_utf8(decoded_df, output_dir / "decoded_instances.csv")

    overlap_metrics = {}
    if metadata.get("patch_csv") and Path(str(metadata["patch_csv"])).exists() and not patch_df.empty:
        input_patch_df = load_patch_table(Path(str(metadata["patch_csv"])))
        input_occ, _, _ = rasterize_patches_to_voxel(
            patch_df=input_patch_df,
            grid=grid,
            thickness_vox=args.thickness_vox_for_compare,
            include_normals=False,
        )
        output_occ, _, _ = rasterize_patches_to_voxel(
            patch_df=patch_df,
            grid=grid,
            thickness_vox=args.thickness_vox_for_compare,
            include_normals=False,
        )
        overlap_metrics = build_voxel_overlap_metrics(
            input_occupancy=input_occ,
            output_occupancy=output_occ,
            threshold=0.5,
        )

    summary = {
        "unit_id": str(metadata.get("unit_id", grid.unit_id)),
        "instance_npz": str(args.instance_npz),
        "output_dir": str(output_dir),
        "roundtrip_patch_csv": str(output_dir / "roundtrip_patches.csv"),
        **decode_summary,
        **overlap_metrics,
    }
    write_json(output_dir / "instance_to_dfn_summary.json", summary)
    print(f"unit_id: {summary['unit_id']}")
    print(f"output_dir: {output_dir}")
    print(f"roundtrip_patch_csv: {output_dir / 'roundtrip_patches.csv'}")


if __name__ == "__main__":
    main()
