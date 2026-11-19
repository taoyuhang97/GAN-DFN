#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export sandstone Sha-3 and Sha-4 multiscale fusion VTK files."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
DEFAULT_STEP8_CSV = (
    CURRENT_DIR.parent
    / "output"
    / "formal_mine_multiscale_flow_v1_seg1"
    / "step8_well_correction"
    / "well_corrected_dfn_fracture_patches.csv"
)

_helper_path = REPO_ROOT / "common" / "layer_scale_vtk_export.py"
_spec = importlib.util.spec_from_file_location(
    "sandstone_layer_scale_vtk_export", _helper_path
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load layer-scale VTK helper: {_helper_path}")
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="导出砂砾岩沙三段、沙四段小/中/大尺度裂缝融合VTK"
    )
    p.add_argument("--step8-csv", type=Path, default=DEFAULT_STEP8_CSV)
    p.add_argument("--output-dir", type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    input_csv = args.step8_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_csv.parent / "layer_scale_fusion_vtk"
    )
    for layer_group, layer_code, filename in (
        ("沙三段", 1, "sandstone_sha3_small_medium_large.vtk"),
        ("沙四段", 2, "sandstone_sha4_small_medium_large.vtk"),
    ):
        helper.print_export_summary(
            helper.export_layer_scale_vtk(
                input_csv,
                output_dir / filename,
                layer_group=layer_group,
                layer_code=layer_code,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
