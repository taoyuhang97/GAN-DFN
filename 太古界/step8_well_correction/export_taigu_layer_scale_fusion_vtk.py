#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Taigu upper-composite and weathering-crust fusion VTK files."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
TAIGU_ROOT = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[1]
DEFAULT_STEP8_CSV = (
    CURRENT_DIR
    / "output"
    / "taigu_step8_attribute_multiscale_v1_quanqu1_seg1"
    / "well_corrected_dfn_fracture_patches.csv"
)

_helper_path = REPO_ROOT / "common" / "layer_scale_vtk_export.py"
_spec = importlib.util.spec_from_file_location(
    "taigu_layer_scale_vtk_export", _helper_path
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load layer-scale VTK helper: {_helper_path}")
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="导出太古界上部复合层、风化壳小/中/大尺度裂缝融合VTK"
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
        ("上部复合层", 1, "taigu_upper_composite_small_medium_large.vtk"),
        ("太古界风化壳", 2, "taigu_weathering_crust_small_medium_large.vtk"),
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
