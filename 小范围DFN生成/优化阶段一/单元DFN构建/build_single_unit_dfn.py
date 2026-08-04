from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

import build_unit_dfn_preview as preview
import run_batch_unit_dfn_preview as batch


DEFAULT_OUTPUT_ROOT = Path(
    "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成_新层位重拆分"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one stage-2 unit DFN with the current layer-constrained seismic fracture-fill logic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--unit-id", type=str, required=True, help="Unit ID, for example BX49_BY5.")
    parser.add_argument("--package-run-dir", type=Path, default=preview.DEFAULT_PACKAGE_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=preview.DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--segy-file", type=Path, default=preview.DEFAULT_SEGY_FILE)
    parser.add_argument("--data-mode", type=str, default="all", choices=["all", "real_virtual", "real_only", "virtual_only"])
    parser.add_argument("--block-size-traces", type=int, default=preview.DEFAULT_BLOCK_SIZE_TRACES)
    parser.add_argument("--block-stride-traces", type=int, default=preview.DEFAULT_BLOCK_STRIDE_TRACES)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing unit_dfn_summary.json for this unit.")

    parser.add_argument("--point-base-length", type=float, default=18.0)
    parser.add_argument("--point-base-height", type=float, default=5.0)
    parser.add_argument("--segment-base-length", type=float, default=26.0)
    parser.add_argument("--segment-base-height", type=float, default=7.0)
    parser.add_argument("--density-gain", type=float, default=0.35)
    parser.add_argument("--segment-length-gain", type=float, default=1.1)
    parser.add_argument("--virtual-point-size-scale", type=float, default=0.78)
    parser.add_argument("--virtual-segment-size-scale", type=float, default=0.88)
    parser.add_argument("--gradient-fill-size-scale", type=float, default=0.72)

    parser.add_argument("--gradient-fill-threshold", type=float, default=0.65)
    parser.add_argument("--gradient-threshold-mode", type=str, default="hybrid", choices=["global", "layer_quantile", "hybrid"])
    parser.add_argument("--gradient-layer-quantile", type=float, default=0.985)
    parser.add_argument("--gradient-layer-threshold-floor", type=float, default=0.60)
    parser.add_argument("--disable-multiscale-gradient", action="store_true")
    parser.add_argument("--gradient-multiscale-sigmas", type=str, default="0,1,2")
    parser.add_argument("--gradient-multiscale-time-scale", type=float, default=1.5)
    parser.add_argument("--gradient-multiscale-combine", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--gradient-fill-radius-m", type=float, default=180.0)
    parser.add_argument("--gradient-fill-vertical-ms", type=float, default=120.0)
    parser.add_argument("--gradient-fill-time-padding-ms", type=float, default=20.0)
    parser.add_argument("--gradient-fill-dbscan-eps-xy", type=float, default=35.0)
    parser.add_argument("--gradient-fill-dbscan-eps-time", type=float, default=14.0)
    parser.add_argument("--gradient-fill-min-samples", type=int, default=2)
    parser.add_argument("--gradient-fill-max-clusters-per-seed", type=int, default=3)
    parser.add_argument("--gradient-fill-max-candidate-voxels-per-seed", type=int, default=450)
    parser.add_argument("--gradient-fill-density-scale", type=float, default=0.82)
    parser.add_argument("--gradient-fill-length-scale", type=float, default=0.90)
    parser.add_argument("--virtual-only-gradient-threshold-bonus", type=float, default=0.0)
    parser.add_argument("--virtual-only-gradient-radius-scale", type=float, default=1.0)
    parser.add_argument("--virtual-only-gradient-vertical-scale", type=float, default=1.0)
    parser.add_argument("--virtual-only-gradient-max-clusters-per-seed", type=int, default=2)

    parser.add_argument("--disable-seedless-layer-fill", action="store_true")
    parser.add_argument("--seedless-layer-max-clusters-per-layer", type=int, default=4)
    parser.add_argument("--seedless-layer-min-samples", type=int, default=2)
    parser.add_argument("--seedless-layer-min-voxels", type=int, default=12)
    parser.add_argument("--seedless-layer-max-candidate-voxels", type=int, default=900)
    parser.add_argument("--seed-proximity-exclusion-xy-m", type=float, default=10.0)
    parser.add_argument("--seed-proximity-exclusion-time-ms", type=float, default=4.0)
    parser.add_argument("--dedup-xy-m", type=float, default=12.5)
    parser.add_argument("--dedup-time-ms", type=float, default=4.0)
    parser.add_argument("--dedup-azimuth-deg", type=float, default=10.0)
    parser.add_argument("--dedup-dip-deg", type=float, default=5.0)

    parser.add_argument("--disable-target-fill-ratio", action="store_true")
    parser.add_argument("--target-fill-to-input-ratio", type=float, default=1.5)
    parser.add_argument("--disable-target-fill-layer-weighted", action="store_true")
    parser.add_argument("--disable-layer-intensity-scaling", action="store_true")
    parser.add_argument(
        "--layer-intensity-profile",
        type=str,
        default=preview.DEFAULT_LAYER_INTENSITY_PROFILE_NAME,
        choices=preview.available_layer_intensity_profiles(),
    )
    parser.add_argument("--layer-intensity-default", type=float, default=1.0)
    parser.add_argument("--target-fill-max-candidate-voxels-per-layer", type=int, default=4000)

    parser.add_argument("--disable-gradient-fill", action="store_true")
    parser.add_argument("--disable-vtk-export", action="store_true")
    parser.add_argument("--vtk-display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-no-invert-time", action="store_true")
    return parser


def select_single_unit(catalog_df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    unit_id = str(args.unit_id).strip()
    if not unit_id:
        raise ValueError("--unit-id cannot be empty")
    matched = catalog_df[catalog_df["UnitID"].astype(str) == unit_id].copy()
    if matched.empty:
        raise ValueError(f"Unit not found in unit_catalog.csv: {unit_id}")
    if str(args.data_mode) != "all":
        matched = matched[matched["DataMode"].astype(str) == str(args.data_mode)].copy()
        if matched.empty:
            raise ValueError(f"Unit {unit_id} does not match requested data mode: {args.data_mode}")
    return matched.iloc[0]


def main() -> int:
    args = build_parser().parse_args()
    start_time = datetime.now()
    package_run_dir = args.package_run_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    catalog_df = preview.load_unit_catalog(package_run_dir)
    unit_row = select_single_unit(catalog_df, args)
    unit_id = str(unit_row["UnitID"])
    unit_output_dir = output_root / unit_id
    summary_path = unit_output_dir / "unit_dfn_summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"[single-unit] skip existing unit={unit_id} summary={summary_path}", flush=True)
        return 0

    print(f"[single-unit] start unit={unit_id}", flush=True)
    print(f"[single-unit] package_run_dir={package_run_dir}", flush=True)
    print(f"[single-unit] output_root={output_root}", flush=True)
    trace_header_df = preview.load_trace_header(args.trace_header_csv.resolve())
    patch_config = batch.build_patch_config(args)
    gradient_config = None if args.disable_gradient_fill else batch.build_gradient_config(args)
    vtk_config = batch.build_vtk_config(args)

    result = batch.process_unit(
        unit_row=unit_row.copy(),
        package_run_dir=package_run_dir,
        output_root=output_root,
        trace_header_df=trace_header_df,
        patch_config=patch_config,
        gradient_config=gradient_config,
        vtk_config=vtk_config,
        block_size_traces=int(args.block_size_traces),
        block_stride_traces=int(args.block_stride_traces),
        export_vtk=not args.disable_vtk_export,
    )

    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": int((datetime.now() - start_time).total_seconds()),
        "unit_id": unit_id,
        "package_run_dir": str(package_run_dir),
        "output_root": str(output_root),
        "unit_output_dir": str(unit_output_dir),
        "trace_header_csv": str(args.trace_header_csv.resolve()),
        "segy_file": str(args.segy_file.resolve()),
        "gradient_fill_enabled": not args.disable_gradient_fill,
        "vtk_export_enabled": not args.disable_vtk_export,
        "overwrite": bool(args.overwrite),
        "result": {key: value for key, value in result.items() if key not in {"LayerFillPlan", "Meta"}},
        "patch_config": batch.to_json_ready(asdict(patch_config)),
        "gradient_config": batch.to_json_ready(asdict(gradient_config)) if gradient_config is not None else None,
        "vtk_config": batch.to_json_ready(asdict(vtk_config)),
    }
    (unit_output_dir / "single_unit_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
