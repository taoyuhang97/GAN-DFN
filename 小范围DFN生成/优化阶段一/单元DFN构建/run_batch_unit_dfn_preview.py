from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import build_unit_dfn_preview as preview


DEFAULT_OUTPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\单元DFN构建\批量生成"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch build per-unit DFN previews; save only <output_root>/<UnitID>/... contents.")
    parser.add_argument("--package-run-dir", type=Path, default=preview.DEFAULT_PACKAGE_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=preview.DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--segy-file", type=Path, default=preview.DEFAULT_SEGY_FILE)
    parser.add_argument("--data-mode", type=str, default="all", choices=["all", "real_virtual", "real_only", "virtual_only"])
    parser.add_argument("--block-size-traces", type=int, default=preview.DEFAULT_BLOCK_SIZE_TRACES)
    parser.add_argument("--block-stride-traces", type=int, default=preview.DEFAULT_BLOCK_STRIDE_TRACES)
    parser.add_argument("--unit-id", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-vtk-export", action="store_true")
    return parser.parse_args()


def select_units(catalog_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    work = catalog_df.copy()
    work = work[work["SeedCount"].fillna(0) > 0].copy()
    if args.data_mode != "all":
        work = work[work["DataMode"].astype(str) == str(args.data_mode)].copy()
    if args.unit_id.strip():
        work = work[work["UnitID"].astype(str) == args.unit_id.strip()].copy()
    work = work.sort_values(["BlockX", "BlockY", "UnitID"]).reset_index(drop=True)
    if args.limit > 0:
        work = work.head(int(args.limit)).copy()
    return work.reset_index(drop=True)


def build_gradient_fill(
    unit_row: pd.Series,
    layers_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    trace_header_df: pd.DataFrame,
    gradient_config: preview.GradientFillConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    gradient_summary: dict[str, object] = {"Enabled": True}
    gradient_fill_df = pd.DataFrame(columns=list(seeds_df.columns) + ["GradientValue", "ClusterPointCount", "ParentSeedID", "ParentSourceKind"])
    merged_seeds_df = seeds_df.copy()
    if layers_df.empty:
        return gradient_fill_df, merged_seeds_df, gradient_summary

    seismic_cube, x_axis, y_axis, time_axis, seismic_summary = preview.load_unit_seismic_cube(
        unit_row,
        layers_df,
        trace_header_df,
        gradient_config,
    )
    gradient_voxel_df, voxel_summary = preview.detect_high_gradient_voxels(
        seismic_cube,
        x_axis,
        y_axis,
        time_axis,
        layers_df,
        gradient_config,
    )
    gradient_fill_df = preview.build_gradient_fill_seeds(
        unit_row,
        layers_df,
        seeds_df,
        gradient_voxel_df,
        gradient_config,
    )
    if not gradient_fill_df.empty:
        merged_seeds_df = pd.concat([seeds_df, gradient_fill_df], ignore_index=True, sort=False)

    gradient_summary.update(seismic_summary)
    gradient_summary.update(voxel_summary)
    gradient_summary["GradientFillSeedCount"] = int(len(gradient_fill_df))
    gradient_summary["GradientFillFromRealCount"] = int(
        gradient_fill_df["ParentSourceKind"].fillna("").astype(str).str.lower().eq("real").sum()
    ) if not gradient_fill_df.empty and "ParentSourceKind" in gradient_fill_df.columns else 0
    gradient_summary["GradientFillFromVirtualCount"] = int(
        gradient_fill_df["ParentSourceKind"].fillna("").astype(str).str.lower().eq("virtual").sum()
    ) if not gradient_fill_df.empty and "ParentSourceKind" in gradient_fill_df.columns else 0
    gradient_summary["GradientFillDirectCount"] = int(
        gradient_fill_df["ParentSourceKind"].fillna("").astype(str).str.lower().eq("layer_direct").sum()
    ) if not gradient_fill_df.empty and "ParentSourceKind" in gradient_fill_df.columns else 0
    gradient_summary["GradientFillTargetBoostCount"] = int(
        gradient_fill_df["SourceName"].fillna("").astype(str).eq("seismic_gradient_fill_target").sum()
    ) if not gradient_fill_df.empty and "SourceName" in gradient_fill_df.columns else 0
    gradient_summary["GradientFillToInputRatio"] = (
        float(len(gradient_fill_df)) / float(len(seeds_df))
    ) if len(seeds_df) > 0 else 0.0
    gradient_summary["GradientFillLayerCounts"] = (
        gradient_fill_df.groupby("GeoIntervalKey")["SeedID"].count().to_dict() if not gradient_fill_df.empty else {}
    )
    return gradient_fill_df, merged_seeds_df, gradient_summary


def process_unit(
    unit_row: pd.Series,
    package_run_dir: Path,
    output_root: Path,
    trace_header_df: pd.DataFrame,
    patch_config: preview.PatchConfig,
    gradient_config: preview.GradientFillConfig,
    vtk_config: preview.VtkExportConfig,
    block_size_traces: int,
    block_stride_traces: int,
    export_vtk: bool,
) -> dict[str, object]:
    unit_id = str(unit_row["UnitID"])
    layers_df, seeds_df, meta = preview.load_unit_package(package_run_dir, unit_id)
    unit_row = preview.derive_unit_bounds_from_block(
        unit_row,
        trace_header_df,
        block_size_traces=block_size_traces,
        block_stride_traces=block_stride_traces,
    )

    gradient_fill_df, merged_seeds_df, gradient_summary = build_gradient_fill(
        unit_row=unit_row,
        layers_df=layers_df,
        seeds_df=seeds_df,
        trace_header_df=trace_header_df,
        gradient_config=gradient_config,
    )

    patch_df = preview.build_unit_dfn_patches(unit_row, layers_df, merged_seeds_df, patch_config)
    patch_df = preview.with_sequential_index(patch_df, "PatchIndex")
    merged_seeds_output_df = preview.with_sequential_index(merged_seeds_df, "SeedIndex")
    seeds_input_output_df = preview.with_sequential_index(seeds_df, "SeedIndex")
    gradient_fill_output_df = preview.with_sequential_index(gradient_fill_df, "SeedIndex")

    summary = preview.build_patch_summary(unit_row, layers_df, merged_seeds_df, patch_df)
    summary["InputSeedCount"] = int(len(seeds_df))
    summary["GradientFillSeedCount"] = int(len(gradient_fill_df))
    summary["MergedSeedCount"] = int(len(merged_seeds_df))
    summary["GradientFillFromRealCount"] = preview.safe_int(gradient_summary.get("GradientFillFromRealCount"))
    summary["GradientFillFromVirtualCount"] = preview.safe_int(gradient_summary.get("GradientFillFromVirtualCount"))
    summary["GradientFillDirectCount"] = preview.safe_int(gradient_summary.get("GradientFillDirectCount"))
    summary["GradientFillTargetBoostCount"] = preview.safe_int(gradient_summary.get("GradientFillTargetBoostCount"))
    summary["GradientFillToInputRatio"] = float(gradient_summary.get("GradientFillToInputRatio", 0.0))
    summary["GradientSummary"] = gradient_summary

    unit_output_dir = output_root / unit_id
    unit_output_dir.mkdir(parents=True, exist_ok=True)
    preview.write_csv_utf8(patch_df, unit_output_dir / "unit_dfn_patches.csv")
    preview.write_csv_utf8(seeds_input_output_df, unit_output_dir / "fracture_seeds_input.csv")
    preview.write_csv_utf8(gradient_fill_output_df, unit_output_dir / "gradient_fill_seeds.csv")
    preview.write_csv_utf8(merged_seeds_output_df, unit_output_dir / "fracture_seeds_merged.csv")
    preview.write_csv_utf8(layers_df, unit_output_dir / "unit_layers_input.csv")
    if not patch_df.empty:
        preview.save_preview_figure(unit_row, layers_df, patch_df, unit_output_dir / "unit_dfn_preview.png")

    vtk_export_summary: dict[str, object] = {"Enabled": export_vtk}
    if export_vtk:
        vtk_export_summary.update(preview.export_patch_vtk_files(patch_df, unit_output_dir, vtk_config))
        vtk_export_summary.update(preview.export_seed_vtk_files(merged_seeds_output_df, unit_output_dir, vtk_config))
        mapping_path = preview.write_vtk_mapping_json(unit_output_dir, vtk_export_summary, vtk_config)
        quickstart_path = preview.write_paraview_quickstart(unit_output_dir, unit_id, vtk_config)
        vtk_export_summary["mapping_json"] = str(mapping_path)
        vtk_export_summary["quickstart_md"] = str(quickstart_path)

    summary["VtkExport"] = vtk_export_summary
    (unit_output_dir / "unit_dfn_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (unit_output_dir / "unit_gradient_summary.json").write_text(json.dumps(gradient_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "UnitID": unit_id,
        "InputSeedCount": int(len(seeds_df)),
        "GradientFillSeedCount": int(len(gradient_fill_df)),
        "GradientFillToInputRatio": float(gradient_summary.get("GradientFillToInputRatio", 0.0)),
        "MergedSeedCount": int(len(merged_seeds_df)),
        "PatchCount": int(len(patch_df)),
        "Meta": meta,
    }


def main() -> int:
    args = parse_args()
    package_run_dir = args.package_run_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    catalog_df = preview.load_unit_catalog(package_run_dir)
    target_units_df = select_units(catalog_df, args)
    if target_units_df.empty:
        raise ValueError("No units matched the current batch filters.")

    patch_config = preview.PatchConfig()
    gradient_config = preview.GradientFillConfig(
        trace_header_csv=args.trace_header_csv.resolve(),
        segy_file=args.segy_file.resolve(),
    )
    vtk_config = preview.VtkExportConfig()
    trace_header_df = preview.load_trace_header(gradient_config.trace_header_csv)

    completed_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    total = len(target_units_df)
    start_timestamp = datetime.now()
    print(f"[batch] start units={total} output_root={output_root}")

    for index, (_, unit_row) in enumerate(target_units_df.iterrows(), start=1):
        unit_id = str(unit_row["UnitID"])
        unit_output_dir = output_root / unit_id
        summary_path = unit_output_dir / "unit_dfn_summary.json"
        if summary_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] skip {unit_id} existing")
            continue
        try:
            result = process_unit(
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
            completed_rows.append(result)
            print(
                f"[{index}/{total}] done {unit_id} "
                f"input={result['InputSeedCount']} fill={result['GradientFillSeedCount']} "
                f"ratio={result['GradientFillToInputRatio']:.3f} merged={result['MergedSeedCount']}"
            )
        except Exception as exc:
            failed_rows.append({"UnitID": unit_id, "Error": str(exc)})
            print(f"[{index}/{total}] fail {unit_id} error={exc}")

    elapsed = datetime.now() - start_timestamp
    completed_count = len(completed_rows)
    failed_count = len(failed_rows)
    skipped_count = total - completed_count - failed_count
    avg_ratio = float(pd.DataFrame(completed_rows)["GradientFillToInputRatio"].mean()) if completed_rows else 0.0
    print(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "output_root": str(output_root),
                "unit_total": total,
                "completed": completed_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "avg_gradient_fill_to_input_ratio": avg_ratio,
                "elapsed_seconds": int(elapsed.total_seconds()),
                "failed_units": failed_rows[:20],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
