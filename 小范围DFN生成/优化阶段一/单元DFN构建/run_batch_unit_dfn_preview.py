from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import build_unit_dfn_preview as preview


DEFAULT_OUTPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成"
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


def build_patch_config(args: argparse.Namespace) -> preview.PatchConfig:
    return preview.PatchConfig(
        point_base_length=args.point_base_length,
        point_base_height=args.point_base_height,
        segment_base_length=args.segment_base_length,
        segment_base_height=args.segment_base_height,
        density_gain=args.density_gain,
        segment_length_gain=args.segment_length_gain,
        virtual_point_scale=args.virtual_point_size_scale,
        virtual_segment_scale=args.virtual_segment_size_scale,
        gradient_fill_scale=args.gradient_fill_size_scale,
    )


def build_gradient_config(args: argparse.Namespace) -> preview.GradientFillConfig:
    layer_intensity_profile = preview.get_layer_intensity_profile(args.layer_intensity_profile)
    return preview.GradientFillConfig(
        trace_header_csv=args.trace_header_csv.resolve(),
        segy_file=args.segy_file.resolve(),
        time_padding_ms=args.gradient_fill_time_padding_ms,
        gradient_threshold=args.gradient_fill_threshold,
        layer_threshold_mode=args.gradient_threshold_mode,
        layer_gradient_quantile=args.gradient_layer_quantile,
        layer_threshold_floor=args.gradient_layer_threshold_floor,
        enable_multiscale_gradient=not args.disable_multiscale_gradient,
        multiscale_sigma_levels=preview.parse_float_tuple(args.gradient_multiscale_sigmas, (0.0, 1.0, 2.0)),
        multiscale_time_sigma_scale=args.gradient_multiscale_time_scale,
        multiscale_combine_mode=args.gradient_multiscale_combine,
        expansion_radius_m=args.gradient_fill_radius_m,
        vertical_radius_ms=args.gradient_fill_vertical_ms,
        dbscan_eps_xy_m=args.gradient_fill_dbscan_eps_xy,
        dbscan_eps_time_ms=args.gradient_fill_dbscan_eps_time,
        min_cluster_samples=args.gradient_fill_min_samples,
        max_clusters_per_seed=args.gradient_fill_max_clusters_per_seed,
        max_candidate_voxels_per_seed=args.gradient_fill_max_candidate_voxels_per_seed,
        fill_density_scale=args.gradient_fill_density_scale,
        fill_length_scale=args.gradient_fill_length_scale,
        virtual_only_threshold_bonus=args.virtual_only_gradient_threshold_bonus,
        virtual_only_radius_scale=args.virtual_only_gradient_radius_scale,
        virtual_only_vertical_scale=args.virtual_only_gradient_vertical_scale,
        virtual_only_max_clusters_per_seed=args.virtual_only_gradient_max_clusters_per_seed,
        enable_seedless_layer_fill=not args.disable_seedless_layer_fill,
        seedless_layer_max_clusters_per_layer=args.seedless_layer_max_clusters_per_layer,
        seedless_layer_min_cluster_samples=args.seedless_layer_min_samples,
        seedless_layer_min_voxels=args.seedless_layer_min_voxels,
        seedless_layer_max_candidate_voxels=args.seedless_layer_max_candidate_voxels,
        seed_proximity_exclusion_xy_m=args.seed_proximity_exclusion_xy_m,
        seed_proximity_exclusion_time_ms=args.seed_proximity_exclusion_time_ms,
        dedup_xy_m=args.dedup_xy_m,
        dedup_time_ms=args.dedup_time_ms,
        dedup_azimuth_deg=args.dedup_azimuth_deg,
        dedup_dip_deg=args.dedup_dip_deg,
        enable_target_fill_ratio=not args.disable_target_fill_ratio,
        target_fill_to_input_ratio=args.target_fill_to_input_ratio,
        target_fill_layer_weighted=not args.disable_target_fill_layer_weighted,
        enable_layer_intensity_scaling=not args.disable_layer_intensity_scaling,
        layer_intensity_profile_name=str(args.layer_intensity_profile),
        layer_intensity_default=max(0.0, float(args.layer_intensity_default)),
        layer_intensity_factors=layer_intensity_profile,
        target_fill_max_candidate_voxels_per_layer=args.target_fill_max_candidate_voxels_per_layer,
    )


def build_vtk_config(args: argparse.Namespace) -> preview.VtkExportConfig:
    return preview.VtkExportConfig(
        display_z_scale=args.vtk_display_z_scale,
        invert_time=not args.vtk_no_invert_time,
    )


def to_json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    return value


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
    gradient_summary.update(preview.build_layer_intensity_summary(gradient_config))
    gradient_summary["LayerFillPlan"] = preview.build_layer_fill_plan(layers_df, seeds_df, gradient_fill_df, gradient_config).to_dict(orient="records")
    return gradient_fill_df, merged_seeds_df, gradient_summary


def process_unit(
    unit_row: pd.Series,
    package_run_dir: Path,
    output_root: Path,
    trace_header_df: pd.DataFrame,
    patch_config: preview.PatchConfig,
    gradient_config: preview.GradientFillConfig | None,
    vtk_config: preview.VtkExportConfig,
    block_size_traces: int,
    block_stride_traces: int,
    export_vtk: bool,
) -> dict[str, object]:
    unit_id = str(unit_row["UnitID"])
    layers_df, seeds_df, meta = preview.load_unit_package(package_run_dir, unit_id)
    seeds_df = preview.assign_seed_layers_by_time(seeds_df, layers_df)
    unit_row = preview.derive_unit_bounds_from_block(
        unit_row,
        trace_header_df,
        block_size_traces=block_size_traces,
        block_stride_traces=block_stride_traces,
    )

    gradient_summary: dict[str, object] = {"Enabled": gradient_config is not None}
    gradient_fill_df = pd.DataFrame(columns=list(seeds_df.columns) + ["GradientValue", "ClusterPointCount", "ParentSeedID", "ParentSourceKind"])
    merged_seeds_df = seeds_df.copy()
    if gradient_config is not None:
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
        "DataMode": str(unit_row.get("DataMode", "")),
        "LayerCount": int(len(layers_df)),
        "RealSeedCount": preview.safe_int(unit_row.get("RealSeedCount")),
        "VirtualSeedCount": preview.safe_int(unit_row.get("VirtualSeedCount")),
        "InputSeedCount": int(len(seeds_df)),
        "GradientFillSeedCount": int(len(gradient_fill_df)),
        "GradientFillToInputRatio": float(gradient_summary.get("GradientFillToInputRatio", 0.0)),
        "MergedSeedCount": int(len(merged_seeds_df)),
        "PatchCount": int(len(patch_df)),
        "GradientFillTargetBoostCount": preview.safe_int(gradient_summary.get("GradientFillTargetBoostCount")),
        "GradientFillDirectCount": preview.safe_int(gradient_summary.get("GradientFillDirectCount")),
        "LayerFillPlan": gradient_summary.get("LayerFillPlan", []),
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

    patch_config = build_patch_config(args)
    gradient_config = None if args.disable_gradient_fill else build_gradient_config(args)
    vtk_config = build_vtk_config(args)
    trace_header_path = args.trace_header_csv.resolve()
    trace_header_df = preview.load_trace_header(trace_header_path)
    preview.write_csv_utf8(target_units_df, output_root / "selected_unit_catalog.csv")

    completed_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    layer_plan_rows: list[dict[str, object]] = []
    total = len(target_units_df)
    start_timestamp = datetime.now()
    print(f"[batch] start units={total} output_root={output_root}")

    for index, (_, unit_row) in enumerate(target_units_df.iterrows(), start=1):
        unit_id = str(unit_row["UnitID"])
        unit_output_dir = output_root / unit_id
        summary_path = unit_output_dir / "unit_dfn_summary.json"
        if summary_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] skip {unit_id} existing")
            skipped_rows.append({"UnitID": unit_id, "Reason": "existing_summary", "SummaryPath": str(summary_path)})
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
            for layer_row in result.get("LayerFillPlan", []) or []:
                layer_plan_rows.append(
                    {
                        "UnitID": unit_id,
                        "DataMode": result.get("DataMode", ""),
                        "GeoIntervalKey": layer_row.get("GeoIntervalKey", ""),
                        "StrataName": layer_row.get("StrataName", ""),
                        "TopSurfaceCode": layer_row.get("TopSurfaceCode", ""),
                        "BaseSurfaceCode": layer_row.get("BaseSurfaceCode", ""),
                        "LayerSurfacePairKey": layer_row.get("LayerSurfacePairKey", ""),
                        "LayerIntensityFactor": layer_row.get("LayerIntensityFactor", 1.0),
                        "BaseTargetRatio": layer_row.get("BaseTargetRatio", 0.0),
                        "AdjustedTargetRatio": layer_row.get("AdjustedTargetRatio", 0.0),
                        "InputSeedCount": layer_row.get("InputSeedCount", 0),
                        "CurrentFillCount": layer_row.get("CurrentFillCount", 0),
                        "BaseTargetFillCount": layer_row.get("BaseTargetFillCount", 0),
                        "AdjustedTargetFillCount": layer_row.get("AdjustedTargetFillCount", 0),
                        "TargetFillCount": layer_row.get("TargetFillCount", 0),
                        "RemainingFillDeficit": layer_row.get("RemainingFillDeficit", 0),
                    }
                )
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
    skipped_count = len(skipped_rows)
    avg_ratio = float(pd.DataFrame(completed_rows)["GradientFillToInputRatio"].mean()) if completed_rows else 0.0
    completed_df = pd.DataFrame(completed_rows)
    failed_df = pd.DataFrame(failed_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    layer_plan_df = pd.DataFrame(layer_plan_rows)
    if not completed_df.empty:
        completed_df = completed_df.drop(columns=["LayerFillPlan", "Meta"], errors="ignore")
        preview.write_csv_utf8(completed_df, output_root / "batch_completed_summary.csv")
    if not failed_df.empty:
        preview.write_csv_utf8(failed_df, output_root / "batch_failed_units.csv")
    if not skipped_df.empty:
        preview.write_csv_utf8(skipped_df, output_root / "batch_skipped_units.csv")
    if not layer_plan_df.empty:
        preview.write_csv_utf8(layer_plan_df, output_root / "batch_layer_fill_plan.csv")

    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "package_run_dir": str(package_run_dir),
        "output_root": str(output_root),
        "trace_header_csv": str(trace_header_path),
        "segy_file": str(args.segy_file.resolve()),
        "data_mode_filter": str(args.data_mode),
        "unit_id_filter": str(args.unit_id),
        "limit": int(args.limit),
        "overwrite": bool(args.overwrite),
        "gradient_fill_enabled": not args.disable_gradient_fill,
        "vtk_export_enabled": not args.disable_vtk_export,
        "unit_total": total,
        "completed": completed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "avg_gradient_fill_to_input_ratio": avg_ratio,
        "elapsed_seconds": int(elapsed.total_seconds()),
        "patch_config": to_json_ready(asdict(patch_config)),
        "gradient_config": to_json_ready(asdict(gradient_config)) if gradient_config is not None else None,
        "vtk_config": to_json_ready(asdict(vtk_config)),
        "failed_units": failed_rows[:20],
        "skipped_units": skipped_rows[:20],
    }
    (output_root / "batch_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
