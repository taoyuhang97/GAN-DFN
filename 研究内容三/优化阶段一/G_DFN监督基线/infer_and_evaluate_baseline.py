# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from baseline_common import (
    DEFAULT_LAYER_DENSITY_SOURCE,
    LAYER_DENSITY_SOURCE_CHOICES,
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SLOTS_PER_VOXEL,
    DEFAULT_UNIT_DFN_ROOT,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_Z_STEP_MS,
    VtkPatchExportConfig,
    apply_layer_density_budget,
    append_lines_to_docx,
    build_unit_layer_segment_table,
    build_window_grid,
    compute_file_sha256,
    decode_window_predictions_with_optional_second_pass,
    dedupe_patch_df,
    load_layer_density_calibration_payload,
    load_sample_manifest,
    prepare_unit_context,
    read_csv_utf8,
    resolve_layer_density_calibration_bounds,
    write_csv_utf8,
    write_json,
    export_patch_comparison_vtk,
    export_patch_vtk_files,
    build_patch_match_metrics,
    scalar_float,
    scalar_int,
)
from baseline_model import SparseInstanceBaselineUNet
from layer_model_registry import (
    LAYER_SURFACE_PAIR_KEY_COL,
    UNIT_LAYER_SEGMENT_KEY_COL,
    build_layer_surface_pair_key_from_row,
    ensure_layer_surface_pair_key_column,
    ensure_unit_layer_segment_key_column,
    build_unit_layer_segment_key_from_row,
    load_layer_model_registry,
    resolve_layer_decode_config,
    resolve_layer_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 G-DFN 监督 baseline 做窗口预测、单元回拼与评估。")
    parser.add_argument("--split-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer-model-registry-py", type=Path)
    parser.add_argument("--layer-surface-pair-key", type=str)
    parser.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "推理评估")
    parser.add_argument("--run-name", type=str, default=f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--center-threshold", type=float, default=0.7)
    parser.add_argument("--decode-mode", type=str, default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--relaxed-min-count", type=int, default=1)
    parser.add_argument("--count-activation-threshold", type=float, default=0.5)
    parser.add_argument("--min-count-if-active", type=int, default=1)
    parser.add_argument("--max-slots-per-voxel", type=int, default=DEFAULT_SLOTS_PER_VOXEL)
    parser.add_argument("--max-total-patches-per-window", type=int, default=256)
    parser.add_argument("--dedupe-xy-tol-m", type=float, default=6.25)
    parser.add_argument("--dedupe-time-tol-ms", type=float, default=0.4)
    parser.add_argument("--dedupe-azimuth-tol-deg", type=float, default=20.0)
    parser.add_argument("--dedupe-dip-tol-deg", type=float, default=12.0)
    parser.add_argument("--disable-layer-density-control", action="store_true")
    parser.add_argument("--layer-density-source", type=str, default=DEFAULT_LAYER_DENSITY_SOURCE, choices=list(LAYER_DENSITY_SOURCE_CHOICES))
    parser.add_argument("--layer-density-scale", type=float, default=1.0)
    parser.add_argument("--layer-density-calibration-json", type=Path)
    parser.add_argument("--layer-density-calibration-min-scale", type=float, default=0.25)
    parser.add_argument("--layer-density-calibration-max-scale", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-units", type=int)
    parser.add_argument("--display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-no-invert-time", action="store_true")
    parser.add_argument("--export-window-csv", action="store_true")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[SparseInstanceBaselineUNet, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = SparseInstanceBaselineUNet(
        in_channels=int(model_config["in_channels"]),
        slots_per_voxel=int(model_config["slots_per_voxel"]),
        base_channels=int(model_config["base_channels"]),
        dx=float(model_config.get("dx", 12.5)),
        dy=float(model_config.get("dy", 12.5)),
        dz=float(model_config.get("dz", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def run_single_sample(
    model: SparseInstanceBaselineUNet,
    sample_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(sample_path, allow_pickle=True)
    input_features = torch.from_numpy(np.asarray(data["input_features"], dtype=np.float32)).unsqueeze(0).to(device)
    valid_z_mask = np.asarray(data["valid_z_mask"], dtype=np.float32)
    with torch.no_grad():
        outputs = model(input_features)
        center_probs = torch.sigmoid(outputs["center_logits"]).squeeze(0).detach().cpu().numpy()
        count_pred = outputs["count_pred"].squeeze(0).detach().cpu().numpy()
        geom_pred = outputs["geom_pred"].squeeze(0).detach().cpu().numpy()
    return center_probs, count_pred, geom_pred, valid_z_mask


def resolve_decode_runtime_config(
    registry_payload: dict[str, Any] | None,
    layer_surface_pair_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    layer_config = resolve_layer_decode_config(registry_payload, layer_surface_pair_key)
    decode_mode = str(layer_config.get("decode_mode", args.decode_mode)).strip().lower()
    if decode_mode not in {"strict", "relaxed"}:
        decode_mode = str(args.decode_mode)
    return {
        "center_threshold": float(layer_config.get("center_threshold", args.center_threshold)),
        "decode_mode": decode_mode,
        "relaxed_min_count": int(layer_config.get("relaxed_min_count", args.relaxed_min_count)),
        "count_activation_threshold": float(
            layer_config.get("count_activation_threshold", args.count_activation_threshold)
        ),
        "min_count_if_active": int(layer_config.get("min_count_if_active", args.min_count_if_active)),
        "max_total_patches_per_window": int(
            layer_config.get("max_total_patches_per_window", args.max_total_patches_per_window)
        ),
        "second_pass": layer_config.get("second_pass", {}),
    }


def compute_activation_score_threshold(
    activation_scores: pd.Series,
    target_active_count: int,
) -> float | None:
    score_values = pd.to_numeric(activation_scores, errors="coerce").to_numpy(dtype=float)
    score_values = score_values[np.isfinite(score_values)]
    if score_values.size <= 0:
        return None
    sorted_scores = np.sort(score_values)[::-1]
    desired_count = int(np.clip(int(target_active_count), 0, int(sorted_scores.size)))
    if desired_count <= 0:
        max_score = float(sorted_scores[0])
        return float(max_score + max(abs(max_score) * 1e-6, 1e-9))
    if desired_count >= int(sorted_scores.size):
        return 0.0
    upper_score = float(sorted_scores[desired_count - 1])
    lower_score = float(sorted_scores[desired_count])
    if upper_score > lower_score:
        return float((upper_score + lower_score) * 0.5)
    return upper_score


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    if args.checkpoint is None and args.layer_model_registry_py is None:
        raise ValueError("please provide --checkpoint or --layer-model-registry-py")
    layer_surface_pair_key = str(args.layer_surface_pair_key).strip() if args.layer_surface_pair_key else ""
    registry_payload = load_layer_model_registry(Path(args.layer_model_registry_py)) if args.layer_model_registry_py else None
    layer_density_calibration_payload = load_layer_density_calibration_payload(args.layer_density_calibration_json)
    model_cache: dict[str, tuple[SparseInstanceBaselineUNet, dict[str, Any]]] = {}

    def get_or_load_model(checkpoint_path: Path) -> tuple[SparseInstanceBaselineUNet, dict[str, Any]]:
        resolved_path = str(Path(checkpoint_path).resolve())
        if resolved_path not in model_cache:
            model_cache[resolved_path] = load_checkpoint_model(Path(resolved_path), device)
        return model_cache[resolved_path]

    split_manifest_csv = Path(args.split_run_dir) / f"{args.split_name}_manifest.csv"
    if not split_manifest_csv.exists():
        raise FileNotFoundError(f"split manifest not found: {split_manifest_csv}")
    split_manifest_sha256 = compute_file_sha256(split_manifest_csv)
    layer_model_registry_sha256 = compute_file_sha256(args.layer_model_registry_py) if args.layer_model_registry_py else ""
    input_layer_density_calibration_json_sha256 = compute_file_sha256(args.layer_density_calibration_json)
    manifest_df = ensure_unit_layer_segment_key_column(
        ensure_layer_surface_pair_key_column(read_csv_utf8(split_manifest_csv), key_col=LAYER_SURFACE_PAIR_KEY_COL),
        key_col=UNIT_LAYER_SEGMENT_KEY_COL,
        pair_key_col=LAYER_SURFACE_PAIR_KEY_COL,
    )
    if layer_surface_pair_key:
        manifest_df = manifest_df[manifest_df[LAYER_SURFACE_PAIR_KEY_COL].astype(str) == layer_surface_pair_key].copy()
    if args.limit_samples:
        manifest_df = manifest_df.head(int(args.limit_samples)).copy()
    if args.limit_units:
        keep_units = manifest_df["UnitID"].astype(str).drop_duplicates().head(int(args.limit_units)).tolist()
        manifest_df = manifest_df[manifest_df["UnitID"].astype(str).isin(keep_units)].copy()
    manifest_df = manifest_df.reset_index(drop=True)
    if manifest_df.empty:
        raise ValueError("evaluation manifest is empty")
    print(
        (
            "[eval] start window inference | "
            f"split={args.split_name}, total_windows={len(manifest_df)}, "
            f"approx_unit_count={manifest_df['UnitID'].astype(str).nunique()}, device={device}"
        ),
        flush=True,
    )

    run_dir = Path(args.output_root) / args.run_name
    units_dir = run_dir / "units"
    aggregated_dir = run_dir / "aggregated"
    units_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    unit_context_cache: dict[str, dict[str, Any]] = {}
    predicted_window_rows: list[dict[str, Any]] = []
    per_unit_patch_frames: dict[str, list[pd.DataFrame]] = {}
    unit_layer_rows: list[dict[str, Any]] = []

    window_progress = tqdm(
        manifest_df.iterrows(),
        total=len(manifest_df),
        desc="Eval windows",
        dynamic_ncols=True,
    )
    for _, row in window_progress:
        unit_id = str(row["UnitID"])
        if unit_id not in unit_context_cache:
            unit_context_cache[unit_id] = prepare_unit_context(unit_id=unit_id, unit_dfn_root=args.unit_dfn_root)
        unit_context = unit_context_cache[unit_id]

        sample_path = Path(str(row["PackagePath"]))
        layer_key = build_layer_surface_pair_key_from_row(row)
        checkpoint_path = (
            resolve_layer_checkpoint(
                registry_payload=registry_payload,
                layer_surface_pair_key=layer_key,
                fallback_checkpoint=args.checkpoint,
            )
            if registry_payload is not None
            else str(Path(args.checkpoint).resolve())
        )
        decode_runtime_config = resolve_decode_runtime_config(
            registry_payload=registry_payload,
            layer_surface_pair_key=layer_key,
            args=args,
        )
        model, checkpoint = get_or_load_model(Path(checkpoint_path))
        center_probs, count_pred, geom_pred, valid_z_mask = run_single_sample(model, sample_path, device)
        window_grid = build_window_grid(
            unit_context=unit_context,
            window_top=float(row["WindowTopTime"]),
            window_size=DEFAULT_WINDOW_SIZE,
            z_step_ms=DEFAULT_Z_STEP_MS,
        )
        window_patch_df, decoded_df, decode_summary, decode_filter_stats = decode_window_predictions_with_optional_second_pass(
            center_probs=center_probs,
            count_pred=count_pred,
            geom_pred=geom_pred,
            valid_z_mask=valid_z_mask,
            grid=window_grid,
            layers_df=unit_context["layers_df"],
            decode_runtime_config=decode_runtime_config,
            max_slots_per_voxel=int(args.max_slots_per_voxel),
            dedupe_xy_tol_m=float(args.dedupe_xy_tol_m),
            dedupe_time_tol_ms=float(args.dedupe_time_tol_ms),
            dedupe_azimuth_tol_deg=float(args.dedupe_azimuth_tol_deg),
            dedupe_dip_tol_deg=float(args.dedupe_dip_tol_deg),
            layer_surface_pair_key=layer_key,
            calibration_payload=layer_density_calibration_payload,
        )
        unit_layer_segment_key = build_unit_layer_segment_key_from_row(row)
        if not window_patch_df.empty:
            if LAYER_SURFACE_PAIR_KEY_COL not in window_patch_df.columns:
                window_patch_df[LAYER_SURFACE_PAIR_KEY_COL] = ""
            pair_fill_mask = window_patch_df[LAYER_SURFACE_PAIR_KEY_COL].fillna("").astype(str).str.strip().eq("")
            if pair_fill_mask.any():
                window_patch_df.loc[pair_fill_mask, LAYER_SURFACE_PAIR_KEY_COL] = layer_key
            if UNIT_LAYER_SEGMENT_KEY_COL not in window_patch_df.columns:
                window_patch_df[UNIT_LAYER_SEGMENT_KEY_COL] = ""
            segment_fill_mask = window_patch_df[UNIT_LAYER_SEGMENT_KEY_COL].fillna("").astype(str).str.strip().eq("")
            if segment_fill_mask.any():
                window_patch_df.loc[segment_fill_mask, UNIT_LAYER_SEGMENT_KEY_COL] = unit_layer_segment_key
        if not window_patch_df.empty:
            window_patch_df["PredWindowIndex"] = int(row["WindowIndex"])
            window_patch_df["PredWindowTopTime"] = float(row["WindowTopTime"])
            window_patch_df["PredWindowBaseTime"] = float(row["WindowBaseTime"])
            window_patch_df["PredSampleID"] = str(row["SampleID"])
        per_unit_patch_frames.setdefault(unit_id, []).append(window_patch_df)
        predicted_window_rows.append(
            {
                "SampleID": str(row["SampleID"]),
                "UnitID": unit_id,
                "GeoIntervalKey": str(row["GeoIntervalKey"]),
                LAYER_SURFACE_PAIR_KEY_COL: layer_key,
                UNIT_LAYER_SEGMENT_KEY_COL: unit_layer_segment_key,
                "CheckpointPath": str(checkpoint_path),
                "WindowIndex": int(row["WindowIndex"]),
                "PredictedPatchCount": int(len(window_patch_df)),
                "DecodedActiveSlotCount": int(decode_summary["active_slot_count"]),
                "RawCandidateCount": int(decode_filter_stats["raw_candidate_count"]),
                "KeptCandidateCount": int(decode_filter_stats["kept_candidate_count"]),
                "MeanPredCount": float(decode_filter_stats["mean_pred_count"]),
                "MaxPredCount": int(decode_filter_stats["max_pred_count"]),
                "ForcedVoxelCount": int(decode_filter_stats["forced_voxel_count"]),
                "DecodeMode": str(decode_filter_stats["decode_mode"]),
                "PrimaryPatchCount": int(decode_filter_stats.get("primary_patch_count", len(window_patch_df))),
                "PrimaryActiveSlotCount": int(decode_filter_stats.get("primary_active_slot_count", 0)),
                "SecondPassTriggered": bool(decode_filter_stats.get("second_pass_triggered", False)),
                "SecondPassReason": str(decode_filter_stats.get("second_pass_reason", "")),
                "SecondPassRawCandidateCount": int(decode_filter_stats.get("second_pass_raw_candidate_count", 0)),
                "SecondPassKeptCandidateCount": int(decode_filter_stats.get("second_pass_kept_candidate_count", 0)),
                "SecondPassPatchCount": int(decode_filter_stats.get("second_pass_patch_count", 0)),
                "SecondPassActiveSlotCount": int(decode_filter_stats.get("second_pass_active_slot_count", 0)),
                "MergedPatchCount": int(decode_filter_stats.get("merged_patch_count", len(window_patch_df))),
                "CenterThreshold": float(decode_runtime_config["center_threshold"]),
                "CountActivationThreshold": float(decode_runtime_config["count_activation_threshold"]),
                "MinCountIfActive": int(decode_runtime_config["min_count_if_active"]),
                "MaxTotalPatchesPerWindow": int(decode_runtime_config["max_total_patches_per_window"]),
                "PackagePath": str(sample_path),
            }
        )
        if args.export_window_csv:
            window_dir = units_dir / unit_id / "windows" / str(row["GeoIntervalKey"])
            window_dir.mkdir(parents=True, exist_ok=True)
            write_csv_utf8(window_patch_df, window_dir / f"W{int(row['WindowIndex']):03d}_predicted_patches.csv")
            write_csv_utf8(decoded_df, window_dir / f"W{int(row['WindowIndex']):03d}_decoded_instances.csv")
        window_progress.set_postfix(unit=unit_id, pred=int(len(window_patch_df)))
    window_progress.close()
    print(
        (
            "[eval] window inference completed | "
            f"processed_windows={len(predicted_window_rows)}, unique_units={len(unit_context_cache)}"
        ),
        flush=True,
    )

    write_csv_utf8(pd.DataFrame(predicted_window_rows), aggregated_dir / "window_prediction_summary.csv")

    vtk_config = VtkPatchExportConfig(
        display_z_scale=float(args.display_z_scale),
        invert_time=not bool(args.vtk_no_invert_time),
    )
    unit_rows: list[dict[str, Any]] = []
    print(f"[eval] start unit aggregation | total_units={len(unit_context_cache)}", flush=True)
    unit_progress = tqdm(
        unit_context_cache.items(),
        total=len(unit_context_cache),
        desc="Eval units",
        dynamic_ncols=True,
    )
    for unit_id, unit_context in unit_progress:
        unit_dir = units_dir / unit_id
        unit_dir.mkdir(parents=True, exist_ok=True)
        predicted_all = pd.concat(per_unit_patch_frames.get(unit_id, []), ignore_index=True, sort=False) if per_unit_patch_frames.get(unit_id) else pd.DataFrame()
        dedup_pred = dedupe_patch_df(
            predicted_all,
            xy_tol_m=float(args.dedupe_xy_tol_m),
            time_tol_ms=float(args.dedupe_time_tol_ms),
            azimuth_tol_deg=float(args.dedupe_azimuth_tol_deg),
            dip_tol_deg=float(args.dedupe_dip_tol_deg),
        )
        predicted_dedup_before_density_count = int(len(dedup_pred))
        dedup_pred, density_control_summary_df = apply_layer_density_budget(
            patch_df=dedup_pred,
            layers_df=unit_context["layers_df"],
            registry_payload=registry_payload,
            density_source=str(args.layer_density_source),
            density_scale=float(args.layer_density_scale),
            calibration_payload=layer_density_calibration_payload,
            calibration_min_scale=float(args.layer_density_calibration_min_scale),
            calibration_max_scale=float(args.layer_density_calibration_max_scale),
            enabled=not bool(args.disable_layer_density_control),
        )
        ground_truth = unit_context["patch_df"].copy()

        write_csv_utf8(predicted_all, unit_dir / "predicted_window_concat_patches.csv")
        write_csv_utf8(density_control_summary_df, unit_dir / "unit_layer_density_control.csv")
        write_csv_utf8(dedup_pred, unit_dir / "predicted_unit_patches.csv")
        write_csv_utf8(ground_truth, unit_dir / "ground_truth_unit_patches.csv")

        gt_vtk = export_patch_vtk_files(
            patch_df=ground_truth,
            output_dir=unit_dir,
            base_name="ground_truth_patches",
            title_prefix="ground_truth_patches",
            config=vtk_config,
        )
        pred_vtk = export_patch_vtk_files(
            patch_df=dedup_pred,
            output_dir=unit_dir,
            base_name="predicted_patches",
            title_prefix="predicted_patches",
            config=vtk_config,
        )
        compare_vtk = export_patch_comparison_vtk(
            input_patch_df=ground_truth,
            output_patch_df=dedup_pred,
            output_dir=unit_dir,
            config=vtk_config,
        )
        match_metrics = build_patch_match_metrics(
            input_patches=ground_truth,
            output_patches=dedup_pred,
            grid=unit_context["grid"],
        )
        unit_summary = {
            "UnitID": unit_id,
            "GroundTruthPatchCount": int(len(ground_truth)),
            "PredictedWindowPatchCount": int(len(predicted_all)),
            "PredictedDedupPatchCountBeforeDensityControl": int(predicted_dedup_before_density_count),
            "PredictedDedupPatchCount": int(len(dedup_pred)),
            "LayerDensityControlEnabled": bool(not args.disable_layer_density_control),
            "LayerDensitySource": str(args.layer_density_source),
            "LayerDensityScale": float(args.layer_density_scale),
            **match_metrics,
            "GroundTruthDisplayVTK": gt_vtk.get("display_vtk", ""),
            "PredictedDisplayVTK": pred_vtk.get("display_vtk", ""),
            "CompareDisplayVTK": compare_vtk.get("display_vtk", ""),
        }
        layer_table = build_unit_layer_segment_table(unit_context["layers_df"])
        gt_counts = ground_truth.groupby(UNIT_LAYER_SEGMENT_KEY_COL, dropna=False).size().to_dict() if not ground_truth.empty else {}
        pred_counts = dedup_pred.groupby(UNIT_LAYER_SEGMENT_KEY_COL, dropna=False).size().to_dict() if not dedup_pred.empty else {}
        gt_counts_by_layer = ground_truth.groupby(LAYER_SURFACE_PAIR_KEY_COL, dropna=False).size().to_dict() if not ground_truth.empty else {}
        pred_counts_by_layer = dedup_pred.groupby(LAYER_SURFACE_PAIR_KEY_COL, dropna=False).size().to_dict() if not dedup_pred.empty else {}
        density_control_rows = (
            density_control_summary_df.set_index(UNIT_LAYER_SEGMENT_KEY_COL).to_dict("index")
            if not density_control_summary_df.empty and UNIT_LAYER_SEGMENT_KEY_COL in density_control_summary_df.columns
            else {}
        )
        density_control_rows_by_layer = (
            density_control_summary_df.drop_duplicates(subset=[LAYER_SURFACE_PAIR_KEY_COL]).set_index(LAYER_SURFACE_PAIR_KEY_COL).to_dict("index")
            if not density_control_summary_df.empty and LAYER_SURFACE_PAIR_KEY_COL in density_control_summary_df.columns
            else {}
        )
        for _, layer_row in layer_table.iterrows():
            segment_key = str(layer_row.get(UNIT_LAYER_SEGMENT_KEY_COL, "")).strip()
            layer_surface_pair_key = str(layer_row.get(LAYER_SURFACE_PAIR_KEY_COL, "")).strip()
            thickness_ms = scalar_float(layer_row.get("LayerThicknessMs"), default=0.0)
            gt_patch_count = int(gt_counts.get(segment_key, gt_counts_by_layer.get(layer_surface_pair_key, 0)))
            pred_patch_count = int(pred_counts.get(segment_key, pred_counts_by_layer.get(layer_surface_pair_key, 0)))
            ground_truth_density = float(gt_patch_count / thickness_ms) if thickness_ms > 0.0 else 0.0
            predicted_density = float(pred_patch_count / thickness_ms) if thickness_ms > 0.0 else 0.0
            density_ratio = (
                float(predicted_density / ground_truth_density)
                if ground_truth_density > 0.0
                else (0.0 if predicted_density <= 0.0 else None)
            )
            suggested_scale = (
                float(ground_truth_density / predicted_density)
                if predicted_density > 0.0
                else (1.0 if ground_truth_density <= 0.0 else None)
            )
            layer_min_scale, layer_max_scale = resolve_layer_density_calibration_bounds(
                registry_payload,
                layer_surface_pair_key,
                default_min_scale=float(args.layer_density_calibration_min_scale),
                default_max_scale=float(args.layer_density_calibration_max_scale),
            )
            if suggested_scale is not None and np.isfinite(float(suggested_scale)):
                suggested_scale_clipped = float(
                    np.clip(
                        float(suggested_scale),
                        float(layer_min_scale),
                        float(layer_max_scale),
                    )
                )
            else:
                suggested_scale_clipped = None
            density_control_row = density_control_rows.get(segment_key, density_control_rows_by_layer.get(layer_surface_pair_key, {}))
            predicted_patch_count_before_density_control = scalar_int(
                density_control_row.get("PredictedPatchCountBefore"),
                default=pred_patch_count,
            )
            predicted_density_before_density_control = (
                float(predicted_patch_count_before_density_control / thickness_ms)
                if thickness_ms > 0.0
                else 0.0
            )
            unit_layer_rows.append(
                {
                    "UnitID": unit_id,
                    "GeoIntervalKey": str(layer_row.get("GeoIntervalKey", "")),
                    LAYER_SURFACE_PAIR_KEY_COL: layer_surface_pair_key,
                    UNIT_LAYER_SEGMENT_KEY_COL: segment_key,
                    "LayerThicknessMs": thickness_ms,
                    "GroundTruthPatchCount": gt_patch_count,
                    "PredictedPatchCount": pred_patch_count,
                    "GroundTruthDensity": ground_truth_density,
                    "PredictedDensity": predicted_density,
                    "GroundTruthNonZero": bool(gt_patch_count > 0),
                    "PredictedNonZero": bool(pred_patch_count > 0),
                    "PredictedPatchCountBeforeDensityControl": predicted_patch_count_before_density_control,
                    "PredictedDensityBeforeDensityControl": predicted_density_before_density_control,
                    "PredictedNonZeroBeforeDensityControl": bool(predicted_patch_count_before_density_control > 0),
                    "DensityRatioPredOverGroundTruth": density_ratio,
                    "SuggestedDensityScale": suggested_scale,
                    "SuggestedDensityScaleClipped": suggested_scale_clipped,
                    "BaseDensity": density_control_row.get("BaseDensity"),
                    "BaseNonZeroSegmentFraction": density_control_row.get("BaseNonZeroSegmentFraction"),
                    "CalibrationScale": density_control_row.get("CalibrationScale"),
                    "EffectiveDensityScale": density_control_row.get("EffectiveDensityScale"),
                    "TargetDensity": density_control_row.get("TargetDensity"),
                    "TargetNonZeroSegmentFraction": density_control_row.get("TargetNonZeroSegmentFraction"),
                    "TargetPatchCount": density_control_row.get("TargetPatchCount"),
                    "PredictedActivationScoreBeforeDensityControl": density_control_row.get("ActivationScoreBeforeDensityControl"),
                    "ActivationScoreThreshold": density_control_row.get("ActivationScoreThreshold"),
                    "ActivationThresholdSource": density_control_row.get("ActivationThresholdSource"),
                    "SuppressedByActivationThreshold": density_control_row.get("SuppressedByActivationThreshold"),
                }
            )
        unit_rows.append(unit_summary)
        write_json(unit_dir / "unit_evaluation_summary.json", unit_summary)
        unit_progress.set_postfix(unit=unit_id, gt=int(len(ground_truth)), pred=int(len(dedup_pred)))
    unit_progress.close()
    print(
        (
            "[eval] unit aggregation completed | "
            f"evaluated_units={len(unit_rows)}, layer_rows={len(unit_layer_rows)}"
        ),
        flush=True,
    )

    unit_eval_df = pd.DataFrame(unit_rows)
    write_csv_utf8(unit_eval_df, aggregated_dir / "unit_evaluation.csv")
    unit_layer_eval_df = pd.DataFrame(unit_layer_rows)
    write_csv_utf8(unit_layer_eval_df, aggregated_dir / "unit_layer_evaluation.csv")
    layer_eval_df = pd.DataFrame()
    layer_calibration_payload: dict[str, Any] = {"layers": {}}
    layer_eval_csv = aggregated_dir / "layer_density_evaluation.csv"
    layer_calibration_json = aggregated_dir / "layer_density_calibration.json"
    if not unit_layer_eval_df.empty:
        layer_eval_df = (
            unit_layer_eval_df.groupby(LAYER_SURFACE_PAIR_KEY_COL, dropna=False)
            .agg(
                UnitCount=("UnitID", lambda values: int(pd.Series(values).astype(str).nunique())),
                SegmentCount=("UnitID", "count"),
                TotalThicknessMs=("LayerThicknessMs", "sum"),
                GroundTruthPatchCount=("GroundTruthPatchCount", "sum"),
                PredictedPatchCount=("PredictedPatchCount", "sum"),
                GroundTruthNonZeroSegmentCount=("GroundTruthNonZero", "sum"),
                PredictedNonZeroSegmentCount=("PredictedNonZero", "sum"),
                PredictedNonZeroSegmentCountBeforeDensityControl=("PredictedNonZeroBeforeDensityControl", "sum"),
                MeanBaseDensity=("BaseDensity", "mean"),
                MeanTargetDensity=("TargetDensity", "mean"),
                MeanBaseNonZeroSegmentFraction=("BaseNonZeroSegmentFraction", "mean"),
                MeanTargetNonZeroSegmentFraction=("TargetNonZeroSegmentFraction", "mean"),
            )
            .reset_index()
        )
        layer_eval_df["GroundTruthDensity"] = [
            float(gt_count / thickness_ms) if float(thickness_ms) > 0.0 else 0.0
            for gt_count, thickness_ms in zip(
                layer_eval_df["GroundTruthPatchCount"],
                layer_eval_df["TotalThicknessMs"],
            )
        ]
        layer_eval_df["PredictedDensity"] = [
            float(pred_count / thickness_ms) if float(thickness_ms) > 0.0 else 0.0
            for pred_count, thickness_ms in zip(
                layer_eval_df["PredictedPatchCount"],
                layer_eval_df["TotalThicknessMs"],
            )
        ]
        layer_eval_df["GroundTruthNonZeroSegmentFraction"] = [
            float(nonzero_count / segment_count) if float(segment_count) > 0.0 else 0.0
            for nonzero_count, segment_count in zip(
                layer_eval_df["GroundTruthNonZeroSegmentCount"],
                layer_eval_df["SegmentCount"],
            )
        ]
        layer_eval_df["PredictedNonZeroSegmentFraction"] = [
            float(nonzero_count / segment_count) if float(segment_count) > 0.0 else 0.0
            for nonzero_count, segment_count in zip(
                layer_eval_df["PredictedNonZeroSegmentCount"],
                layer_eval_df["SegmentCount"],
            )
        ]
        layer_eval_df["PredictedNonZeroSegmentFractionBeforeDensityControl"] = [
            float(nonzero_count / segment_count) if float(segment_count) > 0.0 else 0.0
            for nonzero_count, segment_count in zip(
                layer_eval_df["PredictedNonZeroSegmentCountBeforeDensityControl"],
                layer_eval_df["SegmentCount"],
            )
        ]
        density_ratios: list[float | None] = []
        suggested_scales: list[float | None] = []
        suggested_scales_clipped: list[float | None] = []
        suggested_nonzero_fraction_scales: list[float | None] = []
        suggested_activation_score_thresholds: list[float | None] = []
        for _, row in layer_eval_df.iterrows():
            ground_truth_density = scalar_float(row.get("GroundTruthDensity"), default=0.0)
            predicted_density = scalar_float(row.get("PredictedDensity"), default=0.0)
            if ground_truth_density > 0.0:
                density_ratio = float(predicted_density / ground_truth_density)
            else:
                density_ratio = 0.0 if predicted_density <= 0.0 else None
            if predicted_density > 0.0:
                suggested_scale = float(ground_truth_density / predicted_density)
            else:
                suggested_scale = 1.0 if ground_truth_density <= 0.0 else None
            layer_key = str(row.get(LAYER_SURFACE_PAIR_KEY_COL, "")).strip()
            layer_min_scale, layer_max_scale = resolve_layer_density_calibration_bounds(
                registry_payload,
                layer_key,
                default_min_scale=float(args.layer_density_calibration_min_scale),
                default_max_scale=float(args.layer_density_calibration_max_scale),
            )
            if suggested_scale is not None and np.isfinite(float(suggested_scale)):
                suggested_scale_clipped = float(
                    np.clip(
                        float(suggested_scale),
                        float(layer_min_scale),
                        float(layer_max_scale),
                    )
                )
            else:
                suggested_scale_clipped = None
            ground_truth_nonzero_fraction = scalar_float(
                row.get("GroundTruthNonZeroSegmentFraction"),
                default=0.0,
            )
            predicted_nonzero_fraction_before_density_control = scalar_float(
                row.get("PredictedNonZeroSegmentFractionBeforeDensityControl"),
                default=0.0,
            )
            if predicted_nonzero_fraction_before_density_control > 0.0:
                suggested_nonzero_fraction_scale = float(
                    ground_truth_nonzero_fraction / predicted_nonzero_fraction_before_density_control
                )
            else:
                suggested_nonzero_fraction_scale = 1.0 if ground_truth_nonzero_fraction <= 0.0 else None
            layer_key = str(row.get(LAYER_SURFACE_PAIR_KEY_COL, "")).strip()
            layer_group_df = unit_layer_eval_df[
                unit_layer_eval_df[LAYER_SURFACE_PAIR_KEY_COL].astype(str) == layer_key
            ].copy()
            suggested_activation_score_threshold = compute_activation_score_threshold(
                activation_scores=layer_group_df.get("PredictedActivationScoreBeforeDensityControl", pd.Series(dtype=float)),
                target_active_count=int(layer_group_df.get("GroundTruthNonZero", pd.Series(dtype=bool)).sum()),
            )
            density_ratios.append(density_ratio)
            suggested_scales.append(suggested_scale)
            suggested_scales_clipped.append(suggested_scale_clipped)
            suggested_nonzero_fraction_scales.append(suggested_nonzero_fraction_scale)
            suggested_activation_score_thresholds.append(suggested_activation_score_threshold)
            if layer_key:
                layer_calibration_payload["layers"][layer_key] = {
                    "ground_truth_density": ground_truth_density,
                    "predicted_density": predicted_density,
                    "density_ratio_pred_over_ground_truth": density_ratio,
                    "suggested_density_scale": suggested_scale,
                    "suggested_density_scale_clipped": suggested_scale_clipped,
                    "calibration_min_scale_used": float(layer_min_scale),
                    "calibration_max_scale_used": float(layer_max_scale),
                    "unit_count": scalar_int(row.get("UnitCount"), default=0),
                    "segment_count": scalar_int(row.get("SegmentCount"), default=0),
                    "total_thickness_ms": scalar_float(row.get("TotalThicknessMs"), default=0.0),
                    "ground_truth_patch_count": scalar_int(row.get("GroundTruthPatchCount"), default=0),
                    "predicted_patch_count": scalar_int(row.get("PredictedPatchCount"), default=0),
                    "ground_truth_nonzero_segment_count": scalar_int(row.get("GroundTruthNonZeroSegmentCount"), default=0),
                    "predicted_nonzero_segment_count": scalar_int(row.get("PredictedNonZeroSegmentCount"), default=0),
                    "predicted_nonzero_segment_count_before_density_control": scalar_int(
                        row.get("PredictedNonZeroSegmentCountBeforeDensityControl"),
                        default=0,
                    ),
                    "ground_truth_nonzero_segment_fraction": ground_truth_nonzero_fraction,
                    "predicted_nonzero_segment_fraction": scalar_float(
                        row.get("PredictedNonZeroSegmentFraction"),
                        default=0.0,
                    ),
                    "predicted_nonzero_segment_fraction_before_density_control": predicted_nonzero_fraction_before_density_control,
                    "suggested_nonzero_fraction_scale": suggested_nonzero_fraction_scale,
                    "suggested_activation_score_threshold": suggested_activation_score_threshold,
                    "mean_base_density": scalar_float(row.get("MeanBaseDensity"), default=0.0),
                    "mean_target_density": scalar_float(row.get("MeanTargetDensity"), default=0.0),
                    "mean_base_nonzero_segment_fraction": scalar_float(
                        row.get("MeanBaseNonZeroSegmentFraction"),
                        default=0.0,
                    ),
                    "mean_target_nonzero_segment_fraction": scalar_float(
                        row.get("MeanTargetNonZeroSegmentFraction"),
                        default=0.0,
                    ),
                }
        layer_eval_df["DensityRatioPredOverGroundTruth"] = density_ratios
        layer_eval_df["SuggestedDensityScale"] = suggested_scales
        layer_eval_df["SuggestedDensityScaleClipped"] = suggested_scales_clipped
        layer_eval_df["SuggestedNonZeroFractionScale"] = suggested_nonzero_fraction_scales
        layer_eval_df["SuggestedActivationScoreThreshold"] = suggested_activation_score_thresholds
    write_csv_utf8(layer_eval_df, layer_eval_csv)
    layer_calibration_payload.update(
        {
            "run_dir": str(run_dir),
            "split_name": str(args.split_name),
            "layer_density_source": str(args.layer_density_source),
            "layer_density_scale": float(args.layer_density_scale),
            "layer_density_calibration_min_scale": float(args.layer_density_calibration_min_scale),
            "layer_density_calibration_max_scale": float(args.layer_density_calibration_max_scale),
            "layer_density_evaluation_csv": str(layer_eval_csv.resolve()),
        }
    )
    write_json(layer_calibration_json, layer_calibration_payload)
    layer_density_calibration_json_sha256 = compute_file_sha256(layer_calibration_json)
    aggregate_summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "layer_model_registry_py": str(args.layer_model_registry_py) if args.layer_model_registry_py else "",
        "layer_model_registry_sha256": layer_model_registry_sha256,
        "layer_surface_pair_key": layer_surface_pair_key,
        "split_manifest_csv": str(split_manifest_csv),
        "split_manifest_sha256": split_manifest_sha256,
        "device": str(device),
        "evaluated_unit_count": int(unit_eval_df["UnitID"].nunique()) if not unit_eval_df.empty else 0,
        "evaluated_window_count": int(len(manifest_df)),
        "mean_raw_candidate_count": float(pd.DataFrame(predicted_window_rows)["RawCandidateCount"].mean()) if predicted_window_rows else 0.0,
        "mean_kept_candidate_count": float(pd.DataFrame(predicted_window_rows)["KeptCandidateCount"].mean()) if predicted_window_rows else 0.0,
        "mean_forced_voxel_count": float(pd.DataFrame(predicted_window_rows)["ForcedVoxelCount"].mean()) if predicted_window_rows else 0.0,
        "mean_ground_truth_patch_count": float(unit_eval_df["GroundTruthPatchCount"].mean()) if not unit_eval_df.empty else 0.0,
        "mean_predicted_dedup_before_density_patch_count": float(unit_eval_df["PredictedDedupPatchCountBeforeDensityControl"].mean()) if "PredictedDedupPatchCountBeforeDensityControl" in unit_eval_df.columns and not unit_eval_df.empty else 0.0,
        "mean_predicted_dedup_patch_count": float(unit_eval_df["PredictedDedupPatchCount"].mean()) if not unit_eval_df.empty else 0.0,
        "mean_matched_patch_count": float(unit_eval_df["matched_patch_count"].mean()) if "matched_patch_count" in unit_eval_df.columns and not unit_eval_df.empty else 0.0,
        "median_center_offset": float(unit_eval_df["center_offset_median"].dropna().median()) if "center_offset_median" in unit_eval_df.columns and not unit_eval_df["center_offset_median"].dropna().empty else None,
        "median_azimuth_diff_deg": float(unit_eval_df["azimuth_diff_median_deg"].dropna().median()) if "azimuth_diff_median_deg" in unit_eval_df.columns and not unit_eval_df["azimuth_diff_median_deg"].dropna().empty else None,
        "median_dip_diff_deg": float(unit_eval_df["dip_diff_median_deg"].dropna().median()) if "dip_diff_median_deg" in unit_eval_df.columns and not unit_eval_df["dip_diff_median_deg"].dropna().empty else None,
        "decode_mode": str(args.decode_mode),
        "relaxed_min_count": int(args.relaxed_min_count),
        "count_activation_threshold": float(args.count_activation_threshold),
        "min_count_if_active": int(args.min_count_if_active),
        "layer_density_control_enabled": bool(not args.disable_layer_density_control),
        "layer_density_source": str(args.layer_density_source),
        "layer_density_scale": float(args.layer_density_scale),
        "input_layer_density_calibration_json": str(args.layer_density_calibration_json) if args.layer_density_calibration_json else "",
        "input_layer_density_calibration_json_sha256": input_layer_density_calibration_json_sha256,
        "layer_density_calibration_min_scale": float(args.layer_density_calibration_min_scale),
        "layer_density_calibration_max_scale": float(args.layer_density_calibration_max_scale),
        "layer_density_evaluation_csv": str(layer_eval_csv.resolve()),
        "layer_density_calibration_json": str(layer_calibration_json.resolve()),
        "layer_density_calibration_json_sha256": layer_density_calibration_json_sha256,
    }
    summary_path = aggregated_dir / "evaluation_summary.json"
    write_json(summary_path, aggregate_summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - 推理评估",
        lines=[
            f"split_manifest_csv: {split_manifest_csv}",
            f"checkpoint: {args.checkpoint if args.checkpoint else 'None'}",
            f"layer_model_registry_py: {args.layer_model_registry_py if args.layer_model_registry_py else 'None'}",
            f"layer_surface_pair_key: {layer_surface_pair_key if layer_surface_pair_key else 'ALL'}",
            f"run_dir: {run_dir}",
            f"device: {device}",
            f"vtk_invert_time: {vtk_config.invert_time}",
            f"display_z_scale: {vtk_config.display_z_scale}",
            f"evaluated_unit_count: {aggregate_summary['evaluated_unit_count']}",
            f"evaluated_window_count: {aggregate_summary['evaluated_window_count']}",
            f"mean_raw_candidate_count: {aggregate_summary['mean_raw_candidate_count']}",
            f"mean_kept_candidate_count: {aggregate_summary['mean_kept_candidate_count']}",
            f"mean_forced_voxel_count: {aggregate_summary['mean_forced_voxel_count']}",
            f"decode_mode: {aggregate_summary['decode_mode']}",
            f"relaxed_min_count: {aggregate_summary['relaxed_min_count']}",
            f"count_activation_threshold: {aggregate_summary['count_activation_threshold']}",
            f"min_count_if_active: {aggregate_summary['min_count_if_active']}",
            f"layer_density_control_enabled: {aggregate_summary['layer_density_control_enabled']}",
            f"layer_density_source: {aggregate_summary['layer_density_source']}",
            f"layer_density_scale: {aggregate_summary['layer_density_scale']}",
            f"input_layer_density_calibration_json: {aggregate_summary['input_layer_density_calibration_json'] if aggregate_summary['input_layer_density_calibration_json'] else 'None'}",
            f"layer_density_calibration_min_scale: {aggregate_summary['layer_density_calibration_min_scale']}",
            f"layer_density_calibration_max_scale: {aggregate_summary['layer_density_calibration_max_scale']}",
            f"mean_ground_truth_patch_count: {aggregate_summary['mean_ground_truth_patch_count']}",
            f"mean_predicted_dedup_before_density_patch_count: {aggregate_summary['mean_predicted_dedup_before_density_patch_count']}",
            f"mean_predicted_dedup_patch_count: {aggregate_summary['mean_predicted_dedup_patch_count']}",
            f"mean_matched_patch_count: {aggregate_summary['mean_matched_patch_count']}",
            f"layer_density_evaluation_csv: {layer_eval_csv}",
            f"layer_density_calibration_json: {layer_calibration_json}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"run_dir: {run_dir}")
    print(f"evaluated_unit_count: {aggregate_summary['evaluated_unit_count']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
