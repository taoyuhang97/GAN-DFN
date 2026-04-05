# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from baseline_common import (
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SLOTS_PER_VOXEL,
    DEFAULT_UNIT_DFN_ROOT,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_Z_STEP_MS,
    VtkPatchExportConfig,
    append_lines_to_docx,
    build_window_grid,
    dedupe_patch_df,
    load_sample_manifest,
    prepare_unit_context,
    prediction_to_label_payload,
    read_csv_utf8,
    write_csv_utf8,
    write_json,
    decode_instance_label_to_patches,
    export_patch_comparison_vtk,
    export_patch_vtk_files,
    build_patch_match_metrics,
)
from baseline_model import SparseInstanceBaselineUNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 G-DFN 监督 baseline 做窗口预测、单元回拼与评估。")
    parser.add_argument("--split-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "推理评估")
    parser.add_argument("--run-name", type=str, default=f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--center-threshold", type=float, default=0.7)
    parser.add_argument("--decode-mode", type=str, default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--relaxed-min-count", type=int, default=1)
    parser.add_argument("--max-slots-per-voxel", type=int, default=DEFAULT_SLOTS_PER_VOXEL)
    parser.add_argument("--max-total-patches-per-window", type=int, default=512)
    parser.add_argument("--dedupe-xy-tol-m", type=float, default=6.25)
    parser.add_argument("--dedupe-time-tol-ms", type=float, default=0.4)
    parser.add_argument("--dedupe-azimuth-tol-deg", type=float, default=20.0)
    parser.add_argument("--dedupe-dip-tol-deg", type=float, default=12.0)
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


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint_model(Path(args.checkpoint), device)

    split_manifest_csv = Path(args.split_run_dir) / f"{args.split_name}_manifest.csv"
    if not split_manifest_csv.exists():
        raise FileNotFoundError(f"split manifest not found: {split_manifest_csv}")
    manifest_df = read_csv_utf8(split_manifest_csv)
    if args.limit_samples:
        manifest_df = manifest_df.head(int(args.limit_samples)).copy()
    if args.limit_units:
        keep_units = manifest_df["UnitID"].astype(str).drop_duplicates().head(int(args.limit_units)).tolist()
        manifest_df = manifest_df[manifest_df["UnitID"].astype(str).isin(keep_units)].copy()
    manifest_df = manifest_df.reset_index(drop=True)
    if manifest_df.empty:
        raise ValueError("evaluation manifest is empty")

    run_dir = Path(args.output_root) / args.run_name
    units_dir = run_dir / "units"
    aggregated_dir = run_dir / "aggregated"
    units_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    unit_context_cache: dict[str, dict[str, Any]] = {}
    predicted_window_rows: list[dict[str, Any]] = []
    per_unit_patch_frames: dict[str, list[pd.DataFrame]] = {}

    for _, row in manifest_df.iterrows():
        unit_id = str(row["UnitID"])
        if unit_id not in unit_context_cache:
            unit_context_cache[unit_id] = prepare_unit_context(unit_id=unit_id, unit_dfn_root=args.unit_dfn_root)
        unit_context = unit_context_cache[unit_id]

        sample_path = Path(str(row["PackagePath"]))
        center_probs, count_pred, geom_pred, valid_z_mask = run_single_sample(model, sample_path, device)
        label_payload, decode_filter_stats = prediction_to_label_payload(
            center_probs=center_probs,
            count_pred=count_pred,
            geom_pred=geom_pred,
            valid_z_mask=valid_z_mask,
            threshold=float(args.center_threshold),
            max_slots_per_voxel=int(args.max_slots_per_voxel),
            max_total_patches=int(args.max_total_patches_per_window) if args.max_total_patches_per_window else None,
            decode_mode=str(args.decode_mode),
            relaxed_min_count=int(args.relaxed_min_count),
        )
        window_grid = build_window_grid(
            unit_context=unit_context,
            window_top=float(row["WindowTopTime"]),
            window_size=DEFAULT_WINDOW_SIZE,
            z_step_ms=DEFAULT_Z_STEP_MS,
        )
        window_patch_df, decoded_df, decode_summary = decode_instance_label_to_patches(
            label_payload=label_payload,
            grid=window_grid,
            layers_df=unit_context["layers_df"],
            threshold=float(args.center_threshold),
        )
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
                "WindowIndex": int(row["WindowIndex"]),
                "PredictedPatchCount": int(len(window_patch_df)),
                "DecodedActiveSlotCount": int(decode_summary["active_slot_count"]),
                "RawCandidateCount": int(decode_filter_stats["raw_candidate_count"]),
                "KeptCandidateCount": int(decode_filter_stats["kept_candidate_count"]),
                "MeanPredCount": float(decode_filter_stats["mean_pred_count"]),
                "MaxPredCount": int(decode_filter_stats["max_pred_count"]),
                "ForcedVoxelCount": int(decode_filter_stats["forced_voxel_count"]),
                "DecodeMode": str(decode_filter_stats["decode_mode"]),
                "PackagePath": str(sample_path),
            }
        )
        if args.export_window_csv:
            window_dir = units_dir / unit_id / "windows" / str(row["GeoIntervalKey"])
            window_dir.mkdir(parents=True, exist_ok=True)
            write_csv_utf8(window_patch_df, window_dir / f"W{int(row['WindowIndex']):03d}_predicted_patches.csv")
            write_csv_utf8(decoded_df, window_dir / f"W{int(row['WindowIndex']):03d}_decoded_instances.csv")

    write_csv_utf8(pd.DataFrame(predicted_window_rows), aggregated_dir / "window_prediction_summary.csv")

    vtk_config = VtkPatchExportConfig(
        display_z_scale=float(args.display_z_scale),
        invert_time=not bool(args.vtk_no_invert_time),
    )
    unit_rows: list[dict[str, Any]] = []
    for unit_id, unit_context in unit_context_cache.items():
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
        ground_truth = unit_context["patch_df"].copy()

        write_csv_utf8(predicted_all, unit_dir / "predicted_window_concat_patches.csv")
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
            "PredictedDedupPatchCount": int(len(dedup_pred)),
            **match_metrics,
            "GroundTruthDisplayVTK": gt_vtk.get("display_vtk", ""),
            "PredictedDisplayVTK": pred_vtk.get("display_vtk", ""),
            "CompareDisplayVTK": compare_vtk.get("display_vtk", ""),
        }
        unit_rows.append(unit_summary)
        write_json(unit_dir / "unit_evaluation_summary.json", unit_summary)

    unit_eval_df = pd.DataFrame(unit_rows)
    write_csv_utf8(unit_eval_df, aggregated_dir / "unit_evaluation.csv")
    aggregate_summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(args.checkpoint),
        "split_manifest_csv": str(split_manifest_csv),
        "device": str(device),
        "evaluated_unit_count": int(unit_eval_df["UnitID"].nunique()) if not unit_eval_df.empty else 0,
        "evaluated_window_count": int(len(manifest_df)),
        "mean_raw_candidate_count": float(pd.DataFrame(predicted_window_rows)["RawCandidateCount"].mean()) if predicted_window_rows else 0.0,
        "mean_kept_candidate_count": float(pd.DataFrame(predicted_window_rows)["KeptCandidateCount"].mean()) if predicted_window_rows else 0.0,
        "mean_forced_voxel_count": float(pd.DataFrame(predicted_window_rows)["ForcedVoxelCount"].mean()) if predicted_window_rows else 0.0,
        "mean_ground_truth_patch_count": float(unit_eval_df["GroundTruthPatchCount"].mean()) if not unit_eval_df.empty else 0.0,
        "mean_predicted_dedup_patch_count": float(unit_eval_df["PredictedDedupPatchCount"].mean()) if not unit_eval_df.empty else 0.0,
        "mean_matched_patch_count": float(unit_eval_df["matched_patch_count"].mean()) if "matched_patch_count" in unit_eval_df.columns and not unit_eval_df.empty else 0.0,
        "median_center_offset": float(unit_eval_df["center_offset_median"].dropna().median()) if "center_offset_median" in unit_eval_df.columns and not unit_eval_df["center_offset_median"].dropna().empty else None,
        "median_azimuth_diff_deg": float(unit_eval_df["azimuth_diff_median_deg"].dropna().median()) if "azimuth_diff_median_deg" in unit_eval_df.columns and not unit_eval_df["azimuth_diff_median_deg"].dropna().empty else None,
        "median_dip_diff_deg": float(unit_eval_df["dip_diff_median_deg"].dropna().median()) if "dip_diff_median_deg" in unit_eval_df.columns and not unit_eval_df["dip_diff_median_deg"].dropna().empty else None,
        "decode_mode": str(args.decode_mode),
        "relaxed_min_count": int(args.relaxed_min_count),
    }
    summary_path = aggregated_dir / "evaluation_summary.json"
    write_json(summary_path, aggregate_summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - 推理评估",
        lines=[
            f"split_manifest_csv: {split_manifest_csv}",
            f"checkpoint: {args.checkpoint}",
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
            f"mean_ground_truth_patch_count: {aggregate_summary['mean_ground_truth_patch_count']}",
            f"mean_predicted_dedup_patch_count: {aggregate_summary['mean_predicted_dedup_patch_count']}",
            f"mean_matched_patch_count: {aggregate_summary['mean_matched_patch_count']}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"run_dir: {run_dir}")
    print(f"evaluated_unit_count: {aggregate_summary['evaluated_unit_count']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
