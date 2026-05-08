# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from baseline_common import (
    compute_file_sha256,
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    append_lines_to_docx,
    build_layer_density_prior_payload,
    summarize_layer_density_priors,
    write_csv_utf8,
    write_json,
)
from layer_model_registry import (
    LAYER_SURFACE_PAIR_KEY_COL,
    ensure_layer_surface_pair_key_column,
    sanitize_layer_surface_pair_key,
    summarize_manifest_by_layer_surface_pair,
    write_layer_model_registry_py,
)
from layer_training_config import canonicalize_layer_surface_pair_key, load_layer_training_config


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_supervised_baseline.py"
DEFAULT_LAYER_TRAINING_CONFIG_PY = THIS_DIR / "layerwise_training_plan_demo_v1.py"


def format_running_jobs(running_jobs: list[dict[str, Any]], limit: int = 6) -> str:
    if not running_jobs:
        return "none"
    items = [
        f"{job['layer_surface_pair_key']}@gpu{job['gpu_id']}"
        for job in running_jobs[:limit]
    ]
    if len(running_jobs) > limit:
        items.append(f"...(+{len(running_jobs) - limit})")
    return ", ".join(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 LayerSurfacePairKey 在多 GPU 上并行训练多个监督 baseline。")
    parser.add_argument("--split-run-dir", type=Path, required=True)
    parser.add_argument("--unit-dfn-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "训练结果_分层")
    parser.add_argument("--run-name-prefix", type=str, default="layerwise_baseline")
    parser.add_argument("--registry-output-py", type=Path)
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--layer-training-config-py", type=Path, default=DEFAULT_LAYER_TRAINING_CONFIG_PY)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-concurrent-jobs", type=int)
    parser.add_argument("--layer-surface-pair-key", nargs="+")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--slots-per-voxel", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--torch-num-interop-threads", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--train-limit-samples", type=int)
    parser.add_argument("--val-limit-samples", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-cache-raw-packages", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--center-positive-weight", type=float, default=24.0)
    parser.add_argument("--center-negative-weight", type=float, default=0.25)
    parser.add_argument("--center-focal-gamma", type=float, default=2.0)
    parser.add_argument("--count-positive-weight", type=float, default=12.0)
    parser.add_argument("--count-negative-weight", type=float, default=0.25)
    parser.add_argument("--center-loss-weight", type=float, default=2.5)
    parser.add_argument("--count-loss-weight", type=float, default=1.0)
    parser.add_argument("--area-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--calibration-center-thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--calibration-count-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    )
    parser.add_argument("--calibration-window-weight", type=float, default=0.35)
    return parser


def load_split_manifest(split_run_dir: Path, split_name: str) -> pd.DataFrame:
    manifest_path = Path(split_run_dir) / f"{split_name}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    return ensure_layer_surface_pair_key_column(pd.read_csv(manifest_path, encoding="utf-8-sig"), key_col=LAYER_SURFACE_PAIR_KEY_COL)


def build_base_training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "slots_per_voxel": int(args.slots_per_voxel),
        "base_channels": int(args.base_channels),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "torch_num_threads": int(args.torch_num_threads),
        "torch_num_interop_threads": int(args.torch_num_interop_threads),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "train_limit_samples": int(args.train_limit_samples) if args.train_limit_samples is not None else None,
        "val_limit_samples": int(args.val_limit_samples) if args.val_limit_samples is not None else None,
        "max_train_steps": int(args.max_train_steps) if args.max_train_steps is not None else None,
        "max_val_steps": int(args.max_val_steps) if args.max_val_steps is not None else None,
        "disable_amp": bool(args.disable_amp),
        "no_cache_raw_packages": bool(args.no_cache_raw_packages),
        "no_progress": bool(args.no_progress),
        "center_positive_weight": float(args.center_positive_weight),
        "center_negative_weight": float(args.center_negative_weight),
        "center_focal_gamma": float(args.center_focal_gamma),
        "count_positive_weight": float(args.count_positive_weight),
        "count_negative_weight": float(args.count_negative_weight),
        "center_loss_weight": float(args.center_loss_weight),
        "count_loss_weight": float(args.count_loss_weight),
        "area_loss_weight": float(args.area_loss_weight),
        "calibration_center_thresholds": [float(value) for value in args.calibration_center_thresholds],
        "calibration_count_thresholds": [float(value) for value in args.calibration_count_thresholds],
        "calibration_window_weight": float(args.calibration_window_weight),
    }


def build_training_identity_payload(
    args: argparse.Namespace,
    layer_training_config: dict[str, Any],
) -> dict[str, Any]:
    requested_layer_keys: list[str] = []
    if args.layer_surface_pair_key:
        for raw_value in args.layer_surface_pair_key:
            canonical_key = canonicalize_layer_surface_pair_key(raw_value)
            if canonical_key:
                requested_layer_keys.append(canonical_key)
    return {
        "requested_layer_surface_pair_keys": requested_layer_keys,
        "layer_training_config_py": layer_training_config["config_py"],
        "layer_training_config_sha256": layer_training_config["config_sha256"],
        **build_base_training_config(args),
    }


def build_unit_confidence_map_from_manifest(manifest_df: pd.DataFrame) -> dict[str, float]:
    if manifest_df.empty or "UnitID" not in manifest_df.columns:
        return {}
    confidence_col = "UnitConfidence" if "UnitConfidence" in manifest_df.columns else "SampleConfidence" if "SampleConfidence" in manifest_df.columns else ""
    if not confidence_col:
        return {}
    work_df = manifest_df[["UnitID", confidence_col]].copy()
    work_df["UnitID"] = work_df["UnitID"].astype(str).str.strip()
    work_df[confidence_col] = pd.to_numeric(work_df[confidence_col], errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    grouped = work_df.groupby("UnitID", dropna=False)[confidence_col].max()
    return {str(unit_id): float(value) for unit_id, value in grouped.items() if str(unit_id).strip()}


def build_layer_jobs(
    args: argparse.Namespace,
    layer_training_config: dict[str, Any],
    base_training_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train_df = load_split_manifest(Path(args.split_run_dir), "train")
    val_df = load_split_manifest(Path(args.split_run_dir), "val")
    train_summary = summarize_manifest_by_layer_surface_pair(train_df)
    val_summary = summarize_manifest_by_layer_surface_pair(val_df)
    merged = pd.merge(
        train_summary.add_prefix("Train_"),
        val_summary.add_prefix("Val_"),
        left_on=f"Train_{LAYER_SURFACE_PAIR_KEY_COL}",
        right_on=f"Val_{LAYER_SURFACE_PAIR_KEY_COL}",
        how="outer",
    )
    merged[LAYER_SURFACE_PAIR_KEY_COL] = (
        merged.get(f"Train_{LAYER_SURFACE_PAIR_KEY_COL}", pd.Series(dtype="object"))
        .fillna(merged.get(f"Val_{LAYER_SURFACE_PAIR_KEY_COL}", pd.Series(dtype="object")))
        .astype(str)
    )
    if merged.empty:
        raise ValueError("no layer pair windows found in split manifests")

    if args.layer_surface_pair_key:
        selected_keys = []
        seen_keys: set[str] = set()
        for raw_value in args.layer_surface_pair_key:
            canonical_key = canonicalize_layer_surface_pair_key(raw_value)
            if not canonical_key or canonical_key in seen_keys:
                continue
            seen_keys.add(canonical_key)
            selected_keys.append(canonical_key)
    else:
        selected_keys = [str(layer_key) for layer_key in layer_training_config["allowed_layer_surface_pair_keys"]]
        if not selected_keys:
            selected_keys = (
                merged.sort_values(
                    [f"Train_WindowCount", LAYER_SURFACE_PAIR_KEY_COL],
                    ascending=[False, True],
                    na_position="last",
                )[LAYER_SURFACE_PAIR_KEY_COL]
                .drop_duplicates()
                .astype(str)
                .tolist()
            )

    jobs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    output_root = Path(args.output_root)
    for layer_key in selected_keys:
        train_mask = train_df[LAYER_SURFACE_PAIR_KEY_COL].astype(str) == layer_key
        val_mask = val_df[LAYER_SURFACE_PAIR_KEY_COL].astype(str) == layer_key
        train_window_count = int(train_mask.sum())
        val_window_count = int(val_mask.sum())
        sanitized_key = sanitize_layer_surface_pair_key(layer_key)
        run_name = f"{args.run_name_prefix}__{sanitized_key}"
        run_dir = output_root / run_name
        best_checkpoint = run_dir / "checkpoints" / "best_model.pt"
        log_path = output_root / "launcher_logs" / f"{sanitized_key}.log"
        job_docx_path = output_root / "layer_train_docs" / f"{sanitized_key}.docx"
        train_config_override = dict(layer_training_config["layer_training_overrides"].get(layer_key, {}))
        effective_training_config = dict(base_training_config)
        effective_training_config.update(train_config_override)
        status = "pending"
        reason = ""
        if train_window_count <= 0 or val_window_count <= 0:
            status = "skipped_no_split_data"
            reason = "train_or_val_empty"
        elif best_checkpoint.exists() and not bool(args.overwrite_existing):
            status = "skipped_existing"
            reason = "best_checkpoint_exists"
        else:
            jobs.append(
                {
                    "layer_surface_pair_key": layer_key,
                    "sanitized_key": sanitized_key,
                    "run_name": run_name,
                    "run_dir": run_dir,
                    "best_checkpoint": best_checkpoint,
                    "log_path": log_path,
                    "job_docx_path": job_docx_path,
                    "train_window_count": train_window_count,
                    "val_window_count": val_window_count,
                    "train_config_override": train_config_override,
                    "effective_training_config": effective_training_config,
                }
            )
        summary_rows.append(
            {
                LAYER_SURFACE_PAIR_KEY_COL: layer_key,
                "SanitizedLayerKey": sanitized_key,
                "RunName": run_name,
                "RunDir": str(run_dir),
                "BestCheckpoint": str(best_checkpoint),
                "TrainWindowCount": train_window_count,
                "ValWindowCount": val_window_count,
                "Status": status,
                "Reason": reason,
                "LogPath": str(log_path),
                "JobDocxPath": str(job_docx_path),
                "TrainConfigOverrideJSON": json.dumps(train_config_override, ensure_ascii=False, sort_keys=True),
                "EffectiveTrainConfigJSON": json.dumps(effective_training_config, ensure_ascii=False, sort_keys=True),
            }
        )
    return jobs, pd.DataFrame(summary_rows)


def build_train_command(args: argparse.Namespace, layer_job: dict[str, Any]) -> list[str]:
    effective_training_config = dict(layer_job["effective_training_config"])
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--split-run-dir",
        str(Path(args.split_run_dir).resolve()),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--run-name",
        str(layer_job["run_name"]),
        "--layer-surface-pair-key",
        str(layer_job["layer_surface_pair_key"]),
        "--docx-path",
        str(Path(layer_job["job_docx_path"]).resolve()),
        "--slots-per-voxel",
        str(int(effective_training_config["slots_per_voxel"])),
        "--base-channels",
        str(int(effective_training_config["base_channels"])),
        "--batch-size",
        str(int(effective_training_config["batch_size"])),
        "--num-workers",
        str(int(effective_training_config["num_workers"])),
        "--torch-num-threads",
        str(int(effective_training_config["torch_num_threads"])),
        "--torch-num-interop-threads",
        str(int(effective_training_config["torch_num_interop_threads"])),
        "--epochs",
        str(int(effective_training_config["epochs"])),
        "--learning-rate",
        str(float(effective_training_config["learning_rate"])),
        "--weight-decay",
        str(float(effective_training_config["weight_decay"])),
        "--center-positive-weight",
        str(float(effective_training_config["center_positive_weight"])),
        "--center-negative-weight",
        str(float(effective_training_config["center_negative_weight"])),
        "--center-focal-gamma",
        str(float(effective_training_config["center_focal_gamma"])),
        "--count-positive-weight",
        str(float(effective_training_config["count_positive_weight"])),
        "--count-negative-weight",
        str(float(effective_training_config["count_negative_weight"])),
        "--center-loss-weight",
        str(float(effective_training_config["center_loss_weight"])),
        "--count-loss-weight",
        str(float(effective_training_config["count_loss_weight"])),
        "--area-loss-weight",
        str(float(effective_training_config["area_loss_weight"])),
        "--calibration-center-thresholds",
        *[str(float(value)) for value in effective_training_config["calibration_center_thresholds"]],
        "--calibration-count-thresholds",
        *[str(float(value)) for value in effective_training_config["calibration_count_thresholds"]],
        "--calibration-window-weight",
        str(float(effective_training_config["calibration_window_weight"])),
        "--device",
        "cuda",
        "--seed",
        str(int(effective_training_config["seed"])),
    ]
    if effective_training_config["train_limit_samples"] is not None:
        cmd.extend(["--train-limit-samples", str(int(effective_training_config["train_limit_samples"]))])
    if effective_training_config["val_limit_samples"] is not None:
        cmd.extend(["--val-limit-samples", str(int(effective_training_config["val_limit_samples"]))])
    if effective_training_config["max_train_steps"] is not None:
        cmd.extend(["--max-train-steps", str(int(effective_training_config["max_train_steps"]))])
    if effective_training_config["max_val_steps"] is not None:
        cmd.extend(["--max-val-steps", str(int(effective_training_config["max_val_steps"]))])
    if bool(effective_training_config["disable_amp"]):
        cmd.append("--disable-amp")
    if bool(effective_training_config["no_cache_raw_packages"]):
        cmd.append("--no-cache-raw-packages")
    if bool(effective_training_config["no_progress"]):
        cmd.append("--no-progress")
    return cmd


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "launcher_logs").mkdir(parents=True, exist_ok=True)
    layer_training_config = load_layer_training_config(args.layer_training_config_py)
    base_training_config = build_base_training_config(args)
    jobs, summary_df = build_layer_jobs(
        args,
        layer_training_config=layer_training_config,
        base_training_config=base_training_config,
    )
    initial_status_counts = (
        summary_df.get("Status", pd.Series(dtype="object")).astype(str).value_counts().to_dict()
        if not summary_df.empty
        else {}
    )
    total_layer_count = int(len(summary_df))
    precompleted_count = int(sum(count for status, count in initial_status_counts.items() if status != "pending"))
    print(
        (
            "[train-layerwise] training plan prepared | "
            f"requested_layers={total_layer_count}, pending_train={len(jobs)}, "
            f"skipped_existing={int(initial_status_counts.get('skipped_existing', 0))}, "
            f"skipped_no_split_data={int(initial_status_counts.get('skipped_no_split_data', 0))}"
        ),
        flush=True,
    )
    train_df = load_split_manifest(Path(args.split_run_dir), "train")
    train_unit_ids = train_df["UnitID"].astype(str).drop_duplicates().tolist()
    train_unit_confidence_map = build_unit_confidence_map_from_manifest(train_df)
    density_prior_summary_df, _ = summarize_layer_density_priors(
        unit_dfn_root=Path(args.unit_dfn_root),
        unit_ids=train_unit_ids,
        unit_confidence_map=train_unit_confidence_map,
    )
    density_prior_summary_csv = output_root / f"{args.run_name_prefix}_layer_density_prior_summary.csv"
    write_csv_utf8(density_prior_summary_df, density_prior_summary_csv)
    layer_density_priors = build_layer_density_prior_payload(density_prior_summary_df)

    available_gpus = [int(gpu_id) for gpu_id in args.gpu_ids]
    if not available_gpus:
        raise ValueError("gpu_ids must not be empty")
    max_concurrent_jobs = int(args.max_concurrent_jobs) if args.max_concurrent_jobs else len(available_gpus)
    max_concurrent_jobs = max(1, min(max_concurrent_jobs, len(available_gpus)))

    pending_jobs = list(jobs)
    running_jobs: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    last_status_signature: tuple[int, int, int] | None = None
    last_status_emit_ts = 0.0
    while pending_jobs or running_jobs:
        launched_count = 0
        while pending_jobs and available_gpus and len(running_jobs) < max_concurrent_jobs:
            layer_job = pending_jobs.pop(0)
            gpu_id = available_gpus.pop(0)
            cmd = build_train_command(args, layer_job)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            layer_job["log_path"].parent.mkdir(parents=True, exist_ok=True)
            log_handle = layer_job["log_path"].open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=str(THIS_DIR),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running_jobs.append(
                {
                    **layer_job,
                    "gpu_id": gpu_id,
                    "cmd": cmd,
                    "process": process,
                    "log_handle": log_handle,
                    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            launched_count += 1
            print(
                (
                    "[train-layerwise] launch | "
                    f"layer={layer_job['layer_surface_pair_key']}, gpu={gpu_id}, "
                    f"train_windows={int(layer_job['train_window_count'])}, "
                    f"val_windows={int(layer_job['val_window_count'])}, "
                    f"log={layer_job['log_path']}"
                ),
                flush=True,
            )

        time.sleep(max(float(args.poll_seconds), 0.5))
        still_running: list[dict[str, Any]] = []
        completed_in_cycle = 0
        for item in running_jobs:
            process: subprocess.Popen = item["process"]
            return_code = process.poll()
            if return_code is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            available_gpus.append(int(item["gpu_id"]))
            best_checkpoint = Path(item["best_checkpoint"])
            completed_rows.append(
                {
                    LAYER_SURFACE_PAIR_KEY_COL: item["layer_surface_pair_key"],
                    "SanitizedLayerKey": item["sanitized_key"],
                    "RunName": item["run_name"],
                    "RunDir": str(item["run_dir"]),
                    "BestCheckpoint": str(best_checkpoint),
                    "TrainWindowCount": int(item["train_window_count"]),
                    "ValWindowCount": int(item["val_window_count"]),
                    "Status": "completed" if return_code == 0 and best_checkpoint.exists() else "failed",
                    "Reason": "" if return_code == 0 and best_checkpoint.exists() else f"return_code={return_code}",
                    "GPU": int(item["gpu_id"]),
                    "StartedAt": item["started_at"],
                    "FinishedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "LogPath": str(item["log_path"]),
                    "Command": " ".join(item["cmd"]),
                }
            )
            completed_in_cycle += 1
            print(
                (
                    "[train-layerwise] finish | "
                    f"layer={item['layer_surface_pair_key']}, gpu={item['gpu_id']}, "
                    f"status={'completed' if return_code == 0 and best_checkpoint.exists() else 'failed'}, "
                    f"return_code={return_code}, checkpoint_exists={best_checkpoint.exists()}, "
                    f"log={item['log_path']}"
                ),
                flush=True,
            )
        running_jobs = still_running
        available_gpus.sort()
        status_signature = (len(pending_jobs), len(running_jobs), len(completed_rows))
        now_ts = time.time()
        if (
            launched_count > 0
            or completed_in_cycle > 0
            or status_signature != last_status_signature
            or (now_ts - last_status_emit_ts) >= 30.0
        ):
            failed_count = int(sum(1 for row in completed_rows if str(row.get("Status", "")) == "failed"))
            print(
                (
                    "[train-layerwise] progress | "
                    f"done={precompleted_count + len(completed_rows)}/{total_layer_count}, "
                    f"pending={len(pending_jobs)}, running={len(running_jobs)}, failed={failed_count}, "
                    f"available_gpus={available_gpus}, running_jobs={format_running_jobs(running_jobs)}"
                ),
                flush=True,
            )
            last_status_signature = status_signature
            last_status_emit_ts = now_ts

    merged_summary_df = summary_df.copy()
    if completed_rows:
        completed_df = pd.DataFrame(completed_rows)
        completed_lookup = {
            str(row[LAYER_SURFACE_PAIR_KEY_COL]): row
            for row in completed_df.to_dict("records")
        }
        merged_rows: list[dict[str, Any]] = []
        for row in merged_summary_df.to_dict("records"):
            completed_row = completed_lookup.get(str(row.get(LAYER_SURFACE_PAIR_KEY_COL, "")))
            if completed_row:
                updated = dict(row)
                updated.update(completed_row)
                merged_rows.append(updated)
            else:
                merged_rows.append(row)
        completed_only_keys = {
            str(row.get(LAYER_SURFACE_PAIR_KEY_COL, ""))
            for row in merged_rows
        }
        for row in completed_df.to_dict("records"):
            if str(row.get(LAYER_SURFACE_PAIR_KEY_COL, "")) not in completed_only_keys:
                merged_rows.append(row)
        merged_summary_df = pd.DataFrame(merged_rows)

    # 补回已存在但未重训的模型到注册表。
    registry_models: dict[str, str] = {}
    layer_decode_settings: dict[str, dict[str, Any]] = {}
    for _, row in merged_summary_df.iterrows():
        status = str(row.get("Status", ""))
        best_checkpoint = Path(str(row.get("BestCheckpoint", "")))
        if status in {"completed", "skipped_existing"} and best_checkpoint.exists():
            layer_key = str(row[LAYER_SURFACE_PAIR_KEY_COL])
            registry_models[layer_key] = str(best_checkpoint.resolve())
            training_summary_json = Path(str(row.get("RunDir", ""))) / "training_summary.json"
            if training_summary_json.exists():
                try:
                    payload = json.loads(training_summary_json.read_text(encoding="utf-8"))
                    raw_decode_config = payload.get("recommended_decode_config", {})
                    if isinstance(raw_decode_config, dict) and raw_decode_config:
                        layer_decode_settings[layer_key] = raw_decode_config
                except Exception:
                    pass

    registry_output_py = (
        Path(args.registry_output_py)
        if args.registry_output_py
        else output_root / f"{args.run_name_prefix}_layer_model_registry.py"
    )
    split_run_dir = Path(args.split_run_dir).resolve()
    split_summary_json = split_run_dir / "split_summary.json"
    train_manifest_csv = split_run_dir / "train_manifest.csv"
    val_manifest_csv = split_run_dir / "val_manifest.csv"
    test_manifest_csv = split_run_dir / "test_manifest.csv"
    training_identity = build_training_identity_payload(args, layer_training_config=layer_training_config)
    write_layer_model_registry_py(
        output_path=registry_output_py,
        models=registry_models,
        metadata={
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split_run_dir": str(split_run_dir),
            "unit_dfn_root": str(Path(args.unit_dfn_root).resolve()),
            "output_root": str(output_root.resolve()),
            "layer_training_config_py": layer_training_config["config_py"],
            "layer_training_config_sha256": layer_training_config["config_sha256"],
            "allowed_layer_surface_pair_keys": layer_training_config["allowed_layer_surface_pair_keys"],
            "layer_training_config_metadata": layer_training_config["metadata"],
            "layer_training_overrides": layer_training_config["layer_training_overrides"],
            "layer_density_prior_summary_csv": str(density_prior_summary_csv.resolve()),
            "density_prior_confidence_weighted": bool(train_unit_confidence_map),
            "train_unit_confidence_count": int(len(train_unit_confidence_map)),
            "train_unit_confidence_mean": (
                float(sum(train_unit_confidence_map.values()) / len(train_unit_confidence_map))
                if train_unit_confidence_map
                else 0.0
            ),
            "split_summary_json": str(split_summary_json.resolve()),
            "split_summary_sha256": compute_file_sha256(split_summary_json),
            "train_manifest_csv": str(train_manifest_csv.resolve()),
            "train_manifest_sha256": compute_file_sha256(train_manifest_csv),
            "val_manifest_csv": str(val_manifest_csv.resolve()),
            "val_manifest_sha256": compute_file_sha256(val_manifest_csv),
            "test_manifest_csv": str(test_manifest_csv.resolve()),
            "test_manifest_sha256": compute_file_sha256(test_manifest_csv),
            "training_identity": training_identity,
        },
        layer_decode_settings=layer_decode_settings,
        layer_density_priors=layer_density_priors,
    )

    summary_csv = output_root / f"{args.run_name_prefix}_layer_training_summary.csv"
    summary_json = output_root / f"{args.run_name_prefix}_layer_training_summary.json"
    write_csv_utf8(merged_summary_df, summary_csv)
    payload = {
        "split_run_dir": str(split_run_dir),
        "output_root": str(output_root.resolve()),
        "registry_output_py": str(registry_output_py.resolve()),
        "gpu_ids": [int(gpu_id) for gpu_id in args.gpu_ids],
        "max_concurrent_jobs": int(max_concurrent_jobs),
        "requested_layer_count": int(len(summary_df)),
        "trained_or_reused_layer_count": int(len(registry_models)),
        "layer_training_config_py": layer_training_config["config_py"],
        "layer_training_config_sha256": layer_training_config["config_sha256"],
        "allowed_layer_surface_pair_keys": layer_training_config["allowed_layer_surface_pair_keys"],
        "layer_training_config_metadata": layer_training_config["metadata"],
        "layer_training_overrides": layer_training_config["layer_training_overrides"],
        "layer_decode_settings_count": int(len(layer_decode_settings)),
        "layer_density_prior_count": int(len(layer_density_priors)),
        "layer_density_prior_summary_csv": str(density_prior_summary_csv.resolve()),
        "density_prior_confidence_weighted": bool(train_unit_confidence_map),
        "train_unit_confidence_count": int(len(train_unit_confidence_map)),
        "train_unit_confidence_mean": (
            float(sum(train_unit_confidence_map.values()) / len(train_unit_confidence_map))
            if train_unit_confidence_map
            else 0.0
        ),
        "summary_csv": str(summary_csv.resolve()),
        "split_summary_json": str(split_summary_json.resolve()),
        "split_summary_sha256": compute_file_sha256(split_summary_json),
        "train_manifest_csv": str(train_manifest_csv.resolve()),
        "train_manifest_sha256": compute_file_sha256(train_manifest_csv),
        "val_manifest_csv": str(val_manifest_csv.resolve()),
        "val_manifest_sha256": compute_file_sha256(val_manifest_csv),
        "test_manifest_csv": str(test_manifest_csv.resolve()),
        "test_manifest_sha256": compute_file_sha256(test_manifest_csv),
        "training_identity": training_identity,
    }
    write_json(summary_json, payload)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - 分层并行训练",
        lines=[
            f"split_run_dir: {Path(args.split_run_dir).resolve()}",
            f"output_root: {output_root.resolve()}",
            f"registry_output_py: {registry_output_py.resolve()}",
            f"unit_dfn_root: {Path(args.unit_dfn_root).resolve()}",
            f"layer_training_config_py: {layer_training_config['config_py'] or 'None'}",
            f"gpu_ids: {args.gpu_ids}",
            f"max_concurrent_jobs: {max_concurrent_jobs}",
            f"trained_or_reused_layer_count: {len(registry_models)}",
            f"layer_density_prior_count: {len(layer_density_priors)}",
            f"layer_density_prior_summary_csv: {density_prior_summary_csv.resolve()}",
            f"density_prior_confidence_weighted: {bool(train_unit_confidence_map)}",
            f"train_unit_confidence_count: {len(train_unit_confidence_map)}",
            f"summary_csv: {summary_csv.resolve()}",
            f"summary_json: {summary_json.resolve()}",
        ],
    )

    print(f"registry_output_py: {registry_output_py.resolve()}")
    print(f"trained_or_reused_layer_count: {len(registry_models)}")
    print(f"summary_csv: {summary_csv.resolve()}")

    failed_rows = merged_summary_df[merged_summary_df.get("Status", pd.Series(dtype="object")).astype(str) == "failed"]
    if not failed_rows.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
