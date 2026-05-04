# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _extract_cli_int_arg(flag_name: str) -> int | None:
    for index, token in enumerate(sys.argv):
        if token != flag_name:
            continue
        if index + 1 >= len(sys.argv):
            return None
        try:
            return int(sys.argv[index + 1])
        except (TypeError, ValueError):
            return None
    return None


def _resolve_default_torch_num_threads() -> int:
    requested = _extract_cli_int_arg("--torch-num-threads")
    if requested is not None and requested > 0:
        return int(requested)
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, (cpu_count + 7) // 8))


def _apply_preimport_thread_env_defaults() -> int:
    num_threads = _resolve_default_torch_num_threads()
    for env_key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ.setdefault(env_key, str(num_threads))
    return int(num_threads)


PREIMPORT_TORCH_NUM_THREADS = _apply_preimport_thread_env_defaults()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from baseline_common import (
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SLOTS_PER_VOXEL,
    append_lines_to_docx,
    build_dense_targets_from_sparse_arrays,
    load_sparse_npz_arrays,
    write_csv_utf8,
    write_json,
)
from baseline_model import SparseInstanceBaselineUNet
from layer_model_registry import (
    LAYER_SURFACE_PAIR_KEY_COL,
    ensure_layer_surface_pair_key_column,
    sanitize_layer_surface_pair_key,
)


class SparseWindowBaselineDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        slots_per_voxel: int,
        layer_surface_pair_key: str | None = None,
        limit_samples: int | None = None,
        cache_raw_packages: bool = True,
    ) -> None:
        self.manifest_df = pd.read_csv(manifest_csv, encoding="utf-8-sig")
        self.manifest_df = ensure_layer_surface_pair_key_column(self.manifest_df, key_col=LAYER_SURFACE_PAIR_KEY_COL)
        self.layer_surface_pair_key = str(layer_surface_pair_key).strip() if layer_surface_pair_key else ""
        if self.layer_surface_pair_key:
            self.manifest_df = self.manifest_df[
                self.manifest_df[LAYER_SURFACE_PAIR_KEY_COL].astype(str) == self.layer_surface_pair_key
            ].copy()
        if limit_samples:
            self.manifest_df = self.manifest_df.head(int(limit_samples)).copy()
        self.manifest_df = self.manifest_df.reset_index(drop=True)
        self.rows = self.manifest_df.to_dict("records")
        self.slots_per_voxel = int(slots_per_voxel)
        self.cache_raw_packages = bool(cache_raw_packages)
        self._raw_package_cache: dict[str, dict[str, np.ndarray]] = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        if self.manifest_df.empty:
            raise ValueError(f"empty manifest: {manifest_csv}")

    def __len__(self) -> int:
        return int(len(self.rows))

    def _load_sparse_package(self, package_path: Path) -> dict[str, np.ndarray]:
        cache_key = str(package_path)
        if self.cache_raw_packages:
            cached = self._raw_package_cache.get(cache_key)
            if cached is not None:
                self.cache_hit_count += 1
                return cached
        sparse_arrays = load_sparse_npz_arrays(package_path)
        self.cache_miss_count += 1
        if self.cache_raw_packages:
            self._raw_package_cache[cache_key] = sparse_arrays
        return sparse_arrays

    def get_cache_stats(self) -> dict[str, int | bool]:
        return {
            "enabled": bool(self.cache_raw_packages),
            "cached_package_count": int(len(self._raw_package_cache)),
            "cache_hit_count": int(self.cache_hit_count),
            "cache_miss_count": int(self.cache_miss_count),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        package_path = Path(str(row["PackagePath"]))
        sparse_arrays = self._load_sparse_package(package_path)
        dense = build_dense_targets_from_sparse_arrays(sparse_arrays, slots_per_voxel=self.slots_per_voxel)
        return {
            "input_features": torch.from_numpy(dense["input_features"]).float(),
            "valid_z_mask": torch.from_numpy(dense["valid_z_mask"]).float(),
            "center_target": torch.from_numpy(dense["center_target"]).float(),
            "geom_target": torch.from_numpy(dense["geom_target"]).float(),
            "weight_target": torch.from_numpy(dense["weight_target"]).float(),
            "count_target": torch.from_numpy(dense["count_target"]).float(),
            "overflow_instance_count": torch.from_numpy(dense["overflow_instance_count"]).long(),
            "meta": {
                "SampleID": str(row["SampleID"]),
                "UnitID": str(row["UnitID"]),
                "GeoIntervalKey": str(row["GeoIntervalKey"]),
                LAYER_SURFACE_PAIR_KEY_COL: str(row.get(LAYER_SURFACE_PAIR_KEY_COL, "")),
                "WindowIndex": int(row["WindowIndex"]),
                "PackagePath": str(package_path),
            },
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "input_features",
        "valid_z_mask",
        "center_target",
        "geom_target",
        "weight_target",
        "count_target",
        "overflow_instance_count",
    ]
    collated = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    collated["meta"] = [item["meta"] for item in batch]
    return collated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 G-DFN 监督 baseline。")
    parser.add_argument("--split-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "训练结果")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--layer-surface-pair-key", type=str)
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--slots-per-voxel", type=int, default=DEFAULT_SLOTS_PER_VOXEL)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--torch-num-interop-threads", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--train-limit-samples", type=int)
    parser.add_argument("--val-limit-samples", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
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


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_thread_settings(args: argparse.Namespace) -> tuple[int, int]:
    num_threads = int(args.torch_num_threads) if int(args.torch_num_threads) > 0 else int(PREIMPORT_TORCH_NUM_THREADS)
    num_interop_threads = int(args.torch_num_interop_threads) if int(args.torch_num_interop_threads) > 0 else 1
    return max(1, num_threads), max(1, num_interop_threads)


def apply_runtime_thread_settings(num_threads: int, num_interop_threads: int) -> tuple[int, int]:
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(int(num_interop_threads))
    except RuntimeError as exc:
        print(f"[ThreadConfig] skip set_num_interop_threads({num_interop_threads}): {exc}")
    return int(torch.get_num_threads()), int(torch.get_num_interop_threads())


def resolve_run_dir(
    output_root: Path,
    run_name: str | None,
    resume_checkpoint: Path | None,
    layer_surface_pair_key: str | None,
) -> Path:
    if resume_checkpoint is None:
        if run_name:
            return Path(output_root) / str(run_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if layer_surface_pair_key:
            sanitized_key = sanitize_layer_surface_pair_key(layer_surface_pair_key)
            return Path(output_root) / f"baseline_{sanitized_key}_{timestamp}"
        return Path(output_root) / f"baseline_smoke_{timestamp}"

    checkpoint_run_dir = Path(resume_checkpoint).resolve().parent.parent
    if run_name is None:
        return checkpoint_run_dir

    requested_run_dir = (Path(output_root) / str(run_name)).resolve()
    if requested_run_dir != checkpoint_run_dir:
        raise ValueError(
            f"resume checkpoint belongs to run_dir={checkpoint_run_dir}, "
            f"but requested run_dir={requested_run_dir}. "
            "Omit --run-name to continue in place, or set it to the original run name."
        )
    return checkpoint_run_dir


def load_existing_history_rows(history_csv: Path) -> list[dict[str, Any]]:
    if not history_csv.exists():
        return []
    history_df = pd.read_csv(history_csv, encoding="utf-8-sig")
    if history_df.empty:
        return []
    return history_df.to_dict("records")


def infer_best_val_loss(
    history_rows: list[dict[str, Any]],
    resume_payload: dict[str, Any] | None,
) -> float:
    history_losses: list[float] = []
    for row in history_rows:
        value = row.get("val_loss_total")
        if value is None or pd.isna(value):
            continue
        history_losses.append(float(value))
    if history_losses:
        return min(history_losses)
    if resume_payload:
        if "best_val_loss" in resume_payload and resume_payload["best_val_loss"] is not None:
            return float(resume_payload["best_val_loss"])
        val_metrics = resume_payload.get("val_metrics") or {}
        if "loss_total" in val_metrics and val_metrics["loss_total"] is not None:
            return float(val_metrics["loss_total"])
    return math.inf


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 1e-8:
        return values.sum() * 0.0
    return (values * weights).sum() / denom


def build_loss_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "center_positive_weight": float(max(args.center_positive_weight, 1e-6)),
        "center_negative_weight": float(max(args.center_negative_weight, 1e-6)),
        "center_focal_gamma": float(max(args.center_focal_gamma, 0.0)),
        "count_positive_weight": float(max(args.count_positive_weight, 1e-6)),
        "count_negative_weight": float(max(args.count_negative_weight, 1e-6)),
        "center_loss_weight": float(max(args.center_loss_weight, 0.0)),
        "count_loss_weight": float(max(args.count_loss_weight, 0.0)),
    }


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    cosine = torch.sum(pred * target, dim=2).clamp(-1.0, 1.0)
    return weighted_mean(1.0 - cosine, weight)


def compute_loss_dict(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    center_target = batch["center_target"]
    geom_target = batch["geom_target"]
    weight_target = batch["weight_target"]
    count_target = batch["count_target"]
    valid_z_mask = batch["valid_z_mask"]

    batch_size, slots_per_voxel, nx, ny, nz = center_target.shape
    valid_center = valid_z_mask.view(batch_size, 1, 1, 1, nz)
    center_prob = torch.sigmoid(outputs["center_logits"])
    positive_center_mask = center_target > 0.5
    center_alpha = torch.where(
        positive_center_mask,
        torch.full_like(center_target, float(loss_config["center_positive_weight"])) * (1.0 + weight_target),
        torch.full_like(center_target, float(loss_config["center_negative_weight"])),
    )
    center_loss_map = F.binary_cross_entropy_with_logits(outputs["center_logits"], center_target, reduction="none")
    center_pt = torch.where(positive_center_mask, center_prob, 1.0 - center_prob)
    center_focal = torch.pow(1.0 - center_pt, float(loss_config["center_focal_gamma"]))
    center_loss = weighted_mean(center_loss_map * center_focal, valid_center * center_alpha)

    count_pred = outputs["count_pred"]
    count_mask = valid_z_mask.view(batch_size, 1, 1, 1, nz)
    positive_count_mask = count_target > 0.0
    count_alpha = torch.where(
        positive_count_mask,
        torch.full_like(count_target, float(loss_config["count_positive_weight"]))
        * (1.0 + torch.clamp(count_target, min=0.0, max=4.0) / 4.0),
        torch.full_like(count_target, float(loss_config["count_negative_weight"])),
    )
    count_loss = weighted_mean(F.smooth_l1_loss(count_pred, count_target, reduction="none"), count_mask * count_alpha)

    geom_mask_scalar = center_target * valid_center
    geom_mask_weight = geom_mask_scalar * torch.where(weight_target > 0.0, weight_target, torch.ones_like(weight_target))
    geom_pred = outputs["geom_pred"]

    offset_loss = weighted_mean(
        F.smooth_l1_loss(geom_pred[:, :, 0:3], geom_target[:, :, 0:3], reduction="none").mean(dim=2),
        geom_mask_weight,
    )
    normal_loss = cosine_loss(geom_pred[:, :, 3:6], geom_target[:, :, 3:6], geom_mask_weight)
    udir_loss = cosine_loss(geom_pred[:, :, 6:9], geom_target[:, :, 6:9], geom_mask_weight)
    size_loss = weighted_mean(
        F.smooth_l1_loss(geom_pred[:, :, 9:11], geom_target[:, :, 9:11], reduction="none").mean(dim=2),
        geom_mask_weight,
    )
    confidence_loss = weighted_mean(
        F.smooth_l1_loss(geom_pred[:, :, 11], geom_target[:, :, 11], reduction="none"),
        geom_mask_weight,
    )

    total_loss = (
        float(loss_config["center_loss_weight"]) * center_loss
        + float(loss_config["count_loss_weight"]) * count_loss
        + 1.2 * offset_loss
        + 0.4 * normal_loss
        + 0.4 * udir_loss
        + 0.15 * size_loss
        + 0.10 * confidence_loss
    )

    with torch.no_grad():
        center_prob = torch.sigmoid(outputs["center_logits"])
        center_pred = (center_prob >= 0.5).float() * valid_center
        center_true = center_target * valid_center
        tp = float((center_pred * center_true).sum().detach().cpu())
        pred_pos = float(center_pred.sum().detach().cpu())
        true_pos = float(center_true.sum().detach().cpu())
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / true_pos if true_pos > 0 else 0.0
        count_mae = float((torch.abs(count_pred - count_target) * count_mask).sum().detach().cpu() / max(float(count_mask.sum().detach().cpu()), 1.0))

    metrics = {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_center": float(center_loss.detach().cpu()),
        "loss_count": float(count_loss.detach().cpu()),
        "loss_offset": float(offset_loss.detach().cpu()),
        "loss_normal": float(normal_loss.detach().cpu()),
        "loss_udir": float(udir_loss.detach().cpu()),
        "loss_size": float(size_loss.detach().cpu()),
        "loss_confidence": float(confidence_loss.detach().cpu()),
        "center_precision": precision,
        "center_recall": recall,
        "count_mae": count_mae,
    }
    return total_loss, metrics


def safe_divide(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def f1_score_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    if (precision + recall) <= 0.0:
        return precision, recall, 0.0
    return precision, recall, (2.0 * precision * recall) / (precision + recall)


def build_calibration_threshold_grid(values: list[float]) -> list[float]:
    normalized = sorted({round(float(value), 6) for value in values if np.isfinite(value) and float(value) >= 0.0})
    return normalized or [0.7]


def load_model_checkpoint_for_eval(checkpoint_path: Path, device: torch.device) -> tuple[SparseInstanceBaselineUNet, dict[str, Any]]:
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


def calibrate_decode_config(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    center_thresholds: list[float],
    count_activation_thresholds: list[float],
    window_weight: float,
    max_steps: int | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate_pairs = [
        (float(center_threshold), float(count_activation_threshold))
        for center_threshold in build_calibration_threshold_grid(center_thresholds)
        for count_activation_threshold in build_calibration_threshold_grid(count_activation_thresholds)
    ]
    stats_map = {
        pair: {
            "voxel_tp": 0.0,
            "voxel_fp": 0.0,
            "voxel_fn": 0.0,
            "window_tp": 0.0,
            "window_fp": 0.0,
            "window_fn": 0.0,
        }
        for pair in candidate_pairs
    }
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_steps is not None and batch_idx > int(max_steps):
                break
            batch = {
                key: value.to(device, non_blocking=(device.type == "cuda")) if key != "meta" else value
                for key, value in batch.items()
            }
            outputs = model(batch["input_features"])
            center_probs = torch.sigmoid(outputs["center_logits"])
            max_center_probs = center_probs.amax(dim=1)
            count_pred = outputs["count_pred"].squeeze(1)
            valid_mask = batch["valid_z_mask"].view(batch["valid_z_mask"].shape[0], 1, 1, batch["valid_z_mask"].shape[1]) > 0.5
            true_active = batch["center_target"].amax(dim=1) > 0.5
            true_active = true_active & valid_mask
            true_window = true_active.reshape(true_active.shape[0], -1).any(dim=1)

            for pair in candidate_pairs:
                center_threshold, count_activation_threshold = pair
                pred_active = (max_center_probs >= float(center_threshold)) & (count_pred >= float(count_activation_threshold))
                pred_active = pred_active & valid_mask
                pred_window = pred_active.reshape(pred_active.shape[0], -1).any(dim=1)

                stats = stats_map[pair]
                stats["voxel_tp"] += float((pred_active & true_active).sum().detach().cpu())
                stats["voxel_fp"] += float((pred_active & (~true_active) & valid_mask).sum().detach().cpu())
                stats["voxel_fn"] += float(((~pred_active) & true_active).sum().detach().cpu())
                stats["window_tp"] += float((pred_window & true_window).sum().detach().cpu())
                stats["window_fp"] += float((pred_window & (~true_window)).sum().detach().cpu())
                stats["window_fn"] += float(((~pred_window) & true_window).sum().detach().cpu())

    rows: list[dict[str, Any]] = []
    for pair in candidate_pairs:
        center_threshold, count_activation_threshold = pair
        stats = stats_map[pair]
        voxel_precision, voxel_recall, voxel_f1 = f1_score_from_counts(
            tp=stats["voxel_tp"],
            fp=stats["voxel_fp"],
            fn=stats["voxel_fn"],
        )
        window_precision, window_recall, window_f1 = f1_score_from_counts(
            tp=stats["window_tp"],
            fp=stats["window_fp"],
            fn=stats["window_fn"],
        )
        score = (1.0 - float(window_weight)) * voxel_f1 + float(window_weight) * window_f1
        rows.append(
            {
                "center_threshold": float(center_threshold),
                "count_activation_threshold": float(count_activation_threshold),
                "voxel_precision": float(voxel_precision),
                "voxel_recall": float(voxel_recall),
                "voxel_f1": float(voxel_f1),
                "window_precision": float(window_precision),
                "window_recall": float(window_recall),
                "window_f1": float(window_f1),
                "score": float(score),
            }
        )
    calibration_df = pd.DataFrame(rows)
    if calibration_df.empty:
        fallback = {
            "center_threshold": 0.7,
            "count_activation_threshold": 0.5,
            "min_count_if_active": 1,
            "decode_mode": "strict",
            "relaxed_min_count": 1,
            "score": 0.0,
        }
        return fallback, calibration_df
    calibration_df = calibration_df.sort_values(
        [
            "score",
            "window_f1",
            "voxel_f1",
            "window_precision",
            "voxel_precision",
            "center_threshold",
            "count_activation_threshold",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    best_row = calibration_df.iloc[0]
    recommended = {
        "center_threshold": float(best_row["center_threshold"]),
        "count_activation_threshold": float(best_row["count_activation_threshold"]),
        "min_count_if_active": 1,
        "decode_mode": "strict",
        "relaxed_min_count": 1,
        "score": float(best_row["score"]),
        "voxel_f1": float(best_row["voxel_f1"]),
        "window_f1": float(best_row["window_f1"]),
    }
    return recommended, calibration_df


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    max_steps: int | None,
    epoch_idx: int,
    total_epochs: int,
    split_name: str,
    show_progress: bool,
    loss_config: dict[str, float],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and is_train)
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and is_train)
    metric_sum: dict[str, float] = {}
    step_count = 0
    overflow_sum = 0
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0

    total_steps = len(loader)
    if max_steps is not None:
        total_steps = min(int(total_steps), int(max_steps))
    progress = tqdm(
        total=total_steps,
        desc=f"{split_name} {epoch_idx}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
        disable=not bool(show_progress),
    )

    for batch_idx, batch in enumerate(loader, start=1):
        if max_steps and batch_idx > int(max_steps):
            break
        batch = {
            key: value.to(device, non_blocking=(device.type == "cuda")) if key != "meta" else value
            for key, value in batch.items()
        }
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["input_features"])
            loss, metrics = compute_loss_dict(outputs, batch, loss_config=loss_config)
        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        for key, value in metrics.items():
            metric_sum[key] = metric_sum.get(key, 0.0) + float(value)
        overflow_sum += int(batch["overflow_instance_count"].sum().detach().cpu())
        step_count += 1
        running_loss += float(metrics["loss_total"])
        running_precision += float(metrics["center_precision"])
        running_recall += float(metrics["center_recall"])
        progress.update(1)
        progress.set_postfix(
            loss=f"{running_loss / step_count:.4f}",
            p=f"{running_precision / step_count:.3f}",
            r=f"{running_recall / step_count:.3f}",
        )

    progress.close()

    if step_count <= 0:
        raise ValueError("no batches processed")
    metric_avg = {key: value / step_count for key, value in metric_sum.items()}
    metric_avg["step_count"] = float(step_count)
    metric_avg["overflow_instance_count"] = float(overflow_sum)
    return metric_avg


def main() -> None:
    args = build_parser().parse_args()
    torch_num_threads, torch_num_interop_threads = resolve_thread_settings(args)
    applied_torch_num_threads, applied_torch_num_interop_threads = apply_runtime_thread_settings(
        torch_num_threads,
        torch_num_interop_threads,
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    amp_enabled = (not bool(args.disable_amp)) and device.type == "cuda"
    show_progress = not bool(args.no_progress)
    additional_epochs = int(args.epochs)
    if additional_epochs <= 0:
        raise ValueError("--epochs must be >= 1")
    layer_surface_pair_key = str(args.layer_surface_pair_key).strip() if args.layer_surface_pair_key else ""
    loss_config = build_loss_config(args)

    split_run_dir = Path(args.split_run_dir)
    train_manifest_csv = split_run_dir / "train_manifest.csv"
    val_manifest_csv = split_run_dir / "val_manifest.csv"
    if not train_manifest_csv.exists() or not val_manifest_csv.exists():
        raise FileNotFoundError(f"split manifests not found in: {split_run_dir}")

    resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")
        try:
            resume_payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            resume_payload = torch.load(resume_checkpoint, map_location=device)
        checkpoint_split_run_dir = Path(str(resume_payload.get("split_run_dir", split_run_dir)))
        if checkpoint_split_run_dir.resolve() != split_run_dir.resolve():
            raise ValueError(
                f"resume checkpoint split_run_dir={checkpoint_split_run_dir} "
                f"does not match current split_run_dir={split_run_dir}"
            )
        checkpoint_layer_key = str(resume_payload.get("layer_surface_pair_key", "")).strip()
        if checkpoint_layer_key != layer_surface_pair_key:
            raise ValueError(
                f"resume checkpoint layer_surface_pair_key={checkpoint_layer_key!r} "
                f"does not match current layer_surface_pair_key={layer_surface_pair_key!r}"
            )

    cache_raw_packages = (not bool(args.no_cache_raw_packages)) and int(args.num_workers) == 0
    train_dataset = SparseWindowBaselineDataset(
        manifest_csv=train_manifest_csv,
        slots_per_voxel=int(args.slots_per_voxel),
        layer_surface_pair_key=layer_surface_pair_key,
        limit_samples=args.train_limit_samples,
        cache_raw_packages=cache_raw_packages,
    )
    val_dataset = SparseWindowBaselineDataset(
        manifest_csv=val_manifest_csv,
        slots_per_voxel=int(args.slots_per_voxel),
        layer_surface_pair_key=layer_surface_pair_key,
        limit_samples=args.val_limit_samples,
        cache_raw_packages=cache_raw_packages,
    )
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(args.num_workers),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_batch,
    }
    val_loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_batch,
    }
    if int(args.num_workers) > 0:
        train_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs,
    )

    example_input_channels = int(train_dataset[0]["input_features"].shape[0])
    if resume_payload is not None:
        model_config = resume_payload.get("model_config") or {}
        checkpoint_in_channels = int(model_config.get("in_channels", example_input_channels))
        checkpoint_slots = int(model_config.get("slots_per_voxel", int(args.slots_per_voxel)))
        checkpoint_base_channels = int(model_config.get("base_channels", int(args.base_channels)))
        if checkpoint_in_channels != example_input_channels:
            raise ValueError(
                f"resume checkpoint in_channels={checkpoint_in_channels} "
                f"does not match dataset in_channels={example_input_channels}"
            )
        if checkpoint_slots != int(args.slots_per_voxel):
            raise ValueError(
                f"resume checkpoint slots_per_voxel={checkpoint_slots} "
                f"does not match argument slots_per_voxel={int(args.slots_per_voxel)}"
            )
        if checkpoint_base_channels != int(args.base_channels):
            raise ValueError(
                f"resume checkpoint base_channels={checkpoint_base_channels} "
                f"does not match argument base_channels={int(args.base_channels)}"
            )
    model = SparseInstanceBaselineUNet(
        in_channels=example_input_channels,
        slots_per_voxel=int(args.slots_per_voxel),
        base_channels=int(args.base_channels),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    run_dir = resolve_run_dir(
        output_root=Path(args.output_root),
        run_name=args.run_name,
        resume_checkpoint=resume_checkpoint,
        layer_surface_pair_key=layer_surface_pair_key,
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_csv = run_dir / "history.csv"

    history_rows = load_existing_history_rows(history_csv)
    best_val_loss = infer_best_val_loss(history_rows, resume_payload)
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    last_checkpoint_path = checkpoint_dir / "last_model.pt"
    resumed_from_epoch = 0
    start_epoch = 1

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        optimizer_state = resume_payload.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        resumed_from_epoch = int(resume_payload.get("epoch", 0))
        start_epoch = resumed_from_epoch + 1
        if not history_rows and resumed_from_epoch > 0:
            print(
                f"[Resume] history.csv not found in {run_dir}, "
                f"continuing from checkpoint epoch {resumed_from_epoch} only."
            )

    final_epoch = resumed_from_epoch + additional_epochs

    for epoch_idx in range(start_epoch, final_epoch + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            max_steps=args.max_train_steps,
            epoch_idx=epoch_idx,
            total_epochs=final_epoch,
            split_name="train",
            show_progress=show_progress,
            loss_config=loss_config,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                amp_enabled=False,
                max_steps=args.max_val_steps,
                epoch_idx=epoch_idx,
                total_epochs=final_epoch,
                split_name="val",
                show_progress=show_progress,
                loss_config=loss_config,
            )

        row = {"epoch": epoch_idx}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history_rows.append(row)

        checkpoint_payload = {
            "epoch": int(epoch_idx),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": {
                "in_channels": example_input_channels,
                "slots_per_voxel": int(args.slots_per_voxel),
                "base_channels": int(args.base_channels),
                "dx": 12.5,
                "dy": 12.5,
                "dz": 0.2,
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": float(best_val_loss),
            "split_run_dir": str(split_run_dir),
            "layer_surface_pair_key": layer_surface_pair_key,
        }
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = float(val_metrics["loss_total"])
            checkpoint_payload["best_val_loss"] = float(best_val_loss)
            torch.save(checkpoint_payload, best_checkpoint_path)
        else:
            checkpoint_payload["best_val_loss"] = float(best_val_loss)
        torch.save(checkpoint_payload, last_checkpoint_path)
        print(
            f"[Epoch {epoch_idx}/{final_epoch}] "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"val_loss={val_metrics['loss_total']:.4f} "
            f"train_p={train_metrics['center_precision']:.3f} "
            f"train_r={train_metrics['center_recall']:.3f} "
            f"val_p={val_metrics['center_precision']:.3f} "
            f"val_r={val_metrics['center_recall']:.3f} "
            f"best_val={best_val_loss:.4f}"
        )

    history_df = pd.DataFrame(history_rows)
    if not history_df.empty and "epoch" in history_df.columns:
        history_df = history_df.sort_values("epoch").reset_index(drop=True)
    write_csv_utf8(history_df, history_csv)

    calibration_model = model
    calibration_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
    calibration_checkpoint_payload: dict[str, Any] | None = None
    if calibration_checkpoint_path.exists():
        calibration_model, calibration_checkpoint_payload = load_model_checkpoint_for_eval(calibration_checkpoint_path, device)
    recommended_decode_config, calibration_df = calibrate_decode_config(
        model=calibration_model,
        loader=val_loader,
        device=device,
        center_thresholds=list(args.calibration_center_thresholds),
        count_activation_thresholds=list(args.calibration_count_thresholds),
        window_weight=float(args.calibration_window_weight),
        max_steps=args.max_val_steps,
    )
    calibration_csv = run_dir / "decode_calibration_summary.csv"
    if not calibration_df.empty:
        write_csv_utf8(calibration_df, calibration_csv)

    train_cache_stats = train_dataset.get_cache_stats()
    val_cache_stats = val_dataset.get_cache_stats()
    summary = {
        "run_dir": str(run_dir),
        "split_run_dir": str(split_run_dir),
        "layer_surface_pair_key": layer_surface_pair_key,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resumed_from_epoch": int(resumed_from_epoch),
        "additional_epochs": int(additional_epochs),
        "final_epoch": int(final_epoch),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "epochs": int(final_epoch),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "cache_raw_packages": bool(cache_raw_packages),
        "train_cache_stats": train_cache_stats,
        "val_cache_stats": val_cache_stats,
        "torch_num_threads": int(applied_torch_num_threads),
        "torch_num_interop_threads": int(applied_torch_num_interop_threads),
        "loss_config": loss_config,
        "train_sample_count": int(len(train_dataset)),
        "val_sample_count": int(len(val_dataset)),
        "best_val_loss": float(best_val_loss),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "history_csv": str(history_csv),
        "calibration_checkpoint": str(calibration_checkpoint_path),
        "calibration_summary_csv": str(calibration_csv),
        "recommended_decode_config": recommended_decode_config,
        "calibration_center_thresholds": [float(value) for value in args.calibration_center_thresholds],
        "calibration_count_thresholds": [float(value) for value in args.calibration_count_thresholds],
        "calibration_window_weight": float(args.calibration_window_weight),
    }
    if calibration_checkpoint_payload is not None:
        summary["calibration_checkpoint_epoch"] = int(calibration_checkpoint_payload.get("epoch", 0))
    if history_rows:
        summary["final_train_loss"] = float(history_rows[-1]["train_loss_total"])
        summary["final_val_loss"] = float(history_rows[-1]["val_loss_total"])
        summary["final_val_center_precision"] = float(history_rows[-1]["val_center_precision"])
        summary["final_val_center_recall"] = float(history_rows[-1]["val_center_recall"])
    summary_path = run_dir / "training_summary.json"
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - 监督训练",
        lines=[
            f"split_run_dir: {split_run_dir}",
            f"run_dir: {run_dir}",
            f"layer_surface_pair_key: {layer_surface_pair_key if layer_surface_pair_key else 'ALL'}",
            f"resume_checkpoint: {resume_checkpoint if resume_checkpoint else 'None'}",
            f"resumed_from_epoch: {resumed_from_epoch}",
            f"additional_epochs: {additional_epochs}",
            f"final_epoch: {final_epoch}",
            f"device: {device}",
            f"amp_enabled: {amp_enabled}",
            f"epochs: {final_epoch}",
            f"batch_size: {args.batch_size}",
            f"num_workers: {args.num_workers}",
            f"cache_raw_packages: {cache_raw_packages}",
            f"torch_num_threads: {applied_torch_num_threads}",
            f"torch_num_interop_threads: {applied_torch_num_interop_threads}",
            f"train_sample_count: {len(train_dataset)}",
            f"val_sample_count: {len(val_dataset)}",
            f"best_val_loss: {summary['best_val_loss']}",
            f"best_checkpoint: {best_checkpoint_path}",
            f"recommended_decode_config: {recommended_decode_config}",
            f"decode_calibration_summary_csv: {calibration_csv}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"run_dir: {run_dir}")
    print(f"layer_surface_pair_key: {layer_surface_pair_key if layer_surface_pair_key else 'ALL'}")
    print(f"resume_checkpoint: {resume_checkpoint if resume_checkpoint else 'None'}")
    print(f"resumed_from_epoch: {resumed_from_epoch}")
    print(f"final_epoch: {final_epoch}")
    print(f"cache_raw_packages: {cache_raw_packages}")
    print(f"torch_num_threads: {applied_torch_num_threads}")
    print(f"torch_num_interop_threads: {applied_torch_num_interop_threads}")
    print(f"best_checkpoint: {best_checkpoint_path}")
    print(f"best_val_loss: {summary['best_val_loss']}")
    print(f"recommended_decode_config: {recommended_decode_config}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
