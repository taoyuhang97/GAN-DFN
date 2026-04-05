# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

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
    build_dense_targets_from_sparse_npz,
    write_csv_utf8,
    write_json,
)
from baseline_model import SparseInstanceBaselineUNet


class SparseWindowBaselineDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        slots_per_voxel: int,
        limit_samples: int | None = None,
    ) -> None:
        import pandas as pd

        self.manifest_df = pd.read_csv(manifest_csv, encoding="utf-8-sig")
        if limit_samples:
            self.manifest_df = self.manifest_df.head(int(limit_samples)).copy()
        self.manifest_df = self.manifest_df.reset_index(drop=True)
        self.slots_per_voxel = int(slots_per_voxel)
        if self.manifest_df.empty:
            raise ValueError(f"empty manifest: {manifest_csv}")

    def __len__(self) -> int:
        return int(len(self.manifest_df))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest_df.iloc[int(index)]
        package_path = Path(str(row["PackagePath"]))
        dense = build_dense_targets_from_sparse_npz(package_path, slots_per_voxel=self.slots_per_voxel)
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
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--slots-per-voxel", type=int, default=DEFAULT_SLOTS_PER_VOXEL)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--no-progress", action="store_true")
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


def resolve_run_dir(
    output_root: Path,
    run_name: str | None,
    resume_checkpoint: Path | None,
) -> Path:
    if resume_checkpoint is None:
        if run_name:
            return Path(output_root) / str(run_name)
        return Path(output_root) / f"baseline_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    cosine = torch.sum(pred * target, dim=2).clamp(-1.0, 1.0)
    return weighted_mean(1.0 - cosine, weight)


def compute_loss_dict(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    center_target = batch["center_target"]
    geom_target = batch["geom_target"]
    weight_target = batch["weight_target"]
    count_target = batch["count_target"]
    valid_z_mask = batch["valid_z_mask"]

    batch_size, slots_per_voxel, nx, ny, nz = center_target.shape
    valid_center = valid_z_mask.view(batch_size, 1, 1, 1, nz)
    valid_geom = valid_z_mask.view(batch_size, 1, 1, 1, 1, nz)
    center_weight = valid_center * torch.where(center_target > 0.5, 1.0 + 2.0 * weight_target, torch.ones_like(weight_target))
    center_loss_map = F.binary_cross_entropy_with_logits(outputs["center_logits"], center_target, reduction="none")
    center_loss = weighted_mean(center_loss_map, center_weight)

    count_pred = outputs["count_pred"]
    count_mask = valid_z_mask.view(batch_size, 1, 1, 1, nz)
    count_loss = weighted_mean(F.smooth_l1_loss(count_pred, count_target, reduction="none"), count_mask)

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
        1.0 * center_loss
        + 0.25 * count_loss
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
        batch = {key: value.to(device) if key != "meta" else value for key, value in batch.items()}
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["input_features"])
            loss, metrics = compute_loss_dict(outputs, batch)
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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    amp_enabled = (not bool(args.disable_amp)) and device.type == "cuda"
    show_progress = not bool(args.no_progress)
    additional_epochs = int(args.epochs)
    if additional_epochs <= 0:
        raise ValueError("--epochs must be >= 1")

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

    train_dataset = SparseWindowBaselineDataset(
        manifest_csv=train_manifest_csv,
        slots_per_voxel=int(args.slots_per_voxel),
        limit_samples=args.train_limit_samples,
    )
    val_dataset = SparseWindowBaselineDataset(
        manifest_csv=val_manifest_csv,
        slots_per_voxel=int(args.slots_per_voxel),
        limit_samples=args.val_limit_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
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

    summary = {
        "run_dir": str(run_dir),
        "split_run_dir": str(split_run_dir),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resumed_from_epoch": int(resumed_from_epoch),
        "additional_epochs": int(additional_epochs),
        "final_epoch": int(final_epoch),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "epochs": int(final_epoch),
        "batch_size": int(args.batch_size),
        "train_sample_count": int(len(train_dataset)),
        "val_sample_count": int(len(val_dataset)),
        "best_val_loss": float(best_val_loss),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "history_csv": str(history_csv),
    }
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
            f"resume_checkpoint: {resume_checkpoint if resume_checkpoint else 'None'}",
            f"resumed_from_epoch: {resumed_from_epoch}",
            f"additional_epochs: {additional_epochs}",
            f"final_epoch: {final_epoch}",
            f"device: {device}",
            f"amp_enabled: {amp_enabled}",
            f"epochs: {final_epoch}",
            f"batch_size: {args.batch_size}",
            f"train_sample_count: {len(train_dataset)}",
            f"val_sample_count: {len(val_dataset)}",
            f"best_val_loss: {summary['best_val_loss']}",
            f"best_checkpoint: {best_checkpoint_path}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"run_dir: {run_dir}")
    print(f"resume_checkpoint: {resume_checkpoint if resume_checkpoint else 'None'}")
    print(f"resumed_from_epoch: {resumed_from_epoch}")
    print(f"final_epoch: {final_epoch}")
    print(f"best_checkpoint: {best_checkpoint_path}")
    print(f"best_val_loss: {summary['best_val_loss']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
