# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from baseline_common import (
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    append_lines_to_docx,
    read_csv_utf8,
    write_csv_utf8,
    write_json,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_SPLIT_RUN_DIR = DEFAULT_OUTPUT_ROOT / "数据切分" / "phase1_unit_split_v1_20260330"

# Keep one distant unit plus three patch-rich moderate-size units.
DEFAULT_TRAIN_UNITS = ["BX49_BY5", "BX60_BY54", "BX60_BY55", "BX60_BY52"]
DEFAULT_VAL_UNITS = ["BX61_BY50"]
DEFAULT_TEST_UNITS = ["BX61_BY49"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a CPU-friendly quick subset split for the supervised G-DFN baseline."
    )
    parser.add_argument("--base-split-run-dir", type=Path, default=DEFAULT_BASE_SPLIT_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "快速子集")
    parser.add_argument(
        "--run-name",
        type=str,
        default=f"quick_cpu_subset_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--train-units", nargs="+", default=list(DEFAULT_TRAIN_UNITS))
    parser.add_argument("--val-units", nargs="+", default=list(DEFAULT_VAL_UNITS))
    parser.add_argument("--test-units", nargs="+", default=list(DEFAULT_TEST_UNITS))
    parser.add_argument("--recommended-epochs", type=int, default=3)
    parser.add_argument("--estimated-sec-per-step", type=float, default=4.2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def filter_manifest(manifest_df: pd.DataFrame, units: list[str]) -> pd.DataFrame:
    selected = [str(unit_id) for unit_id in units]
    return manifest_df[manifest_df["UnitID"].astype(str).isin(selected)].copy()


def build_unit_split_table(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_name, manifest_df in split_frames.items():
        grouped = (
            manifest_df.groupby("UnitID", dropna=False)
            .agg(
                WindowCount=("SampleID", "count"),
                PositiveWindowCount=("IsPositiveWindow", "sum"),
                PatchCount=("PatchCount", "sum"),
            )
            .reset_index()
        )
        grouped["Split"] = split_name
        rows.append(grouped)
    unit_split_df = pd.concat(rows, ignore_index=True)
    return unit_split_df[["UnitID", "Split", "WindowCount", "PositiveWindowCount", "PatchCount"]].copy()


def build_command_text(
    run_dir: Path,
    recommended_epochs: int,
    batch_size: int,
    base_channels: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> str:
    train_script = THIS_DIR / "train_supervised_baseline.py"
    eval_script = THIS_DIR / "infer_and_evaluate_baseline.py"
    run_date = datetime.now().strftime("%Y%m%d")
    train_command = " ".join(
        [
            "py -3.12",
            f"\"{train_script}\"",
            f"--split-run-dir \"{run_dir}\"",
            f"--run-name \"quick_cpu_train_{run_date}\"",
            f"--epochs {int(recommended_epochs)}",
            f"--batch-size {int(batch_size)}",
            f"--base-channels {int(base_channels)}",
            f"--learning-rate {float(learning_rate)}",
            f"--weight-decay {float(weight_decay)}",
            f"--device {device}",
        ]
    )
    eval_command = " ".join(
        [
            "py -3.12",
            f"\"{eval_script}\"",
            f"--split-run-dir \"{run_dir}\"",
            "--checkpoint \"<训练输出目录>\\checkpoints\\best_model.pt\"",
            "--split-name test",
            "--center-threshold 0.7",
            "--max-total-patches-per-window 256",
            f"--device {device}",
            f"--run-name \"quick_cpu_eval_{run_date}\"",
        ]
    )
    return "\n".join(
        [
            "# Quick CPU train",
            train_command,
            "",
            "# Quick CPU eval",
            eval_command,
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = read_csv_utf8(Path(args.base_split_run_dir) / "train_manifest.csv")
    val_manifest = read_csv_utf8(Path(args.base_split_run_dir) / "val_manifest.csv")
    test_manifest = read_csv_utf8(Path(args.base_split_run_dir) / "test_manifest.csv")

    split_frames = {
        "train": filter_manifest(train_manifest, list(args.train_units)),
        "val": filter_manifest(val_manifest, list(args.val_units)),
        "test": filter_manifest(test_manifest, list(args.test_units)),
    }
    if any(frame.empty for frame in split_frames.values()):
        raise ValueError("quick subset contains empty split, please check selected units")

    write_csv_utf8(split_frames["train"], run_dir / "train_manifest.csv")
    write_csv_utf8(split_frames["val"], run_dir / "val_manifest.csv")
    write_csv_utf8(split_frames["test"], run_dir / "test_manifest.csv")

    unit_split_df = build_unit_split_table(split_frames)
    write_csv_utf8(unit_split_df, run_dir / "unit_split.csv")

    train_steps = int(len(split_frames["train"]))
    val_steps = int(len(split_frames["val"]))
    test_steps = int(len(split_frames["test"]))
    epoch_steps = train_steps + val_steps
    estimated_sec_per_step = float(args.estimated_sec_per_step)
    recommended_epochs = int(args.recommended_epochs)
    estimated_epoch_minutes = epoch_steps * estimated_sec_per_step / 60.0
    estimated_total_minutes = recommended_epochs * estimated_epoch_minutes
    estimated_test_minutes = test_steps * estimated_sec_per_step / 60.0

    summary = {
        "base_split_run_dir": str(args.base_split_run_dir),
        "run_dir": str(run_dir),
        "train_units": list(args.train_units),
        "val_units": list(args.val_units),
        "test_units": list(args.test_units),
        "train_window_count": train_steps,
        "val_window_count": val_steps,
        "test_window_count": test_steps,
        "train_patch_count": int(split_frames["train"]["PatchCount"].sum()),
        "val_patch_count": int(split_frames["val"]["PatchCount"].sum()),
        "test_patch_count": int(split_frames["test"]["PatchCount"].sum()),
        "recommended_epochs": recommended_epochs,
        "estimated_sec_per_step": estimated_sec_per_step,
        "estimated_epoch_minutes": round(estimated_epoch_minutes, 2),
        "estimated_total_minutes": round(estimated_total_minutes, 2),
        "estimated_test_minutes": round(estimated_test_minutes, 2),
        "recommended_batch_size": int(args.batch_size),
        "recommended_base_channels": int(args.base_channels),
        "recommended_learning_rate": float(args.learning_rate),
        "recommended_weight_decay": float(args.weight_decay),
        "recommended_device": str(args.device),
    }
    summary_path = run_dir / "quick_subset_summary.json"
    write_json(summary_path, summary)

    command_text = build_command_text(
        run_dir=run_dir,
        recommended_epochs=recommended_epochs,
        batch_size=int(args.batch_size),
        base_channels=int(args.base_channels),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
    )
    (run_dir / "recommended_commands.txt").write_text(command_text, encoding="utf-8")

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN supervised baseline - quick CPU subset",
        lines=[
            f"base_split_run_dir: {args.base_split_run_dir}",
            f"run_dir: {run_dir}",
            f"train_units: {', '.join(args.train_units)}",
            f"val_units: {', '.join(args.val_units)}",
            f"test_units: {', '.join(args.test_units)}",
            f"train_window_count: {train_steps}",
            f"val_window_count: {val_steps}",
            f"test_window_count: {test_steps}",
            f"train_patch_count: {summary['train_patch_count']}",
            f"val_patch_count: {summary['val_patch_count']}",
            f"test_patch_count: {summary['test_patch_count']}",
            f"recommended_epochs: {recommended_epochs}",
            f"estimated_epoch_minutes: {summary['estimated_epoch_minutes']}",
            f"estimated_total_minutes: {summary['estimated_total_minutes']}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"run_dir: {run_dir}")
    print(f"train_window_count: {train_steps}")
    print(f"val_window_count: {val_steps}")
    print(f"test_window_count: {test_steps}")
    print(f"train_patch_count: {summary['train_patch_count']}")
    print(f"estimated_total_minutes: {summary['estimated_total_minutes']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
