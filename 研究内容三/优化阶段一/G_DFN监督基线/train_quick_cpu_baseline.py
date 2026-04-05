# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from baseline_common import DEFAULT_OUTPUT_ROOT


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_QUICK_SPLIT_DIR = DEFAULT_OUTPUT_ROOT / "快速子集" / "quick_cpu_subset_v2_20260330"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the supervised G-DFN baseline with a reduced CPU-friendly configuration."
    )
    parser.add_argument("--split-run-dir", type=Path, default=DEFAULT_QUICK_SPLIT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "训练结果")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_script = THIS_DIR / "train_supervised_baseline.py"
    split_run_dir = Path(args.split_run_dir)
    if not split_run_dir.exists():
        raise FileNotFoundError(f"split_run_dir not found: {split_run_dir}")

    command = [
        sys.executable,
        str(train_script),
        "--split-run-dir",
        str(split_run_dir),
        "--output-root",
        str(Path(args.output_root)),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--base-channels",
        str(int(args.base_channels)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--device",
        str(args.device),
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.run_name:
        command.extend(["--run-name", str(args.run_name)])
    if args.resume_checkpoint:
        command.extend(["--resume-checkpoint", str(Path(args.resume_checkpoint))])
    if bool(args.disable_amp):
        command.append("--disable-amp")
    if bool(args.no_progress):
        command.append("--no-progress")

    printable = subprocess.list2cmdline(command)
    print("launch_command:")
    print(printable)

    if bool(args.dry_run):
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
