# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from baseline_common import (
    DEFAULT_DATASET_RUN_DIR,
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    append_lines_to_docx,
    build_unit_level_split,
    load_sample_manifest,
    write_csv_utf8,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 UnitID 对轻量版 G-DFN 样本做训练/验证/测试划分。")
    parser.add_argument("--dataset-run-dir", type=Path, default=DEFAULT_DATASET_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "数据切分")
    parser.add_argument("--run-name", type=str, default=f"phase1_unit_split_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260330)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_sample_manifest(args.dataset_run_dir)
    unit_split_df, split_map = build_unit_level_split(
        manifest_df=manifest_df,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    write_csv_utf8(unit_split_df, run_dir / "unit_split.csv")

    summary = {
        "dataset_run_dir": str(args.dataset_run_dir),
        "run_dir": str(run_dir),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "unit_count": int(unit_split_df["UnitID"].nunique()),
        "window_count": int(len(manifest_df)),
    }
    for split_name in ("train", "val", "test"):
        split_units = set(split_map[split_name])
        split_manifest = manifest_df[manifest_df["UnitID"].astype(str).isin(split_units)].copy()
        write_csv_utf8(split_manifest, run_dir / f"{split_name}_manifest.csv")
        summary[f"{split_name}_unit_count"] = int(len(split_units))
        summary[f"{split_name}_window_count"] = int(len(split_manifest))
        summary[f"{split_name}_positive_window_count"] = int(split_manifest["IsPositiveWindow"].sum())
        summary[f"{split_name}_patch_count"] = int(split_manifest["PatchCount"].sum())

    summary_path = run_dir / "split_summary.json"
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - Unit级数据切分",
        lines=[
            f"dataset_run_dir: {args.dataset_run_dir}",
            f"run_dir: {run_dir}",
            f"seed: {args.seed}",
            f"unit_count: {summary['unit_count']}",
            f"window_count: {summary['window_count']}",
            f"train_unit_count: {summary['train_unit_count']}",
            f"val_unit_count: {summary['val_unit_count']}",
            f"test_unit_count: {summary['test_unit_count']}",
            f"train_window_count: {summary['train_window_count']}",
            f"val_window_count: {summary['val_window_count']}",
            f"test_window_count: {summary['test_window_count']}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"run_dir: {run_dir}")
    print(f"train_units: {summary['train_unit_count']}")
    print(f"val_units: {summary['val_unit_count']}")
    print(f"test_units: {summary['test_unit_count']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
