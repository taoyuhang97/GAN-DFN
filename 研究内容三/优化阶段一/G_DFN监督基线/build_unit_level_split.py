# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from baseline_common import (
    DEFAULT_DATASET_RUN_DIR,
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    append_lines_to_docx,
    build_unit_level_split,
    load_sample_manifest,
    read_csv_utf8,
    write_csv_utf8,
    write_json,
)
from layer_model_registry import summarize_manifest_by_layer_surface_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 UnitID 对轻量版 G-DFN 样本做训练/验证/测试划分。")
    parser.add_argument("--dataset-run-dir", type=Path, default=DEFAULT_DATASET_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "数据切分")
    parser.add_argument("--run-name", type=str, default=f"phase1_unit_split_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-unit-count", type=int)
    parser.add_argument("--val-unit-count", type=int)
    parser.add_argument("--test-unit-count", type=int)
    parser.add_argument("--fixed-unit-split-csv", type=Path)
    parser.add_argument("--seed", type=int, default=20260330)
    return parser


def build_fixed_unit_level_split(
    manifest_df: pd.DataFrame,
    fixed_unit_split_csv: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    split_df = read_csv_utf8(fixed_unit_split_csv).copy()
    required_cols = {"UnitID", "Split"}
    missing_cols = sorted(required_cols - set(split_df.columns))
    if missing_cols:
        raise ValueError(f"fixed split csv missing columns: {missing_cols}")

    split_df["UnitID"] = split_df["UnitID"].astype(str).str.strip()
    split_df["Split"] = split_df["Split"].astype(str).str.strip().str.lower()
    split_df = split_df[split_df["UnitID"] != ""].copy()

    valid_splits = {"train", "val", "test"}
    invalid_split_names = sorted(set(split_df["Split"].tolist()) - valid_splits)
    if invalid_split_names:
        raise ValueError(f"fixed split csv contains invalid split names: {invalid_split_names}")

    duplicate_unit_ids = (
        split_df.loc[split_df["UnitID"].duplicated(keep=False), "UnitID"].astype(str).sort_values().unique().tolist()
    )
    if duplicate_unit_ids:
        raise ValueError(f"fixed split csv contains duplicated UnitID values: {duplicate_unit_ids}")

    manifest_units = set(manifest_df["UnitID"].astype(str).str.strip().tolist())
    split_units = set(split_df["UnitID"].tolist())
    missing_units = sorted(manifest_units - split_units)
    unknown_units = sorted(split_units - manifest_units)
    if missing_units or unknown_units:
        raise ValueError(
            "fixed split csv does not match dataset manifest: "
            f"missing_units={missing_units}, unknown_units={unknown_units}"
        )

    split_map = {
        split_name: split_df.loc[split_df["Split"] == split_name, "UnitID"].astype(str).tolist()
        for split_name in ("train", "val", "test")
    }
    if any(len(unit_ids) <= 0 for unit_ids in split_map.values()):
        raise ValueError(
            "fixed split csv must provide non-empty train/val/test groups: "
            f"train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}"
        )

    rows: list[dict[str, int | str | None]] = []
    for split_name, split_units_ordered in split_map.items():
        for unit_id in split_units_ordered:
            unit_rows = manifest_df[manifest_df["UnitID"].astype(str) == str(unit_id)].copy()
            rows.append(
                {
                    "UnitID": str(unit_id),
                    "Split": split_name,
                    "WindowCount": int(len(unit_rows)),
                    "PositiveWindowCount": int(unit_rows["IsPositiveWindow"].sum()) if "IsPositiveWindow" in unit_rows.columns else None,
                    "PatchCount": int(unit_rows["PatchCount"].sum()) if "PatchCount" in unit_rows.columns else None,
                }
            )
    return pd.DataFrame(rows), split_map


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_sample_manifest(args.dataset_run_dir)
    fixed_unit_split_csv = Path(args.fixed_unit_split_csv).resolve() if args.fixed_unit_split_csv else None
    if fixed_unit_split_csv is not None:
        unit_split_df, split_map = build_fixed_unit_level_split(
            manifest_df=manifest_df,
            fixed_unit_split_csv=fixed_unit_split_csv,
        )
        split_mode = "fixed_csv"
    else:
        unit_split_df, split_map = build_unit_level_split(
            manifest_df=manifest_df,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            train_unit_count=args.train_unit_count,
            val_unit_count=args.val_unit_count,
            test_unit_count=args.test_unit_count,
        )
        split_mode = "random_or_count"
    write_csv_utf8(unit_split_df, run_dir / "unit_split.csv")

    summary = {
        "dataset_run_dir": str(args.dataset_run_dir),
        "run_dir": str(run_dir),
        "split_mode": split_mode,
        "fixed_unit_split_csv": str(fixed_unit_split_csv) if fixed_unit_split_csv is not None else None,
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "train_unit_count_requested": int(args.train_unit_count) if args.train_unit_count is not None else None,
        "val_unit_count_requested": int(args.val_unit_count) if args.val_unit_count is not None else None,
        "test_unit_count_requested": int(args.test_unit_count) if args.test_unit_count is not None else None,
        "unit_count": int(unit_split_df["UnitID"].nunique()),
        "window_count": int(len(manifest_df)),
    }
    layer_pair_summary_frames: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        split_units = set(split_map[split_name])
        split_manifest = manifest_df[manifest_df["UnitID"].astype(str).isin(split_units)].copy()
        write_csv_utf8(split_manifest, run_dir / f"{split_name}_manifest.csv")
        summary[f"{split_name}_unit_count"] = int(len(split_units))
        summary[f"{split_name}_window_count"] = int(len(split_manifest))
        summary[f"{split_name}_positive_window_count"] = int(split_manifest["IsPositiveWindow"].sum())
        summary[f"{split_name}_patch_count"] = int(split_manifest["PatchCount"].sum())
        layer_pair_summary = summarize_manifest_by_layer_surface_pair(split_manifest)
        if not layer_pair_summary.empty:
            layer_pair_summary.insert(0, "Split", split_name)
            layer_pair_summary_frames.append(layer_pair_summary)

    split_layer_pair_summary_df = (
        pd.concat(layer_pair_summary_frames, ignore_index=True, sort=False)
        if layer_pair_summary_frames
        else pd.DataFrame()
    )
    split_layer_pair_summary_csv = run_dir / "split_layer_pair_summary.csv"
    write_csv_utf8(split_layer_pair_summary_df, split_layer_pair_summary_csv)
    summary["split_layer_pair_summary_csv"] = str(split_layer_pair_summary_csv)

    summary_path = run_dir / "split_summary.json"
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - Unit级数据切分",
        lines=[
            f"dataset_run_dir: {args.dataset_run_dir}",
            f"run_dir: {run_dir}",
            f"split_mode: {split_mode}",
            f"fixed_unit_split_csv: {fixed_unit_split_csv}" if fixed_unit_split_csv is not None else "fixed_unit_split_csv: None",
            f"seed: {args.seed}",
            f"train_unit_count_requested: {summary['train_unit_count_requested']}",
            f"val_unit_count_requested: {summary['val_unit_count_requested']}",
            f"test_unit_count_requested: {summary['test_unit_count_requested']}",
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
    print(f"split_mode: {split_mode}")
    print(f"train_units: {summary['train_unit_count']}")
    print(f"val_units: {summary['val_unit_count']}")
    print(f"test_units: {summary['test_unit_count']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
