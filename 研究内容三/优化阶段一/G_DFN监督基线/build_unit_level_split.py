# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from baseline_common import (
    compute_file_sha256,
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
from layer_model_registry import LAYER_SURFACE_PAIR_KEY_COL, ensure_layer_surface_pair_key_column, summarize_manifest_by_layer_surface_pair
from layer_training_config import canonicalize_manifest_layer_surface_pairs, load_layer_training_config


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_LAYER_TRAINING_CONFIG_PY = THIS_DIR / "layerwise_training_plan_demo_v1.py"


def emit_split_progress(stage: str, detail: str | None = None) -> None:
    if detail:
        print(f"[split] {stage} | {detail}", flush=True)
    else:
        print(f"[split] {stage}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 UnitID 对轻量版 G-DFN 样本做训练/验证/测试划分。")
    parser.add_argument("--dataset-run-dir", type=Path, default=DEFAULT_DATASET_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "数据切分")
    parser.add_argument("--run-name", type=str, default=f"phase1_unit_split_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--layer-training-config-py", type=Path, default=DEFAULT_LAYER_TRAINING_CONFIG_PY)
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

    emit_split_progress("开始构建 Unit 级切分", f"dataset_run_dir={Path(args.dataset_run_dir).resolve()}")
    layer_training_config = load_layer_training_config(args.layer_training_config_py)
    allowed_layer_surface_pair_keys = set(layer_training_config["allowed_layer_surface_pair_keys"])
    manifest_df = load_sample_manifest(args.dataset_run_dir)
    manifest_df = ensure_layer_surface_pair_key_column(manifest_df, key_col=LAYER_SURFACE_PAIR_KEY_COL)
    emit_split_progress(
        "manifest 读取完成",
        f"window_count={len(manifest_df)}, allowed_layer_pair_count={len(allowed_layer_surface_pair_keys)}",
    )
    raw_layer_surface_pair_keys = (
        manifest_df[LAYER_SURFACE_PAIR_KEY_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "<EMPTY>")
    )
    manifest_window_count_before_layer_filter = int(len(manifest_df))
    manifest_df = canonicalize_manifest_layer_surface_pairs(manifest_df, key_col=LAYER_SURFACE_PAIR_KEY_COL)
    keep_mask = manifest_df[LAYER_SURFACE_PAIR_KEY_COL].astype(str).isin(allowed_layer_surface_pair_keys)
    dropped_layer_surface_pair_keys = (
        raw_layer_surface_pair_keys.loc[~keep_mask]
        .drop_duplicates()
        .sort_values()
        .astype(str)
        .tolist()
    )
    manifest_df = manifest_df.loc[keep_mask].copy()
    manifest_window_count_after_layer_filter = int(len(manifest_df))
    if manifest_window_count_after_layer_filter <= 0:
        raise ValueError("manifest is empty after layer surface pair filtering")
    emit_split_progress(
        "层位过滤完成",
        (
            f"window_count_after_filter={manifest_window_count_after_layer_filter}, "
            f"dropped_window_count={manifest_window_count_before_layer_filter - manifest_window_count_after_layer_filter}"
        ),
    )

    dataset_run_dir = Path(args.dataset_run_dir).resolve()
    dataset_manifest_csv = dataset_run_dir / "aggregated" / "sample_manifest.csv"
    dataset_summary_json = dataset_run_dir / "aggregated" / "dataset_summary.json"
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
    emit_split_progress(
        "切分完成",
        (
            f"split_mode={split_mode}, unit_count={unit_split_df['UnitID'].nunique()}, "
            f"train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}"
        ),
    )
    unit_split_csv = run_dir / "unit_split.csv"
    write_csv_utf8(unit_split_df, unit_split_csv)

    summary = {
        "dataset_run_dir": str(dataset_run_dir),
        "dataset_manifest_csv": str(dataset_manifest_csv.resolve()),
        "dataset_manifest_sha256": compute_file_sha256(dataset_manifest_csv),
        "dataset_summary_json": str(dataset_summary_json.resolve()),
        "dataset_summary_sha256": compute_file_sha256(dataset_summary_json),
        "run_dir": str(run_dir.resolve()),
        "layer_training_config_py": layer_training_config["config_py"] or None,
        "layer_training_config_sha256": layer_training_config["config_sha256"],
        "allowed_layer_surface_pair_keys": layer_training_config["allowed_layer_surface_pair_keys"],
        "layer_training_config_metadata": layer_training_config["metadata"],
        "split_mode": split_mode,
        "fixed_unit_split_csv": str(fixed_unit_split_csv) if fixed_unit_split_csv is not None else None,
        "fixed_unit_split_csv_sha256": compute_file_sha256(fixed_unit_split_csv) if fixed_unit_split_csv is not None else "",
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "manifest_window_count_before_layer_filter": manifest_window_count_before_layer_filter,
        "manifest_window_count_after_layer_filter": manifest_window_count_after_layer_filter,
        "dropped_window_count_by_layer_filter": (
            manifest_window_count_before_layer_filter - manifest_window_count_after_layer_filter
        ),
        "dropped_layer_surface_pair_keys": dropped_layer_surface_pair_keys,
        "train_unit_count_requested": int(args.train_unit_count) if args.train_unit_count is not None else None,
        "val_unit_count_requested": int(args.val_unit_count) if args.val_unit_count is not None else None,
        "test_unit_count_requested": int(args.test_unit_count) if args.test_unit_count is not None else None,
        "unit_count": int(unit_split_df["UnitID"].nunique()),
        "window_count": int(len(manifest_df)),
        "unit_split_csv": str(unit_split_csv.resolve()),
        "unit_split_csv_sha256": compute_file_sha256(unit_split_csv),
    }
    layer_pair_summary_frames: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        split_units = set(split_map[split_name])
        split_manifest = manifest_df[manifest_df["UnitID"].astype(str).isin(split_units)].copy()
        split_manifest_path = run_dir / f"{split_name}_manifest.csv"
        write_csv_utf8(split_manifest, split_manifest_path)
        summary[f"{split_name}_unit_count"] = int(len(split_units))
        summary[f"{split_name}_window_count"] = int(len(split_manifest))
        summary[f"{split_name}_positive_window_count"] = int(split_manifest["IsPositiveWindow"].sum())
        summary[f"{split_name}_patch_count"] = int(split_manifest["PatchCount"].sum())
        summary[f"{split_name}_manifest_csv"] = str(split_manifest_path.resolve())
        summary[f"{split_name}_manifest_sha256"] = compute_file_sha256(split_manifest_path)
        layer_pair_summary = summarize_manifest_by_layer_surface_pair(split_manifest)
        if not layer_pair_summary.empty:
            layer_pair_summary.insert(0, "Split", split_name)
            layer_pair_summary_frames.append(layer_pair_summary)
        emit_split_progress(
            f"{split_name} manifest 写出完成",
            (
                f"unit_count={len(split_units)}, window_count={len(split_manifest)}, "
                f"patch_count={int(split_manifest['PatchCount'].sum())}"
            ),
        )

    split_layer_pair_summary_df = (
        pd.concat(layer_pair_summary_frames, ignore_index=True, sort=False)
        if layer_pair_summary_frames
        else pd.DataFrame()
    )
    split_layer_pair_summary_csv = run_dir / "split_layer_pair_summary.csv"
    write_csv_utf8(split_layer_pair_summary_df, split_layer_pair_summary_csv)
    summary["split_layer_pair_summary_csv"] = str(split_layer_pair_summary_csv.resolve())
    summary["split_layer_pair_summary_csv_sha256"] = compute_file_sha256(split_layer_pair_summary_csv)

    summary_path = run_dir / "split_summary.json"
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - Unit级数据切分",
        lines=[
            f"dataset_run_dir: {args.dataset_run_dir}",
            f"run_dir: {run_dir}",
            f"layer_training_config_py: {layer_training_config['config_py'] or 'None'}",
            f"split_mode: {split_mode}",
            f"fixed_unit_split_csv: {fixed_unit_split_csv}" if fixed_unit_split_csv is not None else "fixed_unit_split_csv: None",
            f"seed: {args.seed}",
            f"manifest_window_count_before_layer_filter: {manifest_window_count_before_layer_filter}",
            f"manifest_window_count_after_layer_filter: {manifest_window_count_after_layer_filter}",
            f"dropped_window_count_by_layer_filter: {summary['dropped_window_count_by_layer_filter']}",
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
    print(f"layer_training_config_py: {layer_training_config['config_py'] or 'None'}")
    print(f"manifest_window_count_after_layer_filter: {manifest_window_count_after_layer_filter}")
    print(f"train_units: {summary['train_unit_count']}")
    print(f"val_units: {summary['val_unit_count']}")
    print(f"test_units: {summary['test_unit_count']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
