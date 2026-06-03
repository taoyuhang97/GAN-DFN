# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
OPT_STAGE_DIR = THIS_DIR.parent
BASELINE_DIR = OPT_STAGE_DIR / "G_DFN监督基线"

for candidate in (THIS_DIR, BASELINE_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_csv_utf8, write_json
from geophysical_postprocess_multiscale import _df_from_vtk, _df_to_vtk
from merge_unit_dfn_vtks import read_legacy_vtk_polygons


SCALE_ALIAS_TO_CLASS = {
    "large": "macro_core",
    "medium": "meso_link",
    "small": "micro_bg",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将区域 DFN 按大/中/小三个尺度拆分成独立 VTK/CSV。")
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-dir", type=Path, help="直接输出到指定目录，不再额外创建 run-name 子目录。")
    parser.add_argument("--same-dir-as-input", action="store_true", help="直接输出到输入 VTK 所在目录。")
    parser.add_argument("--run-name", type=str, default=f"scale_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def ensure_scale_class(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = df.copy()
    if "ScaleClass" in work.columns:
        scale_series = work["ScaleClass"].astype(str).str.strip()
        valid_mask = scale_series.isin(set(SCALE_ALIAS_TO_CLASS.values()))
        if bool(valid_mask.any()):
            work.loc[valid_mask, "ScaleClass"] = scale_series.loc[valid_mask]
            return work, "vtk_scale_class"

    if "PatchArea" not in work.columns:
        work["PatchArea"] = (
            pd.to_numeric(work.get("PatchLength"), errors="coerce").fillna(0.0)
            * pd.to_numeric(work.get("PatchHeight"), errors="coerce").fillna(0.0)
        )
    area = pd.to_numeric(work["PatchArea"], errors="coerce").fillna(0.0)
    if len(area) <= 0:
        work["ScaleClass"] = "micro_bg"
        return work, "fallback_empty_to_micro"

    q1 = float(area.quantile(1.0 / 3.0))
    q2 = float(area.quantile(2.0 / 3.0))
    scale_class = np.full(len(work), "meso_link", dtype=object)
    scale_class[area.to_numpy(dtype=float) <= q1] = "micro_bg"
    scale_class[area.to_numpy(dtype=float) >= q2] = "macro_core"
    work["ScaleClass"] = scale_class
    return work, "fallback_patch_area_quantile"


def build_subset_summary(scale_alias: str, scale_class: str, subset_df: pd.DataFrame) -> dict[str, Any]:
    patch_area = pd.to_numeric(subset_df.get("PatchArea"), errors="coerce").fillna(0.0)
    patch_length = pd.to_numeric(subset_df.get("PatchLength"), errors="coerce").fillna(0.0)
    patch_height = pd.to_numeric(subset_df.get("PatchHeight"), errors="coerce").fillna(0.0)
    return {
        "scale_alias": scale_alias,
        "scale_class": scale_class,
        "patch_count": int(len(subset_df)),
        "patch_area_sum": float(patch_area.sum()),
        "patch_area_mean": float(patch_area.mean()) if len(subset_df) else 0.0,
        "patch_area_p50": float(patch_area.median()) if len(subset_df) else 0.0,
        "patch_area_p90": float(patch_area.quantile(0.9)) if len(subset_df) else 0.0,
        "patch_length_mean": float(patch_length.mean()) if len(subset_df) else 0.0,
        "patch_height_mean": float(patch_height.mean()) if len(subset_df) else 0.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    input_vtk = Path(args.input_vtk)
    if not input_vtk.exists():
        raise FileNotFoundError(f"input_vtk not found: {input_vtk}")

    direct_output = bool(args.same_dir_as_input) or args.output_dir is not None
    if direct_output:
        run_dir = Path(args.output_dir) if args.output_dir is not None else input_vtk.parent
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        if args.output_root is None:
            raise ValueError("output_root is required unless --output-dir or --same-dir-as-input is used")
        run_dir = Path(args.output_root) / str(args.run_name)
        if run_dir.exists() and not bool(args.overwrite):
            raise FileExistsError(f"output run dir already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)

    payload = read_legacy_vtk_polygons(input_vtk)
    source_df = _df_from_vtk(payload)
    source_df, scale_source = ensure_scale_class(source_df)
    file_prefix = input_vtk.stem

    summary_rows: list[dict[str, Any]] = []
    for scale_alias, scale_class in SCALE_ALIAS_TO_CLASS.items():
        subset_df = source_df[source_df["ScaleClass"].astype(str) == scale_class].copy().reset_index(drop=True)
        subset_csv = run_dir / f"{file_prefix}_{scale_alias}_scale.csv"
        subset_vtk = run_dir / f"{file_prefix}_{scale_alias}_scale.vtk"
        if (subset_csv.exists() or subset_vtk.exists()) and not bool(args.overwrite):
            raise FileExistsError(f"output file already exists, use --overwrite: {subset_vtk}")
        write_csv_utf8(subset_df, subset_csv)
        _df_to_vtk(subset_df, f"regional_dfn_{scale_alias}_scale", subset_vtk)

        row = build_subset_summary(scale_alias=scale_alias, scale_class=scale_class, subset_df=subset_df)
        row["output_csv"] = str(subset_csv)
        row["output_vtk"] = str(subset_vtk)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = run_dir / f"{file_prefix}_scale_split_summary.csv"
    write_csv_utf8(summary_df, summary_csv)

    summary = {
        "input_vtk": str(input_vtk),
        "run_dir": str(run_dir),
        "input_patch_count": int(len(source_df)),
        "scale_source": str(scale_source),
        "summary_csv": str(summary_csv),
        "scale_outputs": summary_rows,
    }
    summary_json = run_dir / f"{file_prefix}_scale_split_summary.json"
    write_json(summary_json, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title=f"regional scale split {args.run_name}",
        lines=[
            f"input_vtk: {input_vtk}",
            f"scale_source: {scale_source}",
            f"input_patch_count: {len(source_df)}",
            f"summary_csv: {summary_csv}",
            f"summary_json: {summary_json}",
        ],
    )

    print(f"summary_csv: {summary_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
