from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


INVALID_SENTINELS = (
    -999.25,
    -9999.0,
    -99999.0,
    9999.0,
    99999.0,
    -9999999.0,
    -9999998.0,
)
WINDOW_COUNT = 63
DEFAULT_ROLE_PRIORITY = [
    "COHERENCE",
    "ANT",
    "CURVATURE",
    "FRACTURE_INVERSION",
    "AMPLITUDE",
    "车西_相干体T4_T7",
    "车西_蚂蚁体T4_T7",
    "车西_最大曲率T4_T7",
    "车西_最大正曲率T4_T7",
    "车西_最大曲率",
    "车西_最大正曲率",
    "车西_相干体",
    "车西_蚂蚁体",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment imaging sample CSVs with SEIS_TRUE/SEIS_0..62 aliases.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            r"E:\项目\石油项目\断缝储\GAN-DFN-upload-first-round-improvement-20260420\模型训练\优化阶段二\near_well_attribute_samples_imaging\aligned_sample_csv_labeled"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"E:\项目\石油项目\断缝储\GAN-DFN-upload-first-round-improvement-20260420\模型训练\优化阶段二\near_well_attribute_samples_imaging\aligned_sample_csv_labeled_stage1_compatible"
        ),
    )
    parser.add_argument(
        "--role-priority",
        type=str,
        default=",".join(DEFAULT_ROLE_PRIORITY),
        help="Comma-separated role prefixes to prefer when multiple volume families exist.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def normalize_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    invalid_mask = ~np.isfinite(out.to_numpy(dtype=np.float64))
    for sentinel in INVALID_SENTINELS:
        invalid_mask |= np.isclose(out.to_numpy(dtype=np.float64), float(sentinel), rtol=0.0, atol=1e-9)
    invalid_mask |= out.to_numpy(dtype=np.float64) <= -999.0
    out = out.mask(invalid_mask)
    return out


def find_role_columns(df: pd.DataFrame, role: str) -> tuple[str, list[str]]:
    center_col = f"{role}_CENTER"
    window_cols = [f"{role}_W{idx:02d}" for idx in range(WINDOW_COUNT)]
    available = [col for col in [center_col, *window_cols] if col in df.columns]
    return center_col, available


def score_role(df: pd.DataFrame, role: str) -> tuple[int, int, int]:
    center_col, cols = find_role_columns(df, role)
    if not cols:
        return 0, 0, 0
    block = pd.concat([normalize_numeric(df[col]) for col in cols], axis=0, ignore_index=True)
    finite = int(np.isfinite(block.to_numpy(dtype=np.float64)).sum())
    total = int(block.size)
    center_finite = 0
    if center_col in df.columns:
        center_values = normalize_numeric(df[center_col])
        center_finite = int(np.isfinite(center_values.to_numpy(dtype=np.float64)).sum())
    return finite, total, center_finite


def augment_one_csv(input_csv: Path, output_csv: Path, role_priority: list[str], overwrite: bool) -> dict[str, object]:
    if output_csv.exists() and not overwrite:
        return {
            "InputCsv": str(input_csv),
            "OutputCsv": str(output_csv),
            "Status": "skipped_exists",
        }

    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    role_scores = []
    for role in role_priority:
        finite, total, center_finite = score_role(df, role)
        if total > 0:
            role_scores.append((center_finite, finite, total, role))
    if not role_scores:
        return {
            "InputCsv": str(input_csv),
            "OutputCsv": str(output_csv),
            "Status": "skipped_no_role_windows",
        }

    role_scores.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected_role = role_scores[0][3]
    center_col, available_cols = find_role_columns(df, selected_role)
    if center_col not in df.columns:
        raise ValueError(f"Selected role missing center column: {selected_role}")

    alias_cols = {"SEIS_SOURCE_ROLE": selected_role}
    center_values = normalize_numeric(df[center_col])
    alias_cols["SEIS_TRUE"] = center_values
    for idx in range(WINDOW_COUNT):
        src_col = f"{selected_role}_W{idx:02d}"
        if src_col in df.columns:
            alias_cols[f"SEIS_{idx}"] = normalize_numeric(df[src_col])
        else:
            alias_cols[f"SEIS_{idx}"] = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

    out_df = df.copy()
    for col_name, values in alias_cols.items():
        out_df[col_name] = values

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return {
        "InputCsv": str(input_csv),
        "OutputCsv": str(output_csv),
        "Status": "ok",
        "SelectedRole": selected_role,
        "CenterFiniteCount": int(role_scores[0][0]),
        "FiniteCount": int(role_scores[0][1]),
        "TotalCount": int(role_scores[0][2]),
        "RowCount": int(len(out_df)),
    }


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    role_priority = [item.strip() for item in str(args.role_priority).split(",") if item.strip()]
    csv_files = sorted(input_dir.glob("*_sample.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_sample.csv files found under: {input_dir}")

    summary_rows: list[dict[str, object]] = []
    for input_csv in csv_files:
        output_csv = output_dir / input_csv.name
        summary_rows.append(augment_one_csv(input_csv, output_csv, role_priority=role_priority, overwrite=bool(args.overwrite)))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "stage1_compat_augmentation_summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
