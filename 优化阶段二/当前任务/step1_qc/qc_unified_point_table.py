from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MIN_REQUIRED_COLUMNS = [
    "WellName",
    "X",
    "Y",
    "TIME",
    "TVD",
    "FractureDensity",
    "IsFracturePoint",
    "SourceType",
]

KEY_COORD_COLUMNS = ["X", "Y", "TIME", "TVD"]
DEFAULT_DUPLICATE_KEY_COLUMNS = ["WellName", "X", "Y", "TIME", "TVD"]


@dataclass
class QcConfig:
    input_path: str
    output_dir: str
    required_columns: list[str]
    key_coord_columns: list[str]
    duplicate_key_columns: list[str]
    source_col: str = "SourceType"
    density_col: str = "FractureDensity"
    fracture_flag_col: str = "IsFracturePoint"
    strict: bool = False


def parse_args() -> QcConfig:
    parser = argparse.ArgumentParser(description="对 unified point table 执行最小质量检查。")
    parser.add_argument("--input", required=True, help="输入样本表路径，支持 csv/parquet。")
    parser.add_argument("--output-dir", required=True, help="QC 输出目录。")
    parser.add_argument(
        "--required-columns",
        nargs="+",
        default=MIN_REQUIRED_COLUMNS,
        help="必需字段列表。",
    )
    parser.add_argument(
        "--key-coord-columns",
        nargs="+",
        default=KEY_COORD_COLUMNS,
        help="关键坐标/深度字段，用于空值检查。",
    )
    parser.add_argument(
        "--duplicate-key-columns",
        nargs="+",
        default=DEFAULT_DUPLICATE_KEY_COLUMNS,
        help="扩展检查：用于重复点检测的字段组合。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="若存在 error 级问题则返回非 0 退出码。",
    )
    args = parser.parse_args()
    return QcConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        required_columns=list(args.required_columns),
        key_coord_columns=list(args.key_coord_columns),
        duplicate_key_columns=list(args.duplicate_key_columns),
        strict=bool(args.strict),
    )


def load_table(path: str) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(input_path)
    elif suffix == ".parquet":
        table = pd.read_parquet(input_path)
    else:
        raise ValueError(f"暂不支持的输入格式: {input_path.suffix}")
    if table.empty:
        raise ValueError("输入 unified point table 为空。")
    return table


def normalize_bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
    }
    normalized = series.copy()
    if pd.api.types.is_bool_dtype(normalized):
        return normalized.astype("Int64")
    normalized = normalized.astype(str).str.strip().str.lower()
    normalized = normalized.replace(mapping)
    return pd.to_numeric(normalized, errors="coerce").astype("Int64")


def to_builtin(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def count_missing(values: pd.Series) -> int:
    return int(values.isna().sum())


def build_source_distribution(table: pd.DataFrame, source_col: str) -> list[dict[str, Any]]:
    distribution = (
        table[source_col]
        .fillna("__MISSING__")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("SourceType")
        .reset_index(name="row_count")
    )
    distribution["row_fraction"] = distribution["row_count"] / max(len(table), 1)
    return distribution.to_dict(orient="records")


def run_qc(table: pd.DataFrame, config: QcConfig) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    present_columns = set(table.columns)
    missing_required = [column for column in config.required_columns if column not in present_columns]

    checks: list[dict[str, Any]] = []

    def add_check(name: str, severity: str, passed: bool, details: dict[str, Any]) -> None:
        checks.append(
            {
                "check_name": name,
                "severity": severity,
                "passed": bool(passed),
                "details": json.dumps({k: to_builtin(v) for k, v in details.items()}, ensure_ascii=False),
            }
        )

    add_check(
        name="required_columns_present",
        severity="error",
        passed=not missing_required,
        details={
            "missing_required_columns": missing_required,
            "present_column_count": len(table.columns),
        },
    )

    missing_by_coord: dict[str, int] = {}
    for column in config.key_coord_columns:
        if column not in table.columns:
            missing_by_coord[column] = len(table)
            continue
        missing_by_coord[column] = count_missing(table[column])
    add_check(
        name="key_coordinate_not_null",
        severity="error",
        passed=all(count == 0 for count in missing_by_coord.values()),
        details=missing_by_coord,
    )

    numeric_issue_counts: dict[str, int] = {}
    normalized = table.copy()
    for column in [config.density_col, *config.key_coord_columns]:
        if column in normalized.columns:
            numeric_values = pd.to_numeric(normalized[column], errors="coerce")
            numeric_issue_counts[column] = int(numeric_values.isna().sum() - normalized[column].isna().sum())
            normalized[column] = numeric_values
    add_check(
        name="numeric_castability",
        severity="warning",
        passed=all(count == 0 for count in numeric_issue_counts.values()),
        details=numeric_issue_counts,
    )

    fracture_flag_missing = len(table)
    invalid_flag_count = len(table)
    point_missing_density = len(table)
    point_non_positive_density = len(table)
    non_point_positive_density = len(table)
    negative_density_count = len(table)

    if config.fracture_flag_col in normalized.columns and config.density_col in normalized.columns:
        flag_series = normalize_bool_series(normalized[config.fracture_flag_col])
        density_series = pd.to_numeric(normalized[config.density_col], errors="coerce")

        fracture_flag_missing = int(flag_series.isna().sum())
        invalid_flag_count = int((~flag_series.isin([0, 1]) & flag_series.notna()).sum())
        point_mask = flag_series == 1
        non_point_mask = flag_series == 0

        point_missing_density = int((point_mask & density_series.isna()).sum())
        point_non_positive_density = int((point_mask & density_series.fillna(0.0).le(0.0)).sum())
        non_point_positive_density = int((non_point_mask & density_series.fillna(0.0).gt(0.0)).sum())
        negative_density_count = int(density_series.lt(0.0).sum())

    add_check(
        name="fracture_density_consistency",
        severity="warning",
        passed=all(
            count == 0
            for count in [
                fracture_flag_missing,
                invalid_flag_count,
                point_missing_density,
                point_non_positive_density,
                non_point_positive_density,
                negative_density_count,
            ]
        ),
        details={
            "fracture_flag_missing": fracture_flag_missing,
            "invalid_flag_count": invalid_flag_count,
            "point_missing_density": point_missing_density,
            "point_non_positive_density": point_non_positive_density,
            "non_point_positive_density": non_point_positive_density,
            "negative_density_count": negative_density_count,
        },
    )

    duplicate_count = 0
    used_duplicate_keys = [column for column in config.duplicate_key_columns if column in normalized.columns]
    if used_duplicate_keys:
        duplicate_count = int(normalized.duplicated(subset=used_duplicate_keys, keep=False).sum())
    add_check(
        name="duplicate_point_keys",
        severity="warning",
        passed=duplicate_count == 0,
        details={
            "duplicate_row_count": duplicate_count,
            "duplicate_key_columns_used": used_duplicate_keys,
        },
    )

    source_distribution_records: list[dict[str, Any]] = []
    if config.source_col in normalized.columns:
        source_distribution_records = build_source_distribution(normalized, config.source_col)
    add_check(
        name="source_type_distribution_available",
        severity="warning",
        passed=bool(source_distribution_records),
        details={
            "distinct_source_count": len(source_distribution_records),
        },
    )

    error_count = sum(1 for item in checks if item["severity"] == "error" and not item["passed"])
    warning_count = sum(1 for item in checks if item["severity"] == "warning" and not item["passed"])
    summary = {
        "status": "fail" if error_count > 0 else ("warn" if warning_count > 0 else "pass"),
        "row_count": int(len(table)),
        "column_count": int(len(table.columns)),
        "missing_required_columns": missing_required,
        "error_count": int(error_count),
        "warning_count": int(warning_count),
        "source_distribution": source_distribution_records,
        "config": asdict(config),
    }
    checks_frame = pd.DataFrame(checks)
    source_frame = pd.DataFrame(source_distribution_records)
    return summary, checks_frame, source_frame


def write_outputs(
    summary: dict[str, Any],
    checks_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_path / "qc_summary.json"
    summary_csv_path = output_path / "qc_summary.csv"
    source_csv_path = output_path / "source_distribution.csv"

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_summary_rows = [
        {"metric": "status", "value": summary["status"]},
        {"metric": "row_count", "value": summary["row_count"]},
        {"metric": "column_count", "value": summary["column_count"]},
        {"metric": "error_count", "value": summary["error_count"]},
        {"metric": "warning_count", "value": summary["warning_count"]},
        {
            "metric": "missing_required_columns",
            "value": ",".join(summary["missing_required_columns"]),
        },
    ]
    summary_frame = pd.concat([pd.DataFrame(flat_summary_rows), checks_frame], ignore_index=False)
    summary_frame.to_csv(summary_csv_path, index=False)

    if not source_frame.empty:
        source_frame.to_csv(source_csv_path, index=False)


def main() -> int:
    config = parse_args()
    table = load_table(config.input_path)
    summary, checks_frame, source_frame = run_qc(table, config)
    write_outputs(summary, checks_frame, source_frame, config.output_dir)
    print(json.dumps(summary, ensure_ascii=False))
    if config.strict and summary["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
