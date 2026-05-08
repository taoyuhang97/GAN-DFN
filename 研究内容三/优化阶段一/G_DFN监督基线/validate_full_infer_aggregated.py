#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate aggregated full_infer shard outputs before downstream merge/postprocess steps."
    )
    parser.add_argument("--aggregated-dir", type=Path, required=True, help="Path to full_infer/aggregated directory.")
    parser.add_argument("--expected-shards", type=int, default=None, help="Expected shard file count for each aggregated pattern.")
    return parser


def list_paths(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def count_csv_rows(path: Path) -> int:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return 0
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        return 0
    return sum(1 for row in reader if any(str(cell).strip() for cell in row))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_pattern_rows(paths: Iterable[Path]) -> tuple[int, list[tuple[str, int]]]:
    details: list[tuple[str, int]] = []
    total = 0
    for path in paths:
        row_count = count_csv_rows(path)
        details.append((path.name, row_count))
        total += row_count
    return total, details


def print_counts(label: str, details: list[tuple[str, int]], total: int) -> None:
    print(f"{label}_files={len(details)}")
    for filename, row_count in details:
        print(f"{label}:{filename}={row_count}")
    print(f"{label}_total={total}")


def main() -> int:
    args = build_parser().parse_args()
    aggregated_dir = args.aggregated_dir.expanduser().resolve()
    if not aggregated_dir.is_dir():
        raise SystemExit(f"aggregated_dir_not_found: {aggregated_dir}")

    selected_paths = list_paths(aggregated_dir, "selected_units*.csv")
    skipped_paths = list_paths(aggregated_dir, "skipped_units*.csv")
    processed_paths = list_paths(aggregated_dir, "unit_prediction*.csv")
    summary_paths = list_paths(aggregated_dir, "production_inference_summary*.json")

    print(f"aggregated_dir={aggregated_dir}")

    if args.expected_shards is not None:
        expected = int(args.expected_shards)
        for label, paths in (
            ("selected_units", selected_paths),
            ("unit_prediction", processed_paths),
            ("production_inference_summary", summary_paths),
        ):
            if len(paths) != expected:
                raise SystemExit(f"{label}_file_count_mismatch: expected={expected} actual={len(paths)}")
        if skipped_paths and len(skipped_paths) != expected:
            raise SystemExit(f"skipped_units_file_count_mismatch: expected={expected} actual={len(skipped_paths)}")

    if not selected_paths:
        raise SystemExit("missing_selected_units_csv")
    if not processed_paths:
        raise SystemExit("missing_unit_prediction_csv")
    if not summary_paths:
        raise SystemExit("missing_production_inference_summary_json")

    selected_total, selected_details = count_pattern_rows(selected_paths)
    skipped_total, skipped_details = count_pattern_rows(skipped_paths)
    processed_total, processed_details = count_pattern_rows(processed_paths)

    print_counts("selected_units", selected_details, selected_total)
    print_counts("skipped_units", skipped_details, skipped_total)
    print_counts("unit_prediction", processed_details, processed_total)

    summary_selected = 0
    summary_skipped = 0
    summary_processed = 0
    for path in summary_paths:
        payload = load_json(path)
        summary_selected += int(payload.get("selected_unit_count", 0))
        summary_skipped += int(payload.get("skipped_unit_count", 0))
        summary_processed += int(payload.get("processed_unit_count", 0))
        print(
            "summary:"
            f"{path.name}"
            f":selected={int(payload.get('selected_unit_count', 0))}"
            f":processed={int(payload.get('processed_unit_count', 0))}"
            f":skipped={int(payload.get('skipped_unit_count', 0))}"
        )
    print(f"summary_selected_total={summary_selected}")
    print(f"summary_processed_total={summary_processed}")
    print(f"summary_skipped_total={summary_skipped}")

    if selected_total <= 0:
        raise SystemExit("selected_units_total<=0")
    if skipped_total > 0 or summary_skipped > 0:
        raise SystemExit(f"full_infer_has_skipped_units: csv={skipped_total} summary={summary_skipped}")
    if processed_total != selected_total:
        raise SystemExit(f"processed_selected_mismatch: selected={selected_total} processed={processed_total}")
    if summary_selected != selected_total:
        raise SystemExit(f"summary_selected_mismatch: summary={summary_selected} csv={selected_total}")
    if summary_processed != processed_total:
        raise SystemExit(f"summary_processed_mismatch: summary={summary_processed} csv={processed_total}")

    print("validation_status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
